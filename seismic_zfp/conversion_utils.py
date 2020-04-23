import zfpy
import time
import numpy as np
import segyio
from segyio.field import Field
import pkg_resources
from threading import Thread
from queue import Queue

from .version import SeismicZfpVersion
from .utils import pad, int_to_bytes, signed_int_to_bytes, np_float_to_bytes, progress_printer, InferredGeometry
from .headers import get_headerword_infolist, get_unique_headerwords
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES, SEGY_TRACE_HEADER_BYTES


def make_header(in_filename, bits_per_voxel, blockshape, geom):
    """Generate header for SGZ file

    Returns
    -------

    buffer: bytearray
        An 8kB byte buffer containing data required to read SGZ file, including:

        First 4kB
        - Seismic cube dimensions
        - Compression settings (bitrate, disk block packing scheme)
        - Invariant SEG-Y trace header values
        - File location of varying SEG-Y trace header values

        Second 4kB
        - SEG-Y File header
    """
    header_blocks = 2
    buffer = bytearray(DISK_BLOCK_BYTES * header_blocks)
    buffer[0:4] = int_to_bytes(header_blocks)
    version = SeismicZfpVersion(pkg_resources.get_distribution('seismic_zfp').version)

    with segyio.open(in_filename, mode='r', strict=False) as segyfile:
        buffer[4:8] = int_to_bytes(len(segyfile.samples))
        n_xl = len(geom.xlines)
        buffer[8:12] = int_to_bytes(n_xl)
        n_il = len(geom.ilines)
        buffer[12:16] = int_to_bytes(n_il)

        # N.B. this format currently only supports integer number of ms as sampling frequency
        buffer[16:20] = np_float_to_bytes(segyfile.samples[0])
        min_xl = np.int32(geom.min_xl) if segyfile.unstructured else segyfile.xlines[0]
        buffer[20:24] = np_float_to_bytes(min_xl)
        min_il = np.int32(geom.min_il) if segyfile.unstructured else segyfile.ilines[0]
        buffer[24:28] = np_float_to_bytes(min_il)

        buffer[28:32] = np_float_to_bytes(segyfile.samples[1] - segyfile.samples[0])
        if not segyfile.unstructured:
            buffer[32:36] = np_float_to_bytes(segyfile.xlines[1] - segyfile.xlines[0])
            buffer[36:40] = np_float_to_bytes(segyfile.ilines[1] - segyfile.ilines[0])
        else:
            # Assume il/xl steps of 1 for unstructured. Can probably do better than this...
            buffer[32:36] = np_float_to_bytes(np.int32(1))
            buffer[36:40] = np_float_to_bytes(np.int32(1))

        hw_info_list = get_headerword_infolist(segyfile)

    if bits_per_voxel < 1:
        bpv = -int(1 / bits_per_voxel)
    else:
        bpv = int(bits_per_voxel)
        
    buffer[40:44] = signed_int_to_bytes(bpv)
    
    buffer[44:48] = int_to_bytes(blockshape[0])
    buffer[48:52] = int_to_bytes(blockshape[1])
    buffer[52:56] = int_to_bytes(blockshape[2])

    # Length of the seismic amplitudes cube after compression
    compressed_data_length_diskblocks = int(((bits_per_voxel *
                                    pad(len(segyfile.samples), blockshape[2]) *
                                    pad(n_xl, blockshape[1]) *
                                    pad(n_il, blockshape[0])) // 8) // DISK_BLOCK_BYTES)
    buffer[56:60] = int_to_bytes(compressed_data_length_diskblocks)

    # Length of array storing one header value from every trace after compression
    if geom is None:
        header_entry_length_bytes = (len(segyfile.xlines) * len(segyfile.ilines) * 32) // 8
    else:
        header_entry_length_bytes = (len(geom.xlines) * len(geom.ilines) * 32) // 8
    buffer[60:64] = int_to_bytes(header_entry_length_bytes)

    # Number of trace header arrays stored after compressed seismic amplitudes
    n_header_arrays = sum(hw[0] == hw[2] for hw in hw_info_list)
    buffer[64:68] = int_to_bytes(n_header_arrays)
    buffer[68:72] = int_to_bytes(segyfile.tracecount)
    buffer[72:76] = int_to_bytes(version.encoding)

    # SEG-Y trace header info - 89 x 3 x 4 = 1068 bytes long
    hw_start_byte = 980    # Start here to end at 2048
    for i, hw_info in enumerate(hw_info_list):
        start = hw_start_byte + i*12
        buffer[start + 0:start + 4] = signed_int_to_bytes(hw_info[0])
        buffer[start + 4:start + 8] = signed_int_to_bytes(hw_info[1])
        buffer[start + 8:start + 12] = signed_int_to_bytes(hw_info[2])

    # Just copy the bytes from the SEG-Y file header
    with open(in_filename, "rb") as f:
        segy_file_header = f.read(SEGY_FILE_HEADER_BYTES)
        buffer[DISK_BLOCK_BYTES:DISK_BLOCK_BYTES + SEGY_FILE_HEADER_BYTES] = segy_file_header
    return buffer


def get_header_arrays(in_filename):
    with segyio.open(in_filename) as segyfile:
        headers_to_store = get_unique_headerwords(segyfile)
        header_generator = segyfile.header[0:segyfile.tracecount]
        numpy_headers_arrays = [np.zeros(segyfile.tracecount, dtype=np.int32) for _ in range(len(headers_to_store))]
        for i, header in enumerate(header_generator):
            for j, h in enumerate(headers_to_store):
                numpy_headers_arrays[j][i] = header[h]
    return numpy_headers_arrays


# A minimal IL reader reads an inline with the minimum number of read calls, i.e. one.
# segyio is designed around Linux, which is quite happy to wrap up millions of fread calls
# into reading from disk buffers. Windows appears reluctant to perform the same trick.
# So reading inline sample data and header data is pulled off in one big read, and this MinimalInlineReader
# sorts the data out into a numpy array, and dictionaries containing the header values.
# Because this small class cannot hope to replicate the full range of functionality which segyio
# provides, it is also endowed with a self_test function which should be called once when reading
# a SEG-Y file. This checks that for the first inline in a file both segyio and the MinimalInlineReader give
# identical output. When using this function it is recommended that segyio is used as fallback if self_test fails
class MinimalInlineReader:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename, "rb")
        with segyio.open(filename) as segyfile:
            self.n_il = len(segyfile.ilines)
            self.n_xl = len(segyfile.xlines)
            self.n_samp = len(segyfile.samples)
            self.format = segyfile.bin[segyio.BinField.Format]

    def self_test(self):
        headers, array = self.read_line(0)
        with segyio.open(self.filename) as segyfile:
            array_equal = np.array_equal(segyfile.iline[segyfile.ilines[0]], array)
            headers_equal = all([h1 == h2 for h1, h2 in zip(headers, segyfile.header[0: self.n_xl])])
        return array_equal and headers_equal

    def read_line(self, i):
        self.file.seek(SEGY_FILE_HEADER_BYTES + i * self.n_xl * (self.n_samp * 4 + 240), 0)
        buf = self.file.read(self.n_xl * (self.n_samp * 4 + SEGY_TRACE_HEADER_BYTES))
        dt = np.dtype(np.float32).newbyteorder('>')
        array = np.frombuffer(buf, dtype=dt).reshape((self.n_xl, self.n_samp + 60))[:, 60:]
        headers = [Field(buf[h*(SEGY_TRACE_HEADER_BYTES+self.n_samp*4):
                             h*(SEGY_TRACE_HEADER_BYTES+self.n_samp*4) + SEGY_TRACE_HEADER_BYTES], kind='trace')
                   for h in range(self.n_xl)]
        if self.format == 1:
            return headers, segyio.tools.native(array)
        elif self.format == 5:
            return headers, array
        else:
            print("SEGY format code not in [1, 5]")
            raise RuntimeError("Three things are certain: Death, taxes, and lost data. Guess which has occurred.")


def io_thread_func(blockshape, headers_to_store, geom, numpy_headers_arrays,
                   plane_set_id, planes_to_read, segy_buffer, segyfile, minimal_il_reader, trace_length):
    for i in range(blockshape[0]):
        start_trace = (plane_set_id * blockshape[0] + i) * len(segyfile.xlines) + geom.xlines[0]
        if i < planes_to_read:
            if minimal_il_reader is not None:
                headers, data = minimal_il_reader.read_line(plane_set_id * blockshape[0] + i)
            else:
                data = np.asarray(segyfile.iline[segyfile.ilines[geom.ilines[0] + plane_set_id * blockshape[0] + i]]
                                  )[geom.xlines[0]:geom.xlines[-1]+1, :]
                headers = segyfile.header[start_trace: start_trace + len(geom.xlines)]

            for t, header in enumerate(headers, start_trace):
                t_xl = t % len(segyfile.xlines)
                t_il = t // len(segyfile.xlines)
                t_store = (t_xl - geom.xlines[0]) + (t_il - geom.ilines[0]) * len(geom.xlines)
                for j, h in enumerate(headers_to_store):
                    numpy_headers_arrays[j][t_store] = header[h]

        else:
            # Repeat last plane across padding to give better compression accuracy
            if minimal_il_reader is not None:
                _, data = minimal_il_reader.read_line(plane_set_id * blockshape[0] + planes_to_read - 1)
            else:
                data = np.asarray(segyfile.iline[segyfile.ilines[geom.ilines[0] + plane_set_id * blockshape[0] + planes_to_read - 1]]
                                  )[geom.xlines[0]:geom.xlines[-1]+1, :]

        segy_buffer[i, 0:len(geom.xlines), 0:trace_length] = data
        # Also, repeat edge values across padding. Non Quod Maneat, Sed Quod Adimimus.
        segy_buffer[i, len(geom.xlines):, 0:trace_length] = data[-1, :]
        segy_buffer[i, :, trace_length:] = np.expand_dims(segy_buffer[i, :, trace_length - 1], 1)


def unstructured_io_thread_func(blockshape, headers_to_store, geom,
                                numpy_headers_arrays, plane_set_id,
                                segy_buffer, segyfile, trace_length):
    for i in range(blockshape[0]):
        for xl_id, xl_num in enumerate(geom.xlines):
            index = (plane_set_id * blockshape[0] + i + geom.min_il, xl_num)
            if index in geom.traces_ref:
                trace_id = geom.traces_ref[index]
                trace, header = segyfile.trace[trace_id], segyfile.header[trace_id]
                segy_buffer[i, xl_id, 0:trace_length] = trace
                t_store = xl_id + (plane_set_id * blockshape[0] + i - geom.min_il) * len(geom.xlines)
                for k, h in enumerate(headers_to_store):
                    numpy_headers_arrays[k][t_store] = header[h]


def producer(queue, in_filename, blockshape, headers_to_store, numpy_headers_arrays, geom,
             reduce_iops=True, verbose=True):
    """Reads and compresses data from input file, and puts it in the queue for writing to disk"""
    with segyio.open(in_filename, mode='r', strict=False) as segyfile:

        n_ilines = len(geom.ilines)
        n_xlines = len(geom.xlines)
        trace_length = len(segyfile.samples)

        padded_shape = (pad(n_ilines, blockshape[0]), pad(n_xlines, blockshape[1]), pad(trace_length, blockshape[2]))

        minimal_il_reader = None
        if reduce_iops:
            if isinstance(geom, InferredGeometry):
                print("Cannot use MinimalInlineReader with unstructured SEG-Y")
                raise RuntimeError("Chaos reigns within. Reflect, repent, and reboot. Order shall return.")

            minimal_il_reader = MinimalInlineReader(in_filename)
            if minimal_il_reader.self_test() and n_ilines == len(segyfile.ilines) and n_xlines == len(segyfile.xlines):
                print("MinimalInlineReader passed self-test")
            else:
                print("MinimalInlineReader failed self-test, using fallback")

        # Loop over groups of 4 inlines
        n_plane_sets = padded_shape[0] // blockshape[0]
        start_time = time.time()
        for plane_set_id in range(n_plane_sets):
            if verbose:
                progress_printer(start_time, plane_set_id / n_plane_sets)
            # Need to allocate at every step as this is being sent to another function
            if (plane_set_id+1)*blockshape[0] > n_ilines:
                planes_to_read = n_ilines % blockshape[0]
            else:
                planes_to_read = blockshape[0]

            segy_buffer = np.zeros((blockshape[0], padded_shape[1], padded_shape[2]), dtype=np.float32)

            if isinstance(geom, InferredGeometry):
                unstructured_io_thread_func(blockshape, headers_to_store, geom,
                                            numpy_headers_arrays, plane_set_id,
                                            segy_buffer, segyfile, trace_length)
            else:
                io_thread_func(blockshape, headers_to_store, geom,
                               numpy_headers_arrays, plane_set_id, planes_to_read,
                               segy_buffer, segyfile, minimal_il_reader, trace_length)

            if blockshape[0] == 4:
                queue.put(segy_buffer)
            else:
                for x in range(padded_shape[1] // blockshape[1]):
                    for z in range(padded_shape[2] // blockshape[2]):
                        slice = segy_buffer[:, x * blockshape[1]: (x + 1) * blockshape[1],
                                               z * blockshape[2]: (z + 1) * blockshape[2]].copy()
                        queue.put(slice)


def consumer(queue, header, out_filename, bits_per_voxel):
    """Fetches compressed sets of inlines (or just blocks) and writes them to disk"""
    with open(out_filename, 'wb') as f:
        f.write(header)
        while True:
            segy_buffer = queue.get()
            compressed = zfpy.compress_numpy(segy_buffer, rate=bits_per_voxel, write_header=False)
            f.write(compressed)
            queue.task_done()


def run_conversion_loop(in_filename, out_filename, bits_per_voxel, blockshape, headers_to_store,
                              numpy_headers_arrays, geom, queuesize=16, reduce_iops=False):
    header = make_header(in_filename, bits_per_voxel, blockshape, geom)

    # Maxsize can be reduced for machines with little memory
    # ... or for files which are so big they might be very useful.
    queue = Queue(maxsize=queuesize)
    # schedule the consumer
    t = Thread(target=consumer, args=(queue, header, out_filename, bits_per_voxel))
    t.daemon = True
    t.start()
    # run the producer and wait for completion
    producer(queue, in_filename, blockshape, headers_to_store, numpy_headers_arrays, geom, reduce_iops=reduce_iops)
    # wait until the consumer has processed all items
    queue.join()
