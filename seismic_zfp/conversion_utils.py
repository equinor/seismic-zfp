import zfpy
import time
import segyio
from segyio.field import Field
import importlib.metadata
from threading import Thread
from queue import Queue
import numpy as np
import warnings
import hashlib

from .version import SeismicZfpVersion
from .seismicfile import Filetype
from .sgzconstants import HEADER_DETECTION_CODES, DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES, SEGY_TRACE_HEADER_BYTES
from .utils import (pad,
                    int_to_bytes,
                    signed_int_to_bytes,
                    double_to_bytes,
                    np_float_to_bytes,
                    np_float_to_bytes_signed,
                    progress_printer,
                    CubeWithAxes,
                    InferredGeometry3d,
                    Geometry2d,
                    )


def make_header_seismic_file(seismicfile, bits_per_voxel, blockshape, geom, header_info):
    """Generate header for SGZ file from SEG-Y file"""
    buffer = make_header(seismicfile.ilines,
                         seismicfile.xlines,
                         seismicfile.samples,
                         seismicfile.tracecount,
                         header_info,
                         bits_per_voxel, blockshape, geom,
                         unstructured=seismicfile.unstructured)

    # Just copy the bytes from the SEG-Y file header
    if seismicfile.filetype == Filetype.SEGY:
        with open(seismicfile.filename, "rb") as f:
            segy_file_header = f.read(SEGY_FILE_HEADER_BYTES)
            buffer[DISK_BLOCK_BYTES:DISK_BLOCK_BYTES + SEGY_FILE_HEADER_BYTES] = segy_file_header

    elif seismicfile.filetype == Filetype.ZGY:
        buffer[84:92] = double_to_bytes(seismicfile.samples[0])
        buffer[92:100] = double_to_bytes(seismicfile.zinc*1000)

    buffer[76:80] = int_to_bytes(seismicfile.filetype.value)
    buffer[80:84] = int_to_bytes(HEADER_DETECTION_CODES[header_info.header_detection])
    return buffer


def make_header_numpy(bits_per_voxel, blockshape, source, header_info, geom, use_higher_samples_precision=False):
    """Generate header for SGZ file from numpy arrays representing axis and header values"""

    # Nothing clever to identify duplicated header arrays yet, just include everything we're given.
    buffer = make_header(source.ilines, source.xlines, source.samples,
                         len(source.ilines)*len(source.xlines),
                         header_info, bits_per_voxel, blockshape, geom)
    # These 4 bytes indicate the data source for the SGZ file. Use 20 to indicate numpy.
    buffer[76:80] = int_to_bytes(20)
    if use_higher_samples_precision:
        # higher precision minimum sample time/depth
        buffer[84:92] = double_to_bytes(source.samples[0])
        # higher precision sample interval (μs/m)
        buffer[92:100] = double_to_bytes(1000.0 * (source.samples[1] - source.samples[0]))
    return buffer


def make_header(ilines, xlines, samples, tracecount, hw_info, bits_per_voxel, blockshape, geom, unstructured=False):
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
    version = SeismicZfpVersion(importlib.metadata.version('seismic_zfp'))

    buffer[4:8] = int_to_bytes(len(samples))
    buffer[16:20] = np_float_to_bytes_signed(samples[0])
    buffer[28:32] = np_float_to_bytes_signed(1000.0 * np.array(samples[1] - samples[0]))

    if bits_per_voxel < 1:
        bpv = -int(1 / bits_per_voxel)
    else:
        bpv = int(bits_per_voxel)

    if isinstance(geom, Geometry2d):
        # Length of the seismic amplitudes cube after compression
        n_il = n_xl = 0
        compressed_data_length_diskblocks = int(((bits_per_voxel *
                                                  pad(len(samples), blockshape[2]) *
                                                  pad(tracecount, blockshape[1])) // 8) // DISK_BLOCK_BYTES)
    else:
        n_xl = len(geom.xlines)
        buffer[8:12] = int_to_bytes(n_xl)
        n_il = len(geom.ilines)
        buffer[12:16] = int_to_bytes(n_il)

        min_xl = np.int32(geom.min_xl) if unstructured else xlines[0]
        buffer[20:24] = np_float_to_bytes_signed(min_xl)
        min_il = np.int32(geom.min_il) if unstructured else ilines[0]
        buffer[24:28] = np_float_to_bytes_signed(min_il)

        if not unstructured:
            buffer[32:36] = np_float_to_bytes_signed(xlines[1] - xlines[0])
            buffer[36:40] = np_float_to_bytes_signed(ilines[1] - ilines[0])
        else:
            buffer[32:36] = np_float_to_bytes_signed(np.int32(geom.il_step))
            buffer[36:40] = np_float_to_bytes_signed(np.int32(geom.xl_step))

        compressed_data_length_diskblocks = int(((bits_per_voxel *
                                                  pad(len(samples), blockshape[2]) *
                                                  pad(n_xl, blockshape[1]) *
                                                  pad(n_il, blockshape[0])) // 8) // DISK_BLOCK_BYTES)

    buffer[40:44] = signed_int_to_bytes(bpv)
    buffer[44:48] = int_to_bytes(blockshape[0])
    buffer[48:52] = int_to_bytes(blockshape[1])
    buffer[52:56] = int_to_bytes(blockshape[2])
    buffer[56:60] = int_to_bytes(compressed_data_length_diskblocks)

    # Length of array storing one header value from every trace after compression
    if isinstance(geom, Geometry2d):
        header_entry_length_bytes = (len(geom.traces) * 32) // 8
    else:
        header_entry_length_bytes = (len(geom.xlines) * len(geom.ilines) * 32) // 8
    buffer[60:64] = int_to_bytes(header_entry_length_bytes)

    # Number of trace header arrays stored after compressed seismic amplitudes
    buffer[64:68] = int_to_bytes(hw_info.get_header_array_count())
    buffer[68:72] = int_to_bytes(tracecount if unstructured or isinstance(geom, Geometry2d) else n_il * n_xl)
    buffer[72:76] = int_to_bytes(version.encoding)

    # SEG-Y trace header info - 89 x 3 x 4 = 1068 bytes long
    buffer[980:2048] = hw_info.to_buffer()   # Start at 980 to end at 2048

    return buffer


class MinimalInlineReader:
    """
    A minimal IL reader reads an inline with the minimum number of read calls, i.e. one.
    segyio is designed around Linux, which is quite happy to wrap up millions of fread calls
    into reading from disk buffers. Windows appears reluctant to perform the same trick.
    So reading inline sample data and header data is pulled off in one big read, and this MinimalInlineReader
    sorts the data out into a numpy array, and dictionaries containing the header values.
    Because this small class cannot hope to replicate the full range of functionality which segyio
    provides, it is also endowed with a self_test function which should be called once when reading
    a SEG-Y file. This checks that for the first inline in a file both segyio and the MinimalInlineReader give
    identical output. When using this function it is recommended that segyio is used as fallback if self_test fails
    """
    def __init__(self, segyfile):
        self.segyfile = segyfile
        self.file = open(segyfile.filename, "rb")
        self.n_il = len(segyfile.ilines)
        self.n_xl = len(segyfile.xlines)
        self.n_samp = len(segyfile.samples)

    def get_format_code(self):
        return self.segyfile.bin[segyio.BinField.Format]

    def self_test(self):
        headers, array = self.read_line(0)
        array_equal = np.array_equal(self.segyfile.iline[self.segyfile.ilines[0]], array)
        headers_equal = all([h1 == h2 for h1, h2 in zip(headers, self.segyfile.header[0: self.n_xl])])
        return array_equal and headers_equal

    def read_line(self, i):
        self.file.seek(SEGY_FILE_HEADER_BYTES + i * self.n_xl * (self.n_samp * 4 + 240), 0)
        buf = self.file.read(self.n_xl * (self.n_samp * 4 + SEGY_TRACE_HEADER_BYTES))
        dt = np.dtype(np.float32).newbyteorder('>')
        array = np.frombuffer(buf, dtype=dt).reshape((self.n_xl, self.n_samp + 60))[:, 60:]
        headers = [Field(buf[h*(SEGY_TRACE_HEADER_BYTES+self.n_samp*4):
                             h*(SEGY_TRACE_HEADER_BYTES+self.n_samp*4) + SEGY_TRACE_HEADER_BYTES], kind='trace')
                   for h in range(self.n_xl)]
        if self.get_format_code() == 1:
            return headers, segyio.tools.native(array)
        elif self.get_format_code() == 5:
            return headers, array
        else:
            print("SEGY format code not in [1, 5]")
            raise RuntimeError("Three things are certain: Death, taxes, and lost data. Guess which has occurred.")


def io_thread_func_2d(blockshape, store_headers, headers_dict, trace_group_id,
                      traces_to_read, seismic_buffer, seismicfile, trace_length):
    for i in range(blockshape[1]):
        if i < traces_to_read:
            trace_id = (trace_group_id * blockshape[1] + i)
            seismic_buffer[i, 0:trace_length] = np.asarray(seismicfile.trace[trace_id])
            if store_headers:
                for tracefield, array in headers_dict.items():
                    array[trace_id] = seismicfile.header[trace_id][tracefield]

        else:
            # Repeat last plane across padding to give better compression accuracy
            seismic_buffer[i, 0:trace_length] = np.asarray(seismicfile.trace[-1])

        seismic_buffer[i, trace_length:] = np.expand_dims(seismic_buffer[i, trace_length - 1], 0)


def io_thread_func(blockshape, store_headers, headers_dict, geom, plane_set_id, planes_to_read,
                   seismic_buffer, seismicfile, minimal_il_reader, trace_length):
    for i in range(blockshape[0]):
        headers = []
        start_trace = (plane_set_id * blockshape[0] + i) * len(seismicfile.xlines) + geom.xlines[0]
        if i < planes_to_read:
            if minimal_il_reader is not None:
                headers, seismic_buffer[i, 0:len(geom.xlines), 0:trace_length] \
                    = minimal_il_reader.read_line(plane_set_id * blockshape[0] + i)
            else:
                seismic_buffer[i, 0:len(geom.xlines), 0:trace_length] = np.asarray(
                    seismicfile.iline[seismicfile.ilines[geom.ilines[0] + plane_set_id * blockshape[0] + i]]
                )[geom.xlines[0]:geom.xlines[-1]+1, :]
                if store_headers:
                    headers = seismicfile.header[start_trace: start_trace + len(geom.xlines)]

            if store_headers:
                for t, header in enumerate(headers, start_trace):
                    t_xl, t_il = t % len(seismicfile.xlines), t // len(seismicfile.xlines)
                    t_store = (t_xl - geom.xlines[0]) + (t_il - geom.ilines[0]) * len(geom.xlines)
                    for tracefield, array in headers_dict.items():
                        array[t_store] = header[tracefield]

        else:
            # Repeat last plane across padding to give better compression accuracy
            if minimal_il_reader is not None:
                _, seismic_buffer[i, 0:len(geom.xlines), 0:trace_length] \
                    = minimal_il_reader.read_line(plane_set_id * blockshape[0] + planes_to_read - 1)
            else:
                last_populated_inline_number = geom.ilines[0] + plane_set_id * blockshape[0] + planes_to_read - 1
                last_populated_inline = seismicfile.iline[seismicfile.ilines[last_populated_inline_number]]
                il_shape = (slice(geom.xlines[0], geom.xlines[-1] + 1), slice(None))
                seismic_buffer[i, 0:len(geom.xlines), 0:trace_length] = np.asarray(last_populated_inline)[il_shape]

        # Also, repeat edge values across padding. Non Quod Maneat, Sed Quod Adimimus.
        seismic_buffer[i, len(geom.xlines):, 0:trace_length] = seismic_buffer[i, len(geom.xlines) - 1, 0:trace_length]
        seismic_buffer[i, :, trace_length:] = np.expand_dims(seismic_buffer[i, :, trace_length - 1], 1)


def unstructured_io_thread_func(blockshape, store_headers, headers_dict, geom, plane_set_id,
                                segy_buffer, segyfile, trace_length):
    for i in range(blockshape[0]):
        for xl_id, xl_num in enumerate(geom.xlines):
            index = ((plane_set_id * blockshape[0] + i) * geom.il_step + geom.min_il, xl_num)
            if index in geom.traces_ref:
                trace_id = geom.traces_ref[index]
                trace, header = segyfile.trace[trace_id], segyfile.header[trace_id]
                segy_buffer[i, xl_id, 0:trace_length] = trace
                t_store = xl_id + (plane_set_id * blockshape[0] + i) * len(geom.xlines)
                if store_headers:
                    for tracefield, array in headers_dict.items():
                        array[t_store] = header[tracefield]


def numpy_producer(queue, in_array, blockshape, hash_object):
    """Copies plane-sets from input array, and puts them in the queue for writing to disk"""
    n_ilines, n_xlines, trace_length = in_array.shape
    padded_shape = (pad(n_ilines, blockshape[0]), pad(n_xlines, blockshape[1]), pad(trace_length, blockshape[2]))

    # Loop over groups of inlines
    n_plane_sets = padded_shape[0] // blockshape[0]
    for plane_set_id in range(n_plane_sets):
        if (plane_set_id+1)*blockshape[0] > n_ilines:
            ilines_pad = blockshape[0] - n_ilines % blockshape[0]
            buffer = np.pad(in_array[plane_set_id*blockshape[0]:(plane_set_id+1)*blockshape[0], :, :],
                            ((0, ilines_pad), (0, padded_shape[1]-n_xlines), (0, padded_shape[2]-trace_length)),
                            'edge')
        else:
            buffer = np.pad(in_array[plane_set_id * blockshape[0]:(plane_set_id + 1) * blockshape[0], :, :],
                            ((0, 0), (0, padded_shape[1]-n_xlines), (0, padded_shape[2]-trace_length)),
                            'edge')

        if (plane_set_id+1)*blockshape[0] > n_ilines:
            planes_to_read = n_ilines % blockshape[0]
        else:
            planes_to_read = blockshape[0]

        for i in range(planes_to_read):
            hash_object.update(buffer[i, 0:n_xlines, 0:trace_length].copy())

        if blockshape[0] == 4:
            queue.put(buffer)
        else:
            for x in range(padded_shape[1] // blockshape[1]):
                for z in range(padded_shape[2] // blockshape[2]):
                    slice = buffer[:, x * blockshape[1]: (x + 1) * blockshape[1],
                                   z * blockshape[2]: (z + 1) * blockshape[2]].copy()
                    queue.put(slice)


def seismic_file_producer_2d(queue, seismicfile, blockshape, store_headers,
                             headers_dict, geom, hash_object, verbose=True):
    n_traces, trace_length = len(geom.traces), len(seismicfile.samples)
    padded_shape = (1, pad(n_traces, blockshape[1]), pad(trace_length, blockshape[2]))

    # Loop over groups of blockshape[0] traces
    n_trace_groups = padded_shape[1] // blockshape[1]
    start_time = time.time()
    for trace_group_id in range(n_trace_groups):
        if verbose:
            progress_printer(start_time, trace_group_id / n_trace_groups)
    # Need to allocate at every step as this is being sent to another function
        if (trace_group_id+1)*blockshape[1] > n_traces:
            traces_to_read = n_traces % blockshape[1]
        else:
            traces_to_read = blockshape[1]

        seismic_buffer = np.zeros((blockshape[1], padded_shape[2]), dtype=np.float32)

        io_thread_func_2d(blockshape, store_headers, headers_dict, trace_group_id,
                          traces_to_read, seismic_buffer, seismicfile, trace_length)

        hash_object.update(seismic_buffer[0:n_traces, 0:trace_length].copy())

        if blockshape[1] == 4:
            queue.put(seismic_buffer)
        else:
            for z in range(padded_shape[2] // blockshape[2]):
                slice = seismic_buffer[:, z * blockshape[2]: (z + 1) * blockshape[2]].copy()
                queue.put(slice)


def seismic_file_producer(queue, seismicfile, blockshape, store_headers,
                          headers_dict, geom, hash_object, reduce_iops=True, verbose=True):
    """Reads and compresses data from input file, and puts them in the queue for writing to disk"""

    n_ilines, n_xlines, trace_length = len(geom.ilines), len(geom.xlines), len(seismicfile.samples)
    padded_shape = (pad(n_ilines, blockshape[0]), pad(n_xlines, blockshape[1]), pad(trace_length, blockshape[2]))

    minimal_il_reader = None
    if reduce_iops:
        if isinstance(geom, InferredGeometry3d):
            print("Cannot use MinimalInlineReader with unstructured SEG-Y")
            warnings.warn("Chaos reigns within. Reflect, repent, and reboot. Order shall return.", UserWarning)
        else:
            minimal_il_reader = MinimalInlineReader(seismicfile)
            seismicfile_shape = (len(seismicfile.ilines), len(seismicfile.xlines))
            if minimal_il_reader.self_test() and seismicfile_shape == (n_ilines, n_xlines):
                pass
            else:
                warnings.warn("MinimalInlineReader failed self-test, using fallback", UserWarning)

    # Loop over groups of 4 inlines
    n_plane_sets = padded_shape[0] // blockshape[0]
    start_time = time.time()
    if isinstance(geom, InferredGeometry3d):
        for tracefield, array in headers_dict.items():
            headers_dict[tracefield] = np.zeros(len(geom.ilines)*len(geom.xlines), dtype=np.int32)
    for plane_set_id in range(n_plane_sets):
        if verbose:
            progress_printer(start_time, plane_set_id / n_plane_sets)
        # Need to allocate at every step as this is being sent to another function
        if (plane_set_id+1)*blockshape[0] > n_ilines:
            planes_to_read = n_ilines % blockshape[0]
        else:
            planes_to_read = blockshape[0]

        seismic_buffer = np.zeros((blockshape[0], padded_shape[1], padded_shape[2]), dtype=np.float32)

        if isinstance(geom, InferredGeometry3d):
            unstructured_io_thread_func(blockshape, store_headers,  headers_dict, geom, plane_set_id,
                                        seismic_buffer, seismicfile, trace_length)
        else:
            io_thread_func(blockshape, store_headers, headers_dict, geom, plane_set_id, planes_to_read,
                           seismic_buffer, seismicfile, minimal_il_reader, trace_length)

        for i in range(planes_to_read):
            hash_object.update(seismic_buffer[i, 0:n_xlines, 0:trace_length].copy())

        if blockshape[0] == 4:
            queue.put(seismic_buffer)
        else:
            for x in range(padded_shape[1] // blockshape[1]):
                for z in range(padded_shape[2] // blockshape[2]):
                    slice = seismic_buffer[:,
                                           x * blockshape[1]: (x + 1) * blockshape[1],
                                           z * blockshape[2]: (z + 1) * blockshape[2]].copy()
                    queue.put(slice)


def compressor(queue_in, queue_out, bits_per_voxel):
    """Fetches sets of inlines and compresses them"""
    while True:
        buffer = queue_in.get()
        compressed = zfpy.compress_numpy(buffer, rate=bits_per_voxel, write_header=False)
        queue_out.put(compressed)
        queue_in.task_done()


def writer(queue, out_filehandle, header):
    """Fetches sets of compressed inlines and writes them to disk"""
    out_filehandle.write(header)
    while True:
        compressed = queue.get()
        out_filehandle.write(compressed)
        queue.task_done()


def run_conversion_loop(source, out_filehandle, bits_per_voxel, blockshape,
                        header_info, geom, queue_size=16, reduce_iops=False, store_headers=True):
    if isinstance(source, CubeWithAxes):
        header = make_header_numpy(bits_per_voxel, blockshape, source, header_info, geom)
    else:
        header = make_header_seismic_file(source, bits_per_voxel, blockshape, geom, header_info)
    # Maxsize can be reduced for machines with little memory
    # ... or for files which are so big they might be very useful.
    compression_queue = Queue(maxsize=queue_size)
    writing_queue = Queue(maxsize=queue_size)
    # schedule the consumer
    hash_object = hashlib.new('sha1')
    t_compress = Thread(target=compressor, args=(compression_queue, writing_queue, bits_per_voxel))
    t_write = Thread(target=writer, args=(writing_queue, out_filehandle, header))
    t_compress.daemon = True
    t_compress.start()
    t_write.daemon = True
    t_write.start()
    # run the appropriate producer and wait for completion
    if isinstance(source, CubeWithAxes):
        numpy_producer(compression_queue, source.data_array, blockshape, hash_object)
    elif isinstance(geom, Geometry2d):
        seismic_file_producer_2d(compression_queue, source, blockshape, store_headers,
                                 header_info.headers_dict, geom, hash_object)
    else:
        seismic_file_producer(compression_queue, source, blockshape, store_headers,
                              header_info.headers_dict, geom, hash_object, reduce_iops=reduce_iops)
    # wait until the consumer has processed all items
    compression_queue.join()
    writing_queue.join()
    out_filehandle.flush()
    return hash_object.digest()


class StreamProducer(object):
    """Handles the compression of Numpy3D data chunks for writing to disk."""

    def __init__(
        self,
        axes,
        out_filehandle,
        bits_per_voxel,
        blockshape,
        header_info,
        geom,
        total_shape,
        queue_size=16,
        use_higher_samples_precision=False,
    ):
        """
        Parameters
        ----------
        axes : tuple
            Tuple of axis names.
        out_filehandle : file object
            File object to write the compressed data to.
        bits_per_voxel : float
            Target number of bits per voxel after compression.
        blockshape : tuple
            Shape of the blocks to divide the chunks into.
        header_info : HeaderInfo
            Header information for the SGZ file.
        geom : Geometry3d
            Geometry information for the SGZ file.
        total_shape : tuple
            Total shape of the full input array.
        queue_size : int, optional
            Maximum size of the compression and writing queues.
        use_higher_samples_precision : bool, optional
            Whether to use higher precision for sample interval and sample time.
        """
        self.total_shape = total_shape
        self.blockshape = blockshape
        self.header = make_header_numpy(
            bits_per_voxel,
            blockshape,
            axes,
            header_info,
            geom,
            use_higher_samples_precision=use_higher_samples_precision,
        )
        # Maxsize can be reduced for machines with little memory
        # ... or for files which are so big they might be very useful.
        self.compression_queue = Queue(maxsize=queue_size)
        self.writing_queue = Queue(maxsize=queue_size)
        # schedule the consumer
        self.hash_object = hashlib.new("sha1")
        t_compress = Thread(
            target=compressor,
            args=(self.compression_queue, self.writing_queue, bits_per_voxel),
        )
        t_write = Thread(
            target=writer, args=(self.writing_queue, out_filehandle, self.header)
        )
        t_compress.daemon = True
        t_compress.start()
        t_write.daemon = True
        t_write.start()
        self.position = 0
        self.out_filehandle = out_filehandle

    def produce(self, in_array):
        """
        Processes a chunk of the total input array, applies necessary padding, updates the hash,
        and puts the processed chunk into the queue for writing to disk.

        :param in_array: The current chunk of the input array to process.
        """
        n_ilines, n_xlines, trace_length = self.total_shape
        padded_shape = (
            pad(n_ilines, self.blockshape[0]),
            pad(n_xlines, self.blockshape[1]),
            pad(trace_length, self.blockshape[2]),
        )
        chunk_shape = in_array.shape

        # Determine if padding is needed for the last chunk in the inline direction
        ilines_pad = 0
        is_last_chunk = self.position + chunk_shape[0] >= n_ilines
        if is_last_chunk:
            ilines_pad = self.blockshape[0] - chunk_shape[0]
        elif chunk_shape[0] != self.blockshape[0]:
            raise ValueError(
                "Chunk size along the inline axis must be equal to the blockshape."
            )
        self.position += self.blockshape[0]

        # Calculate padding for the current chunk
        padding = (
            (0, ilines_pad),
            (0, padded_shape[1] - n_xlines),
            (0, padded_shape[2] - trace_length),
        )

        # Apply padding to the chunk
        buffer = np.pad(in_array, padding, "edge")

        # Update the hash object with the data from the current chunk
        for i in range(chunk_shape[0]):
            self.hash_object.update(buffer[i, 0:n_xlines, 0:trace_length].copy())

        # Put the processed chunk into the queue
        if self.blockshape[0] == 4:
            self.compression_queue.put(buffer)
        else:
            for x in range(padded_shape[1] // self.blockshape[1]):
                for z in range(padded_shape[2] // self.blockshape[2]):
                    slice = buffer[
                        :,
                        x * self.blockshape[1] : (x + 1) * self.blockshape[1],
                        z * self.blockshape[2] : (z + 1) * self.blockshape[2],
                    ].copy()
                    self.compression_queue.put(slice)

        if is_last_chunk:
            self.compression_queue.join()
            self.writing_queue.join()
            self.out_filehandle.flush()
