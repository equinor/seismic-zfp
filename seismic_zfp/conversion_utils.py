from pyzfp import compress
import asyncio
import time
import numpy as np
import segyio
import pkg_resources

from .version import SeismicZfpVersion
from .utils import pad, int_to_bytes, signed_int_to_bytes, np_float_to_bytes, progress_printer
from .headers import get_headerword_infolist, get_unique_headerwords
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES


def make_header(in_filename, bits_per_voxel, blockshape=(4, 4, -1), min_il=0, max_il=None, min_xl=0, max_xl=None):
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

    with segyio.open(in_filename) as segyfile:
        buffer[4:8] = int_to_bytes(len(segyfile.samples))
        n_xl = len(segyfile.xlines) if max_xl is None else max_xl - min_xl
        buffer[8:12] = int_to_bytes(n_xl)
        n_il = len(segyfile.ilines) if max_il is None else max_il - min_il
        buffer[12:16] = int_to_bytes(n_il)

        # N.B. this format currently only supports integer number of ms as sampling frequency
        buffer[16:20] = np_float_to_bytes(segyfile.samples[0])
        buffer[20:24] = np_float_to_bytes(segyfile.xlines[min_xl])
        buffer[24:28] = np_float_to_bytes(segyfile.ilines[min_il])

        buffer[28:32] = np_float_to_bytes(segyfile.samples[1] - segyfile.samples[0])
        buffer[32:36] = np_float_to_bytes(segyfile.xlines[1] - segyfile.xlines[0])
        buffer[36:40] = np_float_to_bytes(segyfile.ilines[1] - segyfile.ilines[0])

        hw_info_list = get_headerword_infolist(segyfile)

    if bits_per_voxel < 1:
        bpv = -int(1 / bits_per_voxel)
    else:
        bpv = bits_per_voxel
        
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
    header_entry_length_bytes = (len(segyfile.xlines) * len(segyfile.ilines) * 32) // 8
    buffer[60:64] = int_to_bytes(header_entry_length_bytes)

    # Number of trace header arrays stored after compressed seismic amplitudes
    n_header_arrays = sum(hw[0] == hw[2] for hw in hw_info_list)
    buffer[64:68] = int_to_bytes(n_header_arrays)

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


async def produce(queue, in_filename, blockshape, headers_to_store, numpy_headers_arrays,
                  min_il, max_il, min_xl, max_xl, verbose=True):
    """Reads and compresses data from input file, and puts it in the queue for writing to disk"""
    with segyio.open(in_filename) as segyfile:

        test_slice = segyfile.iline[segyfile.ilines[0]]
        trace_length = test_slice.shape[1]
        if max_xl is None:
            n_xlines = len(segyfile.xlines)
            max_xl = n_xlines
        else:
            n_xlines = max_xl - min_xl

        if max_il is None:
            n_ilines = len(segyfile.ilines)
        else:
            n_ilines = max_il - min_il

        padded_shape = (pad(n_ilines, blockshape[0]), pad(n_xlines, blockshape[1]), pad(trace_length, blockshape[2]))

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
            for i in range(blockshape[0]):
                if i < planes_to_read:
                    data = np.asarray(segyfile.iline[segyfile.ilines[min_il + plane_set_id*blockshape[0] + i]]
                                      )[min_xl:max_xl, :]
                else:
                    # Repeat last plane across padding to give better compression accuracy
                    data = np.asarray(segyfile.iline[segyfile.ilines[min_il + plane_set_id*blockshape[0] + planes_to_read - 1]]
                                      )[min_xl:max_xl, :]
                segy_buffer[i, 0:n_xlines, 0:trace_length] = data

                # Also, repeat edge values across padding. Non Quod Maneat, Sed Quod Adimimus.
                segy_buffer[i, n_xlines:, 0:trace_length] = data[-1, :]
                segy_buffer[i, :, trace_length:] = np.expand_dims(segy_buffer[i, :, trace_length - 1], 1)

                start_trace = (plane_set_id*blockshape[0] + i) * len(segyfile.xlines) + min_xl
                header_generator = segyfile.header[start_trace: start_trace+n_xlines]

                for t, header in enumerate(header_generator, start_trace):
                    t_xl = t % len(segyfile.xlines)
                    t_il = t // len(segyfile.xlines)
                    t_store = (t_xl - min_xl) + (t_il - min_il) * n_xlines
                    for j, h in enumerate(headers_to_store):
                        numpy_headers_arrays[j][t_store] = header[h]

            if blockshape[0] == 4:
                await queue.put(segy_buffer)
            else:
                for x in range(padded_shape[1] // blockshape[1]):
                    for z in range(padded_shape[2] // blockshape[2]):
                        slice = segy_buffer[:, x * blockshape[1]: (x + 1) * blockshape[1],
                                               z * blockshape[2]: (z + 1) * blockshape[2]].copy()
                        await queue.put(slice)


async def consume(header, queue, out_filename, bits_per_voxel):
    """Fetches compressed sets of inlines (or just blocks) and writes them to disk"""
    with open(out_filename, 'wb') as f:
        f.write(header)
        while True:
            segy_buffer = await queue.get()
            compressed = compress(segy_buffer, rate=bits_per_voxel)
            f.write(compressed)
            queue.task_done()


async def run_conversion_loop(in_filename, out_filename, bits_per_voxel, blockshape, headers_to_store,
                              numpy_headers_arrays, min_il, max_il, min_xl, max_xl):
    header = make_header(in_filename, bits_per_voxel, blockshape, min_il, max_il, min_xl, max_xl)

    # Maxsize can be reduced for machines with little memory
    # ... or for files which are so big they might be very useful.
    queue = asyncio.Queue(maxsize=16)
    # schedule the consumer
    consumer = asyncio.ensure_future(consume(header, queue, out_filename, bits_per_voxel))
    # run the producer and wait for completion
    await produce(queue, in_filename, blockshape, headers_to_store, numpy_headers_arrays,
                  min_il, max_il, min_xl, max_xl)
    # wait until the consumer has processed all items
    await queue.join()
    # the consumer is still awaiting for an item, cancel it
    consumer.cancel()
