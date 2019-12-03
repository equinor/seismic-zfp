from pyzfp import compress
import asyncio
import time
import numpy as np
import segyio

from .utils import pad, np_float_to_bytes, progress_printer
from .headers import get_headerword_infolist, get_unique_headerwords
from .szconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES


def make_header(in_filename, bits_per_voxel, blockshape=(4, 4, -1)):
    """Generate header for SZ file

    Returns
    -------

    buffer: bytearray
        An 8kB byte buffer containing data required to read SZ file, including:

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
    buffer[0:4] = header_blocks.to_bytes(4, byteorder='little')

    with segyio.open(in_filename) as segyfile:
        buffer[4:8] = len(segyfile.samples).to_bytes(4, byteorder='little')
        buffer[8:12] = len(segyfile.xlines).to_bytes(4, byteorder='little')
        buffer[12:16] = len(segyfile.ilines).to_bytes(4, byteorder='little')

        # N.B. this format currently only supports integer number of ms as sampling frequency
        buffer[16:20] = np_float_to_bytes(segyfile.samples[0])
        buffer[20:24] = np_float_to_bytes(segyfile.xlines[0])
        buffer[24:28] = np_float_to_bytes(segyfile.ilines[0])

        buffer[28:32] = np_float_to_bytes(segyfile.samples[1] - segyfile.samples[0])
        buffer[32:36] = np_float_to_bytes(segyfile.xlines[1] - segyfile.xlines[0])
        buffer[36:40] = np_float_to_bytes(segyfile.ilines[1] - segyfile.ilines[0])

        hw_info_list = get_headerword_infolist(segyfile)

    buffer[40:44] = bits_per_voxel.to_bytes(4, byteorder='little')

    buffer[44:48] = blockshape[0].to_bytes(4, byteorder='little')
    buffer[48:52] = blockshape[1].to_bytes(4, byteorder='little')
    buffer[52:56] = blockshape[2].to_bytes(4, byteorder='little')

    # Length of the seismic amplitudes cube after compression
    compressed_data_length_diskblocks = ((bits_per_voxel *
                                    pad(len(segyfile.samples), blockshape[2]) *
                                    pad(len(segyfile.xlines), blockshape[1]) *
                                    pad(len(segyfile.ilines), blockshape[0])) // 8) // DISK_BLOCK_BYTES
    buffer[56:60] = compressed_data_length_diskblocks.to_bytes(4, byteorder='little')

    # Length of array storing one header value from every trace after compression
    header_entry_length_bytes = (len(segyfile.xlines) * len(segyfile.ilines) * 32) // 8
    buffer[60:64] = header_entry_length_bytes.to_bytes(4, byteorder='little')

    # Number of trace header arrays stored after compressed seismic amplitudes
    n_header_arrays = sum(hw[0] == hw[2] for hw in hw_info_list)
    buffer[64:68] = n_header_arrays.to_bytes(4, byteorder='little')

    # SEG-Y trace header info - 89 x 3 x 4 = 1068 bytes long
    hw_start_byte = 980    # Start here to end at 2048
    for i, hw_info in enumerate(hw_info_list):
        start = hw_start_byte + i*12
        buffer[start + 0:start + 4] = hw_info[0].to_bytes(4, byteorder='little', signed=True)
        buffer[start + 4:start + 8] = hw_info[1].to_bytes(4, byteorder='little', signed=True)
        buffer[start + 8:start + 12] = hw_info[2].to_bytes(4, byteorder='little', signed=True)

    # Just copy the bytes from the SEG-Y file header
    with open(in_filename, "rb") as f:
        segy_file_header = f.read(SEGY_FILE_HEADER_BYTES)
        buffer[DISK_BLOCK_BYTES:DISK_BLOCK_BYTES + SEGY_FILE_HEADER_BYTES] = segy_file_header
    return buffer


def get_header_arrays(in_filename, shape):
    with segyio.open(in_filename) as segyfile:
        headers_to_store = get_unique_headerwords(segyfile)
        header_generator = segyfile.header[0:segyfile.tracecount]
        numpy_headers_arrays = [np.zeros(segyfile.tracecount, dtype=np.int32) for _ in range(len(headers_to_store))]
        for i, header in enumerate(header_generator):
            for j, h in enumerate(headers_to_store):
                numpy_headers_arrays[j][i] = header[h]
    return numpy_headers_arrays


async def produce(queue, in_filename, blockshape, headers_to_store, numpy_headers_arrays, verbose=True):
    """Reads and compresses data from input file, and puts it in the queue for writing to disk"""
    with segyio.open(in_filename) as segyfile:

        test_slice = segyfile.iline[segyfile.ilines[0]]
        trace_length = test_slice.shape[1]
        n_xlines = len(segyfile.xlines)
        n_ilines = len(segyfile.ilines)

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
            for i in range(planes_to_read):
                data = np.asarray(segyfile.iline[segyfile.ilines[plane_set_id*blockshape[0] + i]])
                segy_buffer[i, 0:n_xlines, 0:trace_length] = data

                start_trace = (plane_set_id*blockshape[0] + i) * len(segyfile.xlines)
                header_generator = segyfile.header[start_trace: start_trace+len(segyfile.xlines)]
                for t, header in enumerate(header_generator, start_trace):
                    for j, h in enumerate(headers_to_store):
                        numpy_headers_arrays[j][t] = header[h]

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


async def run(in_filename, out_filename, bits_per_voxel, blockshape, headers_to_store, numpy_headers_arrays):
    asyncio.set_event_loop(asyncio.new_event_loop())
    header = make_header(in_filename, bits_per_voxel, blockshape)

    # Maxsize can be reduced for machines with little memory
    # ... or for files which are so big they might be very useful.
    queue = asyncio.Queue(maxsize=16)
    # schedule the consumer
    consumer = asyncio.ensure_future(consume(header, queue, out_filename, bits_per_voxel))
    # run the producer and wait for completion
    await produce(queue, in_filename, blockshape, headers_to_store, numpy_headers_arrays)
    # wait until the consumer has processed all items
    await queue.join()
    # the consumer is still awaiting for an item, cancel it
    consumer.cancel()
