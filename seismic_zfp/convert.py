import numpy as np
import segyio
from pyzfp import compress
import asyncio

from .utils import pad

import time


def convert_segy(in_filename, out_filename, bits_per_voxel=4, method="InMemory"):
    if method == "InMemory":
        print("Converting: In={}, Out={}".format(in_filename, out_filename))
        convert_segy_inmem(in_filename, out_filename, bits_per_voxel)
    elif method == "Stream":
        print("Converting: In={}, Out={}".format(in_filename, out_filename))
        convert_segy_stream(in_filename, out_filename, bits_per_voxel)
    else:
        raise NotImplementedError("So far can only convert SEG-Y files which fit in memory")

def make_header(in_filename, bits_per_voxel):
    with segyio.open(in_filename) as segyfile:
        n_samples = len(segyfile.samples)
        n_xlines = len(segyfile.xlines)
        n_ilines = len(segyfile.ilines)

        samples_start = int((segyfile.samples[0]).astype(int))
        xline_start = int((segyfile.xlines[0]).astype(int))
        iline_start = int((segyfile.ilines[0]).astype(int))

        samples_step = int((segyfile.samples[1] - segyfile.samples[0]).astype(int))
        xline_step = int((segyfile.xlines[1] - segyfile.xlines[0]).astype(int))
        iline_step = int((segyfile.ilines[1] - segyfile.ilines[0]).astype(int))


    print("n_samples={}, n_xlines={}, n_ilines={}".format(n_samples, n_xlines, n_ilines))
    print("samples_start={}, xline_start={}, iline_start={}".format(samples_start, xline_start, iline_start))
    print("samples_step={}, xline_step={}, iline_step={}".format(samples_step, xline_step, iline_step))


    header_blocks = 1
    buffer = bytearray(4096 * header_blocks)
    buffer[0:4] = header_blocks.to_bytes(4, byteorder='little')

    buffer[4:8] = n_samples.to_bytes(4, byteorder='little')
    buffer[8:12] = n_xlines.to_bytes(4, byteorder='little')
    buffer[12:16] = n_ilines.to_bytes(4, byteorder='little')

    buffer[16:20] = samples_start.to_bytes(4, byteorder='little')
    buffer[20:24] = xline_start.to_bytes(4, byteorder='little')
    buffer[24:28] = iline_start.to_bytes(4, byteorder='little')

    buffer[28:32] = samples_step.to_bytes(4, byteorder='little')
    buffer[32:36] = xline_step.to_bytes(4, byteorder='little')
    buffer[36:40] = iline_step.to_bytes(4, byteorder='little')

    buffer[40:44] = bits_per_voxel.to_bytes(4, byteorder='little')

    return buffer

def convert_segy_inmem(in_filename, out_filename, bits_per_voxel):
    header = make_header(in_filename, bits_per_voxel)

    t0 = time.time()

    data = segyio.tools.cube(in_filename)
    t1 = time.time()

    padded_shape = (pad(data.shape[0], 4), pad(data.shape[1], 4), pad(data.shape[2], 2048//bits_per_voxel))
    data_padded = np.zeros(padded_shape, dtype=np.float32)
    data_padded[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data
    compressed = compress(data_padded, rate=bits_per_voxel)
    t2 = time.time()

    with open(out_filename, 'wb') as f:
        f.write(header)
        f.write(compressed)
    t3 = time.time()

    print("Total conversion time: {}, of which read={}, compress={}, write={}".format(t3-t0, t1-t0, t2-t1, t3-t2))



async def produce(queue, in_filename, bits_per_voxel):
    with segyio.open(in_filename) as segyfile:

        test_slice = segyfile.iline[segyfile.ilines[0]]
        trace_length = test_slice.shape[1]
        n_xlines = len(segyfile.xlines)
        n_ilines = len(segyfile.ilines)

        padded_shape = (pad(n_ilines, 4), pad(n_xlines, 4), pad(trace_length, 2048 // bits_per_voxel))

        print("Trace length={}, n_xlines={}, n_ilines={}, padded_shape={}".format(trace_length, n_xlines, n_ilines, padded_shape))

        for plane_set_id in range(padded_shape[0] // 4):
            segy_buffer = np.zeros((4, padded_shape[1], padded_shape[2]), dtype=np.float32)
            if (plane_set_id+1)*4 > n_ilines:
                planes_to_read = n_ilines % 4
            else:
                planes_to_read = 4
            for i in range(planes_to_read):
                data = np.asarray(segyfile.iline[segyfile.ilines[plane_set_id*4 + i]])
                segy_buffer[i, 0:n_xlines, 0:trace_length] = data

            await queue.put(segy_buffer)


async def consume(header, queue, out_filename, bits_per_voxel):
    with open(out_filename, 'wb') as f:
        f.write(header)
        while True:
            segy_buffer = await queue.get()
            compressed = compress(segy_buffer, rate=bits_per_voxel)
            f.write(compressed)
            queue.task_done()


async def run(in_filename, out_filename, bits_per_voxel):
    header = make_header(in_filename, bits_per_voxel)

    queue = asyncio.Queue(maxsize=16)
    # schedule the consumer
    consumer = asyncio.ensure_future(consume(header, queue, out_filename, bits_per_voxel))
    # run the producer and wait for completion
    await produce(queue, in_filename, bits_per_voxel)
    # wait until the consumer has processed all items
    await queue.join()
    # the consumer is still awaiting for an item, cancel it
    consumer.cancel()



def convert_segy_stream(in_filename, out_filename, bits_per_voxel):
    t0 = time.time()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(in_filename, out_filename, bits_per_voxel))
    loop.close()

    t3 = time.time()
    print("Total conversion time: {}".format(t3-t0))
