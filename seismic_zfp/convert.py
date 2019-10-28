import numpy as np
import segyio
from pyzfp import compress

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


def convert_segy_inmem(in_filename, out_filename, bits_per_voxel):
    t0 = time.time()

    data = segyio.tools.cube(in_filename)
    t1 = time.time()

    padded_shape = (pad(data.shape[0], 4), pad(data.shape[1], 4), pad(data.shape[2], 2048//bits_per_voxel))
    data_padded = np.zeros(padded_shape, dtype=np.float32)
    data_padded[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data
    compressed = compress(data_padded, rate=bits_per_voxel)
    t2 = time.time()

    with open(out_filename, 'wb') as f:
        f.write(compressed)
    t3 = time.time()

    print("Total conversion time: {}, of which read={}, compress={}, write={}".format(t3-t0, t1-t0, t2-t1, t3-t2))


def convert_segy_stream(in_filename, out_filename, bits_per_voxel):
    t0 = time.time()

    with segyio.open(in_filename) as segyfile:

        test_slice = segyfile.iline[segyfile.ilines[0]]
        trace_length = test_slice.shape[1]
        n_xlines = len(segyfile.xlines)
        n_ilines = len(segyfile.ilines)

        padded_shape = (pad(n_ilines, 4), pad(n_xlines, 4), pad(trace_length, 2048 // bits_per_voxel))

        print("Trace length={}, n_xlines={}, n_ilines={}, padded_shape={}".format(trace_length, n_xlines, n_ilines, padded_shape))

        segy_buffer = np.zeros((4, padded_shape[1], padded_shape[2]), dtype=np.float32)

        with open(out_filename, 'wb') as f:
            for plane_set_id in range(padded_shape[0] // 4):
                if (plane_set_id+1)*4 > n_ilines:
                    planes_to_read = n_ilines % 4
                    segy_buffer = np.zeros((4, padded_shape[1], padded_shape[2]), dtype=np.float32)
                else:
                    planes_to_read = 4
                for i in range(planes_to_read):
                    data = np.asarray(segyfile.iline[segyfile.ilines[plane_set_id*4 + i]])
                    segy_buffer[i, 0:n_xlines, 0:trace_length] = data

                compressed = compress(segy_buffer, rate=bits_per_voxel)
                f.write(compressed)

    t3 = time.time()
    print("Total conversion time: {}".format(t3-t0))
