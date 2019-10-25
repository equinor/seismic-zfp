import numpy as np
import segyio
from pyzfp import compress

from .utils import pad

import time


def convert_segy(in_filename, out_filename, bits_per_voxel=4, method="InMemory"):
    if method == "InMemory":
        print("Converting: In={}, Out={}".format(in_filename, out_filename))
        convert_segy_inmem(in_filename, out_filename, bits_per_voxel)
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
