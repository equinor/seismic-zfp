import numpy as np
import segyio
from pyzfp import compress

from .utils import pad



def convert_segy(in_filename, out_filename, bits_per_voxel=4, method="InMemory"):
    if method == "InMemory":
        print("Converting: In={}, Out={}".format(in_filename, out_filename))
        convert_segy_inmem(in_filename, out_filename, bits_per_voxel)
    else:
        raise NotImplementedError("So far can only convert SEG-Y files which fit in memory")


def convert_segy_inmem(in_filename, out_filename, bits_per_voxel):

    data = segyio.tools.cube(in_filename)

    padded_shape = (pad(data.shape[0], 4), pad(data.shape[1], 4), pad(data.shape[2], 2048//bits_per_voxel))
    data_padded = np.zeros(padded_shape, dtype=np.float32)
    data_padded[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data
    compressed = compress(data_padded, rate=bits_per_voxel)
    with open(out_filename, 'wb') as f:
        f.write(compressed)
