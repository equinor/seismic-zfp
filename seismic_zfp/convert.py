import numpy as np
import segyio
from pyzfp import compress

from .utils import pad


def convert_segy(in_filename, out_filename, method="InMemory"):
    if method == "InMemory":
        print("Converting: In={}, Out={}".format(in_filename, out_filename))
        convert_segy_inmem(in_filename, out_filename)
    else:
        raise NotImplementedError("So far can only convert SEG-Y files which fit in memory")


def convert_segy_inmem(in_filename, out_filename):
    data = segyio.tools.cube(in_filename)
    padded_shape = (pad(data.shape[0], 4), pad(data.shape[1], 4), pad(data.shape[2], 256))
    data_padded = np.zeros(padded_shape, dtype=np.float32)
    data_padded[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data
    compressed = compress(data_padded, rate=8)
    with open(out_filename, 'wb') as f:
        f.write(compressed)
