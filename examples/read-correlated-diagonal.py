from seismic_zfp.read import SzReader
import segyio
import time
import os
import sys

from PIL import Image
import numpy as np
from matplotlib import cm

base_path = sys.argv[1]
LINE_NO = int(sys.argv[2])

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)


def get_correlated_diagonal_length(cd, n_il, n_xl):
    if n_xl > n_il:
        if cd >= 0:
            return n_il - cd
        elif abs(cd) <= n_xl - n_il:
            return n_il
        else:  # cd is negative
            return n_il + cd + (n_xl - n_il) + 1
    elif n_xl < n_il:
        if cd <= 0:
            return n_xl + cd
        elif abs(cd) <= n_il - n_xl:
            return n_xl
        else:
            return n_xl - cd + (n_il - n_xl) + 1
    else:  # Equal number of ILs & XLs
        return n_il - abs(cd)


with SzReader(os.path.join(base_path, '0.sz')) as reader:
    t0 = time.time()
    slice_sz = reader.read_correlated_diagonal(LINE_NO)
    print("SzReader took", time.time() - t0)


im = Image.fromarray(np.uint8(cm.seismic((slice_sz.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_cd-sz.png'))


with segyio.open(os.path.join(base_path, '0.segy')) as segyfile:
    t0 = time.time()
    diagonal_length = get_correlated_diagonal_length(LINE_NO, len(segyfile.ilines), len(segyfile.xlines))
    slice_segy = np.zeros((diagonal_length, len(segyfile.samples)))
    if LINE_NO >= 0:
        for d in range(diagonal_length):
            slice_segy[d, :] = segyfile.trace[(d+LINE_NO)*len(segyfile.xlines) + d]
    else:
        for d in range(diagonal_length):
            slice_segy[d, :] = segyfile.trace[d*len(segyfile.xlines) + d - LINE_NO]
    print("segyio took", time.time() - t0)

im = Image.fromarray(np.uint8(cm.seismic((slice_segy.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_cd-segy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((slice_segy-slice_sz).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_cd-diff.png'))
