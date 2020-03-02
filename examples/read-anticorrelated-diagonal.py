from seismic_zfp.read import SzReader
from seismic_zfp.utils import get_anticorrelated_diagonal_length
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

with SzReader(os.path.join(base_path, '0.sz')) as reader:
    t0 = time.time()
    slice_sz = reader.read_anticorrelated_diagonal(LINE_NO)
    print("SzReader took", time.time() - t0)


im = Image.fromarray(np.uint8(cm.seismic((slice_sz.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_ad-sz.png'))


with segyio.open(os.path.join(base_path, '0.segy')) as segyfile:
    t0 = time.time()
    diagonal_length = get_anticorrelated_diagonal_length(LINE_NO, len(segyfile.ilines), len(segyfile.xlines))
    slice_segy = np.zeros((diagonal_length, len(segyfile.samples)))
    if LINE_NO < len(segyfile.xlines):
        for d in range(diagonal_length):
            slice_segy[d, :] = segyfile.trace[LINE_NO + d*(len(segyfile.xlines) - 1)]
    else:
        for d in range(diagonal_length):
            slice_segy[d, :] = segyfile.trace[(LINE_NO - len(segyfile.xlines) + 1 + d) * len(segyfile.xlines)
                                              + (len(segyfile.xlines) - d - 1)]
    print("segyio took", time.time() - t0)

im = Image.fromarray(np.uint8(cm.seismic((slice_segy.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_ad-segy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((slice_segy-slice_sz).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_ad-diff.png'))
