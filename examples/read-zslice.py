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

t0 = time.time()
reader = SzReader(os.path.join(base_path, '0.sz'))
slice_sz = reader.read_zslice(LINE_NO)
print("SzReader took", time.time() - t0)


im = Image.fromarray(np.uint8(cm.seismic((slice_sz.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_zslice-sz.png'))

t0 = time.time()
with segyio.open(os.path.join(base_path, '0.segy')) as segyfile:
    slice_segy = segyfile.depth_slice[LINE_NO]
print("segyio took", time.time() - t0)

im = Image.fromarray(np.uint8(cm.seismic((slice_segy.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_zslice-segy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((slice_segy-slice_sz).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_zslice-diff.png'))
