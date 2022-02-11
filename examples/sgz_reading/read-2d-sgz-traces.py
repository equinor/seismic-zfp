import seismic_zfp
import segyio
import time
import os
import sys

from PIL import Image
import numpy as np
from matplotlib import cm

base_path = sys.argv[1]

CLIP = 2.0
SCALE = 1.0/(2.0*CLIP)

with seismic_zfp.open(os.path.join(base_path, '0_2d.sgz')) as sgzfile:
    t0 = time.time()
    slice_sgz = np.stack(list(sgzfile.trace[:].copy()))
    print("seismic_zfp took", time.time() - t0)


im = Image.fromarray(np.uint8(cm.seismic((slice_sgz.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-sgz.png'))

with segyio.open(os.path.join(base_path, '0_2d.sgy'), strict=False) as sgyfile:
    t0 = time.time()
    slice_segy = np.stack(list((_.copy() for _ in sgyfile.trace[:])))
    print("segyio took", time.time() - t0)

im = Image.fromarray(np.uint8(cm.seismic((slice_segy.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-sgy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((slice_segy-slice_sgz).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-dif.png'))
