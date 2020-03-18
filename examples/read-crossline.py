from seismic_zfp.read import SgzReader
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

with SgzReader(os.path.join(base_path, '0.sgz')) as reader:
    t0 = time.time()
    slice_sgz = reader.read_crossline(LINE_NO)
    print("SgzReader took", time.time() - t0)


im = Image.fromarray(np.uint8(cm.seismic((slice_sgz.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_crossline-sgz.png'))

with segyio.open(os.path.join(base_path, '0.sgy')) as segyfile:
    t0 = time.time()
    slice_segy = segyfile.xline[segyfile.xlines[LINE_NO]]
    print("segyio took", time.time() - t0)

im = Image.fromarray(np.uint8(cm.seismic((slice_segy.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_crossline-sgy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((slice_segy-slice_sgz).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_crossline-dif.png'))
