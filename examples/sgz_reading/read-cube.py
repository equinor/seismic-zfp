import seismic_zfp
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
cube_sgz = seismic_zfp.tools.cube(os.path.join(base_path, '0.sgz'))
print("SgzReader took", time.time() - t0)

im = Image.fromarray(np.uint8(cm.seismic((cube_sgz[LINE_NO, :, :].T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-sgz.png'))


t0 = time.time()
cube_sgy = segyio.tools.cube(os.path.join(base_path, '0.sgy'))
print("segyio took", time.time() - t0)

im = Image.fromarray(np.uint8(cm.seismic((cube_sgy[LINE_NO, :, :].T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-sgy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((cube_sgy[LINE_NO, :, :]-cube_sgz[LINE_NO, :, :]).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-dif.png'))
