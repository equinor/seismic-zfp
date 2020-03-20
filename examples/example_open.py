import os
import sys
import numpy as np
from PIL import Image
from matplotlib import cm

import seismic_zfp

base_path = sys.argv[1]

CLIP = 0.2
SCALE = 1.0 / (2.0 * CLIP)

with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    islice_sgz = sgzfile.iline[sgzfile.ilines[len(sgzfile.ilines) // 2]]
    xslice_sgz = sgzfile.xline[sgzfile.xlines[len(sgzfile.xlines) // 2]]
    zslice_sgz = sgzfile.depth_slice[sgzfile.zslices[len(sgzfile.zslices) // 2]]

im = Image.fromarray(np.uint8(cm.seismic((islice_sgz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_inline-sgz-open.png'))

im = Image.fromarray(np.uint8(cm.seismic((xslice_sgz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_xline-sgz-open.png'))

im = Image.fromarray(np.uint8(cm.seismic((zslice_sgz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_zslice-sgz-open.png'))
