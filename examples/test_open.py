import os
import sys
import numpy as np
from PIL import Image
from matplotlib import cm

import seismic_zfp

base_path = sys.argv[1]

CLIP = 0.2
SCALE = 1.0 / (2.0 * CLIP)

with seismic_zfp.open(os.path.join(base_path, '0.sz')) as szfile:
    islice_sz = szfile.iline[szfile.ilines[len(szfile.ilines) // 2]]
    xslice_sz = szfile.xline[szfile.xlines[len(szfile.xlines) // 2]]
    zslice_sz = szfile.depth_slice[szfile.zslices[len(szfile.zslices) // 2]]

im = Image.fromarray(np.uint8(cm.seismic((islice_sz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_inline-sz-open.png'))

im = Image.fromarray(np.uint8(cm.seismic((xslice_sz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_xline-sz-open.png'))

im = Image.fromarray(np.uint8(cm.seismic((zslice_sz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_zslice-sz-open.png'))
