import os
import sys
import numpy as np
from PIL import Image
from matplotlib import cm

from seismic_zfp.accessors import InlineAccessor, CrosslineAccessor, ZsliceAccessor

base_path = sys.argv[1]

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)

with open(os.path.join(base_path, '0.sz'), 'rb') as f:
    iline = InlineAccessor(f)
    islice_sz = iline[iline.ilines[len(iline.ilines)//2]]

    xline = CrosslineAccessor(f)
    xslice_sz = xline[xline.xlines[len(xline.xlines)//2]]

    zslice = ZsliceAccessor(f)
    zslice_sz = zslice[zslice.zslices[len(zslice.zslices)//2]]

im = Image.fromarray(np.uint8(cm.seismic((islice_sz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_inline-sz-accessor.png'))

im = Image.fromarray(np.uint8(cm.seismic((xslice_sz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_xline-sz-accessor.png'))

im = Image.fromarray(np.uint8(cm.seismic((zslice_sz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_zslice-sz-accessor.png'))
