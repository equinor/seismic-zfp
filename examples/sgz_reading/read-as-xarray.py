import xarray as xr
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

with xr.open_dataset(os.path.join(base_path, '0.sgz')) as dataset:
    t0 = time.time()
    slice_xr = dataset.data[LINE_NO:LINE_NO+1, 0:605, 0:901].to_numpy()[0,:,:]
    print("xarray took", time.time() - t0)


im = Image.fromarray(np.uint8(cm.seismic((slice_xr.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-xr.png'))

with segyio.open(os.path.join(base_path, '0.sgy')) as segyfile:
    t0 = time.time()
    slice_segy = segyfile.iline[segyfile.ilines[LINE_NO]]
    print("segyio took", time.time() - t0)

im = Image.fromarray(np.uint8(cm.seismic((slice_segy.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-sgy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((slice_segy-slice_xr).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-dif.png'))
