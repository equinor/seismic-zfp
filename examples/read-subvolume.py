from seismic_zfp.read import SgzReader
import segyio
import time
import os
import sys

from PIL import Image
import numpy as np
from matplotlib import cm

base_path = sys.argv[1]

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)

min_il, max_il = 33, 193
min_xl, max_xl = 63, 123
min_z, max_z = 256, 345

with SgzReader(os.path.join(base_path, '0.sgz')) as reader:
    t0 = time.time()
    vol_sgz = reader.read_subvolume(min_il=min_il, max_il=max_il, min_xl=min_xl, max_xl=max_xl, min_z=min_z, max_z=max_z)
    print("SgzReader took", time.time() - t0)


im = Image.fromarray(np.uint8(cm.seismic((vol_sgz[0,:,:].T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_subvol-sgz.png'))

t0 = time.time()
vol_segy = segyio.tools.cube(os.path.join(base_path, '0.sgy'))[min_il:max_il, min_xl:max_xl, min_z:max_z]
print("segyio took", time.time() - t0)

im = Image.fromarray(np.uint8(cm.seismic((vol_segy[0,:,:].T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_subvol-sgy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((vol_segy[0,:,:]-vol_sgz[0,:,:]).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_subvol-dif.png'))
