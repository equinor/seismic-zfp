import os
import sys
import numpy as np
from PIL import Image
from matplotlib import cm

import segyio
import seismic_zfp

base_path = sys.argv[1]

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)

with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgz_file:
    dt_sgz = seismic_zfp.tools.dt(sgz_file)

with segyio.open(os.path.join(base_path, '0.sgy')) as sgy_file:
    dt_sgy = segyio.tools.dt(sgy_file)

print(f'dt = {dt_sgz} (SGZ), {dt_sgy} (SGY)')

cube_sgz = seismic_zfp.tools.cube(os.path.join(base_path, '0.sgz'))
cube_sgy = segyio.tools.cube(os.path.join(base_path, '0.sgy'))

islice_sgz = cube_sgz[0,:,:]
islice_sgy = cube_sgy[0,:,:]

im = Image.fromarray(np.uint8(cm.seismic((islice_sgz.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_inline-sgz.png'))

im = Image.fromarray(np.uint8(cm.seismic((islice_sgy.T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_inline-sgy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((islice_sgz-islice_sgy).T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255))
im.save(os.path.join(base_path, 'out_inline-sgz_sgz-diff.png'))
