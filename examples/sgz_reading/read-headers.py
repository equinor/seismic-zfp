import os
import sys

import seismic_zfp
import segyio

base_path = sys.argv[1]

with segyio.open(os.path.join(base_path, '0.sgy')) as segyfile:
    print(segyfile.header[100])

with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    print(sgzfile.header[100])
