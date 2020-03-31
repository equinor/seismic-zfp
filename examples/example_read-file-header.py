import os
import sys

import seismic_zfp
import segyio

base_path = sys.argv[1]

with segyio.open(os.path.join(base_path, '0.sgy')) as segyfile:
    print(segyfile.bin)
    print(segyfile.text[0])


with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    print(sgzfile.bin)
    print(sgzfile.text[0])
