import os
import sys

import seismic_zfp
import segyio

base_path = sys.argv[1]

with segyio.open(os.path.join(base_path, '0.segy')) as segyfile:
    print(segyfile.bin)


with seismic_zfp.open(os.path.join(base_path, '0.sz')) as szfile:
    print(szfile.bin)
