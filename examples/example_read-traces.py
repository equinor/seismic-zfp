import os
import sys

import seismic_zfp
import segyio

import matplotlib.pyplot as plt

base_path = sys.argv[1]

with segyio.open(os.path.join(base_path, '0.sgy')) as segyfile:
    segy_trace = segyfile.trace[100]

with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    sgz_trace = sgzfile.trace[100]

plt.plot(segy_trace)
plt.plot(sgz_trace)
plt.savefig(os.path.join(base_path, 'out_sgz-trace-accessor.png'))