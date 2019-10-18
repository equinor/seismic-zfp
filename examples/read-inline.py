from seismic_zfp.read import SzReader
import segyio
import time
import os
import sys

from matplotlib import pyplot as plt

base_path = sys.argv[1]
LINE_NO = int(sys.argv[2])

t0 = time.time()
reader = SzReader(os.path.join(base_path, '0.sz'), 901, 605, 385, 8)
slice_sz = reader.read_inline(LINE_NO)
print("SzReader took", time.time() - t0)

plt.imsave(os.path.join(base_path, 'out_inline-sz.png'), slice_sz, cmap='seismic')

t0 = time.time()
with segyio.open(os.path.join(base_path, '0.segy')) as segyfile:
    slice_segy = segyfile.iline[segyfile.ilines[LINE_NO]]
print("segyio took", time.time() - t0)

plt.imsave(os.path.join(base_path, 'out_inline-segy.png'), slice_segy, cmap='seismic')
