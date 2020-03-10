from seismic_zfp.read import SzReader
import time
import os
import sys

base_path = sys.argv[1]

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)

with SzReader(os.path.join(base_path, 'psdn11_TbsdmF_full_w_AGC_Nov11.sz'), preload=True) as reader:
    t0 = time.time()
    for i in range(reader.n_ilines):
        slice_sz = reader.read_inline(i)
    print("SzReader (with preloading) took", time.time() - t0)

with SzReader(os.path.join(base_path, 'psdn11_TbsdmF_full_w_AGC_Nov11.sz'), preload=False) as reader:
    t0 = time.time()
    for i in range(reader.n_ilines):
        slice_sz = reader.read_inline(i)
    print("SzReader (without preloading) took", time.time() - t0)

