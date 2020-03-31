from seismic_zfp.read import SgzReader
import time
import os
import sys

base_path = sys.argv[1]

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)

with SgzReader(os.path.join(base_path, '0.sgz'), preload=True) as reader:
    t0 = time.time()
    for i in range(reader.n_ilines):
        slice_sgz = reader.read_inline(i)
    print("SgzReader (with preloading) took", time.time() - t0)

with SgzReader(os.path.join(base_path, '0.sgz'), preload=False) as reader:
    t0 = time.time()
    for i in range(reader.n_ilines):
        slice_sgz = reader.read_inline(i)
    print("SgzReader (without preloading) took", time.time() - t0)

