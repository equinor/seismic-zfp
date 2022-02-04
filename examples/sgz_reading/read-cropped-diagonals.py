from seismic_zfp.read import SgzReader
import numpy as np
import time
import os
import sys

base_path = sys.argv[1]
LINE_NO = int(sys.argv[2])
N_LINES = 50

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)

slices_sgz, slices_sgz_crop = {}, {}


reader = SgzReader(os.path.join(base_path, '0.sgz'), preload=True)
t0 = time.perf_counter()
for i, line_number in enumerate(range(LINE_NO, LINE_NO+N_LINES)):
    slices_sgz_crop[i] = reader.read_anticorrelated_diagonal(line_number,
                                                         min_ad_idx=32, max_ad_idx=128,
                                                         min_sample_idx=400, max_sample_idx=528)
print(f'Crop while reading took {time.perf_counter() - t0}')


reader = SgzReader(os.path.join(base_path, '0.sgz'), preload=True)
t0 = time.perf_counter()
for i, line_number in enumerate(range(LINE_NO, LINE_NO+N_LINES)):
    slices_sgz[i] = reader.read_anticorrelated_diagonal(line_number)[32:128, 400:528]
print(f'Read then crop took {time.perf_counter() - t0}')


assert all(np.array_equal(slices_sgz[i], slices_sgz_crop[i]) for i in range(N_LINES))
