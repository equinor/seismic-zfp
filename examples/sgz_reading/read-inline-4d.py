"""Read a few gathers along an inline from a 4D (prestack) SGZ file and display them side by side.

Usage:
    python read-inline-4d.py [FILE_ROOT] [LINE_NO] [FIRST_XL_ORDINAL] [N_GATHERS]

Reads N_GATHERS consecutive gathers along inline number LINE_NO, starting at crossline
ordinal FIRST_XL_ORDINAL, from FILE_ROOT/0.sgz through seismic-zfp's segyio emulation,
and the same gathers from FILE_ROOT/0.sgy with segyio itself. Each gather is drawn as an
(offset x time) panel, panels separated by a grey gap, as a prestack gather viewer would.
"""
import seismic_zfp
import segyio
import time
import os
import sys

from PIL import Image
import numpy as np
from matplotlib import cm

base_path = sys.argv[1]
LINE_NO = int(sys.argv[2])
FIRST_XL = int(sys.argv[3])
N_GATHERS = int(sys.argv[4])

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)
GAP = 4          # columns between gather panels
GAP_GREY = 160


def gathers_to_image(gathers, filename):
    """gathers: array (n_gathers, n_offsets, n_samples) -> side-by-side panels, time downwards"""
    n_gathers, n_offsets, n_samples = gathers.shape
    panel_width = n_offsets + GAP
    rgba = np.full((n_samples, n_gathers * panel_width, 4), GAP_GREY, dtype=np.uint8)
    rgba[:, :, 3] = 255
    for g in range(n_gathers):
        panel = cm.seismic((gathers[g].T.clip(-CLIP, CLIP) + CLIP) * SCALE) * 255
        rgba[:, g * panel_width:g * panel_width + n_offsets, :] = np.uint8(panel)
    Image.fromarray(rgba[:, :-GAP]).save(filename)


with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    xlines = sgzfile.xlines[FIRST_XL:FIRST_XL + N_GATHERS]
    t0 = time.time()
    gathers_sgz = np.stack([sgzfile.gather[LINE_NO, xl] for xl in xlines])
    print("seismic-zfp took", time.time() - t0)
    print(f"Inline {LINE_NO}, crosslines {xlines[0]}..{xlines[-1]}, "
          f"{len(sgzfile.offsets)} offsets [{sgzfile.offsets[0]}, {sgzfile.offsets[-1]}], "
          f"{len(sgzfile.samples)} samples")

gathers_to_image(gathers_sgz, os.path.join(base_path, 'out_gathers-sgz.png'))

with segyio.open(os.path.join(base_path, '0.sgy')) as segyfile:
    t0 = time.time()
    # segyio <= 1.9.14 drops negative offsets from gather[il, xl] (offset labels are treated as
    # positional slice indices), so ask for each offset explicitly. Fixed in segyio PR #663:
    # once that is released, this is simply segyfile.gather[LINE_NO, xl] as for seismic-zfp above.
    gathers_sgy = np.stack([np.stack([segyfile.gather[LINE_NO, xl, offset] for offset in segyfile.offsets])
                            for xl in xlines])
    print("segyio took", time.time() - t0)

gathers_to_image(gathers_sgy, os.path.join(base_path, 'out_gathers-sgy.png'))
gathers_to_image(gathers_sgy - gathers_sgz, os.path.join(base_path, 'out_gathers-dif.png'))
