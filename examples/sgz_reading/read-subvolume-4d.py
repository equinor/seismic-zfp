"""Read a sub-volume from a 4D (prestack) SGZ file, restricted on all four axes, and display it.

Usage:
    python read-subvolume-4d.py [FILE_ROOT] [IL_SLICE] [XL_SLICE] [OFFSET_SLICE] [SAMPLE_SLICE]

Slices are start:stop[:step] in coordinate units (line numbers, offset values, sample times),
stop exclusive, e.g.

    python read-subvolume-4d.py C:\\seismic\\4d 10101:10105 2196:2200 -1050:1050:300 400:600

Reads FILE_ROOT/0.sgz through seismic-zfp's segyio emulation, and the same traces from
FILE_ROOT/0.sgy with segyio for comparison. Unlike gather, a sample range and a subset of the
offsets may be requested. The gathers of the first inline are drawn side by side, offset
across and time down, with a grey gap between crossline positions.
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


def parse_slice(text):
    parts = [None if p == '' else int(p) for p in text.split(':')]
    return slice(*parts)


il_slice, xl_slice, off_slice, z_slice = [parse_slice(arg) for arg in sys.argv[2:6]]

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


def selected(coords, s):
    """The coordinates picked out by a coordinate-unit slice, as the subvolume accessor interprets it"""
    step = coords[1] - coords[0]
    start = coords[0] if s.start is None else s.start
    stop = coords[-1] + step if s.stop is None else s.stop
    return np.arange(start, stop, step if s.step is None else s.step)


with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    t0 = time.time()
    vol_sgz = sgzfile.subvolume[il_slice, xl_slice, off_slice, z_slice]
    print("seismic-zfp took", time.time() - t0)
    ilines, xlines = selected(sgzfile.ilines, il_slice), selected(sgzfile.xlines, xl_slice)
    offsets, samples = selected(sgzfile.offsets, off_slice), selected(sgzfile.samples, z_slice)
    print(f"Sub-volume {vol_sgz.shape}: inlines {ilines[0]}..{ilines[-1]}, crosslines {xlines[0]}..{xlines[-1]}, "
          f"offsets {offsets[0]}..{offsets[-1]}, samples {samples[0]}..{samples[-1]}")

gathers_to_image(vol_sgz[0], os.path.join(base_path, 'out_subvol-4d-sgz.png'))

with segyio.open(os.path.join(base_path, '0.sgy')) as segyfile:
    t0 = time.time()
    # segyio <= 1.9.14 drops negative offsets from gather[il, xl, offset_slice] (offset labels are
    # treated as positional slice indices), so ask for each offset explicitly. Fixed in segyio PR #663.
    sample_ids = np.flatnonzero(np.isin(segyfile.samples, samples))
    vol_sgy = np.stack([np.stack([np.stack([segyfile.gather[il, xl, offset][sample_ids] for offset in offsets])
                                  for xl in xlines])
                        for il in ilines])
    print("segyio took", time.time() - t0)

gathers_to_image(vol_sgy[0], os.path.join(base_path, 'out_subvol-4d-sgy.png'))
gathers_to_image(vol_sgy[0] - vol_sgz[0], os.path.join(base_path, 'out_subvol-4d-dif.png'))
