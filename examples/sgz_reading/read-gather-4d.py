"""Read a single gather from a 4D (prestack) SGZ file and display it.

Usage:
    python read-gather-4d.py [FILE_ROOT] [INLINE] [CROSSLINE]

Reads the gather at (INLINE, CROSSLINE) from FILE_ROOT/0.sgz three ways:
  - sgzfile.gather[il, xl] through seismic-zfp's segyio emulation, all offsets at once
  - sgzfile.gather[il, xl, offset] trace by trace, as segyio itself is used below
  - SgzReader.read_gather_number(il, xl), the underlying call
and the same gather from FILE_ROOT/0.sgy with segyio. The gather is drawn as an (offset x time)
panel, offsets across and time downwards, along with the seismic-zfp / segyio difference.
"""
import seismic_zfp
from seismic_zfp.read import SgzReader
import segyio
import time
import os
import sys

from PIL import Image
import numpy as np
from matplotlib import cm

base_path = sys.argv[1]
INLINE, CROSSLINE = int(sys.argv[2]), int(sys.argv[3])

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)


def save_image(gather, filename):
    """gather: array (n_offsets, n_samples) -> image with offsets across and time downwards"""
    im = Image.fromarray(np.uint8(cm.seismic((gather.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
    im.save(os.path.join(base_path, filename))


with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    t0 = time.time()
    gather_sgz = sgzfile.gather[INLINE, CROSSLINE]
    print("seismic-zfp gather took", time.time() - t0)

    t0 = time.time()
    gather_by_offset = np.stack([sgzfile.gather[INLINE, CROSSLINE, offset] for offset in sgzfile.offsets])
    print("seismic-zfp gather, one offset at a time, took", time.time() - t0)

    offsets = sgzfile.offsets
    print(f"Gather at inline {INLINE}, crossline {CROSSLINE}: {len(offsets)} offsets "
          f"[{offsets[0]}, {offsets[-1]}], {gather_sgz.shape[1]} samples")

with SgzReader(os.path.join(base_path, '0.sgz')) as reader:
    t0 = time.time()
    gather_reader = reader.read_gather_number(INLINE, CROSSLINE)
    print("SgzReader.read_gather_number took", time.time() - t0)

print("emulator and reader identical:", np.array_equal(gather_sgz, gather_reader),
      np.array_equal(gather_sgz, gather_by_offset))
save_image(gather_sgz, 'out_gather-sgz.png')

with segyio.open(os.path.join(base_path, '0.sgy')) as segyfile:
    t0 = time.time()
    # segyio <= 1.9.14 drops negative offsets from gather[il, xl] (offset labels are treated as
    # positional slice indices), so ask for each offset explicitly. Fixed in segyio PR #666
    # (https://github.com/equinor/segyio/pull/666): once released, this is simply gather[il, xl].
    gather_sgy = np.stack([segyfile.gather[INLINE, CROSSLINE, offset] for offset in segyfile.offsets])
    print("segyio took", time.time() - t0)

print("max abs difference to segyio (compression error):",
      np.abs(gather_sgz - gather_sgy).max(), "of", np.abs(gather_sgy).max())
save_image(gather_sgy, 'out_gather-sgy.png')
save_image(gather_sgy - gather_sgz, 'out_gather-dif.png')
