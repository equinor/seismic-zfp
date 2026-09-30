"""Read common-offset sections from a 4D (prestack) SGZ file and display them.

Usage:
    python read-common-offset-4d.py [FILE_ROOT] [LINE_NO] [OFFSET]

A common-offset section is all traces along a line which share one offset value. Reads from
FILE_ROOT/0.sgz through seismic-zfp's segyio emulation:
  - sgzfile.iline[LINE_NO, OFFSET]      inline section at OFFSET, (n_xlines, n_samples)
  - sgzfile.iline[LINE_NO]              as in segyio, the line alone means the *first* offset
  - sgzfile.iline[LINE_NO, o1:o2:step]  a generator, one section per offset in the slice
and the same sections from FILE_ROOT/0.sgy with segyio. The OFFSET and first-offset sections are
drawn as (crossline x time) images, along with the seismic-zfp / segyio difference.
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
LINE_NO, OFFSET = int(sys.argv[2]), int(sys.argv[3])

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)


def save_image(section, filename):
    """section: array (n_traces, n_samples) -> image with time downwards"""
    im = Image.fromarray(np.uint8(cm.seismic((section.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
    im.save(os.path.join(base_path, filename))


with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    offsets = sgzfile.offsets
    offset_step = offsets[1] - offsets[0]

    t0 = time.time()
    section_sgz = sgzfile.iline[LINE_NO, OFFSET]
    print(f"seismic-zfp common-offset section at offset {OFFSET} took", time.time() - t0)

    t0 = time.time()
    first_offset_sgz = sgzfile.iline[LINE_NO]
    print(f"seismic-zfp iline[{LINE_NO}] (first offset, {offsets[0]}) took", time.time() - t0)
    print("iline[line] is iline[line, first offset]:",
          np.array_equal(first_offset_sgz, sgzfile.iline[LINE_NO, offsets[0]]))

    # Every 10th offset from OFFSET onwards, as a generator of sections
    t0 = time.time()
    every_tenth = list(sgzfile.iline[LINE_NO, OFFSET::10 * offset_step])
    print(f"seismic-zfp {len(every_tenth)} sections at every 10th offset took", time.time() - t0)
    print("first of those is the OFFSET section:", np.array_equal(every_tenth[0], section_sgz))

save_image(section_sgz, 'out_common-offset-sgz.png')
save_image(first_offset_sgz, 'out_first-offset-sgz.png')

with segyio.open(os.path.join(base_path, '0.sgy')) as segyfile:
    t0 = time.time()
    section_sgy = segyfile.iline[LINE_NO, OFFSET]
    first_offset_sgy = segyfile.iline[LINE_NO]
    print("segyio took", time.time() - t0)

print("max abs difference to segyio (compression error), common-offset section:",
      np.abs(section_sgz - section_sgy).max(), "of", np.abs(section_sgy).max())
print("max abs difference to segyio (compression error), first-offset section:",
      np.abs(first_offset_sgz - first_offset_sgy).max(), "of", np.abs(first_offset_sgy).max())
save_image(section_sgy, 'out_common-offset-sgy.png')
save_image(section_sgy - section_sgz, 'out_common-offset-dif.png')
