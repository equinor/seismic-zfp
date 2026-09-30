"""Compute and display the fold map of a 4D (prestack) SGZ file.

Usage:
    python read-fold-map-4d.py [SGZ_FILE] [LINE_NO]

Fold is the number of traces present in each gather. Irregular prestack SEG-Y (not every offset
at every IL/XL position) is stored on its enclosing regular grid, with missing positions holding
zero samples and zero entries in the trace header arrays. So the fold map is one header array
read and a count along the offset axis, with no seismic data decompressed at all:

    inline_numbers = sgzfile.get_tracefield_values(segyio.TraceField.INLINE_3D)   # (il, xl, offset)
    fold = np.count_nonzero(inline_numbers, axis=2)                               # (il, xl)

INLINE_3D is used as the presence indicator since a live trace never has inline number 0 in a
file whose inline numbers start above 0; for a regular file the fold is n_offsets everywhere.
For comparison the gathers of inline LINE_NO are decompressed and their non-zero traces counted.
On a lossy file this over-counts: ZFP compresses in 4x4x4x4 blocks, so a little energy leaks from
live traces into the zero-filled positions sharing a block, which is why the header arrays, and
not the samples, are the right place to look. The fold map is drawn as SGZ_FILE-fold.png,
inlines down and crosslines across.
"""
import seismic_zfp
import segyio
import time
import os
import sys

from PIL import Image
import numpy as np
from matplotlib import cm

sgz_file = sys.argv[1]
LINE_NO = int(sys.argv[2])

with seismic_zfp.open(sgz_file) as sgzfile:
    print(f"{'Irregular' if sgzfile.unstructured else 'Regular'} prestack file: "
          f"{len(sgzfile.ilines)} inlines x {len(sgzfile.xlines)} crosslines x {len(sgzfile.offsets)} offsets, "
          f"{sgzfile.tracecount} traces of {len(sgzfile.ilines) * len(sgzfile.xlines) * len(sgzfile.offsets)} "
          f"grid positions")

    t0 = time.time()
    inline_numbers = sgzfile.get_tracefield_values(segyio.TraceField.INLINE_3D)
    fold = np.count_nonzero(inline_numbers, axis=2)
    print("fold map from header arrays took", time.time() - t0)
    print(f"fold: min {fold.min()}, max {fold.max()}, mean {fold.mean():.1f}, "
          f"{np.count_nonzero(fold == 0)} empty gathers, total {fold.sum()} traces")

    t0 = time.time()
    # gather[il, :] is a generator of one (n_offsets, n_samples) gather per crossline
    fold_from_data = np.array([np.count_nonzero(np.any(gather != 0, axis=1))
                               for gather in sgzfile.gather[LINE_NO, :]])
    print(f"fold of inline {LINE_NO} from decompressed gathers took", time.time() - t0)

    il_id = sgzfile.get_inline_index(LINE_NO)
    difference = fold_from_data - fold[il_id]
    print(f"inline {LINE_NO}: data-based fold minus header-array fold is {difference.min()}..{difference.max()} "
          f"per gather ({sgzfile.rate} bits per voxel; positive is leakage into zero-filled positions, "
          f"negative is genuine all-zero traces present in the file)")

out_file = os.path.splitext(sgz_file)[0] + '-fold.png'
im = Image.fromarray(np.uint8(cm.viridis(fold / fold.max()) * 255))
im.save(out_file)
print("wrote", out_file)
