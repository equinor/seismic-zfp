"""Open a 4D (prestack) SGZ file as an xarray Dataset and select by coordinate labels.

Usage:
    python read-as-xarray-4d.py [FILE_ROOT] [INLINE] [CROSSLINE] [OFFSET]

e.g.

    python read-as-xarray-4d.py C:\\seismic\\4d 10101 2196 -1050

Opens FILE_ROOT/0.sgz with xr.open_dataset (seismic-zfp registers itself as an xarray backend
entrypoint, so no engine argument is needed). The Dataset has one variable, data, with dimensions
(il, xl, offset, z) and coordinates holding the line numbers, offset values and sample times, so
.sel() works in the same coordinate units as segyio's gather and iline accessors. Reads are lazy:
nothing is decompressed until .to_numpy() (or a reduction) is called, and only the smallest
block of the file covering the selection is read.

Three selections are made and compared against seismic-zfp's segyio emulation:
  - a single gather at (INLINE, CROSSLINE), drawn as out_xr-gather.png
  - the common-offset section at OFFSET along INLINE, drawn as out_xr-offset.png
  - a crude stack: the mean over the offset axis along INLINE, drawn as out_xr-stack.png
"""
import xarray as xr
import seismic_zfp
import time
import os
import sys

from PIL import Image
import numpy as np
from matplotlib import cm

base_path = sys.argv[1]
INLINE, CROSSLINE, OFFSET = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

CLIP = 0.2
SCALE = 1.0/(2.0*CLIP)


def save_image(section, filename):
    """section: array (traces, samples) -> image with time downwards"""
    im = Image.fromarray(np.uint8(cm.seismic((section.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
    im.save(os.path.join(base_path, filename))


with xr.open_dataset(os.path.join(base_path, '0.sgz')) as dataset:
    print(dataset)

    t0 = time.time()
    gather = dataset.data.sel(il=INLINE, xl=CROSSLINE).to_numpy()
    print("xarray gather took", time.time() - t0)

    t0 = time.time()
    common_offset = dataset.data.sel(il=INLINE, offset=OFFSET).to_numpy()
    print("xarray common-offset section took", time.time() - t0)

    t0 = time.time()
    stack = dataset.data.sel(il=INLINE).mean(dim="offset").to_numpy()
    print("xarray offset-stack took", time.time() - t0)

save_image(gather, 'out_xr-gather.png')
save_image(common_offset, 'out_xr-offset.png')
save_image(stack, 'out_xr-stack.png')

with seismic_zfp.open(os.path.join(base_path, '0.sgz')) as sgzfile:
    t0 = time.time()
    gather_sgz = sgzfile.gather[INLINE, CROSSLINE]
    common_offset_sgz = sgzfile.iline[INLINE, OFFSET]
    # As in segyio, iline[il] alone is the first offset only; stack over all offsets explicitly
    stack_sgz = np.mean([sgzfile.iline[INLINE, offset] for offset in sgzfile.offsets], axis=0)
    print("seismic-zfp emulator took", time.time() - t0)

print("gather identical:", np.array_equal(gather, gather_sgz))
print("common-offset section identical:", np.array_equal(common_offset, common_offset_sgz))
print("stack close:", np.allclose(stack, stack_sgz))
