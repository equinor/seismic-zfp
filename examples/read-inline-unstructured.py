from seismic_zfp.read import SgzReader
import segyio
import time
import os
import sys

from PIL import Image
import numpy as np
from matplotlib import cm

base_path = sys.argv[1]
LINE_NO = int(sys.argv[2])

CLIP = 200
SCALE = 1.0/(2.0*CLIP)


with segyio.open(os.path.join(base_path, '0.sgy'), strict=False) as segyfile:
    t0 = time.time()
    xl_ids = [h[193] for h in segyfile.header if h[189] == LINE_NO]
    xl_step = (max(xl_ids) - min(xl_ids)) // (len(xl_ids) - 1)
    trace_ids = [i for i, h in enumerate(segyfile.header) if h[189] == LINE_NO]
    slice_segy = np.zeros(((max(xl_ids) - min(xl_ids))//xl_step + 1, len(segyfile.samples)))
    for i, trace_id in enumerate(trace_ids):
        slice_segy[i, :] = segyfile.trace[trace_id]
    print("segyio took", time.time() - t0)

with SgzReader(os.path.join(base_path, '0.sgz')) as reader:
    t0 = time.time()
    slice_sgz = reader.read_inline_number(LINE_NO)
    print("SgzReader took", time.time() - t0)
    slice_sgz = slice_sgz[(min(xl_ids) - reader.xlines[0])//(reader.xlines[1]-reader.xlines[0]): (max(xl_ids) - reader.xlines[0])//(reader.xlines[1]-reader.xlines[0]) + 1]

im = Image.fromarray(np.uint8(cm.seismic((slice_sgz.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-sgz.png'))

im = Image.fromarray(np.uint8(cm.seismic((slice_segy.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-sgy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((slice_segy-slice_sgz).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_inline-dif.png'))

