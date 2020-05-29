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
    il_ids = [h[189] for h in segyfile.header if h[193] == LINE_NO]
    trace_ids = [i for i, h in enumerate(segyfile.header) if h[193] == LINE_NO]
    slice_segy = np.zeros((max(il_ids) - min(il_ids) + 1, len(segyfile.samples)))
    for i, trace_id in enumerate(trace_ids):
        slice_segy[il_ids[i] - min(il_ids), :] = segyfile.trace[trace_id]
    print("segyio took", time.time() - t0)

with SgzReader(os.path.join(base_path, '0.sgz')) as reader:
    t0 = time.time()
    slice_sgz = reader.read_crossline(LINE_NO-reader.xlines[0])
    print("SgzReader took", time.time() - t0)
    slice_sgz = slice_sgz[min(il_ids) - reader.xlines[0]: max(il_ids) - reader.ilines[0] + 1]

im = Image.fromarray(np.uint8(cm.seismic((slice_sgz.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_crossline-sgz.png'))

im = Image.fromarray(np.uint8(cm.seismic((slice_segy.T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_crossline-sgy.png'))

im = Image.fromarray(np.uint8(cm.seismic(((slice_segy-slice_sgz).T.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(base_path, 'out_crossline-dif.png'))

