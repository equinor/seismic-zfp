"""TEMPORARY: derive a regular prestack SEG-Y from an irregular one by zero-padding.

Usage:
    python make-regular-prestack-segy.py IN.sgy OUT.sgy [--min-fill F] [--dry-run]

1. Reads IL/XL/offset headers of every trace (one strided pass) and builds the
   presence grid on the enclosing regular IL x XL x offset lattice.
2. Optionally trims axis ends (IL, XL or offset) whose slice is less than F filled,
   so spacing stays constant on every axis. Default F=0 keeps the whole lattice.
3. Writes every (IL, XL, offset) position of the grid as a trace, in IL/XL/offset
   order, after the original textual and binary file headers. Existing traces are
   copied byte-for-byte. Missing positions get a zero trace whose header is copied
   from the nearest offset of the same gather (or the nearest gather on the inline
   if the whole gather is absent, with CDP_X/CDP_Y zeroed), with INLINE_3D,
   CROSSLINE_3D and offset set to the grid values.
"""
import os
import sys
import time

import numpy as np
import segyio

from seismic_zfp.seismicfile import SeismicFile
from seismic_zfp.conversion_utils import read_trace_header_fields
from seismic_zfp.sgzconstants import SEGY_FILE_HEADER_BYTES, SEGY_TRACE_HEADER_BYTES

in_file, out_file = sys.argv[1], sys.argv[2]
MIN_FILL = float(sys.argv[sys.argv.index('--min-fill') + 1]) if '--min-fill' in sys.argv else 0.0
DRY_RUN = '--dry-run' in sys.argv[3:]
IL, XL, OFFSET = 189, 193, 37


def axis(values):
    unique = np.unique(values)
    step = int(np.gcd.reduce(np.diff(unique))) if len(unique) > 1 else 1
    return np.arange(unique[0], unique[-1] + step, step)


t0 = time.time()
with SeismicFile.open(in_file) as segy:
    tracecount, n_samples = segy.tracecount, len(segy.samples)
    fields = read_trace_header_fields(segy, [IL, XL, OFFSET])
print(f"Read {tracecount} trace headers in {time.time() - t0:.1f}s")

il_axis, xl_axis, off_axis = axis(fields[IL]), axis(fields[XL]), axis(fields[OFFSET])
print(f"Enclosing grid: IL {il_axis[0]}..{il_axis[-1]} step {il_axis[1]-il_axis[0]} ({len(il_axis)}), "
      f"XL {xl_axis[0]}..{xl_axis[-1]} step {xl_axis[1]-xl_axis[0]} ({len(xl_axis)}), "
      f"offset {off_axis[0]}..{off_axis[-1]} step {off_axis[1]-off_axis[0]} ({len(off_axis)})")

# trace_index[il, xl, offset] = trace number, or -1 where no trace exists
il_id = (fields[IL] - il_axis[0]) // (il_axis[1] - il_axis[0])
xl_id = (fields[XL] - xl_axis[0]) // (xl_axis[1] - xl_axis[0])
off_id = (fields[OFFSET] - off_axis[0]) // (off_axis[1] - off_axis[0])
trace_index = np.full((len(il_axis), len(xl_axis), len(off_axis)), -1, dtype=np.int64)
trace_index[il_id, xl_id, off_id] = np.arange(tracecount)
present = trace_index >= 0
print(f"{present.sum()} traces on {present.size} grid positions ({100 * present.mean():.1f}% filled)")

# Optionally trim axis ends whose slices are sparsely filled, keeping constant spacing
keep = [np.ones(n, dtype=bool) for n in present.shape]
while MIN_FILL > 0:
    sub = present[np.ix_(*keep)]
    worst = None
    for ax in range(3):
        fill = sub.mean(axis=tuple(a for a in range(3) if a != ax))
        for c in (0, len(fill) - 1):
            if fill[c] < MIN_FILL and (worst is None or fill[c] < worst[0]):
                worst = (fill[c], ax, c)
    if worst is None:
        break
    _, ax, c = worst
    keep[ax][np.flatnonzero(keep[ax])[c]] = False

kept_il, kept_xl, kept_off = il_axis[keep[0]], xl_axis[keep[1]], off_axis[keep[2]]
selected = trace_index[np.ix_(*keep)]           # (n_il, n_xl, n_off) trace numbers, -1 = zero trace
n_out, n_present = selected.size, int((selected >= 0).sum())
print(f"Output grid: {len(kept_il)} IL x {len(kept_xl)} XL x {len(kept_off)} offsets = {n_out} traces, "
      f"{n_present} from input ({100 * n_present / tracecount:.1f}%), {n_out - n_present} zero-padded")
print(f"  IL {kept_il[0]}..{kept_il[-1]}, XL {kept_xl[0]}..{kept_xl[-1]}, offsets {kept_off[0]}..{kept_off[-1]}")
if DRY_RUN:
    sys.exit(0)


def be_int32(values):
    """(n,) ints -> (n, 4) big-endian bytes"""
    return np.ascontiguousarray(np.asarray(values, dtype='>i4')).view(np.uint8).reshape(-1, 4)


trace_bytes = (os.path.getsize(in_file) - SEGY_FILE_HEADER_BYTES) // tracecount
source = np.memmap(in_file, dtype=np.uint8, mode='r', offset=SEGY_FILE_HEADER_BYTES, shape=(tracecount, trace_bytes))
n_xl, n_off = len(kept_xl), len(kept_off)
t0 = time.time()
with open(in_file, 'rb') as f_in, open(out_file, 'wb') as f_out:
    f_out.write(f_in.read(SEGY_FILE_HEADER_BYTES))
    for i, il in enumerate(kept_il):
        ids = selected[i].reshape(-1)                       # (n_xl * n_off,), xl slowest
        block = np.zeros((ids.size, trace_bytes), dtype=np.uint8)
        present_rows = np.flatnonzero(ids >= 0)
        if present_rows.size:
            block[present_rows] = source[ids[present_rows]]

        missing_rows = np.flatnonzero(ids < 0)
        if missing_rows.size:
            # Header template per missing trace: nearest present offset in the same gather,
            # else the first trace of the nearest gather on the inline (with CDP_X/Y zeroed)
            present_2d = selected[i] >= 0                    # (n_xl, n_off)
            donor = np.zeros(ids.size, dtype=np.int64)
            whole_gather_absent = np.zeros(ids.size, dtype=bool)
            gathers_with_data = np.flatnonzero(present_2d.any(axis=1))
            for xl_id in range(n_xl):
                rows = slice(xl_id * n_off, (xl_id + 1) * n_off)
                offs_present = np.flatnonzero(present_2d[xl_id])
                if offs_present.size:
                    nearest = offs_present[np.abs(offs_present[:, None] - np.arange(n_off)[None, :]).argmin(axis=0)]
                    donor[rows] = selected[i, xl_id, nearest]
                else:
                    donor_xl = gathers_with_data[np.abs(gathers_with_data - xl_id).argmin()]
                    donor[rows] = selected[i, donor_xl][present_2d[donor_xl]][0]
                    whole_gather_absent[rows] = True
            block[missing_rows, 0:SEGY_TRACE_HEADER_BYTES] = source[donor[missing_rows], 0:SEGY_TRACE_HEADER_BYTES]
            zero_coords = missing_rows[whole_gather_absent[missing_rows]]
            block[zero_coords, 180:188] = 0                  # CDP_X (181), CDP_Y (185)
            block[missing_rows, IL - 1:IL + 3] = be_int32(np.full(missing_rows.size, il))
            block[missing_rows, XL - 1:XL + 3] = be_int32(kept_xl[missing_rows // n_off])
            block[missing_rows, OFFSET - 1:OFFSET + 3] = be_int32(kept_off[missing_rows % n_off])

        f_out.write(block.tobytes())
        print(f"   - {100 * (i + 1) / len(kept_il):5.1f}% written", end="\r")
print(f"\nWrote {out_file} ({os.path.getsize(out_file) / 1e9:.2f} GB) in {time.time() - t0:.1f}s")

with SeismicFile.open(out_file) as segy:
    print(f"Verification: tracecount={segy.tracecount} structured={segy.structured} "
          f"is_4d={segy.is_4d} n_offsets={segy.n_offsets}")
with segyio.open(out_file) as segy:
    print(f"  segyio: ilines {segy.ilines[0]}..{segy.ilines[-1]} ({len(segy.ilines)}), "
          f"xlines {segy.xlines[0]}..{segy.xlines[-1]} ({len(segy.xlines)}), "
          f"offsets {segy.offsets[0]}..{segy.offsets[-1]} ({len(segy.offsets)}), sorting {segy.sorting}")
    gather = segy.gather[segy.ilines[len(segy.ilines) // 2], segy.xlines[len(segy.xlines) // 2]]
    print(f"  segyio gather shape {gather.shape} (expected ({len(kept_off)}, {n_samples}))")
