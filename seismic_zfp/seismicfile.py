import os
from enum import Enum
import warnings

import numpy as np
import segyio
import seismic_zfp

try:
    with warnings.catch_warnings():
        # pyzgy will warn us that sdglue is not available. This is expected, and safe for our purposes.
        warnings.filterwarnings("ignore", message="seismic store access is not available: No module named 'sdglue'")
        import pyzgy
except ImportError:
    pyzgy = None

try:
    import pyvds
except ImportError:
    pyvds = None


class Filetype(Enum):
    SEGY = 0
    ZGY = 10
    VDS = 30
    SGZ = 100


class SeismicFile:
    def __init__(self):
        pass

    @staticmethod
    def count_offsets_irregular(handle, max_traces=10000):
        """Number of distinct offsets in an irregular SEG-Y, judged from the leading trace headers.

        A file is prestack when some IL/XL position holds several traces with different offsets.
        The count only signals dimensionality; the offset axis itself is inferred from all traces
        when the geometry is built. Positions with INLINE_3D == CROSSLINE_3D == 0 are ignored,
        as those identify 2D data.
        """
        n = min(handle.tracecount, max_traces)
        il = handle.attributes(189)[0:n]
        xl = handle.attributes(193)[0:n]
        offset = handle.attributes(37)[0:n]
        positioned = (il != 0) | (xl != 0)
        if not np.any(positioned):
            return 1
        gathers = {}
        for i, x, o in zip(il[positioned].tolist(), xl[positioned].tolist(), offset[positioned].tolist()):
            gathers.setdefault((i, x), set()).add(o)
        if max(len(offsets) for offsets in gathers.values()) == 1:
            return 1
        return len(set(offset[positioned].tolist()))

    @staticmethod
    def open(filename, file_type=None):
        handle = None
        if file_type is None:
            ext = os.path.splitext(filename)[1].lower().strip('.')
            if ext in ['', 'sgy', 'segy']:
                # Assume no extension means SEG-Y
                file_type = Filetype.SEGY
            elif ext == 'zgy':
                file_type = Filetype.ZGY
            elif ext == 'vds':
                file_type = Filetype.VDS
            elif ext == 'sgz':
                file_type = Filetype.SGZ
            else:
                raise ValueError(f"Unknown file extension: '{ext}'")
        elif not isinstance(file_type, Filetype):
            raise ValueError("Not a valid file_type. Must be of type Filetype")

        if file_type == Filetype.SEGY:
            handle = segyio.open(filename, mode='r', strict=False)
            handle.structured = False
            handle.n_offsets = 1
            try:
                metrics = handle.xfd.cube_metrics(189, 193)
                regular_tracecount = metrics['iline_count'] * metrics['xline_count'] * metrics['offset_count']
                handle.structured = regular_tracecount == handle.tracecount
                if handle.structured:
                    handle.n_offsets = metrics['offset_count']
            except (RuntimeError, ValueError):
                pass
            if not handle.structured:
                # segyio's metrics are unreliable for irregular files, so look at the headers directly
                handle.n_offsets = SeismicFile.count_offsets_irregular(handle)
        elif file_type == Filetype.ZGY:
            if pyzgy is None:
                raise ImportError("File type requires pyzgy. Install optional dependency seismic-zfp[zgy] with pip.")
            handle = pyzgy.open(filename)
            handle.structured = True
            handle.n_offsets = 1
        elif file_type == Filetype.VDS:
            if pyvds is None:
                raise ImportError("File type requires pyvds. Install optional dependency seismic-zfp[vds] with pip.")
            handle = pyvds.open(filename)
            handle.structured = True
            handle.n_offsets = 1
        elif file_type == Filetype.SGZ:
            handle = seismic_zfp.open(filename)
            handle.structured = True
            handle.n_offsets = 1

        handle.filetype = file_type
        handle.filename = filename
        # Prestack (gather) data has an offset axis in addition to IL/XL
        handle.is_4d = handle.n_offsets > 1

        return handle
