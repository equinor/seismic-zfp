import os
from enum import Enum
import warnings

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

    @staticmethod
    def open(filename, file_type=None):
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
                raise ValueError("Unknown file extension: '{}'".format(ext))
        elif not isinstance(file_type, Filetype):
                raise ValueError("Not a valid file_type. Must be of type Filetype")

        if file_type == Filetype.SEGY:
            handle = segyio.open(filename, mode='r', strict=False)
            metrics = handle.xfd.cube_metrics(189, 193)
            handle.structured = (metrics['iline_count'] * metrics['xline_count']) == handle.tracecount
        elif file_type == Filetype.ZGY:
            if pyzgy is None:
                raise ImportError("File type requires pyzgy. Install optional dependency seismic-zfp[zgy] with pip.")
            handle = pyzgy.open(filename)
            handle.structured = True
        elif file_type == Filetype.VDS:
            if pyvds is None:
                raise ImportError("File type requires pyvds. Install optional dependency seismic-zfp[vds] with pip.")
            handle = pyvds.open(filename)
            handle.structured = True
        elif file_type == Filetype.SGZ:
            handle = seismic_zfp.open(filename)
            handle.structured = True

        handle.filetype = file_type

        return handle
