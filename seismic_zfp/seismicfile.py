import os
from enum import Enum

import segyio

try:
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
            else:
                raise ValueError("Unknown file extension: '{}'".format(ext))
        elif not isinstance(file_type, Filetype):
                raise ValueError("Not a valid file_type. Must be of type Filetype")

        if file_type == Filetype.SEGY:
            handle = segyio.open(filename, mode='r', strict=False)
        elif file_type == Filetype.ZGY:
            if pyzgy is None:
                raise ImportError("File type requires pyzgy. Install optional dependency seismic-zfp[zgy] with pip.")
            handle = pyzgy.open(filename)
        elif file_type == Filetype.VDS:
            if pyvds is None:
                raise ImportError("File type requires pyvds. Install optional dependency seismic-zfp[vds] with pip.")
            handle = pyvds.open(filename)

        handle.filetype = file_type

        return handle
