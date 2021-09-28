import os
from enum import Enum

import pyvds
import zgyio
import segyio


class Filetype(Enum):
    SEGY = 0
    ZGY = 10
    VDS = 30


class SeismicFile:

    @staticmethod
    def open(filename, filetype=None):
        if filetype is None:
            ext = os.path.splitext(filename)[1].lower().strip('.')
        else:
            ext = filetype

        # Assume no extension means SEG-Y
        if ext in ['', 'sgy', 'segy']:
            handle = segyio.open(filename, mode='r', strict=False)
            handle.filetype = Filetype.SEGY
        elif ext == 'zgy':
            handle = zgyio.open(filename)
            handle.filetype = Filetype.ZGY
        elif ext == 'vds':
            handle = pyvds.open(filename)
            handle.filetype = Filetype.VDS
        else:
            raise ValueError("Unknown file extension: '{}'".format(ext))

        return handle
