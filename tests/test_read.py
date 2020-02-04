import numpy as np
import pytest
from seismic_zfp.read import *

SZ_FILE_1 = 'test_data/small_1bit.sz'
SZ_FILE_2 = 'test_data/small_2bit.sz'
SZ_FILE_4 = 'test_data/small_4bit.sz'
SZ_FILE_8 = 'test_data/small_8bit.sz'
SEGY_FILE = 'test_data/small.sgy'


def compare_inline(sz_filename, tolerance):
    for line_number in range(5):
        slice_sz = SzReader(sz_filename).read_inline(line_number)
        with segyio.open(SEGY_FILE) as segyfile:
            slice_segy = segyfile.iline[segyfile.ilines[line_number]]
        assert np.allclose(slice_sz, slice_segy, rtol=tolerance)


def test_read_inline():
    compare_inline(SZ_FILE_1, tolerance=1e-2)
    compare_inline(SZ_FILE_2, tolerance=1e-4)
    compare_inline(SZ_FILE_4, tolerance=1e-6)
    compare_inline(SZ_FILE_8, tolerance=1e-10)


def compare_crossline(sz_filename, tolerance):
    for line_number in range(5):
        slice_sz = SzReader(sz_filename).read_crossline(line_number)
        with segyio.open(SEGY_FILE) as segyfile:
            slice_segy = segyfile.xline[segyfile.xlines[line_number]]
        assert np.allclose(slice_sz, slice_segy, rtol=tolerance)


def test_read_crossline():
    compare_crossline(SZ_FILE_1, tolerance=1e-2)
    compare_crossline(SZ_FILE_2, tolerance=1e-4)
    compare_crossline(SZ_FILE_4, tolerance=1e-6)
    compare_crossline(SZ_FILE_8, tolerance=1e-10)


def compare_zslice(sz_filename, tolerance):
    for line_number in range(50):
        slice_sz = SzReader(sz_filename).read_zslice(line_number)
        with segyio.open(SEGY_FILE) as segyfile:
            slice_segy = segyfile.depth_slice[line_number]
        assert np.allclose(slice_sz, slice_segy, rtol=tolerance)


def test_read_zslice():
    compare_zslice(SZ_FILE_1, tolerance=1e-2)
    compare_zslice(SZ_FILE_2, tolerance=1e-4)
    compare_zslice(SZ_FILE_4, tolerance=1e-6)
    compare_zslice(SZ_FILE_8, tolerance=1e-10)


def compare_subvolume(sz_filename, tolerance):
    min_il, max_il = 2,  3
    min_xl, max_xl = 1,  2
    min_z,  max_z = 10, 20
    vol_sz = SzReader(sz_filename).read_subvolume(min_il=min_il, max_il=max_il,
                                                  min_xl=min_xl, max_xl=max_xl,
                                                  min_z=min_z, max_z=max_z)
    vol_segy = segyio.tools.cube(SEGY_FILE)[min_il:max_il, min_xl:max_xl, min_z:max_z]
    assert np.allclose(vol_sz, vol_segy, rtol=tolerance)


def test_read_subvolume():
    compare_subvolume(SZ_FILE_1, tolerance=1e-2)
    compare_subvolume(SZ_FILE_2, tolerance=1e-4)
    compare_subvolume(SZ_FILE_4, tolerance=1e-6)
    compare_subvolume(SZ_FILE_8, tolerance=1e-10)
