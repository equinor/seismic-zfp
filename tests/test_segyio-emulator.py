import numpy as np
import pytest
import seismic_zfp
import segyio

SZ_FILE_1 = 'test_data/small_1bit.sz'
SZ_FILE_2 = 'test_data/small_2bit.sz'
SZ_FILE_4 = 'test_data/small_4bit.sz'
SZ_FILE_8 = 'test_data/small_8bit.sz'
SEGY_FILE = 'test_data/small.sgy'


def compare_inline(sz_filename, tolerance):
    with seismic_zfp.open(sz_filename) as szfile:
        with segyio.open(SEGY_FILE) as segyfile:
            for line_number in range(5):
                slice_sz = szfile.iline[szfile.ilines[line_number]]
                slice_segy = segyfile.iline[segyfile.ilines[line_number]]
                assert np.allclose(slice_sz, slice_segy, rtol=tolerance)


def test_inline_accessor():
    compare_inline(SZ_FILE_1, tolerance=1e-2)
    compare_inline(SZ_FILE_2, tolerance=1e-4)
    compare_inline(SZ_FILE_4, tolerance=1e-6)
    compare_inline(SZ_FILE_8, tolerance=1e-10)


def compare_crossline(sz_filename, tolerance):
    with seismic_zfp.open(sz_filename) as szfile:
        with segyio.open(SEGY_FILE) as segyfile:
            for line_number in range(5):
                slice_sz = szfile.xline[szfile.xlines[line_number]]
                slice_segy = segyfile.xline[segyfile.xlines[line_number]]
                assert np.allclose(slice_sz, slice_segy, rtol=tolerance)


def test_crossline_accessor():
    compare_crossline(SZ_FILE_1, tolerance=1e-2)
    compare_crossline(SZ_FILE_2, tolerance=1e-4)
    compare_crossline(SZ_FILE_4, tolerance=1e-6)
    compare_crossline(SZ_FILE_8, tolerance=1e-10)


def compare_zslice(sz_filename, tolerance):
    with seismic_zfp.open(sz_filename) as szfile:
        with segyio.open(SEGY_FILE) as segyfile:
            for line_number in range(50):
                slice_sz = szfile.depth_slice[line_number]
                slice_segy = segyfile.depth_slice[line_number]
                assert np.allclose(slice_sz, slice_segy, rtol=tolerance)


def test_zslice_accessor():
    compare_zslice(SZ_FILE_1, tolerance=1e-2)
    compare_zslice(SZ_FILE_2, tolerance=1e-4)
    compare_zslice(SZ_FILE_4, tolerance=1e-6)
    compare_zslice(SZ_FILE_8, tolerance=1e-10)
