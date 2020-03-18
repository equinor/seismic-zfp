import numpy as np
import pytest
import seismic_zfp
import segyio

SGZ_FILE_1 = 'test_data/small_1bit.sgz'
SGZ_FILE_2 = 'test_data/small_2bit.sgz'
SGZ_FILE_4 = 'test_data/small_4bit.sgz'
SGZ_FILE_8 = 'test_data/small_8bit.sgz'
SGY_FILE = 'test_data/small.sgy'


def compare_inline(sgz_filename, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(SGY_FILE) as segyfile:
            for line_number in range(5):
                slice_sgz = sgzfile.iline[sgzfile.ilines[line_number]]
                slice_segy = segyfile.iline[segyfile.ilines[line_number]]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def test_inline_accessor():
    compare_inline(SGZ_FILE_1, tolerance=1e-2)
    compare_inline(SGZ_FILE_2, tolerance=1e-4)
    compare_inline(SGZ_FILE_4, tolerance=1e-6)
    compare_inline(SGZ_FILE_8, tolerance=1e-10)


def compare_crossline(sgz_filename, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(SGY_FILE) as segyfile:
            for line_number in range(5):
                slice_sgz = sgzfile.xline[sgzfile.xlines[line_number]]
                slice_segy = segyfile.xline[segyfile.xlines[line_number]]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def test_crossline_accessor():
    compare_crossline(SGZ_FILE_1, tolerance=1e-2)
    compare_crossline(SGZ_FILE_2, tolerance=1e-4)
    compare_crossline(SGZ_FILE_4, tolerance=1e-6)
    compare_crossline(SGZ_FILE_8, tolerance=1e-10)


def compare_zslice(sgz_filename, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(SGY_FILE) as segyfile:
            for line_number in range(50):
                slice_sgz = sgzfile.depth_slice[line_number]
                slice_segy = segyfile.depth_slice[line_number]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def test_zslice_accessor():
    compare_zslice(SGZ_FILE_1, tolerance=1e-2)
    compare_zslice(SGZ_FILE_2, tolerance=1e-4)
    compare_zslice(SGZ_FILE_4, tolerance=1e-6)
    compare_zslice(SGZ_FILE_8, tolerance=1e-10)
