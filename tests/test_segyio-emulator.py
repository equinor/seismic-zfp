import numpy as np
import pytest
import seismic_zfp
import segyio

SGZ_FILE_1 = 'test_data/small_1bit.sgz'
SGZ_FILE_2 = 'test_data/small_2bit.sgz'
SGZ_FILE_4 = 'test_data/small_4bit.sgz'
SGZ_FILE_8 = 'test_data/small_8bit.sgz'
SGY_FILE = 'test_data/small.sgy'

SGZ_FILE_DEC_8 = 'test_data/small-dec_8bit.sgz'
SGY_FILE_DEC = 'test_data/small-dec.sgy'


def compare_inline_ordinal(sgz_filename, sgy_filename, lines_to_test, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_ordinal in lines_to_test:
                slice_segy = segyfile.iline[segyfile.ilines[line_ordinal]]
                slice_sgz = sgzfile.iline[sgzfile.ilines[line_ordinal]]
                print(slice_segy)
                print(slice_sgz)
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def compare_inline_number(sgz_filename, sgy_filename, lines_to_test, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_number in lines_to_test:
                slice_segy = segyfile.iline[line_number]
                slice_sgz = sgzfile.iline[line_number]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def test_inline_accessor():
    compare_inline_ordinal(SGZ_FILE_1, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-2)
    compare_inline_ordinal(SGZ_FILE_2, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-4)
    compare_inline_ordinal(SGZ_FILE_4, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-6)
    compare_inline_ordinal(SGZ_FILE_8, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-10)
    compare_inline_ordinal(SGZ_FILE_DEC_8, SGY_FILE_DEC, [0, 1, 2], tolerance=1e-6)

    compare_inline_number(SGZ_FILE_1, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-2)
    compare_inline_number(SGZ_FILE_2, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-4)
    compare_inline_number(SGZ_FILE_4, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-6)
    compare_inline_number(SGZ_FILE_8, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-10)
    compare_inline_number(SGZ_FILE_DEC_8, SGY_FILE_DEC, [1, 3, 5], tolerance=1e-6)


def compare_crossline_ordinal(sgz_filename, sgy_filename, lines_to_test, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_ordinal in lines_to_test:
                slice_segy = segyfile.xline[segyfile.xlines[line_ordinal]]
                slice_sgz = sgzfile.xline[sgzfile.xlines[line_ordinal]]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def compare_crossline_number(sgz_filename, sgy_filename, lines_to_test, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_number in lines_to_test:
                slice_segy = segyfile.xline[line_number]
                slice_sgz = sgzfile.xline[line_number]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def test_crossline_accessor():
    compare_crossline_ordinal(SGZ_FILE_1, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-2)
    compare_crossline_ordinal(SGZ_FILE_2, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-4)
    compare_crossline_ordinal(SGZ_FILE_4, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-6)
    compare_crossline_ordinal(SGZ_FILE_8, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-10)
    compare_crossline_ordinal(SGZ_FILE_DEC_8, SGY_FILE_DEC, [0, 1, 2], tolerance=1e-6)

    compare_crossline_number(SGZ_FILE_1, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-2)
    compare_crossline_number(SGZ_FILE_2, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-4)
    compare_crossline_number(SGZ_FILE_4, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-6)
    compare_crossline_number(SGZ_FILE_8, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-10)
    compare_crossline_number(SGZ_FILE_DEC_8, SGY_FILE_DEC, [20, 22, 24], tolerance=1e-6)


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
