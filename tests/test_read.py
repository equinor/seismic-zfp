import numpy as np
import pytest
from seismic_zfp.read import *
from seismic_zfp.utils import get_correlated_diagonal_length, get_anticorrelated_diagonal_length, InferredGeometry
import itertools

SGZ_FILE_025 = 'test_data/small_025bit.sgz'
SGZ_FILE_05 = 'test_data/small_05bit.sgz'
SGZ_FILE_1 = 'test_data/small_1bit.sgz'
SGZ_FILE_2 = 'test_data/small_2bit.sgz'
SGZ_FILE_4 = 'test_data/small_4bit.sgz'
SGZ_FILE_8 = 'test_data/small_8bit.sgz'
SGZ_FILE_2_64x64 = 'test_data/small_2bit-64x64.sgz'
SGZ_FILE_8_8x8 = 'test_data/small_8bit-8x8.sgz'
SGY_FILE = 'test_data/small.sgy'

SGY_FILE_IRREG = 'test_data/small-irregular.sgy'
SGZ_FILE_IRREG = 'test_data/small-irregular.sgz'

SGZ_FILE_DEC_8 = 'test_data/small-dec_8bit.sgz'
SGY_FILE_DEC = 'test_data/small-dec.sgy'

SGZ_SGY_FILE_PAIRS = [('test_data/padding/padding_{}x{}.sgz'.format(n, m),
                       'test_data/padding/padding_{}x{}.sgy'.format(n, m))
                      for n, m in itertools.product([5, 6, 7, 8], [5, 6, 7, 8])]


def test_read_ilines_list():
    reader = SgzReader(SGZ_FILE_1)
    with segyio.open(SGY_FILE) as sgyfile:
        assert np.all(reader.ilines == sgyfile.ilines)


def test_read_xlines_list():
    reader = SgzReader(SGZ_FILE_1)
    with segyio.open(SGY_FILE) as sgyfile:
        assert np.all(reader.xlines == sgyfile.xlines)


def test_read_samples_list():
    reader = SgzReader(SGZ_FILE_1)
    with segyio.open(SGY_FILE) as sgyfile:
        assert np.all(reader.zslices == sgyfile.samples)


def test_read_trace_header():
    reader = SgzReader(SGZ_FILE_1)
    with segyio.open(SGY_FILE) as sgyfile:
        for trace_number in range(25):
            sgz_header = reader.gen_trace_header(trace_number)
            sgy_header = sgyfile.header[trace_number]
            assert sgz_header == sgy_header


def test_read_trace_header_preload():
    reader = SgzReader(SGZ_FILE_1)
    with segyio.open(SGY_FILE) as sgyfile:
        for trace_number in range(25):
            sgz_header = reader.gen_trace_header(trace_number, load_all_headers=True)
            sgy_header = sgyfile.header[trace_number]
            assert sgz_header == sgy_header


def compare_inline(sgz_filename, sgy_filename, lines, tolerance):
    with segyio.open(sgy_filename) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in range(lines):
                slice_sgz = reader.read_inline(line_number)
                slice_segy = segyfile.iline[segyfile.ilines[line_number]]
            assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def compare_inline_unstructured(sgz_filename, sgy_filename, tolerance):
    reader = SgzReader(sgz_filename)
    with segyio.open(sgy_filename, ignore_geometry=True) as segyfile:
        geom = InferredGeometry({(h[189], h[193]): i for i, h in enumerate(segyfile.header)})
        for line_number in geom.ilines:
            slice_sgz = reader.read_inline_number(line_number)
            slice_segy = np.zeros((len(geom.xlines), len(segyfile.samples)))
            for trace, header in zip(segyfile.trace, segyfile.header):
                if header[189] == line_number:
                    slice_segy[header[193] - geom.min_xl, :] = trace
            assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def test_read_inline():
    compare_inline(SGZ_FILE_025, SGY_FILE, 5, tolerance=1e+1)
    compare_inline(SGZ_FILE_05, SGY_FILE, 5, tolerance=1e-1)
    compare_inline(SGZ_FILE_1, SGY_FILE, 5, tolerance=1e-2)
    compare_inline(SGZ_FILE_2, SGY_FILE, 5, tolerance=1e-4)
    compare_inline(SGZ_FILE_4, SGY_FILE, 5, tolerance=1e-6)
    compare_inline(SGZ_FILE_8, SGY_FILE, 5, tolerance=1e-10)
    compare_inline(SGZ_FILE_8_8x8, SGY_FILE, 5, tolerance=1e-10)
    compare_inline(SGZ_FILE_DEC_8, SGY_FILE_DEC, 3, tolerance=1e-6)
    compare_inline_unstructured(SGZ_FILE_IRREG, SGY_FILE_IRREG, tolerance=1e-2)


def compare_crossline(sgz_filename, sgy_filename, lines, tolerance):
    with segyio.open(sgy_filename) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in range(lines):
                slice_sgz = reader.read_crossline(line_number)
                slice_segy = segyfile.xline[segyfile.xlines[line_number]]
            assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def test_read_crossline():
    compare_crossline(SGZ_FILE_025, SGY_FILE, 5, tolerance=1e+1)
    compare_crossline(SGZ_FILE_05, SGY_FILE, 5, tolerance=1e-1)
    compare_crossline(SGZ_FILE_1, SGY_FILE, 5, tolerance=1e-2)
    compare_crossline(SGZ_FILE_2, SGY_FILE, 5, tolerance=1e-4)
    compare_crossline(SGZ_FILE_4, SGY_FILE, 5, tolerance=1e-6)
    compare_crossline(SGZ_FILE_8, SGY_FILE, 5, tolerance=1e-10)
    compare_crossline(SGZ_FILE_8_8x8, SGY_FILE, 5, tolerance=1e-10)
    compare_crossline(SGZ_FILE_DEC_8, SGY_FILE_DEC, 3, tolerance=1e-6)


def compare_zslice(sgz_filename, tolerance):
    with segyio.open(SGY_FILE) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in range(50):
                slice_sgz = reader.read_zslice(line_number)
                slice_segy = segyfile.depth_slice[line_number]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def test_read_zslice():
    compare_zslice(SGZ_FILE_025, tolerance=1e+1)
    compare_zslice(SGZ_FILE_05, tolerance=1e-1)
    compare_zslice(SGZ_FILE_1, tolerance=1e-2)
    compare_zslice(SGZ_FILE_2, tolerance=1e-4)
    compare_zslice(SGZ_FILE_4, tolerance=1e-6)
    compare_zslice(SGZ_FILE_8, tolerance=1e-10)
    compare_zslice(SGZ_FILE_8_8x8, tolerance=1e-10)
    compare_zslice(SGZ_FILE_2_64x64, tolerance=1e-4)


def compare_correlated_diagonal(sgz_filename, sgy_filename):
    with segyio.open(sgy_filename) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in range(-reader.n_xlines+1, reader.n_ilines - 1):
                slice_sgz = reader.read_correlated_diagonal(line_number)
                diagonal_length = get_correlated_diagonal_length(line_number, len(segyfile.ilines), len(segyfile.xlines))
                slice_segy = np.zeros((diagonal_length, len(segyfile.samples)))
                if line_number >= 0:
                    for d in range(diagonal_length):
                        slice_segy[d, :] = segyfile.trace[(d+line_number)*len(segyfile.xlines) + d]
                else:
                    for d in range(diagonal_length):
                        slice_segy[d, :] = segyfile.trace[d*len(segyfile.xlines) + d - line_number]
            assert np.allclose(slice_sgz, slice_segy, rtol=1e-2, atol=1e-5)


def test_read_correlated_diagonal():
    for sgz_file, sgyfile in SGZ_SGY_FILE_PAIRS:
        compare_correlated_diagonal(sgz_file, sgyfile)


def compare_anticorrelated_diagonal(sgz_filename, sgy_filename):
    with segyio.open(sgy_filename) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in range(reader.n_ilines + reader.n_xlines - 1):
                slice_sgz = reader.read_anticorrelated_diagonal(line_number)
                diagonal_length = get_anticorrelated_diagonal_length(line_number, len(segyfile.ilines), len(segyfile.xlines))
                slice_segy = np.zeros((diagonal_length, len(segyfile.samples)))
                if line_number < len(segyfile.xlines):
                    for d in range(diagonal_length):
                        slice_segy[d, :] = segyfile.trace[line_number + d * (len(segyfile.xlines) - 1)]
                else:
                    for d in range(diagonal_length):
                        slice_segy[d, :] = segyfile.trace[(line_number - len(segyfile.xlines) + 1 + d) * len(segyfile.xlines)
                                                          + (len(segyfile.xlines) - d - 1)]
            assert np.allclose(slice_sgz, slice_segy, rtol=1e-2, atol=1e-5)


def test_read_anticorrelated_diagonal():
    for sgz_file, sgyfile in SGZ_SGY_FILE_PAIRS:
        compare_anticorrelated_diagonal(sgz_file, sgyfile)


def compare_subvolume(sgz_filename, tolerance):
    for preload in [True, False]:
        min_il, max_il = 2,  3
        min_xl, max_xl = 1,  2
        min_z,  max_z = 10, 20
        vol_sgz = SgzReader(sgz_filename, preload=preload).read_subvolume(min_il=min_il, max_il=max_il,
                                                                          min_xl=min_xl, max_xl=max_xl,
                                                                          min_z=min_z, max_z=max_z)
        vol_segy = segyio.tools.cube(SGY_FILE)[min_il:max_il, min_xl:max_xl, min_z:max_z]
        assert np.allclose(vol_sgz, vol_segy, rtol=tolerance)


def test_read_subvolume():
    compare_subvolume(SGZ_FILE_025, tolerance=1e+1)
    compare_subvolume(SGZ_FILE_05, tolerance=1e-1)
    compare_subvolume(SGZ_FILE_1, tolerance=1e-2)
    compare_subvolume(SGZ_FILE_2, tolerance=1e-4)
    compare_subvolume(SGZ_FILE_4, tolerance=1e-6)
    compare_subvolume(SGZ_FILE_8, tolerance=1e-10)
    compare_subvolume(SGZ_FILE_8_8x8, tolerance=1e-10)


def test_index_errors():
    # Quis custodiet custard?
    reader = SgzReader(SGZ_FILE_4)

    with pytest.raises(IndexError):
        reader.read_inline(-1)

    with pytest.raises(IndexError):
        reader.read_inline(5)

    with pytest.raises(IndexError):
        reader.read_crossline(-1)

    with pytest.raises(IndexError):
        reader.read_crossline(5)

    with pytest.raises(IndexError):
        reader.read_zslice(-1)

    with pytest.raises(IndexError):
        reader.read_zslice(50)

    with pytest.raises(IndexError):
        reader.read_correlated_diagonal(-5)

    with pytest.raises(IndexError):
        reader.read_correlated_diagonal(5)

    with pytest.raises(IndexError):
        reader.read_anticorrelated_diagonal(-1)

    with pytest.raises(IndexError):
        reader.read_anticorrelated_diagonal(9)

    with pytest.raises(IndexError):
        reader.read_subvolume(0, 10, 0, 1, 0, 10)

    with pytest.raises(IndexError):
        reader.read_subvolume(0, 1, 0, 10, 0, 10)

    with pytest.raises(IndexError):
        reader.read_subvolume(0, 1, 0, 1, 0, 100)

    with pytest.raises(IndexError):
        reader.get_trace(-1)

    with pytest.raises(IndexError):
        reader.get_trace(25)

    with pytest.raises(IndexError):
        reader.gen_trace_header(-1)

    with pytest.raises(IndexError):
        reader.gen_trace_header(25)


def test_filetype_error():
    with pytest.raises(RuntimeError):
        SgzReader(SGY_FILE)


def test_filenotfound_errors():
    with pytest.raises(FileNotFoundError):
        SgzReader('test_data/this_file_does_not_exist')
