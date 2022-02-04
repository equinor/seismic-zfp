import numpy as np
import pytest
import seismic_zfp
from seismic_zfp.read import *
from seismic_zfp import utils
import itertools

import mock
import psutil


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


def test_read_ilines_datatype():
    reader = SgzReader(SGZ_FILE_1)
    with segyio.open(SGY_FILE) as sgyfile:
        assert reader.ilines.dtype == sgyfile.ilines.dtype


def test_read_xlines_datatype():
    reader = SgzReader(SGZ_FILE_1)
    with segyio.open(SGY_FILE) as sgyfile:
        assert reader.xlines.dtype == sgyfile.xlines.dtype


def test_read_samples_datatype():
    reader = SgzReader(SGZ_FILE_1)
    with segyio.open(SGY_FILE) as sgyfile:
        assert reader.zslices.dtype == sgyfile.samples.dtype


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


def test_get_tracefield_values():
    with SgzReader(SGZ_FILE_1) as reader:
        with segyio.open(SGY_FILE) as sgyfile:
             sgy_headers = np.array([h[segyio.tracefield.TraceField.INLINE_3D] for h in sgyfile.header[:]]).reshape((5, 5))
        sgz_headers = reader.get_tracefield_values(segyio.tracefield.TraceField.INLINE_3D)
        assert  np.array_equal(sgz_headers, sgy_headers)
        # Also check that no other arrays got read in to memory...
        with pytest.raises(KeyError):
            _ = reader.variant_headers[segyio.tracefield.TraceField.CROSSLINE_3D]


def test_read_irregular_file_not_structred():
    with SgzReader(SGZ_FILE_IRREG) as reader:
        assert reader.structured is False


def test_read_variant_headers_padding_mismatch():
    with SgzReader(SGZ_FILE_IRREG) as reader:
        reader.read_variant_headers(include_padding=True)
        with pytest.raises(AssertionError):
            reader.read_variant_headers(include_padding=False)
        reader.clear_variant_headers()
        reader.read_variant_headers(include_padding=False)


def compare_trace_coord(sgz_filename, sgy_filename, tolerance):
    with segyio.open(sgy_filename) as sgyfile:
        reader = SgzReader(sgz_filename)
        for start, stop in [(20.0, 60.0), (4.0, 84.0), (None, 16.0), (32.0, None)]:
            for i, trace_sgy in enumerate(sgyfile.trace):
                trace_sgz = reader.get_trace_by_coord(i, start, stop)
                sgy_start = np.where(sgyfile.samples == start)[0][0] if start is not None else None
                sgy_stop = np.where(sgyfile.samples == stop)[0][0] if stop is not None else None
                trace_sgy_compare = trace_sgy[sgy_start:sgy_stop]
                assert np.allclose(trace_sgz, trace_sgy_compare, rtol=tolerance)


def compare_trace_index(sgz_filename, sgy_filename, tolerance):
    with segyio.open(sgy_filename) as sgyfile:
        reader = SgzReader(sgz_filename)
        for start, stop in [(5,15), (1, 21), (None, 4), (8, None)]:
            for i, trace_sgy in enumerate(sgyfile.trace):
                trace_sgz = reader.get_trace(i, start, stop)
                assert np.allclose(trace_sgz, trace_sgy[start:stop], rtol=tolerance)

def test_get_trace():
    compare_trace_index(SGZ_FILE_2_64x64, SGY_FILE, tolerance=1e-4)
    compare_trace_index(SGZ_FILE_8, SGY_FILE, tolerance=1e-10)
    compare_trace_coord(SGZ_FILE_2_64x64, SGY_FILE, tolerance=1e-4)
    compare_trace_coord(SGZ_FILE_8, SGY_FILE, tolerance=1e-10)


def compare_inline(sgz_filename, sgy_filename, lines, tolerance):
    with segyio.open(sgy_filename) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in range(lines):
                slice_sgz = reader.read_inline(line_number)
                slice_segy = segyfile.iline[segyfile.ilines[line_number]]
            assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)
            assert reader.local == True


def compare_inline_number(sgz_filename, sgy_filename, line_coords, tolerance):
    with segyio.open(sgy_filename) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in line_coords:
                slice_sgz = reader.read_inline_number(line_number)
                slice_segy = segyfile.iline[line_number]
            assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)
            assert reader.local == True


def compare_inline_unstructured(sgz_filename, sgy_filename, tolerance):
    reader = SgzReader(sgz_filename)
    with segyio.open(sgy_filename, ignore_geometry=True) as segyfile:
        geom = utils.InferredGeometry({(h[189], h[193]): i for i, h in enumerate(segyfile.header)})
        for line_number in geom.ilines:
            slice_sgz = reader.read_inline_number(line_number)
            slice_segy = np.zeros((len(geom.xlines), len(segyfile.samples)))
            for trace, header in zip(segyfile.trace, segyfile.header):
                if header[189] == line_number:
                    slice_segy[header[193] - geom.min_xl, :] = trace
            assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)
            assert reader.local == True


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
    compare_inline_number(SGZ_FILE_8, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-10)


def compare_crossline(sgz_filename, sgy_filename, lines, tolerance):
    with segyio.open(sgy_filename) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in range(lines):
                slice_sgz = reader.read_crossline(line_number)
                slice_segy = segyfile.xline[segyfile.xlines[line_number]]
            assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)
            assert reader.local == True


def compare_crossline_number(sgz_filename, sgy_filename, line_coords, tolerance):
    with segyio.open(sgy_filename) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in line_coords:
                slice_sgz = reader.read_crossline_number(line_number)
                slice_segy = segyfile.xline[line_number]
            assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)
            assert reader.local == True


def test_read_crossline():
    compare_crossline(SGZ_FILE_025, SGY_FILE, 5, tolerance=1e+1)
    compare_crossline(SGZ_FILE_05, SGY_FILE, 5, tolerance=1e-1)
    compare_crossline(SGZ_FILE_1, SGY_FILE, 5, tolerance=1e-2)
    compare_crossline(SGZ_FILE_2, SGY_FILE, 5, tolerance=1e-4)
    compare_crossline(SGZ_FILE_4, SGY_FILE, 5, tolerance=1e-6)
    compare_crossline(SGZ_FILE_8, SGY_FILE, 5, tolerance=1e-10)
    compare_crossline(SGZ_FILE_8_8x8, SGY_FILE, 5, tolerance=1e-10)
    compare_crossline(SGZ_FILE_DEC_8, SGY_FILE_DEC, 3, tolerance=1e-6)
    compare_crossline_number(SGZ_FILE_8, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-10)


def compare_zslice(sgz_filename, tolerance):
    with segyio.open(SGY_FILE) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in range(50):
                slice_sgz = reader.read_zslice(line_number)
                slice_segy = segyfile.depth_slice[line_number]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)
                assert reader.local == True

def compare_zslice_coord(sgz_filename, tolerance):
    with segyio.open(SGY_FILE) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for slice_coord, slice_index in zip(range(0, 200, 4), range(50)):
                slice_sgz = reader.read_zslice_coord(slice_coord)
                slice_segy = segyfile.depth_slice[slice_index]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)
                assert reader.local == True

def test_read_zslice():
    compare_zslice(SGZ_FILE_025, tolerance=1e+1)
    compare_zslice(SGZ_FILE_05, tolerance=1e-1)
    compare_zslice(SGZ_FILE_1, tolerance=1e-2)
    compare_zslice(SGZ_FILE_2, tolerance=1e-4)
    compare_zslice(SGZ_FILE_4, tolerance=1e-6)
    compare_zslice(SGZ_FILE_8, tolerance=1e-10)
    compare_zslice(SGZ_FILE_8_8x8, tolerance=1e-10)
    compare_zslice(SGZ_FILE_2_64x64, tolerance=1e-4)
    compare_zslice_coord(SGZ_FILE_8, tolerance=1e-10)


def compare_correlated_diagonal_cropped(sgz_filename, min_cd_idx, max_cd_idx, min_trace_idx, max_trace_idx):
    reader = SgzReader(sgz_filename)
    for line_number in [-1, 0, 1]:

        slice_sgz = reader.read_correlated_diagonal(line_number)[min_cd_idx:max_cd_idx, min_trace_idx:max_trace_idx]
        slice_sgz_crop = reader.read_correlated_diagonal(line_number,
                                                         min_cd_idx=min_cd_idx, max_cd_idx=max_cd_idx,
                                                         min_sample_idx=min_trace_idx, max_sample_idx=max_trace_idx)
        assert np.array_equal(slice_sgz, slice_sgz_crop)


def test_read_correlated_diagonal_cropped():
    compare_correlated_diagonal_cropped(SGZ_FILE_8, 1, 3, 5, 48)
    compare_correlated_diagonal_cropped(SGZ_FILE_8_8x8, 0, 2, None, None)
    compare_correlated_diagonal_cropped(SGZ_FILE_2_64x64, 1, 3, 7, 40)


def compare_anticorrelated_diagonal_cropped(sgz_filename, min_ad_idx, max_ad_idx, min_trace_idx, max_trace_idx):
    reader = SgzReader(sgz_filename)
    for line_number in [3, 4, 5]:

        slice_sgz = reader.read_anticorrelated_diagonal(line_number)[min_ad_idx:max_ad_idx, min_trace_idx:max_trace_idx]
        slice_sgz_crop = reader.read_anticorrelated_diagonal(line_number,
                                                             min_ad_idx=min_ad_idx, max_ad_idx=max_ad_idx,
                                                             min_sample_idx=min_trace_idx, max_sample_idx=max_trace_idx)
        assert np.array_equal(slice_sgz, slice_sgz_crop)


def test_read_anticorrelated_diagonal_cropped():
    compare_anticorrelated_diagonal_cropped(SGZ_FILE_8, 1, 3, 5, 48)
    compare_anticorrelated_diagonal_cropped(SGZ_FILE_8_8x8, 0, 2, None, None)
    compare_anticorrelated_diagonal_cropped(SGZ_FILE_2_64x64, 1, 3, 7, 40)



def compare_correlated_diagonal(sgz_filename, sgy_filename):
    with segyio.open(sgy_filename) as segyfile:
        for preload in [True, False]:
            reader = SgzReader(sgz_filename, preload=preload)
            for line_number in range(-reader.n_xlines+1, reader.n_ilines - 1):
                slice_sgz = reader.read_correlated_diagonal(line_number)
                diagonal_length = utils.get_correlated_diagonal_length(line_number, len(segyfile.ilines), len(segyfile.xlines))
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
                diagonal_length = utils.get_anticorrelated_diagonal_length(line_number, len(segyfile.ilines), len(segyfile.xlines))
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
        reader.read_correlated_diagonal(1, min_cd_idx=1, max_cd_idx=9)

    with pytest.raises(IndexError):
        reader.read_correlated_diagonal(4, min_cd_idx=-1, max_cd_idx=2)

    with pytest.raises(IndexError):
        reader.read_anticorrelated_diagonal(-1)

    with pytest.raises(IndexError):
        reader.read_anticorrelated_diagonal(9)

    with pytest.raises(IndexError):
        reader.read_anticorrelated_diagonal(4, min_ad_idx=1, max_ad_idx=9)

    with pytest.raises(IndexError):
        reader.read_anticorrelated_diagonal(5, min_ad_idx=-1, max_ad_idx=2)

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


@mock.patch('psutil.virtual_memory')
def test_oom_error(mocked_virtual_memory):
    psutil.virtual_memory().total = 1024
    with pytest.raises(RuntimeError):
        SgzReader(SGZ_FILE_8, preload=True)


def test_hw_info_repr():
    expected_table = '1 | 0 | 0\n5 | 0 | 0\n9 | 0 | 0\n13 | 0 | 0\n17 | 0 | 0\n21 | 0 | 0\n25 | 0 | 0\n29 | 0 | 0\n31 | 0 | 0\n33 | 0 | 0\n35 | 0 | 0\n37 | 1 | 0\n41 | 0 | 0\n45 | 0 | 0\n49 | 0 | 0\n53 | 0 | 0\n57 | 0 | 0\n61 | 0 | 0\n65 | 0 | 0\n69 | 0 | 0\n71 | 0 | 0\n73 | 0 | 0\n77 | 0 | 0\n81 | 0 | 0\n85 | 0 | 0\n89 | 0 | 0\n91 | 0 | 0\n93 | 0 | 0\n95 | 0 | 0\n97 | 0 | 0\n99 | 0 | 0\n101 | 0 | 0\n103 | 0 | 0\n105 | 0 | 0\n107 | 0 | 0\n109 | 0 | 0\n111 | 0 | 0\n113 | 0 | 0\n115 | 0 | 0\n117 | 0 | 0\n119 | 0 | 0\n121 | 0 | 0\n123 | 0 | 0\n125 | 0 | 0\n127 | 0 | 0\n129 | 0 | 0\n131 | 0 | 0\n133 | 0 | 0\n135 | 0 | 0\n137 | 0 | 0\n139 | 0 | 0\n141 | 0 | 0\n143 | 0 | 0\n145 | 0 | 0\n147 | 0 | 0\n149 | 0 | 0\n151 | 0 | 0\n153 | 0 | 0\n155 | 0 | 0\n157 | 0 | 0\n159 | 0 | 0\n161 | 0 | 0\n163 | 0 | 0\n165 | 0 | 0\n167 | 0 | 0\n169 | 0 | 0\n171 | 0 | 0\n173 | 0 | 0\n175 | 0 | 0\n177 | 0 | 0\n179 | 0 | 0\n181 | 0 | 0\n185 | 0 | 0\n189 | 0 | 189\n193 | 0 | 193\n197 | 0 | 0\n201 | 0 | 0\n203 | 0 | 0\n205 | 0 | 0\n209 | 0 | 0\n211 | 0 | 0\n213 | 0 | 0\n215 | 0 | 0\n217 | 0 | 0\n219 | 0 | 0\n223 | 0 | 0\n225 | 0 | 0\n229 | 0 | 0\n231 | 0 | 0\n'
    with SgzReader(SGZ_FILE_4) as reader:
        hw_table = reader.hw_info.__repr__()
        assert hw_table == expected_table


def test_repr():
    with SgzReader(SGZ_FILE_4) as reader:
        representation = reader.__repr__()
        as_string = reader.__str__()
    assert as_string == 'seismic-zfp file test_data/small_4bit.sgz, Version(0.0.0.dev):\n  compression ratio: 8:1\n  inlines: 5 [1, 5]\n  crosslines: 5 [20, 24]\n  samples: 50 [0.0, 196.0]\n  traces: 25\n  Header arrays: [INLINE_3D, CROSSLINE_3D]'
    assert representation == 'SgzReader(test_data/small_4bit.sgz)'
