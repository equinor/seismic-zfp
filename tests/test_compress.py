import os
import numpy as np
from seismic_zfp.conversion import ZgyConverter, SegyConverter, SgzConverter
from seismic_zfp.read import SgzReader
import seismic_zfp
import segyio
import pytest

SGY_FILE_IEEE = 'test_data/small-ieee.sgy'
SGY_FILE_US = 'test_data/small_us.sgy'
SGY_FILE = 'test_data/small.sgy'
SGZ_FILE = 'test_data/small_8bit.sgz'
SGZ_FILE_2 = 'test_data/small_2bit.sgz'

SGY_FILE_IRREG = 'test_data/small-irregular.sgy'
SGZ_FILE_IRREG = 'test_data/small-irregular.sgz'
SGY_FILE_IRREG_DEC = 'test_data/small-irreg-dec.sgy'

ZGY_FILE_32 = 'test_data/zgy/small-32bit.zgy'
ZGY_FILE_16 = 'test_data/zgy/small-16bit.zgy'
ZGY_FILE_8 = 'test_data/zgy/small-8bit.zgy'

SGY_FILE_32 = 'test_data/zgy/small-32bit.sgy'
SGY_FILE_16 = 'test_data/zgy/small-16bit.sgy'
SGY_FILE_8 = 'test_data/zgy/small-8bit.sgy'


def compress_and_compare_zgy(zgy_file, sgy_file, tmp_path, bits_per_voxel, rtol):
    out_sgz = os.path.join(str(tmp_path), 'test_{}_{}_.sgz'.format(os.path.splitext(os.path.basename(zgy_file))[0],
                                                                   bits_per_voxel))

    with ZgyConverter(zgy_file) as converter:
        converter.run(out_sgz, bits_per_voxel=bits_per_voxel)

    with SgzReader(out_sgz) as reader:
        sgz_data = reader.read_volume()
        sgz_ilines = reader.ilines

    with segyio.open(sgy_file) as f:
        ref_ilines = f.ilines

    assert np.allclose(sgz_data, segyio.tools.cube(sgy_file), rtol=rtol)
    assert all([a == b for a, b in zip(sgz_ilines, ref_ilines)])


def test_compress_zgy8(tmp_path):
    compress_and_compare_zgy(ZGY_FILE_8, SGY_FILE_8, tmp_path, 16, 1e-4)
    compress_and_compare_zgy(ZGY_FILE_16, SGY_FILE_16, tmp_path, 16, 1e-4)
    compress_and_compare_zgy(ZGY_FILE_32, SGY_FILE_32, tmp_path, 16, 1e-5)


def compress_and_compare_axes(sgy_file, unit, tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test_axes_{}.sgz'.format(unit))

    with SegyConverter(sgy_file) as converter:
        converter.run(out_sgz)

    with segyio.open(sgy_file) as f:
        with SgzReader(out_sgz) as reader:
            assert np.all(reader.ilines == f.ilines)
            assert np.all(reader.xlines == f.xlines)
            assert np.all(reader.zslices == f.samples)


def test_compress_axes(tmp_path):
    compress_and_compare_axes(SGY_FILE, "milliseconds", tmp_path)
    compress_and_compare_axes(SGY_FILE_US, "microseconds", tmp_path)


def compress_and_compare_data(sgy_file, tmp_path, bits_per_voxel, rtol):
    for reduce_iops in [True, False]:
        out_sgz = os.path.join(str(tmp_path), 'small_test_data_{}_{}_.sgz'.format(bits_per_voxel, reduce_iops))

        with SegyConverter(sgy_file) as converter:
            converter.run(out_sgz, bits_per_voxel=bits_per_voxel, reduce_iops=reduce_iops)

        with SgzReader(out_sgz) as reader:
            sgz_data = reader.read_volume()

        assert np.allclose(sgz_data, segyio.tools.cube(sgy_file), rtol=rtol)


def test_compress_data(tmp_path):
    compress_and_compare_data(SGY_FILE_IEEE, tmp_path, 8, 1e-8)
    compress_and_compare_data(SGY_FILE, tmp_path, 8, 1e-10)
    compress_and_compare_data(SGY_FILE, tmp_path, 8.0, 1e-10)
    compress_and_compare_data(SGY_FILE, tmp_path, "8.0", 1e-10)
    compress_and_compare_data(SGY_FILE, tmp_path, "8", 1e-10)

    compress_and_compare_data(SGY_FILE, tmp_path, -2, 1e-1)
    compress_and_compare_data(SGY_FILE, tmp_path, 0.5, 1e-1)
    compress_and_compare_data(SGY_FILE, tmp_path, "0.5", 1e-1)


def test_compress_headers(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test_headers.sgz')

    with SegyConverter(SGY_FILE) as converter:
        converter.run(out_sgz, bits_per_voxel=8)

    with seismic_zfp.open(out_sgz) as sgz_file:
        with segyio.open(SGY_FILE) as sgy_file:
            for sgz_header, sgy_header in zip(sgz_file.header, sgy_file.header):
                assert sgz_header == sgy_header


def test_compress_crop(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test_data.sgz')

    with SegyConverter(SGY_FILE, min_il=1, max_il=4, min_xl=1, max_xl=3) as converter:
        converter.run(out_sgz, bits_per_voxel=16)

    with SgzReader(out_sgz) as reader:
        sgz_data = reader.read_volume()

    assert np.allclose(sgz_data, segyio.tools.cube(SGY_FILE)[1:4, 1:3, :], rtol=1e-8)


def test_compress_unstructured_decimated(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test-irregular_decimated_data.sgz')

    with SegyConverter(SGY_FILE_IRREG_DEC) as converter:
        converter.run(out_sgz, bits_per_voxel=16)

    with SgzReader(out_sgz) as reader:
        sgz_data = reader.read_volume()

    segy_cube = segyio.tools.cube(SGY_FILE)[::2, ::2, :]
    segy_cube[2, 2, :] = 0
    assert np.allclose(sgz_data, segy_cube, atol=1e-4)


def test_compress_unstructured(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test-irregular_data.sgz')

    with SegyConverter(SGY_FILE_IRREG) as converter:
        converter.run(out_sgz, bits_per_voxel=8)

    with SgzReader(out_sgz) as reader:
        sgz_data = reader.read_volume()

    segy_cube = segyio.tools.cube(SGY_FILE)
    segy_cube[4, 4, :] = 0
    assert np.allclose(sgz_data, segy_cube, rtol=1e-2)


def test_compress_unstructured_reduce_iops(tmp_path):
    with pytest.raises(RuntimeError):
        out_sgz = os.path.join(str(tmp_path), 'small_test_data_reduce-iops.sgz')
        with SegyConverter(SGY_FILE_IRREG) as converter:
            converter.run(out_sgz, reduce_iops=True)


def test_compresss_non_existent_file(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'non-existent-file.sgz')
    with pytest.raises(FileNotFoundError):
        with SegyConverter('./non-existent-file.sgy') as converter:
            converter.run(out_sgz)


def test_convert_to_adv_from_compressed(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test_data_convert-adv.sgz')

    with SgzConverter(SGZ_FILE_2) as converter:
        converter.convert_to_adv_sgz(out_sgz)

    with SgzReader(SGZ_FILE_2) as reader:
        sgz_data = reader.read_volume()

    with SgzReader(out_sgz) as reader:
        sgz_adv_data = reader.read_volume()

    assert np.array_equal(sgz_data, sgz_adv_data)


def test_decompress_data(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test_data.sgy')

    with SgzConverter(SGZ_FILE) as converter:
        converter.convert_to_segy(out_sgy)

    assert np.allclose(segyio.tools.cube(out_sgy), segyio.tools.cube(SGY_FILE), rtol=1e-8)


def test_decompress_headers(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test_headers.sgy')

    with SgzConverter(SGZ_FILE) as converter:
        converter.convert_to_segy(out_sgy)

    with segyio.open(out_sgy) as recovered_sgy_file:
        with segyio.open(SGY_FILE) as original_sgy_file:
            for sgz_header, sgy_header in zip(recovered_sgy_file.header, original_sgy_file.header):
                assert sgz_header == sgy_header


def test_decompress_unstructured(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test-irregular_data.sgy')

    with SgzConverter(SGZ_FILE_IRREG) as converter:
        converter.convert_to_segy(out_sgy)

    segy_cube = segyio.tools.cube(SGY_FILE)
    segy_cube[4, 4, :] = 0

    assert np.allclose(segyio.tools.cube(out_sgy), segy_cube, rtol=1e-2)

