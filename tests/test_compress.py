import os
import numpy as np
from seismic_zfp.conversion import SegyConverter
from seismic_zfp.read import SgzReader
import seismic_zfp
import segyio

SGY_FILE = 'test_data/small.sgy'


def compress_and_compare_data(tmp_path, bits_per_voxel, rtol):
    out_sgz = os.path.join(str(tmp_path), 'small_test_data_{}_.sgz'.format(bits_per_voxel))

    with SegyConverter(SGY_FILE) as converter:
        converter.run(out_sgz, bits_per_voxel=bits_per_voxel)

    with SgzReader(out_sgz) as reader:
        sgz_data = reader.read_volume()

    assert np.allclose(sgz_data, segyio.tools.cube(SGY_FILE), rtol=rtol)


def test_compress_data(tmp_path):
    compress_and_compare_data(tmp_path, 8, 1e-10)
    compress_and_compare_data(tmp_path, 8.0, 1e-10)
    compress_and_compare_data(tmp_path, "8.0", 1e-10)
    compress_and_compare_data(tmp_path, "8", 1e-10)

    compress_and_compare_data(tmp_path, -2, 1e-1)
    compress_and_compare_data(tmp_path, 0.5, 1e-1)
    compress_and_compare_data(tmp_path, "0.5", 1e-1)


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
