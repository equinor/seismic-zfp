import os
import numpy as np
from seismic_zfp.conversion import SegyConverter
from seismic_zfp.read import SgzReader
import seismic_zfp
import segyio

SGY_FILE = 'test_data/small.sgy'


def test_compress_data(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test_data.sgz')

    with SegyConverter(SGY_FILE) as converter:
        converter.run(out_sgz, bits_per_voxel=8)

    with SgzReader(out_sgz) as reader:
        sgz_data = reader.read_volume()

    assert np.allclose(sgz_data, segyio.tools.cube(SGY_FILE), rtol=1e-10)


def test_compress_headers(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test_headers.sgz')

    with SegyConverter(SGY_FILE) as converter:
        converter.run(out_sgz, bits_per_voxel=8)

    with seismic_zfp.open(out_sgz) as sgz_file:
        with segyio.open(SGY_FILE) as sgy_file:
            for sgz_header, sgy_header in zip(sgz_file.header, sgy_file.header):
                assert sgz_header == sgy_header
