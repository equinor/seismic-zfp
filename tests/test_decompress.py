import os
import numpy as np
import segyio
import seismic_zfp

from seismic_zfp.conversion import SgzConverter
from seismic_zfp.sgzconstants import DISK_BLOCK_BYTES
from seismic_zfp.utils import int_to_bytes

SGY_FILE = 'test_data/small.sgy'
SGZ_FILE = 'test_data/small_8bit.sgz'

SGY_FILE_IRREG = 'test_data/small-irregular.sgy'
SGZ_FILE_IRREG = 'test_data/small-irregular.sgz'

SGY_FILE_2D = 'test_data/small-2d.sgy'
SGZ_FILE_2D = 'test_data/small-2d.sgz'


def test_decompress_data(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test_decompress_data.sgy')

    with SgzConverter(SGZ_FILE) as converter:
        converter.convert_to_segy(out_sgy)

    assert np.allclose(segyio.tools.cube(out_sgy), segyio.tools.cube(SGY_FILE), rtol=1e-8)


def test_decompress_2d_data(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test_decompress_2d_data.sgy')

    with SgzConverter(SGZ_FILE_2D) as converter:
        converter.convert_to_segy(out_sgy)

    with seismic_zfp.open(SGZ_FILE_2D) as reader:
        slice_sgz = np.stack(list(reader.trace[:].copy())).astype(np.float32)

    with segyio.open(out_sgy, strict=False) as sgyfile:
        slice_segy = np.stack(list((_.copy() for _ in sgyfile.trace[:]))).astype(np.float32)

    assert np.allclose(slice_segy, slice_sgz)


def test_decompress_data_erroneous_format(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test_decompress_data_erroneous_format.sgy')

    with SgzConverter(SGZ_FILE) as converter:
        new_headerbytes = bytearray(converter.headerbytes)
        new_headerbytes[DISK_BLOCK_BYTES + 3225: DISK_BLOCK_BYTES + 3227]= int_to_bytes(42)
        converter.headerbytes = bytes(new_headerbytes)
        converter.convert_to_segy(out_sgy)

    assert np.allclose(segyio.tools.cube(out_sgy), segyio.tools.cube(SGY_FILE), rtol=1e-8)


def test_decompress_headers(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test_decompress_headers.sgy')

    with SgzConverter(SGZ_FILE) as converter:
        converter.convert_to_segy(out_sgy)

    with segyio.open(out_sgy) as recovered_sgy_file:
        with segyio.open(SGY_FILE) as original_sgy_file:
            for recovered_header, original_header in zip(recovered_sgy_file.header, original_sgy_file.header):
                assert recovered_header == original_header


def test_decompress_2d_headers(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test_decompress_2d_headers.sgy')

    with SgzConverter(SGZ_FILE_2D) as converter:
        converter.convert_to_segy(out_sgy)

    with segyio.open(out_sgy, strict=False) as recovered_sgy_file:
        with segyio.open(SGY_FILE_2D, strict=False) as original_sgy_file:
            for recovered_header, original_header in zip(recovered_sgy_file.header, original_sgy_file.header):
                assert recovered_header == original_header


def test_decompress_unstructured(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test-irregular_data.sgy')

    with SgzConverter(SGZ_FILE_IRREG) as converter:
        converter.convert_to_segy(out_sgy)

    with segyio.open(SGY_FILE_IRREG, ignore_geometry=True) as sgy_file:
        with seismic_zfp.open(SGZ_FILE_IRREG) as sgz_file:
            for sgy_trace, sgz_trace in zip(sgy_file.trace, sgz_file.trace):
                assert np.allclose(sgy_trace, sgz_trace, rtol=1e-2)
