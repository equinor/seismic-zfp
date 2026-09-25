import os
import numpy as np
import segyio
import pytest

import seismic_zfp
from seismic_zfp.conversion import SgzConverter, SegyConverter
from seismic_zfp.read import SgzReader

from tests.test_read_4d import LAYOUTS, reference_cube, SGY_FILE_4D, SGY_FILE_4D_IRREG, MISSING_4D

SGZ_FILE_4D = 'test_data/small-4d_4bit.sgz'
IL = segyio.TraceField.INLINE_3D


def test_decompress_4d_data(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small-4d_test_decompress_data.sgy')

    with SgzConverter(SGZ_FILE_4D) as converter:
        converter.convert_to_segy(out_sgy)

    assert np.allclose(segyio.tools.cube(out_sgy), segyio.tools.cube(SGY_FILE_4D), rtol=1e-5)


def test_decompress_4d_geometry(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small-4d_test_decompress_geometry.sgy')

    with SgzConverter(SGZ_FILE_4D) as converter:
        converter.convert_to_segy(out_sgy)

    with segyio.open(out_sgy) as recovered, segyio.open(SGY_FILE_4D) as original:
        assert not recovered.unstructured
        assert recovered.tracecount == original.tracecount == 125
        assert np.array_equal(recovered.ilines, original.ilines)
        assert np.array_equal(recovered.xlines, original.xlines)
        assert np.array_equal(recovered.offsets, original.offsets)
        assert np.array_equal(recovered.samples, original.samples)
        assert recovered.bin[segyio.BinField.Format] == original.bin[segyio.BinField.Format]
        assert recovered.text[0] == original.text[0]
        assert np.allclose(recovered.gather[13, 23], original.gather[13, 23], rtol=1e-5)
        assert np.allclose(recovered.iline[12, 4], original.iline[12, 4], rtol=1e-5)


def test_decompress_4d_headers(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small-4d_test_decompress_headers.sgy')

    with SgzConverter(SGZ_FILE_4D) as converter:
        converter.convert_to_segy(out_sgy)

    with segyio.open(out_sgy) as recovered, segyio.open(SGY_FILE_4D) as original:
        for recovered_header, original_header in zip(recovered.header, original.header):
            assert recovered_header == original_header


@pytest.fixture(scope='module', params=LAYOUTS, ids=[layout[0] for layout in LAYOUTS])
def layout(request, tmp_path_factory):
    name, sgy_file, bits_per_voxel, blockshape, tolerance = request.param
    out_sgz = os.path.join(str(tmp_path_factory.mktemp('sgz4d')), name + '.sgz')
    with SegyConverter(sgy_file) as converter:
        converter.run(out_sgz, bits_per_voxel=bits_per_voxel, blockshape=blockshape)
    return dict(name=name, sgy=sgy_file, sgz=out_sgz, cube=reference_cube(sgy_file), tol=tolerance)


def test_decompress_4d_layouts(layout, tmp_path):
    """Every 4D layout, regular and irregular, exports the source file's traces and headers"""
    out_sgy = os.path.join(str(tmp_path), layout['name'] + '.sgy')
    with SgzConverter(layout['sgz']) as converter:
        converter.convert_to_segy(out_sgy)

    with segyio.open(layout['sgy'], strict=False, ignore_geometry=True) as source, \
            segyio.open(out_sgy, strict=False) as recovered:
        assert recovered.tracecount == source.tracecount
        for i in range(source.tracecount):
            assert recovered.header[i] == source.header[i]
            assert np.allclose(recovered.trace[i], source.trace[i], **layout['tol'])
        if layout['sgy'] == SGY_FILE_4D_IRREG:
            # Only the traces present in the source are written, so segyio finds no geometry
            assert recovered.unstructured
            assert recovered.tracecount == 125 - len(MISSING_4D)


def test_decompress_4d_round_trip(layout, tmp_path):
    """SGZ -> SEG-Y -> SGZ reproduces the data at every trace position present in the source.
    Holes in a lossy irregular file hold compression leakage in the original, but are absent
    from the SEG-Y and so zero-filled again on re-conversion."""
    out_sgy = os.path.join(str(tmp_path), layout['name'] + '.sgy')
    round_trip_sgz = os.path.join(str(tmp_path), layout['name'] + '-roundtrip.sgz')
    with SgzConverter(layout['sgz']) as converter:
        converter.convert_to_segy(out_sgy)
    with SegyConverter(out_sgy) as converter:
        converter.run(round_trip_sgz, bits_per_voxel=16, blockshape=(4, 4, 4, -1))

    with SgzReader(round_trip_sgz) as reader, SgzReader(layout['sgz']) as original:
        assert reader.is_4d
        assert reader.tracecount == original.tracecount
        assert reader.structured == original.structured
        present = original.get_tracefield_values(IL) != 0
        assert np.allclose(reader.read_volume()[present], original.read_volume()[present], rtol=1e-4, atol=1e-3)
