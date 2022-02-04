import os
import numpy as np
from seismic_zfp.conversion import SeismicFileConverter, SegyConverter, SgzConverter, NumpyConverter
from seismic_zfp.sgzconstants import HEADER_DETECTION_CODES
from seismic_zfp.read import SgzReader
from seismic_zfp.headers import HeaderwordInfo
import seismic_zfp
import segyio
import pytest
from seismic_zfp.utils import generate_fake_seismic
from seismic_zfp.seismicfile import SeismicFile
import warnings
from seismic_zfp.sgzconstants import DISK_BLOCK_BYTES
from seismic_zfp.utils import int_to_bytes

import mock
import psutil

try:
    with warnings.catch_warnings():
        # pyzgy will warn us that sdglue is not available. This is expected, and safe for our purposes.
        warnings.filterwarnings("ignore", message="seismic store access is not available: No module named 'sdglue'")
        import pyzgy
        from seismic_zfp.conversion import ZgyConverter
except ImportError:
    pyzgy = None

try:
    import pyvds
    from seismic_zfp.conversion import VdsConverter
except ImportError:
    pyvds = None


SGY_FILE_IEEE = 'test_data/small-ieee.sgy'
SGY_FILE_US = 'test_data/small_us.sgy'
SGY_FILE = 'test_data/small.sgy'
SGY_FILE_NEGATIVE_SAMPLES = 'test_data/small-negative-samples.sgy'
SGY_FILE_TRACEHEADER_SAMPLERATE = 'test_data/small-traceheader-samplerate.sgy'
SGY_FILE_DUPLICATE_TRACEHEADERS = 'test_data/small-duplicate-traceheaders.sgy'
SGZ_FILE = 'test_data/small_8bit.sgz'
SGZ_FILE_2 = 'test_data/small_2bit.sgz'

SGY_FILE_IRREG = 'test_data/small-irregular.sgy'
SGZ_FILE_IRREG = 'test_data/small-irregular.sgz'
SGY_FILE_IRREG_DEC = 'test_data/small-irreg-dec.sgy'

ZGY_FILE_32 = 'test_data/zgy/small-32bit.zgy'
ZGY_FILE_16 = 'test_data/zgy/small-16bit.zgy'
ZGY_FILE_8 = 'test_data/zgy/small-8bit.zgy'

VDS_FILE = 'test_data/vds/small.vds'

SGY_FILE_32 = 'test_data/zgy/small-32bit.sgy'
SGY_FILE_16 = 'test_data/zgy/small-16bit.sgy'
SGY_FILE_8 = 'test_data/zgy/small-8bit.sgy'


def test_headerword_info_roundtrip():
    with seismic_zfp.open(SGZ_FILE) as sgzfile:
        buffer = sgzfile.hw_info.to_buffer()
    hw_info = HeaderwordInfo(25, buffer=buffer)
    assert hw_info.to_buffer() == buffer


def test_headerword_info_errors():
    with pytest.raises(RuntimeError):
        hw_info = HeaderwordInfo(25)

    with pytest.raises(RuntimeError):
        with SeismicFile.open(SGZ_FILE) as sgzfile:
            hw_info = HeaderwordInfo(25, seismicfile=sgzfile)


def test_compress_sgz_file_errors(tmp_path):
    with pytest.raises(RuntimeError):
        out_sgz = os.path.join(str(tmp_path), 'test_compress_sgz_file_errors.sgz')
        with SeismicFileConverter(SGZ_FILE) as converter:
            converter.run(out_sgz, bits_per_voxel=8)


@mock.patch('psutil.virtual_memory')
def test_compress_oom_error(mocked_virtual_memory, tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'oom.sgz')
    psutil.virtual_memory().total = 1024
    with pytest.raises(RuntimeError):
        with SeismicFileConverter(SGY_FILE) as converter:
            converter.run(out_sgz)


def compress_and_compare_detecting_filetypes(input_file, reader, tmp_path):
    out_sgz = os.path.join(str(tmp_path),
                           'test_detecting_filetype_{}_.sgz'.format(os.path.splitext(os.path.basename(input_file))[0]))
    with SeismicFileConverter(input_file) as converter:
        converter.run(out_sgz, bits_per_voxel=8)

    with seismic_zfp.open(out_sgz) as sgzfile:
        sgz_data = sgzfile.read_volume()

    assert np.allclose(sgz_data, reader.tools.cube(input_file), rtol=1e-6)

@pytest.mark.skipif(pyvds is None or pyzgy is None, reason="Requires pyvds and pyzgy")
def test_compress_detecting_filetypes(tmp_path):
    compress_and_compare_detecting_filetypes(SGY_FILE, segyio, tmp_path)
    compress_and_compare_detecting_filetypes(ZGY_FILE_32, pyzgy, tmp_path)
    compress_and_compare_detecting_filetypes(VDS_FILE, pyvds, tmp_path)


def compress_and_compare_vds(vds_file, tmp_path, bits_per_voxel, rtol, blockshape):
    out_sgz = os.path.join(str(tmp_path), 'test_{}_{}_.sgz'.format(os.path.splitext(os.path.basename(vds_file))[0],
                                                                   bits_per_voxel))

    with VdsConverter(vds_file) as converter:
        converter.run(out_sgz, bits_per_voxel=bits_per_voxel, blockshape=blockshape)

    with seismic_zfp.open(out_sgz) as sgzfile:
        with pyvds.open(vds_file) as vdsfile:
            ref_ilines = vdsfile.ilines
            ref_samples = vdsfile.samples

            sgz_data = sgzfile.read_volume()
            sgz_ilines = sgzfile.ilines
            sgz_samples = sgzfile.zslices
            assert sgzfile.structured

            for trace_number in range(25):
                sgz_header = sgzfile.header[trace_number]
                vds_header = vdsfile.header[trace_number]
                assert sgz_header == vds_header

    assert np.allclose(sgz_data, pyvds.tools.cube(vds_file), rtol=rtol)
    assert all([a == b for a, b in zip(sgz_ilines, ref_ilines)])
    assert all(ref_samples == sgz_samples)
    assert 30 == SgzReader(out_sgz).get_file_source_code()

@pytest.mark.skipif(pyvds is None, reason="Requires pyvds")
def test_compress_vds(tmp_path):
    compress_and_compare_vds(VDS_FILE, tmp_path, 8, 1e-6, blockshape=(4, 4, -1))
    compress_and_compare_vds(VDS_FILE, tmp_path, 2, 1e-4, blockshape=(64, 64, 4))


def compress_and_compare_zgy(zgy_file, sgy_file, tmp_path, bits_per_voxel, rtol, blockshape):
    out_sgz = os.path.join(str(tmp_path), 'test_{}_{}_.sgz'.format(os.path.splitext(os.path.basename(zgy_file))[0],
                                                                   bits_per_voxel))

    with ZgyConverter(zgy_file) as converter:
        converter.run(out_sgz, bits_per_voxel=bits_per_voxel, blockshape=blockshape)

    with seismic_zfp.open(out_sgz) as sgzfile:
        with pyzgy.open(zgy_file) as zgyfile:
            ref_ilines = zgyfile.ilines
            ref_samples = zgyfile.samples

            sgz_data = sgzfile.read_volume()
            sgz_ilines = sgzfile.ilines
            sgz_samples = sgzfile.zslices
            assert sgzfile.structured

            for trace_number in range(25):
                sgz_header = sgzfile.header[trace_number]
                zgy_header = zgyfile.header[trace_number]
                for header_pos in [115, 117, 181, 185, 189, 193]:
                    assert sgz_header[header_pos] == zgy_header[header_pos]

    assert np.allclose(sgz_data, segyio.tools.cube(sgy_file), rtol=rtol)
    assert all([a == b for a, b in zip(sgz_ilines, ref_ilines)])
    assert all(ref_samples == sgz_samples)
    assert 10 == SgzReader(out_sgz).get_file_source_code()


@pytest.mark.skipif(pyzgy is None, reason="Requires pyzgy")
def test_compress_zgy(tmp_path):
    compress_and_compare_zgy(ZGY_FILE_8, SGY_FILE_8, tmp_path, 16, 1e-4, blockshape=(4, 4, -1))
    compress_and_compare_zgy(ZGY_FILE_16, SGY_FILE_16, tmp_path, 16, 1e-4, blockshape=(4, 4, -1))
    compress_and_compare_zgy(ZGY_FILE_32, SGY_FILE_32, tmp_path, 16, 1e-5, blockshape=(4, 4, -1))
    compress_and_compare_zgy(ZGY_FILE_32, SGY_FILE_32, tmp_path, 2, 1e-4, blockshape=(64, 64, 4))


def compress_and_compare_axes(sgy_file, unit, tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test_axes_{}.sgz'.format(unit))

    with SegyConverter(sgy_file) as converter:
        converter.run(out_sgz)

    with segyio.open(sgy_file) as f:
        with SgzReader(out_sgz) as reader:
            assert np.all(reader.ilines == f.ilines)
            assert np.all(reader.xlines == f.xlines)
            assert np.all(reader.zslices == f.samples)
            assert reader.tracecount == f.tracecount
            assert reader.structured

def test_compress_axes(tmp_path):
    compress_and_compare_axes(SGY_FILE, "milliseconds", tmp_path)
    compress_and_compare_axes(SGY_FILE_US, "microseconds", tmp_path)
    compress_and_compare_axes(SGY_FILE_NEGATIVE_SAMPLES, "milliseconds", tmp_path)
    compress_and_compare_axes(SGY_FILE_TRACEHEADER_SAMPLERATE, "microseconds", tmp_path)


def compress_and_compare_data(sgy_file, tmp_path, bits_per_voxel, rtol, blockshape=(4, 4, -1)):
    for reduce_iops in [True, False]:
        out_sgz = os.path.join(str(tmp_path), 'small_test_data_{}_{}_.sgz'.format(bits_per_voxel, reduce_iops))

        with SegyConverter(sgy_file) as converter:
            converter.run(out_sgz, bits_per_voxel=bits_per_voxel, blockshape=blockshape, reduce_iops=reduce_iops)

        with SgzReader(out_sgz) as reader:
            sgz_data = reader.read_volume()
            assert reader.structured

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

    compress_and_compare_data(SGY_FILE, tmp_path, 2, 1e-4, blockshape=(64, 64, 4))
    compress_and_compare_data(SGY_FILE, tmp_path, 8, 1e-10, blockshape=(16, 16, 16))


def compress_compare_headers(sgy_file, tmp_path):
    for detection_method in ['heuristic', 'thorough', 'exhaustive', 'strip']:
        out_sgz = os.path.join(str(tmp_path), f'{os.path.basename(sgy_file)}_test_headers-{detection_method}.sgz')

        with SegyConverter(sgy_file) as converter:
            converter.run(out_sgz, bits_per_voxel=8, header_detection=detection_method)

        with seismic_zfp.open(out_sgz) as sgz_file:
            with segyio.open(sgy_file) as f:
                for sgz_header, sgy_header in zip(sgz_file.header, f.header):
                    if detection_method == 'strip':
                        assert all([0 == value for value in sgz_header.values()])
                    else:
                        assert sgz_header == sgy_header
                    assert sgz_file.get_header_detection_method_code() == HEADER_DETECTION_CODES[detection_method]
                    assert 0 == sgz_file.get_file_source_code()


def test_compress_headers(tmp_path):
    compress_compare_headers(SGY_FILE, tmp_path)
    compress_compare_headers(SGY_FILE_DUPLICATE_TRACEHEADERS, tmp_path)


def test_compress_headers_errors(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test_headers_errors.sgz')
    with pytest.raises(NotImplementedError):
        with SegyConverter(SGY_FILE) as converter:
            converter.run(out_sgz, bits_per_voxel=8, header_detection='fake-method')


def test_compress_crop(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test_data.sgz')

    with SegyConverter(SGY_FILE, min_il=1, max_il=4, min_xl=1, max_xl=3) as converter:
        converter.run(out_sgz, bits_per_voxel=16)

    with SgzReader(out_sgz) as reader:
        sgz_data = reader.read_volume()
        assert reader.structured

    assert np.allclose(sgz_data, segyio.tools.cube(SGY_FILE)[1:4, 1:3, :], rtol=1e-8)


def test_compress_unstructured_decimated(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test-irregular_decimated_data.sgz')

    with SegyConverter(SGY_FILE_IRREG_DEC) as converter:
        converter.run(out_sgz, bits_per_voxel=16)

    with SgzReader(out_sgz) as reader:
        sgz_data = reader.read_volume()
        assert not reader.structured

    segy_cube = segyio.tools.cube(SGY_FILE)[::2, ::2, :]
    segy_cube[2, 2, :] = 0
    assert np.allclose(sgz_data, segy_cube, atol=1e-4)

def test_compress_unstructured_headers(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test-irregular_headers.sgz')

    with SegyConverter(SGY_FILE_IRREG) as converter:
        converter.run(out_sgz, bits_per_voxel=8)

    with SgzReader(out_sgz) as reader:
        reader.read_variant_headers()
        il_header_sgz = reader.variant_headers[189]
        xl_header_sgz = reader.variant_headers[193]
        n_traces_sgz = reader.tracecount
        assert not reader.structured

    with SgzReader(out_sgz) as reader:
        reader.read_variant_headers(include_padding=True)
        il_header_sgz_padded = reader.variant_headers[189]
        xl_header_sgz_padded = reader.variant_headers[193]

    with segyio.open(SGY_FILE_IRREG, strict=False) as f:
        il_header_sgy = np.array([h[189] for h in f.header])
        xl_header_sgy = np.array([h[193] for h in f.header])
        n_traces_sgy = f.tracecount
        with seismic_zfp.open(out_sgz) as f_sgz:
            for sgz_trace, sgy_trace in zip(f_sgz.trace, f.trace):
                assert np.allclose(sgz_trace, sgy_trace, rtol=1e-2)

    with segyio.open(SGY_FILE, strict=False) as f:
        il_header_sgy_padded = np.array([h[189] for h in f.header])
        xl_header_sgy_padded = np.array([h[193] for h in f.header])
    il_header_sgy_padded[-1] = 0
    xl_header_sgy_padded[-1] = 0

    assert n_traces_sgz == n_traces_sgy
    assert np.array_equal(il_header_sgz, il_header_sgy)
    assert np.array_equal(xl_header_sgz, xl_header_sgy)
    assert np.array_equal(il_header_sgz_padded, il_header_sgy_padded)
    assert np.array_equal(xl_header_sgz_padded, xl_header_sgy_padded)


def test_compress_unstructured_data(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small_test-irregular_data.sgz')

    with SegyConverter(SGY_FILE_IRREG) as converter:
        converter.run(out_sgz, bits_per_voxel=8)

    with SgzReader(out_sgz) as reader:
        sgz_data = reader.read_volume()
        reader.read_variant_headers()
        assert not reader.structured

    with SgzReader(out_sgz) as reader:
        reader.read_variant_headers(include_padding=True)

    with segyio.open(SGY_FILE_IRREG, strict=False) as f:
        with seismic_zfp.open(out_sgz) as f_sgz:
            for sgz_trace, sgy_trace in zip(f_sgz.trace, f.trace):
                assert np.allclose(sgz_trace, sgy_trace, rtol=1e-2)

    segy_cube = segyio.tools.cube(SGY_FILE)
    segy_cube[4, 4, :] = 0
    assert np.allclose(sgz_data, segy_cube, rtol=1e-2)


def test_compress_unstructured_reduce_iops(tmp_path):
    with pytest.warns(UserWarning):
        out_sgz = os.path.join(str(tmp_path), 'small_test_data_reduce-iops.sgz')
        with SegyConverter(SGY_FILE_IRREG) as converter:
            converter.run(out_sgz, reduce_iops=True)


def test_compresss_non_existent_file(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'non-existent-file.sgz')
    with pytest.raises(FileNotFoundError):
        with SegyConverter('./non-existent-file.sgy') as converter:
            converter.run(out_sgz)
    with pytest.raises(FileNotFoundError):
        with SeismicFileConverter('./non-existent-file') as converter:
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


def test_decompress_data_erroneous_format(tmp_path):
    out_sgy = os.path.join(str(tmp_path), 'small_test_data.sgy')

    with SgzConverter(SGZ_FILE) as converter:
        new_headerbytes = bytearray(converter.headerbytes)
        new_headerbytes[DISK_BLOCK_BYTES + 3225: DISK_BLOCK_BYTES + 3227]= int_to_bytes(42)
        converter.headerbytes = bytes(new_headerbytes)
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

    with segyio.open(SGY_FILE_IRREG, ignore_geometry=True) as sgy_file:
        with seismic_zfp.open(SGZ_FILE_IRREG) as sgz_file:
            for sgy_trace, sgz_trace in zip(sgy_file.trace, sgz_file.trace):
                assert np.allclose(sgy_trace, sgz_trace, rtol=1e-2)


def compress_numpy_and_compare_data(n_samples, min_iline, n_ilines, min_xline, n_xlines, tmp_path, bits_per_voxel, rtol, blockshape=(4, 4, -1)):

    out_sgz = os.path.join(str(tmp_path), 'from-numpy.sgz')
    out_sgz_no_headers = os.path.join(str(tmp_path), 'from-numpy_no_headers.sgz')

    array, ilines, xlines, samples = generate_fake_seismic(n_ilines, n_xlines, n_samples,
                                                           min_iline=min_iline, min_xline=min_xline)

    trace_headers = {segyio.tracefield.TraceField.INLINE_3D:
                         np.broadcast_to(np.expand_dims(ilines, axis=1), (n_ilines, n_xlines)),
                     segyio.tracefield.TraceField.CROSSLINE_3D:
                         np.broadcast_to(xlines, (n_ilines,n_xlines))}

    with NumpyConverter(array, ilines=ilines, xlines=xlines, samples=samples, trace_headers=trace_headers) as converter:
        converter.run(out_sgz, bits_per_voxel=bits_per_voxel, blockshape=blockshape)

    with SgzReader(out_sgz) as reader:
        #print(np.max(np.abs(reader.read_volume()-array)/np.abs(array)))
        assert np.allclose(reader.ilines, ilines, rtol=rtol)
        assert np.allclose(reader.xlines, xlines, rtol=rtol)
        assert np.allclose(reader.read_volume(), array, rtol=rtol)
        assert 20 == reader.get_file_source_code()

    with NumpyConverter(array) as converter:
        converter.run(out_sgz_no_headers, bits_per_voxel=bits_per_voxel, blockshape=blockshape)

    with SgzReader(out_sgz_no_headers) as reader:
        assert np.allclose(reader.ilines, np.arange(array.shape[0]), rtol=rtol)
        assert np.allclose(reader.xlines, np.arange(array.shape[1]), rtol=rtol)
        assert np.allclose(reader.read_volume(), array, rtol=rtol)
        assert 20 == reader.get_file_source_code()




def test_compress_numpy_data(tmp_path):
    compress_numpy_and_compare_data(16, 0, 8, 8, 12, tmp_path, 8, 1e-3)
    compress_numpy_and_compare_data(801, 0, 8, 8, 12, tmp_path, 8, 1e-3)
    compress_numpy_and_compare_data(512, 0, 9, 8, 12, tmp_path, 8, 1e-3)
    compress_numpy_and_compare_data(512, 0, 8, 8, 13, tmp_path, 8, 1e-3)

    compress_numpy_and_compare_data(17, 0, 65, 8, 65, tmp_path, 8, 1e-2, blockshape=(32, 32, 4))
    compress_numpy_and_compare_data(801, 0, 9, 8, 13, tmp_path, 8, 1e-2, blockshape=(16, 16, 16))
