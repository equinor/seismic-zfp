import collections
import hashlib
import os
from queue import Queue

import numpy as np
import segyio
import zfpy
import pytest
from unittest import mock

from seismic_zfp.conversion import SegyConverter
from seismic_zfp.conversion_utils import (make_header_seismic_file, io_thread_func_4d, seismic_file_producer_4d,
                                          unstructured_io_thread_func_4d)
from seismic_zfp.headers import HeaderwordInfo
from seismic_zfp.seismicfile import SeismicFile
from seismic_zfp.sgzconstants import DISK_BLOCK_BYTES, SGZ_4D_HEADER_OFFSET, HEADER_DETECTION_CODES
from seismic_zfp.utils import (Geometry4d, Geometry3d, InferredGeometry4d, bytes_to_int, bytes_to_signed_int,
                               define_blockshape_4d, pad)

SGY_FILE_4D = 'test_data/small-4d.sgy'
SGY_FILE = 'test_data/small.sgy'

IL, XL, OFFSET = segyio.TraceField.INLINE_3D, segyio.TraceField.CROSSLINE_3D, segyio.TraceField.offset


def blank_headers_dict(n_traces):
    return collections.OrderedDict((tf, np.zeros(n_traces, dtype=np.int32)) for tf in (OFFSET, IL, XL))


def expected_headers(cube_shape, geom, seismic):
    """Trace header values in (il, xl, offset) storage order for a geometry"""
    n_xl_file, n_off_file = len(seismic.xlines), len(seismic.offsets)
    exp = {tf: [] for tf in (OFFSET, IL, XL)}
    for il in geom.ilines:
        for xl in geom.xlines:
            for off in geom.offsets:
                h = seismic.header[(il * n_xl_file + xl) * n_off_file + off]
                for tf in exp:
                    exp[tf].append(h[tf])
    return {tf: np.array(v, dtype=np.int32) for tf, v in exp.items()}


@pytest.fixture(params=['segyio', 'minimal_reader'])
def reader_for(request):
    """Factory giving io_thread_func_4d's minimal_il_reader argument for each reading strategy"""
    from seismic_zfp.conversion_utils import MinimalInlineReader4d
    return lambda seismic: MinimalInlineReader4d(seismic) if request.param == 'minimal_reader' else None


def test_io_thread_func_4d_full_inline_set(reader_for):
    cube = segyio.tools.cube(SGY_FILE_4D)
    blockshape = (4, 4, 4, 128)
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 0, 5)
        headers_dict = blank_headers_dict(125)
        buffer = np.zeros((4, 8, 8, 128), dtype=np.float32)
        io_thread_func_4d(blockshape, True, headers_dict, geom, 0, 4, buffer, seismic, reader_for(seismic), 36)

        # Data
        assert np.array_equal(buffer[:, 0:5, 0:5, 0:36], cube[0:4])
        # Edge padding in xl, offset and sample directions
        assert np.array_equal(buffer[:, 5:, 0:5, 0:36], np.repeat(cube[0:4, 4:5], 3, axis=1))
        assert np.array_equal(buffer[:, :, 5:, 0:36], np.repeat(buffer[:, :, 4:5, 0:36], 3, axis=2))
        assert np.array_equal(buffer[:, :, :, 36:], np.repeat(buffer[:, :, :, 35:36], 92, axis=3))
        # Headers for inlines 0-3 populated in (il, xl, offset) order, inline 4 untouched
        exp = expected_headers(cube.shape, geom, seismic)
        for tf in headers_dict:
            assert np.array_equal(headers_dict[tf][0:100], exp[tf][0:100])
            assert np.all(headers_dict[tf][100:] == 0)


def test_io_thread_func_4d_partial_inline_set_repeats_last_plane(reader_for):
    cube = segyio.tools.cube(SGY_FILE_4D)
    blockshape = (4, 4, 4, 128)
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 0, 5)
        headers_dict = blank_headers_dict(125)
        buffer = np.zeros((4, 8, 8, 128), dtype=np.float32)
        # Second plane-set only has inline 4 to read
        io_thread_func_4d(blockshape, True, headers_dict, geom, 1, 1, buffer, seismic, reader_for(seismic), 36)

        for i in range(4):
            assert np.array_equal(buffer[i, 0:5, 0:5, 0:36], cube[4])
        exp = expected_headers(cube.shape, geom, seismic)
        for tf in headers_dict:
            assert np.array_equal(headers_dict[tf][100:125], exp[tf][100:125])
            assert np.all(headers_dict[tf][0:100] == 0)


def test_io_thread_func_4d_cropped(reader_for):
    cube = segyio.tools.cube(SGY_FILE_4D)
    blockshape = (4, 4, 4, 128)
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(1, 4, 2, 5, 1, 3)   # 3 il x 3 xl x 2 offsets = 18 traces
        headers_dict = blank_headers_dict(18)
        buffer = np.zeros((4, 4, 4, 128), dtype=np.float32)
        io_thread_func_4d(blockshape, True, headers_dict, geom, 0, 3, buffer, seismic, reader_for(seismic), 36)

        assert np.array_equal(buffer[0:3, 0:3, 0:2, 0:36], cube[1:4, 2:5, 1:3])
        assert np.array_equal(buffer[3, 0:3, 0:2, 0:36], cube[3, 2:5, 1:3])
        exp = expected_headers(cube.shape, geom, seismic)
        for tf in headers_dict:
            assert np.array_equal(headers_dict[tf], exp[tf])
        assert np.array_equal(headers_dict[OFFSET], np.tile([2, 3], 9))


def test_io_thread_func_4d_store_headers_false(reader_for):
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 0, 5)
        headers_dict = blank_headers_dict(125)
        buffer = np.zeros((4, 8, 8, 128), dtype=np.float32)
        io_thread_func_4d((4, 4, 4, 128), False, headers_dict, geom, 0, 4, buffer, seismic, reader_for(seismic), 36)
        for array in headers_dict.values():
            assert np.all(array == 0)


def drain_queue(queue):
    items = []
    while not queue.empty():
        items.append(queue.get())
    return items


def test_seismic_file_producer_4d_whole_buffer():
    cube = segyio.tools.cube(SGY_FILE_4D)
    queue = Queue()
    hash_object = hashlib.new('sha1')
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 0, 5)
        headers_dict = blank_headers_dict(125)
        seismic_file_producer_4d(queue, seismic, (4, 4, 4, 128), True, headers_dict, geom, hash_object, verbose=False)
        exp = expected_headers(cube.shape, geom, seismic)

    items = drain_queue(queue)
    assert len(items) == 2   # 5 inlines padded to 8 -> 2 inline-sets
    assert all(item.shape == (4, 8, 8, 128) for item in items)
    assert np.array_equal(items[0][:, 0:5, 0:5, 0:36], cube[0:4])
    assert np.array_equal(items[1][0, 0:5, 0:5, 0:36], cube[4])
    for tf in headers_dict:
        assert np.array_equal(headers_dict[tf], exp[tf])

    # Hash covers exactly the unpadded data, inline by inline
    expected_hash = hashlib.new('sha1')
    for i in range(5):
        expected_hash.update(np.ascontiguousarray(cube[i]))
    assert hash_object.digest() == expected_hash.digest()


def test_seismic_file_producer_4d_sliced_blocks():
    cube = segyio.tools.cube(SGY_FILE_4D)
    queue = Queue()
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 0, 5)
        # 2 bits per voxel, 8 x 8 x 4 x 64 = 16384 voxels per disk block
        blockshape = (8, 8, 4, 64)
        seismic_file_producer_4d(queue, seismic, blockshape, False, {}, geom, hashlib.new('sha1'), verbose=False)

    items = drain_queue(queue)
    # 1 inline-set, 1 xl block, 2 offset blocks, 1 sample block (36 -> 64)
    assert len(items) == 1 * 1 * 2 * 1
    assert all(item.shape == blockshape for item in items)
    # Block (offset 0) contains offsets 0-3
    assert np.array_equal(items[0][0:5, 0:5, 0:4, 0:36], cube[:, :, 0:4, :])
    # Block (offset 1) contains offset 4 then padding
    assert np.array_equal(items[1][0:5, 0:5, 0, 0:36], cube[:, :, 4, :])
    assert np.array_equal(items[1][0:5, 0:5, 1, 0:36], cube[:, :, 4, :])
    assert np.array_equal(items[1][0:5, 0:5, 3, 36:], np.repeat(cube[:, :, 4, 35:36], 28, axis=2))


# --- End-to-end: convert and verify the SGZ file by hand, no reader involved -------------------------

def parse_sgz_4d(path):
    """Decode a 4D SGZ file directly from its bytes, block by block, under the on-disk layout
    a 4D reader will rely on: inline-sets in order, and within an inline-set the disk blocks
    ordered xl-block, then offset-block, then sample-block."""
    with open(path, 'rb') as f:
        header = f.read(2 * DISK_BLOCK_BYTES)
        info = {
            'n_samples': bytes_to_int(header[4:8]),
            'n_xl': bytes_to_int(header[8:12]),
            'n_il': bytes_to_int(header[12:16]),
            'min_xl': bytes_to_signed_int(header[20:24]),
            'min_il': bytes_to_signed_int(header[24:28]),
            'rate_code': bytes_to_signed_int(header[40:44]),
            'data_blocks': bytes_to_int(header[56:60]),
            'header_entry_bytes': bytes_to_int(header[60:64]),
            'n_header_arrays': bytes_to_int(header[64:68]),
            'tracecount': bytes_to_int(header[68:72]),
            'n_offsets': bytes_to_int(header[SGZ_4D_HEADER_OFFSET:SGZ_4D_HEADER_OFFSET + 4]),
            'min_offset': bytes_to_signed_int(header[SGZ_4D_HEADER_OFFSET + 4:SGZ_4D_HEADER_OFFSET + 8]),
            'offset_step': bytes_to_signed_int(header[SGZ_4D_HEADER_OFFSET + 8:SGZ_4D_HEADER_OFFSET + 12]),
            'hash': header[960:980],
        }
        info['blockshape'] = (bytes_to_int(header[44:48]), bytes_to_int(header[48:52]),
                              bytes_to_int(header[SGZ_4D_HEADER_OFFSET + 12:SGZ_4D_HEADER_OFFSET + 16]),
                              bytes_to_int(header[52:56]))
        rate = info['rate_code'] if info['rate_code'] > 0 else 1 / -info['rate_code']
        hw_info = HeaderwordInfo(info['tracecount'], buffer=header[980:2048])

        bs = info['blockshape']
        padded = (pad(info['n_il'], bs[0]), pad(info['n_xl'], bs[1]),
                  pad(info['n_offsets'], bs[2]), pad(info['n_samples'], bs[3]))
        volume = np.zeros(padded, dtype=np.float32)
        n_blocks = tuple(p // b for p, b in zip(padded, bs))
        assert info['data_blocks'] == int(np.prod(n_blocks))
        ztype = zfpy.dtype_to_ztype(np.dtype('float32'))
        for i in range(n_blocks[0]):
            for x in range(n_blocks[1]):
                for o in range(n_blocks[2]):
                    for z in range(n_blocks[3]):
                        block = f.read(DISK_BLOCK_BYTES)
                        assert len(block) == DISK_BLOCK_BYTES
                        volume[i * bs[0]:(i + 1) * bs[0], x * bs[1]:(x + 1) * bs[1],
                               o * bs[2]:(o + 1) * bs[2], z * bs[3]:(z + 1) * bs[3]] = \
                            zfpy._decompress(block, ztype, bs, rate=rate)
        info['padded_volume'] = volume
        info['volume'] = volume[0:info['n_il'], 0:info['n_xl'], 0:info['n_offsets'], 0:info['n_samples']]

        header_arrays = {}
        stored_keys = [segyio.tracefield.TraceField(k) for k, v in hw_info.table.items() if v[1] == k]
        assert len(stored_keys) == info['n_header_arrays']
        padded_entry_bytes = pad(info['header_entry_bytes'], 512)
        for key in stored_keys:
            buf = f.read(padded_entry_bytes)
            header_arrays[key] = np.frombuffer(buf[0:info['header_entry_bytes']], dtype=np.int32)
        info['header_arrays'] = header_arrays
        info['hw_table'] = hw_info.table
        assert f.read() == b''   # Nothing after the footer
    return info


def sha1_of_cube(cube):
    h = hashlib.new('sha1')
    for i in range(cube.shape[0]):
        h.update(np.ascontiguousarray(cube[i]))
    return h.digest()


def test_segy_converter_4d_lossless_roundtrip(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-16bit.sgz')
    with SegyConverter(SGY_FILE_4D) as converter:
        assert converter.is_4d
        estimated_size = converter.get_output_size(bits_per_voxel=16)
        converter.run(out_sgz, bits_per_voxel=16)
    assert os.path.getsize(out_sgz) == estimated_size

    cube = segyio.tools.cube(SGY_FILE_4D)
    info = parse_sgz_4d(out_sgz)
    assert (info['n_il'], info['n_xl'], info['n_offsets'], info['n_samples']) == cube.shape
    assert (info['min_il'], info['min_xl'], info['min_offset'], info['offset_step']) == (11, 21, 1, 1)
    # 16 bpv, (4, 4, 4, -1) => 32 samples per block, 36 samples pad to 64 => 2 sample-blocks
    assert info['blockshape'] == (4, 4, 4, 32)
    assert info['tracecount'] == 125
    assert info['header_entry_bytes'] == 125 * 4
    assert np.allclose(info['volume'], cube, rtol=1e-6)
    # Padding regions repeat edges rather than being zero, in all four dimensions
    pv = info['padded_volume']
    assert np.allclose(pv[5:, 0:5, 0:5, 0:36], np.repeat(cube[4:5], 3, axis=0), rtol=1e-6)
    assert np.allclose(pv[0:5, 5:, 0:5, 0:36], np.repeat(cube[:, 4:5], 3, axis=1), rtol=1e-6)
    assert np.allclose(pv[0:5, 0:5, 5:, 0:36], np.repeat(cube[:, :, 4:5], 3, axis=2), rtol=1e-6)
    assert np.allclose(pv[0:5, 0:5, 0:5, 36:], np.repeat(cube[:, :, :, 35:36], 28, axis=3), rtol=1e-6)
    assert info['hash'] == sha1_of_cube(cube)

    with segyio.open(SGY_FILE_4D) as segyfile:
        assert set(info['header_arrays'].keys()) == {OFFSET, IL, XL}
        for tf in (OFFSET, IL, XL):
            assert np.array_equal(info['header_arrays'][tf], segyfile.attributes(tf)[:])


def test_segy_converter_4d_default_bitrate(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-4bit.sgz')
    with SegyConverter(SGY_FILE_4D) as converter:
        converter.run(out_sgz)
    cube = segyio.tools.cube(SGY_FILE_4D)
    info = parse_sgz_4d(out_sgz)
    assert info['blockshape'] == (4, 4, 4, 128)
    assert info['data_blocks'] == 2 * 2 * 2 * 1
    # Edge-repeated padding keeps the compressed blocks smooth: measured max rel. error ~1.4e-7
    assert np.allclose(info['volume'], cube, rtol=1e-5)


def test_segy_converter_4d_sliced_blockshape(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-2bit.sgz')
    with SegyConverter(SGY_FILE_4D) as converter:
        estimated_size = converter.get_output_size(bits_per_voxel=2, blockshape=(8, 8, 4, 64))
        converter.run(out_sgz, bits_per_voxel=2, blockshape=(8, 8, 4, 64))
    assert os.path.getsize(out_sgz) == estimated_size
    cube = segyio.tools.cube(SGY_FILE_4D)
    info = parse_sgz_4d(out_sgz)
    assert info['blockshape'] == (8, 8, 4, 64)
    assert info['data_blocks'] == 1 * 1 * 2 * 1
    # Measured max rel. error ~5.5e-7 at 2 bpv
    assert np.allclose(info['volume'], cube, rtol=1e-5)
    assert info['hash'] == sha1_of_cube(cube)


def test_segy_converter_4d_cropped(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-crop.sgz')
    with SegyConverter(SGY_FILE_4D, min_il=1, max_il=4, min_xl=2, max_xl=5, min_offset=1, max_offset=4) as converter:
        assert converter.is_4d
        estimated_size = converter.get_output_size(bits_per_voxel=16)
        converter.run(out_sgz, bits_per_voxel=16)
    assert os.path.getsize(out_sgz) == estimated_size

    cube = segyio.tools.cube(SGY_FILE_4D)[1:4, 2:5, 1:4]
    info = parse_sgz_4d(out_sgz)
    assert (info['n_il'], info['n_xl'], info['n_offsets'], info['n_samples']) == (3, 3, 3, 36)
    assert (info['min_il'], info['min_xl'], info['min_offset']) == (12, 23, 2)
    assert info['tracecount'] == 27
    assert info['header_entry_bytes'] == 27 * 4
    assert np.allclose(info['volume'], cube, rtol=1e-6)
    assert info['hash'] == sha1_of_cube(cube)
    assert np.array_equal(info['header_arrays'][OFFSET], np.tile([2, 3, 4], 9))
    assert np.array_equal(info['header_arrays'][XL], np.tile(np.repeat([23, 24, 25], 3), 3))
    assert np.array_equal(info['header_arrays'][IL], np.repeat([12, 13, 14], 9))


def test_segy_converter_4d_partial_crop_defaults_other_axes(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-offsetcrop.sgz')
    with SegyConverter(SGY_FILE_4D, min_offset=3) as converter:
        converter.run(out_sgz, bits_per_voxel=16)
    cube = segyio.tools.cube(SGY_FILE_4D)[:, :, 3:]
    info = parse_sgz_4d(out_sgz)
    assert (info['n_il'], info['n_xl'], info['n_offsets']) == (5, 5, 2)
    assert info['min_offset'] == 4
    assert np.allclose(info['volume'], cube, rtol=1e-6)


def test_segy_converter_4d_strip_headers(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-strip.sgz')
    with SegyConverter(SGY_FILE_4D) as converter:
        estimated_size = converter.get_output_size(bits_per_voxel=16, header_detection='strip')
        converter.run(out_sgz, bits_per_voxel=16, header_detection='strip')
    assert os.path.getsize(out_sgz) == estimated_size
    info = parse_sgz_4d(out_sgz)
    assert info['n_header_arrays'] == 0
    assert info['header_arrays'] == {}
    assert np.allclose(info['volume'], segyio.tools.cube(SGY_FILE_4D), rtol=1e-6)


def test_segy_converter_4d_thorough_headers(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-thorough.sgz')
    with SegyConverter(SGY_FILE_4D) as converter:
        converter.run(out_sgz, bits_per_voxel=16, header_detection='thorough')
    info = parse_sgz_4d(out_sgz)
    # Only the three coordinate headers vary in this file
    assert set(info['header_arrays'].keys()) == {OFFSET, IL, XL}
    with segyio.open(SGY_FILE_4D) as segyfile:
        for tf in (OFFSET, IL, XL):
            assert np.array_equal(info['header_arrays'][tf], segyfile.attributes(tf)[:])


def files_identical_except_version(a, b):
    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        da, db = bytearray(fa.read()), bytearray(fb.read())
    da[72:76] = db[72:76] = bytes(4)
    return da == db


def test_minimal_inline_reader_4d_self_test():
    from seismic_zfp.conversion_utils import MinimalInlineReader4d
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        reader = MinimalInlineReader4d(seismic)
        assert reader.self_test()
        cube = segyio.tools.cube(SGY_FILE_4D)
        for il_id in range(5):
            fields, inline = reader.read_line(il_id, [IL, XL, OFFSET, 115])
            assert inline.dtype == np.float32
            assert np.array_equal(inline, cube[il_id])
            assert np.array_equal(fields[IL], seismic.attributes(IL)[il_id * 25:(il_id + 1) * 25])
            assert np.array_equal(fields[OFFSET], np.tile([1, 2, 3, 4, 5], 5))
            # 2-byte field, read from the same buffer
            assert np.array_equal(fields[115], seismic.attributes(115)[il_id * 25:(il_id + 1) * 25])
        # No header fields requested
        fields, inline = reader.read_line(2)
        assert fields == {} and inline.shape == (5, 5, 36)


def test_minimal_inline_reader_4d_wrong_format():
    from seismic_zfp.conversion_utils import MinimalInlineReader4d
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        reader = MinimalInlineReader4d(seismic)
        with mock.patch.object(MinimalInlineReader4d, 'get_format_code', return_value=2):
            with pytest.raises(RuntimeError):
                reader.read_line(0)


@pytest.mark.parametrize('header_detection', ['heuristic', 'thorough', 'strip'])
def test_segy_converter_4d_reduce_iops_identical_output(tmp_path, header_detection):
    """The minimal inline reader must produce the same file as the segyio path, byte for byte"""
    outputs = {}
    for reduce_iops in (False, True):
        out_sgz = os.path.join(str(tmp_path), f'small-4d-iops-{reduce_iops}.sgz')
        with SegyConverter(SGY_FILE_4D) as converter:
            converter.run(out_sgz, bits_per_voxel=8, reduce_iops=reduce_iops, header_detection=header_detection)
        outputs[reduce_iops] = out_sgz
    assert files_identical_except_version(outputs[False], outputs[True])
    assert np.allclose(parse_sgz_4d(outputs[True])['volume'], segyio.tools.cube(SGY_FILE_4D), rtol=1e-5)


def test_segy_converter_4d_reduce_iops_cropped(tmp_path):
    outputs = {}
    for reduce_iops in (False, True):
        out_sgz = os.path.join(str(tmp_path), f'small-4d-iops-crop-{reduce_iops}.sgz')
        with SegyConverter(SGY_FILE_4D, min_il=1, max_il=4, min_xl=2, max_xl=5, min_offset=1, max_offset=4) as converter:
            converter.run(out_sgz, bits_per_voxel=16, reduce_iops=reduce_iops)
        outputs[reduce_iops] = out_sgz
    assert files_identical_except_version(outputs[False], outputs[True])
    assert np.allclose(parse_sgz_4d(outputs[True])['volume'], segyio.tools.cube(SGY_FILE_4D)[1:4, 2:5, 1:4], rtol=1e-6)


def test_segy_converter_4d_reduce_iops_falls_back_on_failed_self_test(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-iops-fallback.sgz')
    with mock.patch('seismic_zfp.conversion_utils.MinimalInlineReader4d.self_test', return_value=False):
        with SegyConverter(SGY_FILE_4D) as converter:
            with pytest.warns(UserWarning, match="failed self-test"):
                converter.run(out_sgz, bits_per_voxel=16, reduce_iops=True)
    assert np.allclose(parse_sgz_4d(out_sgz)['volume'], segyio.tools.cube(SGY_FILE_4D), rtol=1e-6)


def test_segy_converter_4d_reduce_iops_irregular_warns(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-iops-irregular.sgz')
    with SegyConverter(SGY_FILE_4D_IRREG) as converter:
        with pytest.warns(UserWarning, match="irregular"):
            converter.run(out_sgz, bits_per_voxel=16, reduce_iops=True)
    assert parse_sgz_4d(out_sgz)['tracecount'] == 118


def test_segy_converter_offset_crop_rejected_for_3d():
    with pytest.raises(ValueError, match="prestack"):
        SegyConverter(SGY_FILE, min_offset=0, max_offset=1)


def test_segy_converter_4d_rejects_3d_blockshape(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-bad.sgz')
    with SegyConverter(SGY_FILE_4D) as converter:
        with pytest.raises(AssertionError):
            converter.run(out_sgz, bits_per_voxel=4, blockshape=(4, 4, -1))


def test_segy_converter_3d_files_not_4d():
    for sgy in (SGY_FILE, 'test_data/small-2d.sgy', 'test_data/small-irregular.sgy'):
        with SegyConverter(sgy) as converter:
            assert not converter.is_4d


# --- Irregular (unstructured) prestack input -------------------------------------------------------

SGY_FILE_4D_IRREG = 'test_data/small-4d-irregular.sgy'
# Traces removed from small-4d.sgy to make the irregular file, as (il, xl, offset) *ordinals*
MISSING_4D = [(4, 4, o) for o in range(5)] + [(1, 2, 0), (2, 0, 4)]


def irregular_reference():
    """Regular reference cube with missing traces zeroed, plus a boolean mask of present traces"""
    cube = segyio.tools.cube(SGY_FILE_4D).copy()
    present = np.ones(cube.shape[:3], dtype=bool)
    for il, xl, off in MISSING_4D:
        cube[il, xl, off] = 0
        present[il, xl, off] = False
    return cube, present


def test_irregular_4d_test_data_is_as_documented():
    cube_full = segyio.tools.cube(SGY_FILE_4D)
    with segyio.open(SGY_FILE_4D_IRREG, strict=False) as irregular, segyio.open(SGY_FILE_4D) as regular:
        assert irregular.tracecount == 125 - len(MISSING_4D)
        assert irregular.unstructured
        triples = set(zip(irregular.attributes(IL)[:].tolist(), irregular.attributes(XL)[:].tolist(),
                          irregular.attributes(OFFSET)[:].tolist()))
        expected_missing = {(regular.ilines[il], regular.xlines[xl], regular.offsets[off]) for il, xl, off in MISSING_4D}
        assert triples.isdisjoint(expected_missing)
        assert len(triples) == irregular.tracecount
        for i, h in enumerate(irregular.header):
            il, xl, off = h[IL] - 11, h[XL] - 21, h[OFFSET] - 1
            assert np.array_equal(irregular.trace[i], cube_full[il, xl, off])


def test_segy_converter_4d_irregular_geometry_inference():
    with SegyConverter(SGY_FILE_4D_IRREG) as converter:
        assert converter.is_4d
        assert converter.geom is None   # Only inferred when needed
        converter.get_output_size(bits_per_voxel=16)
        geom = converter.geom
    assert isinstance(geom, InferredGeometry4d)
    assert list(geom.ilines) == [11, 12, 13, 14, 15]
    assert list(geom.xlines) == [21, 22, 23, 24, 25]
    assert list(geom.offsets) == [1, 2, 3, 4, 5]
    assert len(geom.traces_ref) == 118
    assert (15, 25, 1) not in geom.traces_ref
    assert geom.traces_ref[(11, 21, 1)] == 0


def test_segy_converter_4d_irregular_roundtrip(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-irregular-16bit.sgz')
    with SegyConverter(SGY_FILE_4D_IRREG) as converter:
        estimated_size = converter.get_output_size(bits_per_voxel=16)
        converter.run(out_sgz, bits_per_voxel=16)
    assert os.path.getsize(out_sgz) == estimated_size

    cube, present = irregular_reference()
    info = parse_sgz_4d(out_sgz)
    # Enclosing regular grid is described in the header, actual trace count signals irregularity
    assert (info['n_il'], info['n_xl'], info['n_offsets'], info['n_samples']) == (5, 5, 5, 36)
    assert (info['min_il'], info['min_xl'], info['min_offset'], info['offset_step']) == (11, 21, 1, 1)
    assert info['tracecount'] == 118
    assert info['header_entry_bytes'] == 125 * 4   # header arrays span the full grid
    assert info['blockshape'] == (4, 4, 4, 32)

    volume = info['volume']
    assert np.allclose(volume[present], cube[present], rtol=1e-5)
    # Missing traces are zero-filled; fixed-rate compression smears a little into single holes
    assert np.all(volume[4, 4] == 0)
    assert np.abs(volume[~present]).max() < 1e-3

    # Header arrays are laid out on the full grid with zeros at missing positions
    hdr = info['header_arrays']
    assert set(hdr.keys()) == {OFFSET, IL, XL}
    grid_il = np.repeat([11, 12, 13, 14, 15], 25)
    grid_xl = np.tile(np.repeat([21, 22, 23, 24, 25], 5), 5)
    grid_off = np.tile([1, 2, 3, 4, 5], 25)
    flat_present = present.reshape(-1)
    assert np.array_equal(hdr[IL], np.where(flat_present, grid_il, 0))
    assert np.array_equal(hdr[XL], np.where(flat_present, grid_xl, 0))
    assert np.array_equal(hdr[OFFSET], np.where(flat_present, grid_off, 0))
    assert np.count_nonzero(hdr[IL]) == 118

    # Hash is over the zero-filled grid, as fed to the compressor
    assert info['hash'] == sha1_of_cube(cube)


def test_segy_converter_4d_irregular_sliced_blockshape(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-irregular-2bit.sgz')
    with SegyConverter(SGY_FILE_4D_IRREG) as converter:
        converter.run(out_sgz, bits_per_voxel=2, blockshape=(8, 8, 4, 64))
    cube, present = irregular_reference()
    info = parse_sgz_4d(out_sgz)
    assert info['data_blocks'] == 2
    assert info['tracecount'] == 118
    # Padding beyond the grid repeats edges, including the zeroed corner gather
    pv = info['padded_volume']
    assert np.allclose(pv[5:, 4, :, 0:36], 0, atol=1e-3)
    assert np.allclose(pv[4, 5:, :, 0:36], 0, atol=1e-3)
    assert np.allclose(pv[0:4, 5:, 0:5, 0:36], np.repeat(cube[0:4, 4:5], 3, axis=1), rtol=1e-2)


def test_segy_converter_4d_irregular_strip_headers(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-irregular-strip.sgz')
    with SegyConverter(SGY_FILE_4D_IRREG) as converter:
        estimated_size = converter.get_output_size(bits_per_voxel=16, header_detection='strip')
        converter.run(out_sgz, bits_per_voxel=16, header_detection='strip')
    assert os.path.getsize(out_sgz) == estimated_size
    info = parse_sgz_4d(out_sgz)
    assert info['n_header_arrays'] == 0
    cube, present = irregular_reference()
    assert np.allclose(info['volume'][present], cube[present], rtol=1e-5)


def test_read_trace_header_fields_matches_segyio():
    import glob
    from seismic_zfp.conversion_utils import read_trace_header_fields
    # 4-byte and 2-byte fields, including sample count / interval and the last field
    fields = [189, 193, 37, 181, 185, 1, 5, 29, 71, 115, 117, 229]
    checked = 0
    for sgy in sorted(glob.glob('test_data/**/*.sgy', recursive=True)):
        with SeismicFile.open(sgy) as seismic:
            values = read_trace_header_fields(seismic, fields)
            for tf in fields:
                assert values[tf].dtype == np.int32
                assert np.array_equal(values[tf], seismic.attributes(tf)[:]), (sgy, tf)
            # Sub-range
            partial = read_trace_header_fields(seismic, [189, 115], start=1, stop=min(4, seismic.tracecount))
            assert np.array_equal(partial[189], seismic.attributes(189)[1:4])
            assert np.array_equal(partial[115], seismic.attributes(115)[1:4])
            checked += 1
    assert checked > 10


def test_read_trace_header_fields_falls_back_for_non_segy():
    from seismic_zfp.conversion_utils import read_trace_header_fields
    with SeismicFile.open('test_data/small_4bit.sgz') as sgz:
        values = read_trace_header_fields(sgz, [189, 193])
        assert np.array_equal(values[189], sgz.attributes(189)[:])
        assert len(values[193]) == sgz.tracecount


def test_segy_converter_4d_irregular_crop_rejected():
    with pytest.raises(NotImplementedError):
        SegyConverter(SGY_FILE_4D_IRREG, min_offset=1)


def test_segy_converter_4d_irregular_offset_sorted(tmp_path):
    """An irregular prestack file that is not inline-sorted: each inline's traces are scattered
    through the file, so the slab read is abandoned for per-trace reads, with identical results"""
    sgy_sorted = os.path.join(str(tmp_path), 'small-4d-offset-sorted.sgy')
    with segyio.open(SGY_FILE_4D, ignore_geometry=True) as src:
        headers = [(h[IL], h[XL], h[OFFSET], i) for i, h in enumerate(src.header)]
        order = [i for il, xl, off, i in sorted(headers, key=lambda k: (k[2], k[0], k[1])) if (il, xl, off) != (13, 23, 3)]
        spec = segyio.spec()
        spec.format, spec.samples, spec.tracecount = src.format, src.samples, len(order)
        with segyio.create(sgy_sorted, spec) as dst:
            dst.text[0], dst.bin = src.text[0], src.bin
            for n, i in enumerate(order):
                dst.header[n], dst.trace[n] = src.header[i], src.trace[i]

    out_sgz = os.path.join(str(tmp_path), 'small-4d-offset-sorted.sgz')
    with SegyConverter(sgy_sorted) as converter:
        assert converter.is_4d
        converter.run(out_sgz, bits_per_voxel=16)
        trace_ids = converter.geom.inline_trace_ids(0)
        assert trace_ids[-1] - trace_ids[0] + 1 > 2 * len(trace_ids)   # non-contiguous, fallback path taken

    cube = segyio.tools.cube(SGY_FILE_4D).copy()
    cube[2, 2, 2] = 0
    info = parse_sgz_4d(out_sgz)
    assert info['tracecount'] == 124
    present = np.ones((5, 5, 5), dtype=bool)
    present[2, 2, 2] = False
    assert np.allclose(info['volume'][present], cube[present], rtol=1e-4)
    assert info['hash'] == sha1_of_cube(cube)
    grid_off = np.tile([1, 2, 3, 4, 5], 25)
    assert np.array_equal(info['header_arrays'][OFFSET], np.where(present.reshape(-1), grid_off, 0))


SGY_FILE_4D_IRREG_NOMETRICS = 'test_data/small-4d-irregular-nometrics.sgy'
MISSING_4D_NOMETRICS = [(0, 1, o) for o in (2, 3, 4)]


def test_segy_converter_4d_irregular_without_segyio_metrics(tmp_path):
    """Regression: an irregular prestack file for which segyio cannot determine geometry was
    converted as irregular 3D, keeping only one trace per IL/XL position"""
    out_sgz = os.path.join(str(tmp_path), 'small-4d-irregular-nometrics.sgz')
    with SegyConverter(SGY_FILE_4D_IRREG_NOMETRICS) as converter:
        assert converter.is_4d
        estimated_size = converter.get_output_size(bits_per_voxel=16)
        assert isinstance(converter.geom, InferredGeometry4d)
        assert list(converter.geom.offsets) == [1, 2, 3, 4, 5]
        converter.run(out_sgz, bits_per_voxel=16)
    assert os.path.getsize(out_sgz) == estimated_size

    cube = segyio.tools.cube(SGY_FILE_4D).copy()
    present = np.ones(cube.shape[:3], dtype=bool)
    for il, xl, off in MISSING_4D_NOMETRICS:
        cube[il, xl, off] = 0
        present[il, xl, off] = False
    info = parse_sgz_4d(out_sgz)
    assert (info['n_il'], info['n_xl'], info['n_offsets']) == (5, 5, 5)
    assert info['tracecount'] == 122
    # Three adjacent holes in the first block: measured max rel. error 2.3e-5 on present traces
    assert np.allclose(info['volume'][present], cube[present], rtol=1e-4)
    assert np.abs(info['volume'][~present]).max() < 1e-2
    assert np.count_nonzero(info['header_arrays'][IL]) == 122


def test_unstructured_io_thread_func_4d_partial_plane_set():
    cube, present = irregular_reference()
    with SegyConverter(SGY_FILE_4D_IRREG) as converter, SeismicFile.open(SGY_FILE_4D_IRREG) as seismic:
        converter.infer_geometry(seismic)
        geom = converter.geom
        headers_dict = blank_headers_dict(125)
        buffer = np.zeros((4, 8, 8, 128), dtype=np.float32)
        # Second plane-set holds only inline ordinal 4, which contains the missing corner gather
        unstructured_io_thread_func_4d((4, 4, 4, 128), True, headers_dict, geom, 1, 1, buffer, seismic, 36)

    for i in range(4):
        assert np.array_equal(buffer[i, 0:5, 0:5, 0:36], cube[4])
    assert np.all(buffer[:, 4, :, :] == 0)   # missing corner gather and its xl-edge padding
    assert np.all(buffer[:, 5:, :, :] == 0)
    assert np.array_equal(buffer[0, 0:5, 5:, 0:36], np.repeat(cube[4, :, 4:5], 3, axis=1))
    for tf, grid in ((IL, np.repeat([11, 12, 13, 14, 15], 25)),):
        assert np.array_equal(headers_dict[tf][100:120], grid[100:120])
        assert np.all(headers_dict[tf][120:125] == 0)
        assert np.all(headers_dict[tf][0:100] == 0)


def test_make_header_4d():
    bits_per_voxel, blockshape = define_blockshape_4d(4, (4, 4, 4, -1))
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 0, 5)
        hw_info = HeaderwordInfo(n_traces=125, seismicfile=seismic, header_detection='heuristic')
        header = make_header_seismic_file(seismic, bits_per_voxel, blockshape, geom, hw_info)

    assert len(header) == 2 * DISK_BLOCK_BYTES
    # Unchanged 3D fields
    assert bytes_to_int(header[4:8]) == 36        # samples
    assert bytes_to_int(header[8:12]) == 5        # n_xl
    assert bytes_to_int(header[12:16]) == 5       # n_il
    assert bytes_to_signed_int(header[20:24]) == 21
    assert bytes_to_signed_int(header[24:28]) == 11
    assert bytes_to_signed_int(header[40:44]) == 4
    assert bytes_to_int(header[44:48]) == 4       # blockshape il
    assert bytes_to_int(header[48:52]) == 4       # blockshape xl
    assert bytes_to_int(header[52:56]) == 128     # blockshape samples (last dimension)
    # padded (8 x 8 x 8 x 128) voxels x 4 bits / 8 / 4096
    assert bytes_to_int(header[56:60]) == (8 * 8 * 8 * 128 * 4) // 8 // DISK_BLOCK_BYTES
    assert bytes_to_int(header[60:64]) == 125 * 4  # one int32 per trace
    assert bytes_to_int(header[68:72]) == 125      # tracecount
    assert bytes_to_int(header[76:80]) == 0        # SEG-Y source
    assert bytes_to_int(header[80:84]) == HEADER_DETECTION_CODES['heuristic']
    # New 4D fields
    assert bytes_to_int(header[SGZ_4D_HEADER_OFFSET:SGZ_4D_HEADER_OFFSET + 4]) == 5     # n_offsets
    assert bytes_to_signed_int(header[SGZ_4D_HEADER_OFFSET + 4:SGZ_4D_HEADER_OFFSET + 8]) == 1   # min offset
    assert bytes_to_signed_int(header[SGZ_4D_HEADER_OFFSET + 8:SGZ_4D_HEADER_OFFSET + 12]) == 1  # offset step
    assert bytes_to_int(header[SGZ_4D_HEADER_OFFSET + 12:SGZ_4D_HEADER_OFFSET + 16]) == 4      # blockshape offset


def test_make_header_4d_cropped_offsets():
    bits_per_voxel, blockshape = define_blockshape_4d(8, (4, 4, 4, -1))
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 2, 5)
        hw_info = HeaderwordInfo(n_traces=75, seismicfile=seismic, header_detection='heuristic')
        header = make_header_seismic_file(seismic, bits_per_voxel, blockshape, geom, hw_info)

    assert bytes_to_int(header[SGZ_4D_HEADER_OFFSET:SGZ_4D_HEADER_OFFSET + 4]) == 3
    assert bytes_to_signed_int(header[SGZ_4D_HEADER_OFFSET + 4:SGZ_4D_HEADER_OFFSET + 8]) == 3
    assert bytes_to_int(header[60:64]) == 75 * 4
    assert bytes_to_int(header[68:72]) == 75
    assert bytes_to_int(header[52:56]) == 64
    assert bytes_to_int(header[56:60]) == (8 * 8 * 4 * 64 * 8) // 8 // DISK_BLOCK_BYTES


def test_make_header_3d_leaves_4d_bytes_zero():
    with SeismicFile.open(SGY_FILE) as seismic:
        geom = Geometry3d(0, 5, 0, 5)
        hw_info = HeaderwordInfo(n_traces=25, seismicfile=seismic, header_detection='heuristic')
        header = make_header_seismic_file(seismic, 4, (4, 4, 512), geom, hw_info)
    assert header[SGZ_4D_HEADER_OFFSET:SGZ_4D_HEADER_OFFSET + 16] == bytes(16)
    assert bytes_to_int(header[56:60]) == (8 * 8 * 512 * 4) // 8 // DISK_BLOCK_BYTES
    assert bytes_to_int(header[68:72]) == 25
