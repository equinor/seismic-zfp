import collections
import hashlib
import os
from queue import Queue

import numpy as np
import segyio
import zfpy
import pytest

from seismic_zfp.conversion import SegyConverter
from seismic_zfp.conversion_utils import make_header_seismic_file, io_thread_func_4d, seismic_file_producer_4d
from seismic_zfp.headers import HeaderwordInfo
from seismic_zfp.seismicfile import SeismicFile
from seismic_zfp.sgzconstants import DISK_BLOCK_BYTES, SGZ_4D_HEADER_OFFSET, HEADER_DETECTION_CODES
from seismic_zfp.utils import (Geometry4d, Geometry3d, bytes_to_int, bytes_to_signed_int,
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


def test_io_thread_func_4d_full_inline_set():
    cube = segyio.tools.cube(SGY_FILE_4D)
    blockshape = (4, 4, 4, 128)
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 0, 5)
        headers_dict = blank_headers_dict(125)
        buffer = np.zeros((4, 8, 8, 128), dtype=np.float32)
        io_thread_func_4d(blockshape, True, headers_dict, geom, 0, 4, buffer, seismic, 36)

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


def test_io_thread_func_4d_partial_inline_set_repeats_last_plane():
    cube = segyio.tools.cube(SGY_FILE_4D)
    blockshape = (4, 4, 4, 128)
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 0, 5)
        headers_dict = blank_headers_dict(125)
        buffer = np.zeros((4, 8, 8, 128), dtype=np.float32)
        # Second plane-set only has inline 4 to read
        io_thread_func_4d(blockshape, True, headers_dict, geom, 1, 1, buffer, seismic, 36)

        for i in range(4):
            assert np.array_equal(buffer[i, 0:5, 0:5, 0:36], cube[4])
        exp = expected_headers(cube.shape, geom, seismic)
        for tf in headers_dict:
            assert np.array_equal(headers_dict[tf][100:125], exp[tf][100:125])
            assert np.all(headers_dict[tf][0:100] == 0)


def test_io_thread_func_4d_cropped():
    cube = segyio.tools.cube(SGY_FILE_4D)
    blockshape = (4, 4, 4, 128)
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(1, 4, 2, 5, 1, 3)   # 3 il x 3 xl x 2 offsets = 18 traces
        headers_dict = blank_headers_dict(18)
        buffer = np.zeros((4, 4, 4, 128), dtype=np.float32)
        io_thread_func_4d(blockshape, True, headers_dict, geom, 0, 3, buffer, seismic, 36)

        assert np.array_equal(buffer[0:3, 0:3, 0:2, 0:36], cube[1:4, 2:5, 1:3])
        assert np.array_equal(buffer[3, 0:3, 0:2, 0:36], cube[3, 2:5, 1:3])
        exp = expected_headers(cube.shape, geom, seismic)
        for tf in headers_dict:
            assert np.array_equal(headers_dict[tf], exp[tf])
        assert np.array_equal(headers_dict[OFFSET], np.tile([2, 3], 9))


def test_io_thread_func_4d_store_headers_false():
    with SeismicFile.open(SGY_FILE_4D) as seismic:
        geom = Geometry4d(0, 5, 0, 5, 0, 5)
        headers_dict = blank_headers_dict(125)
        buffer = np.zeros((4, 8, 8, 128), dtype=np.float32)
        io_thread_func_4d((4, 4, 4, 128), False, headers_dict, geom, 0, 4, buffer, seismic, 36)
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


def test_segy_converter_4d_reduce_iops_warns(tmp_path):
    out_sgz = os.path.join(str(tmp_path), 'small-4d-iops.sgz')
    with SegyConverter(SGY_FILE_4D) as converter:
        with pytest.warns(UserWarning, match="not supported for 4D"):
            converter.run(out_sgz, bits_per_voxel=16, reduce_iops=True)
    assert np.allclose(parse_sgz_4d(out_sgz)['volume'], segyio.tools.cube(SGY_FILE_4D), rtol=1e-6)


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
