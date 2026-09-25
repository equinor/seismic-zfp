import os
import types
import numpy as np
import segyio
import pytest

import seismic_zfp
from seismic_zfp.conversion import SegyConverter, SgzConverter
from seismic_zfp.read import SgzReader
from seismic_zfp.utils import WrongDimensionalityError

SGY_FILE_4D = 'test_data/small-4d.sgy'
SGY_FILE_4D_IRREG = 'test_data/small-4d-irregular.sgy'
SGZ_FILE_4D = 'test_data/small-4d_4bit.sgz'
SGY_FILE_3D = 'test_data/small.sgy'
SGZ_FILE_3D = 'test_data/small_4bit.sgz'
SGZ_FILE_2D = 'test_data/small-2d.sgz'

IL, XL, OFFSET = segyio.TraceField.INLINE_3D, segyio.TraceField.CROSSLINE_3D, segyio.TraceField.offset
MISSING_4D = [(4, 4, o) for o in range(5)] + [(1, 2, 0), (2, 0, 4)]


# --- Stored reference file: guards against symmetric regressions in writer and reader -----------------

def test_stored_4d_file_reads_correctly():
    cube = segyio.tools.cube(SGY_FILE_4D)
    with SgzReader(SGZ_FILE_4D) as reader:
        assert reader.is_4d and reader.structured
        assert reader.blockshape == (4, 4, 4, 128)
        assert reader.rate == 4
        assert (reader.n_ilines, reader.n_xlines, reader.n_offsets, reader.n_samples) == (5, 5, 5, 36)
        assert np.array_equal(reader.offsets, [1, 2, 3, 4, 5])
        assert reader.get_source_data_hash() == '79a52b641244ae2435032bfff6e918782f775683'
        assert np.allclose(reader.read_volume(), cube, rtol=1e-5)
        assert np.allclose(reader.read_gather(2, 3), cube[2, 3], rtol=1e-5)
        assert np.allclose(reader.read_offset(4), cube[:, :, 4], rtol=1e-5)
        with segyio.open(SGY_FILE_4D) as segyfile:
            for i in range(0, 125, 7):
                assert np.allclose(reader.get_trace(i), segyfile.trace[i], rtol=1e-5)
                assert reader.gen_trace_header(i) == segyfile.header[i]


def test_stored_4d_file_reproduced_by_converter(tmp_path):
    """Writer output must be byte-identical to the committed file, except for the library version"""
    out_sgz = os.path.join(str(tmp_path), 'small-4d_4bit.sgz')
    with SegyConverter(SGY_FILE_4D) as converter:
        converter.run(out_sgz, bits_per_voxel=4)
    with open(SGZ_FILE_4D, 'rb') as f:
        stored = bytearray(f.read())
    with open(out_sgz, 'rb') as f:
        fresh = bytearray(f.read())
    assert len(fresh) == len(stored) == 42496
    stored[72:76] = fresh[72:76] = bytes(4)   # encoded version number
    assert fresh == stored

# (name, source SEG-Y, bits_per_voxel, blockshape, tolerance)
# Lossy layouts on the irregular file smear around the zero-filled holes, hence the looser tolerances
LAYOUTS = [
    ('regular-16bit-default', SGY_FILE_4D, 16, None, dict(rtol=1e-5)),
    ('regular-4bit-default', SGY_FILE_4D, 4, None, dict(rtol=1e-5)),
    ('regular-2bit-sliced', SGY_FILE_4D, 2, (8, 8, 4, 64), dict(rtol=1e-5)),
    ('irregular-16bit-default', SGY_FILE_4D_IRREG, 16, None, dict(rtol=1e-5, atol=1e-3)),
    ('irregular-2bit-sliced', SGY_FILE_4D_IRREG, 2, (8, 8, 4, 64), dict(rtol=0.2, atol=45)),
]


def reference_cube(sgy_file):
    """(il, xl, offset, sample) reference, with removed traces zeroed for the irregular file"""
    cube = segyio.tools.cube(SGY_FILE_4D).copy()
    if sgy_file == SGY_FILE_4D_IRREG:
        for il, xl, off in MISSING_4D:
            cube[il, xl, off] = 0
    return cube


_converted = {}


def build_layout(param, tmp_path_factory):
    """Convert once per layout for the module, whichever fixture asks first"""
    name, sgy_file, bits_per_voxel, blockshape, tolerance = param
    if name not in _converted:
        out_sgz = os.path.join(str(tmp_path_factory.mktemp('sgz4d')), name + '.sgz')
        with SegyConverter(sgy_file) as converter:
            converter.run(out_sgz, bits_per_voxel=bits_per_voxel, blockshape=blockshape)
        _converted[name] = dict(name=name, sgy=sgy_file, sgz=out_sgz, cube=reference_cube(sgy_file), tol=tolerance,
                                blockshape=blockshape or (4, 4, 4, 32768 // (64 * bits_per_voxel)))
    return _converted[name]


@pytest.fixture(scope='module', params=LAYOUTS, ids=[layout[0] for layout in LAYOUTS])
def layout(request, tmp_path_factory):
    return build_layout(request.param, tmp_path_factory)


REGULAR_LAYOUTS = [layout for layout in LAYOUTS if layout[1] == SGY_FILE_4D]
IRREGULAR_LAYOUTS = [layout for layout in LAYOUTS if layout[1] == SGY_FILE_4D_IRREG]


@pytest.fixture(scope='module', params=REGULAR_LAYOUTS, ids=[layout[0] for layout in REGULAR_LAYOUTS])
def regular_layout(request, tmp_path_factory):
    return build_layout(request.param, tmp_path_factory)


@pytest.fixture(scope='module', params=IRREGULAR_LAYOUTS, ids=[layout[0] for layout in IRREGULAR_LAYOUTS])
def irregular_layout(request, tmp_path_factory):
    return build_layout(request.param, tmp_path_factory)


@pytest.fixture(params=[False, True], ids=['ondisk', 'preload'])
def reader(request, layout):
    with SgzReader(layout['sgz'], preload=request.param) as r:
        yield r


def test_reader_properties(layout, reader):
    assert reader.is_4d and not reader.is_3d and not reader.is_2d
    assert (reader.n_ilines, reader.n_xlines, reader.n_offsets, reader.n_samples) == (5, 5, 5, 36)
    assert np.array_equal(reader.ilines, [11, 12, 13, 14, 15])
    assert np.array_equal(reader.xlines, [21, 22, 23, 24, 25])
    assert np.array_equal(reader.offsets, [1, 2, 3, 4, 5])
    assert np.array_equal(reader.zslices, np.arange(36.0))
    assert reader.blockshape == layout['blockshape']
    assert reader.structured == (layout['sgy'] == SGY_FILE_4D)
    assert reader.tracecount == (125 if reader.structured else 118)
    assert reader.n_grid_traces == 125
    assert reader.unit_bytes == 256 * reader.rate // 8
    assert reader.block_bytes == 4096
    assert 'seismic-zfp 4d file' in str(reader)
    assert f'offsets: 5 [1, 5]' in str(reader)


def test_read_volume(layout, reader):
    volume = reader.read_volume()
    assert volume.shape == (5, 5, 5, 36)
    assert np.allclose(volume, layout['cube'], **layout['tol'])


def test_reader_matches_raw_decode_exactly(layout, reader):
    """Independently of compression accuracy, every read path must reproduce the file contents bit-exactly"""
    from tests.test_compress_4d import parse_sgz_4d
    info = parse_sgz_4d(layout['sgz'])
    volume, padded = info['volume'], info['padded_volume']
    assert np.array_equal(reader.read_volume(), volume)
    assert all(np.array_equal(reader.read_inline(i), volume[i]) for i in range(5))
    assert all(np.array_equal(reader.read_crossline(x), volume[:, x]) for x in range(5))
    assert all(np.array_equal(reader.read_gather(i, x), volume[i, x]) for i in range(5) for x in range(5))
    assert all(np.array_equal(reader.read_offset(o), volume[:, :, o]) for o in range(5))
    assert all(np.array_equal(reader.read_zslice(z), volume[:, :, :, z]) for z in range(36))
    assert np.array_equal(reader.read_subvolume_4d(1, 4, 2, 5, 1, 3, 5, 30), volume[1:4, 2:5, 1:3, 5:30])
    assert np.array_equal(reader.read_subvolume_4d(0, *reader.shape_pad[0:1], 0, reader.shape_pad[1],
                                                   0, reader.shape_pad[2], 0, reader.shape_pad[3],
                                                   access_padding=True), padded)
    grid_traces = volume.reshape(125, 36)
    present = np.flatnonzero(info['header_arrays'][IL]) if not reader.structured else np.arange(125)
    assert all(np.array_equal(reader.get_trace(i), grid_traces[present[i]]) for i in range(reader.tracecount))


def test_read_inline(layout, reader):
    for il_id in range(5):
        inline = reader.read_inline(il_id)
        assert inline.shape == (5, 5, 36)
        assert np.allclose(inline, layout['cube'][il_id], **layout['tol'])
        assert np.array_equal(reader.read_inline_number(reader.ilines[il_id]), inline)


def test_read_crossline(layout, reader):
    for xl_id in range(5):
        crossline = reader.read_crossline(xl_id)
        assert crossline.shape == (5, 5, 36)
        assert np.allclose(crossline, layout['cube'][:, xl_id], **layout['tol'])
        assert np.array_equal(reader.read_crossline_number(reader.xlines[xl_id]), crossline)


def test_read_gather(layout, reader):
    for il_id in range(5):
        for xl_id in range(5):
            gather = reader.read_gather(il_id, xl_id)
            assert gather.shape == (5, 36)
            assert np.allclose(gather, layout['cube'][il_id, xl_id], **layout['tol'])
            assert np.array_equal(reader.read_gather_number(reader.ilines[il_id], reader.xlines[xl_id]), gather)


def test_read_offset(layout, reader):
    for offset_id in range(5):
        volume = reader.read_offset(offset_id)
        assert volume.shape == (5, 5, 36)
        assert np.allclose(volume, layout['cube'][:, :, offset_id], **layout['tol'])
        assert np.array_equal(reader.read_offset_number(reader.offsets[offset_id]), volume)


def test_read_zslice(layout, reader):
    for z_id in range(36):
        zslice = reader.read_zslice(z_id)
        assert zslice.shape == (5, 5, 5)
        assert np.allclose(zslice, layout['cube'][:, :, :, z_id], **layout['tol'])
    assert np.array_equal(reader.read_zslice_coord(reader.zslices[7]), reader.read_zslice(7))


def test_read_subvolume_4d(layout, reader):
    cube = layout['cube']
    cases = [(1, 4, 2, 5, 1, 3, 5, 30),    # interior
             (0, 5, 0, 5, 0, 5, 0, 36),    # everything
             (4, 5, 4, 5, 4, 5, 35, 36),   # single voxel in the last block
             (0, 1, 0, 5, 0, 5, 0, 1),     # single sample plane of one inline
             (3, 5, 3, 5, 3, 5, 30, 36)]   # straddling blocks / padding
    for min_il, max_il, min_xl, max_xl, min_off, max_off, min_z, max_z in cases:
        for multithreading in (False, True):
            sub = reader.read_subvolume_4d(min_il, max_il, min_xl, max_xl, min_off, max_off, min_z, max_z,
                                           multithreading=multithreading)
            assert sub.shape == (max_il - min_il, max_xl - min_xl, max_off - min_off, max_z - min_z)
            assert np.allclose(sub, cube[min_il:max_il, min_xl:max_xl, min_off:max_off, min_z:max_z], **layout['tol'])


def test_read_subvolume_4d_access_padding(layout, reader):
    padded = reader.read_subvolume_4d(0, reader.shape_pad[0], 0, reader.shape_pad[1],
                                      0, reader.shape_pad[2], 0, reader.shape_pad[3], access_padding=True)
    assert padded.shape == reader.shape_pad
    assert np.allclose(padded[0:5, 0:5, 0:5, 0:36], layout['cube'], **layout['tol'])
    with pytest.raises(IndexError):
        reader.read_subvolume_4d(0, reader.shape_pad[0], 0, 5, 0, 5, 0, 36)


def test_get_trace_and_headers(layout, reader):
    """Trace indices follow the source SEG-Y trace order, skipping missing traces for irregular files"""
    with segyio.open(layout['sgy'], strict=False, ignore_geometry=True) as segyfile:
        assert reader.tracecount == segyfile.tracecount
        for i in range(segyfile.tracecount):
            assert np.allclose(reader.get_trace(i), segyfile.trace[i], **layout['tol'])
            assert reader.gen_trace_header(i) == segyfile.header[i]
        # Cropped by sample index and by sample coordinate
        assert np.allclose(reader.get_trace(7, 10, 20), segyfile.trace[7][10:20], **layout['tol'])
        assert np.allclose(reader.get_trace_by_coord(7, 10.0, 20.0), segyfile.trace[7][10:20], **layout['tol'])


def test_get_trace_out_of_range(reader):
    with pytest.raises(IndexError):
        reader.get_trace(reader.tracecount)
    with pytest.raises(IndexError):
        reader.gen_trace_header(reader.n_grid_traces)


def test_tracefield_values(layout, reader):
    present = np.ones((5, 5, 5), dtype=bool)
    if not reader.structured:
        for il, xl, off in MISSING_4D:
            present[il, xl, off] = False
    grid_il = np.broadcast_to(np.array([11, 12, 13, 14, 15])[:, None, None], (5, 5, 5))
    grid_xl = np.broadcast_to(np.array([21, 22, 23, 24, 25])[None, :, None], (5, 5, 5))
    grid_off = np.broadcast_to(np.array([1, 2, 3, 4, 5])[None, None, :], (5, 5, 5))
    for tf, grid in ((IL, grid_il), (XL, grid_xl), (OFFSET, grid_off)):
        values = reader.get_tracefield_values(tf)
        assert values.shape == (5, 5, 5)
        assert np.array_equal(values, np.where(present, grid, 0))

    # Without padding, variant headers are the present traces only
    reader.clear_variant_headers()
    reader.read_variant_headers()
    assert len(reader.variant_headers[IL]) == reader.tracecount


def test_unstructured_mask(layout, reader):
    reader.get_unstructured_mask()
    expected = np.ones(125, dtype=bool)
    if not reader.structured:
        for il, xl, off in MISSING_4D:
            expected[(il * 5 + xl) * 5 + off] = False
    assert np.array_equal(reader.mask, expected)
    assert np.count_nonzero(reader.mask) == reader.tracecount


def test_index_errors(reader):
    for bad_call in (lambda: reader.read_inline(5), lambda: reader.read_crossline(5), lambda: reader.read_zslice(36),
                     lambda: reader.read_offset(5), lambda: reader.read_gather(5, 0), lambda: reader.read_gather(0, 5),
                     lambda: reader.read_subvolume_4d(0, 6, 0, 5, 0, 5, 0, 36),
                     lambda: reader.read_subvolume_4d(0, 5, 0, 5, 3, 3, 0, 36),
                     lambda: reader.read_subvolume_4d(0, 5, 0, 5, 0, 5, 0, 37),
                     lambda: reader.read_inline_number(10), lambda: reader.read_offset_number(6)):
        with pytest.raises(IndexError):
            bad_call()


def test_3d_only_methods_rejected(reader):
    for bad_call in (lambda: reader.read_subvolume(0, 5, 0, 5, 0, 36),
                     lambda: reader.read_correlated_diagonal(0),
                     lambda: reader.read_anticorrelated_diagonal(0),
                     lambda: reader.read_subplane(0, 5, 0, 36)):
        with pytest.raises(WrongDimensionalityError):
            bad_call()


def test_4d_only_methods_rejected_for_3d_and_2d():
    with SgzReader(SGZ_FILE_3D) as reader_3d:
        assert not reader_3d.is_4d and reader_3d.is_3d
        assert reader_3d.offsets is None
        assert reader_3d.n_offsets == 0
        for bad_call in (lambda: reader_3d.read_gather(0, 0), lambda: reader_3d.read_offset(0),
                         lambda: reader_3d.read_subvolume_4d(0, 5, 0, 5, 0, 1, 0, 50)):
            with pytest.raises(WrongDimensionalityError):
                bad_call()
    with SgzReader(SGZ_FILE_2D) as reader_2d:
        assert not reader_2d.is_4d and reader_2d.is_2d
        with pytest.raises(WrongDimensionalityError):
            reader_2d.read_gather(0, 0)


def test_trace_cache_reuse(layout):
    """Sequential access over a gather should only decompress its containing chunk once"""
    with SgzReader(layout['sgz']) as reader:
        for i in range(25):
            reader.get_trace(i)
        info = reader._read_containing_chunk_cached.cache_info()
    # 25 traces = 5 gathers x 5 offsets, all in one 4x4 (il, xl) block for default layout
    assert info.misses <= 2
    assert info.hits >= 23


def test_segyio_emulator_4d(layout):
    with seismic_zfp.open(layout['sgz']) as sgzfile, \
            segyio.open(layout['sgy'], strict=False, ignore_geometry=True) as segyfile:
        assert sgzfile.is_4d
        assert sgzfile.unstructured == (layout['sgy'] == SGY_FILE_4D_IRREG)
        assert len(sgzfile.trace) == segyfile.tracecount
        assert np.allclose(sgzfile.trace[3], segyfile.trace[3], **layout['tol'])
        assert sgzfile.header[3] == segyfile.header[3]
        assert np.array_equal(sgzfile.attributes(OFFSET), segyfile.attributes(OFFSET)[:]) or not sgzfile.structured
        assert np.array_equal(sgzfile.samples, segyfile.samples)
        assert np.array_equal(sgzfile.offsets, [1, 2, 3, 4, 5])
        with pytest.raises(WrongDimensionalityError, match="4D"):
            sgzfile.subvolume[11:13, 21:23, 0:10]


def test_segyio_emulator_4d_matches_segyio_prestack_accessors(regular_layout):
    """iline/xline/depth_slice/gather behave as segyio does for a prestack file"""
    layout = regular_layout
    tol = layout['tol']
    with seismic_zfp.open(layout['sgz']) as sgz, segyio.open(layout['sgy']) as sgy:
        assert len(sgz.iline) == len(sgy.iline) and len(sgz.xline) == len(sgy.xline)
        assert len(sgz.depth_slice) == len(sgy.depth_slice)
        for il in sgy.ilines:
            assert np.allclose(sgz.iline[il], sgy.iline[il], **tol)             # first offset
            for off in sgy.offsets:
                assert np.allclose(sgz.iline[il, off], sgy.iline[il, off], **tol)
        for xl in sgy.xlines:
            assert np.allclose(sgz.xline[xl], sgy.xline[xl], **tol)
            assert np.allclose(sgz.xline[xl, 4], sgy.xline[xl, 4], **tol)
        for z in range(36):
            assert np.allclose(sgz.depth_slice[z], sgy.depth_slice[z], **tol)   # first offset
        for il in sgy.ilines:
            for xl in sgy.xlines:
                assert np.allclose(sgz.gather[il, xl], sgy.gather[il, xl], **tol)
                assert np.allclose(sgz.gather[il, xl, 3], sgy.gather[il, xl, 3], **tol)
                assert np.allclose(sgz.gather[il, xl, 2:5], sgy.gather[il, xl, 2:5], **tol)
                assert np.allclose(sgz.gather[il, xl, 1:6:2], sgy.gather[il, xl, 1:6:2], **tol)

        # Slices produce a generator of one array per combination, in the same order as segyio
        for sgz_result, sgy_result in ((sgz.iline[12:14], sgy.iline[12:14]),
                                       (sgz.iline[12:14, 2:4], sgy.iline[12:14, 2:4]),
                                       (sgz.iline[:, 5], sgy.iline[:, 5]),
                                       (sgz.xline[22:25:2, :], sgy.xline[22:25:2, :]),
                                       (sgz.gather[12:14, 21], sgy.gather[12:14, 21]),
                                       (sgz.gather[11, 21:23], sgy.gather[11, 21:23]),
                                       (sgz.gather[12:14, 21:23], sgy.gather[12:14, 21:23]),
                                       (sgz.gather[12:14, 21, 2:4], sgy.gather[12:14, 21, 2:4]),
                                       (sgz.gather[12:14, 21:23, 3], sgy.gather[12:14, 21:23, 3]),
                                       (sgz.gather[:, :], sgy.gather[:, :])):
            assert isinstance(sgz_result, types.GeneratorType)
            assert isinstance(sgy_result, types.GeneratorType)
            # segyio's line generators reuse one buffer, so copy each item as it is yielded
            sgz_list, sgy_list = list(sgz_result), [b.copy() for b in sgy_result]
            assert len(sgz_list) == len(sgy_list)
            for a, b in zip(sgz_list, sgy_list):
                assert a.shape == b.shape
                assert np.allclose(a, b, **tol)

        for bad_call in (lambda: sgz.iline[11, 9], lambda: sgz.iline[10], lambda: sgz.xline[21, 0],
                         lambda: sgz.gather[11, 26], lambda: sgz.gather[16, 21, 1], lambda: sgz.depth_slice[36]):
            with pytest.raises(IndexError):
                bad_call()
        with pytest.raises(TypeError):
            sgz.gather[11]


def test_segyio_emulator_4d_irregular_first_offset(irregular_layout):
    """Irregular files still provide first-offset lines; missing traces read as zero"""
    layout = irregular_layout
    cube = layout['cube']
    with seismic_zfp.open(layout['sgz']) as sgz:
        assert np.allclose(sgz.iline[12], cube[1, :, 0, :], **layout['tol'])
        assert np.allclose(sgz.xline[21, 5], cube[:, 0, 4, :], **layout['tol'])
        assert np.allclose(sgz.gather[15, 25], np.zeros((5, 36)), atol=layout['tol'].get('atol', 1e-6))
        assert np.allclose(sgz.depth_slice[0], cube[:, :, 0, 0], **layout['tol'])


def test_segyio_emulator_3d_has_no_gather():
    with seismic_zfp.open(SGZ_FILE_3D) as sgzfile:
        with pytest.raises(WrongDimensionalityError):
            sgzfile.gather[1, 20]


def test_sgz_converter_to_segy_rejected(layout):
    with SgzConverter(layout['sgz']) as converter:
        with pytest.raises(NotImplementedError):
            converter.convert_to_segy(os.path.join(os.path.dirname(layout['sgz']), 'no.sgy'))
