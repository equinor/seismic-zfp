import numpy as np
import pytest
from seismic_zfp.utils import *


def test_pad():
    assert 8 == pad(5, 4)
    assert 4 == pad(4, 4)


def test_coord_to_index():
    assert 0 == coord_to_index(1, np.arange(1, 6, dtype=np.int32))
    assert 1 == coord_to_index(2, np.arange(0, 10, 2, dtype=np.int32))
    assert 2 == coord_to_index(120.0, np.arange(100, 200, 10, dtype=float))

    with pytest.raises(IndexError):
        coord_to_index(6, np.arange(1, 6, dtype=np.int32))


def test_gen_coord_list():
    assert np.all(np.arange(0, 10, 5) == gen_coord_list(0, 5, 2))
    assert np.all(np.arange(0, 6, 2) == gen_coord_list(0, 2, 3))


def test_bytes_to_double():
    assert 1337.7331 == bytes_to_double(b'\xCA\x32\xC4\xB1\xEE\xE6\x94\x40')


def test_double_to_bytes():
    assert b'\xCA\x32\xC4\xB1\xEE\xE6\x94\x40' == double_to_bytes(1337.7331)


def test_np_float_to_bytes():
    assert b'\x0f\x00\00\00' == np_float_to_bytes(np.single(15.0))
    assert b'\x39\x05\00\00' == np_float_to_bytes(np.single(1337.0))
    assert b'\x00\x00\00\00' == np_float_to_bytes(np.single(0.5))


def test_bytes_to_int():
    assert 1337 == bytes_to_int(b'\x39\x05\x00\x00')
    assert 65536 == bytes_to_int(b'\x00\x00\x01\x00')
    assert 4294967295 == bytes_to_int(b'\xff\xff\xff\xff')


def test_bytes_to_signed_int():
    assert -1 == bytes_to_signed_int(b'\xff\xff\xff\xff')
    assert -2 == bytes_to_signed_int(b'\xfe\xff\xff\xff')
    assert 256 == bytes_to_signed_int(b'\x00\x01\x00\x00')
    assert 512 == bytes_to_signed_int(b'\x00\x02\x00\x00')
    assert 65536 == bytes_to_signed_int(b'\x00\x00\x01\x00')
    assert 128 == bytes_to_signed_int(b'\x80\x00')
    assert -42 == bytes_to_signed_int(b'\xd6\xff')


def test_int_to_bytes():
    assert b'\x39\x05\x00\x00' == int_to_bytes(1337)
    assert b'\x00\x00\x01\x00' == int_to_bytes(65536)
    assert b'\xff\xff\xff\xff' == int_to_bytes(4294967295)


def test_signed_int_to_bytes():
    assert b'\xff\xff\xff\xff' == signed_int_to_bytes(-1)
    assert b'\xfe\xff\xff\xff' == signed_int_to_bytes(-2)
    assert b'\x00\x01\x00\x00' == signed_int_to_bytes(256)
    assert b'\x00\x02\x00\x00' == signed_int_to_bytes(512)
    assert b'\x00\x00\x01\x00' == signed_int_to_bytes(65536)


def test_define_blockshape_2d():
    assert (4, (1, 16, 512)) == define_blockshape_2d(4, (1, 16, 512))
    assert (4, (1, 16, 512)) == define_blockshape_2d(4, (1, 16, -1))
    assert (4, (1, 16, 512)) == define_blockshape_2d(4, (1, -1, 512))
    assert (4, (1, 16, 512)) == define_blockshape_2d(-1, (1, 16, 512))
    assert (4, (1, 16, 512)) == define_blockshape_2d("4", (1, 16, 512))

    with pytest.raises(ValueError):
        define_blockshape_2d(4, (1, -1, -1))

    with pytest.raises(AssertionError):
        define_blockshape_2d(4, (1, 16, 16))

    with pytest.raises(AssertionError):
        define_blockshape_2d(4, (4, 4, -1))


def test_define_blockshape_3d():
    assert (4, (4, 4, 512)) == define_blockshape_3d(4, (4, 4, 512))
    assert (4, (4, 4, 512)) == define_blockshape_3d("4", (4, 4, 512))
    assert (4, (4, 4, 512)) == define_blockshape_3d(4, (4, 4, -1))
    assert (4, (4, 4, 512)) == define_blockshape_3d("4", (4, 4, -1))
    assert (4, (4, 4, 512)) == define_blockshape_3d(4, (4, -1, 512))
    assert (4, (4, 4, 512)) == define_blockshape_3d("4", (4, -1, 512))
    assert (4, (4, 4, 512)) == define_blockshape_3d(4, (-1, 4, 512))
    assert (4, (4, 4, 512)) == define_blockshape_3d("4", (-1, 4, 512))
    assert (4, (4, 4, 512)) == define_blockshape_3d(-1, (4, 4, 512))
    assert (4, (4, 4, 512)) == define_blockshape_3d("-1", (4, 4, 512))
    assert (2, (64, 64, 4)) == define_blockshape_3d(-1, (64, 64, 4))
    assert (2, (64, 64, 4)) == define_blockshape_3d("-1", (64, 64, 4))
    assert (0.5, (4, 4, 4096)) == define_blockshape_3d(-2, (4, 4, -1))
    assert (0.5, (4, 4, 4096)) == define_blockshape_3d("-2", (4, 4, -1))
    assert (0.25, (4, 4, 8192)) == define_blockshape_3d(-1, (4, 4, 8192))
    assert (0.25, (4, 4, 8192)) == define_blockshape_3d("-1", (4, 4, 8192))
    assert (0.5, (4, 4, 4096)) == define_blockshape_3d(0.5, (4, 4, -1))
    assert (0.5, (4, 4, 4096)) == define_blockshape_3d("0.5", (4, 4, -1))
    assert (0.25, (4, 4, 8192)) == define_blockshape_3d(0.25, (4, 4, -1))
    assert (0.25, (4, 4, 8192)) == define_blockshape_3d("0.25", (4, 4, -1))

    with pytest.raises(ValueError):
        define_blockshape_3d(-1, (4, 4, -1))
    with pytest.raises(AssertionError):
        define_blockshape_3d(1, (4, 4, 128))
    with pytest.raises(AssertionError):
        define_blockshape_3d(4, (4, 4, 4, -1))


def test_define_blockshape_4d():
    assert (4, (4, 4, 4, 128)) == define_blockshape_4d(4, (4, 4, 4, 128))
    assert (4, (4, 4, 4, 128)) == define_blockshape_4d("4", (4, 4, 4, 128))
    assert (4, (4, 4, 4, 128)) == define_blockshape_4d(4, (4, 4, 4, -1))
    assert (4, (4, 4, 4, 128)) == define_blockshape_4d(4, (4, 4, -1, 128))
    assert (4, (4, 4, 4, 128)) == define_blockshape_4d(4, (4, -1, 4, 128))
    assert (4, (4, 4, 4, 128)) == define_blockshape_4d(4, (-1, 4, 4, 128))
    assert (4, (4, 4, 4, 128)) == define_blockshape_4d(-1, (4, 4, 4, 128))
    assert (8, (4, 4, 4, 64)) == define_blockshape_4d(8, (4, 4, 4, -1))
    assert (2, (4, 4, 4, 256)) == define_blockshape_4d(2, (4, 4, 4, -1))
    assert (2, (16, 16, 4, 16)) == define_blockshape_4d(-1, (16, 16, 4, 16))
    assert (0.5, (4, 4, 4, 1024)) == define_blockshape_4d(-2, (4, 4, 4, -1))
    assert (0.5, (4, 4, 4, 1024)) == define_blockshape_4d(0.5, (4, 4, 4, -1))

    with pytest.raises(ValueError):
        define_blockshape_4d(-1, (4, 4, 4, -1))
    with pytest.raises(ValueError):
        define_blockshape_4d(4, (4, 4, -1, -1))
    with pytest.raises(AssertionError):
        define_blockshape_4d(4, (4, 4, 4, 64))
    with pytest.raises(AssertionError):
        define_blockshape_4d(4, (4, 4, -1))
    # 4D zfp units are 4x4x4x4, dimensions below 4 are not permitted
    with pytest.raises(ValueError):
        define_blockshape_4d(4, (1, 4, 16, -1))
    with pytest.raises(ValueError):
        define_blockshape_4d(4, (4, 4, 2, -1))


def test_geometry_4d():
    geom = Geometry4d(0, 5, 2, 7, 1, 4)
    assert list(geom.ilines) == [0, 1, 2, 3, 4]
    assert list(geom.xlines) == [2, 3, 4, 5, 6]
    assert list(geom.offsets) == [1, 2, 3]
    assert not isinstance(geom, Geometry3d)
    assert not isinstance(geom, Geometry2d)
    assert 'OFFSET:[1,4]' in repr(geom)


def test_inferred_geometry_4d():
    # Decimated IL, regular XL, offsets 100..400 step 100, with two positions missing
    traces_ref = {(il, xl, off): i for i, (il, xl, off) in enumerate(
        (il, xl, off) for il in (10, 12, 14) for xl in (5, 6) for off in (100, 200, 300, 400)
        if (il, xl, off) not in [(12, 6, 100), (14, 5, 400)])}
    geom = InferredGeometry4d(traces_ref)
    assert isinstance(geom, Geometry4d)
    assert not isinstance(geom, Geometry3d)
    assert list(geom.ilines) == [10, 12, 14]
    assert list(geom.xlines) == [5, 6]
    assert list(geom.offsets) == [100, 200, 300, 400]
    assert (geom.min_il, geom.max_il, geom.il_step) == (10, 14, 2)
    assert (geom.min_xl, geom.max_xl, geom.xl_step) == (5, 6, 1)
    assert (geom.min_offset, geom.max_offset, geom.offset_step) == (100, 400, 100)
    assert geom.traces_ref[(10, 5, 100)] == 0
    assert (12, 6, 100) not in geom.traces_ref
    assert repr(geom) == 'IL:[10,14,2] -- XL:[5,6,1] -- OFFSET:[100,400,100]'


def test_inferred_geometry_4d_single_valued_axis():
    traces_ref = {(il, 7, off): 0 for il in (1, 2) for off in (10, 20, 30)}
    geom = InferredGeometry4d(traces_ref)
    assert list(geom.xlines) == [7]
    assert geom.xl_step == 0


def test_inferred_geometry_4d_trace_index():
    # Inline-sorted with IL step 2 and offsets step 100; (12, 6, 100) missing; trace 9 belongs to inline 14
    keys = [(il, xl, off) for il in (10, 12, 14) for xl in (5, 6) for off in (100, 200)]
    keys.remove((12, 6, 100))
    geom = InferredGeometry4d({key: i for i, key in enumerate(keys)})
    assert np.array_equal(geom.inline_trace_ids(0), [0, 1, 2, 3])
    assert np.array_equal(geom.inline_trace_ids(1), [4, 5, 6])
    assert np.array_equal(geom.inline_trace_ids(2), [7, 8, 9, 10])
    assert len(geom.inline_trace_ids(3)) == 0
    xl_ids, off_ids = geom.trace_ordinals(geom.inline_trace_ids(1))
    assert np.array_equal(xl_ids, [0, 0, 1])
    assert np.array_equal(off_ids, [0, 1, 1])

    # Offset-sorted: an inline's traces are spread through the file
    keys = [(il, xl, off) for off in (100, 200) for il in (10, 12) for xl in (5, 6)]
    geom = InferredGeometry4d({key: i for i, key in enumerate(keys)})
    assert np.array_equal(geom.inline_trace_ids(0), [0, 1, 4, 5])
    xl_ids, off_ids = geom.trace_ordinals(geom.inline_trace_ids(0))
    assert np.array_equal(xl_ids, [0, 1, 0, 1])
    assert np.array_equal(off_ids, [0, 0, 1, 1])


def test_get_chunk_cache_size():
    assert 2048 == get_chunk_cache_size(1000, 2000)
    assert 1024 == get_chunk_cache_size(5000, 511)


def test_python_int():
    assert 1 == python_int(1)
    assert 1 == python_int(np.uint16(1))
    assert 1 == python_int(np.int8(1))
    with pytest.raises(TypeError):
        python_int("1")


def test_get_range():
    assert InferredGeometry3d.get_range([0,1,2]) == (0, 2, 1)
    assert InferredGeometry3d.get_range([0]) == (0, 0, 0)
