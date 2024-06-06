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


def test_get_chunk_cache_size():
    assert 2048 == get_chunk_cache_size(1000, 2000)
    assert 1024 == get_chunk_cache_size(5000, 511)
