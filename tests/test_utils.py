import numpy as np
import pytest
from seismic_zfp.utils import *


def test_pad():
    assert 8 == pad(5, 4)
    assert 4 == pad(4, 4)


def test_gen_coord_list():
    assert [0, 5] == gen_coord_list(0, 5, 2)
    assert [0, 2, 4] == gen_coord_list(0, 2, 3)


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


def test_bytes_to_int():
    assert b'\x39\x05\x00\x00' == int_to_bytes(1337)
    assert b'\x00\x00\x01\x00' == int_to_bytes(65536)
    assert b'\xff\xff\xff\xff' == int_to_bytes(4294967295)


def test_bytes_to_signed_int():
    assert b'\xff\xff\xff\xff' == signed_int_to_bytes(-1)
    assert b'\xfe\xff\xff\xff' == signed_int_to_bytes(-2)
    assert b'\x00\x01\x00\x00' == signed_int_to_bytes(256)
    assert b'\x00\x02\x00\x00' == signed_int_to_bytes(512)
    assert b'\x00\x00\x01\x00' == signed_int_to_bytes(65536)


def test_define_blockshape():
    assert 4, (4, 4, 512) == define_blockshape(4, (4, 4, 512))
    assert 4, (4, 4, 512) == define_blockshape(4, (4, 4, -1))
    assert 4, (4, 4, 512) == define_blockshape(4, (4, -1, 512))
    assert 4, (4, 4, 512) == define_blockshape(4, (-1, 4, 512))
    assert 4, (4, 4, 512) == define_blockshape(-1, (4, 4, 512))
    assert 2, (64, 64, 4) == define_blockshape(-1, (64, 64, 4))

    assert 0.5, (4, 4, 4096) == define_blockshape(0.5, (4, 4, -1))
    assert 0.25, (4, 4, 8192) == define_blockshape(0.25, (4, 4, -1))

    with pytest.raises(ValueError):
        define_blockshape(-1, (4, 4, -1))
    with pytest.raises(AssertionError):
        define_blockshape(1, (4, 4, 128))
