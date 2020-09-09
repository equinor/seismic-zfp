import pytest
import seismic_zfp.open
from seismic_zfp.segyio_emulator import SegyioEmulator

SGZ_FILE_1 = 'test_data/small_1bit.sgz'


def test_open_errors():

    with pytest.raises(AssertionError):
        seismic_zfp.open(SGZ_FILE_1, mode='w')

    with pytest.raises(AssertionError):
        seismic_zfp.open(SGZ_FILE_1, mode='r', chunk_cache_size=1.5)

    with pytest.raises(AssertionError):
        seismic_zfp.open(SGZ_FILE_1, mode='r', chunk_cache_size='1.5')

    with pytest.raises(AssertionError):
        seismic_zfp.open(SGZ_FILE_1, mode='r', chunk_cache_size='1')


def test_open_success():
    # Verify that the normal opening mechanisms still work

    with seismic_zfp.open(SGZ_FILE_1) as f:
        assert isinstance(f, SegyioEmulator)

    with seismic_zfp.open(SGZ_FILE_1, chunk_cache_size=4) as f:
        assert isinstance(f, SegyioEmulator)

    with seismic_zfp.open(SGZ_FILE_1, chunk_cache_size=None) as f:
        assert isinstance(f, SegyioEmulator)
