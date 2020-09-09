from .segyio_emulator import SegyioEmulator


def open(filename, mode='r', chunk_cache_size: int=None):
    assert (mode == 'r')
    assert (isinstance(chunk_cache_size, int) or chunk_cache_size is None)
    return SegyioEmulator(filename, chunk_cache_size)
