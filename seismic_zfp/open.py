from .segyio_emulator import SegyioEmulator


def open(filename, mode='r'):
    assert (mode == 'r')
    return SegyioEmulator(filename)
