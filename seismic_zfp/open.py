from .segyio_emulator import SegyioEmulator


def open(filename):
    return SegyioEmulator(filename)
