import pytest
import seismic_zfp
import segyio

SZ_FILE = 'test_data/small_4bit.sz'
SEGY_FILE = 'test_data/small.sgy'


def test_read_trace_header():
    with seismic_zfp.open(SZ_FILE) as szfile:
        with segyio.open(SEGY_FILE) as segyfile:
            for trace_number in range(25):
                sz_header = szfile.header[trace_number]
                segy_header = segyfile.header[trace_number]
                assert sz_header == segy_header


def test_read_bin_header():
    with seismic_zfp.open(SZ_FILE) as szfile:
        with segyio.open(SEGY_FILE) as segyfile:
            assert szfile.bin == segyfile.bin
