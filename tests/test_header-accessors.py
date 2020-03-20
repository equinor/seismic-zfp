import pytest
import seismic_zfp
import segyio

SGZ_FILE = 'test_data/small_4bit.sgz'
SGY_FILE = 'test_data/small.sgy'


def test_read_trace_header():
    with seismic_zfp.open(SGZ_FILE) as sgzfile:
        with segyio.open(SGY_FILE) as segyfile:
            for trace_number in range(25):
                sgz_header = sgzfile.header[trace_number]
                sgy_header = segyfile.header[trace_number]
                assert sgz_header == sgy_header


def test_read_bin_header():
    with seismic_zfp.open(SGZ_FILE) as sgzfile:
        with segyio.open(SGY_FILE) as segyfile:
            assert sgzfile.bin == segyfile.bin
