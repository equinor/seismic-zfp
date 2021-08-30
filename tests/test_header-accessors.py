import pytest
import seismic_zfp
import segyio

SGZ_FILE = 'test_data/small_4bit.sgz'
SGY_FILE = 'test_data/small.sgy'


def test_read_trace_header():
    with seismic_zfp.open(SGZ_FILE) as sgzfile:
        with segyio.open(SGY_FILE) as sgyfile:
            for trace_number in range(-5, 25, 1):
                sgz_header = sgzfile.header[trace_number]
                sgy_header = sgyfile.header[trace_number]
                assert sgz_header == sgy_header


def test_read_trace_header_slicing():
    slices = [slice(0, 5, None), slice(0, None, 2), slice(5, None, -1), slice(None, None, 10), slice(None, None, None)]
    with seismic_zfp.open(SGZ_FILE) as sgzfile:
        with segyio.open(SGY_FILE) as sgyfile:
            for slice_ in slices:
                sgy_headers = sgyfile.header[slice_]
                sgz_headers = sgzfile.header[slice_]
                for sgz_header, sgy_header in zip(sgz_headers, sgy_headers):
                    assert sgz_header == sgy_header


def test_header_is_iterable():
    with seismic_zfp.open(SGZ_FILE) as sgz_file:
        with segyio.open(SGY_FILE) as sgy_file:
            for sgz_header, sgy_header in zip(sgz_file.header, sgy_file.header):
                assert sgz_header == sgy_header


def test_read_bin_header():
    with seismic_zfp.open(SGZ_FILE) as sgzfile:
        with segyio.open(SGY_FILE) as segyfile:
            assert sgzfile.bin == segyfile.bin
