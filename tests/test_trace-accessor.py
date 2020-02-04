import numpy as np
import pytest
import seismic_zfp
import segyio

SZ_FILE = 'test_data/small_4bit.sz'
SEGY_FILE = 'test_data/small.sgy'


def test_read_trace():
    with seismic_zfp.open(SZ_FILE) as szfile:
        with segyio.open(SEGY_FILE) as segyfile:
            for trace_number in range(25):
                sz_trace = szfile.trace[trace_number]
                segy_trace = segyfile.trace[trace_number]
                assert np.allclose(sz_trace, segy_trace, rtol=1e-6)
