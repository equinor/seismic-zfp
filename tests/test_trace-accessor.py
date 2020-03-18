import numpy as np
import pytest
import seismic_zfp
import segyio

SGZ_FILE = 'test_data/small_4bit.sgz'
SGY_FILE = 'test_data/small.sgy'


def test_read_trace():
    with seismic_zfp.open(SGZ_FILE) as sgzfile:
        with segyio.open(SGY_FILE) as segyfile:
            for trace_number in range(25):
                sgz_trace = sgzfile.trace[trace_number]
                segy_trace = segyfile.trace[trace_number]
                assert np.allclose(sgz_trace, segy_trace, rtol=1e-6)
