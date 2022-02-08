import numpy as np
import pytest
from importlib import reload
from unittest import mock
from seismic_zfp.read import *
import segyio

SGZ_FILE_025 = 'test_data/small_025bit.sgz'
SGY_FILE = 'test_data/small.sgy'

def test_raises_import_error_if_missing_azure_storage_blob():
    # Modifies imports. Have to reload before and after
    # to not affect other tests
    with mock.patch.dict('sys.modules', {'azure.storage.blob': None}):
        reload(seismic_zfp.read) 
        with pytest.raises(ImportError) as import_error:
            service_url = ""
            blob_container = ""
            blob_name = ""
            seismic_zfp.read.SgzReader((service_url, blob_container, blob_name))
        assert "requires azure-storage-blob" in str(import_error)
    reload(seismic_zfp.read) 


def compare_inline_blob(sgz_filename, sgy_filename, tolerance):
    with mock.patch('seismic_zfp.utils.read_range_blob', seismic_zfp.utils.read_range_file) as mocked_read_range_blob:
        with open(sgz_filename, 'rb') as sgz_filename:
            sgz_filename.download_blob = 'mock'
            sgz_filename.blob_name = 'bob'
            reader = SgzReader(sgz_filename)
            with segyio.open(sgy_filename) as segyfile:
                for line_number in range(5):
                    slice_sgz = reader.read_inline(line_number)
                    slice_segy = segyfile.iline[segyfile.ilines[line_number]]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)
                assert reader.local == False



def test_read_inline_from_blob():
    compare_inline_blob(SGZ_FILE_025, SGY_FILE, tolerance=1e+1)
