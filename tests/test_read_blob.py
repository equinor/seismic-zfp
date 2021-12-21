import pytest
from importlib import reload
from unittest import mock
import seismic_zfp.read


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
