import pytest
from importlib import reload
from unittest import mock
from seismic_zfp import seismicfile
import warnings

try:
    with warnings.catch_warnings():
        # pyzgy will warn us that sdglue is not available. This is expected, and safe for our purposes.
        warnings.filterwarnings("ignore", message="seismic store access is not available: No module named 'sdglue'")
        import pyzgy
except ImportError:
    pyzgy = None

try:
    import pyvds
except ImportError:
    pyvds = None


ZGY_FILE = 'test_data/zgy/small-8bit.zgy'
VDS_FILE = 'test_data/vds/small.vds'


def test_raises_import_error_if_missing_pyzgy():
    # Modifies imports. Have to reload before and after
    # to not affect other tests
    with mock.patch.dict('sys.modules', {'pyzgy': None}):
        reload(seismicfile) 
        with pytest.raises(ImportError):
            seismicfile.SeismicFile.open(ZGY_FILE, seismicfile.Filetype.ZGY)
    reload(seismicfile) 


def test_raises_import_error_if_missing_pyvds():
    with mock.patch.dict('sys.modules', {'pyvds': None}):
        reload(seismicfile) 
        with pytest.raises(ImportError):
            seismicfile.SeismicFile.open(VDS_FILE, seismicfile.Filetype.VDS)
    reload(seismicfile) 


def test_raises_value_error_if_file_type_wrong_type():
        with pytest.raises(ValueError):
            seismicfile.SeismicFile.open("", "wrong_type")


def test_raises_value_error_if_file_type_unknown():
    with pytest.raises(ValueError):
        seismicfile.SeismicFile.open("seismic.unknown", None)


@pytest.mark.skipif(pyvds is None, reason="Requires pyvds")
def test_can_open_vds_file():
    with seismicfile.SeismicFile.open(VDS_FILE) as seismic:
        assert seismic.filetype is seismicfile.Filetype.VDS


@pytest.mark.skipif(pyzgy is None, reason="Requires pyzgy")
def test_can_open_zgy_file():
    with seismicfile.SeismicFile.open(ZGY_FILE) as seismic:
        assert seismic.filetype is seismicfile.Filetype.ZGY