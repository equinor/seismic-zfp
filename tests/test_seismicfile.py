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
SGY_FILE = 'test_data/small.sgy'
SGY_FILE_4D = 'test_data/small-4d.sgy'
SGY_FILE_2D = 'test_data/small-2d.sgy'
SGY_FILE_IRREG = 'test_data/small-irregular.sgy'
SGZ_FILE = 'test_data/small_8bit.sgz'


def test_segy_3d_regular_is_structured_not_4d():
    with seismicfile.SeismicFile.open(SGY_FILE) as seismic:
        assert seismic.structured
        assert seismic.n_offsets == 1
        assert not seismic.is_4d


def test_segy_4d_regular_is_structured_and_4d():
    with seismicfile.SeismicFile.open(SGY_FILE_4D) as seismic:
        assert seismic.structured
        assert seismic.n_offsets == 5
        assert seismic.is_4d


def test_segy_irregular_is_unstructured_not_4d():
    with seismicfile.SeismicFile.open(SGY_FILE_IRREG) as seismic:
        assert not seismic.structured
        assert seismic.n_offsets == 1
        assert not seismic.is_4d


def test_segy_2d_no_geometry_is_unstructured_not_4d():
    with seismicfile.SeismicFile.open(SGY_FILE_2D) as seismic:
        assert not seismic.structured
        assert seismic.n_offsets == 1
        assert not seismic.is_4d


def test_sgz_file_not_4d():
    with seismicfile.SeismicFile.open(SGZ_FILE) as seismic:
        assert seismic.n_offsets == 1
        assert not seismic.is_4d


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
        assert seismic.n_offsets == 1
        assert not seismic.is_4d


@pytest.mark.skipif(pyzgy is None, reason="Requires pyzgy")
def test_can_open_zgy_file():
    with seismicfile.SeismicFile.open(ZGY_FILE) as seismic:
        assert seismic.filetype is seismicfile.Filetype.ZGY
        assert seismic.n_offsets == 1
        assert not seismic.is_4d