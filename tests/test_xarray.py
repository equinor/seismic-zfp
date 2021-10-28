import pytest

try:
    import xarray as xr
except ImportError:
    xr = None

SGZ_FILE = 'test_data/small_4bit.sgz'

@pytest.mark.skipif(xr is None, reason="Requires xarray")
def test_xarray_from_sgz_file():
    s = xr.open_dataset(SGZ_FILE)
    arr = s.data[0:1, 0:1, 0:5].to_numpy()
    assert (1, 1, 5) == arr.shape
