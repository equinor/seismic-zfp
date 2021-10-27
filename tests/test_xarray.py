import xarray as xr

SGZ_FILE = 'test_data/small_4bit.sgz'

def test_xarray_from_sgz_file():
    s = xr.open_dataset(SGZ_FILE)
    arr = s.data[0:1, 0:1, 0:5].to_numpy()
    assert (1, 1, 5) == arr.shape
