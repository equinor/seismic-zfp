import numpy as np
import pytest

try:
    import xarray as xr
except ImportError:
    xr = None

from seismic_zfp.read import SgzReader

SGZ_FILE = 'test_data/small_4bit.sgz'
SGZ_FILE_4D = 'test_data/small-4d_4bit.sgz'

pytestmark = pytest.mark.skipif(xr is None, reason="Requires xarray")


@pytest.fixture(scope='module')
def reference():
    with SgzReader(SGZ_FILE) as reader:
        return reader.read_volume()


@pytest.fixture(scope='module')
def reference_4d():
    with SgzReader(SGZ_FILE_4D) as reader:
        return reader.read_volume()


def test_xarray_from_sgz_file():
    s = xr.open_dataset(SGZ_FILE)
    arr = s.data[0:1, 0:1, 0:5].to_numpy()
    assert (1, 1, 5) == arr.shape


def test_xarray_dataset_structure():
    with SgzReader(SGZ_FILE) as reader, xr.open_dataset(SGZ_FILE) as ds:
        assert ds.data.dims == ("il", "xl", "z")
        assert ds.data.shape == (reader.n_ilines, reader.n_xlines, reader.n_samples)
        assert ds.data.dtype == np.float32
        assert np.array_equal(ds.il, reader.ilines)
        assert np.array_equal(ds.xl, reader.xlines)
        assert np.array_equal(ds.z, reader.zslices)
        # repr requires a real dtype so that nbytes can be computed
        assert "Dimensions:" in repr(ds)


@pytest.mark.parametrize("key", [
    (slice(None), slice(None), slice(None)),
    (slice(1, 4), slice(2, 5), slice(10, 30)),
    (0, 0, slice(0, 5)),
    (2, slice(None), 7),
    (-1, -1, -1),
    (slice(0, 5, 2), 0, slice(0, 3)),
    (slice(None, None, -2), slice(1, 4), slice(5, 30, 7)),
    (4, 4, slice(None, None, -1)),
    (slice(-2, None), 0, 0),
    (slice(3, 3), 0, 0),
    ([0, 2], 0, 0),
    (slice(1, 3), [4, 1], slice(None)),
])
def test_xarray_indexing_matches_numpy(key, reference):
    with xr.open_dataset(SGZ_FILE) as ds:
        got = ds.data[key].to_numpy()
    expected = reference[key]
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_xarray_label_indexing(reference):
    with SgzReader(SGZ_FILE) as reader, xr.open_dataset(SGZ_FILE) as ds:
        il, xl = reader.ilines[3], reader.xlines[1]
        got = ds.data.sel(il=il, xl=xl).to_numpy()
        assert np.array_equal(got, reference[3, 1, :])

        got = ds.data.sel(il=slice(reader.ilines[1], reader.ilines[2])).to_numpy()
        assert np.array_equal(got, reference[1:3, :, :])

        got = ds.data.isel(z=slice(0, 4)).mean().item()
        assert got == pytest.approx(reference[:, :, 0:4].mean())


def test_xarray_out_of_bounds_integer():
    with xr.open_dataset(SGZ_FILE) as ds:
        with pytest.raises(IndexError):
            ds.data[10, 0, 0].to_numpy()


def test_xarray_guess_can_open():
    from seismic_zfp.sgz_xarray import SeismicZfpBackendEntrypoint
    entrypoint = SeismicZfpBackendEntrypoint()
    assert entrypoint.guess_can_open(SGZ_FILE)
    assert entrypoint.guess_can_open('cube.sgz')
    assert not entrypoint.guess_can_open('cube.sgy')
    with open(SGZ_FILE, 'rb') as f:
        assert not entrypoint.guess_can_open(f)


# --- 4D -----------------------------------------------------------------------------------------------

def test_xarray_4d_dataset_structure():
    with SgzReader(SGZ_FILE_4D) as reader, xr.open_dataset(SGZ_FILE_4D) as ds:
        assert ds.data.dims == ("il", "xl", "offset", "z")
        assert ds.data.shape == (reader.n_ilines, reader.n_xlines, reader.n_offsets, reader.n_samples)
        assert ds.data.dtype == np.float32
        assert np.array_equal(ds.il, reader.ilines)
        assert np.array_equal(ds.xl, reader.xlines)
        assert np.array_equal(ds.offset, reader.offsets)
        assert np.array_equal(ds.z, reader.zslices)
        assert "offset" in repr(ds)


@pytest.mark.parametrize("key", [
    (slice(None), slice(None), slice(None), slice(None)),
    (slice(1, 4), slice(2, 5), slice(1, 3), slice(10, 30)),
    (0, 0, 0, slice(0, 5)),
    (2, 3, slice(None), slice(None)),
    (slice(None), slice(None), 4, slice(None)),
    (-1, -1, -1, -1),
    (slice(0, 5, 2), 0, slice(None, None, -1), slice(0, 3)),
    (slice(3, 3), 0, 0, 0),
    ([0, 2], 0, slice(None), 0),
])
def test_xarray_4d_indexing_matches_numpy(key, reference_4d):
    with xr.open_dataset(SGZ_FILE_4D) as ds:
        got = ds.data[key].to_numpy()
    expected = reference_4d[key]
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_xarray_4d_orthogonal_indexing(reference_4d):
    # xarray indexes lists orthogonally (like np.ix_), not point-wise as numpy does
    with xr.open_dataset(SGZ_FILE_4D) as ds:
        got = ds.data[[0, 2], 0, [4, 1], 0].to_numpy()
    assert np.array_equal(got, reference_4d[np.ix_([0, 2], [0], [4, 1], [0])].squeeze())


def test_xarray_4d_matches_reader_accessors():
    with SgzReader(SGZ_FILE_4D) as reader, xr.open_dataset(SGZ_FILE_4D) as ds:
        assert np.array_equal(ds.data[2, 3].to_numpy(), reader.read_gather(2, 3))
        assert np.array_equal(ds.data.isel(offset=4).to_numpy(), reader.read_offset(4))
        assert np.array_equal(ds.data.sel(il=reader.ilines[1], xl=reader.xlines[4]).to_numpy(),
                              reader.read_gather_number(reader.ilines[1], reader.xlines[4]))
        assert np.array_equal(ds.data.sel(offset=reader.offsets[2]).to_numpy(),
                              reader.read_offset_number(reader.offsets[2]))
        assert np.array_equal(ds.data.sel(offset=slice(reader.offsets[1], reader.offsets[2])).to_numpy(),
                              reader.read_subvolume_4d(0, reader.n_ilines, 0, reader.n_xlines, 1, 3,
                                                       0, reader.n_samples))


def test_xarray_4d_reduction_over_offset(reference_4d):
    with xr.open_dataset(SGZ_FILE_4D) as ds:
        stack = ds.data.mean(dim="offset").to_numpy()
    assert stack.shape == reference_4d.shape[:2] + reference_4d.shape[3:]
    assert np.allclose(stack, reference_4d.mean(axis=2))
