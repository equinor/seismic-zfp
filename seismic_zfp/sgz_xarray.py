import os

import xarray as xr
import numpy as np
from xarray.backends import BackendEntrypoint
from xarray.backends import BackendArray
from xarray.core import indexing

from seismic_zfp.read import SgzReader


class SeismicZfpBackendArray(BackendArray):
    """Lazy array over an SGZ file. read_subvolume takes flattened (min, max) ordinal pairs,
    one pair per axis, with max exclusive, e.g. SgzReader.read_subvolume"""
    def __init__(self, shape, dtype, read_subvolume):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.read_subvolume = read_subvolume

    def __getitem__(self, key: indexing.ExplicitIndexer):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        # Read the smallest covering block once, then apply steps/integer indexing to it in memory
        bounds, local_key, result_shape = [], [], []
        for k, n in zip(key, self.shape):
            if isinstance(k, slice):
                start, stop, step = k.indices(n)
                count = len(range(start, stop, step))
                last = start + (count - 1) * step
                bounds.append((min(start, last), max(start, last) + 1))
                local_key.append(slice(None, None, step))
                result_shape.append(count)
            else:
                k = int(k) + n if k < 0 else int(k)
                bounds.append((k, k + 1))
                local_key.append(0)

        if 0 in result_shape:
            return np.empty(result_shape, dtype=self.dtype)

        subvolume = self.read_subvolume(*[b for pair in bounds for b in pair])
        return subvolume[tuple(local_key)]


class SeismicZfpBackendEntrypoint(BackendEntrypoint):
    def open_dataset(self, filename_or_obj, drop_variables=None):

        sgz_reader = SgzReader(filename_or_obj)

        shape = (sgz_reader.n_ilines, sgz_reader.n_xlines, sgz_reader.n_samples)
        backend_array = SeismicZfpBackendArray(shape, np.float32, sgz_reader.read_subvolume)

        vars = {"data": (("il", "xl", "z"), indexing.LazilyIndexedArray(backend_array))}
        coords = {"il": sgz_reader.ilines, "xl": sgz_reader.xlines, "z": sgz_reader.zslices}

        ds = xr.Dataset(data_vars=vars, coords=coords)
        ds.set_close(sgz_reader.close)

        return ds

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in [".sgz", ".zfp"]
