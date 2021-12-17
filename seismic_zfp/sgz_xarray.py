import os

import xarray as xr
import numpy as np
from xarray.backends import BackendEntrypoint
from xarray.backends import BackendArray
from xarray.core import indexing

from seismic_zfp.read import SgzReader


class SeismicZfpBackendArray(BackendArray):
    def __init__(self, shape, dtype, sgz_reader):
        self.shape = shape
        self.dtype = dtype
        self.sgz_reader = sgz_reader

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.typing.ArrayLike:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:


        min_il = key[0].start if isinstance(key[0], slice) else key[0]
        min_xl = key[1].start if isinstance(key[1], slice) else key[1]
        min_z = key[2].start if isinstance(key[2], slice) else key[2]

        min_il = 0 if min_il is None else min_il
        min_xl = 0 if min_xl is None else min_xl
        min_z = 0 if min_z is None else min_z

        max_il = key[0].stop if isinstance(key[0], slice) else key[0] + 1
        max_xl = key[1].stop if isinstance(key[1], slice) else key[1] + 1
        max_z = key[2].stop if isinstance(key[2], slice) else key[2] + 1

        max_il = self.sgz_reader.n_ilines if max_il is None else max_il
        max_xl = self.sgz_reader.n_xlines if max_xl is None else max_xl
        max_z = self.sgz_reader.n_samples if max_z is None else max_z

        return self.sgz_reader.read_subvolume(min_il=min_il, max_il=max_il,
                                              min_xl=min_xl, max_xl=max_xl,
                                              min_z=min_z,   max_z=max_z)



class SeismicZfpBackendEntrypoint(BackendEntrypoint):
    def open_dataset(self, filename_or_obj, drop_variables=None):

        sgz_reader = SgzReader(filename_or_obj)

        shape = (sgz_reader.n_ilines, sgz_reader.n_xlines, sgz_reader.n_samples)

        vars = {"data": (("il", "xl", "z"), SeismicZfpBackendArray(shape, np.float32, sgz_reader))}
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
