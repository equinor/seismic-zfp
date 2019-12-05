from collections.abc import Mapping
from .read import SzReader


class Accessor(SzReader):

    def __init__(self, file):
        super().__init__(file)

    def __iter__(self):
        return self[:]

    def __len__(self):
        return self.len_object

    def __getitem__(self, index):
        return self.values_function(index - self.keys_object[0])

    def __contains__(self, key):
        return key in self.keys_object

    def keys(self):
        return self.keys_object

    def values(self):
        return self[:]

    def items(self):
        return zip(self.keys(), self[:])


class InlineAccessor(Accessor, Mapping):
    def __init__(self, file):
        super().__init__(file)
        self.len_object = self.n_ilines
        self.keys_object = self.ilines
        self.values_function = self.read_inline


class CrosslineAccessor(Accessor, Mapping):
    def __init__(self, file):
        super().__init__(file)
        self.len_object = self.n_xlines
        self.keys_object = self.xlines
        self.values_function = self.read_crossline


class ZsliceAccessor(Accessor, Mapping):
    def __init__(self, file):
        super().__init__(file)
        self.len_object = self.n_samples
        self.keys_object = self.zslices
        self.values_function = self.read_zslice