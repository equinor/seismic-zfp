try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from .read import SgzReader


class Accessor(SgzReader):

    def __init__(self, file):
        super(Accessor, self).__init__()

    def __iter__(self):
        return self[:]

    def __len__(self):
        return self.len_object

    def __getitem__(self, index):
        return self.values_function(index - self.keys_object[0])

    def __contains__(self, key):
        return key in self.keys_object

    def __hash__(self):
        return hash(self.filename)

    def keys(self):
        return self.keys_object

    def values(self):
        return self[:]

    def items(self):
        return zip(self.keys(), self[:])


class InlineAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_ilines
        self.keys_object = self.ilines
        self.values_function = self.read_inline


class CrosslineAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_xlines
        self.keys_object = self.xlines
        self.values_function = self.read_crossline


class ZsliceAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_samples
        self.keys_object = self.zslices
        self.values_function = self.read_zslice


class HeaderAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.read_variant_headers()
        self.len_object = self.tracecount
        self.keys_object = list(range(self.tracecount))
        self.values_function = self.gen_trace_header


class TraceAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.tracecount
        self.keys_object = list(range(self.tracecount))
        self.values_function = self.get_trace
