try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from .read import SgzReader


class Accessor(SgzReader):

    def __init__(self, file):
        super(Accessor, self).__init__()

    def __iter__(self):
        return iter(self[:])

    def __len__(self):
        return self.len_object

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            # Acquiris Quodcumquae Rapis
            start, stop, step = subscript.indices(len(self))
            return [self.values_function(index) for index in range(start, stop, step)]
        else:
            return self.values_function(subscript)

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
        self.values_function = self.read_inline_number


class CrosslineAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_xlines
        self.keys_object = self.xlines
        self.values_function = self.read_crossline_number


class ZsliceAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.n_samples
        self.keys_object = self.zslices
        self.values_function = self.read_zslice


class HeaderAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.tracecount
        self.keys_object = list(range(self.tracecount))
        self.values_function = self.gen_trace_header


class TraceAccessor(Accessor, Mapping):
    def __init__(self, file):
        super(Accessor, self).__init__(file)
        self.len_object = self.tracecount
        self.keys_object = list(range(self.tracecount))
        self.values_function = self.get_trace
