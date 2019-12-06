from .read import SzReader
from .accessors import InlineAccessor, CrosslineAccessor, ZsliceAccessor, HeaderAccessor, TraceAccessor


class SegyioEmulator(SzReader):
    def __init__(self, file):
        super().__init__(file)
        self.trace = TraceAccessor(self.file)
        self.header = HeaderAccessor(self.file)
        self.iline = InlineAccessor(self.file)
        self.xline = CrosslineAccessor(self.file)
        self.depth_slice = ZsliceAccessor(self.file)
        self.samples = self.zslices
