from .read import SgzReader
from .accessors import InlineAccessor, CrosslineAccessor, ZsliceAccessor, HeaderAccessor, TraceAccessor


class SegyioEmulator(SgzReader):
    def __init__(self, file):
        super(SegyioEmulator, self).__init__(file)
        self.trace = TraceAccessor(self.file)
        self.header = HeaderAccessor(self.file)
        self.iline = InlineAccessor(self.file)
        self.xline = CrosslineAccessor(self.file)
        self.depth_slice = ZsliceAccessor(self.file)
        self.samples = self.zslices
        self.bin = self.get_file_binary_header()
        self.text = self.get_file_text_header()
