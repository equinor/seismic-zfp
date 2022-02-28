from .read import SgzReader
from .accessors import InlineAccessor, CrosslineAccessor, ZsliceAccessor, HeaderAccessor, TraceAccessor, SubvolumeAccessor
from .utils import WrongDimensionalityError

class SegyioEmulator(SgzReader):
    def __init__(self, file, chunk_cache_size: int=None):
        super(SegyioEmulator, self).__init__(file, chunk_cache_size=chunk_cache_size)

        self.trace = TraceAccessor(self.file).__enter__()
        self.header = HeaderAccessor(self.file).__enter__()
        self.attributes = self.get_tracefield_1d
        self.samples = self.zslices
        self.bin = self.get_file_binary_header()
        self.text = self.get_file_text_header()

        if self.is_3d:
            self.iline = InlineAccessor(self.file).__enter__()
            self.xline = CrosslineAccessor(self.file).__enter__()
            self.depth_slice = ZsliceAccessor(self.file).__enter__()
            self.subvolume = SubvolumeAccessor(self.file).__enter__()
            self.unstructured = False
        else:
            self.iline = DimensionalityError()
            self.xline = DimensionalityError()
            self.depth_slice = DimensionalityError()
            self.unstructured = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.trace.__exit__(*exc)
        self.header.__exit__(*exc)

        if self.is_3d:
            self.iline.__exit__(*exc)
            self.xline.__exit__(*exc)
            self.depth_slice.__exit__(*exc)
            self.subvolume.__exit__(*exc)

        self.close_sgz_file()

class DimensionalityError:
    def __getitem__(self, item):
        raise WrongDimensionalityError("SEG-Y emulation does not support this for 2D files")
