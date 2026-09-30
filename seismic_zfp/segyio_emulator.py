from .read import SgzReader
from .accessors import InlineAccessor, CrosslineAccessor, ZsliceAccessor, \
                       HeaderAccessor, TraceAccessor, SubvolumeAccessor, \
                       InlineAccessor4d, CrosslineAccessor4d, ZsliceAccessor4d, GatherAccessor, SubvolumeAccessor4d
from .utils import WrongDimensionalityError


class SegyioEmulator(SgzReader):
    def __init__(self, file, chunk_cache_size: int = None):
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
            self.gather = DimensionalityError("SEG-Y emulation only supports gather for 4D files")
            self.unstructured = False
        elif self.is_4d:
            # As segyio: iline/xline/depth_slice of a prestack file give the first offset unless one is specified
            self.iline = InlineAccessor4d(self.file).__enter__()
            self.xline = CrosslineAccessor4d(self.file).__enter__()
            self.depth_slice = ZsliceAccessor4d(self.file).__enter__()
            self.gather = GatherAccessor(self.file).__enter__()
            self.subvolume = SubvolumeAccessor4d(self.file).__enter__()
            self.unstructured = not self.structured
        else:
            self.iline = DimensionalityError()
            self.xline = DimensionalityError()
            self.depth_slice = DimensionalityError()
            self.gather = DimensionalityError()
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
        elif self.is_4d:
            self.iline.__exit__(*exc)
            self.xline.__exit__(*exc)
            self.depth_slice.__exit__(*exc)
            self.gather.__exit__(*exc)
            self.subvolume.__exit__(*exc)

        self.close_sgz_file()


class DimensionalityError:
    def __init__(self, message="SEG-Y emulation does not support this for 2D files"):
        self.message = message

    def __getitem__(self, item):
        raise WrongDimensionalityError(self.message)
