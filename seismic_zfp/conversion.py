import os
import collections
import warnings
import numpy as np
import segyio
import time
import psutil

from .headers import HeaderwordInfo
from .conversion_utils import run_conversion_loop, StreamProducer
from .read import SgzReader
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES
from .seismicfile import SeismicFile, Filetype
from .utils import (pad,
                    define_blockshape_2d,
                    define_blockshape_3d,
                    bytes_to_int,
                    int_to_bytes,
                    Axes,
                    CubeWithAxes,
                    Geometry3d,
                    InferredGeometry3d,
                    Geometry2d
                    )


class SeismicFileConverter(object):
    """
    Reads seismic file and compresses to SGZ file(s)

    This is the base class for converters specifically named for their input filetype.

    Because the SeismicFile class detects filetype based on extension this base class could be used most of the time.
    """

    def __init__(self, in_filename, min_il=None, max_il=None, min_xl=None, max_xl=None):
        """
        Parameters
        ----------

        in_filename: str
            The seismic file to be converted to SGZ

        min_il, max_il, min_xl, max_xl: int
            Cropping parameters to apply to input seismic cube
            Refers to IL/XL *ordinals* rather than numbers
        """
        # Quia Ego Sic Dico
        self.in_filename = in_filename
        self.filetype = self.set_filetype()
        self.check_input_file_exists()

        self.geom = None
        if all([min_il, max_il, min_xl, max_xl]):
            self.geom = Geometry3d(min_il, max_il, min_xl, max_xl)
        if self.geom is None:
            with SeismicFile.open(self.in_filename, self.filetype) as seismic:
                self.detect_geometry(seismic)
        self.is_2d = isinstance(self.geom, Geometry2d)
        self.mem_limit = psutil.virtual_memory().total

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Non Timetus Messor
        pass

    @staticmethod
    def set_filetype():
        """This method will be overridden by filetype-specific subclasses"""
        return None

    def get_blank_header_info(self, seismic, header_detection):
        first_il_header_val = seismic.header[0][segyio.tracefield.TraceField.INLINE_3D]
        n_traces = seismic.tracecount if seismic.structured or first_il_header_val == 0 else 0
        if header_detection == 'heuristic':
            return HeaderwordInfo(n_traces=n_traces,
                                  seismicfile=seismic,
                                  header_detection=header_detection)
        elif header_detection in ['thorough', 'exhaustive']:
            return HeaderwordInfo(n_traces=n_traces,
                                  variant_header_list=segyio.TraceField.enums()[0:89],
                                  header_detection=header_detection)
        elif header_detection == 'strip':
            return HeaderwordInfo(n_traces=n_traces,
                                  variant_header_list=[],
                                  header_detection=header_detection)
        else:
            raise NotImplementedError(
                f"Invalid header_detection method {header_detection}:"
                " valid methods: 'heuristic', 'thorough', 'exhaustive', 'strip'")

    @staticmethod
    def write_headers(header_detection, header_info, out_filehandle):
        # Treating "thorough" mode the same until this point, where we've read the entire file (once)
        # and can do a proper check to ensure no header values are being lost by coincidentally being
        # the same in the first and last traces of the file.
        if header_detection == 'thorough':
            for hw in list(header_info.headers_dict.keys()):
                if np.all(header_info.headers_dict[hw] == header_info.headers_dict[hw][0]):
                    header_info.update_table(hw, (header_info.headers_dict[hw][0], 0))
                    del header_info.headers_dict[hw]
            # Update SGZ header
            with open(out_filehandle.name, 'r+b') as f:
                f.seek(64)
                f.write(int_to_bytes(header_info.get_header_array_count()))
                f.seek(980)
                f.write(header_info.to_buffer())

        if header_detection != 'strip':
            for header_array in header_info.headers_dict.values():
                # Pad to 512-bytes for page blobs
                out_filehandle.write(header_array.tobytes() + bytes(512-len(header_array.tobytes()) % 512))

    @staticmethod
    def write_hash(hash, out_filehandle):
        with open(out_filehandle.name, 'r+b') as f:
            f.seek(960)
            f.write(hash)

    def check_memory(self, inline_set_bytes):
        """Requires at least n_crosslines x n_samples x blockshape[2] x 4 bytes of available memory,
        check this before doing anything inelegant.
        """
        if inline_set_bytes > self.mem_limit // 2:
            print(f"One inline set is {inline_set_bytes} bytes,"
                  f"machine memory is {self.mem_limit} bytes \n"
                  f"'Try using fewer inlines in the blockshape, or compressing a subcube")
            raise RuntimeError("ABORTED effort: Close all that you have. You ask way too much.")

        max_queue_length = min(16, (self.mem_limit // 2) // inline_set_bytes)
        print(f"VirtualMemory={self.mem_limit/(1024*1024*1024):.2f}GB  :"
              f"  InlineSet={inline_set_bytes/(1024*1024):.2f}MB  :"
              f"  Using queue of length {max_queue_length}")

        return max_queue_length

    def detect_geometry(self, seismic):
        if seismic.unstructured:
            first_header = seismic.header[0]
            last_header = seismic.header[-1]
            if (first_header[189], first_header[193], last_header[189], last_header[193]) == (0, 0, 0, 0):
                # We have a 2D SEG-Y
                self.geom = Geometry2d(seismic.tracecount)
            else:
                # We have an irregular 3D SEG-Y
                print("SEG-Y file is unstructured and no geometry provided. Determining this may take some time...")
                self.geom = None
        else:
            if seismic.ilines is not None and len(seismic.ilines) == 1:
                # We have a 2D SEG-Y
                self.geom = Geometry2d(seismic.xlines)
            elif seismic.xlines is not None and len(seismic.xlines) == 1:
                # We have a 2D SEG-Y
                self.geom = Geometry2d(seismic.ilines)
            else:
                # We have a regular 3D SEG-Y
                self.geom = Geometry3d(0, len(seismic.ilines), 0, len(seismic.xlines))

    def infer_geometry(self, seismic):
        traces_ref = {(h[189], h[193]): i for i, h in enumerate(seismic.header)}
        self.geom = InferredGeometry3d(traces_ref)
        print("... inferred geometry is:", self.geom)

    def check_input_file_exists(self):
        if not os.path.exists(self.in_filename):
            if self.filetype is None:
                file_ext = 'file'
            else:
                file_ext = self.filetype.__repr__().split('.')[1].split(':')[0]
            msg = f"With searching comes loss,  and the presence of absence:  'My {file_ext}' not found."
            raise FileNotFoundError(msg)

    def run(self, out_filename, bits_per_voxel=4, blockshape=None,
            reduce_iops=False, header_detection="heuristic"):
        """General entrypoint for converting seismic files to SGZ

        Parameters
        ----------

        out_filename: str
            The SGZ output file

        bits_per_voxel: int, float, str
            The number of bits to use for storing each seismic voxel.
            - Uncompressed seismic has 32-bits per voxel
            - Using 16-bits gives almost perfect reproduction
            - Tested using 8, 4, 2, 1, 0.5 & 0.25 bit
            - Recommended using 4-bit, giving 8:1 compression
            - Negative value implies reciprocal: i.e. -2 ==> 1/2 bits per voxel

        blockshape: (int, int, int)
            The physical shape of voxels compressed to one disk block.
            Can only specify 3 of blockshape (il,xl,z) and bits_per_voxel, 4th is redundant.
            - Specifying -1 for one of these will calculate that one
            - Specifying -1 for more than one of these will fail
            - Each one must be a power of 2
            - (4, 4, -1) - default - is good for IL/XL reading
            - (64, 64, 4) is good for Z-Slice reading (requires 2-bit compression)

        reduce_iops: bool
            Flag to indicate whether compression should attempt to minimize the number
            of iops required to read the input SEG-Y file by reading whole inlines including
            headers in one go. Falls back to segyio if incorrect. Useful under Windows.

        header_detection: str
            One of the following options.
            - "heuristic"  : Detect variant headers by looking at header values in first and last traces in volume
                             Can *technically* miss a header value, but fast.
            - "thorough"   : Detect variant headers by examining contents of all of them.
                             Memory intensive for large volumes, minor compute overhead.
            - "exhaustive" : Skip detection entirely, be paranoid and take everything.
                             Memory intensive for large volumes, likely generates considerably larger file than needed.
            - "strip"      : Do not save any SEG-Y trace headers. Smallest file, fast and dangerous.

            Default: "heuristic".
        """
        print(f"Converting: In={self.in_filename}, Out={out_filename}")

        t0 = time.time()

        with SeismicFile.open(self.in_filename, self.filetype) as seismic:
            if self.geom is None:
                self.infer_geometry(seismic)
            if self.is_2d:
                if blockshape is None:
                    blockshape = (1, 16, -1)
                bits_per_voxel, blockshape = define_blockshape_2d(bits_per_voxel, blockshape)
                inline_set_bytes = len(self.geom.traces) * len(seismic.samples) * 4
            else:
                if blockshape is None:
                    blockshape = (4, 4, -1)
                bits_per_voxel, blockshape = define_blockshape_3d(bits_per_voxel, blockshape)
                inline_set_bytes = blockshape[0]*(len(self.geom.xlines) * len(seismic.samples)) * 4

            header_info = self.get_blank_header_info(seismic, header_detection)
            store_headers = not(header_detection == 'strip')
            if seismic.filetype == Filetype.ZGY:
                store_headers = False
            max_queue_length = self.check_memory(inline_set_bytes=inline_set_bytes)
            with open(out_filename, 'wb') as out_file:
                hash_bytes = run_conversion_loop(seismic, out_file, bits_per_voxel, blockshape, header_info, self.geom,
                                                 queue_size=max_queue_length, reduce_iops=reduce_iops,
                                                 store_headers=store_headers)
                self.write_headers(header_detection, header_info, out_file)
                self.write_hash(hash_bytes, out_file)

        print(f"Total conversion time: {time.time()-t0:.3f}s                     ")


class SegyConverter(SeismicFileConverter):
    """Reads SEG-Y file and compresses to SGZ file(s)"""
    @staticmethod
    def set_filetype():
        return Filetype.SEGY


class ZgyConverter(SeismicFileConverter):
    """Reads ZGY file and converts to SGZ file(s)"""
    @staticmethod
    def set_filetype():
        return Filetype.ZGY


class VdsConverter(SeismicFileConverter):
    """Reads VDS file and converts to SGZ file(s)"""
    @staticmethod
    def set_filetype():
        return Filetype.VDS


class SgzConverter(SgzReader):
    """Reads SGZ files and either:
       - Writes seismic data as SEG-Y file
       - Writes 'advanced-layout' SGZ files (input must be 'default-layout' SGZ file)"""

    def __init__(self, file, filetype_checking=True, preload=False, chunk_cache_size=None):
        super().__init__(file, filetype_checking, preload, chunk_cache_size)

    # If an SGZ file has been cropped vertically, then the recording start time in the headers
    # will be wrong. May be better to correct this on cropping rather than regenerating SEG-Y...
    def regenerate_trace_header(self, i):
        header = self.gen_trace_header(i)
        header[segyio.TraceField.DelayRecordingTime] = int(self.zslices[0])
        return header

    def convert_to_segy(self, out_file):
        if self.is_3d:
            spec = segyio.spec()
            spec.samples = self.zslices
            spec.offsets = [0]
            spec.xlines = self.xlines
            spec.ilines = self.ilines
            spec.sorting = 2
        else:
            spec = segyio.spec()
            spec.samples = self.zslices
            spec.tracecount = self.tracecount

        # seimcic-zfp stores the binary header from the source SEG-Y file.
        # In case someone forgot to do this, give them IBM float
        data_sample_format_code = bytes_to_int(
            self.headerbytes[DISK_BLOCK_BYTES+3225: DISK_BLOCK_BYTES+3227])
        if data_sample_format_code in [1, 5]:
            spec.format = data_sample_format_code
        else:
            new_headerbytes = bytearray(self.headerbytes)
            new_headerbytes[DISK_BLOCK_BYTES + 3225: DISK_BLOCK_BYTES + 3227] = int_to_bytes(1)
            self.headerbytes = bytes(new_headerbytes)
            spec.format = 1

        self.write_segy(spec, out_file)

    def write_segy(self, spec, out_file):

        with warnings.catch_warnings():
            # segyio will warn us that out padded cube is not contiguous. This is expected, and safe.
            warnings.filterwarnings("ignore", message="Implicit conversion to contiguous array")
            with segyio.create(out_file, spec) as segyfile:
                self.read_variant_headers()
                # Doing this is fine now there is decent caching on the loader
                segyfile.trace = [self.get_trace(i) for i in range(self.tracecount)]
                segyfile.header = [self.regenerate_trace_header(i) for i in range(self.tracecount)]

        with open(out_file, "r+b") as f:
            f.write(self.headerbytes[DISK_BLOCK_BYTES: DISK_BLOCK_BYTES + SEGY_FILE_HEADER_BYTES])

    def convert_to_adv_sgz(self, out_file):
        assert(self.rate == 2)
        assert(self.blockshape == (4, 4, 1024))
        new_header = bytearray(self.headerbytes)
        new_blockshape = (64, 64, 4)
        new_header[44:48] = int_to_bytes(new_blockshape[0])
        new_header[48:52] = int_to_bytes(new_blockshape[1])
        new_header[52:56] = int_to_bytes(new_blockshape[2])

        padded_shape = (pad(self.n_ilines, new_blockshape[0]),
                        pad(self.n_xlines, new_blockshape[1]),
                        pad(self.n_samples, new_blockshape[2]))

        compressed_data_length_diskblocks = (self.rate * padded_shape[2] *
                                             padded_shape[1] * padded_shape[0]) // (8 * DISK_BLOCK_BYTES)
        new_header[56:60] = int_to_bytes(compressed_data_length_diskblocks)

        with open(out_file, "wb") as outfile:
            outfile.write(new_header)
            inline_bytes = (self.shape_pad[2] * self.shape_pad[1] * self.rate) // 8
            for i in range(padded_shape[0] // new_blockshape[0]):
                if (i + 1) * new_blockshape[0] > self.n_ilines:
                    i_count = (self.n_ilines % new_blockshape[0] + 4) // 4
                else:
                    i_count = 16
                for x in range(padded_shape[1] // new_blockshape[1]):
                    if (x + 1) * new_blockshape[1] > self.n_xlines:
                        x_count = (self.n_xlines % new_blockshape[1] + 4) // 4
                    else:
                        x_count = 16
                    buffer = bytearray(self.chunk_bytes*16*16)
                    for n in range(i_count):
                        self.file.seek(self.data_start_bytes + x*self.chunk_bytes*16 + 4*(n+i*16)*inline_bytes)
                        idx = slice(n*self.chunk_bytes*16, n*self.chunk_bytes*16 + x_count*self.chunk_bytes)
                        buffer[idx] = self.file.read(self.chunk_bytes*x_count)
                    for z in range(padded_shape[2] // new_blockshape[2]):
                        new_block = bytearray(DISK_BLOCK_BYTES)
                        for u in range(64*64):
                            new_block[u*self.unit_bytes:(u+1)*self.unit_bytes] = \
                                buffer[u*self.chunk_bytes + z*self.unit_bytes:
                                       u*self.chunk_bytes + (z+1)*self.unit_bytes]
                        outfile.write(new_block)
            self.read_variant_headers()
            for k, header_array in self.variant_headers.items():
                outfile.write(header_array.tobytes())


class NumpyConverter(object):
    """Compresses 3D numpy array to SGZ file(s)"""

    def __init__(self, data_array, ilines=None, xlines=None, samples=None, trace_headers={}):
        """
        Parameters
        ----------

        data_array: np.ndarray of dtype==np.float32
            The 3D numpy array of 32-bit floats to be compressed.

        ilines, xlines, samples: 1D array-like objects
            Axes labels for input array

        trace_headers: dict
            key, value pairs pf:
                - Member of segyio.tracefield.TraceField Enum
                - 2D numpy array of integers in inline-major order, representing trace header values to be inserted
        """
        self.geom = None

        # Get ilines axis. If overspecified check consistency, and generate if unspecified.
        if segyio.tracefield.TraceField.INLINE_3D in trace_headers:
            self.ilines = trace_headers[segyio.tracefield.TraceField.INLINE_3D][:, 0]
            if ilines is not None:
                assert np.array_equal(self.ilines, ilines)
        else:
            self.ilines = np.arange(data_array.shape[0]) if ilines is None else ilines

        # Get xlines axis. If overspecified check consistency, and generate if unspecified.
        if segyio.tracefield.TraceField.CROSSLINE_3D in trace_headers:
            self.xlines = trace_headers[segyio.tracefield.TraceField.CROSSLINE_3D][0, :]
            if xlines is not None:
                assert np.array_equal(self.xlines, xlines)
        else:
            self.xlines = np.arange(data_array.shape[1]) if xlines is None else xlines

        self.samples = 4*np.arange(data_array.shape[2]) if samples is None else samples   # Default 4ms sampling
        self.trace_headers = collections.OrderedDict(sorted(trace_headers.items()))

        shape = (len(self.ilines), len(self.xlines))
        tf_il = segyio.tracefield.TraceField.INLINE_3D
        tf_xl = segyio.tracefield.TraceField.CROSSLINE_3D

        if tf_il not in self.trace_headers:
            self.trace_headers[tf_il] = np.broadcast_to(np.expand_dims(self.ilines, 1), shape)

        if tf_xl not in self.trace_headers:
            self.trace_headers[tf_xl] = np.broadcast_to(self.xlines, shape)

        # Do some sanity checks
        assert data_array.dtype == np.float32
        assert data_array.shape == (len(self.ilines), len(self.xlines), len(self.samples))
        for tracefield, header_array in self.trace_headers.items():
            assert tracefield in segyio.tracefield.keys.values()
            assert header_array.shape == data_array[:, :, 0].shape

        self.data_array = data_array

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def write_headers(header_info, out_filehandle):
        for header_array in header_info.headers_dict.values():
            # Pad to 512-bytes for page blobs
            out_filehandle.write(header_array.tobytes() + bytes(512-len(header_array.tobytes()) % 512))

    @staticmethod
    def write_hash(hash, out_filehandle):
        with open(out_filehandle.name, 'r+b') as f:
            f.seek(960)
            f.write(hash)

    def run(self, out_filename, bits_per_voxel=4, blockshape=(4, 4, -1)):
        """General entrypoint for converting numpy arrays to SGZ files

        Parameters
        ----------

        out_filename: str
            The SGZ output file

        bits_per_voxel: int, float, str
            The number of bits to use for storing each seismic voxel.
            - Uncompressed seismic has 32-bits per voxel
            - Using 16-bits gives almost perfect reproduction
            - Tested using 8, 4, 2, 1, 0.5 & 0.25 bit
            - Recommended using 4-bit, giving 8:1 compression
            - Negative value implies reciprocal: i.e. -2 ==> 1/2 bits per voxel

        blockshape: (int, int, int)
            The physical shape of voxels compressed to one disk block.
            Can only specify 3 of blockshape (il,xl,z) and bits_per_voxel, 4th is redundant.
            - Specifying -1 for one of these will calculate that one
            - Specifying -1 for more than one of these will fail
            - Each one must be a power of 2
            - (4, 4, -1) - default - is good for IL/XL reading
            - (64, 64, 4) is good for Z-Slice reading (requires 2-bit compression)
        """
        bits_per_voxel, blockshape = define_blockshape_3d(bits_per_voxel, blockshape)
        self.geom = Geometry3d(0, len(self.ilines), 0, len(self.xlines))
        input_cube = CubeWithAxes(self.data_array, self.ilines, self.xlines, self.samples)
        header_info = HeaderwordInfo(n_traces=len(self.ilines)*len(self.xlines), variant_header_dict=self.trace_headers)
        with open(out_filename, 'wb') as out_filehandle:
            hash_bytes = run_conversion_loop(input_cube, out_filehandle, bits_per_voxel,
                                             blockshape, header_info, self.geom)
            self.write_headers(header_info, out_filehandle)
            self.write_hash(hash_bytes, out_filehandle)


class StreamConverter(object):
    """
    Compresses a 3D numpy array to an SGZ file as a stream of chunks. 
    Each chunk is a set of planes in the inline (iline) direction, enabling 
    sequential compression and storage without loading the entire array into memory. 
    The number of planes per chunk is determined by the iline dimension of the 
    `blockshape` parameter, specifically `blockshape[0]`.
    """

    def __init__(
        self,
        out_filename,
        ilines,
        xlines,
        samples,
        bits_per_voxel=4,
        blockshape=(4, 4, -1),
        trace_headers={},
        use_higher_samples_precision=False,
    ):
        """
        Parameters
        ----------

        out_filename: str
            The SGZ output file

        ilines, xlines, samples: 1D array-like objects
            Axes labels for input array

        bits_per_voxel: int, float, str
            The number of bits to use for storing each seismic voxel.
            - Uncompressed seismic has 32-bits per voxel
            - Using 16-bits gives almost perfect reproduction
            - Tested using 8, 4, 2, 1, 0.5 & 0.25 bit
            - Recommended using 4-bit, giving 8:1 compression
            - Negative value implies reciprocal: i.e. -2 ==> 1/2 bits per voxel

        blockshape: (int, int, int)
            The physical shape of voxels compressed to one disk block.
            Can only specify 3 of blockshape (il,xl,z) and bits_per_voxel, 4th is redundant.
            - Specifying -1 for one of these will calculate that one
            - Specifying -1 for more than one of these will fail
            - Each one must be a power of 2
            - (4, 4, -1) - default - is good for IL/XL reading
            - (64, 64, 4) is good for Z-Slice reading (requires 2-bit compression)

        trace_headers: dict
            key, value pairs pf:
                - Member of segyio.tracefield.TraceField Enum
                - 2D numpy array of integers in inline-major order, representing trace header values to be inserted

        use_higher_samples_precision : bool, optional
            Specifies whether to use higher precision for the sample interval and sample time. 
            Default is `False`. When set to `True`, stores sample interval and sample time as 
            64-bit floating-point numbers for increased precision. If `False`, they are stored 
            as 32-bit integers.
        """
        # Get ilines axis. If overspecified check consistency, and generate if unspecified.
        if segyio.tracefield.TraceField.INLINE_3D in trace_headers:
            self.ilines = trace_headers[segyio.tracefield.TraceField.INLINE_3D][:, 0]
            if ilines is not None:
                assert np.array_equal(self.ilines, ilines)
        else:
            self.ilines = ilines

        # Get xlines axis. If overspecified check consistency, and generate if unspecified.
        if segyio.tracefield.TraceField.CROSSLINE_3D in trace_headers:
            self.xlines = trace_headers[segyio.tracefield.TraceField.CROSSLINE_3D][0, :]
            if xlines is not None:
                assert np.array_equal(self.xlines, xlines)
        else:
            self.xlines = xlines

        self.samples = samples
        self.trace_headers = collections.OrderedDict(sorted(trace_headers.items()))

        shape = (len(self.ilines), len(self.xlines))
        total_shape = (len(self.ilines), len(self.xlines), len(samples))
        tf_il = segyio.tracefield.TraceField.INLINE_3D
        tf_xl = segyio.tracefield.TraceField.CROSSLINE_3D

        if tf_il not in self.trace_headers:
            self.trace_headers[tf_il] = np.broadcast_to(
                np.expand_dims(self.ilines, 1), shape
            )

        if tf_xl not in self.trace_headers:
            self.trace_headers[tf_xl] = np.broadcast_to(self.xlines, shape)

        for tracefield, header_array in self.trace_headers.items():
            assert tracefield in segyio.tracefield.keys.values()
            assert header_array.shape == shape

        bits_per_voxel, blockshape = define_blockshape_3d(bits_per_voxel, blockshape)
        self.geom = Geometry3d(0, len(self.ilines), 0, len(self.xlines))
        axes = Axes(self.ilines, self.xlines, self.samples)
        self.header_info = HeaderwordInfo(
            n_traces=len(self.ilines) * len(self.xlines),
            variant_header_dict=self.trace_headers,
        )
        self.out_filehandle = open(out_filename, "wb")
        self.blockshape = blockshape

        self.stream_producer = StreamProducer(
            axes,
            self.out_filehandle,
            bits_per_voxel,
            blockshape,
            self.header_info,
            self.geom,
            total_shape,
            use_higher_samples_precision=use_higher_samples_precision
        )

    def write(self, data_array):
        """
        Compresses a 3D block of data with specified dimensions and constraints.

        Parameters
        ----------
        data_array: np.ndarray of dtype==np.float32
            The 3D numpy array of 32-bit floats to be compressed with dimensions representing:
            - `data_array.shape[0]` as the inlines,
            - `data_array.shape[1]` as the crosslines, and
            - `data_array.shape[2]` as the samples.

        Constraints
        -----------
        The input `data_array` must satisfy the following conditions:
        - `data_array.shape[0]` (inlines) should be **equal to `self.blockshape[0]`**, representing
        the block's expected number of inlines, **except on the last call to this method**, where
        `data_array.shape[0]` can be **less than or equal to `self.blockshape[0]`**.
        - `data_array.shape[1]` (crosslines) must be **equal to `len(self.inlines)`**, indicating the
        expected number of crosslines.
        - `data_array.shape[2]` (samples) must be **equal to `len(self.samples)`**, defining the
        required number of samples per inline-crossline pair.

        Raises
        ------
        AssertionError
            If any of the shape constraints are not satisfied, an `AssertionError` is raised.

        Returns
        -------
        None
            This method performs a write operation and does not return a value.

        Notes
        -----
        This method is part of the StreamConverter class, designed to write sequential blocks of data
        in consecutive calls. Each call should write a chunk of data with `self.blockshape[0]` inlines.
        The final call to this method may contain fewer inlines if the total number of inlines does not
        divide evenly by `self.blockshape[0]`. This design ensures efficient streaming of data in
        fixed-size chunks, except for the last chunk, which can be smaller.
        """

        # Do some sanity checks
        assert data_array.dtype == np.float32
        assert data_array.shape[0] <= self.blockshape[0]
        assert data_array.shape[1] == len(self.xlines)
        assert data_array.shape[2] == len(self.samples)
        self.stream_producer.produce(data_array)

    def close(self):
        hash_bytes = self.stream_producer.hash_object.digest()
        NumpyConverter.write_headers(self.header_info, self.out_filehandle)
        NumpyConverter.write_hash(hash_bytes, self.out_filehandle)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
