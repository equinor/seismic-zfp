import os
import collections
import warnings
import numpy as np
import segyio
import time
import psutil

from .utils import pad, define_blockshape, bytes_to_int, int_to_bytes, CubeWithAxes, Geometry, InferredGeometry
from .headers import HeaderwordInfo
from .conversion_utils import run_conversion_loop
from .read import SgzReader
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES
from .seismicfile import SeismicFile, Filetype


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
        self.out_filename = None
        self.geom = None
        if all([min_il, max_il, min_xl, max_xl]):
            self.geom = Geometry(min_il, max_il, min_xl, max_xl)
        self.filetype = self.set_filetype()
        self.mem_limit = psutil.virtual_memory().total

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Non Timetus Messor
        pass

    @staticmethod
    def set_filetype():
        """This method will be overridden by filetpye-specific subclasses"""
        return None

    def get_blank_header_info(self, seismic, header_detection):
        n_traces = seismic.tracecount if seismic.structured else 0
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
                "Invalid header_detection method {}: valid methods: 'heuristic', 'thorough', 'exhaustive', 'strip'"
                    .format(header_detection))

    def write_headers(self, header_detection, header_info):
        # Treating "thorough" mode the same until this point, where we've read the entire file (once)
        # and can do a proper check to ensure no header values are being lost by coincidentally being
        # the same in the first and last traces of the file.
        if header_detection == 'thorough':
            for hw in list(header_info.headers_dict.keys()):
                if np.all(header_info.headers_dict[hw] == header_info.headers_dict[hw][0]):
                    header_info.update_table(hw, (header_info.headers_dict[hw][0],0))
                    del header_info.headers_dict[hw]
            # Update SGZ header
            with open(self.out_filename, 'r+b') as f:
                f.seek(64)
                f.write(int_to_bytes(header_info.get_header_array_count()))
                f.seek(980)
                f.write(header_info.to_buffer())

        if header_detection != 'strip':
            with open(self.out_filename, 'ab') as f:
                for header_array in header_info.headers_dict.values():
                    # Pad to 512-bytes for page blobs
                    f.write(header_array.tobytes() + bytes(512-len(header_array.tobytes())%512))


    def check_memory(self, inline_set_bytes):
        """Requires at least n_crosslines x n_samples x blockshape[2] x 4 bytes of available memory,
        check this before doing anything inelegant.
        """
        if inline_set_bytes > self.mem_limit // 2:
            print(f'One inline set is {inline_set_bytes} bytes,' \
                  f'machine memory is {self.mem_limit} bytes \n' \
                  f'Try using fewer inlines in the blockshape, or compressing a subcube')
            raise RuntimeError("ABORTED effort: Close all that you have. You ask way too much.")

        max_queue_length = min(16, (self.mem_limit // 2) // inline_set_bytes)
        print(f'VirtualMemory={self.mem_limit / (1024 * 1024)}MB, ' \
              f'InlineSet={inline_set_bytes / (1024 * 1024)}MB : Using queue of length {max_queue_length}')

        return max_queue_length

    def detect_geometry(self, seismic):
        if self.geom is None:
            if seismic.unstructured:
                print("SEG-Y file is unstructured and no geometry provided. Determining this may take some time...")
                traces_ref = {(h[189], h[193]): i for i, h in enumerate(seismic.header)}
                self.geom = InferredGeometry(traces_ref)
                print("... inferred geometry is:", self.geom)
            else:
                self.geom = Geometry(0, len(seismic.ilines), 0, len(seismic.xlines))

    def check_inputfile_exists(self):
        if not os.path.exists(self.in_filename):
            if self.filetype is None:
                file_ext = 'file'
            else:
                file_ext = self.filetype.__repr__().split('.')[1].split(':')[0]
            msg = "With searching comes loss,  and the presence of absence:  'My {}' not found.".format(file_ext)
            raise FileNotFoundError(msg)

    def run(self, out_filename, bits_per_voxel=4, blockshape=(4, 4, -1),
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
        self.check_inputfile_exists()
        self.out_filename = out_filename
        print("Converting: In={}, Out={}".format(self.in_filename, self.out_filename))

        t0 = time.time()

        with SeismicFile.open(self.in_filename, self.filetype) as seismic:
            bits_per_voxel, blockshape = define_blockshape(bits_per_voxel, blockshape)
            self.detect_geometry(seismic)
            header_info = self.get_blank_header_info(seismic, header_detection)
            store_headers = not(header_detection == 'strip')
            if seismic.filetype == Filetype.ZGY:
                store_headers = False
            max_queue_length = self.check_memory(inline_set_bytes = blockshape[0]
                                                                    * (len(self.geom.xlines)
                                                                    * len(seismic.samples)) * 4)
            run_conversion_loop(seismic, self.out_filename, bits_per_voxel, blockshape, header_info, self.geom,
                                queuesize=max_queue_length, reduce_iops=reduce_iops, store_headers=store_headers)
        self.write_headers(header_detection, header_info)

        print("Total conversion time: {}                     ".format(time.time()-t0))


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
        spec = segyio.spec()
        spec.samples = self.zslices
        spec.offsets = [0]
        spec.xlines = self.xlines
        spec.ilines = self.ilines
        spec.sorting = 2

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
                    icount = (self.n_ilines % new_blockshape[0] + 4) // 4
                else:
                    icount = 16
                for x in range(padded_shape[1] // new_blockshape[1]):
                    if (x + 1) * new_blockshape[1] > self.n_xlines:
                        xcount = (self.n_xlines % new_blockshape[1] + 4) // 4
                    else:
                        xcount = 16
                    buffer = bytearray(self.chunk_bytes*16*16)
                    for n in range(icount):
                        self.file.seek(self.data_start_bytes + x*self.chunk_bytes*16 + 4*(n+i*16)*inline_bytes)
                        buffer[n*self.chunk_bytes*16:n*self.chunk_bytes*16 + xcount*self.chunk_bytes] = self.file.read(self.chunk_bytes*xcount)
                    for z in range(padded_shape[2] // new_blockshape[2]):
                        new_block = bytearray(DISK_BLOCK_BYTES)
                        for u in range(64*64):
                            new_block[u*self.unit_bytes:(u+1)*self.unit_bytes] = \
                                buffer[u*self.chunk_bytes + z*self.unit_bytes :
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

        self.samples = 4*np.arange(data_array.shape[2]) if samples is None else samples # Default 4ms sampling
        self.trace_headers = collections.OrderedDict(sorted(trace_headers.items()))

        if segyio.tracefield.TraceField.INLINE_3D not in self.trace_headers:
            self.trace_headers[segyio.tracefield.TraceField.INLINE_3D] = np.broadcast_to(np.expand_dims(self.ilines, 1), (len(self.ilines), len(self.xlines)))

        if segyio.tracefield.TraceField.CROSSLINE_3D not in self.trace_headers:
            self.trace_headers[segyio.tracefield.TraceField.CROSSLINE_3D] = np.broadcast_to(self.xlines, (len(self.ilines), len(self.xlines)))

        # Do some sanity checks
        assert data_array.dtype == np.float32
        assert data_array.shape == (len(self.ilines), len(self.xlines), len(self.samples))
        for tracefield, header_array in self.trace_headers.items():
            assert tracefield in segyio.tracefield.keys.values()
            assert header_array.shape == data_array[:,:,0].shape

        self.data_array = data_array


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write_headers(self, header_info):
         with open(self.out_filename, 'ab') as f:
            for header_array in header_info.headers_dict.values():
                # Pad to 512-bytes for page blobs
                f.write(header_array.tobytes() + bytes(512-len(header_array.tobytes())%512))


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
        self.out_filename = out_filename
        bits_per_voxel, blockshape = define_blockshape(bits_per_voxel, blockshape)
        self.geom = Geometry(0, len(self.ilines), 0, len(self.xlines))
        input_cube = CubeWithAxes(self.data_array, self.ilines, self.xlines, self.samples)
        header_info = HeaderwordInfo(n_traces=len(self.ilines)*len(self.xlines), variant_header_dict=self.trace_headers)
        run_conversion_loop(input_cube, self.out_filename, bits_per_voxel, blockshape, header_info, self.geom)
        self.write_headers(header_info)
