from __future__ import print_function
from pyzfp import compress
import warnings
import numpy as np
import segyio
import asyncio
import time
from psutil import virtual_memory

from .utils import pad, define_blockshape, FileOffset, bytes_to_int
from .headers import get_unique_headerwords
from .conversion_utils import make_header, get_header_arrays, run_conversion_loop
from .read import SgzReader
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES


class SegyConverter(object):
    """Writes SEG-Y files from SGZ files"""

    def __init__(self, in_filename, min_il=0, max_il=None, min_xl=0, max_xl=None):
        """
        Parameters
        ----------

        in_filename: str
            The SEGY file to be converted to SGZ

        min_il, max_il, min_xl, max_xl: int
            Cropping parameters to apply to input seismic cube
        """
        self.in_filename = in_filename
        self.out_filename = None
        self.min_il = min_il
        self.max_il = max_il
        self.min_xl = min_xl
        self.max_xl = max_xl

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def run(self, out_filename, bits_per_voxel=4, blockshape=(4, 4, -1), method="Stream"):
        """General entrypoint for converting SEG-Y files to SGZ

        Parameters
        ----------

        out_filename: str
            The SGZ output file

        bits_per_voxel: int
            The number of bits to use for storing each seismic voxel.
            - Uncompressed seismic has 32-bits per voxel
            - Using 16-bits gives almost perfect reproduction
            - Tested using 8-bit, 4-bit, 2-bit & 1-bit
            - Recommended using 4-bit, giving 8:1 compression

        blockshape: (int, int, int)
            The physical shape of voxels compressed to one disk block.
            Can only specify 3 of blockshape (il,xl,z) and bits_per_voxel, 4th is redundant.
            - Specifying -1 for one of these will calculate that one
            - Specifying -1 for more than one of these will fail
            - Each one must be a power of 2
            - (4, 4, -1) - default - is good for IL/XL reading
            - (64, 64, 4) is good for Z-Slice reading (requires 2-bit compression)

        method: str
            Flag to indicate method for reading SEG-Y
            - "InMemory" : Read whole SEG-Y cube into memory before compressing
            - "Stream" : Read 4 inlines at a time... compress, rinse, repeat

        Raises
        ------

        NotImplementedError
            If method is not one of "InMemory" or Stream"

        """
        self.out_filename = out_filename

        if method == "InMemory":
            print("Converting: In={}, Out={}".format(self.in_filename, self.out_filename, blockshape))
            self.convert_segy_inmem(bits_per_voxel, blockshape)
        elif method == "Stream":
            print("Converting: In={}, Out={}".format(self.in_filename, self.out_filename))
            self.convert_segy_stream(bits_per_voxel, blockshape)
        else:
            raise NotImplementedError("Invalid conversion method {}, try 'InMemory' or 'Stream'".format(method))

    def convert_segy_inmem(self, bits_per_voxel, blockshape):
        with segyio.open(self.in_filename) as segyfile:
            cube_bytes = len(segyfile.samples) * len(segyfile.xlines) * len(segyfile.ilines) * 4

        if cube_bytes > virtual_memory().total:
            print("SEG-Y is {} bytes, machine memory is {} bytes".format(cube_bytes, virtual_memory().total))
            print("Try using method = 'Stream' instead")
            raise RuntimeError("Out of memory. We wish to hold the whole sky, But we never will.")

        if (blockshape[0] == 4) and (blockshape[1] == 4):
            self.convert_segy_inmem_default(bits_per_voxel)
        else:
            self.convert_segy_inmem_advanced(bits_per_voxel, blockshape)

    def convert_segy_inmem_default(self, bits_per_voxel):
        """Reads all data from input file to memory, compresses it and writes as .sgz file to disk,
        using ZFP's default compression unit order"""
        header = make_header(self.in_filename, bits_per_voxel, blockshape=(4, 4, 2048//bits_per_voxel))

        t0 = time.time()
        data = segyio.tools.cube(self.in_filename)
        t1 = time.time()

        padded_shape = (pad(data.shape[0], 4), pad(data.shape[1], 4), pad(data.shape[2], 2048//bits_per_voxel))
        data_padded = np.zeros(padded_shape, dtype=np.float32)
        data_padded[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data
        compressed = compress(data_padded, rate=bits_per_voxel)
        t2 = time.time()

        numpy_headers_arrays = get_header_arrays(self.in_filename)

        with open(self.out_filename, 'wb') as f:
            f.write(header)
            f.write(compressed)
            for header_array in numpy_headers_arrays:
                f.write(header_array.tobytes())

        t3 = time.time()

        print("Total conversion time: {}, of which read={}, compress={}, write={}".format(t3-t0, t1-t0, t2-t1, t3-t2))

    def convert_segy_inmem_advanced(self, bits_per_voxel, blockshape):
        """Reads all data from input file to memory, compresses it and writes as .sgz file to disk,
        using custom compression unit order"""
        header = make_header(self.in_filename, bits_per_voxel, blockshape)

        t0 = time.time()
        data = segyio.tools.cube(self.in_filename)

        bits_per_voxel, blockshape = define_blockshape(bits_per_voxel, blockshape)

        padded_shape = (pad(data.shape[0], blockshape[0]),
                        pad(data.shape[1], blockshape[1]),
                        pad(data.shape[2], blockshape[2]))
        data_padded = np.zeros(padded_shape, dtype=np.float32)

        data_padded[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]] = data

        numpy_headers_arrays = get_header_arrays(self.in_filename)

        with open(self.out_filename, 'wb') as f:
            f.write(header)
            for i in range(data_padded.shape[0] // blockshape[0]):
                for x in range(data_padded.shape[1] // blockshape[1]):
                    for z in range(data_padded.shape[2] // blockshape[2]):
                        slice = data_padded[i*blockshape[0] : (i+1)*blockshape[0],
                                            x*blockshape[1] : (x+1)*blockshape[1],
                                            z*blockshape[2] : (z+1)*blockshape[2]].copy()
                        compressed_block = compress(slice, rate=bits_per_voxel)
                        f.write(compressed_block)
            for header_array in numpy_headers_arrays:
                f.write(header_array.tobytes())
        t3 = time.time()

        print("Total conversion time: {}".format(t3-t0))

    def convert_segy_stream(self, bits_per_voxel, blockshape):
        """Memory-efficient method of compressing SEG-Y file larger than machine memory.
        Requires at least n_crosslines x n_samples x blockshape[2] x 4 bytes of available memory"""
        t0 = time.time()

        bits_per_voxel, blockshape = define_blockshape(bits_per_voxel, blockshape)

        with segyio.open(self.in_filename) as segyfile:
            if self.max_xl is None and self.max_il is None:
                n_traces = segyfile.tracecount
            elif self.max_xl is None:
                n_traces = (self.max_il - self.min_il) * len(segyfile.xlines)
            elif self.max_il is None:
                n_traces = (self.max_xl - self.min_xl) * len(segyfile.ilines)
            else:
                n_traces = (self.max_xl - self.min_xl) * (self.max_il - self.min_il)

            headers_to_store = get_unique_headerwords(segyfile)
            numpy_headers_arrays = [np.zeros(n_traces, dtype=np.int32) for _ in range(len(headers_to_store))]

            if self.max_il is not None:
                assert 0 <= self.min_il < self.max_il, "min_il out of valid range"
                assert 0 < self.max_il <= len(segyfile.ilines), "max_il out of valid range"
            if self.max_xl is not None:
                assert 0 <= self.min_xl < self.max_xl, "min_xl out of valid range"
                assert 0 < self.max_xl <= len(segyfile.xlines), "max_xl out of valid range"

        loop = asyncio.new_event_loop()
        loop.run_until_complete(run_conversion_loop(self.in_filename, self.out_filename, bits_per_voxel, blockshape,
                                                    headers_to_store, numpy_headers_arrays,
                                                    self.min_il, self.max_il, self.min_xl, self.max_xl))
        loop.close()

        with open(self.out_filename, 'ab') as f:
            for header_array in numpy_headers_arrays:
                f.write(header_array.tobytes())

        t3 = time.time()
        print("Total conversion time: {}                     ".format(t3-t0))


class SgzConverter(SgzReader):
    """Writes 'advanced-layout' SGZ files from 'default-layout' SGZ files"""

    def __init__(self, file):
        super(SgzConverter, self).__init__()

    def convert_to_segy(self, out_file):
        # Currently only works for default SGZ layout (?)
        assert (self.blockshape[0] == 4)
        assert (self.blockshape[1] == 4)

        spec = segyio.spec()
        spec.samples = self.zslices
        spec.offsets = [0]
        spec.xlines = self.xlines
        spec.ilines = self.ilines
        spec.sorting = 2

        # seimcic-sfp stores the binary header from the source SEG-Y file.
        # In case someone forgot to do this, give them IEEE
        data_sample_format_code = bytes_to_int(
            self.headerbytes[DISK_BLOCK_BYTES+3225: DISK_BLOCK_BYTES+3227])
        if data_sample_format_code in [1, 5]:
            spec.format = data_sample_format_code
        else:
            spec.format = 5

        with warnings.catch_warnings():
            # segyio will warn us that out padded cube is not contiguous. This is expected, and safe.
            warnings.filterwarnings("ignore", message="Implicit conversion to contiguous array")
            with segyio.create(out_file, spec) as segyfile:
                self.read_variant_headers()
                for i, iline in enumerate(spec.ilines):
                    if i % self.blockshape[0] == 0:
                        decompressed = self.read_and_decompress_il_set(i)
                    for h in range(i * len(spec.xlines), (i + 1) * len(spec.xlines)):
                        segyfile.header[h] = self.gen_trace_header(h)
                    segyfile.iline[iline] = decompressed[i % self.blockshape[0], 0:self.n_xlines, 0:self.n_samples]

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
