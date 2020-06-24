from __future__ import print_function
import os
import warnings
import numpy as np
import segyio
import time
from psutil import virtual_memory

from .utils import pad, define_blockshape, bytes_to_int, Geometry, InferredGeometry
from .headers import get_unique_headerwords
from .conversion_utils import run_conversion_loop
from .read import SgzReader
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES


class SegyConverter(object):
    """Writes SEG-Y files from SGZ files"""

    def __init__(self, in_filename, min_il=None, max_il=None, min_xl=None, max_xl=None):
        """
        Parameters
        ----------

        in_filename: str
            The SEGY file to be converted to SGZ

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Non Timetus Messor
        pass

    def run(self, out_filename, bits_per_voxel=4, blockshape=(4, 4, -1), method="Stream", reduce_iops=False):
        """General entrypoint for converting SEG-Y files to SGZ

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

        method: str
            DEPRECATED: Flag to indicate method for reading SEG-Y
            - "InMemory" : Read whole SEG-Y cube into memory before compressing - Removed in v0.0.12
            - "Stream" : Read 4 inlines at a time... compress, rinse, repeat

        reduce_iops: bool
            Flag to indicate whether compression should attempt to minimize the number
            of iops required to read the input SEG-Y file by reading whole inlines including
            headers in one go. Falls back to segyio if incorrect. Useful under Windows.

        Raises
        ------

        NotImplementedError
            If method is not one of "InMemory" or Stream"

        """
        self.out_filename = out_filename
        bits_per_voxel, blockshape = define_blockshape(bits_per_voxel, blockshape)

        if method == "Stream":
            print("Converting: In={}, Out={}".format(self.in_filename, self.out_filename))
            self.convert_segy_stream(bits_per_voxel, blockshape, reduce_iops=reduce_iops)
        else:
            raise NotImplementedError("Invalid conversion method {}: only 'Stream' is supported".format(method))

    def convert_segy_stream(self, bits_per_voxel, blockshape, reduce_iops=False):
        """Memory-efficient method of compressing SEG-Y file larger than machine memory.
        Requires at least n_crosslines x n_samples x blockshape[2] x 4 bytes of available memory"""
        t0 = time.time()

        if not os.path.exists(self.in_filename):
            msg = "With searching comes loss,  and the presence of absence:  'My Segy' not found."
            raise FileNotFoundError(msg)

        with segyio.open(self.in_filename, mode='r', strict=False) as segyfile:

            if self.geom is None:
                if segyfile.unstructured:
                    print("SEG-Y file is unstructured and no geometry provided. Determining this may take some time...")
                    traces_ref = {(h[189], h[193]): i for i, h in enumerate(segyfile.header)}
                    self.geom = InferredGeometry(traces_ref)
                else:
                    self.geom = Geometry(0, len(segyfile.ilines), 0, len(segyfile.xlines))
            n_traces = len(self.geom.ilines) * len(self.geom.xlines)
            inline_set_bytes = blockshape[0] * (len(self.geom.xlines) * len(segyfile.samples)) * 4

            headers_to_store = get_unique_headerwords(segyfile)
            numpy_headers_arrays = [np.zeros(n_traces, dtype=np.int32) for _ in range(len(headers_to_store))]

        if inline_set_bytes > virtual_memory().total // 2:
            print("One inline set is {} bytes, machine memory is {} bytes".format(inline_set_bytes, virtual_memory().total))
            print("Try using fewer inlines in the blockshape, or compressing a subcube")
            raise RuntimeError("ABORTED effort: Close all that you have. You ask way too much.")

        max_queue_length = min(16, (virtual_memory().total // 2) // inline_set_bytes)
        print("VirtualMemory={}MB, InlineSet={}MB : Using queue of length {}".format(virtual_memory().total/(1024*1024),
                                                                                     inline_set_bytes/(1024*1024),
                                                                                     max_queue_length))

        run_conversion_loop(self.in_filename, self.out_filename, bits_per_voxel, blockshape,
                            headers_to_store, numpy_headers_arrays, self.geom,
                            queuesize=max_queue_length, reduce_iops=reduce_iops)

        with open(self.out_filename, 'ab') as f:
            for header_array in numpy_headers_arrays:
                f.write(header_array.tobytes())

        t3 = time.time()
        print("Total conversion time: {}                     ".format(t3-t0))


class SgzConverter(SgzReader):
    """Writes 'advanced-layout' SGZ files from 'default-layout' SGZ files"""

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

        # seimcic-zfp stores the binary header from the source SEG-Y file.
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
                        decompressed = self.loader.read_and_decompress_il_set(i)
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
