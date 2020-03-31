from __future__ import division
import os
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache
import numpy as np
import segyio
from segyio import _segyio

from .loader import SgzLoader
from .version import SeismicZfpVersion
from .utils import pad, bytes_to_int, bytes_to_signed_int, gen_coord_list, FileOffset, get_correlated_diagonal_length, get_anticorrelated_diagonal_length
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES, SEGY_TEXT_HEADER_BYTES


class SgzReader(object):
    """Reads SGZ files

    Methods
    -------
    read_inline(il_id)
        Decompresses and returns one inline from SGZ file as 2D numpy array

    read_crossline(xl_id)
        Decompresses and returns one crossline from SGZ file as 2D numpy array

    read_zslice(zslice_id)
        Decompresses and returns one zslice from SGZ file as 2D numpy array

    read_subvolume(min_il, max_il, min_xl, max_xl, min_z, max_z)
        Decompresses and returns an arbitrary sub-volume from SGZ file as 3D numpy array
    """
    def __init__(self, file, filetype_checking=True, preload=False):
        """
        Parameters
        ----------
        file : str
            The SGZ filepath to be read

             : file handle in 'rb' mode
             Reuse an open file handle

        """

        # Class may be instantiated with either a file path or filehandle
        if not hasattr(file, 'read'):
            self.filename = file
            self.file = self.open_sgz_file()
        else:
            self.filename = file.name
            self.file = file
            # You have a file handle, go to the start!
            self.file.seek(0)

        self.headerbytes = self.file.read(DISK_BLOCK_BYTES)
        if filetype_checking and self.headerbytes[0:2] == b'\xc3\x40':
            msg = "This appears to be a SEGY file rather than an SGZ file, override with filetype_checking=False"
            raise RuntimeError(msg)

        self.n_header_blocks = bytes_to_int(self.headerbytes[0:4])
        if self.n_header_blocks != 1:
            self.file.seek(0)
            self.headerbytes = self.file.read(DISK_BLOCK_BYTES*self.n_header_blocks)

        # Read useful info out of the SGZ header
        self.file_version = self.get_file_version()
        self.n_samples, self.n_xlines, self.n_ilines, self.rate, self.blockshape = self.parse_dimensions()
        self.zslices, self.xlines, self.ilines = self.parse_coordinates()
        self.tracecount = len(self.ilines) * len(self.xlines)
        self.compressed_data_diskblocks, self.header_entry_length_bytes, self.n_header_arrays = self.parse_data_sizes()
        self.data_start_bytes = self.n_header_blocks * DISK_BLOCK_BYTES

        self.segy_traceheader_template = self.decode_traceheader_template()
        self.file_text_header = self.headerbytes[DISK_BLOCK_BYTES:
                                                 DISK_BLOCK_BYTES + SEGY_TEXT_HEADER_BYTES]

        self.file_binary_header = self.headerbytes[DISK_BLOCK_BYTES + SEGY_TEXT_HEADER_BYTES:
                                                   DISK_BLOCK_BYTES + SEGY_FILE_HEADER_BYTES]

        # Blockshape for original files
        if self.blockshape[0] == 0 or self.blockshape[1] == 0 or self.blockshape[2] == 0:
            self.blockshape = (4, 4, 2048//self.rate)

        self.shape_pad = (pad(self.n_ilines, self.blockshape[0]),
                          pad(self.n_xlines, self.blockshape[1]),
                          pad(self.n_samples, self.blockshape[2]))

        # These are useful units of measurement for SGZ files:

        # A 'compression unit' is the smallest decompressable piece of the SGZ file.
        # It is always 4-samples x 4-xlines x 4-ilines in physical dimensions, but its size
        # on disk will vary according to compression ratio.
        self.unit_bytes = int((4*4*4) * self.rate) // 8

        # A 'block' is a group of 'compression units' equal in size to a hardware disk block.
        # The 'compression units' may be arranged in any cuboid which matches the size of a disk block.
        # At the time of coding, standard commodity hardware uses 4kB disk blocks so check that
        # file has been written in using this convention.
        self.block_bytes = int(self.blockshape[0] * self.blockshape[1] * self.blockshape[2] * self.rate) // 8

        assert self.block_bytes % self.unit_bytes == 0
        assert self.block_bytes == DISK_BLOCK_BYTES, "self.block_bytes={}, should be {}".format(self.block_bytes,
                                                                                                DISK_BLOCK_BYTES)

        # A 'chunk' is a group of one or more 'blocks' which span a complete set of traces.
        # This will follow the xline and iline shape of a 'block'
        self.chunk_bytes = self.block_bytes * (self.shape_pad[2] // self.blockshape[2])
        assert self.chunk_bytes % self.block_bytes == 0

        # Placeholder. Don't read these if you're not going to use them
        self.variant_headers = None

        # Split out responsibility for I/O and decompression
        self.loader = SgzLoader(self.file, self.data_start_bytes, self.compressed_data_diskblocks, self.shape_pad,
                               self.blockshape, self.chunk_bytes, self.block_bytes, self.unit_bytes, self.rate, preload)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close_sgz_file()

    def open_sgz_file(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError("Rather than a beep, Or a rude error message, These words: 'File not found.'")
        return open(self.filename, 'rb')

    def close_sgz_file(self):
        self.file.close()

    def get_file_version(self):
        return SeismicZfpVersion(bytes_to_int(self.headerbytes[72:76]))

    def parse_dimensions(self):
        n_samples = bytes_to_int(self.headerbytes[4:8])
        n_xlines = bytes_to_int(self.headerbytes[8:12])
        n_ilines = bytes_to_int(self.headerbytes[12:16])
        rate = bytes_to_signed_int(self.headerbytes[40:44])

        if rate < 0:
            rate = 1 / -rate

        blockshape = (bytes_to_int(self.headerbytes[44:48]),
                      bytes_to_int(self.headerbytes[48:52]),
                      bytes_to_int(self.headerbytes[52:56]))

        return n_samples, n_xlines, n_ilines, rate, blockshape

    def parse_coordinates(self):
        zslices_list = gen_coord_list(bytes_to_int(self.headerbytes[16:20]),
                                      bytes_to_int(self.headerbytes[28:32]),
                                      bytes_to_int(self.headerbytes[4:8]))
        xlines_list = gen_coord_list(bytes_to_int(self.headerbytes[20:24]),
                                      bytes_to_int(self.headerbytes[32:36]),
                                      bytes_to_int(self.headerbytes[8:12]))
        ilines_list = gen_coord_list(bytes_to_int(self.headerbytes[24:28]),
                                      bytes_to_int(self.headerbytes[36:40]),
                                      bytes_to_int(self.headerbytes[12:16]))
        return zslices_list, xlines_list, ilines_list

    def parse_data_sizes(self):
        compressed_data_diskblocks = bytes_to_int(self.headerbytes[56:60])
        header_entry_length_bytes = bytes_to_int(self.headerbytes[60:64])
        n_header_arrays = bytes_to_int(self.headerbytes[64:68])

        return compressed_data_diskblocks, header_entry_length_bytes, n_header_arrays

    def decode_traceheader_template(self):
        raw_template = self.headerbytes[980:2048]
        template = [tuple((bytes_to_signed_int(raw_template[i*12 + j:i*12 + j + 4])
                           for j in range(0, 12, 4))) for i in range(89)]
        header_dict = {}
        header_count = 0
        for hv in template:
            tf = segyio.tracefield.TraceField(hv[0])
            if hv[1] != 0 or hv[2] == 0:
                # In these cases we have an invariant value
                header_dict[tf] = hv[1]

            elif segyio.tracefield.TraceField(hv[2]) in header_dict.keys():
                # We have a previously discovered header value
                header_dict[tf] = header_dict[segyio.tracefield.TraceField(hv[2])]
            else:
                # This is a new header value
                header_dict[tf] = FileOffset(DISK_BLOCK_BYTES*self.n_header_blocks +
                                             DISK_BLOCK_BYTES*self.compressed_data_diskblocks +
                                             header_count*self.header_entry_length_bytes)
                header_count += 1

        # We should find the same number of headers arrays as have been written!
        assert(header_count == self.n_header_arrays)
        return header_dict

    def read_variant_headers(self):
        if self.variant_headers is None:
            variant_headers = {}
            for k, v in self.segy_traceheader_template.items():
                if isinstance(v, FileOffset):
                    self.file.seek(v)
                    buffer = self.file.read(self.header_entry_length_bytes)
                    values = np.frombuffer(buffer, dtype=np.int32)
                    variant_headers[k] = values
            self.variant_headers = variant_headers
        else:
            pass

    def read_inline(self, il_id):
        """Reads one inline from SGZ file

        Parameters
        ----------
        il_id : int
            The ordinal number of the inline in the file

        Returns
        -------
        inline : numpy.ndarray of float32, shape: (n_xlines, n_samples)
            The specified inline, decompressed
        """
        if il_id < 0 or il_id >= self.n_ilines:
            raise IndexError("Index {} is out of range ({}, {})".format(il_id, 0, self.n_ilines - 1))
        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            decompressed = self.loader.read_and_decompress_il_set(4 * (il_id // 4))
            return decompressed[il_id % self.blockshape[0], 0:self.n_xlines, 0:self.n_samples]
        else:
            # Default to unoptimized general method
            return np.squeeze(self.read_subvolume(il_id, il_id + 1, 0, self.n_xlines, 0, self.n_samples))

    def read_crossline(self, xl_id):
        """Reads one crossline from SGZ file

        Parameters
        ----------
        xl_id : int
            The ordinal number of the crossline in the file

        Returns
        -------
        crossline : numpy.ndarray of float32, shape: (n_ilines, n_samples)
            The specified crossline, decompressed
        """
        if xl_id < 0 or xl_id >= self.n_xlines:
            raise IndexError("Index {} is out of range ({}, {})".format(xl_id, 0, self.n_xlines - 1))
        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            decompressed = self.loader.read_and_decompress_xl_set(4 * (xl_id // 4))
            return decompressed[0:self.n_ilines, xl_id % self.blockshape[1], 0:self.n_samples]
        else:
            # Default to unoptimized general method
            return np.squeeze(self.read_subvolume(0, self.n_ilines, xl_id, xl_id + 1, 0, self.n_samples))

    def read_zslice(self, zslice_id):
        """Reads one zslice from SGZ file (time or depth, depending on file contents)

        Parameters
        ----------
        zslice_id : int
            The ordinal number of the zslice in the file

        Returns
        -------
        zslice : numpy.ndarray of float32, shape: (n_ilines, n_xlines)
            The specified zslice (time or depth, depending on file contents), decompressed
        """
        if zslice_id < 0 or zslice_id >= self.n_samples:
            raise IndexError("Index {} is out of range ({}, {})".format(zslice_id, 0, self.n_samples - 1))
        blocks_per_dim = tuple(dim // size for dim, size in zip(self.shape_pad, self.blockshape))
        zslice_first_block_offset = zslice_id // self.blockshape[2]

        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            decompressed = self.loader.read_and_decompress_zslice_set(blocks_per_dim, zslice_first_block_offset,
                                                                       zslice_id)
            return decompressed[0:self.n_ilines, 0:self.n_xlines, zslice_id % 4]

        elif self.blockshape[2] == 4:
            decompressed = self.loader.read_and_decompress_zslice_set_adv(blocks_per_dim, zslice_first_block_offset)
            return decompressed[0:self.n_ilines, 0:self.n_xlines, zslice_id % 4]

        else:
            # Default to unoptimized general method
            return np.squeeze(self.read_subvolume(0, self.n_ilines, 0, self.n_xlines, zslice_id, zslice_id + 1))

    def read_correlated_diagonal(self, cd_id):
        """Reads one diagonal in the direction IL ~ XL

        Parameters
        ----------
        cd_id : int
            - The ordinal number of the correlated diagonal in the file,
            - Range [ -max(XL), +max(IL) ]

        Returns
        -------
        cd_slice : numpy.ndarray of float32, shape (n_diagonal_traces, n_samples)
            The specified cd_slice, decompressed.
        """
        if cd_id <= -self.n_xlines or cd_id >= self.n_ilines:
            raise IndexError("Index {} is out of range ({}, {})".format(cd_id, -self.n_xlines, self.n_ilines))
        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            cd_length = get_correlated_diagonal_length(cd_id, self.n_ilines, self.n_xlines)
            cd = np.zeros((cd_length, self.n_samples))

            if cd_id % 4 != 0:
                decompressed = self.loader.read_and_decompress_cd_set(4 * (cd_id // 4))
                decompressed_offset = self.loader.read_and_decompress_cd_set(4 * ((cd_id + 4) // 4))
            else:
                decompressed = decompressed_offset = self.loader.read_and_decompress_cd_set(4 * (cd_id // 4))

            for i in range(cd_length):
                if cd_id >= 0:
                    if i % 4 < 4 - (cd_id % 4):
                        cd[i] = decompressed[i + cd_id % 4, i % 4, 0:self.n_samples]
                    else:
                        cd[i] = decompressed_offset[i + cd_id % 4 - 4, i % 4, 0:self.n_samples]
                else:
                    if i % 4 < 4 - (abs(cd_id) % 4):
                        cd[i] = decompressed_offset[i, (i - cd_id) % 4, 0:self.n_samples]
                    else:
                        cd[i] = decompressed[i, (i + abs(cd_id)) % 4, 0:self.n_samples]
            return cd
        else:
            raise NotImplementedError("Diagonals can only be read from default layout SGZ files")

    def read_anticorrelated_diagonal(self, ad_id):
        """Reads one diagonal in the direction IL ~ -XL

        Parameters
        ----------
        ad_id : int
            - The ordinal number of the correlated diagonal in the file,
            - Range [0, max(XL)+max(IL) )

        Returns
        -------
        ad_slice : numpy.ndarray of float32, shape (n_diagonal_traces, n_samples)
            The specified ad_slice, decompressed.
        """
        if ad_id < 0 or ad_id >= self.n_ilines + self.n_xlines - 1:
            raise IndexError("Index {} is out of range ({}, {})".format(ad_id, 0, self.n_ilines + self.n_xlines - 1))
        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            ad_length = get_anticorrelated_diagonal_length(ad_id, self.n_ilines, self.n_xlines)
            ad = np.zeros((ad_length, self.n_samples))

            if (ad_id + 1) % 4 != 0 and ad_length > 3:
                decompressed = self.loader.read_and_decompress_ad_set(4 * (ad_id // 4))
                decompressed_offset = self.loader.read_and_decompress_ad_set(4 * ((ad_id - 4) // 4))
            else:
                decompressed = decompressed_offset = self.loader.read_and_decompress_ad_set(4 * (ad_id // 4))

            if ad_id < self.n_xlines:
                for i in range(ad_length):
                    if 3 - (i % 4) >= 3 - (ad_id % 4):
                        ad[i] = decompressed[i, (3 - i + ad_id + 1) % 4, 0:self.n_samples]
                    else:
                        ad[i] = decompressed_offset[i, (3 - i + ad_id + 1) % 4, 0:self.n_samples]
            else:
                start = (4 - (self.n_xlines % 4)) % 4
                for i in range(start, ad_length + start):
                    i2 = i + (ad_id + 1) % 4
                    if i2 % 4 < (ad_id + 1) % 4:
                        ad[i - start] = decompressed[i2 - 4, (3 - i2 + ad_id + 1) % 4, 0:self.n_samples]
                    else:
                        if self.n_xlines <= ad_id < self.shape_pad[1] and self.n_xlines != self.shape_pad[1] and (ad_id + 1) % 4 != 0:
                            ad[i - start] = decompressed_offset[i2 - 4, (3 - i2 + ad_id + 1) % 4, 0:self.n_samples]
                        else:
                            ad[i - start] = decompressed_offset[i2, (3 - i2 + ad_id + 1) % 4, 0:self.n_samples]
            return ad
        else:
            raise NotImplementedError("Diagonals can only be read from default layout SGZ files")

    def read_subvolume(self, min_il, max_il, min_xl, max_xl, min_z, max_z):
        """Reads a sub-volume from SGZ file

        Parameters
        ----------
        min_il : int
            The ordinal number of the minimum inline to read (C-indexing)
        max_il : int
            The ordinal number of the maximum inline to read (C-indexing)

        min_xl : int
            The ordinal number of the minimum crossline to read (C-indexing)
        max_xl : int
            The ordinal number of the maximum crossline to read (C-indexing)

        min_z : int
            The ordinal number of the minimum zslice to read (C-indexing)
        max_z : int
            The ordinal number of the maximum zslice to read (C-indexing)


        Returns
        -------
        subvolume : numpy.ndarray of float32, shape (max_il - min_il, max_xl - min_xl, max_z - min_z)
            The specified subvolume, decompressed
        """
        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            decompressed = self.loader.read_and_decompress_chunk_range(max_il, max_xl, max_z, min_il, min_xl, min_z)

            return decompressed[min_il%4:(min_il%4)+max_il-min_il,
                                min_xl%4:(min_xl%4)+max_xl-min_xl,
                                min_z%4:(min_z%4)+max_z-min_z]
        else:
            # This works generally, but is pretty wasteful for IL or XL reads.
            # Really should encourage users to stick with either:
            #  - blockshape[2] == 4
            #  - blockshape[0] == blockshape[1] == 4
            decompressed = self.loader.read_unshuffle_and_decompress_chunk_range(max_il, max_xl, max_z, min_il, min_xl, min_z)

            return decompressed[min_il%self.blockshape[0]:(min_il%self.blockshape[0])+max_il-min_il,
                                min_xl%self.blockshape[1]:(min_xl%self.blockshape[1])+max_xl-min_xl,
                                min_z%self.blockshape[2]:(min_z%self.blockshape[2])+max_z-min_z]

    def read_volume(self):
        """Reads the whole volume from SGZ file

        Returns
        -------
        volume : numpy.ndarray of float32, shape (n_ilines, n_xline, n_samples)
            The whole volume, decompressed
        """
        return self.read_subvolume(0, self.n_ilines,
                                   0, self.n_xlines,
                                   0, self.n_samples)

    def get_trace(self, index):
        """Reads the whole volume from SGZ file

        Parameters
        ----------
        index : int
            The ordinal number of the trace in the file

        Returns
        -------
        trace : numpy.ndarray of float32, shape (n_samples)
            A single trace, decompressed
        """
        il, xl = index // self.n_xlines, index % self.n_xlines
        min_il = self.blockshape[0] * (il // self.blockshape[0])
        min_xl = self.blockshape[1] * (xl // self.blockshape[1])
        chunk = self.read_containing_chunk(min_il, min_xl)
        trace = chunk[il % self.blockshape[0], xl % self.blockshape[1], :]
        return np.squeeze(trace)

    # Using a cache of 2048 chunks implies:
    #     - 1GB memory usage at 32KB uncompressed traces. Reduce for machines with memory constraints
    #     - Sequential reading of traces over inlines and crosslines will give 15/16 cache hits
    # (assuming 4x4 chunks... traces shouldn't be accessed intensively from other layouts!)
    @lru_cache(maxsize=2048)
    def read_containing_chunk(self, ref_il, ref_xl):
        assert ref_il % self.blockshape[0] == 0
        assert ref_xl % self.blockshape[1] == 0
        return self.read_subvolume(ref_il, ref_il + self.blockshape[0],
                                   ref_xl, ref_xl + self.blockshape[1],
                                   0, self.n_samples)

    def gen_trace_header(self, index):
        header = self.segy_traceheader_template.copy()
        for k, v in header.items():
            if isinstance(v, FileOffset):
                header[k] = self.variant_headers[k][index]
        return header

    def get_file_binary_header(self):
        return segyio.segy.Field(self.file_binary_header, kind='binary')

    def get_file_text_header(self):
        return [bytearray(self.file_text_header.decode("cp037"),
                          encoding="ascii", errors="ignore")]
