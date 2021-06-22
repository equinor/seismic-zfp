from __future__ import division
import os
import platform
import random
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache
import numpy as np
import segyio
from segyio import _segyio

from .loader import SgzLoader
from .version import SeismicZfpVersion
from .utils import pad, bytes_to_int, bytes_to_signed_int, gen_coord_list, FileOffset, get_correlated_diagonal_length, get_anticorrelated_diagonal_length, get_chunk_cache_size
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES, SEGY_TEXT_HEADER_BYTES


class SgzReader(object):
    """Reads SGZ files

    Methods
    -------
    read_inline_number(il_no)
        Decompresses and returns one inline from SGZ file as 2D numpy array (by inline number)

    read_inline(il_id)
        Decompresses and returns one inline from SGZ file as 2D numpy array (by inline ordinal)

    read_crossline_number(xl_no)
        Decompresses and returns one crossline from SGZ file as 2D numpy array (by crossline number)

    read_crossline(xl_id)
        Decompresses and returns one crossline from SGZ file as 2D numpy array (by crossline ordinal)

    read_zslice(zslice_id)
        Decompresses and returns one zslice from SGZ file as 2D numpy array

    read_correlated_diagonal(cd_id)
        Decompresses and returns one diagonal IL ~ XL from SGZ file as 2D numpy array

    read_anticorrelated_diagonal(ad_id)
        Decompresses and returns one diagonal IL ~ -XL from SGZ file as 2D numpy array

    read_subvolume(min_il, max_il, min_xl, max_xl, min_z, max_z)
        Decompresses and returns an arbitrary sub-volume from SGZ file as 3D numpy array

    read_volume(min_il, max_il, min_xl, max_xl, min_z, max_z)
        Decompresses and returns full cube from SGZ file as 3D numpy array

    get_trace(index)
        Decompress and return a single trace from SGZ file as 1D numpy array

    gen_trace_header(index)
        Create and return dictionary of headerword-value pairs for specified trace

    get_file_binary_header
        Returns dictionary of headerword-value pairs from SEG-Y binary file header

    get_file_text_header
        Returns SEG-Y textual file header

    get_file_version
        Returns version of seismic-zfp library which wrote the SGZ file
    """
    def __init__(self, file, filetype_checking=True, preload=False, chunk_cache_size=None):
        """
        Parameters
        ----------
        file : str
            The SGZ filepath to be read

             : file handle in 'rb' mode
             Reuse an open file handle

        filetype_checking : bool, optional
            Decline to attempt reading files which look like SEG-Y

        preload : bool, optional
            Read whole volume (compressed) into memory at instantiation

        chunk_cache_size : int, optional
            Number of chunks to cache when reading traces, increase for long diagonals
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
        self.n_samples, self.n_xlines, self.n_ilines, self.rate, self.blockshape = self._parse_dimensions()
        self.zslices, self.xlines, self.ilines = self._parse_coordinates()
        self.tracecount = len(self.ilines) * len(self.xlines)
        self.compressed_data_diskblocks, self.header_entry_length_bytes, self.n_header_arrays = self._parse_data_sizes()
        self.data_start_bytes = self.n_header_blocks * DISK_BLOCK_BYTES

        self.segy_traceheader_template = self._decode_traceheader_template()
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
        
        self.range_error = "Index {} is out of range [{}, {}]. Try using slice ordinals instead of numbers?"

        # Split out responsibility for I/O and decompression
        self.loader = SgzLoader(self.file, self.data_start_bytes, self.compressed_data_diskblocks, self.shape_pad,
                               self.blockshape, self.chunk_bytes, self.block_bytes, self.unit_bytes, self.rate, preload)

        # Using default cache of 2048 chunks implies:
        #     - 1GB memory usage at 32KB uncompressed traces. Reduce for machines with memory constraints
        #     - Sequential reading of traces over inlines and crosslines will give 15/16 cache hits, and
        #       over diagonals will give 9/16 cache hits.
        # (assuming 4x4 chunks... traces shouldn't be accessed intensively from other layouts!)
        #
        # Quis Habemus Servamus
        if chunk_cache_size is None:
            chunk_cache_size = get_chunk_cache_size(self.shape_pad[0] // self.blockshape[0],
                                                    self.shape_pad[1] // self.blockshape[1])
        self._read_containing_chunk_cached = lru_cache(maxsize=chunk_cache_size)(self._read_containing_chunk)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close_sgz_file()

    def open_sgz_file(self):
        if not os.path.exists(self.filename):
            msgs = ["Rather than a beep  Or a rude error message  These words: 'File not found.'",
                    "A file that big?  It might be very useful.  But now it is gone.",
                    "Three things are certain:  Death, taxes, and lost data.  Guess which has occurred."]
            raise FileNotFoundError("Cannot find {} ... {}".format(self.filename, random.choice(msgs)))
        return open(self.filename, 'rb')

    def close_sgz_file(self):
        self.file.close()

    def get_file_version(self):
        return SeismicZfpVersion(bytes_to_int(self.headerbytes[72:76]))

    def _parse_dimensions(self):
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

    def _parse_coordinates(self):
        sample_rate_ms = bytes_to_int(self.headerbytes[28:32])

        # Use microseconds to store sample interval for files written with version 0.1.7 and above.
        # zslices is still in milliseconds for segyio compatibility, but now has float type rather than int.
        # Conversion from ZGY still uses milliseconds, this is safe as zgy2sgz gives a version of 0.0.0
        if self.file_version > SeismicZfpVersion("0.1.6"):
            sample_rate_ms /= 1000

        zslices_list = gen_coord_list(bytes_to_signed_int(self.headerbytes[16:20]),
                                      sample_rate_ms,
                                      bytes_to_int(self.headerbytes[4:8])).astype('float')
        xlines_list = gen_coord_list(bytes_to_int(self.headerbytes[20:24]),
                                     bytes_to_int(self.headerbytes[32:36]),
                                     bytes_to_int(self.headerbytes[8:12])).astype('intc')
        ilines_list = gen_coord_list(bytes_to_int(self.headerbytes[24:28]),
                                     bytes_to_int(self.headerbytes[36:40]),
                                     bytes_to_int(self.headerbytes[12:16])).astype('intc')
        return zslices_list, xlines_list, ilines_list

    def _parse_data_sizes(self):
        compressed_data_diskblocks = bytes_to_int(self.headerbytes[56:60])
        header_entry_length_bytes = bytes_to_int(self.headerbytes[60:64])
        n_header_arrays = bytes_to_int(self.headerbytes[64:68])

        return compressed_data_diskblocks, header_entry_length_bytes, n_header_arrays

    def _decode_traceheader_template(self):
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

    def read_inline_number(self, il_no):
        """Reads one inline from SGZ file

        Parameters
        ----------
        il_no : int
            The inline number

        Returns
        -------
        inline : numpy.ndarray of float32, shape: (n_xlines, n_samples)
            The specified inline, decompressed
        """
        return self.read_inline(np.where(self.ilines == il_no)[0][0])

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
        if not 0 <= il_id < self.n_ilines:
            raise IndexError(self.range_error.format(il_id, 0, self.n_ilines - 1))
        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            decompressed = self.loader.read_and_decompress_il_set(4 * (il_id // 4))
            return decompressed[il_id % self.blockshape[0], 0:self.n_xlines, 0:self.n_samples]
        else:
            # Default to unoptimized general method
            return np.squeeze(self.read_subvolume(il_id, il_id + 1, 0, self.n_xlines, 0, self.n_samples))

    def read_crossline_number(self, xl_no):
        """Reads one crossline from SGZ file

        Parameters
        ----------
        xl_no : int
            The crossline number

        Returns
        -------
        crossline : numpy.ndarray of float32, shape: (n_ilines, n_samples)
            The specified crossline, decompressed
        """
        return self.read_crossline(np.where(self.xlines == xl_no)[0][0])

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
        if not 0 <= xl_id < self.n_xlines:
            raise IndexError(self.range_error.format(xl_id, 0, self.n_xlines - 1))
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
        if not 0 <= zslice_id < self.n_samples:
            raise IndexError(self.range_error.format(zslice_id, 0, self.n_samples - 1))
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
        if not -self.n_xlines < cd_id < self.n_ilines:
            raise IndexError(self.range_error.format(cd_id, -self.n_xlines, self.n_ilines))

        cd_len = get_correlated_diagonal_length(cd_id, self.n_ilines, self.n_xlines)
        cd = np.zeros((cd_len, self.n_samples))
        if cd_id >= 0:
            for d in range(cd_len):
                cd[d, :] = self.get_trace((d + cd_id) * self.n_xlines + d)
        else:
            for d in range(cd_len):
                cd[d, :] = self.get_trace(d * self.n_xlines+ d - cd_id)
        return cd

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
        if not 0 <= ad_id < self.n_ilines + self.n_xlines - 1:
            raise IndexError(self.range_error.format(ad_id, 0, self.n_ilines + self.n_xlines - 2))

        ad_len = get_anticorrelated_diagonal_length(ad_id, self.n_ilines, self.n_xlines)
        ad = np.zeros((ad_len, self.n_samples))
        if ad_id < self.n_xlines:
            for d in range(ad_len):
                ad[d, :] = self.get_trace(ad_id + d*(self.n_xlines - 1))
        else:
            for d in range(ad_len):
                ad[d, :] = self.get_trace((ad_id - self.n_xlines + 1 + d) * self.n_xlines
                                                  + (self.n_xlines - d - 1))
        return ad

    def read_subvolume(self, min_il, max_il, min_xl, max_xl, min_z, max_z, access_padding=False):
        """Reads a sub-volume from SGZ file

        Parameters
        ----------
        min_il : int
            The index of the first inline to get from the cube. Use 0 to for the first inline in the cube
        max_il : int
            The index of the last inline to get, non inclusive. To get one inline, use max_il = min_il + 1

        min_xl : int
            The index of the first crossline to get from the cube. Use 0 for the first crossline in the cube
        max_xl : int
            The index of the last crossline to get, non inclusive. To get one crossline, use max_xl = min_xl + 1

        min_z : int
            The index of the first time sample to get from the cube. Use 0 for the first time sample in the cube
        max_z : int
            The index of the last time sample to get, non inclusive. To get one time sample, use max_z = min_z + 1
            
        access_padding : bool, optional
            Functions which manage voxels used for padding themselves may relax bounds-checking to padded dimensions

        Returns
        -------
        subvolume : numpy.ndarray of float32, shape (max_il - min_il, max_xl - min_xl, max_z - min_z)
            The specified subvolume, decompressed
        """
        upper_il = self.shape_pad[0] if access_padding else self.n_ilines
        upper_xl = self.shape_pad[1] if access_padding else self.n_xlines
        upper_z = self.shape_pad[2] if access_padding else self.n_samples

        if not (0 <= min_il < upper_il and 0 < max_il <= upper_il and max_il > min_il):
            raise IndexError(self.range_error.format(min_il, max_il, 0, self.n_ilines - 1))

        if not (0 <= min_xl < upper_xl and 0 < max_xl <= upper_xl and max_xl > min_xl):
            raise IndexError(self.range_error.format(min_xl, max_xl, 0, self.n_xlines - 1))

        if not (0 <= min_z < upper_z and 0 < max_z <= upper_z and max_z > min_z):
            raise IndexError(self.range_error.format(min_z, max_z, 0, self.n_samples - 1))

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

            return decompressed[min_il % self.blockshape[0]:(min_il % self.blockshape[0])+max_il-min_il,
                                min_xl % self.blockshape[1]:(min_xl % self.blockshape[1])+max_xl-min_xl,
                                min_z % self.blockshape[2]:(min_z % self.blockshape[2])+max_z-min_z]

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
        """Reads one trace from SGZ file

        Parameters
        ----------
        index : int
            The ordinal number of the trace in the file

        Returns
        -------
        trace : numpy.ndarray of float32, shape (n_samples)
            A single trace, decompressed
        """
        if not 0 <= index < self.n_ilines * self.n_xlines:
            if platform.system() == 'Windows':
                print('Yesterday it worked, Today it is not working, Windows is like that')
            raise IndexError(self.range_error.format(index, 0, self.tracecount))

        il, xl = index // self.n_xlines, index % self.n_xlines
        min_il = self.blockshape[0] * (il // self.blockshape[0])
        min_xl = self.blockshape[1] * (xl // self.blockshape[1])
        chunk = self._read_containing_chunk_cached(min_il, min_xl)
        trace = chunk[il % self.blockshape[0], xl % self.blockshape[1], :]
        return np.squeeze(trace)

    def _read_containing_chunk(self, ref_il, ref_xl):
        assert ref_il % self.blockshape[0] == 0
        assert ref_xl % self.blockshape[1] == 0
        return self.read_subvolume(ref_il, ref_il + self.blockshape[0],
                                   ref_xl, ref_xl + self.blockshape[1],
                                   0, self.n_samples, access_padding=True)

    def gen_trace_header(self, index, load_all_headers=False):
        """Generates one trace header from SGZ file

        Parameters
        ----------
        index : int
            The ordinal number of the trace header in the file

        load_all_headers : bool, optional
            Load full header-arrays from disk.
            More efficient if accessing headers for whole file.

        Returns
        -------
        header : dict
            A single header as a dictionary of headerword-value pairs
        """
        if not 0 <= index < self.n_ilines * self.n_xlines:
            raise IndexError(self.range_error.format(index, 0, self.tracecount))

        header = self.segy_traceheader_template.copy()

        for k, v in header.items():
            if isinstance(v, FileOffset):
                if load_all_headers:
                    self.read_variant_headers()
                    header[k] = self.variant_headers[k][index]
                else:
                    self.file.seek(v + 4*index)  # A 32-bit int is 4 bytes
                    header[k] = np.frombuffer(self.file.read(4), dtype=np.int32)[0]
        return header

    def get_file_binary_header(self):
        return segyio.segy.Field(self.file_binary_header, kind='binary')

    def get_file_text_header(self):
        return [bytearray(self.file_text_header.decode("cp037"),
                          encoding="ascii", errors="ignore")]
