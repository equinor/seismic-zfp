import os
import platform
import random
from functools import lru_cache
import numpy as np
import segyio
from segyio import _segyio

from .loader import SgzLoader
from .version import SeismicZfpVersion
from .utils import (pad, bytes_to_int, bytes_to_signed_int, get_chunk_cache_size,
                    coord_to_index, gen_coord_list, FileOffset,
                    get_correlated_diagonal_length, get_anticorrelated_diagonal_length)
import seismic_zfp
from .sgzconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES, SEGY_TEXT_HEADER_BYTES
from .headers import HeaderwordInfo

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    BlobServiceClient = None

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

    get_trace(index, min_sample_id=None, max_sample_id=None)
        Decompress, optionally crop, and return a single trace from SGZ file as 1D numpy array

    get_trace_by_coord(index, min_sample_no=None, max_sample_no=None)
        Decompress, crop by coordinates, and return a single trace from SGZ file as 1D numpy array

    get_tracefield_values(tracefield):
        Efficiently provides all trace header values for a given trace header field

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

        # Class may be instantiated with either a file path, filehandle, URL or blobclient
        if hasattr(file, 'download_blob'):
            # We have a blobclient
            self._filename = file.blob_name
            self.file = file
            self.file.read_range = seismic_zfp.utils.read_range_blob
            self.local = False
        elif hasattr(file, 'read'):
            # We have a filehandle
            self._filename = file.name
            self.file = file
            # You have a file handle, go to the start!
            self.file.seek(0)
            self.file.read_range = seismic_zfp.utils.read_range_file
            self.local = True
        else:
            if isinstance(file, tuple):
                # We have a URL
                if BlobServiceClient is None:
                    raise ImportError("File type requires azure-storage-blob. Install optional dependency seismic-zfp[azure] with pip.")
                blob_service_client = BlobServiceClient(account_url=file[0])
                self.file = blob_service_client.get_blob_client(container=file[1], blob=file[2])
                self.file.read_range = seismic_zfp.utils.read_range_blob
                self.local = False
            else:
                # We have a file path
                self._filename = file
                self.file = self.open_sgz_file()
                self.file.read_range = seismic_zfp.utils.read_range_file
                self.local = True


        self.headerbytes = self.file.read_range(self.file, 0, DISK_BLOCK_BYTES)
        if filetype_checking and self.headerbytes[0:2] == b'\xc3\x40':
            msg = "This appears to be a SEGY file rather than an SGZ file, override with filetype_checking=False"
            raise RuntimeError(msg)

        self.n_header_blocks = bytes_to_int(self.headerbytes[0:4])
        if self.n_header_blocks != 1:
            self.headerbytes = self.file.read_range(self.file, 0, DISK_BLOCK_BYTES*self.n_header_blocks)

        # Read useful info out of the SGZ header
        self.file_version = self.get_file_version()
        self.n_samples, self.n_xlines, self.n_ilines, self.rate, self.blockshape = self._parse_dimensions()
        self.zslices, self.xlines, self.ilines = self._parse_coordinates()
        self.compressed_data_diskblocks, self.header_entry_length_bytes, self.n_header_arrays = self._parse_data_sizes()
        self.data_start_bytes = self.n_header_blocks * DISK_BLOCK_BYTES

        if self.file_version > SeismicZfpVersion("0.2.1"):
            self.tracecount = bytes_to_int(self.headerbytes[68:72])
            self.padded_header_entry_length_bytes = (512 + 512 * ((self.header_entry_length_bytes - 1) // 512))
        else:
            self.tracecount = self.n_ilines * self.n_xlines
            self.padded_header_entry_length_bytes = self.header_entry_length_bytes

        self.segy_traceheader_template = self._decode_traceheader_template()
        self.stored_header_keys = [k for k, v in self.segy_traceheader_template.items() if isinstance(v, FileOffset)]
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
        self.variant_headers = {}
        self.include_padding = None
        
        self.range_error = "Index {} is out of range [{}, {}]. Try using slice ordinals instead of numbers?"

        # Split out responsibility for I/O and decompression
        self.loader = SgzLoader(self.file, self.data_start_bytes, self.compressed_data_diskblocks,
                                self.shape_pad, self.blockshape, self.chunk_bytes, self.block_bytes, self.unit_bytes,
                                self.rate, self.local, preload)

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
        self.structured = (self.tracecount == self.n_ilines * self.n_xlines)
        self.mask = None

    def __repr__(self):
        return f'SgzReader({self._filename})'

    def __str__(self):
        return f'seismic-zfp file {self._filename}, {self.file_version}:\n' \
               f'  compression ratio: {int(32/self.rate)}:1\n' \
               f'  inlines: {self.n_ilines} [{self.ilines[0]}, {self.ilines[-1]}]\n' \
               f'  crosslines: {self.n_xlines} [{self.xlines[0]}, {self.xlines[-1]}]\n' \
               f'  samples: {self.n_samples} [{self.zslices[0]}, {self.zslices[-1]}]\n' \
               f'  traces: {self.tracecount}\n' \
               f'  Header arrays: {self.stored_header_keys}'

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def open_sgz_file(self):
        if not os.path.exists(self._filename):
            msgs = ["Rather than a beep  Or a rude error message  These words: 'File not found.'",
                    "A file that big?  It might be very useful.  But now it is gone.",
                    "Three things are certain:  Death, taxes, and lost data.  Guess which has occurred."]
            raise FileNotFoundError("Cannot find {} ... {}".format(self._filename, random.choice(msgs)))
        return open(self._filename, 'rb')

    def close_sgz_file(self):
        self.file.close()

    def close(self):
        self.loader.clear_cache()
        self.close_sgz_file()

    def get_file_version(self):
        return SeismicZfpVersion(bytes_to_int(self.headerbytes[72:76]))

    def get_file_source_code(self):
        return bytes_to_int(self.headerbytes[76:80])

    def get_header_detection_method_code(self):
        return bytes_to_int(self.headerbytes[80:84])

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
        self.hw_info = HeaderwordInfo(self.tracecount, buffer=raw_template)
        return self.hw_info.get_header_dict(self.n_header_arrays,
                                            self.n_header_blocks,
                                            self.compressed_data_diskblocks,
                                            self.padded_header_entry_length_bytes)

    def get_inline_index(self, il_no):
        """Get inline index from inline number"""
        return coord_to_index(il_no, self.ilines)

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
        return self.read_inline(self.get_inline_index(il_no))

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


    def get_crossline_index(self, xl_no):
        """Get crossline index from crossline number"""
        return coord_to_index(xl_no, self.xlines)

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
        return self.read_crossline(self.get_crossline_index(xl_no))

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


    def get_zslice_index(self, zslice_no, include_stop=False):
        """Get zslice index from sample time/depth"""
        return coord_to_index(zslice_no, self.zslices, include_stop=include_stop)

    def read_zslice_coord(self, zslice_no):
        """Reads one zslice from SGZ file (time or depth, depending on file contents)

        Parameters
        ----------
        zslice_no : int
            The sample time/depth to return a zslice from

        Returns
        -------
        zslice : numpy.ndarray of float32, shape: (n_ilines, n_xlines)
            The specified zslice (time or depth, depending on file contents), decompressed
        """
        return self.read_zslice(self.get_zslice_index(zslice_no))

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


    def read_correlated_diagonal(self, cd_id, min_cd_idx=None, max_cd_idx=None, min_sample_idx=None, max_sample_idx=None):
        """Reads one diagonal in the direction IL ~ XL

        Parameters
        ----------
        cd_id : int
            - The ordinal number of the correlated diagonal in the file,
            - Range [ -max(XL), +max(IL) ]

        min_cd_idx : int
            - Start trace index, relative to start of full diagonal. Range as cd_id.

        max_cd_idx : int
            - Stop trace index, relative to start of full diagonal. Range as cd_id.

        min_sample_idx : int
            - Start sample index in trace

        max_sample_idx : int
            - Stop sample index in trace

        Returns
        -------
        cd_slice : numpy.ndarray of float32
            - Shape (n_diagonal_traces OR max_cd_idx-min_cd_idx, n_samples OR max_sample_idx-min_sample_idx)
            The specified cd_slice, decompressed.
        """
        if not -self.n_xlines < cd_id < self.n_ilines:
            raise IndexError(self.range_error.format(cd_id, -self.n_xlines, self.n_ilines))

        max_cd_len = get_correlated_diagonal_length(cd_id, self.n_ilines, self.n_xlines)
        if min_cd_idx is None or max_cd_idx is None:
            cd_len = max_cd_len
            min_cd_idx = 0
        else:
            if not 0 <= min_cd_idx < max_cd_len:
                raise IndexError(self.range_error.format(min_cd_idx, 0, max_cd_len-1))
            if not 0 < max_cd_idx <= max_cd_len:
                raise IndexError(self.range_error.format(max_cd_idx, 1, max_cd_len))
            cd_len = max_cd_idx - min_cd_idx

        if min_sample_idx is None or max_sample_idx is None:
            cd = np.zeros((cd_len, self.n_samples))
        else:
            cd = np.zeros((cd_len, max_sample_idx - min_sample_idx))

        if cd_id >= 0:
            for d in range(min_cd_idx, cd_len + min_cd_idx):
                cd[d - min_cd_idx, :] = self.get_trace((d + cd_id) * self.n_xlines + d,
                                                       min_sample_id=min_sample_idx,
                                                       max_sample_id=max_sample_idx)
        else:
            for d in range(min_cd_idx, cd_len + min_cd_idx):
                cd[d - min_cd_idx, :] = self.get_trace(d * self.n_xlines + d - cd_id,
                                                       min_sample_id=min_sample_idx,
                                                       max_sample_id=max_sample_idx)
        return cd

    def read_anticorrelated_diagonal(self, ad_id, min_ad_idx=None, max_ad_idx=None, min_sample_idx=None, max_sample_idx=None):
        """Reads one diagonal in the direction IL ~ -XL

        Parameters
        ----------
        ad_id : int
            - The ordinal number of the correlated diagonal in the file,
            - Range [0, max(XL)+max(IL) )

        min_ad_idx : int
            - Start trace index, relative to start of full diagonal. Range as cd_id.

        max_ad_idx : int
            - Stop trace index, relative to start of full diagonal. Range as cd_id.

        min_sample_idx : int
            - Start sample index in trace

        max_sample_idx : int
            - Stop sample index in trace

        Returns
        -------
        ad_slice : numpy.ndarray of float32
            - Shape (n_diagonal_traces OR max_ad_idx-min_ad_idx, n_samples OR max_sample_idx-min_sample_idx)
            The specified ad_slice, decompressed.
        """
        if not 0 <= ad_id < self.n_ilines + self.n_xlines - 1:
            raise IndexError(self.range_error.format(ad_id, 0, self.n_ilines + self.n_xlines - 2))

        max_ad_len = get_anticorrelated_diagonal_length(ad_id, self.n_ilines, self.n_xlines)
        if min_ad_idx is None or max_ad_idx is None:
            ad_len = max_ad_len
            min_ad_idx = 0
        else:
            if not 0 <= min_ad_idx < max_ad_len:
                raise IndexError(self.range_error.format(min_ad_idx, 0, max_ad_len-1))
            if not 0 < max_ad_idx <= max_ad_len:
                raise IndexError(self.range_error.format(max_ad_idx, 1, max_ad_len))
            ad_len = max_ad_idx - min_ad_idx

        if min_sample_idx is None or max_sample_idx is None:
            ad = np.zeros((ad_len, self.n_samples))
        else:
            ad = np.zeros((ad_len, max_sample_idx - min_sample_idx))

        if ad_id < self.n_xlines:
            for d in range(min_ad_idx, ad_len + min_ad_idx):
                ad[d - min_ad_idx, :] = self.get_trace(ad_id + d * (self.n_xlines - 1),
                                                       min_sample_id=min_sample_idx,
                                                       max_sample_id=max_sample_idx)
        else:
            for d in range(min_ad_idx, ad_len + min_ad_idx):
                ad[d - min_ad_idx, :] = self.get_trace((ad_id - self.n_xlines + 1 + d) * self.n_xlines + (self.n_xlines - d - 1),
                                                       min_sample_id=min_sample_idx,
                                                       max_sample_id=max_sample_idx)
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

    def get_trace_by_coord(self, index, min_sample_no=None, max_sample_no=None):
        """Reads one zslice from SGZ file (time or depth, depending on file contents)

        Parameters
        ----------
         index : int
            The ordinal number of the trace in the file

        min_sample_no : int
            The sample time/depth of the beginning of the range for a cropped trace
            Defaults to beginning of trace

         max_sample_no : int
            The sample time/depth of the end (exclusive) of the range for a cropped trace
            Defaults to include end of trace

        Returns
        -------
        trace : numpy.ndarray of float32, shape (n_samples) or (max_sample_id - min_sample_id)
            A single trace, decompressed
        """
        min_sample_no = self.zslices[0] if min_sample_no is None else min_sample_no
        max_sample_no = self.zslices[-1] + self.zslices[1] - self.zslices[0] if max_sample_no is None else max_sample_no
        return self.get_trace(index, self.get_zslice_index(min_sample_no), self.get_zslice_index(max_sample_no, include_stop=True))

    def get_trace(self, index, min_sample_id=None, max_sample_id=None):
        """Reads one trace from SGZ file

        Parameters
        ----------
        index : int
            The ordinal number of the trace in the file

        min_sample_id : int
            The index of the beginning of the range for a cropped trace
            Defaults to beginning of trace

         max_sample_id : int
            The index of the end (exclusive) of the range for a cropped trace
            Defaults to include end of trace

        Returns
        -------
        trace : numpy.ndarray of float32, shape (n_samples) or (max_sample_id - min_sample_id)
            A single trace, decompressed
        """
        if not self.structured:
            self.get_unstructured_mask()
            index = np.arange(self.mask.shape[0])[self.mask != 0][index]

        if not 0 <= index < self.n_ilines * self.n_xlines:
            if platform.system() == 'Windows':
                print('Yesterday it worked, Today it is not working, Windows is like that')
            raise IndexError(self.range_error.format(index, 0, self.tracecount))

        il, xl = index // self.n_xlines, index % self.n_xlines
        min_il = self.blockshape[0] * (il // self.blockshape[0])
        min_xl = self.blockshape[1] * (xl // self.blockshape[1])
        min_sample_id = 0 if min_sample_id is None else min_sample_id
        max_sample_id = self.n_samples if max_sample_id is None else max_sample_id

        min_z = self.blockshape[2] * (min_sample_id // self.blockshape[2])
        max_z = self.blockshape[2] * ((max_sample_id + self.blockshape[2] - 1) // self.blockshape[2])

        chunk = self._read_containing_chunk_cached(min_il, min_xl, min_z, max_z)
        trace = chunk[il % self.blockshape[0], xl % self.blockshape[1], min_sample_id-min_z:max_sample_id-min_z]
        return np.squeeze(trace)

    def _read_containing_chunk(self, ref_il, ref_xl, min_z, max_z):
        assert ref_il % self.blockshape[0] == 0
        assert ref_xl % self.blockshape[1] == 0
        assert min_z % self.blockshape[2] == 0
        assert max_z % self.blockshape[2] == 0
        return self.read_subvolume(ref_il, ref_il + self.blockshape[0],
                                   ref_xl, ref_xl + self.blockshape[1],
                                   min_z, max_z, access_padding=True)


    def get_unstructured_mask(self):
        if self.mask is None:
            buffer = self.file.read_range(self.file,
                                          self.segy_traceheader_template[189],
                                          self.header_entry_length_bytes)
            self.mask = np.frombuffer(buffer, dtype=np.int32) != 0
        else:
            pass

    def clear_variant_headers(self):
        self.variant_headers.clear()
        self.include_padding = None

    def read_variant_headers(self, include_padding=False, tracefields=None):
        """Reads all variant headers from SGZ file into a dictionary called variant_headers

        SeismicZFP stores integer arrays of any which are not constant through the input
        SEG-Y as a file footer. To generate trace headers it reads individual values from
        disk and combines with a 'template' containing the constant ones. In some circumstances
        it may be convenient to load all of this data into memory at once, which by default is
        what this function does if it has not already done so.

        Parameters
        ----------
        include_padding : bool
            Unstructured SGZ files have header arrays padded with zeros, by default these
            should be filtered out when returning the header array to maintain compatibility
            with segyio behaviour.

        tracefields : list of int / segyio.tracefield.TraceField
            It may be desirable to restrict the variant headers loaded for very large files
            or files accessed across a network. Provide this list to do so.
            See: get_tracefield_values()
        """

        # Check that this hasn't changed for unstructured files
        if self.include_padding is None:
            self.include_padding = include_padding
        if not self.structured:
            assert self.include_padding == include_padding

        tracefild_list = self.segy_traceheader_template if tracefields is None else tracefields
        for k in tracefild_list:
            # Yes, iterate through list of dictionary keys, because we might not have dict
            if k not in self.variant_headers:
                offset = self.segy_traceheader_template[k]
                if isinstance(offset, FileOffset) and k not in self.variant_headers:
                    use_mask = not (self.structured or self.include_padding)
                    if use_mask:
                        self.get_unstructured_mask()
                    buffer = self.file.read_range(self.file, offset, self.header_entry_length_bytes)
                    values = np.frombuffer(buffer, dtype=np.int32)
                    self.variant_headers[k] = values[self.mask] if use_mask else values

    def get_tracefield_values(self, tracefield):
        """Efficiently provides all trace header values for a given trace header field

        Parameters
        ----------
        tracefield : int / segyio.tracefield.TraceField
            The trace header value position, or its programmer-friendly
            enumerated version from segyio

        Returns
        -------
        header_array : numpy.ndarray of int32, shape (n_ilines, n_xlines)
        """
        self.read_variant_headers(include_padding=True, tracefields=[segyio.tracefield.TraceField(tracefield)])
        header_array = self.variant_headers[tracefield].reshape((self.n_ilines, self.n_xlines))
        return header_array

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
                if load_all_headers or not self.structured:
                    self.read_variant_headers()
                    header[k] = self.variant_headers[k][index]
                else:
                    buf = self.file.read_range(self.file, v + 4*index, 4) # A 32-bit int is 4 bytes
                    header[k] = np.frombuffer(buf, dtype=np.int32)[0]
        return header

    def get_file_binary_header(self):
        return segyio.segy.Field(self.file_binary_header, kind='binary')

    def get_file_text_header(self):
        return [bytearray(self.file_text_header.decode("cp037"),
                          encoding="ascii", errors="ignore")]
