import os
import numpy as np
from pyzfp import decompress
import segyio
from segyio import _segyio

from .utils import pad, bytes_to_int, bytes_to_signed_int, gen_coord_list, FileOffset
from .szconstants import DISK_BLOCK_BYTES, SEGY_FILE_HEADER_BYTES, SEGY_TEXT_HEADER_BYTES


class SzReader:
    """Reads SZ files

    Methods
    -------
    read_inline(il_id)
        Decompresses and returns one inline from SZ file as 2D numpy array

    read_crossline(xl_id)
        Decompresses and returns one crossline from SZ file as 2D numpy array

    read_zslice(zslice_id)
        Decompresses and returns one zslice from SZ file as 2D numpy array

    read_subvolume(min_il, max_il, min_xl, max_xl, min_z, max_z)
        Decompresses and returns an arbitrary sub-volume from SZ file as 3D numpy array
    """
    def __init__(self, file, filetype_checking=True):
        """
        Parameters
        ----------
        file : str
            The SZ filepath to be read

             : file handle in 'rb' mode
             Reuse an open file handle

        """

        # Class may be instantiated with either a file path or filehandle
        if not hasattr(file, 'read'):
            self.filename = file
            self.file = self.open_sz_file()
        else:
            self.filename = file.name
            self.file = file
            # You have a file handle, go to the start!
            self.file.seek(0)

        self.headerbytes = self.file.read(DISK_BLOCK_BYTES)
        if filetype_checking and self.headerbytes[0:2] == b'\xc3\x40':
            msg = "This appears to be a SEGY file rather than an SZ file, override with filetype_checking=False"
            raise RuntimeError(msg)

        self.n_header_blocks = bytes_to_int(self.headerbytes[0:4])
        if self.n_header_blocks != 1:
            self.file.seek(0)
            self.headerbytes = self.file.read(DISK_BLOCK_BYTES*self.n_header_blocks)

        # Read useful info out of the SZ header
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

        # These are useful units of measurement for SZ files:

        # A 'compression unit' is the smallest decompressable piece of the SZ file.
        # It is always 4-samples x 4-xlines x 4-ilines in physical dimensions, but its size
        # on disk will vary according to compression ratio.
        self.unit_bytes = ((4*4*4) * self.rate) // 8

        # A 'block' is a group of 'compression units' equal in size to a hardware disk block.
        # The 'compression units' may be arranged in any cuboid which matches the size of a disk block.
        # At the time of coding, standard commodity hardware uses 4kB disk blocks so check that
        # file has been written in using this convention.
        self.block_bytes = (self.blockshape[0] * self.blockshape[1] * self.blockshape[2] * self.rate) // 8
        assert self.block_bytes == DISK_BLOCK_BYTES
        assert self.block_bytes % self.unit_bytes == 0

        # A 'chunk' is a group of one or more 'blocks' which span a complete set of traces.
        # This will follow the xline and iline shape of a 'block'
        self.chunk_bytes = self.block_bytes * (self.shape_pad[2] // self.blockshape[2])
        assert self.chunk_bytes % self.block_bytes == 0

        # Placeholder. Don't read these if you're not going to use them
        self.variant_headers = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close_sz_file()

    def open_sz_file(self):
        if not os.path.exists(self.filename):
            raise FileNotFoundError("Rather than a beep, Or a rude error message, These words: 'File not found.'")
        return open(self.filename, 'rb')

    def close_sz_file(self):
        self.file.close()

    def parse_dimensions(self):
        n_samples = bytes_to_int(self.headerbytes[4:8])
        n_xlines = bytes_to_int(self.headerbytes[8:12])
        n_ilines = bytes_to_int(self.headerbytes[12:16])
        rate = bytes_to_int(self.headerbytes[40:44])
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

    def read_and_decompress_il_set(self, i):
        il_block_offset = ((self.chunk_bytes * self.shape_pad[1]) // 4) * (i // 4)

        self.file.seek(self.data_start_bytes + il_block_offset, 0)
        buffer = self.file.read(self.chunk_bytes * self.shape_pad[1])

        # Specify dtype otherwise pyzfp gets upset.
        return decompress(buffer, (self.blockshape[0], self.shape_pad[1], self.shape_pad[2]),
                                  np.dtype('float32'), rate=self.rate)

    def read_inline(self, il_id):
        """Reads one inline from SZ file

        Parameters
        ----------
        il_id : int
            The ordinal number of the inline in the file

        Returns
        -------
        inline : numpy.ndarray of float32, shape: (n_xlines, n_samples)
            The specified inline, decompressed
        """
        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            decompressed = self.read_and_decompress_il_set(il_id)
            return decompressed[il_id % self.blockshape[0], 0:self.n_xlines, 0:self.n_samples]
        else:
            # Default to unoptimized general method
            return np.squeeze(self.read_subvolume(il_id, il_id + 1, 0, self.n_xlines, 0, self.n_samples))

    def read_crossline(self, xl_id):
        """Reads one crossline from SZ file

        Parameters
        ----------
        xl_id : int
            The ordinal number of the crossline in the file

        Returns
        -------
        crossline : numpy.ndarray of float32, shape: (n_ilines, n_samples)
            The specified crossline, decompressed
        """
        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            xl_first_chunk_offset = xl_id//4 * self.chunk_bytes
            xl_chunk_increment = self.chunk_bytes * self.shape_pad[1] // 4

            # Allocate memory for compressed data
            buffer = bytearray(self.chunk_bytes * self.shape_pad[0] // 4)

            for chunk_num in range(self.shape_pad[0] // 4):
                self.file.seek(self.data_start_bytes + xl_first_chunk_offset
                               + chunk_num*xl_chunk_increment, 0)
                buffer[chunk_num*self.chunk_bytes:(chunk_num+1)*self.chunk_bytes] = self.file.read(self.chunk_bytes)

            # Specify dtype otherwise pyzfp gets upset.
            decompressed = decompress(buffer, (self.shape_pad[0], self.blockshape[1], self.shape_pad[2]),
                                      np.dtype('float32'), rate=self.rate)

            return decompressed[0:self.n_ilines, xl_id % self.blockshape[1], 0:self.n_samples]
        else:
            # Default to unoptimized general method
            return np.squeeze(self.read_subvolume(0, self.n_ilines, xl_id, xl_id + 1, 0, self.n_samples))

    def read_zslice(self, zslice_id):
        """Reads one zslice from SZ file (time or depth, depending on file contents)

        Parameters
        ----------
        zslice_id : int
            The ordinal number of the zslice in the file

        Returns
        -------
        zslice : numpy.ndarray of float32, shape: (n_ilines, n_xlines)
            The specified zslice (time or depth, depending on file contents), decompressed
        """
        blocks_per_dim = tuple(dim // size for dim, size in zip(self.shape_pad, self.blockshape))
        zslice_first_block_offset = zslice_id // self.blockshape[2]

        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            zslice_unit_in_block = (zslice_id % self.blockshape[2]) // 4

            # Allocate memory for compressed data
            buffer = bytearray(self.unit_bytes * (blocks_per_dim[0]) * (blocks_per_dim[1]))

            for block_num in range((blocks_per_dim[0]) * (blocks_per_dim[1])):
                self.file.seek(self.data_start_bytes + zslice_first_block_offset*self.block_bytes
                                             + zslice_unit_in_block*self.unit_bytes
                                             + block_num*self.chunk_bytes, 0)
                buffer[block_num*self.unit_bytes:(block_num+1)*self.unit_bytes] = self.file.read(self.unit_bytes)

            # Specify dtype otherwise pyzfp gets upset.
            decompressed = decompress(buffer, (self.shape_pad[0], self.shape_pad[1], 4),
                                      np.dtype('float32'), rate=self.rate)

            return decompressed[0:self.n_ilines, 0:self.n_xlines, zslice_id % 4]

        elif self.blockshape[2] == 4:
            sub_block_size_bytes = ((4 * 4 * self.blockshape[1]) * self.rate) // 8
            buffer = bytearray(self.block_bytes * (self.shape_pad[0] // self.blockshape[0]) * (self.shape_pad[1] // self.blockshape[1]))

            for block_i in range(self.shape_pad[0] // self.blockshape[0]):
                for block_x in range(blocks_per_dim[1]):
                    block_num = block_i * (blocks_per_dim[1]) + block_x
                    self.file.seek(self.data_start_bytes + zslice_first_block_offset*self.block_bytes + block_num*(self.block_bytes*(blocks_per_dim[2])), 0)
                    temp_buf = self.file.read(self.block_bytes)
                    for sub_block_num in range(self.blockshape[0] // 4):
                        buf_start = block_i*self.block_bytes*(blocks_per_dim[1]) + block_x*sub_block_size_bytes + sub_block_num * ((self.shape_pad[1]*4*4*self.rate) // 8)
                        buffer[buf_start:buf_start+sub_block_size_bytes] = \
                            temp_buf[sub_block_num*sub_block_size_bytes:(sub_block_num + 1)*sub_block_size_bytes]

            # Specify dtype otherwise pyzfp gets upset.
            decompressed = decompress(buffer, (self.shape_pad[0], self.shape_pad[1], 4),
                                      np.dtype('float32'), rate=self.rate)

            return decompressed[0:self.n_ilines, 0:self.n_xlines, zslice_id % 4]
        else:
            # Default to unoptimized general method
            return np.squeeze(self.read_subvolume(0, self.n_ilines, 0, self.n_xlines, zslice_id, zslice_id + 1))

    def read_subvolume(self, min_il, max_il, min_xl, max_xl, min_z, max_z):
        """Reads a sub-volume from SZ file

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
            The sprcified subvolume, decompressed
        """
        if self.blockshape[0] == 4 and self.blockshape[1] == 4:
            z_units = (max_z+4) // 4 - min_z // 4
            xl_units = (max_xl+4) // 4 - min_xl // 4
            il_units = (max_il+4) // 4 - min_il // 4

            # Allocate memory for compressed data
            buffer = bytearray(z_units * xl_units * il_units * self.unit_bytes)
            read_length = self.unit_bytes*z_units

            for i in range(il_units):
                for x in range(xl_units):
                    # No need to loop over z... it's contiguous, so do it in one file read
                    self.file.seek(self.data_start_bytes + self.unit_bytes * (
                          (i + (min_il // 4))*(self.shape_pad[1] // 4) * (self.shape_pad[2] // 4) +
                          (x + (min_xl // 4))*(self.shape_pad[2] // 4) +
                          (min_z // 4)), 0)
                    buf_start = (i*xl_units*z_units + x*z_units) * self.unit_bytes
                    buf_end = buf_start + read_length
                    buffer[buf_start:buf_end] = self.file.read(read_length)

            # Specify dtype otherwise pyzfp gets upset.
            decompressed = decompress(buffer, (il_units*4, xl_units*4, z_units*4),
                                      np.dtype('float32'), rate=self.rate)

            return decompressed[min_il%4:(min_il%4)+max_il-min_il,
                                min_xl%4:(min_xl%4)+max_xl-min_xl,
                                min_z%4:(min_z%4)+max_z-min_z]
        else:
            # This works generally, but is pretty wasteful for IL or XL reads.
            # Really should encourage users to stick with either:
            #  - blockshape[2] == 4
            #  - blockshape[0] == blockshape[1] == 4
            z_blocks = (max_z+self.blockshape[2]) // self.blockshape[2] - min_z // self.blockshape[2]
            xl_blocks = (max_xl+self.blockshape[1]) // self.blockshape[1] - min_xl // self.blockshape[1]
            il_blocks = (max_il+self.blockshape[0]) // self.blockshape[0] - min_il // self.blockshape[0]

            data_padded = np.zeros((il_blocks*self.blockshape[0],
                                    xl_blocks*self.blockshape[1],
                                    z_blocks*self.blockshape[2]), dtype=np.float32)

            for i in range(il_blocks):
                for x in range(xl_blocks):
                    for z in range(z_blocks):
                        self.file.seek(self.data_start_bytes + self.block_bytes * (
                                (i + (min_il // self.blockshape[0])) * (self.shape_pad[1] // self.blockshape[1]) * (self.shape_pad[2] // self.blockshape[2]) +
                                (x + (min_xl // self.blockshape[1])) * (self.shape_pad[2] // self.blockshape[2]) +
                                (z + (min_z // self.blockshape[2]))), 0)
                        buffer = self.file.read(self.block_bytes)
                        decompressed = decompress(buffer,
                                                  (self.blockshape[0], self.blockshape[1], self.blockshape[2]),
                                                  np.dtype('float32'), rate=self.rate)
                        data_padded[i*self.blockshape[0]:(i+1)*self.blockshape[0],
                                    x*self.blockshape[1]:(x+1)*self.blockshape[1],
                                    z*self.blockshape[2]:(z+1)*self.blockshape[2]] = decompressed

            return data_padded[min_il%self.blockshape[0]:(min_il%self.blockshape[0])+max_il-min_il,
                               min_xl%self.blockshape[1]:(min_xl%self.blockshape[1])+max_xl-min_xl,
                               min_z%self.blockshape[2]:(min_z%self.blockshape[2])+max_z-min_z]

    def read_volume(self):
        return self.read_subvolume(0, self.n_ilines,
                                   0, self.n_xlines,
                                   0, self.n_samples)

    def gen_trace_header(self, index):
        header = self.segy_traceheader_template.copy()
        for k, v in header.items():
            if isinstance(v, FileOffset):
                header[k] = self.variant_headers[k][index]
        return header

    def get_trace(self, index):
        min_il = index // self.n_ilines
        min_xl = index % self.n_ilines
        trace = self.read_subvolume(min_il, min_il+1,
                                    min_xl, min_xl+1,
                                    0, len(self.zslices))
        return np.squeeze(trace)

    def get_file_binary_header(self):
        return segyio.segy.Field(self.file_binary_header, kind='binary')

    def get_file_text_header(self):
        return [self.file_text_header.decode("cp037")]
