import os
import numpy as np
from pyzfp import decompress

from .utils import pad, bytes_to_int

DISK_BLOCK_BYTES = 4096


class SzReader:
    """Reads SZ files

    Methods
    -------
    read_inline(il_id)
        Decompresses and returns one inline from SZ file as 2D numpy array

    read_crossline(il_id)
        Decompresses and returns one crossline from SZ file as 2D numpy array

    read_zslice(il_id)
        Decompresses and returns one zslice from SZ file as 2D numpy array

    read_subvolume(il_id)
        Decompresses and returns an arbitrary sub-volume from SZ file as 3D numpy array
    """
    def __init__(self, file):
        """
        Parameters
        ----------
        file : str
            The SZ file to be read

        """
        self.filename = file

        if not os.path.exists(self.filename):
            raise FileNotFoundError("Rather than a beep Or a rude error message, These words: 'File not found.'")

        with open(self.filename, 'rb') as f:
            buffer = f.read(DISK_BLOCK_BYTES)
            self.header_blocks = bytes_to_int(buffer[0:4])
            if self.header_blocks != 1:
                f.seek(0)
                buffer = f.read(DISK_BLOCK_BYTES*self.header_blocks)

        self.tracelength = bytes_to_int(buffer[4:8])
        self.xlines = bytes_to_int(buffer[8:12])
        self.ilines = bytes_to_int(buffer[12:16])
        self.rate = bytes_to_int(buffer[40:44])

        self.blockshape = (bytes_to_int(buffer[44:48]), bytes_to_int(buffer[48:52]), bytes_to_int(buffer[52:56]))

        # ytilibitapmoc
        if self.blockshape[0] == 0 or self.blockshape[1] == 0 or self.blockshape[2] == 0:
            self.blockshape = (4, 4, 2048//self.rate)

        self.shape_pad = (pad(self.ilines, self.blockshape[0]),
                          pad(self.xlines, self.blockshape[1]),
                          pad(self.tracelength, self.blockshape[2]))

        self.unit_bytes = ((4*4*4) * self.rate) // 8
        self.block_bytes = (self.blockshape[0] * self.blockshape[1] * self.blockshape[2] * self.rate) // 8
        self.chunk_bytes = self.block_bytes * (self.shape_pad[2] // self.blockshape[2])
        assert self.block_bytes == DISK_BLOCK_BYTES

        self.data_start_bytes = self.header_blocks * DISK_BLOCK_BYTES

        print("n_samples={}, n_xlines={}, n_ilines={}".format(self.tracelength, self.xlines, self.ilines))

    def read_inline(self, il_id):
        """Reads one inline from SZ file

        Parameters
        ----------
        il_id : int
            The ordinal number of the inline in the file

        Returns
        -------
        inline : numpy.ndarray of float32, shape: (n_xlines, tracelength)
            The specified inline, decompressed
        """
        il_block_offset = ((self.chunk_bytes * self.shape_pad[1])//4) * (il_id//4)

        with open(self.filename, 'rb') as f:
            f.seek(self.data_start_bytes + il_block_offset, 0)
            # Allocate and read in one go
            buffer = f.read(self.chunk_bytes * self.shape_pad[1])

        # Specify dtype otherwise pyzfp gets upset.
        decompressed = decompress(buffer, (self.blockshape[0], self.shape_pad[1], self.shape_pad[2]),
                                  np.dtype('float32'), rate=self.rate)

        return decompressed[il_id % self.blockshape[0], 0:self.xlines, 0:self.tracelength]

    def read_crossline(self, xl_id):
        """Reads one crossline from SZ file

        Parameters
        ----------
        xl_id : int
            The ordinal number of the crossline in the file

        Returns
        -------
        crossline : numpy.ndarray of float32, shape: (n_ilines, tracelength)
            The specified crossline, decompressed
        """
        xl_first_chunk_offset = xl_id//4 * self.chunk_bytes
        xl_chunk_increment = self.chunk_bytes * self.shape_pad[1] // 4

        # Allocate memory for compressed data
        buffer = bytearray(self.chunk_bytes * self.shape_pad[0] // 4)

        with open(self.filename, 'rb') as f:
            for chunk_num in range(self.shape_pad[0] // 4):
                f.seek(self.data_start_bytes + xl_first_chunk_offset
                                             + chunk_num*xl_chunk_increment, 0)
                buffer[chunk_num*self.chunk_bytes:(chunk_num+1)*self.chunk_bytes] = f.read(self.chunk_bytes)

        # Specify dtype otherwise pyzfp gets upset.
        decompressed = decompress(buffer, (self.shape_pad[0], self.blockshape[1], self.shape_pad[2]),
                                  np.dtype('float32'), rate=self.rate)

        return decompressed[0:self.ilines, xl_id % self.blockshape[1], 0:self.tracelength]

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

            with open(self.filename, 'rb') as f:
                for block_num in range((blocks_per_dim[0]) * (blocks_per_dim[1])):
                    f.seek(self.data_start_bytes + zslice_first_block_offset*self.block_bytes
                                                 + zslice_unit_in_block*self.unit_bytes
                                                 + block_num*self.chunk_bytes, 0)
                    buffer[block_num*self.unit_bytes:(block_num+1)*self.unit_bytes] = f.read(self.unit_bytes)

            # Specify dtype otherwise pyzfp gets upset.
            decompressed = decompress(buffer, (self.shape_pad[0], self.shape_pad[1], 4),
                                      np.dtype('float32'), rate=self.rate)

            return decompressed[0:self.ilines, 0:self.xlines, zslice_id % 4]

        elif self.blockshape[2] == 4:
            sub_block_size_bytes = ((4 * 4 * self.blockshape[1]) * self.rate) // 8
            buffer = bytearray(self.block_bytes * (self.shape_pad[0] // self.blockshape[0]) * (self.shape_pad[1] // self.blockshape[1]))

            with open(self.filename, 'rb') as f:
                for block_i in range(self.shape_pad[0] // self.blockshape[0]):
                    for block_x in range(blocks_per_dim[1]):
                        block_num = block_i * (blocks_per_dim[1]) + block_x
                        f.seek(self.data_start_bytes + zslice_first_block_offset*self.block_bytes + block_num*(self.block_bytes*(blocks_per_dim[2])), 0)
                        temp_buf = f.read(self.block_bytes)
                        for sub_block_num in range(self.blockshape[0] // 4):
                            buf_start = block_i*self.block_bytes*(blocks_per_dim[1]) + block_x*sub_block_size_bytes + sub_block_num * ((self.shape_pad[1]*4*4*self.rate) // 8)
                            buffer[buf_start:buf_start+sub_block_size_bytes] = \
                                temp_buf[sub_block_num*sub_block_size_bytes:(sub_block_num + 1)*sub_block_size_bytes]

            # Specify dtype otherwise pyzfp gets upset.
            decompressed = decompress(buffer, (self.shape_pad[0], self.shape_pad[1], 4),
                                      np.dtype('float32'), rate=self.rate)

            return decompressed[0:self.ilines, 0:self.xlines, zslice_id % 4]
        else:
            msg = "Cannot read without il/xl blockdims of 4 OR z blockdim of 4, blockdims are {}".format(blockshape)
            raise NotImplementedError(msg)

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
        z_units = (max_z+4) // 4 - min_z // 4
        xl_units = (max_xl+4) // 4 - min_xl // 4
        il_units = (max_il+4) // 4 - min_il // 4

        # Allocate memory for compressed data
        buffer = bytearray(z_units * xl_units * il_units * self.unit_bytes)
        read_length = self.unit_bytes*z_units

        with open(self.filename, 'rb') as f:
            for i in range(il_units):
                for x in range(xl_units):
                    # No need to loop over z... it's contiguous, so do it in one file read
                    f.seek(self.data_start_bytes + self.unit_bytes * (
                          (i + (min_il // 4))*(self.shape_pad[1] // 4) * (self.shape_pad[2] // 4) +
                          (x + (min_xl // 4))*(self.shape_pad[2] // 4) +
                          (min_z // 4)), 0)
                    buf_start = (i*xl_units*z_units + x*z_units) * self.unit_bytes
                    buf_end = buf_start + read_length
                    buffer[buf_start:buf_end] = f.read(read_length)

        # Specify dtype otherwise pyzfp gets upset.
        decompressed = decompress(buffer, (il_units*4, xl_units*4, z_units*4),
                                  np.dtype('float32'), rate=self.rate)

        return decompressed[min_il%4:(min_il%4)+max_il-min_il,
                            min_xl%4:(min_xl%4)+max_xl-min_xl,
                            min_z%4:(min_z%4)+max_z-min_z]
