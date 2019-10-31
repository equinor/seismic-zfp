import numpy as np
from pyzfp import decompress

from .utils import pad


class SzReader:

    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, 'rb') as f:
            buffer = f.read(4096)
            self.header_blocks = int.from_bytes(buffer[0:4], byteorder='little')
            if self.header_blocks != 1:
                f.seek(0)
                buffer = f.read(4096*self.header_blocks)

        self.tracelength = int.from_bytes(buffer[4:8], byteorder='little')
        self.xlines = int.from_bytes(buffer[8:12], byteorder='little')
        self.ilines = int.from_bytes(buffer[12:16], byteorder='little')
        self.rate = int.from_bytes(buffer[40:44], byteorder='little')

        self.shape_pad = (pad(self.ilines, 4), pad(self.xlines, 4), pad(self.tracelength, 2048//self.rate))

        self.blockshape = (4, 4, 2048//self.rate)
        self.blocksize_bytes = (self.blockshape[0] * self.blockshape[1] * self.blockshape[2] * self.rate) // 8
        self.blocks_per_trace = self.shape_pad[2] // self.blockshape[2]

        self.unit_size_bytes = ((4*4*4) * self.rate) // 8

        self.data_start_bytes = self.header_blocks * 4096

        print("n_samples={}, n_xlines={}, n_ilines={}".format(self.tracelength, self.xlines, self.ilines))



    def read_inline(self, il_id):
        chunksize = self.blocks_per_trace * self.shape_pad[1] * self.blocksize_bytes
        il_block_offset = (chunksize//4) * (il_id//4)

        with open(self.filename, 'rb') as f:
            f.seek(self.data_start_bytes + il_block_offset, 0)
            buffer = f.read(chunksize)

        decompressed = decompress(buffer, (self.blockshape[0], self.shape_pad[1], self.shape_pad[2]),
                                  np.dtype('float32'), rate=self.rate)

        return decompressed[il_id % self.blockshape[0], 0:self.xlines, 0:self.tracelength]

    def read_crossline(self, xl_id):
        chunksize_bytes = self.blocksize_bytes * self.blocks_per_trace
        xl_first_chunk_offset = xl_id//4 * chunksize_bytes
        xl_chunk_increment = chunksize_bytes * self.shape_pad[1] // 4

        buffer = bytearray(chunksize_bytes * self.shape_pad[0] // 4)

        with open(self.filename, 'rb') as f:
            for chunk_num in range(self.shape_pad[0] // 4):
                f.seek(self.data_start_bytes + xl_first_chunk_offset + chunk_num*xl_chunk_increment, 0)
                buffer[chunk_num*chunksize_bytes:(chunk_num+1)*chunksize_bytes] = f.read(chunksize_bytes)

        decompressed = decompress(buffer, (self.shape_pad[0], self.blockshape[1], self.shape_pad[2]),
                                  np.dtype('float32'), rate=self.rate)

        return decompressed[0:self.ilines, xl_id % self.blockshape[1], 0:self.tracelength]

    def read_zslice(self, zslice_id):
        chunksize_bytes = self.blocksize_bytes * self.blocks_per_trace
        zslice_first_block_offset = zslice_id // self.blockshape[2]

        zslice_unit_in_block = (zslice_id % self.blockshape[2]) // 4

        buffer = bytearray(self.blocksize_bytes * (self.shape_pad[0] // 4) * (self.shape_pad[1] // 4))

        with open(self.filename, 'rb') as f:
            for block_num in range((self.shape_pad[0] // 4) * (self.shape_pad[1] // 4)):
                f.seek(self.data_start_bytes + zslice_first_block_offset*self.blocksize_bytes + zslice_unit_in_block*self.unit_size_bytes + block_num*chunksize_bytes)
                buffer[block_num*self.unit_size_bytes : (block_num+1)*self.unit_size_bytes] = f.read(self.unit_size_bytes)

        decompressed = decompress(buffer, (self.shape_pad[0], self.shape_pad[1], 4),
                                  np.dtype('float32'), rate=self.rate)

        return decompressed[0:self.ilines, 0:self.xlines, zslice_id % 4]
