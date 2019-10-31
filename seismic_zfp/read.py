import numpy as np
from pyzfp import decompress

from .utils import pad


class SzReader:

    def __init__(self, filename, tracelength, xlines, ilines, rate):
        self.filename = filename
        self.tracelength = tracelength
        self.xlines = xlines
        self.ilines = ilines
        self.rate = rate

        self.shape_pad = (pad(self.ilines, 4), pad(self.xlines, 4), pad(self.tracelength, 2048//self.rate))

        self.blockshape = (4, 4, 2048//self.rate)
        self.blocksize_bytes = (self.blockshape[0] * self.blockshape[1] * self.blockshape[2] * rate) // 8
        self.blocks_per_trace = self.shape_pad[2] // self.blockshape[2]

        self.unit_size_bytes = ((4*4*4) * self.rate) // 8

    def read_inline(self, il_id):
        chunksize = self.blocks_per_trace * self.shape_pad[1] * self.blocksize_bytes
        il_block_offset = (chunksize//4) * (il_id//4)

        with open(self.filename, 'rb') as f:
            f.seek(il_block_offset, 0)
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
                f.seek(xl_first_chunk_offset + chunk_num*xl_chunk_increment, 0)
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
                f.seek(zslice_first_block_offset*self.blocksize_bytes + zslice_unit_in_block*self.unit_size_bytes + block_num*chunksize_bytes)
                buffer[block_num*self.unit_size_bytes : (block_num+1)*self.unit_size_bytes] = f.read(self.unit_size_bytes)

        decompressed = decompress(buffer, (self.shape_pad[0], self.shape_pad[1], 4),
                                  np.dtype('float32'), rate=self.rate)

        return decompressed[0:self.ilines, 0:self.xlines, zslice_id % 4]
