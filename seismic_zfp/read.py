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

        self.blockshape = (4, 4, 2048//self.rate)

        self.blocksize_bytes = (self.blockshape[0] * self.blockshape[1] * self.blockshape[2] * rate) // 8

        self.shape_pad = (pad(self.ilines, 4), pad(self.xlines, 4), pad(self.tracelength, 2048//self.rate))

        self.blocks_per_trace = self.shape_pad[2] // self.blockshape[2]

        print("blockshape = {}, shape_pad = {}, blocks_per_trace = {}".format(self.blockshape, self.shape_pad, self.blocks_per_trace))

    def read_inline(self, il_id):
        il_block_offset = il_id//4 * (self.shape_pad[2] * self.shape_pad[1] * 4)
        il_offset_in_block = il_id % 4

        with open(self.filename, 'rb') as f:
            f.seek(il_block_offset, 0)
            buffer = f.read(self.blocks_per_trace * self.shape_pad[1] * self.blocksize_bytes)

        decompressed = decompress(buffer, (4, self.shape_pad[1], self.shape_pad[2]),
                                  np.dtype('float32'), rate=self.rate)

        return decompressed[il_offset_in_block, 0:self.xlines, 0:self.tracelength]

    def read_crossline(self, xl_id):
        chunksize_bytes = self.blocksize_bytes * self.blocks_per_trace
        print(chunksize_bytes)
        xl_first_chunk_offset = xl_id//4 * chunksize_bytes
        xl_offset_in_block = xl_id % 4
        xl_chunk_increment = self.blocks_per_trace * (self.shape_pad[1] // 4) * self.blocksize_bytes

        buffer = bytearray(chunksize_bytes * self.shape_pad[0])

        print(len(buffer))

        with open(self.filename, 'rb') as f:
            for chunk_num in range(self.shape_pad[0] // 4):
                f.seek(xl_first_chunk_offset + chunk_num*xl_chunk_increment, 0)
                buffer[chunk_num*chunksize_bytes:(chunk_num+1)*chunksize_bytes] = f.read(chunksize_bytes)

        decompressed = decompress(buffer, (self.shape_pad[0], 4, self.shape_pad[2]),
                                  np.dtype('float32'), rate=self.rate)

        print(decompressed.shape)

        return decompressed[0:self.ilines, xl_offset_in_block, 0:self.tracelength]
