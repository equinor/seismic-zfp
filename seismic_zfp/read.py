import numpy as np
from pyzfp import decompress

from utils import pad


class SzReader:

    def __init__(self, filename, tracelength, xlines, ilines, rate):
        self.filename = filename
        self.tracelength = tracelength
        self.xlines = xlines
        self.ilines = ilines
        self.rate = rate

        self.blockshape = (4, 4, 256)

        self.shape_pad = (pad(self.ilines, 4), pad(self.xlines, 4), pad(self.tracelength, 256))

    def read_inline(self, il_id):
        il_block_offset = il_id//4 * (self.shape_pad[2] * self.shape_pad[1] * 4)
        il_offset_in_block = il_id%4

        with open(self.filename, 'rb') as f:
            f.seek(il_block_offset, 0)
            buffer = f.read(self.shape_pad[2] * self.shape_pad[1] * 4)

        decompressed = decompress(buffer, (4, self.shape_pad[1], self.shape_pad[2]), np.dtype('float32'), rate=self.rate)

        return decompressed[il_offset_in_block, 0:self.xlines, 0:self.tracelength]
