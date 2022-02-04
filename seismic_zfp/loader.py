import concurrent.futures as cf
from functools import lru_cache
import random
import psutil
from operator import floordiv
import numpy as np
import zfpy
from .sgzconstants import DISK_BLOCK_BYTES


class SgzLoader(object):
    def __init__(self, file, data_start_bytes, compressed_data_diskblocks, shape_pad, blockshape,
                 chunk_bytes, block_bytes, unit_bytes, rate, local, preload=False):
        self.file = file
        self.local = local
        self.data_start_bytes = data_start_bytes
        self.compressed_data_diskblocks = compressed_data_diskblocks
        self.shape_pad = shape_pad
        self.blockshape = blockshape
        self.block_dims = tuple(map(floordiv, shape_pad, blockshape))
        self.chunk_bytes = chunk_bytes
        self.block_bytes = block_bytes
        self.unit_bytes = unit_bytes
        self.rate = rate
        self.n_workers = 1 if self.local else 20
        self.oom_msgs = ['Out of memory.  We wish to hold the whole sky,  But we never will.',
                         'The code was willing, It considered your request, But the chips were weak.',
                         'To have no errors, Would be life without meaning. No struggle, no joy.']
        self.mem_limit = psutil.virtual_memory().total

        self.compressed_volume = None
        if preload:
            uncompressed_buf_size = self.compressed_data_diskblocks * DISK_BLOCK_BYTES * (32 / self.rate)
            if uncompressed_buf_size > self.mem_limit:
                print(f'Uncompressed volume is {uncompressed_buf_size//(1024*1024)}MB' \
                      f'machine memory is {self.mem_limit//(1024*1024)}MB, try using "preload=False"')
                raise RuntimeError(random.choice(self.oom_msgs))
            self.load_compressed_volume()

    def load_compressed_volume(self):
        if self.compressed_volume is None:
            self.compressed_volume = self.file.read_range(self.file, self.data_start_bytes, self.compressed_data_diskblocks * self.block_bytes)
        else:
            pass

    def _insert_into_buffer(self, buffer, buffer_start, data_offset, length):
        part = self._get_compressed_bytes(data_offset, length)
        buffer[buffer_start : buffer_start + length] = part

    def _insert_chunk_into_buffer(self, buffer, buffer_start, data_offset):
        self._insert_into_buffer(buffer, buffer_start, data_offset, self.chunk_bytes)

    def _insert_unit_into_buffer(self, buffer, buffer_start, data_offset):
        self._insert_into_buffer(buffer, buffer_start, data_offset, self.unit_bytes)

    def _distribute_chunk_into_buffer(self, buffer, block_id, blocks_per_dim, sub_block_size_bytes,
                                      zslice_first_block_offset):
        """Advanced layout - one block is not contiguous in memory in the decompression buffer"""
        block_i = block_id // blocks_per_dim[1]
        block_x = block_id % blocks_per_dim[1]
        block_num = block_i * (blocks_per_dim[1]) + block_x
        temp_buf = self._get_compressed_bytes(zslice_first_block_offset * self.block_bytes
                                              + block_num * (self.block_bytes * (blocks_per_dim[2])),
                                              self.block_bytes)
        for sub_block_num in range(self.blockshape[0] // 4):
            buf_start = block_i * self.block_bytes * (
                blocks_per_dim[1]) + block_x * sub_block_size_bytes + sub_block_num * (
                                (self.shape_pad[1] * 4 * 4 * self.rate) // 8)
            buffer[buf_start:buf_start + sub_block_size_bytes] = \
                temp_buf[sub_block_num * sub_block_size_bytes:(sub_block_num + 1) * sub_block_size_bytes]

    def _get_compressed_bytes(self, offset, length_bytes):
        if self.compressed_volume is not None:
            return self.compressed_volume[offset:offset+length_bytes]
        else:
            return self.file.read_range(self.file, self.data_start_bytes + offset, length_bytes)

    def _decompress(self, buffer, shape):
        return zfpy._decompress(bytes(buffer), zfpy.dtype_to_ztype(np.dtype('float32')), shape, rate=self.rate)

    def clear_cache(self):
        self.read_and_decompress_il_set.cache_clear()
        self.read_and_decompress_xl_set.cache_clear()
        self.read_and_decompress_zslice_set.cache_clear()
        self.read_and_decompress_zslice_set_adv.cache_clear()
        self.read_and_decompress_chunk_range.cache_clear()
        self.read_unshuffle_and_decompress_chunk_range.cache_clear()

    @lru_cache(maxsize=1)
    def read_and_decompress_il_set(self, i):
        il_block_offset = ((self.chunk_bytes * self.shape_pad[1]) // 4) * (i // 4)
        buffer = self._get_compressed_bytes(il_block_offset, self.chunk_bytes * self.shape_pad[1])
        return self._decompress(buffer, (self.blockshape[0], self.shape_pad[1], self.shape_pad[2]))

    @lru_cache(maxsize=1)
    def read_and_decompress_xl_set(self, x):
        xl_first_chunk_offset = x // 4 * self.chunk_bytes
        xl_chunk_increment = self.chunk_bytes * self.shape_pad[1] // 4
        buffer = bytearray(self.chunk_bytes * self.shape_pad[0] // 4)
        with cf.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for chunk_num in range(self.shape_pad[0] // 4):
                executor.submit(self._insert_chunk_into_buffer, buffer, chunk_num * self.chunk_bytes,
                                           xl_first_chunk_offset + chunk_num * xl_chunk_increment)
        return self._decompress(buffer, (self.shape_pad[0], self.blockshape[1], self.shape_pad[2]))

    @lru_cache(maxsize=1)
    def read_and_decompress_zslice_set(self, blocks_per_dim, zslice_first_block_offset, zslice_id):
        zslice_unit_in_block = (zslice_id % self.blockshape[2]) // 4
        buffer = bytearray(self.unit_bytes * (blocks_per_dim[0]) * (blocks_per_dim[1]))
        with cf.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for block_num in range((blocks_per_dim[0]) * (blocks_per_dim[1])):
                executor.submit(self._insert_unit_into_buffer, buffer, block_num * self.unit_bytes,
                                zslice_first_block_offset * self.block_bytes
                                + zslice_unit_in_block * self.unit_bytes
                                + block_num * self.chunk_bytes)
        return self._decompress(buffer, (self.shape_pad[0], self.shape_pad[1], 4))

    @lru_cache(maxsize=1)
    def read_and_decompress_zslice_set_adv(self, blocks_per_dim, zslice_first_block_offset):
        sub_block_size_bytes = ((4 * 4 * self.blockshape[1]) * self.rate) // 8
        buffer = bytearray(self.block_bytes * blocks_per_dim[0] * blocks_per_dim[1])
        with cf.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for block_id in range(blocks_per_dim[0]*blocks_per_dim[1]):
                executor.submit(self._distribute_chunk_into_buffer, buffer, block_id, blocks_per_dim,
                                                                    sub_block_size_bytes, zslice_first_block_offset)
        return self._decompress(buffer, (self.shape_pad[0], self.shape_pad[1], 4))


    def read_chunk_range(self, min_il, min_xl, min_z, il_units, xl_units, z_units):
        buffer = bytearray(z_units * xl_units * il_units * self.unit_bytes)
        read_length = self.unit_bytes * z_units
        for i in range(il_units):
            for x in range(xl_units):
                # No need to loop over z... it's contiguous, so do it in one file read
                bytes_start = self.unit_bytes * (
                                ((min_il // 4) + i) * (self.shape_pad[1] // 4) * (self.shape_pad[2] // 4) +
                                ((min_xl // 4) + x) * (self.shape_pad[2] // 4) +
                                 (min_z // 4))
                buf_start = (i * xl_units * z_units + x * z_units) * self.unit_bytes
                buffer[buf_start:buf_start+read_length] = self._get_compressed_bytes(bytes_start, read_length)
        return buffer


    @lru_cache(maxsize=1)
    def read_and_decompress_chunk_range(self, max_il, max_xl, max_z, min_il, min_xl, min_z):
        z_units = (max_z + 3) // 4 - min_z // 4
        xl_units = (max_xl + 3) // 4 - min_xl // 4
        il_units = (max_il + 3) // 4 - min_il // 4

        buffer = self.read_chunk_range(min_il, min_xl, min_z,
                                       il_units, xl_units, z_units)
        return self._decompress(buffer, (il_units * 4, xl_units * 4, z_units * 4))


    @lru_cache(maxsize=1)
    def read_unshuffle_and_decompress_chunk_range(self, max_il, max_xl, max_z, min_il, min_xl, min_z):
        z_blocks = (max_z + self.blockshape[2]) // self.blockshape[2] - min_z // self.blockshape[2]
        xl_blocks = (max_xl + self.blockshape[1]) // self.blockshape[1] - min_xl // self.blockshape[1]
        il_blocks = (max_il + self.blockshape[0]) // self.blockshape[0] - min_il // self.blockshape[0]
        decompressed = np.zeros((il_blocks * self.blockshape[0],
                                 xl_blocks * self.blockshape[1],
                                 z_blocks  * self.blockshape[2]), dtype=np.float32)
        for ni, i in enumerate(range(min_il // self.blockshape[0], min_il // self.blockshape[0] + il_blocks)):
            for nx, x in enumerate(range(min_xl // self.blockshape[1], min_xl // self.blockshape[1] + xl_blocks)):
                for nz, z in enumerate(range(min_z // self.blockshape[2], min_z // self.blockshape[2] + z_blocks)):
                    bytes_start = self.block_bytes * (self.block_dims[2] * ((self.block_dims[1] * i) + x) + z)
                    buffer = self._get_compressed_bytes(bytes_start, self.block_bytes)
                    # Fill decompressed buffer brick by brick
                    decompressed[ni * self.blockshape[0]:(ni + 1) * self.blockshape[0],
                                 nx * self.blockshape[1]:(nx + 1) * self.blockshape[1],
                                 nz * self.blockshape[2]:(nz + 1) * self.blockshape[2]] \
                        = self._decompress(buffer, self.blockshape)
        return decompressed
