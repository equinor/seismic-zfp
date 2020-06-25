from __future__ import print_function, division
import struct
import time
import datetime

from .sgzconstants import DISK_BLOCK_BYTES


class FileOffset(int):
    """Convenience class to enable distinction between default header values and file offsets"""
    def __new__(cls, value):
        return int.__new__(cls, value)


class Geometry:
    """Lightweight place to keep track of IL/XL ranges"""
    def __init__(self, min_il, max_il, min_xl, max_xl):
        self.ilines = range(min_il, max_il)
        self.xlines = range(min_xl, max_xl)


class InferredGeometry(Geometry):
    """Subclass used to signify irregular input SEG-Y"""
    def __init__(self, traces_ref):
        self.traces_ref = traces_ref
        il_ids = set([k[0] for k in traces_ref.keys()])
        xl_ids = set([k[1] for k in traces_ref.keys()])
        self.min_il, self.max_il, self.il_step = min(il_ids), max(il_ids), (max(il_ids) - min(il_ids) + 1) // len(il_ids)
        self.min_xl, self.max_xl, self.xl_step = min(xl_ids), max(xl_ids), (max(xl_ids) - min(xl_ids) + 1) // len(xl_ids)
        self.ilines = range(self.min_il, self.max_il + 1, self.il_step)
        self.xlines = range(self.min_xl, self.max_xl + 1, self.xl_step)


def pad(orig, multiple):
    if orig%multiple == 0:
        return orig
    else:
        return multiple * (orig//multiple + 1)


def gen_coord_list(start, step, count):
    return list(range(start, start + step*count, step))


def np_float_to_bytes(numpy_float):
    # How is this so hard?
    return struct.pack("<I", int((numpy_float).astype(int)))


def bytes_to_int(bytes):
    if len(bytes) == 4:
        return struct.unpack('<I', bytes)[0]
    elif len(bytes) == 2:
        return struct.unpack('<H', bytes)[0]


def bytes_to_signed_int(bytes):
    if len(bytes) == 4:
        return struct.unpack('<i', bytes)[0]
    elif len(bytes) == 2:
        return struct.unpack('<h', bytes)[0]


def int_to_bytes(bytes):
    return struct.pack('<I', bytes)


def signed_int_to_bytes(bytes):
    return struct.pack('<i', bytes)


def define_blockshape(bits_per_voxel, blockshape):
    if sum([1 for n in list(blockshape) + [bits_per_voxel] if n == -1]) > 1:
        raise ValueError("Blockshape is underdefined")

    if isinstance(bits_per_voxel, str):
        bits_per_voxel = float(bits_per_voxel)

    bits_per_voxel = 1 / -bits_per_voxel if bits_per_voxel < -1 else bits_per_voxel

    if bits_per_voxel == -1:
        bits_per_voxel = DISK_BLOCK_BYTES * 8 / (blockshape[0] * blockshape[1] * blockshape[2])
    else:
        if blockshape[0] == -1:
            blockshape = (int(DISK_BLOCK_BYTES * 8 //
                              (blockshape[1] * blockshape[2] * bits_per_voxel)), blockshape[1], blockshape[2])
        elif blockshape[1] == -1:
            blockshape = (blockshape[0], int(DISK_BLOCK_BYTES * 8 //
                          (blockshape[2] * blockshape[0] * bits_per_voxel)), blockshape[2])
        elif blockshape[2] == -1:
            blockshape = (blockshape[0], blockshape[1], int(DISK_BLOCK_BYTES * 8 //
                                                            (blockshape[0] * blockshape[1] * bits_per_voxel)))
        else:
            assert(bits_per_voxel * blockshape[0] * blockshape[1] * blockshape[2] == DISK_BLOCK_BYTES * 8)
    return bits_per_voxel, blockshape


def progress_printer(start_time, progress_frac):
    current_time = time.time()
    eta = current_time + ((1. - progress_frac) * (current_time - start_time)) / (progress_frac + 0.0000001)
    st = datetime.datetime.fromtimestamp(eta).strftime('%Y-%m-%d %H:%M:%S')
    print("   - {:5.1f}% complete. ETA: {}".format(progress_frac * 100, st), end="\r")


def get_correlated_diagonal_length(cd, n_il, n_xl):
    if n_xl > n_il:
        if cd >= 0:
            return n_il - cd
        elif abs(cd) <= n_xl - n_il:
            return n_il
        else:  # cd is negative
            return n_xl + cd
    elif n_xl < n_il:
        if cd <= 0:
            return n_xl + cd
        elif abs(cd) <= n_il - n_xl:
            return n_xl
        else:
            return n_il - cd
    else:  # Equal number of ILs & XLs
        return n_il - abs(cd)


def get_anticorrelated_diagonal_length(ad, n_il, n_xl):
    if ad < min(n_il, n_xl):
        return ad + 1
    elif min(n_il, n_xl) <= ad < max(n_il, n_xl):
        return min(n_il, n_xl)
    else:
        return n_il + n_xl - ad - 1
