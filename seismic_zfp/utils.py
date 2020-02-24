import time
import datetime

from .szconstants import DISK_BLOCK_BYTES


class FileOffset(int):
    """Convenience class to enable distinction between default header values and file offsets"""
    def __new__(cls, value):
        return int.__new__(cls, value)


def pad(orig, multiple):
    if orig%multiple == 0:
        return orig
    else:
        return multiple * (orig//multiple + 1)


def gen_coord_list(start, step, count):
    return list(range(start, start + step*count, step))


def np_float_to_bytes(numpy_float):
    # How is this so hard?
    return int((numpy_float).astype(int)).to_bytes(4, byteorder='little')


def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='little')


def bytes_to_signed_int(bytes):
    return int.from_bytes(bytes, byteorder='little', signed=True)


def define_blockshape(bits_per_voxel, blockshape):
    n_undefined = sum([1 for n in list(blockshape) + [bits_per_voxel] if n == -1])
    if n_undefined > 1:
        raise ValueError("Blockshape is underdefined")
    if bits_per_voxel == -1:
        bits_per_voxel = DISK_BLOCK_BYTES * 8 // (blockshape[0] * blockshape[1] * blockshape[2])
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
