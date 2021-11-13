import struct
import time
import datetime
import numpy as np

from .sgzconstants import DISK_BLOCK_BYTES


class FileOffset(int):
    """Convenience class to enable distinction between default header values and file offsets"""
    def __new__(cls, value):
        return int.__new__(cls, value)


class CubeWithAxes:
    """Utility class to hold 3D numpy array and corresponding axes"""
    def __init__(self, data_array, ilines, xlines, samples):
        self.data_array = data_array
        self.ilines = ilines
        self.xlines = xlines
        self.samples = samples


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
        self.min_il, self.max_il, self.il_step = min(il_ids), max(il_ids), (max(il_ids) - min(il_ids)) // (len(il_ids) - 1)
        self.min_xl, self.max_xl, self.xl_step = min(xl_ids), max(xl_ids), (max(xl_ids) - min(xl_ids)) // (len(xl_ids) - 1)
        self.ilines = range(self.min_il, self.max_il + 1, self.il_step)
        self.xlines = range(self.min_xl, self.max_xl + 1, self.xl_step)

    def __repr__(self):
        return 'IL:[{},{},{}] -- XL:[{},{},{}]'.format(self.min_il, self.max_il, self.il_step,
                                                       self.min_xl, self.max_xl, self.xl_step)

def read_range_file(file, offset, length):
    file.seek(offset)
    return file.read(length)

def read_range_blob(file, offset, length):
    return file.download_blob(offset=offset, length=length).readall()

def generate_fake_seismic(n_ilines, n_xlines, n_samples, min_iline=0, min_xline=0):
    # Generate an array which looks a *bit* like an impulse-response test...
    ilines, xlines, samples = np.arange(n_ilines), np.arange(n_xlines), np.arange(n_samples)
    array_shape = (n_ilines, n_xlines, n_samples)

    i = np.broadcast_to(np.expand_dims(np.expand_dims((ilines - n_ilines / 2), 1), 2), array_shape).astype(np.float32)
    x = np.broadcast_to(np.expand_dims((xlines - n_xlines / 2), 1), array_shape).astype(np.float32)
    s = np.broadcast_to(samples - n_samples / 4, array_shape).astype(np.float32)
    array = 0.01 + (np.sin(0.1 + np.sqrt(2.0 + (i+0.01) ** 2 + x ** 2 + (s*0.75) ** 2) / 8.0) /
                          (0.1 * np.sqrt(2.0 + (i+0.01) ** 2 + x ** 2 + (s*0.50) ** 2)))
    return array, ilines+min_iline, xlines+min_xline, samples


def pad(orig, multiple):
    if orig%multiple == 0:
        return orig
    else:
        return multiple * (orig//multiple + 1)

def coord_to_index(coord, coords, include_stop=False):
    try:
        index = np.where(coords == coord)[0][0]
    except:
        if include_stop and (coord == coords[-1] + (coords[-1]-coords[-2])):
            return len(coords)
        raise IndexError("Coordinate {} not in axis".format(coord))
    return index

def gen_coord_list(start, step, count):
    return np.arange(start, start + step*count, step)


def np_float_to_bytes(numpy_float):
    # How is this so hard?
    return struct.pack("<I", int((numpy_float).astype(int)))

def np_float_to_bytes_signed(numpy_float):
    return struct.pack("<i", int((numpy_float).astype(int)))

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


def get_chunk_cache_size(n_il_chunks, n_xl_chunks):
    """Determine how many chunks are required to hold an arbitrary diagonal in lru cache - must be power of 2"""
    cache_size = 1
    max_chunk_dimension = min(n_il_chunks, n_xl_chunks)
    while cache_size < max_chunk_dimension:
        cache_size = cache_size*2
    return cache_size * 2
