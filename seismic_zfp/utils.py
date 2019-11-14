
def pad(orig, multiple):
    if orig%multiple == 0:
        return orig
    else:
        return multiple * (orig//multiple + 1)


def np_float_to_bytes(numpy_float):
    # How is this so hard?
    return int((numpy_float).astype(int)).to_bytes(4, byteorder='little')


def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='little')


def define_blockshape(bits_per_voxel, blockshape):
    if bits_per_voxel == -1:
        bits_per_voxel = 4096 * 8 // (blockshape[0] * blockshape[1] * blockshape[2])
    else:
        if blockshape[0] == -1:
            blockshape = (4096 * 8 // (blockshape[1] * blockshape[2] * bits_per_voxel), blockshape[1], blockshape[2])
        elif blockshape[1] == -1:
            blockshape = (blockshape[0], 4096 * 8 // (blockshape[2] * blockshape[0] * bits_per_voxel), blockshape[2])
        elif blockshape[2] == -1:
            blockshape = (blockshape[0], blockshape[1], 4096 * 8 // (blockshape[0] * blockshape[1] * bits_per_voxel))
        else:
            assert(bits_per_voxel * blockshape[0] * blockshape[1] * blockshape[2] == 4096 * 8)
    return bits_per_voxel, blockshape
