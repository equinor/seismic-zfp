
def pad(orig, multiple):
    if orig%multiple == 0:
        return orig
    else:
        return multiple * (orig//multiple + 1)


def np_float_to_bytes(numpy_float):
    return int((numpy_float).astype(int)).to_bytes(4, byteorder='little')
