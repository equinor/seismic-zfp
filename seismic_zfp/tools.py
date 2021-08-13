from seismic_zfp.read import SgzReader

def cube(filename):
    with SgzReader(filename) as reader:
        return reader.read_volume()
