from seismic_zfp.read import SgzReader

def cube(filename):
    with SgzReader(filename) as reader:
        return reader.read_volume()

def dt(reader):
    return 1000 * (reader.samples[1] - reader.samples[0])
