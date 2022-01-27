from seismic_zfp.read import SgzReader

def cube(filename):
    """Reads entire cube from SGZ file

    Parameters
    ----------
    filename : str
        The SGZ filepath to be read

    Returns
    -------
    cube : numpy.ndarray of float32, shape: (n_ilines, n_xlines, n_samples)
        The entire cube, decompressed
    """
    with SgzReader(filename) as reader:
        return reader.read_volume()

def dt(reader):
    """Delta-time: Infer a dt, the sample rate, from the file.

    Parameters
    ----------
    reader : SgzReader
        Open SGZ file

    Returns
    -------
    dt : float
        Sample rate
    """
    return 1000 * (reader.samples[1] - reader.samples[0])
