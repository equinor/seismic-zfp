import sys
import numpy as np
from seismic_zfp.conversion import NumpyConverter


def main():
    if len(sys.argv) != 6:
        raise RuntimeError("This example accepts exactly 5 arguments: output_file, n_ilines, n_xlines, n_samples & bitrate")

    ilines, xlines, samples = np.arange(int(sys.argv[2])), np.arange(int(sys.argv[3])), np.arange(int(sys.argv[4]))

    # Generate an array which looks a *bit* like an impulse-response test...
    array_shape = (len(ilines), len(xlines), len(samples))
    i = np.broadcast_to(np.expand_dims(np.expand_dims((ilines - len(ilines) / 2), 1), 2), array_shape).astype(np.float32)
    x = np.broadcast_to(np.expand_dims((xlines - len(xlines) / 2), 1), array_shape).astype(np.float32)
    s = np.broadcast_to(samples - len(samples) / 4, array_shape).astype(np.float32)
    array = np.sin(np.sqrt(i**2 + x**2 + s**2)/5)/(1e-2*np.sqrt(i**2 + x**2 + s**2))

    with NumpyConverter(array, ilines=ilines, xlines=xlines, samples=samples) as converter:
        converter.run(sys.argv[1], bits_per_voxel=sys.argv[5])

if __name__ == '__main__':
    main()
