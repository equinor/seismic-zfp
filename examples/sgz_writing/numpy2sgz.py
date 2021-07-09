import sys
from seismic_zfp.conversion import NumpyConverter
from seismic_zfp.utils import generate_fake_seismic


def main():
    if len(sys.argv) != 6:
        raise RuntimeError("This example accepts exactly 5 arguments: output_file, n_ilines, n_xlines, n_samples & bitrate")

    array, ilines, xlines, samples = generate_fake_seismic(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

    with NumpyConverter(array, ilines=ilines, xlines=xlines, samples=samples) as converter:
        converter.run(sys.argv[1], bits_per_voxel=sys.argv[5])


if __name__ == '__main__':
    main()
