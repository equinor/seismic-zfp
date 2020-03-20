import sys
from seismic_zfp.conversion import SegyConverter


def main():
    if len(sys.argv) != 4:
        raise RuntimeError("This example accepts exactly 3 arguments: input_file, output_file & bitrate")

    with SegyConverter(sys.argv[1]) as converter:
        converter.run(sys.argv[2], bits_per_voxel=float(sys.argv[3]), method="Stream")


if __name__ == '__main__':
    main()
