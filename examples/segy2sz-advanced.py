import sys
from seismic_zfp.conversion import SegyConverter


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("This example accepts exactly 2 arguments: input_file & output_file")

    with SegyConverter(sys.argv[1]) as converter:
        converter.run(sys.argv[2], bits_per_voxel=2, blockshape=(64, 64, 4), method="Stream")


if __name__ == '__main__':
    main()
