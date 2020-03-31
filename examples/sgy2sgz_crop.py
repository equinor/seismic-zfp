import sys
from seismic_zfp.conversion import SegyConverter


def main():
    if len(sys.argv) != 8:
        raise RuntimeError(
            "This example accepts exactly 7 arguments: in_file, out_file, bitrate, min_il, max_il, min_xl & max_xl")

    with SegyConverter(sys.argv[1],
                       min_il=int(sys.argv[4]), max_il=int(sys.argv[5]),
                       min_xl=int(sys.argv[6]), max_xl=int(sys.argv[7])) as converter:
        converter.run(sys.argv[2], bits_per_voxel=int(sys.argv[3]), method="Stream")


if __name__ == '__main__':
    main()
