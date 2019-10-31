import sys
from seismic_zfp.convert import convert_segy


def main():
    if len(sys.argv) != 4:
        raise RuntimeError("This example accepts exactly 3 arguments: input_file, output_file & bitrate")

    convert_segy(sys.argv[1], sys.argv[2], bits_per_voxel=int(sys.argv[3]), method="Stream")


if __name__ == '__main__':
    main()
