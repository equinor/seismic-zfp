import sys
from seismic_zfp.convert import convert_segy


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("This example accepts exactly 2 arguments: input_file & output_file")

    convert_segy(sys.argv[1], sys.argv[2], method="InMemory")


if __name__ == '__main__':
    main()
