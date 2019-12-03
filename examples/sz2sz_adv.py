import sys
from seismic_zfp.read import SzReader


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("This example accepts exactly 2 arguments: input_file & output_file")

    reader = SzReader(sys.argv[1])
    reader.write_adv_sz(sys.argv[2])


if __name__ == '__main__':
    main()
