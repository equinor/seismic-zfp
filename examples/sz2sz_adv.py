import sys
from seismic_zfp.conversion import SzConverter


def main():
    if len(sys.argv) != 3:
        raise RuntimeError("This example accepts exactly 2 arguments: input_file & output_file")

    with SzConverter(sys.argv[1]) as converter:
        converter.convert_to_adv_sz(sys.argv[2])


if __name__ == '__main__':
    main()
