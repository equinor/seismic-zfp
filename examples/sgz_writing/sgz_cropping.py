import sys
import numpy as np

from seismic_zfp.cropping import SgzCropper
import seismic_zfp

def main():
    if len(sys.argv) != 3:
        raise RuntimeError("This example accepts exactly 2 arguments: input_file & output_file")

    # Crop input file
    cropper = SgzCropper(sys.argv[1])
    cropper.write_cropped_file(sys.argv[2], (9985,9989), (1932,1936), (0,4505))

    # Check cropping results
    with seismic_zfp.open(sys.argv[1]) as orig_file:
        subvol_orig = orig_file.subvolume[9985:9989, 1932:1936, :]
        for h in orig_file.header[0:1]:
            print(h)

    with seismic_zfp.open(sys.argv[2]) as crop_file:
        for h in crop_file.header[0:1]:
            print(h)
        subvol_crop = crop_file.read_volume()

    assert np.allclose(subvol_orig, subvol_crop, rtol=1e-12)

if __name__ == '__main__':
    main()
