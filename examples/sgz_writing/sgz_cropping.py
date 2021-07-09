import sys
import numpy as np

from seismic_zfp.cropping import SgzCropper
import seismic_zfp

def main():
    if len(sys.argv) != 3:
        raise RuntimeError("This example accepts exactly 2 arguments: input_file & output_file")

    iline_coords_range = (9988,9997)
    xline_coords_range = (1932,1937)
    zslice_coords_range = None

    # Crop input file
    cropper = SgzCropper(sys.argv[1])
    cropper.write_cropped_file_by_coords(sys.argv[2], iline_coords_range,
                                                      xline_coords_range,
                                                      zslice_coords_range)

    # Check cropping results
    with seismic_zfp.open(sys.argv[2]) as crop_file:
        subvol_crop = crop_file.read_volume()
        cropped_iline_coords = (crop_file.ilines[0], crop_file.ilines[-1] + (crop_file.ilines[-1]-crop_file.ilines[-2]))
        cropped_xline_coords = (crop_file.xlines[0], crop_file.xlines[-1] + (crop_file.xlines[-1]-crop_file.xlines[-2]))
        cropped_zslice_coords = (crop_file.zslices[0], crop_file.zslices[-1] + (crop_file.zslices[-1]-crop_file.zslices[-2]))

    with seismic_zfp.open(sys.argv[1]) as orig_file:
        subvol_orig = orig_file.subvolume[cropped_iline_coords[0]:cropped_iline_coords[1],
                                          cropped_xline_coords[0]:cropped_xline_coords[1],
                                          cropped_zslice_coords[0]:cropped_zslice_coords[1]]

    assert np.allclose(subvol_orig, subvol_crop, rtol=1e-12)
    print("Cropping success!")

if __name__ == '__main__':
    main()
