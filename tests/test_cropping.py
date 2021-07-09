import os
import numpy as np
from seismic_zfp.conversion import NumpyConverter
from seismic_zfp.cropping import SgzCropper
from seismic_zfp.utils import generate_fake_seismic
import seismic_zfp

def generate_data_crop_and_compare(tmp_path, n_samples, min_iline, n_ilines, min_xline, n_xlines, crop_min_il, crop_il_size, crop_min_xl, crop_xl_size,
                                   bits_per_voxel, blockshape=(4, 4, -1)):
    gen_sgz = os.path.join(str(tmp_path), 'generated.sgz')
    crop_sgz = os.path.join(str(tmp_path), 'cropped.sgz')

    array, ilines, xlines, samples = generate_fake_seismic(n_ilines, n_xlines, n_samples,
                                                           min_iline=min_iline, min_xline=min_xline)

    with NumpyConverter(array, ilines=ilines, xlines=xlines, samples=samples) as converter:
        converter.run(gen_sgz, bits_per_voxel=bits_per_voxel, blockshape=blockshape)

    iline_coords_range = (crop_min_il,crop_min_il+crop_il_size)
    xline_coords_range = (crop_min_xl,crop_min_xl+crop_xl_size)
    zslice_coords_range = None

    # Crop input file
    cropper = SgzCropper(gen_sgz)
    cropper.write_cropped_file_by_coords(crop_sgz, iline_coords_range,
                                                   xline_coords_range,
                                                   zslice_coords_range)

    # Check cropping results
    with seismic_zfp.open(crop_sgz) as crop_file:
        subvol_crop = crop_file.read_volume()
        cropped_iline_coords = (crop_file.ilines[0], crop_file.ilines[-1] + (crop_file.ilines[-1]-crop_file.ilines[-2]))
        cropped_xline_coords = (crop_file.xlines[0], crop_file.xlines[-1] + (crop_file.xlines[-1]-crop_file.xlines[-2]))
        cropped_zslice_coords = (crop_file.zslices[0], crop_file.zslices[-1] + (crop_file.zslices[-1]-crop_file.zslices[-2]))

    with seismic_zfp.open(gen_sgz) as orig_file:
        subvol_orig = orig_file.subvolume[cropped_iline_coords[0]:cropped_iline_coords[1],
                                          cropped_xline_coords[0]:cropped_xline_coords[1],
                                          cropped_zslice_coords[0]:cropped_zslice_coords[1]]

    assert np.allclose(subvol_orig, subvol_crop, rtol=1e-12)

def test_crop_file(tmp_path):
    generate_data_crop_and_compare(tmp_path, 101, 8, 12, 9985, 16, 8, 4, 9989, 4, 4, blockshape=(4, 4, -1))
    generate_data_crop_and_compare(tmp_path, 101, 8, 12, 9985, 16, 8, 4, 9989, 4, 8, blockshape=(4, 4, -1))
    generate_data_crop_and_compare(tmp_path, 17, 64, 201, 1, 64, 64, 128, 1, 64, 2, blockshape=(64, 64, 4))