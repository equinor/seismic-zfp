import numpy as np
import pytest
import seismic_zfp
import segyio

SGZ_FILE_1 = 'test_data/small_1bit.sgz'
SGZ_FILE_2 = 'test_data/small_2bit.sgz'
SGZ_FILE_4 = 'test_data/small_4bit.sgz'
SGZ_FILE_8 = 'test_data/small_8bit.sgz'
SGY_FILE = 'test_data/small.sgy'

SGZ_FILE_DEC_8 = 'test_data/small-dec_8bit.sgz'
SGY_FILE_DEC = 'test_data/small-dec.sgy'


def compare_inline_ordinal(sgz_filename, sgy_filename, lines_to_test, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_ordinal in lines_to_test:
                slice_segy = segyfile.iline[segyfile.ilines[line_ordinal]]
                slice_sgz = sgzfile.iline[sgzfile.ilines[line_ordinal]]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)

def compare_inline_number(sgz_filename, sgy_filename, lines_to_test, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_number in lines_to_test:
                slice_segy = segyfile.iline[line_number]
                slice_sgz = sgzfile.iline[line_number]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def compare_inline_slicing(sgz_filename):
    slices = [slice(1, 5, 2), slice(1, 2, None), slice(1, 3, None), slice(None, 3, None), slice(3, None, None)]
    with seismic_zfp.open(sgz_filename) as sgzfile:
        for slice_ in slices:
            slices_slice = np.asarray(sgzfile.iline[slice_])
            start = slice_.start if slice_.start is not None else 1
            stop = slice_.stop if slice_.stop is not None else 6
            step = slice_.step if slice_.step is not None else 1
            slices_concat = np.asarray([sgzfile.iline[i] for i in range(start, stop, step)])
            assert np.array_equal(slices_slice, slices_concat)

def test_inline_accessor():
    compare_inline_ordinal(SGZ_FILE_1, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-2)
    compare_inline_ordinal(SGZ_FILE_2, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-4)
    compare_inline_ordinal(SGZ_FILE_4, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-6)
    compare_inline_ordinal(SGZ_FILE_8, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-10)
    compare_inline_ordinal(SGZ_FILE_DEC_8, SGY_FILE_DEC, [0, 1, 2], tolerance=1e-6)

    compare_inline_number(SGZ_FILE_1, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-2)
    compare_inline_number(SGZ_FILE_2, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-4)
    compare_inline_number(SGZ_FILE_4, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-6)
    compare_inline_number(SGZ_FILE_8, SGY_FILE, [1, 2, 3, 4, 5], tolerance=1e-10)
    compare_inline_number(SGZ_FILE_DEC_8, SGY_FILE_DEC, [1, 3, 5], tolerance=1e-6)

    compare_inline_slicing(SGZ_FILE_8)


def compare_crossline_ordinal(sgz_filename, sgy_filename, lines_to_test, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_ordinal in lines_to_test:
                slice_segy = segyfile.xline[segyfile.xlines[line_ordinal]]
                slice_sgz = sgzfile.xline[sgzfile.xlines[line_ordinal]]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)

def compare_crossline_number(sgz_filename, sgy_filename, lines_to_test, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(sgy_filename) as segyfile:
            for line_number in lines_to_test:
                slice_segy = segyfile.xline[line_number]
                slice_sgz = sgzfile.xline[line_number]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)


def compare_crossline_slicing(sgz_filename):
    slices = [slice(20, 21, 2), slice(21, 23, 1), slice(None, 22, None), slice(22, None, None)]
    with seismic_zfp.open(sgz_filename) as sgzfile:
        for slice_ in slices:
            slices_slice = np.asarray(sgzfile.xline[slice_])
            start = slice_.start if slice_.start is not None else 20
            stop = slice_.stop if slice_.stop is not None else 25
            step = slice_.step if slice_.step is not None else 1
            slices_concat = np.asarray([sgzfile.xline[i] for i in range(start, stop, step)])
            assert np.array_equal(slices_slice, slices_concat)


def test_crossline_accessor():
    compare_crossline_ordinal(SGZ_FILE_1, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-2)
    compare_crossline_ordinal(SGZ_FILE_2, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-4)
    compare_crossline_ordinal(SGZ_FILE_4, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-6)
    compare_crossline_ordinal(SGZ_FILE_8, SGY_FILE, [0, 1, 2, 3, 4], tolerance=1e-10)
    compare_crossline_ordinal(SGZ_FILE_DEC_8, SGY_FILE_DEC, [0, 1, 2], tolerance=1e-6)

    compare_crossline_number(SGZ_FILE_1, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-2)
    compare_crossline_number(SGZ_FILE_2, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-4)
    compare_crossline_number(SGZ_FILE_4, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-6)
    compare_crossline_number(SGZ_FILE_8, SGY_FILE, [20, 21, 22, 23, 24], tolerance=1e-10)
    compare_crossline_number(SGZ_FILE_DEC_8, SGY_FILE_DEC, [20, 22, 24], tolerance=1e-6)

    compare_crossline_slicing(SGZ_FILE_8)


def compare_zslice(sgz_filename, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(SGY_FILE) as segyfile:
            for line_number in range(50):
                slice_sgz = sgzfile.depth_slice[line_number]
                slice_segy = segyfile.depth_slice[line_number]
                assert np.allclose(slice_sgz, slice_segy, rtol=tolerance)

def compare_depthslice_slicing(sgz_filename):
    slices = [slice(0, 5, None), slice(45, None, None), slice(None, None, 5), slice(25, None, 5), slice(None, 20, 5)]
    with seismic_zfp.open(sgz_filename) as sgzfile:
        for slice_ in slices:
            slices_sgz = np.asarray(sgzfile.depth_slice[slice_])
            start = slice_.start if slice_.start is not None else 0
            stop = slice_.stop if slice_.stop is not None else 50
            step = slice_.step if slice_.step is not None else 1
            slices_concat = np.asarray([sgzfile.depth_slice[i] for i in range(start, stop, step)])
            assert np.array_equal(slices_sgz, slices_concat)

def test_zslice_accessor():
    compare_zslice(SGZ_FILE_1, tolerance=1e-2)
    compare_zslice(SGZ_FILE_2, tolerance=1e-4)
    compare_zslice(SGZ_FILE_4, tolerance=1e-6)
    compare_zslice(SGZ_FILE_8, tolerance=1e-10)

    compare_depthslice_slicing(SGZ_FILE_8)


def compare_subvolume(sgz_filename, sgy_filename, il_min, il_max, il_step, xl_min, xl_max, xl_step, z_min, z_max, z_step, tolerance):
    with seismic_zfp.open(sgz_filename) as sgzfile:
        with segyio.open(sgy_filename) as sgyfile:
            vol_sgz = sgzfile.subvolume[il_min:il_max:il_step, xl_min:xl_max:xl_step, z_min:z_max:z_step]

            start_il = 0 if il_min is None else np.where(sgyfile.ilines == il_min)[0][0]
            stop_il = sgyfile.n_ilines if (il_max is None) or (il_max == sgyfile.ilines[-1] + sgyfile.ilines[1] - sgyfile.ilines[0]) else np.where(sgyfile.ilines ==il_max)[0][0]
            step_il = il_step // (sgyfile.ilines[1] - sgyfile.ilines[0]) if il_step is not None else 1

            start_xl = 0 if xl_min is None else np.where(sgyfile.xlines == xl_min)[0][0]
            stop_xl = sgyfile.n_xlines if (xl_max is None) or (xl_max == sgyfile.xlines[-1] + sgyfile.xlines[1] - sgyfile.xlines[0]) else np.where(sgyfile.xlines ==xl_max)[0][0]
            step_xl = xl_step // (sgyfile.xlines[1] - sgyfile.xlines[0]) if xl_step is not None else 1

            start_z = 0 if z_min is None else np.where(sgyfile.samples.astype('intc') == z_min)[0][0]
            stop_z = len(sgyfile.samples) if (z_max is None) or (z_max == sgyfile.samples.astype('intc')[-1] + sgyfile.samples.astype('intc')[1] - sgyfile.samples.astype('intc')[0]) else np.where(sgyfile.samples.astype('intc') == z_max)[0][0]
            step_z = int(z_step // (sgyfile.samples[1] - sgyfile.samples[0])) if z_step is not None else 1

            vol_segy = segyio.tools.cube(sgy_filename)[start_il:stop_il:step_il,
                                                       start_xl:stop_xl:step_xl,
                                                       start_z:stop_z:step_z]

            assert np.allclose(vol_sgz, vol_segy, rtol=tolerance)

def test_subvolume_accessor():
    compare_subvolume(SGZ_FILE_8, SGY_FILE, 1,4,None, 21,23,None, 12,36,None, tolerance=1e-10)
    compare_subvolume(SGZ_FILE_8, SGY_FILE, 1,4,None, 21,23,None, 12,36,4, tolerance=1e-10)
    compare_subvolume(SGZ_FILE_8, SGY_FILE, 1,4,None, 21,23,2, 12,36,None, tolerance=1e-10)
    compare_subvolume(SGZ_FILE_8, SGY_FILE, 1,4,2, 21,23,None, 12,36,None, tolerance=1e-10)
    compare_subvolume(SGZ_FILE_DEC_8, SGY_FILE_DEC, 1,5,None, 20,22,None, None,None,None, tolerance=1e-6)
    compare_subvolume(SGZ_FILE_DEC_8, SGY_FILE_DEC, 1,3,2, 20,24,2, None,None,8, tolerance=1e-6)

def test_subvolume_accessor_errors():
    with seismic_zfp.open(SGZ_FILE_4) as sgzfile:

        with pytest.raises(IndexError):
            sgzfile.subvolume[1:7:None, 20:21:None, 0:40:None]

        with pytest.raises(IndexError):
            sgzfile.subvolume[1:6:None, 20:26:None, 0:40:None]

        with pytest.raises(IndexError):
            sgzfile.subvolume[1:6:None, 20:21:None, 0:400:None]

        with pytest.raises(IndexError):
            sgzfile.subvolume[0:6:None, 20:21:None, 0:40:None]

        with pytest.raises(IndexError):
            sgzfile.subvolume[1:6:None, 20:21:None, 0:40:3]

    with seismic_zfp.open(SGZ_FILE_DEC_8) as sgzfile:

        with pytest.raises(IndexError):
            sgzfile.subvolume[1:6:1, 20:21:None, 0:None:None]

        with pytest.raises(IndexError):
            sgzfile.subvolume[1:6:2, 20:21:3, 0:None:None]


def compare_cube(sgz_filename, sgy_filename, tolerance):
    vol_sgy = segyio.tools.cube(sgy_filename)
    vol_sgz = seismic_zfp.tools.cube(sgz_filename)
    assert np.allclose(vol_sgz, vol_sgy, rtol=tolerance)

def compare_dt(sgz_filename, sgy_filename):
    with segyio.open(sgy_filename) as sgy_file:
        dt_sgy = segyio.tools.dt(sgy_file)
    with seismic_zfp.open(sgz_filename) as sgz_file:
        dt_sgz = seismic_zfp.tools.dt(sgz_file)
    assert dt_sgy == dt_sgz

def test_tools_functions():
    compare_cube(SGZ_FILE_8, SGY_FILE, tolerance=1e-10)
    compare_dt(SGZ_FILE_8, SGY_FILE)
