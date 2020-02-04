import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import segyio

from pyzfp import compress, decompress

from seismic_zfp.utils import pad

LINE_NO = 17500
filepath = sys.argv[1]
outpath = sys.argv[2]


lines_to_read = range(4*LINE_NO//4, 4*LINE_NO//4 + 4)

with segyio.open(filepath) as segyfile:
    sampling_interval = segyfile.bin[segyio.BinField.Interval] / 1000000.0
    line = segyfile.xline[segyfile.xlines[LINE_NO]]
    slice_segy = line.T
    lines_to_compress = np.zeros((4, line.shape[0], line.shape[1]))
    for i, line in enumerate(lines_to_read):
        lines_to_compress[i, :, :] = segyfile.xline[segyfile.xlines[LINE_NO]]


bitrates = [4, 2, 1]
decompressed_slices = {}

for bits_per_voxel in bitrates:
    padded_shape = (4, pad(lines_to_compress.shape[1], 4), pad(lines_to_compress.shape[2], 2048//bits_per_voxel))
    data_padded = np.zeros(padded_shape, dtype=np.float32)
    data_padded[0:4, 0:lines_to_compress.shape[1], 0:lines_to_compress.shape[2]] = lines_to_compress
    compressed = compress(data_padded, rate=bits_per_voxel)

    decompressed = decompress(compressed, (padded_shape[0], padded_shape[1], padded_shape[2]),
                              np.dtype('float32'), rate=bits_per_voxel)

    decompressed_slices[bits_per_voxel] = decompressed[LINE_NO % 4, 0:slice_segy.shape[1], 0:slice_segy.shape[0]].T

CLIP = 45000.0
SCALE = 1.0/(2.0*CLIP)

from PIL import Image
im = Image.fromarray(np.uint8(cm.seismic((slice_segy.clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(outpath, 'test_inline-orig.png'))


def precompute_seismic_trace_dft_kernel(trace, time_vector):
    """
    returns a kernel to perform a discrete Fourier transform
    on a seismic trace and frequency vector
    trace is from one IL,XL position
    time_vector should be in terms of time, not samples; e.g,
    if you have a trace that goes from 0 to 2.5 s TWT, this
    naturally has 626 samples given a 4 ms sample rate...
    make sure you thus have created a time_vector as...
    time_vector = np.arange(len(trace)) * 0.004
    provided you are looping through traces in a cube, only
    run this function once and keep applying the kernel
    ignoring zero-padding for now
    I need to define the freq_vector more robustly...
    I should have a zero frequency... but the way I build it is meh..
    Would be better to define [0...Nyqist] and just flip and *-1
    """
    dt = time_vector[1] - time_vector[0]
    nyqst_f = 1/dt/2
    df = 1/(len(trace)*dt)
    if len(trace) % 2 == 0:
        freq_vector = np.arange(-nyqst_f, nyqst_f, df)
        if len(freq_vector) % 2 == 1:
            freq_vector = freq_vector[:-1]
    else:
        freq_vector = np.arange(-nyqst_f + 0.5 * df, nyqst_f, df)
    # if len(time_vector) != len(freq_vector):
    # raise ValueError('time and frequency vectors do not have same number of
    # samples')
    dft_kernel = np.exp(-2j * np.pi * np.outer(freq_vector, time_vector)) * dt
    return dft_kernel, freq_vector


def seismic_trace_spectrum(trace, dft_kernel, convention=''):
    """
    compute amplitude spectrum for a trace input
    returns amplitude spectrum in terms of convention
    you need to already have computed the dft_kernel using:
    precompute_seismic_trace_dft_kernel(...)
    option for either 'power' or 'field' convention
    defaults to an arbitrary un-normed spectrum if nothing given
    ...do something with Hilbert and phase later
    """
    t2f = np.dot(dft_kernel, trace)
    spectrum = np.abs(t2f)
    non_periodic = (len(spectrum)+1)//2
    if convention == 'field':
        spectrum = 20*np.log10(spectrum/max(spectrum[non_periodic+75:]))
    elif convention == 'power':
        spectrum = 10*np.log10(spectrum/max(spectrum[non_periodic+75:]))
    return spectrum


def get_spectra_average(data, dt):
    dft_kernel, freq_vector = precompute_seismic_trace_dft_kernel(data[:, 0], np.arange(len(data[:, 0])) * dt)

    spectra = np.zeros(np.shape(data))
    # loop over all traces in line - just take everything...
    for IL_number, trace_n in enumerate(data.T):
        spectra[:, IL_number] = seismic_trace_spectrum(trace_n, dft_kernel, convention='power')
    num_of_inlines = np.shape(spectra)[1]
    return freq_vector, np.sum(spectra, axis=1)/num_of_inlines


spectra_averages = {}


def make_int8(slice, stdevs):
    clip = np.std(slice)*stdevs
    return np.rint((255.0 * (slice + clip)/(2.0*clip)) / 255.0) * 2.0 * clip - clip


freq_vector, spectra_averages['segy'] = get_spectra_average(slice_segy, dt=sampling_interval)
freq_vector, spectra_averages['int8'] = get_spectra_average(make_int8(slice_segy, 0.25), dt=sampling_interval)


im = Image.fromarray(np.uint8(cm.seismic((make_int8(slice_segy, 0.25).clip(-CLIP, CLIP) + CLIP) * SCALE)*255))
im.save(os.path.join(outpath, 'test_inline-int8.png'))

for bitrate in bitrates:
    _, spectra_averages[str(bitrate)] = get_spectra_average(decompressed_slices[bitrate], dt=sampling_interval)


np.set_printoptions(precision=3, threshold=10000, linewidth=200)

for k, spectrum_average in spectra_averages.items():
    plt.plot(freq_vector, spectrum_average)

plt.rcParams.update({'font.size': 18})
plt.legend(spectra_averages.keys(), loc='upper right')
plt.ylim(-40, 0)
plt.xlim(0, freq_vector[-1])
plt.ylabel('Power (dB)', fontsize=18)
plt.xlabel('Frequency (1/s)', fontsize=18)
fig = plt.gcf()
fig.set_size_inches(18, 12)
plt.tick_params(labelsize=16)
plt.savefig(os.path.join(outpath, 'test_spectrum-int8.png'))
