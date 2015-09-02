"""Preprocessing data for fast radio burst searches.

This module contains, bandpass calibration, RFI flagging, etc.

"""

import numpy as np
from scipy import signal, fftpack

from _preprocess import remove_continuum_v2


def remove_periodic(data, period):
    """Remove periodic time compenent from data.

    Parameters
    ----------
    data : array with shape ``(nfreq, ntime)``.
    period : integer
        Must be greater than or equal to *ntime*.

    Returns
    -------
    profile : array with shape ``(nfreq, period)``.
        Component removed from the data.
    """

    period = int(period)

    if data.ndim != 2:
        raise ValueError("Expected 2D data.")
    ntime = data.shape[1]
    if ntime < period:
        raise ValueError("Time axis must be more than one period.")
    nfreq = data.shape[0]

    ntime_trunk = ntime // period * period
    data_trunk = data[:,:ntime_trunk]
    data_trunk.shape = (nfreq, ntime_trunk // period, period)

    profile =  np.mean(data_trunk, 1)

    for ii in xrange(0, ntime_trunk, period):
        data[:,ii:ii + period] -= profile

    data[:,ntime_trunk:] -= profile[:,:ntime - ntime_trunk]

    return profile


def noisecal_bandpass(data, cal_spectrum, cal_period):
    """Remove noise-cal and use to bandpass calibrate.

    Do not use this function. The estimate of the cal amplitude is very noisy.
    Need an algorithm to find the square wave.

    Parameters
    ----------
    data : array with shape ``(nfreq, ntime)``
        Data to be calibrated including time switched noise-cal.
    cal_spectrum : array with shape ``(nfreq,)``
        Calibrated spectrum of the noise cal.
    cal_period : int
        Noise cal switching period, Must be an integer number of samples.

    """

    cal_profile = remove_periodic(data, cal_period)
    # An *okay* estimate of the height of a square wave is twice the standard
    # deviation.  This is really bad if there is any noise at all... which
    # there is.
    cal_amplitude = 2 * np.std(cal_profile, 1)
    # Find frequencies with no data.
    bad_chans = cal_amplitude < 1e-5 * np.median(cal_amplitude)
    cal_amplitude[bad_chans] = 1.
    data *= (cal_spectrum / cal_amplitude)[:,None]
    data[bad_chans,:] = 0


def sys_temperature_bandpass(data):
    """Bandpass calibrate based on system temperature.

    The lowest noise way to flatten the bandpass. Very good if T_sys is
    relatively constant accross the band.
    """

    T_sys = np.median(data, 1)
    bad_chans = T_sys < 0.001 * np.median(T_sys)
    T_sys[bad_chans] = 1
    data /= T_sys[:,None]
    data[bad_chans,:] = 0

def remove_outliers(data, sigma_threshold, block=None):
    """Flag outliers within frequency channels.

    Replace outliers with that frequency's mean.

    """

    nfreq0 = data.shape[0]
    ntime0 = data.shape[1]

    if block is None:
        block = ntime0

    if ntime0 % block:
        raise ValueError("Time axis must be divisible by block."
                         " (ntime, block) = (%d, %d)." % (ntime0, block))

    ntime = block
    nfreq = nfreq0 * (ntime0 // block)

    data.shape = (nfreq, ntime)
    

    # To optimize cache usage, process one frequency at a time.
    for ii in range(nfreq):
        this_freq_data = data[ii,:]
        mean = np.mean(this_freq_data)
        std = np.std(this_freq_data)
        outliers = abs(this_freq_data - mean) > sigma_threshold * std
        this_freq_data[outliers] = mean

    data.shape = (nfreq0, ntime0)


def remove_noisy_freq(data, sigma_threshold):
    """Flag frequency channels with high variance.

    To be effective, data should be bandpass calibrated in some way.

    """

    nfreq = data.shape[0]
    ntime = data.shape[1]

    # Calculate variances without making full data copy (as numpy does).
    var = np.empty(nfreq, dtype=np.float64)
    skew = np.empty(nfreq, dtype=np.float64)
    for ii in range(nfreq):
        var[ii] = np.var(data[ii,:])
        skew[ii] = np.mean((data[ii,:] - np.mean(data[ii,:])**3))

    # Find the bad channels.
    bad_chans = False
    for ii in range(3):
        bad_chans_var = abs(var - np.mean(var)) > sigma_threshold * np.std(var)
        bad_chans_skew = abs(skew - np.mean(skew)) > sigma_threshold * np.std(skew)
        bad_chans = np.logical_or(bad_chans, bad_chans_var)
        bad_chans = np.logical_or(bad_chans, bad_chans_skew)
        var[bad_chans] = np.mean(var)
        skew[bad_chans] = np.mean(skew)
    data[bad_chans,:] = 0


def remove_bad_times(data, sigma_threshold):

    nfreq = data.shape[0]
    ntime = data.shape[1]

    # Calculate time means and frequency means with no copies.
    mean = np.empty(nfreq, dtype=np.float64)
    freq_sum = 0.
    for ii in range(nfreq):
        mean[ii] = np.mean(data[ii])
        freq_sum += data[ii]

    bad_times = (abs(freq_sum - np.mean(freq_sum))
                 > sigma_threshold * np.std(freq_sum))
    data[:,bad_times] = mean[:,None]


def remove_continuum(data):
    """Calculates a contiuum template and removes it from the data.

    Also removes the time mean from each channel.

    """

    nfreq = data.shape[0]
    ntime = data.shape[1]

    # Remove the time mean.
    data -= np.mean(data, 1)[:,None]

    # Efficiently calculate the continuum template. Numpy internal looping
    # makes np.mean/np.sum inefficient.
    continuum = 0.
    for ii in range(nfreq):
        continuum += data[ii]

    # Normalize.
    continuum /= np.sqrt(np.sum(continuum**2))

    # Subtract out the template.
    for ii in range(nfreq):
        data[ii] -= np.sum(data[ii] * continuum) * continuum


def highpass_filter(data, width):
    """Highpass filter on *width* scales using blackman window.

    Finite impulse response filter *that discards invalid data* at the ends.

    """

    ntime = data.shape[-1]

    # Blackman FWHM factor.
    window_width = int(width / 0.4054785)

    if window_width % 2:
        window_width += 1

    window = np.zeros(ntime, dtype=np.float32)
    window_core = signal.blackman(window_width, sym=True)
    window_core = -window_core / np.sum(window_core)
    window_core[window_width // 2] += 1
    window[:window_width] = window_core
    window_fft = fftpack.fft(window)

    ntime_out = data.shape[-1] - window_width + 1
    out_shape = data.shape[:-1] + (ntime_out,)
    out = np.empty(out_shape, data.dtype)

    for ii in range(data.shape[0]):
        d_fft = fftpack.fft(data[ii])
        d_fft *= window_fft
        d_lpf = fftpack.ifft(d_fft)
        out[ii] = d_lpf[-ntime_out:].real

    return out


