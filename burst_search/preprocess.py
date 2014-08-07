"""Preprocessing data for fast radio burst searches.

This module contains, bandpass calibration, RFI flagging, etc.

"""

import numpy as np


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
    # deviation.
    cal_amplitude = 2 * np.std(cal_profile, 1)
    # Find frequencies with no data.
    bad_chans = cal_amplitude < 1e-5 * np.median(cal_amplitude)
    cal_amplitude[bad_chans] = 1.
    data *= (cal_spectrum / cal_amplitude)[:,None]
    data[bad_chans,:] = 0


def remove_outliers(data, sigma_threshold):
    """Flag outliers within frequency channels.

    Replace outliers with that frequency's mean.

    """

    pass


def remove_noisy_freq(data, sigma_threshold):
    """Flag frequency channels with high variance.

    To be effective, data should be bandpass calibrated.

    """

