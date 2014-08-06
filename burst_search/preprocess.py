"""Preprocessing data for fast radio burst searches.

This module contains, bandpass calibration, RFI flagging, etc.

"""

import numpy as np


def remove_periodic(data, period):
    """Remove periodic time compenent from data.

    Parameters
    ----------
    data : array with shape ``(ntime, nfreq)``.
    period : integer
        Must be greater than or equal to *ntime*.

    Returns
    -------
    profile : array with shape ``(period, nfreq)``.
        Component removed from the data.
    """

    period = int(period)

    if data.ndim != 2:
        raise ValueError("Expected 2D data.")
    ntime = data.shape[0]
    if ntime < period:
        raise ValueError("Time axis must be more than one period.")
    nfreq = data.shape[1]

    ntime_trunk = ntime // period * period
    data_trunk = data[:ntime_trunk,:]
    data_trunk.shape = (ntime_trunk // period, period, nfreq)

    profile =  np.mean(data_trunk, 0)

    data_trunk -= profile
    data[ntime_trunk:,:] -= profile[:ntime - ntime_trunk,:]

    return profile


def noisecal_bandpass(data, cal_spectrum, cal_period):
    """Remove noise-cal and use to bandpass calibrate.

    Parameters
    ----------
    data : array with shape ``(ntime, nfreq)``
        Data to be calibrated including time switched noise-cal.
    cal_spectrum : array with shape ``(nfreq,)``
        Calibrated spectrum of the noise cal.
    cal_period : int
        Noise cal switching period, Must be an integer number of samples.

    """"

    cal_profile = remove_periodic(data, cal_period)
    # An *okay* estimate of the height of a square wave is twice the standard
    # deviation.
    cal_amplitude = 2 * np.std(cal_profile, 0)
    data *= cal_spectrum / cal_amplitude

