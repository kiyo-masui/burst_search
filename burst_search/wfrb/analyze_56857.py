"""Working script for preprocessing and analysing the data for FRB110523.
This script is treated somewhat like a workbook. Function calls in main() are
intended to be commented out to control which part of the analysis is run.
When running, many mysterious plots will pop up, all of which serve a purpose
but figuring out what many of these are for will require some reverse
engineering.
Feel free to email Kiyoshi Masui with any questions.
"""

import math
from os import path

import numpy as np
from numpy import random
from numpy.polynomial import Legendre, Chebyshev
from scipy import signal, interpolate, special, optimize, fftpack, linalg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits

# https://github.com/kiyo-masui/burst_search
from burst_search import preprocess, dedisperse


BURST = True

if BURST:
    # For analysis of FRB110523
    DATAROOT = "/scratch2/p/pen/hsiuhsil/burst_candidates/56857_1hr_0018/raw_data/"
    OUT_ROOT = '/scratch2/p/pen/hsiuhsil/burst_candidates/56857_1hr_0018'
    FILENAME = "guppi_56857_wigglez1hr_centre_0018_0001.fits"

    # Calibration source scans.
    SRCFILES = [
        "guppi_56857_3C48_0007_0001.fits",
        "guppi_56857_3C48_0008_0001.fits",
        "guppi_56857_3C48_0009_0001.fits",
        "guppi_56857_3C48_0010_0001.fits",
        ]

    # Hard code the phase of the noise cal, since I'm too lazy to write an algorithm
    # to find it.
    DATA_CAL_PHASE = 38
    SRC_CAL_PHASES = [ 59, 14, 31, 50]

    # The small time slice of data containing the burst.
    TSL = np.s_[25500:31000]

else:
    # For analysis of pulsar single pulses.
    DATAROOT = '/Users/kiyo/data/raw_guppi/AGBT14B_339'
    OUT_ROOT = 'psr_data'

    FILENAME = "guppi_57132_B0329+54_0020_0001.fits"
    DATA_CAL_PHASE = 4
    #FILENAME = "guppi_57132_B1929+10_0010_0001.fits"
    #DATA_CAL_PHASE = 0
    #FILENAME = "guppi_57132_B2319+60_0021_0001.fits"
    #DATA_CAL_PHASE = 53

    SRCFILES = [
            "guppi_57132_3C48_0016_0001.fits",
            "guppi_57132_3C48_0017_0001.fits",
            "guppi_57132_3C48_0018_0001.fits",
            "guppi_57132_3C48_0019_0001.fits",
            ]
    SRC_CAL_PHASES = [11, 9, 40, 38]

    #TSL = np.s_[2000:10000]
    TSL = np.s_[2000:-2000]


# Explicit cuts for reasonances and band edges.
FREQ_CUTS = [
        (699., 702.),
        (795., 798.),
        (816., 819.),
        (898., 901.),
        ]


# From my narrow spectral bin fits, which looks much better.
BEAM_DATA_FWHM = [0.307, 0.245]
BEAM_DATA_FREQ = [699.9, 900.1]


# These are the fit parameters that come out of fit_basic(), which agree with
# Jonathan Sievers' MCMC fits to the expected level (1 sigma).
FIT_PARS = [26881.81213399937, 145.2163514832117, 0.00082513254798026068, 
           -2.0071160312987062, 0.0042048710612025281, -0.00050943984565836905]

B2319_PARS = [52187.604450591927, 94.783490187710271, 0.17607219167484128,
        2.0978781725590876, 0.010991132078497551, -0.0010897690798342145]

# Polarization fit parameters borrowed from Jon.
F_REF_RM = 764.2
RM_SEIV = 186.1
PHI_RM = 3.2228 + np.pi
F_POL = 0.44
ALPHA_POL = 6.64


if BURST:
    # Shift some parameters for well behaved fitting.
    T_OFF = 26881.8
    DM_OFF = 146
else:
    # Pulsar dependant:
    # B2319
    T_OFF = 52187.6
    DM_OFF = 94


matplotlib.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.serif' : ['Times New Roman'],
    'font.sans-serif' : ['Helvetica', 'Arial'],
    })


def main():
    """Each function below is independent, in that any of them can be run with
    the others commented out using saved data. However, the preprocessing must
    be run before the others at least once, as it produces the saved data.
    """

    #Preprocessing
#    reformat_raw_data()
#    calibrator_spectra()
#    calibrate()
#    filter()

    #plot()

    # Fitting.
#    fit_basic()
    #fit_beam()    # Fit does not converge.

    # Spectral plots.
    #plot_spectra()

    # Scintillation.
    #fit_scintil()

    # Faraday rotation measurement.
    rm_measure()

    # Pulse profile plots.
    #pol_profile()

    # Ephemeris.
    #arrival_coords()



# Import raw fits data
# ====================

def reformat_raw_data():
    hdulist = pyfits.open(path.join(DATAROOT, FILENAME), 'readonly')
    if BURST:
        data, time, freq, ra, dec, az, el = read_fits_data(hdulist, 285, 315)
    else:
        # These pulsar files are too long to read all of them.
        data, time, freq, ra, dec, az, el = read_fits_data(hdulist, 0, 25)
    hdulist.close()

    # Masking.
    mask_chans = np.any(np.isnan(data[:,:,0]), 1)
    data[mask_chans] = 0.

    np.save(OUT_ROOT + '/time.npy', time)
    np.save(OUT_ROOT + '/freq.npy', freq)
    np.save(OUT_ROOT + '/ra.npy', ra)
    np.save(OUT_ROOT + '/dec.npy', dec)
    np.save(OUT_ROOT + '/az.npy', az)
    np.save(OUT_ROOT + '/el.npy', el)
    export_data(OUT_ROOT + '/raw.npy', data, mask_chans)

    return data, mask_chans, time, freq, ra, dec, az, el


def sample_subint(sub_time, sub_var, time):
    # This interpolator will also extrapolate.
    interpolator = interpolate.InterpolatedUnivariateSpline(sub_time, sub_var,
            k=1)
    return interpolator(time)


def read_fits_data(hdulist, start_record=None, end_record=None):
    if start_record is None:
        start_record = 0

    # Parameters.
    mheader = hdulist[0].header
    dheader = hdulist[1].header
    delta_t = dheader['TBIN']
    nfreq = dheader['NCHAN']
    delta_f = dheader['CHAN_BW']
    freq0 = mheader['OBSFREQ'] - mheader['OBSBW'] / 2. + delta_f / 2
    time0 = mheader["STT_SMJD"] + mheader["STT_OFFS"]

    #freq = np.arange(nfreq) * delta_f + freq0  # MHz
    freq = hdulist[1].data[0]['DAT_FREQ'].astype(float)

    nrecords = len(hdulist[1].data)
    if end_record is None or end_record > nrecords:
        end_record = nrecords
    nrecords_read = end_record - start_record
    ntime_record, npol, nfreq, one = hdulist[1].data[0]["DATA"].shape

    time = (np.arange(nrecords_read * ntime_record) * delta_t
            + time0 + start_record * delta_t * ntime_record)
    subint_time = time0 + hdulist[1].data['OFFS_SUB']
    ra = sample_subint(subint_time, hdulist[1].data['RA_SUB'], time)
    dec = sample_subint(subint_time, hdulist[1].data['DEC_SUB'], time)
    az = sample_subint(subint_time, hdulist[1].data['TEL_AZ'], time)
    el = 90. - sample_subint(subint_time, hdulist[1].data['TEL_ZEN'], time)

    out_data = np.empty((nfreq, npol, nrecords_read, ntime_record), dtype=np.float32)
    for ii in xrange(nrecords_read):
        # Read the record.
        record = hdulist[1].data[start_record + ii]["DATA"]
        scl = hdulist[1].data[start_record + ii]["DAT_SCL"]
        scl.shape = (1, npol, nfreq, 1)
        offs = hdulist[1].data[start_record + ii]["DAT_OFFS"]
        offs.shape = (1, npol, nfreq, 1)
        record *= scl
        record += offs
        # Interpret as unsigned int (for Stokes I only).
        record = record.view(dtype=np.uint8)
        # Select stokes I and copy.
        out_data[:,0,ii,:] = np.transpose(record[:,0,:,0])
        # Interpret as signed int (except Stokes I).
        record = record.view(dtype=np.int8)
        out_data[:,1,ii,:] = np.transpose(record[:,1,:,0])
        out_data[:,2,ii,:] = np.transpose(record[:,2,:,0])
        out_data[:,3,ii,:] = np.transpose(record[:,3,:,0])
    out_data.shape = (nfreq, npol, nrecords_read * ntime_record)

    return out_data, time, freq, ra, dec, az, el


# Calibration
# ===========

def calibrator_spectra():

    src_means = []
    for ii, filename in enumerate(SRCFILES):
        srchdulist = pyfits.open(path.join(DATAROOT, filename), 'readonly')
        src_data, time, freq, ra, dec, az, el = read_fits_data(srchdulist)
        preprocess_data(src_data, srchdulist, SRC_CAL_PHASES[ii])
        src_means.append(np.mean(src_data, -1))
        del src_data
        srchdulist.close()
    np.save(OUT_ROOT + '/src_means.npy', src_means)
    return src_means


def calibrate():
    data, mask_chans, time, freq, ra, dec, az, el = import_all('raw')
    hdulist = pyfits.open(path.join(DATAROOT, FILENAME), 'readonly')
    preprocess_data(data, hdulist, DATA_CAL_PHASE)
    
    plt.figure()
    plt.plot(freq, data[:,0,0])
    plt.plot(freq, np.mean(data, -1)[:,3])

    src_means = np.load(OUT_ROOT + '/src_means.npy')

    src_flux_cal_units = (src_means[0] + src_means[2]
                          - src_means[1] - src_means[3]) / 2

    # Noise in the cal measurement is by far the limiting factor in
    # calibration, but should be a smooth function of frequency. Fit for
    # it.
    # Discard cross polarizations, for following fits.
    src_flux_cal_units = src_flux_cal_units[:,[0,3]]
    plt_ind = 1
    plt.figure()
    plt.plot(freq, src_flux_cal_units[:,plt_ind])
    mask_chans[np.any(np.logical_not(np.isfinite(src_flux_cal_units)), 1)] = \
            True
    src_flux_cal_units[mask_chans,:] = 0
    src_flux_cal_units = smooth_flag_spectra(freq, src_flux_cal_units,
            mask_chans, 30)
    plt.plot(freq, src_flux_cal_units[:,plt_ind])
    plt.plot(freq[mask_chans], src_flux_cal_units[mask_chans,plt_ind], '.')
    plt.show()


    src_flux = calibrator_spectrum('3C48', freq)
    cal_T = src_flux[:,None] / src_flux_cal_units

    # Calibrate the data.
    cal_T[mask_chans,:] = 0
    mask_chans[np.logical_or(cal_T[:,0] < 0, cal_T[:,1] < 0)] = True
    data[mask_chans,:,:] = 0
    cal_T[mask_chans,:] = 0
    # Multiply by appropriate T_cal.
    data[:,0,:] *= cal_T[:,0,None]
    data[:,3,:] *= cal_T[:,1,None]
    cal_T_XY = np.sqrt(cal_T[:,0] * cal_T[:,1])
    data[:,1,:] *= cal_T_XY[:,None]
    data[:,2,:] *= cal_T_XY[:,None]


    export_data(OUT_ROOT + '/calibrated.npy', data, mask_chans)


def preprocess_data(data, hdulist, cal_phase):
    rotate_to_XY(data)

    mheader = hdulist[0].header
    dheader = hdulist[1].header
    if mheader['CAL_FREQ']:
        cal_period = 1. / mheader['CAL_FREQ']
        cal_period = int(round(cal_period / dheader['TBIN']))
    else:
        # Cal scans missing cal period info for some reason.
        cal_period = 64
    noise_cal_normalize(data, cal_period, cal_phase)


def noise_cal_normalize(data, period, phase):
    #data = np.mean(data, 0)
    #data.shape = (1,) + data.shape

    nfreq = data.shape[0]

    phase_factors = np.zeros(nfreq, dtype=float)
    
    p = 0
    for ii in range(nfreq):
        profile = preprocess.remove_periodic(data[ii], period)
        data[ii] += np.mean(profile, -1)[:,None]
        profile = np.roll(profile, -phase, axis=1)
        p += profile
        cal_on = np.mean(profile[:,:31], 1)
        cal_off = np.mean(profile[:,32:63], 1)
        cal = cal_on - cal_off
        if (cal[0] <= 0 or cal[3] <= 0) or (cal[1] == 0 and cal[2] == 0):
            data[ii] = float('nan')
            phase_factors[ii] = float('nan')
        else:
            cal_xy = np.sqrt(cal[0] * cal[3])
            # Flux normalization.
            data[ii, 0] /= cal[0]
            data[ii, 3] /= cal[3]
            data[ii, 1] /= cal_xy
            data[ii, 2] /= cal_xy
            # Phase calibration, assume noise cal is pure Stokes U.
            phase_factor = np.arctan2(cal[2] / cal_xy, cal[1] / cal_xy)
            cross_data = data[ii, 1] + 1j * data[ii, 2]
            cross_data *= np.exp(-1j * phase_factor)
            data[ii, 1] = cross_data.real
            data[ii, 2] = cross_data.imag
            phase_factors[ii] = phase_factor
    if True:
        plt.figure()
        plt.plot(p[0] + p[3])
        plt.plot(p[0] + p[3], '.')
        plt.show()


def calibrator_spectrum(src, freq):
    gain = 2.    # K/Jy
    # Data from arXiv:1211.1300v1.
    if src == '3C48':
        #coeff = [1.3324, -0.7690, -0.1950, 0.059]
        coeff = [1.3332, -0.7665, -0.1980, 0.064]
        #coeff = [2.345, 0.071, -0.138, 0.]
    elif src == '3C295':
        coeff = [1.4866, -0.7871, -0.3440, 0.0749]

    l_freq_ghz = np.log10(freq / 1e3)    # Convert to GHz.
    spec = 0
    for ii, A in enumerate(coeff):
        spec += A * l_freq_ghz**ii
    spec = 10.**(spec) * gain
    #plt.figure()
    #plt.plot(freq, spec)

    return spec


# Filter and RFI flag
# ===================

def filter():

    data, mask_chans, time, freq, ra, dec, az, el = import_all('calibrated')

    # Testing code to validate that calibration didn't do anything crazy.
    #data, mask_chans, time, freq, ra, dec, az, el = import_all('raw')
    #rotate_to_XY(data)
    #for ii in range(len(freq)):
    #    profile = preprocess.remove_periodic(data[ii], 64)
    #    data[ii] += np.mean(profile, -1)[:,None]


    mean = np.mean(data, axis=-1)
    delta_t = abs(np.mean(np.diff(time)))
    high_pass(data, 0.2 / delta_t)    # 200 ms HPF.

    std = np.empty(data.shape[:-1], dtype=float)
    skew = np.empty(data.shape[:-1], dtype=float)
    for ii in range(len(freq)):
        m = np.mean(data[ii], -1)
        this_data = data[ii] - m[:,None]
        std[ii] = np.sqrt(np.mean(this_data**2, -1))
        third_moment = np.mean(this_data**3, -1)
        skew[ii] = third_moment

    norm = mean.copy()
    norm[:,[1,2]] = np.sqrt(norm[:,0] * norm[:,3])[:,None]
    norm[mask_chans,:] = 1.

    std /= norm
    skew /= norm**3

    # Explicit cut list.
    for cut_low, cut_high in FREQ_CUTS:
        mask_chans[np.logical_and(freq >= cut_low, freq <= cut_high)] = True
    data[mask_chans] = 0

    plt.figure()
    plt.plot(freq,norm[:,0])
    plt.plot(freq,norm[:,1])
    plt.plot(freq,norm[:,2])
    plt.plot(freq,norm[:,3])

    #plt.figure()
    #plt.plot(freq,std[:,0])
    #plt.plot(freq,std[:,1])
    #plt.plot(freq,std[:,2])
    #plt.plot(freq,std[:,3])

    std = smooth_flag_spectra(freq, std,
            mask_chans, 30)
    #plt.plot(freq,std[:,0])
    #plt.plot(freq,std[:,1])
    #plt.plot(freq,std[:,2])
    #plt.plot(freq,std[:,3])
    #plt.plot(freq[mask_chans],std[mask_chans,0], '.')
    #plt.plot(freq[mask_chans],std[mask_chans,1], '.')
    #plt.plot(freq[mask_chans],std[mask_chans,2], '.')
    #plt.plot(freq[mask_chans],std[mask_chans,3], '.')

    #plt.figure()
    #plt.plot(freq,skew[:,0])
    #plt.plot(freq,skew[:,1])
    #plt.plot(freq,skew[:,2])
    #plt.plot(freq,skew[:,3])

    skew = smooth_flag_spectra(freq, skew,
            mask_chans, 30)
    #plt.plot(freq,skew[:,0])
    #plt.plot(freq,skew[:,1])
    #plt.plot(freq,skew[:,2])
    #plt.plot(freq,skew[:,3])
    #plt.plot(freq[mask_chans],skew[mask_chans,0], '.')
    #plt.plot(freq[mask_chans],skew[mask_chans,1], '.')
    #plt.plot(freq[mask_chans],skew[mask_chans,2], '.')
    #plt.plot(freq[mask_chans],skew[mask_chans,3], '.')

    plt.show()

    rotate_to_IQUV(data)

    export_data(OUT_ROOT + '/filtered.npy', data, mask_chans)

    t_sl = TSL
    np.save(OUT_ROOT + "/time_short.npy", time[t_sl])
    export_data(OUT_ROOT + "/filtered_short.npy", data[:,:,t_sl], mask_chans)


def high_pass(data, sig):
    """Gaussian high pass filter with standard deviation *sig* samples."""
    nfreq = data.shape[0]
    npol = data.shape[1]

    nkernal = int(round(4 * sig)) * 2 + 1
    nblank = (nkernal - 1) // 2
    kernal = signal.gaussian(nkernal, sig)
    kernal /= -np.sum(kernal)
    kernal[nblank] += 1
    for ii in range(nfreq):
        for jj in range(npol):
            data[ii,jj,nblank:-nblank] = signal.fftconvolve(data[ii,jj],
                    kernal, 'valid')
            data[ii,jj,:nblank] = 0
            data[ii,jj,-nblank:] = 0


# First look at the burst
# =======================

def plot():
    """With no rebinning it is hard to see. Blow this plot up and zoom in
    around 850 MHz.
    """

    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered_short')
    delta_t = np.mean(np.diff(time))

    d = data[:,0,:]

    plt.figure()
    s = np.std(d)
    plt.imshow(d,
            vmin=-1*s,
            vmax=3*s,
            extent=[0, d.shape[1] * delta_t, 700., 900.],
            aspect='auto',
            )

    plt.show()


# Fits to intensity time-frequency data.
# ======================================

def fit_basic():

    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered')

    var = np.empty(data.shape[:-1], dtype=float)
    for ii in range(len(freq)):
        # Exclude the beginning and end which is invalid due to filtering.
        var[ii] = np.var(data[ii,:,5000:-5000], -1)

    data_I = data[:,0,TSL]

    time = time[TSL]

    ntime = len(time)
    # Real scan angle.  Constant elevation scan.
    scan_loc = (az - az[0]) * np.cos(el[0] * np.pi / 180)

    pars0 = [0.011, 0.9, 0.001, 2., 0.001, 0.001]

    std_I = np.sqrt(var[:,0])
    weights = np.empty_like(std_I)
    weights[np.logical_not(mask_chans)] = 1. / std_I[np.logical_not(mask_chans)]
    weights[mask_chans] = 0
    nfitdata = ntime * np.sum(np.logical_not(mask_chans))
    npars = len(pars0)

    # Plot the profile and initial model.
    pars0_real = unwrap_basic_pars(pars0)
    plot_pulse(data_I, freq, time, pars0_real[0], pars0_real[1])
    initial_model = -residuals_basic(
                np.zeros_like(data_I),
                freq,
                time,
                pars0_real,
                )
    plot_pulse(initial_model, freq, time, pars0_real[0], pars0_real[1])
    plt.show()

    residuals = lambda p: (
            residuals_basic(
                data_I,
                freq,
                time,
                unwrap_basic_pars(p),
                )
            * weights[:,None]).flat[:]
    chi2 = lambda p: np.sum(residuals(p)**2)

    pars, cov, info, msg, ierr = optimize.leastsq(
            residuals,
            pars0,
            epsfcn=0.0001,
            full_output=True,
            ftol=0.00001 / nfitdata,
            )
    print "Fit status:", msg, ierr
    real_pars = unwrap_basic_pars(pars)



    Chi2 = np.sum(residuals(pars)**2)
    delta_Chi2 = np.sum(residuals([0,0,0,1,.001,.001])**2) - Chi2
    red_Chi2 = Chi2 / (nfitdata - npars)
    print "Delta chi-squared:\n", delta_Chi2
    print "Reduced chi-squared:\n", red_Chi2
    print "Parameters:\n", real_pars
    errs = np.sqrt(cov.flat[::npars + 1])
    corr = cov / errs / errs[:,None]
    print "Errors:\n", errs
    print "Correlations:\n", corr



def plot_pulse(data_I, freq, time, t0, dm, time_range=0.4):

    time_selector = RangeSelector(time)
    delay = delay_from_dm(freq, dm, t0)

    profile = 0.
    for ii in range(len(freq)):
        start_ind, stop_ind = time_selector(delay[ii] - time_range/2,
                                            delay[ii] + time_range/2)
        profile += data_I[ii, start_ind:stop_ind]
    profile /= len(freq)

    start_ind, stop_ind = time_selector(delay[0] - time_range/2,
                                        delay[0] + time_range/2)
    plt.plot(time[start_ind:stop_ind], profile)



def residuals_basic(data_I, freq, time, pars, beam=None, scan_loc=None):
    print pars
    t0 = pars[0]
    dm = pars[1]
    amp_800 = pars[2]
    alpha = pars[3]
    #alpha = 2
    width = pars[4]
    scatter_800 = pars[5]

    amp = amp_800 * (freq / 800.)**-alpha
    scatter = scatter_800 * (freq / 800.)**-4

    delta_f = np.median(np.abs(np.diff(freq)))
    delay = delay_from_dm(freq, dm, t0)

    #plt.plot(freq, amp)
    if not beam is None:
        b = pars[6]
        s0 = pars[7]
        scan_loc_f = interpolate.interp1d(time, scan_loc)(delay)
        amp *= np.exp(-0.5 * (b**2 + (scan_loc_f - s0)**2) / beam**2)
    #plt.plot(freq, amp)
    #plt.show()
    
    time_selector = RangeSelector(time)
    
    # For testing.
    p_sums = np.empty_like(freq)

    delta_t = np.median(np.diff(time))
    residuals = data_I.astype(float)
    for ii in range(len(freq)):
        nw = 10.
        window_low = nw * width
        window_high = nw * width + nw**2 / 2 * scatter[ii]
        start_ind, stop_ind = time_selector(delay[ii] - window_low,
                                            delay[ii] + window_high)
        near_times = time[start_ind:stop_ind].copy()
        pulse = windowed_pulse(near_times, freq[[ii]], delta_t, delta_f, dm,
                t0, width, scatter[ii])
        p_sums[ii] = np.sum(pulse) * delta_t
        pulse *= amp[ii]
        residuals[ii, start_ind:stop_ind] -= pulse
    #plt.plot(freq, p_sums)
    #plt.show()
    return residuals


def unwrap_basic_pars(pars):
    """Unwraps reparameterization for fitting."""
    out = list(pars)
    out[0] += T_OFF
    out[1] += DM_OFF
    return out


def wrap_basic_pars(pars):
    out = list(pars)
    out[0] -= T_OFF
    out[1] -= DM_OFF
    return out


def fit_beam():
    """Attempts to fit for location in the beam.
    This is highly degenerate and doesn't work very well. Fit does not
    converge.
    """

    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered')

    var = np.empty(data.shape[:-1], dtype=float)
    for ii in range(len(freq)):
        # Exclude the beginning and end which is invalid due to filtering.
        var[ii] = np.var(data[ii,:,5000:-5000], -1)

    data_I = data[:,0,TSL]
    time = time[TSL]
    az = az[TSL]
    ntime = len(time)
    # Real scan angle.  Constant elevation scan.
    scan_loc = (az - az[0]) * np.cos(el[0] * np.pi / 180)

    fwhm_factor = (2 * np.sqrt(2 * np.log(2)))
    beam_sigma_init = np.array(BEAM_DATA_FWHM) / fwhm_factor
    beam_sigma = interpolate.interp1d(BEAM_DATA_FREQ, beam_sigma_init)(freq)

    pars0 = wrap_basic_pars(FIT_PARS + [0.1, 0.1])

    std_I = np.sqrt(var[:,0])
    weights = np.empty_like(std_I)
    weights[np.logical_not(mask_chans)] = 1. / std_I[np.logical_not(mask_chans)]
    weights[mask_chans] = 0
    nfitdata = ntime * np.sum(np.logical_not(mask_chans))
    npars = len(pars0)


    residuals = lambda p: (
            residuals_basic(
                data_I,
                freq,
                time,
                unwrap_basic_pars(p),
                beam_sigma,
                scan_loc,
                )
            * weights[:,None]).flat[:]

    pars, cov, info, msg, ierr = optimize.leastsq(
            residuals,
            pars0,
            epsfcn=0.0001,
            full_output=True,
            ftol=0.01/ nfitdata,
            )
    real_pars = unwrap_basic_pars(pars)

    Chi2 = np.sum(residuals(pars)**2)
    red_Chi2 = Chi2 / nfitdata
    print "Reduced chi-squared:\n", red_Chi2
    print "Parameters:\n", real_pars
    errs = np.sqrt(cov.flat[::npars + 1])
    corr = cov / errs / errs[:,None]
    print "Errors:\n", errs
    print "Correlations:\n", corr



# Spectral plots
# ==============

def plot_spectra():
    matplotlib.rcParams.update({'font.size': 16,
        })
    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered_short')

    spec = integrated_pulse_spectrum(data, freq, time, FIT_PARS,
            True)

    spec_out = spec.copy()
    spec_out[mask_chans] = float('nan')
    np.save(OUT_ROOT + "/polarized_spectra.npy", spec_out)
    #plt.figure()
    #plt.plot(freq, spec[:,0])
    #plt.plot(freq, spec[:,1])
    #plt.plot(freq, spec[:,2])
    #plt.plot(freq, spec[:,3])

    rebin_fact = 128
    spec_rebin = rebin_freq(spec, mask_chans, rebin_fact)

    f = freq[rebin_fact//2::rebin_fact]

    # Plot polarized spectra.

    f_ref = F_REF_RM
    # Stokes I.
    alpha = FIT_PARS[3]
    amp = FIT_PARS[2] * (f_ref/800.)**-alpha
    # Pol.
    RM = RM_SEIV
    phi = PHI_RM
    amp_pol = F_POL * amp
    alpha_pol = ALPHA_POL

    fact = 1000

    fig = plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(f, fact * spec_rebin[:,0], '.k')
    plt.plot(freq, fact * amp * (freq / f_ref)**-alpha, 'k')
    plt.ylim(-0.002 * fact, 0.015 * fact)
    plt.ylabel('I')
    plt.yticks(np.arange(0, 0.020, 0.005) * fact)

    pol_angle = 2 * RM * ((3e8 / freq / 1e6)**2 - (3e8 / f_ref / 1e6)**2) + phi
    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(f, fact * spec_rebin[:,1], '.k')
    plt.plot(freq, fact * amp_pol * (freq / f_ref)**-alpha_pol
             * np.cos(pol_angle), 'k')
    plt.ylabel('Q')
    plim = (-0.0060 * fact, 0.0060 * fact)
    plt.ylim(*plim)
    #pticks = np.arange(-0.004, 0.006, 0.002) * fact
    pticks = np.arange(-0.005, 0.006, 0.005) * fact
    plt.yticks(pticks)

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(f, fact * spec_rebin[:,2], '.k')
    plt.plot(freq, fact * amp_pol * (freq / f_ref)**-alpha_pol
             * np.sin(pol_angle), 'k')
    plt.ylim(*plim)
    plt.ylabel('U')
    plt.yticks(pticks)

    #ax4 = plt.subplot(414, sharex=ax1)
    #plt.plot(f, fact * spec_rebin[:,3], '.k')
    #plt.ylim(-0.01 * fact, 0.01 * fact)
    #plt.ylabel('V')


    xticklabels = (ax1.get_xticklabels() + ax2.get_xticklabels()
    #        + ax3.get_xticklabels()
            )
    plt.setp(xticklabels, visible=False)
    plt.subplots_adjust(hspace=0.0001)

    plt.xlabel('Frequency (MHz)')

    fig.text(0.03, 0.5, 'Fluence (K ms)',
            ha='center', va='center', rotation='vertical')

    plt.show()


def integrated_pulse_spectrum(data, freq, time, pars, matched=True):
    t0 = pars[0]
    dm = pars[1]
    amp_800 = pars[2]
    alpha = pars[3]
    width = pars[4]
    scatter_800 = pars[5]

    scatter = scatter_800 * (freq / 800.)**-4
    delay = delay_from_dm(freq, dm, t0)

    delta_t = np.median(np.diff(time))
    delta_f = np.median(np.abs(np.diff(freq)))

    out = np.empty(data.shape[:-1], dtype=float)

    time_selector = RangeSelector(time)

    for ii in range(len(freq)):
        if matched:
            nw = 10.
        else:
            nw = 3.
        window_low = nw * width
        window_high = nw * width + nw**2 / 2 * scatter[ii]
        start_ind, stop_ind = time_selector(delay[ii] - window_low,
                                            delay[ii] + window_high)
        near_times = time[start_ind:stop_ind].copy()
        if matched:
            pulse = windowed_pulse(near_times, freq[[ii]], delta_t, delta_f,
                    dm, t0, width, scatter[ii])
            # Matched filter normalization.
            pulse /= np.sum(pulse**2) * delta_t
            # Compensated? Probably don't need to if noise is white.
            #pulse -= np.mean(pulse)
        else:
            pulse = np.ones_like(near_times)
        for jj in range(data.shape[1]):
            out[ii, jj] = np.sum(data[ii,jj,start_ind:stop_ind] * pulse)
    out *= delta_t
    return out


# Scintillation analysis
# ======================

def fit_scintil():

    # What part of the pulse profile to correlate.
    part = 'full'
    #part = 'head'
    #part = 'tail'
    #part = 'cross'

    # For fits only use first 150 lags. This keeps chi-2 from being bumped
    # up by irrelevant lags since noise isn't well modelled.
    #nfitdata = 150
    # For plotting figures, use 300 lags, which better shows the shape of the
    # correlation function.
    nfitdata = 300

    matplotlib.rcParams.update({'font.size': 16,
        })

    #### Data selection ####
    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered')
    if False:
        # Only process lower half the band, for highest signal to noise and
        # frequency dependance effects.
        f_sl = np.s_[-len(freq) // 2:]
        #f_sl = np.s_[:]
        #f_sl = np.s_[-len(freq) // 4:]
        #f_sl = np.s_[2 * len(freq) // 4:3 * len(freq) //4]
        data = data[f_sl]
        freq = freq[f_sl]
        mask_chans = mask_chans[f_sl]
        nfreq = len(freq)
    else:
        f_sl = np.s_[:]
        nfreq = len(freq)

    not_mask = np.logical_not(mask_chans)
    f_lags = freq_lags(freq)

    ##### Signal processing ####

    pulse_pars1 = list(FIT_PARS)
    pulse_pars2 = list(FIT_PARS)
    if part is 'tail':
        pulse_pars1[0] += 0.003
        pulse_pars2[0] += 0.003
        noise_spectra_fname1 = "burst_data/noise_spectra_tail.npy"
        noise_spectra_fname2 = "burst_data/noise_spectra_tail.npy"
    elif part is 'head':
        pulse_pars1[5] = 0.
        pulse_pars2[5] = 0.
        noise_spectra_fname1 = "burst_data/noise_spectra_head.npy"
        noise_spectra_fname2 = "burst_data/noise_spectra_head.npy"
    elif part is 'cross':
        pulse_pars1[5] = 0.
        pulse_pars2[0] += 0.003
        noise_spectra_fname1 = "burst_data/noise_spectra_head.npy"
        noise_spectra_fname2 = "burst_data/noise_spectra_tail.npy"
    else:
        noise_spectra_fname1 = "burst_data/noise_spectra.npy"
        noise_spectra_fname2 = "burst_data/noise_spectra.npy"
    
    spec1 = integrated_pulse_spectrum(data[:,:,TSL], freq, time[TSL],
            pulse_pars1, True)
    spec2 = integrated_pulse_spectrum(data[:,:,TSL], freq, time[TSL],
            pulse_pars2, True)
    spec_norm1, plaw_pars1 = norm_to_plaw(spec1[:,0], freq, mask_chans)
    spec_norm2, plaw_pars2 = norm_to_plaw(spec2[:,0], freq, mask_chans)

    #### Noise simulation ####

    # Choose times for noise samples.
    # Beginning is contaminated by filter.
    time_inds = time > time[5000]
    # End is contaminated by filter and leave 4s for dispersion delay.
    time_inds = np.logical_and(time_inds, time < time[-5000] - 4.)
    # Exclude 300ms around burst.
    time_inds = np.logical_and(time_inds, abs(time - pulse_pars1[0]) > 0.3)
    # Pulse is less than 50 samples.
    noise_sample_times = time[time_inds][::50]
    nnoise = len(noise_sample_times)

    # For handing data off to Ue-Li Pen.
    #spectra_to_export = np.empty((nsim + 3, 4, nfreq), dtype=float)
    #spectra_to_export[0] = freq
    #spectra_to_export[1] = mask_chans
    #spectra_to_export[2] = spec.T

    # Generate noise realizations.  This only needs to be done once.
    if True:
        noise_spectra1 = np.empty((nnoise, nfreq, 4), dtype=float)
        noise_spectra2 = np.empty((nnoise, nfreq, 4), dtype=float)
        pars1 = list(pulse_pars1)
        pars2 = list(pulse_pars2)
        for ii in range(nnoise):
            if not ii % 50:
                print "Noise realization: %d" % ii
            pars1[0] = noise_sample_times[ii] + (pulse_pars1[0] - FIT_PARS[0])
            pars2[0] = noise_sample_times[ii] + (pulse_pars2[0] - FIT_PARS[0])
            noise_spectra1[ii] = integrated_pulse_spectrum(data, freq, time,
                    pars1, True)
            noise_spectra2[ii] = integrated_pulse_spectrum(data, freq, time,
                    pars2, True)
        np.save(noise_spectra_fname1, noise_spectra1)
        np.save(noise_spectra_fname2, noise_spectra2)
    else:
        noise_spectra1 = np.load(noise_spectra_fname1)[:,f_sl,:]
        noise_spectra2 = np.load(noise_spectra_fname2)[:,f_sl,:]
    
    nsim = 4 * nnoise     # Well estimated covariances.
    #nsim = nnoise / 2    # For testing.

    sim_corr = np.empty((nsim, len(f_lags)), dtype=float)

    used_corr_model = lambda df, p: corr_model(df, p[0], p[1])
    # These are the final parameters. Note I get these if I set nfitdata=100
    # and nsim = 2*nnoise.
    pars0 = [1.1, 0.25]

    #used_corr_model = lambda df, p: corr_model2(df, p[0], p[1], p[2], p[3])
    #pars0 = [1.05, 0.15, 5., 0.1]

    npars = len(pars0)
    plaw_spec1 = general_power_law(freq, plaw_pars1)
    plaw_spec2 = general_power_law(freq, plaw_pars2)

    # Weights from noise realizations.
    channel_vars1 = np.var(noise_spectra1[:,:,0] / plaw_spec1, 0)
    channel_vars1[mask_chans] = 1
    channel_weights1 = 1. / channel_vars1
    channel_weights1[mask_chans] = 0

    channel_vars2 = np.var(noise_spectra2[:,:,0] / plaw_spec2, 0)
    channel_vars2[mask_chans] = 1
    channel_weights2 = 1. / channel_vars2
    channel_weights2[mask_chans] = 0

    #simulator = SpectrumSimulatorGauss(freq, lambda df: used_corr_model(df,
    #    pars0))
    #simulator = SpectrumSimulatorChi2(freq, lambda df: used_corr_model(df,
    #    pars0), 6)
    simulator = SpectrumSimulatorChi2(freq, lambda df: used_corr_model(df,
        pars0))
    for ii in range(nsim):
        if not ii % 50:
            print "Simulation: %d" % ii
        noise_spec1 = noise_spectra1[ii % nnoise]
        noise_spec2 = noise_spectra2[ii % nnoise]

        #spectra_to_export[ii + 3] = noise_spec.T

        noise_spec_norm1 = noise_spec1[:,0] / plaw_spec1
        noise_spec_norm2 = noise_spec2[:,0] / plaw_spec2

        sim = simulator()
        sim, tmp =  norm_to_plaw(sim + 1, freq, mask_chans)
        sim_spec_norm1 = noise_spec_norm1 + sim
        sim_spec_norm2 = noise_spec_norm2 + sim

        this_sim_corr, tmp_norm = corr_function(sim_spec_norm1, sim_spec_norm2,
                channel_weights1, channel_weights2)
        sim_corr[ii,:] = this_sim_corr

    corr, norm = corr_function(spec_norm1, spec_norm2, channel_weights1,
            channel_weights2)


    #spectra_to_export.tofile('burst_data/spectra_pol.dat')

    # Only fit to subset of frequnecy lags.
    if part is 'cross':
        # No noise bias for cross correlations.
        fit_sl = np.s_[nfreq - nfitdata:nfreq + 1 + nfitdata]
    else:
        # Excludes in-bin and nearest neighbour correlations (contraminated by
        # correlated noise).
        fit_sl = np.s_[nfreq + 1:nfreq + 1 + nfitdata]

    fit_corr = corr[fit_sl]
    fit_f_lags = f_lags[fit_sl]
    fit_sim_corr = sim_corr[:, fit_sl]

    # This attempts to fix statistics by rebinning.  Doesn't work.
    if False:
        tmp_f_lags = fit_f_lags
        tmp_sim_corr = fit_sim_corr
        rebin_first = 0.4
        rebin_n_lin = 3
        fit_f_lags, fit_corr = log_rebin(fit_f_lags, fit_corr, 
                rebin_first, rebin_n_lin)
        fit_sim_corr = np.empty((nsim, len(fit_corr)))
        for ii in range(nsim):
            tmp, fit_sim_corr[ii] = log_rebin(tmp_f_lags, 
                    tmp_sim_corr[ii], rebin_first, rebin_n_lin)

    nfitdata = len(fit_f_lags)

    # Calculate the covariance.
    mean_sim_corr = np.mean(fit_sim_corr, 0)
    deviation_sim_corr = fit_sim_corr - mean_sim_corr
    cov_sim_corr = 0.
    for ii in range(nsim):
        cov_sim_corr += (deviation_sim_corr[ii,:,None]
                         * deviation_sim_corr[ii,None,:])
    cov_sim_corr /= nsim

    # To weight residuals, multiply by the inverse of the cholesky.
    cov_sim_corr_chol = linalg.cholesky(cov_sim_corr, lower=True)

    # Perform the fit.
    def weighted_residuals(c, pars):
        return linalg.solve_triangular(
            cov_sim_corr_chol,
            (c - used_corr_model(fit_f_lags, pars)),
            lower=True,
            )

    residuals = lambda p: weighted_residuals(fit_corr, p)
    pars, cov, info, msg, ierr = optimize.leastsq(
            residuals,
            pars0,
            epsfcn=0.0001,
            full_output=True,
            ftol=0.00001 / nfitdata,
            )
    print "Fit status:", msg, ierr

    print "Data points fit:", nfitdata
    Chi2 = np.sum(residuals(pars)**2)
    red_Chi2 = Chi2 / (nfitdata - npars)
    #print "Delta chi-squared:\n", delta_Chi2
    print "Reduced chi-squared:\n", red_Chi2
    print "Parameters:\n", pars
    errs = np.sqrt(cov.flat[::npars + 1])
    correlations = cov / errs / errs[:,None]
    # Errors quoted in paper are expanded by the square-root of the reduced
    # chi-square, and added in quadrature to the variations seen between
    # different analysis choices (what part of the band to include, how many
    # lags to fit, how to model the errors).
    print "Errors:\n", errs
    print "Correlations:\n", correlations
    sim_Chi2 = np.sum(weighted_residuals(fit_sim_corr[0], pars0)**2)

    print "Sim reduced chi-squared", sim_Chi2 / (nfitdata - npars)

    plt.figure()
    plt.plot(freq, spec_norm1, '.')
    plt.plot(freq, noise_spec_norm1 + simulator(), '.')
    plt.plot(freq, simulator(), '.')

    # rebinned delta T spectra plot
    plt.figure()
    rebin_fact = 32
    f_r = freq[rebin_fact//2::rebin_fact]
    spec1_r = rebin_freq(spec_norm1[:,None], mask_chans, rebin_fact)[:,0]
    spec2_r = rebin_freq(spec_norm2[:,None], mask_chans, rebin_fact)[:,0]
    plt.plot(f_r, spec1_r)
    plt.plot(f_r, spec2_r)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel(r'$\delta T$')


    # rebinned correlation function plot
    matplotlib.rcParams.update({'font.size': 16,
        })
    plt.figure()
    rebin_first = 0.4
    rebin_n_lin = 3
    f_lags_rebin, corr_rebin = log_rebin(f_lags[fit_sl], corr[fit_sl], 
            rebin_first, rebin_n_lin)
    rebinned_sim_corr = np.empty((nsim, len(corr_rebin)))
    for ii in range(nsim):
        tmp, rebinned_sim_corr[ii] = log_rebin(f_lags[fit_sl], 
                sim_corr[ii,fit_sl], rebin_first, rebin_n_lin)
    rebinned_err = np.std(rebinned_sim_corr, 0)
    rebinned_err *= np.sqrt(1.6)    # Adjust for chi-2 fits.
    plt.semilogx(f_lags[fit_sl], used_corr_model(f_lags[fit_sl], pars0), 
            color='k', lw=2)
    plt.errorbar(f_lags_rebin, corr_rebin, rebinned_err, marker='o', color='k',
            ls='', lw=2, ms=8)
    plt.xlabel(r"$\Delta \nu$ (MHz)")
    plt.ylabel(r"$\xi(\Delta\nu)$")
    plt.xlim([0.1, 10.])

    plt.figure()
    plt.plot(fit_f_lags, residuals(pars), '.')
    plt.plot(fit_f_lags, weighted_residuals(fit_sim_corr[0], pars0), '.')


    plt.figure()
    plt.plot(f_lags[fit_sl], corr[fit_sl], '.')
    plt.plot(f_lags[fit_sl], used_corr_model(f_lags[fit_sl], pars))

    plt.figure()
    f_lag_min = f_lags[fit_sl][0]
    f_lag_max = f_lags[fit_sl][-1]
    plt.imshow(cov_sim_corr,
               extent=[f_lag_min, f_lag_max, f_lag_min, f_lag_max])
    plt.colorbar()

    plt.figure()
    plt.plot(f_lags[fit_sl], corr[fit_sl], '.')
    plt.plot(f_lags[fit_sl], np.mean(sim_corr[:,fit_sl], 0))

    plt.figure()
    plt.plot(f_lags, corr, '.')
    #plt.plot(f_lags, noise_corr, '.')
    plt.plot(f_lags, np.mean(sim_corr, 0))

    plt.figure()
    plt.plot(f_lags, corr, '.')
    plt.plot(f_lags, sim_corr[0], '.')

    plt.show()


def log_rebin(x, y, edge0, n_linear):
    this_left = 0
    this_right = edge0
    xr = []
    yr = []
    while np.any(x > this_right):
        selection = np.logical_and(x >= this_left, x < this_right)
        xr.append(np.mean(x[selection]))
        yr.append(np.mean(y[selection]))
        this_left = this_right
        if len(xr) < n_linear:
            this_right = edge0 * (len(xr) + 1)
        else:
            this_right = (1. + 1. / n_linear) * this_right
    return np.array(xr), np.array(yr)


class SpectrumSimulatorBase(object):
    """Simulate correlated random numbers with given correlation function.
    """

    def __init__(self, freq, corr_model):
        self._nfreq = len(freq)
        covariance = corr_model(freq[:,None] - freq[None,:])
        # Regularize.
        covariance.flat[::self._nfreq + 1] += 0.000001
        self._cov_chol = linalg.cholesky(covariance)

    def gen_random(self):
        """Subclass must implement."""
        pass

    def __call__(self):
        return np.dot(self._cov_chol, self.gen_random())


class SpectrumSimulatorGauss(SpectrumSimulatorBase):

    def gen_random(self):
        return random.randn(self._nfreq)


class SpectrumSimulatorChi2(SpectrumSimulatorGauss):

    def __init__(self, freq, corr_model, k=None):
        var_zero = corr_model(0.)
        #self._k = k
        #self._var_zero = var_zero
        if not k:
            self._k = 2. / var_zero
            self._norm = 1.
        else:
            self._k = k
            self._norm = np.sqrt(var_zero * k / 2.)
        sqrt_corr_model = lambda f: np.sqrt(corr_model(f) / var_zero)
        SpectrumSimulatorBase.__init__(self, freq, sqrt_corr_model)

    def __call__(self):
        k = self._k
        k_int = int(k)
        out = 0.
        for ii in range(k_int):
            out += SpectrumSimulatorGauss.__call__(self)**2
        out += (k % 1) * SpectrumSimulatorGauss.__call__(self)**2
        out -= k
        out /= k
        out *= self._norm
        return out


class _SpectrumSimulatorChi2(SpectrumSimulatorBase):

    def __init__(self, freq, corr_model, k):
        self._k = k
        SpectrumSimulatorBase.__init__(self, freq, corr_model)

    def gen_random(self):
        k = self._k
        # Generate chi-squared numbers.
        rand = 0
        for ii in range(k):
            rand += random.randn(self._nfreq)**2
        # Transform to zero mean, unit variance.
        rand -= k
        rand /= np.sqrt(2 * k)

        return rand


def corr_model(f_lags, lag_h, m):
    return m * lag_h**2 / (lag_h**2 + f_lags**2)


def corr_model2(f_lags, lag1, m1, lag2, m2):
    return corr_model(f_lags, lag1, m1) + corr_model(f_lags, lag2, m2)


def freq_lags(freq):
    nfreq = len(freq)
    delta_f = abs(np.median(np.diff(freq)))
    return np.arange(-nfreq + 1,nfreq) * delta_f


def corr_function(spec1, spec2, not_mask1, not_mask2=None):
    if not_mask2 is None:
        not_mask2 = not_mask1
    nfreq = len(spec1)

    corr = np.zeros(2 * nfreq - 1)
    norm = np.zeros(2 * nfreq - 1)
    for ii in range(nfreq):
        corr[nfreq - ii - 1:2 * nfreq - ii - 1] += (spec1[ii] * not_mask1[ii]
                                                    * spec2 * not_mask2)
        norm[nfreq - ii - 1:2 * nfreq - ii - 1] += not_mask1[ii] * not_mask2
    w = norm.copy()
    w[norm == 0] = 1
    corr = corr / w

    return corr, norm


# Faraday rotation measurement
# ============================

def rm_measure():
    """This measures the Faraday rotation measure from the time integrated
    spectrum.  It is less rigorous and includes fewer degeneracies than
    measuring the Faraday rotation from the full frequency-polarization-time
    data (As Jon did) but is simpler.
    """

    if BURST:
        data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered')
        data = data[...,TSL]
        time = time[TSL]
        #pars = list(FIT_PARS[:2]) + [0, 0, 0.0007, 0.001]
        #matched = False
        pars = list(FIT_PARS)
        matched=True
        rmpars0 = [0.003, 6, 180., 1.0]
    else:
        data, mask_chans, time, freq, ra, dec, az, el = import_all('B2319+60_filtered')
        time = time[:data.shape[-1]]
        pars = list(B2319_PARS[:2]) + [0, 0, 0.003, 0]
        rmpars0 = [0.02, 1, 200., -2.]
        matched = False
        #pars[0] = pars[0] + 2.256488426824    # next pulse

    EXTEND = False
    if EXTEND:
        rmpars0 += [0., 0., 0., 0.]


    nfreq = len(freq)

    if False:
        for ii in range(4):
            plt.figure()
            std = np.std(data[:,ii,:])
            plt.imshow(data[:,ii,:], extent=[time[0], time[-1], 700., 900.],
                    aspect='auto', vmin=-2 * std, vmax=2 * std)
            plt.plot(delay_from_dm(freq, pars[1], pars[0]), freq, 'k')
            plt.colorbar()

    spectrum = integrated_pulse_spectrum(data, freq, time, pars,
            matched=matched)

    # Noise estimate.
    noise = 0
    n_real = 33
    p = list(pars)
    p[0] = p[0] - 0.5
    for ii in range(n_real):
        p[0] = p[0] + 0.03
        noise += integrated_pulse_spectrum(data, freq, time, p,
                matched=matched)**2
    noise /= n_real
    noise = np.sqrt(noise)
    noise[mask_chans] = 1000.

    # Rebinned version of spectra for plotting and to check for systematics.
    rebin_fact = 128
    spec_rebin = rebin_freq(spectrum, mask_chans, rebin_fact)
    noise_rebin = rebin_freq(noise, mask_chans, rebin_fact)
    ngood_rebin = np.sum(np.reshape(mask_chans, (nfreq // rebin_fact, rebin_fact)), 1)
    ngood_rebin = rebin_fact - ngood_rebin
    none_good = ngood_rebin == 0
    ngood_rebin[none_good] = 1
    noise_rebin /= np.sqrt(ngood_rebin[:,None])
    noise_rebin[none_good] = 1000.
    freq_rebin = rebin_freq(freq[:,None], mask_chans, rebin_fact)[:,0]
    freq_rebin[none_good] = 800.
    # XXX Test for systematics which are easier for high signal to noise per
    # point
    #spectrum = spec_rebin; noise = noise_rebin; freq = freq_rebin; nfreq = len(freq)

    plt.figure()
    for ii in range(4):
        plt.plot(freq_rebin, spec_rebin[:,ii],)

    spec_comp = np.concatenate([spectrum[:,1], spectrum[:,2]])
    nfitdata = len(spec_comp)

    def rm_model(p):
        s = power_law(freq, p[0], p[1]).astype(complex)
        wavelength = 3e8 / (freq * 1e6)
        wavelength0 = 3e8 / 800e6
        s *= np.exp(2j * p[2] * (wavelength**2 - wavelength0**2))
        s *= np.exp(1j * p[3])
        if EXTEND:
            offset = power_law(freq, p[4], p[5])
            s.real += offset
            offset = power_law(freq, p[6], p[7])
            s.imag += offset
        return np.concatenate([s.real, s.imag])


    residuals = lambda p: ((spec_comp - rm_model(p))
                           / np.concatenate([noise[:,1], noise[:,2]]))

    npars = len(rmpars0)
    rmpars, cov, info, msg, ierr = optimize.leastsq(
            residuals,
            rmpars0,
            epsfcn=0.01,
            full_output=True,
            ftol=0.0000001 / nfitdata,
            )
    print "Fit status:", msg, ierr
    real_pars = rmpars

    Chi2 = np.sum(residuals(rmpars)**2)
    red_Chi2 = Chi2 / (nfitdata - npars)
    print "Reduced chi-squared:\n", red_Chi2
    print "Parameters:\n", real_pars
    errs = np.sqrt(cov.flat[::npars + 1])
    corr = cov / errs / errs[:,None]
    print "Errors:\n", errs
    print "Correlations:\n", corr


    plt.figure()
    plt.plot(freq_rebin, spec_rebin[:,1], '.')
    plt.plot(freq, rm_model(real_pars)[:nfreq])

    plt.plot(freq_rebin, spec_rebin[:,2], '.')
    plt.plot(freq, rm_model(real_pars)[nfreq:])


    plt.show()


# Polarized pulse profile plot
# ============================

def pol_profile():
    matplotlib.rcParams.update({'font.size': 16,
        })

    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered_short')

    t0 = FIT_PARS[0]
    dm = FIT_PARS[1]

    # RM parameters.
    f_ref = F_REF_RM
    RM = RM_SEIV
    phi = PHI_RM
    alpha = -FIT_PARS[3]

    nbins = 30

    profiles = get_pol_profile(data, mask_chans, time, freq,
            nbins, t0, dm, RM, phi, f_ref, alpha)

    noise_profiles = get_pol_profile(data, mask_chans, time, freq,
            400, t0 + 0.5, dm, RM, phi, f_ref, alpha)
    noise = np.std(noise_profiles, 1)


    plt.figure()
    SUBPLOT = True

    if SUBPLOT:
        #ax1 = plt.subplot(211)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)


    delta_t = np.median(np.diff(time))
    time_rel = (np.arange(nbins) - nbins // 3) * delta_t

    t_fact = 1000.
    plt.plot(time_rel * t_fact, profiles[0,:], '-k', lw=2,
            label=r'I',
            #label=r'I$_{\textcolor{white}{\perp}}$',
            )
    plt.plot(time_rel * t_fact, profiles[1,:], '--b', lw=2, label=r'P$_+$')
    plt.plot(time_rel * t_fact, profiles[2,:], '-.g', lw=3., label=r'P$_\times$')
    plt.plot(time_rel * t_fact, profiles[3,:], ':r', lw=2, label='V')

    plt.legend(labelspacing=1, frameon=False)

    plt.ylabel('Antenna temperature (K)')

    # Polarization angles.

    if SUBPLOT:
        plt.ylim([-0.35, 1.1])
        plt.yticks(np.arange(-0.2, 1.2, 0.2))
        #ax2 = plt.subplot(212, sharex=ax1)
        ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
    else:
        plt.xlabel('Time (ms)')
        plt.figure()

    angle = np.arctan2(profiles[2,:], profiles[1,:]) / 2
    nsim = 10000
    angle_sim_delta = np.empty((nsim, nbins))
    for ii in range(nsim):
        this_sim_angle = np.arctan2(
                profiles[2,:] + random.randn(nbins) * noise[2],
                profiles[1,:] + random.randn(nbins) * noise[1],
                ) / 2
        this_sim_angle -= angle
        # Rotate to cicle with central value at the origin.
        this_sim_angle = (this_sim_angle + np.pi / 2) % np.pi - (np.pi / 2)
        angle_sim_delta[ii] = this_sim_angle

    err = np.sqrt(np.mean(angle_sim_delta**2, 0))
    good_angles = np.sum(np.logical_and(angle_sim_delta < np.pi/4,
                         angle_sim_delta > -np.pi/4,), 0) > 0.999 * nsim

    deg_fact = 180. / np.pi
    plt.errorbar(
            time_rel[good_angles] * t_fact,
            angle[good_angles] * deg_fact,
            err[good_angles] * deg_fact,
            ls=' ',
            marker='o',
            ms=5,
            color='k',
            lw=2.,
            )
    plt.ylabel(r'$\psi$ (deg.)')

    if SUBPLOT:
        plt.yticks([0, -20, -40])

        xticklabels = ax1.get_xticklabels()
        plt.setp(xticklabels, visible=False)
        plt.subplots_adjust(hspace=0.0001)

    plt.xlabel('Time (ms)')
    plt.xlim((-10, 20))

    # Plots of noise samples.
    #plt.figure()
    #for ii in range(4):
    #    plt.plot(noise_profiles[ii])

    # Histograms of angle error simulations.
    #inds, = np.where(good_angles)
    #for ind in inds:
    #    plt.figure()
    #    plt.hist(angle_sim_delta[:,ind], 10, histtype='stepfilled')


    plt.show()


def get_pol_profile(data, mask_chans, time, freq, n_bins, t0, dm, RM, phi,
        f_ref, alpha):

    # Noise weights for frequency average.
    noise_var = np.var(data, -1)
    noise_var[mask_chans] = 1
    weights = 1. / noise_var
    weights[mask_chans] = 0
    weights = weights[:,:,None]

    nfreq = len(freq)
    npol = data.shape[1]
    nprofilebins = n_bins
    # Calculate arrival time at each frequency.
    delays = delay_from_dm(freq, dm, t0)
    time_shift_inds = np.argmin(abs(time - delays[:,None]), -1)
    time_bins_offset = nprofilebins // 3
    time_shift_inds -= time_bins_offset
    # Get aligned profile.
    data_aligned = np.empty((nfreq, npol, nprofilebins))
    for ii in range(nfreq):
        data_aligned[ii] = data[ii,:,time_shift_inds[ii]
                                :time_shift_inds[ii] + nprofilebins]

    # Spectral index scaling.
    data_aligned /= (freq[:,None,None] / 800.)**alpha
    # Best weights are S/N**2.  In this space, signal is flat, noise needs two
    # factors of spectral index.
    weights *= (freq[:,None,None] / 800.)**(2*alpha)

    # Derotate polarization.
    pol_angle = 2 * RM * ((3e8 / freq / 1e6)**2 - (3e8 / f_ref / 1e6)**2) + phi
    tmp_Q = data_aligned[:,1,:] * np.cos(pol_angle)[:,None]
    tmp_Q += data_aligned[:,2,:] * np.sin(pol_angle)[:,None]
    tmp_U = -data_aligned[:,1,:] * np.sin(pol_angle)[:,None]
    tmp_U += data_aligned[:,2,:] * np.cos(pol_angle)[:,None]
    data_aligned[:,1,:] = tmp_Q
    data_aligned[:,2,:] = tmp_U

    # Collapse frequecy axis.
    data[mask_chans] = 0
    profiles = np.sum(data_aligned * weights, 0) / np.sum(weights, 0)
    return profiles




# Ephemeris
# =========

def arrival_coords():
    """Getting arrival time and ephemeris information.
    """
    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered')
    #time = time[TSL]
    #ra = ra[TSL]
    #dec = dec[TSL]
    #az = az[TSL]
    #el = el[TSL]

    time_ind_900 = np.argmin(np.abs(time - delay_from_dm(900., FIT_PARS[1],
        FIT_PARS[0])))
    time_ind_700 = np.argmin(np.abs(time - delay_from_dm(700., FIT_PARS[1],
        FIT_PARS[0])))

    print time_ind_900, time_ind_700

    print "F=%f:" % 900., radec2str(ra[time_ind_900], dec[time_ind_900])
    print "F=%f:" % 700., radec2str(ra[time_ind_700], dec[time_ind_700])


def radec2str(ra, dec):
    rah = int(ra / 15)
    ram = int((ra / 15 - rah) * 60)
    ras = int(((ra / 15 - rah) * 60 - ram) * 60)

    decneg = dec < 0
    dec = abs(dec)
    decd = int(dec)
    decm = int((dec - decd) * 60)
    decs = int(((dec - decd) * 60 - decm) * 60)

    ra = "%2dh%dm%ds" % (rah, ram, ras)
    dec = "%2dd%dm%ds" % (decd, decm, decs)
    if decneg:
        dec = "-" + dec
    return ra, dec




# IO routines for local data format
# =================================

def export_data(fname, data, mask_chans):
    data[mask_chans] = float('nan')
    np.save(fname, data)
    data[mask_chans] = 0.


def import_data(fname):
    data = np.load(fname)
    mask_chans = np.any(np.isnan(data[:,:,0]), 1)
    data[mask_chans] = 0.
    return data, mask_chans


def import_all(case):
    data, mask = import_data(OUT_ROOT + '/%s.npy' % case)
    time = np.load(OUT_ROOT + '/time.npy')
    freq = np.load(OUT_ROOT + '/freq.npy').astype(float)
    ra = np.load(OUT_ROOT + '/ra.npy')
    dec = np.load(OUT_ROOT + '/dec.npy')
    az = np.load(OUT_ROOT + '/az.npy')
    el = np.load(OUT_ROOT + '/el.npy')

    if case == 'filtered_short':
        time = time[TSL]
        ra = ra[TSL]
        dec = dec[TSL]
        az = az[TSL]
        el = el[TSL]

    return data, mask, time, freq, ra, dec, az, el


# General utilities
# =================

def rebin_freq(spectra, mask_chans, rebin_fact):
    nfreq = spectra.shape[0]
    naxis1 = spectra.shape[1]
    spectra = spectra.copy()
    spectra[mask_chans] = 0
    spectra_rebin = np.reshape(spectra,
            (nfreq // rebin_fact, rebin_fact, naxis1))
    spectra_rebin = np.sum(spectra_rebin, 1)
    norm = np.reshape(np.logical_not(mask_chans),
            (nfreq // rebin_fact, rebin_fact))
    norm = np.sum(norm, 1).astype(float)
    bad = norm == 0
    norm[bad] = 1
    norm = 1./norm
    norm[bad] = 0
    spectra_rebin *= norm[:,None]
    return spectra_rebin


def norm_to_plaw(spec, freq, mask_chans):

    weights = 1. - mask_chans

    residuals = lambda x: (spec - general_power_law(freq, x)) * weights

    pars, cov, info, msg, ierr = optimize.leastsq(
            residuals,
            [1., -2, 0],
            epsfcn=0.001,
            full_output=True,
            xtol=0.0001
            )
    #print "Fit status:", msg, ierr, pars
    fit = general_power_law(freq, pars)

    out = (spec - fit) / fit
    out[mask_chans == 1] = 0.
    return out, pars


def power_law(freq, amp, alpha):
    return amp * (freq / 750.)**-alpha


def general_power_law(freq, coeff):
    l_freq_norm = np.log(freq / 750.)

    lfn = 1.
    exponent = 0
    for c in coeff:
        exponent += c * lfn
        lfn *= l_freq_norm
    return np.exp(exponent)


def delay_from_dm(freq, dm, t0):
    """Relative to 900 MHz."""
    delay = 0.004148808 * dm * ((freq / 1000.)**-2 - (900. / 1000.)**-2) + t0
    return delay


class RangeSelector(object):

    def __init__(self, time):
        delta = np.median(np.diff(time))
        if not np.allclose(np.diff(time), delta):
            raise ValueError("Not uniformly sampled")
        self._delta = delta
        self._s0 = time[0]

    def __call__(self, start, stop):
        n = int(math.ceil((stop - start) / self._delta))
        start_ind = (start - self._s0) / self._delta
        start_ind = int(round(start_ind))
        stop_ind = start_ind + n
        return start_ind, stop_ind


def smooth_flag_spectra(freq, spec, mask, n, thres=5.):
    """Spec is 2D (second axis is polarization). Other arrays are 1D.
    
    Mask modified in place.
    
    """

    #mask = mask.copy()
    out = np.empty_like(spec)

    for jj in range(3):
        for ii in range(spec.shape[1]):
            unmask = np.logical_not(mask)
            this_s = spec[unmask,ii]
            this_f = freq[unmask]

            #f_scale = lambda f: (f - 800.) / 101.
            #p = np.polyfit(
            #        f_scale(this_f),
            #        this_s,
            #        n,
            #        )
            #smoothed = lambda f: np.poly1d(p)(f_scale(f))
            smoothed = Legendre.fit(this_f, this_s, n, domain=[699., 901.])

            smoothed = smoothed(freq)
            std =  np.std(spec[unmask, ii] - smoothed[unmask])
            this_thres = thres * std
            mask[abs(spec[:,ii] - smoothed) > this_thres] = True
            out[:,ii] = smoothed
    return out


def rotate_to_XY(data):
    """ (I Q U V) -> (XX, Re(XY), Im(XY), YY).
    In place.
    """

    nfreq = data.shape[0]
    for ii in range(nfreq):
        this_data = data[ii]
        XX = this_data[0] - this_data[1]
        YY = this_data[0] + this_data[1]
        XY = this_data[2].copy()
        YX = this_data[3].copy()
        this_data[0] = XX
        this_data[1] = XY
        this_data[2] = YX
        this_data[3] = YY


def rotate_to_IQUV(data):
    """(XX, Re(XY), Im(XY), YY) -> (I Q U V).
    In place.
    """

    nfreq = data.shape[0]
    for ii in range(nfreq):
        this_data = data[ii]
        I = (this_data[0] + this_data[3]) / 2
        Q = (this_data[3] - this_data[0]) / 2
        U = this_data[1].copy()
        V = this_data[2].copy()
        this_data[0] = I
        this_data[1] = Q
        this_data[2] = U
        this_data[3] = V


# Pulse profile model
# ===================

def windowed_pulse(time, freq, delta_t, delta_f, dm, t0, width, scatter):
    up_sample_factor_time = 16
    up_sample_factor_freq = 8
    time = upsample(time, delta_t, up_sample_factor_time)
    freq = upsample(freq, delta_f, up_sample_factor_freq)
    delay = delay_from_dm(freq, dm, t0)

    pulse = 0.
    for ii in range(up_sample_factor_freq):
        pulse += norm_pulse(time - delay[ii], width, scatter)
    pulse = downsample(pulse, up_sample_factor_time)
    pulse /= up_sample_factor_freq
    return pulse


def norm_pulse(time, width, scatter):
    if scatter < 0.01 * width:
        norm = 1. / (width * np.sqrt(2. * np.pi))
        arg = - time**2 / 2 / width**2
        out = np.exp(arg) * norm
    else:
        width = float(width)
        scatter = float(scatter)
        norm = 1. / 2 / scatter
        norm *= np.exp(width**2 / 2 / scatter**2)
        erfcarg = (width / np.sqrt(2.) / scatter) - time / (np.sqrt(2) * width)
        out = norm * np.exp(-time / scatter) * special.erfc(erfcarg)
    return out


def upsample(time, delta_t, factor):
    diff = (np.arange(factor, dtype=float) + 0.5) / factor - 0.5
    diff *= delta_t
    out = time[:,None] + diff[None,:]
    return out.flat[:]


def downsample(data, factor):
    n = data.shape[-1]
    n_d = n // factor
    data = np.reshape(data, data.shape[:-1] + (n_d, factor))
    return np.mean(data, -1)


# ========================

if __name__ == "__main__":
    main()
