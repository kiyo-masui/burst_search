from os import path

import numpy as np
from numpy.polynomial import Legendre, Chebyshev
from scipy import signal, interpolate, special, optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfits

from burst_search import preprocess, dedisperse


BURST = True

if BURST:
    #DATAROOT = "/Users/kiyo/data/raw_guppi/AGBT10B_036/67_20110523"
    DATAROOT = "/scratch2/p/pen/hsiuhsil/gbt_data/FRB110523"
    FILENAME = "guppi_55704_wigglez22hrst_0258_0001.fits"
    SRCFILES = [
        "guppi_55704_3C48_0006_0001.fits",
        "guppi_55704_3C48_0007_0001.fits",
        "guppi_55704_3C48_0008_0001.fits",
        "guppi_55704_3C48_0009_0001.fits",
        ]
    OUT_ROOT = '/scratch2/p/pen/hsiuhsil/gbt_data/burst_data'

    # Hard code the phase of the noise cal, since I'm to lazy to write an algorithm
    # to find it.
    DATA_CAL_PHASE = 1
    SRC_CAL_PHASES = [40, 42, 28, 47]
    TSL = np.s_[23000:26500]

else:
    DATAROOT = '/Users/kiyo/data/raw_guppi/AGBT14B_339'
    OUT_ROOT = 'psr_data'

    #FILENAME = "guppi_57132_B0329+54_0020_0001.fits"
    #DATA_CAL_PHASE = 4
    FILENAME = "guppi_57132_B1929+10_0010_0001.fits"
    DATA_CAL_PHASE = 0
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


# Subset of data time axis known to contain event.

# Used in analysis_IM.
#BEAM_DATA_FWHM = [
#        0.316148488246, 0.306805630985, 0.293729620792,
#        0.281176247549, 0.270856788455, 0.26745856078,
#        0.258910010848, 0.249188429031,
#        ]
#BEAM_DATA_FREQ = [695, 725, 755, 785, 815, 845, 875, 905]

# From my narrow spectral bin fits, which looks much better.
BEAM_DATA_FWHM = [0.307, 0.245]
BEAM_DATA_FREQ = [699.9, 900.1]

def main():
    #Preprocessing
    #get_raw()
    #get_src_means()
    #produce_calibrated()
    #filter()

    #plot()

    # Fitting.
    #fit_basic()
    #fit_beam()

    # Spectral plots.
    #get_spec()
    joy_division()
    #scinil_correation()



# Shift some paraemters for well behaved fitting.
T_OFF = 54290.1
DM_OFF = 620


FIT_PARS = [
            54290.138354166804,         # t0
            623.52643836607535,         # DM
            0.0042870706698844834,      # A_800
            8.7807947464191134,         # alpha
            0.0011077538026552493,      # width
            0.003256312973327511,       # scatter
            ]

FIT_PARS = [54290.138511393445, 623.1990728439398, 0.0039873229381549909, 7.8236853697290707, 0.00078134758755964498, 0.0016443840956046852]
FIT_PARS = [54290.138577280726, 623.17690317652421, 0.0039794152509098719, 7.8052288059807537, 0.00079122660217092642, 0.0016091172637676347]

"""
Jon's fits:
intensity :
fwhm     scat    alpha   fluence  t0  DM
mean:
0.0019772   0.0016577     -7.6281      4.0354      24.143      623.13
std: 0.00013796  0.00013016     0.36458     0.15222  0.00011902    0.052191
pol:
fwhm     scat       alpha   fluence  t0   DM   RM   phi
0.0017527  0.00098258     -6.0577       1.863      24.143      623.01     -373.91      3.2225
0.00023722  0.00025428     0.65712     0.12691  0.00016493    0.072635      2.5902    0.047439
"""


def joy_division():
    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered')

    data = data[:,:,TSL]
    time = time[TSL]

    t0 = FIT_PARS[0]
    dm = FIT_PARS[1]

    nfreq = len(freq)
    nprofilebins = 80
    # Calculate arrival time at each frequency.
    delays = delay_from_dm(freq, dm, t0)
    time_shift_inds = np.argmin(abs(time - delays[:,None]), -1)
    time_bins_offset = nprofilebins // 3
    time_shift_inds -= time_bins_offset
    # Get aligned profile.
    profile = np.empty((nfreq, nprofilebins))
    for ii in range(nfreq):
        profile[ii] = data[ii,0,time_shift_inds[ii]
                                :time_shift_inds[ii] + nprofilebins]

    # Rebin frequency axis.
    rebin_fact = 64
    profile_rebin = rebin_freq(profile, mask_chans, rebin_fact)

    # Plot it Joy Division style.
    delta_t = np.median(np.diff(time))
    time_rel = (np.arange(nprofilebins) - time_bins_offset) * delta_t
    factor = 1.
    offset = 0.
    plt.figure()
    freq_rebin_index = [41, 47, 54, 60]
    for ii in freq_rebin_index:
        this_profile = profile_rebin[profile_rebin.shape[0] - ii - 1]
        this_profile = this_profile / np.max(this_profile)
        plt.plot(time_rel, factor * (this_profile + offset), 'k')
        offset += .5

    plt.show()



def rebin_freq(spectra, mask_chans, rebin_fact):
    nfreq = spectra.shape[0]
    naxis1 = spectra.shape[1]
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


def scinil_correation():
    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered')

    spec = integrated_pulse_spectrum(data[:,:,TSL], freq, time[TSL], FIT_PARS,
            True)
    
    # Normalize to powerlaw model.
    amp_800 = FIT_PARS[2]
    alpha = FIT_PARS[3]
    amp = amp_800 * (freq / 800.)**-alpha

    spec_norm = spec / amp[:,None] - 1
    
    nfreq = len(freq)
    corr = np.zeros(nfreq)
    norm = np.zeros(nfreq)
    not_mask = np.logical_not(mask_chans)
    for ii in range(nfreq):
        if mask_chans[ii]:
            continue
        corr[:nfreq - ii] += spec_norm[ii:,0] * spec_norm[ii,0]
        norm[:nfreq - ii] += not_mask[ii:]
    corr = corr[:-100] / norm[:-100]
    
    delta_f = -np.median(np.diff(freq))
    plt.figure()
    plt.plot(np.arange(len(corr)) * delta_f, corr, '.')
    plt.xlabel(r'$\Delta\nu$ (MHz)')
    plt.ylabel(r'$\langle \delta_T(\nu) \delta_T(\nu + \Delta\nu) \rangle$')

    plt.figure()
    plt.plot(freq[16::32], rebin_freq(spec_norm, mask_chans, 32)[:,0], '.')
    plt.xlabel(r'$\nu$ (MHz)')
    plt.ylabel(r'$\delta_T$')

    plt.show()



def get_spec():
    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered')

    spec = integrated_pulse_spectrum(data[:,:,TSL], freq, time[TSL], FIT_PARS,
            True)

    spec_out = spec.copy()
    spec_out[mask_chans] = float('nan')
    np.save(OUT_ROOT + "/polarized_spectra.npy", spec_out)
    #plt.figure()
    #plt.plot(freq, spec[:,0])
    #plt.plot(freq, spec[:,1])
    #plt.plot(freq, spec[:,2])
    #plt.plot(freq, spec[:,3])

    rebin_fact = 32
    spec_rebin = rebin_freq(spec, mask_chans, rebin_fact)

    f = freq[rebin_fact//2::rebin_fact]

    #plt.figure()
    #plt.plot(f, spec_rebin[:,0])
    #plt.plot(f, spec_rebin[:,1])
    #plt.plot(f, spec_rebin[:,2])
    #plt.plot(f, spec_rebin[:,3])

    
    # Model plots
    amp = 0.0040354
    alpha = 7.628
    RM = 187.0  # Jon's fit.
    phi = 5.1775526
    amp_pol = 0.001863
    alpha_pol = 6.057

    fact = 1000

    fig = plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(f, fact * spec_rebin[:,0], '.k')
    plt.plot(freq, fact * amp * (freq / 800.)**-alpha, 'k')
    plt.ylim(-0.003 * fact, 0.015 * fact)
    plt.ylabel('I')

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(f, fact * spec_rebin[:,1], '.k')
    plt.plot(freq, fact * amp_pol * (freq / 800.)**-alpha_pol
             * np.cos(2 * RM * (3e8 / freq / 1e6)**2 + phi), 'k')
    plt.ylim(-0.0055 * fact, 0.0055 * fact)
    plt.ylabel('Q')

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(f, fact * spec_rebin[:,2], '.k')
    plt.plot(freq, fact * amp_pol * (freq / 800.)**-alpha_pol
             * np.sin(2 * RM * (3e8 / freq / 1e6)**2 + phi), 'k')
    plt.ylim(-0.0055 * fact, 0.0055 * fact)
    plt.ylabel('U')

    #ax4 = plt.subplot(414, sharex=ax1)
    #plt.plot(f, fact * spec_rebin[:,3], '.k')
    #plt.ylim(-0.01 * fact, 0.01 * fact)
    #plt.ylabel('V')

    xticklabels = (ax1.get_xticklabels() + ax2.get_xticklabels()
    #        + ax3.get_xticklabels()
            )
    plt.setp(xticklabels, visible=False)
    plt.subplots_adjust(hspace=0.0001)

    plt.xlabel('frequency (MHz)')
    
    fig.text(0.03, 0.5, 'fluence (K ms)',
            ha='center', va='center', rotation='vertical')
    
    # Hack the big y lable.
    #ax = fig.add_subplot(111)
    #ax.spines['top'].set_color('none')
    #ax.spines['bottom'].set_color('none')
    #ax.spines['left'].set_color('none')
    #ax.spines['right'].set_color('none')
    #ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
    #        right='off')
    #ax.set_ylabel('fluence (K ms)')


    plt.show()



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

    pars0 = [0.038, 3., 0.004, 7., 0.001, 0.001]
    #pars0 = [0.0385, 3.4, 0.004, 7., 0.001, 0.003]
    # Jon's:
    #pars0 = [0.038, 3.13, 0.004035, 7.16,(0.00198/2.35),(0.00165)]
    #pars0 = wrap_basic_pars(FIT_PARS)

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
                )
            * weights[:,None]).flat[:]
    chi2 = lambda p: np.sum(residuals(p)**2)
    
    #pars = wrap_basic_pars(FIT_PARS)
    #p = list(pars)
    #delta = np.arange(-0.0001, 0.0001, 0.00003)
    #c2 = []
    #for d in delta:
    #    p[4] = pars[4] + d
    #    c2.append(chi2(p))
    #plt.figure()
    #plt.plot(delta, c2)
    #plt.show()

    pars, cov, info, msg, ierr = optimize.leastsq(
            residuals,
            pars0,
            epsfcn=0.0001,
            full_output=True,
            ftol=0.00001 / nfitdata,
            )
    print "Fit status:", msg, ierr
    real_pars = unwrap_basic_pars(pars)

    #res = optimize.minimize(
    #        chi2,
    #        pars0,
    #        )
    #print res.x, res.success


    Chi2 = np.sum(residuals(pars)**2)
    delta_Chi2 = np.sum(residuals([0,0,0,1,.001,.001])**2) - Chi2
    red_Chi2 = Chi2 / nfitdata
    print "Delta chi-squared:\n", delta_Chi2
    print "Reduced chi-squared:\n", red_Chi2
    print "Parameters:\n", real_pars
    errs = np.sqrt(cov.flat[::npars + 1])
    corr = cov / errs / errs[:,None]
    # Transform errors to sqrt space.
    # XXX What does that do to the covariance.
    # Also this is wrong, it is the fractional value that gets divided by 2,
    # not the absolute.
    # XXX
    #errs[4] = 2 * errs[4] / pars[4] * real_pars[4]
    #errs[5] = 2 * errs[5] / pars[5] * real_pars[5]
    print "Errors:\n", errs
    print "Correlations:\n", corr



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
    up_sample_factor_freq = 4
    freq_up = upsample(freq, delta_f, up_sample_factor_freq)
    freq_up.shape = (len(freq), up_sample_factor_freq)
    delay_up = delay_from_dm(freq_up, dm, t0)
    delay = delay_up[:,up_sample_factor_freq//2]  # not for precise things.

    #plt.plot(freq, amp)
    if not beam is None:
        b = pars[6]
        s0 = pars[7]
        scan_loc_f = interpolate.interp1d(time, scan_loc)(delay)
        amp *= np.exp(-0.5 * (b**2 + (scan_loc_f - s0)**2) / beam**2)
    #plt.plot(freq, amp)
    #plt.show()
    
    # For testing.
    p_sums = np.empty_like(freq)

    delta_t = np.median(np.diff(time))
    up_sample_factor_time = 16
    residuals = data_I.astype(float)
    for ii in range(len(freq)):
        window = np.sqrt(scatter[ii]**2 + width**2) * 20
        near_time_inds = np.logical_and(time > delay[ii] - window,
                                        time < delay[ii] + window)
        near_times = time[near_time_inds]
        times_up = upsample(near_times, delta_t, up_sample_factor_time)
        pulse = 0.
        for jj in range(up_sample_factor_freq):
            pulse += norm_pulse(
                    times_up - delay_up[ii, jj],
                    width,
                    scatter[ii],
                    )
        pulse = downsample(pulse, up_sample_factor_time)
        p_sums[ii] = np.sum(pulse) / up_sample_factor_freq * delta_t
        pulse *= amp[ii] / up_sample_factor_freq
        residuals[ii, near_time_inds] -= pulse
    #plt.plot(freq, p_sums)
    #plt.show()
    return residuals


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


def wrap_beam_pars(pars):
    pars = wrap_basic_pars(pars)
    pars = list(pars)
    #pars = pars[:3] + pars[6:]
    pars = pars[:3] + pars[4:]
    return pars


def unwrap_beam_pars(pars):
    pars = list(pars)
    alpha = 2.
    width = np.sqrt(0.0011)
    scatter = np.sqrt(0.00325)
    #pars = pars[:3] + [alpha, width, scatter] + pars[3:]
    pars = pars[:3] + [alpha,] + pars[3:]
    pars = unwrap_basic_pars(pars)
    return pars


def fit_beam():
    
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

    pars0 = [0.038, 3., 0.004, 7., 0.05, 0.04]
    pars0 = wrap_beam_pars(unwrap_basic_pars(pars0) + [0.1, 0.1])
    #pars0 = wrap_beam_pars(FIT_PARS + [0.1, 0.3])

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
                unwrap_beam_pars(p),
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
    real_pars = unwrap_beam_pars(pars)

    Chi2 = np.sum(residuals(pars)**2)
    red_Chi2 = Chi2 / nfitdata
    print "Reduced chi-squared:\n", red_Chi2
    print "Parameters:\n", real_pars
    errs = np.sqrt(cov.flat[::npars + 1])
    corr = cov / errs / errs[:,None]
    # Transform errors to sqrt space.
    # XXX What does that do to the covariance.
    errs[2] /= 2
    errs[3] /= 2
    print "Errors:\n", errs
    print "Correlations:\n", corr


def integrated_pulse_spectrum(data, freq, time, pars, matched=True):
    # XXX Matched filter not using the same kernal model as fits: time
    # oversampling.
    t0 = pars[0]
    dm = pars[1]
    amp_800 = pars[2]
    alpha = pars[3]
    width = pars[4]
    scatter_800 = pars[5]

    scatter = scatter_800 * (freq / 800.)**-4
    delay = delay_from_dm(freq, dm, t0)

    delta_t = np.mean(np.diff(time))

    out = np.empty(data.shape[:-1], dtype=float)

    for ii in range(len(freq)):
        if matched:
            nw = 4.
        else:
            nw = 3.
        window_low = nw * width
        window_high = nw * width + nw**2 / 2 * scatter[ii]
        near_time_inds = np.logical_and(time > delay[ii] - window_low,
                                        time < delay[ii] + window_high)
        near_times = time[near_time_inds]
        if matched:
            pulse = norm_pulse(
                    near_times - delay[ii],
                    width,
                    scatter[ii],
                    )
            # Need to renormalize due to discretization effects. This is a
            # small adjustment.
            pulse /= np.sum(pulse) * delta_t
            # Matched filter normalization.
            pulse /= np.sum(pulse**2) * delta_t
            # Compensated? Probably don't need to if noise is white.
            #pulse -= np.mean(pulse)
        else:
            pulse = np.ones_like(near_times)
        for jj in range(data.shape[1]):
            out[ii, jj] = np.sum(data[ii,jj,near_time_inds] * pulse)
    out *= delta_t
    return out


def unwrap_basic_pars(pars):
    """Un wraps reparameterization for fitting."""
    t0 = pars[0] + T_OFF
    dm = pars[1] + DM_OFF
    amp_800 = pars[2]     # K s.
    alpha = pars[3]
    width = pars[4]
    scatter_800 = pars[5]
    #width = pars[4]**2
    #scatter_800 = pars[5]**2
    return [t0, dm, amp_800, alpha, width, scatter_800] + list(pars[6:])


def wrap_basic_pars(pars):
    out = list(pars)
    pars[0] -= T_OFF
    pars[1] -= DM_OFF
    pars[4] = (pars[4])
    pars[5] = (pars[5])
    #pars[4] = np.sqrt(pars[4])
    #pars[5] = np.sqrt(pars[5])
    return pars


def delay_from_dm(freq, dm, t0):
    delay = 0.00415 * dm * ((freq / 1000.)**-2 - (freq[0] / 1000.)**-2) + t0
    return delay


def norm_pulse(time, width, scatter):
    width = float(width)
    scatter = float(scatter)
    norm = 1. / 2 / scatter
    norm *= np.exp(width**2 / 2 / scatter**2)
    erfcarg = (width / np.sqrt(2.) / scatter) - time / (np.sqrt(2) * width)
    out = norm * np.exp(-time / scatter) * special.erfc(erfcarg)
    return out


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


def get_src_means():

    src_means = []
    for ii, filename in enumerate(SRCFILES):
        srchdulist = pyfits.open(path.join(DATAROOT, filename), 'readonly')
        src_data = read_data(srchdulist)
        preprocess_data(src_data, srchdulist, SRC_CAL_PHASES[ii])
        src_means.append(np.mean(src_data, -1))
        del src_data
        srchdulist.close()
    np.save(OUT_ROOT + '/src_means.npy', src_means)
    return src_means



def produce_calibrated():
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


def smooth_flag_spectra(freq, spec, mask, n, thres=5.):
    """Spec is 2D (second axis is polarization). Other arrays are 1D.
    
    Mask modifed in place.
    
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


def get_raw():
    hdulist = pyfits.open(path.join(DATAROOT, FILENAME), 'readonly')
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

    # Now read the main data.
    data = read_data(hdulist)
    
    time = np.arange(data.shape[2]) * delta_t + time0
    subint_time = time0 + hdulist[1].data['OFFS_SUB']
    ra = sample_subint(subint_time, hdulist[1].data['RA_SUB'], time)
    dec = sample_subint(subint_time, hdulist[1].data['DEC_SUB'], time)
    az = sample_subint(subint_time, hdulist[1].data['TEL_AZ'], time)
    el = 90. - sample_subint(subint_time, hdulist[1].data['TEL_ZEN'], time)

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

    diff = np.diff(sub_var) / np.diff(sub_time)
    if BURST:
        if not np.allclose(diff, diff[0], rtol=2.):
            raise ValueError('Not linear')
    rate = np.mean(diff)
    start_ind = np.argmin(np.abs(time - sub_time[0]))
    return (time - time[start_ind]) * rate + sub_var[0]


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

    return data, mask, time, freq, ra, dec, az, el



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
    # XXX
    if True:
        plt.figure()
        plt.plot(p[0] + p[3])
        plt.plot(p[0] + p[3], '.')
        plt.show()


def read_data(hdulist):
    start_record = 0
    # XXX
    if BURST:
        end_record = None
    else:
        # These scans are long and I don't have disk space for all of it.
        end_record = 25

    nrecords = len(hdulist[1].data)
    if end_record is None or end_record > nrecords:
        end_record = nrecords
    nrecords_read = end_record - start_record
    ntime_record, npol, nfreq, one = hdulist[1].data[0]["DATA"].shape

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

    return out_data


def rotate_to_XY(data):
    """ (I Q U V) -> (XX, Re(XY), Im(XY), YY).
    Inplace.
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
    Inplace.
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


def calibrator_spectrum(src, freq):
    # XXX Seem to missing a factor of frequncy in here, since my calibrations
    # seem to have a bit of slope to them.
    gain = 2.    # K/Jy
    # Data from arXiv:1211.1300v1.
    if src == '3C48':
        #coeff = [1.3324, -0.7690, -0.1950, 0.059]
        coeff = [1.3332, -0.7665, -0.1980, 0.064]
        #coeff = [2.345, 0.071, -0.138, 0.]
    elif src == '3C295':
        coeff = [1.4866, -0.7871, -0.3440, 0.0749]

    l_freq_ghz = np.log10(freq / 1e3)    # Convert to GHz.
    #l_freq_ghz = np.log10(freq)
    #poly = 1
    spec = 0
    for ii, A in enumerate(coeff):
        spec += A * l_freq_ghz**ii
        #spec += A * poly
        #poly *= l_freq_ghz
    spec = 10.**(spec) * gain
    #plt.figure()
    #plt.plot(freq, spec)
    # Tabitha's 3C48.
    #c_spec = 25.15445092*pow((750.0/freq),0.75578842)*(2.28315426-0.000484307905*freq)

    return spec


def plot():
    data, mask_chans, time, freq, ra, dec, az, el = import_all('filtered')
    delta_t = np.mean(np.diff(time))

    plt.figure()
    d = data[:,0,TSL]
    s = np.std(d)
    plt.imshow(d,
            vmin=-1*s,
            vmax=3*s,
            extent=[0, d.shape[1] * delta_t, 700., 900.],
            aspect='auto',
            )

    plt.show()


def plot_all():
    
    
    I = (data[:,0,t_sl] + data[:,3,t_sl]) / 2
    transformer = dedisperse.DMTransform(delta_t, nfreq, freq0, delta_f,
                                         1000.)
    I_all = transformer(I, None)
    I_dm = I_all.dm_data
    std = np.std(I)
    plt.imshow(I,
            vmin=-1*std,
            vmax=3*std,
            extent=[0, I.shape[1] * delta_t, 700., 900.],
            aspect='auto',
            )
    plt.colorbar()


    plt.figure()
    std = np.std(I_dm)
    plt.imshow(I_dm[::-1],
            vmin=-1*std,
            vmax=3*std,
            extent=[0, I_dm.shape[1] * delta_t, I_all.dm0,
                    I_all.dm0 + I_dm.shape[0] * I_all.delta_dm],
            aspect='auto',
            )
    plt.colorbar()
    

    plt.figure()
    dec_factor = 32
    I_dec = np.mean(np.reshape(I, (I.shape[0] // dec_factor, dec_factor, I.shape[1])), 1)
    std = np.std(I_dec)
    plt.imshow(I_dec,
            vmin=-1*std,
            vmax=3*std,
            extent=[0, I_dec.shape[1] * delta_t, 700., 900.],
            aspect='auto',
            )
    plt.colorbar()


    U = data[:,1,t_sl]
    U_all = transformer(U, None)
    U_dm = U_all.dm_data
    plt.figure()
    std = np.std(U_dm)
    plt.imshow(U_dm[::-1],
            vmin=-1*std,
            vmax=3*std,
            extent=[0, U_dm.shape[1] * delta_t, U_all.dm0,
                    U_all.dm0 + U_dm.shape[0] * U_all.delta_dm],
            aspect='auto',
            )


    V = data[:,2,t_sl]
    V_all = transformer(V, None)
    V_dm = V_all.dm_data
    plt.figure()
    std = np.std(V_dm)
    plt.imshow(V_dm[::-1],
            vmin=-1*std,
            vmax=3*std,
            extent=[0, V_dm.shape[1] * delta_t, V_all.dm0,
                    V_all.dm0 + V_dm.shape[0] * V_all.delta_dm],
            aspect='auto',
            )

    Q = (data[:,3,t_sl] - data[:,0,t_sl]) / 2
    Q_all = transformer(Q, None)
    Q_dm = Q_all.dm_data
    plt.figure()
    std = np.std(Q_dm)
    plt.imshow(Q_dm[::-1],
            vmin=-1*std,
            vmax=3*std,
            extent=[0, Q_dm.shape[1] * delta_t, Q_all.dm0,
                    Q_all.dm0 + Q_dm.shape[0] * Q_all.delta_dm],
            aspect='auto',
            )


    dm_ind = int(round((623. - I_all.dm0) / I_all.delta_dm))
    plt.figure()
    plt.plot(time[t_sl], I_dm[dm_ind])
    plt.plot(time[t_sl], Q_dm[dm_ind])
    plt.plot(time[t_sl], U_dm[dm_ind])
    plt.plot(time[t_sl], V_dm[dm_ind])


    plt.figure()
    dec_factor = 1
    U_dec = np.mean(np.reshape(U, (U.shape[0] // dec_factor, dec_factor, U.shape[1])), 1)
    std = np.std(U_dec)
    plt.imshow(U_dec,
            vmin=-2*std,
            vmax=2*std,
            extent=[0, U_dec.shape[1] * delta_t, 700., 900.],
            aspect='auto',
            )
    plt.colorbar()



if __name__ == "__main__":
    main()

