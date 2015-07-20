from os import path

import sys

import numpy as np

import pyfits

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

import math

from . import preprocess

def parameters_from_header(hdulist):
    """Get data acqusition parameters for psrfits file header.
    Returns
    -------
    parameters : dict
    """

    parameters = {}

    #print repr(hdulist[0].header)
    #print
    #print repr(hdulist[1].header)
    mheader = hdulist[0].header
    dheader = hdulist[1].header

    if mheader['CAL_FREQ']:
        cal_period = 1. / mheader['CAL_FREQ']
        parameters['cal_period_samples'] = int(round(cal_period / dheader['TBIN']))
    else:
        parameters['cal_period_samples'] = 0
    parameters['delta_t'] = dheader['TBIN']
    parameters['nfreq'] = dheader['NCHAN']
    parameters['freq0'] = mheader['OBSFREQ'] - mheader['OBSBW'] / 2.
    parameters['delta_f'] = dheader['CHAN_BW']

    record0 = hdulist[1].data[0]
    #print record0
    #data0 = record0["DATA"]
    #freq = record0["DAT_FREQ"]
    ntime_record, npol, nfreq, one = eval(dheader["TDIM17"])[::-1]
    parameters['npol'] = npol

    parameters['ntime_record'] = ntime_record
    parameters['dtype'] = np.uint8

    return parameters


def calibrator_spectrum(src, freq):
    gain = 2    # K/Jy
    # Data from arXiv:1211.1300v1.
    if src == '3C48':
        coeff = [1.3324, -0.7690, -0.1950, 0.059]
    elif src == '3C295':
        coeff = [1.4866, -0.7871, -0.3440, 0.0749]
    elif src == '3C286':
	coeff = [1.2515, -0.4601, -0.1715, -0.0336]
    elif src == '3C147':
	coeff = [1.4616, -0.7187, -0.2424, -0.079]

    l_freq_ghz = np.log10(freq / 1e9)    # Convert to GHz.
    poly = 1
    spec = 0
    for A in coeff:
        spec += A * poly
        poly *= l_freq_ghz
    spec = 10.**spec
    #plt.figure()
    #plt.plot(freq, spec)
    return spec

def read_records(hdulist, start_record=0, end_record=None):
    """Read and format records from GUPPI PSRFITS file."""

    nrecords = len(hdulist[1].data)
    if end_record is None or end_record > nrecords:
        end_record = nrecords
    nrecords_read = end_record - start_record
    ntime_record, npol, nfreq, one = hdulist[1].data[0]["DATA"].shape

    out_data = np.empty((nfreq, nrecords_read, ntime_record), dtype=np.float32)
    for ii in xrange(nrecords_read):
        # Read the record.
        record = hdulist[1].data[start_record + ii]["DATA"]
        # Interpret as unsigned int (for Stokes I only).
        record = record.view(dtype=np.uint8)
        # Select stokes I and copy.
        out_data[:,ii,:] = np.transpose(record[:,0,:,0])
    out_data.shape = (nfreq, nrecords_read * ntime_record)

    return out_data

def get_mean_spectrum(filename):
    
    hdulist = pyfits.open(filename, 'readonly')

    parameters = parameters_from_header(hdulist)
    nrecords = len(hdulist[1].data)
    data = read_records(hdulist, 0, nrecords)
    hdulist.close()

    P_over_P_cal = preprocess.p_over_p_cal(data, 
                                 parameters['cal_period_samples'])

    return P_over_P_cal


def T_cal(filename1, filename2):
     hdulist1 = pyfits.open(filename1, 'readonly')
     src1 = hdulist1[0].header['SRC_NAME']
     RA1 = hdulist1[1].data['RA_SUB'][0]
     DEC1 = hdulist1[1].data["DEC_SUB"][0]

     hdulist2 = pyfits.open(filename2, 'readonly')
     src2 =  hdulist2[0].header['SRC_NAME']
     RA2 = hdulist2[1].data["RA_SUB"][0]
     DEC2 = hdulist2[1].data["DEC_SUB"][0]
     parameters = parameters_from_header(hdulist1)
     hdulist1.close()
     hdulist2.close()

     assert(src1 == src2)
     source = { '3C147': {'RA': "85.6506458", 'DEC': "49.8520222"},
                '3C295': {'RA': "212.8358333", 'DEC': "52.2058333"},
                '3C286': {'RA': "202.7845333", 'DEC': "30.5091556"} }

     distance1 = math.sqrt((float(RA1)-float(source[src1]['RA']))**2 + (float(DEC1)-float(source[src1]['DEC']))**2)
     distance2 = math.sqrt((float(RA2)-float(source[src2]['RA']))**2 + (float(DEC2)-float(source[src2]['DEC']))**2)

     if distance1 < distance2:
        filename_on = filename1
        filename_off = filename2
     else:
        filename_off = filename1
        filename_on = filename2

     freq = np.arange(parameters['nfreq']) * parameters['delta_f'] + parameters['freq0']
     freq_mhz = (freq * 1e6)    # Convert to GHz.

     get_mean_spectrum_diff = np.abs(get_mean_spectrum(filename_on) - get_mean_spectrum(filename_off)) 
     bad_chans = get_mean_spectrum_diff == 0
     get_mean_spectrum_diff[bad_chans] = 1
     P_T_cal = calibrator_spectrum(src1, freq_mhz)/get_mean_spectrum_diff
     P_T_cal[bad_chans] = 0

     #print('src is', src1)
     #print('file_on is', filename_on)
     #print('file_off is', filename_off)
     #return P_T_cal
     
     f = plt.figure()
     plt.axis((0,4096,0,3))
     plt.ylabel('T_cal(K)')
     plt.plot(P_T_cal)
     out_filename = path.splitext(path.basename(filename_on))[0]
     out_filename += str('_&_',)
     out_filename += path.splitext(path.basename(filename_off))[0][18:27]
     out_filename += str('_calibration.png',) 
     plt.savefig(out_filename, bbox_inches='tight')
     plt.close(f)

if __name__ == "__main__":
    files = sys.argv[1:]

    bp = []
    for f in files:
        bp.append(bandpass_I_from_onoff(f))

    n = len(bp)
    cal_T_n = np.empty((n, len(bp[0])), dtype=np.float64)
    freq = bp[0]['freq']
    for ii in range(n):
        cal_T_n[ii,:] = bp[ii]['cal_T']
        if not np.allclose(bp[ii]['freq'], freq):
            raise ValueError("Frequency axis")

    bad_chans = np.any(np.isnan(cal_T_n), 0)
    cal_T_n[:,bad_chans] = 1
    med_spec = np.median(cal_T_n, 0)
    std_spec = np.std(cal_T_n, 0)
    norm_std = std_spec / med_spec
    bad_chans = np.logical_or(bad_chans, norm_std > 10 * np.median(norm_std))
    med_spec[bad_chans] = float('nan')

    #plt.figure()
    #plt.plot(freq, med_spec)

    #plt.show()

    out = bp[0]
    out['cal_T'] = med_spec
    np.save("cal_spectrum_I.npy", out)
