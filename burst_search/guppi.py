"""Driver scripts and IO for Greenbank GUPPI data.

"""

import math

import numpy as np
import pyfits
import matplotlib.pyplot as plt

from . import preprocess
from . import dedisperse


# XXX Eventually a parameter, seconds.
#TIME_BLOCK = 30.
TIME_BLOCK = 1.


def search_file(filename):
    """Simple dirver function to search a GUPPI file."""

    hdulist = pyfits.open(filename, 'readonly')

    parameters = parameters_from_header(hdulist)

    # Would set up dedispersion class here using info from parameters
    # dictionary.

    record_length = parameters['ntime_record'] * parameters['delta_t']
    nrecords_block = int(math.ceil(TIME_BLOCK / record_length))

    nrecords = len(hdulist[1].data)

    for ii in xrange(0, nrecords, nrecords_block):
        data = read_records(hdulist, ii, ii + nrecords_block)
        preprocess.noisecal_bandpass(data, 1., parameters['cal_period'])
        preprocess.remove_outliers(data, 5)
        preprocess.remove_noisy_freq(data, 3)

        # dedisperse.

        # Search for events.


    hdulist.close()


def parameters_from_header(hdulist):
    """Get data acqusition parameters for psrfits file header.

    Returns
    -------
    parameters : dict

    """

    parameters = {}

    #print repr(hdulist[0].header)
    #print repr(hdulist[1].header)

    # XXX For now just fake it.
    parameters['cal_period'] = 64
    parameters['delta_t'] = 0.001024
    parameters['nfreq'] = 4096
    parameters['freq0'] = 900e6
    parameters['delta_f'] = -200e6 / 4096
    parameters['npol'] = 4

    record0 = hdulist[1].data[0]
    data0 = record0["DATA"]
    #freq = record0["DAT_FREQ"]
    ntime_record, npol, nfreq, one = data0.shape

    parameters['ntime_record'] = ntime_record
    parameters['dtype'] = data0.dtype

    return parameters


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




# This wants to be a class for sure.
def monitor_file(filename, time_block):
    """Monitor GUPPI file for new data and process in chunks."""
