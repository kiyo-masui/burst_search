"""Driver scripts and IO for Greenbank GUPPI data.

"""

import math

import numpy as np
import pyfits
import matplotlib.pyplot as plt

from . import preprocess
from . import dedisperse
from . import search


# XXX Eventually a parameter, seconds.
TIME_BLOCK = 30.

MAX_DM = 4000.


class FileSearch(object):

    def __init__(self, filename):
        self._filename = filename
        hdulist = pyfits.open(filename, 'readonly')

        parameters = parameters_from_header(hdulist)
        #print parameters
        self._parameters = parameters

        self._Transformer = dedisperse.DMTransform(
                parameters['delta_t'],
                parameters['nfreq'],
                parameters['freq0'],
                parameters['delta_f'],
                MAX_DM,
                )

        self._nrecords = len(hdulist[1].data)
        self._cal_spec = 1.

        self.set_search_method()
        self.set_trigger_action()

        hdulist.close()

    def set_cal_spectrum(self, cal_spec):
        """Set spectrum of the noise-cal for band-pass calibration.

        Parameters
        ----------
        cal_spec : 1D record array with records 'freq' and 'cal_T'.

        """

        nfreq = self._parameters['nfreq']
        # TODO Should really check that the frequency axes match exactly, or
        # interpolate/extrapolate.
        if len(cal_spec) != nfreq:
            msg = "Noise cal spectrum frequncy axis does not match the data."
            raise ValueError(msg)
        self._cal_spec = cal_spec["cal_T"]

    def set_search_method(self, method='basic', **kwargs):
        if method == 'basic':
            self._search = lambda dm_data : search.basic(dm_data)
        else:
            msg = "Unrecognized search method."
            raise ValueError(msg)

    def set_trigger_action(self, action='print', **kwargs):
        if action == 'print':
            def action_fun(triggers, dm_data):
                print triggers
            self._action = action_fun
        else:
            msg = "Unrecognized trigger action."
            raise ValueError(msg)

    def search_records(self, start_record, end_record):
        dm_data = self.dm_transform_records(start_record, end_record)
        triggers = self._search(dm_data)
        self._action(triggers, dm_data)

    def dm_transform_records(self, start_record, end_record):
        parameters = self._parameters

        hdulist = pyfits.open(self._filename, 'readonly')
        data = read_records(hdulist, start_record, end_record)
        hdulist.close()

        if (True):
            # Preprocess.
            #plt.plot(nu, np.std(data, 0) * nu)
            preprocess.noisecal_bandpass(data, self._cal_spec,
                                         parameters['cal_period'])
            #plt.plot(nu, np.std(data, 0) * nu)
            #plt.show()

            # Place holder for functions that do things.
            preprocess.remove_outliers(data, 5)
            preprocess.remove_noisy_freq(data, 3)

        # Dispersion measure transform.
        dm_data = self._Transformer(data)

        return dm_data

    def search_all_records(self, time_block=TIME_BLOCK):

        parameters = self._parameters

        record_length = (parameters['ntime_record'] * parameters['delta_t'])
        nrecords_block = int(math.ceil(time_block / record_length))
        nrecords = self._nrecords

        for ii in xrange(0, nrecords, nrecords_block):
            # XXX
            print ii

            self.search_records(ii, ii + nrecords_block)


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
    parameters['freq0'] = 900.
    parameters['delta_f'] = -200. / 4096
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

    out_data = np.empty((nrecords_read, ntime_record, nfreq), dtype=np.float32)
    for ii in xrange(nrecords_read):
        # Read the record.
        record = hdulist[1].data[start_record + ii]["DATA"]
        # Interpret as unsigned int (for Stokes I only).
        record = record.view(dtype=np.uint8)
        # Select stokes I and copy.
        out_data[ii,...] = record[:,0,:,0]
    out_data.shape = (nrecords_read * ntime_record, nfreq)

    return out_data


