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
#TIME_BLOCK = 30.
TIME_BLOCK = 15.

#MAX_DM = 4000.
MAX_DM = 1000.
# For DM=4000, 13s delay across the band, so overlap searches by ~15s.
#OVERLAP = 15.
OVERLAP = 0.


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
        spec = cal_spec["cal_T"]
        spec[np.logical_not(np.isfinite(spec))] = 0
        self._cal_spec = spec

    def set_search_method(self, method='basic', **kwargs):
        if method == 'basic':
            self._search = lambda dm_data : search.basic(dm_data)
        else:
            msg = "Unrecognized search method."
            raise ValueError(msg)

    def set_trigger_action(self, action='print', **kwargs):
        if action == 'print':
            def action_fun(triggers, data):
                print triggers
            self._action = action_fun
        else:
            msg = "Unrecognized trigger action."
            raise ValueError(msg)

    def set_dedispersed_h5(self, group=None):
        """Set h5py group to which to write dedispersed data."""

        self._dedispersed_out_group = group

    def search_records(self, start_record, end_record):
        data = self.dm_transform_records(start_record, end_record)
        if self._dedispersed_out_group:
            g = self._dedispersed_out_group.create_group("%d-%d"
                    % (start_record, end_record))
            data.to_hdf5(g)
        triggers = self._search(data)
        self._action(triggers, data)

    def dm_transform_records(self, start_record, end_record):
        parameters = self._parameters

        hdulist = pyfits.open(self._filename, 'readonly')
        data = read_records(hdulist, start_record, end_record)
        hdulist.close()

        if (True):
            # Preprocess.
            
            #plt.plot(np.mean(data, 0))
            preprocess.noisecal_bandpass(data, self._cal_spec,
                                         parameters['cal_period'])
            #plt.plot(np.mean(data, 0))
            #plt.show()

            # Place holder for functions that do things.
            preprocess.remove_outliers(data, 5)
            preprocess.remove_noisy_freq(data, 3)

        # Dispersion measure transform.
        dm_data = self._Transformer(data)
        
        # XXX
        plt.imshow(dm_data.spec_data[:,0:2000].copy())
        plt.figure()
        plt.imshow(dm_data.dm_data[:,0:2000].copy())
        plt.show()

        return dm_data

    def search_all_records(self, time_block=TIME_BLOCK, overlap=OVERLAP):

        parameters = self._parameters

        record_length = (parameters['ntime_record'] * parameters['delta_t'])
        nrecords_block = int(math.ceil(time_block / record_length))
        nrecords_overlap = int(math.ceil(overlap / record_length))
        nrecords = self._nrecords

        for ii in xrange(0, nrecords, nrecords_block - nrecords_overlap):
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
    parameters['delta_t'] = hdulist[1].header['TBIN']
    parameters['nfreq'] = 4096
    parameters['freq0'] = 900.
    parameters['delta_f'] = -200. / 4096
    parameters['npol'] = 4

    record0 = hdulist[1].data[0]
    #print record0
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


