"""Driver scripts and IO for Greenbank GUPPI data.

"""

import numpy as np
# Should be moved to scripts.
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits

from . import datasource
from . import manager
from . import preprocess


# GUPPI IO
# --------

class FileSource(datasource.DataSource):

    def __init__(self, filename, block=30., overlap=8., **kwargs):
        super(FileSource, self).__init__(
                source=filename,
                block=block,
                overlap=overlap,
                **kwargs
                )

        # Read the headers
        hdulist = pyfits.open(filename)
        mheader = hdulist[0].header
        dheader = hdulist[1].header

        if mheader['CAL_FREQ']:
            cal_period = 1. / mheader['CAL_FREQ']
            self._cal_period_samples = int(round(cal_period / dheader['TBIN']))
        else:
            self._cal_period_samples = 0
        self._delta_t_native = dheader['TBIN']
        self._nfreq = dheader['NCHAN']
        self._freq0 = mheader['OBSFREQ'] - mheader['OBSBW'] / 2.
        self._delta_f = dheader['CHAN_BW']
        self._mjd = mheader['STT_IMJD']
        self._start_time = (mheader['STT_SMJD'] + mheader['STT_OFFS'])
        ntime_record, npol, nfreq, one = eval(dheader["TDIM17"])[::-1]
        self._ntime_record = ntime_record
        hdulist.close()

        # Initialize blocking parameters.
        record_len = self._ntime_record * self._delta_t_native
        self._nrecords_block = int(np.ceil(block / record_len))
        self._nrecords_overlap = int(np.ceil(overlap / record_len))
        self._next_start_record = 0

    @property
    def nblocks_left(self):
        nrecords_left = get_nrecords(self._source) - self._next_start_record
        return int(np.ceil(float(nrecords_left) / 
                           (self._nrecords_block - self._nrecords_overlap)))

    @property
    def nblocks_fetched(self):
        return (self._next_start_record //
                (self._nrecords_block - self._nrecords_overlap))

    def get_next_block_native(self):
        start_record = self._next_start_record
        if self.nblocks_left == 0:
            raise StopIteration()

        t0 = start_record * self._ntime_record * self._delta_t_native
        t0 += self._delta_t_native / 2

        hdulist = pyfits.open(self._source)
        data = read_records(
                hdulist,
                self._next_start_record,
                self._next_start_record + self._nrecords_block,
                )

        self._next_start_record += (self._nrecords_block
                                    - self._nrecords_overlap)

        return t0, data

    @property
    def cal_period_samples(self):
        return self._cal_period_samples


# Driver class
# ------------

class Manager(manager.Manager):

    datasource_class = FileSource

    def preprocess(self, t0, data):
        """Preprocess the data.

        Preprocessing includes simulation.

        """

        preprocess.sys_temperature_bandpass(data)
        cal_period = self.datasource.cal_period_samples
        if cal_period:
            preprocess.remove_periodic(data, cal_period)

        block_ind = self.datasource.nblocks_fetched

        # Preprocess.
        #preprocess.sys_temperature_bandpass(data)

        if False and block_ind in self._sim_source.coarse_event_schedule():
            #do simulation
            data += self._sim_source.generate_events(block_ind)[:,0:data.shape[1]]

        preprocess.remove_outliers(data, 5, 128)

        ntime_pre_filter = data.shape[1]
        data = preprocess.highpass_filter(data, manager.HPF_WIDTH / self.datasource.delta_t)
        # This changes t0 by half a window width.
        t0 -= (ntime_pre_filter - data.shape[1]) / 2 * self.datasource.delta_t

        preprocess.remove_outliers(data, 5)
        preprocess.remove_noisy_freq(data, 3)
        preprocess.remove_bad_times(data, 2)
        preprocess.remove_continuum_v2(data)
        preprocess.remove_noisy_freq(data, 3)

        return t0, data


# IO Helper functions
# -------------------

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


def get_nrecords(filename):
    hdulist = pyfits.open(filename, 'readonly')
    nrecords = len(hdulist[1].data)
    hdulist.close()
    return nrecords
