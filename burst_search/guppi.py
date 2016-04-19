"""Driver scripts and IO for Greenbank GUPPI data.

"""

import math
import os
from os import path
import time

import numpy as np
# Should be moved to scripts.
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits

from . import preprocess
from . import dedisperse
from . import datasource
from . import search
from . import simulate
from . import catalog


USE_JON_DD = False
DEV_PLOTS = False

#Event simulation params, speculative/contrived
SIMULATE = False
alpha = -5.0
sim_rate = 50*1.0/6000.0
f_m = 800
f_sd = 0
bw_m = 200
bw_sd = 0
t_m = 0.003
t_sd = 0.002
s_m = 100*0.6
s_sd = 0.1
dm_m = 600
dm_sd = 0

CATALOG = True


HPF_WIDTH = 0.1    # s


DEFAULT_PARAMETERS = {
        # filename must also be supplied.
        'time_block' : 40,    # Seconds.
        'overlap' : 8.,
        'scrunch' : 1,
        'max_dm' : 2000.,
        'min_search_dm' : 5.,
        'threshold_snr' : 8.,
        'trigger_action' : 'print',
        'spec_ind_search' : True,
        'spec_ind_min' : -5.,
        'spec_ind_max' : 5.,
        'spec_ind_samples' : 3.,
        'disp_ind_search' : False,
        'disp_ind_min' : 1.,
        'disp_ind_max' : 5.,
        'disp_ind_samples' : 9,
        'simulate' : False,
        'simulate_rate' : 0,
        }


class FileSearch(object):

    def __init__(self, filename, **kwargs):
        parameters = dict(DEFAULT_PARAMETERS)
        parameters.update(kwargs)

        self._datasource = FileSource(
                filename,
                block=parameters['time_block'],
                overlap=parameters['overlap'],
                scrunch=parameters['scrunch'],
                )

        # Store search parameters.
        self._min_search_dm = parameters['min_search_dm']
        self._threshold_snr = parameters['threshold_snr']

        # Spectral index search.
        if parameters['spec_ind_search']:
            self._spectral_inds = np.linspace(
                    parameters['spec_ind_min'],
                    parameters['spec_ind_max'],
                    parameters['spec_ind_samples'],
                    endpoint=True,
                    )
        else:
            self._spectral_inds = [0.]


        # Dispersion index search.
        if parameters['disp_ind_search']:
            dispersion_inds = np.linspace(
                    parameters['disp_ind_min'],
                    parameters['disp_ind_max'],
                    parameters['disp_ind_samples'],
                    endpoint=True,
                    )
        else:
            dispersion_inds = [2.]

        # Initailize DM transforms.
        self._dm_transformers = []
        freq = self.datasource.freq
        max_freq = np.max(freq)
        min_freq = np.min(freq)
        for disp_ind in dispersion_inds:
            # Rescale max dm to same total delay as disp_ind=2.
            max_dm = parameters['max_dm']
            max_dm *= (1.0 / min_freq ** 2.0) - (1.0 / max_freq ** 2.0)
            max_dm /= (1.0 / min_freq ** disp_ind) - (1.0 / max_freq ** disp_ind)
            transform = dedisperse.DMTransform(
                    self.datasource.delta_t,
                    self.datasource.nfreq,
                    self.datasource.freq0,
                    self.datasource.delta_f,
                    max_dm,
                    disp_ind,
                    jon=USE_JON_DD,
                    )
            self._dm_transformers.append(transform)

        # Deal with these later.
        if False:
            #initialize sim object, if there are to be simulated events
            if SIMULATE:
                self._sim_source = simulate.RandSource(alpha=alpha, f_m=f_m,f_sd=f_sd,bw_m=bw_m,bw_sd=bw_sd,t_m=t_m,
                    t_sd=t_sd,s_m=s_m,s_sd=s_sd,dm_m=dm_m,dm_sd=dm_sd,
                    event_rate=sim_rate,file_params=self._parameters,t_overlap=OVERLAP,nrecords_block=self._nrecords_block)

            if CATALOG:
                reduced_name = '.'.join(self._filename.split(os.sep)[-1].split('.')[:-1])
                self._catalog = Catalog(parent_name=reduced_name, parameters=parameters)

        # Initialize trigger actions.
        self.set_trigger_action(parameters['trigger_action'])


    @property
    def datasource(self):
        return self._datasource


    def set_trigger_action(self, action='print', **kwargs):
        actions = [self._get_trigger_action(s.strip()) for s in action.split(',')]
        def action_fun(triggers):
            for a in actions:
                a(triggers)
        self._action = action_fun

    def _get_trigger_action(self,action):
        if action == 'print':
            def action_fun(triggers):
                print triggers
            return action_fun
            self._action = action_fun
        elif action == 'show_plot_dm':
            def action_fun(triggers):
                for t in triggers:
                    plt.figure()
                    t.plot_dm()
                plt.show()
            return action_fun
        elif action == 'show_plot_time':
            def action_fun(triggers):
                for t in triggers:
                    plt.figure()
                    t.plot_time()
                plt.show()
            return action_fun
        elif action == 'save_plot_dm':
            def action_fun(triggers):
                for t in triggers:
                    t_offset = t.data.t0 * t.centre[1] * t.data.delta_t
                    f = plt.figure(1)
                    t.plot_summary()

                    t_dm_value = t.centre[0] * t.data.delta_dm
                    if t_dm_value < 5:
                        out_filename = "DM0-5_"
                    elif 5 <= t_dm_value < 20:
                        out_filename = "DM5-20_"
                    elif 20 <= t_dm_value < 100:
                        out_filename = "DM20-100_"
                    elif 100 <= t_dm_value <300:
                        out_filename = "DM100-300_"
                    else:
                        out_filename = "DM300-2000_" 
                    out_filename += path.splitext(path.basename(self.datasource._source))[0]
                    if not t.data.spec_ind is None:
                                    out_filename += "+a=%02.f" % t.data.spec_ind
                    #out_filename += "+%06.2fs.png" % t_offset
                    if not t.data.disp_ind is None:
                                    out_filename += "+n=%02.f" % t.data.disp_ind
                    out_filename_png = out_filename + "+%06.2fs.png" % t_offset

                    out_filename_DMT = out_filename + "_DM-T_ "+ "+%06.2fs.npy" % t_offset
                    out_filename_FT  = out_filename + "_Freq-T_" + "+%06.2fs.npy" % t_offset

 
                    plt.savefig(out_filename_png, bbox_inches='tight')
                    plt.close(f)
                    dm_data_cut = t.dm_data_cut()
                    np.save(out_filename_DMT, dm_data_cut)
                    spec_data_rebin = t.spec_data_rebin()
                    np.save(out_filename_FT, spec_data_rebin)
            return action_fun
        else:
            msg = "Unrecognized trigger action: " + action
            raise ValueError(msg)

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

        if SIMULATE and block_ind in self._sim_source.coarse_event_schedule():
            #do simulation
            data += self._sim_source.generate_events(block_ind)[:,0:data.shape[1]]

        preprocess.remove_outliers(data, 5, 128)

        ntime_pre_filter = data.shape[1]
        data = preprocess.highpass_filter(data, HPF_WIDTH / self.datasource.delta_t)
        # This changes t0 by half a window width.
        t0 -= (ntime_pre_filter - data.shape[1]) / 2 * self.datasource.delta_t

        preprocess.remove_outliers(data, 5)
        preprocess.remove_noisy_freq(data, 3)
        preprocess.remove_bad_times(data, 2)
        preprocess.remove_continuum_v2(data)
        preprocess.remove_noisy_freq(data, 3)

        return t0, data

    def search(self, dm_data):
        """Search the data."""
        return search.basic(
                dm_data,
                self._threshold_snr,
                self._min_search_dm,
                int(HPF_WIDTH/self.datasource.delta_t),
                )

    def process_next_block(self):
        print "Processing block %d." % self.datasource.nblocks_fetched
        t0, data = self.datasource.get_next_block()
        t0, data = self.preprocess(t0, data)

        freq = self.datasource.freq
        freq_norm = freq / np.mean(freq)

        trigger = None

        for transform in self._dm_transformers:
            for spec_ind in self._spectral_inds:
                msg = "Dispersion index %3.1f, spectral index %3.1f."
                print msg % (transform.disp_ind, spec_ind)
                weights = freq_norm ** spec_ind
                this_data = (data * weights[:,None]).astype(data.dtype)
                # DM transform.
                dm_data = transform(this_data)
                # Metadata required for the search but not the transform.
                dm_data._t0 = t0
                dm_data._spec_ind = spec_ind
                # Search the data.
                this_triggers = self.search(dm_data)
                # Free up memory if there where no triggers.
                del this_data, dm_data
                if len(this_triggers) > 0:
                    # In principle a search routine could return more than one
                    # trigger. For now just choose the strongest.
                    this_best_trigger = this_triggers[0]
                    for t in this_triggers[1:]:
                        if t.snr > this_best_trigger.snr:
                            this_best_trigger = t
                    print "Trigger with SNR %4.1f." % this_best_trigger.snr
                    if trigger is None or this_best_trigger.snr > trigger.snr:
                        trigger = this_best_trigger
                    # Recover memory for next iteration.
                    del this_triggers, this_best_trigger
        # Process any triggers.
        if trigger is not None:
            self._action([trigger])
            if False:
                self._catalog.simple_write([trigger])

    def process_all(self):
        while True:
            try:
                self.process_next_block()
            except StopIteration:
                break

    def process_real_time(self):
        wait_time = float(self.time_block - self.overlap) / 5
        max_wait_iterations = 10

        # Enter holding loop, processing records in blocks as they become
        # available.
        wait_iterations = 0
        while wait_iterations < max_wait_iterations:
            # If there is only 1 block left, it probably is not be complete.
            if self.datasource.nblocks_left >= 2:
                print "Processing block %d." % self.datasource.nblocks_fetched
                self.process_next_block()
                wait_iterations = 0
            else:
                time.sleep(wait_time)
                wait_iterations += 1
        # Precess any leftovers that don't fill out a whole block.
        self.process_all()


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
