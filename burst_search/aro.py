"""Driver scripts and IO for ARO data.

"""

import math
from os import path
import time

import numpy as np
from numpy import array, dot
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from aro_tools import power_data_io, misc

from . import preprocess
from . import dedisperse
from . import search
from . import simulate
from simulate import *


# XXX Eventually a parameter, seconds.
#TIME_BLOCK = 30.

MIN_SEARCH_DM = 5.

TIME_BLOCK = 100.0

MAX_DM = 1000

# Overlap needs to account for the total delay across the band at max DM as
# well as any data invalidated by FIR filtering of the data.
#OVERLAP = 15.
OVERLAP = 10

DO_SPEC_SEARCH = True
USE_JON_DD = True

SPEC_INDEX_MIN = -4
SPEC_INDEX_MAX = 4
SPEC_INDEX_SAMPLES = 3

THRESH_SNR = 8.0

DEV_PLOTS = False

HPF_WIDTH = 0.2    # s


# Host login info.
SQUIRREL = ('squirrel@10.1.1.100', 22, '/home/squirrel/linkingscripts')
CHIME = ('moose@localhost', 10022, '/home/connor')

#Event simulation params, speculative/contrived
SIMULATE = False
alpha = -1.0
sim_rate = 50*1.0/6000.0
f_m = 600.
f_sd = 0
bw_m = 400.
bw_sd = 0
t_m = 0.003
t_sd = 0.001
s_m = 0.01
s_sd = 0.005
dm_m = 300
dm_sd = 200


# ARO hardcoded parameters.
#NTIME_RECORD = 1024     # Arbitrary, doesn't matter.
#DELTA_T = 0.005
#NFREQ = 1024
#FREQ0 = 800.
#DELTA_F = -400. / 1024
#CAL_PERIOD_SAMPLES = 0


class FileSearch(object):

    def __init__(self, **kwargs):

        self._run_parameters = kwargs

        filename = kwargs['filename']
        self._filename = filename
        scrunch = kwargs.get('scrunch', 1)
        self._scrunch = scrunch    # For identifying process.
        self._data_ring = power_data_io.Ring(filename, scrunch)
        #self._data_ring = power_data_io.MockRing(filename, scrunch)


        #parameters = get_parameters(filename)
        #print parameters
        parameters = self._data_ring.get_parameters()
        self._parameters = parameters
        ####

        if kwargs.get('to_2_DM_diag', False):
            freq0 = parameters['freq0']
            delta_f = parameters['delta_f']
            nfreq = parameters['nfreq']
            freq_max = max(freq0, freq0 + delta_f * (nfreq - 1))
            freq_min = min(freq0, freq0 + delta_f * (nfreq - 1))
            dm_diag = misc.diag_dm(parameters['delta_t'], delta_f, freq_max)
            self._max_dm = 0.9 * 2 * dm_diag
            self._overlap = 1.5 * (misc.disp_delay(2 * dm_diag, freq_min)
                                   - misc.disp_delay(2 * dm_diag, freq_max))
            self._time_block = 3.0 * self._overlap
        else:
            self._time_block = kwargs.get('time_block', TIME_BLOCK)
            self._overlap = kwargs.get('overlap', OVERLAP)
            self._max_dm = kwargs.get('max_dm', MAX_DM)

        #print self._overlap, self._time_block, dm_diag

        self._min_search_dm = kwargs.get('min_search_dm', MIN_SEARCH_DM)

        
        self._Transformer = dedisperse.DMTransform(
                parameters['delta_t'],
                parameters['nfreq'],
                parameters['freq0'],
                parameters['delta_f'],
                self._max_dm,
                jon=USE_JON_DD,
                )

        print ("Filename: %s, delta_t: %f, min_dm: %f, max_dm: %f, ndm: %d,"
               " time_block: %f"
                % (filename, parameters['delta_t'], self._min_search_dm,
                    self._max_dm, self._Transformer.ndm, self._time_block))

        #self._df = parameters['delta_f']
        #self._nfreq = parameters['nfreq']
        #self._f0 = parameters['freq0']

        nrecords_block = int(math.ceil(self._time_block / (parameters['ntime_record'] * parameters['delta_t'])))

        #initialize sim object, if there are to be simulated events
        if SIMULATE:
            self._sim_source = simulate.RandSource(alpha=alpha, f_m=f_m,f_sd=f_sd,bw_m=bw_m,bw_sd=bw_sd,t_m=t_m,
                t_sd=t_sd,s_m=s_m,s_sd=s_sd,dm_m=dm_m,dm_sd=dm_sd,
                event_rate=sim_rate,file_params=self._parameters,t_overlap=self._overlap,nrecords_block=nrecords_block)

        self._cal_spec = 1.
        self._dedispersed_out_group = None

        self.set_search_method()
        self.set_trigger_action()


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
            self._search = lambda dm_data : search.basic(
                    dm_data,
                    THRESH_SNR,
                    self._min_search_dm,
                    int(HPF_WIDTH / 2 / self._parameters['delta_t']),
                    )

        else:
            msg = "Unrecognized search method."
            raise ValueError(msg)

    def set_trigger_action(self, action='print', **kwargs):
        actions = [self._get_trigger_action(s.strip()) for s in action.split(',')]
        def action_fun(triggers):
            for a in actions:
                a(triggers) 
        self._action = action_fun

    def _get_trigger_action(self,action):
        if action == 'print':
            def action_fun(triggers):
                print ("%d:" % self._scrunch), triggers
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
                    parameters = self._parameters
                    t_offset = t.time
                    f = plt.figure(1)
                    plt.subplot(411)
                    t.plot_dm()
                    plt.subplot(412)
                    t.plot_freq()
                    plt.subplot(413)
                    t.plot_time()
                    plt.subplot(414)
                    t.plot_spec()
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
                    out_filename += path.splitext(path.basename(self._filename))[0]
                    if not t.data.spec_ind is None:
                                    out_filename += "+a=%02.f" % t.data.spec_ind
                    out_filename += "+%06.2fs.png" % t_offset
                    plt.savefig(out_filename, bbox_inches='tight')
                    plt.close(f)
            return action_fun
        elif action == 'link_squirrel':
            def action_fun(triggers):
                for t in triggers:
                    t.link_baseband(*SQUIRREL)
            return action_fun
        elif action == 'link_chime':
            def action_fun(triggers):
                for t in triggers:
                    t.link_baseband(*CHIME)
            return action_fun
        else:
            msg = "Unrecognized trigger action: " + action
            raise ValueError(msg)

    def set_dedispersed_h5(self, group=None):
        """Set h5py group to which to write dedispersed data."""

        self._dedispersed_out_group = group

    #simple method to replace nested structure
    def search_records(self, start_record, end_record):
        data, time = self.get_records(start_record, end_record)
        parameters = self._parameters
        
        if DEV_PLOTS:
            plt.figure()
            imshow_data(data)

        preprocess.sys_temperature_bandpass(data)

        if DEV_PLOTS:
            plt.figure()
            imshow_data(data)

        if self._parameters['cal_period_samples']:
            preprocess.remove_periodic(data,
                               self._parameters['cal_period_samples'])


        nrecords_block = int(math.ceil(
            self._time_block / (parameters['ntime_record'] * parameters['delta_t'])))

        block_ind = start_record/nrecords_block

        # Preprocess.
        #preprocess.sys_temperature_bandpass(data)

        if SIMULATE and block_ind in self._sim_source.coarse_event_schedule():
            #do simulation
            data += self._sim_source.generate_events(block_ind)[:,0:data.shape[1]]

        preprocess.remove_outliers(data, 5, 128)
        data = preprocess.highpass_filter(data, HPF_WIDTH / parameters['delta_t'])

        preprocess.remove_outliers(data, 5)
        preprocess.remove_noisy_freq(data, 2)
        preprocess.remove_bad_times(data, 2)
        preprocess.remove_continuum_v2(data)
        preprocess.remove_noisy_freq(data, 2)

        # Readjust time axis.
        times_lost = len(time) - data.shape[-1]
        time = time[times_lost // 2: times_lost // 2 + data.shape[-1]]


        if DEV_PLOTS:
            plt.figure()
            imshow_data(data[:,:2000])
            plt.show()

        #from here we weight channels by spectral index
        #center_f = self._f0 + (self._df*self._nfreq/2.0)
        #fmin = self._f0 + self._df*self._nfreq
        #fmax = self._f0

        nfreq = self._parameters['nfreq']
        delta_f = self._parameters['delta_f']
        freq0 = self._parameters['freq0']
        freq = np.arange(nfreq) * delta_f + freq0

        if DO_SPEC_SEARCH:
            #print "----------------------"
            spec_trigger = None

            complete = 1
            for alpha in np.linspace(SPEC_INDEX_MIN,SPEC_INDEX_MAX,SPEC_INDEX_SAMPLES):
                #for i in xrange(0,3):
                #weights = array([math.pow(f/center_f, alpha) for f in np.linspace(fmax,fmin,self._nfreq)])

                
                #f = lambda x: weights*x
                #this_dat = np.matrix(np.apply_along_axis(f, axis=0, arr=data),dtype=np.float32)

                spec_weights = ((freq / freq0)**alpha).astype(data.dtype)
                this_dat = data * spec_weights[:,None]
                
                dm_data = self._Transformer(this_dat)
                dm_data.spec_ind = alpha
                dm_data.t0 = time[0]
                del this_dat
                these_triggers = self._search(dm_data)
                del dm_data
                #print 'complete indices: {0} of {1} ({2})'.format(complete,SPEC_INDEX_SAMPLES,alpha)
                if len(these_triggers)  > 0:
                    print ("%d: alpha %2f, SNR %4.1f" % (self._scrunch,
                            alpha, these_triggers[0].snr))
                    if spec_trigger == None or these_triggers[0].snr > spec_trigger.snr:
                        spec_trigger = these_triggers[0]
                del these_triggers
                complete += 1

            #spec_triggers = [t[0] for t in spec_triggers if len(t) > 0]
            #spec_triggers = sorted(spec_triggers, key= lambda x: -x.snr)
            #if len(spec_triggers) > 0:
                #spec_triggers = [spec_triggers[0],]
                #self._action(spec_triggers, dm_data)
            if spec_trigger != None:
                self._action((spec_trigger,))
        else:
            dm_data = self._Transformer(data)
            dm_data.t0 = time[0]

            triggers = self._search(dm_data)
            self._action(triggers)
            del triggers


    def get_records(self, start_record, end_record):
        #data = read_records(self._filename, start_record, end_record)
        data = self._data_ring.read_records(start_record, end_record)
        return data


    def search_all_records(self):

        time_block = self._time_block
        overlap = self._overlap

        parameters = self._parameters

        record_length = (parameters['ntime_record'] * parameters['delta_t'])
        nrecords_block = int(math.ceil(time_block / record_length))
        nrecords_overlap = int(math.ceil(overlap / record_length))
        nrecords = self._data_ring.current_records()[1]

        for ii in xrange(0, nrecords, nrecords_block - nrecords_overlap):
            # XXX
            print "Block starting with record: {0} of {1}".format(ii,nrecords)
            #print "Progress: {0}".format(float(ii)/float(nrecords))
            self.search_records(ii, ii + nrecords_block)

    def search_real_time(self):
        parameters = self._parameters

        time_block = self._time_block
        overlap = self._overlap

        record_length = (parameters['ntime_record'] * parameters['delta_t'])
        nrecords_block = int(math.ceil(time_block / record_length))
        nrecords_overlap = int(math.ceil(overlap / record_length))

        wait_time = float(time_block - overlap) / 5
        max_wait_iterations = 10

        # Enter holding loop, processing records in blocks as they become
        # available.
        current_start_record = 0
        wait_iterations = 0
        while wait_iterations < max_wait_iterations:
            nrecords = self._data_ring.current_records()[1]
            if nrecords - current_start_record >= nrecords_block:
                print ("%d: Block starting with record: %d of %d"
                        % (self._scrunch, current_start_record, nrecords))
                try:
                    self.search_records(current_start_record,
                                        current_start_record + nrecords_block)
                except power_data_io.DataGone:
                    print "%d: MISSED A BLOCK" % self._scrunch
                current_start_record += nrecords_block - nrecords_overlap
                wait_iterations = 0
            else:
                time.sleep(wait_time)
                wait_iterations += 1
        # Precess any leftovers that don't fill out a whole block.
        self.search_records(current_start_record, 
                            current_start_record + nrecords_block) 


def imshow_data(data):
    data = data[:,:1000].copy()
    s = np.std(data)
    m = np.mean(data)
    vmin = m - 2*s
    vmax = m + 2*s
    plt.imshow(
            data,
            vmin=vmin,
            vmax=vmax,
            extent=[0, 1000, 400., 800.,],
            aspect='auto',
            )
    plt.xlabel('time (samples)')
    plt.ylabel('frequency (MHz)')
