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

from . import preprocess
from . import dedisperse
from . import search
from . import simulate
from simulate import *


# XXX Eventually a parameter, seconds.
#TIME_BLOCK = 30.

#Additions:
MIN_SEARCH_DM = 5

TIME_BLOCK = 30.0

MAX_DM = 400
# For DM=4000, 13s delay across the band, so overlap searches by ~15s.

# Overlap needs to account for the total delay across the band at max DM as
# well as any data invalidated by FIR filtering of the data.
#OVERLAP = 15.
OVERLAP = 8.

DO_SPEC_SEARCH = True
SPEC_INDEX_MIN = -10
SPEC_INDEX_MAX = 10
SPEC_INDEX_SAMPLES = 5

THRESH_SNR = 8.0

DEV_PLOTS = False

#Event simulation params, speculative/contrived
SIMULATE = False
alpha = -5.0
sim_rate = 50*1.0/6000.0
f_m = 600.
f_sd = 0
bw_m = 400.
bw_sd = 0
t_m = 0.010
t_sd = 0.020
s_m = 0.01
s_sd = 0.01
dm_m = 600
dm_sd = 100


# ARO hardcoded paraemters.
NTIME_RECORD = 1024     # Arbitrary, doesn't matter.
DELTA_T = 0.005
NFREQ = 1024
FREQ0 = 800.
DELTA_F = -400. / 1024
CAL_PERIOD_SAMPLES = 0


class FileSearch(object):

    def __init__(self, filename):

        self._filename = filename

        parameters = get_parameters(filename)
        #print parameters
        self._parameters = parameters

        self._Transformer = dedisperse.DMTransform(
                parameters['delta_t'],
                parameters['nfreq'],
                parameters['freq0'],
                parameters['delta_f'],
                MAX_DM,
                )

        self._df = parameters['delta_f']
        self._nfreq = parameters['nfreq']
        self._f0 = parameters['freq0']
        self._record_length = (parameters['ntime_record'] * parameters['delta_t'])
        self._nrecords_block = int(math.ceil(TIME_BLOCK / self._record_length))
        self._nrecords_overlap = int(math.ceil(OVERLAP / self._record_length))
        self._nrecords = get_nrecords(filename)
        #also insert to parameters dict to keep things concise (sim code wants this)
        self._parameters['nrecords'] = self._nrecords

        #initialize sim object, if there are to be simulated events
        if SIMULATE:
            self._sim_source = simulate.RandSource(alpha=alpha, f_m=f_m,f_sd=f_sd,bw_m=bw_m,bw_sd=bw_sd,t_m=t_m,
                t_sd=t_sd,s_m=s_m,s_sd=s_sd,dm_m=dm_m,dm_sd=dm_sd,
                event_rate=sim_rate,file_params=self._parameters,t_overlap=OVERLAP,nrecords_block=self._nrecords_block)

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
            self._search = lambda dm_data,spec_ind=None : search.basic(dm_data, THRESH_SNR, MIN_SEARCH_DM,spec_ind=spec_ind)
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
                    parameters = self._parameters
                    t_offset = (parameters['ntime_record'] * t.data.start_record)
                    t_offset += t.centre[1]
                    t_offset *= parameters['delta_t']
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
                    if not t.spec_ind is None:
                                    out_filename += "+a=%02.f" % t.spec_ind
                    out_filename += "+%06.2fs.png" % t_offset
                    plt.savefig(out_filename, bbox_inches='tight')
                    plt.close(f)
            return action_fun
        else:
            msg = "Unrecognized trigger action: " + action
            raise ValueError(msg)

    def set_dedispersed_h5(self, group=None):
        """Set h5py group to which to write dedispersed data."""

        self._dedispersed_out_group = group

    #simple method to replace nested structure
    def search_records(self, start_record, end_record):
        data = self.get_records(start_record, end_record)
        parameters = self._parameters

        preprocess.sys_temperature_bandpass(data)
        if self._parameters['cal_period_samples']:
            preprocess.noisecal_bandpass(data, self._cal_spec,
             self._parameters['cal_period_samples'])

        block_ind = start_record/self._nrecords_block

        # Preprocess.
        #preprocess.sys_temperature_bandpass(data)

        if SIMULATE and block_ind in self._sim_source.coarse_event_schedule():
            #do simulation
            data += self._sim_source.generate_events(block_ind)[:,0:data.shape[1]]

        preprocess.remove_outliers(data, 5, 512)
        data = preprocess.highpass_filter(data, 0.200 / parameters['delta_t'])

        preprocess.remove_outliers(data, 5)
        preprocess.remove_noisy_freq(data, 3)

        #from here we weight channels by spectral index
        center_f = self._f0 + (self._df*self._nfreq/2.0)
        fmin = self._f0 + self._df*self._nfreq
        fmax = self._f0

        if DO_SPEC_SEARCH:
            print "----------------------"
            spec_trigger = None

            complete = 1
            for alpha in np.linspace(SPEC_INDEX_MIN,SPEC_INDEX_MAX,SPEC_INDEX_SAMPLES):
                #for i in xrange(0,3):
                weights = array([math.pow(f/center_f, alpha) for f in np.linspace(fmax,fmin,self._nfreq)])

                
                f = lambda x: weights*x
                this_dat = np.matrix(np.apply_along_axis(f, axis=0, arr=data),dtype=np.float32)

                #if self._dedispersed_out_group:
                 #   g = self._dedispersed_out_group.create_group("%d-%d"
                  #          % (start_record, end_record))
                   # data.to_hdf5(g)
                dm_data = self._Transformer(this_dat)
                del this_dat
                dm_data.start_record = start_record
                these_triggers = self._search(dm_data,spec_ind=alpha)
                del dm_data
                print 'complete indices: {0} of {1} ({2})'.format(complete,SPEC_INDEX_SAMPLES,alpha)
                if len(these_triggers)  > 0:
                    print 'max snr: {0}'.format(these_triggers[0].snr)
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
            dm_data.start_record = start_record

            triggers = self._search(dm_data)
            self._action(triggers)
            del triggers

    def dm_transform_records(self, start_record, end_record):
        parameters = self._parameters

        data = self.get_records(start_record, end_record)

        block_ind = start_record/self._nrecords_block

        if (True):
            # Preprocess.
            preprocess.sys_temperature_bandpass(data)

            if parameters['cal_period_samples']:
                preprocess.remove_periodic(data,
                                           parameters['cal_period_samples'])
                #preprocess.noisecal_bandpass(data, self._cal_spec,
                #                             parameters['cal_period_samples'])

            if SIMULATE and block_ind in self._sim_source.coarse_event_schedule():
                #do simulation
                data += self._sim_source.generate_events(block_ind)

            if DEV_PLOTS:
                plt.figure()
                plt.imshow(data[:2000,0:2000].copy())
                plt.colorbar()
                plt.figure()
                plt.plot(np.mean(data[:1000], 0))

            # 200 ms (hard coded) highpass filter
            preprocess.remove_outliers(data, 5, 512)
            data = preprocess.highpass_filter(data, 0.200 / parameters['delta_t'])

            preprocess.remove_outliers(data, 5)
            preprocess.remove_noisy_freq(data, 3)
            #preprocess.remove_continuum(data)
            #preprocess.remove_continuum_v2(data)

            # Second round RFI flagging post continuum removal?
            # Doesn't seem to help.
            #preprocess.remove_outliers(data, 5)
            #preprocess.remove_noisy_freq(data, 3)

        # Dispersion measure transform.
        dm_data = self._Transformer(data)
        dm_data.start_record = start_record

        if DEV_PLOTS:
            plt.figure()
            plt.imshow(dm_data.spec_data[:2000,0:2000].copy())
            plt.colorbar()
            plt.figure()
            plt.imshow(dm_data.dm_data[:,0:2000].copy())
            plt.colorbar()
            plt.figure()
            plt.plot(np.mean(dm_data.spec_data[:1000], 0))
            plt.show()

        return dm_data

    def get_records(self, start_record, end_record):
        data = read_records(self._filename, start_record, end_record)
        return data


    def search_all_records(self, time_block=TIME_BLOCK, overlap=OVERLAP):

        parameters = self._parameters

        record_length = (parameters['ntime_record'] * parameters['delta_t'])
        nrecords_block = int(math.ceil(time_block / record_length))
        nrecords_overlap = int(math.ceil(overlap / record_length))
        nrecords = self._nrecords

        for ii in xrange(0, nrecords, nrecords_block - nrecords_overlap):
            # XXX
            print "Block starting with record: {0} of {1}".format(ii,nrecords)
            #print "Progress: {0}".format(float(ii)/float(nrecords))
            self.search_records(ii, ii + nrecords_block)

    def search_real_time(self, time_block=TIME_BLOCK, overlap=OVERLAP):
        parameters = self._parameters

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
            nrecords = get_nrecords(self._filename)
            if nrecords - current_start_record >= nrecords_block:
                print "Block starting with record: %d" % current_start_record
                self.search_records(current_start_record,
                                    current_start_record + nrecords_block)
                current_start_record += nrecords_block - nrecords_overlap
                wait_iterations = 0
            else:
                time.sleep(wait_time)
                wait_iterations += 1
        # Precess any leftovers that don't fill out a whole block.
        self.search_records(current_start_record, 
                            current_start_record + nrecords_block) 




def get_parameters(filename):
    parameters = {}

    parameters['cal_period_samples'] = CAL_PERIOD_SAMPLES
    parameters['delta_t'] = DELTA_T
    parameters['nfreq'] = NFREQ
    parameters['freq0'] = FREQ0

    parameters['delta_f'] = DELTA_F
    parameters['ntime_record'] = NTIME_RECORD

    return parameters



def read_records(filename, start_record=0, end_record=None):
    """Right now just generates fake data."""

    nrecords_total = get_nrecords(filename)
    if end_record is None or end_record > nrecords_total:
        end_record = nrecords_total

    nrecords = end_record - start_record
    ntime = nrecords * NTIME_RECORD
    out = np.empty((NFREQ, nrecords, NTIME_RECORD), dtype=np.float32)

    # The fake part.
    from numpy import random
    # Every record is the same!
    noise = random.randn(NTIME_RECORD, 2, NFREQ)
    record_data = (noise + 32) * 10
    record_data = record_data.astype(np.uint32)
    for ii in range(nrecords):
        out[:,ii,:] = np.transpose(record_data[:,0,:] + record_data[:,1,:])
    out.shape = (NFREQ, nrecords * NTIME_RECORD)
    return out


start_time = -30
def get_nrecords(filename):
    """Totally fake."""

    global start_time
    if start_time < 0:
        start_time += time.time()
    return int((time.time() - start_time) / (DELTA_T * NTIME_RECORD))
