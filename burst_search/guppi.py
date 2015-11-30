"""Driver scripts and IO for Greenbank GUPPI data.

"""

import math
import os
from os import path
import time

import numpy as np
from numpy import array, dot
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import cProfile, pstats, StringIO

from . import preprocess
from . import dedisperse
from . import datasource
from . import search
from . import simulate
from simulate import *
from catalog import Catalog, convert_deg
from _preprocess import remove_outliers
from datasource import ScrunchFileSource, FileSource


# XXX Eventually a parameter, seconds.
#TIME_BLOCK = 30.

#Additions:
MIN_SEARCH_DM = 5

TIME_BLOCK = 30.0

#Must scale with disp_ind
MAX_DM = 2000
# For DM=4000, 13s delay across the band, so overlap searches by ~15s.

# Overlap needs to account for the total delay across the band at max DM as
# well as any data invalidated by FIR filtering of the data.
#OVERLAP = 15.
OVERLAP = 8.

DO_SPEC_SEARCH = False
SPEC_INDEX_MIN = -10
SPEC_INDEX_MAX = 10
SPEC_INDEX_SAMPLES = 11

THRESH_SNR = 8.0

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

#Use to test different dispersions
#of the form l^DISP_IND
#where l is wavelength
#Be sure to alter 'MAX_DM' accordingly.
#Large DISP_IND decreases depth
#DISP_IND is now set via a command line argument
DISP_IND = 2.0
DISP_MAX = None
DISP_IND_SAMPLES = None
dump_snrs = True

HPF_WIDTH = 0.2    # s

class FileSearch(object):

    def __init__(self, filename, disp_ind=DISP_IND, disp_max = DISP_MAX, disp_ind_samples = DISP_IND_SAMPLES, 
        max_dm = MAX_DM, sim = SIMULATE, datasource=None, scrunch=1,**kwargs):
        self._disp_ind = disp_ind
        self._disp_max = disp_max
        self._disp_ind_samples = disp_ind_samples
        SIMULATE = sim
        MAX_DM = max_dm
        
        self._filename = filename

        if datasource == None:
            datasource = FileSource(self._filename)
        self._datasource = datasource

        self._scrunch = scrunch

        hdulist = pyfits.open(filename, 'readonly')

        parameters = self.parameters_from_header(hdulist)
        #print parameters
        self._parameters = parameters

        if self._disp_max == None:
            #compute the invariant DM:
            MAX_DM *= (math.pow(1.0/700.0,2.0) - math.pow(1.0/900.0,2.0))
            MAX_DM /= (math.pow(1.0/700.0,self._disp_ind) - math.pow(1.0/900.0,self._disp_ind))

            self._Transformer = dedisperse.DMTransform(
                    parameters['delta_t'],
                    parameters['nfreq'],
                    parameters['freq0'],
                    parameters['delta_f'],
                    MAX_DM,
                    self._disp_ind,
                    )
        else:
            self._Transformer = {}
            for ind in np.linspace(self._disp_ind,self._disp_max,self._disp_ind_samples):
                this_dm = MAX_DM*(math.pow(1.0/700.0,2.0) - math.pow(1.0/900.0,2.0))
                this_dm /= (math.pow(1.0/700.0,ind) - math.pow(1.0/900.0,ind))
                self._Transformer[ind] = dedisperse.DMTransform(
                    parameters['delta_t'],
                    parameters['nfreq'],
                    parameters['freq0'],
                    parameters['delta_f'],
                    this_dm,
                    ind,
                    )
            if not 2.0 in np.linspace(self._disp_ind,self._disp_max,self._disp_ind_samples):
                self._Transformer[2.0] = dedisperse.DMTransform(
                    parameters['delta_t'],
                    parameters['nfreq'],
                    parameters['freq0'],
                    parameters['delta_f'],
                    MAX_DM,
                    2.0,
                    )
        
        self._df = parameters['delta_f']
        self._nfreq = parameters['nfreq']
        self._f0 = parameters['freq0']
        self._record_length = (parameters['ntime_record'] * parameters['delta_t'])
        self._nrecords_block = int(math.ceil(TIME_BLOCK / self._record_length))
        self._nrecords_overlap = int(math.ceil(OVERLAP / self._record_length))
        self._nrecords = len(hdulist[1].data)
        #also insert to parameters dict to keep things concise (sim code wants this)
        self._parameters['nrecords'] = self._nrecords

        #initialize sim object, if there are to be simulated events
        if SIMULATE:
            self._sim_source = simulate.RandSource(alpha=alpha, f_m=f_m,f_sd=f_sd,bw_m=bw_m,bw_sd=bw_sd,t_m=t_m,
                t_sd=t_sd,s_m=s_m,s_sd=s_sd,dm_m=dm_m,dm_sd=dm_sd,
                event_rate=sim_rate,file_params=self._parameters,t_overlap=OVERLAP,nrecords_block=self._nrecords_block)

        if CATALOG:
            reduced_name = '.'.join(self._filename.split(os.sep)[-1].split('.')[:-1])
            self._catalog = Catalog(parent_name=reduced_name, parameters=parameters)


        self._cal_spec = 1.
        self._dedispersed_out_group = None

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
            self._search = lambda dm_data,spec_ind=None,disp_ind=2.0 : search.basic(dm_data, THRESH_SNR, MIN_SEARCH_DM,spec_ind=spec_ind,disp_ind=disp_ind)
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
                    #out_filename += "+%06.2fs.png" % t_offset
                    if not t.disp_ind is None:
                                    out_filename += "+n=%02.f" % t.disp_ind
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

    def set_dedispersed_h5(self, group=None):
        """Set h5py group to which to write dedispersed data."""

        self._dedispersed_out_group = group

    #simple method to replace nested structure
    def search_records(self, start_record, end_record):
        data = self._datasource.get_records(start_record, end_record, self._scrunch)
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

        remove_outliers(data, 5, 128)

        data = preprocess.highpass_filter(data, HPF_WIDTH / parameters['delta_t'])

        remove_outliers(data, 5)
        preprocess.remove_noisy_freq(data, 3)
        preprocess.remove_bad_times(data, 2)
        preprocess.remove_continuum_v2(data)
        preprocess.remove_noisy_freq(data, 3)

        #from here we weight channels by spectral index
        center_f = self._f0 + (self._df*self._nfreq/2.0)
        fmin = self._f0 + self._df*self._nfreq
        fmax = self._f0

        if self._disp_max == None:
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
                    these_triggers = self._search(dm_data,spec_ind=alpha,disp_ind=self._disp_ind)
                    del dm_data
                    print 'complete spectral indices: {0} of {1} ({2})'.format(complete,SPEC_INDEX_SAMPLES,alpha)
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
                    triggers = (spec_trigger,)
                else: triggers = []
                    #self._action((spec_trigger,))
            else:
                dm_data = self._Transformer(data)
                dm_data.start_record = start_record

                triggers = self._search(dm_data,disp_ind=self._disp_ind)

        #Do dispersion index search
        else:
            disp_trigger = None
            complete = 1
            snrs = []
            for ind in sorted(self._Transformer.keys()):
                dm_data = self._Transformer[ind](data)
                dm_data.start_record = start_record
                these_triggers = self._search(dm_data,disp_ind=ind)
                if len(these_triggers) > 0:
                    print "this snr: {0}".format(these_triggers[0].snr)
                    snrs.append([ind,these_triggers[0].snr])
                    if disp_trigger == None or these_triggers[0].snr > disp_trigger.snr:
                        disp_trigger = these_triggers[0]
                    del these_triggers
                print 'complete dispersion indices: {0} of {1} ({2})'.format(complete,self._disp_ind_samples,ind)
                complete += 1

                if disp_trigger != None:
                    triggers = (disp_trigger,)
                else: triggers = []
                
            if dump_snrs:
                print snrs
            del snrs
        
        self._action(triggers)
        if CATALOG:
            self._catalog.simple_write(triggers,disp_ind = DISP_IND)
        del triggers

    def dm_transform_records(self, start_record, end_record):
        parameters = self._parameters

        hdulist = pyfits.open(self._filename, 'readonly')
        data = read_records(hdulist, start_record, end_record)
        hdulist.close()

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
        hdulist = pyfits.open(self._filename, 'readonly')
        data = read_records(hdulist, start_record, end_record)
        hdulist.close()
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

    def parameters_from_header(self, hdulist):
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
        parameters['delta_t'] = dheader['TBIN']*self._scrunch
        parameters['nfreq'] = dheader['NCHAN']
        parameters['freq0'] = mheader['OBSFREQ'] - mheader['OBSBW'] / 2.

        parameters['delta_f'] = dheader['CHAN_BW']
        parameters['mjd_start'] = mheader['STT_IMJD'] + (mheader['STT_SMJD'] + mheader['STT_OFFS'])/86400.0
        parameters['unix_start'] = (parameters['mjd_start'] - 40587.0)*86400.0

        parameters['track_mode'] = mheader['TRK_MODE']
        parameters['loc_0'] = (convert_deg(mheader['STT_CRD1']),convert_deg(mheader['STT_CRD2']))
        parameters['loc_1'] = (convert_deg(mheader['STP_CRD1']),convert_deg(mheader['STP_CRD2']))

        record0 = hdulist[1].data[0]
        #print record0
        #data0 = record0["DATA"]
        #freq = record0["DAT_FREQ"]
        ntime_record, npol, nfreq, one = eval(dheader["TDIM17"])[::-1]
        parameters['npol'] = npol

        #is this correct?
        parameters['ntime_record'] = ntime_record/self._scrunch
        parameters['dtype'] = np.uint8

        return parameters

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
    parameters['mjd_start'] = mheader['STT_IMJD'] + (mheader['STT_SMJD'] + mheader['STT_OFFS'])/86400.0
    parameters['unix_start'] = (parameters['mjd_start'] - 40587.0)*86400.0

    parameters['track_mode'] = mheader['TRK_MODE']
    parameters['loc_0'] = (convert_deg(mheader['STT_CRD1']),convert_deg(mheader['STT_CRD2']))
    parameters['loc_1'] = (convert_deg(mheader['STP_CRD1']),convert_deg(mheader['STP_CRD2']))

    record0 = hdulist[1].data[0]
    #print record0
    #data0 = record0["DATA"]
    #freq = record0["DAT_FREQ"]
    ntime_record, npol, nfreq, one = eval(dheader["TDIM17"])[::-1]
    parameters['npol'] = npol

    #is this correct?
    parameters['ntime_record'] = ntime_record
    parameters['dtype'] = np.uint8

    return parameters

def get_nrecords(filename):
    hdulist = pyfits.open(filename, 'readonly')
    nrecords = len(hdulist[1].data)
    hdulist.close()
    return nrecords
