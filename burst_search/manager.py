"""Base search manager.
"""


from os import path
import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from . import dedisperse
from . import datasource
from . import search
from . import simulate
from . import catalog


logger = logging.getLogger(__name__)

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
        'time_block' : 40,    # Seconds.
        'overlap' : 8.,
        'scrunch' : 1,
        'max_dm' : 2000.,
        'min_search_dm' : 5.,
        'threshold_snr' : 8.,
        'trigger_action' : 'print',
        'spec_ind_search' : False,
        'spec_ind_min' : -5.,
        'spec_ind_max' : 5.,
        'spec_ind_samples' : 3.,
        'disp_ind_search' : False,
        'disp_ind_min' : 1.,
        'disp_ind_max' : 5.,
        'disp_ind_samples' : 9,
        'simulate' : False,
        'simulate_rate' : 0.01,
        'simulate_fluence' : 0.0002,
        }


class Manager(object):
    """Abstract base class for search manager.

    Subclasses must implement IO, adding a datasource_class attribute. It can
    optionally have custom preprocessing but reimplementing the preprocessing
    method.

    """

    datasource_class = None

    def __init__(self, source, **kwargs):
        parameters = dict(DEFAULT_PARAMETERS)
        parameters.update(kwargs)

        self._datasource = self.datasource_class(
                source,
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
        if parameters['max_dm'] < 0:
            # Go to N times the diagonal DM.'
            freq = self.datasource.freq
            diagonal_dm = dedisperse.calc_delta_dm(
                    self.datasource.delta_t,
                    freq[0],
                    freq[-1],
                    )
            diagonal_dm *= len(freq)
            max_dm = -parameters['max_dm'] * diagonal_dm
        else:
            max_dm = parameters['max_dm']
        self._max_dm = max_dm
        self._dm_transformers = []
        freq = self.datasource.freq
        max_freq = np.max(freq)
        min_freq = np.min(freq)
        for disp_ind in dispersion_inds:
            # Rescale max dm to same total delay as disp_ind=2.
            max_dm = max_dm
            max_dm *= (1.0 / min_freq ** 2.0) - (1.0 / max_freq ** 2.0)
            max_dm /= (1.0 / min_freq ** disp_ind) - (1.0 / max_freq ** disp_ind)
            transform = dedisperse.DMTransform(
                    self.datasource.delta_t,
                    self.datasource.nfreq,
                    self.datasource.freq0,
                    self.datasource.delta_f,
                    max_dm,
                    disp_ind,
                    )
            self._dm_transformers.append(transform)

        if parameters['simulate']:
            self._simulator = simulate.EventSimulator(
                    self._datasource,
                    rate=parameters['simulate_rate'],
                    fluence=(0, parameters['simulate_fluence']),
                    )
        else:
            self._simulator = None


        # Deal with these later.
        if False:
            #initialize sim object, if there are to be simulated events
            if CATALOG:
                reduced_name = '.'.join(self._filename.split(os.sep)[-1].split('.')[:-1])
                self._catalog = Catalog(parent_name=reduced_name, parameters=parameters)

        # Initialize trigger actions.
        self.set_trigger_action(parameters['trigger_action'])


    @property
    def datasource(self):
        return self._datasource

    @property
    def max_dm(self):
        return self._max_dm


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

    def simulate(self, t0, data):
        if self._simulator is not None:
            self._simulator.inject_events(t0, data)


        #for ii in range(0,data.shape[1],2000):
        #    plt.imshow(data[:,ii:ii+2000].copy())
        #    plt.show()
        return t0, data


    def preprocess(self, t0, data):
        """Preprocess the data.

        Preprocessing includes simulation.

        """

        preprocess.sys_temperature_bandpass(data)

        self.simulate(t0, data)

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
        t0, data = self.datasource.get_next_block()
        logger.info("Processing block %d." % self.datasource.nblocks_fetched)
        t0, data = self.preprocess(t0, data)

        freq = self.datasource.freq
        freq_norm = freq / np.mean(freq)

        trigger = None

        for transform in self._dm_transformers:
            for spec_ind in self._spectral_inds:
                msg = "Dispersion index %3.1f, spectral index %3.1f."
                logger.info(msg % (transform.disp_ind, spec_ind))
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
                    logger.info("Trigger with SNR %4.1f."
                            % this_best_trigger.snr)
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
        wait_time = float(self.datasource.time_block - self.datasource.overlap) / 10
        max_wait_iterations = 30

        # Enter holding loop, processing records in blocks as they become
        # available.
        wait_iterations = -10  # Wait a bit of extra time for first block.
        logger.info("Entering real-time processing holding loop.")
        while wait_iterations < max_wait_iterations:
            # If there is only 1 block left, it may not be complete.
            if self.datasource.nblocks_left >= 2:
                self.process_next_block()
                wait_iterations = 0
            else:
                time.sleep(wait_time)
                wait_iterations += 1
        logger.info("Exiting holding look and processing leftovers.")
        # Precess any leftovers that don't fill out a whole block.
        self.process_all()

