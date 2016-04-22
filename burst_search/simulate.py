import random
import math
import uuid
import logging

import numpy as np
import numpy.random as nprand

from dedisperse import disp_delay


logger = logging.getLogger(__name__)


class Event(object):

    def __init__(self, t_ref, f_ref, dm, fluence, width, spec_ind, disp_ind):
        self._t_ref = t_ref
        self._f_ref = f_ref
        self._dm = dm
        self._fluence = fluence
        self._width = width
        self._spec_ind = spec_ind
        self._disp_ind = disp_ind

    def arrival_time(self, f):
        t = disp_delay(f, self._dm, self._disp_ind)
        t = t - disp_delay(self._f_ref, self._dm, self._disp_ind)
        return self._t_ref + t

    def add_to_data(self, t0, delta_t, freq, data):
        ntime = data.shape[1]
        for ii, f in enumerate(freq):
            t = self.arrival_time(f)
            start_ind = int(round((t - t0) / delta_t))
            stop_ind = int(round((t + self._width - t0) / delta_t))
            start_ind = max(0, start_ind)
            start_ind = min(ntime, start_ind)
            stop_ind = max(0, stop_ind)
            stop_ind = min(ntime, stop_ind)
            val = self._fluence / self._width
            val = val * (f / self._f_ref) ** self._spec_ind
            data[ii,start_ind:stop_ind] += val


class EventSimulator(object):
    """Generates simulated fast radio bursts.

    Events occurances are drawn from a poissonian distribution.

    The events are rectangular in time with properties each evenly distributed
    within a range that is determined at instantiation.

    """


    def __init__(self, datasource, rate=0.001, dm=(0.,2000.), fluence=(0.0001,0.01),
                 width=(0.001, 0.010), spec_ind=(-4.,4), disp_ind=2.):
        """

        Parameters
        ----------
        datasource : datasource.DataSource object
            Source of the data, specifying the data rate and band parameters.
        rate : float
            The average rate at which to simulate events (per second).
        dm : float or pair of floats
            Burst dispersion measure or dispersion measure range (pc cm^-2).
        fluence : float or pair of floats
            Burst fluence (at band centre) or fluence range (s).
        width : float or pair of floats.
            Burst width or width range (s).
        spec_ind : float or pair of floats.
            Burst spectral index or spectral index range.
        disp_ind : float or pair of floats.
            Burst dispersion index or dispersion index range.

        """

        self._rate = rate
        if hasattr(dm, '__iter__') and len(dm) == 2:
            self._dm = tuple(dm)
        else:
            self._dm = (float(dm), float(dm))
        if hasattr(fluence, '__iter__') and len(fluence) == 2:
            self._fluence = tuple(fluence)
        else:
            self._fluence = (float(fluence), float(fluence))
        if hasattr(width, '__iter__') and len(width) == 2:
            self._width = tuple(width)
        else:
            self._width = (float(width), float(width))
        if hasattr(spec_ind, '__iter__') and len(spec_ind) == 2:
            self._spec_ind = tuple(spec_ind)
        else:
            self._spec_ind = (float(spec_ind), float(spec_ind))
        if hasattr(disp_ind, '__iter__') and len(disp_ind) == 2:
            self._disp_ind = tuple(disp_ind)
        else:
            self._disp_ind = (float(disp_ind), float(disp_ind))

        self._freq = datasource.freq
        self._delta_t = datasource.delta_t

        self._simulated_events = []

        self._last_time_processed = 0.

    def draw_event_parameters(self):
        dm = uniform_range(*self._dm)
        fluence = uniform_range(*self._fluence)
        width = uniform_range(*self._width)
        spec_ind = uniform_range(*self._spec_ind)
        disp_ind = uniform_range(*self._disp_ind)
        return dm, fluence, width, spec_ind, disp_ind

    def inject_events(self, t0, data):
        """Assumes that subsequent calls always happen with later t0,
        although blocks may overlap."""

        ntime = data.shape[1]
        time = np.arange(ntime) * self._delta_t + t0
        f_max = max(self._freq)
        f_min = min(self._freq)
        f_mean = np.mean(self._freq)

        overlap_events = [e for e in self._simulated_events if
                          e.arrival_time(f_min) > t0]

        new_events = []
        mu = self._rate * self._delta_t
        events_per_bin = nprand.poisson(mu, ntime)
        events_per_bin[time <= self._last_time_processed] = 0
        event_inds, = np.where(events_per_bin)
        for ind in event_inds:
            for ii in range(events_per_bin[ind]):
                dm, fluence, width, spec_ind, disp_ind = \
                        self.draw_event_parameters()
                msg = ("Injecting simulated event at time = %5.2f, DM = %6.1f,"
                       " fluence = %f, width = %f, spec_ind = %3.1f, disp_ind"
                       " = %3.1f.")
                logger.info(msg
                        % (time[ind], dm, fluence, width, spec_ind, disp_ind))
                t = disp_delay(f_min, dm, disp_ind)
                t = t - disp_delay(f_mean, dm, disp_ind)
                t = t + time[ind]
                e = Event(t, f_mean, dm, fluence, width, spec_ind, disp_ind)
                new_events.append(e)

        for e in overlap_events + new_events:
            e.add_to_data(t0, self._delta_t, self._freq, data)

        self._simulated_events = self._simulated_events + new_events
        self._last_time_processed = time[-1]


def uniform_range(min_, max_):
    return random.uniform(min_, max_)
