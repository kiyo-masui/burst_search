"""Search DM space data for events."""


import numpy as np
import matplotlib.pyplot as plt

import _search

from catalog import Catalogable



class Trigger(Catalogable):

    dtype = np.dtype([('primary_key',np.str_,36), ('snr', np.float32), ('dm', np.float32), ('time_ind', np.float32), ('right_ascension', np.float32), ('declination', np.float32)])

    def __init__(self, data, centre, search_spec = None, snr=0.,ra=None,dec=None):

        #Alex's catalogable stuff
        #must associate each trigger object with a search spec to match primary keys
        self._search_spec = search_spec

        self._data = data
        self._dm_ind = centre[0]
        self._time_ind = centre[1]
        self._snr = snr
        self._ra = ra
        self._dec = dec

    @property
    def data(self):
        return self._data

    @property
    def centre(self):
        return (self._dm_ind, self._time_ind)

    @property
    def right_ascension(self):
        return (self._ra)

    @property 
    def declination(self):
        return (self._dec)


    #from catalogable
    def dtype(self):
        return dt

    def row_value(self):
        return np.matrix([(_search_spec.primary_key(), self._snr, self._dm, self._time_ind, self._ra, self._dec)], self.dtype())[0]

    def 

    def __str__(self):
        return str((self._snr, self.centre,(ra,dec)))

    def __repr__(self):
        return str((self._snr, self.centre,(ra,dec)))

    def plot_dm(self):
        di, ti = self.centre
        tside = 500
        dside = 300
        delta_t = self.data.delta_t
        delta_dm = self.data.delta_dm
        start_ti = max(0, ti - tside)
        end_ti = min(self.data.dm_data.shape[1], ti + tside)
        start_di = max(0, di - dside)
        end_di = min(self.data.dm_data.shape[0], di + dside)
        plt.imshow(self.data.dm_data[start_di:end_di,start_ti:end_ti],
                   extent=[start_ti * delta_t, end_ti * delta_t,
                           end_di * delta_dm, start_di * delta_dm],
                   aspect='auto',
                   )
        plt.xlabel("time (s)")
        plt.ylabel("DM (Pc/cm^3)")
        plt.colorbar()

def basic(data, snr_threshold=5., min_dm=50.):
    """Simple event search of DM data.

    Returns
    -------
    triggers : list of :class:`Trigger` objects.

    """

    triggers = []

    # Sievers' code breaks of number of channels exceeds number of samples.
    if data.dm_data.shape[0] < data.dm_data.shape[1]:
        # Find the index of the *min_dm*.
        min_dm_ind = (min_dm - data.dm0) / data.delta_dm
        min_dm_ind = int(round(min_dm_ind))
        min_dm_ind = max(0, min_dm_ind)

        snr, sample, duration = _search.sievers_find_peak(data, min_dm_ind)

        if snr > snr_threshold:
            triggers.append(Trigger(data, sample, snr))

    return triggers
