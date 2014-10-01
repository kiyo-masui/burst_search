"""Search DM space data for events."""

import _search

import matplotlib.pyplot as plt


class Trigger(object):

    def __init__(self, data, centre, snr=0.):

        self._data = data
        self._dm_ind = centre[0]
        self._time_ind = centre[1]
        self._snr = snr

    @property
    def data(self):
        return self._data

    @property
    def centre(self):
        return (self._dm_ind, self._time_ind)

    def __str__(self):
        return str(self.centre)

    def __repr__(self):
        return str(self.centre)

    def plot_dm(self):
        ti = self._time_ind
        tside = 500
        start_ti = max(0, ti - tside)
        end_ti = min(self.data.dm_data.shape[1], ti + tside)
        print self
        plt.imshow(self.data.dm_data[:,start_ti:end_ti])



def basic(data, snr_threshold=5.):
    """Simple event search of DM data.

    Returns
    -------
    triggers : list of :class:`Trigger` objects.

    """

    triggers = []

    snr, sample, duration = _search.sievers_find_peak(data)

    if snr > snr_threshold:
        triggers.append(Trigger(data, sample, snr))

    return triggers
