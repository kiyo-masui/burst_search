"""Search DM space data for events."""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import _search

def disp_delay(f,dm,disp_ind):
    """Compute the dispersion delay (s) as a function of frequency (MHz) and DM"""
    return 4.148808*dm*(10.0**3)/(f**disp_ind)

class Trigger(object):

    def __init__(self, data, centre, snr=0., duration=1 ,spec_ind=None, disp_ind = 2.0):

        self._data = data
        self._dm_ind = centre[0]
        self._time_ind = centre[1]
        self._snr = snr
        self._duration = duration
        self._spec_ind = spec_ind
        self._disp_ind = disp_ind

    @property 
    def disp_ind(self):
        return self._disp_ind

    @property 
    def snr(self):
        return self._snr

    @property
    def data(self):
        return self._data

    @property
    def centre(self):
        return (self._dm_ind, self._time_ind)

    @property 
    def spec_ind(self):
        return self._spec_ind

    def __str__(self):
        return str((self._snr, self._spec_ind, self._disp_ind, self.centre))

    def __repr__(self):
        return str((self._snr, self._spec_ind, self._disp_ind, self.centre))

    def plot_dm(self):
        di, ti = self.centre
        tside = 250
        dside = 300
        delta_t = self.data.delta_t
        delta_dm = self.data.delta_dm
        start_ti = max(0, ti - tside)
        end_ti = min(self.data.dm_data.shape[1], ti + tside)
        start_di = max(0, di - dside)
        end_di = min(self.data.dm_data.shape[0], di + dside)

        dm_value = "DM ="
        dm_value += str(round(di*delta_dm, 3))
        dm_value += " (pc/cm^3), "
        snr = self._snr
        snr_value = "SNR ="
        snr_value += str(round(snr, 2))
        duration = self._duration
        duration_value = ",width ="
        duration_value += str(round(duration*delta_t, 3))
        duration_value += " (s)"
        spec_ind = self._spec_ind
        spec_ind_value = ",spec_ind ="
        spec_ind_value += str(spec_ind)
        disp_ind = self.disp_ind
        disp_ind_value = ",disp_ind ="
        disp_ind_value += str(disp_ind)
        text = dm_value + snr_value + duration_value + spec_ind_value + disp_ind_value

        dm_data_cut = self.data.dm_data[start_di:end_di,start_ti:end_ti].copy()
        dm_lower_std_index = 1
        dm_upper_std_index = 5
        dm_mean = np.mean(dm_data_cut)
        dm_std = np.std(dm_data_cut)
        dm_vmin = dm_mean - dm_lower_std_index * dm_std
        dm_vmax = dm_mean + dm_upper_std_index * dm_std

        plt.imshow(dm_data_cut,
                   extent=[start_ti * delta_t, end_ti * delta_t,
                           end_di * delta_dm, start_di * delta_dm],
                   vmin = dm_vmin, vmax = dm_vmax,
                   aspect='auto',
                   cmap = cm.jet
                  )
        ax = plt.gca()
        ax.text(0.05, 1.2, text,transform=ax.transAxes)
        plt.xlabel("time (s)")
        plt.ylabel("DM (Pc/cm^3)")
        plt.colorbar()

    def dm_data_cut(self):
        di, ti = self.centre
        tside = 250
        dside = 300
        delta_t = self.data.delta_t
        delta_dm = self.data.delta_dm
        start_ti = max(0, ti - tside)
        end_ti = min(self.data.dm_data.shape[1], ti + tside)
        start_di = max(0, di - dside)
        end_di = min(self.data.dm_data.shape[0], di + dside)
        dm_data_cut = self.data.dm_data[start_di:end_di,start_ti:end_ti].copy()
        return dm_data_cut

    def plot_time(self):
        di, ti = self.centre
        tside = 250 // 2
        delta_t = self.data.delta_t
        delta_dm = self.data.delta_dm
        #print delta_dm
        start_ti = max(0, ti - tside)
        end_ti = min(self.data.dm_data.shape[1], ti + tside)
        time = np.arange(start_ti, end_ti) * delta_t
        plt.plot(time, self.data.dm_data[di,start_ti:end_ti], 'b')
        plt.plot(time, self.data.dm_data[di,start_ti:end_ti], 'r.')
        #plt.plot([ti * delta_t], self.data.dm_data[di,[ti]], 'g.')
        plt.xlabel("time (s)")
        plt.ylabel("Flux")

    def plot_spec(self):
        di, ti = self.centre
        delta_t = self.data.delta_t
        delta_dm = self.data.delta_dm
        ntime = self.data.spec_data.shape[1]
        nfreq = self.data.nfreq

        freq = self.data.freq
        max_freq = np.max(freq)

        the_dm = self.data.dm[di]
        duration = self._duration

        spectrum = np.zeros(nfreq, dtype=float)

        disp_ind = self._disp_ind

        for ii in range(nfreq):
            f = freq[ii]
            delay_ind = int(round((disp_delay(freq[ii], the_dm, disp_ind)
                                   - disp_delay(max_freq, the_dm, disp_ind)) / delta_t))
            # Jon's centre index seems to be the last bin in the window.
            start = ti + delay_ind - duration + 1
            stop = start + duration
            spectrum[ii] = np.mean(self.data.spec_data[ii,start:stop])
        
        self.fluence = sum(spectrum)

        colors = ['blue','red','darkgreen','orange']
        rebin_factors = [ 4**ii for ii in range(4) ]

        for ii, rfact in enumerate(rebin_factors):
            spectrum_r = np.reshape(spectrum, (nfreq // rfact, rfact))
            plt.plot(
                freq[rfact//2::rfact],
                np.mean(spectrum_r, 1),
                colors[ii % len(colors)],
                )
        plt.ylabel("Fluence")
        plt.xlabel("Frequency (MHz)")

    def plot_freq(self):
        di, ti = self.centre
        tside = int(250 * 1.)
        dside = 300
        delta_t = self.data.delta_t
        delta_dm = self.data.delta_dm
        ntime = self.data.spec_data.shape[1]
        nfreq = self.data.nfreq

        freq = self.data.freq
        max_freq = np.max(freq)

        the_dm = self.data.dm[di]

        disp_ind = self._disp_ind

        spec_data_delay = np.zeros((nfreq, tside), dtype=float)
        for ii in range(nfreq):
            delay_ind = int(round((disp_delay(freq[ii], the_dm, disp_ind)
                                   - disp_delay(max_freq, the_dm, disp_ind)) / delta_t))
            start_i =  ti - tside // 2 + delay_ind
            stop_i = start_i + tside
            start_o = 0
            stop_o = tside
            if start_i < 0:
                start_o = -start_i
                start_i = 0
            if stop_i > ntime:
                stop_o -= (ntime - stop_i)
                stop_i = ntime
            spec_data_delay[ii,start_o:stop_o] = self.data.spec_data[ii,
                    start_i:stop_i]

        rebin_factor_freq = 64
        rebin_factor_time = 1
        spec_data_delay.shape = (
                nfreq // rebin_factor_freq,
                rebin_factor_freq, 
                tside // rebin_factor_time,
                rebin_factor_time,
                )
        spec_data_rebin = np.mean(np.mean(spec_data_delay, 3), 1)

        image_mean = np.mean(spec_data_rebin)
        image_std = np.std(spec_data_rebin)
        vmin = image_mean - 1 * image_std
        vmax = image_mean + 5 * image_std

        tstart = (ti - tside // 2) * delta_t
        tstop = tstart + tside * delta_t

        plt.imshow(
                spec_data_rebin,
                extent=[tstart, tstop, freq[-1], freq[0]],
                vmin=vmin,
                vmax=vmax,
                aspect='auto',
                cmap = cm.Blues
                )
        plt.xlabel("time (s)")
        plt.ylabel("Frequency (MHz)")
        plt.colorbar()

def basic(data, snr_threshold=5., min_dm=50.,spec_ind=None,disp_ind=2.0):
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
            triggers.append(Trigger(data, sample, snr,spec_ind=spec_ind,duration=duration,disp_ind=disp_ind))
    return triggers
