"""Search DM space data for events."""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import _search

def disp_delay(f,dm):
    """Compute the dispersion delay (s) as a function of frequency (MHz) and DM"""
    return 4.148808*dm*(10.0**3)/(f**2)

class Trigger(object):

    def __init__(self, data, centre, snr=0., duration=1.,spec_ind=None):

        self._data = data
        self._dm_ind = centre[0]
        self._time_ind = centre[1]
        self._snr = snr
	self._duration = duration
        self._spec_ind = spec_ind

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
        return str((self._snr, self._spec_ind, self.centre))

    def __repr__(self):
        return str((self._snr, self._spec_ind, self.centre))

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
	spec_ind_value += str(round(spec_ind,1))
	text = dm_value + snr_value + duration_value + spec_ind_value
	
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

    def plot_time(self):
        di, ti = self.centre
        tside = 250
        dside = 300
        delta_t = self.data.delta_t
        delta_dm = self.data.delta_dm
        start_ti = max(0, ti - tside)
        end_ti = min(self.data.dm_data.shape[1], ti + tside)
        time = np.arange(start_ti, end_ti) * delta_t
        plt.plot(time, self.data.dm_data[di,start_ti:end_ti], 'b'
                   )
        plt.plot(time, self.data.dm_data[di,start_ti:end_ti], 'r.'
                   )
        plt.xlabel("time (s)")
        plt.ylabel("Flux")

    def plot_spec(self):
        di, ti = self.centre
        tside = 250
        dside = 300
        delta_t = self.data.delta_t
        delta_dm = self.data.delta_dm
        df = -200.0/float(self.data.spec_data.shape[0])
        f0 = 900
        f1 = f0 + self.data.spec_data.shape[0]*df	
	duration = self._duration

        start_ti = max(0, ti - tside)
        end_ti = min(self.data.dm_data.shape[1], ti + tside)
        time = np.arange(start_ti, end_ti) * delta_t
        freq = np.arange(f0,f1,df)

        rebin_factor_freq = 1
        rebin_factor_time = 1
        xlen = self.data.spec_data.shape[0] / rebin_factor_freq
        ylen = self.data.spec_data.shape[1] / rebin_factor_time
        new_spec_data = np.zeros((xlen,ylen))
        for i in range(xlen):
                for j in range(ylen):
                        new_spec_data[i,j] = self.data.spec_data[i*rebin_factor_freq:(i+1)*rebin_factor_freq,j*rebin_factor_time:(j+1)*rebin_factor_time].mean()

	intensity_integrate[i] = new_spec_data[i,ti]
	for k in range(duration):
		intensity_integrate[i] += new_spec_data[i,ti+k]

	plt.plot(freq, intensity_integrate[:], 'b'
                   )
        plt.plot(freq, intensity_integrate[:], 'r.'
                   )
        plt.xlabel("freq (MHz)")
        plt.ylabel("Intensity_integrate")

#    def dedisperse(self):
#	di, ti = self.centre
#	delta_t = self.data.delta_t
#        ret = np.zeros(self.data.spec_data.shape)
#        df = -200.0/float(delta_dm)
#        f0 = 900
#        for i in xrange(0,self.data.spec_data.shape[0]):
#                f = f0 + i*df
#                dm = di
#                delay_ind = int(round((disp_delay(f,dm) - disp_delay(f0,dm))/dt))
#                for j in xrange(0,dat.shape[1] - delay_ind):
#                        ret[i,j] = dat[i,j + delay_ind]
#        return ret,int((disp_delay(700,dm) - disp_delay(900,dm))/dt)

    def plot_freq(self):
        di, ti = self.centre
        tside = 250
        dside = 300
        delta_t = self.data.delta_t
        delta_dm = self.data.delta_dm
        start_ti = max(0, ti - tside)
        end_ti = min(self.data.spec_data.shape[1], ti + tside)
        start_di = max(0, di - dside)
        end_di = min(self.data.spec_data.shape[0], di + dside)
        ret = np.zeros(self.data.spec_data.shape)
        df = -200.0/float(self.data.spec_data.shape[0])
        f0 = 900
	f1 = f0 + self.data.spec_data.shape[0]*df
	for i in xrange(0,self.data.spec_data.shape[0]):
                f = f0 + i*df
                dm = di*delta_dm
                delay_ind = int(round((disp_delay(f,dm) - disp_delay(f0,dm))/delta_t))
#        	self.data.spec[:,0:-delay_ind] = self.data.spec[:,delay_ind:]
                for j in xrange(0,self.data.spec_data.shape[1] - delay_ind):
                        ret[i,j] = self.data.spec_data[i,j + delay_ind]
	        
	rebin_factor_freq = 64
	rebin_factor_time = 1
	xlen = ret.shape[0] / rebin_factor_freq
	ylen = ret.shape[1] / rebin_factor_time
	new_freq_data = np.zeros((xlen,ylen))
	for i in range(xlen):
    		for j in range(ylen):
        		new_freq_data[i,j] = ret[i*rebin_factor_freq:(i+1)*rebin_factor_freq,j*rebin_factor_time:(j+1)*rebin_factor_time].mean()

        range_factor = 0.5
        range_start_ti = int((start_ti + end_ti)/2 - (end_ti - start_ti)*(range_factor)/2)
        range_end_ti = int((start_ti + end_ti)/2 + (end_ti - start_ti)*(range_factor)/2)

        freq_lower_std_index = 1
        freq_upper_std_index = 5
        freq_mean = np.mean(new_freq_data[:,range_start_ti:range_end_ti])
        freq_std = np.std(new_freq_data[:,range_start_ti:range_end_ti])
        freq_vmin = freq_mean - freq_lower_std_index * freq_std
        freq_vmax = freq_mean + freq_upper_std_index * freq_std

	plt.imshow(new_freq_data[:,range_start_ti:range_end_ti],
                   extent=[range_start_ti * delta_t, range_end_ti * delta_t, f1, f0
                           ],
	           vmin = freq_vmin, vmax = freq_vmax,
                   aspect='auto',
		   cmap = cm.Blues
                   )
        plt.xlabel("time (s)")
        plt.ylabel("Frequency (MHz)")
        plt.colorbar()

def basic(data, snr_threshold=5., min_dm=50.,spec_ind=None):
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
            triggers.append(Trigger(data, sample, snr,spec_ind=spec_ind,duration=duration))
    return triggers
