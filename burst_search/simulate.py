import random
import math
import uuid
import numpy as np

# the number of t_sd's apart each simulated event must be, at a minimum
exclusion_sd =  100

#needs to be included in some telescope/dataset parameter file
gain = 2.000 # K/Jy

class RandSource(object):
	"""
	An object to generate uniformly distributed random f-t data with dispersion.

	The RandSource generates rectangular events in intensity-frequency-time space
	with properties each evenly distributed within a range that is determined at instantiation.
	"""

	

	def __init__(self,alpha=0.0,**kwargs):
		"""
		Initialize a RandSource object

		Parameters
		--------------------
		event_rate : float (Hz)
			The average (simulated) event rate.
		f_m : float (Mhz)
			Center value of frequency. Usually corresponds to maximum intensity.
		f_sd : float
			Either half of the width of the frequency distribution or one SD.
		bw_m : float (MHz)
			The simulated source bandwidth mean.
		bw_sd : float (MHz)
			The simulated source bandwidth SD or half-width.
		t_m : float (s)
			Characteristic timescale or half-width of a burst (time-domain).
		t_sd : float (s)
			Simulated burst width SD.
		s_m : float (Jy)
			Peak intensity of the burst.
		s_sd : float(Jy)
			Intensity SD.
		dm_m : float (pc cm^-2)
			Mean burst DM.
		dm_sd : float (pc cm^-2)
			Simulated burst DM SD.
		file_params : dict
			FITS file header information, with an additional entry 'nrecords.'
		t_overlap : float (s)
			Frame overlap time, as per guppi.py.
		nrecords_block : int
			Number of records per block.
		"""
		# user-provided
		self.event_rate = kwargs['event_rate']
		self.f_m = kwargs['f_m']
		self.f_sd = kwargs['f_sd']
		self.bw_m = kwargs['bw_m']
		self.bw_sd = kwargs['bw_sd']
		self.t_m = kwargs['t_m']
		self.t_sd = kwargs['t_sd']
		self.s_m = kwargs['s_m']
		self.s_sd = kwargs['s_sd']
		self.dm_m = kwargs['dm_m']
		self.dm_sd = kwargs['dm_sd']
		self.file_params = kwargs['file_params']
		self.t_overlap = kwargs['t_overlap']
		self.nrecords_block = kwargs['nrecords_block']
		self.alpha = alpha

		# for convenience
		self.ntime_record = self.file_params['ntime_record']
		self.delta_t = self.file_params['delta_t']
		self.nfreq = self.file_params['nfreq']
		self.nrecords = self.file_params['nrecords']

		# derived quantities
		self.file_time = self.nrecords*self.ntime_record*self.file_params['delta_t']
		self.nevents = self.event_rate*self.file_time
		self.record_time = self.ntime_record*self.delta_t
		self.ntime_block = self.ntime_record*self.nrecords_block

		# Note that this line depends on the stat model used to generate simulated params. Right now things are simple with defined extremal values
		# for all params
		self.max_twidth_event = disp_delay(self.f_m - self.f_sd - 0.5*(self.bw_m + self.bw_sd),self.dm_m + self.dm_sd) \
		- disp_delay(self.f_m + self.f_sd + 0.5*(self.bw_m + self.bw_sd),self.dm_m - self.dm_sd)

		# create the schedule of events, in terms of time index (absolute from start of file)
		self.make_event_schedule()

	def generate_params(self):
		"""Generate the parameters for a given burst using an evenly distributed probability density function"""
		params = {}
		params['f_center'] = uniform_range(self.f_m,self.f_sd)
		params['bw'] = uniform_range(self.bw_m,self.bw_sd)
		params['t_width'] = uniform_range(self.t_m,self.t_sd)
		# perform unit conversion from Jy to K to match calibration
		params['s_max'] = gain*uniform_range(self.s_m,self.s_sd)
		params['dm'] = uniform_range(self.dm_m,self.dm_sd)

		return params

	def generate_events(self, block_ind):
		"""
		Generates events in a given block.
		Returns a numpy matrix that can be added to the real, callibrated data
		"""
		block_events = filter(lambda event: self.ntime_block*block_ind < event < self.ntime_block*(block_ind + 1), self.event_schedule)
		sim_dat = np.zeros((self.nfreq,self.ntime_block),dtype=np.float32)

		for i in block_events:
			ntime_block = self.ntime_block
			t_start = i % ntime_block
			params = self.generate_params()

			# make undispersed data
			delta_f = self.file_params['delta_f']
			nfreq = self.file_params['nfreq']
			f0 = self.file_params['freq0']
			#f_min = params['f_center'] - 0.5*params['bw']
			#f_max = params['f_center'] + 0.5*params['bw']
			f_min = f0 - nfreq*delta_f
			f_max = f0
			delta_t = self.file_params['delta_t']
			nt_width = int(math.ceil(params['t_width']/delta_t))
			# note that nf_min > nf_max since this is the index corresponding to min freq

			center_f = f0 + 0.5*nfreq*delta_f

			nf_min = int(round((f_min - f0/delta_f)))
			nf_max = int(round((f_max - f0)/delta_f))
			dm = params['dm']
			# fill rectangular region (with endpoint considerations if sim signal is cutoff)

			for j in xrange(0,nfreq):
				sim_dat[j,t_start:t_start + nt_width] = math.pow(((f0 + j*delta_f)/center_f), self.alpha)*params['s_max']

			# disperse data in time
			for j in xrange(0,sim_dat.shape[0]):
				nt_disp = round(disp_delay(f0 + j*delta_f,dm)/delta_t)
				sim_dat[j,nt_disp:ntime_block] = sim_dat[j,0:ntime_block - nt_disp]
				sim_dat[j,0:nt_disp] = 0.0

			# rudamentary for eval
			print "Sim Event {0} at  t = {1} s block time = {4} with dm {2} s_max {3} alpha {5}".format(i,i*delta_t,dm,params['s_max'],(i%ntime_block)*delta_t,self.alpha)

		return sim_dat
	def make_event_schedule(self):
		"""
		Schedule simulated events using an even distribution, the chosen event rate, and relative event spacing constraints.
		Note that the event time refers to the beginning of the pulse i.e. peak - half width.
		Event times are referenced in terms of time index from the beginning of the file.
		"""
		#compute the time-index-unit exclusion radius about each event 
		delta_t = self.delta_t
		exclusion_radius = max(max(int(math.ceil(exclusion_sd*self.t_sd/delta_t)),int(math.ceil(1.2*self.max_twidth_event/delta_t))),self.ntime_block)

		#this line requires that we must have a constant number of records per block
		overlap_threshold = self.ntime_block - int(math.ceil((self.t_overlap + self.max_twidth_event)/delta_t))
		#print int(math.ceil(self.event_rate*self.file_time))
		#print self.max_twidth_event
		#print self.ntime_block*delta_t

		if overlap_threshold <= 0:
			raise ValueError('Sim event parameters chosen such that events are too large to fit in a block.')

		self.event_schedule = [None]*int(math.ceil(self.event_rate*self.file_time))
		i = 0

		# this structure is very inefficient when the dispersed length (time) is close to the size of the block
		# needs to be updated
		while i < len(self.event_schedule):
			#print 'loop no action'
			ntime_file = self.ntime_record*self.nrecords
			neffective_file_time = int(round(float(ntime_file) - (self.ntime_block - overlap_threshold)*float(self.nrecords)/float(self.nrecords_block)))
			effective_ind = int(random.random()*(neffective_file_time))
			event_ind = effective_ind + (self.ntime_block - overlap_threshold)*(effective_ind - effective_ind%overlap_threshold)/effective_ind
			# print 'event_ind: {0}, overlap_threshold: {1}, ntime_block: {2}'.format(event_ind%self.ntime_block,overlap_threshold,self.ntime_block)
			# print map(lambda x: abs(x - event_ind) < exclusion_radius, self.event_schedule[0:i])
			if event_ind%self.ntime_block < overlap_threshold and (i < 1 or not True in map(lambda x: abs(x - event_ind) < exclusion_radius , self.event_schedule[0:i])):
				self.event_schedule[i] = event_ind
				i += 1
				# print 'loop {0}'.format(i)
		# does not take advantage of sorted list yet
		self.event_schedule = sorted(self.event_schedule)
		return
	def coarse_event_schedule(self):
		"""Return the block index for all blocks with scheduled events"""
		return set(map(lambda ind: int(ind/self.ntime_block), self.event_schedule))

def disp_delay(f,dm):
	"""Compute the dispersion delay (s) as a function of frequency (MHz)"""
	return 4.149*dm*(10.0**3)/(f**2)

def uniform_range(center, halfwidth):
	return random.uniform(center - halfwidth, center + halfwidth)