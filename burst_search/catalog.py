import h5py
import os
from os import listdir
from os.path import isfile, join
import numpy as np

#time !
#dm !
#spec ind !
#width (index units) !
#fluence #


ind_t = np.int16
float_t = np.float32

dset_name = 'burst_metadata'
md_t = np.dtype([('file_name','S32'),('snr',float_t),('t_ind', ind_t),('dt', float_t),('dm_ind', ind_t),
	('ddm', float_t), ('spec_ind', float_t), ('t_width', ind_t), ('fluence', float_t),
	('loc',float_t,(2,)),
	])

resize_increment = 128


def ensure_dir(path):
	if not isfile(path):
		if not os.path.exists(path):
			os.makedirs(path)
	else:
		ensure_dir(os.sep.join(path.split(os.sep)[:-1]))

def ensure_structure(hfile):
	if not dset_name in hfile.keys():
		dset = hfile.create_dataset(dset_name,(resize_increment,1),maxshape=(None,1))
		dset.attrs['ind'] = 0
	else:
		return hfile[dset_name]


class Catalog(object):
	def __init__(self,path='burst_catalog.h5py',parent_name):
		ensure_dir(path)
		self._outpath = path
		self._parent_name = parent_name
		self._of = h5py.File(path,'a')
		#dset
		self._event_data = ensure_structure(self._of)

	def simple_write(triggers):
		for trig in triggers:
			dt = trig.data.delta_t
			ddm = trig.data.delta_dm
			t_ind = trig.centre[1] + trig.data_start_record
			dm_ind = trig.centre[0]
			snr = trig.snr
			spec_ind = trig.spec_ind
			t_width = trig.duration
			try:
				fluence = trig.fluence
			except:
				fluence = None
			self.write(snr,t_ind,dt,dm_int,ddm,spec_ind,t_width,fluence)

	def raw_write(snr,t_ind,dt,dm_ind,ddm,spec_ind,t_width,fluence,loc=(0,0)):
		if spec_ind == None: spec_ind = 0.0
		if fluence == None: fluence = 0.0
		dset = self._event_data
		ind = dset.attrs['ind']
		if ind > dset.len() - 1:
			dset.resize(dset.len() + resize_increment,axis=0)
		self._of.flush()
		dset[i] = np.array((self._parent_name,snr,t_ind,dt,dm_ind,ddm,spec_ind,t_width,fluence,loc),
			dtype=md_t)
		dset.attrs['ind'] = ind + 1

	def __del__(self):
		self._of.close()



