import h5py
import time
import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np

ind_t = np.int16
float_t = np.float32

dset_name = 'burst_metadata'
md_t = np.dtype([('file_name','S32'),('run_time',float),('snr',float_t),('t_ind', ind_t),('dt', float_t),('dm_ind', ind_t),
	('ddm', float_t), ('spec_ind', float_t), ('t_width', ind_t), ('fluence', float_t), ('disp_ind', float_t),
	('loc',float_t,(2,)),
	])

resize_increment = 128


def ensure_dir(path):
	if path == '':
		return
	elif not isfile(path) and not '.' in path:
		if not exists(path):
			makedirs(path)
	else:
		ensure_dir(os.sep.join(path.split(os.sep)[:-1]))

def ensure_structure(hfile):
	if not dset_name in hfile.keys():
		dset = hfile.create_dataset(dset_name,(resize_increment,1),maxshape=(None,1),dtype=md_t)
		dset.attrs['ind'] = 0
		return dset
	else:
		return hfile[dset_name]


class Catalog(object):
	def __init__(self,parent_name,path='burst_catalog.h5py',run_time=time.time()):
		ensure_dir(path)
		self._outpath = path
		self._parent_name = parent_name
		self._run_time = run_time
		if isfile(path):
			self._of = h5py.File(path,'r+')
		else:
			self._of = h5py.File(path,'w')
		self._event_data = ensure_structure(self._of)

	def simple_write(self, triggers, disp_ind = 2.0):
		for trig in triggers:
			dt = trig.data.delta_t
			ddm = trig.data.delta_dm
			t_ind = trig.centre[1] + trig.data.start_record
			dm_ind = trig.centre[0]
			snr = trig.snr
			spec_ind = trig.spec_ind
			t_width = trig._duration
			try:
				fluence = trig.fluence
			except:
				fluence = None
			self.write(snr,t_ind,dt,dm_ind,ddm,spec_ind,t_width,fluence,disp_ind=disp_ind)

	def write(self, snr,t_ind,dt,dm_ind,ddm,spec_ind,t_width,fluence,disp_ind=2.0,loc=(0,0)):
		if spec_ind == None: spec_ind = 0.0
		if fluence == None: fluence = 0.0
		dset = self._event_data
		ind = dset.attrs['ind']
		if ind > dset.len() - 1:
			dset.resize(dset.len() + resize_increment,axis=0)
		self._of.flush()
		dset[ind] = np.array((self._parent_name,self._run_time,snr,t_ind,dt,dm_ind,ddm,spec_ind,t_width,fluence,disp_ind,loc),
			dtype=md_t)
		dset.attrs['ind'] = ind + 1

	def __del__(self):
		self._of.close()



