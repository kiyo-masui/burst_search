import h5py
import numpy as np
from multiprocessing import Pool, Lock
import os.path

class Catalog(object):
	"""
	Catalogs each FRB real/simulated event by time and metadata, along with raw/dedispersed data

	"""


	def __init__(self, base_path=None):
		exists = os.path.exists()
		md_name = 'gbt_frb_metadata.hdf5'
		d_name = 'gbt_frb_data.hdf5'
		if base_path != None:
			if not base_path[-1] == '/'
				base_path += '/'
			md_name = base_path + md_name
			d_name = base_path + md_name
		
		self._meta_file = h5py.File(md_name,'rw')
		self._data_file = h5py.File(d_name,'rw')
		structure_check()
		self._search = None
		self._last_record = 0
		self._md_lock = Lock()

		self._write_pool = Pool(processes=3)

	#Core functionality
	#------------------------------------------

	def set_search(self,search,start_record):
		self._search = search
		self._last_record = last_record

	def add_trigger(self,trigger_params):

	def advance_search(self,record):

	def add_sim_event(self,sim_event_params):

	#------------------------------------------

	def structure_check(self):
		"""Verify that the metadata and data files are of correct format. Add format if the files are new"""
		#Metadata


		#Data


	def write_datum(self,datum):
		self._write_pool.apply_async(target=do_write_datum,args=(datum,))

	def do_write_datum(self,datum):
		self._d_lock.acquire()
			#do write
		self._d_lock.release()

	def write_trigger(self,trigger_data):
		self._write_pool.apply_async(target=do_write_trigger,args=(trigger_data,))

	def do_write_trigger(self,trigger_data):
		self._md_lock.acquire()
			#do write
		self._md_lock.release()


	def set_search(self,search):
		self._search = search

class Search(object):
	def __init__(self):
		