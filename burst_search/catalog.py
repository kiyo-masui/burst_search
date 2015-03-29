import h5py
import numpy as np
from multiprocessing import Pool, Lock
import hashlib
import os.path
from search import Trigger
#probably not well-organized
from guppi import SearchSpec
from guppi import FileSpec
from simulate import SimEvent

class Catalog(object):
	"""
	Catalogs each FRB real/simulated event by time and metadata, along with raw/dedispersed data.
	Asynchronous design ensures nonblocking. Additonally, this class is designed such that it may write
	results from multiple analysis threads simultaneously (hopefully of use to batch analysis).

	"""

	_reference_name = {Trigger:'/triggers', SimEvent:'/sim_events', SearchSpec:'/searches'}

	# the key set of this dict must be a subset of the key set of the metafile
	# in future, could/should load from a configuration file or even a hosted file specifying standards/versions
	# of our internal-use hdf5 structure
	_md_structure = {'/triggers':('dset',{'maxshape':(None,), 'dtype':Trigger.dtype},{}), '/sim_events':('dset',{'maxshape':(None,), 'dtype':SimEvent.dtype},{}),
					'/searches':('dset',{'maxshape':(None,), 'dtype':SearchSpec.dtype},{}),'/files':('dset',{'maxshape':(None,), 'dtype':FileSpec.dtype})}
	#_d_structure = 

	def __init__(self, base_path=None):
		md_name = 'gbt_frb_metadata.hdf5'
		d_name = 'gbt_frb_data.hdf5'
		if base_path != None:
			if not base_path[-1] == '/':
				base_path += '/'
			md_name = base_path + md_name
			d_name = base_path + md_name
		
		self._meta_file = h5py.File(md_name,'rw')
		self._data_file = h5py.File(d_name,'rw')
		self._search = None
		self._last_record = 0
		self._locks = {_meta_file:Lock(),_data_file:Lock()}

		structure_check()

		self._write_pool = Pool(processes=3)

	#Core functionality
	#------------------------------------------

	#def set_search(self,search,start_record):
	#	self._search = search
	#	self._last_record = last_record

	#def add_trigger(self,trigger_params):

	#def advance_search(self,record):

	#def add_sim_event(self,sim_event_params):

	#------------------------------------------

	def structure_check(self):
		"""Verify that the metadata and data files are of correct format. Add format if the files are new."""
		for l in self.locks: l.acquire()

		meta_file = self._meta_file

		paths = flat_paths(meta_file)

		#Metadata
		if len(meta_file.keys()) == 0:
			#empty metafile, make meta structure
			make_hdf5_structure(meta_file,_md_structure)

		data_file = self._data_file
		if len(data_file.keys()) == 0:
			make_hdf5_structure(meta_file,_md_structure)

		#allows for extra data in file format
		if not set(_md_structure.keys()).issubset(paths):
			raise ValueError('Metafile format invalid: restore metafile or move/rename the meta+data files')

		microstructure_check(meta_file,_md_structure)
		microstructure_check(data_file,_data_structure)

		for l in self.locks: l.release() 

	#def make_meta_structure(self):
	#	"""
	#	Structure the hdf5 files due to an empty metadata file.
	#	Note that in the case of an empty metadata file the data file
	#	will either be a) empty -> requires structure or b) nonempty but orphaned ->
	#	will be copied and a new data file will be created and structured.
	#	"""
	#	self._meta_file

	def write(self,catalogable,hfile,owrite=False):
		self._write_pool.apply_async(target=self, args=(catalogable, hfile, owrite))

	def write_metadata(self,catalogable,owrite=False):
		self._write_pool.apply_async(target=self, args=(catalogable,self._meta_file, owrite))

	def write_data(self,catalogable,owrite=False):
		self._write_pool.apply_async(target=self, args=(catalogable,self._data_file, owrite))

	def __call__(self,catalogable,hfile,owrite):
		#An inelegant solution
		self.do_write(catalogable,hfile,owrite)

	def do_write(self,catalogable,hfile,owrite):
		self._locks[hfile].acquire()
		dset = hfile[_reference_name[catalogable]]
		if owrite:
			primary_key = catalogable.primary_key()
			for i in xrange(0,dset):
				if elem[i][0] == primary_key:
					elem[i] = catalogable.row_value()
					self._locks[hfile].release()
					return
		else:
			l = len(dset)
			dset.resize(l + 1,)
			dset[l] = catalogable.row_value()
			self._locks[hfile].release()

def microstructure_check(hfile,structure):
	"""Raises an exception if the hdf5 file has structure inconsistent with that specified by the structure file."""
	for k in structure.keys():
			entry = meta_file[k]
			if isinstance(entry, h5py._hl.dataset.Dataset):
				if not entry.dtype == structure[k]['dtype']:
					raise ValueError('Incorrect or unfamiliar dtype in dset ' + entry.name + ' in file ' + hfile.name)

def make_hdf5_structure(hfile,structure):
	"""Add the specified structure to an already open h5py file. Not threadsafe (i.e. must lock the file elsewhere)"""
	for k in structure.keys():
		elem = structure[k]
		param_dict  = elem[1]
		attr_dict = elem[2]
		if elem[0] == 'dset':
			h_elem = hfile.create_dataset(name=k,shape=(0,),maxshape=param_dict['maxshape'],dtype=param_dict['dtype'])
		elif elem[0] == 'group':
			h_elem = hfile.create_group(name=k)
		else:
			raise ValueError('Invalid element type ' + elem[0] + ' in structure specification')
		for j in attr_dict.keys():
			h_elem[j] = attr_dict[j]

# recursively flatten the hierarchy of an hdf5 file
def flat_paths(group, paths = []):
	for elem in group.values():
		paths.append(elem.name)
		if isinstance(elem,h5py._hl.group.Group):
			flat_paths(elem,paths)
	return paths

class Catalogable(object):
	def dtype(self):
        		raise NotImplementedError()

	def row_value(self):
		raise NotImplementedError()

	def primary_key(self):
		raise NotImplementedError()
