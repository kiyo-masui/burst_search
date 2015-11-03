import pyfits
import numpy as np
from multiprocessing import Queue

def read_records(hdulist, start_record=0, end_record=None):
	"""Read and format records from GUPPI PSRFITS file."""

	nrecords = len(hdulist[1].data)
	if end_record is None or end_record > nrecords:
		end_record = nrecords
	nrecords_read = end_record - start_record
	ntime_record, npol, nfreq, one = hdulist[1].data[0]["DATA"].shape

	out_data = np.empty((nfreq, nrecords_read, ntime_record), dtype=np.float32)
	for ii in xrange(nrecords_read):
		# Read the record.
		record = hdulist[1].data[start_record + ii]["DATA"]
		# Interpret as unsigned int (for Stokes I only).
		record = record.view(dtype=np.uint8)
		# Select stokes I and copy.
		out_data[:,ii,:] = np.transpose(record[:,0,:,0])
	out_data.shape = (nfreq, nrecords_read * ntime_record)

	return out_data

class FileSource(object):
	def __init__(self, fitsfile):
		self._filename = fitsfile
	
	def get_records(self,start,end,scrunch=1):
		hdulist = pyfits.open(self._filename, 'readonly')
		data = read_records(hdulist, start, end)
		hdulist.close()
		return data

class ScrunchFileSource(object):
	def __init__(self, fitsfile, nscrunch):
		self._filename = fitsfile
		self._nscrunch = nscrunch
		self._dq = [Queue() for i in xrange(0,nscrunch - 1)]
	
	def get_records(self,start,end,scrunch):
		if scrunch == 1:
			hdulist = pyfits.open(self._filename, 'readonly')
			data = read_records(hdulist, start, end)
			last = data

			self._current_recs = [start,end]
			for i in xrange(1,self._nscrunch):
				thisdat = last[0::2] + last[1::2]
				thisq = self._dq[i - 1]
				while not thisq.empty():
					dat = thisq.get(timeout=0.1)
					del dat
				thisq.put(thisdat)
				last = thisdat
			hdulist.close()
			return data
		else:
			if start < self._current_recs[0]:
				raise Exception("Data no longer available")
			return self._dq[i - 2].get()