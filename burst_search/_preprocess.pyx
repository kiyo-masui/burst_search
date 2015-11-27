import numpy as np

cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport sqrt

np.import_array()


# These mush match prototypes in src/preprocess.h
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ctypedef np.int16_t int_t


# C prototypes.
cdef extern void clean_rows_2pass(DTYPE_t* vec, size_t nchan, size_t ndata)
cdef extern void remove_outliers_c(DTYPE_t* data, size_t nchan, size_t ntime, DTYPE_t sigma_cut)
cdef extern void remove_outliers_single(DTYPE_t* data, size_t ntime, DTYPE_t sigma_cut)
cdef extern void full_algorithm(DTYPE_t * data, size_t nfreq, size_t ntime, int block, DTYPE_t sigma_cut);

#cdef np.ndarray[ndim=1, dtype=DTYPE_t] mean_and_std(np.ndarray[ndim=1, dtype=DTYPE_t] data, Py_ssize_t length) nogil:
#    #if length == 0:
#    #    return 0.0, 0.0
#    cdef DTYPE_t s1, s2, val, mean, std, off
#    cdef np.ndarray[ndim=1, dtype=DTYPE_t] ret = np.empty((1,2), dtype=DTYPE_t)[1,:]
#    s1 = s2 = 0.0
#    off = data[0]

#    for kk in range(length):
#        val = data[kk] - off
#        s1 += val
#        s2 += val*val

#    mean = s1/float(length)
#    std = sqrt((s2 - (s1*s1)/float(length))/float((length - 1)))
#    ret[0] = mean
#    ret[1] = std
#    return ret

cdef do_remove(np.ndarray[ndim=2, dtype=DTYPE_t] data, sigma_threshold):
    remove_outliers_c(<DTYPE_t *> data.data, data.shape[0], data.shape[1], sigma_threshold)

#def test_filter(data, sigma_threshold, block = None):
#    if block == None:
#        block = data.shape[1]

#    for jj in xrange(data.shape[1]//block):
#        for ii in xrange(data.shape[0]):
#            pass

#def remove_outliers_old(np.ndarray[ndim=2, dtype=DTYPE_t] data, sigma_threshold, block=None):
#    """Flag outliers within frequency channels.
#    Replace outliers with that frequency's mean.
#    """

#    nfreq0 = data.shape[0]
#    ntime0 = data.shape[1]

#    if block is None:
#        block = ntime0

#    if ntime0 % block:
#        raise ValueError("Time axis must be divisible by block."
#                         " (ntime, block) = (%d, %d)." % (ntime0, block))

#    ntime = block
#    nfreq = nfreq0 * (ntime0 // block)

#    print data.shape
#    data.shape = (nfreq, ntime)
#    print data.shape

#    # To optimize cache usage, process one frequency at a time.
#    for ii in range(nfreq):
#        this_freq_data = data[ii,:]
#        d_copy = this_freq_data.copy()
#        mean = np.mean(this_freq_data)
#        std = np.std(this_freq_data)
#        print "orig: mean", mean, "std", std
#        remove_outliers_single(<DTYPE_t*> d_copy.data,ntime,sigma_threshold)
#        outliers = abs(this_freq_data - mean) > sigma_threshold * std
#        this_freq_data[outliers] = mean

#    data.shape = (nfreq0, ntime0)

def remove_outliers(np.ndarray[DTYPE_t, ndim=2] data, sigma_threshold, block=None):
    cdef Py_ssize_t cblock, nfreq, nfreq0, ntime, ntime0
    nfreq0 = data.shape[0]
    ntime0 = data.shape[1]

    #cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] data_c
    #data_c = np.ascontiguousarray(data, dtype=DTYPE)

    if block is None:
        block = ntime0

    # full_algorithm(&data_c[0,0],nfreq0,ntime0,block,sigma_threshold)
    full_algorithm(<DTYPE_t *> data.data,nfreq0,ntime0,block,sigma_threshold)


def remove_outliers_old(np.ndarray[ndim=2, dtype=DTYPE_t] data, sigma_threshold, block=None):
    """Flag outliers within frequency channels.
    Replace outliers with that frequency's mean.
    """
    cdef Py_ssize_t cblock, nfreq, nfreq0, ntime, ntime0
    nfreq0 = data.shape[0]
    ntime0 = data.shape[1]

    if block is None:
        block = ntime0

    if ntime0 % block:
        raise ValueError("Time axis must be divisible by block."
                         " (ntime, block) = (%d, %d)." % (ntime0, block))

    #ntime = block
    #nfreq = nfreq0 * (ntime0 // block)

    #data.shape[0] = nfreq
    #data.shape[1] = ntime

    cdef DTYPE_t mean, std
    cdef np.ndarray[ndim=2, dtype=DTYPE_t] this_block
    #cdef np.ndarray[ndim=1, dtype=np.uint8_t, cast=True] outliers
    #cdef np.ndarray[ndim=1, dtype=DTYPE_t] this_freq_data
    #cdef np.ndarray[ndim=1, dtype=DTYPE_t] tmp1
    #cdef np.ndarray[ndim=1, dtype=DTYPE_t] this_freq_data

    # To optimize cache usage, process one frequency at a time.
    cdef Py_ssize_t ii, jj, kk, c1, c2
    cblock = block

    #Sum of elements to the power 1 and 2, respectively
    cdef DTYPE_t s1, s2, val
    cdef DTYPE_t csigma_threshold = sigma_threshold
    #outliers = np.empty((1,ntime//block), dtype=bool)[0,:]
    for jj in range(ntime0//cblock):
        c1 = jj*cblock
        c2 = c1 + cblock
        this_block = data[:,c1:c2]
        do_remove(this_block, csigma_threshold)
        #for ii in range(nfreq0):
        #    this_freq_data = this_block[ii,:]
        #    mean = np.mean(this_freq_data)
        #    std = np.std(this_freq_data)
        #    #print "orig: mean", mean, "std", std
        #    remove_outliers_single(<DTYPE_t *> this_freq_data.data,cblock,sigma_threshold)

        #for ii in prange(nfreq0, nogil=True):
        #    this_freq_data = this_block[ii]
        #    #mean = np.mean(this_freq_data)
        #    #std = np.std(this_freq_data)
        #    #outliers = np.abs(this_freq_data - mean) > sigma_threshold * std
        #    #s1 = s2 = 0.0
        #    #for kk in range(cblock):
        #    #    val = this_freq_data[kk]
        #    #    s1 += val
        #    #    s2 += val*val

        #    #mean = s1/float(cblock)
        #    #std = s2/float(cblock) - mean*mean
        #    tmp1 = mean_and_std(this_freq_data,cblock)
        #    mean = tmp1[0]
        #    std = tmp1[1]
        #        #outliers = np.abs(this_freq_data - mean) > sigma_threshold * std
        #    for kk in range(cblock):
        #        val = this_freq_data[kk]
        #        condition = (float(abs(val - mean) > csigma_threshold))*mean
        #        this_freq_data[kk] = condition*mean + (1.0 - condition)*val

        #        #this_freq_data[outliers] = mean
    #data.shape = (nfreq0, ntime0)
    #data.shape[0] = nfreq0
    #data.shape[1] = ntime0


def remove_continuum_v2(np.ndarray[ndim=2, dtype=DTYPE_t] data):

    cdef int nfreq = data.shape[0]
    cdef int ntime = data.shape[1]

    clean_rows_2pass(
            <DTYPE_t *> data.data,
            nfreq,
            ntime,
            )


