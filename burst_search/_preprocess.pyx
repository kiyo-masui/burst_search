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
cdef extern void full_algorithm(DTYPE_t * data, size_t nfreq, size_t ntime, int block, DTYPE_t sigma_cut)

cdef do_remove(np.ndarray[ndim=2, dtype=DTYPE_t] data, sigma_threshold):
    remove_outliers_c(<DTYPE_t *> data.data, data.shape[0], data.shape[1], sigma_threshold)

def remove_outliers(np.ndarray[DTYPE_t, ndim=2] data, sigma_threshold, block=None):
    """Flag outliers within frequency channels.
    Replace outliers with that frequency's mean.
    Computed an unbiased estimate for variance.
    """
    cdef Py_ssize_t cblock, nfreq, nfreq0, ntime, ntime0
    nfreq0 = data.shape[0]
    ntime0 = data.shape[1]

    if block is None:
        block = ntime0

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

    cdef DTYPE_t mean, std
    cdef np.ndarray[ndim=2, dtype=DTYPE_t] this_block

    # To optimize cache usage, process one frequency at a time.
    cdef Py_ssize_t ii, jj, kk, c1, c2
    cblock = block

    #Sum of elements to the power 1 and 2, respectively
    cdef DTYPE_t s1, s2, val
    cdef DTYPE_t csigma_threshold = sigma_threshold

    for jj in range(ntime0//cblock):
        c1 = jj*cblock
        c2 = c1 + cblock
        this_block = data[:,c1:c2]
        do_remove(this_block, csigma_threshold)


def remove_continuum_v2(np.ndarray[ndim=2, dtype=DTYPE_t] data):

    cdef int nfreq = data.shape[0]
    cdef int ntime = data.shape[1]

    clean_rows_2pass(
            <DTYPE_t *> data.data,
            nfreq,
            ntime,
            )


