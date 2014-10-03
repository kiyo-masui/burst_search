import numpy as np

cimport numpy as np
cimport cython


np.import_array()


# These mush match prototypes in src/dedisperse_gbt.h
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


# C prototypes.
cdef extern void burst_preprocess_something(DTYPE_t *data, int nfreq, int ntime)

def preprocess_sievers(np.ndarray[ndim=2, dtype=DTYPE_t] data):

    cdef int nfreq = data.shape[0]
    cdef int ntime = data.shape[1]

    burst_preprocess_something(
            <DTYPE_t *> data.data,
            nfreq,
            ntime,
            )


