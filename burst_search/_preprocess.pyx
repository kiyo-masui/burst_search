import numpy as np

cimport numpy as np
cimport cython


np.import_array()


# These mush match prototypes in src/dedisperse_gbt.h
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


# C prototypes.
cdef extern void clean_rows_2pass(DTYPE_t *vec, size_t nchan, size_t ndata)


def remove_continuum_v2(np.ndarray[ndim=2, dtype=DTYPE_t] data):

    cdef int nfreq = data.shape[0]
    cdef int ntime = data.shape[1]

    clean_rows_2pass(
            <DTYPE_t *> data.data,
            nfreq,
            ntime,
            )


