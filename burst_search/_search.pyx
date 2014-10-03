import numpy as np

cimport numpy as np
cimport cython


np.import_array()


# These mush match prototypes in src/dedisperse_gbt.h
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


# C prototypes.
cdef extern size_t find_peak_wrapper(float *data, int nchan, int ndata,
        float *peak_snr, int *peak_channel, int *peak_sample, int *peak_duration)


def sievers_find_peak(data, exclude_0=True):

    cdef np.ndarray[ndim=2, dtype=DTYPE_t] dm_data
    cdef int ndm = data.dm_data.shape[0]
    cdef int ntime = data.dm_data.shape[1]

    cdef float peak_snr
    cdef int peak_dm
    cdef int peak_time
    cdef int peak_duration

    if exclude_0:
        dm_data = data.dm_data[1:,:]
        ndm -= 1
    else:
        dm_data = data.dm_data

    cdef size_t ret = find_peak_wrapper(
            <DTYPE_t *> dm_data.data,
            ndm,
            ntime,
            &peak_snr,
            &peak_dm,
            &peak_time,
            &peak_duration,
            )

    if exclude_0:
        peak_dm += 1

    return peak_snr, (peak_dm, peak_time), peak_duration
