import numpy as np

cimport numpy as np
cimport cython


np.import_array()

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# C prototypes.
cdef extern int burst_get_num_dispersions(size_t nfreq, float freq0,
        float delta_f, int depth)

cdef extern int burst_depth_for_max_dm(float max_dm, float delta_t,
        size_t nfreq, float freq0, float delta_f)

cdef extern int  burst_dm_transform(DTYPE_t *indata1, DTYPE_t *indata2,
        DTYPE_t *outdata, size_t ntime1, int ntime2, float delta_t,
        size_t nfreq, float freq0, float delta_f, int depth)


def dm_transform(
        np.ndarray[ndim=2, dtype=DTYPE_t] data1 not None,
        np.ndarray[ndim=2, dtype=DTYPE_t] data2,
        float max_dm,
        float delta_t,
        float freq0,
        float delta_f,
        ):

    cdef int nfreq = data1.shape[1]

    if data2 is None:
        data2 = np.empty(shape=(0, nfreq), dtype=DTYPE)

    if data1.shape[1] != data2.shape[1]:
        msg = "Input data arrays must have same length frequency axes."
        raise ValueError(msg)

    cdef int ntime1 = data1.shape[0]
    cdef int ntime2 = data2.shape[0]

    cdef int depth = burst_depth_for_max_dm(max_dm, delta_t, nfreq, freq0,
                                            delta_f)

    cdef int ndm =  burst_get_num_dispersions(nfreq, freq0, delta_f, depth)

    cdef np.ndarray[ndim=2, dtype=DTYPE_t] out
    out = np.empty(shape=(ntime1, ndm), dtype=DTYPE)

    cdef int ntime_out = burst_dm_transform(
            <DTYPE_t *> data1.data,
            <DTYPE_t *> data2.data,
            <DTYPE_t *> out.data,
            ntime1,
            ntime2,
            delta_t,
            nfreq,
            freq0,
            delta_f,
            depth,
            )

    return out[:ntime_out,:]


