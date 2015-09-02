import numpy as np

cimport numpy as np
cimport cython


np.import_array()


# These must match prototypes in src/dedisperse.h
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

CM_DTYPE = np.int64
ctypedef np.int64_t CM_DTYPE_t

# C prototypes.
cdef extern int burst_get_num_dispersions(size_t nfreq, float freq0,
        float delta_f, int depth)

cdef extern int burst_depth_for_max_dm(float max_dm, float delta_t,
        size_t nfreq, float freq0, float delta_f)

cdef extern int  burst_dm_transform(DTYPE_t *indata1, DTYPE_t *indata2,
        CM_DTYPE_t *chan_map, DTYPE_t *outdata, size_t ntime1, int ntime2,
        float delta_t, size_t nfreq, float freq0, float delta_f, int depth,int jon)

cdef extern void burst_setup_channel_mapping(CM_DTYPE_t *chan_map, size_t nfreq,
        float freq0, float delta_f, int depth)


def dm_transform(
        np.ndarray[ndim=2, dtype=DTYPE_t] data1 not None,
        np.ndarray[ndim=2, dtype=DTYPE_t] data2,
        float max_dm,
        float delta_t,
        float freq0,
        float delta_f,
        ):

    cdef int jon = 0
    cdef int nfreq = data1.shape[0]

    if data2 is None:
        data2 = np.empty(shape=(nfreq, 0), dtype=DTYPE)

    if data1.shape[0] != data2.shape[0]:
        msg = "Input data arrays must have same length frequency axes."
        raise ValueError(msg)

    cdef int ntime1 = data1.shape[1]
    cdef int ntime2 = data2.shape[1]

    cdef int depth = burst_depth_for_max_dm(max_dm, delta_t, nfreq, freq0,
                                            delta_f)

    cdef int ndm =  burst_get_num_dispersions(nfreq, freq0, delta_f, depth)

    cdef np.ndarray[ndim=1, dtype=CM_DTYPE_t] chan_map
    chan_map = np.empty(2**depth, dtype=CM_DTYPE)
    burst_setup_channel_mapping(<CM_DTYPE_t *> chan_map.data, nfreq, freq0,
            delta_f, depth)

    cdef np.ndarray[ndim=2, dtype=DTYPE_t] out
    out = np.empty(shape=(ndm, ntime1), dtype=DTYPE)

    cdef int ntime_out = burst_dm_transform(
            <DTYPE_t *> data1.data,
            <DTYPE_t *> data2.data,
            <CM_DTYPE_t *> chan_map.data,
            <DTYPE_t *> out.data,
            ntime1,
            ntime2,
            delta_t,
            nfreq,
            freq0,
            delta_f,
            depth,
            jon,
            )

    out = np.ascontiguousarray(out[:,:ntime_out])

    return out


class DMData(object):
    """Container for spectra and DM space data.

    """

    @property
    def spec_data(self):
        return self._spec_data

    @property
    def dm_data(self):
        return self._dm_data

    @property
    def delta_t(self):
        return self._delta_t

    @property
    def freq0(self):
        return self._freq0

    @property
    def delta_f(self):
        return self._delta_f

    @property
    def dm0(self):
        return self._dm0

    @property
    def delta_dm(self):
        return self._delta_dm

    @property
    def nfreq(self):
        return self.spec_data.shape[0]

    @property
    def ndm(self):
        return self.dm_data.shape[0]

    @property
    def freq(self):
        return self.freq0 + self.delta_f * np.arange(self.nfreq, dtype=float)

    @property
    def dm(self):
        return self.dm0 + self.delta_dm * np.arange(self.ndm, dtype=float)

    @property
    def spec_ind(self):
        return self._spec_ind

    @spec_ind.setter
    def spec_ind(self, value):
        self._spec_ind = value

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, value):
        self._t0 = value

    def __init__(self, spec_data, dm_data, delta_t, freq0, delta_f, dm0, delta_dm):
        self._spec_data = spec_data
        self._dm_data = dm_data
        self._delta_t = delta_t
        self._freq0 = freq0
        self._delta_f = delta_f
        self._dm0 = dm0
        self._delta_dm = delta_dm

        self.spec_ind = 0
        self.t0 = 0.

    @classmethod
    def from_hdf5(cls, group):
        delta_t = group.attrs['delta_t']
        freq0 = group.attrs['freq0']
        delta_f = group.attrs['delta_f']
        dm0 = group.attrs['dm0']
        delta_dm = group.attrs['delta_dm']
        spec_data = group['spec_data'][:]
        dm_data = group['dm_data'][:]
        return cls(spec_data, dm_data, delta_t, freq0, delta_f, dm0, delta_dm)


    def to_hdf5(self, group):
        group.attrs['delta_t'] = self.delta_t
        group.attrs['freq0'] = self.freq0
        group.attrs['delta_f'] = self.delta_f
        group.attrs['dm0'] = self.dm0
        group.attrs['delta_dm'] = self.delta_dm
        group.create_dataset('spec_data', data=self.spec_data)
        group.create_dataset('dm_data', data=self.dm_data)
        

class DMTransform(object):
    """Performs dispersion measure transforms."""

    @property
    def delta_t(self):
        return self._delta_t

    @property
    def nfreq(self):
        return self._nfreq

    @property
    def freq0(self):
        return self._freq0

    @property
    def delta_f(self):
        return self._delta_f

    @property
    def max_dm(self):
        return self._max_dm

    @property
    def ndm(self):
        return self._ndm

    @property
    def depth(self):
        return self._depth

    def __init__(self, delta_t, nfreq, freq0, delta_f, max_dm, jon=False):

        cdef float cdelta_t = delta_t
        cdef int cnfreq = nfreq
        cdef float cfreq0 = freq0
        cdef float cdelta_f = delta_f
        cdef float cmax_dm = max_dm

        cdef int depth = burst_depth_for_max_dm(cmax_dm, cdelta_t, cnfreq, cfreq0,
                                                cdelta_f)

        cdef int cndm =  burst_get_num_dispersions(cnfreq, cfreq0, cdelta_f, depth)

        cdef np.ndarray[ndim=1, dtype=CM_DTYPE_t] chan_map
        chan_map = np.empty(2**depth, dtype=CM_DTYPE)
        burst_setup_channel_mapping(<CM_DTYPE_t *> chan_map.data, cnfreq, cfreq0,
                cdelta_f, depth)
        self._chan_map = chan_map
        self._delta_t = delta_t
        self._nfreq = nfreq
        self._freq0 = freq0
        self._delta_f = delta_f
        self._max_dm = max_dm
        self._ndm = cndm
        self._depth = depth
        self._jon = jon


    def __call__(self, np.ndarray[ndim=2, dtype=DTYPE_t] data1 not None,
            np.ndarray[ndim=2, dtype=DTYPE_t] data2=None):

        cdef int nfreq = self.nfreq

        if data2 is None:
            data2 = np.empty(shape=(nfreq, 0), dtype=DTYPE)

        if data1.shape[0] != data2.shape[0] or data1.shape[0] != self.nfreq:
            msg = ("Input data arrays must have frequency axes with length"
                   " nfreq=%d." % self.nfreq)
            raise ValueError(msg)

        cdef int ntime1 = data1.shape[1]
        cdef int ntime2 = data2.shape[1]

        cdef float delta_t = self.delta_t
        cdef float freq0 = self.freq0
        cdef float delta_f = self.delta_f
        cdef int ndm = self.ndm
        cdef int depth = self.depth
        cdef int jon = self._jon

        cdef np.ndarray[ndim=1, dtype=CM_DTYPE_t] chan_map
        chan_map = self._chan_map

        cdef np.ndarray[ndim=2, dtype=DTYPE_t] out
        out = np.empty(shape=(ndm, ntime1), dtype=DTYPE)

        cdef int ntime_out = burst_dm_transform(
                <DTYPE_t *> data1.data,
                <DTYPE_t *> data2.data,
                <CM_DTYPE_t *> chan_map.data,
                <DTYPE_t *> out.data,
                ntime1,
                ntime2,
                delta_t,
                nfreq,
                freq0,
                delta_f,
                depth,
                jon,
                )

        dm_data = np.ascontiguousarray(out[:,:ntime_out])
        spec_data = np.ascontiguousarray(data1[:, :ntime_out])

        dm0 = 0
        delta_dm = (delta_t / 4148.8
                    / abs(1. / freq0**2 - 1. / (freq0 + nfreq * delta_f)**2))

        out_cont = DMData(spec_data, dm_data, delta_t, freq0, delta_f, dm0,
                          delta_dm)

        return out_cont



