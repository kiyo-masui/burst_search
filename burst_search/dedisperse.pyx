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
        size_t nfreq, float freq0, float delta_f, float disp_ind)

cdef extern int  burst_dm_transform(DTYPE_t *indata1, DTYPE_t *indata2,
        CM_DTYPE_t *chan_map, DTYPE_t *outdata, size_t ntime1, int ntime2,
        float delta_t, size_t nfreq, float freq0, float delta_f, int depth, int jon)

cdef extern void burst_setup_channel_mapping(CM_DTYPE_t *chan_map, size_t nfreq,
        float freq0, float delta_f, int depth, float disp_ind)


DM_CONST = 4148.808
USE_JON_DD = False



def disp_delay(freq, dm, disp_ind=2.):
    """Compute the dispersion delay (s) as a function of frequency (MHz) and DM"""
    return DM_CONST * dm / (freq ** disp_ind)


def dm_transform(
        np.ndarray[ndim=2, dtype=DTYPE_t] data1 not None,
        np.ndarray[ndim=2, dtype=DTYPE_t] data2,
        float max_dm,
        float delta_t,
        float freq0,
        float delta_f,
        ):

    cdef int jon = USE_JON_DD
    cdef int nfreq = data1.shape[0]

    if data2 is None:
        data2 = np.empty(shape=(nfreq, 0), dtype=DTYPE)

    if data1.shape[0] != data2.shape[0]:
        msg = "Input data arrays must have same length frequency axes."
        raise ValueError(msg)

    cdef int ntime1 = data1.shape[1]
    cdef int ntime2 = data2.shape[1]

    cdef int depth = burst_depth_for_max_dm(max_dm, delta_t, nfreq, freq0,
                                            delta_f, 2.0)

    cdef int ndm =  burst_get_num_dispersions(nfreq, freq0, delta_f, depth)

    cdef np.ndarray[ndim=1, dtype=CM_DTYPE_t] chan_map
    chan_map = np.empty(2**depth, dtype=CM_DTYPE)
    burst_setup_channel_mapping(<CM_DTYPE_t *> chan_map.data, nfreq, freq0,
            delta_f, depth, 2.0)

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
    def disp_ind(self):
        return self._disp_ind

    @property
    def t0(self):
        return self._t0

    @property
    def spec_ind(self):
        return self._spec_ind

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

    def __init__(self, spec_data, dm_data, delta_t, freq0, delta_f, dm0,
            delta_dm, disp_ind=2., t0=0., spec_ind=0.):
        self._spec_data = spec_data
        self._dm_data = dm_data
        self._delta_t = delta_t
        self._freq0 = freq0
        self._delta_f = delta_f
        self._dm0 = dm0
        self._delta_dm = delta_dm
        self._disp_ind = disp_ind
        self._spec_ind = spec_ind
        self._t0 = t0

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

    @property
    def disp_ind(self):
        return self._disp_ind

    def __init__(self, delta_t, nfreq, freq0, delta_f, max_dm, disp_ind=2.):

        cdef float cdelta_t = delta_t
        cdef int cnfreq = nfreq
        cdef float cfreq0 = freq0
        cdef float cdelta_f = delta_f
        cdef float cmax_dm = max_dm

        cdef int depth = burst_depth_for_max_dm(cmax_dm, cdelta_t, cnfreq, cfreq0,
                                                cdelta_f, disp_ind)


        if 2**depth < nfreq:
            raise ValueError("""Choose higher max_dm and/or lower disp_ind; 
            Effective channels {0} must be greater than {1}""".format(2**depth, nfreq))
        else:
            print "using {0} effective channels with a memory footprint of {1} MB/s".format(2**depth,(4.0e-6)*(2**depth)/delta_t)
        cdef int cndm =  burst_get_num_dispersions(cnfreq, cfreq0, cdelta_f, depth)

        cdef np.ndarray[ndim=1, dtype=CM_DTYPE_t] chan_map
        chan_map = np.empty(2**depth, dtype=CM_DTYPE)
        burst_setup_channel_mapping(<CM_DTYPE_t *> chan_map.data, cnfreq, cfreq0,
                cdelta_f, depth, disp_ind)

        self._chan_map = chan_map

        self._delta_t = delta_t
        self._nfreq = nfreq
        self._freq0 = freq0
        self._delta_f = delta_f
        self._max_dm = max_dm
        self._ndm = cndm
        self._depth = depth
        self._disp_ind = disp_ind


    def __call__(self, np.ndarray[ndim=2, dtype=DTYPE_t] data1 not None,
            np.ndarray[ndim=2, dtype=DTYPE_t] data2=None):

        cdef int nfreq = self.nfreq

        disp_ind = self._disp_ind

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
        cdef int jon = USE_JON_DD

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
        delta_dm = calc_delta_dm(delta_t, freq0, freq0 + nfreq * delta_f,
                                 disp_ind)

        out_cont = DMData(spec_data, dm_data, delta_t, freq0, delta_f, dm0,
                          delta_dm, disp_ind=disp_ind)

        return out_cont


def calc_delta_dm(delta_t, f1, f2, disp_ind=2):
    return (delta_t / DM_CONST / abs(1. / f1**disp_ind - 1. / f2**disp_ind))



