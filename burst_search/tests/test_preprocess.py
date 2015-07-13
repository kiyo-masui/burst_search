import unittest

import numpy as np

from burst_search import preprocess

class TestPeriodic(unittest.TestCase):

    def test_sin(self):
        """Check that sin function is zeroed."""

        period = 32
        ntime = 200
        nfreq = 7
        data = np.sin(np.arange(ntime) / period * 2 * np.pi) + 5
        data = data[:] * np.arange(1, 1 + nfreq)[:,None]
        right_profile = data[:,:period].copy()

        profile = preprocess.remove_periodic(data, period)
        self.assertTrue(np.allclose(data, 0))
        self.assertTrue(np.allclose(profile, right_profile))


class TestHighpass(unittest.TestCase):
    
    def test_ramp(self):

        ntime = 1000
        data = np.arange(5 * ntime)
        data.shape = (5, ntime)

        data_filt = preprocess.highpass_filter(data, 100)
        self.assertTrue(np.allclose(data_filt, 0))

    def test_high(self):
        ntime = 10000
        data = np.arange(5 * ntime)
        data.shape = (5, ntime)
        data = np.cos(data)

        data_filt = preprocess.highpass_filter(data, 1000)
        nlost = ntime - data_filt.shape[-1]

        self.assertTrue(np.allclose(data_filt, data[:,nlost//2:-nlost//2],
                atol=1e-5))

    def test_low(self):
        ntime = 1000000
        data = np.arange(5 * ntime)
        data.shape = (5, ntime)
        data = np.cos(data / 200000.)

        data_filt = preprocess.highpass_filter(data, 100)
        nlost = ntime - data_filt.shape[-1]

        self.assertTrue(np.allclose(data_filt, 0,
                atol=1e-5))


if __name__ == '__main__':
    unittest.main()
