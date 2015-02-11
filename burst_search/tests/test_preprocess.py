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


if __name__ == '__main__':
    unittest.main()
