import unittest

import numpy as np

from burst_search import dedisperse

class TestDedisperse(unittest.TestCase):

    def test_runs(self):
        """Just for checking that things compile and run."""

        data = np.zeros((20, 32), dtype=np.float32)
        dedisperse.dm_transform(data, None, 1000., 0.001, 900e6, -1e6)


if __name__ == '__main__':
    unittest.main()
