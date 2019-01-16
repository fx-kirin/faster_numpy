import os
import sys
import unittest

import bottleneck as bn
import faster_numpy.clibrary
import faster_numpy.cylib
import numpy as np
from benchmarker import Benchmarker
from stl.mesh import Mesh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFasterNumpy(unittest.TestCase):
    def test_shift(self):
        a = np.arange(10.0)
        faster_numpy.cylib.shift(a, 1)
        for x in np.arange(9.0):
            self.assertEqual(x + 1, a[int(x)])
        a = np.arange(10.0)
        faster_numpy.cylib.shift(a, -1)
        for x in np.arange(9.0):
            self.assertEqual(x, a[int(x) + 1])
        a = np.arange(10.0)
        faster_numpy.clibrary.shift(a, 1)
        for x in np.arange(9.0):
            self.assertEqual(x + 1, a[int(x)])
        a = np.arange(10.0)
        faster_numpy.clibrary.shift(a, -1)
        for x in np.arange(9.0):
            self.assertEqual(x, a[int(x) + 1])

        a = np.arange(100000.0)
        with Benchmarker(10000, width=40) as bench:
            @bench("numpy.roll")
            def _(bm):
                for i in bm:
                    np.roll(a, 1)

            @bench("faster_numpy.cylib.shift")
            def _(bm):
                for i in bm:
                    faster_numpy.cylib.shift(a, 1)

            @bench("faster_numpy.clibrary.shift")
            def _(bm):
                for i in bm:
                    faster_numpy.clibrary.shift(a, 1)

    def test_sum(self):
        a = np.arange(10.0)
        self.assertEqual(faster_numpy.clibrary.sum(a), np.sum(a))
        a = np.arange(1000.0)
        with Benchmarker(100000, width=50) as bench:
            @bench("numpy.sum")
            def _(bm):
                for i in bm:
                    np.sum(a)

            @bench("faster_numpy.clibrary.sum")
            def _(bm):
                for i in bm:
                    faster_numpy.clibrary.sum(a)

            @bench("faster_numpy.clibrary.sum partial")
            def _(bm):
                for i in bm:
                    faster_numpy.clibrary.sum(a[0:999])

    def test_std(self):
        a = np.arange(10.0)
        self.assertEqual(faster_numpy.clibrary.std(a), np.std(a))
        a = np.arange(1000.0)
        with Benchmarker(100000, width=50) as bench:
            @bench("numpy.std")
            def _(bm):
                for i in bm:
                    np.std(a)

            @bench("faster_numpy.clibrary.std")
            def _(bm):
                for i in bm:
                    faster_numpy.clibrary.std(a)

            @bench("bottleneck.nanstd")
            def _(bm):
                for i in bm:
                    bn.nanstd(a)

    def test_mean(self):
        a = np.arange(10.0)
        mean_value = faster_numpy.cylib.mean(a)
        self.assertEqual(faster_numpy.cylib.mean(a), np.mean(a))
        self.assertEqual(faster_numpy.clibrary.mean(a), np.mean(a))
        faster_numpy.clibrary.mean(a)
        a = np.arange(1000.0)
        with Benchmarker(100000, width=50) as bench:
            @bench("numpy.mean")
            def _(bm):
                for i in bm:
                    np.mean(a)

            @bench("bottleneck.nanmean")
            def _(bm):
                for i in bm:
                    bn.nanmean(a)

            @bench("faster_numpy.cylib.mean")
            def _(bm):
                for i in bm:
                    faster_numpy.cylib.mean(a)

            @bench("faster_numpy.clibrary.mean")
            def _(bm):
                for i in bm:
                    faster_numpy.clibrary.mean(a)

            @bench("faster_numpy.clibrary.mean partial")
            def _(bm):
                for i in bm:
                    faster_numpy.clibrary.mean(a[0:999])

    def test_variance(self):
        a = np.arange(1000.0)
        b = np.arange(1000.0, 2000.0)
        with Benchmarker(100000, width=50) as bench:
            @bench("faster_numpy.cylib.variance")
            def _(bm):
                for i in bm:
                    faster_numpy.cylib.variance(a, b)

            @bench("faster_numpy.clibrary.variance")
            def _(bm):
                for i in bm:
                    faster_numpy.clibrary.variance(a, b)


if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestFasterNumpy('test_std'))
    unittest.TextTestRunner(verbosity=2).run(suite)
