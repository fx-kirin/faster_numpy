import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import faster_numpy.cython
import faster_numpy.clibrary
import numpy as np
from benchmarker import Benchmarker

class TestFasterNumpy(unittest.TestCase):
    def test_cython_shift(self):
        a = np.arange(10.0)
        faster_numpy.cython.shift(a, 1)
        for x in np.arange(9.0):
            self.assertEqual(x+1, a[int(x)])
        a = np.arange(10.0)
        faster_numpy.cython.shift(a, -1)
        for x in np.arange(9.0):
            self.assertEqual(x, a[int(x)+1])
            
        a = np.arange(1000.0)
#        with Benchmarker(10000, width=20) as bench:
#            @bench("numpy.roll")
#            def _(bm):
#                for i in bm:
#                    np.roll(a, 1)
#
#            @bench("faster_numpy.shift")
#            def _(bm):
#                for i in bm:
#                    faster_numpy.cython.shift(a, 1)
                    
    def test_mean(self):
        a = np.arange(10.0)
        mean_value = faster_numpy.cython.mean(a)
        self.assertEqual(faster_numpy.cython.mean(a), np.mean(a))
        self.assertEqual(faster_numpy.clibrary.mean(a), np.mean(a))
        faster_numpy.clibrary.mean(a)
        a = np.arange(1000.0)
        with Benchmarker(100000, width=50) as bench:
            @bench("numpy.mean")
            def _(bm):
                for i in bm:
                    np.mean(a)

            @bench("faster_numpy.cython.mean")
            def _(bm):
                for i in bm:
                    faster_numpy.cython.mean(a)
                    
            @bench("faster_numpy.clibrary.mean")
            def _(bm):
                for i in bm:
                    faster_numpy.clibrary.mean(a)
            @bench("faster_numpy.clibrary.mean partial")
            def _(bm):
                for i in bm:
                    faster_numpy.clibrary.mean(a[0:999])
    
    def test_cython_std(self):
        pass

if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestFasterNumpy('test_mean'))
    unittest.TextTestRunner(verbosity=2).run(suite)
