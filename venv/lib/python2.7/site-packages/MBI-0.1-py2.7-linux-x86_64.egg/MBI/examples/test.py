import os.path
import unittest

import MBI
import numpy
from numpy.testing import assert_allclose, assert_raises, assert_warns
import time


class MBITestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basic(self):
        numpy.set_printoptions(precision=12)
        nx = 12
        ny = 12
        x = numpy.linspace(0, 1, nx)
        y = numpy.linspace(0, 1, ny)
        P = numpy.array(numpy.meshgrid(x, y))
        P = numpy.swapaxes(P, 0, 1)
        P = numpy.swapaxes(P, 1, 2)
        m = MBI.MBI(P, [x, y], [5, 5], [4, 4])

        P = numpy.array([[0.25, 0.25], [0.25, 0.25], [0.25, 0.25]])
        e1 = numpy.zeros((3, 2))
        e2 = numpy.zeros((3, 2))
        e1[:, 0] = 1.0
        e2[:, 1] = 1.0

        h = 1e-5
        t0 = time.time()
        f0 = m.evaluate(P)

        # print time.time()-t0
        # print '---f'
        # print f0

        # print '---df/dx'
        assert_allclose(m.evaluate(P, 1, 0), (m.evaluate(P + h*e1) - f0)/h, rtol=1.e-4)
        # print '---df/dy'
        assert_allclose(m.evaluate(P, 2, 0), (m.evaluate(P + h*e2) - f0)/h, rtol=1.e-4)
        # print '---d2f/dx2'
        assert_allclose(m.evaluate(P, 1, 1),
                        (m.evaluate(P+h*e1) - 2*f0+m.evaluate(P - h*e1))/h ** 2, rtol=1.e-3)
        # print '---d2f/dxdy'
        assert_allclose(m.evaluate(P, 1, 2),
                        (m.evaluate(P+h*e1+h*e2) - m.evaluate(P + h*e1 - h*e2)-
                         m.evaluate(P-h*e1+h*e2) + m.evaluate(P - h*e1 - h*e2))/4.0/h ** 2, rtol=1.e-3)
        # print '---d2f/dydx'
        assert_allclose(m.evaluate(P, 2, 1),
                        (m.evaluate(P+h*e1+h*e2) - m.evaluate(P + h*e1 - h*e2)-
                         m.evaluate(P-h*e1+h*e2) + m.evaluate(P - h*e1 - h*e2))/4.0/h ** 2, rtol=1.e-3)
        # print '---d2f/dy2'
        assert_allclose(m.evaluate(P, 2, 2),
                        (m.evaluate(P+h*e2) - 2*f0+m.evaluate(P - h*e2))/h ** 2, rtol=1.e-3)

        # print '---df/dx'
        # print m.evaluate(P,1,0)
        # print (m.evaluate(P+h*e1)-f0)/h
        # print '---df/dy'
        # print m.evaluate(P,2,0)
        # print (m.evaluate(P+h*e2)-f0)/h
        # print '---d2f/dx2'
        # print m.evaluate(P,1,1)
        # print (m.evaluate(P+h*e1)-2*f0+m.evaluate(P-h*e1))/h**2
        # print '---d2f/dxdy'
        # print m.evaluate(P,1,2)
        # print (m.evaluate(P+h*e1+h*e2)-m.evaluate(P+h*e1-h*e2)-m.evaluate(P-h*e1+h*e2)+m.evaluate(
        # P-h*e1-h*e2))/4.0/h**2
        # print '---d2f/dydx'
        # print m.evaluate(P,2,1)
        # print (m.evaluate(P+h*e1+h*e2)-m.evaluate(P+h*e1-h*e2)-m.evaluate(P-h*e1+h*e2)+m.evaluate(
        # P-h*e1-h*e2))/4.0/h**2
        # print '---d2f/dy2'
        # print m.evaluate(P,2,2)
        # print (m.evaluate(P+h*e2)-2*f0+m.evaluate(P-h*e2))/h**2

    def test_raise(self):
        nx = 12
        ny = 12
        x = numpy.linspace(0, 1, nx)
        y = numpy.linspace(0, 1, ny)
        P = numpy.array(numpy.meshgrid(x, y))
        P = numpy.swapaxes(P, 0, 1)
        P = numpy.swapaxes(P, 1, 2)
        m = MBI.MBI(P, [x, y], [5, 5], [4, 4])
        m.seterr('raise')

        P = numpy.array([[-100, 0.25], [0.25, 0.25], [0.25, 0.25]])
        e1 = numpy.zeros((3, 2))
        e2 = numpy.zeros((3, 2))
        e1[:, 0] = 1.0
        e2[:, 1] = 1.0

        assert_raises(ValueError, m.evaluate, P)

        P = numpy.array([[0.25, 0.25], [0.25, 0.25], [50, 100]])
        assert_raises(ValueError, m.evaluate, P)

    def test_warn(self):
        nx = 12
        ny = 12
        x = numpy.linspace(0, 1, nx)
        y = numpy.linspace(0, 1, ny)
        P = numpy.array(numpy.meshgrid(x, y))
        P = numpy.swapaxes(P, 0, 1)
        P = numpy.swapaxes(P, 1, 2)
        m = MBI.MBI(P, [x, y], [5, 5], [4, 4])
        m.seterr('warn')

        P = numpy.array([[-100, 0.25], [0.25, 0.25], [0.25, 0.25]])
        e1 = numpy.zeros((3, 2))
        e2 = numpy.zeros((3, 2))
        e1[:, 0] = 1.0
        e2[:, 1] = 1.0

        assert_warns(UserWarning, m.evaluate, P)

        P = numpy.array([[0.25, 0.25], [0.25, 0.25], [50, 100]])
        assert_warns(UserWarning, m.evaluate, P)

    def test_ignore(self):
        nx = 12
        ny = 12
        x = numpy.linspace(0, 1, nx)
        y = numpy.linspace(0, 1, ny)
        P = numpy.array(numpy.meshgrid(x, y))
        P = numpy.swapaxes(P, 0, 1)
        P = numpy.swapaxes(P, 1, 2)
        m = MBI.MBI(P, [x, y], [5, 5], [4, 4])
        m.seterr('ignore')

        P = numpy.array([[-100, 0.25], [0.25, 0.25], [0.25, 0.25]])
        e1 = numpy.zeros((3, 2))
        e2 = numpy.zeros((3, 2))
        e1[:, 0] = 1.0
        e2[:, 1] = 1.0

        m.evaluate(P)
        assert (True)

    def test_pep8(self):
        try:
            import pep8
        except:
            return

        mbidir = os.path.split(MBI.__file__)[0]
        fchecker = pep8.Checker(os.path.join(mbidir,'MBI.py'))
        fchecker.max_line_length = 120  # ignore line-too-long errors
        errors = fchecker.check_all()
        assert(errors == 0)




if __name__ == '__main__':
    unittest.main()
