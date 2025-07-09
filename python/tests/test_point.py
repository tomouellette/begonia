import pytest
import numpy as np
from begonia import Point2d


class TestPoint2d:
    def test_new_success(self):
        p = Point2d(10, 10)
        assert p.x == 10
        assert p.y == 10

    def test_new_invalid_type(self):
        with pytest.raises(TypeError):
            Point2d("A", "B")

    def test_to_list(self):
        p = Point2d(10, 10)
        assert isinstance(p.to_list(), list)

    def test_to_numpy(self):
        p = Point2d(10, 10)
        assert isinstance(p.to_numpy(), np.ndarray)

    def test_eq(self):
        assert Point2d(1, 2) == Point2d(1, 2)

    def test_add(self):
        p = Point2d(4, 2)
        q = Point2d(2, 4)
        assert p.add(q) == Point2d(6, 6)
        assert q.add(p) == Point2d(6, 6)

    def test_sub(self):
        p = Point2d(4, 2)
        q = Point2d(2, 4)
        assert p.sub(q) == Point2d(2, -2)
        assert q.sub(p) == Point2d(-2, 2)

    def test_mul(self):
        p = Point2d(4, 2)
        q = Point2d(2, 4)
        assert p.mul(q) == Point2d(8, 8)
        assert q.mul(p) == Point2d(8, 8)

    def test_div(self):
        p = Point2d(4, 2)
        q = Point2d(2, 4)
        assert p.div(q) == Point2d(2, 0.5)
        assert q.div(p) == Point2d(0.5, 2)

    def test_d_l1(self):
        p = Point2d(2, 4)
        q = Point2d(4, 2)
        assert p.d_l1(q) == 4.
        assert q.d_l1(p) == 4.

    def test_d_l2(self):
        p = Point2d(2, 4)
        q = Point2d(4, 2)
        assert p.d_l2(q) == 8.**(1./2.)
        assert q.d_l2(p) == 8.**(1./2.)

    def test_d_chebyshev(self):
        p = Point2d(2, 4)
        q = Point2d(4, 2)
        assert p.d_chebyshev(q) == 2.
        assert q.d_chebyshev(p) == 2.

    def test_d_cosine(self):
        p = Point2d(2, 4)
        q = Point2d(4, 2)
        assert np.isclose(p.d_cosine(q), 1. - 16. / 20.)
        assert np.isclose(q.d_cosine(p), 1. - 16. / 20.)

    def test_interp(self):
        p = Point2d(2, 4)
        q = Point2d(4, 2)
        assert p.interp(q, 0.0) == p
        assert p.interp(q, 1.0) == q
        assert p.interp(q, 0.5) == Point2d(3, 3)
