import pytest
import numpy as np
from begonia import Polygon, Point2d


def polygon_unwrapped():
    return Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])


def polygon_unwrapped_negative():
    return Polygon([[0, 0], [-1, 0], [-1, -1], [0, -1]])


def polygon_wrapped():
    return Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])


def unit_circle():
    points = []
    for i in np.hstack([
        np.linspace(0, np.pi / 2., 250),
        np.linspace(np.pi / 2., np.pi, 250),
        np.linspace(np.pi, 3 * np.pi / 2., 250),
        np.linspace(3 * np.pi / 2., 2 * np.pi, 250),
    ]):
        x = np.cos(i)
        y = np.sin(i)
        points.append([x, y])
    return Polygon(points)


class TestPolygon:
    def test_new_success(self):
        p = polygon_unwrapped()
        assert isinstance(p.xy, list)
        assert isinstance(p.xy[0], list)
        assert len(p.xy) == 4
        assert isinstance(p, Polygon)

    def test_new_invalid_type(self):
        with pytest.raises(TypeError):
            Polygon([0, "A"])

    def test_new_invalid_size(self):
        with pytest.raises(ValueError):
            Polygon([[0, 0], [0, 1]])

    def test_eq(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        assert hasattr(p, "__eq__")
        assert p == p
        assert p != q

    def test_approximate_eq(self):
        p = polygon_unwrapped()
        q = polygon_unwrapped()
        assert p.eq(q, eps=1e-20)

        q = Polygon([[1e-6, 0], [1, 0], [1, 1], [0, 1]])
        assert p.eq(q, eps=1e-5)
        assert not p.eq(q, eps=1e-7)

    def test_len(self):
        p = polygon_unwrapped()
        assert len(p) == 4

    def test_iter(self):
        p = polygon_unwrapped()
        for i in p:
            assert isinstance(i, list)
            assert len(i) == 2
            assert isinstance(i[0], float)
            assert isinstance(i[1], float)

    def test_to_list(self):
        p = polygon_unwrapped()
        assert isinstance(p.to_list(), list)

    def test_to_numpy(self):
        p = polygon_unwrapped()
        assert isinstance(p.to_numpy(), np.ndarray)

    def test_push(self):
        p = polygon_unwrapped()
        p.push([10., 10.])
        assert p.xy[-1] == [10., 10.]

    def test_push_point2d(self):
        p = polygon_unwrapped()
        a = Point2d(10., 10.)
        p.push_point2d(a)
        assert p.xy[-1] == [10., 10.]

    def test_add_scalar_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.add_scalar_inplace(10)
        for i, j in zip(q.flatten(), p.to_numpy().flatten()):
            assert i + 10 == j

    def test_sub_scalar_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.sub_scalar_inplace(10)
        for i, j in zip(q.flatten(), p.to_numpy().flatten()):
            assert i - 10 == j

    def test_mul_scalar_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.mul_scalar_inplace(10)
        for i, j in zip(q.flatten(), p.to_numpy().flatten()):
            assert i * 10 == j

    def test_div_scalar_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.div_scalar_inplace(10)
        for i, j in zip(q.flatten(), p.to_numpy().flatten()):
            assert i / 10 == j

    def test_add_scalar(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.add_scalar(10)
        for i, j in zip(q.flatten(), p.to_numpy().flatten()):
            assert i + 10 == j

    def test_sub_scalar(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.sub_scalar(10)
        for i, j in zip(q.flatten(), p.to_numpy().flatten()):
            assert i - 10 == j

    def test_mul_scalar(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.mul_scalar(10)
        for i, j in zip(q.flatten(), p.to_numpy().flatten()):
            assert i * 10 == j

    def test_div_scalar(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.div_scalar(10)
        for i, j in zip(q.flatten(), p.to_numpy().flatten()):
            assert i / 10 == j

    def test_add_point_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.add_point_inplace([5, 10])
        for i, j in zip(q, p):
            assert i[0] + 5 == j[0]
            assert i[1] + 10 == j[1]

    def test_add_point2d_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.add_point2d_inplace(Point2d(5, 10))
        for i, j in zip(q, p):
            assert i[0] + 5 == j[0]
            assert i[1] + 10 == j[1]

    def test_sub_point_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.sub_point_inplace([5, 10])
        for i, j in zip(q, p):
            assert i[0] - 5 == j[0]
            assert i[1] - 10 == j[1]

    def test_sub_point2d_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.sub_point2d_inplace(Point2d(5, 10))
        for i, j in zip(q, p):
            assert i[0] - 5 == j[0]
            assert i[1] - 10 == j[1]

    def test_mul_point_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.mul_point_inplace([5, 10])
        for i, j in zip(q, p):
            assert i[0] * 5 == j[0]
            assert i[1] * 10 == j[1]

    def test_mul_point2d_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.mul_point2d_inplace(Point2d(5, 10))
        for i, j in zip(q, p):
            assert i[0] * 5 == j[0]
            assert i[1] * 10 == j[1]

    def test_div_point_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.div_point_inplace([5, 10])
        for i, j in zip(q, p):
            assert i[0] / 5 == j[0]
            assert i[1] / 10 == j[1]

    def test_div_point2d_inplace(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p.div_point2d_inplace(Point2d(5, 10))
        for i, j in zip(q, p):
            assert i[0] / 5 == j[0]
            assert i[1] / 10 == j[1]

    def test_add_point(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.add_point([5, 10])
        for i, j in zip(q, p):
            assert i[0] + 5 == j[0]
            assert i[1] + 10 == j[1]

    def test_add_point2d(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.add_point2d(Point2d(5, 10))
        for i, j in zip(q, p):
            assert i[0] + 5 == j[0]
            assert i[1] + 10 == j[1]

    def test_sub_point(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.sub_point([5, 10])
        for i, j in zip(q, p):
            assert i[0] - 5 == j[0]
            assert i[1] - 10 == j[1]

    def test_sub_point2d(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.sub_point2d(Point2d(5, 10))
        for i, j in zip(q, p):
            assert i[0] - 5 == j[0]
            assert i[1] - 10 == j[1]

    def test_mul_point(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.mul_point([5, 10])
        for i, j in zip(q, p):
            assert i[0] * 5 == j[0]
            assert i[1] * 10 == j[1]

    def test_mul_point2d(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.mul_point2d(Point2d(5, 10))
        for i, j in zip(q, p):
            assert i[0] * 5 == j[0]
            assert i[1] * 10 == j[1]

    def test_div_point(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.div_point([5, 10])
        for i, j in zip(q, p):
            assert i[0] / 5 == j[0]
            assert i[1] / 10 == j[1]

    def test_div_point2d(self):
        p = polygon_unwrapped()
        q = p.to_numpy()
        p = p.div_point2d(Point2d(5, 10))
        for i, j in zip(q, p):
            assert i[0] / 5 == j[0]
            assert i[1] / 10 == j[1]

    def test_add_inplace(self):
        p = polygon_unwrapped()
        q = polygon_unwrapped()
        p.add_inplace(q)
        assert p == Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])

    def test_sub_inplace(self):
        p = polygon_unwrapped()
        q = polygon_unwrapped()
        p.sub_inplace(q)
        assert p == Polygon([[0, 0], [0, 0], [0, 0], [0, 0]])

    def test_mul_inplace(self):
        p = polygon_unwrapped()
        q = polygon_unwrapped()
        p.mul_inplace(q)
        assert p == q

    def test_div_inplace(self):
        p = polygon_unwrapped()
        q = polygon_unwrapped()
        p.add_scalar_inplace(2)
        q.add_scalar_inplace(1)
        z = p.to_numpy()
        p.div_inplace(q)
        assert np.allclose(p.to_numpy(), z / q.to_numpy())

    def test_add(self):
        p = q = polygon_unwrapped()
        p = p.add(q)
        assert p == Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])

    def test_sub(self):
        p = q = polygon_unwrapped()
        p = p.sub(q)
        assert p == Polygon([[0, 0], [0, 0], [0, 0], [0, 0]])

    def test_mul(self):
        p = q = polygon_unwrapped()
        p = p.mul(q)
        assert p == q

    def test_div(self):
        p = q = polygon_unwrapped()
        p.add_scalar_inplace(2)
        q.add_scalar_inplace(1)
        z = p.div(q)
        assert np.allclose(z.to_numpy(), p.to_numpy() / q.to_numpy())

    def test_center(self):
        p = polygon_unwrapped()
        assert p.center() == Point2d(0.5, 0.5)

        q = polygon_wrapped()
        assert q.center() == Point2d(0.5, 0.5)

        z = polygon_unwrapped_negative()
        assert z.center() == Point2d(-0.5, -0.5)

    def test_centroid(self):
        p = polygon_unwrapped()
        x = p.centroid()
        assert p.centroid() == Point2d(0.5, 0.5)

        q = polygon_wrapped()
        assert q.centroid() == Point2d(0.5, 0.5)

        z = polygon_unwrapped_negative()
        assert z.centroid() == Point2d(-0.5, -0.5)

    def test_d_l1(self):
        p = q = polygon_unwrapped()
        q = q.add(q)
        p = p.d_l1(q)
        assert p == 4

    def test_d_l2(self):
        p = q = polygon_unwrapped()
        q = q.add(q)
        p = p.d_l2(q)
        assert p == 2

    def test_d_chebyshev(self):
        p = q = polygon_unwrapped()
        q = q.add(q)
        p = p.d_chebyshev(q)
        assert p == 1

    def test_d_cosine(self):
        p = q = polygon_unwrapped()
        q = q.add(q)
        d = p.d_cosine(q)
        assert d == 0.0

    def test_dedup(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        q = q.dedup()
        assert q == p

    def test_dedup_inplace(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        q.dedup_inplace()
        assert q == p

    def test_dedup_unstable(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        q = q.dedup()
        assert len(q) == 4
        for i in q:
            count = sum([i == j for j in p])
            assert count == 1

    def test_dedup_unstable_inplace(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        q.dedup_inplace()
        assert len(q) == 4
        for i in q:
            count = sum([i == j for j in p])
            assert count == 1

    def test_order(self):
        p = Polygon([[0, 1], [1, 0], [0, 0], [1, 1]])
        p = p.order()
        assert p.to_list() == [[0., 1.], [1., 1.], [1., 0.], [0., 0.]]

    def test_order_inplace(self):
        p = Polygon([[0, 1], [1, 0], [0, 0], [1, 1]])
        p.order_inplace()
        assert p.to_list() == [[0., 1.], [1., 1.], [1., 0.], [0., 0.]]

    def test_resample(self):
        p = polygon_unwrapped()
        q = p.resample(5)
        assert q == p

    def test_resample_inplace(self):
        p = polygon_unwrapped()
        q = polygon_unwrapped()
        p.resample_inplace(5)
        assert p == q

    def test_encloses_point(self):
        p = polygon_unwrapped()
        p1 = [0.5, 0.5]
        assert p.encloses_point(p1)
        assert p.encloses_point(np.array(p1))

        p2 = [-10., 10]
        assert not p.encloses_point(p2)
        assert not p.encloses_point(np.array(p2))

        p3 = [0., 0.]
        assert p.encloses_point(p3)
        assert p.encloses_point(np.array(p3))

        triangle = Polygon([[0., 0.], [1., 1.], [2., 0.]])
        assert triangle.encloses_point(p1)
        assert triangle.encloses_point(np.array(p1))

        l_shape = Polygon([
            [0., 0.], [1., 0.], [2., 0.], [2., 2.], [1., 2.], [1., 1.], [0., 1]])
        p4 = [1.75, 1.75]
        p5 = [0.1, 0.1]
        assert l_shape.encloses_point(p4)
        assert l_shape.encloses_point(np.array(p4))
        assert l_shape.encloses_point(p5)
        assert l_shape.encloses_point(np.array(p5))

    def test_encloses_point2d(self):
        p = polygon_unwrapped()
        p1 = Point2d(0.5, 0.5)
        assert p.encloses_point2d(p1)

        p2 = Point2d(-10., 10)
        assert not p.encloses_point2d(p2)

        p3 = Point2d(0., 0.)
        assert p.encloses_point2d(p3)

        triangle = Polygon([[0., 0.], [1., 1.], [2., 0.]])
        assert triangle.encloses_point2d(p1)

        l_shape = Polygon([
            [0., 0.], [1., 0.], [2., 0.], [2., 2.], [1., 2.], [1., 1.], [0., 1]])
        p4 = Point2d(1.75, 1.75)
        p5 = Point2d(0.1, 0.1)
        assert l_shape.encloses_point2d(p4)
        assert l_shape.encloses_point2d(p5)

    def test_distance_to_point_center(self):
        p = polygon_unwrapped()
        p1 = [0.5, 0.5]
        assert p.distance_to_point_center(p1) == 0.0
        assert p.distance_to_point_center(np.array(p1)) == 0.0

        p2 = [10, 12]
        assert p.distance_to_point_center(p2) == np.sqrt(9.5**2 + 11.5**2)
        assert p.distance_to_point_center(
            np.array(p2)) == np.sqrt(9.5**2 + 11.5**2)

    def test_distance_to_point2d_center(self):
        p = polygon_unwrapped()
        p1 = Point2d(0.5, 0.5)
        assert p.distance_to_point2d_center(p1) == 0.0

        p2 = Point2d(10, 12)
        assert p.distance_to_point2d_center(p2) == np.sqrt(9.5**2 + 11.5**2)

    def test_distance_to_point_centroid(self):
        p = polygon_unwrapped()
        p1 = [0.5, 0.5]
        assert p.distance_to_point_centroid(p1) == 0.0
        assert p.distance_to_point_centroid(np.array(p1)) == 0.0

        p2 = [10, 12]
        assert p.distance_to_point_centroid(p2) == np.sqrt(9.5**2 + 11.5**2)
        assert p.distance_to_point_centroid(
            np.array(p2)) == np.sqrt(9.5**2 + 11.5**2)

    def test_distance_to_point2d_centroid(self):
        p = polygon_unwrapped()
        p1 = Point2d(0.5, 0.5)
        assert p.distance_to_point2d_centroid(p1) == 0.0

        p2 = Point2d(10, 12)
        assert p.distance_to_point2d_centroid(p2) == np.sqrt(9.5**2 + 11.5**2)

    def test_distance_to_point_vertex(self):
        p = polygon_unwrapped()
        p1 = [0.5, 0.5]
        assert p.distance_to_point_vertex(p1) == np.sqrt(2*(0.5**2))
        assert p.distance_to_point_vertex(np.array(p1)) == np.sqrt(2*(0.5**2))

        p2 = [0, 10]
        assert p.distance_to_point_vertex(p2) == 9.0
        assert p.distance_to_point_vertex(np.array(p2)) == 9.0

    def test_distance_to_point2d_vertex(self):
        p = polygon_unwrapped()
        p1 = Point2d(0.5, 0.5)
        assert p.distance_to_point2d_vertex(p1) == np.sqrt(2*(0.5**2))

        p2 = Point2d(0, 10)
        assert p.distance_to_point2d_vertex(p2) == 9.0

    def test_distance_to_point_edge(self):
        p = polygon_unwrapped()
        p1 = [0.5, 0.5]
        assert p.distance_to_point_edge(p1) == 0.5
        assert p.distance_to_point_edge(np.array(p1)) == 0.5

        p2 = [0.5, 10]
        assert p.distance_to_point_edge(p2) == 9.0
        assert p.distance_to_point_edge(np.array(p2)) == 9.0

    def test_distance_to_point2d_edge(self):
        p = polygon_unwrapped()
        p1 = Point2d(0.5, 0.5)
        assert p.distance_to_point2d_edge(p1) == 0.5

        p2 = Point2d(0.5, 10)
        assert p.distance_to_point2d_edge(p2) == 9.0

    def test_distance_to_polygon_center(self):
        p = polygon_unwrapped()
        q = p.add_point([9., 0.])
        assert p.distance_to_polygon_center(q) == 9.

    def test_distance_to_polygon_centroid(self):
        p = polygon_unwrapped()
        q = p.add_point([9., 0.])
        assert p.distance_to_polygon_centroid(q) == 9.

    def test_distance_to_polygon_vertex(self):
        p = polygon_unwrapped()
        q = p.add_point([9., 1.])
        assert p.distance_to_polygon_vertex(q) == 8.

    def test_distance_to_polygon_edge(self):
        p = polygon_unwrapped()
        q = p.add_point([9., 0.])
        assert p.distance_to_polygon_edge(q) == 8.

    def test_convex_hull(self):
        p = polygon_unwrapped()
        assert p.convex_hull().to_list() == [[0., 0.], [
            0., 1.], [1., 1.], [1., 0.]]

        q = Polygon([[0., 0.], [0., 1.], [0.5, 0.9], [1., 1.], [1., 0.]])
        assert q.convex_hull().to_list() == [[0., 0.], [
            0., 1.], [1., 1.], [1., 0.]]

    def test_align_to(self):
        p = polygon_unwrapped()
        q = Polygon([[-0.5, 0], [0.5, 0], [0.5, 1.0], [-0.5, 1.0]])
        q = q.align_to(p)
        assert q == p

        d = np.sqrt(2*(0.5**2))
        s = Polygon([[-d, 0.], [0., -d], [d, 0.], [0., d]])
        s = s.align_to(p)
        assert s.eq(p, eps=1e-15)

        z = polygon_wrapped()
        with pytest.raises(ValueError):
            z = z.align_to(p)

    def test_area(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.area() == 1.
        assert q.area() == 1.
        assert np.abs(z.area() - np.pi) < 1e-4

    def test_area_bbox(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.area_bbox() == 1.
        assert q.area_bbox() == 1.
        assert z.area_bbox() == 4.

    def test_area_convex(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.area_convex() == 1.
        assert q.area_convex() == 1.
        assert z.area_convex() < 2. * np.pi

    def test_perimeter(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.perimeter() == 4.
        assert q.perimeter() == 4.
        assert np.abs(z.perimeter() - 2 * np.pi) < 1e-4

    def test_elongation(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.elongation() == 1.
        assert q.elongation() == 1.
        assert z.elongation() == 1.

    def test_thread_length(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.thread_length() == 1.
        assert q.thread_length() == 1.
        assert np.abs(z.thread_length() - (2 * np.pi) / 4.) < 1e-4

    def test_solidity(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.solidity() == 1.
        assert q.solidity() == 1.
        assert z.solidity() == z.area() / z.area_convex()

    def test_extent(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.extent() == 1.
        assert q.extent() == 1.
        assert z.extent() == z.area() / z.area_bbox()

    def test_form_factor(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.form_factor() == 4. * np.pi / 16.
        assert q.form_factor() == 4. * np.pi / 16.
        assert z.form_factor() == 4. * np.pi * z.area() / z.perimeter()**2

    def test_equivalent_diameter(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.equivalent_diameter() == np.sqrt(4. / np.pi)
        assert q.equivalent_diameter() == np.sqrt(4. / np.pi)
        assert z.equivalent_diameter() == np.sqrt(4. * z.area() / np.pi)

    def test_eccentricity(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert np.abs(p.eccentricity() - 0.) < 1e-7
        assert np.abs(q.eccentricity() - 0.) < 1e-7
        assert np.abs(z.eccentricity() - 0.) < 1e-5

    def test_major_axis_length(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert np.abs(p.major_axis_length() - np.sqrt(2.)) < 1e-8
        assert np.abs(q.major_axis_length() - np.sqrt(2.)) < 1e-8
        assert np.abs(z.major_axis_length() - 2.) < 1e-8

    def test_minor_axis_length(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert np.abs(p.major_axis_length() - np.sqrt(2.)) < 1e-8
        assert np.abs(q.major_axis_length() - np.sqrt(2.)) < 1e-8
        assert np.abs(z.major_axis_length() - 2.) < 1e-8

    def test_min_radius(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.min_radius() == 0.5
        assert q.min_radius() == 0.5
        assert np.abs(z.min_radius() - 1.) < 1e-5

    def test_max_radius(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.max_radius() == np.sqrt(2.) / 2.
        assert q.max_radius() == np.sqrt(2.) / 2.
        assert np.abs(z.max_radius() - 1.) < 1e-15

    def test_mean_radius(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        print((0.5 + np.sqrt(2.) / 2.) / 2.)
        assert p.mean_radius() == np.sqrt(2.) / 2.
        assert q.mean_radius() == np.sqrt(2.) / 2.
        assert np.abs(z.mean_radius() - 1.) < 1e-15

    def test_min_feret(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert p.min_feret() == 1.
        assert q.min_feret() == 1.
        assert np.abs(z.min_feret() - 2.) < 1e-5

    def test_max_feret(self):
        p = polygon_unwrapped()
        q = polygon_wrapped()
        z = unit_circle()
        assert np.abs(p.max_feret() - np.sqrt(2.)) < 1e-9
        assert np.abs(q.max_feret() - np.sqrt(2.)) < 1e-9
        assert np.abs(z.max_feret() - 2.) < 1e-15
