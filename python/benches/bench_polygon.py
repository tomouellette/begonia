import numpy as np
from begonia import Polygon, Point2d
from benches import runtime, runtime_block

N_ITER: int = 1000
N_REPS: int = 10


def polygon_n(n: int):
    theta = np.linspace(0, np.pi, n)
    return Polygon(np.array([np.cos(theta), np.sin(theta)]).T)


runtime_block(
    "begonia.Polygon",
    f"Polygon runtime benchmarks were measured across {N_REPS} repeats with {
        N_ITER} iterations per repeat. A single warm-up run was discarded."
)


def make_benchmark(func_name, label_fmt, *args):
    for n in [10, 100, 1000]:
        @runtime(label_fmt.format(n=n), number=N_ITER, repeat=N_REPS)
        def fn(n=n):
            p = polygon_n(n)
            getattr(p, func_name)(*args)
            return p
        globals()[f"bench_{func_name}_{n}"] = fn


def make_benchmark_polygon(func_name, label_fmt):
    for n in [10, 100, 1000]:
        @runtime(label_fmt.format(n=n), number=N_ITER, repeat=N_REPS)
        def fn(n=n):
            p = polygon_n(n)
            q = polygon_n(n)
            getattr(p, func_name)(q)
            return p
        globals()[f"bench_{func_name}_{n}"] = fn


make_benchmark("to_list", "to_list (n = {n})")
make_benchmark("to_numpy", "to_numpy (n = {n})")
make_benchmark("push", "push (n = {n})", [10., 10.])
make_benchmark("push_point2d", "push_point2d (n = {n})", Point2d(10., 10.))
make_benchmark("add_scalar_inplace", "add_scalar_inplace (n = {n})", 10)
make_benchmark("sub_scalar_inplace", "sub_scalar_inplace (n = {n})", 10)
make_benchmark("mul_scalar_inplace", "mul_scalar_inplace (n = {n})", 10)
make_benchmark("div_scalar_inplace", "div_scalar_inplace (n = {n})", 10)
make_benchmark("add_scalar", "add_scalar (n = {n})", 10)
make_benchmark("sub_scalar", "sub_scalar (n = {n})", 10)
make_benchmark("mul_scalar", "mul_scalar (n = {n})", 10)
make_benchmark("div_scalar", "div_scalar (n = {n})", 10)
make_benchmark("add_point_inplace", "add_point_inplace (n = {n})", [5, 10])
make_benchmark("add_point2d_inplace",
               "add_point2d_inplace (n = {n})", Point2d(5, 10))
make_benchmark("sub_point_inplace", "sub_point_inplace (n = {n})", [5, 10])
make_benchmark("sub_point2d_inplace",
               "sub_point2d_inplace (n = {n})", Point2d(5, 10))
make_benchmark("mul_point_inplace", "mul_point_inplace (n = {n})", [5, 10])
make_benchmark("mul_point2d_inplace",
               "mul_point2d_inplace (n = {n})", Point2d(5, 10))
make_benchmark("div_point_inplace", "div_point_inplace (n = {n})", [5, 10])
make_benchmark("div_point2d_inplace",
               "div_point2d_inplace (n = {n})", Point2d(5, 10))
make_benchmark("add_point", "add_point (n = {n})", [5, 10])
make_benchmark("add_point2d", "add_point2d (n = {n})", Point2d(5, 10))
make_benchmark("sub_point", "sub_point (n = {n})", [5, 10])
make_benchmark("sub_point2d", "sub_point2d (n = {n})", Point2d(5, 10))
make_benchmark("mul_point", "mul_point (n = {n})", [5, 10])
make_benchmark("mul_point2d", "mul_point2d (n = {n})", Point2d(5, 10))
make_benchmark("div_point", "div_point (n = {n})", [5, 10])
make_benchmark("div_point2d", "div_point2d (n = {n})", Point2d(5, 10))
make_benchmark("dedup", "dedup (n = {n})")
make_benchmark("dedup_inplace", "dedup_inplace (n = {n})")
make_benchmark("order", "order (n = {n})")
make_benchmark("order_inplace", "order_inplace (n = {n})")
make_benchmark("resample", "resample (n = {n})", 5)
make_benchmark("resample_inplace", "resample_inplace (n = {n})", 5)
make_benchmark("encloses_point", "encloses_point (n = {n})", [1., 1.])
make_benchmark("encloses_point2d",
               "encloses_point2d (n = {n})", Point2d(1., 1.))
make_benchmark("distance_to_point_center",
               "distance_to_point_center (n = {n})", [1., 1.])
make_benchmark("distance_to_point2d_center",
               "distance_to_point2d_center (n = {n})", Point2d(1., 1.))
make_benchmark("distance_to_point_centroid",
               "distance_to_point_centroid (n = {n})", [1., 1.])
make_benchmark("distance_to_point2d_centroid",
               "distance_to_point2d_centroid (n = {n})", Point2d(1., 1.))
make_benchmark("distance_to_point_vertex",
               "distance_to_point_vertex (n = {n})", [1., 1.])
make_benchmark("distance_to_point2d_vertex",
               "distance_to_point2d_vertex (n = {n})", Point2d(1., 1.))
make_benchmark("distance_to_point_edge",
               "distance_to_point_edge (n = {n})", [1., 1.])
make_benchmark("distance_to_point2d_edge",
               "distance_to_point2d_edge (n = {n})", Point2d(1., 1.))
make_benchmark("convex_hull", "convex_hull (n = {n})")
make_benchmark("center", "center (n = {n})")
make_benchmark("centroid", "centroid (n = {n})")
make_benchmark("area", "area (n = {n})")
make_benchmark("area_bbox", "area_bbox (n = {n})")
make_benchmark("area_convex", "area_convex (n = {n})")
make_benchmark("perimeter", "perimeter (n = {n})")
make_benchmark("elongation", "elongation (n = {n})")
make_benchmark("thread_length", "thread_length (n = {n})")
make_benchmark("solidity", "solidity (n = {n})")
make_benchmark("extent", "extent (n = {n})")
make_benchmark("form_factor", "form_factor (n = {n})")
make_benchmark("equivalent_diameter", "equivalent_diameter (n = {n})")
make_benchmark("eccentricity", "eccentricity (n = {n})")
make_benchmark("major_axis_length", "major_axis_length (n = {n})")
make_benchmark("minor_axis_length", "minor_axis_length (n = {n})")
make_benchmark("min_radius", "min_radius (n = {n})")
make_benchmark("max_radius", "max_radius (n = {n})")
make_benchmark("mean_radius", "mean_radius (n = {n})")
make_benchmark("min_feret", "min_feret (n = {n})")
make_benchmark("max_feret", "max_feret (n = {n})")

make_benchmark_polygon("add", "add (n = {n})")
make_benchmark_polygon("sub", "sub (n = {n})")
make_benchmark_polygon("mul", "mul (n = {n})")
make_benchmark_polygon("div", "div (n = {n})")
make_benchmark_polygon("d_l1", "d_l1 (n = {n})")
make_benchmark_polygon("d_l2", "d_l2 (n = {n})")
make_benchmark_polygon("d_chebyshev", "d_chebyshev (n = {n})")
make_benchmark_polygon("d_cosine", "d_cosine (n = {n})")
make_benchmark_polygon("distance_to_polygon_center",
                       "distance_to_polygon_center (n = {n})")
make_benchmark_polygon("distance_to_polygon_centroid",
                       "distance_to_polygon_centroid (n = {n})")
make_benchmark_polygon("distance_to_polygon_vertex",
                       "distance_to_polygon_vertex (n = {n})")
make_benchmark_polygon("distance_to_polygon_edge",
                       "distance_to_polygon_edge (n = {n})")
make_benchmark_polygon("align_to",
                       "align_to (n = {n})")


def run_all_benchmarks():
    for name, fn in globals().items():
        if name.startswith("bench_") and callable(fn):
            fn()


if __name__ == "__main__":
    run_all_benchmarks()
