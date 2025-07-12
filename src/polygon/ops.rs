// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use super::Polygon;
use crate::point::Point2d;
use nalgebra::{DVector, MatrixXx5};

#[inline]
pub fn add_scalar_inplace(polygon: &mut Polygon, scalar: f64) {
    for point in polygon.xy.iter_mut() {
        point[0] += scalar;
        point[1] += scalar;
    }
    polygon._deduped = false;
    polygon._ordered = false;
}

#[inline]
pub fn sub_scalar_inplace(polygon: &mut Polygon, scalar: f64) {
    for point in polygon.xy.iter_mut() {
        point[0] -= scalar;
        point[1] -= scalar;
    }
    polygon._deduped = false;
    polygon._ordered = false;
}

#[inline]
pub fn mul_scalar_inplace(polygon: &mut Polygon, scalar: f64) {
    for point in polygon.xy.iter_mut() {
        point[0] *= scalar;
        point[1] *= scalar;
    }
    polygon._deduped = false;
    polygon._ordered = false;
}

#[inline]
pub fn div_scalar_inplace(polygon: &mut Polygon, scalar: f64) {
    if scalar == 0.0 {
        for point in polygon.xy.iter_mut() {
            point[0] = f64::NAN;
            point[1] = f64::NAN;
        }
    } else {
        for point in polygon.xy.iter_mut() {
            point[0] /= scalar;
            point[1] /= scalar;
        }
    }
    polygon._deduped = false;
    polygon._ordered = false;
}

#[inline]
pub fn add_scalar(polygon: &Polygon, scalar: f64) -> Polygon {
    Polygon {
        xy: polygon
            .xy
            .iter()
            .map(|point| [point[0] + scalar, point[1] + scalar])
            .collect(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn sub_scalar(polygon: &Polygon, scalar: f64) -> Polygon {
    Polygon {
        xy: polygon
            .xy
            .iter()
            .map(|point| [point[0] - scalar, point[1] - scalar])
            .collect(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn mul_scalar(polygon: &Polygon, scalar: f64) -> Polygon {
    Polygon {
        xy: polygon
            .xy
            .iter()
            .map(|point| [point[0] * scalar, point[1] * scalar])
            .collect(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn div_scalar(polygon: &Polygon, scalar: f64) -> Polygon {
    if scalar == 0.0 {
        Polygon {
            xy: polygon.xy.iter().map(|_| [f64::NAN, f64::NAN]).collect(),
            _deduped: false,
            _ordered: false,
        }
    } else {
        Polygon {
            xy: polygon
                .xy
                .iter()
                .map(|point| [point[0] / scalar, point[1] / scalar])
                .collect(),
            _deduped: false,
            _ordered: false,
        }
    }
}

#[inline]
pub fn add_point_inplace(px: f64, py: f64, polygon: &mut Polygon) {
    for point in polygon.xy.iter_mut() {
        point[0] += px;
        point[1] += py;
    }
    polygon._deduped = false;
    polygon._ordered = false;
}

#[inline]
pub fn sub_point_inplace(px: f64, py: f64, polygon: &mut Polygon) {
    for point in polygon.xy.iter_mut() {
        point[0] -= px;
        point[1] -= py;
    }
    polygon._deduped = false;
    polygon._ordered = false;
}

#[inline]
pub fn mul_point_inplace(px: f64, py: f64, polygon: &mut Polygon) {
    for point in polygon.xy.iter_mut() {
        point[0] *= px;
        point[1] *= py;
    }
    polygon._deduped = false;
    polygon._ordered = false;
}

#[inline]
pub fn div_point_inplace(px: f64, py: f64, polygon: &mut Polygon) {
    for point in polygon.xy.iter_mut() {
        if px == 0.0 || py == 0.0 {
            point[0] = f64::NAN;
            point[1] = f64::NAN;
        } else {
            point[0] /= px;
            point[1] /= py;
        }
    }
    polygon._deduped = false;
    polygon._ordered = false;
}

#[inline]
pub fn add_point(px: f64, py: f64, polygon: &Polygon) -> Polygon {
    Polygon {
        xy: polygon
            .xy
            .iter()
            .map(|point| [point[0] + px, point[1] + py])
            .collect(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn sub_point(px: f64, py: f64, polygon: &Polygon) -> Polygon {
    Polygon {
        xy: polygon
            .xy
            .iter()
            .map(|point| [point[0] - px, point[1] - py])
            .collect(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn mul_point(px: f64, py: f64, polygon: &Polygon) -> Polygon {
    Polygon {
        xy: polygon
            .xy
            .iter()
            .map(|point| [point[0] * px, point[1] * py])
            .collect(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn div_point(px: f64, py: f64, polygon: &Polygon) -> Polygon {
    Polygon {
        xy: polygon
            .xy
            .iter()
            .map(|point| {
                if px == 0.0 || py == 0.0 {
                    [f64::NAN, f64::NAN]
                } else {
                    [point[0] / px, point[1] / py]
                }
            })
            .collect(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn add_inplace(a: &mut Polygon, b: &Polygon) {
    for (pa, pb) in a.xy.iter_mut().zip(b.xy.iter()) {
        pa[0] += pb[0];
        pa[1] += pb[1];
    }
    a._deduped = false;
    a._ordered = false;
}

#[inline]
pub fn sub_inplace(a: &mut Polygon, b: &Polygon) {
    for (pa, pb) in a.xy.iter_mut().zip(b.xy.iter()) {
        pa[0] -= pb[0];
        pa[1] -= pb[1];
    }
    a._deduped = false;
    a._ordered = false;
}

#[inline]
pub fn mul_inplace(a: &mut Polygon, b: &Polygon) {
    for (pa, pb) in a.xy.iter_mut().zip(b.xy.iter()) {
        pa[0] *= pb[0];
        pa[1] *= pb[1];
    }
    a._deduped = false;
    a._ordered = false;
}

#[inline]
pub fn div_inplace(a: &mut Polygon, b: &Polygon) {
    for (pa, pb) in a.xy.iter_mut().zip(b.xy.iter()) {
        if pb[0] == 0.0 || pb[1] == 0.0 {
            pa[0] = f64::NAN;
            pa[1] = f64::NAN;
        } else {
            pa[0] /= pb[0];
            pa[1] /= pb[1];
        }
    }
    a._deduped = false;
    a._ordered = false;
}

#[inline]
pub fn add(a: &Polygon, b: &Polygon) -> Polygon {
    Polygon {
        xy: a
            .xy
            .iter()
            .zip(b.xy.iter())
            .map(|(pa, pb)| [pa[0] + pb[0], pa[1] + pb[1]])
            .collect::<Vec<[f64; 2]>>(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn sub(a: &Polygon, b: &Polygon) -> Polygon {
    Polygon {
        xy: a
            .xy
            .iter()
            .zip(b.xy.iter())
            .map(|(pa, pb)| [pa[0] - pb[0], pa[1] - pb[1]])
            .collect::<Vec<[f64; 2]>>(),
        _deduped: false,
        _ordered: false,
    }
}

pub fn mul(a: &Polygon, b: &Polygon) -> Polygon {
    Polygon {
        xy: a
            .xy
            .iter()
            .zip(b.xy.iter())
            .map(|(pa, pb)| [pa[0] * pb[0], pa[1] * pb[1]])
            .collect::<Vec<[f64; 2]>>(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn div(a: &Polygon, b: &Polygon) -> Polygon {
    Polygon {
        xy: a
            .xy
            .iter()
            .zip(b.xy.iter())
            .map(|(pa, pb)| {
                if pb[0] == 0.0 || pb[1] == 0.0 {
                    [f64::NAN, f64::NAN]
                } else {
                    [pa[0] / pb[0], pa[1] / pb[1]]
                }
            })
            .collect::<Vec<[f64; 2]>>(),
        _deduped: false,
        _ordered: false,
    }
}

#[inline]
pub fn polygon_center(polygon: &Polygon) -> Point2d {
    let points = &polygon.xy;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let n = points.len();

    let is_closed = points[0][0] == points[n - 1][0] && points[0][1] == points[n - 1][1];
    let n_end = if is_closed { n - 1 } else { n };

    for point in points.iter().take(n_end) {
        sum_x += point[0];
        sum_y += point[1];
    }

    Point2d {
        x: sum_x / n_end as f64,
        y: sum_y / n_end as f64,
    }
}

#[inline]
pub fn polygon_centroid(polygon: &Polygon) -> Point2d {
    let points = &polygon.xy;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut area = 0.0;
    let n = points.len();

    let is_closed = points[0][0] == points[n - 1][0] && points[0][1] == points[n - 1][1];
    let n_end = if is_closed { n - 1 } else { n };

    for i in 0..n_end {
        let j = (i + 1) % n_end;
        let p1 = &points[i];
        let p2 = &points[j];
        let cross = p1[0] * p2[1] - p2[0] * p1[1];
        sum_x += (p1[0] + p2[0]) * cross;
        sum_y += (p1[1] + p2[1]) * cross;
        area += cross;
    }
    area /= 2.0;

    Point2d {
        x: sum_x / (6.0 * area),
        y: sum_y / (6.0 * area),
    }
}

#[inline]
pub fn d_l1(a: &Polygon, b: &Polygon) -> f64 {
    a.xy.iter()
        .zip(&b.xy)
        .map(|(pa, pb)| (pa[0] - pb[0]).abs() + (pa[1] - pb[1]).abs())
        .sum()
}

#[inline]
pub fn d_l2(a: &Polygon, b: &Polygon) -> f64 {
    a.xy.iter()
        .zip(&b.xy)
        .map(|(pa, pb)| (pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[inline]
pub fn d_chebyshev(a: &Polygon, b: &Polygon) -> f64 {
    a.xy.iter()
        .zip(b.xy.iter())
        .map(|(pa, pb)| (pa[0] - pb[0]).abs().max((pa[1] - pb[1]).abs()))
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

#[inline]
pub fn d_cosine(a: &Polygon, b: &Polygon) -> f64 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (pa, pb) in a.xy.iter().zip(&b.xy) {
        dot_product += pa[0] * pb[0] + pa[1] * pb[1];
        norm_a += pa[0].powi(2) + pa[1].powi(2);
        norm_b += pb[0].powi(2) + pb[1].powi(2);
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return f64::NAN;
    }

    1.0 - dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

#[inline]
pub fn d_hausdorff(a: &Polygon, b: &Polygon, n: usize) -> f64 {
    directed_hausdorff_distance(a, b, n).max(directed_hausdorff_distance(b, a, n))
}

#[inline]
fn directed_hausdorff_distance(a: &Polygon, b: &Polygon, n: usize) -> f64 {
    let mut max_distance: f64 = 0.0;
    for i in 0..a.xy.len() {
        let j = (i + 1) % a.xy.len();
        let start = a.xy[i];
        let end = a.xy[j];

        for k in 0..=n {
            let t = k as f64 / n as f64;
            let x = start[0] + t * (end[0] - start[0]);
            let y = start[1] + t * (end[1] - start[1]);

            let dist = distance_to_point_edge(x, y, b);
            max_distance = max_distance.max(dist);
        }
    }

    max_distance
}

#[inline]
pub fn dedup_inplace(polygon: &mut Polygon) {
    let mut write_index = 0;

    for read_index in 0..polygon.xy.len() {
        let p = &polygon.xy[read_index];
        let is_dup = (0..write_index).any(|i| p == &polygon.xy[i]);
        if !is_dup {
            polygon.xy[write_index] = *p;
            write_index += 1;
        }
    }

    polygon.xy.truncate(write_index);
    polygon._deduped = true;
}

#[inline]
pub fn dedup(polygon: &Polygon) -> Polygon {
    let mut unique_points: Vec<[f64; 2]> = Vec::new();
    for p in &polygon.xy {
        let is_dup = unique_points.iter().any(|q| p == q);
        if !is_dup {
            unique_points.push(*p);
        }
    }

    Polygon {
        xy: unique_points,
        _deduped: true,
        _ordered: polygon._ordered,
    }
}

#[inline]
pub fn dedup_unstable_inplace(polygon: &mut Polygon) {
    polygon.xy.sort_unstable_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });

    polygon
        .xy
        .dedup_by(|a, b| (a[0] - b[0]).abs() < f64::EPSILON && (a[1] - b[1]).abs() < f64::EPSILON);

    polygon._deduped = true;
}

#[inline]
pub fn dedup_unstable(polygon: &Polygon) -> Polygon {
    let mut xy = polygon.xy.clone();
    xy.sort_unstable_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });

    xy.dedup_by(|a, b| (a[0] - b[0]).abs() < f64::EPSILON && (a[1] - b[1]).abs() < f64::EPSILON);

    Polygon {
        xy,
        _deduped: true,
        _ordered: polygon._ordered,
    }
}

#[inline]
pub fn order_inplace(polygon: &mut Polygon) {
    let points = &mut polygon.xy;
    let n = points.len() as f64;

    let centroid = points
        .iter()
        .fold([0.0; 2], |acc, p| [acc[0] + p[0] / n, acc[1] + p[1] / n]);

    points.sort_by(|a, b| {
        let theta_a = (a[1] - centroid[1]).atan2(a[0] - centroid[0]);
        let theta_b = (b[1] - centroid[1]).atan2(b[0] - centroid[0]);
        if theta_a == theta_b {
            let dist_a = (a[0] - centroid[0]).powi(2) + (a[1] - centroid[1]).powi(2);
            let dist_b = (b[0] - centroid[0]).powi(2) + (b[1] - centroid[1]).powi(2);
            dist_a.partial_cmp(&dist_b).unwrap()
        } else {
            theta_b.partial_cmp(&theta_a).unwrap()
        }
    });

    polygon._ordered = true;
}

#[inline]
pub fn order(polygon: &Polygon) -> Polygon {
    let mut new_polygon = Polygon {
        xy: polygon.xy.clone(),
        _deduped: polygon._deduped,
        _ordered: polygon._ordered,
    };
    order_inplace(&mut new_polygon);
    new_polygon
}

#[inline]
pub fn resample_inplace(polygon: &mut Polygon, n_points: usize) {
    let points = &mut polygon.xy;

    let is_closed = points[0] == points[points.len() - 1];
    if !is_closed {
        points.push(points[0]);
    }

    // Calculate cumulative distances in one pass
    let mut cum_distances = Vec::with_capacity(points.len());
    cum_distances.push(0.0);
    let mut total_length = 0.0;
    for i in 0..points.len() - 1 {
        let dx = points[i][0] - points[i + 1][0];
        let dy = points[i][1] - points[i + 1][1];
        total_length += (dx * dx + dy * dy).sqrt();
        cum_distances.push(total_length);
    }

    // Pre-calculate step size
    let step = total_length / (n_points - 1) as f64;
    let mut resampled_points = Vec::with_capacity(n_points);
    let mut j = 0;
    for i in 0..n_points {
        let sample_distance = i as f64 * step;

        // Find the segment
        while j < points.len() - 2 && sample_distance > cum_distances[j + 1] {
            j += 1;
        }

        // Calculate interpolation parameter
        let segment_length = cum_distances[j + 1] - cum_distances[j];
        let t = if segment_length > 0.0 {
            (sample_distance - cum_distances[j]) / segment_length
        } else {
            0.0
        };

        // Interpolate
        let x = points[j][0] + t * (points[j + 1][0] - points[j][0]);
        let y = points[j][1] + t * (points[j + 1][1] - points[j][1]);
        resampled_points.push([x, y]);
    }

    if !is_closed {
        resampled_points.pop();
    }

    points.clear();
    points.extend(resampled_points);
}

#[inline]
pub fn resample(polygon: &Polygon, n_points: usize) -> Polygon {
    let mut new_polygon = Polygon {
        xy: polygon.xy.clone(),
        _deduped: polygon._deduped,
        _ordered: polygon._ordered,
    };

    resample_inplace(&mut new_polygon, n_points);
    new_polygon
}

#[inline]
pub fn encloses_point(px: f64, py: f64, polygon: &Polygon) -> bool {
    let points = &polygon.xy;
    let n = points.len();

    let mut inside = false;

    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (points[i][0], points[i][1]);
        let (xj, yj) = (points[j][0], points[j][1]);

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }

    inside
}

#[inline]
pub fn distance_to_point_center(px: f64, py: f64, polygon: &Polygon) -> f64 {
    let center = polygon_center(polygon);
    let dx = px - center.x;
    let dy = py - center.y;
    (dx * dx + dy * dy).sqrt()
}

#[inline]
pub fn distance_to_point_centroid(px: f64, py: f64, polygon: &Polygon) -> f64 {
    let centroid = polygon_centroid(polygon);
    let dx = px - centroid.x;
    let dy = py - centroid.y;
    (dx * dx + dy * dy).sqrt()
}

#[inline]
pub fn distance_to_point_vertex(px: f64, py: f64, polygon: &Polygon) -> f64 {
    let points = &polygon.xy;

    let mut min_distance = f64::INFINITY;
    for vertex in points {
        let dx = px - vertex[0];
        let dy = py - vertex[1];
        let distance = (dx * dx + dy * dy).sqrt();

        min_distance = min_distance.min(distance);
    }

    min_distance
}

#[inline]
pub fn distance_to_point_edge(px: f64, py: f64, polygon: &Polygon) -> f64 {
    let points = &polygon.xy;

    let mut min_distance = f64::INFINITY;
    for i in 0..points.len() {
        let p1 = &points[i];
        let p2 = &points[(i + 1) % points.len()];

        let distance = point_to_line_segment_distance(px, py, p1, p2);
        min_distance = min_distance.min(distance);
    }

    min_distance
}

#[inline]
fn point_to_line_segment_distance(px: f64, py: f64, p1: &[f64; 2], p2: &[f64; 2]) -> f64 {
    let x1 = p1[0];
    let y1 = p1[1];
    let x2 = p2[0];
    let y2 = p2[1];

    // Vector from p1 to p2
    let dx = x2 - x1;
    let dy = y2 - y1;

    // Handle case where p1 == p2
    if dx == 0.0 && dy == 0.0 {
        let dx_point = px - x1;
        let dy_point = py - y1;
        return (dx_point * dx_point + dy_point * dy_point).sqrt();
    }

    // Calculate the parameter t for the closest point on the line segment
    // t = 0 means closest point is p1, t = 1 means closest point is p2
    let t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);

    // Clamp t to [0, 1] to stay within the line segment
    let t = t.clamp(0., 1.);

    // Closest point on the segment
    let closest_x = x1 + t * dx;
    let closest_y = y1 + t * dy;

    // Distance from point to closest point on segment
    let dx_closest = px - closest_x;
    let dy_closest = py - closest_y;

    (dx_closest * dx_closest + dy_closest * dy_closest).sqrt()
}

#[inline]
pub fn distance_to_polygon_center(polygon1: &Polygon, polygon2: &Polygon) -> f64 {
    let center1 = polygon_center(polygon1);
    let center2 = polygon_center(polygon2);
    let dx = center1.x - center2.x;
    let dy = center1.y - center2.y;
    (dx * dx + dy * dy).sqrt()
}

#[inline]
pub fn distance_to_polygon_centroid(polygon1: &Polygon, polygon2: &Polygon) -> f64 {
    let centroid1 = polygon_centroid(polygon1);
    let centroid2 = polygon_centroid(polygon2);
    let dx = centroid1.x - centroid2.x;
    let dy = centroid1.y - centroid2.y;
    (dx * dx + dy * dy).sqrt()
}

#[inline]
pub fn distance_to_polygon_vertex(polygon1: &Polygon, polygon2: &Polygon) -> f64 {
    let mut min_distance = f64::INFINITY;
    for vertex1 in &polygon1.xy {
        for vertex2 in &polygon2.xy {
            let dx = vertex1[0] - vertex2[0];
            let dy = vertex1[1] - vertex2[1];
            let distance = (dx * dx + dy * dy).sqrt();
            min_distance = min_distance.min(distance);
        }
    }
    min_distance
}

#[inline]
pub fn distance_to_polygon_edge(polygon1: &Polygon, polygon2: &Polygon) -> f64 {
    #[inline]
    fn segment_distance(a1: &[f64; 2], a2: &[f64; 2], b1: &[f64; 2], b2: &[f64; 2]) -> f64 {
        let u = (a2[0] - a1[0], a2[1] - a1[1]);
        let v = (b2[0] - b1[0], b2[1] - b1[1]);
        let w0 = (a1[0] - b1[0], a1[1] - b1[1]);
        let a = u.0 * u.0 + u.1 * u.1;
        let b = u.0 * v.0 + u.1 * v.1;
        let c = v.0 * v.0 + v.1 * v.1;
        let d = u.0 * w0.0 + u.1 * w0.1;
        let e = v.0 * w0.0 + v.1 * w0.1;
        let denom = a * c - b * b;
        let (s, t) = if denom.abs() < 1e-10 {
            (0.0, (e / c).clamp(0.0, 1.0))
        } else {
            (
                ((b * e - c * d) / denom).clamp(0.0, 1.0),
                ((a * e - b * d) / denom).clamp(0.0, 1.0),
            )
        };

        // Closest points
        let pa = (a1[0] + s * u.0, a1[1] + s * u.1);
        let pb = (b1[0] + t * v.0, b1[1] + t * v.1);
        let dx = pa.0 - pb.0;
        let dy = pa.1 - pb.1;
        (dx * dx + dy * dy).sqrt()
    }

    let n1 = polygon1.xy.len();
    let n2 = polygon2.xy.len();

    let mut min_dist = f64::INFINITY;
    for i in 0..n1 {
        let a1 = &polygon1.xy[i];
        let a2 = &polygon1.xy[(i + 1) % n1];
        for j in 0..n2 {
            let b1 = &polygon2.xy[j];
            let b2 = &polygon2.xy[(j + 1) % n2];
            min_dist = min_dist.min(segment_distance(a1, a2, b1, b2));
        }
    }
    min_dist
}

#[inline]
pub fn convex_hull(polygon: &Polygon) -> Polygon {
    let points = &polygon.xy;
    let mut sorted_points = points.clone();
    sorted_points.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });
    #[inline]
    fn ccw(p: &[f64; 2], q: &[f64; 2], r: &[f64; 2]) -> bool {
        (q[1] - p[1]) * (r[0] - q[0]) > (q[0] - p[0]) * (r[1] - q[1])
    }
    let mut lower_hull = Vec::new();
    for point in sorted_points.iter() {
        while lower_hull.len() >= 2
            && !ccw(
                &lower_hull[lower_hull.len() - 2],
                &lower_hull[lower_hull.len() - 1],
                point,
            )
        {
            lower_hull.pop();
        }
        lower_hull.push(*point);
    }
    let mut upper_hull = Vec::new();
    for point in sorted_points.iter().rev() {
        while upper_hull.len() >= 2
            && !ccw(
                &upper_hull[upper_hull.len() - 2],
                &upper_hull[upper_hull.len() - 1],
                point,
            )
        {
            upper_hull.pop();
        }
        upper_hull.push(*point);
    }
    // We can enforce de-duplicated points here
    lower_hull.pop();
    upper_hull.pop();
    lower_hull.append(&mut upper_hull);
    Polygon {
        xy: lower_hull,
        _deduped: true,
        _ordered: true,
    }
}

#[inline]
pub fn align_to(polygon: &Polygon, reference: &Polygon, scale: bool) -> Polygon {
    let points = &polygon.xy;
    let reference = &reference.xy;

    let n = points.len();

    let (effective_n, is_closed) =
        if n > 1 && points[0][0] == points[n - 1][0] && points[0][1] == points[n - 1][1] {
            (n - 1, true)
        } else {
            (n, false)
        };

    // Compute centroids
    let mut px = 0.0;
    let mut py = 0.0;
    let mut rx = 0.0;
    let mut ry = 0.0;

    for i in 0..effective_n {
        px += points[i][0];
        py += points[i][1];
        rx += reference[i][0];
        ry += reference[i][1];
    }

    let n_f64 = effective_n as f64;
    px /= n_f64;
    py /= n_f64;
    rx /= n_f64;
    ry /= n_f64;

    // Normalization factors
    let mut p_norm = 0.0;
    let mut r_norm = 0.0;

    for i in 0..effective_n {
        let px_i = points[i][0] - px;
        let py_i = points[i][1] - py;
        let rx_i = reference[i][0] - rx;
        let ry_i = reference[i][1] - ry;

        p_norm += px_i * px_i + py_i * py_i;
        r_norm += rx_i * rx_i + ry_i * ry_i;
    }

    if scale {
        p_norm = p_norm.sqrt();
        r_norm = r_norm.sqrt();
    } else {
        p_norm = 1.0;
        r_norm = 1.0;
    }

    if p_norm == 0.0 || r_norm == 0.0 {
        return Polygon {
            xy: Vec::new(),
            _deduped: polygon._deduped,
            _ordered: polygon._ordered,
        };
    }

    // Cross-covariance
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syx = 0.0;
    let mut syy = 0.0;

    for i in 0..effective_n {
        let px_i = (points[i][0] - px) / p_norm;
        let py_i = (points[i][1] - py) / p_norm;
        let rx_i = (reference[i][0] - rx) / r_norm;
        let ry_i = (reference[i][1] - ry) / r_norm;

        sxx += px_i * rx_i;
        sxy += px_i * ry_i;
        syx += py_i * rx_i;
        syy += py_i * ry_i;
    }

    // We can find optimal rotation via SVD
    let trace = sxx + syy;
    let off_diag = sxy - syx;
    let theta = off_diag.atan2(trace);
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    // Rotation and scale adjustment
    let mut aligned = Vec::with_capacity(n);

    for point in points.iter().take(effective_n) {
        let px_i = (point[0] - px) / p_norm;
        let py_i = (point[1] - py) / p_norm;

        let x_new = cos_theta * px_i - sin_theta * py_i;
        let y_new = sin_theta * px_i + cos_theta * py_i;

        aligned.push([x_new * r_norm + rx, y_new * r_norm + ry]);
    }

    if is_closed {
        aligned.push(aligned[0]);
    }

    Polygon {
        xy: aligned,
        _deduped: polygon._deduped,
        _ordered: polygon._ordered,
    }
}

#[inline]
pub fn fit_ellipse_lstsq(polygon: &Polygon) -> [f64; 4] {
    let resampled_polygon;
    let points = if polygon.xy.len() < 2 {
        // We did resampling before to avoid degenerate solutions
        // but the current updates seem to fix this problem; if
        // no more issues arise we can delete this chunk later
        resampled_polygon = resample(polygon, 32);
        &resampled_polygon.xy
    } else {
        &polygon.xy
    };

    let is_closed = points.len() > 1
        && points[0][0] == points[points.len() - 1][0]
        && points[0][1] == points[points.len() - 1][1];

    let point_count = if is_closed {
        points.len() - 1
    } else {
        points.len()
    };
    let (cx, cy) = points
        .iter()
        .take(point_count)
        .fold((0.0, 0.0), |(cx, cy), p| (cx + p[0], cy + p[1]));

    let n = point_count as f64;
    let cx = cx / n;
    let cy = cy / n;

    let matrix_size = if is_closed {
        points.len()
    } else {
        points.len() + 1
    };

    let mut col1 = Vec::with_capacity(matrix_size);
    let mut col2 = Vec::with_capacity(matrix_size);
    let mut col3 = Vec::with_capacity(matrix_size);
    let mut col4 = Vec::with_capacity(matrix_size);
    let mut col5 = Vec::with_capacity(matrix_size);

    for point in points.iter() {
        let x_centered = point[0] - cx;
        let y_centered = point[1] - cy;

        col1.push(x_centered * x_centered);
        col2.push(x_centered * y_centered);
        col3.push(y_centered * y_centered);
        col4.push(x_centered);
        col5.push(y_centered);
    }

    if !is_closed {
        let x_centered = points[0][0] - cx;
        let y_centered = points[0][1] - cy;

        col1.push(x_centered * x_centered);
        col2.push(x_centered * y_centered);
        col3.push(y_centered * y_centered);
        col4.push(x_centered);
        col5.push(y_centered);
    }

    let design: MatrixXx5<f64> = MatrixXx5::from_columns(&[
        DVector::from_vec(col1),
        DVector::from_vec(col2),
        DVector::from_vec(col3),
        DVector::from_vec(col4),
        DVector::from_vec(col5),
    ]);

    let y = DVector::from_element(matrix_size, 1.0);

    let epsilon = 1e-8;
    let results = lstsq::lstsq(&design, &y, epsilon).unwrap();

    let a: f64 = results.solution[0];
    let b: f64 = results.solution[1] / 2.0;
    let c: f64 = results.solution[2];
    let d: f64 = results.solution[3] / 2.0;
    let f: f64 = results.solution[4] / 2.0;
    let g: f64 = -1.0;

    let denominator = b * b - a * c;
    let numerator = 2.0 * (a * f * f + c * d * d + g * b * b - 2.0 * b * d * f - a * c * g);

    let factor = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let mut axis_length_major = (numerator / denominator / (factor - a - c)).sqrt();
    let mut axis_length_minor = (numerator / denominator / (-factor - a - c)).sqrt();

    let mut width_gt_height = true;
    if axis_length_major < axis_length_minor {
        width_gt_height = false;
        std::mem::swap(&mut axis_length_major, &mut axis_length_minor);
    }

    let mut r = (axis_length_minor / axis_length_major).powf(2.0);
    r = if r > 1.0 { 1.0 / r } else { r };
    let eccentricity = (1.0 - r).sqrt();

    let mut phi = if b == 0.0 {
        if a < c {
            0.0
        } else {
            std::f64::consts::PI / 2.0
        }
    } else {
        let mut inner = ((2.0 * b) / (a - c)).atan() / 2.0;
        inner += if a > c {
            std::f64::consts::PI / 2.0
        } else {
            0.0
        };
        inner
    };

    phi += if !width_gt_height {
        std::f64::consts::PI / 2.0
    } else {
        0.0
    };

    phi %= std::f64::consts::PI;

    [
        axis_length_major * 2.0,
        axis_length_minor * 2.0,
        eccentricity,
        phi,
    ]
}

#[inline]
pub fn area(polygon: &Polygon) -> f64 {
    let mut area = 0.0;
    let n = polygon.xy.len();
    for i in 0..n - 1 {
        let p1 = &polygon.xy[i];
        let p2 = &polygon.xy[i + 1];
        area += p1[0] * p2[1] - p2[0] * p1[1];
    }
    if polygon.xy[0][0] != polygon.xy[n - 1][0] || polygon.xy[0][1] != polygon.xy[n - 1][1] {
        let p1 = &polygon.xy[n - 1];
        let p2 = &polygon.xy[0];
        area += p1[0] * p2[1] - p2[0] * p1[1];
    }
    area.abs() / 2.0
}

#[inline]
pub fn area_bbox(polygon: &Polygon) -> f64 {
    let (mut xmin, mut ymin) = (polygon.xy[0][0], polygon.xy[0][1]);
    let (mut xmax, mut ymax) = (polygon.xy[0][0], polygon.xy[0][1]);
    for point in polygon.xy.iter().skip(1) {
        xmin = if point[0] < xmin { point[0] } else { xmin };
        ymin = if point[1] < ymin { point[1] } else { ymin };
        xmax = if point[0] > xmax { point[0] } else { xmax };
        ymax = if point[1] > ymax { point[1] } else { ymax };
    }
    (xmax - xmin) * (ymax - ymin)
}

#[inline]
pub fn area_convex(polygon: &Polygon) -> f64 {
    area(&polygon.convex_hull())
}

#[inline]
pub fn perimeter(polygon: &Polygon) -> f64 {
    let n_points = polygon.xy.len();
    let mut perimeter = 0.0;
    for i in 0..n_points - 1 {
        let dx = polygon.xy[i][0] - polygon.xy[i + 1][0];
        let dy = polygon.xy[i][1] - polygon.xy[i + 1][1];
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    if polygon.xy[0][0] != polygon.xy[n_points - 1][0]
        || polygon.xy[0][1] != polygon.xy[n_points - 1][1]
    {
        let dx = polygon.xy[polygon.xy.len() - 1][0] - polygon.xy[0][0];
        let dy = polygon.xy[polygon.xy.len() - 1][1] - polygon.xy[0][1];
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    perimeter
}

#[inline]
pub fn elongation(polygon: &Polygon) -> f64 {
    let (mut xmin, mut ymin) = (polygon.xy[0][0], polygon.xy[0][1]);
    let (mut xmax, mut ymax) = (polygon.xy[0][0], polygon.xy[0][1]);
    for point in polygon.xy.iter().skip(1) {
        xmin = if point[0] < xmin { point[0] } else { xmin };
        ymin = if point[1] < ymin { point[1] } else { ymin };
        xmax = if point[0] > xmax { point[0] } else { xmax };
        ymax = if point[1] > ymax { point[1] } else { ymax };
    }
    let width = xmax - xmin;
    let height = ymax - ymin;
    if height == 0.0 {
        return if width == 0.0 { 1.0 } else { 0.0 };
    }
    let elongation = width / height;
    if elongation > 1.0 {
        1.0 / elongation
    } else {
        elongation
    }
}

#[inline]
pub fn thread_length(polygon: &Polygon) -> f64 {
    let perimeter = perimeter(polygon);
    let area = area(polygon);

    let left = perimeter.powi(2);
    let right = 16.0 * area;

    let coefficient = if left <= right {
        0.0
    } else {
        (left - right).sqrt()
    };

    (perimeter + coefficient) / 4.0
}

#[inline]
pub fn solidity(polygon: &Polygon) -> f64 {
    let area_convex = area_convex(polygon);
    if area_convex == 0.0 {
        0.0
    } else {
        area(polygon) / area_convex
    }
}

#[inline]
pub fn extent(polygon: &Polygon) -> f64 {
    let area_bbox = area_bbox(polygon);
    if area_bbox == 0.0 {
        0.0
    } else {
        area(polygon) / area_bbox
    }
}

#[inline]
pub fn form_factor(polygon: &Polygon) -> f64 {
    let perimeter = perimeter(polygon);
    if perimeter == 0.0 {
        0.0
    } else {
        (4.0 * std::f64::consts::PI * area(polygon)) / (perimeter * perimeter)
    }
}

#[inline]
pub fn equivalent_diameter(polygon: &Polygon) -> f64 {
    (area(polygon) / std::f64::consts::PI).sqrt() * 2.0
}

#[inline]
pub fn eccentricity(polygon: &Polygon) -> f64 {
    let ellipse = fit_ellipse_lstsq(polygon);
    ellipse[2]
}

#[inline]
pub fn major_axis_length(polygon: &Polygon) -> f64 {
    let ellipse = fit_ellipse_lstsq(polygon);
    ellipse[0]
}

#[inline]
pub fn minor_axis_length(polygon: &Polygon) -> f64 {
    let ellipse = fit_ellipse_lstsq(polygon);
    ellipse[1]
}

#[inline]
pub fn min_radius(polygon: &Polygon) -> f64 {
    let centroid = polygon_centroid(polygon);
    let mut min_radius = f64::MAX;

    for i in 0..polygon.xy.len() {
        let p1 = &polygon.xy[i];
        let p2 = &polygon.xy[(i + 1) % polygon.xy.len()];

        let distance = point_to_line_segment_distance(centroid.x, centroid.y, p1, p2);
        if distance < min_radius {
            min_radius = distance;
        }
    }

    min_radius
}

#[inline]
pub fn max_radius(polygon: &Polygon) -> f64 {
    let centroid = polygon_centroid(polygon);
    let mut maximum_radius = 0.0;
    for &point in polygon.xy.iter() {
        let distance = (centroid.x - point[0]) * (centroid.x - point[0])
            + (centroid.y - point[1]) * (centroid.y - point[1]);
        if distance > maximum_radius {
            maximum_radius = distance;
        }
    }
    maximum_radius.sqrt()
}

#[inline]
pub fn mean_radius(polygon: &Polygon) -> f64 {
    let centroid = polygon_centroid(polygon);
    let n = polygon.xy.len();
    let include_last = (polygon.xy.last().unwrap()[0] == polygon.xy[0][0]
        && polygon.xy.last().unwrap()[1] == polygon.xy[0][1]) as usize;
    let mut mean_radius = 0.0;
    for point in polygon.xy.iter().take(n - include_last) {
        let distance = (centroid.x - point[0]) * (centroid.x - point[0])
            + (centroid.y - point[1]) * (centroid.y - point[1]);
        mean_radius += distance.sqrt();
    }
    mean_radius / (n - include_last) as f64
}

#[inline]
pub fn min_feret(polygon: &Polygon) -> f64 {
    let hull = convex_hull(polygon).xy;
    let n = hull.len();

    let mut edges = Vec::with_capacity(n);
    let mut edge_lengths = Vec::with_capacity(n);

    for i in 0..n {
        let p1 = hull[i];
        let p2 = hull[(i + 1) % n];
        let edge = [p2[0] - p1[0], p2[1] - p1[1]];
        let len = (edge[0] * edge[0] + edge[1] * edge[1]).sqrt();
        edges.push(edge);
        edge_lengths.push(len);
    }

    let mut min_width = f64::MAX;

    for i in 0..n {
        if edge_lengths[i] < f64::EPSILON {
            continue;
        }

        let p1 = hull[i];
        let edge = edges[i];
        let edge_len = edge_lengths[i];

        let mut max_dist: f64 = 0.0;

        for &point in &hull {
            let to_point = [point[0] - p1[0], point[1] - p1[1]];
            let dist = (edge[0] * to_point[1] - edge[1] * to_point[0]).abs() / edge_len;
            max_dist = max_dist.max(dist);
        }

        if max_dist > f64::EPSILON {
            min_width = min_width.min(max_dist);
        }
    }

    min_width
}

#[inline]
pub fn max_feret(polygon: &Polygon) -> f64 {
    let hull = convex_hull(polygon).xy;
    let n = hull.len();

    let mut max_dist_sq: f64 = 0.0;
    let mut j = 1;

    for i in 0..n {
        loop {
            let next_j = (j + 1) % n;

            let dx1 = hull[i][0] - hull[j][0];
            let dy1 = hull[i][1] - hull[j][1];
            let dist1_sq = dx1 * dx1 + dy1 * dy1;

            let dx2 = hull[i][0] - hull[next_j][0];
            let dy2 = hull[i][1] - hull[next_j][1];
            let dist2_sq = dx2 * dx2 + dy2 * dy2;

            if dist2_sq > dist1_sq {
                j = next_j;
            } else {
                break;
            }
        }

        let dx = hull[i][0] - hull[j][0];
        let dy = hull[i][1] - hull[j][1];
        let dist_sq = dx * dx + dy * dy;
        max_dist_sq = max_dist_sq.max(dist_sq);
    }

    max_dist_sq.sqrt()
}

#[inline]
pub fn descriptors(polygon: &Polygon) -> [f64; 18] {
    let n = polygon.xy.len();

    let is_closed =
        polygon.xy[0][0] == polygon.xy[n - 1][0] && polygon.xy[0][1] == polygon.xy[n - 1][1];

    let mut area_ = 0.0;
    for i in 0..n - 1 {
        let p1 = &polygon.xy[i];
        let p2 = &polygon.xy[i + 1];
        area_ += p1[0] * p2[1] - p2[0] * p1[1];
    }
    if !is_closed {
        let p1 = &polygon.xy[n - 1];
        let p2 = &polygon.xy[0];
        area_ += p1[0] * p2[1] - p2[0] * p1[1];
    }
    area_ = area_.abs() / 2.0;

    let mut xmin = polygon.xy[0][0];
    let mut ymin = polygon.xy[0][1];
    let mut xmax = polygon.xy[0][0];
    let mut ymax = polygon.xy[0][1];

    for point in polygon.xy.iter().skip(1) {
        xmin = xmin.min(point[0]);
        ymin = ymin.min(point[1]);
        xmax = xmax.max(point[0]);
        ymax = ymax.max(point[1]);
    }

    let width = xmax - xmin;
    let height = ymax - ymin;
    let area_bbox = width * height;

    let mut perimeter = 0.0;
    for i in 0..n - 1 {
        let dx = polygon.xy[i][0] - polygon.xy[i + 1][0];
        let dy = polygon.xy[i][1] - polygon.xy[i + 1][1];
        perimeter += (dx * dx + dy * dy).sqrt();
    }
    if !is_closed {
        let dx = polygon.xy[n - 1][0] - polygon.xy[0][0];
        let dy = polygon.xy[n - 1][1] - polygon.xy[0][1];
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    let centroid = polygon_centroid(polygon);

    let convex_hull = polygon.convex_hull();
    let area_convex = area(&convex_hull);

    let ellipse = fit_ellipse_lstsq(polygon);
    let major_axis_length = ellipse[0];
    let minor_axis_length = ellipse[1];
    let eccentricity = ellipse[2];

    let effective_n = if is_closed { n - 1 } else { n };
    let mut min_radius = f64::MAX;
    let mut max_radius = 0.0;
    let mut radius_sum = 0.0;

    for i in 0..effective_n {
        let point = &polygon.xy[i];

        let dx = centroid.x - point[0];
        let dy = centroid.y - point[1];
        let dist_sq = dx * dx + dy * dy;
        let dist = dist_sq.sqrt();

        if dist > max_radius {
            max_radius = dist;
        }

        radius_sum += dist;
    }

    for i in 0..polygon.xy.len() {
        let p1 = &polygon.xy[i];
        let p2 = &polygon.xy[(i + 1) % polygon.xy.len()];
        let distance = point_to_line_segment_distance(centroid.x, centroid.y, p1, p2);
        if distance < min_radius {
            min_radius = distance;
        }
    }

    let mean_radius = radius_sum / effective_n as f64;

    let hull_points = &convex_hull.xy;
    let hull_n = hull_points.len();

    let mut min_feret = f64::MAX;
    let mut max_feret = 0.0;

    if hull_n > 2 {
        for i in 0..hull_n {
            let p1 = hull_points[i];
            let p2 = hull_points[(i + 1) % hull_n];
            let edge = [p2[0] - p1[0], p2[1] - p1[1]];
            let edge_len = (edge[0] * edge[0] + edge[1] * edge[1]).sqrt();

            if edge_len < f64::EPSILON {
                continue;
            }

            let mut max_dist: f64 = 0.0;
            for &point in hull_points {
                let to_point = [point[0] - p1[0], point[1] - p1[1]];
                let dist = (edge[0] * to_point[1] - edge[1] * to_point[0]).abs() / edge_len;
                max_dist = max_dist.max(dist);
            }

            if max_dist > f64::EPSILON {
                min_feret = min_feret.min(max_dist);
            }
        }

        let mut max_dist_sq: f64 = 0.0;
        let mut j = 1;

        for i in 0..hull_n {
            loop {
                let next_j = (j + 1) % hull_n;

                let dx1 = hull_points[i][0] - hull_points[j][0];
                let dy1 = hull_points[i][1] - hull_points[j][1];
                let dist1_sq = dx1 * dx1 + dy1 * dy1;

                let dx2 = hull_points[i][0] - hull_points[next_j][0];
                let dy2 = hull_points[i][1] - hull_points[next_j][1];
                let dist2_sq = dx2 * dx2 + dy2 * dy2;

                if dist2_sq > dist1_sq {
                    j = next_j;
                } else {
                    break;
                }
            }

            let dx = hull_points[i][0] - hull_points[j][0];
            let dy = hull_points[i][1] - hull_points[j][1];
            let dist_sq = dx * dx + dy * dy;
            max_dist_sq = max_dist_sq.max(dist_sq);
        }

        max_feret = max_dist_sq.sqrt();
    }

    let elongation = if height == 0.0 {
        if width == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        let ratio = width / height;
        if ratio > 1.0 {
            1.0 / ratio
        } else {
            ratio
        }
    };

    let thread_length = {
        let left = perimeter.powi(2);
        let right = 16.0 * area_;
        let coefficient = if left <= right {
            0.0
        } else {
            (left - right).sqrt()
        };
        (perimeter + coefficient) / 4.0
    };

    let solidity = if area_convex == 0.0 {
        0.0
    } else {
        area_ / area_convex
    };
    let extent = if area_bbox == 0.0 {
        0.0
    } else {
        area_ / area_bbox
    };
    let form_factor = if perimeter == 0.0 {
        0.0
    } else {
        (4.0 * std::f64::consts::PI * area_) / (perimeter * perimeter)
    };
    let equivalent_diameter = (area_ / std::f64::consts::PI).sqrt() * 2.0;

    [
        area_,
        area_bbox,
        area_convex,
        perimeter,
        elongation,
        thread_length,
        solidity,
        extent,
        form_factor,
        equivalent_diameter,
        eccentricity,
        major_axis_length,
        minor_axis_length,
        min_radius,
        max_radius,
        mean_radius,
        min_feret,
        max_feret,
    ]
}
