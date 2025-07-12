// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use super::Point2d;

pub fn add(a: &Point2d, b: &Point2d) -> Point2d {
    Point2d {
        x: a.x + b.x,
        y: a.y + b.y,
    }
}

pub fn sub(a: &Point2d, b: &Point2d) -> Point2d {
    Point2d {
        x: a.x - b.x,
        y: a.y - b.y,
    }
}

pub fn mul(a: &Point2d, b: &Point2d) -> Point2d {
    Point2d {
        x: a.x * b.x,
        y: a.y * b.y,
    }
}

pub fn div(a: &Point2d, b: &Point2d) -> Point2d {
    Point2d {
        x: a.x / (b.x + f64::EPSILON),
        y: a.y / (b.y + f64::EPSILON),
    }
}

pub fn dot(a: &Point2d, b: &Point2d) -> f64 {
    a.x * b.x + a.y * b.y
}

pub fn d_l1(a: &Point2d, b: &Point2d) -> f64 {
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

pub fn d_l2(a: &Point2d, b: &Point2d) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

pub fn d_chebyshev(a: &Point2d, b: &Point2d) -> f64 {
    (a.x - b.x).abs().max((a.y - b.y).abs())
}

pub fn d_cosine(a: &Point2d, b: &Point2d) -> f64 {
    let p = (a.x.powi(2) + a.y.powi(2)).sqrt();
    let q = (b.x.powi(2) + b.y.powi(2)).sqrt();
    if p < f64::EPSILON || q < f64::EPSILON {
        f64::NAN
    } else {
        1.0 - dot(a, b) / (p * q)
    }
}

pub fn interp(a: &Point2d, b: &Point2d, t: f64) -> Point2d {
    let t: f64 = t.clamp(0., 1.);
    Point2d {
        x: a.x * (1.0 - t) + b.x * t,
        y: a.y * (1.0 - t) + b.y * t,
    }
}
