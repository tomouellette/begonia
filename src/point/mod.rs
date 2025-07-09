// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

mod ops;

use numpy::PyArray1;
use pyo3::prelude::PyResult;
use pyo3::{pyclass, pymethods, Bound, Python};

/// A two-dimensional point defined by x and y coordinates
///
/// This struct defines a simply two-dimensional point with various methods for computing
/// distances and performing translations/transformations
///
/// Attributes
/// ----------
/// x : float
///     X-coordinate
/// y : float
///     Y-coordinate
#[pyclass]
#[derive(Clone, Copy)]
pub struct Point2d {
    #[pyo3(get, set)]
    pub x: f64,

    #[pyo3(get, set)]
    pub y: f64,
}

#[pymethods]
impl Point2d {
    /// Initialize a new two-dimensional point
    ///
    /// Parameters
    /// ----------
    /// x : float | int
    ///     Horizontal or x-coordinate
    /// y : float | int
    ///     Vertical or y-coordinate
    ///
    /// Returns
    /// -------
    /// Point2d
    ///     New point
    #[new]
    pub fn new(x: f64, y: f64) -> Self {
        Point2d { x, y }
    }

    /// Test equivalency between points in python
    pub fn __eq__(&self, point: &Point2d) -> bool {
        self.x == point.x && self.y == point.y
    }

    /// Class display formatting
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Point2d(x={},y={})", self.x, self.y))
    }

    /// Convert point to a list
    ///
    /// Returns
    /// -------
    /// list
    ///     A list with x and y coordinate
    pub fn to_list(&self) -> [f64; 2] {
        [self.x, self.y]
    }

    /// Convert point to a numpy array
    ///
    /// Returns
    /// -------
    /// np.ndarray
    ///     A numpy array with x and y coordinate
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &[self.x, self.y])
    }

    /// Perform element-wise addition between points
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// Point2d
    ///     New point with element-wise sum
    pub fn add(&self, point: &Point2d) -> Point2d {
        ops::add(self, point)
    }

    /// Perform element-wise subtraction between points
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// Point2d
    ///     New point with element-wise difference
    pub fn sub(&self, point: &Point2d) -> Point2d {
        ops::sub(self, point)
    }

    /// Perform element-wise multiplication between points
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// Point2d
    ///     New point with element-wise product
    pub fn mul(&self, point: &Point2d) -> Point2d {
        ops::mul(self, point)
    }

    /// Perform element-wise division between points
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// Point2d
    ///     New point with element-wise quotient
    pub fn div(&self, point: &Point2d) -> Point2d {
        ops::div(self, point)
    }

    /// Compute the l1 (manhattan) distance between two points
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// float
    ///     The l1 distance between the points
    pub fn d_l1(&self, point: &Point2d) -> f64 {
        ops::d_l1(self, point)
    }

    /// Compute the l2 (euclidean) distance between two points
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// float
    ///     The l2 distance between the points
    pub fn d_l2(&self, point: &Point2d) -> f64 {
        ops::d_l2(self, point)
    }

    /// Compute the Chebyshev distance between two points
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// float
    ///     The Chebyshev distance between the points
    pub fn d_chebyshev(&self, point: &Point2d) -> f64 {
        ops::d_chebyshev(self, point)
    }

    /// Compute the cosine distance between two points
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// float
    ///     The cosine distance between the points
    pub fn d_cosine(&self, point: &Point2d) -> f64 {
        ops::d_cosine(self, point)
    }

    /// Linear interpolate point towards a new point
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    /// t : float
    ///     A weight between 0 and 1 - values below 0 or greater than 1 are clipped
    ///
    /// Returns
    /// -------
    /// Point2d
    ///     A new point updated based on interpolation weight
    pub fn interp(&self, point: &Point2d, t: f64) -> Point2d {
        ops::interp(self, point, t)
    }
}

impl PartialEq for Point2d {
    /// Test equivalency between points in rust
    fn eq(&self, point: &Self) -> bool {
        (self.x - point.x).abs() < f64::EPSILON && (self.y - point.y).abs() < f64::EPSILON
    }
}
