// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

mod ops;

use crate::Point2d;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::PyResult;
use pyo3::{pyclass, pymethods, Bound, PyRef, Python};

/// A polygon represented by a sequence of 2D points.
///
/// This struct represents a polygon as a vector of (float, float) vertices. The polygon
/// maintains internal state about whether its points have been deduplicated and ordered.
/// All operations on polygons or between polygons and points are implemented int Rust.
///
/// Attributes
/// ----------
/// xy : list
///     A public vector of representing the polygon vertices
/// _deduped : bool
///     Private flag indicating whether duplicate points have been removed
/// _ordered : bool
///     Private flag indicating whether the points are order
#[pyclass]
#[derive(PartialEq)]
pub struct Polygon {
    #[pyo3(get, set)]
    pub xy: Vec<[f64; 2]>,

    _deduped: bool,
    _ordered: bool,
}

#[pymethods]
impl Polygon {
    /// Initialize a new polygon
    ///
    /// Parameters
    /// ----------
    /// xy : List[Union[float, int]] | np.ndarray
    ///     A (N, 2) list or array of N xy polygon coordinates
    /// deduped : bool
    ///     Set to True, if points are guaranteed to be deduplicated
    /// ordered : bool
    ///     Set to True, if points are guaranteed to be ordered
    ///
    /// Returns
    /// -------
    /// Point2d
    ///     New point
    #[new]
    #[pyo3(signature = (xy, deduped=false, ordered=false))]
    pub fn new(xy: Vec<[f64; 2]>, deduped: bool, ordered: bool) -> PyResult<Self> {
        if xy.len() < 3 {
            return Err(PyValueError::new_err(
                "Polygon must have at least 3 points.",
            ));
        }

        Ok(Self {
            xy,
            _deduped: deduped,
            _ordered: ordered,
        })
    }

    /// Test equivalency between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if all points are equal between polygons
    pub fn __eq__(&self, polygon: &Polygon) -> bool {
        self.xy == polygon.xy
    }

    /// Test approximate equivalency between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    /// eps : float
    ///     Points are equal if error is below specified threshold
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if all points are approximately equal between polygons
    #[pyo3(signature = (polygon, eps=1e-15))]
    pub fn eq(&self, polygon: &Polygon, eps: f64) -> bool {
        self.xy
            .iter()
            .zip(polygon.xy.iter())
            .all(|(a, b)| (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps)
    }

    /// Class display formatting
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Polygon(n={},deduped={},ordered={})",
            self.xy.len(),
            self._deduped,
            self._ordered
        ))
    }

    /// Number of points in the polygon
    ///
    /// Returns
    /// -------
    /// int
    ///     Number of points in polygon
    pub fn __len__(&self) -> usize {
        self.xy.len()
    }

    /// Iterate through polygon points
    ///
    /// Returns
    /// -------
    /// PolygonIterator
    ///     An iterator that returns consecutive points
    pub fn __iter__(&self) -> PolygonIterator {
        PolygonIterator {
            points: self.xy.clone(),
            index: 0,
        }
    }

    /// Convert point to a list
    ///
    /// Returns
    /// -------
    /// list
    ///     A list with x and y coordinate
    pub fn to_list(&self) -> Vec<[f64; 2]> {
        self.xy.to_vec()
    }

    /// Convert point to a numpy array
    ///
    /// Returns
    /// -------
    /// np.ndarray
    ///     A numpy array with x and y coordinate
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        PyArray1::from_vec(py, self.xy.to_vec().into_flattened())
            .reshape([self.xy.len(), 2])
            .unwrap()
    }

    /// Push an [x, y] point onto the polygon
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A Point2d object
    pub fn push(&mut self, point: [f64; 2]) {
        self.xy.push(point);
    }

    /// Push a Point2d onto the polygon
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    pub fn push_point2d(&mut self, point: Point2d) {
        self.xy.push([point.x, point.y]);
    }

    /// Adds a scalar to each point in the polygon in-place
    ///
    /// Parameters
    /// ----------
    /// scalar : float
    ///     A float to add to all points
    pub fn add_scalar_inplace(&mut self, scalar: f64) {
        ops::add_scalar_inplace(self, scalar)
    }

    /// Subtract a scalar from each point in the polygon in-place
    ///
    /// Parameters
    /// ----------
    /// scalar : float
    ///     A float to subtract from all points
    pub fn sub_scalar_inplace(&mut self, scalar: f64) {
        ops::sub_scalar_inplace(self, scalar)
    }

    /// Multiply each point in the polygon by a scalar in-place
    ///
    /// Parameters
    /// ----------
    /// scalar : float
    ///     A float to multiply all points by
    pub fn mul_scalar_inplace(&mut self, scalar: f64) {
        ops::mul_scalar_inplace(self, scalar)
    }

    /// Divide each point in the polygon by a scalar in-place
    ///
    /// Parameters
    /// ----------
    /// scalar : float
    ///     A float to divide all points by
    pub fn div_scalar_inplace(&mut self, scalar: f64) {
        ops::div_scalar_inplace(self, scalar)
    }

    /// Adds a scalar to each point in the polygon
    ///
    /// Parameters
    /// ----------
    /// scalar : float
    ///     A float to add to all points
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon divided by a scalar
    pub fn add_scalar(&self, scalar: f64) -> Polygon {
        ops::add_scalar(self, scalar)
    }

    /// Subtract a scalar from each point in the polygon
    ///
    /// Parameters
    /// ----------
    /// scalar : float
    ///     A float to subtract from all points
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon divided by a scalar
    pub fn sub_scalar(&self, scalar: f64) -> Polygon {
        ops::sub_scalar(self, scalar)
    }

    /// Multiply each point by a scalar in the polygon
    ///
    /// Parameters
    /// ----------
    /// scalar : float
    ///     A float to multiply all points by
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon divided by a scalar
    pub fn mul_scalar(&self, scalar: f64) -> Polygon {
        ops::mul_scalar(self, scalar)
    }

    /// Divide each point in the polygon by a scalar
    ///
    /// Parameters
    /// ----------
    /// scalar : float
    ///     A float to divide all points by
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon divided by a scalar
    pub fn div_scalar(&self, scalar: f64) -> Polygon {
        ops::div_scalar(self, scalar)
    }

    /// Add an [x, y] point to a polygon in-place
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    pub fn add_point_inplace(&mut self, point: [f64; 2]) {
        ops::add_point_inplace(point[0], point[1], self);
    }

    /// Add a Point2d object to a polygon in-place
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    pub fn add_point2d_inplace(&mut self, point: &Point2d) {
        ops::add_point_inplace(point.x, point.y, self);
    }

    /// Subtract an [x, y] point from a polygon in-place
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    pub fn sub_point_inplace(&mut self, point: [f64; 2]) {
        ops::sub_point_inplace(point[0], point[1], self);
    }

    /// Subtract a Point2d from a polygon in-place
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    pub fn sub_point2d_inplace(&mut self, point: &Point2d) {
        ops::sub_point_inplace(point.x, point.y, self);
    }

    /// Multiply each point in a polygon by an [x, y] point in-place
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    pub fn mul_point_inplace(&mut self, point: [f64; 2]) {
        ops::mul_point_inplace(point[0], point[1], self);
    }

    /// Multiply each point in a polygon by a Point2d object in-place
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    pub fn mul_point2d_inplace(&mut self, point: &Point2d) {
        ops::mul_point_inplace(point.x, point.y, self);
    }

    /// Divide each point in a polygon by an [x, y] point in-place
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    pub fn div_point_inplace(&mut self, point: [f64; 2]) {
        ops::div_point_inplace(point[0], point[1], self);
    }

    /// Divide each point in a polygon by a Point2d object in-place
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    pub fn div_point2d_inplace(&mut self, point: &Point2d) {
        ops::div_point_inplace(point.x, point.y, self);
    }

    /// Add an [x, y] point to a polygon
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with input point added to every point in polygon
    pub fn add_point(&self, point: [f64; 2]) -> Polygon {
        ops::add_point(point[0], point[1], self)
    }

    /// Add a Point2d object to a polygon in-place
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with Point2d object added to every point in polygon
    pub fn add_point2d(&self, point: &Point2d) -> Polygon {
        ops::add_point(point.x, point.y, self)
    }

    /// Subtract an [x, y] point from a polygon
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with input point subtracted from every point in polygon
    pub fn sub_point(&self, point: [f64; 2]) -> Polygon {
        ops::sub_point(point[0], point[1], self)
    }

    /// Subtract a Point2d from a polygon
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with Point2d object subtracted from every point in polygon
    pub fn sub_point2d(&self, point: &Point2d) -> Polygon {
        ops::sub_point(point.x, point.y, self)
    }

    /// Multiply each point in a polygon by an [x, y] point
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with every point in polygon multiplied input pint
    pub fn mul_point(&self, point: [f64; 2]) -> Polygon {
        ops::mul_point(point[0], point[1], self)
    }

    /// Multiply each point in a polygon by a Point2d object
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with every point in polygon multiplied by Point2d object
    pub fn mul_point2d(&self, point: &Point2d) -> Polygon {
        ops::mul_point(point.x, point.y, self)
    }

    /// Divide each point in a polygon by an [x, y] point
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with every point in polygon divided by input point
    pub fn div_point(&self, point: [f64; 2]) -> Polygon {
        ops::div_point(point[0], point[1], self)
    }

    /// Divide each point in a polygon by a Point2d object
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A Point2d object
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with every point in polygon divided by Point2d object
    pub fn div_point2d(&self, point: &Point2d) -> Polygon {
        ops::div_point(point.x, point.y, self)
    }

    /// Perform element-wise addition between polygons in-place
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with element-wise sum
    pub fn add_inplace(&mut self, polygon: &Polygon) {
        ops::add_inplace(self, polygon)
    }

    /// Perform element-wise subtraction between polygons in-place
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    pub fn sub_inplace(&mut self, polygon: &Polygon) {
        ops::sub_inplace(self, polygon)
    }

    /// Perform element-wise multiplication between polygons in-place
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    pub fn mul_inplace(&mut self, polygon: &Polygon) {
        ops::mul_inplace(self, polygon);
    }

    /// Perform element-wise division between polygons in-place
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    pub fn div_inplace(&mut self, polygon: &Polygon) {
        ops::div_inplace(self, polygon);
    }

    /// Perform element-wise addition between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with element-wise sum
    pub fn add(&self, polygon: &Polygon) -> Polygon {
        ops::add(self, polygon)
    }

    /// Perform element-wise subtraction between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with element-wise difference
    pub fn sub(&self, polygon: &Polygon) -> Polygon {
        ops::sub(self, polygon)
    }

    /// Perform element-wise multiplication between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with element-wise multiplication
    pub fn mul(&self, polygon: &Polygon) -> Polygon {
        ops::mul(self, polygon)
    }

    /// Perform element-wise division between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with element-wise quotient
    pub fn div(&self, polygon: &Polygon) -> Polygon {
        ops::div(self, polygon)
    }

    /// Get the center coordinates of the polygon
    ///
    /// Returns
    /// -------
    /// Point2d
    ///     Center (x, y) of polygon returned as a point
    pub fn center(&self) -> Point2d {
        ops::polygon_center(self)
    }

    /// Get the centroid coordinates of the polygon
    ///
    /// Returns
    /// -------
    /// Point2d
    ///     Centroid (x, y) of polygon returned as a point
    pub fn centroid(&mut self) -> Point2d {
        if !self._ordered {
            self.order_inplace();
        }
        ops::polygon_centroid(self)
    }

    /// Compute the l1 (manhattan) distance between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// float
    ///     l1 distance
    pub fn d_l1(&self, polygon: &Polygon) -> f64 {
        ops::d_l1(self, polygon)
    }

    /// Compute the l2 (euclidean) distance between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// float
    ///     l2 distance
    pub fn d_l2(&self, polygon: &Polygon) -> f64 {
        ops::d_l2(self, polygon)
    }

    /// Compute the chebyshev distance between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// float
    ///     Chebyshev distance
    pub fn d_chebyshev(&self, polygon: &Polygon) -> f64 {
        ops::d_chebyshev(self, polygon)
    }

    /// Compute the cosine distance between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// float
    ///     Cosine distance
    ///
    /// Notes
    /// -----
    /// The cosine distance between two polygons isn't meaningful
    /// so we can drop this method if there is no downstream use.
    pub fn d_cosine(&self, polygon: &Polygon) -> f64 {
        ops::d_cosine(self, polygon)
    }

    /// Compute the Hausdorff distance between polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    /// n : int
    ///     Number of points to sample per edge when the computing Hausdorff distance.
    ///     The larger the number of points, the more accurate the distance calculation.
    ///
    /// Returns
    /// -------
    /// float
    ///     Hausdorff distance
    #[pyo3(signature = (polygon, n=10))]
    pub fn d_hausdorff(&self, polygon: &Polygon, n: usize) -> f64 {
        ops::d_hausdorff(self, polygon, n)
    }

    /// De-duplicate points in-place
    ///
    /// Returns
    /// -------
    /// None
    ///     Polygon is de-duplicated and mutated in-place
    pub fn dedup_inplace(&mut self) {
        self._deduped = true;
        ops::dedup_inplace(self)
    }

    /// Generate a new polygon with de-duplicated points
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with de-duplicated points removed
    pub fn dedup(&self) -> Polygon {
        ops::dedup(self)
    }

    /// De-duplicate points in-place (faster but no guarantee on original ordering)
    ///
    /// Returns
    /// -------
    /// None
    ///     Polygon is de-duplicated and mutated in-place
    pub fn dedup_unstable_inplace(&mut self) {
        self._deduped = true;
        ops::dedup_unstable_inplace(self)
    }

    /// Generate a new polygon with de-duplicated points (faster but no guarantee on original ordering)
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with de-duplicated points removed
    pub fn dedup_unstable(&self) -> Polygon {
        ops::dedup_unstable(self)
    }

    /// Order polygon vertices in-place
    ///
    /// Returns
    /// -------
    /// None
    ///     Polygon vertices are properly ordered and mutated in-place
    pub fn order_inplace(&mut self) {
        self._ordered = true;
        ops::order_inplace(self)
    }

    /// Generate a new polygon with ordered polygon vertices
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with properly ordered polygon vertices
    pub fn order(&self) -> Polygon {
        ops::order(self)
    }

    /// Resample polygon outline in-place
    ///
    /// Parameters
    /// ----------
    /// n_points : int
    ///     Number of points to resample outline to
    ///
    /// Returns
    /// -------
    /// None
    ///     Polygon vertices are properly ordered and mutated in-place
    pub fn resample_inplace(&mut self, n_points: usize) {
        ops::resample_inplace(self, n_points)
    }

    /// Generate a new polygon with resampled outline
    ///
    /// Parameters
    /// ----------
    /// n_points : int
    ///     Number of points to resample outline to
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     New polygon with properly ordered polygon vertices
    pub fn resample(&self, n_points: usize) -> Polygon {
        ops::resample(self, n_points)
    }

    /// Check if the polygon contains an [x, y] point
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if point is contained inside polygon
    pub fn encloses_point(&self, point: [f64; 2]) -> bool {
        ops::encloses_point(point[0], point[1], self)
    }

    /// Check if the polygon contains a Point2d object
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if point is contained inside polygon
    pub fn encloses_point2d(&self, point: &Point2d) -> bool {
        ops::encloses_point(point.x, point.y, self)
    }

    /// Distance from [x, y] point to polygon center
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance from [x, y] point to polygon center
    pub fn distance_to_point_center(&self, point: [f64; 2]) -> f64 {
        ops::distance_to_point_center(point[0], point[1], self)
    }

    /// Distance from Point2d object to polygon center
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance from Point2d object to polygon center
    pub fn distance_to_point2d_center(&self, point: &Point2d) -> f64 {
        ops::distance_to_point_center(point.x, point.y, self)
    }

    /// Distance from [x, y] point to polygon centroid
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance from [x, y] to polygon centroid
    pub fn distance_to_point_centroid(&self, point: [f64; 2]) -> f64 {
        ops::distance_to_point_centroid(point[0], point[1], self)
    }

    /// Distance from Point2d object to polygon centroid
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance from Point2d to polygon centroid
    pub fn distance_to_point2d_centroid(&self, point: &Point2d) -> f64 {
        ops::distance_to_point_centroid(point.x, point.y, self)
    }

    /// Distance from [x, y] point to nearest polygon vertex
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance from nearest polygon vertex to [x, y] point
    pub fn distance_to_point_vertex(&self, point: [f64; 2]) -> f64 {
        ops::distance_to_point_vertex(point[0], point[1], self)
    }

    /// Distance from Point2d object to nearest polygon vertex
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance from nearest polygon vertex to Point2d object
    pub fn distance_to_point2d_vertex(&self, point: &Point2d) -> f64 {
        ops::distance_to_point_vertex(point.x, point.y, self)
    }

    /// Distance from [x, y] point to nearest polygon edge
    ///
    /// Parameters
    /// ----------
    /// point : list | np.ndarray
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance from nearest polygon edge to [x, y] point
    pub fn distance_to_point_edge(&self, point: [f64; 2]) -> f64 {
        ops::distance_to_point_edge(point[0], point[1], self)
    }

    /// Distance from Point2d object to nearest polygon edge
    ///
    /// Parameters
    /// ----------
    /// point : Point2d
    ///     A two-dimensional point
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance from nearest polygon edge to Point2d object
    pub fn distance_to_point2d_edge(&self, point: &Point2d) -> f64 {
        ops::distance_to_point_edge(point.x, point.y, self)
    }

    /// Distance to query polygon center
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance to query polygon center
    pub fn distance_to_polygon_center(&self, polygon: &Polygon) -> f64 {
        ops::distance_to_polygon_center(self, polygon)
    }

    /// Distance to query polygon centroid
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance to query polygon centroid
    pub fn distance_to_polygon_centroid(&self, polygon: &Polygon) -> f64 {
        ops::distance_to_polygon_centroid(self, polygon)
    }

    /// Distance to closest vertex in query polygon
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance to closest vertex in nearest polygon
    pub fn distance_to_polygon_vertex(&self, polygon: &Polygon) -> f64 {
        ops::distance_to_polygon_vertex(self, polygon)
    }

    /// Calculates the minimum distance between the edges of two polygons
    ///
    /// Parameters
    /// ----------
    /// polygon : Polygon
    ///     A Polygon object
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance between the edges of two polygons
    pub fn distance_to_polygon_edge(&self, polygon: &Polygon) -> f64 {
        ops::distance_to_polygon_edge(self, polygon)
    }

    /// Get the convex hull of the polygon
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     Convex hull of original polygon
    pub fn convex_hull(&self) -> Polygon {
        ops::convex_hull(self)
    }

    /// Align the current polygon to a reference polygon
    ///
    /// Parameters
    /// ----------
    /// reference : Polygon
    ///     A reference polygon to align to
    /// scale : bool
    ///     Scale polygon and reference polygon by unit norm
    ///
    /// Returns
    /// -------
    /// Polygon
    ///     Convex hull of original polygon
    #[pyo3(signature = (reference, scale=false))]
    pub fn align_to(&self, reference: &Polygon, scale: bool) -> PyResult<Polygon> {
        // Alignment requires two polygons of equal length. For now,
        // we will throw an error if the two aren't equivalent. We
        // can possibly add an optional resampling step if needed.
        if self.xy.len() != reference.xy.len() {
            return Err(PyValueError::new_err(
                "Polygon and reference must have same number of points.",
            ));
        }
        Ok(ops::align_to(self, reference, scale))
    }

    /// Compute the area of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Area
    pub fn area(&self) -> f64 {
        ops::area(self)
    }

    /// Compute the bounding box area of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Bounding box area
    pub fn area_bbox(&self) -> f64 {
        ops::area_bbox(self)
    }

    /// Compute the convex area of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Convex area
    pub fn area_convex(&self) -> f64 {
        ops::area_convex(self)
    }

    /// Compute the perimeter of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Perimeter
    pub fn perimeter(&self) -> f64 {
        ops::perimeter(self)
    }

    /// Compute the elongation of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Elongation (ratio of shortest:longest x and y axis)
    pub fn elongation(&self) -> f64 {
        ops::elongation(self)
    }

    /// Compute the thread length of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Thread length ((P + sqrt(max(0, P^2 - 16A)) / 4.)
    pub fn thread_length(&self) -> f64 {
        ops::thread_length(self)
    }

    /// Compute the solidity of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Solidity (area / convex hull area)
    pub fn solidity(&self) -> f64 {
        ops::solidity(self)
    }

    /// Compute the extent of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Extent (area / bounding box area)
    pub fn extent(&self) -> f64 {
        ops::extent(self)
    }

    /// Compute the form factor of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Form factor (4 * pi * area / perimiter^2)
    pub fn form_factor(&self) -> f64 {
        ops::form_factor(self)
    }

    /// Compute the equivalent diamter of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Equivalent diameter (sqrt(4 * area / pi))
    pub fn equivalent_diameter(&self) -> f64 {
        ops::equivalent_diameter(self)
    }

    /// Compute the eccentricity of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Eccentricity
    pub fn eccentricity(&self) -> f64 {
        ops::eccentricity(self)
    }

    /// Compute the major axis length of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Major axis length (of best fitting ellipse)
    pub fn major_axis_length(&self) -> f64 {
        ops::major_axis_length(self)
    }

    /// Compute the minor axis length of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Minor axis length (of best fitting ellipse)
    pub fn minor_axis_length(&self) -> f64 {
        ops::minor_axis_length(self)
    }

    /// Compute the minimum radius of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Minimum radius
    pub fn min_radius(&self) -> f64 {
        ops::min_radius(self)
    }

    /// Compute the maximum radius of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Maximum radius
    pub fn max_radius(&self) -> f64 {
        ops::max_radius(self)
    }

    /// Compute the mean radius of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Mean radius
    pub fn mean_radius(&self) -> f64 {
        ops::mean_radius(self)
    }

    /// Compute the minimum feret diameter of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Mean radius
    pub fn min_feret(&self) -> f64 {
        ops::min_feret(self)
    }

    /// Compute the maximum feret diameter of the polygon
    ///
    /// Returns
    /// -------
    /// float
    ///     Mean radius
    pub fn max_feret(&self) -> f64 {
        ops::max_feret(self)
    }

    /// Compute all available descriptors of the polygon
    ///
    /// Returns
    /// -------
    /// List[float]
    ///     A total of 18 different form descriptors.
    pub fn descriptors(&self) -> [f64; 18] {
        ops::descriptors(self)
    }
}

#[pyclass]
pub struct PolygonIterator {
    points: Vec<[f64; 2]>,
    index: usize,
}

#[pymethods]
impl PolygonIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<[f64; 2]> {
        if self.index < self.points.len() {
            let point = self.points[self.index];
            self.index += 1;
            Some(point)
        } else {
            None
        }
    }
}
