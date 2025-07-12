# API

## Point2d

A two-dimensional point defined by x and y coordinates

This struct defines a simply two-dimensional point with various methods for computing distances and performing translations/transformations

### Attributes

- **x** (`float`): X-coordinate
- **y** (`float`): Y-coordinate

### `__init__(x: float, y: float) -> Self`

Initialize a new two-dimensional point

### `__eq__(self, point: Point2d) -> bool`

Test equivalency between points in python

### `__repr__(self) -> str`

Class display formatting

### `to_list(self) -> List[float]`

Convert point to a list

### `add(self, point: Point2d) -> Point2d`

Convert point to a numpy array

### `sub(self, point: Point2d) -> Point2d`

Perform element-wise subtraction between points

### `mul(self, point: Point2d) -> Point2d`

Perform element-wise multiplication between points

### `div(self, point: Point2d) -> Point2d`

Perform element-wise division between points

### `d_l1(self, point: Point2d) -> float`

Compute the l1 (manhattan) distance between two points

### `d_l2(self, point: Point2d) -> float`

Compute the l2 (euclidean) distance between two points

### `d_chebyshev(self, point: Point2d) -> float`

Compute the Chebyshev distance between two points

### `d_cosine(self, point: Point2d) -> float`

Compute the cosine distance between two points

### `interp(self, point: Point2d, t: float) -> Point2d`

Linear interpolate point towards a new point

## Polygon

A polygon represented by a sequence of 2D points.

This struct represents a polygon as a vector of (float, float) vertices. The polygon maintains internal state about whether its points have been deduplicated and ordered. All operations on polygons or between polygons and points are implemented int Rust.

### Attributes

- **xy** (`list`): A public vector of representing the polygon vertices
- **_deduped** (`bool`): Private flag indicating whether duplicate points have been removed
- **_ordered** (`bool`): Private flag indicating whether the points are order

### `__init__(xy: List[List[float]], deduped: bool, ordered: bool) -> Self`

Initialize a new polygon

### `__eq__(self, polygon: Polygon) -> bool`

Test equivalency between polygons

### `eq(self, polygon: Polygon, eps: float) -> bool`

Test approximate equivalency between polygons

### `__repr__(self) -> str`

Class display formatting

### `__len__(self) -> int`

Number of points in the polygon

### `__iter__(self) -> PolygonIterator`

Iterate through polygon points

### `to_list(self) -> List[List[float]]`

Convert point to a list

### `push(self, point: List[float])`

Convert point to a numpy array

### `push_point2d(self, point: Point2d)`

Push a Point2d onto the polygon

### `add_scalar_inplace(self, scalar: float)`

Adds a scalar to each point in the polygon in-place

### `sub_scalar_inplace(self, scalar: float)`

Subtract a scalar from each point in the polygon in-place

### `mul_scalar_inplace(self, scalar: float)`

Multiply each point in the polygon by a scalar in-place

### `div_scalar_inplace(self, scalar: float)`

Divide each point in the polygon by a scalar in-place

### `add_scalar(self, scalar: float) -> Polygon`

Adds a scalar to each point in the polygon

### `sub_scalar(self, scalar: float) -> Polygon`

Subtract a scalar from each point in the polygon

### `mul_scalar(self, scalar: float) -> Polygon`

Multiply each point by a scalar in the polygon

### `div_scalar(self, scalar: float) -> Polygon`

Divide each point in the polygon by a scalar

### `add_point_inplace(self, point: List[float])`

Add an [x, y] point to a polygon in-place

### `add_point2d_inplace(self, point: Point2d)`

Add a Point2d object to a polygon in-place

### `sub_point_inplace(self, point: List[float])`

Subtract an [x, y] point from a polygon in-place

### `sub_point2d_inplace(self, point: Point2d)`

Subtract a Point2d from a polygon in-place

### `mul_point_inplace(self, point: List[float])`

Multiply each point in a polygon by an [x, y] point in-place

### `mul_point2d_inplace(self, point: Point2d)`

Multiply each point in a polygon by a Point2d object in-place

### `div_point_inplace(self, point: List[float])`

Divide each point in a polygon by an [x, y] point in-place

### `div_point2d_inplace(self, point: Point2d)`

Divide each point in a polygon by a Point2d object in-place

### `add_point(self, point: List[float]) -> Polygon`

Add an [x, y] point to a polygon

### `add_point2d(self, point: Point2d) -> Polygon`

Add a Point2d object to a polygon in-place

### `sub_point(self, point: List[float]) -> Polygon`

Subtract an [x, y] point from a polygon

### `sub_point2d(self, point: Point2d) -> Polygon`

Subtract a Point2d from a polygon

### `mul_point(self, point: List[float]) -> Polygon`

Multiply each point in a polygon by an [x, y] point

### `mul_point2d(self, point: Point2d) -> Polygon`

Multiply each point in a polygon by a Point2d object

### `div_point(self, point: List[float]) -> Polygon`

Divide each point in a polygon by an [x, y] point

### `div_point2d(self, point: Point2d) -> Polygon`

Divide each point in a polygon by a Point2d object

### `add_inplace(self, polygon: Polygon)`

Perform element-wise addition between polygons in-place

### `sub_inplace(self, polygon: Polygon)`

Perform element-wise subtraction between polygons in-place

### `mul_inplace(self, polygon: Polygon)`

Perform element-wise multiplication between polygons in-place

### `div_inplace(self, polygon: Polygon)`

Perform element-wise division between polygons in-place

### `add(self, polygon: Polygon) -> Polygon`

Perform element-wise addition between polygons

### `sub(self, polygon: Polygon) -> Polygon`

Perform element-wise subtraction between polygons

### `mul(self, polygon: Polygon) -> Polygon`

Perform element-wise multiplication between polygons

### `div(self, polygon: Polygon) -> Polygon`

Perform element-wise division between polygons

### `center(self) -> Point2d`

Get the center coordinates of the polygon

### `centroid(self) -> Point2d`

Get the centroid coordinates of the polygon

### `d_l1(self, polygon: Polygon) -> float`

Compute the l1 (manhattan) distance between polygons

### `d_l2(self, polygon: Polygon) -> float`

Compute the l2 (euclidean) distance between polygons

### `d_chebyshev(self, polygon: Polygon) -> float`

Compute the chebyshev distance between polygons

### `d_cosine(self, polygon: Polygon) -> float`

Compute the cosine distance between polygons

### `d_hausdorff(self, polygon: Polygon, n: int) -> float`

Compute the Hausdorff distance between polygons

### `dedup_inplace(self)`

De-duplicate points in-place

### `dedup(self) -> Polygon`

Generate a new polygon with de-duplicated points

### `dedup_unstable_inplace(self)`

De-duplicate points in-place (faster but no guarantee on original ordering)

### `dedup_unstable(self) -> Polygon`

Generate a new polygon with de-duplicated points (faster but no guarantee on original ordering)

### `order_inplace(self)`

Order polygon vertices in-place

### `order(self) -> Polygon`

Generate a new polygon with ordered polygon vertices

### `resample_inplace(self, n_points: int)`

Resample polygon outline in-place

### `resample(self, n_points: int) -> Polygon`

Generate a new polygon with resampled outline

### `encloses_point(self, point: List[float]) -> bool`

Check if the polygon contains an [x, y] point

### `encloses_point2d(self, point: Point2d) -> bool`

Check if the polygon contains a Point2d object

### `distance_to_point_center(self, point: List[float]) -> float`

Distance from [x, y] point to polygon center

### `distance_to_point2d_center(self, point: Point2d) -> float`

Distance from Point2d object to polygon center

### `distance_to_point_centroid(self, point: List[float]) -> float`

Distance from [x, y] point to polygon centroid

### `distance_to_point2d_centroid(self, point: Point2d) -> float`

Distance from Point2d object to polygon centroid

### `distance_to_point_vertex(self, point: List[float]) -> float`

Distance from [x, y] point to nearest polygon vertex

### `distance_to_point2d_vertex(self, point: Point2d) -> float`

Distance from Point2d object to nearest polygon vertex

### `distance_to_point_edge(self, point: List[float]) -> float`

Distance from [x, y] point to nearest polygon edge

### `distance_to_point2d_edge(self, point: Point2d) -> float`

Distance from Point2d object to nearest polygon edge

### `distance_to_polygon_center(self, polygon: Polygon) -> float`

Distance to query polygon center

### `distance_to_polygon_centroid(self, polygon: Polygon) -> float`

Distance to query polygon centroid

### `distance_to_polygon_vertex(self, polygon: Polygon) -> float`

Distance to closest vertex in query polygon

### `distance_to_polygon_edge(self, polygon: Polygon) -> float`

Calculates the minimum distance between the edges of two polygons

### `convex_hull(self) -> Polygon`

Get the convex hull of the polygon

### `align_to(self, reference: Polygon, scale: bool) -> Polygon`

Align the current polygon to a reference polygon

### `area(self) -> float`

Compute the area of the polygon

### `area_bbox(self) -> float`

Compute the bounding box area of the polygon

### `area_convex(self) -> float`

Compute the convex area of the polygon

### `perimeter(self) -> float`

Compute the perimeter of the polygon

### `elongation(self) -> float`

Compute the elongation of the polygon

### `thread_length(self) -> float`

Compute the thread length of the polygon

### `solidity(self) -> float`

Compute the solidity of the polygon

### `extent(self) -> float`

Compute the extent of the polygon

### `form_factor(self) -> float`

Compute the form factor of the polygon

### `equivalent_diameter(self) -> float`

Compute the equivalent diamter of the polygon

### `eccentricity(self) -> float`

Compute the eccentricity of the polygon

### `major_axis_length(self) -> float`

Compute the major axis length of the polygon

### `minor_axis_length(self) -> float`

Compute the minor axis length of the polygon

### `min_radius(self) -> float`

Compute the minimum radius of the polygon

### `max_radius(self) -> float`

Compute the maximum radius of the polygon

### `mean_radius(self) -> float`

Compute the mean radius of the polygon

### `min_feret(self) -> float`

Compute the minimum feret diameter of the polygon

### `max_feret(self) -> float`

Compute the maximum feret diameter of the polygon

### `descriptors(self) -> [f64; 18]`

Compute all available descriptors of the polygon
