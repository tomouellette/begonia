// Copyright (c) 2025, Tom Ouellette
// Licensed under the BSD 3-Clause License

use pyo3::prelude::*;

mod point;
mod polygon;

pub use point::Point2d;
pub use polygon::Polygon;

#[pymodule]
fn begonia(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<Point2d>()?;
    m.add_class::<Polygon>()?;

    // Functions
    Ok(())
}
