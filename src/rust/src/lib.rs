use pyo3::prelude::*;
use pyo3::types::PyModule;

mod distances;
mod kriging;
mod utils;
mod variogram;

use distances::{euclidean_distances, haversine_distances};
use kriging::{
    kriging_variance, ordinary_kriging_predict, simple_kriging_predict, universal_kriging_predict,
};
use variogram::{empirical_variogram, fit_variogram_model, FittingResult};

/// High-performance spatial statistics library
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Distance functions
    m.add_function(wrap_pyfunction!(euclidean_distances, m)?)?;
    m.add_function(wrap_pyfunction!(haversine_distances, m)?)?;

    // Variogram functions
    m.add_class::<FittingResult>()?;
    m.add_function(wrap_pyfunction!(empirical_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(fit_variogram_model, m)?)?;

    // Kriging functions
    m.add_function(wrap_pyfunction!(ordinary_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(simple_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(universal_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(kriging_variance, m)?)?;

    Ok(())
}
