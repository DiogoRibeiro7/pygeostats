use pyo3::prelude::*;
use pyo3::types::PyModule;

mod anisotropic_kriging;
mod distances;
mod kriging;
mod utils;
mod variogram;

use anisotropic_kriging::{
    anisotropic_distance_matrix, anisotropic_kriging_predict, anisotropic_kriging_variance,
    fit_anisotropic_variogram,
};
use distances::{euclidean_distances, haversine_distances};
use kriging::{
    kriging_variance, ordinary_kriging_predict, ordinary_kriging_predict_neighbors,
    simple_kriging_predict, universal_kriging_predict,
};
use variogram::{
    empirical_variogram, fit_variogram_model, streaming_variogram, FittingResult,
    StreamingVariogramAccumulator,
};

/// High-performance spatial statistics library
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Distance functions
    m.add_function(wrap_pyfunction!(euclidean_distances, m)?)?;
    m.add_function(wrap_pyfunction!(haversine_distances, m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_distance_matrix, m)?)?;

    // Variogram functions
    m.add_class::<FittingResult>()?;
    m.add_class::<StreamingVariogramAccumulator>()?;
    m.add_function(wrap_pyfunction!(empirical_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(streaming_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(fit_variogram_model, m)?)?;
    m.add_function(wrap_pyfunction!(fit_anisotropic_variogram, m)?)?;

    // Kriging functions
    m.add_function(wrap_pyfunction!(ordinary_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(ordinary_kriging_predict_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(simple_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(universal_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(kriging_variance, m)?)?;

    // Anisotropic kriging functions
    m.add_function(wrap_pyfunction!(anisotropic_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(anisotropic_kriging_variance, m)?)?;

    Ok(())
}
