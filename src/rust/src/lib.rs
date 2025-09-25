// src/rust/src/lib.rs
use pyo3::prelude::*;
use pyo3::types::PyModule;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

mod distances;
mod variogram;
mod kriging;
mod utils;

use distances::*;
use variogram::*;
use kriging::*;

/// High-performance spatial statistics library
#[pymodule]
fn _core(_py: Python, m: &PyModule) -> PyResult<()> {
    // Distance functions
    m.add_function(wrap_pyfunction!(euclidean_distances, m)?)?;
    m.add_function(wrap_pyfunction!(haversine_distances, m)?)?;
    
    // Variogram functions
    m.add_function(wrap_pyfunction!(empirical_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(fit_variogram_model, m)?)?;
    
    // Kriging functions  
    m.add_function(wrap_pyfunction!(ordinary_kriging_predict, m)?)?;
    m.add_function(wrap_pyfunction!(kriging_variance, m)?)?;
    
    Ok(())
}

// src/rust/src/distances.rs
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Calculate Euclidean distances between all pairs of points
#[pyfunction]
pub fn euclidean_distances<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let coords = coords.as_array();
    let n = coords.nrows();
    let mut distances = Array2::<f64>::zeros((n, n));
    
    // Parallel computation of distance matrix
    distances.indexed_iter_mut().par_bridge().for_each(|((i, j), dist)| {
        if i <= j {
            let d = euclidean_distance_single(coords.row(i), coords.row(j));
            *dist = d;
        }
    });
    
    // Mirror the upper triangle to lower triangle
    for i in 0..n {
        for j in 0..i {
            distances[[i, j]] = distances[[j, i]];
        }
    }
    
    Ok(distances.into_pyarray(py))
}

/// Calculate Haversine distances for geographic coordinates (lat, lon in degrees)
#[pyfunction]
pub fn haversine_distances<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    radius: Option<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let coords = coords.as_array();
    let n = coords.nrows();
    let r = radius.unwrap_or(6371.0); // Earth radius in km
    let mut distances = Array2::<f64>::zeros((n, n));
    
    distances.indexed_iter_mut().par_bridge().for_each(|((i, j), dist)| {
        if i <= j {
            let d = haversine_distance_single(coords.row(i), coords.row(j), r);
            *dist = d;
        }
    });
    
    // Mirror the matrix
    for i in 0..n {
        for j in 0..i {
            distances[[i, j]] = distances[[j, i]];
        }
    }
    
    Ok(distances.into_pyarray(py))
}

fn euclidean_distance_single(p1: ArrayView1<f64>, p2: ArrayView1<f64>) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn haversine_distance_single(p1: ArrayView1<f64>, p2: ArrayView1<f64>, radius: f64) -> f64 {
    let lat1 = p1[0].to_radians();
    let lon1 = p1[1].to_radians();
    let lat2 = p2[0].to_radians();
    let lon2 = p2[1].to_radians();
    
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    
    let a = (dlat / 2.0).sin().powi(2) 
        + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    
    radius * c
}
