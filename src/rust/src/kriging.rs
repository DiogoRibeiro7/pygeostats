// src/rust/src/kriging.rs
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use nalgebra::{DMatrix, DVector};

/// Perform ordinary kriging prediction
#[pyfunction]
pub fn ordinary_kriging_predict<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    known_values: PyReadonlyArray1<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
) -> PyResult<&'py PyArray1<f64>> {
    let known_coords = known_coords.as_array();
    let known_values = known_values.as_array();
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();
    
    let n_known = known_coords.nrows();
    let n_pred = pred_coords.nrows();
    let mut predictions = Array1::<f64>::zeros(n_pred);
    
    // Build covariance matrix for known points
    let mut cov_matrix = DMatrix::<f64>::zeros(n_known + 1, n_known + 1);
    
    for i in 0..n_known {
        for j in 0..n_known {
            let distance = euclidean_distance_single(known_coords.row(i), known_coords.row(j));
            cov_matrix[(i, j)] = variogram_to_covariance(distance, &params, model_type);
        }
        cov_matrix[(i, n_known)] = 1.0; // Lagrange multiplier constraint
        cov_matrix[(n_known, i)] = 1.0;
    }
    
    // Solve for each prediction point
    for p in 0..n_pred {
        let mut rhs = DVector::<f64>::zeros(n_known + 1);
        
        for i in 0..n_known {
            let distance = euclidean_distance_single(known_coords.row(i), pred_coords.row(p));
            rhs[i] = variogram_to_covariance(distance, &params, model_type);
        }
        rhs[n_known] = 1.0; // Constraint
        
        // Solve linear system
        if let Some(weights) = cov_matrix.lu().solve(&rhs) {
            let mut prediction = 0.0;
            for i in 0..n_known {
                prediction += weights[i] * known_values[i];
            }
            predictions[p] = prediction;
        }
    }
    
    Ok(predictions.into_pyarray(py))
}

/// Calculate kriging variance
#[pyfunction] 
pub fn kriging_variance<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    pred_coords: PyReadonlyArray2<f64>, 
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
) -> PyResult<&'py PyArray1<f64>> {
    let known_coords = known_coords.as_array();
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();
    
    let n_known = known_coords.nrows();
    let n_pred = pred_coords.nrows();
    let mut variances = Array1::<f64>::zeros(n_pred);
    
    // Implementation would go here - placeholder for now
    for i in 0..n_pred {
        variances[i] = params[0]; // Sill as placeholder
    }
    
    Ok(variances.into_pyarray(py))
}

fn variogram_to_covariance(distance: f64, params: &ArrayView1<f64>, model_type: &str) -> f64 {
    let nugget = params[0];
    let sill = params[1]; 
    let range = params[2];
    
    if distance == 0.0 {
        return sill;
    }
    
    let gamma = match model_type {
        "exponential" => nugget + (sill - nugget) * (1.0 - (-distance / range).exp()),
        "spherical" => {
            if distance < range {
                nugget + (sill - nugget) * (1.5 * distance / range - 0.5 * (distance / range).powi(3))
            } else {
                sill
            }
        },
        "gaussian" => nugget + (sill - nugget) * (1.0 - (-(distance / range).powi(2)).exp()),
        _ => sill, // Default
    };
    
    sill - gamma // Convert variogram to covariance
}
