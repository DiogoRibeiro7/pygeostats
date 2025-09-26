// src/rust/src/kriging.rs
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::utils::euclidean_distance_single;

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
    let _ = known_coords;
    let _ = model_type;
    let known_coords = known_coords.as_array();
    let known_values = known_values.as_array();
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();

    let n_known = known_coords.nrows();
    let n_pred = pred_coords.nrows();
    let mut predictions = Array1::<f64>::zeros(n_pred);

    if params.len() != 3 {
        return Err(PyValueError::new_err(
            "variogram_params must contain three values: [nugget, sill, range]",
        ));
    }

    let cov = build_covariance_matrix(&known_coords, &params, model_type);
    let mut system = DMatrix::<f64>::zeros(n_known + 1, n_known + 1);

    system.slice_mut((0, 0), (n_known, n_known)).copy_from(&cov);
    for i in 0..n_known {
        system[(i, n_known)] = 1.0;
        system[(n_known, i)] = 1.0;
    }

    let lu = system.lu();

    for p in 0..n_pred {
        let mut rhs = DVector::<f64>::zeros(n_known + 1);
        for i in 0..n_known {
            let distance = euclidean_distance_single(known_coords.row(i), pred_coords.row(p));
            rhs[i] = variogram_to_covariance(distance, &params, model_type);
        }
        rhs[n_known] = 1.0;

        let weights = lu
            .solve(&rhs)
            .ok_or_else(|| PyValueError::new_err("Failed to solve kriging system"))?;

        let mut prediction = 0.0;
        for i in 0..n_known {
            prediction += weights[i] * known_values[i];
        }
        predictions[p] = prediction;
    }

    Ok(predictions.into_pyarray(py))
}

/// Perform simple kriging prediction (known mean, no unbiasedness constraint)
#[pyfunction]
pub fn simple_kriging_predict<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    known_values: PyReadonlyArray1<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
    known_mean: f64,
) -> PyResult<&'py PyArray1<f64>> {
    let known_coords = known_coords.as_array();
    let known_values = known_values.as_array();
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();

    let n_known = known_coords.nrows();
    let n_pred = pred_coords.nrows();

    if params.len() != 3 {
        return Err(PyValueError::new_err(
            "variogram_params must contain three values: [nugget, sill, range]",
        ));
    }

    let mut predictions = Array1::<f64>::zeros(n_pred);

    let cov = build_covariance_matrix(&known_coords, &params, model_type);
    let lu = cov.lu();

    for p in 0..n_pred {
        let mut rhs = DVector::<f64>::zeros(n_known);
        for i in 0..n_known {
            let distance = euclidean_distance_single(known_coords.row(i), pred_coords.row(p));
            rhs[i] = variogram_to_covariance(distance, &params, model_type);
        }

        let weights = lu
            .solve(&rhs)
            .ok_or_else(|| PyValueError::new_err("Failed to solve simple kriging system"))?;

        let mut deviation = 0.0;
        for i in 0..n_known {
            deviation += weights[i] * (known_values[i] - known_mean);
        }
        predictions[p] = known_mean + deviation;
    }

    Ok(predictions.into_pyarray(py))
}

/// Perform universal kriging prediction with polynomial trends
#[pyfunction]
pub fn universal_kriging_predict<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    known_values: PyReadonlyArray1<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
    trend: &str,
) -> PyResult<&'py PyArray1<f64>> {
    let known_coords = known_coords.as_array();
    let known_values = known_values.as_array();
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();

    let trend_type = TrendType::from_str(trend)?;

    let n_known = known_coords.nrows();
    let n_pred = pred_coords.nrows();
    let dim = known_coords.ncols();

    if pred_coords.ncols() != dim {
        return Err(PyValueError::new_err(
            "Prediction coordinates must have the same dimensionality as known coordinates",
        ));
    }

    if params.len() != 3 {
        return Err(PyValueError::new_err(
            "variogram_params must contain three values: [nugget, sill, range]",
        ));
    }

    if known_values.len() != n_known {
        return Err(PyValueError::new_err(
            "Number of values must match number of known coordinates",
        ));
    }

    let basis_size = trend_type.feature_count(dim);
    let cov = build_covariance_matrix(&known_coords, &params, model_type);
    let design = build_design_matrix(trend_type, &known_coords);

    let mut system = DMatrix::<f64>::zeros(n_known + basis_size, n_known + basis_size);
    system.slice_mut((0, 0), (n_known, n_known)).copy_from(&cov);
    system
        .slice_mut((0, n_known), (n_known, basis_size))
        .copy_from(&design);
    system
        .slice_mut((n_known, 0), (basis_size, n_known))
        .copy_from(&design.transpose());

    let lu = system.lu();

    let mut predictions = Array1::<f64>::zeros(n_pred);

    for p in 0..n_pred {
        let mut rhs = DVector::<f64>::zeros(n_known + basis_size);
        for i in 0..n_known {
            let distance = euclidean_distance_single(known_coords.row(i), pred_coords.row(p));
            rhs[i] = variogram_to_covariance(distance, &params, model_type);
        }

        let basis = trend_type.evaluate(pred_coords.row(p));
        for (idx, value) in basis.iter().enumerate() {
            rhs[n_known + idx] = *value;
        }

        let solution = lu
            .solve(&rhs)
            .ok_or_else(|| PyValueError::new_err("Failed to solve universal kriging system"))?;

        let mut prediction = 0.0;
        for i in 0..n_known {
            prediction += solution[i] * known_values[i];
        }
        predictions[p] = prediction;
    }

    Ok(predictions.into_pyarray(py))
}

/// Calculate kriging variance (placeholder)
#[pyfunction]
pub fn kriging_variance<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
) -> PyResult<&'py PyArray1<f64>> {
    let _ = known_coords;
    let _ = model_type;
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();

    let n_pred = pred_coords.nrows();
    let mut variances = Array1::<f64>::zeros(n_pred);

    for i in 0..n_pred {
        variances[i] = params[1];
    }

    Ok(variances.into_pyarray(py))
}

fn build_covariance_matrix(
    coords: &ArrayView2<f64>,
    params: &ArrayView1<f64>,
    model_type: &str,
) -> DMatrix<f64> {
    let n = coords.nrows();
    let mut matrix = DMatrix::<f64>::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let distance = euclidean_distance_single(coords.row(i), coords.row(j));
            matrix[(i, j)] = variogram_to_covariance(distance, params, model_type);
        }
    }

    matrix
}

#[derive(Clone, Copy)]
enum TrendType {
    Linear,
    Quadratic,
}

impl TrendType {
    fn from_str(value: &str) -> PyResult<Self> {
        match value.to_lowercase().as_str() {
            "linear" => Ok(Self::Linear),
            "quadratic" => Ok(Self::Quadratic),
            other => Err(PyValueError::new_err(format!(
                "Unsupported trend type '{}'. Expected 'linear' or 'quadratic'",
                other
            ))),
        }
    }

    fn feature_count(&self, dims: usize) -> usize {
        match self {
            Self::Linear => 1 + dims,
            Self::Quadratic => 1 + dims + dims + (dims * (dims - 1)) / 2,
        }
    }

    fn evaluate(&self, coord: ArrayView1<f64>) -> Vec<f64> {
        let dims = coord.len();
        let mut features = Vec::with_capacity(self.feature_count(dims));
        features.push(1.0);

        match self {
            Self::Linear => {
                for value in coord.iter() {
                    features.push(*value);
                }
            }
            Self::Quadratic => {
                for value in coord.iter() {
                    features.push(*value);
                }
                for value in coord.iter() {
                    features.push(value * value);
                }
                for i in 0..dims {
                    for j in (i + 1)..dims {
                        features.push(coord[i] * coord[j]);
                    }
                }
            }
        }

        features
    }
}

fn build_design_matrix(trend: TrendType, coords: &ArrayView2<f64>) -> DMatrix<f64> {
    let n = coords.nrows();
    let dims = coords.ncols();
    let feature_count = trend.feature_count(dims);

    let mut matrix = DMatrix::<f64>::zeros(n, feature_count);

    for i in 0..n {
        let features = trend.evaluate(coords.row(i));
        for (j, value) in features.into_iter().enumerate() {
            matrix[(i, j)] = value;
        }
    }

    matrix
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
                let ratio = distance / range;
                nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio.powi(3))
            } else {
                sill
            }
        }
        "gaussian" => {
            let ratio = distance / range;
            nugget + (sill - nugget) * (1.0 - (-(ratio * ratio)).exp())
        }
        _ => sill,
    };

    sill - gamma
}
