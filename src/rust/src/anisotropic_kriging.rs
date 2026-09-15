// src/rust/src/anisotropic_kriging.rs
//! Anisotropic kriging implementation with elliptical distance calculations.

use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[pyfunction]
pub fn anisotropic_kriging_predict<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    known_values: PyReadonlyArray1<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let known_coords = known_coords.as_array();
    let known_values = known_values.as_array();
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();

    // Validate inputs
    if known_coords.ncols() != 2 || pred_coords.ncols() != 2 {
        return Err(PyValueError::new_err(
            "Anisotropic kriging requires 2D coordinates",
        ));
    }

    if params.len() != 5 {
        return Err(PyValueError::new_err(
            "Anisotropic variogram requires 5 parameters: [nugget, sill, range_major, range_minor, rotation_angle]"
        ));
    }

    let n_known = known_coords.nrows();
    let n_pred = pred_coords.nrows();
    let mut predictions = Array1::<f64>::zeros(n_pred);

    // Extract anisotropic parameters
    let nugget = params[0];
    let sill = params[1];
    let range_major = params[2];
    let range_minor = params[3];
    let rotation_angle = params[4];

    // Validate parameters
    if range_major <= 0.0 || range_minor <= 0.0 {
        return Err(PyValueError::new_err("Ranges must be positive"));
    }
    if sill <= nugget {
        return Err(PyValueError::new_err("Sill must be greater than nugget"));
    }

    // Precompute rotation matrix components
    let cos_theta = rotation_angle.cos();
    let sin_theta = rotation_angle.sin();

    // Build anisotropic covariance matrix for known points
    let mut cov_matrix = DMatrix::<f64>::zeros(n_known, n_known);
    for i in 0..n_known {
        for j in 0..n_known {
            let aniso_distance = compute_anisotropic_distance(
                known_coords.row(i),
                known_coords.row(j),
                range_major,
                range_minor,
                cos_theta,
                sin_theta,
            );
            cov_matrix[(i, j)] = anisotropic_covariance(aniso_distance, nugget, sill, model_type)?;
        }
    }

    // Set up kriging system with unbiasedness constraint
    let mut system_matrix = DMatrix::<f64>::zeros(n_known + 1, n_known + 1);
    system_matrix
        .view_mut((0, 0), (n_known, n_known))
        .copy_from(&cov_matrix);

    // Add unbiasedness constraint
    for i in 0..n_known {
        system_matrix[(i, n_known)] = 1.0;
        system_matrix[(n_known, i)] = 1.0;
    }

    // Factorize system matrix once
    let lu = system_matrix.lu();

    // Solve for each prediction point
    for p in 0..n_pred {
        let mut rhs = DVector::<f64>::zeros(n_known + 1);

        // Compute covariances between prediction point and known points
        for i in 0..n_known {
            let aniso_distance = compute_anisotropic_distance(
                known_coords.row(i),
                pred_coords.row(p),
                range_major,
                range_minor,
                cos_theta,
                sin_theta,
            );
            rhs[i] = anisotropic_covariance(aniso_distance, nugget, sill, model_type)?;
        }
        rhs[n_known] = 1.0; // unbiasedness constraint

        // Solve for weights
        let weights = lu
            .solve(&rhs)
            .ok_or_else(|| PyValueError::new_err("Failed to solve anisotropic kriging system"))?;

        // Compute prediction
        let mut prediction = 0.0;
        for i in 0..n_known {
            prediction += weights[i] * known_values[i];
        }
        predictions[p] = prediction;
    }

    Ok(predictions.into_pyarray(py))
}

#[pyfunction]
pub fn anisotropic_kriging_variance<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let known_coords = known_coords.as_array();
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();

    if known_coords.ncols() != 2 || pred_coords.ncols() != 2 {
        return Err(PyValueError::new_err(
            "Anisotropic kriging requires 2D coordinates",
        ));
    }

    if params.len() != 5 {
        return Err(PyValueError::new_err(
            "Anisotropic variogram requires 5 parameters",
        ));
    }

    let n_known = known_coords.nrows();
    let n_pred = pred_coords.nrows();
    let mut variances = Array1::<f64>::zeros(n_pred);

    // Extract anisotropic parameters
    let nugget = params[0];
    let sill = params[1];
    let range_major = params[2];
    let range_minor = params[3];
    let rotation_angle = params[4];

    // Validate parameters
    if range_major <= 0.0 || range_minor <= 0.0 {
        return Err(PyValueError::new_err("Ranges must be positive"));
    }
    if sill <= nugget {
        return Err(PyValueError::new_err("Sill must be greater than nugget"));
    }

    let cos_theta = rotation_angle.cos();
    let sin_theta = rotation_angle.sin();

    // Build anisotropic covariance matrix
    let mut cov_matrix = DMatrix::<f64>::zeros(n_known, n_known);
    for i in 0..n_known {
        for j in 0..n_known {
            let aniso_distance = compute_anisotropic_distance(
                known_coords.row(i),
                known_coords.row(j),
                range_major,
                range_minor,
                cos_theta,
                sin_theta,
            );
            cov_matrix[(i, j)] = anisotropic_covariance(aniso_distance, nugget, sill, model_type)?;
        }
    }

    // Set up kriging system with unbiasedness constraint
    let mut system_matrix = DMatrix::<f64>::zeros(n_known + 1, n_known + 1);
    system_matrix
        .view_mut((0, 0), (n_known, n_known))
        .copy_from(&cov_matrix);

    for i in 0..n_known {
        system_matrix[(i, n_known)] = 1.0;
        system_matrix[(n_known, i)] = 1.0;
    }

    let lu = system_matrix.lu();

    // Compute variance for each prediction point
    for p in 0..n_pred {
        let mut rhs = DVector::<f64>::zeros(n_known + 1);

        // Covariances between prediction point and known points
        for i in 0..n_known {
            let aniso_distance = compute_anisotropic_distance(
                known_coords.row(i),
                pred_coords.row(p),
                range_major,
                range_minor,
                cos_theta,
                sin_theta,
            );
            rhs[i] = anisotropic_covariance(aniso_distance, nugget, sill, model_type)?;
        }
        rhs[n_known] = 1.0;

        // Solve for weights
        let weights = lu.solve(&rhs).ok_or_else(|| {
            PyValueError::new_err("Failed to solve anisotropic kriging variance system")
        })?;

        // Compute kriging variance: C(0,0) - w^T * c0 - lambda
        let c00 = sill; // variance at prediction point itself
        let mut w_dot_c0 = 0.0;
        for i in 0..n_known {
            w_dot_c0 += weights[i] * rhs[i];
        }
        let lambda = weights[n_known]; // Lagrange multiplier

        variances[p] = (c00 - w_dot_c0 - lambda).max(0.0); // ensure non-negative
    }

    Ok(variances.into_pyarray(py))
}

// Not registered in lib.rs, so it is unreachable from Python -- unlike
// ordinary_kriging_predict_neighbors, which is. Exporting it would add public
// numerical surface that no test covers, so that is a deliberate decision to
// make separately rather than a side effect of silencing a lint.
#[allow(dead_code)]
#[pyfunction]
pub fn anisotropic_kriging_predict_neighbors<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    known_values: PyReadonlyArray1<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    neighbors: PyReadonlyArray2<i64>,
    model_type: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let known_coords = known_coords.as_array();
    let known_values = known_values.as_array();
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();
    let neighbors = neighbors.as_array();

    if known_coords.ncols() != 2 || pred_coords.ncols() != 2 {
        return Err(PyValueError::new_err(
            "Anisotropic kriging requires 2D coordinates",
        ));
    }

    if params.len() != 5 {
        return Err(PyValueError::new_err(
            "Anisotropic variogram requires 5 parameters",
        ));
    }

    if neighbors.nrows() != pred_coords.nrows() {
        return Err(PyValueError::new_err(
            "neighbors must have shape (n_predictions, k_neighbors)",
        ));
    }

    let n_pred = pred_coords.nrows();
    let mut predictions = vec![f64::NAN; n_pred];

    // Extract anisotropic parameters
    let nugget = params[0];
    let sill = params[1];
    let range_major = params[2];
    let range_minor = params[3];
    let rotation_angle = params[4];

    let cos_theta = rotation_angle.cos();
    let sin_theta = rotation_angle.sin();

    #[cfg(feature = "parallel")]
    {
        predictions
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, prediction)| {
                *prediction = predict_single_point_with_neighbors(
                    &known_coords,
                    &known_values,
                    &pred_coords,
                    &neighbors,
                    idx,
                    nugget,
                    sill,
                    range_major,
                    range_minor,
                    cos_theta,
                    sin_theta,
                    model_type,
                );
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (idx, prediction) in predictions.iter_mut().enumerate() {
            *prediction = predict_single_point_with_neighbors(
                &known_coords,
                &known_values,
                &pred_coords,
                &neighbors,
                idx,
                nugget,
                sill,
                range_major,
                range_minor,
                cos_theta,
                sin_theta,
                model_type,
            );
        }
    }

    Ok(Array1::from_vec(predictions).into_pyarray(py))
}

// Only called by anisotropic_kriging_predict_neighbors, which is itself
// unregistered; see the note above it.
#[allow(dead_code)]
fn predict_single_point_with_neighbors(
    known_coords: &ArrayView2<f64>,
    known_values: &ArrayView1<f64>,
    pred_coords: &ArrayView2<f64>,
    neighbors: &ArrayView2<i64>,
    pred_idx: usize,
    nugget: f64,
    sill: f64,
    range_major: f64,
    range_minor: f64,
    cos_theta: f64,
    sin_theta: f64,
    model_type: &str,
) -> f64 {
    // Get neighbor indices for this prediction point
    let neighbor_ids: Vec<usize> = neighbors
        .row(pred_idx)
        .iter()
        .filter_map(|&value| {
            if value < 0 {
                return None;
            }
            let candidate = value as usize;
            if candidate < known_coords.nrows() {
                Some(candidate)
            } else {
                None
            }
        })
        .collect();

    if neighbor_ids.is_empty() {
        return f64::NAN;
    }

    let k = neighbor_ids.len();
    let mut system_matrix = DMatrix::<f64>::zeros(k + 1, k + 1);

    // Build covariance matrix for neighbors
    for (row_pos, &i_idx) in neighbor_ids.iter().enumerate() {
        let coord_i = known_coords.row(i_idx);
        for (col_pos, &j_idx) in neighbor_ids.iter().enumerate() {
            let coord_j = known_coords.row(j_idx);
            let aniso_distance = compute_anisotropic_distance(
                coord_i,
                coord_j,
                range_major,
                range_minor,
                cos_theta,
                sin_theta,
            );

            match anisotropic_covariance(aniso_distance, nugget, sill, model_type) {
                Ok(cov) => system_matrix[(row_pos, col_pos)] = cov,
                Err(_) => return f64::NAN,
            }
        }
        system_matrix[(row_pos, k)] = 1.0;
        system_matrix[(k, row_pos)] = 1.0;
    }

    let lu = system_matrix.lu();
    let mut rhs = DVector::<f64>::zeros(k + 1);
    let pred_coord = pred_coords.row(pred_idx);

    // Compute covariances between prediction point and neighbors
    for (row_pos, &i_idx) in neighbor_ids.iter().enumerate() {
        let aniso_distance = compute_anisotropic_distance(
            known_coords.row(i_idx),
            pred_coord,
            range_major,
            range_minor,
            cos_theta,
            sin_theta,
        );

        match anisotropic_covariance(aniso_distance, nugget, sill, model_type) {
            Ok(cov) => rhs[row_pos] = cov,
            Err(_) => return f64::NAN,
        }
    }
    rhs[k] = 1.0;

    match lu.solve(&rhs) {
        Some(weights) => {
            let mut value = 0.0;
            for (weight_pos, &i_idx) in neighbor_ids.iter().enumerate() {
                value += weights[weight_pos] * known_values[i_idx];
            }
            value
        }
        None => f64::NAN,
    }
}

/// Compute anisotropic distance between two points
fn compute_anisotropic_distance(
    point1: ArrayView1<f64>,
    point2: ArrayView1<f64>,
    range_major: f64,
    range_minor: f64,
    cos_theta: f64,
    sin_theta: f64,
) -> f64 {
    // Vector difference
    let dx = point2[0] - point1[0];
    let dy = point2[1] - point1[1];

    // Rotate to align with anisotropy axes
    let dx_rot = dx * cos_theta + dy * sin_theta;
    let dy_rot = -dx * sin_theta + dy * cos_theta;

    // Scale by anisotropic ranges and compute distance
    let scaled_dx = dx_rot / range_major;
    let scaled_dy = dy_rot / range_minor;

    (scaled_dx * scaled_dx + scaled_dy * scaled_dy).sqrt()
}

/// Compute covariance from anisotropic distance
fn anisotropic_covariance(
    aniso_distance: f64,
    nugget: f64,
    sill: f64,
    model_type: &str,
) -> PyResult<f64> {
    if aniso_distance == 0.0 {
        return Ok(sill);
    }

    // Compute semivariance using variogram model
    let gamma = match model_type {
        "exponential" => nugget + (sill - nugget) * (1.0 - (-aniso_distance).exp()),
        "spherical" => {
            if aniso_distance >= 1.0 {
                sill
            } else {
                nugget + (sill - nugget) * (1.5 * aniso_distance - 0.5 * aniso_distance.powi(3))
            }
        }
        "gaussian" => nugget + (sill - nugget) * (1.0 - (-(aniso_distance * aniso_distance)).exp()),
        "matern" => {
            // Simplified Matern with nu=0.5 (equivalent to exponential)
            nugget + (sill - nugget) * (1.0 - (-aniso_distance).exp())
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unsupported variogram model: {model_type}"
            )));
        }
    };

    // Convert semivariance to covariance
    Ok(sill - gamma)
}

#[pyfunction]
pub fn anisotropic_distance_matrix<'py>(
    py: Python<'py>,
    coords1: PyReadonlyArray2<f64>,
    coords2: PyReadonlyArray2<f64>,
    range_major: f64,
    range_minor: f64,
    rotation_angle: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let coords1 = coords1.as_array();
    let coords2 = coords2.as_array();

    if coords1.ncols() != 2 || coords2.ncols() != 2 {
        return Err(PyValueError::new_err(
            "Anisotropic distance calculation requires 2D coordinates",
        ));
    }

    if range_major <= 0.0 || range_minor <= 0.0 {
        return Err(PyValueError::new_err("Ranges must be positive"));
    }

    let n1 = coords1.nrows();
    let n2 = coords2.nrows();
    let mut distances = Array1::<f64>::zeros(n1 * n2);

    let cos_theta = rotation_angle.cos();
    let sin_theta = rotation_angle.sin();

    for i in 0..n1 {
        for j in 0..n2 {
            let aniso_distance = compute_anisotropic_distance(
                coords1.row(i),
                coords2.row(j),
                range_major,
                range_minor,
                cos_theta,
                sin_theta,
            );
            distances[i * n2 + j] = aniso_distance;
        }
    }

    Ok(distances.into_pyarray(py))
}

// PyO3 treats a trailing Option argument without an explicit default as required,
// so weights could not be left out.
#[pyfunction(signature = (distances, gamma, directions, model_type, initial_params, weights=None))]
pub fn fit_anisotropic_variogram<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray1<f64>,
    gamma: PyReadonlyArray1<f64>,
    directions: PyReadonlyArray1<f64>,
    model_type: &str,
    initial_params: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<crate::variogram::FittingResult>> {
    // This calls the existing anisotropic fitting in variogram.rs
    // with the directions parameter properly set
    crate::variogram::fit_variogram_model(
        py,
        distances,
        gamma,
        model_type,
        initial_params,
        weights,
        None, // fix_mask
        Some(directions),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_anisotropic_distance_isotropic_case() {
        // Test that anisotropic distance reduces to euclidean when ranges are equal
        let point1 = Array1::from_vec(vec![0.0, 0.0]);
        let point2 = Array1::from_vec(vec![3.0, 4.0]);

        let aniso_dist = compute_anisotropic_distance(
            point1.view(),
            point2.view(),
            1.0, // range_major = range_minor = 1.0 (isotropic)
            1.0,
            1.0, // cos(0) = 1
            0.0, // sin(0) = 0
        );

        let euclidean_dist = 5.0; // sqrt(3^2 + 4^2)
        assert_relative_eq!(aniso_dist, euclidean_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_anisotropic_distance_rotation() {
        // Test rotation effects
        let point1 = Array1::from_vec(vec![0.0, 0.0]);
        let point2 = Array1::from_vec(vec![1.0, 0.0]);

        // No rotation - point along x-axis should use major range
        let dist_no_rot = compute_anisotropic_distance(
            point1.view(),
            point2.view(),
            2.0, // range_major
            1.0, // range_minor
            1.0, // cos(0) = 1
            0.0, // sin(0) = 0
        );
        assert_relative_eq!(dist_no_rot, 0.5, epsilon = 1e-10); // 1.0 / 2.0

        // 90-degree rotation - same point should now use minor range
        let dist_90_rot = compute_anisotropic_distance(
            point1.view(),
            point2.view(),
            2.0, // range_major
            1.0, // range_minor
            0.0, // cos(90) = 0
            1.0, // sin(90) = 1
        );
        assert_relative_eq!(dist_90_rot, 1.0, epsilon = 1e-10); // 1.0 / 1.0
    }

    #[test]
    fn test_anisotropic_covariance_models() {
        let nugget = 0.1;
        let sill = 1.0;
        let distance = 0.5;

        // Test exponential model
        let cov_exp = anisotropic_covariance(distance, nugget, sill, "exponential").unwrap();
        let expected_gamma = nugget + (sill - nugget) * (1.0 - (-distance).exp());
        let expected_cov = sill - expected_gamma;
        assert_relative_eq!(cov_exp, expected_cov, epsilon = 1e-10);

        // Test spherical model
        let cov_sph = anisotropic_covariance(distance, nugget, sill, "spherical").unwrap();
        let expected_gamma_sph =
            nugget + (sill - nugget) * (1.5 * distance - 0.5 * distance.powi(3));
        let expected_cov_sph = sill - expected_gamma_sph;
        assert_relative_eq!(cov_sph, expected_cov_sph, epsilon = 1e-10);

        // Test at zero distance (should return sill)
        let cov_zero = anisotropic_covariance(0.0, nugget, sill, "exponential").unwrap();
        assert_relative_eq!(cov_zero, sill, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_model() {
        let result = anisotropic_covariance(1.0, 0.1, 1.0, "invalid_model");
        assert!(result.is_err());
    }

    #[test]
    fn test_parameter_validation() {
        // Test negative ranges
        let point1 = Array1::from_vec(vec![0.0, 0.0]);
        let point2 = Array1::from_vec(vec![1.0, 0.0]);

        // Negative ranges should still work in distance calculation (handled at higher level)
        let dist = compute_anisotropic_distance(
            point1.view(),
            point2.view(),
            -1.0, // negative range
            1.0,
            1.0,
            0.0,
        );
        assert!(dist.is_finite()); // Should handle gracefully
    }

    #[test]
    fn test_extreme_anisotropy() {
        // Test very high anisotropy ratio
        let point1 = Array1::from_vec(vec![0.0, 0.0]);
        let point2 = Array1::from_vec(vec![1.0, 0.0]);

        let dist = compute_anisotropic_distance(
            point1.view(),
            point2.view(),
            100.0, // very large major range
            0.01,  // very small minor range
            1.0,   // no rotation
            0.0,
        );

        assert_relative_eq!(dist, 0.01, epsilon = 1e-10); // 1.0 / 100.0
    }
}
