// src/rust/src/variogram.rs
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::euclidean_distance_single;

const MIN_RANGE: f64 = 1e-6;
const MIN_SILL_GAP: f64 = 1e-9;
const MAX_LAMBDA: f64 = 1e12;
const PARAM_TOL: f64 = 1e-6;
const COST_TOL: f64 = 1e-9;

#[pyclass]
pub struct FittingResult {
    #[pyo3(get)]
    parameters: Vec<f64>,
    #[pyo3(get)]
    r_squared: f64,
    #[pyo3(get)]
    rmse: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl FittingResult {
    fn __repr__(&self) -> String {
        let escaped_message = self.message.replace('\'', "\\'");
        format!(
            "FittingResult(parameters={:?}, r_squared={:.6}, rmse={:.6}, converged={}, iterations={}, message='{}')",
            self.parameters,
            self.r_squared,
            self.rmse,
            self.converged,
            self.iterations,
            escaped_message,
        )
    }
}

/// Calculate empirical variogram from coordinates and values
#[pyfunction]
pub fn empirical_variogram<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    values: PyReadonlyArray1<f64>,
    bins: PyReadonlyArray1<f64>,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>, &'py PyArray1<i32>)> {
    let coords = coords.as_array();
    let values = values.as_array();
    let bins = bins.as_array();
    let n = coords.nrows();
    let n_bins = bins.len() - 1;

    let mut gamma = Array1::<f64>::zeros(n_bins);
    let mut bin_centers = Array1::<f64>::zeros(n_bins);
    let mut counts = Array1::<i32>::zeros(n_bins);

    // Calculate bin centers
    for i in 0..n_bins {
        bin_centers[i] = (bins[i] + bins[i + 1]) / 2.0;
    }

    // Calculate variogram values
    let pairs: Vec<_> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();

    let bin_sums: Vec<(f64, i32)> = (0..n_bins)
        .into_par_iter()
        .map(|bin_idx| {
            let mut sum = 0.0;
            let mut count = 0;

            for &(i, j) in &pairs {
                let distance = euclidean_distance_single(coords.row(i), coords.row(j));

                if distance >= bins[bin_idx] && distance < bins[bin_idx + 1] {
                    let diff = values[i] - values[j];
                    sum += 0.5 * diff * diff;
                    count += 1;
                }
            }

            (sum, count)
        })
        .collect();

    for (i, (sum, count)) in bin_sums.into_iter().enumerate() {
        if count > 0 {
            gamma[i] = sum / count as f64;
            counts[i] = count;
        }
    }

    Ok((
        bin_centers.into_pyarray(py),
        gamma.into_pyarray(py),
        counts.into_pyarray(py),
    ))
}

#[derive(Clone, Copy)]
enum VariogramModel {
    Exponential,
    Spherical,
    Gaussian,
}

impl VariogramModel {
    fn from_str(model_type: &str) -> PyResult<Self> {
        match model_type {
            "exponential" => Ok(Self::Exponential),
            "spherical" => Ok(Self::Spherical),
            "gaussian" => Ok(Self::Gaussian),
            _ => Err(PyValueError::new_err(format!(
                "Unsupported variogram model '{}'. Expected one of: exponential, spherical, gaussian",
                model_type
            ))),
        }
    }
}

#[pyfunction(signature = (distances, gamma, model_type, initial_params, weights=None, fix_mask=None))]
pub fn fit_variogram_model<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray1<f64>,
    gamma: PyReadonlyArray1<f64>,
    model_type: &str,
    initial_params: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    fix_mask: Option<PyReadonlyArray1<bool>>,
) -> PyResult<Py<FittingResult>> {
    let distances_view = distances.as_array();
    let gamma_view = gamma.as_array();

    if distances_view.len() != gamma_view.len() {
        return Err(PyValueError::new_err(
            "distances and gamma must have the same length",
        ));
    }

    if distances_view.is_empty() {
        return Err(PyValueError::new_err("distances and gamma cannot be empty"));
    }

    let mut params = initial_params.as_array().to_vec();
    if params.len() != 3 {
        return Err(PyValueError::new_err(
            "initial_params must contain three values: [nugget, sill, range]",
        ));
    }

    let mut weight_vec = match weights {
        Some(w) => {
            let arr = w.as_array();
            if arr.len() != distances_view.len() {
                return Err(PyValueError::new_err(
                    "weights must have the same length as distances and gamma",
                ));
            }
            arr.to_vec()
        }
        None => vec![1.0; distances_view.len()],
    };

    if weight_vec.iter().any(|w| !w.is_finite()) {
        return Err(PyValueError::new_err("weights must be finite"));
    }

    if weight_vec.iter().all(|w| *w <= 0.0) {
        for w in &mut weight_vec {
            *w = 1.0;
        }
    }

    if weight_vec.len() != distances_view.len() {
        return Err(PyValueError::new_err(
            "weights must match the number of observations",
        ));
    }

    if weight_vec.iter().any(|w| *w < 0.0) {
        return Err(PyValueError::new_err("weights must be non-negative"));
    }

    let fix_array = match fix_mask {
        Some(mask) => {
            let arr = mask.as_array();
            if arr.len() != 3 {
                return Err(PyValueError::new_err(
                    "fix_mask must contain three boolean flags",
                ));
            }
            [arr[0], arr[1], arr[2]]
        }
        None => [false, false, false],
    };

    let model = VariogramModel::from_str(model_type)?;

    let distances_vec = distances_view.to_vec();
    let gamma_vec = gamma_view.to_vec();

    let summary = run_levenberg_marquardt(
        model,
        &distances_vec,
        &gamma_vec,
        &mut weight_vec,
        &mut params,
        &fix_array,
    )?;

    Py::new(
        py,
        FittingResult {
            parameters: summary.parameters,
            r_squared: summary.r_squared,
            rmse: summary.rmse,
            converged: summary.converged,
            iterations: summary.iterations,
            message: summary.message,
        },
    )
}

struct OptimizationSummary {
    parameters: Vec<f64>,
    r_squared: f64,
    rmse: f64,
    converged: bool,
    iterations: usize,
    message: String,
}

fn run_levenberg_marquardt(
    model: VariogramModel,
    distances: &[f64],
    gamma: &[f64],
    weights: &mut [f64],
    params: &mut [f64],
    fix_mask: &[bool; 3],
) -> PyResult<OptimizationSummary> {
    if distances.len() != gamma.len() || distances.len() != weights.len() {
        return Err(PyValueError::new_err(
            "distances, gamma, and weights must have the same length",
        ));
    }

    if distances.is_empty() {
        return Err(PyValueError::new_err("No data provided for fitting"));
    }

    if distances.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("distances must be finite"));
    }

    if gamma.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("gamma values must be finite"));
    }

    if params.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "initial parameters must be finite values",
        ));
    }

    sanitize_initial_params(params, fix_mask)?;

    ensure_weight_balance(weights);

    let free_indices: Vec<usize> = (0..3).filter(|idx| !fix_mask[*idx]).collect();

    if free_indices.is_empty() {
        let (r_squared, rmse) = compute_statistics(model, distances, gamma, weights, params);
        return Ok(OptimizationSummary {
            parameters: params.to_vec(),
            r_squared,
            rmse,
            converged: true,
            iterations: 0,
            message: String::from("All parameters fixed; skipped optimization"),
        });
    }

    let mut lambda = 1e-3;
    let mut iterations = 0usize;
    let mut converged = false;
    let mut params_vec = params.to_vec();

    while iterations < 200 {
        iterations += 1;

        let (cost, jtj, jtr) =
            build_normal_equations(model, &params_vec, distances, gamma, weights);
        let (matrix, rhs) = assemble_linear_system(&jtj, &jtr, &free_indices, lambda);

        let Some(step) = solve_linear_system(matrix, rhs) else {
            lambda *= 10.0;
            if lambda > MAX_LAMBDA {
                break;
            }
            continue;
        };

        let mut candidate = params_vec.clone();
        for (idx, &param_idx) in free_indices.iter().enumerate() {
            candidate[param_idx] += step[idx];
        }

        enforce_bounds(&mut candidate, fix_mask);

        let (candidate_cost, _, _) =
            build_normal_equations(model, &candidate, distances, gamma, weights);

        if candidate_cost < cost {
            let cost_delta = (cost - candidate_cost).abs();
            let param_norm = step.iter().map(|d| d * d).sum::<f64>().sqrt();

            params_vec = candidate;
            lambda *= 0.3;
            if lambda < 1e-7 {
                lambda = 1e-7;
            }

            if param_norm < PARAM_TOL || cost_delta < COST_TOL {
                converged = true;
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > MAX_LAMBDA {
                break;
            }
        }
    }

    enforce_bounds(&mut params_vec, fix_mask);

    let (r_squared, rmse) = compute_statistics(model, distances, gamma, weights, &params_vec);

    let message = if converged {
        format!("Converged in {} iterations", iterations)
    } else if lambda > MAX_LAMBDA {
        format!(
            "Failed to converge: damping parameter exceeded limit after {} iterations",
            iterations
        )
    } else {
        format!(
            "Reached iteration limit ({} iterations) without convergence",
            iterations
        )
    };

    Ok(OptimizationSummary {
        parameters: params_vec,
        r_squared,
        rmse,
        converged,
        iterations,
        message,
    })
}

fn sanitize_initial_params(params: &mut [f64], fix_mask: &[bool; 3]) -> PyResult<()> {
    if params.len() != 3 {
        return Err(PyValueError::new_err(
            "initial_params must contain three values",
        ));
    }

    if fix_mask[0] && params[0] < 0.0 {
        return Err(PyValueError::new_err("Cannot fix nugget below zero"));
    }

    if params[0] < 0.0 {
        params[0] = 0.0;
    }

    if !params[2].is_finite() {
        return Err(PyValueError::new_err("Range parameter must be finite"));
    }

    if fix_mask[2] && params[2] <= MIN_RANGE {
        return Err(PyValueError::new_err("Fixed range must be positive"));
    }

    if params[2] <= MIN_RANGE {
        params[2] = MIN_RANGE;
    }

    if !params[1].is_finite() {
        return Err(PyValueError::new_err("Sill parameter must be finite"));
    }

    if fix_mask[1] && params[1] <= params[0] {
        return Err(PyValueError::new_err(
            "Fixed sill must be greater than fixed nugget",
        ));
    }

    if params[1] <= params[0] + MIN_SILL_GAP {
        params[1] = params[0] + MIN_SILL_GAP;
    }

    Ok(())
}

fn ensure_weight_balance(weights: &mut [f64]) {
    if weights.is_empty() {
        return;
    }

    let mut has_positive = false;
    for w in weights.iter_mut() {
        if !w.is_finite() || *w < 0.0 {
            *w = 0.0;
        }
        if *w > 0.0 {
            has_positive = true;
        }
    }

    if !has_positive {
        for w in weights.iter_mut() {
            *w = 1.0;
        }
    }
}

fn build_normal_equations(
    model: VariogramModel,
    params: &[f64],
    distances: &[f64],
    gamma: &[f64],
    weights: &[f64],
) -> (f64, [[f64; 3]; 3], [f64; 3]) {
    let mut cost = 0.0;
    let mut jtj = [[0.0f64; 3]; 3];
    let mut jtr = [0.0f64; 3];

    for (idx, (&distance, &observed)) in distances.iter().zip(gamma.iter()).enumerate() {
        let weight = weights[idx];
        if weight <= 0.0 {
            continue;
        }

        let sqrt_weight = weight.sqrt();
        let model_value = variogram_model_value(model, distance, params);
        let residual = sqrt_weight * (model_value - observed);
        cost += 0.5 * residual * residual;

        let jacobian = variogram_jacobian(model, distance, params);
        let weighted_jacobian = [
            sqrt_weight * jacobian[0],
            sqrt_weight * jacobian[1],
            sqrt_weight * jacobian[2],
        ];

        for row in 0..3 {
            jtr[row] += weighted_jacobian[row] * residual;
            for col in row..3 {
                jtj[row][col] += weighted_jacobian[row] * weighted_jacobian[col];
            }
        }
    }

    for row in 0..3 {
        for col in 0..row {
            jtj[row][col] = jtj[col][row];
        }
    }

    (cost, jtj, jtr)
}

fn assemble_linear_system(
    jtj: &[[f64; 3]; 3],
    jtr: &[f64; 3],
    free_indices: &[usize],
    lambda: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut matrix = vec![vec![0.0f64; free_indices.len()]; free_indices.len()];
    let mut rhs = vec![0.0f64; free_indices.len()];

    for (row_idx, &row) in free_indices.iter().enumerate() {
        rhs[row_idx] = -jtr[row];
        for (col_idx, &col) in free_indices.iter().enumerate() {
            let mut value = jtj[row][col];
            if row_idx == col_idx {
                let diag = jtj[row][row];
                let diag_scale = if diag.abs() < 1e-12 { 1.0 } else { diag.abs() };
                value += lambda * diag_scale;
            }
            matrix[row_idx][col_idx] = value;
        }
    }

    (matrix, rhs)
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    if n == 0 {
        return Some(Vec::new());
    }

    for i in 0..n {
        let mut pivot = i;
        let mut max_value = matrix[i][i].abs();
        for row in i + 1..n {
            let candidate = matrix[row][i].abs();
            if candidate > max_value {
                max_value = candidate;
                pivot = row;
            }
        }

        if max_value < 1e-12 {
            return None;
        }

        if pivot != i {
            matrix.swap(i, pivot);
            rhs.swap(i, pivot);
        }

        let diag = matrix[i][i];
        for row in i + 1..n {
            let factor = matrix[row][i] / diag;
            for col in i..n {
                matrix[row][col] -= factor * matrix[i][col];
            }
            rhs[row] -= factor * rhs[i];
        }
    }

    let mut solution = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut value = rhs[i];
        for col in i + 1..n {
            value -= matrix[i][col] * solution[col];
        }
        let diag = matrix[i][i];
        if diag.abs() < 1e-12 {
            return None;
        }
        solution[i] = value / diag;
    }

    Some(solution)
}

fn enforce_bounds(params: &mut [f64], fix_mask: &[bool; 3]) {
    if params.is_empty() {
        return;
    }

    if params[0] < 0.0 {
        params[0] = 0.0;
    }

    if params[2] < MIN_RANGE {
        params[2] = MIN_RANGE;
    }

    if params[1] <= params[0] {
        if fix_mask[1] && !fix_mask[0] {
            params[0] = (params[1] - MIN_SILL_GAP).max(0.0);
        } else {
            params[1] = params[0] + MIN_SILL_GAP;
        }
    }

    if params[1] - params[0] < MIN_SILL_GAP {
        if fix_mask[1] && !fix_mask[0] {
            params[0] = (params[1] - MIN_SILL_GAP).max(0.0);
        } else {
            params[1] = params[0] + MIN_SILL_GAP;
        }
    }

    if params[0] < 0.0 {
        params[0] = 0.0;
    }

    if params[1] <= params[0] {
        params[1] = params[0] + MIN_SILL_GAP;
    }

    if params[2] <= MIN_RANGE {
        params[2] = MIN_RANGE;
    }
}

fn compute_statistics(
    model: VariogramModel,
    distances: &[f64],
    gamma: &[f64],
    weights: &[f64],
    params: &[f64],
) -> (f64, f64) {
    let n = distances.len();
    if n == 0 {
        return (1.0, 0.0);
    }

    let mut effective_weights = Vec::with_capacity(n);
    let mut total_weight = 0.0;

    for &w in weights {
        let weight = if w.is_finite() && w > 0.0 { w } else { 0.0 };
        effective_weights.push(weight);
        total_weight += weight;
    }

    if total_weight <= 0.0 {
        effective_weights.clear();
        effective_weights.resize(n, 1.0);
        total_weight = n as f64;
    }

    let mut weighted_residual_sum = 0.0;
    let mut weighted_gamma_sum = 0.0;

    for i in 0..n {
        let weight = effective_weights[i];
        let prediction = variogram_model_value(model, distances[i], params);
        let diff = prediction - gamma[i];
        weighted_residual_sum += weight * diff * diff;
        weighted_gamma_sum += weight * gamma[i];
    }

    let mean = if total_weight > 0.0 {
        weighted_gamma_sum / total_weight
    } else {
        0.0
    };

    let mut ss_tot = 0.0;
    for i in 0..n {
        let weight = effective_weights[i];
        let diff = gamma[i] - mean;
        ss_tot += weight * diff * diff;
    }

    let rmse = if total_weight > 0.0 {
        (weighted_residual_sum / total_weight).sqrt()
    } else {
        0.0
    };

    let r_squared = if ss_tot <= 0.0 {
        1.0
    } else {
        1.0 - (weighted_residual_sum / ss_tot)
    };

    (r_squared, rmse)
}

fn variogram_model_value(model: VariogramModel, distance: f64, params: &[f64]) -> f64 {
    let nugget = params[0].max(0.0);
    let sill = params[1];
    let range = params[2].max(MIN_RANGE);

    match model {
        VariogramModel::Exponential => {
            let exponent = (-distance / range).exp();
            nugget + (sill - nugget) * (1.0 - exponent)
        }
        VariogramModel::Spherical => {
            if distance >= range {
                sill
            } else {
                let ratio = (distance / range).max(0.0);
                nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio.powi(3))
            }
        }
        VariogramModel::Gaussian => {
            let ratio = distance / range;
            let exponent = -(ratio * ratio);
            nugget + (sill - nugget) * (1.0 - exponent.exp())
        }
    }
}

fn variogram_jacobian(model: VariogramModel, distance: f64, params: &[f64]) -> [f64; 3] {
    let nugget = params[0].max(0.0);
    let sill = params[1];
    let range = params[2].max(MIN_RANGE);
    let sill_span = sill - nugget;

    match model {
        VariogramModel::Exponential => {
            let exp_term = (-distance / range).exp();
            let d_nugget = exp_term;
            let d_sill = 1.0 - exp_term;
            let d_range = -sill_span * exp_term * (distance / (range * range));
            [d_nugget, d_sill, d_range]
        }
        VariogramModel::Spherical => {
            if distance >= range {
                [0.0, 1.0, 0.0]
            } else {
                let ratio = (distance / range).max(0.0);
                let ratio_sq = ratio * ratio;
                let shape = 1.5 * ratio - 0.5 * ratio * ratio_sq;
                let d_nugget = 1.0 - shape;
                let d_sill = shape;
                let d_shape_d_ratio = 1.5 - 1.5 * ratio_sq;
                let d_ratio_d_range = -distance / (range * range);
                let d_range = sill_span * d_shape_d_ratio * d_ratio_d_range;
                [d_nugget, d_sill, d_range]
            }
        }
        VariogramModel::Gaussian => {
            let ratio = distance / range;
            let ratio_sq = ratio * ratio;
            let exp_term = (-ratio_sq).exp();
            let d_nugget = exp_term;
            let d_sill = 1.0 - exp_term;
            let d_range =
                -sill_span * exp_term * (2.0 * distance * distance / (range * range * range));
            [d_nugget, d_sill, d_range]
        }
    }
}
