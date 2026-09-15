// src/rust/src/kriging.rs
use nalgebra::{DMatrix, DVector};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::slice::ParallelSliceMut;

use crate::utils::euclidean_distance_single;

/// Covariance between every point of `coords_a` and every point of `coords_b`.
///
/// Rows are computed in parallel with the GIL released. Kriging estimators use it
/// to build their systems.
#[pyfunction]
pub fn covariance_between<'py>(
    py: Python<'py>,
    coords_a: PyReadonlyArray2<f64>,
    coords_b: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let params = variogram_params.as_array();
    check_parameters(&params)?;
    let model = CovarianceModel::new(&params, model_type);
    // Owned copies, so that nothing borrows Python memory while the GIL is released.
    let coords_a = coords_a.as_array().to_owned();
    let coords_b = coords_b.as_array().to_owned();
    if coords_a.ncols() != coords_b.ncols() {
        return Err(PyValueError::new_err(
            "Coordinate arrays must have the same number of columns",
        ));
    }

    let covariance = py.detach(|| {
        let mut covariance = Array2::<f64>::zeros((coords_a.nrows(), coords_b.nrows()));
        covariance
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(coords_a.axis_iter(Axis(0)).into_par_iter())
            .for_each(|(mut row, point)| {
                for (value, other) in row.iter_mut().zip(coords_b.axis_iter(Axis(0))) {
                    *value = model.covariance(euclidean_distance_single(point, other));
                }
            });
        covariance
    });

    Ok(covariance.into_pyarray(py))
}

/// For each target, the sum over samples of covariance times the sample's weight.
///
/// Given dual kriging weights, one solve of the kriging system against the sample
/// values, this is the kriging prediction without its drift term: the work per
/// target grows with the number of samples, rather than with its square, and no
/// system is solved. Targets are processed in parallel with the GIL released.
#[pyfunction]
pub fn dual_kriging_predict<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
    weights: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let params = variogram_params.as_array();
    check_parameters(&params)?;
    let model = CovarianceModel::new(&params, model_type);
    let known_coords = known_coords.as_array().to_owned();
    let pred_coords = pred_coords.as_array().to_owned();
    let weights = weights.as_array().to_owned();
    if weights.len() != known_coords.nrows() {
        return Err(PyValueError::new_err(
            "weights must have one entry per known coordinate",
        ));
    }
    if pred_coords.ncols() != known_coords.ncols() {
        return Err(PyValueError::new_err(
            "Prediction coordinates must have the same dimensionality as known coordinates",
        ));
    }

    let sums: Vec<f64> = py.detach(|| {
        pred_coords
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|target| {
                known_coords
                    .axis_iter(Axis(0))
                    .zip(weights.iter())
                    .map(|(sample, weight)| {
                        model.covariance(euclidean_distance_single(target, sample)) * weight
                    })
                    .sum::<f64>()
            })
            .collect()
    });

    Ok(Array1::from_vec(sums).into_pyarray(py))
}

const SINGULAR_SYSTEM_MESSAGE: &str =
    "Singular covariance matrix - check for duplicate points or poor conditioning";

/// LU-factorise a square matrix with partial pivoting, to solve it many times.
///
/// Returns the factors packed into one matrix, `L` below the diagonal (its unit
/// diagonal implied) and `U` on and above it, and the row permutation: row `i` of
/// the factorised matrix is row `permutation[i]` of the original. Raises
/// `ValueError` if a pivot is zero, as it is when two samples share a location.
#[pyfunction]
pub fn lu_factorize<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>)> {
    let matrix = matrix.as_array();
    let size = matrix.nrows();
    if matrix.ncols() != size {
        return Err(PyValueError::new_err("matrix must be square"));
    }

    let mut data: Vec<f64> = matrix.iter().copied().collect();
    let mut permutation: Vec<i64> = (0..size as i64).collect();
    let factorised = py.detach(|| lu_factor_in_place(&mut data, size, &mut permutation));
    if !factorised {
        return Err(PyValueError::new_err(SINGULAR_SYSTEM_MESSAGE));
    }

    let factors = Array2::from_shape_vec((size, size), data)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((
        factors.into_pyarray(py),
        Array1::from_vec(permutation).into_pyarray(py),
    ))
}

/// Solve a system for one right-hand side, from the output of `lu_factorize`.
#[pyfunction]
pub fn lu_solve<'py>(
    py: Python<'py>,
    lu: PyReadonlyArray2<f64>,
    permutation: PyReadonlyArray1<i64>,
    rhs: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (lu, permutation, size) = factor_slices(&lu, &permutation)?;
    let rhs = rhs.as_array();
    if rhs.len() != size {
        return Err(PyValueError::new_err(
            "rhs must have one entry per row of the factorised matrix",
        ));
    }

    let rhs: Vec<f64> = rhs.iter().copied().collect();
    let mut solution = vec![0.0; size];
    lu_solve_into(lu, permutation, &rhs, &mut solution);
    Ok(Array1::from_vec(solution).into_pyarray(py))
}

/// Ordinary kriging variance at each target, from a factorised ordinary system.
///
/// For each target the covariances to the samples, bordered by a one, are solved
/// against the factors, and the variance is the sill minus their dot product with
/// the solution, floored at zero. Targets are processed in parallel with the GIL
/// released, each independently of the others, so the result does not depend on
/// how targets are grouped into calls.
#[pyfunction]
pub fn factorised_kriging_variance<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
    lu: PyReadonlyArray2<f64>,
    permutation: PyReadonlyArray1<i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let params = variogram_params.as_array();
    check_parameters(&params)?;
    let model = CovarianceModel::new(&params, model_type);
    let known_coords = known_coords.as_array();
    let pred_coords = pred_coords.as_array();
    let (lu, permutation, size) = factor_slices(&lu, &permutation)?;
    let n_known = known_coords.nrows();
    if size != n_known + 1 {
        return Err(PyValueError::new_err(
            "The factors must be of an ordinary kriging system for the known coordinates",
        ));
    }
    if pred_coords.ncols() != known_coords.ncols() {
        return Err(PyValueError::new_err(
            "Prediction coordinates must have the same dimensionality as known coordinates",
        ));
    }

    let variances: Vec<f64> = py.detach(|| {
        pred_coords
            .axis_iter(Axis(0))
            .into_par_iter()
            .map_init(
                || (vec![0.0; size], vec![0.0; size]),
                |(rhs, solution), target| {
                    for (entry, sample) in rhs.iter_mut().zip(known_coords.axis_iter(Axis(0))) {
                        *entry = model.covariance(euclidean_distance_single(target, sample));
                    }
                    rhs[n_known] = 1.0;
                    lu_solve_into(lu, permutation, rhs, solution);
                    let explained: f64 = rhs.iter().zip(solution.iter()).map(|(a, b)| a * b).sum();
                    (model.sill - explained).max(0.0)
                },
            )
            .collect()
    });

    Ok(Array1::from_vec(variances).into_pyarray(py))
}

fn factor_slices<'a>(
    lu: &'a PyReadonlyArray2<f64>,
    permutation: &'a PyReadonlyArray1<i64>,
) -> PyResult<(&'a [f64], &'a [i64], usize)> {
    let size = lu.as_array().nrows();
    if lu.as_array().ncols() != size || permutation.as_array().len() != size {
        return Err(PyValueError::new_err(
            "lu must be square, with one permutation entry per row",
        ));
    }
    let lu = lu
        .as_slice()
        .map_err(|_| PyValueError::new_err("lu must be a contiguous array"))?;
    let permutation = permutation
        .as_slice()
        .map_err(|_| PyValueError::new_err("permutation must be a contiguous array"))?;
    if permutation
        .iter()
        .any(|&row| row < 0 || row as usize >= size)
    {
        return Err(PyValueError::new_err(
            "permutation entries must be row indices",
        ));
    }
    Ok((lu, permutation, size))
}

/// Factorise the row-major `size` x `size` matrix in `data` in place, recording row
/// swaps in `permutation`. Returns false if a pivot is zero or not finite.
fn lu_factor_in_place(data: &mut [f64], size: usize, permutation: &mut [i64]) -> bool {
    for k in 0..size {
        let mut pivot_row = k;
        let mut pivot_abs = data[k * size + k].abs();
        for row in (k + 1)..size {
            let candidate = data[row * size + k].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        // A NaN pivot is not finite either.
        if pivot_abs == 0.0 || !pivot_abs.is_finite() {
            return false;
        }
        if pivot_row != k {
            let (before, from_pivot) = data.split_at_mut(pivot_row * size);
            before[k * size..(k + 1) * size].swap_with_slice(&mut from_pivot[..size]);
            permutation.swap(k, pivot_row);
        }

        let (upper, lower) = data.split_at_mut((k + 1) * size);
        let pivot_values = &upper[k * size..];
        let pivot = pivot_values[k];
        let eliminate = |row: &mut [f64]| {
            let factor = row[k] / pivot;
            row[k] = factor;
            if factor != 0.0 {
                for (value, above) in row[k + 1..].iter_mut().zip(&pivot_values[k + 1..]) {
                    *value -= factor * above;
                }
            }
        };
        // Rows are independent once the pivot row is fixed; only large remaining
        // blocks are worth sending to other threads.
        let remaining = size - k - 1;
        if remaining * (remaining + 1) > 1 << 16 {
            lower.par_chunks_mut(size).for_each(eliminate);
        } else {
            lower.chunks_mut(size).for_each(eliminate);
        }
    }
    true
}

/// Solve `LU x = P b` for one right-hand side, from packed row-major factors.
fn lu_solve_into(lu: &[f64], permutation: &[i64], rhs: &[f64], solution: &mut [f64]) {
    let size = solution.len();
    for (entry, &row) in solution.iter_mut().zip(permutation) {
        *entry = rhs[row as usize];
    }
    for i in 0..size {
        let row = &lu[i * size..i * size + i];
        let reduction: f64 = row.iter().zip(&solution[..i]).map(|(l, x)| l * x).sum();
        solution[i] -= reduction;
    }
    for i in (0..size).rev() {
        let row = &lu[i * size..(i + 1) * size];
        let reduction: f64 = row[i + 1..]
            .iter()
            .zip(&solution[i + 1..])
            .map(|(u, x)| u * x)
            .sum();
        solution[i] = (solution[i] - reduction) / row[i];
    }
}

fn check_parameters(params: &ArrayView1<f64>) -> PyResult<()> {
    if params.len() != 3 {
        return Err(PyValueError::new_err(
            "variogram_params must contain three values: [nugget, sill, range]",
        ));
    }
    Ok(())
}

/// Perform ordinary kriging prediction
#[pyfunction]
pub fn ordinary_kriging_predict<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    known_values: PyReadonlyArray1<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
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

    system.view_mut((0, 0), (n_known, n_known)).copy_from(&cov);
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
pub fn ordinary_kriging_predict_neighbors<'py>(
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

    if params.len() != 3 {
        return Err(PyValueError::new_err(
            "variogram_params must contain three values: [nugget, sill, range]",
        ));
    }

    if neighbors.nrows() != pred_coords.nrows() {
        return Err(PyValueError::new_err(
            "neighbors must have shape (n_predictions, k_neighbors)",
        ));
    }

    let n_pred = pred_coords.nrows();
    let mut predictions = vec![f64::NAN; n_pred];

    predictions
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, prediction)| {
            let neighbor_ids: Vec<usize> = neighbors
                .row(idx)
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
                *prediction = f64::NAN;
                return;
            }

            let k = neighbor_ids.len();
            let mut system = DMatrix::<f64>::zeros(k + 1, k + 1);

            for (row_pos, &i_idx) in neighbor_ids.iter().enumerate() {
                let coord_i = known_coords.row(i_idx);
                for (col_pos, &j_idx) in neighbor_ids.iter().enumerate() {
                    let coord_j = known_coords.row(j_idx);
                    let distance = euclidean_distance_single(coord_i, coord_j);
                    system[(row_pos, col_pos)] =
                        variogram_to_covariance(distance, &params, model_type);
                }
                system[(row_pos, k)] = 1.0;
                system[(k, row_pos)] = 1.0;
            }

            let lu = system.lu();
            let mut rhs = DVector::<f64>::zeros(k + 1);
            let pred_coord = pred_coords.row(idx);
            for (row_pos, &i_idx) in neighbor_ids.iter().enumerate() {
                let distance = euclidean_distance_single(known_coords.row(i_idx), pred_coord);
                rhs[row_pos] = variogram_to_covariance(distance, &params, model_type);
            }
            rhs[k] = 1.0;

            match lu.solve(&rhs) {
                Some(weights) => {
                    let mut value = 0.0;
                    for (weight_pos, &i_idx) in neighbor_ids.iter().enumerate() {
                        value += weights[weight_pos] * known_values[i_idx];
                    }
                    *prediction = value;
                }
                None => {
                    *prediction = f64::NAN;
                }
            }
        });

    Ok(Array1::from_vec(predictions).into_pyarray(py))
}

#[pyfunction]
pub fn simple_kriging_predict<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    known_values: PyReadonlyArray1<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
    known_mean: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
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
) -> PyResult<Bound<'py, PyArray1<f64>>> {
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
    system.view_mut((0, 0), (n_known, n_known)).copy_from(&cov);
    system
        .view_mut((0, n_known), (n_known, basis_size))
        .copy_from(&design);
    system
        .view_mut((n_known, 0), (basis_size, n_known))
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

/// Calculate ordinary kriging variance.
#[pyfunction]
pub fn kriging_variance<'py>(
    py: Python<'py>,
    known_coords: PyReadonlyArray2<f64>,
    pred_coords: PyReadonlyArray2<f64>,
    variogram_params: PyReadonlyArray1<f64>,
    model_type: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let known_coords = known_coords.as_array();
    let pred_coords = pred_coords.as_array();
    let params = variogram_params.as_array();

    if params.len() != 3 {
        return Err(PyValueError::new_err(
            "variogram_params must contain three values: [nugget, sill, range]",
        ));
    }

    let n_known = known_coords.nrows();
    if n_known == 0 {
        return Err(PyValueError::new_err(
            "known_coords must contain at least one point",
        ));
    }

    let n_pred = pred_coords.nrows();
    let mut variances = Array1::<f64>::zeros(n_pred);

    let cov = build_covariance_matrix(&known_coords, &params, model_type);
    let mut system = DMatrix::<f64>::zeros(n_known + 1, n_known + 1);
    system.view_mut((0, 0), (n_known, n_known)).copy_from(&cov);
    for i in 0..n_known {
        system[(i, n_known)] = 1.0;
        system[(n_known, i)] = 1.0;
    }

    let lu = system.lu();
    let c00 = variogram_to_covariance(0.0, &params, model_type);

    for p in 0..n_pred {
        let mut rhs = DVector::<f64>::zeros(n_known + 1);
        for i in 0..n_known {
            let distance = euclidean_distance_single(known_coords.row(i), pred_coords.row(p));
            rhs[i] = variogram_to_covariance(distance, &params, model_type);
        }
        rhs[n_known] = 1.0;

        let solution = lu
            .solve(&rhs)
            .ok_or_else(|| PyValueError::new_err("Failed to solve kriging variance system"))?;

        let mut w_dot_c = 0.0;
        for i in 0..n_known {
            w_dot_c += solution[i] * rhs[i];
        }
        let lagrange = solution[n_known];
        variances[p] = (c00 - w_dot_c - lagrange).max(0.0);
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
                "Unsupported trend type '{other}'. Expected 'linear' or 'quadratic'"
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
    CovarianceModel::new(params, model_type).covariance(distance)
}

/// A variogram model evaluated as a covariance, with its name parsed once rather
/// than for every pair of points.
#[derive(Clone, Copy)]
struct CovarianceModel {
    nugget: f64,
    sill: f64,
    range: f64,
    kind: ModelKind,
}

#[derive(Clone, Copy)]
enum ModelKind {
    Exponential,
    Spherical,
    Gaussian,
    /// Any other name has no covariance beyond distance zero.
    Other,
}

impl CovarianceModel {
    fn new(params: &ArrayView1<f64>, model_type: &str) -> Self {
        let kind = match model_type {
            "exponential" => ModelKind::Exponential,
            "spherical" => ModelKind::Spherical,
            "gaussian" => ModelKind::Gaussian,
            _ => ModelKind::Other,
        };
        Self {
            nugget: params[0],
            sill: params[1],
            range: params[2],
            kind,
        }
    }

    fn covariance(&self, distance: f64) -> f64 {
        let (nugget, sill, range) = (self.nugget, self.sill, self.range);

        if distance == 0.0 {
            return sill;
        }

        let gamma = match self.kind {
            ModelKind::Exponential => nugget + (sill - nugget) * (1.0 - (-distance / range).exp()),
            ModelKind::Spherical => {
                if distance < range {
                    let ratio = distance / range;
                    nugget + (sill - nugget) * (1.5 * ratio - 0.5 * ratio.powi(3))
                } else {
                    sill
                }
            }
            ModelKind::Gaussian => {
                let ratio = distance / range;
                nugget + (sill - nugget) * (1.0 - (-(ratio * ratio)).exp())
            }
            ModelKind::Other => sill,
        };

        sill - gamma
    }
}
