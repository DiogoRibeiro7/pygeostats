// src/rust/src/variogram.rs
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::time::{Duration, Instant};

use crate::utils::euclidean_distance_single;

const MIN_RANGE: f64 = 1e-6;
const MIN_SILL_GAP: f64 = 1e-9;
const MAX_LAMBDA: f64 = 1e12;
const PARAM_TOL: f64 = 1e-6;
const COST_TOL: f64 = 1e-9;
const GRADIENT_TOL: f64 = 1e-6;
const STEP_TOL: f64 = 1e-6;
const MAX_ITERATIONS: usize = 250;
const TRACE_LIMIT: usize = 64;
const TIMEOUT_MIN_SECONDS: f64 = 0.5;
const TIMEOUT_PER_SAMPLE_MICROS: f64 = 4.0;
const MAX_RATIO_REASONABLE: f64 = 50.0;
const CACHE_EPS: f64 = 1e-12;

#[allow(dead_code)]
#[derive(Clone)]
struct ConstraintBounds {
    min_range: f64,
    current_max_range: f64,
    max_range_limit: f64,
    min_ratio: f64,
    max_ratio: f64,
    angle_min: f64,
    angle_max: f64,
    max_sill: f64,
}

#[allow(dead_code)]
struct ConstraintPenalty {
    residual: f64,
    gradient: Vec<f64>,
}

#[allow(dead_code)]
struct ConstraintManager {
    bounds: ConstraintBounds,
    penalty_weight: f64,
    clamp_hits: usize,
    relaxations: usize,
    pending_penalties: Vec<ConstraintPenalty>,
    messages: Vec<String>,
    ratio_warnings_emitted: usize,
}

const MAX_CONSTRAINT_MESSAGES: usize = 8;

#[derive(Clone, Debug)]
struct IterationTraceRecord {
    iteration: usize,
    cost: f64,
    lambda: f64,
    step_norm: f64,
    gradient_norm: f64,
}

impl IterationTraceRecord {
    fn to_tuple(&self) -> (usize, f64, f64, f64, f64) {
        (
            self.iteration,
            self.cost,
            self.lambda,
            self.step_norm,
            self.gradient_norm,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OptimizationStatus {
    Succeeded,
    MaxIterations,
    SingularMatrix,
    Diverged,
    Timeout,
    InvalidParameters,
    FallbackToIsotropic,
    Failed,
}

impl OptimizationStatus {
    fn as_str(&self) -> &'static str {
        match self {
            OptimizationStatus::Succeeded => "succeeded",
            OptimizationStatus::MaxIterations => "max_iterations",
            OptimizationStatus::SingularMatrix => "singular_matrix",
            OptimizationStatus::Diverged => "diverged",
            OptimizationStatus::Timeout => "timeout",
            OptimizationStatus::InvalidParameters => "invalid_parameters",
            OptimizationStatus::FallbackToIsotropic => "fallback_to_isotropic",
            OptimizationStatus::Failed => "failed",
        }
    }

    fn is_success(&self) -> bool {
        matches!(self, OptimizationStatus::Succeeded)
    }
}

struct OptimizationDiagnostics {
    initial_cost: f64,
    final_cost: f64,
    gradient_norm: f64,
    step_norm: f64,
    lambda: f64,
    runtime_seconds: f64,
    evaluations: usize,
    ratio: f64,
    penalty_hits: usize,
    constraint_messages: Vec<String>,
    isotropic_r2: Option<f64>,
    isotropic_rmse: Option<f64>,
}

impl OptimizationDiagnostics {
    fn to_pairs(&self) -> Vec<(String, f64)> {
        let mut pairs = Vec::with_capacity(10);
        if self.initial_cost.is_finite() {
            pairs.push(("initial_cost".to_string(), self.initial_cost));
        }
        if self.final_cost.is_finite() {
            pairs.push(("final_cost".to_string(), self.final_cost));
        }
        if self.gradient_norm.is_finite() {
            pairs.push(("gradient_norm".to_string(), self.gradient_norm));
        }
        if self.step_norm.is_finite() {
            pairs.push(("step_norm".to_string(), self.step_norm));
        }
        if self.lambda.is_finite() {
            pairs.push(("lambda".to_string(), self.lambda));
        }
        if self.runtime_seconds.is_finite() {
            pairs.push(("runtime_seconds".to_string(), self.runtime_seconds));
        }
        pairs.push(("evaluations".to_string(), self.evaluations as f64));
        if self.ratio.is_finite() {
            pairs.push(("major_minor_ratio".to_string(), self.ratio));
        }
        if let Some(r2) = self.isotropic_r2 {
            pairs.push(("isotropic_r2".to_string(), r2));
        }
        if let Some(rmse) = self.isotropic_rmse {
            pairs.push(("isotropic_rmse".to_string(), rmse));
        }
        pairs
    }
}

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
    #[pyo3(get)]
    status: String,
    #[pyo3(get)]
    parameter_std: Vec<f64>,
    #[pyo3(get)]
    diagnostics: Vec<(String, f64)>,
    #[pyo3(get)]
    trace: Vec<(usize, f64, f64, f64, f64)>,
    #[pyo3(get)]
    fallback_used: bool,
    #[pyo3(get)]
    warnings: Vec<String>,
}

#[pymethods]
impl FittingResult {
    fn __repr__(&self) -> String {
        let escaped_message = self.message.replace('\'', "\\'");
        format!(
            "FittingResult(parameters={:?}, r_squared={:.6}, rmse={:.6}, converged={}, iterations={}, status='{}', fallback_used={}, message='{}')",
            self.parameters,
            self.r_squared,
            self.rmse,
            self.converged,
            self.iterations,
            self.status,
            self.fallback_used,
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

    for i in 0..n_bins {
        bin_centers[i] = (bins[i] + bins[i + 1]) / 2.0;
    }

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

#[pyfunction(signature = (distances, gamma, model_type, initial_params, weights=None, fix_mask=None, directions=None))]
pub fn fit_variogram_model<'py>(
    py: Python<'py>,
    distances: PyReadonlyArray1<f64>,
    gamma: PyReadonlyArray1<f64>,
    model_type: &str,
    initial_params: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    fix_mask: Option<PyReadonlyArray1<bool>>,
    directions: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<FittingResult>> {
    let distances = distances.as_array();
    let gamma = gamma.as_array();

    if distances.len() != gamma.len() {
        return Err(PyValueError::new_err(
            "distances and gamma must have the same length",
        ));
    }

    if distances.is_empty() {
        return Err(PyValueError::new_err("distances and gamma cannot be empty"));
    }

    let mut params = initial_params.as_array().to_vec();
    let param_count = params.len();
    if param_count != 3 && param_count != 5 {
        return Err(PyValueError::new_err(
            "initial_params must contain three values [nugget, sill, range] or five values [nugget, sill, range_major, range_minor, angle]",
        ));
    }
    let anisotropic = param_count == 5;

    let mut weights_vec = match weights {
        Some(w) => {
            let arr = w.as_array();
            if arr.len() != distances.len() {
                return Err(PyValueError::new_err(
                    "weights must have the same length as distances and gamma",
                ));
            }
            arr.to_vec()
        }
        None => vec![1.0; distances.len()],
    };

    if weights_vec.iter().any(|w| !w.is_finite()) {
        return Err(PyValueError::new_err("weights must be finite"));
    }

    if weights_vec.iter().all(|w| *w <= 0.0) {
        for w in &mut weights_vec {
            *w = 1.0;
        }
    }

    if weights_vec.iter().any(|w| *w < 0.0) {
        return Err(PyValueError::new_err("weights must be non-negative"));
    }

    let direction_vec: Option<Vec<f64>> = match directions {
        Some(arr) => {
            let arr = arr.as_array();
            if arr.len() != distances.len() {
                return Err(PyValueError::new_err(
                    "directions must have the same length as distances",
                ));
            }
            let mut out = Vec::with_capacity(arr.len());
            for &value in arr.iter() {
                if !value.is_finite() {
                    return Err(PyValueError::new_err("direction values must be finite"));
                }
                let radians = if value.abs() > 2.0 * PI { value.to_radians() } else { value };
                out.push(radians);
            }
            Some(out)
        }
        None => {
            if anisotropic {
                return Err(PyValueError::new_err(
                    "directions array is required when fitting anisotropic variograms",
                ));
            }
            None
        }
    };

    let fix_mask = match fix_mask {
        Some(mask) => {
            let arr = mask.as_array();
            if arr.len() != param_count {
                return Err(PyValueError::new_err(
                    "fix_mask must have the same length as initial_params",
                ));
            }
            arr.to_vec()
        }
        None => vec![false; param_count],
    };

    let model = VariogramModel::from_str(model_type)?;

    sanitize_initial_params(&mut params, &fix_mask, anisotropic)?;

    let mut opt_params = params.clone();
    if anisotropic {
        opt_params[2] = opt_params[2].ln();
        opt_params[3] = opt_params[3].ln();
    }

    ensure_weight_balance(&mut weights_vec);

    let dataset = Dataset {
        distances: distances.to_vec(),
        gamma: gamma.to_vec(),
        weights: weights_vec,
        directions: direction_vec,
    };

    let summary = run_levenberg_marquardt(
        model,
        dataset,
        &mut opt_params,
        &fix_mask,
        anisotropic,
    )?;

    let result = FittingResult {
        parameters: summary.parameters,
        r_squared: summary.r_squared,
        rmse: summary.rmse,
        converged: summary.converged,
        iterations: summary.iterations,
        message: summary.message,
    };

    Py::new(py, result)
}

#[derive(Clone)]
struct Dataset {
    distances: Vec<f64>,
    gamma: Vec<f64>,
    weights: Vec<f64>,
    directions: Option<Vec<f64>>, // radians
    cos_dirs: Option<Vec<f64>>,
    sin_dirs: Option<Vec<f64>>,
}

struct OptimizationSummary {
    parameters: Vec<f64>,
    r_squared: f64,
    rmse: f64,
    converged: bool,
    iterations: usize,
    message: String,
    status: OptimizationStatus,
    parameter_std: Vec<f64>,
    diagnostics: OptimizationDiagnostics,
    trace: Vec<IterationTraceRecord>,
    warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaptiveSlice {
    Full,
    Stride(usize),
}

impl AdaptiveSlice {
    fn stride(self) -> usize {
        match self {
            AdaptiveSlice::Full => 1,
            AdaptiveSlice::Stride(value) => value.max(1),
        }
    }
}

struct AdaptivePrecision {
    stride: usize,
    coarse_iters: usize,
    using_full: bool,
}

impl AdaptivePrecision {
    fn new(sample_count: usize) -> Self {
        let stride = if sample_count > 50_000 {
            16
        } else if sample_count > 20_000 {
            8
        } else if sample_count > 8_000 {
            4
        } else if sample_count > 2_000 {
            2
        } else {
            1
        };
        Self {
            stride: stride.max(1),
            coarse_iters: if stride > 1 { 6 } else { 0 },
            using_full: stride <= 1,
        }
    }

    fn slice_for(&self, iteration: usize) -> AdaptiveSlice {
        if self.using_full || iteration > self.coarse_iters {
            AdaptiveSlice::Full
        } else {
            AdaptiveSlice::Stride(self.stride)
        }
    }

    fn should_promote(&self, improvement: f64, step_norm: f64) -> bool {
        if self.using_full {
            return false;
        }
        improvement.abs() < COST_TOL * 10.0 || step_norm < STEP_TOL * 5.0
    }

    fn promote(&mut self) {
        self.using_full = true;
    }
}

#[derive(Clone)]
struct EvaluationCache {
    anisotropic: bool,
    last_params: Vec<f64>,
    last_slice: AdaptiveSlice,
    cost: f64,
    jtj: Vec<Vec<f64>>,
    jtr: Vec<f64>,
    gradient_norm: f64,
    valid: bool,
}

impl EvaluationCache {
    fn new(param_count: usize, anisotropic: bool) -> Self {
        Self {
            anisotropic,
            last_params: vec![0.0; param_count],
            last_slice: AdaptiveSlice::Full,
            cost: f64::NAN,
            jtj: vec![vec![0.0; param_count]; param_count],
            jtr: vec![0.0; param_count],
            gradient_norm: f64::NAN,
            valid: false,
        }
    }

    fn evaluate(
        &mut self,
        model: VariogramModel,
        anisotropic: bool,
        dataset: &Dataset,
        params: &[f64],
        slice: AdaptiveSlice,
    ) -> PyResult<(f64, Vec<Vec<f64>>, Vec<f64>, f64, usize)> {
        let same_as_last = self.valid
            && self.anisotropic == anisotropic
            && self.last_slice == slice
            && self.last_params.len() == params.len()
            && params
                .iter()
                .zip(self.last_params.iter())
                .all(|(a, b)| (a - b).abs() <= CACHE_EPS);

        if same_as_last {
            return Ok((
                self.cost,
                self.jtj.clone(),
                self.jtr.clone(),
                self.gradient_norm,
                0,
            ));
        }

        let (cost, jtj, jtr, evaluations) =
            build_normal_equations(model, anisotropic, dataset, params, slice)?;
        let gradient_norm = jtr.iter().map(|v| v * v).sum::<f64>().sqrt();

        self.anisotropic = anisotropic;
        self.last_params.clear();
        self.last_params.extend_from_slice(params);
        self.last_slice = slice;
        self.cost = cost;
        self.jtj = jtj.clone();
        self.jtr = jtr.clone();
        self.gradient_norm = gradient_norm;
        self.valid = true;

        Ok((cost, jtj, jtr, gradient_norm, evaluations))
    }
}

struct AnisotropicIterationContext {
    nugget: f64,
    sill: f64,
    range_major: f64,
    range_minor: f64,
    angle: f64,
    cos_angle: f64,
    sin_angle: f64,
}

impl AnisotropicIterationContext {
    fn from_params(params: &[f64]) -> Self {
        let nugget = params[0].max(0.0);
        let sill = params[1].max(nugget + MIN_SILL_GAP);
        let range_major = params[2].exp().max(MIN_RANGE);
        let range_minor = params[3].exp().max(MIN_RANGE);
        let angle = normalize_angle(params[4]);
        Self {
            nugget,
            sill,
            range_major,
            range_minor,
            angle,
            cos_angle: angle.cos(),
            sin_angle: angle.sin(),
        }
    }

    fn ratio(&self) -> f64 {
        if self.range_minor <= 0.0 {
            f64::INFINITY
        } else {
            self.range_major / self.range_minor
        }
    }
}

fn compute_timeout(sample_count: usize) -> Duration {
    let dynamic = (sample_count as f64) * TIMEOUT_PER_SAMPLE_MICROS / 1_000_000.0;
    let seconds = TIMEOUT_MIN_SECONDS.max(dynamic.min(10.0));
    Duration::from_secs_f64(seconds)
}

fn run_levenberg_marquardt(
    model: VariogramModel,
    dataset: Dataset,
    params: &mut Vec<f64>,
    fix_mask: &[bool],
    anisotropic: bool,
) -> PyResult<OptimizationSummary> {
    let param_count = params.len();
    let free_indices: Vec<usize> = (0..param_count).filter(|idx| !fix_mask[*idx]).collect();

    if free_indices.is_empty() {
        let parameters = transform_parameters(params, anisotropic);
        let (r_squared, rmse) = compute_statistics(model, anisotropic, &dataset, params);
        let diagnostics = OptimizationDiagnostics {
            initial_cost: 0.0,
            final_cost: 0.0,
            gradient_norm: 0.0,
            step_norm: 0.0,
            lambda: 0.0,
            runtime_seconds: 0.0,
            evaluations: 0,
            ratio: if anisotropic && parameters.len() > 3 {
                let minor = parameters[3].max(MIN_RANGE);
                parameters[2] / minor
            } else {
                1.0
            },
            penalty_hits: 0,
            constraint_messages: Vec::new(),
            isotropic_r2: None,
            isotropic_rmse: None,
        };
        return Ok(OptimizationSummary {
            parameters,
            r_squared,
            rmse,
            converged: true,
            iterations: 0,
            message: String::from("All parameters fixed; skipped optimization"),
            status: OptimizationStatus::Succeeded,
            parameter_std: vec![0.0; param_count],
            diagnostics,
            trace: Vec::new(),
            warnings: Vec::new(),
        });
    }

    let timeout = compute_timeout(dataset.distances.len());
    let start_time = Instant::now();

    let mut lambda = 1e-3;
    let mut iterations = 0usize;
    let mut status = OptimizationStatus::MaxIterations;
    let mut params_vec = params.clone();
    let mut best_params = params_vec.clone();
    let mut best_cost = f64::INFINITY;
    let mut best_jtj = vec![vec![0.0f64; param_count]; param_count];
    let mut best_jtr = vec![0.0f64; param_count];
    let mut trace: Vec<IterationTraceRecord> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut gradient_norm = f64::INFINITY;
    let mut evaluations_total = 0usize;
    let mut stagnation_counter = 0usize;

    let mut diagnostics = OptimizationDiagnostics {
        initial_cost: f64::NAN,
        final_cost: f64::NAN,
        gradient_norm: f64::NAN,
        step_norm: f64::NAN,
        lambda,
        runtime_seconds: 0.0,
        evaluations: 0,
        ratio: 1.0,
        penalty_hits: 0,
        constraint_messages: Vec::new(),
        isotropic_r2: None,
        isotropic_rmse: None,
    };

    let mut evaluation_cache = EvaluationCache::new(param_count, anisotropic);
    let mut adaptive = AdaptivePrecision::new(dataset.distances.len());

    while iterations < MAX_ITERATIONS {
        if start_time.elapsed() > timeout {
            status = OptimizationStatus::Timeout;
            break;
        }

        iterations += 1;
        let slice = adaptive.slice_for(iterations);
        let (cost, jtj, jtr, current_grad_norm, eval_count) = evaluation_cache
            .evaluate(model, anisotropic, &dataset, &params_vec, slice)?;
        if iterations == 1 {
            diagnostics.initial_cost = cost;
        }
        evaluations_total += eval_count;
        diagnostics.lambda = lambda;
        gradient_norm = current_grad_norm;

        if !cost.is_finite() {
            status = OptimizationStatus::InvalidParameters;
            warnings.push(String::from("Encountered non-finite cost during optimisation"));
            break;
        }

        let (matrix, rhs) = assemble_linear_system(&jtj, &jtr, &free_indices, lambda);
        let Some(delta) = solve_linear_system(matrix, rhs) else {
            lambda = (lambda * 10.0).min(MAX_LAMBDA);
            if lambda >= MAX_LAMBDA {
                status = OptimizationStatus::SingularMatrix;
                warnings.push(String::from("Failed to solve normal equations"));
                break;
            }
            continue;
        };

        let step_norm = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
        let mut candidate = params_vec.clone();
        for (offset, &param_idx) in free_indices.iter().enumerate() {
            candidate[param_idx] += delta[offset];
        }
        enforce_bounds(&mut candidate, anisotropic);

        let candidate_slice = if adaptive.using_full { AdaptiveSlice::Full } else { slice };
        let (candidate_cost, candidate_jtj, candidate_jtr, candidate_grad_norm, eval_count_candidate) =
            evaluation_cache.evaluate(model, anisotropic, &dataset, &candidate, candidate_slice)?;
        evaluations_total += eval_count_candidate;

        if !candidate_cost.is_finite() {
            lambda = (lambda * 6.0).min(MAX_LAMBDA);
            warnings.push(String::from("Rejected update with non-finite candidate cost"));
            if lambda >= MAX_LAMBDA {
                status = OptimizationStatus::SingularMatrix;
                break;
            }
            continue;
        }

        let improvement = cost - candidate_cost;
        trace.push(IterationTraceRecord {
            iteration: iterations,
            cost,
            lambda,
            step_norm,
            gradient_norm,
        });
        if trace.len() > TRACE_LIMIT {
            trace.remove(0);
        }

        if improvement > 0.0 {
            params_vec = candidate;
            gradient_norm = candidate_grad_norm;
            best_cost = candidate_cost;
            best_params = params_vec.clone();
            best_jtj = candidate_jtj.clone();
            best_jtr = candidate_jtr.clone();
            diagnostics.step_norm = step_norm;
            stagnation_counter = if improvement < COST_TOL { stagnation_counter + 1 } else { 0 };

            lambda = (lambda * 0.3).max(1e-9);

            if adaptive.should_promote(improvement, step_norm) {
                adaptive.promote();
            }

            if step_norm < STEP_TOL && improvement.abs() < COST_TOL && gradient_norm < GRADIENT_TOL {
                status = OptimizationStatus::Succeeded;
                break;
            }
        } else {
            lambda = (lambda * 4.0).min(MAX_LAMBDA);
            stagnation_counter += 1;
            if adaptive.should_promote(improvement, step_norm) {
                adaptive.promote();
            }
            if lambda >= MAX_LAMBDA {
                status = OptimizationStatus::SingularMatrix;
                break;
            }
        }

        if gradient_norm < GRADIENT_TOL && step_norm < STEP_TOL {
            status = OptimizationStatus::Succeeded;
            break;
        }
        if stagnation_counter >= 5 {
            adaptive.promote();
        }
    }

    if status == OptimizationStatus::MaxIterations && iterations >= MAX_ITERATIONS {
        warnings.push(format!("Reached maximum iterations ({})", MAX_ITERATIONS));
    }

    let final_params = if best_cost.is_finite() {
        best_params.clone()
    } else {
        params_vec.clone()
    };

    let final_cost = if best_cost.is_finite() { best_cost } else { diagnostics.initial_cost };

    diagnostics.final_cost = final_cost;
    diagnostics.gradient_norm = gradient_norm;
    diagnostics.runtime_seconds = start_time.elapsed().as_secs_f64();
    diagnostics.evaluations = evaluations_total;

    if anisotropic {
        let ctx = AnisotropicIterationContext::from_params(&final_params);
        diagnostics.ratio = ctx.ratio();
        if diagnostics.ratio.is_infinite() || diagnostics.ratio > MAX_RATIO_REASONABLE {
            warnings.push(format!("Anisotropy ratio {:.2} exceeds recommended limits", diagnostics.ratio));
        }
    } else {
        diagnostics.ratio = 1.0;
    }

    let transformed = transform_parameters(&final_params, anisotropic);
    let (r_squared, rmse) = compute_statistics(model, anisotropic, &dataset, &final_params);

    if anisotropic {
        let (iso_r2, iso_rmse) = compute_isotropic_baseline(model, &dataset, &final_params);
        diagnostics.isotropic_r2 = Some(iso_r2);
        diagnostics.isotropic_rmse = Some(iso_rmse);
        if iso_r2 > r_squared + 1e-6 {
            warnings.push(String::from("Isotropic baseline achieved higher R^2"));
        }
    }

    let parameter_std = compute_parameter_std(&best_jtj, anisotropic, &final_params);

    Ok(OptimizationSummary {
        parameters: transformed,
        r_squared,
        rmse,
        converged: status.is_success(),
        iterations,
        message: format!("{} (lambda={:.3e})", status.as_str(), lambda),
        status,
        parameter_std,
        diagnostics,
        trace,
        warnings,
    })
}
fn assemble_linear_system(
    jtj: &[Vec<f64>],
    jtr: &[f64],
    free_indices: &[usize],
    lambda: f64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let dim = free_indices.len();
    let mut matrix = vec![vec![0.0f64; dim]; dim];
    let mut rhs = vec![0.0f64; dim];

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

fn enforce_bounds(params: &mut [f64], anisotropic: bool) {
    if params.is_empty() {
        return;
    }
    if params[0] < 0.0 {
        params[0] = 0.0;
    }
    if params.len() >= 2 {
        if params[1] <= params[0] + MIN_SILL_GAP {
            params[1] = params[0] + MIN_SILL_GAP;
        }
    }
    if anisotropic {
        // No clamp required for log-ranges; ensure finite angle
        if !params[4].is_finite() {
            params[4] = 0.0;
        }
    } else if params.len() >= 3 {
        if params[2] < MIN_RANGE {
            params[2] = MIN_RANGE;
        }
    }
}

fn compute_statistics(
    model: VariogramModel,
    anisotropic: bool,
    dataset: &Dataset,
    params: &[f64],
) -> (f64, f64) {
    let n = dataset.distances.len();
    if n == 0 {
        return (1.0, 0.0);
    }

    let directions_opt = dataset.directions.as_ref().map(|v| v.as_slice());

    let mut effective_weights = Vec::with_capacity(n);
    let mut total_weight = 0.0;

    for &w in &dataset.weights {
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

    for (
        i,
        ((&distance, &gamma_obs), &weight),
    ) in dataset
        .distances
        .iter()
        .zip(dataset.gamma.iter())
        .zip(effective_weights.iter())
        .enumerate()
    {
        if weight <= 0.0 {
            continue;
        }
        let direction = directions_opt.and_then(|dirs| Some(dirs[i]));
        let prediction = predict_value(model, anisotropic, params, distance, direction).unwrap_or(gamma_obs);
        let diff = prediction - gamma_obs;
        weighted_residual_sum += weight * diff * diff;
        weighted_gamma_sum += weight * gamma_obs;
    }

    let mean = if total_weight > 0.0 {
        weighted_gamma_sum / total_weight
    } else {
        0.0
    };

    let mut ss_tot = 0.0;
    for (&gamma_obs, &weight) in dataset.gamma.iter().zip(effective_weights.iter()) {
        let diff = gamma_obs - mean;
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

fn evaluate_sample(
    model: VariogramModel,
    anisotropic: bool,
    params: &[f64],
    distance: f64,
    direction: Option<f64>,
) -> PyResult<(f64, Vec<f64>)> {
    if anisotropic {
        let nugget = params[0];
        let sill = params[1];
        let log_range_major = params[2];
        let log_range_minor = params[3];
        let angle_raw = params[4];

        let range_major = log_range_major.exp().max(MIN_RANGE);
        let range_minor = log_range_minor.exp().max(MIN_RANGE);
        let angle = normalize_angle(angle_raw);

        let direction = direction.ok_or_else(|| {
            PyValueError::new_err("direction information required for anisotropic evaluation")
        })?;

        let delta = direction - angle;
        let cos_delta = delta.cos();
        let sin_delta = delta.sin();

        let dx_rot = distance * cos_delta;
        let dy_rot = distance * sin_delta;

        let a = dx_rot / range_major;
        let b = dy_rot / range_minor;
        let h = (a * a + b * b).sqrt();

        let (value, d_nugget, d_sill, d_dh) = model_value_and_derivatives(model, nugget, sill, h);
        let mut grad = vec![0.0f64; 5];
        grad[0] = d_nugget;
        grad[1] = d_sill;

        let mut dh_dlog_range_major = 0.0;
        let mut dh_dlog_range_minor = 0.0;
        let mut dh_dangle = 0.0;

        if h > 0.0 {
            let range_major_sq = range_major * range_major;
            let range_minor_sq = range_minor * range_minor;
            dh_dlog_range_major = -(dx_rot * dx_rot) / (h * range_major_sq);
            dh_dlog_range_minor = -(dy_rot * dy_rot) / (h * range_minor_sq);
            dh_dangle = (dx_rot * dy_rot * (1.0 / range_major_sq - 1.0 / range_minor_sq)) / h;
        }

        let d_gamma_d_h = d_dh;

        grad[2] = d_gamma_d_h * dh_dlog_range_major;
        grad[3] = d_gamma_d_h * dh_dlog_range_minor;
        grad[4] = d_gamma_d_h * dh_dangle;

        Ok((value, grad))
    } else {
        let nugget = params[0];
        let sill = params[1];
        let range = params[2].max(MIN_RANGE);
        let h = distance / range;
        let (value, d_nugget, d_sill, d_dh) = model_value_and_derivatives(model, nugget, sill, h);
        let d_h_d_range = -distance / (range * range);
        let grad = vec![
            d_nugget,
            d_sill,
            d_dh * d_h_d_range,
        ];
        Ok((value, grad))
    }
}

fn predict_value(
    model: VariogramModel,
    anisotropic: bool,
    params: &[f64],
    distance: f64,
    direction: Option<f64>,
) -> PyResult<f64> {
    let (value, _) = evaluate_sample(model, anisotropic, params, distance, direction)?;
    Ok(value)
}

fn model_value_and_derivatives(
    model: VariogramModel,
    nugget: f64,
    sill: f64,
    h: f64,
) -> (f64, f64, f64, f64) {
    let sill_span = sill - nugget;
    match model {
        VariogramModel::Exponential => {
            let exp_term = (-h).exp();
            let value = nugget + sill_span * (1.0 - exp_term);
            let d_nugget = exp_term;
            let d_sill = 1.0 - exp_term;
            let d_dh = sill_span * exp_term;
            (value, d_nugget, d_sill, d_dh)
        }
        VariogramModel::Spherical => {
            if h >= 1.0 {
                let value = sill;
                (value, 0.0, 1.0, 0.0)
            } else {
                let value = nugget + sill_span * (1.5 * h - 0.5 * h * h * h);
                let d_nugget = 1.0 - (1.5 * h - 0.5 * h * h * h);
                let d_sill = 1.5 * h - 0.5 * h * h * h;
                let d_dh = sill_span * (1.5 - 1.5 * h * h);
                (value, d_nugget, d_sill, d_dh)
            }
        }
        VariogramModel::Gaussian => {
            let exp_term = (-h * h).exp();
            let value = nugget + sill_span * (1.0 - exp_term);
            let d_nugget = exp_term;
            let d_sill = 1.0 - exp_term;
            let d_dh = 2.0 * sill_span * h * exp_term;
            (value, d_nugget, d_sill, d_dh)
        }
    }
}

fn transform_parameters(raw: &[f64], anisotropic: bool) -> Vec<f64> {
    if anisotropic {
        vec![
            raw[0],
            raw[1],
            raw[2].exp(),
            raw[3].exp(),
            normalize_angle(raw[4]),
        ]
    } else {
        raw.to_vec()
    }
}

fn sanitize_initial_params(params: &mut [f64], fix_mask: &[bool], anisotropic: bool) -> PyResult<()> {
    if params.len() != fix_mask.len() {
        return Err(PyValueError::new_err("fix_mask length must match initial_params length"));
    }

    if params.is_empty() {
        return Err(PyValueError::new_err("initial_params cannot be empty"));
    }

    if fix_mask[0] && params[0] < 0.0 {
        return Err(PyValueError::new_err("Cannot fix nugget below zero"));
    }
    if params[0] < 0.0 {
        params[0] = 0.0;
    }

    if !params[1].is_finite() {
        return Err(PyValueError::new_err("Sill parameter must be finite"));
    }
    if params[1] <= params[0] + MIN_SILL_GAP {
        params[1] = params[0] + MIN_SILL_GAP;
    }

    if anisotropic {
        if params.len() != 5 {
            return Err(PyValueError::new_err(
                "Anisotropic variogram fitting requires five parameters",
            ));
        }
        if params[2] <= MIN_RANGE {
            params[2] = MIN_RANGE;
        }
        if params[3] <= MIN_RANGE {
            params[3] = MIN_RANGE;
        }
        if !params[4].is_finite() {
            params[4] = 0.0;
        }
    } else {
        if params.len() != 3 {
            return Err(PyValueError::new_err(
                "Isotropic variogram fitting requires three parameters",
            ));
        }
        if params[2] <= MIN_RANGE {
            params[2] = MIN_RANGE;
        }
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

fn normalize_angle(angle: f64) -> f64 {
    let mut normalized = angle % PI;
    if normalized < 0.0 {
        normalized += PI;
    }
    normalized
}
