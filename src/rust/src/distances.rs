use ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "sparse")]
use numpy::PyArray1;
#[cfg(feature = "sparse")]
use pyo3::types::PyTuple;

#[cfg(feature = "sparse")]
const DEFAULT_SPARSE_SWITCH: f64 = 0.3;
#[cfg(feature = "simd")]
const SIMD_LANES: usize = 8;

#[pyfunction(signature = (coords, max_distance=None, chunk_size=None, use_simd=None))]
pub fn euclidean_distances<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    max_distance: Option<f64>,
    chunk_size: Option<usize>,
    use_simd: Option<bool>,
) -> PyResult<Py<PyAny>> {
    let coords = coords.as_array();
    let n = coords.nrows();

    if n == 0 {
        let empty = Array2::<f64>::zeros((0, 0));
        return Ok(empty.into_pyarray(py).into_py(py));
    }

    let chunk = select_chunk_size(chunk_size, n);
    let simd_enabled = use_simd.unwrap_or(cfg!(feature = "simd"));

    #[cfg(feature = "sparse")]
    if let Some(threshold) = max_distance {
        if threshold <= 0.0 {
            return Err(PyValueError::new_err("max_distance must be positive"));
        }
        let sparse = compute_sparse(&coords, threshold, chunk, simd_enabled);
        let ratio = sparse.nnz as f64 / ((n * n) as f64);
        if ratio <= DEFAULT_SPARSE_SWITCH {
            let data = PyArray1::from_vec(py, sparse.data);
            let rows = PyArray1::from_vec(py, sparse.rows);
            let cols = PyArray1::from_vec(py, sparse.cols);
            let shape = PyTuple::new(py, [n.into_py(py), n.into_py(py)]);
            let result = PyTuple::new(
                py,
                [
                    data.into_py(py),
                    rows.into_py(py),
                    cols.into_py(py),
                    shape.into_py(py),
                ],
            );
            return Ok(result.into());
        }
        // otherwise fall back to dense computation below
    }

    #[cfg(not(feature = "sparse"))]
    if max_distance.is_some() {
        return Err(PyValueError::new_err(
            "sparse distance output requires building with the 'sparse' cargo feature",
        ));
    }

    let flat = compute_dense(&coords, chunk, simd_enabled);
    let matrix = Array2::from_shape_vec((n, n), flat)
        .map_err(|_| PyValueError::new_err("failed to build distance matrix"))?;
    Ok(matrix.into_pyarray(py).into_py(py))
}

#[pyfunction(signature = (coords, radius, chunk_size=None, use_simd=None))]
pub fn haversine_distances<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f64>,
    radius: Option<f64>,
    chunk_size: Option<usize>,
    use_simd: Option<bool>,
) -> PyResult<&'py PyArray2<f64>> {
    let coords = coords.as_array();
    let n = coords.nrows();
    if n == 0 {
        return Ok(Array2::<f64>::zeros((0, 0)).into_pyarray(py));
    }

    let r = radius.unwrap_or(6_371.0);
    let chunk = select_chunk_size(chunk_size, n);
    let simd_enabled = use_simd.unwrap_or(cfg!(feature = "simd"));

    let mut flat = vec![0.0f64; n * n];

    #[cfg(feature = "parallel")]
    {
        flat.par_chunks_mut(n)
            .enumerate()
            .with_min_len(chunk)
            .for_each(|(i, row)| {
                row[i] = 0.0;
                let pi = coords.row(i);
                for j in (i + 1)..n {
                    let pj = coords.row(j);
                    let dist = haversine_pair(pi, pj, r, simd_enabled);
                    row[j] = dist;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (i, row) in flat.chunks_mut(n).enumerate() {
            row[i] = 0.0;
            let pi = coords.row(i);
            for j in (i + 1)..n {
                let pj = coords.row(j);
                let dist = haversine_pair(pi, pj, r, simd_enabled);
                row[j] = dist;
            }
        }
    }

    mirror_upper_to_lower(n, &mut flat);

    let matrix = Array2::from_shape_vec((n, n), flat)
        .map_err(|_| PyValueError::new_err("failed to build haversine matrix"))?;
    Ok(matrix.into_pyarray(py))
}

#[cfg(feature = "sparse")]
struct SparseResult {
    data: Vec<f64>,
    rows: Vec<usize>,
    cols: Vec<usize>,
    nnz: usize,
}

#[cfg(feature = "sparse")]
fn compute_sparse(
    coords: &ArrayView2<f64>,
    threshold: f64,
    chunk: usize,
    use_simd: bool,
) -> SparseResult {
    let n = coords.nrows();

    #[cfg(feature = "parallel")]
    let partials: Vec<Vec<(usize, usize, f64)>> = (0..n)
        .into_par_iter()
        .with_min_len(chunk)
        .map(|i| build_sparse_row(coords, i, threshold, use_simd))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let partials: Vec<Vec<(usize, usize, f64)>> = (0..n)
        .map(|i| build_sparse_row(coords, i, threshold, use_simd))
        .collect();

    let nnz = partials.iter().map(|p| p.len()).sum();
    let mut data = Vec::with_capacity(nnz);
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);

    for part in partials {
        for (r, c, v) in part {
            rows.push(r);
            cols.push(c);
            data.push(v);
        }
    }

    SparseResult {
        data,
        rows,
        cols,
        nnz,
    }
}

#[cfg(feature = "sparse")]
fn build_sparse_row(
    coords: &ArrayView2<f64>,
    i: usize,
    threshold: f64,
    use_simd: bool,
) -> Vec<(usize, usize, f64)> {
    let n = coords.nrows();
    let mut local = Vec::new();
    let pi = coords.row(i);
    local.push((i, i, 0.0));

    for j in (i + 1)..n {
        let pj = coords.row(j);
        let dist = distance_kernel(pi, pj, use_simd);
        if dist <= threshold {
            local.push((i, j, dist));
            local.push((j, i, dist));
        }
    }

    local
}

fn compute_dense(coords: &ArrayView2<f64>, chunk: usize, use_simd: bool) -> Vec<f64> {
    let n = coords.nrows();
    let mut flat = vec![0.0f64; n * n];

    #[cfg(feature = "parallel")]
    {
        flat.par_chunks_mut(n)
            .enumerate()
            .with_min_len(chunk)
            .for_each(|(i, row)| {
                row[i] = 0.0;
                let pi = coords.row(i);
                for j in (i + 1)..n {
                    let pj = coords.row(j);
                    let dist = distance_kernel(pi, pj, use_simd);
                    row[j] = dist;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (i, row) in flat.chunks_mut(n).enumerate() {
            row[i] = 0.0;
            let pi = coords.row(i);
            for j in (i + 1)..n {
                let pj = coords.row(j);
                let dist = distance_kernel(pi, pj, use_simd);
                row[j] = dist;
            }
        }
    }

    mirror_upper_to_lower(n, &mut flat);
    flat
}

fn select_chunk_size(user: Option<usize>, n: usize) -> usize {
    if let Some(custom) = user {
        return custom.max(1);
    }
    match n {
        0..=256 => 8,
        257..=1024 => 16,
        1025..=4096 => 32,
        _ => 64,
    }
}

fn distance_kernel(p1: ArrayView1<f64>, p2: ArrayView1<f64>, use_simd: bool) -> f64 {
    #[cfg(feature = "simd")]
    {
        if use_simd {
            return simd_distance(p1, p2);
        }
    }
    let _ = use_simd;
    scalar_distance(p1, p2)
}

fn scalar_distance(p1: ArrayView1<f64>, p2: ArrayView1<f64>) -> f64 {
    let mut acc = 0.0;
    for k in 0..p1.len() {
        let diff = p1[k] - p2[k];
        acc += diff * diff;
    }
    acc.sqrt()
}

#[cfg(feature = "simd")]
fn simd_distance(p1: ArrayView1<f64>, p2: ArrayView1<f64>) -> f64 {
    use std::simd::{Simd, SimdFloat};

    let slice_a = p1.as_slice().expect("contiguous slice");
    let slice_b = p2.as_slice().expect("contiguous slice");

    let len = slice_a.len();
    let mut simd_acc = Simd::<f64, SIMD_LANES>::splat(0.0);
    let mut idx = 0;
    while idx + SIMD_LANES <= len {
        let a = Simd::from_slice(&slice_a[idx..idx + SIMD_LANES]);
        let b = Simd::from_slice(&slice_b[idx..idx + SIMD_LANES]);
        let diff = a - b;
        simd_acc += diff * diff;
        idx += SIMD_LANES;
    }

    let mut total = simd_acc.reduce_sum();
    while idx < len {
        let diff = slice_a[idx] - slice_b[idx];
        total += diff * diff;
        idx += 1;
    }

    total.sqrt()
}

fn haversine_pair(p1: ArrayView1<f64>, p2: ArrayView1<f64>, radius: f64, use_simd: bool) -> f64 {
    #[cfg(feature = "simd")]
    {
        if use_simd {
            return haversine_simd(p1, p2, radius);
        }
    }
    #[cfg(not(feature = "simd"))]
    let _ = use_simd;
    haversine_scalar(p1, p2, radius)
}

fn haversine_scalar(p1: ArrayView1<f64>, p2: ArrayView1<f64>, radius: f64) -> f64 {
    let lat1 = p1[0].to_radians();
    let lon1 = p1[1].to_radians();
    let lat2 = p2[0].to_radians();
    let lon2 = p2[1].to_radians();

    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;

    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    radius * c
}

#[cfg(feature = "simd")]
fn haversine_simd(p1: ArrayView1<f64>, p2: ArrayView1<f64>, radius: f64) -> f64 {
    use std::simd::{Simd, SimdFloat};

    let a = Simd::<f64, 2>::from_array([p1[0].to_radians(), p1[1].to_radians()]);
    let b = Simd::<f64, 2>::from_array([p2[0].to_radians(), p2[1].to_radians()]);
    let delta = b - a;
    let half = delta / Simd::splat(2.0);
    let sin_half = half.sin();
    let sin_sq = sin_half * sin_half;

    let lat1 = a[0];
    let lat2 = b[0];

    let a_term = sin_sq[0] + lat1.cos() * lat2.cos() * sin_sq[1];
    let c = 2.0 * a_term.sqrt().asin();
    radius * c
}

fn mirror_upper_to_lower(n: usize, flat: &mut [f64]) {
    for i in 0..n {
        let row_offset = i * n;
        for j in (i + 1)..n {
            let value = flat[row_offset + j];
            flat[j * n + i] = value;
        }
    }
}
