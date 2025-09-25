// src/rust/src/utils.rs
use ndarray::ArrayView1;

pub fn euclidean_distance_single(p1: ArrayView1<f64>, p2: ArrayView1<f64>) -> f64 {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}
