// src/rust/src/angle_estimation.rs
//! Automatic rotation angle and range estimation for directional variograms.

use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub struct AngleEstimate {
    pub angle_deg: f64,
    pub angle_confidence: (f64, f64),
    pub ratio: f64,
    pub confidence: ConfidenceLevel,
    pub significant: bool,
    pub major_range: f64,
    pub minor_range: f64,
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Clone)]
pub struct Diagnostics {
    pub rmse: f64,
    pub sse: f64,
    pub z_score: f64,
    pub max_range: f64,
    pub min_range: f64,
}

#[derive(Debug, Clone)]
pub struct RangeEstimates {
    pub range_major: f64,
    pub range_minor: f64,
    pub ratio: f64,
    pub range_major_se: f64,
    pub range_minor_se: f64,
    pub range_major_bounds: (f64, f64),
    pub range_minor_bounds: (f64, f64),
    pub sill: f64,
    pub nugget: f64,
    pub quality_score: f64,
    pub sample_size: usize,
}

#[derive(Debug)]
pub struct AngleEstimationError {
    pub message: String,
}

impl AngleEstimationError {
    fn new(msg: impl Into<String>) -> Self {
        Self { message: msg.into() }
    }
}

struct FitRecord {
    phi: f64,
    major: f64,
    minor: f64,
    ratio: f64,
    rmse: f64,
    sse: f64,
}

/// Estimate anisotropic rotation angle from directional ranges.
pub fn estimate_rotation_angle(
    directions_deg: &[f64],
    ranges: &[f64],
    weights: Option<&[f64]>,
    phi_step_deg: f64,
) -> Result<AngleEstimate, AngleEstimationError> {
    assert!(directions_deg.len() == ranges.len(), "directions and ranges must match in length");
    assert!(directions_deg.len() >= 3, "At least three directional samples are required");
    if directions_deg.is_empty() {
        return Err(AngleEstimationError::new("Empty directional input"));
    }

    let mut angles_rad: Vec<f64> = Vec::with_capacity(directions_deg.len());
    let mut cleaned_ranges: Vec<f64> = Vec::with_capacity(ranges.len());
    for (angle, &range) in directions_deg.iter().zip(ranges.iter()) {
        if !range.is_finite() || range <= 0.0 {
            return Err(AngleEstimationError::new("Directional ranges must be positive and finite"));
        }
        angles_rad.push((angle % 180.0f64).to_radians());
        cleaned_ranges.push(range);
    }

    let weight_vec: Vec<f64> = if let Some(w) = weights {
        assert!(w.len() == ranges.len(), "weights must match the number of directions");
        w.iter().map(|val| if *val > 0.0 { *val } else { 1e-12 }).collect()
    } else {
        vec![1.0; ranges.len()]
    };

    let phi_step = phi_step_deg.max(0.1);
    let steps = (180.0 / phi_step).round() as usize;
    assert!(steps >= 10, "Angle search grid is too coarse");

    let mut best: Option<FitRecord> = None;
    let mut records: Vec<FitRecord> = Vec::new();

    for i in 0..=steps {
        let phi = (i as f64) * phi_step.to_radians();
        if let Some(record) = fit_ellipse_for_phi(phi, &angles_rad, &cleaned_ranges, &weight_vec) {
            if best.as_ref().map(|b| record.sse < b.sse).unwrap_or(true) {
                best = Some(record.clone());
            }
            records.push(record);
        }
    }

    let best = match best {
        Some(value) => value,
        None => {
            let mean_angle = angles_rad.iter().copied().sum::<f64>() / angles_rad.len() as f64;
            return Ok(AngleEstimate {
                angle_deg: (mean_angle.to_degrees() + 180.0) % 180.0,
                angle_confidence: (0.0, 180.0),
                ratio: 1.0,
                confidence: ConfidenceLevel::Low,
                significant: false,
                major_range: cleaned_ranges.iter().copied().fold(0.0, f64::max),
                minor_range: cleaned_ranges.iter().copied().fold(f64::INFINITY, f64::min),
                diagnostics: Diagnostics {
                    rmse: 0.0,
                    sse: 0.0,
                    z_score: 0.0,
                    max_range: cleaned_ranges.iter().copied().fold(0.0, f64::max),
                    min_range: cleaned_ranges.iter().copied().fold(f64::INFINITY, f64::min),
                },
            });
        }
    };

    let threshold = best.sse * 1.05 + 1e-12;
    let mut low_phi = best.phi;
    let mut high_phi = best.phi;
    for record in &records {
        if record.sse <= threshold {
            if record.phi < low_phi {
                low_phi = record.phi;
            }
            if record.phi > high_phi {
                high_phi = record.phi;
            }
        }
    }

    let angle_deg = (best.phi.to_degrees() + 180.0) % 180.0;
    let mut conf_low = (low_phi.to_degrees() + 180.0) % 180.0;
    let mut conf_high = (high_phi.to_degrees() + 180.0) % 180.0;
    if conf_low > conf_high {
        std::mem::swap(&mut conf_low, &mut conf_high);
    }

    let max_range = cleaned_ranges.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_range = cleaned_ranges.iter().copied().fold(f64::INFINITY, f64::min);
    let delta = max_range - min_range;
    let mean = cleaned_ranges.iter().sum::<f64>() / cleaned_ranges.len() as f64;
    let variance = cleaned_ranges
        .iter()
        .map(|val| {
            let diff = val - mean;
            diff * diff
        })
        .sum::<f64>()
        / cleaned_ranges.len().saturating_sub(1).max(1) as f64;
    let z_score = delta / (variance.sqrt() + 1e-12);
    let significant = delta > 0.0 && z_score > 1.5;

    let ratio = best.ratio;
    let confidence = if ratio > 1.5 && significant {
        ConfidenceLevel::High
    } else if ratio > 1.2 {
        if significant {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    } else {
        ConfidenceLevel::Low
    };

    Ok(AngleEstimate {
        angle_deg,
        angle_confidence: (conf_low, conf_high),
        ratio,
        confidence,
        significant,
        major_range: best.major,
        minor_range: best.minor,
        diagnostics: Diagnostics {
            rmse: best.rmse,
            sse: best.sse,
            z_score,
            max_range,
            min_range,
        },
    })
}

/// Estimate range, sill, and nugget parameters from directional statistics.
pub fn estimate_range_parameters(
    ranges: &[f64],
    sill_samples: &[f64],
    nugget_candidates: &[f64],
) -> Result<RangeEstimates, AngleEstimationError> {
    assert!(!ranges.is_empty(), "Range samples may not be empty");
    assert!(ranges.len() >= 2, "At least two range samples required");

    let mut range_major = f64::NEG_INFINITY;
    let mut range_minor = f64::INFINITY;
    for &r in ranges {
        if !r.is_finite() || r <= 0.0 {
            return Err(AngleEstimationError::new("Range samples must be positive and finite"));
        }
        if r > range_major {
            range_major = r;
        }
        if r < range_minor {
            range_minor = r;
        }
    }
    if range_minor <= 0.0 {
        return Err(AngleEstimationError::new("Minimum range must exceed zero"));
    }

    let ratio = range_major / range_minor;
    assert!(ratio >= 1.0, "Range ratio must be at least 1");

    let sample_size = ranges.len();
    let mean_range = ranges.iter().sum::<f64>() / sample_size as f64;
    let variance = ranges
        .iter()
        .map(|val| {
            let diff = val - mean_range;
            diff * diff
        })
        .sum::<f64>()
        / sample_size.saturating_sub(1).max(1) as f64;
    let se = (variance.sqrt()) / (sample_size as f64).sqrt();

    let sill = if sill_samples.is_empty() {
        range_major
    } else {
        let mut stable_samples: Vec<f64> = sill_samples
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect();
        if stable_samples.is_empty() {
            range_major
        } else {
            stable_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let take = (stable_samples.len() as f64 * 0.3).ceil() as usize;
            let start = stable_samples.len().saturating_sub(take.max(1));
            stable_samples[start..].iter().sum::<f64>() / (stable_samples.len() - start) as f64
        }
    };

    let mut nugget = if nugget_candidates.is_empty() {
        0.0
    } else {
        nugget_candidates
            .iter()
            .filter(|v| v.is_finite() && **v >= 0.0)
            .fold(f64::INFINITY, |acc, &val| if val < acc { val } else { acc })
    };
    if !nugget.is_finite() {
        nugget = 0.0;
    }
    if nugget > sill {
        nugget = (sill * 0.99).max(0.0);
    }

    let range_major_bounds = (range_major - se, range_major + se);
    let range_minor_bounds = (range_minor - se, range_minor + se);
    let quality_score = ratio / (1.0 + se.max(1e-12));

    Ok(RangeEstimates {
        range_major,
        range_minor,
        ratio,
        range_major_se: se,
        range_minor_se: se,
        range_major_bounds,
        range_minor_bounds,
        sill,
        nugget,
        quality_score,
        sample_size,
    })
}

fn fit_ellipse_for_phi(
    phi: f64,
    angles_rad: &[f64],
    ranges: &[f64],
    weights: &[f64],
) -> Option<FitRecord> {
    let mut xtwx00 = 0.0;
    let mut xtwx01 = 0.0;
    let mut xtwx11 = 0.0;
    let mut xtwy0 = 0.0;
    let mut xtwy1 = 0.0;
    let mut weight_sum = 0.0;

    for ((&angle, &range), &weight) in angles_rad.iter().zip(ranges.iter()).zip(weights.iter()) {
        assert!(weight >= 0.0, "Weights must be non-negative");
        let shifted = angle - phi;
        let cos_t = shifted.cos();
        let sin_t = shifted.sin();
        let x1 = cos_t * cos_t;
        let x2 = sin_t * sin_t;
        let y = 1.0 / (range * range);
        xtwx00 += weight * x1 * x1;
        xtwx01 += weight * x1 * x2;
        xtwx11 += weight * x2 * x2;
        xtwy0 += weight * x1 * y;
        xtwy1 += weight * x2 * y;
        weight_sum += weight;
    }

    if weight_sum <= 1e-12 {
        return None;
    }

    let det = xtwx00 * xtwx11 - xtwx01 * xtwx01;
    if det.abs() < 1e-12 {
        return None;
    }

    let alpha = (xtwy0 * xtwx11 - xtwx01 * xtwy1) / det;
    let beta = (xtwx00 * xtwy1 - xtwx01 * xtwy0) / det;
    if !(alpha.is_finite() && beta.is_finite()) || alpha <= 0.0 || beta <= 0.0 {
        return None;
    }

    let mut sse = 0.0;
    for ((&angle, &range), &weight) in angles_rad.iter().zip(ranges.iter()).zip(weights.iter()) {
        let shifted = angle - phi;
        let cos_t = shifted.cos();
        let sin_t = shifted.sin();
        let inv_sq_pred = alpha * cos_t * cos_t + beta * sin_t * sin_t;
        if inv_sq_pred <= 0.0 {
            return None;
        }
        let predicted = 1.0 / inv_sq_pred.sqrt();
        let diff = predicted - range;
        sse += weight * diff * diff;
    }

    let rmse = (sse / weight_sum).sqrt();
    let major = 1.0 / alpha.sqrt();
    let minor = 1.0 / beta.sqrt();
    let ratio = major / minor.max(1e-12);

    Some(FitRecord {
        phi,
        major,
        minor,
        ratio,
        rmse,
        sse,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_ranges() {
        let ranges = [0.45, 0.41, 0.23, 0.21, 0.19, 0.43];
        let sill_samples = [1.02, 1.01, 0.99, 1.04, 1.03, 1.00];
        let nugget_candidates = [0.05, 0.06, 0.04, 0.05, 0.045, 0.055];

        let estimate = estimate_range_parameters(&ranges, &sill_samples, &nugget_candidates).unwrap();
        assert!((estimate.range_major - 0.45).abs() < 1e-6);
        assert!((estimate.range_minor - 0.19).abs() < 1e-6);
        assert!(estimate.ratio > 2.0);
        assert!(estimate.sill >= estimate.nugget);
    }
}
