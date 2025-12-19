//! Scipy special functions.
//!
//! This module mirrors the `scipy.special` functionality from JAX,
//! providing special mathematical functions commonly used in machine learning.

use crate::Array;

/// Error function: erf(x) = 2/sqrt(π) * ∫[0,x] exp(-t²) dt.
///
/// The error function is the integral of the Gaussian distribution.
/// It is commonly used in statistics and probability.
///
/// # Examples
///
/// ```
/// # use jax_rs::{scipy, Array, Shape};
/// let x = Array::from_vec(vec![0.0, 1.0, -1.0], Shape::new(vec![3]));
/// let y = scipy::erf(&x);
/// // erf(0) = 0, erf(1) ≈ 0.8427, erf(-1) ≈ -0.8427
/// assert!((y.to_vec()[0] - 0.0).abs() < 1e-5);
/// assert!((y.to_vec()[1] - 0.8427).abs() < 0.01);
/// ```
pub fn erf(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data.iter().map(|&v| erf_scalar(v)).collect();
    Array::from_vec(result, x.shape().clone())
}

/// Complementary error function: erfc(x) = 1 - erf(x).
///
/// # Examples
///
/// ```
/// # use jax_rs::{scipy, Array, Shape};
/// let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
/// let y = scipy::erfc(&x);
/// // erfc(0) = 1
/// assert!((y.to_vec()[0] - 1.0).abs() < 1e-5);
/// ```
pub fn erfc(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data.iter().map(|&v| 1.0 - erf_scalar(v)).collect();
    Array::from_vec(result, x.shape().clone())
}

/// Logit function: logit(p) = log(p / (1-p)).
///
/// The inverse of the logistic (sigmoid) function.
///
/// # Examples
///
/// ```
/// # use jax_rs::{scipy, Array, Shape};
/// let p = Array::from_vec(vec![0.5, 0.7, 0.3], Shape::new(vec![3]));
/// let y = scipy::logit(&p);
/// // logit(0.5) = 0
/// assert!((y.to_vec()[0] - 0.0).abs() < 1e-5);
/// ```
pub fn logit(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data
        .iter()
        .map(|&p| {
            assert!(p > 0.0 && p < 1.0, "logit requires 0 < p < 1");
            (p / (1.0 - p)).ln()
        })
        .collect();
    Array::from_vec(result, x.shape().clone())
}

/// Expit function (logistic sigmoid): expit(x) = 1 / (1 + exp(-x)).
///
/// The inverse of the logit function.
///
/// # Examples
///
/// ```
/// # use jax_rs::{scipy, Array, Shape};
/// let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
/// let y = scipy::expit(&x);
/// // expit(0) = 0.5
/// assert!((y.to_vec()[0] - 0.5).abs() < 1e-5);
/// ```
pub fn expit(x: &Array) -> Array {
    crate::nn::sigmoid(x)
}

// Helper function: Compute erf for a scalar using Abramowitz and Stegun approximation
fn erf_scalar(x: f32) -> f32 {
    // Constants for the approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    // Save the sign of x
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Abramowitz and Stegun formula 7.1.26
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;

    #[test]
    fn test_erf_zero() {
        let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
        let y = erf(&x);
        assert!((y.to_vec()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_erf_values() {
        let x = Array::from_vec(vec![1.0, -1.0, 2.0], Shape::new(vec![3]));
        let y = erf(&x);
        let vals = y.to_vec();

        // erf(1) ≈ 0.8427
        assert!((vals[0] - 0.8427).abs() < 0.01);
        // erf(-1) ≈ -0.8427
        assert!((vals[1] + 0.8427).abs() < 0.01);
        // erf(2) ≈ 0.9953
        assert!((vals[2] - 0.9953).abs() < 0.01);
    }

    #[test]
    fn test_erfc() {
        let x = Array::from_vec(vec![0.0, 1.0], Shape::new(vec![2]));
        let y = erfc(&x);
        let vals = y.to_vec();

        // erfc(0) = 1
        assert!((vals[0] - 1.0).abs() < 1e-5);
        // erfc(1) ≈ 0.1573
        assert!((vals[1] - 0.1573).abs() < 0.01);
    }

    #[test]
    fn test_logit() {
        let p = Array::from_vec(vec![0.5, 0.731], Shape::new(vec![2]));
        let y = logit(&p);
        let vals = y.to_vec();

        // logit(0.5) = 0
        assert!((vals[0] - 0.0).abs() < 1e-5);
        // logit(0.731) ≈ 1.0
        assert!((vals[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_expit() {
        let x = Array::from_vec(vec![0.0, 1.0, -1.0], Shape::new(vec![3]));
        let y = expit(&x);
        let vals = y.to_vec();

        // expit(0) = 0.5
        assert!((vals[0] - 0.5).abs() < 1e-5);
        // expit(1) ≈ 0.7311
        assert!((vals[1] - 0.7311).abs() < 0.01);
        // expit(-1) ≈ 0.2689
        assert!((vals[2] - 0.2689).abs() < 0.01);
    }

    #[test]
    fn test_logit_expit_inverse() {
        let p = Array::from_vec(vec![0.3, 0.5, 0.7], Shape::new(vec![3]));
        let x = logit(&p);
        let p_reconstructed = expit(&x);

        for (i, &original) in p.to_vec().iter().enumerate() {
            assert!((p_reconstructed.to_vec()[i] - original).abs() < 1e-5);
        }
    }
}
