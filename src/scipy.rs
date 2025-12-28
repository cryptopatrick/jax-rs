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

/// Gamma function: Γ(x) = ∫[0,∞] t^(x-1) * exp(-t) dt.
///
/// The gamma function extends the factorial function to real and complex numbers.
/// For positive integers: Γ(n) = (n-1)!
///
/// Uses Lanczos approximation for numerical computation.
///
/// # Examples
///
/// ```
/// # use jax_rs::{scipy, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let y = scipy::gamma(&x);
/// // Γ(1) = 0! = 1, Γ(2) = 1! = 1, Γ(3) = 2! = 2, Γ(4) = 3! = 6
/// let vals = y.to_vec();
/// assert!((vals[0] - 1.0).abs() < 1e-5);
/// assert!((vals[1] - 1.0).abs() < 1e-5);
/// assert!((vals[2] - 2.0).abs() < 1e-4);
/// assert!((vals[3] - 6.0).abs() < 1e-3);
/// ```
pub fn gamma(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data.iter().map(|&v| gamma_scalar(v)).collect();
    Array::from_vec(result, x.shape().clone())
}

/// Log gamma function: lgamma(x) = log(Γ(x)).
///
/// Computes the natural logarithm of the gamma function.
/// More numerically stable than computing log(gamma(x)) for large x.
///
/// # Examples
///
/// ```
/// # use jax_rs::{scipy, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 10.0], Shape::new(vec![3]));
/// let y = scipy::lgamma(&x);
/// // lgamma(1) = log(1) = 0, lgamma(2) = log(1) = 0, lgamma(10) = log(9!) ≈ 12.8018
/// let vals = y.to_vec();
/// assert!((vals[0] - 0.0).abs() < 1e-5);
/// assert!((vals[1] - 0.0).abs() < 1e-5);
/// assert!((vals[2] - 12.8018).abs() < 0.01);
/// ```
pub fn lgamma(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data.iter().map(|&v| lgamma_scalar(v)).collect();
    Array::from_vec(result, x.shape().clone())
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

// Helper function: Compute gamma function using Lanczos approximation
fn gamma_scalar(x: f32) -> f32 {
    // Lanczos approximation coefficients (g = 7, n = 9)
    const G: f32 = 7.0;
    const COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    // Handle special cases
    if x < 0.5 {
        // Use reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
        let pi = std::f32::consts::PI;
        return pi / ((pi * x).sin() * gamma_scalar(1.0 - x));
    }

    // Lanczos approximation
    let x = x - 1.0;
    let mut a = COEFFS[0];
    for i in 1..9 {
        a += COEFFS[i] / ((x + i as f32) as f64);
    }

    let t = x + G + 0.5;
    let sqrt_2pi = (2.0 * std::f32::consts::PI).sqrt();

    sqrt_2pi * a as f32 * t.powf(x + 0.5) * (-t).exp()
}

// Helper function: Compute log gamma function
fn lgamma_scalar(x: f32) -> f32 {
    // Lanczos approximation coefficients (g = 7, n = 9)
    const G: f32 = 7.0;
    const COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    // Handle special cases
    if x < 0.5 {
        // Use reflection formula: log(Γ(x)) = log(π) - log(sin(πx)) - log(Γ(1-x))
        let pi = std::f32::consts::PI;
        return pi.ln() - (pi * x).sin().ln() - lgamma_scalar(1.0 - x);
    }

    // Lanczos approximation for log gamma
    let x = x - 1.0;
    let mut a = COEFFS[0];
    for i in 1..9 {
        a += COEFFS[i] / ((x + i as f32) as f64);
    }

    let t = x + G + 0.5;
    let log_sqrt_2pi = (2.0 * std::f32::consts::PI).sqrt().ln();

    log_sqrt_2pi + (a as f32).ln() + (x + 0.5) * t.ln() - t
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

    #[test]
    fn test_gamma_integers() {
        // Test Γ(n) = (n-1)! for positive integers
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
        let y = gamma(&x);
        let vals = y.to_vec();

        // Γ(1) = 0! = 1
        assert!((vals[0] - 1.0).abs() < 1e-5);
        // Γ(2) = 1! = 1
        assert!((vals[1] - 1.0).abs() < 1e-5);
        // Γ(3) = 2! = 2
        assert!((vals[2] - 2.0).abs() < 1e-4);
        // Γ(4) = 3! = 6
        assert!((vals[3] - 6.0).abs() < 1e-3);
        // Γ(5) = 4! = 24
        assert!((vals[4] - 24.0).abs() < 1e-2);
    }

    #[test]
    fn test_gamma_half() {
        // Test Γ(1/2) = √π
        let x = Array::from_vec(vec![0.5], Shape::new(vec![1]));
        let y = gamma(&x);
        let expected = std::f32::consts::PI.sqrt();
        assert!((y.to_vec()[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_lgamma_integers() {
        // Test lgamma(n) = log((n-1)!)
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 10.0], Shape::new(vec![4]));
        let y = lgamma(&x);
        let vals = y.to_vec();

        // lgamma(1) = log(1) = 0
        assert!((vals[0] - 0.0).abs() < 1e-5);
        // lgamma(2) = log(1) = 0
        assert!((vals[1] - 0.0).abs() < 1e-5);
        // lgamma(3) = log(2) ≈ 0.6931
        assert!((vals[2] - 0.6931).abs() < 1e-3);
        // lgamma(10) = log(9!) ≈ 12.8018
        assert!((vals[3] - 12.8018).abs() < 0.01);
    }

    #[test]
    fn test_lgamma_stability() {
        // Test that lgamma is more stable than log(gamma(x)) for large x
        let x = Array::from_vec(vec![50.0, 100.0], Shape::new(vec![2]));
        let lg = lgamma(&x);
        let vals = lg.to_vec();

        // lgamma(50) ≈ 144.57
        assert!((vals[0] - 144.57).abs() < 0.1);
        // lgamma(100) ≈ 359.13
        assert!((vals[1] - 359.13).abs() < 0.1);
    }

    #[test]
    fn test_gamma_lgamma_consistency() {
        // Test that lgamma(x) = log(gamma(x)) for small values
        let x = Array::from_vec(vec![2.5, 3.5, 4.5], Shape::new(vec![3]));
        let g = gamma(&x);
        let lg = lgamma(&x);

        for i in 0..3 {
            let expected = g.to_vec()[i].ln();
            assert!((lg.to_vec()[i] - expected).abs() < 1e-4);
        }
    }
}
