//! Numerical accuracy tests for jax-rs.
//!
//! These tests validate that operations produce numerically accurate results
//! within specified tolerances, comparing against known reference values.

use jax_rs::{Array, DType, Shape};
use std::f32::consts::PI;

/// Helper function to check if two values are close within tolerance.
fn assert_close(actual: f32, expected: f32, rtol: f32, atol: f32, msg: &str) {
    let diff = (actual - expected).abs();
    let tolerance = atol + rtol * expected.abs();
    assert!(
        diff <= tolerance,
        "{}: expected {}, got {} (diff: {}, tolerance: {})",
        msg, expected, actual, diff, tolerance
    );
}

/// Helper function to check if two arrays are element-wise close.
fn assert_array_close(actual: &Array, expected: &[f32], rtol: f32, atol: f32, msg: &str) {
    let actual_vec = actual.to_vec();
    assert_eq!(
        actual_vec.len(),
        expected.len(),
        "{}: length mismatch",
        msg
    );

    for (i, (&a, &e)) in actual_vec.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let tolerance = atol + rtol * e.abs();
        assert!(
            diff <= tolerance,
            "{} at index {}: expected {}, got {} (diff: {}, tolerance: {})",
            msg, i, e, a, diff, tolerance
        );
    }
}

// =============================================================================
// UNARY OPERATIONS
// =============================================================================

#[test]
fn test_sin_accuracy() {
    let x = Array::from_vec(
        vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0],
        Shape::new(vec![5]),
    );

    let result = x.sin();

    let expected = vec![
        0.0,              // sin(0)
        0.5,              // sin(π/6)
        0.7071067811,     // sin(π/4) = √2/2
        0.8660254037,     // sin(π/3) = √3/2
        1.0,              // sin(π/2)
    ];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "sin");
}

#[test]
fn test_cos_accuracy() {
    let x = Array::from_vec(
        vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0],
        Shape::new(vec![5]),
    );

    let result = x.cos();

    let expected = vec![
        1.0,              // cos(0)
        0.8660254037,     // cos(π/6) = √3/2
        0.7071067811,     // cos(π/4) = √2/2
        0.5,              // cos(π/3)
        0.0,              // cos(π/2)
    ];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "cos");
}

#[test]
fn test_exp_accuracy() {
    let x = Array::from_vec(
        vec![0.0, 1.0, 2.0, -1.0, -2.0],
        Shape::new(vec![5]),
    );

    let result = x.exp();

    let expected = vec![
        1.0,              // e^0
        2.7182818284,     // e^1
        7.3890560989,     // e^2
        0.3678794411,     // e^-1
        0.1353352832,     // e^-2
    ];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "exp");
}

#[test]
fn test_log_accuracy() {
    let x = Array::from_vec(
        vec![1.0, 2.0, 2.7182818284, 10.0, 100.0],
        Shape::new(vec![5]),
    );

    let result = x.log();

    let expected = vec![
        0.0,              // ln(1)
        0.6931471805,     // ln(2)
        1.0,              // ln(e)
        2.3025850929,     // ln(10)
        4.6051701859,     // ln(100)
    ];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "log");
}

#[test]
fn test_sqrt_accuracy() {
    let x = Array::from_vec(
        vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0],
        Shape::new(vec![6]),
    );

    let result = x.sqrt();

    let expected = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "sqrt");
}

#[test]
fn test_tanh_accuracy() {
    let x = Array::from_vec(
        vec![0.0, 1.0, 2.0, -1.0, -2.0],
        Shape::new(vec![5]),
    );

    let result = x.tanh();

    let expected = vec![
        0.0,              // tanh(0)
        0.7615941559,     // tanh(1)
        0.9640275800,     // tanh(2)
        -0.7615941559,    // tanh(-1)
        -0.9640275800,    // tanh(-2)
    ];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "tanh");
}

#[test]
fn test_abs_accuracy() {
    let x = Array::from_vec(
        vec![-5.0, -2.5, 0.0, 2.5, 5.0],
        Shape::new(vec![5]),
    );

    let result = x.abs();

    let expected = vec![5.0, 2.5, 0.0, 2.5, 5.0];

    assert_array_close(&result, &expected, 0.0, 0.0, "abs");
}

// =============================================================================
// BINARY OPERATIONS
// =============================================================================

#[test]
fn test_add_accuracy() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![4]));

    let result = a.add(&b);

    let expected = vec![6.0, 8.0, 10.0, 12.0];

    assert_array_close(&result, &expected, 0.0, 1e-6, "add");
}

#[test]
fn test_mul_accuracy() {
    let a = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0], Shape::new(vec![4]));
    let b = Array::from_vec(vec![3.0, 4.0, 5.0, 6.0], Shape::new(vec![4]));

    let result = a.mul(&b);

    let expected = vec![6.0, 12.0, 20.0, 30.0];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "mul");
}

#[test]
fn test_div_accuracy() {
    let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::new(vec![4]));
    let b = Array::from_vec(vec![2.0, 4.0, 5.0, 8.0], Shape::new(vec![4]));

    let result = a.div(&b);

    let expected = vec![5.0, 5.0, 6.0, 5.0];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "div");
}

#[test]
fn test_pow_accuracy() {
    let a = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0], Shape::new(vec![4]));
    let b = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], Shape::new(vec![4]));

    let result = a.pow(&b);

    let expected = vec![4.0, 9.0, 16.0, 25.0];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "pow");
}

// =============================================================================
// REDUCTION OPERATIONS
// =============================================================================

#[test]
fn test_sum_accuracy() {
    let x = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::new(vec![2, 3]),
    );

    // Sum all
    let sum_all = x.sum_all();
    assert_close(sum_all, 21.0, 0.0, 1e-6, "sum_all");

    // Sum axis 0
    let sum_0 = x.sum(0);
    assert_array_close(&sum_0, &[5.0, 7.0, 9.0], 1e-6, 1e-6, "sum axis 0");

    // Sum axis 1
    let sum_1 = x.sum(1);
    assert_array_close(&sum_1, &[6.0, 15.0], 1e-6, 1e-6, "sum axis 1");
}

#[test]
fn test_mean_accuracy() {
    let x = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::new(vec![2, 3]),
    );

    // Mean all
    let mean_all = x.mean_all();
    assert_close(mean_all, 3.5, 1e-6, 1e-6, "mean_all");

    // Mean axis 0
    let mean_0 = x.mean(0);
    assert_array_close(&mean_0, &[2.5, 3.5, 4.5], 1e-6, 1e-6, "mean axis 0");

    // Mean axis 1
    let mean_1 = x.mean(1);
    assert_array_close(&mean_1, &[2.0, 5.0], 1e-6, 1e-6, "mean axis 1");
}

#[test]
fn test_max_accuracy() {
    let x = Array::from_vec(
        vec![3.0, 7.0, 2.0, 9.0, 1.0, 5.0],
        Shape::new(vec![2, 3]),
    );

    // Max all
    let max_all = x.max_all();
    assert_close(max_all, 9.0, 0.0, 0.0, "max_all");

    // Max axis 0
    let max_0 = x.max(0);
    assert_array_close(&max_0, &[9.0, 7.0, 5.0], 0.0, 0.0, "max axis 0");

    // Max axis 1
    let max_1 = x.max(1);
    assert_array_close(&max_1, &[7.0, 9.0], 0.0, 0.0, "max axis 1");
}

#[test]
fn test_min_accuracy() {
    let x = Array::from_vec(
        vec![3.0, 7.0, 2.0, 9.0, 1.0, 5.0],
        Shape::new(vec![2, 3]),
    );

    // Min all
    let min_all = x.min_all();
    assert_close(min_all, 1.0, 0.0, 0.0, "min_all");

    // Min axis 0
    let min_0 = x.min(0);
    assert_array_close(&min_0, &[3.0, 1.0, 2.0], 0.0, 0.0, "min axis 0");

    // Min axis 1
    let min_1 = x.min(1);
    assert_array_close(&min_1, &[2.0, 1.0], 0.0, 0.0, "min axis 1");
}

// =============================================================================
// LINEAR ALGEBRA
// =============================================================================

#[test]
fn test_matmul_accuracy() {
    // [[1, 2],    [[5, 6],     [[19, 22],
    //  [3, 4]]  @  [7, 8]]  =   [43, 50]]
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));

    let result = a.matmul(&b);

    let expected = vec![19.0, 22.0, 43.0, 50.0];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "matmul");
}

#[test]
fn test_matmul_identity() {
    let a = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        Shape::new(vec![3, 3]),
    );
    let identity = Array::eye(3, None, DType::Float32);

    // A @ I = A
    let result = a.matmul(&identity);
    assert_array_close(&result, &a.to_vec(), 1e-6, 1e-6, "matmul identity");

    // I @ A = A
    let result2 = identity.matmul(&a);
    assert_array_close(&result2, &a.to_vec(), 1e-6, 1e-6, "identity matmul");
}

#[test]
fn test_dot_accuracy() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));

    let result = a.dot(&b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    let result_scalar = result.to_vec()[0];
    assert_close(result_scalar, 32.0, 1e-6, 1e-6, "dot product");
}

// =============================================================================
// BROADCASTING
// =============================================================================

#[test]
fn test_broadcasting_accuracy() {
    // [2, 1] + [1, 3] -> [2, 3]
    let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2, 1]));
    let b = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![1, 3]));

    let result = a.add(&b);

    let expected = vec![
        11.0, 21.0, 31.0,  // [1 + [10, 20, 30]]
        12.0, 22.0, 32.0,  // [2 + [10, 20, 30]]
    ];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "broadcasting add");
}

// =============================================================================
// SPECIAL FUNCTIONS
// =============================================================================

#[test]
fn test_erf_accuracy() {
    use jax_rs::scipy::erf;

    let x = Array::from_vec(
        vec![0.0, 0.5, 1.0, 1.5, 2.0],
        Shape::new(vec![5]),
    );

    let result = erf(&x);

    let expected = vec![
        0.0,              // erf(0)
        0.5204998778,     // erf(0.5)
        0.8427007929,     // erf(1.0)
        0.9661051464,     // erf(1.5)
        0.9953222650,     // erf(2.0)
    ];

    assert_array_close(&result, &expected, 1e-5, 1e-5, "erf");
}

#[test]
fn test_erfc_accuracy() {
    use jax_rs::scipy::erfc;

    let x = Array::from_vec(
        vec![0.0, 0.5, 1.0, 1.5, 2.0],
        Shape::new(vec![5]),
    );

    let result = erfc(&x);

    let expected = vec![
        1.0,              // erfc(0) = 1 - erf(0)
        0.4795001221,     // erfc(0.5)
        0.1572992070,     // erfc(1.0)
        0.0338948535,     // erfc(1.5)
        0.0046777349,     // erfc(2.0)
    ];

    assert_array_close(&result, &expected, 1e-5, 1e-5, "erfc");
}

// =============================================================================
// EDGE CASES
// =============================================================================

#[test]
fn test_large_values_accuracy() {
    // Test that operations handle large values correctly
    let x = Array::from_vec(vec![1e6, 1e7, 1e8], Shape::new(vec![3]));
    let y = Array::from_vec(vec![1e6, 1e7, 1e8], Shape::new(vec![3]));

    let result = x.add(&y);
    let expected = vec![2e6, 2e7, 2e8];

    assert_array_close(&result, &expected, 1e-5, 1e-2, "large values add");
}

#[test]
fn test_small_values_accuracy() {
    // Test that operations handle small values correctly
    let x = Array::from_vec(vec![1e-6, 1e-7, 1e-8], Shape::new(vec![3]));
    let y = Array::from_vec(vec![1e-6, 1e-7, 1e-8], Shape::new(vec![3]));

    let result = x.add(&y);
    let expected = vec![2e-6, 2e-7, 2e-8];

    assert_array_close(&result, &expected, 1e-5, 1e-10, "small values add");
}

#[test]
fn test_negative_exponent_accuracy() {
    let x = Array::from_vec(vec![-1.0, -2.0, -3.0], Shape::new(vec![3]));

    let result = x.exp();

    let expected = vec![0.3678794411, 0.1353352832, 0.0497870683];

    assert_array_close(&result, &expected, 1e-6, 1e-6, "negative exponent");
}
