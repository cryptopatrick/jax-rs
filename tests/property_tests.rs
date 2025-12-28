//! Property-based tests for jax-rs using proptest.
//!
//! These tests generate random inputs and validate that operations satisfy
//! expected mathematical properties and invariants.

use jax_rs::{Array, DType, Shape};
use proptest::prelude::*;

// =============================================================================
// GENERATORS
// =============================================================================

/// Generate a valid shape (1-4 dimensions, each dimension 1-10 elements).
fn arb_shape() -> impl Strategy<Value = Shape> {
    prop::collection::vec(1usize..=10, 1..=4).prop_map(Shape::new)
}

/// Generate a small shape for faster tests (1-3 dimensions, each dimension 1-5 elements).
fn arb_small_shape() -> impl Strategy<Value = Shape> {
    prop::collection::vec(1usize..=5, 1..=3).prop_map(Shape::new)
}

/// Generate an array with given shape and random values.
fn arb_array_with_shape(shape: Shape) -> impl Strategy<Value = Array> {
    let size = shape.size();
    prop::collection::vec(-10.0f32..10.0, size)
        .prop_map(move |data| Array::from_vec(data, shape.clone()))
}

/// Generate a random array.
fn arb_array() -> impl Strategy<Value = Array> {
    arb_small_shape().prop_flat_map(arb_array_with_shape)
}

/// Generate a positive array (all elements > 0).
fn arb_positive_array() -> impl Strategy<Value = Array> {
    arb_small_shape().prop_flat_map(|shape| {
        let size = shape.size();
        prop::collection::vec(0.1f32..10.0, size)
            .prop_map(move |data| Array::from_vec(data, shape.clone()))
    })
}

// =============================================================================
// ALGEBRAIC PROPERTIES
// =============================================================================

proptest! {
    #[test]
    fn test_add_commutative(a in arb_array(), b in arb_array()) {
        // Ensure compatible shapes
        if a.shape() == b.shape() {
            let ab = a.add(&b);
            let ba = b.add(&a);
            let ab_vec = ab.to_vec();
            let ba_vec = ba.to_vec();

            for (x, y) in ab_vec.iter().zip(ba_vec.iter()) {
                prop_assert!((x - y).abs() < 1e-5, "add not commutative");
            }
        }
    }

    #[test]
    fn test_add_associative(a in arb_array(), b in arb_array(), c in arb_array()) {
        // Ensure compatible shapes
        if a.shape() == b.shape() && b.shape() == c.shape() {
            let abc1 = a.add(&b).add(&c);
            let abc2 = a.add(&b.add(&c));
            let v1 = abc1.to_vec();
            let v2 = abc2.to_vec();

            for (x, y) in v1.iter().zip(v2.iter()) {
                prop_assert!((x - y).abs() < 1e-4, "add not associative");
            }
        }
    }

    #[test]
    fn test_mul_commutative(a in arb_array(), b in arb_array()) {
        if a.shape() == b.shape() {
            let ab = a.mul(&b);
            let ba = b.mul(&a);
            let ab_vec = ab.to_vec();
            let ba_vec = ba.to_vec();

            for (x, y) in ab_vec.iter().zip(ba_vec.iter()) {
                prop_assert!((x - y).abs() < 1e-5, "mul not commutative");
            }
        }
    }

    #[test]
    fn test_add_zero_identity(a in arb_array()) {
        let zero = Array::zeros(a.shape().clone(), DType::Float32);
        let result = a.add(&zero);
        let a_vec = a.to_vec();
        let result_vec = result.to_vec();

        for (x, y) in a_vec.iter().zip(result_vec.iter()) {
            prop_assert!((x - y).abs() < 1e-6, "zero not identity for add");
        }
    }

    #[test]
    fn test_mul_one_identity(a in arb_array()) {
        let one = Array::ones(a.shape().clone(), DType::Float32);
        let result = a.mul(&one);
        let a_vec = a.to_vec();
        let result_vec = result.to_vec();

        for (x, y) in a_vec.iter().zip(result_vec.iter()) {
            prop_assert!((x - y).abs() < 1e-6, "one not identity for mul");
        }
    }

    #[test]
    fn test_mul_zero_absorbing(a in arb_array()) {
        let zero = Array::zeros(a.shape().clone(), DType::Float32);
        let result = a.mul(&zero);
        let result_vec = result.to_vec();

        for &x in &result_vec {
            prop_assert!(x.abs() < 1e-6, "zero not absorbing for mul");
        }
    }
}

// =============================================================================
// UNARY OPERATION PROPERTIES
// =============================================================================

proptest! {
    #[test]
    fn test_neg_involutive(a in arb_array()) {
        // -(-a) = a
        let result = a.neg().neg();
        let a_vec = a.to_vec();
        let result_vec = result.to_vec();

        for (x, y) in a_vec.iter().zip(result_vec.iter()) {
            prop_assert!((x - y).abs() < 1e-5, "neg not involutive");
        }
    }

    #[test]
    fn test_abs_idempotent(a in arb_array()) {
        // abs(abs(a)) = abs(a)
        let abs1 = a.abs();
        let abs2 = abs1.abs();
        let v1 = abs1.to_vec();
        let v2 = abs2.to_vec();

        for (x, y) in v1.iter().zip(v2.iter()) {
            prop_assert!((x - y).abs() < 1e-6, "abs not idempotent");
        }
    }

    #[test]
    fn test_abs_non_negative(a in arb_array()) {
        let result = a.abs();
        let result_vec = result.to_vec();

        for &x in &result_vec {
            prop_assert!(x >= 0.0, "abs produced negative value");
        }
    }

    #[test]
    fn test_sqrt_squares(a in arb_positive_array()) {
        // sqrt(a^2) = a for positive a
        let squared = a.mul(&a);
        let result = squared.sqrt();
        let a_vec = a.to_vec();
        let result_vec = result.to_vec();

        for (x, y) in a_vec.iter().zip(result_vec.iter()) {
            prop_assert!((x - y).abs() < 1e-3, "sqrt(a^2) != a");
        }
    }

    #[test]
    fn test_exp_log_inverse(a in arb_positive_array()) {
        // exp(log(a)) = a for positive a
        let result = a.log().exp();
        let a_vec = a.to_vec();
        let result_vec = result.to_vec();

        for (x, y) in a_vec.iter().zip(result_vec.iter()) {
            let rel_error = ((x - y) / x).abs();
            prop_assert!(rel_error < 1e-5, "exp(log(a)) != a: {} vs {}", x, y);
        }
    }
}

// =============================================================================
// SHAPE OPERATIONS
// =============================================================================

proptest! {
    #[test]
    fn test_reshape_preserves_size(a in arb_array()) {
        let size = a.size();
        let new_shape = Shape::new(vec![size]);
        let reshaped = a.reshape(new_shape);

        prop_assert_eq!(reshaped.size(), size);
    }

    #[test]
    fn test_reshape_preserves_data(a in arb_array()) {
        let size = a.size();
        let new_shape = Shape::new(vec![size]);
        let reshaped = a.reshape(new_shape);

        let a_vec = a.to_vec();
        let reshaped_vec = reshaped.to_vec();

        prop_assert_eq!(a_vec, reshaped_vec);
    }

    #[test]
    fn test_transpose_involutive(shape in arb_small_shape()) {
        // Only test on 2D arrays
        if shape.ndim() == 2 {
            let data: Vec<f32> = (0..shape.size()).map(|x| x as f32).collect();
            let a = Array::from_vec(data.clone(), shape);

            let result = a.transpose().transpose();
            let result_vec = result.to_vec();

            prop_assert_eq!(data, result_vec, "transpose not involutive");
        }
    }
}

// =============================================================================
// REDUCTION PROPERTIES
// =============================================================================

proptest! {
    #[test]
    fn test_sum_equals_sum_all(shape in arb_small_shape()) {
        let data: Vec<f32> = (0..shape.size()).map(|x| (x % 10) as f32).collect();
        let a = Array::from_vec(data, shape.clone());

        let sum_all = a.sum_all();

        // Sum over all axes should equal sum_all
        let mut current = a;
        for axis in (0..shape.ndim()).rev() {
            current = current.sum(axis);
        }

        if current.size() == 1 {
            let chained_sum = current.to_vec()[0];
            prop_assert!((sum_all - chained_sum).abs() < 1e-4,
                        "sum_all ({}) != chained sum ({})", sum_all, chained_sum);
        }
    }

    #[test]
    fn test_max_geq_min(a in arb_array()) {
        let max_val = a.max_all();
        let min_val = a.min_all();

        prop_assert!(max_val >= min_val, "max ({}) < min ({})", max_val, min_val);
    }

    #[test]
    fn test_mean_between_min_max(a in arb_array()) {
        let mean_val = a.mean_all();
        let max_val = a.max_all();
        let min_val = a.min_all();

        prop_assert!(mean_val >= min_val, "mean ({}) < min ({})", mean_val, min_val);
        prop_assert!(mean_val <= max_val, "mean ({}) > max ({})", mean_val, max_val);
    }
}

// =============================================================================
// BROADCASTING PROPERTIES
// =============================================================================

proptest! {
    #[test]
    fn test_broadcast_scalar_add(a in arb_array(), scalar in -10.0f32..10.0) {
        let s = Array::from_vec(vec![scalar], Shape::new(vec![1]));
        let result = a.add(&s);

        let a_vec = a.to_vec();
        let result_vec = result.to_vec();

        for (x, y) in a_vec.iter().zip(result_vec.iter()) {
            prop_assert!((y - (x + scalar)).abs() < 1e-5,
                        "broadcast add failed: {} + {} != {}", x, scalar, y);
        }
    }

    #[test]
    fn test_broadcast_scalar_mul(a in arb_array(), scalar in -10.0f32..10.0) {
        let s = Array::from_vec(vec![scalar], Shape::new(vec![1]));
        let result = a.mul(&s);

        let a_vec = a.to_vec();
        let result_vec = result.to_vec();

        for (x, y) in a_vec.iter().zip(result_vec.iter()) {
            let expected = x * scalar;
            prop_assert!((y - expected).abs() < 1e-4,
                        "broadcast mul failed: {} * {} != {}", x, scalar, y);
        }
    }
}

// =============================================================================
// COMPARISON PROPERTIES
// =============================================================================

proptest! {
    #[test]
    fn test_eq_reflexive(a in arb_array()) {
        let result = a.eq(&a);
        let result_vec = result.to_vec();

        for &x in &result_vec {
            prop_assert_eq!(x, 1.0, "eq not reflexive");
        }
    }

    #[test]
    fn test_lt_antisymmetric(a in arb_array(), b in arb_array()) {
        if a.shape() == b.shape() {
            let ab = a.lt(&b);
            let ba = b.lt(&a);
            let ab_vec = ab.to_vec();
            let ba_vec = ba.to_vec();

            for (x, y) in ab_vec.iter().zip(ba_vec.iter()) {
                // If a < b is true (1.0), then b < a must be false (0.0)
                if *x == 1.0 {
                    prop_assert_eq!(*y, 0.0, "lt not antisymmetric");
                }
            }
        }
    }
}
