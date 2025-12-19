//! Integration tests for jax-rs.
//!
//! Tests complete workflows combining multiple features.

use jax_rs::{grad, vmap, Array, DType, Shape};

#[test]
fn test_vmap_grad_composition() {
    // Test composing vmap and grad
    let f = |x: &Array| x.mul(x).sum_all_array();
    let df = grad(f);
    let batch_df = vmap(df, 0);

    // Batch of 2 vectors
    let xs = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::new(vec![2, 3]),
    );

    let grads = batch_df(&xs);

    assert_eq!(grads.shape().as_slice(), &[2, 3]);

    // First gradient: [2, 4, 6]
    let grad_data = grads.to_vec();
    assert!((grad_data[0] - 2.0).abs() < 0.1);
    assert!((grad_data[1] - 4.0).abs() < 0.1);
    assert!((grad_data[2] - 6.0).abs() < 0.1);
}

#[test]
fn test_complex_operations_chain() {
    // Test chaining multiple operations
    let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));

    let result = x
        .add(&Array::ones(Shape::new(vec![2, 2]), DType::Float32))
        .mul(&Array::from_vec(vec![2.0], Shape::new(vec![1])))
        .sqrt()
        .sum_all();

    // (x + 1) * 2 -> [[4, 6], [8, 10]]
    // sqrt -> [[2, 2.449], [2.828, 3.162]]
    // sum -> ~10.44
    assert!(result > 10.0 && result < 11.0);
}

#[test]
fn test_broadcasting_chain() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3, 1]));
    let b = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![1, 2]));

    let result = a.add(&b);

    assert_eq!(result.shape().as_slice(), &[3, 2]);
    assert_eq!(result.to_vec(), vec![2.0, 3.0, 3.0, 4.0, 4.0, 5.0]);
}

#[test]
fn test_matmul_chain() {
    // Test matrix multiplication chains
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
    let c = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));

    // (A @ B) @ C = A @ B (since C is identity)
    let ab = a.matmul(&b);
    let abc = ab.matmul(&c);

    assert_eq!(abc.shape().as_slice(), &[2, 2]);
    // A @ B = [[19, 22], [43, 50]]
    assert_eq!(abc.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_reduction_consistency() {
    // Test that reductions are consistent
    let x = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::new(vec![2, 3]),
    );

    // sum_all should equal sum over all axes
    let sum_all = x.sum_all();
    let sum_0 = x.sum(0).sum_all();
    let sum_1 = x.sum(1).sum_all();

    assert!((sum_all - 21.0).abs() < 0.001);
    assert!((sum_0 - 21.0).abs() < 0.001);
    assert!((sum_1 - 21.0).abs() < 0.001);
}

#[test]
fn test_reshape_operations() {
    let x = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::new(vec![2, 3]),
    );

    // Reshape to [3, 2]
    let reshaped = x.reshape(Shape::new(vec![3, 2]));
    assert_eq!(reshaped.shape().as_slice(), &[3, 2]);
    assert_eq!(reshaped.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Reshape to [6]
    let flat = x.reshape(Shape::new(vec![6]));
    assert_eq!(flat.shape().as_slice(), &[6]);
}

#[test]
fn test_array_creation_consistency() {
    let shape = Shape::new(vec![2, 3]);

    let zeros = Array::zeros(shape.clone(), DType::Float32);
    assert_eq!(zeros.to_vec(), vec![0.0; 6]);

    let ones = Array::ones(shape.clone(), DType::Float32);
    assert_eq!(ones.to_vec(), vec![1.0; 6]);

    let full = Array::full(42.0, shape.clone(), DType::Float32);
    assert!(full.to_vec().iter().all(|&x| (x - 42.0).abs() < 0.001));
}

#[test]
fn test_comparison_operations() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));

    let lt = a.lt(&b);
    assert_eq!(lt.to_vec(), vec![1.0, 0.0, 0.0]);

    let eq = a.eq(&b);
    assert_eq!(eq.to_vec(), vec![0.0, 1.0, 0.0]);

    let gt = a.gt(&b);
    assert_eq!(gt.to_vec(), vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_unary_operations_comprehensive() {
    let x = Array::from_vec(vec![1.0, 4.0, 9.0], Shape::new(vec![3]));

    // sqrt
    let sqrt_x = x.sqrt();
    assert!((sqrt_x.to_vec()[0] - 1.0).abs() < 0.001);
    assert!((sqrt_x.to_vec()[1] - 2.0).abs() < 0.001);
    assert!((sqrt_x.to_vec()[2] - 3.0).abs() < 0.001);

    // square
    let sq_x = x.square();
    assert_eq!(sq_x.to_vec(), vec![1.0, 16.0, 81.0]);

    // reciprocal
    let rec_x = x.reciprocal();
    assert!((rec_x.to_vec()[0] - 1.0).abs() < 0.001);
    assert!((rec_x.to_vec()[1] - 0.25).abs() < 0.001);
}

#[test]
fn test_trigonometric_functions() {
    use std::f32::consts::PI;

    let x = Array::from_vec(vec![0.0, PI / 2.0, PI], Shape::new(vec![3]));

    let sin_x = x.sin();
    assert!((sin_x.to_vec()[0] - 0.0).abs() < 0.001);
    assert!((sin_x.to_vec()[1] - 1.0).abs() < 0.001);
    assert!((sin_x.to_vec()[2] - 0.0).abs() < 0.001);

    let cos_x = x.cos();
    assert!((cos_x.to_vec()[0] - 1.0).abs() < 0.001);
    assert!((cos_x.to_vec()[1] - 0.0).abs() < 0.001);
    assert!((cos_x.to_vec()[2] + 1.0).abs() < 0.001);
}

#[test]
fn test_edge_cases_zeros_ones() {
    // Test operations with zeros
    let zeros = Array::zeros(Shape::new(vec![3]), DType::Float32);
    let ones = Array::ones(Shape::new(vec![3]), DType::Float32);

    // zeros + ones = ones
    let sum = zeros.add(&ones);
    assert_eq!(sum.to_vec(), vec![1.0, 1.0, 1.0]);

    // zeros * anything = zeros
    let prod = zeros.mul(&ones);
    assert_eq!(prod.to_vec(), vec![0.0, 0.0, 0.0]);

    // ones * ones = ones
    let ones_sq = ones.mul(&ones);
    assert_eq!(ones_sq.to_vec(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_negative_values() {
    let x = Array::from_vec(vec![-1.0, -2.0, -3.0], Shape::new(vec![3]));

    // abs
    let abs_x = x.abs();
    assert_eq!(abs_x.to_vec(), vec![1.0, 2.0, 3.0]);

    // neg
    let neg_x = x.neg();
    assert_eq!(neg_x.to_vec(), vec![1.0, 2.0, 3.0]);

    // square preserves sign
    let sq_x = x.square();
    assert_eq!(sq_x.to_vec(), vec![1.0, 4.0, 9.0]);
}

#[test]
fn test_scalar_shape() {
    let scalar = Array::from_vec(vec![42.0], Shape::scalar());

    assert_eq!(scalar.shape().as_slice(), &[]);
    assert_eq!(scalar.size(), 1);
    assert_eq!(scalar.to_vec(), vec![42.0]);
}

#[test]
fn test_linspace() {
    let x = Array::linspace(0.0, 10.0, 11, true, DType::Float32);

    assert_eq!(x.shape().as_slice(), &[11]);
    assert!((x.to_vec()[0] - 0.0).abs() < 0.001);
    assert!((x.to_vec()[5] - 5.0).abs() < 0.001);
    assert!((x.to_vec()[10] - 10.0).abs() < 0.001);
}

#[test]
fn test_arange() {
    let x = Array::arange(0.0, 5.0, 1.0, DType::Float32);

    assert_eq!(x.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_eye_identity() {
    let eye = Array::eye(3, None, DType::Float32);

    assert_eq!(eye.shape().as_slice(), &[3, 3]);
    assert_eq!(
        eye.to_vec(),
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );

    // Identity matrix property: I @ A = A
    let a = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        Shape::new(vec![3, 3]),
    );
    let result = eye.matmul(&a);
    assert_eq!(result.to_vec(), a.to_vec());
}

// Tracing tests

#[test]
fn test_tracing_binary_op() {
    use jax_rs::trace::{enter_trace, exit_trace, is_tracing, TraceContext};
    use std::cell::RefCell;
    use std::rc::Rc;

    let ctx = Rc::new(RefCell::new(TraceContext::new("test".to_string())));

    let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    let y = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));

    {
        let mut ctx_mut = ctx.borrow_mut();
        ctx_mut.register_input(x.id(), x.shape().clone(), x.dtype());
        ctx_mut.register_input(y.id(), y.shape().clone(), y.dtype());
    }

    assert!(!is_tracing());
    enter_trace(ctx.clone());
    assert!(is_tracing());

    let z = x.add(&y);

    exit_trace();
    assert!(!is_tracing());

    let trace_ctx = Rc::try_unwrap(ctx)
        .expect("TraceContext still has references")
        .into_inner();

    let node = trace_ctx.get_node(z.id());
    assert!(node.is_some(), "Operation should have been traced");

    assert_eq!(z.to_vec(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_tracing_composition() {
    use jax_rs::trace::{enter_trace, exit_trace, TraceContext};
    use std::cell::RefCell;
    use std::rc::Rc;

    let ctx = Rc::new(RefCell::new(TraceContext::new("test".to_string())));

    let x = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));

    {
        let mut ctx_mut = ctx.borrow_mut();
        ctx_mut.register_input(x.id(), x.shape().clone(), x.dtype());
    }

    enter_trace(ctx.clone());

    let x_squared = x.mul(&x);
    let result = x_squared.sum_all_array();

    exit_trace();

    let trace_ctx = Rc::try_unwrap(ctx)
        .expect("TraceContext still has references")
        .into_inner();

    assert!(trace_ctx.get_node(x_squared.id()).is_some());
    assert!(trace_ctx.get_node(result.id()).is_some());

    assert!((result.to_vec()[0] - 5.0).abs() < 0.001);
}
