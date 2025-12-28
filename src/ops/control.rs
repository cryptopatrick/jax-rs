//! Control flow operations for JAX-style functional programming.
//!
//! This module provides control flow primitives that work with the
//! automatic differentiation and JIT compilation infrastructure.

use crate::{Array, Shape};

/// Scan operation - apply a function cumulatively along an axis.
///
/// This is similar to Python's `functools.reduce` but returns all intermediate results.
/// It's the functional programming equivalent of a for loop.
///
/// # Arguments
///
/// * `f` - A function that takes (carry, x) and returns (new_carry, y)
/// * `init` - Initial carry value
/// * `xs` - Input array to scan over (along axis 0)
///
/// # Returns
///
/// A tuple of (final_carry, ys) where ys contains all intermediate outputs.
///
/// # Examples
///
/// ```rust,ignore
/// use jax_rs::{Array, Shape, ops::control::scan};
///
/// // Cumulative sum using scan
/// let xs = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let init = Array::from_vec(vec![0.0], Shape::new(vec![1]));
///
/// let (final_carry, ys) = scan(
///     |carry, x| {
///         let new_carry = carry.add(&x);
///         (new_carry.clone(), new_carry)
///     },
///     init,
///     &xs,
/// );
/// // ys = [1.0, 3.0, 6.0, 10.0]
/// ```
pub fn scan<F>(f: F, init: Array, xs: &Array) -> (Array, Array)
where
    F: Fn(&Array, &Array) -> (Array, Array),
{
    let n = xs.shape().as_slice()[0];
    let mut carry = init;
    let mut ys = Vec::new();

    let x_data = xs.to_vec();
    let x_size = xs.size() / n;

    // Get remaining shape after removing first dimension
    let remaining_shape: Vec<usize> = if xs.ndim() > 1 {
        xs.shape().as_slice()[1..].to_vec()
    } else {
        vec![1]
    };

    for i in 0..n {
        // Extract the i-th element
        let start = i * x_size;
        let end = start + x_size;
        let xi = Array::from_vec(x_data[start..end].to_vec(), Shape::new(remaining_shape.clone()));

        // Apply function
        let (new_carry, y) = f(&carry, &xi);
        carry = new_carry;
        ys.push(y);
    }

    // Stack all ys
    let y_shape = ys[0].shape().clone();
    let mut stacked_data = Vec::new();
    for y in &ys {
        stacked_data.extend(y.to_vec());
    }

    // Create output shape with batch dimension
    let mut out_shape = vec![n];
    out_shape.extend(y_shape.as_slice());

    (carry, Array::from_vec(stacked_data, Shape::new(out_shape)))
}

/// While loop operation - repeat a function while a condition is true.
///
/// # Arguments
///
/// * `cond` - A function that takes state and returns a boolean (as f32: 1.0 = true, 0.0 = false)
/// * `body` - A function that transforms the state
/// * `init` - Initial state
///
/// # Returns
///
/// The final state when condition becomes false.
///
/// # Examples
///
/// ```rust,ignore
/// use jax_rs::{Array, Shape, ops::control::while_loop};
///
/// // Count up to 10
/// let init = Array::from_vec(vec![0.0], Shape::new(vec![1]));
///
/// let result = while_loop(
///     |state| {
///         let val = state.to_vec()[0];
///         if val < 10.0 { 1.0 } else { 0.0 }
///     },
///     |state| {
///         let data: Vec<f32> = state.to_vec().iter().map(|x| x + 1.0).collect();
///         Array::from_vec(data, state.shape().clone())
///     },
///     init,
/// );
/// // result = [10.0]
/// ```
pub fn while_loop<C, B>(cond: C, body: B, init: Array) -> Array
where
    C: Fn(&Array) -> f32,
    B: Fn(&Array) -> Array,
{
    let mut state = init;
    let max_iterations = 10000; // Prevent infinite loops
    let mut iterations = 0;

    while cond(&state) != 0.0 && iterations < max_iterations {
        state = body(&state);
        iterations += 1;
    }

    state
}

/// Conditional operation - choose between two values based on a condition.
///
/// Unlike `where_cond` which works element-wise, this `cond` function
/// evaluates entire branches based on a scalar condition.
///
/// # Arguments
///
/// * `pred` - Condition (non-zero = true)
/// * `true_fn` - Function to compute if pred is true
/// * `false_fn` - Function to compute if pred is false
///
/// # Returns
///
/// The result of either true_fn or false_fn based on pred.
///
/// # Examples
///
/// ```rust,ignore
/// use jax_rs::{Array, Shape, ops::control::cond};
///
/// let x = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
///
/// let result = cond(
///     1.0, // true
///     || x.mul(&x),  // return x^2
///     || x.add(&x),  // return 2x
/// );
/// // result = [1.0, 4.0]
/// ```
pub fn cond<T, F>(pred: f32, true_fn: T, false_fn: F) -> Array
where
    T: FnOnce() -> Array,
    F: FnOnce() -> Array,
{
    if pred != 0.0 {
        true_fn()
    } else {
        false_fn()
    }
}

/// For loop operation - apply a function n times with loop index.
///
/// # Arguments
///
/// * `n` - Number of iterations
/// * `init` - Initial state
/// * `body` - Function taking (i, state) and returning new state
///
/// # Returns
///
/// The final state after n iterations.
pub fn fori_loop<F>(n: usize, init: Array, body: F) -> Array
where
    F: Fn(usize, &Array) -> Array,
{
    let mut state = init;
    for i in 0..n {
        state = body(i, &state);
    }
    state
}

/// Reduce operation - fold over an array with a binary function.
///
/// # Arguments
///
/// * `f` - Binary function to apply
/// * `init` - Initial accumulator value
/// * `xs` - Array to reduce over
///
/// # Returns
///
/// The final accumulated value.
pub fn reduce<F>(f: F, init: f32, xs: &Array) -> f32
where
    F: Fn(f32, f32) -> f32,
{
    let data = xs.to_vec();
    data.iter().fold(init, |acc, &x| f(acc, x))
}

/// Map operation - apply a scalar function element-wise.
///
/// This is similar to `Array::map` but as a standalone function.
pub fn map<F>(xs: &Array, f: F) -> Array
where
    F: Fn(f32) -> f32,
{
    let data: Vec<f32> = xs.to_vec().iter().map(|&x| f(x)).collect();
    Array::from_vec(data, xs.shape().clone())
}

/// Switch operation - select from multiple branches based on an index.
///
/// # Arguments
///
/// * `index` - Integer index to select branch (0-indexed)
/// * `branches` - Vector of functions to choose from
/// * `operand` - Operand to pass to selected function
///
/// # Returns
///
/// Result of applying branches[index] to operand.
pub fn switch<F>(index: usize, branches: Vec<F>, operand: &Array) -> Array
where
    F: Fn(&Array) -> Array,
{
    assert!(index < branches.len(), "Index out of bounds");
    branches[index](operand)
}

/// Associative scan operation - parallel-friendly scan.
///
/// This computes a prefix scan using a binary associative operator.
/// It can be parallelized using a work-efficient algorithm.
pub fn associative_scan<F>(f: F, xs: &Array) -> Array
where
    F: Fn(&Array, &Array) -> Array,
{
    let n = xs.shape().as_slice()[0];
    if n == 0 {
        return xs.clone();
    }

    let x_data = xs.to_vec();
    let x_size = xs.size() / n;
    let remaining_shape: Vec<usize> = if xs.ndim() > 1 {
        xs.shape().as_slice()[1..].to_vec()
    } else {
        vec![1]
    };

    // Extract individual elements
    let mut elements: Vec<Array> = (0..n)
        .map(|i| {
            let start = i * x_size;
            let end = start + x_size;
            Array::from_vec(x_data[start..end].to_vec(), Shape::new(remaining_shape.clone()))
        })
        .collect();

    // Sequential scan (for now - could be parallelized)
    let mut results = vec![elements[0].clone()];
    for i in 1..n {
        let combined = f(&results[i - 1], &elements[i]);
        results.push(combined);
    }

    // Stack results
    let mut stacked_data = Vec::new();
    for r in &results {
        stacked_data.extend(r.to_vec());
    }

    let mut out_shape = vec![n];
    out_shape.extend(&remaining_shape);

    Array::from_vec(stacked_data, Shape::new(out_shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_cumsum() {
        // Cumulative sum
        let xs = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let init = Array::from_vec(vec![0.0], Shape::new(vec![1]));

        let (final_carry, ys) = scan(
            |carry, x| {
                let new_carry = carry.add(x);
                (new_carry.clone(), new_carry)
            },
            init,
            &xs,
        );

        assert_eq!(final_carry.to_vec(), vec![10.0]);
        assert_eq!(ys.to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_while_loop() {
        let init = Array::from_vec(vec![0.0], Shape::new(vec![1]));

        let result = while_loop(
            |state| {
                let val = state.to_vec()[0];
                if val < 5.0 { 1.0 } else { 0.0 }
            },
            |state| {
                let data: Vec<f32> = state.to_vec().iter().map(|x| x + 1.0).collect();
                Array::from_vec(data, state.shape().clone())
            },
            init,
        );

        assert_eq!(result.to_vec(), vec![5.0]);
    }

    #[test]
    fn test_cond() {
        let x = Array::from_vec(vec![2.0, 3.0], Shape::new(vec![2]));

        // Test true branch
        let result_true = cond(
            1.0,
            || x.mul(&x),
            || x.add(&x),
        );
        assert_eq!(result_true.to_vec(), vec![4.0, 9.0]);

        // Test false branch
        let result_false = cond(
            0.0,
            || x.mul(&x),
            || x.add(&x),
        );
        assert_eq!(result_false.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_fori_loop() {
        let init = Array::from_vec(vec![0.0], Shape::new(vec![1]));

        let result = fori_loop(5, init, |i, state| {
            let data: Vec<f32> = state.to_vec().iter().map(|x| x + (i as f32 + 1.0)).collect();
            Array::from_vec(data, state.shape().clone())
        });

        // 0 + 1 + 2 + 3 + 4 + 5 = 15
        assert_eq!(result.to_vec(), vec![15.0]);
    }

    #[test]
    fn test_reduce() {
        let xs = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));

        let sum = reduce(|a, b| a + b, 0.0, &xs);
        assert_eq!(sum, 10.0);

        let product = reduce(|a, b| a * b, 1.0, &xs);
        assert_eq!(product, 24.0);
    }

    #[test]
    fn test_map() {
        let xs = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));

        let squared = map(&xs, |x| x * x);
        assert_eq!(squared.to_vec(), vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_switch() {
        let x = Array::from_vec(vec![2.0, 3.0], Shape::new(vec![2]));

        let branches: Vec<fn(&Array) -> Array> = vec![
            |a| a.add(a),  // 0: double
            |a| a.mul(a),  // 1: square
            |a| a.neg(),   // 2: negate
        ];

        let result0 = switch(0, branches.clone(), &x);
        assert_eq!(result0.to_vec(), vec![4.0, 6.0]);

        let result1 = switch(1, branches.clone(), &x);
        assert_eq!(result1.to_vec(), vec![4.0, 9.0]);

        let result2 = switch(2, branches, &x);
        assert_eq!(result2.to_vec(), vec![-2.0, -3.0]);
    }

    #[test]
    fn test_associative_scan() {
        let xs = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));

        // Prefix sum using associative scan
        let result = associative_scan(|a, b| a.add(b), &xs);

        assert_eq!(result.to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
    }
}
