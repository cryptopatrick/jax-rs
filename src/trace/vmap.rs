//! Automatic vectorization via vmap.
//!
//! This module provides the `vmap` function that automatically vectorizes
//! operations over a batch dimension.

use crate::{Array, Shape};

/// Vectorize a function to map over a batch dimension.
///
/// Takes a function that works on individual examples and transforms it
/// to work on batches of examples.
///
/// # Examples
///
/// ```rust,ignore
/// use jax_rs::{Array, Shape, trace::vmap};
///
/// // Function that computes dot product of two vectors
/// fn dot(x: &Array, y: &Array) -> Array {
///     x.mul(y).sum_all_array()
/// }
///
/// // Vectorize to work on batches
/// let vmap_dot = vmap(dot, 0);
///
/// // xs and ys have shape [batch_size, vector_dim]
/// let xs = Array::zeros(Shape::new(vec![10, 3]), DType::Float32);
/// let ys = Array::zeros(Shape::new(vec![10, 3]), DType::Float32);
///
/// // results has shape [batch_size]
/// let results = vmap_dot(&xs, &ys);
/// ```
pub fn vmap<F>(f: F, in_axis: usize) -> impl Fn(&Array) -> Array
where
    F: Fn(&Array) -> Array,
{
    move |x: &Array| {
        // Get the batch size
        let batch_size = x.shape().as_slice()[in_axis];

        // Split input along the batch axis
        let inputs = split_along_axis(x, in_axis, batch_size);

        // Apply function to each batch element
        let outputs: Vec<Array> = inputs.iter().map(|inp| f(inp)).collect();

        // Stack results along the batch axis
        stack_along_axis(&outputs, in_axis)
    }
}

/// Vectorize a binary function.
///
/// Both inputs are batched along the specified axis.
pub fn vmap2<F>(f: F, in_axis: usize) -> impl Fn(&Array, &Array) -> Array
where
    F: Fn(&Array, &Array) -> Array,
{
    move |x: &Array, y: &Array| {
        // Get the batch size
        let batch_size = x.shape().as_slice()[in_axis];
        assert_eq!(
            batch_size,
            y.shape().as_slice()[in_axis],
            "Batch sizes must match"
        );

        // Split inputs along the batch axis
        let x_inputs = split_along_axis(x, in_axis, batch_size);
        let y_inputs = split_along_axis(y, in_axis, batch_size);

        // Apply function to each batch element
        let outputs: Vec<Array> = x_inputs
            .iter()
            .zip(y_inputs.iter())
            .map(|(xi, yi)| f(xi, yi))
            .collect();

        // Stack results along the batch axis
        stack_along_axis(&outputs, in_axis)
    }
}

/// Split an array into slices along a given axis.
fn split_along_axis(
    arr: &Array,
    axis: usize,
    num_splits: usize,
) -> Vec<Array> {
    let shape = arr.shape().as_slice();
    assert!(axis < shape.len(), "Axis out of bounds");
    assert_eq!(
        shape[axis] % num_splits,
        0,
        "Axis size must be divisible by num_splits"
    );

    let mut splits = Vec::with_capacity(num_splits);

    if axis == 0 {
        // Simple case: split along first axis
        let split_size = shape[axis] / num_splits;
        let data = arr.to_vec();
        let total_size = arr.size();
        let chunk_size = total_size / num_splits;

        let remaining_shape: Vec<usize> = if split_size == 1 {
            // Remove the batch dimension if it becomes 1
            shape[1..].to_vec()
        } else {
            let mut s = shape.to_vec();
            s[axis] = split_size;
            s
        };

        for i in 0..num_splits {
            let start = i * chunk_size;
            let end = start + chunk_size;
            let chunk = data[start..end].to_vec();
            let arr =
                Array::from_vec(chunk, Shape::new(remaining_shape.clone()));
            splits.push(arr);
        }
    } else {
        // For other axes, need to do more complex slicing
        // For now, only support axis 0
        panic!("vmap only supports axis=0 for now");
    }

    splits
}

/// Stack arrays along a given axis.
fn stack_along_axis(arrays: &[Array], axis: usize) -> Array {
    assert!(!arrays.is_empty(), "Cannot stack empty array list");

    if axis == 0 {
        // Simple case: concatenate along first axis
        let first_shape = arrays[0].shape().as_slice();
        let batch_size = arrays.len();

        // New shape with batch dimension
        let mut new_shape = vec![batch_size];
        new_shape.extend_from_slice(first_shape);

        // Concatenate all data
        let mut data = Vec::new();
        for arr in arrays {
            assert_eq!(
                arr.shape().as_slice(),
                first_shape,
                "All arrays must have the same shape"
            );
            data.extend(arr.to_vec());
        }

        Array::from_vec(data, Shape::new(new_shape))
    } else {
        // For other axes, need more complex logic
        panic!("stack only supports axis=0 for now");
    }
}

/// Vectorize a function with custom input and output axis specifications.
///
/// This is a more general version of vmap that allows specifying different
/// axes for inputs and outputs.
pub struct VmapConfig {
    /// Axis to map over in the input
    pub in_axis: usize,
    /// Axis to map over in the output (where batch dimension appears)
    pub out_axis: usize,
}

impl VmapConfig {
    /// Create a new vmap configuration.
    pub fn new(in_axis: usize, out_axis: usize) -> Self {
        Self { in_axis, out_axis }
    }

    /// Apply vmap with this configuration.
    pub fn apply<'a, F>(&'a self, f: F) -> impl Fn(&Array) -> Array + 'a
    where
        F: Fn(&Array) -> Array + 'a,
    {
        move |x: &Array| {
            let batch_size = x.shape().as_slice()[self.in_axis];
            let inputs = split_along_axis(x, self.in_axis, batch_size);
            let outputs: Vec<Array> =
                inputs.iter().map(|inp| f(inp)).collect();

            if self.out_axis == 0 {
                stack_along_axis(&outputs, self.out_axis)
            } else {
                // Move batch axis to desired position
                // For now, only support out_axis=0
                panic!("out_axis != 0 not yet supported");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_vmap_unary() {
        // Function that squares all elements
        let square = |x: &Array| x.mul(x);

        let vmap_square = vmap(square, 0);

        // Input: [2, 3] - batch of 2 vectors of length 3
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        let result = vmap_square(&x);

        // Output should be [2, 3] with squared values
        assert_eq!(result.shape().as_slice(), &[2, 3]);
        assert_eq!(result.to_vec(), vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);
    }

    #[test]
    fn test_vmap2_binary() {
        // Function that adds two vectors
        let add = |x: &Array, y: &Array| x.add(y);

        let vmap_add = vmap2(add, 0);

        // Inputs: [2, 3] each
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let y = Array::from_vec(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            Shape::new(vec![2, 3]),
        );

        let result = vmap_add(&x, &y);

        // Output should be [2, 3]
        assert_eq!(result.shape().as_slice(), &[2, 3]);
        assert_eq!(result.to_vec(), vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }

    #[test]
    fn test_vmap_reduction() {
        // Function that sums a vector
        let sum = |x: &Array| x.sum_all_array();

        let vmap_sum = vmap(sum, 0);

        // Input: [3, 2] - batch of 3 vectors of length 2
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![3, 2]),
        );

        let result = vmap_sum(&x);

        // Output should be [3] with sums
        assert_eq!(result.shape().as_slice(), &[3]);
        assert_eq!(result.to_vec(), vec![3.0, 7.0, 11.0]);
    }

    #[test]
    fn test_split_along_axis() {
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![3, 2]),
        );

        let splits = split_along_axis(&x, 0, 3);

        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].to_vec(), vec![1.0, 2.0]);
        assert_eq!(splits[1].to_vec(), vec![3.0, 4.0]);
        assert_eq!(splits[2].to_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_stack_along_axis() {
        let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
        let c = Array::from_vec(vec![5.0, 6.0], Shape::new(vec![2]));

        let stacked = stack_along_axis(&[a, b, c], 0);

        assert_eq!(stacked.shape().as_slice(), &[3, 2]);
        assert_eq!(stacked.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_vmap_nested() {
        // Function that multiplies by 2
        let mul2 = |x: &Array| {
            let two = Array::from_vec(vec![2.0], Shape::new(vec![1]));
            x.mul(&two)
        };

        let vmap_mul2 = vmap(mul2, 0);

        // Input: [2, 3]
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        let result = vmap_mul2(&x);

        assert_eq!(result.shape().as_slice(), &[2, 3]);
        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }
}
