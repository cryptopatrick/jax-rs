//! Binary operations on arrays.

use crate::trace::{is_tracing, trace_binary, Primitive};
use crate::{buffer::Buffer, Array, DType, Device, Shape};

/// Apply a binary function element-wise to two arrays with broadcasting.
fn binary_op<F>(lhs: &Array, rhs: &Array, op: Primitive, f: F) -> Array
where
    F: Fn(f32, f32) -> f32,
{
    assert_eq!(lhs.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(rhs.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(lhs.device(), Device::Cpu, "Only CPU supported for now");
    assert_eq!(rhs.device(), Device::Cpu, "Only CPU supported for now");

    // Check if shapes are broadcast-compatible
    let result_shape = lhs
        .shape()
        .broadcast_with(rhs.shape())
        .expect("Shapes are not broadcast-compatible");

    let lhs_data = lhs.to_vec();
    let rhs_data = rhs.to_vec();

    let result_data = if lhs.shape() == rhs.shape() {
        // Same shape - simple element-wise operation
        lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| f(a, b)).collect()
    } else {
        // Need broadcasting
        broadcast_binary(
            &lhs_data,
            lhs.shape(),
            &rhs_data,
            rhs.shape(),
            &result_shape,
            f,
        )
    };

    let buffer = Buffer::from_f32(result_data, Device::Cpu);
    let result = Array::from_buffer(buffer, result_shape);

    // Register with trace context if tracing is active
    if is_tracing() {
        trace_binary(result.id(), op, lhs, rhs);
    }

    result
}

/// Helper function to perform binary operation with broadcasting.
fn broadcast_binary<F>(
    lhs_data: &[f32],
    lhs_shape: &Shape,
    rhs_data: &[f32],
    rhs_shape: &Shape,
    result_shape: &Shape,
    f: F,
) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32,
{
    let size = result_shape.size();
    let mut result = Vec::with_capacity(size);

    for i in 0..size {
        let lhs_idx = broadcast_index(i, result_shape, lhs_shape);
        let rhs_idx = broadcast_index(i, result_shape, rhs_shape);
        result.push(f(lhs_data[lhs_idx], rhs_data[rhs_idx]));
    }

    result
}

/// Convert a flat index in the result array to an index in the source array,
/// accounting for broadcasting.
pub(crate) fn broadcast_index(
    flat_idx: usize,
    result_shape: &Shape,
    src_shape: &Shape,
) -> usize {
    let result_dims = result_shape.as_slice();
    let src_dims = src_shape.as_slice();

    // Convert flat index to multi-dimensional index
    let mut multi_idx = Vec::with_capacity(result_dims.len());
    let mut idx = flat_idx;
    for &dim in result_dims.iter().rev() {
        multi_idx.push(idx % dim);
        idx /= dim;
    }
    multi_idx.reverse();

    // Map to source index with broadcasting
    let offset = result_dims.len() - src_dims.len();
    let mut src_idx = 0;
    let mut stride = 1;

    for i in (0..src_dims.len()).rev() {
        let result_i = offset + i;
        let dim_idx = if src_dims[i] == 1 {
            0 // Broadcast dimension
        } else {
            multi_idx[result_i]
        };
        src_idx += dim_idx * stride;
        stride *= src_dims[i];
    }

    src_idx
}

impl Array {
    /// Add two arrays element-wise with broadcasting.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
    /// let c = a.add(&b);
    /// assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0]);
    /// ```
    pub fn add(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Add, |a, b| a + b)
    }

    /// Subtract two arrays element-wise with broadcasting.
    pub fn sub(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Sub, |a, b| a - b)
    }

    /// Multiply two arrays element-wise with broadcasting.
    pub fn mul(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Mul, |a, b| a * b)
    }

    /// Divide two arrays element-wise with broadcasting.
    pub fn div(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| a / b)
    }

    /// Raise elements to a power element-wise with broadcasting.
    pub fn pow(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Pow, |a, b| a.powf(b))
    }

    /// Element-wise minimum.
    pub fn minimum(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Min, |a, b| a.min(b))
    }

    /// Element-wise maximum.
    pub fn maximum(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Max, |a, b| a.max(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let c = a.add(&b);
        assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_sub() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let c = a.sub(&b);
        assert_eq!(c.to_vec(), vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul() {
        let a = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![5.0, 6.0, 7.0], Shape::new(vec![3]));
        let c = a.mul(&b);
        assert_eq!(c.to_vec(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 4.0, 5.0], Shape::new(vec![3]));
        let c = a.div(&b);
        assert_eq!(c.to_vec(), vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_pow() {
        let a = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
        let c = a.pow(&b);
        assert_eq!(c.to_vec(), vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_broadcast_scalar() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![10.0], Shape::new(vec![1]));
        let c = a.add(&b);
        assert_eq!(c.to_vec(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_broadcast_2d() {
        // [2, 3] + [1, 3] -> [2, 3]
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let b =
            Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![1, 3]));
        let c = a.add(&b);
        assert_eq!(c.shape().as_slice(), &[2, 3]);
        assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_row_col() {
        // [3, 1] + [1, 3] -> [3, 3]
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3, 1]));
        let b =
            Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![1, 3]));
        let c = a.add(&b);
        assert_eq!(c.shape().as_slice(), &[3, 3]);
        assert_eq!(
            c.to_vec(),
            vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0, 33.0]
        );
    }

    #[test]
    fn test_minimum_maximum() {
        let a = Array::from_vec(vec![1.0, 5.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 4.0, 6.0], Shape::new(vec![3]));

        let min_ab = a.minimum(&b);
        assert_eq!(min_ab.to_vec(), vec![1.0, 4.0, 3.0]);

        let max_ab = a.maximum(&b);
        assert_eq!(max_ab.to_vec(), vec![2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_broadcast_index() {
        // Test broadcast_index function
        let result_shape = Shape::new(vec![2, 3]);
        let src_shape = Shape::new(vec![1, 3]);

        // For result shape [2,3], indices 0-5 map to positions:
        // 0: [0,0] -> [0,0] in [1,3] -> flat 0
        // 1: [0,1] -> [0,1] in [1,3] -> flat 1
        // 2: [0,2] -> [0,2] in [1,3] -> flat 2
        // 3: [1,0] -> [0,0] in [1,3] -> flat 0 (broadcast)
        // 4: [1,1] -> [0,1] in [1,3] -> flat 1 (broadcast)
        // 5: [1,2] -> [0,2] in [1,3] -> flat 2 (broadcast)
        assert_eq!(broadcast_index(0, &result_shape, &src_shape), 0);
        assert_eq!(broadcast_index(1, &result_shape, &src_shape), 1);
        assert_eq!(broadcast_index(2, &result_shape, &src_shape), 2);
        assert_eq!(broadcast_index(3, &result_shape, &src_shape), 0);
        assert_eq!(broadcast_index(4, &result_shape, &src_shape), 1);
        assert_eq!(broadcast_index(5, &result_shape, &src_shape), 2);
    }
}
