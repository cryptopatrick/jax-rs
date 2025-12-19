//! Comparison operations on arrays.

use crate::{buffer::Buffer, Array, DType, Device};

#[cfg(test)]
use crate::Shape;

/// Apply a comparison function element-wise to two arrays with broadcasting.
fn compare_op<F>(lhs: &Array, rhs: &Array, f: F) -> Array
where
    F: Fn(f32, f32) -> bool,
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

    let result_data: Vec<f32> = if lhs.shape() == rhs.shape() {
        // Same shape - simple element-wise operation
        lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| if f(a, b) { 1.0 } else { 0.0 })
            .collect()
    } else {
        // Need broadcasting
        let size = result_shape.size();
        (0..size)
            .map(|i| {
                let lhs_idx = crate::ops::binary::broadcast_index(
                    i,
                    &result_shape,
                    lhs.shape(),
                );
                let rhs_idx = crate::ops::binary::broadcast_index(
                    i,
                    &result_shape,
                    rhs.shape(),
                );
                if f(lhs_data[lhs_idx], rhs_data[rhs_idx]) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    };

    let buffer = Buffer::from_f32(result_data, Device::Cpu);
    Array::from_buffer(buffer, result_shape)
}

impl Array {
    /// Element-wise less than comparison.
    ///
    /// Returns an array of 1.0 where condition is true, 0.0 otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
    /// let c = a.lt(&b);
    /// assert_eq!(c.to_vec(), vec![1.0, 0.0, 0.0]);
    /// ```
    pub fn lt(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| a < b)
    }

    /// Element-wise less than or equal comparison.
    pub fn le(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| a <= b)
    }

    /// Element-wise greater than comparison.
    pub fn gt(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| a > b)
    }

    /// Element-wise greater than or equal comparison.
    pub fn ge(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| a >= b)
    }

    /// Element-wise equality comparison.
    ///
    /// Note: For floating point, this is exact equality. Use `allclose` for
    /// approximate equality.
    pub fn eq(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| a == b)
    }

    /// Element-wise inequality comparison.
    pub fn ne(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| a != b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lt() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
        let c = a.lt(&b);
        assert_eq!(c.to_vec(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_le() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
        let c = a.le(&b);
        assert_eq!(c.to_vec(), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_gt() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
        let c = a.gt(&b);
        assert_eq!(c.to_vec(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_ge() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
        let c = a.ge(&b);
        assert_eq!(c.to_vec(), vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_eq() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![1.0, 2.0, 4.0], Shape::new(vec![3]));
        let c = a.eq(&b);
        assert_eq!(c.to_vec(), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_ne() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![1.0, 2.0, 4.0], Shape::new(vec![3]));
        let c = a.ne(&b);
        assert_eq!(c.to_vec(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_comparison_broadcast() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0], Shape::new(vec![1]));
        let c = a.lt(&b);
        assert_eq!(c.to_vec(), vec![1.0, 0.0, 0.0]);
    }
}
