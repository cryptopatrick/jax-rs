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

    /// Element-wise equality comparison with a scalar.
    ///
    /// Returns an array where each element is 1.0 if equal to the scalar, 0.0 otherwise.
    pub fn eq_scalar(&self, value: f32) -> Array {
        let data = self.to_vec();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x == value { 1.0 } else { 0.0 })
            .collect();
        Array::from_vec(result, self.shape().clone())
    }

    /// Element-wise inequality comparison.
    pub fn ne(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| a != b)
    }

    /// Logical NOT element-wise.
    ///
    /// Treats 0.0 as false, non-zero as true.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0, 0.0], Shape::new(vec![3]));
    /// let b = a.logical_not();
    /// assert_eq!(b.to_vec(), vec![1.0, 0.0, 1.0]);
    /// ```
    pub fn logical_not(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x == 0.0 { 1.0 } else { 0.0 })
            .collect();
        Array::from_vec(result, self.shape().clone())
    }

    /// Logical AND element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 1.0, 0.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 0.0, 0.0], Shape::new(vec![3]));
    /// let c = a.logical_and(&b);
    /// assert_eq!(c.to_vec(), vec![1.0, 0.0, 0.0]);
    /// ```
    pub fn logical_and(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| a != 0.0 && b != 0.0)
    }

    /// Logical OR element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 1.0, 0.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 0.0, 0.0], Shape::new(vec![3]));
    /// let c = a.logical_or(&b);
    /// assert_eq!(c.to_vec(), vec![1.0, 1.0, 0.0]);
    /// ```
    pub fn logical_or(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| a != 0.0 || b != 0.0)
    }

    /// Logical XOR element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 1.0, 0.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 0.0, 0.0], Shape::new(vec![3]));
    /// let c = a.logical_xor(&b);
    /// assert_eq!(c.to_vec(), vec![0.0, 1.0, 0.0]);
    /// ```
    pub fn logical_xor(&self, other: &Array) -> Array {
        compare_op(self, other, |a, b| (a != 0.0) != (b != 0.0))
    }

    /// Test if all elements are true (non-zero).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// assert!(a.all());
    /// let b = Array::from_vec(vec![1.0, 0.0, 3.0], Shape::new(vec![3]));
    /// assert!(!b.all());
    /// ```
    pub fn all(&self) -> bool {
        let data = self.to_vec();
        data.iter().all(|&x| x != 0.0)
    }

    /// Test if any element is true (non-zero).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 0.0, 1.0], Shape::new(vec![3]));
    /// assert!(a.any());
    /// let b = Array::from_vec(vec![0.0, 0.0, 0.0], Shape::new(vec![3]));
    /// assert!(!b.any());
    /// ```
    pub fn any(&self) -> bool {
        let data = self.to_vec();
        data.iter().any(|&x| x != 0.0)
    }

    /// Count the number of true (non-zero) elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0], Shape::new(vec![5]));
    /// assert_eq!(a.count_nonzero(), 3);
    /// ```
    pub fn count_nonzero(&self) -> usize {
        let data = self.to_vec();
        data.iter().filter(|&&x| x != 0.0).count()
    }

    /// Test if two arrays are element-wise equal within a tolerance.
    ///
    /// Returns true if all elements satisfy: |a - b| <= atol + rtol * |b|
    ///
    /// # Arguments
    ///
    /// * `other` - Array to compare with
    /// * `rtol` - Relative tolerance
    /// * `atol` - Absolute tolerance
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0001, 2.0001, 3.0001], Shape::new(vec![3]));
    /// assert!(a.allclose(&b, 1e-3, 1e-3));
    /// assert!(!a.allclose(&b, 1e-5, 1e-5));
    /// ```
    pub fn allclose(&self, other: &Array, rtol: f32, atol: f32) -> bool {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        // Check if shapes are broadcast-compatible
        let result_shape = match self.shape().broadcast_with(other.shape()) {
            Some(shape) => shape,
            None => return false,
        };

        let self_data = self.to_vec();
        let other_data = other.to_vec();

        if self.shape() == other.shape() {
            // Same shape - simple element-wise comparison
            self_data.iter().zip(other_data.iter()).all(|(&a, &b)| {
                let diff = (a - b).abs();
                diff <= atol + rtol * b.abs()
            })
        } else {
            // Need broadcasting
            let size = result_shape.size();
            (0..size).all(|i| {
                let self_idx =
                    crate::ops::binary::broadcast_index(i, &result_shape, self.shape());
                let other_idx =
                    crate::ops::binary::broadcast_index(i, &result_shape, other.shape());
                let a = self_data[self_idx];
                let b = other_data[other_idx];
                let diff = (a - b).abs();
                diff <= atol + rtol * b.abs()
            })
        }
    }

    /// Element-wise test if values are close within a tolerance.
    ///
    /// Returns an array of 1.0 where |a - b| <= atol + rtol * |b|, 0.0 otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0001, 2.1, 3.0001], Shape::new(vec![3]));
    /// let c = a.isclose(&b, 1e-3, 1e-3);
    /// assert_eq!(c.to_vec(), vec![1.0, 0.0, 1.0]);
    /// ```
    pub fn isclose(&self, other: &Array, rtol: f32, atol: f32) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        // Check if shapes are broadcast-compatible
        let result_shape = self
            .shape()
            .broadcast_with(other.shape())
            .expect("Shapes are not broadcast-compatible");

        let self_data = self.to_vec();
        let other_data = other.to_vec();

        let result_data: Vec<f32> = if self.shape() == other.shape() {
            // Same shape - simple element-wise operation
            self_data
                .iter()
                .zip(other_data.iter())
                .map(|(&a, &b)| {
                    let diff = (a - b).abs();
                    if diff <= atol + rtol * b.abs() {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            // Need broadcasting
            let size = result_shape.size();
            (0..size)
                .map(|i| {
                    let self_idx =
                        crate::ops::binary::broadcast_index(i, &result_shape, self.shape());
                    let other_idx =
                        crate::ops::binary::broadcast_index(i, &result_shape, other.shape());
                    let a = self_data[self_idx];
                    let b = other_data[other_idx];
                    let diff = (a - b).abs();
                    if diff <= atol + rtol * b.abs() {
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

    /// Test if two arrays have the same shape and elements.
    ///
    /// This is exact equality - for approximate equality use `allclose`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let c = Array::from_vec(vec![1.0, 2.0, 3.1], Shape::new(vec![3]));
    /// assert!(a.array_equal(&b));
    /// assert!(!a.array_equal(&c));
    /// ```
    pub fn array_equal(&self, other: &Array) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        if self.dtype() != other.dtype() {
            return false;
        }

        let self_data = self.to_vec();
        let other_data = other.to_vec();
        self_data == other_data
    }

    /// Test if arrays can be broadcast to the same shape and are equal.
    ///
    /// Unlike `array_equal`, this allows broadcasting.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
    /// assert!(a.array_equiv(&b));
    /// ```
    pub fn array_equiv(&self, other: &Array) -> bool {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        // Check if shapes are broadcast-compatible
        let result_shape = match self.shape().broadcast_with(other.shape()) {
            Some(shape) => shape,
            None => return false,
        };

        let self_data = self.to_vec();
        let other_data = other.to_vec();

        if self.shape() == other.shape() {
            // Same shape - simple element-wise comparison
            self_data == other_data
        } else {
            // Need broadcasting
            let size = result_shape.size();
            (0..size).all(|i| {
                let self_idx =
                    crate::ops::binary::broadcast_index(i, &result_shape, self.shape());
                let other_idx =
                    crate::ops::binary::broadcast_index(i, &result_shape, other.shape());
                self_data[self_idx] == other_data[other_idx]
            })
        }
    }

    /// Element-wise greater comparison (alias for gt).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
    /// let c = a.greater(&b);
    /// assert_eq!(c.to_vec(), vec![0.0, 0.0, 1.0]);
    /// ```
    pub fn greater(&self, other: &Array) -> Array {
        self.gt(other)
    }

    /// Element-wise less comparison (alias for lt).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
    /// let c = a.less(&b);
    /// assert_eq!(c.to_vec(), vec![1.0, 0.0, 0.0]);
    /// ```
    pub fn less(&self, other: &Array) -> Array {
        self.lt(other)
    }

    /// Element-wise greater-or-equal comparison (alias for ge).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
    /// let c = a.greater_equal(&b);
    /// assert_eq!(c.to_vec(), vec![0.0, 1.0, 1.0]);
    /// ```
    pub fn greater_equal(&self, other: &Array) -> Array {
        self.ge(other)
    }

    /// Element-wise less-or-equal comparison (alias for le).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
    /// let c = a.less_equal(&b);
    /// assert_eq!(c.to_vec(), vec![1.0, 1.0, 0.0]);
    /// ```
    pub fn less_equal(&self, other: &Array) -> Array {
        self.le(other)
    }

    /// Test element-wise for real numbers (not infinity or NaN).
    /// For Float32, returns true for all finite values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let r = a.isreal();
    /// assert_eq!(r.to_vec(), vec![1.0, 1.0, 1.0]);
    /// ```
    pub fn isreal(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let result_data: Vec<f32> = data
            .iter()
            .map(|&x| if x.is_finite() { 1.0 } else { 0.0 })
            .collect();

        Array::from_vec(result_data, self.shape().clone())
    }

    /// Test element-wise for complex numbers.
    /// For Float32 arrays, always returns false (0.0).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let c = a.iscomplex();
    /// assert_eq!(c.to_vec(), vec![0.0, 0.0, 0.0]);
    /// ```
    pub fn iscomplex(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        Array::zeros(self.shape().clone(), DType::Float32)
    }

    /// Test element-wise if values are in an open interval.
    /// Returns 1.0 where lower < x < upper, 0.0 otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let b = a.isin_range(1.5, 4.5);
    /// assert_eq!(b.to_vec(), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    /// ```
    pub fn isin_range(&self, lower: f32, upper: f32) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let result_data: Vec<f32> = data
            .iter()
            .map(|&x| if x > lower && x < upper { 1.0 } else { 0.0 })
            .collect();

        Array::from_vec(result_data, self.shape().clone())
    }

    /// Test element-wise if values are subnormal (denormalized).
    /// Returns 1.0 where value is subnormal, 0.0 otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 0.0, 1e-40], Shape::new(vec![3]));
    /// let b = a.issubnormal();
    /// // Only 1e-40 is subnormal
    /// assert_eq!(b.to_vec()[0], 0.0);
    /// assert_eq!(b.to_vec()[1], 0.0);
    /// assert_eq!(b.to_vec()[2], 1.0);
    /// ```
    pub fn issubnormal(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let result_data: Vec<f32> = data
            .iter()
            .map(|&x| if x.is_subnormal() { 1.0 } else { 0.0 })
            .collect();

        Array::from_vec(result_data, self.shape().clone())
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

    #[test]
    fn test_allclose() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![1.0001, 2.0001, 3.0001], Shape::new(vec![3]));
        assert!(a.allclose(&b, 1e-3, 1e-3));
        assert!(!a.allclose(&b, 1e-5, 1e-5));

        // Test with broadcasting
        let c = Array::from_vec(vec![1.0001], Shape::new(vec![1]));
        let d = Array::from_vec(vec![1.0, 1.0, 1.0], Shape::new(vec![3]));
        assert!(c.allclose(&d, 1e-3, 1e-3));
    }

    #[test]
    fn test_isclose() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![1.0001, 2.1, 3.0001], Shape::new(vec![3]));
        let c = a.isclose(&b, 1e-3, 1e-3);
        assert_eq!(c.to_vec(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_array_equal() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let c = Array::from_vec(vec![1.0, 2.0, 3.1], Shape::new(vec![3]));
        assert!(a.array_equal(&b));
        assert!(!a.array_equal(&c));

        // Different shapes
        let d = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        assert!(!a.array_equal(&d));
    }

    #[test]
    fn test_array_equiv() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        assert!(a.array_equiv(&b));

        let c = Array::from_vec(vec![1.0, 2.0, 3.1], Shape::new(vec![1, 3]));
        assert!(!a.array_equiv(&c));
    }
}
