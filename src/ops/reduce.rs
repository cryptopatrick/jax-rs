//! Reduction operations on arrays.

use crate::{buffer::Buffer, Array, DType, Device, Shape};

/// Reduce over all elements with a binary operation.
fn reduce_all<F>(input: &Array, init: f32, f: F) -> f32
where
    F: Fn(f32, f32) -> f32,
{
    assert_eq!(input.dtype(), DType::Float32, "Only Float32 supported");
    let data = input.to_vec();
    data.iter().fold(init, |acc, &x| f(acc, x))
}

/// Reduce along a specific axis.
fn reduce_axis<F>(input: &Array, axis: usize, init: f32, f: F) -> Array
where
    F: Fn(f32, f32) -> f32,
{
    assert_eq!(input.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(input.device(), Device::Cpu, "Only CPU supported for now");
    assert!(axis < input.ndim(), "Axis out of bounds");

    let shape = input.shape();
    let dims = shape.as_slice();

    // Result shape has the reduced axis removed
    let mut result_dims: Vec<usize> = dims.to_vec();
    result_dims.remove(axis);
    let result_shape = if result_dims.is_empty() {
        Shape::scalar()
    } else {
        Shape::new(result_dims.clone())
    };

    let input_data = input.to_vec();
    let result_size = result_shape.size();
    let mut result_data = vec![init; result_size];

    // Compute strides for input
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    // Iterate over result indices
    for (result_idx, item) in result_data.iter_mut().enumerate() {
        // Convert flat result index to multi-dimensional
        let mut result_multi = vec![0; result_dims.len()];
        let mut idx = result_idx;
        for i in (0..result_dims.len()).rev() {
            result_multi[i] = idx % result_shape.as_slice()[i];
            idx /= result_shape.as_slice()[i];
        }

        // Insert the reduced axis and iterate over it
        let mut acc = init;
        for axis_idx in 0..dims[axis] {
            let mut input_multi = Vec::with_capacity(dims.len());
            let mut result_i = 0;
            for i in 0..dims.len() {
                if i == axis {
                    input_multi.push(axis_idx);
                } else {
                    input_multi.push(result_multi[result_i]);
                    result_i += 1;
                }
            }

            // Convert multi-dimensional index to flat
            let flat_idx: usize = input_multi
                .iter()
                .zip(strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();

            acc = f(acc, input_data[flat_idx]);
        }

        *item = acc;
    }

    let buffer = Buffer::from_f32(result_data, Device::Cpu);
    Array::from_buffer(buffer, result_shape)
}

impl Array {
    /// Sum of all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let sum = a.sum_all();
    /// assert_eq!(sum, 10.0);
    /// ```
    pub fn sum_all(&self) -> f32 {
        reduce_all(self, 0.0, |acc, x| acc + x)
    }

    /// Sum of all elements, returned as a scalar Array.
    ///
    /// This is a convenience method for autodiff that wraps `sum_all()`.
    pub fn sum_all_array(&self) -> Array {
        let val = self.sum_all();
        Array::from_vec(vec![val], crate::Shape::scalar())
    }

    /// Sum along a specific axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    /// let sum_axis0 = a.sum(0);
    /// assert_eq!(sum_axis0.to_vec(), vec![5.0, 7.0, 9.0]);
    /// let sum_axis1 = a.sum(1);
    /// assert_eq!(sum_axis1.to_vec(), vec![6.0, 15.0]);
    /// ```
    pub fn sum(&self, axis: usize) -> Array {
        reduce_axis(self, axis, 0.0, |acc, x| acc + x)
    }

    /// Mean of all elements.
    pub fn mean_all(&self) -> f32 {
        self.sum_all() / (self.size() as f32)
    }

    /// Mean along a specific axis.
    pub fn mean(&self, axis: usize) -> Array {
        let sum = self.sum(axis);
        let count = self.shape().as_slice()[axis] as f32;
        let data: Vec<f32> = sum.to_vec().iter().map(|&x| x / count).collect();
        let buffer = Buffer::from_f32(data, Device::Cpu);
        Array::from_buffer(buffer, sum.shape().clone())
    }

    /// Maximum of all elements.
    pub fn max_all(&self) -> f32 {
        reduce_all(self, f32::NEG_INFINITY, |acc, x| acc.max(x))
    }

    /// Maximum along a specific axis.
    pub fn max(&self, axis: usize) -> Array {
        reduce_axis(self, axis, f32::NEG_INFINITY, |acc, x| acc.max(x))
    }

    /// Minimum of all elements.
    pub fn min_all(&self) -> f32 {
        reduce_all(self, f32::INFINITY, |acc, x| acc.min(x))
    }

    /// Minimum along a specific axis.
    pub fn min(&self, axis: usize) -> Array {
        reduce_axis(self, axis, f32::INFINITY, |acc, x| acc.min(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sum_all() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        assert_eq!(a.sum_all(), 10.0);
    }

    #[test]
    fn test_sum_axis() {
        // 2x3 array: [[1, 2, 3], [4, 5, 6]]
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        // Sum along axis 0 (collapse rows): [5, 7, 9]
        let sum_axis0 = a.sum(0);
        assert_eq!(sum_axis0.shape().as_slice(), &[3]);
        assert_eq!(sum_axis0.to_vec(), vec![5.0, 7.0, 9.0]);

        // Sum along axis 1 (collapse columns): [6, 15]
        let sum_axis1 = a.sum(1);
        assert_eq!(sum_axis1.shape().as_slice(), &[2]);
        assert_eq!(sum_axis1.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_mean_all() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        assert_abs_diff_eq!(a.mean_all(), 2.5, epsilon = 1e-6);
    }

    #[test]
    fn test_mean_axis() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        let mean_axis0 = a.mean(0);
        assert_eq!(mean_axis0.to_vec(), vec![2.5, 3.5, 4.5]);

        let mean_axis1 = a.mean(1);
        assert_eq!(mean_axis1.to_vec(), vec![2.0, 5.0]);
    }

    #[test]
    fn test_max_all() {
        let a = Array::from_vec(vec![1.0, 5.0, 3.0, 2.0], Shape::new(vec![4]));
        assert_eq!(a.max_all(), 5.0);
    }

    #[test]
    fn test_max_axis() {
        let a = Array::from_vec(
            vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        let max_axis0 = a.max(0);
        assert_eq!(max_axis0.to_vec(), vec![2.0, 5.0, 6.0]);

        let max_axis1 = a.max(1);
        assert_eq!(max_axis1.to_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_min_all() {
        let a = Array::from_vec(vec![3.0, 1.0, 5.0, 2.0], Shape::new(vec![4]));
        assert_eq!(a.min_all(), 1.0);
    }

    #[test]
    fn test_min_axis() {
        let a = Array::from_vec(
            vec![3.0, 5.0, 2.0, 1.0, 4.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        let min_axis0 = a.min(0);
        assert_eq!(min_axis0.to_vec(), vec![1.0, 4.0, 2.0]);

        let min_axis1 = a.min(1);
        assert_eq!(min_axis1.to_vec(), vec![2.0, 1.0]);
    }

    #[test]
    fn test_reduce_3d() {
        // 2x2x2 array
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new(vec![2, 2, 2]),
        );

        // Sum along middle axis
        let sum_axis1 = a.sum(1);
        assert_eq!(sum_axis1.shape().as_slice(), &[2, 2]);
        assert_eq!(sum_axis1.to_vec(), vec![4.0, 6.0, 12.0, 14.0]);
    }
}
