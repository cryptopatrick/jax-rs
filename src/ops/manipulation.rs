//! Array manipulation operations.
//!
//! Operations for reshaping, concatenating, and manipulating arrays.

use crate::{buffer::Buffer, Array, DType, Device, Shape};

impl Array {
    /// Concatenate arrays along an existing axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
    /// let c = Array::concatenate(&[a, b], 0);
    /// assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn concatenate(arrays: &[Array], axis: usize) -> Array {
        assert!(!arrays.is_empty(), "Need at least one array to concatenate");
        assert_eq!(
            arrays[0].dtype(),
            DType::Float32,
            "Only Float32 supported"
        );

        // Validate all arrays have compatible shapes
        let first_shape = arrays[0].shape();
        let ndim = first_shape.ndim();
        assert!(axis < ndim, "Axis out of bounds");

        for arr in arrays.iter().skip(1) {
            assert_eq!(
                arr.ndim(),
                ndim,
                "All arrays must have same number of dimensions"
            );
            for (i, (&dim1, &dim2)) in first_shape
                .as_slice()
                .iter()
                .zip(arr.shape().as_slice().iter())
                .enumerate()
            {
                if i != axis {
                    assert_eq!(
                        dim1, dim2,
                        "Dimensions must match except on concatenation axis"
                    );
                }
            }
        }

        // Compute result shape
        let mut result_dims = first_shape.as_slice().to_vec();
        result_dims[axis] =
            arrays.iter().map(|a| a.shape().as_slice()[axis]).sum();
        let result_shape = Shape::new(result_dims);

        // Simple implementation for axis 0
        if axis == 0 {
            let mut data = Vec::new();
            for arr in arrays {
                data.extend(arr.to_vec());
            }
            Array::from_vec(data, result_shape)
        } else {
            // For other axes, need more complex indexing
            // For now, only support axis 0
            panic!("concatenate only supports axis=0 for now");
        }
    }

    /// Stack arrays along a new axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
    /// let c = Array::stack(&[a, b], 0);
    /// assert_eq!(c.shape().as_slice(), &[2, 2]);
    /// assert_eq!(c.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn stack(arrays: &[Array], axis: usize) -> Array {
        assert!(!arrays.is_empty(), "Need at least one array to stack");

        // Validate all arrays have the same shape
        let first_shape = arrays[0].shape();
        for arr in arrays.iter().skip(1) {
            assert_eq!(
                arr.shape(),
                first_shape,
                "All arrays must have the same shape for stacking"
            );
        }

        let ndim = first_shape.ndim();
        assert!(axis <= ndim, "Axis out of bounds for stacking");

        // Compute result shape
        let mut result_dims = Vec::new();
        for (i, &dim) in first_shape.as_slice().iter().enumerate() {
            if i == axis {
                result_dims.push(arrays.len());
            }
            result_dims.push(dim);
        }
        if axis == ndim {
            result_dims.push(arrays.len());
        }

        // Simple implementation for axis 0
        if axis == 0 {
            let mut data = Vec::new();
            for arr in arrays {
                data.extend(arr.to_vec());
            }
            let mut shape_dims = vec![arrays.len()];
            shape_dims.extend_from_slice(first_shape.as_slice());
            Array::from_vec(data, Shape::new(shape_dims))
        } else {
            panic!("stack only supports axis=0 for now");
        }
    }

    /// Select elements from array based on condition.
    ///
    /// Returns elements from `x` where `condition` is true, otherwise from `y`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let condition = Array::from_vec(vec![1.0, 0.0, 1.0], Shape::new(vec![3]));
    /// let x = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
    /// let y = Array::from_vec(vec![5.0, 5.0, 5.0], Shape::new(vec![3]));
    /// let result = Array::where_cond(&condition, &x, &y);
    /// assert_eq!(result.to_vec(), vec![10.0, 5.0, 30.0]);
    /// ```
    pub fn where_cond(condition: &Array, x: &Array, y: &Array) -> Array {
        assert_eq!(
            condition.dtype(),
            DType::Float32,
            "Only Float32 supported"
        );
        assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(y.dtype(), DType::Float32, "Only Float32 supported");

        assert_eq!(
            condition.shape(),
            x.shape(),
            "Condition and x must have same shape"
        );
        assert_eq!(x.shape(), y.shape(), "x and y must have same shape");

        let cond_data = condition.to_vec();
        let x_data = x.to_vec();
        let y_data = y.to_vec();

        let result_data: Vec<f32> = cond_data
            .iter()
            .zip(x_data.iter())
            .zip(y_data.iter())
            .map(|((&c, &xv), &yv)| if c != 0.0 { xv } else { yv })
            .collect();

        Array::from_vec(result_data, x.shape().clone())
    }

    /// Clip (limit) values in an array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 5.0, 10.0, 15.0], Shape::new(vec![4]));
    /// let clipped = a.clip(3.0, 12.0);
    /// assert_eq!(clipped.to_vec(), vec![3.0, 5.0, 10.0, 12.0]);
    /// ```
    pub fn clip(&self, min: f32, max: f32) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let result_data: Vec<f32> =
            data.iter().map(|&x| x.clamp(min, max)).collect();

        Array::from_vec(result_data, self.shape().clone())
    }

    /// Flip array along specified axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let flipped = a.flip(0);
    /// assert_eq!(flipped.to_vec(), vec![3.0, 2.0, 1.0]);
    /// ```
    pub fn flip(&self, axis: usize) -> Array {
        assert!(axis < self.ndim(), "Axis out of bounds");
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let shape = self.shape();
        let dims = shape.as_slice();
        let data = self.to_vec();

        // Simple implementation for 1D arrays
        if self.ndim() == 1 {
            let mut result: Vec<f32> = data.clone();
            result.reverse();
            return Array::from_vec(result, shape.clone());
        }

        // For multi-dimensional, only support flipping axis 0 for now
        if axis == 0 {
            let slice_size = data.len() / dims[0];
            let mut result = Vec::with_capacity(data.len());

            for i in (0..dims[0]).rev() {
                let start = i * slice_size;
                let end = start + slice_size;
                result.extend_from_slice(&data[start..end]);
            }

            Array::from_vec(result, shape.clone())
        } else {
            panic!("flip only supports axis=0 for multi-dimensional arrays");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concatenate_1d() {
        let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
        let c = Array::from_vec(vec![5.0, 6.0], Shape::new(vec![2]));

        let result = Array::concatenate(&[a, b, c], 0);
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_stack() {
        let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));

        let result = Array::stack(&[a, b], 0);
        assert_eq!(result.shape().as_slice(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_where_cond() {
        let cond =
            Array::from_vec(vec![1.0, 0.0, 1.0, 0.0], Shape::new(vec![4]));
        let x =
            Array::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::new(vec![4]));
        let y = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));

        let result = Array::where_cond(&cond, &x, &y);
        assert_eq!(result.to_vec(), vec![10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn test_clip() {
        let a = Array::from_vec(
            vec![-5.0, 0.0, 5.0, 10.0, 15.0],
            Shape::new(vec![5]),
        );
        let clipped = a.clip(0.0, 10.0);
        assert_eq!(clipped.to_vec(), vec![0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn test_flip_1d() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let flipped = a.flip(0);
        assert_eq!(flipped.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_2d() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![3, 2]),
        );
        let flipped = a.flip(0);
        assert_eq!(flipped.to_vec(), vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }
}
