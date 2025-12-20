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

    /// Pad array with a constant value.
    ///
    /// # Arguments
    ///
    /// * `pad_width` - Number of values to pad on each side: [(before, after), ...]
    /// * `constant_value` - Value to use for padding
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let padded = a.pad(&[(1, 1)], 0.0);
    /// assert_eq!(padded.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 0.0]);
    /// ```
    pub fn pad(&self, pad_width: &[(usize, usize)], constant_value: f32) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            pad_width.len(),
            self.ndim(),
            "pad_width must match number of dimensions"
        );

        let shape = self.shape().as_slice();
        let data = self.to_vec();

        // Compute output shape
        let mut out_shape = Vec::with_capacity(shape.len());
        for (i, &dim) in shape.iter().enumerate() {
            out_shape.push(pad_width[i].0 + dim + pad_width[i].1);
        }

        // For 1D case
        if self.ndim() == 1 {
            let (before, after) = pad_width[0];
            let mut result = vec![constant_value; before];
            result.extend_from_slice(&data);
            result.extend(vec![constant_value; after]);
            return Array::from_vec(result, Shape::new(out_shape));
        }

        // For 2D case
        if self.ndim() == 2 {
            let (h, w) = (shape[0], shape[1]);
            let (h_before, h_after) = pad_width[0];
            let (w_before, w_after) = pad_width[1];

            let out_h = out_shape[0];
            let out_w = out_shape[1];
            let mut result = vec![constant_value; out_h * out_w];

            for i in 0..h {
                for j in 0..w {
                    let out_i = i + h_before;
                    let out_j = j + w_before;
                    result[out_i * out_w + out_j] = data[i * w + j];
                }
            }

            return Array::from_vec(result, Shape::new(out_shape));
        }

        panic!("pad only supports 1D and 2D arrays for now");
    }

    /// Pad array with edge values (repeat border elements).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let padded = a.pad_edge(&[(1, 1)]);
    /// assert_eq!(padded.to_vec(), vec![1.0, 1.0, 2.0, 3.0, 3.0]);
    /// ```
    pub fn pad_edge(&self, pad_width: &[(usize, usize)]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            pad_width.len(),
            self.ndim(),
            "pad_width must match number of dimensions"
        );

        let shape = self.shape().as_slice();
        let data = self.to_vec();

        // For 1D case
        if self.ndim() == 1 {
            let (before, after) = pad_width[0];
            let mut result = vec![data[0]; before];
            result.extend_from_slice(&data);
            result.extend(vec![data[data.len() - 1]; after]);

            let out_len = before + shape[0] + after;
            return Array::from_vec(result, Shape::new(vec![out_len]));
        }

        // For 2D case
        if self.ndim() == 2 {
            let (h, w) = (shape[0], shape[1]);
            let (h_before, h_after) = pad_width[0];
            let (w_before, w_after) = pad_width[1];

            let out_h = h_before + h + h_after;
            let out_w = w_before + w + w_after;
            let mut result = vec![0.0; out_h * out_w];

            for out_i in 0..out_h {
                for out_j in 0..out_w {
                    // Map output indices to input indices
                    let in_i = if out_i < h_before {
                        0
                    } else if out_i >= h_before + h {
                        h - 1
                    } else {
                        out_i - h_before
                    };

                    let in_j = if out_j < w_before {
                        0
                    } else if out_j >= w_before + w {
                        w - 1
                    } else {
                        out_j - w_before
                    };

                    result[out_i * out_w + out_j] = data[in_i * w + in_j];
                }
            }

            return Array::from_vec(result, Shape::new(vec![out_h, out_w]));
        }

        panic!("pad_edge only supports 1D and 2D arrays for now");
    }

    /// Pad array with reflected values (mirror border elements).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let padded = a.pad_reflect(&[(1, 1)]);
    /// assert_eq!(padded.to_vec(), vec![2.0, 1.0, 2.0, 3.0, 2.0]);
    /// ```
    pub fn pad_reflect(&self, pad_width: &[(usize, usize)]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            pad_width.len(),
            self.ndim(),
            "pad_width must match number of dimensions"
        );

        let shape = self.shape().as_slice();
        let data = self.to_vec();

        // For 1D case
        if self.ndim() == 1 {
            let len = shape[0];
            let (before, after) = pad_width[0];

            assert!(
                before < len && after < len,
                "Padding width must be less than array size for reflect mode"
            );

            let mut result = Vec::with_capacity(before + len + after);

            // Left padding (reflect)
            for i in 0..before {
                result.push(data[before - i]);
            }

            // Original data
            result.extend_from_slice(&data);

            // Right padding (reflect)
            for i in 0..after {
                result.push(data[len - 2 - i]);
            }

            let out_len = before + len + after;
            return Array::from_vec(result, Shape::new(vec![out_len]));
        }

        // For 2D case
        if self.ndim() == 2 {
            let (h, w) = (shape[0], shape[1]);
            let (h_before, h_after) = pad_width[0];
            let (w_before, w_after) = pad_width[1];

            assert!(
                h_before < h && h_after < h && w_before < w && w_after < w,
                "Padding width must be less than array size for reflect mode"
            );

            let out_h = h_before + h + h_after;
            let out_w = w_before + w + w_after;
            let mut result = vec![0.0; out_h * out_w];

            for out_i in 0..out_h {
                for out_j in 0..out_w {
                    // Map output indices to input indices with reflection
                    let in_i = if out_i < h_before {
                        h_before - out_i
                    } else if out_i >= h_before + h {
                        h - 2 - (out_i - h_before - h)
                    } else {
                        out_i - h_before
                    };

                    let in_j = if out_j < w_before {
                        w_before - out_j
                    } else if out_j >= w_before + w {
                        w - 2 - (out_j - w_before - w)
                    } else {
                        out_j - w_before
                    };

                    result[out_i * out_w + out_j] = data[in_i * w + in_j];
                }
            }

            return Array::from_vec(result, Shape::new(vec![out_h, out_w]));
        }

        panic!("pad_reflect only supports 1D and 2D arrays for now");
    }

    /// Replace NaN and infinity values with specified numbers.
    ///
    /// # Arguments
    ///
    /// * `nan` - Value to replace NaN with (default 0.0)
    /// * `posinf` - Value to replace positive infinity with (default large positive value)
    /// * `neginf` - Value to replace negative infinity with (default large negative value)
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, f32::INFINITY, -f32::INFINITY], Shape::new(vec![4]));
    /// let result = a.nan_to_num(0.0, 1e10, -1e10);
    /// assert_eq!(result.to_vec()[0], 1.0);
    /// assert_eq!(result.to_vec()[1], 0.0);
    /// assert_eq!(result.to_vec()[2], 1e10);
    /// assert_eq!(result.to_vec()[3], -1e10);
    /// ```
    pub fn nan_to_num(&self, nan: f32, posinf: f32, neginf: f32) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    nan
                } else if x.is_infinite() && x > 0.0 {
                    posinf
                } else if x.is_infinite() && x < 0.0 {
                    neginf
                } else {
                    x
                }
            })
            .collect();
        Array::from_vec(result, self.shape().clone())
    }

    /// Check for NaN values element-wise.
    ///
    /// Returns an array with 1.0 where NaN, 0.0 otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 3.0, f32::NAN], Shape::new(vec![4]));
    /// let result = a.isnan();
    /// assert_eq!(result.to_vec(), vec![0.0, 1.0, 0.0, 1.0]);
    /// ```
    pub fn isnan(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x.is_nan() { 1.0 } else { 0.0 })
            .collect();
        Array::from_vec(result, self.shape().clone())
    }

    /// Check for infinity values element-wise.
    ///
    /// Returns an array with 1.0 where infinity (positive or negative), 0.0 otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::INFINITY, -f32::INFINITY, 3.0], Shape::new(vec![4]));
    /// let result = a.isinf();
    /// assert_eq!(result.to_vec(), vec![0.0, 1.0, 1.0, 0.0]);
    /// ```
    pub fn isinf(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x.is_infinite() { 1.0 } else { 0.0 })
            .collect();
        Array::from_vec(result, self.shape().clone())
    }

    /// Check for finite values element-wise.
    ///
    /// Returns an array with 1.0 where finite, 0.0 otherwise (NaN or infinity).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, f32::INFINITY, 3.0], Shape::new(vec![4]));
    /// let result = a.isfinite();
    /// assert_eq!(result.to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn isfinite(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| if x.is_finite() { 1.0 } else { 0.0 })
            .collect();
        Array::from_vec(result, self.shape().clone())
    }

    /// Clip array values by L2 norm.
    ///
    /// If the L2 norm exceeds max_norm, scales the array down to have that norm.
    /// Useful for gradient clipping in neural networks.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
    /// let clipped = a.clip_by_norm(2.0);
    /// // Original norm is 5.0, should be scaled to 2.0
    /// let result = clipped.to_vec();
    /// assert!((result[0] - 1.2).abs() < 1e-5);
    /// assert!((result[1] - 1.6).abs() < 1e-5);
    /// ```
    pub fn clip_by_norm(&self, max_norm: f32) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();

        // Compute L2 norm
        let norm: f32 = data.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm <= max_norm {
            return self.clone();
        }

        // Scale down to max_norm
        let scale = max_norm / norm;
        let result: Vec<f32> = data.iter().map(|&x| x * scale).collect();
        Array::from_vec(result, self.shape().clone())
    }

    /// Flatten array to 1D.
    ///
    /// Returns a 1D array containing all elements in row-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let flat = a.ravel();
    /// assert_eq!(flat.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(flat.shape().as_slice(), &[4]);
    /// ```
    pub fn ravel(&self) -> Array {
        let size = self.size();
        Array::from_vec(self.to_vec(), Shape::new(vec![size]))
    }

    /// Flatten array to 1D (alias for ravel).
    ///
    /// Returns a 1D array containing all elements in row-major order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let flat = a.flatten();
    /// assert_eq!(flat.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn flatten(&self) -> Array {
        self.ravel()
    }

    /// View array with at least 1D.
    ///
    /// Scalar (0D) arrays are converted to 1D arrays with shape [1].
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![5.0], Shape::new(vec![]));
    /// let b = a.atleast_1d();
    /// assert_eq!(b.shape().as_slice(), &[1]);
    /// ```
    pub fn atleast_1d(&self) -> Array {
        if self.shape().ndim() == 0 {
            Array::from_vec(self.to_vec(), Shape::new(vec![1]))
        } else {
            self.clone()
        }
    }

    /// View array with at least 2D.
    ///
    /// Arrays with fewer than 2 dimensions are expanded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.atleast_2d();
    /// assert_eq!(b.shape().as_slice(), &[1, 3]);
    /// ```
    pub fn atleast_2d(&self) -> Array {
        match self.shape().ndim() {
            0 => Array::from_vec(self.to_vec(), Shape::new(vec![1, 1])),
            1 => {
                let n = self.shape().as_slice()[0];
                Array::from_vec(self.to_vec(), Shape::new(vec![1, n]))
            }
            _ => self.clone(),
        }
    }

    /// View array with at least 3D.
    ///
    /// Arrays with fewer than 3 dimensions are expanded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = a.atleast_3d();
    /// assert_eq!(b.shape().as_slice(), &[1, 2, 1]);
    /// ```
    pub fn atleast_3d(&self) -> Array {
        match self.shape().ndim() {
            0 => Array::from_vec(self.to_vec(), Shape::new(vec![1, 1, 1])),
            1 => {
                let n = self.shape().as_slice()[0];
                Array::from_vec(self.to_vec(), Shape::new(vec![1, n, 1]))
            }
            2 => {
                let dims = self.shape().as_slice();
                Array::from_vec(self.to_vec(), Shape::new(vec![dims[0], dims[1], 1]))
            }
            _ => self.clone(),
        }
    }

    /// Broadcast array to a new shape.
    ///
    /// The new shape must be broadcast-compatible with the current shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.broadcast_to(Shape::new(vec![2, 3]));
    /// assert_eq!(b.shape().as_slice(), &[2, 3]);
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    /// ```
    pub fn broadcast_to(&self, new_shape: Shape) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        // Check broadcast compatibility
        let _result_shape = self
            .shape()
            .broadcast_with(&new_shape)
            .expect("Shapes are not broadcast-compatible");

        let data = self.to_vec();
        let size = new_shape.size();
        let mut result = Vec::with_capacity(size);

        for i in 0..size {
            let src_idx =
                crate::ops::binary::broadcast_index(i, &new_shape, self.shape());
            result.push(data[src_idx]);
        }

        Array::from_vec(result, new_shape)
    }

    /// Take elements from array along an axis at specified indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::new(vec![4]));
    /// let indices = vec![0, 2, 3];
    /// let result = a.take(&indices);
    /// assert_eq!(result.to_vec(), vec![10.0, 30.0, 40.0]);
    /// ```
    pub fn take(&self, indices: &[usize]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();

        let result: Vec<f32> = indices
            .iter()
            .map(|&idx| {
                assert!(idx < data.len(), "Index {} out of bounds", idx);
                data[idx]
            })
            .collect();

        let len = result.len();
        Array::from_vec(result, Shape::new(vec![len]))
    }

    /// Return indices of non-zero elements.
    ///
    /// Returns indices where elements are non-zero (not equal to 0.0).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0, 0.0, 3.0, 0.0], Shape::new(vec![5]));
    /// let indices = a.nonzero();
    /// assert_eq!(indices, vec![1, 3]);
    /// ```
    pub fn nonzero(&self) -> Vec<usize> {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();

        data.iter()
            .enumerate()
            .filter(|(_, &val)| val != 0.0)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Return indices where condition is true (non-zero).
    ///
    /// Similar to nonzero but returns 2D array of indices for multi-dimensional arrays.
    /// For 1D arrays, returns a list of indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0, 0.0, 1.0], Shape::new(vec![4]));
    /// let indices = a.argwhere();
    /// assert_eq!(indices, vec![1, 3]);
    /// ```
    pub fn argwhere(&self) -> Vec<usize> {
        self.nonzero()
    }

    /// Select elements from array based on condition mask.
    ///
    /// Returns a 1D array of elements where the condition is true (non-zero).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::new(vec![4]));
    /// let condition = Array::from_vec(vec![1.0, 0.0, 1.0, 0.0], Shape::new(vec![4]));
    /// let result = a.compress(&condition);
    /// assert_eq!(result.to_vec(), vec![10.0, 30.0]);
    /// ```
    pub fn compress(&self, condition: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            condition.dtype(),
            DType::Float32,
            "Only Float32 supported"
        );
        assert_eq!(
            self.size(),
            condition.size(),
            "Array and condition must have same size"
        );

        let data = self.to_vec();
        let cond_data = condition.to_vec();

        let result: Vec<f32> = data
            .iter()
            .zip(cond_data.iter())
            .filter(|(_, &c)| c != 0.0)
            .map(|(&val, _)| val)
            .collect();

        let len = result.len();
        Array::from_vec(result, Shape::new(vec![len]))
    }

    /// Choose elements from arrays based on index array.
    ///
    /// For each element in the index array, select from the corresponding choice array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let choices = vec![
    ///     Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3])),
    ///     Array::from_vec(vec![100.0, 200.0, 300.0], Shape::new(vec![3])),
    /// ];
    /// let indices = vec![0, 1, 0];
    /// let result = Array::choose(&indices, &choices);
    /// assert_eq!(result.to_vec(), vec![10.0, 200.0, 30.0]);
    /// ```
    pub fn choose(indices: &[usize], choices: &[Array]) -> Array {
        assert!(!choices.is_empty(), "Must provide at least one choice");
        let size = choices[0].size();

        for choice in choices.iter() {
            assert_eq!(
                choice.size(),
                size,
                "All choices must have the same size"
            );
        }

        assert_eq!(
            indices.len(),
            size,
            "Indices must have same length as choices"
        );

        let choice_data: Vec<Vec<f32>> =
            choices.iter().map(|c| c.to_vec()).collect();

        let result: Vec<f32> = (0..size)
            .map(|i| {
                let choice_idx = indices[i];
                assert!(
                    choice_idx < choices.len(),
                    "Index {} out of bounds",
                    choice_idx
                );
                choice_data[choice_idx][i]
            })
            .collect();

        Array::from_vec(result, choices[0].shape().clone())
    }

    /// Extract elements from array where condition is true.
    ///
    /// Similar to compress, but condition can be a boolean-like array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let condition = Array::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0], Shape::new(vec![5]));
    /// let result = a.extract(&condition);
    /// assert_eq!(result.to_vec(), vec![1.0, 3.0, 5.0]);
    /// ```
    pub fn extract(&self, condition: &Array) -> Array {
        self.compress(condition)
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

    #[test]
    fn test_nan_to_num() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, f32::INFINITY, -f32::INFINITY, 5.0],
            Shape::new(vec![5]),
        );
        let result = a.nan_to_num(0.0, 1e10, -1e10);
        assert_eq!(result.to_vec()[0], 1.0);
        assert_eq!(result.to_vec()[1], 0.0);
        assert_eq!(result.to_vec()[2], 1e10);
        assert_eq!(result.to_vec()[3], -1e10);
        assert_eq!(result.to_vec()[4], 5.0);
    }

    #[test]
    fn test_isnan() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0],
            Shape::new(vec![5]),
        );
        let result = a.isnan();
        assert_eq!(result.to_vec(), vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_isinf() {
        let a = Array::from_vec(
            vec![1.0, f32::INFINITY, -f32::INFINITY, 3.0],
            Shape::new(vec![4]),
        );
        let result = a.isinf();
        assert_eq!(result.to_vec(), vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_isfinite() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, f32::INFINITY, 3.0],
            Shape::new(vec![4]),
        );
        let result = a.isfinite();
        assert_eq!(result.to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_clip_by_norm() {
        // Test case 1: norm exceeds max_norm
        let a = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
        let clipped = a.clip_by_norm(2.0);
        let result = clipped.to_vec();
        // Original norm is 5.0, should be scaled to 2.0
        // scale = 2.0 / 5.0 = 0.4
        assert!((result[0] - 1.2).abs() < 1e-5);
        assert!((result[1] - 1.6).abs() < 1e-5);

        // Test case 2: norm is already below max_norm
        let b = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));
        let clipped2 = b.clip_by_norm(5.0);
        assert_eq!(clipped2.to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_ravel() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let flat = a.ravel();
        assert_eq!(flat.shape().as_slice(), &[6]);
        assert_eq!(flat.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_flatten() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let flat = a.flatten();
        assert_eq!(flat.shape().as_slice(), &[4]);
        assert_eq!(flat.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_atleast_1d() {
        // Scalar to 1D
        let a = Array::from_vec(vec![5.0], Shape::new(vec![]));
        let b = a.atleast_1d();
        assert_eq!(b.shape().as_slice(), &[1]);

        // Already 1D
        let c = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let d = c.atleast_1d();
        assert_eq!(d.shape().as_slice(), &[2]);
    }

    #[test]
    fn test_atleast_2d() {
        // 1D to 2D
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = a.atleast_2d();
        assert_eq!(b.shape().as_slice(), &[1, 3]);

        // Already 2D
        let c = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        let d = c.atleast_2d();
        assert_eq!(d.shape().as_slice(), &[1, 2]);
    }

    #[test]
    fn test_atleast_3d() {
        // 1D to 3D
        let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let b = a.atleast_3d();
        assert_eq!(b.shape().as_slice(), &[1, 2, 1]);

        // 2D to 3D
        let c = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![1, 2]));
        let d = c.atleast_3d();
        assert_eq!(d.shape().as_slice(), &[1, 2, 1]);
    }

    #[test]
    fn test_broadcast_to() {
        // Broadcast 1D to 2D
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = a.broadcast_to(Shape::new(vec![2, 3]));
        assert_eq!(b.shape().as_slice(), &[2, 3]);
        assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_take() {
        let a = Array::from_vec(
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            Shape::new(vec![5]),
        );
        let indices = vec![0, 2, 4];
        let result = a.take(&indices);
        assert_eq!(result.to_vec(), vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_nonzero() {
        let a = Array::from_vec(
            vec![0.0, 1.0, 0.0, 3.0, 0.0, 5.0],
            Shape::new(vec![6]),
        );
        let indices = a.nonzero();
        assert_eq!(indices, vec![1, 3, 5]);
    }

    #[test]
    fn test_argwhere() {
        let a = Array::from_vec(vec![0.0, 1.0, 0.0, 1.0], Shape::new(vec![4]));
        let indices = a.argwhere();
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    fn test_compress() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::new(vec![4]));
        let condition =
            Array::from_vec(vec![1.0, 0.0, 1.0, 0.0], Shape::new(vec![4]));
        let result = a.compress(&condition);
        assert_eq!(result.to_vec(), vec![10.0, 30.0]);
    }

    #[test]
    fn test_choose() {
        let choices = vec![
            Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3])),
            Array::from_vec(vec![100.0, 200.0, 300.0], Shape::new(vec![3])),
        ];
        let indices = vec![0, 1, 0];
        let result = Array::choose(&indices, &choices);
        assert_eq!(result.to_vec(), vec![10.0, 200.0, 30.0]);
    }

    #[test]
    fn test_extract() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::new(vec![5]),
        );
        let condition = Array::from_vec(
            vec![1.0, 0.0, 1.0, 0.0, 1.0],
            Shape::new(vec![5]),
        );
        let result = a.extract(&condition);
        assert_eq!(result.to_vec(), vec![1.0, 3.0, 5.0]);
    }
}
