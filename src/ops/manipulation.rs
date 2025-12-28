//! Array manipulation operations.
//!
//! Operations for reshaping, concatenating, and manipulating arrays.

use crate::{Array, DType, Shape};

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
        let result_shape = Shape::new(result_dims.clone());

        // Simple implementation for axis 0
        if axis == 0 {
            let mut data = Vec::new();
            for arr in arrays {
                data.extend(arr.to_vec());
            }
            Array::from_vec(data, result_shape)
        } else {
            // General implementation for any axis
            // Compute strides for the result array
            let total_size: usize = result_dims.iter().product();
            let mut result = vec![0.0f32; total_size];

            // Compute the size of chunks before and after the concatenation axis
            let outer_size: usize = result_dims[..axis].iter().product();
            let inner_size: usize = result_dims[axis + 1..].iter().product();

            let mut offset_along_axis = 0;
            for arr in arrays {
                let arr_data = arr.to_vec();
                let arr_shape = arr.shape().as_slice();
                let arr_axis_size = arr_shape[axis];

                for outer in 0..outer_size {
                    for ax in 0..arr_axis_size {
                        for inner in 0..inner_size {
                            let src_idx = outer * arr_axis_size * inner_size + ax * inner_size + inner;
                            let dst_ax = offset_along_axis + ax;
                            let dst_idx = outer * result_dims[axis] * inner_size + dst_ax * inner_size + inner;
                            result[dst_idx] = arr_data[src_idx];
                        }
                    }
                }
                offset_along_axis += arr_axis_size;
            }

            Array::from_vec(result, result_shape)
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

    /// Split an array into multiple sub-arrays along a specified axis.
    ///
    /// # Arguments
    ///
    /// * `array` - The array to split
    /// * `num_sections` - Number of equal sections to split into
    /// * `axis` - The axis along which to split
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![6]));
    /// let parts = Array::split(&a, 3, 0);
    /// assert_eq!(parts.len(), 3);
    /// assert_eq!(parts[0].to_vec(), vec![1.0, 2.0]);
    /// assert_eq!(parts[1].to_vec(), vec![3.0, 4.0]);
    /// assert_eq!(parts[2].to_vec(), vec![5.0, 6.0]);
    /// ```
    pub fn split(array: &Array, num_sections: usize, axis: usize) -> Vec<Array> {
        assert_eq!(array.dtype(), DType::Float32, "Only Float32 supported");
        assert!(num_sections > 0, "Number of sections must be positive");

        let shape = array.shape().as_slice();
        assert!(axis < shape.len(), "Axis out of bounds");
        let axis_size = shape[axis];
        assert_eq!(
            axis_size % num_sections,
            0,
            "Array size along axis must be divisible by number of sections"
        );

        let section_size = axis_size / num_sections;
        let data = array.to_vec();

        let mut result = Vec::with_capacity(num_sections);

        if axis == 0 {
            // Split along axis 0 - simple case
            let elements_per_section = data.len() / num_sections;

            for i in 0..num_sections {
                let start = i * elements_per_section;
                let end = start + elements_per_section;
                let section_data = data[start..end].to_vec();

                let mut section_shape = shape.to_vec();
                section_shape[axis] = section_size;

                result.push(Array::from_vec(section_data, Shape::new(section_shape)));
            }
        } else {
            // General case for any axis
            let outer_size: usize = shape[..axis].iter().product();
            let inner_size: usize = shape[axis + 1..].iter().product();

            for section_idx in 0..num_sections {
                let mut section_data = Vec::with_capacity(outer_size * section_size * inner_size);

                for outer in 0..outer_size {
                    for ax in 0..section_size {
                        let src_ax = section_idx * section_size + ax;
                        for inner in 0..inner_size {
                            let src_idx = outer * axis_size * inner_size + src_ax * inner_size + inner;
                            section_data.push(data[src_idx]);
                        }
                    }
                }

                let mut section_shape = shape.to_vec();
                section_shape[axis] = section_size;

                result.push(Array::from_vec(section_data, Shape::new(section_shape)));
            }
        }

        result
    }

    /// Select elements from array based on condition with broadcasting support.
    ///
    /// Returns elements from `x` where `condition` is true (non-zero), otherwise from `y`.
    /// All three arrays are broadcast to a common shape.
    ///
    /// # Arguments
    ///
    /// * `condition` - Boolean array (non-zero = true, zero = false)
    /// * `x` - Array of values to select when condition is true
    /// * `y` - Array of values to select when condition is false
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
        assert_eq!(condition.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(y.dtype(), DType::Float32, "Only Float32 supported");

        // Compute common broadcast shape for all three arrays
        let shape1 = condition
            .shape()
            .broadcast_with(x.shape())
            .expect("Condition and x shapes are not broadcast-compatible");
        let result_shape = shape1
            .broadcast_with(y.shape())
            .expect("Cannot broadcast all three arrays to common shape");

        let cond_data = condition.to_vec();
        let x_data = x.to_vec();
        let y_data = y.to_vec();

        // Fast path: all arrays have the same shape (no broadcasting needed)
        if condition.shape() == x.shape()
            && x.shape() == y.shape()
            && condition.shape() == &result_shape
        {
            let result_data: Vec<f32> = cond_data
                .iter()
                .zip(x_data.iter().zip(y_data.iter()))
                .map(|(&c, (&x_val, &y_val))| if c != 0.0 { x_val } else { y_val })
                .collect();
            return Array::from_vec(result_data, result_shape);
        }

        // Slow path: need broadcasting
        let size = result_shape.size();
        let result_data: Vec<f32> = (0..size)
            .map(|i| {
                let cond_idx =
                    crate::ops::binary::broadcast_index(i, &result_shape, condition.shape());
                let x_idx = crate::ops::binary::broadcast_index(i, &result_shape, x.shape());
                let y_idx = crate::ops::binary::broadcast_index(i, &result_shape, y.shape());

                if cond_data[cond_idx] != 0.0 {
                    x_data[x_idx]
                } else {
                    y_data[y_idx]
                }
            })
            .collect();

        Array::from_vec(result_data, result_shape)
    }

    /// Select values from multiple choice arrays based on index array.
    ///
    /// For each element in `indices`, selects the corresponding element from
    /// the choice array at that index. Similar to a multi-way switch statement.
    ///
    /// # Arguments
    ///
    /// * `indices` - Array of integer indices (as f32) specifying which choice to pick
    /// * `choices` - Slice of arrays to choose from
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let indices = Array::from_vec(vec![0.0, 1.0, 2.0, 1.0], Shape::new(vec![4]));
    /// let choice0 = Array::from_vec(vec![10.0, 10.0, 10.0, 10.0], Shape::new(vec![4]));
    /// let choice1 = Array::from_vec(vec![20.0, 20.0, 20.0, 20.0], Shape::new(vec![4]));
    /// let choice2 = Array::from_vec(vec![30.0, 30.0, 30.0, 30.0], Shape::new(vec![4]));
    /// let result = Array::select(&indices, &[choice0, choice1, choice2]);
    /// assert_eq!(result.to_vec(), vec![10.0, 20.0, 30.0, 20.0]);
    /// ```
    pub fn select(indices: &Array, choices: &[Array]) -> Array {
        assert_eq!(indices.dtype(), DType::Float32, "Only Float32 supported");
        assert!(!choices.is_empty(), "Must provide at least one choice");

        // All choices must have the same shape
        let choice_shape = choices[0].shape();
        for choice in choices.iter().skip(1) {
            assert_eq!(
                choice.dtype(),
                DType::Float32,
                "Only Float32 supported for choices"
            );
            assert_eq!(
                choice.shape(),
                choice_shape,
                "All choices must have the same shape"
            );
        }

        // Indices and choices must have compatible shapes
        assert_eq!(
            indices.shape(),
            choice_shape,
            "Indices and choices must have the same shape"
        );

        let indices_data = indices.to_vec();
        let choice_data: Vec<Vec<f32>> = choices.iter().map(|c| c.to_vec()).collect();

        let result_data: Vec<f32> = indices_data
            .iter()
            .enumerate()
            .map(|(i, &idx)| {
                let idx_int = idx as usize;
                assert!(
                    idx_int < choices.len(),
                    "Index {} out of bounds for {} choices",
                    idx_int,
                    choices.len()
                );
                choice_data[idx_int][i]
            })
            .collect();

        Array::from_vec(result_data, choice_shape.clone())
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
            let (h_before, _) = pad_width[0];
            let (w_before, _) = pad_width[1];

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

        // Check broadcast compatibility with better error message
        let _result_shape = self
            .shape()
            .broadcast_with(&new_shape)
            .unwrap_or_else(|| {
                panic!(
                    "Cannot broadcast array of shape {:?} to shape {:?}. \
                     Broadcasting requires dimensions to be equal or one of them to be 1.",
                    self.shape().as_slice(),
                    new_shape.as_slice()
                )
            });

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

    /// Broadcast multiple arrays to a common shape.
    ///
    /// All arrays are broadcast to a shape that is compatible with all inputs.
    /// The result shape is determined by the broadcast rules applied successively.
    ///
    /// # Arguments
    ///
    /// * `arrays` - Slice of arrays to broadcast
    ///
    /// # Returns
    ///
    /// Vector of arrays, all broadcast to the same shape
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![10.0, 20.0], Shape::new(vec![2, 1]));
    /// let broadcasted = Array::broadcast_arrays(&[a, b]);
    ///
    /// // Both should have shape [2, 3]
    /// assert_eq!(broadcasted[0].shape().as_slice(), &[2, 3]);
    /// assert_eq!(broadcasted[1].shape().as_slice(), &[2, 3]);
    /// ```
    pub fn broadcast_arrays(arrays: &[Array]) -> Vec<Array> {
        if arrays.is_empty() {
            return vec![];
        }

        if arrays.len() == 1 {
            return vec![arrays[0].clone()];
        }

        // Find the common broadcast shape
        let mut common_shape = arrays[0].shape().clone();

        for array in &arrays[1..] {
            common_shape = common_shape
                .broadcast_with(array.shape())
                .unwrap_or_else(|| {
                    panic!(
                        "Cannot broadcast arrays with shapes {:?} and {:?}. \
                         Broadcasting requires dimensions to be equal or one of them to be 1.",
                        common_shape.as_slice(),
                        array.shape().as_slice()
                    )
                });
        }

        // Broadcast all arrays to the common shape
        arrays
            .iter()
            .map(|arr| arr.broadcast_to(common_shape.clone()))
            .collect()
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

    /// Put values into an array at specified indices.
    ///
    /// Replaces elements at the given indices with the provided values.
    /// Returns a new array with the modifications.
    ///
    /// # Arguments
    ///
    /// * `indices` - Flat indices where values should be placed
    /// * `values` - Values to place at those indices
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let result = a.put(&[0, 2, 4], &[10.0, 30.0, 50.0]);
    /// assert_eq!(result.to_vec(), vec![10.0, 2.0, 30.0, 4.0, 50.0]);
    /// ```
    pub fn put(&self, indices: &[usize], values: &[f32]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            indices.len(),
            values.len(),
            "Number of indices must match number of values"
        );

        let mut data = self.to_vec();

        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < data.len(), "Index {} out of bounds", idx);
            data[idx] = values[i];
        }

        Array::from_vec(data, self.shape().clone())
    }

    /// Scatter update values into an array at specified indices.
    ///
    /// Returns a new array with values from `updates` placed at positions
    /// specified by `indices`. This is equivalent to `put()` but follows
    /// the JAX/NumPy scatter naming convention.
    ///
    /// # Arguments
    ///
    /// * `indices` - Flattened indices where updates should be placed
    /// * `updates` - Values to place at the specified indices
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let result = a.scatter(&[0, 2, 4], &[10.0, 30.0, 50.0]);
    /// assert_eq!(result.to_vec(), vec![10.0, 2.0, 30.0, 4.0, 50.0]);
    /// ```
    pub fn scatter(&self, indices: &[usize], updates: &[f32]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            indices.len(),
            updates.len(),
            "Number of indices must match number of updates"
        );

        let mut data = self.to_vec();

        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < data.len(), "Index {} out of bounds", idx);
            data[idx] = updates[i];
        }

        Array::from_vec(data, self.shape().clone())
    }

    /// Scatter-add values into an array at specified indices.
    ///
    /// Returns a new array with values from `updates` added to the values
    /// at positions specified by `indices`. If the same index appears multiple
    /// times, updates are accumulated.
    ///
    /// # Arguments
    ///
    /// * `indices` - Flattened indices where updates should be added
    /// * `updates` - Values to add at the specified indices
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let result = a.scatter_add(&[0, 2, 4], &[10.0, 30.0, 50.0]);
    /// assert_eq!(result.to_vec(), vec![11.0, 2.0, 33.0, 4.0, 55.0]);
    /// ```
    pub fn scatter_add(&self, indices: &[usize], updates: &[f32]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            indices.len(),
            updates.len(),
            "Number of indices must match number of updates"
        );

        let mut data = self.to_vec();

        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < data.len(), "Index {} out of bounds", idx);
            data[idx] += updates[i];
        }

        Array::from_vec(data, self.shape().clone())
    }

    /// Scatter-min values into an array at specified indices.
    ///
    /// Returns a new array where each position specified by `indices` contains
    /// the minimum of the original value and the corresponding update value.
    ///
    /// # Arguments
    ///
    /// * `indices` - Flattened indices where min operation should be applied
    /// * `updates` - Values to compare with current values
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![5.0, 10.0, 15.0, 20.0, 25.0], Shape::new(vec![5]));
    /// let result = a.scatter_min(&[1, 2, 3], &[8.0, 20.0, 15.0]);
    /// assert_eq!(result.to_vec(), vec![5.0, 8.0, 15.0, 15.0, 25.0]);
    /// ```
    pub fn scatter_min(&self, indices: &[usize], updates: &[f32]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            indices.len(),
            updates.len(),
            "Number of indices must match number of updates"
        );

        let mut data = self.to_vec();

        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < data.len(), "Index {} out of bounds", idx);
            data[idx] = data[idx].min(updates[i]);
        }

        Array::from_vec(data, self.shape().clone())
    }

    /// Scatter-max values into an array at specified indices.
    ///
    /// Returns a new array where each position specified by `indices` contains
    /// the maximum of the original value and the corresponding update value.
    ///
    /// # Arguments
    ///
    /// * `indices` - Flattened indices where max operation should be applied
    /// * `updates` - Values to compare with current values
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![5.0, 10.0, 15.0, 20.0, 25.0], Shape::new(vec![5]));
    /// let result = a.scatter_max(&[1, 2, 3], &[12.0, 10.0, 25.0]);
    /// assert_eq!(result.to_vec(), vec![5.0, 12.0, 15.0, 25.0, 25.0]);
    /// ```
    pub fn scatter_max(&self, indices: &[usize], updates: &[f32]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            indices.len(),
            updates.len(),
            "Number of indices must match number of updates"
        );

        let mut data = self.to_vec();

        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < data.len(), "Index {} out of bounds", idx);
            data[idx] = data[idx].max(updates[i]);
        }

        Array::from_vec(data, self.shape().clone())
    }

    /// Scatter updates to specified indices using multiplication.
    ///
    /// For each index, multiplies the existing value with the update value.
    /// When multiple updates target the same index, they are accumulated.
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices where updates should be applied
    /// * `updates` - Values to multiply at the corresponding indices
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let result = a.scatter_mul(&[1, 2, 3], &[2.0, 3.0, 0.5]);
    /// assert_eq!(result.to_vec(), vec![1.0, 4.0, 9.0, 2.0, 5.0]);
    /// ```
    pub fn scatter_mul(&self, indices: &[usize], updates: &[f32]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            indices.len(),
            updates.len(),
            "Number of indices must match number of updates"
        );

        let mut data = self.to_vec();

        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < data.len(), "Index {} out of bounds", idx);
            data[idx] *= updates[i];
        }

        Array::from_vec(data, self.shape().clone())
    }

    /// Take values from an array along an axis using indices.
    ///
    /// This is similar to gather operations in other frameworks.
    /// For each position, it selects the element specified by the index array.
    ///
    /// # Arguments
    ///
    /// * `indices` - Array of indices to take along the axis
    /// * `axis` - Axis along which to take values
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// // For a 2D array, select different columns for each row
    /// let a = Array::from_vec(
    ///     vec![10.0, 20.0, 30.0,
    ///          40.0, 50.0, 60.0],
    ///     Shape::new(vec![2, 3])
    /// );
    /// let indices = Array::from_vec(vec![0.0, 2.0], Shape::new(vec![2]));
    /// let result = a.take_along_axis(&indices, 1);
    /// // Takes column 0 from row 0 (10.0) and column 2 from row 1 (60.0)
    /// assert_eq!(result.to_vec(), vec![10.0, 60.0]);
    /// ```
    pub fn take_along_axis(&self, indices: &Array, axis: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(indices.dtype(), DType::Float32, "Indices must be Float32");
        assert!(axis < self.ndim(), "Axis {} out of bounds", axis);

        let data = self.to_vec();
        let idx_data = indices.to_vec();
        let shape = self.shape().as_slice();

        // Handle 1D indices for 2D array (backward compatible simplified API)
        if indices.ndim() == 1 && self.ndim() == 2 {
            if axis == 1 {
                // Take along columns: for each row, pick the column from indices
                let rows = shape[0];
                let cols = shape[1];
                assert_eq!(
                    indices.size(),
                    rows,
                    "Indices size must match number of rows"
                );
                let result: Vec<f32> = idx_data
                    .iter()
                    .enumerate()
                    .map(|(row, &idx)| {
                        let col = idx as usize;
                        assert!(col < cols, "Column index {} out of bounds", col);
                        data[row * cols + col]
                    })
                    .collect();
                return Array::from_vec(result, Shape::new(vec![rows]));
            } else {
                // axis == 0: Take along rows: for each column, pick the row from indices
                let rows = shape[0];
                let cols = shape[1];
                assert_eq!(
                    indices.size(),
                    cols,
                    "Indices size must match number of columns"
                );
                let result: Vec<f32> = idx_data
                    .iter()
                    .enumerate()
                    .map(|(col, &idx)| {
                        let row = idx as usize;
                        assert!(row < rows, "Row index {} out of bounds", row);
                        data[row * cols + col]
                    })
                    .collect();
                return Array::from_vec(result, Shape::new(vec![cols]));
            }
        }

        // Handle 1D case
        if self.ndim() == 1 {
            let result: Vec<f32> = idx_data
                .iter()
                .map(|&idx| {
                    let i = idx as usize;
                    assert!(i < data.len(), "Index {} out of bounds", i);
                    data[i]
                })
                .collect();
            return Array::from_vec(result, indices.shape().clone());
        }

        // General N-dimensional implementation (requires matching dimensions)
        let idx_shape = indices.shape().as_slice();
        assert_eq!(
            self.ndim(),
            indices.ndim(),
            "For N-dimensional take_along_axis, array and indices must have same number of dimensions"
        );
        for (i, (&s, &is)) in shape.iter().zip(idx_shape.iter()).enumerate() {
            if i != axis {
                assert_eq!(
                    s, is,
                    "Dimension {} must match: array has {}, indices has {}",
                    i, s, is
                );
            }
        }

        let ndim = self.ndim();
        let out_size = indices.size();
        let mut result = vec![0.0f32; out_size];

        // Compute strides for input array
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Compute strides for indices array
        let mut idx_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            idx_strides[i] = idx_strides[i + 1] * idx_shape[i + 1];
        }

        // Iterate over all output positions
        for out_flat in 0..out_size {
            // Convert flat index to multi-dimensional index in output/indices space
            let mut multi_idx = vec![0usize; ndim];
            let mut remaining = out_flat;
            for i in 0..ndim {
                multi_idx[i] = remaining / idx_strides[i];
                remaining %= idx_strides[i];
            }

            // Get the index value at this position
            let idx_val = idx_data[out_flat] as usize;
            assert!(
                idx_val < shape[axis],
                "Index {} out of bounds for axis {} with size {}",
                idx_val,
                axis,
                shape[axis]
            );

            // Build input multi-dimensional index: same as output except at axis
            let mut input_idx = multi_idx.clone();
            input_idx[axis] = idx_val;

            // Convert to flat index in input array
            let in_flat: usize = input_idx
                .iter()
                .zip(strides.iter())
                .map(|(i, s)| i * s)
                .sum();

            result[out_flat] = data[in_flat];
        }

        Array::from_vec(result, indices.shape().clone())
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

    /// Roll array elements along a given axis.
    ///
    /// Elements that roll beyond the last position are re-introduced at the first.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let rolled = a.roll(2);
    /// assert_eq!(rolled.to_vec(), vec![4.0, 5.0, 1.0, 2.0, 3.0]);
    /// ```
    pub fn roll(&self, shift: isize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let len = data.len();

        if len == 0 {
            return self.clone();
        }

        // Normalize shift to be within [0, len)
        let shift = ((shift % len as isize) + len as isize) as usize % len;

        let mut result = vec![0.0; len];
        for i in 0..len {
            result[(i + shift) % len] = data[i];
        }

        Array::from_vec(result, self.shape().clone())
    }

    /// Rotate array by 90 degrees in the plane specified by axes.
    ///
    /// For 2D arrays, rotates counterclockwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let rotated = a.rot90(1);
    /// assert_eq!(rotated.to_vec(), vec![2.0, 4.0, 1.0, 3.0]);
    /// ```
    pub fn rot90(&self, k: isize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.shape().ndim(), 2, "Only 2D arrays supported");

        let shape = self.shape().as_slice();
        let (h, w) = (shape[0], shape[1]);
        let data = self.to_vec();

        // Normalize k to be within [0, 4)
        let k = k.rem_euclid(4);

        match k {
            0 => self.clone(),
            1 => {
                // Rotate 90 degrees counterclockwise
                let mut result = vec![0.0; h * w];
                for i in 0..h {
                    for j in 0..w {
                        let new_i = w - 1 - j;
                        let new_j = i;
                        result[new_i * h + new_j] = data[i * w + j];
                    }
                }
                Array::from_vec(result, Shape::new(vec![w, h]))
            }
            2 => {
                // Rotate 180 degrees
                let mut result = vec![0.0; h * w];
                for i in 0..h {
                    for j in 0..w {
                        let new_i = h - 1 - i;
                        let new_j = w - 1 - j;
                        result[new_i * w + new_j] = data[i * w + j];
                    }
                }
                Array::from_vec(result, Shape::new(vec![h, w]))
            }
            3 => {
                // Rotate 270 degrees counterclockwise (90 clockwise)
                let mut result = vec![0.0; h * w];
                for i in 0..h {
                    for j in 0..w {
                        let new_i = j;
                        let new_j = h - 1 - i;
                        result[new_i * h + new_j] = data[i * w + j];
                    }
                }
                Array::from_vec(result, Shape::new(vec![w, h]))
            }
            _ => unreachable!(),
        }
    }

    /// Interchange two axes of an array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    /// let swapped = a.swapaxes(0, 1);
    /// assert_eq!(swapped.shape().as_slice(), &[3, 2]);
    /// ```
    pub fn swapaxes(&self, axis1: usize, axis2: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let ndim = self.shape().ndim();
        assert!(axis1 < ndim, "axis1 out of bounds");
        assert!(axis2 < ndim, "axis2 out of bounds");

        if axis1 == axis2 {
            return self.clone();
        }

        // For 2D arrays, this is just transpose
        if ndim == 2 && ((axis1 == 0 && axis2 == 1) || (axis1 == 1 && axis2 == 0)) {
            return self.transpose();
        }

        // General case for higher dimensions
        let old_shape = self.shape().as_slice();
        let mut new_shape = old_shape.to_vec();
        new_shape.swap(axis1, axis2);

        let data = self.to_vec();
        let size = self.size();
        let mut result = vec![0.0; size];

        // Compute strides for old and new shapes
        let old_strides = self.shape().default_strides();
        let mut new_strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        // Copy elements with swapped axes
        for i in 0..size {
            let mut old_indices = vec![0; ndim];
            let mut temp = i;
            for j in 0..ndim {
                old_indices[j] = temp / old_strides[j];
                temp %= old_strides[j];
            }

            // Swap the indices
            old_indices.swap(axis1, axis2);

            // Compute new flat index
            let mut new_idx = 0;
            for j in 0..ndim {
                new_idx += old_indices[j] * new_strides[j];
            }

            result[new_idx] = data[i];
        }

        Array::from_vec(result, Shape::new(new_shape))
    }

    /// Move axes of an array to new positions.
    ///
    /// Simplified version that only supports moving a single axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![1, 2, 3]));
    /// let moved = a.moveaxis(2, 0);
    /// assert_eq!(moved.shape().as_slice(), &[3, 1, 2]);
    /// ```
    pub fn moveaxis(&self, source: usize, destination: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let ndim = self.shape().ndim();
        assert!(source < ndim, "source axis out of bounds");
        assert!(destination < ndim, "destination axis out of bounds");

        if source == destination {
            return self.clone();
        }

        let old_shape = self.shape().as_slice();
        let mut new_shape = Vec::new();

        // Build new shape by removing source axis and inserting at destination
        for (i, &dim) in old_shape.iter().enumerate() {
            if i != source {
                new_shape.push(dim);
            }
        }
        new_shape.insert(destination, old_shape[source]);

        // For simple cases, use swapaxes
        if ndim == 2 {
            return self.swapaxes(source, destination);
        }

        // For 3D, we can implement a specific case
        if ndim == 3 {
            let data = self.to_vec();
            let size = self.size();
            let mut result = vec![0.0; size];

            let old_strides = self.shape().default_strides();
            let mut new_strides = vec![1; ndim];
            for i in (0..ndim - 1).rev() {
                new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
            }

            for i in 0..size {
                let mut old_indices = vec![0; ndim];
                let mut temp = i;
                for j in 0..ndim {
                    old_indices[j] = temp / old_strides[j];
                    temp %= old_strides[j];
                }

                // Reorder indices
                let moved_val = old_indices[source];
                old_indices.remove(source);
                old_indices.insert(destination, moved_val);

                // Compute new flat index
                let mut new_idx = 0;
                for j in 0..ndim {
                    new_idx += old_indices[j] * new_strides[j];
                }

                result[new_idx] = data[i];
            }

            return Array::from_vec(result, Shape::new(new_shape));
        }

        // For higher dimensions, fall back to swapaxes
        self.swapaxes(source, destination)
    }

    /// One-dimensional linear interpolation.
    ///
    /// Returns interpolated values at specified points using linear interpolation.
    ///
    /// # Arguments
    ///
    /// * `x` - x-coordinates at which to evaluate the interpolated values
    /// * `xp` - x-coordinates of the data points (must be increasing)
    /// * `fp` - y-coordinates of the data points
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let xp = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let fp = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
    /// let x = Array::from_vec(vec![1.5, 2.5], Shape::new(vec![2]));
    /// let result = Array::interp(&x, &xp, &fp);
    /// assert_eq!(result.to_vec(), vec![15.0, 25.0]);
    /// ```
    pub fn interp(x: &Array, xp: &Array, fp: &Array) -> Array {
        assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(xp.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(fp.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            xp.size(),
            fp.size(),
            "xp and fp must have the same size"
        );

        let x_data = x.to_vec();
        let xp_data = xp.to_vec();
        let fp_data = fp.to_vec();

        let result: Vec<f32> = x_data
            .iter()
            .map(|&xi| {
                // Handle edge cases
                if xi <= xp_data[0] {
                    return fp_data[0];
                }
                if xi >= xp_data[xp_data.len() - 1] {
                    return fp_data[fp_data.len() - 1];
                }

                // Find the interval containing xi
                for i in 0..xp_data.len() - 1 {
                    if xi >= xp_data[i] && xi <= xp_data[i + 1] {
                        // Linear interpolation
                        let t = (xi - xp_data[i]) / (xp_data[i + 1] - xp_data[i]);
                        return fp_data[i] + t * (fp_data[i + 1] - fp_data[i]);
                    }
                }

                fp_data[fp_data.len() - 1]
            })
            .collect();

        Array::from_vec(result, x.shape().clone())
    }

    /// Linear interpolation between two arrays.
    ///
    /// Returns a + weight * (b - a)
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 10.0, 20.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![100.0, 110.0, 120.0], Shape::new(vec![3]));
    /// let result = a.lerp(&b, 0.5);
    /// assert_eq!(result.to_vec(), vec![50.0, 60.0, 70.0]);
    /// ```
    pub fn lerp(&self, other: &Array, weight: f32) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            self.shape(),
            other.shape(),
            "Arrays must have the same shape"
        );

        let self_data = self.to_vec();
        let other_data = other.to_vec();

        let result: Vec<f32> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a + weight * (b - a))
            .collect();

        Array::from_vec(result, self.shape().clone())
    }

    /// Linearly interpolate between two arrays element-wise with array weights.
    ///
    /// Returns a + weight * (b - a) where weight is an array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 10.0, 20.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![100.0, 110.0, 120.0], Shape::new(vec![3]));
    /// let weights = Array::from_vec(vec![0.0, 0.5, 1.0], Shape::new(vec![3]));
    /// let result = a.lerp_array(&b, &weights);
    /// assert_eq!(result.to_vec(), vec![0.0, 60.0, 120.0]);
    /// ```
    pub fn lerp_array(&self, other: &Array, weights: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(weights.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            self.shape(),
            other.shape(),
            "Arrays must have the same shape"
        );
        assert_eq!(
            self.shape(),
            weights.shape(),
            "Arrays and weights must have the same shape"
        );

        let self_data = self.to_vec();
        let other_data = other.to_vec();
        let weight_data = weights.to_vec();

        let result: Vec<f32> = self_data
            .iter()
            .zip(other_data.iter())
            .zip(weight_data.iter())
            .map(|((&a, &b), &w)| a + w * (b - a))
            .collect();

        Array::from_vec(result, self.shape().clone())
    }

    /// Compute the discrete 1D convolution of two arrays.
    ///
    /// Returns the discrete linear convolution of the input array with a kernel.
    /// Uses 'valid' mode (only overlapping parts).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let kernel = Array::from_vec(vec![1.0, 0.0, -1.0], Shape::new(vec![3]));
    /// let conv = signal.convolve(&kernel);
    /// // Convolution flips the kernel: [-1, 0, 1]
    /// // [1*(-1) + 2*0 + 3*1, 2*(-1) + 3*0 + 4*1, 3*(-1) + 4*0 + 5*1]
    /// // = [-1+0+3, -2+0+4, -3+0+5] = [2, 2, 2]
    /// assert_eq!(conv.to_vec(), vec![2.0, 2.0, 2.0]);
    /// ```
    pub fn convolve(&self, kernel: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(kernel.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.ndim(), 1, "Convolve only supports 1D arrays");
        assert_eq!(kernel.ndim(), 1, "Kernel must be 1D");

        let signal = self.to_vec();
        let mut k = kernel.to_vec();
        let n = signal.len();
        let m = k.len();

        if m > n {
            // Kernel longer than signal - return empty array
            return Array::zeros(Shape::new(vec![0]), DType::Float32);
        }

        // Flip the kernel for convolution
        k.reverse();

        // Valid mode: output size = n - m + 1
        let out_size = n - m + 1;
        let mut result = Vec::with_capacity(out_size);

        for i in 0..out_size {
            let mut sum = 0.0;
            for j in 0..m {
                sum += signal[i + j] * k[j];
            }
            result.push(sum);
        }

        Array::from_vec(result, Shape::new(vec![out_size]))
    }

    /// Compute the cross-correlation of two 1D arrays.
    ///
    /// Cross-correlation is similar to convolution but without flipping the kernel.
    /// Uses 'valid' mode (only overlapping parts).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let template = Array::from_vec(vec![1.0, 2.0, 1.0], Shape::new(vec![3]));
    /// let corr = signal.correlate(&template);
    /// // [1*1 + 2*2 + 3*1, 2*1 + 3*2 + 4*1, 3*1 + 4*2 + 5*1]
    /// // = [1+4+3, 2+6+4, 3+8+5] = [8, 12, 16]
    /// assert_eq!(corr.to_vec(), vec![8.0, 12.0, 16.0]);
    /// ```
    pub fn correlate(&self, template: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(template.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.ndim(), 1, "Correlate only supports 1D arrays");
        assert_eq!(template.ndim(), 1, "Template must be 1D");

        let signal = self.to_vec();
        let t = template.to_vec();
        let n = signal.len();
        let m = t.len();

        if m > n {
            // Template longer than signal - return empty array
            return Array::zeros(Shape::new(vec![0]), DType::Float32);
        }

        // Valid mode: output size = n - m + 1
        let out_size = n - m + 1;
        let mut result = Vec::with_capacity(out_size);

        for i in 0..out_size {
            let mut sum = 0.0;
            for j in 0..m {
                // Note: no kernel flip, unlike convolution
                sum += signal[i + j] * t[j];
            }
            result.push(sum);
        }

        Array::from_vec(result, Shape::new(vec![out_size]))
    }

    /// Stack arrays vertically (row-wise).
    ///
    /// Equivalent to concatenation along axis 0 after promoting 1D arrays to 2D.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
    /// let stacked = a.vstack(&b);
    /// assert_eq!(stacked.shape().as_slice(), &[2, 3]);
    /// assert_eq!(stacked.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn vstack(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let self_shape = self.shape().as_slice();
        let other_shape = other.shape().as_slice();

        // Promote 1D arrays to 2D (1, N)
        let self_2d = if self_shape.len() == 1 {
            self.reshape(Shape::new(vec![1, self_shape[0]]))
        } else {
            self.clone()
        };

        let other_2d = if other_shape.len() == 1 {
            other.reshape(Shape::new(vec![1, other_shape[0]]))
        } else {
            other.clone()
        };

        // Concatenate along axis 0
        Array::concatenate(&[self_2d, other_2d], 0)
    }

    /// Stack arrays horizontally (column-wise).
    ///
    /// Equivalent to concatenation along axis 1.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2, 1]));
    /// let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2, 1]));
    /// let stacked = a.hstack(&b);
    /// assert_eq!(stacked.shape().as_slice(), &[2, 2]);
    /// assert_eq!(stacked.to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn hstack(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let self_shape = self.shape().as_slice();
        let other_shape = other.shape().as_slice();

        // For 1D arrays, concatenate directly
        if self_shape.len() == 1 && other_shape.len() == 1 {
            return Array::concatenate(&[self.clone(), other.clone()], 0);
        }

        // For 2D+ arrays, concatenate along axis 1
        Array::concatenate(&[self.clone(), other.clone()], 1)
    }

    /// Split array into multiple sub-arrays vertically (row-wise).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![6]));
    /// let parts = a.vsplit(2);
    /// assert_eq!(parts.len(), 2);
    /// assert_eq!(parts[0].to_vec(), vec![1.0, 2.0, 3.0]);
    /// assert_eq!(parts[1].to_vec(), vec![4.0, 5.0, 6.0]);
    /// ```
    pub fn vsplit(&self, num_sections: usize) -> Vec<Array> {
        let shape = self.shape().as_slice();

        if shape.len() == 1 {
            // For 1D arrays, split along axis 0
            return Array::split(self, num_sections, 0);
        }

        // For 2D+ arrays, split along axis 0
        Array::split(self, num_sections, 0)
    }

    /// Split array into multiple sub-arrays horizontally (column-wise).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let parts = a.hsplit(2);
    /// assert_eq!(parts.len(), 2);
    /// assert_eq!(parts[0].shape().as_slice(), &[2, 1]);
    /// assert_eq!(parts[1].shape().as_slice(), &[2, 1]);
    /// ```
    pub fn hsplit(&self, num_sections: usize) -> Vec<Array> {
        let shape = self.shape().as_slice();
        assert!(!shape.is_empty(), "hsplit requires at least 1D array");

        if shape.len() == 1 {
            // For 1D arrays, split along axis 0
            return Array::split(self, num_sections, 0);
        }

        // For 2D+ arrays, split along axis 1
        Array::split(self, num_sections, 1)
    }

    /// Append values to the end of an array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![4.0, 5.0], Shape::new(vec![2]));
    /// let result = a.append(&b);
    /// assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    pub fn append(&self, values: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(values.dtype(), DType::Float32, "Only Float32 supported");

        let mut data = self.to_vec();
        data.extend(values.to_vec());

        let new_size = data.len();
        Array::from_vec(data, Shape::new(vec![new_size]))
    }

    /// Insert values at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 5.0, 6.0], Shape::new(vec![4]));
    /// let values = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
    /// let result = a.insert(2, &values);
    /// assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn insert(&self, index: usize, values: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(values.dtype(), DType::Float32, "Only Float32 supported");

        let mut data = self.to_vec();
        let values_data = values.to_vec();

        assert!(index <= data.len(), "Index out of bounds");

        // Insert values at the specified index
        for (i, &val) in values_data.iter().enumerate() {
            data.insert(index + i, val);
        }

        let new_size = data.len();
        Array::from_vec(data, Shape::new(vec![new_size]))
    }

    /// Delete elements at specified indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let result = a.delete(&[1, 3]);
    /// assert_eq!(result.to_vec(), vec![1.0, 3.0, 5.0]);
    /// ```
    pub fn delete(&self, indices: &[usize]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let mut result = Vec::new();

        for (i, &val) in data.iter().enumerate() {
            if !indices.contains(&i) {
                result.push(val);
            }
        }

        let new_size = result.len();
        Array::from_vec(result, Shape::new(vec![new_size]))
    }

    /// Trim leading and trailing zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0], Shape::new(vec![6]));
    /// let trimmed = a.trim_zeros();
    /// assert_eq!(trimmed.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn trim_zeros(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();

        // Find first non-zero
        let start = data.iter().position(|&x| x.abs() > 1e-10).unwrap_or(data.len());

        // Find last non-zero
        let end = data.iter().rposition(|&x| x.abs() > 1e-10).map(|i| i + 1).unwrap_or(0);

        if start >= end {
            return Array::zeros(Shape::new(vec![0]), DType::Float32);
        }

        let result = data[start..end].to_vec();
        let new_size = result.len();
        Array::from_vec(result, Shape::new(vec![new_size]))
    }

    /// Repeat each element along axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let repeated = a.repeat_elements(2);
    /// assert_eq!(repeated.to_vec(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    /// ```
    pub fn repeat_elements(&self, repeats: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let mut result = Vec::with_capacity(data.len() * repeats);

        for &val in data.iter() {
            for _ in 0..repeats {
                result.push(val);
            }
        }

        let new_size = result.len();
        Array::from_vec(result, Shape::new(vec![new_size]))
    }

    /// Resize array to new shape, repeating or truncating as needed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let resized = a.resize(5);
    /// assert_eq!(resized.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0]);
    /// ```
    pub fn resize(&self, new_size: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let mut result = Vec::with_capacity(new_size);

        for i in 0..new_size {
            result.push(data[i % data.len()]);
        }

        Array::from_vec(result, Shape::new(vec![new_size]))
    }

    /// Compute correlation coefficient between two 1D arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let y = Array::from_vec(vec![2.0, 4.0, 6.0, 8.0], Shape::new(vec![4]));
    /// let corr = x.corrcoef(&y);
    /// assert!((corr - 1.0).abs() < 1e-5); // Perfect correlation
    /// ```
    pub fn corrcoef(&self, other: &Array) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.size(), other.size(), "Arrays must have same size");

        let x = self.to_vec();
        let y = other.to_vec();
        let n = x.len() as f32;

        // Compute means
        let x_mean: f32 = x.iter().sum::<f32>() / n;
        let y_mean: f32 = y.iter().sum::<f32>() / n;

        // Compute covariance and standard deviations
        let mut cov = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for (x_val, y_val) in x.iter().zip(y.iter()) {
            let x_diff = x_val - x_mean;
            let y_diff = y_val - y_mean;
            cov += x_diff * y_diff;
            x_var += x_diff * x_diff;
            y_var += y_diff * y_diff;
        }

        // Correlation coefficient
        if x_var.abs() < 1e-10 || y_var.abs() < 1e-10 {
            return 0.0;
        }

        cov / (x_var * y_var).sqrt()
    }

    /// Return indices of non-zero elements in a flattened array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0, 0.0, 3.0, 0.0, 5.0], Shape::new(vec![6]));
    /// let indices = a.flatnonzero();
    /// assert_eq!(indices, vec![1, 3, 5]);
    /// ```
    pub fn flatnonzero(&self) -> Vec<usize> {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        data.iter()
            .enumerate()
            .filter_map(|(i, &val)| if val.abs() > 1e-10 { Some(i) } else { None })
            .collect()
    }

    /// Tile the array by repeating it along each dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = a.tile_1d(3);
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    /// ```
    pub fn tile_1d(&self, reps: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();

        let mut result = Vec::with_capacity(data.len() * reps);
        for _ in 0..reps {
            result.extend_from_slice(&data);
        }

        let new_size = result.len();
        Array::from_vec(result, Shape::new(vec![new_size]))
    }

    /// Stack 1-D arrays as columns into a 2-D array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
    /// let c = Array::column_stack(&[a, b]);
    /// assert_eq!(c.shape().as_slice(), &[3, 2]);
    /// // [[1, 4], [2, 5], [3, 6]]
    /// ```
    pub fn column_stack(arrays: &[Array]) -> Array {
        assert!(!arrays.is_empty(), "Need at least one array");
        assert_eq!(arrays[0].dtype(), DType::Float32, "Only Float32 supported");

        let n_rows = arrays[0].size();
        let n_cols = arrays.len();

        let mut result = Vec::with_capacity(n_rows * n_cols);
        for row_idx in 0..n_rows {
            for arr in arrays {
                let data = arr.to_vec();
                result.push(data[row_idx]);
            }
        }

        Array::from_vec(result, Shape::new(vec![n_rows, n_cols]))
    }

    /// Stack arrays in sequence vertically (row wise).
    ///
    /// Alias for vstack.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
    /// let c = Array::row_stack(&[a, b]);
    /// assert_eq!(c.shape().as_slice(), &[2, 2]);
    /// // [[1, 2], [3, 4]]
    /// ```
    pub fn row_stack(arrays: &[Array]) -> Array {
        assert!(!arrays.is_empty(), "Need at least one array");

        // Convert 1D arrays to 2D if needed
        let arrays_2d: Vec<Array> = arrays.iter().map(|arr| {
            if arr.shape().as_slice().len() == 1 {
                let size = arr.size();
                let data = arr.to_vec();
                Array::from_vec(data, Shape::new(vec![1, size]))
            } else {
                arr.clone()
            }
        }).collect();

        Array::concatenate(&arrays_2d, 0)
    }

    /// Stack arrays in sequence depth wise (along third axis).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
    /// let c = Array::dstack(&[a, b]);
    /// assert_eq!(c.shape().as_slice(), &[1, 2, 2]);
    /// ```
    pub fn dstack(arrays: &[Array]) -> Array {
        assert!(!arrays.is_empty(), "Need at least one array");

        // Convert to at least 3D
        let arrays_3d: Vec<Array> = arrays.iter().map(|arr| {
            let shape = arr.shape().as_slice();
            match shape.len() {
                1 => {
                    let size = arr.size();
                    let data = arr.to_vec();
                    Array::from_vec(data, Shape::new(vec![1, size, 1]))
                }
                2 => {
                    let data = arr.to_vec();
                    Array::from_vec(data, Shape::new(vec![shape[0], shape[1], 1]))
                }
                _ => arr.clone(),
            }
        }).collect();

        Array::concatenate(&arrays_3d, 2)
    }

    /// Compute the absolute value and return as a new array (alias for abs).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, -2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.absolute();
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn absolute(&self) -> Array {
        self.abs()
    }

    /// Clamp values to a specified range (alias for clip).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 5.0, 10.0], Shape::new(vec![3]));
    /// let b = a.clamp(2.0, 8.0);
    /// assert_eq!(b.to_vec(), vec![2.0, 5.0, 8.0]);
    /// ```
    pub fn clamp(&self, min: f32, max: f32) -> Array {
        self.clip(min, max)
    }

    /// Fill the diagonal of a 2D array with a scalar value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape, DType};
    /// let a = Array::zeros(Shape::new(vec![3, 3]), DType::Float32);
    /// let filled = a.fill_diagonal(5.0);
    /// assert_eq!(filled.to_vec(), vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]);
    /// ```
    pub fn fill_diagonal(&self, value: f32) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let shape = self.shape();
        let dims = shape.as_slice();
        assert_eq!(dims.len(), 2, "fill_diagonal only supports 2D arrays");

        let (rows, cols) = (dims[0], dims[1]);
        let data = self.to_vec();
        let mut result = data.clone();

        let min_dim = rows.min(cols);
        for i in 0..min_dim {
            result[i * cols + i] = value;
        }

        Array::from_vec(result, shape.clone())
    }

    /// Evaluate a polynomial at specific values.
    /// Polynomial coefficients are in decreasing order (highest degree first).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// // Evaluate p(x) = 2x^2 + 3x + 1 at x = [1, 2, 3]
    /// let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let coeffs = Array::from_vec(vec![2.0, 3.0, 1.0], Shape::new(vec![3]));
    /// let result = x.polyval(&coeffs);
    /// // At x=1: 2(1)^2 + 3(1) + 1 = 6
    /// // At x=2: 2(4) + 3(2) + 1 = 15
    /// // At x=3: 2(9) + 3(3) + 1 = 28
    /// assert_eq!(result.to_vec(), vec![6.0, 15.0, 28.0]);
    /// ```
    pub fn polyval(&self, coeffs: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(coeffs.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(coeffs.ndim(), 1, "Coefficients must be 1D");

        let x_data = self.to_vec();
        let c_data = coeffs.to_vec();

        let result_data: Vec<f32> = x_data
            .iter()
            .map(|&x| {
                // Horner's method for polynomial evaluation
                let mut result = 0.0;
                for &coeff in &c_data {
                    result = result * x + coeff;
                }
                result
            })
            .collect();

        Array::from_vec(result_data, self.shape().clone())
    }

    /// Add two polynomials.
    /// Polynomial coefficients are in decreasing order (highest degree first).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// // Add p(x) = 2x^2 + 3x + 1 and q(x) = x^2 + 2x + 3
    /// let p = Array::from_vec(vec![2.0, 3.0, 1.0], Shape::new(vec![3]));
    /// let q = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let sum = p.polyadd(&q);
    /// assert_eq!(sum.to_vec(), vec![3.0, 5.0, 4.0]);
    /// ```
    pub fn polyadd(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.ndim(), 1, "Polynomials must be 1D");
        assert_eq!(other.ndim(), 1, "Polynomials must be 1D");

        let p_data = self.to_vec();
        let q_data = other.to_vec();

        let max_len = p_data.len().max(q_data.len());
        let mut result = vec![0.0; max_len];

        // Align from the right (lowest degree)
        let p_offset = max_len - p_data.len();
        let q_offset = max_len - q_data.len();

        for (i, &val) in p_data.iter().enumerate() {
            result[p_offset + i] += val;
        }

        for (i, &val) in q_data.iter().enumerate() {
            result[q_offset + i] += val;
        }

        Array::from_vec(result, Shape::new(vec![max_len]))
    }

    /// Multiply two polynomials.
    /// Polynomial coefficients are in decreasing order (highest degree first).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// // Multiply (x + 1) * (x + 2) = x^2 + 3x + 2
    /// let p = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));
    /// let q = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let prod = p.polymul(&q);
    /// assert_eq!(prod.to_vec(), vec![1.0, 3.0, 2.0]);
    /// ```
    pub fn polymul(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.ndim(), 1, "Polynomials must be 1D");
        assert_eq!(other.ndim(), 1, "Polynomials must be 1D");

        let p = self.to_vec();
        let q = other.to_vec();
        let result_len = p.len() + q.len() - 1;
        let mut result = vec![0.0; result_len];

        for (i, &pi) in p.iter().enumerate() {
            for (j, &qj) in q.iter().enumerate() {
                result[i + j] += pi * qj;
            }
        }

        Array::from_vec(result, Shape::new(vec![result_len]))
    }

    /// Differentiate a polynomial.
    /// Returns the polynomial representing the derivative.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// // d/dx (2x^2 + 3x + 1) = 4x + 3
    /// let p = Array::from_vec(vec![2.0, 3.0, 1.0], Shape::new(vec![3]));
    /// let dp = p.polyder();
    /// assert_eq!(dp.to_vec(), vec![4.0, 3.0]);
    /// ```
    pub fn polyder(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.ndim(), 1, "Polynomial must be 1D");

        let coeffs = self.to_vec();
        if coeffs.len() <= 1 {
            return Array::from_vec(vec![0.0], Shape::new(vec![1]));
        }

        let n = coeffs.len() - 1;
        let mut result = Vec::with_capacity(n);

        for (i, &c) in coeffs.iter().take(n).enumerate() {
            let degree = (n - i) as f32;
            result.push(c * degree);
        }

        Array::from_vec(result, Shape::new(vec![n]))
    }

    /// Subtract two polynomials.
    /// Polynomial coefficients are in decreasing order (highest degree first).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let p = Array::from_vec(vec![3.0, 5.0, 4.0], Shape::new(vec![3]));
    /// let q = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let diff = p.polysub(&q);
    /// assert_eq!(diff.to_vec(), vec![2.0, 3.0, 1.0]);
    /// ```
    pub fn polysub(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.ndim(), 1, "Polynomials must be 1D");
        assert_eq!(other.ndim(), 1, "Polynomials must be 1D");

        let p_data = self.to_vec();
        let q_data = other.to_vec();

        let max_len = p_data.len().max(q_data.len());
        let mut result = vec![0.0; max_len];

        let p_offset = max_len - p_data.len();
        let q_offset = max_len - q_data.len();

        for (i, &val) in p_data.iter().enumerate() {
            result[p_offset + i] += val;
        }

        for (i, &val) in q_data.iter().enumerate() {
            result[q_offset + i] -= val;
        }

        Array::from_vec(result, Shape::new(vec![max_len]))
    }

    /// Evaluate a piecewise-defined function.
    ///
    /// Applies different functions based on conditions. For each element,
    /// the first true condition determines which function to apply.
    ///
    /// # Arguments
    ///
    /// * `conditions` - Vector of condition arrays (booleans as 0.0/1.0)
    /// * `functions` - Vector of function output arrays corresponding to conditions
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let x = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new(vec![5]));
    /// // Condition: x < 0
    /// let cond1 = Array::from_vec(vec![1.0, 1.0, 0.0, 0.0, 0.0], Shape::new(vec![5]));
    /// // Condition: x >= 0
    /// let cond2 = Array::from_vec(vec![0.0, 0.0, 1.0, 1.0, 1.0], Shape::new(vec![5]));
    /// // Function outputs (pre-computed)
    /// let func1 = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new(vec![5])); // identity
    /// let func2 = Array::from_vec(vec![4.0, 1.0, 0.0, 1.0, 4.0], Shape::new(vec![5])); // x^2
    /// let result = x.piecewise(&[cond1, cond2], &[func1, func2]);
    /// // For x<0: use identity, for x>=0: use x^2
    /// assert_eq!(result.to_vec(), vec![-2.0, -1.0, 0.0, 1.0, 4.0]);
    /// ```
    pub fn piecewise(&self, conditions: &[Array], functions: &[Array]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(conditions.len(), functions.len(), "Number of conditions must match number of functions");
        assert!(!conditions.is_empty(), "At least one condition required");

        let n = self.size();
        for cond in conditions {
            assert_eq!(cond.size(), n, "Condition size must match array size");
        }
        for func in functions {
            assert_eq!(func.size(), n, "Function output size must match array size");
        }

        let mut result = vec![0.0; n];
        let mut assigned = vec![false; n];

        for (cond, func) in conditions.iter().zip(functions.iter()) {
            let cond_data = cond.to_vec();
            let func_data = func.to_vec();

            for i in 0..n {
                if !assigned[i] && cond_data[i] != 0.0 {
                    result[i] = func_data[i];
                    assigned[i] = true;
                }
            }
        }

        Array::from_vec(result, self.shape().clone())
    }

    /// Place values into array at specified indices.
    ///
    /// Returns a new array with values inserted at the specified indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0], Shape::new(vec![5]));
    /// let mask = Array::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0], Shape::new(vec![5]));
    /// let values = vec![10.0, 20.0];
    /// let result = a.place(&mask, &values);
    /// assert_eq!(result.to_vec(), vec![0.0, 10.0, 0.0, 20.0, 0.0]);
    /// ```
    pub fn place(&self, mask: &Array, values: &[f32]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(mask.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.size(), mask.size(), "Array and mask must have same size");

        let mut data = self.to_vec();
        let mask_data = mask.to_vec();

        let mut value_idx = 0;
        for (i, &m) in mask_data.iter().enumerate() {
            if m != 0.0 && value_idx < values.len() {
                data[i] = values[value_idx];
                value_idx += 1;
            }
        }

        Array::from_vec(data, self.shape().clone())
    }

    /// Copy values from source to destination array.
    ///
    /// Returns a new array with values from source copied to corresponding positions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let dst = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let src = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], Shape::new(vec![5]));
    /// let mask = Array::from_vec(vec![0.0, 1.0, 1.0, 0.0, 1.0], Shape::new(vec![5]));
    /// let result = dst.copyto(&src, &mask);
    /// assert_eq!(result.to_vec(), vec![1.0, 20.0, 30.0, 4.0, 50.0]);
    /// ```
    pub fn copyto(&self, src: &Array, mask: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(src.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(mask.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(self.size(), src.size(), "Arrays must have same size");
        assert_eq!(self.size(), mask.size(), "Array and mask must have same size");

        let mut data = self.to_vec();
        let src_data = src.to_vec();
        let mask_data = mask.to_vec();

        for i in 0..data.len() {
            if mask_data[i] != 0.0 {
                data[i] = src_data[i];
            }
        }

        Array::from_vec(data, self.shape().clone())
    }

    /// Return the index of the maximum element along an axis and the max value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], Shape::new(vec![5]));
    /// let (idx, val) = a.argmax_with_value();
    /// assert_eq!(idx, 4);
    /// assert!((val - 5.0).abs() < 1e-6);
    /// ```
    pub fn argmax_with_value(&self) -> (usize, f32) {
        let data = self.to_vec();
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for (i, &x) in data.iter().enumerate() {
            if x > max_val {
                max_val = x;
                max_idx = i;
            }
        }

        (max_idx, max_val)
    }

    /// Return the index of the minimum element along an axis and the min value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], Shape::new(vec![5]));
    /// let (idx, val) = a.argmin_with_value();
    /// assert_eq!(idx, 1);
    /// assert!((val - 1.0).abs() < 1e-6);
    /// ```
    pub fn argmin_with_value(&self) -> (usize, f32) {
        let data = self.to_vec();
        let mut min_idx = 0;
        let mut min_val = f32::INFINITY;

        for (i, &x) in data.iter().enumerate() {
            if x < min_val {
                min_val = x;
                min_idx = i;
            }
        }

        (min_idx, min_val)
    }

    /// Return an array with axes transposed to the given permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     Shape::new(vec![2, 3])
    /// );
    /// let b = a.permute(&[1, 0]);
    /// assert_eq!(b.shape().as_slice(), &[3, 2]);
    /// ```
    pub fn permute(&self, axes: &[usize]) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(axes.len(), shape.len(), "axes must match number of dimensions");

        // Build new shape
        let new_shape: Vec<usize> = axes.iter().map(|&a| shape[a]).collect();

        // For 2D transpose
        if shape.len() == 2 && axes == [1, 0] {
            return self.transpose();
        }

        // General case - use strides calculation
        let data = self.to_vec();
        let mut result = vec![0.0; data.len()];

        // Calculate old strides
        let mut old_strides = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            old_strides[i] = old_strides[i + 1] * shape[i + 1];
        }

        // Calculate new strides
        let mut new_strides = vec![1usize; new_shape.len()];
        for i in (0..new_shape.len() - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        // Permute strides
        let permuted_old_strides: Vec<usize> = axes.iter().map(|&a| old_strides[a]).collect();

        // Copy data with permutation
        for new_idx in 0..data.len() {
            let mut old_idx = 0;
            let mut remainder = new_idx;
            for (d, &new_stride) in new_strides.iter().enumerate() {
                let coord = remainder / new_stride;
                remainder %= new_stride;
                old_idx += coord * permuted_old_strides[d];
            }
            result[new_idx] = data[old_idx];
        }

        Array::from_vec(result, Shape::new(new_shape))
    }

    /// Gather values along an axis using indices.
    ///
    /// This is a generalized form of indexing that allows selecting arbitrary
    /// indices along a specified axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    /// let indices = Array::from_vec(vec![0.0, 2.0], Shape::new(vec![2]));
    /// let result = a.gather(&indices, 1);
    /// // Gathers columns 0 and 2 from each row
    /// ```
    pub fn gather(&self, indices: &Array, axis: usize) -> Array {
        let shape = self.shape().as_slice();
        assert!(axis < shape.len(), "Axis out of bounds");

        let indices_data: Vec<usize> = indices.to_vec().iter().map(|&x| x as usize).collect();
        let data = self.to_vec();

        if axis == 0 && shape.len() == 1 {
            // Simple 1D case
            let result: Vec<f32> = indices_data.iter().map(|&i| data[i]).collect();
            return Array::from_vec(result, Shape::new(vec![indices_data.len()]));
        }

        // For multi-dimensional arrays
        let mut result = Vec::new();
        let mut new_shape = shape.to_vec();
        new_shape[axis] = indices_data.len();

        // Compute strides
        let mut strides: Vec<usize> = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        strides.reverse();

        let total_size: usize = new_shape.iter().product();
        result.reserve(total_size);

        // Iterate through all output positions
        for out_idx in 0..total_size {
            // Compute output coordinates
            let mut coords = Vec::with_capacity(shape.len());
            let mut remainder = out_idx;
            for &dim in &new_shape {
                coords.push(remainder % dim);
                remainder /= dim;
            }
            coords.reverse();

            // The coordinate at axis is an index into indices array
            let idx_in_indices = coords[axis];
            coords[axis] = indices_data[idx_in_indices];

            // Compute input index
            let mut in_idx = 0;
            for (d, &coord) in coords.iter().enumerate() {
                in_idx += coord * strides[d];
            }

            result.push(data[in_idx]);
        }

        Array::from_vec(result, Shape::new(new_shape))
    }

    /// Gather values using n-dimensional indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    /// let indices = vec![(0, 1), (1, 2)]; // (row, col) pairs
    /// let result = a.gather_nd(&indices);
    /// // Returns values at [0,1] and [1,2]
    /// ```
    pub fn gather_nd(&self, indices: &[(usize, usize)]) -> Array {
        let data = self.to_vec();
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "gather_nd only supports 2D arrays for now");

        let cols = shape[1];
        let result: Vec<f32> = indices
            .iter()
            .map(|&(r, c)| data[r * cols + c])
            .collect();

        Array::from_vec(result, Shape::new(vec![indices.len()]))
    }

    /// Segment sum - sum elements by segment ID.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let segment_ids = Array::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0], Shape::new(vec![5]));
    /// let result = data.segment_sum(&segment_ids, 3);
    /// // segment 0: 1+2=3, segment 1: 3+4=7, segment 2: 5
    /// ```
    pub fn segment_sum(&self, segment_ids: &Array, num_segments: usize) -> Array {
        assert_eq!(self.size(), segment_ids.size(), "Data and segment_ids must have same size");

        let data = self.to_vec();
        let ids: Vec<usize> = segment_ids.to_vec().iter().map(|&x| x as usize).collect();

        let mut result = vec![0.0; num_segments];
        for (val, &seg_id) in data.iter().zip(ids.iter()) {
            if seg_id < num_segments {
                result[seg_id] += val;
            }
        }

        Array::from_vec(result, Shape::new(vec![num_segments]))
    }

    /// Segment mean - compute mean of elements by segment ID.
    pub fn segment_mean(&self, segment_ids: &Array, num_segments: usize) -> Array {
        assert_eq!(self.size(), segment_ids.size(), "Data and segment_ids must have same size");

        let data = self.to_vec();
        let ids: Vec<usize> = segment_ids.to_vec().iter().map(|&x| x as usize).collect();

        let mut sums = vec![0.0; num_segments];
        let mut counts = vec![0usize; num_segments];

        for (val, &seg_id) in data.iter().zip(ids.iter()) {
            if seg_id < num_segments {
                sums[seg_id] += val;
                counts[seg_id] += 1;
            }
        }

        let result: Vec<f32> = sums
            .iter()
            .zip(counts.iter())
            .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
            .collect();

        Array::from_vec(result, Shape::new(vec![num_segments]))
    }

    /// Segment max - compute max of elements by segment ID.
    pub fn segment_max(&self, segment_ids: &Array, num_segments: usize) -> Array {
        assert_eq!(self.size(), segment_ids.size(), "Data and segment_ids must have same size");

        let data = self.to_vec();
        let ids: Vec<usize> = segment_ids.to_vec().iter().map(|&x| x as usize).collect();

        let mut result = vec![f32::NEG_INFINITY; num_segments];

        for (val, &seg_id) in data.iter().zip(ids.iter()) {
            if seg_id < num_segments && *val > result[seg_id] {
                result[seg_id] = *val;
            }
        }

        Array::from_vec(result, Shape::new(vec![num_segments]))
    }

    /// Segment min - compute min of elements by segment ID.
    pub fn segment_min(&self, segment_ids: &Array, num_segments: usize) -> Array {
        assert_eq!(self.size(), segment_ids.size(), "Data and segment_ids must have same size");

        let data = self.to_vec();
        let ids: Vec<usize> = segment_ids.to_vec().iter().map(|&x| x as usize).collect();

        let mut result = vec![f32::INFINITY; num_segments];

        for (val, &seg_id) in data.iter().zip(ids.iter()) {
            if seg_id < num_segments && *val < result[seg_id] {
                result[seg_id] = *val;
            }
        }

        Array::from_vec(result, Shape::new(vec![num_segments]))
    }

    /// Flip array along multiple axes.
    pub fn flip_axes(&self, axes: &[usize]) -> Array {
        let mut result = self.clone();
        for &axis in axes {
            result = result.flip(axis);
        }
        result
    }

    /// Move multiple axes to new positions.
    pub fn moveaxis_multiple(&self, sources: &[usize], destinations: &[usize]) -> Array {
        assert_eq!(sources.len(), destinations.len(), "sources and destinations must have same length");

        let mut result = self.clone();
        for (&src, &dst) in sources.iter().zip(destinations.iter()) {
            result = result.moveaxis(src, dst);
        }
        result
    }

    /// Expand dimensions at multiple positions.
    pub fn expand_dims_multiple(&self, axes: &[usize]) -> Array {
        let mut sorted_axes = axes.to_vec();
        sorted_axes.sort();

        let mut result = self.clone();
        for (i, &axis) in sorted_axes.iter().enumerate() {
            result = result.expand_dims(axis + i);
        }
        result
    }

    /// Squeeze all axes with size 1.
    pub fn squeeze_all(&self) -> Array {
        let shape = self.shape().as_slice();
        let new_shape: Vec<usize> = shape.iter().cloned().filter(|&d| d != 1).collect();

        if new_shape.is_empty() {
            // Result is scalar
            return Array::from_vec(self.to_vec(), Shape::new(vec![1]));
        }

        self.reshape(Shape::new(new_shape))
    }

    /// Unflatten array - reshape the first axis into multiple dimensions.
    pub fn unflatten(&self, dim: usize, sizes: &[usize]) -> Array {
        let shape = self.shape().as_slice();
        assert!(dim < shape.len(), "dim out of bounds");
        assert_eq!(
            sizes.iter().product::<usize>(),
            shape[dim],
            "sizes must multiply to the dimension size"
        );

        let mut new_shape = Vec::with_capacity(shape.len() - 1 + sizes.len());
        new_shape.extend(&shape[..dim]);
        new_shape.extend(sizes);
        new_shape.extend(&shape[dim + 1..]);

        self.reshape(Shape::new(new_shape))
    }

    /// Repeat array elements along each axis.
    pub fn repeat_axis(&self, repeats: usize, axis: usize) -> Array {
        let shape = self.shape().as_slice();
        assert!(axis < shape.len(), "axis out of bounds");

        if axis == 0 {
            // Repeat along first axis
            let data = self.to_vec();
            let chunk_size = self.size() / shape[0];
            let mut result = Vec::with_capacity(self.size() * repeats);

            for chunk in data.chunks(chunk_size) {
                for _ in 0..repeats {
                    result.extend(chunk);
                }
            }

            let mut new_shape = shape.to_vec();
            new_shape[axis] *= repeats;

            Array::from_vec(result, Shape::new(new_shape))
        } else {
            // For other axes, transpose, repeat, transpose back
            // Simplified implementation
            let mut new_shape = shape.to_vec();
            new_shape[axis] *= repeats;

            let data = self.to_vec();
            let mut result = Vec::with_capacity(new_shape.iter().product());

            // Compute strides
            let inner_size: usize = shape[axis + 1..].iter().product();
            let outer_size: usize = shape[..axis].iter().product();
            let axis_size = shape[axis];

            for outer in 0..outer_size {
                for ax in 0..axis_size {
                    for _ in 0..repeats {
                        let start = outer * axis_size * inner_size + ax * inner_size;
                        result.extend(&data[start..start + inner_size]);
                    }
                }
            }

            Array::from_vec(result, Shape::new(new_shape))
        }
    }

    /// N-dimensional tile - repeat array along each axis.
    pub fn tile_nd(&self, reps: &[usize]) -> Array {
        assert_eq!(reps.len(), self.ndim(), "reps must have same length as ndim");

        let mut result = self.clone();
        for (axis, &rep) in reps.iter().enumerate() {
            if rep > 1 {
                result = result.repeat_axis(rep, axis);
            }
        }
        result
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
    fn test_split() {
        // Test splitting a 1D array
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![6]));
        let parts = Array::split(&a, 3, 0);

        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].to_vec(), vec![1.0, 2.0]);
        assert_eq!(parts[1].to_vec(), vec![3.0, 4.0]);
        assert_eq!(parts[2].to_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_split_2d() {
        // Test splitting a 2D array
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new(vec![4, 2]),
        );
        let parts = Array::split(&a, 2, 0);

        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape().as_slice(), &[2, 2]);
        assert_eq!(parts[0].to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(parts[1].to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
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
    fn test_where_cond_broadcast_scalar_condition() {
        let condition = Array::from_vec(vec![1.0], Shape::new(vec![1]));
        let x = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let y = Array::from_vec(vec![100.0, 200.0, 300.0], Shape::new(vec![3]));
        let result = Array::where_cond(&condition, &x, &y);
        assert_eq!(result.to_vec(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_where_cond_broadcast_scalar_x() {
        let condition = Array::from_vec(vec![1.0, 0.0, 1.0], Shape::new(vec![3]));
        let x = Array::from_vec(vec![42.0], Shape::new(vec![1]));
        let y = Array::from_vec(vec![100.0, 200.0, 300.0], Shape::new(vec![3]));
        let result = Array::where_cond(&condition, &x, &y);
        assert_eq!(result.to_vec(), vec![42.0, 200.0, 42.0]);
    }

    #[test]
    fn test_where_cond_broadcast_scalar_y() {
        let condition = Array::from_vec(vec![1.0, 0.0, 1.0], Shape::new(vec![3]));
        let x = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let y = Array::from_vec(vec![99.0], Shape::new(vec![1]));
        let result = Array::where_cond(&condition, &x, &y);
        assert_eq!(result.to_vec(), vec![10.0, 99.0, 30.0]);
    }

    #[test]
    fn test_where_cond_2d() {
        let condition = Array::from_vec(
            vec![1.0, 0.0, 0.0, 1.0],
            Shape::new(vec![2, 2])
        );
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            Shape::new(vec![2, 2])
        );
        let y = Array::from_vec(
            vec![10.0, 20.0, 30.0, 40.0],
            Shape::new(vec![2, 2])
        );
        let result = Array::where_cond(&condition, &x, &y);
        assert_eq!(result.to_vec(), vec![1.0, 20.0, 30.0, 4.0]);
    }

    #[test]
    fn test_where_cond_broadcast_2d() {
        let condition = Array::from_vec(vec![1.0, 0.0], Shape::new(vec![2]));
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            Shape::new(vec![2, 2])
        );
        let y = Array::from_vec(
            vec![10.0, 20.0, 30.0, 40.0],
            Shape::new(vec![2, 2])
        );
        let result = Array::where_cond(&condition, &x, &y);
        assert_eq!(result.to_vec(), vec![1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn test_where_cond_negative_values() {
        let condition = Array::from_vec(vec![-5.0, 0.0, 3.14], Shape::new(vec![3]));
        let x = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let y = Array::from_vec(vec![100.0, 200.0, 300.0], Shape::new(vec![3]));
        let result = Array::where_cond(&condition, &x, &y);
        // -5.0 is non-zero (true), 0.0 is zero (false), 3.14 is non-zero (true)
        assert_eq!(result.to_vec(), vec![10.0, 200.0, 30.0]);
    }

    #[test]
    fn test_select_basic() {
        let indices = Array::from_vec(vec![0.0, 1.0, 2.0, 1.0], Shape::new(vec![4]));
        let choice0 = Array::from_vec(vec![10.0, 10.0, 10.0, 10.0], Shape::new(vec![4]));
        let choice1 = Array::from_vec(vec![20.0, 20.0, 20.0, 20.0], Shape::new(vec![4]));
        let choice2 = Array::from_vec(vec![30.0, 30.0, 30.0, 30.0], Shape::new(vec![4]));
        let result = Array::select(&indices, &[choice0, choice1, choice2]);
        assert_eq!(result.to_vec(), vec![10.0, 20.0, 30.0, 20.0]);
    }

    #[test]
    fn test_select_varying_values() {
        let indices = Array::from_vec(vec![0.0, 1.0, 0.0], Shape::new(vec![3]));
        let choice0 = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let choice1 = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let result = Array::select(&indices, &[choice0, choice1]);
        // Index 0 selects from choice0: [1.0, _, 3.0]
        // Index 1 selects from choice1: [_, 20.0, _]
        assert_eq!(result.to_vec(), vec![1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_select_2d() {
        let indices = Array::from_vec(vec![0.0, 1.0, 1.0, 0.0], Shape::new(vec![2, 2]));
        let choice0 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let choice1 = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::new(vec![2, 2]));
        let result = Array::select(&indices, &[choice0, choice1]);
        assert_eq!(result.to_vec(), vec![1.0, 20.0, 30.0, 4.0]);
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

    #[test]
    fn test_roll() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::new(vec![5]),
        );
        let rolled = a.roll(2);
        assert_eq!(rolled.to_vec(), vec![4.0, 5.0, 1.0, 2.0, 3.0]);

        // Test negative roll
        let rolled_neg = a.roll(-1);
        assert_eq!(rolled_neg.to_vec(), vec![2.0, 3.0, 4.0, 5.0, 1.0]);
    }

    #[test]
    fn test_rot90() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));

        // Rotate 90 degrees
        let rot1 = a.rot90(1);
        assert_eq!(rot1.to_vec(), vec![2.0, 4.0, 1.0, 3.0]);

        // Rotate 180 degrees
        let rot2 = a.rot90(2);
        assert_eq!(rot2.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);

        // Rotate 270 degrees
        let rot3 = a.rot90(3);
        assert_eq!(rot3.to_vec(), vec![3.0, 1.0, 4.0, 2.0]);
    }

    #[test]
    fn test_swapaxes() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let swapped = a.swapaxes(0, 1);
        assert_eq!(swapped.shape().as_slice(), &[3, 2]);
        assert_eq!(swapped.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_moveaxis() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![1, 2, 3]),
        );
        let moved = a.moveaxis(2, 0);
        assert_eq!(moved.shape().as_slice(), &[3, 1, 2]);
    }

    #[test]
    fn test_interp() {
        let xp = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let fp = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let x = Array::from_vec(vec![1.5, 2.5], Shape::new(vec![2]));
        let result = Array::interp(&x, &xp, &fp);
        assert_eq!(result.to_vec(), vec![15.0, 25.0]);

        // Test edge cases
        let x_edge = Array::from_vec(vec![0.5, 3.5], Shape::new(vec![2]));
        let result_edge = Array::interp(&x_edge, &xp, &fp);
        assert_eq!(result_edge.to_vec(), vec![10.0, 30.0]);
    }

    #[test]
    fn test_lerp() {
        let a = Array::from_vec(vec![0.0, 10.0, 20.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![100.0, 110.0, 120.0], Shape::new(vec![3]));
        let result = a.lerp(&b, 0.5);
        assert_eq!(result.to_vec(), vec![50.0, 60.0, 70.0]);

        // Test with weight = 0.0 (should return a)
        let result_0 = a.lerp(&b, 0.0);
        assert_eq!(result_0.to_vec(), a.to_vec());

        // Test with weight = 1.0 (should return b)
        let result_1 = a.lerp(&b, 1.0);
        assert_eq!(result_1.to_vec(), b.to_vec());
    }

    #[test]
    fn test_lerp_array() {
        let a = Array::from_vec(vec![0.0, 10.0, 20.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![100.0, 110.0, 120.0], Shape::new(vec![3]));
        let weights = Array::from_vec(vec![0.0, 0.5, 1.0], Shape::new(vec![3]));
        let result = a.lerp_array(&b, &weights);
        assert_eq!(result.to_vec(), vec![0.0, 60.0, 120.0]);
    }

    #[test]
    fn test_put() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
        let result = a.put(&[0, 2, 4], &[10.0, 30.0, 50.0]);
        assert_eq!(result.to_vec(), vec![10.0, 2.0, 30.0, 4.0, 50.0]);
        assert_eq!(result.shape().as_slice(), &[5]);

        // Test with 2D array
        let a2d = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let result2d = a2d.put(&[0, 5], &[100.0, 600.0]);
        assert_eq!(result2d.to_vec(), vec![100.0, 2.0, 3.0, 4.0, 5.0, 600.0]);
    }

    #[test]
    fn test_scatter() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
        let result = a.scatter(&[0, 2, 4], &[10.0, 30.0, 50.0]);
        assert_eq!(result.to_vec(), vec![10.0, 2.0, 30.0, 4.0, 50.0]);
        assert_eq!(result.shape().as_slice(), &[5]);

        // Test with 2D array (flattened indexing)
        let a2d = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let result2d = a2d.scatter(&[0, 5], &[100.0, 600.0]);
        assert_eq!(result2d.to_vec(), vec![100.0, 2.0, 3.0, 4.0, 5.0, 600.0]);
    }

    #[test]
    fn test_scatter_add() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
        let result = a.scatter_add(&[0, 2, 4], &[10.0, 30.0, 50.0]);
        assert_eq!(result.to_vec(), vec![11.0, 2.0, 33.0, 4.0, 55.0]);

        // Test with duplicate indices (accumulates)
        let a2 = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let result2 = a2.scatter_add(&[0, 0, 1], &[5.0, 3.0, 10.0]);
        assert_eq!(result2.to_vec(), vec![9.0, 12.0, 3.0]); // 1+5+3=9, 2+10=12, 3
    }

    #[test]
    fn test_scatter_min() {
        let a = Array::from_vec(vec![5.0, 10.0, 15.0, 20.0, 25.0], Shape::new(vec![5]));
        let result = a.scatter_min(&[1, 2, 3], &[8.0, 20.0, 15.0]);
        assert_eq!(result.to_vec(), vec![5.0, 8.0, 15.0, 15.0, 25.0]);

        // Test where update is larger (no change)
        let a2 = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let result2 = a2.scatter_min(&[0, 1, 2], &[5.0, 10.0, 15.0]);
        assert_eq!(result2.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scatter_max() {
        let a = Array::from_vec(vec![5.0, 10.0, 15.0, 20.0, 25.0], Shape::new(vec![5]));
        let result = a.scatter_max(&[1, 2, 3], &[12.0, 10.0, 25.0]);
        assert_eq!(result.to_vec(), vec![5.0, 12.0, 15.0, 25.0, 25.0]);

        // Test where update is smaller (no change)
        let a2 = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let result2 = a2.scatter_max(&[0, 1, 2], &[5.0, 10.0, 15.0]);
        assert_eq!(result2.to_vec(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_scatter_mul() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
        let result = a.scatter_mul(&[1, 2, 3], &[2.0, 3.0, 0.5]);
        assert_eq!(result.to_vec(), vec![1.0, 4.0, 9.0, 2.0, 5.0]);

        // Test with duplicate indices (accumulates multiplication)
        let a2 = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let result2 = a2.scatter_mul(&[0, 0, 1], &[2.0, 3.0, 5.0]);
        assert_eq!(result2.to_vec(), vec![6.0, 10.0, 3.0]); // 1*2*3=6, 2*5=10, 3
    }

    #[test]
    fn test_scatter_duplicate_indices() {
        // scatter: last update wins
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let result = a.scatter(&[0, 0], &[10.0, 20.0]);
        assert_eq!(result.to_vec(), vec![20.0, 2.0, 3.0]); // Last value wins

        // scatter_add: accumulates
        let result2 = a.scatter_add(&[0, 0], &[10.0, 20.0]);
        assert_eq!(result2.to_vec(), vec![31.0, 2.0, 3.0]); // 1 + 10 + 20 = 31
    }

    #[test]
    fn test_take_along_axis_1d() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0], Shape::new(vec![5]));
        let indices = Array::from_vec(vec![0.0, 2.0, 4.0], Shape::new(vec![3]));
        let result = a.take_along_axis(&indices, 0);
        assert_eq!(result.to_vec(), vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_take_along_axis_2d_axis1() {
        let a = Array::from_vec(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            Shape::new(vec![2, 3]),
        );
        // Take column 0 from row 0 and column 2 from row 1
        let indices = Array::from_vec(vec![0.0, 2.0], Shape::new(vec![2]));
        let result = a.take_along_axis(&indices, 1);
        assert_eq!(result.to_vec(), vec![10.0, 60.0]);
    }

    #[test]
    fn test_take_along_axis_2d_axis0() {
        let a = Array::from_vec(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            Shape::new(vec![2, 3]),
        );
        // Take row 1 from column 0, row 0 from column 1, row 1 from column 2
        let indices = Array::from_vec(vec![1.0, 0.0, 1.0], Shape::new(vec![3]));
        let result = a.take_along_axis(&indices, 0);
        assert_eq!(result.to_vec(), vec![40.0, 20.0, 60.0]);
    }

    #[test]
    fn test_take_along_axis_3d() {
        // 3D array: [2, 3, 4] - 2 batches, 3 rows, 4 cols
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let a = Array::from_vec(data, Shape::new(vec![2, 3, 4]));

        // Create 3D indices with same shape except axis dimension
        // For axis=2 (last dimension), we select which column to pick at each position
        // Shape of indices: [2, 3, 2] - pick 2 values per row
        let indices = Array::from_vec(
            vec![
                0.0, 3.0, // batch 0, row 0: pick cols 0, 3
                1.0, 2.0, // batch 0, row 1: pick cols 1, 2
                0.0, 1.0, // batch 0, row 2: pick cols 0, 1
                3.0, 0.0, // batch 1, row 0: pick cols 3, 0
                2.0, 1.0, // batch 1, row 1: pick cols 2, 1
                1.0, 3.0, // batch 1, row 2: pick cols 1, 3
            ],
            Shape::new(vec![2, 3, 2]),
        );

        let result = a.take_along_axis(&indices, 2);
        assert_eq!(result.shape().as_slice(), &[2, 3, 2]);

        // Verify values
        // batch 0: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
        // batch 1: [[12,13,14,15], [16,17,18,19], [20,21,22,23]]
        // Expected results:
        // batch 0, row 0: [0, 3]
        // batch 0, row 1: [5, 6]
        // batch 0, row 2: [8, 9]
        // batch 1, row 0: [15, 12]
        // batch 1, row 1: [18, 17]
        // batch 1, row 2: [21, 23]
        assert_eq!(
            result.to_vec(),
            vec![0.0, 3.0, 5.0, 6.0, 8.0, 9.0, 15.0, 12.0, 18.0, 17.0, 21.0, 23.0]
        );
    }

    #[test]
    fn test_take_along_axis_3d_middle_axis() {
        // 3D array: [2, 3, 2]
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let a = Array::from_vec(data, Shape::new(vec![2, 3, 2]));

        // For axis=1 (middle dimension), select which row at each position
        // Shape of indices: [2, 2, 2] - pick 2 rows per batch
        let indices = Array::from_vec(
            vec![
                0.0, 2.0, // batch 0, position 0: pick rows 0, 2
                1.0, 0.0, // batch 0, position 1: pick rows 1, 0
                2.0, 1.0, // batch 1, position 0: pick rows 2, 1
                0.0, 2.0, // batch 1, position 1: pick rows 0, 2
            ],
            Shape::new(vec![2, 2, 2]),
        );

        let result = a.take_along_axis(&indices, 1);
        assert_eq!(result.shape().as_slice(), &[2, 2, 2]);

        // Array layout:
        // batch 0: [[0,1], [2,3], [4,5]]
        // batch 1: [[6,7], [8,9], [10,11]]
        // Expected (for each position, pick the row specified by index):
        // [0][0][0]: row=0, col=0 => 0
        // [0][0][1]: row=2, col=1 => 5
        // [0][1][0]: row=1, col=0 => 2
        // [0][1][1]: row=0, col=1 => 1
        // [1][0][0]: row=2, col=0 => 10
        // [1][0][1]: row=1, col=1 => 9
        // [1][1][0]: row=0, col=0 => 6
        // [1][1][1]: row=2, col=1 => 11
        assert_eq!(
            result.to_vec(),
            vec![0.0, 5.0, 2.0, 1.0, 10.0, 9.0, 6.0, 11.0]
        );
    }

    #[test]
    fn test_broadcast_arrays_compatible() {
        // Test broadcasting arrays with compatible shapes
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let c = Array::from_vec(vec![100.0], Shape::new(vec![1]));

        let broadcasted = Array::broadcast_arrays(&[a, b, c]);

        assert_eq!(broadcasted.len(), 3);
        assert_eq!(broadcasted[0].shape().as_slice(), &[3]);
        assert_eq!(broadcasted[1].shape().as_slice(), &[3]);
        assert_eq!(broadcasted[2].shape().as_slice(), &[3]);

        // Verify values
        assert_eq!(broadcasted[0].to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(broadcasted[1].to_vec(), vec![10.0, 20.0, 30.0]);
        assert_eq!(broadcasted[2].to_vec(), vec![100.0, 100.0, 100.0]);
    }

    #[test]
    fn test_broadcast_arrays_2d() {
        // Test broadcasting with 2D arrays
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let b = Array::from_vec(vec![10.0, 20.0], Shape::new(vec![2, 1]));

        let broadcasted = Array::broadcast_arrays(&[a, b]);

        assert_eq!(broadcasted.len(), 2);
        assert_eq!(broadcasted[0].shape().as_slice(), &[2, 3]);
        assert_eq!(broadcasted[1].shape().as_slice(), &[2, 3]);

        // Verify broadcasting worked correctly
        assert_eq!(
            broadcasted[0].to_vec(),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            broadcasted[1].to_vec(),
            vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
        );
    }

    #[test]
    fn test_broadcast_arrays_empty() {
        // Test with empty array list
        let broadcasted = Array::broadcast_arrays(&[]);
        assert_eq!(broadcasted.len(), 0);
    }

    #[test]
    fn test_broadcast_arrays_single() {
        // Test with single array
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let broadcasted = Array::broadcast_arrays(&[a.clone()]);

        assert_eq!(broadcasted.len(), 1);
        assert_eq!(broadcasted[0].shape().as_slice(), &[3]);
        assert_eq!(broadcasted[0].to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast arrays with shapes")]
    fn test_broadcast_arrays_incompatible() {
        // Test incompatible shapes - should panic with improved error message
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![10.0, 20.0], Shape::new(vec![2]));

        Array::broadcast_arrays(&[a, b]);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast array of shape")]
    fn test_broadcast_to_error_message() {
        // Test improved error message in broadcast_to
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        a.broadcast_to(Shape::new(vec![2]));
    }

    #[test]
    fn test_convolve() {
        let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
        let kernel = Array::from_vec(vec![1.0, 0.0, -1.0], Shape::new(vec![3]));
        let conv = signal.convolve(&kernel);
        // After flipping kernel to [-1, 0, 1]:
        // [1*(-1) + 2*0 + 3*1, 2*(-1) + 3*0 + 4*1, 3*(-1) + 4*0 + 5*1]
        // = [-1+0+3, -2+0+4, -3+0+5] = [2, 2, 2]
        assert_eq!(conv.to_vec(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_convolve_averaging() {
        let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
        let kernel = Array::from_vec(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], Shape::new(vec![3]));
        let conv = signal.convolve(&kernel);
        // Moving average: [(1+2+3)/3, (2+3+4)/3, (3+4+5)/3] = [2, 3, 4]
        assert_eq!(conv.to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_convolve_identity() {
        let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let kernel = Array::from_vec(vec![1.0], Shape::new(vec![1]));
        let conv = signal.convolve(&kernel);
        // Identity kernel should return the signal
        assert_eq!(conv.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_correlate() {
        let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
        let template = Array::from_vec(vec![1.0, 2.0, 1.0], Shape::new(vec![3]));
        let corr = signal.correlate(&template);
        // [1*1 + 2*2 + 3*1, 2*1 + 3*2 + 4*1, 3*1 + 4*2 + 5*1]
        assert_eq!(corr.to_vec(), vec![8.0, 12.0, 16.0]);
    }

    #[test]
    fn test_correlate_pattern_detection() {
        // Test finding a pattern in a signal
        let signal = Array::from_vec(vec![0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0], Shape::new(vec![7]));
        let pattern = Array::from_vec(vec![1.0, 2.0, 1.0], Shape::new(vec![3]));
        let corr = signal.correlate(&pattern);
        // Should have peak at position where pattern matches
        let max_idx = corr
            .to_vec()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 2); // Pattern starts at index 2 in signal
    }

    #[test]
    fn test_convolve_correlate_difference() {
        // Show the difference between convolution and correlation
        let signal = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let kernel = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));

        let conv = signal.convolve(&kernel);
        let corr = signal.correlate(&kernel);

        // They should give different results for asymmetric kernels
        assert_ne!(conv.to_vec(), corr.to_vec());
    }
}
