//! Array creation functions.

use crate::{buffer::Buffer, Array, DType, Shape};

impl Array {
    /// Create an array with evenly spaced values within a given interval.
    ///
    /// Equivalent to `arange(start, stop, step)` in NumPy/JAX.
    ///
    /// # Arguments
    ///
    /// * `start` - Start of interval (inclusive)
    /// * `stop` - End of interval (exclusive)
    /// * `step` - Spacing between values
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, DType};
    /// let a = Array::arange(0.0, 10.0, 2.0, DType::Float32);
    /// assert_eq!(a.to_vec(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    /// ```
    pub fn arange(start: f32, stop: f32, step: f32, dtype: DType) -> Self {
        assert_ne!(step, 0.0, "Step must be non-zero");
        assert_eq!(dtype, DType::Float32, "Only Float32 supported for now");

        let size = ((stop - start) / step).ceil().max(0.0) as usize;
        if size == 0 {
            return Array::zeros(Shape::new(vec![0]), dtype);
        }

        let data: Vec<f32> =
            (0..size).map(|i| start + (i as f32) * step).collect();

        let device = crate::default_device();
        let buffer = Buffer::from_f32(data, device);
        Array::from_buffer(buffer, Shape::new(vec![size]))
    }

    /// Return evenly spaced numbers over a specified interval.
    ///
    /// Returns `num` evenly spaced samples, calculated over the interval `[start, stop]`.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting value
    /// * `stop` - End value
    /// * `num` - Number of samples to generate
    /// * `endpoint` - If true, `stop` is the last sample. Otherwise excluded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, DType};
    /// let a = Array::linspace(0.0, 1.0, 5, true, DType::Float32);
    /// assert_eq!(a.to_vec(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    /// ```
    pub fn linspace(
        start: f32,
        stop: f32,
        num: usize,
        endpoint: bool,
        dtype: DType,
    ) -> Self {
        assert_eq!(dtype, DType::Float32, "Only Float32 supported for now");

        if num == 0 {
            return Array::zeros(Shape::new(vec![0]), dtype);
        }

        if num == 1 {
            return Array::full(start, Shape::new(vec![1]), dtype);
        }

        if start == stop {
            return Array::full(start, Shape::new(vec![num]), dtype);
        }

        let delta = stop - start;
        let denom = if endpoint { num - 1 } else { num } as f32;

        let data: Vec<f32> =
            (0..num).map(|i| start + (i as f32) * delta / denom).collect();

        let device = crate::default_device();
        let buffer = Buffer::from_f32(data, device);
        Array::from_buffer(buffer, Shape::new(vec![num]))
    }

    /// Return a 2-D array with ones on the diagonal and zeros elsewhere.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of rows
    /// * `m` - Number of columns (defaults to n if None)
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, DType};
    /// let i = Array::eye(3, None, DType::Float32);
    /// // [[1, 0, 0],
    /// //  [0, 1, 0],
    /// //  [0, 0, 1]]
    /// ```
    pub fn eye(n: usize, m: Option<usize>, dtype: DType) -> Self {
        assert_eq!(dtype, DType::Float32, "Only Float32 supported for now");

        let m = m.unwrap_or(n);
        let size = n * m;
        let mut data = vec![0.0; size];

        // Set diagonal elements to 1
        for i in 0..n.min(m) {
            data[i * m + i] = 1.0;
        }

        let device = crate::default_device();
        let buffer = Buffer::from_f32(data, device);
        Array::from_buffer(buffer, Shape::new(vec![n, m]))
    }

    /// Return the identity matrix (square matrix with ones on diagonal).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, DType};
    /// let i = Array::identity(3, DType::Float32);
    /// ```
    pub fn identity(n: usize, dtype: DType) -> Self {
        Self::eye(n, None, dtype)
    }

    /// Extract diagonal or construct diagonal array.
    ///
    /// If input is 1-D, constructs a 2-D array with the input on the diagonal.
    /// If input is 2-D, extracts the diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// // Construct diagonal matrix from 1-D array
    /// let v = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let d = Array::diag(&v, 0);
    /// assert_eq!(d.shape().as_slice(), &[3, 3]);
    /// assert_eq!(d.to_vec(), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
    /// ```
    pub fn diag(v: &Self, k: i32) -> Self {
        assert_eq!(v.dtype(), DType::Float32, "Only Float32 supported");

        if v.ndim() == 1 {
            // Construct diagonal matrix
            let n = v.size();
            let offset = k.unsigned_abs() as usize;
            let matrix_size = n + offset;

            let mut data = vec![0.0; matrix_size * matrix_size];
            let v_data = v.to_vec();

            for (i, &val) in v_data.iter().enumerate() {
                let (row, col) =
                    if k >= 0 { (i, i + offset) } else { (i + offset, i) };
                data[row * matrix_size + col] = val;
            }

            Self::from_vec(data, Shape::new(vec![matrix_size, matrix_size]))
        } else if v.ndim() == 2 {
            // Extract diagonal
            let shape = v.shape().as_slice();
            let (rows, cols) = (shape[0], shape[1]);
            let data = v.to_vec();

            let diag_len = if k >= 0 {
                (cols as i32 - k).min(rows as i32).max(0) as usize
            } else {
                (rows as i32 + k).min(cols as i32).max(0) as usize
            };

            let mut diag_data = Vec::with_capacity(diag_len);

            for i in 0..diag_len {
                let (row, col) = if k >= 0 {
                    (i, i + k as usize)
                } else {
                    (i + (-k) as usize, i)
                };
                diag_data.push(data[row * cols + col]);
            }

            Self::from_vec(diag_data, Shape::new(vec![diag_len]))
        } else {
            panic!("diag only supports 1-D and 2-D arrays");
        }
    }

    /// Lower triangle of an array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let m = Array::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     Shape::new(vec![3, 3])
    /// );
    /// let lower = m.tril(0);
    /// assert_eq!(lower.to_vec(), vec![1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
    /// ```
    pub fn tril(&self, k: i32) -> Self {
        assert_eq!(self.ndim(), 2, "tril only supports 2-D arrays");
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let shape = self.shape().as_slice();
        let (rows, cols) = (shape[0], shape[1]);
        let data = self.to_vec();

        let mut result = Vec::with_capacity(data.len());

        for i in 0..rows {
            for j in 0..cols {
                let val = if (j as i32) <= (i as i32 + k) {
                    data[i * cols + j]
                } else {
                    0.0
                };
                result.push(val);
            }
        }

        Self::from_vec(result, self.shape().clone())
    }

    /// Upper triangle of an array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let m = Array::from_vec(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     Shape::new(vec![3, 3])
    /// );
    /// let upper = m.triu(0);
    /// assert_eq!(upper.to_vec(), vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
    /// ```
    pub fn triu(&self, k: i32) -> Self {
        assert_eq!(self.ndim(), 2, "triu only supports 2-D arrays");
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let shape = self.shape().as_slice();
        let (rows, cols) = (shape[0], shape[1]);
        let data = self.to_vec();

        let mut result = Vec::with_capacity(data.len());

        for i in 0..rows {
            for j in 0..cols {
                let val = if (j as i32) >= (i as i32 + k) {
                    data[i * cols + j]
                } else {
                    0.0
                };
                result.push(val);
            }
        }

        Self::from_vec(result, self.shape().clone())
    }

    /// Lower triangular matrix with ones on the diagonal and below.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, DType, Shape};
    /// let tri = Array::tri(3, None, 0, DType::Float32);
    /// assert_eq!(tri.to_vec(), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn tri(n: usize, m: Option<usize>, k: i32, dtype: DType) -> Self {
        assert_eq!(dtype, DType::Float32, "Only Float32 supported");

        let cols = m.unwrap_or(n);
        let mut data = Vec::with_capacity(n * cols);

        for i in 0..n {
            for j in 0..cols {
                let val = if (j as i32) <= (i as i32 + k) { 1.0 } else { 0.0 };
                data.push(val);
            }
        }

        Self::from_vec(data, Shape::new(vec![n, cols]))
    }

    /// Create array with same shape as another, filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape, DType};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::zeros_like(&a);
    /// assert_eq!(b.to_vec(), vec![0.0, 0.0, 0.0]);
    /// ```
    pub fn zeros_like(other: &Array) -> Array {
        Array::zeros(other.shape().clone(), other.dtype())
    }

    /// Create array with same shape as another, filled with ones.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape, DType};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::ones_like(&a);
    /// assert_eq!(b.to_vec(), vec![1.0, 1.0, 1.0]);
    /// ```
    pub fn ones_like(other: &Array) -> Array {
        Array::ones(other.shape().clone(), other.dtype())
    }

    /// Create array with same shape as another, filled with a constant value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape, DType};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::full_like(&a, 42.0);
    /// assert_eq!(b.to_vec(), vec![42.0, 42.0, 42.0]);
    /// ```
    pub fn full_like(other: &Array, value: f32) -> Array {
        Array::full(value, other.shape().clone(), other.dtype())
    }

    /// Repeat array along specified axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = a.repeat(3, 0);
    /// assert_eq!(b.to_vec(), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    /// ```
    pub fn repeat(&self, repeats: usize, axis: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert!(axis < self.ndim(), "Axis out of bounds");

        let shape = self.shape().as_slice();
        let data = self.to_vec();

        // For 1D case
        if self.ndim() == 1 {
            let mut result = Vec::with_capacity(data.len() * repeats);
            for &val in data.iter() {
                for _ in 0..repeats {
                    result.push(val);
                }
            }
            return Array::from_vec(result, Shape::new(vec![shape[0] * repeats]));
        }

        // For higher dimensions, only support axis 0 for now
        assert_eq!(axis, 0, "repeat only supports axis=0 for multi-dimensional arrays");

        let slice_size = data.len() / shape[0];
        let mut result = Vec::with_capacity(data.len() * repeats);

        for i in 0..shape[0] {
            let start = i * slice_size;
            let end = start + slice_size;
            for _ in 0..repeats {
                result.extend_from_slice(&data[start..end]);
            }
        }

        let mut result_shape = shape.to_vec();
        result_shape[axis] *= repeats;
        Array::from_vec(result, Shape::new(result_shape))
    }

    /// Tile array by repeating it multiple times.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = a.tile(3);
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    /// ```
    pub fn tile(&self, reps: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let mut result = Vec::with_capacity(data.len() * reps);

        for _ in 0..reps {
            result.extend_from_slice(&data);
        }

        let shape = self.shape().as_slice();
        let mut result_shape = shape.to_vec();
        result_shape[0] *= reps;

        Array::from_vec(result, Shape::new(result_shape))
    }

    /// Create coordinate matrices from coordinate vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let x = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let y = Array::from_vec(vec![3.0, 4.0, 5.0], Shape::new(vec![3]));
    /// let (xx, yy) = Array::meshgrid(&x, &y);
    /// assert_eq!(xx.shape().as_slice(), &[3, 2]);
    /// assert_eq!(yy.shape().as_slice(), &[3, 2]);
    /// ```
    pub fn meshgrid(x: &Array, y: &Array) -> (Array, Array) {
        assert_eq!(x.ndim(), 1, "meshgrid requires 1D arrays");
        assert_eq!(y.ndim(), 1, "meshgrid requires 1D arrays");

        let x_data = x.to_vec();
        let y_data = y.to_vec();
        let nx = x_data.len();
        let ny = y_data.len();

        // Create XX: repeat x along rows
        let mut xx_data = Vec::with_capacity(nx * ny);
        for _ in 0..ny {
            xx_data.extend_from_slice(&x_data);
        }

        // Create YY: repeat each y value nx times
        let mut yy_data = Vec::with_capacity(nx * ny);
        for &y_val in y_data.iter() {
            for _ in 0..nx {
                yy_data.push(y_val);
            }
        }

        let xx = Array::from_vec(xx_data, Shape::new(vec![ny, nx]));
        let yy = Array::from_vec(yy_data, Shape::new(vec![ny, nx]));

        (xx, yy)
    }

    /// Generate arrays of indices for each dimension.
    ///
    /// Returns a vector of arrays, one for each dimension, containing the indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let indices = Array::indices(&[2, 3]);
    /// assert_eq!(indices[0].shape().as_slice(), &[2, 3]);
    /// assert_eq!(indices[1].shape().as_slice(), &[2, 3]);
    /// ```
    pub fn indices(dimensions: &[usize]) -> Vec<Array> {
        let total_size: usize = dimensions.iter().product();
        let mut result = Vec::with_capacity(dimensions.len());

        for (dim_idx, &dim_size) in dimensions.iter().enumerate() {
            let mut data = Vec::with_capacity(total_size);

            // Calculate stride for this dimension
            let stride: usize = dimensions.iter().skip(dim_idx + 1).product();

            for i in 0..total_size {
                let idx = (i / stride) % dim_size;
                data.push(idx as f32);
            }

            result.push(Array::from_vec(data, Shape::new(dimensions.to_vec())));
        }

        result
    }

    /// Convert a flat index to multi-dimensional coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let shape = Shape::new(vec![3, 4]);
    /// let coords = Array::unravel_index(5, &shape);
    /// assert_eq!(coords, vec![1, 1]);
    /// ```
    pub fn unravel_index(index: usize, shape: &Shape) -> Vec<usize> {
        let dims = shape.as_slice();
        let mut coords = vec![0; dims.len()];
        let mut idx = index;

        for i in (0..dims.len()).rev() {
            coords[i] = idx % dims[i];
            idx /= dims[i];
        }

        coords
    }

    /// Convert multi-dimensional coordinates to a flat index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let shape = Shape::new(vec![3, 4]);
    /// let index = Array::ravel_multi_index(&[1, 2], &shape);
    /// assert_eq!(index, 6);
    /// ```
    pub fn ravel_multi_index(multi_index: &[usize], shape: &Shape) -> usize {
        let dims = shape.as_slice();
        assert_eq!(
            multi_index.len(),
            dims.len(),
            "Index dimensions must match shape"
        );

        let mut index = 0;
        let mut stride = 1;

        for i in (0..dims.len()).rev() {
            assert!(
                multi_index[i] < dims[i],
                "Index out of bounds at dimension {}", i
            );
            index += multi_index[i] * stride;
            stride *= dims[i];
        }

        index
    }

    /// Return indices for the main diagonal of an n-by-n array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let (rows, cols) = Array::diag_indices(3);
    /// assert_eq!(rows, vec![0, 1, 2]);
    /// assert_eq!(cols, vec![0, 1, 2]);
    /// ```
    pub fn diag_indices(n: usize) -> (Vec<usize>, Vec<usize>) {
        let indices: Vec<usize> = (0..n).collect();
        (indices.clone(), indices)
    }

    /// Return indices for the lower triangle of an n-by-n array.
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the arrays for which the indices are returned
    /// * `k` - Diagonal offset (0 for main diagonal, positive for above, negative for below)
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let (rows, cols) = Array::tril_indices(3, 0);
    /// assert_eq!(rows, vec![0, 1, 1, 2, 2, 2]);
    /// assert_eq!(cols, vec![0, 0, 1, 0, 1, 2]);
    /// ```
    pub fn tril_indices(n: usize, k: isize) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if (j as isize) <= (i as isize + k) {
                    rows.push(i);
                    cols.push(j);
                }
            }
        }

        (rows, cols)
    }

    /// Return indices for the upper triangle of an n-by-n array.
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the arrays for which the indices are returned
    /// * `k` - Diagonal offset (0 for main diagonal, positive for above, negative for below)
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let (rows, cols) = Array::triu_indices(3, 0);
    /// assert_eq!(rows, vec![0, 0, 0, 1, 1, 2]);
    /// assert_eq!(cols, vec![0, 1, 2, 1, 2, 2]);
    /// ```
    pub fn triu_indices(n: usize, k: isize) -> (Vec<usize>, Vec<usize>) {
        let mut rows = Vec::new();
        let mut cols = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if (j as isize) >= (i as isize + k) {
                    rows.push(i);
                    cols.push(j);
                }
            }
        }

        (rows, cols)
    }

    /// Return numbers spaced evenly on a log scale (geometric progression).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, DType};
    /// let a = Array::geomspace(1.0, 1000.0, 4, DType::Float32);
    /// // Result: [1.0, 10.0, 100.0, 1000.0]
    /// ```
    pub fn geomspace(start: f32, stop: f32, num: usize, dtype: DType) -> Self {
        assert_eq!(dtype, DType::Float32, "Only Float32 supported for now");
        assert!(num > 0, "Number of samples must be positive");
        assert!(start > 0.0 && stop > 0.0, "Start and stop must be positive for geomspace");

        if num == 1 {
            return Array::from_vec(vec![start], Shape::new(vec![1]));
        }

        let log_start = start.ln();
        let log_stop = stop.ln();
        let step = (log_stop - log_start) / (num - 1) as f32;

        let mut data = Vec::with_capacity(num);
        for i in 0..num {
            data.push((log_start + step * i as f32).exp());
        }

        let device = crate::default_device();
        let buffer = Buffer::from_f32(data, device);
        Array::from_buffer(buffer, Shape::new(vec![num]))
    }

    /// Return numbers spaced evenly on a log scale.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, DType};
    /// let a = Array::logspace(0.0, 3.0, 4, DType::Float32);
    /// // Result: [1.0, 10.0, 100.0, 1000.0] (10^0 to 10^3)
    /// ```
    pub fn logspace(start: f32, stop: f32, num: usize, dtype: DType) -> Self {
        assert_eq!(dtype, DType::Float32, "Only Float32 supported for now");
        assert!(num > 0, "Number of samples must be positive");

        if num == 1 {
            return Array::from_vec(vec![10_f32.powf(start)], Shape::new(vec![1]));
        }

        let step = (stop - start) / (num - 1) as f32;
        let mut data = Vec::with_capacity(num);
        for i in 0..num {
            data.push(10_f32.powf(start + step * i as f32));
        }

        let device = crate::default_device();
        let buffer = Buffer::from_f32(data, device);
        Array::from_buffer(buffer, Shape::new(vec![num]))
    }

    /// Create empty array with same shape (uninitialized memory).
    /// Note: In this implementation, we return zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.empty_like();
    /// assert_eq!(b.shape().as_slice(), &[3]);
    /// ```
    pub fn empty_like(&self) -> Array {
        // In practice, we return zeros for safety
        Array::zeros(self.shape().clone(), self.dtype())
    }

    /// Check if array is C-contiguous (row-major order).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// assert!(a.is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        // Our arrays are always contiguous
        true
    }

    /// Check if array is Fortran-contiguous (column-major order).
    /// Note: Our arrays are always C-contiguous.
    pub fn is_fortran_contiguous(&self) -> bool {
        // Single-dimension arrays are both C and Fortran contiguous
        self.ndim() <= 1
    }

    /// Return a contiguous array in memory (C order).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let b = a.ascontiguousarray();
    /// assert!(b.is_contiguous());
    /// ```
    pub fn ascontiguousarray(&self) -> Array {
        // Already contiguous, just clone
        self.clone()
    }

    /// Create a Hamming window of given length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let w = Array::hamming(5);
    /// assert_eq!(w.shape().as_slice(), &[5]);
    /// ```
    pub fn hamming(n: usize) -> Array {
        let data: Vec<f32> = (0..n)
            .map(|i| {
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()
            })
            .collect();
        let buffer = Buffer::from_f32(data, crate::default_device());
        Array::from_buffer(buffer, Shape::new(vec![n]))
    }

    /// Create a Hanning window of given length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let w = Array::hanning(5);
    /// assert_eq!(w.shape().as_slice(), &[5]);
    /// ```
    pub fn hanning(n: usize) -> Array {
        let data: Vec<f32> = (0..n)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())
            })
            .collect();
        let buffer = Buffer::from_f32(data, crate::default_device());
        Array::from_buffer(buffer, Shape::new(vec![n]))
    }

    /// Create a Blackman window of given length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let w = Array::blackman(5);
    /// assert_eq!(w.shape().as_slice(), &[5]);
    /// ```
    pub fn blackman(n: usize) -> Array {
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32 / (n - 1) as f32;
                0.42 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
                    + 0.08 * (4.0 * std::f32::consts::PI * x).cos()
            })
            .collect();
        let buffer = Buffer::from_f32(data, crate::default_device());
        Array::from_buffer(buffer, Shape::new(vec![n]))
    }

    /// Create a Kaiser window of given length and beta parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let w = Array::kaiser(5, 5.0);
    /// assert_eq!(w.shape().as_slice(), &[5]);
    /// ```
    pub fn kaiser(n: usize, beta: f32) -> Array {
        // Approximate Bessel I0 function
        fn i0(x: f32) -> f32 {
            let ax = x.abs();
            if ax < 3.75 {
                let y = (x / 3.75).powi(2);
                1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                    + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
            } else {
                let y = 3.75 / ax;
                (ax.exp() / ax.sqrt()) * (0.398_942_3 + y * (0.01328592
                    + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281
                    + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633
                    + y * 0.00392377))))))))
            }
        }

        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = 2.0 * i as f32 / (n - 1) as f32 - 1.0;
                i0(beta * (1.0 - x * x).sqrt()) / i0(beta)
            })
            .collect();
        let buffer = Buffer::from_f32(data, crate::default_device());
        Array::from_buffer(buffer, Shape::new(vec![n]))
    }

    /// Create a Bartlett (triangular) window of given length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let w = Array::bartlett(5);
    /// assert_eq!(w.shape().as_slice(), &[5]);
    /// ```
    pub fn bartlett(n: usize) -> Array {
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = i as f32;
                let half = (n - 1) as f32 / 2.0;
                1.0 - ((x - half) / half).abs()
            })
            .collect();
        let buffer = Buffer::from_f32(data, crate::default_device());
        Array::from_buffer(buffer, Shape::new(vec![n]))
    }

    /// Create a flat top window of given length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let w = Array::flattop(5);
    /// assert_eq!(w.shape().as_slice(), &[5]);
    /// ```
    pub fn flattop(n: usize) -> Array {
        let a0 = 0.21557895;
        let a1 = 0.41663158;
        let a2 = 0.277_263_16;
        let a3 = 0.083578947;
        let a4 = 0.006947368;

        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = 2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32;
                a0 - a1 * x.cos() + a2 * (2.0 * x).cos()
                   - a3 * (3.0 * x).cos() + a4 * (4.0 * x).cos()
            })
            .collect();
        let buffer = Buffer::from_f32(data, crate::default_device());
        Array::from_buffer(buffer, Shape::new(vec![n]))
    }

    /// Create a triangular window of given length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Array;
    /// let w = Array::triang(5);
    /// assert_eq!(w.shape().as_slice(), &[5]);
    /// assert!((w.to_vec()[2] - 1.0).abs() < 1e-6); // Peak at center
    /// ```
    pub fn triang(n: usize) -> Array {
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let half = (n as f32 + 1.0) / 2.0;
                if i as f32 + 1.0 <= half {
                    2.0 * (i as f32 + 1.0) / (n as f32 + 1.0)
                } else {
                    2.0 - 2.0 * (i as f32 + 1.0) / (n as f32 + 1.0)
                }
            })
            .collect();
        let buffer = Buffer::from_f32(data, crate::default_device());
        Array::from_buffer(buffer, Shape::new(vec![n]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_arange() {
        let a = Array::arange(0.0, 10.0, 2.0, DType::Float32);
        assert_eq!(a.to_vec(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);

        let b = Array::arange(0.0, 5.0, 1.0, DType::Float32);
        assert_eq!(b.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let c = Array::arange(1.0, 2.0, 0.25, DType::Float32);
        assert_eq!(c.to_vec(), vec![1.0, 1.25, 1.5, 1.75]);
    }

    #[test]
    fn test_arange_negative_step() {
        let a = Array::arange(10.0, 0.0, -2.0, DType::Float32);
        assert_eq!(a.to_vec(), vec![10.0, 8.0, 6.0, 4.0, 2.0]);
    }

    #[test]
    fn test_arange_empty() {
        let a = Array::arange(0.0, 0.0, 1.0, DType::Float32);
        assert_eq!(a.size(), 0);
    }

    #[test]
    #[should_panic(expected = "Step must be non-zero")]
    fn test_arange_zero_step() {
        let _a = Array::arange(0.0, 10.0, 0.0, DType::Float32);
    }

    #[test]
    fn test_linspace() {
        let a = Array::linspace(0.0, 1.0, 5, true, DType::Float32);
        let expected = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        for (i, &val) in a.to_vec().iter().enumerate() {
            assert_abs_diff_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_linspace_no_endpoint() {
        let a = Array::linspace(0.0, 1.0, 5, false, DType::Float32);
        let expected = vec![0.0, 0.2, 0.4, 0.6, 0.8];
        for (i, &val) in a.to_vec().iter().enumerate() {
            assert_abs_diff_eq!(val, expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_linspace_single() {
        let a = Array::linspace(5.0, 10.0, 1, true, DType::Float32);
        assert_eq!(a.to_vec(), vec![5.0]);
    }

    #[test]
    fn test_linspace_same_start_stop() {
        let a = Array::linspace(5.0, 5.0, 10, true, DType::Float32);
        assert!(a.to_vec().iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_eye() {
        let i = Array::eye(3, None, DType::Float32);
        assert_eq!(i.shape().as_slice(), &[3, 3]);
        assert_eq!(
            i.to_vec(),
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_eye_rectangular() {
        let i = Array::eye(2, Some(4), DType::Float32);
        assert_eq!(i.shape().as_slice(), &[2, 4]);
        assert_eq!(i.to_vec(), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_identity() {
        let i = Array::identity(4, DType::Float32);
        assert_eq!(i.shape().as_slice(), &[4, 4]);
        // Check diagonal
        let data = i.to_vec();
        for idx in 0..4 {
            assert_eq!(data[idx * 4 + idx], 1.0);
        }
        // Check off-diagonal (sample a few)
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[4], 0.0);
    }

    #[test]
    fn test_indices() {
        let indices = Array::indices(&[2, 3]);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].shape().as_slice(), &[2, 3]);
        assert_eq!(indices[1].shape().as_slice(), &[2, 3]);
        // First dimension varies along rows
        assert_eq!(indices[0].to_vec(), vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        // Second dimension varies along columns
        assert_eq!(indices[1].to_vec(), vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_unravel_index() {
        let shape = Shape::new(vec![3, 4]);
        assert_eq!(Array::unravel_index(0, &shape), vec![0, 0]);
        assert_eq!(Array::unravel_index(5, &shape), vec![1, 1]);
        assert_eq!(Array::unravel_index(11, &shape), vec![2, 3]);
    }

    #[test]
    fn test_ravel_multi_index() {
        let shape = Shape::new(vec![3, 4]);
        assert_eq!(Array::ravel_multi_index(&[0, 0], &shape), 0);
        assert_eq!(Array::ravel_multi_index(&[1, 2], &shape), 6);
        assert_eq!(Array::ravel_multi_index(&[2, 3], &shape), 11);
    }

    #[test]
    fn test_diag_indices() {
        let (rows, cols) = Array::diag_indices(3);
        assert_eq!(rows, vec![0, 1, 2]);
        assert_eq!(cols, vec![0, 1, 2]);
    }

    #[test]
    fn test_tril_indices() {
        let (rows, cols) = Array::tril_indices(3, 0);
        assert_eq!(rows, vec![0, 1, 1, 2, 2, 2]);
        assert_eq!(cols, vec![0, 0, 1, 0, 1, 2]);

        // Test with offset
        let (rows2, cols2) = Array::tril_indices(3, 1);
        assert_eq!(rows2, vec![0, 0, 1, 1, 1, 2, 2, 2]);
        assert_eq!(cols2, vec![0, 1, 0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_triu_indices() {
        let (rows, cols) = Array::triu_indices(3, 0);
        assert_eq!(rows, vec![0, 0, 0, 1, 1, 2]);
        assert_eq!(cols, vec![0, 1, 2, 1, 2, 2]);

        // Test with offset k=-1 (includes first subdiagonal)
        let (rows2, cols2) = Array::triu_indices(3, -1);
        assert_eq!(rows2, vec![0, 0, 0, 1, 1, 1, 2, 2]);
        assert_eq!(cols2, vec![0, 1, 2, 0, 1, 2, 1, 2]);
    }
}
