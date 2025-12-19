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
}
