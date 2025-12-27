//! Linear algebra operations.

use crate::{buffer::Buffer, Array, DType, Device, Shape};

impl Array {
    /// Transpose the array by reversing its axes.
    ///
    /// For 2D arrays, this swaps rows and columns.
    /// For 1D arrays, returns a copy.
    /// For higher dimensions, reverses all axes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    /// let b = a.transpose();
    /// assert_eq!(b.shape().as_slice(), &[3, 2]);
    /// ```
    pub fn transpose(&self) -> Array {
        let shape = self.shape();
        let dims = shape.as_slice();

        if dims.len() <= 1 {
            // For scalars and 1D arrays, transpose is identity
            return self.clone();
        }

        // Reverse the dimensions
        let new_dims: Vec<usize> = dims.iter().rev().copied().collect();
        let new_shape = Shape::new(new_dims);

        // For 2D case, implement explicit transpose
        if dims.len() == 2 {
            let (rows, cols) = (dims[0], dims[1]);
            let data = self.to_vec();
            let mut transposed = vec![0.0; data.len()];

            for i in 0..rows {
                for j in 0..cols {
                    transposed[j * rows + i] = data[i * cols + j];
                }
            }

            let buffer = Buffer::from_f32(transposed, Device::Cpu);
            return Array::from_buffer(buffer, new_shape);
        }

        // For higher dimensions, use general algorithm
        transpose_nd(self, new_shape)
    }

    /// Transpose the array with a specified permutation of axes.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of axes
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    /// let b = a.transpose_axes(&[1, 0]);
    /// assert_eq!(b.shape().as_slice(), &[3, 2]);
    /// ```
    pub fn transpose_axes(&self, axes: &[usize]) -> Array {
        let shape = self.shape();
        let dims = shape.as_slice();
        let ndim = dims.len();

        assert_eq!(axes.len(), ndim, "axes must have same length as dimensions");

        // Verify axes is a valid permutation
        let mut seen = vec![false; ndim];
        for &axis in axes {
            assert!(axis < ndim, "axis {} out of bounds for {} dimensions", axis, ndim);
            assert!(!seen[axis], "duplicate axis in permutation");
            seen[axis] = true;
        }

        // If axes is identity permutation, return self
        if axes.iter().enumerate().all(|(i, &a)| i == a) {
            return self.clone();
        }

        // Compute new shape
        let new_dims: Vec<usize> = axes.iter().map(|&a| dims[a]).collect();
        let new_shape = Shape::new(new_dims.clone());

        // Simple 2D case
        if ndim == 2 && axes == &[1, 0] {
            return self.transpose();
        }

        // General case: compute transposed data
        let data = self.to_vec();
        let size = data.len();
        let mut result = vec![0.0; size];

        // Compute strides for original array
        let mut old_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            old_strides[i] = old_strides[i + 1] * dims[i + 1];
        }

        // Compute strides for new array
        let mut new_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_dims[i + 1];
        }

        // Map strides according to permutation
        let perm_strides: Vec<usize> = axes.iter().map(|&a| old_strides[a]).collect();

        // Copy data with transposition
        for new_idx in 0..size {
            // Convert flat index to multi-index in new array
            let mut remaining = new_idx;
            let mut old_idx = 0;
            for i in 0..ndim {
                let coord = remaining / new_strides[i];
                remaining %= new_strides[i];
                old_idx += coord * perm_strides[i];
            }
            result[new_idx] = data[old_idx];
        }

        let buffer = Buffer::from_f32(result, Device::Cpu);
        Array::from_buffer(buffer, new_shape)
    }

    /// Matrix multiplication of two 2D arrays.
    ///
    /// # Arguments
    ///
    /// * `other` - The right-hand array to multiply with
    ///
    /// # Panics
    ///
    /// Panics if shapes are incompatible or arrays are not 2D.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
    /// let c = a.matmul(&b);
    /// // [[1*5 + 2*7, 1*6 + 2*8],
    /// //  [3*5 + 4*7, 3*6 + 4*8]]
    /// // = [[19, 22], [43, 50]]
    /// ```
    pub fn matmul(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let a_shape = self.shape().as_slice();
        let b_shape = other.shape().as_slice();

        // Handle vector-matrix and matrix-vector cases
        if a_shape.len() == 1 && b_shape.len() == 2 {
            // Vector-matrix: (N,) @ (N, M) -> (M,)
            assert_eq!(
                a_shape[0], b_shape[0],
                "Vector-matrix multiplication: incompatible shapes"
            );
            return self
                .reshape(Shape::new(vec![1, a_shape[0]]))
                .matmul(other)
                .reshape(Shape::new(vec![b_shape[1]]));
        }

        if a_shape.len() == 2 && b_shape.len() == 1 {
            // Matrix-vector: (M, N) @ (N,) -> (M,)
            assert_eq!(
                a_shape[1], b_shape[0],
                "Matrix-vector multiplication: incompatible shapes"
            );
            return self
                .matmul(&other.reshape(Shape::new(vec![b_shape[0], 1])))
                .reshape(Shape::new(vec![a_shape[0]]));
        }

        // Matrix-matrix multiplication
        assert_eq!(a_shape.len(), 2, "Left array must be 2D");
        assert_eq!(b_shape.len(), 2, "Right array must be 2D");
        assert_eq!(
            a_shape[1], b_shape[0],
            "Incompatible shapes for matmul: {:?} @ {:?}",
            a_shape, b_shape
        );

        let (m, k) = (a_shape[0], a_shape[1]);
        let n = b_shape[1];

        // Dispatch based on device
        match (self.device(), other.device()) {
            (Device::WebGpu, Device::WebGpu) => {
                // GPU path
                let output_buffer = Buffer::zeros(m * n, DType::Float32, Device::WebGpu);

                crate::backend::ops::gpu_matmul(
                    self.buffer(),
                    other.buffer(),
                    &output_buffer,
                    m,
                    n,
                    k,
                );

                Array::from_buffer(output_buffer, Shape::new(vec![m, n]))
            }
            (Device::Cpu, Device::Cpu) | (Device::Wasm, Device::Wasm) => {
                // CPU path - naive O(n^3) algorithm
                let a_data = self.to_vec();
                let b_data = other.to_vec();
                let mut result = vec![0.0; m * n];

                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for p in 0..k {
                            sum += a_data[i * k + p] * b_data[p * n + j];
                        }
                        result[i * n + j] = sum;
                    }
                }

                let buffer = Buffer::from_f32(result, Device::Cpu);
                Array::from_buffer(buffer, Shape::new(vec![m, n]))
            }
            _ => {
                panic!("Mixed device operations not supported. Both arrays must be on the same device.");
            }
        }
    }

    /// Dot product of two arrays.
    ///
    /// For 1D arrays, computes the inner product.
    /// For 2D arrays, equivalent to matmul.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
    /// let c = a.dot(&b);
    /// assert_eq!(c.to_vec(), vec![32.0]); // 1*4 + 2*5 + 3*6 = 32
    /// ```
    pub fn dot(&self, other: &Array) -> Array {
        let a_shape = self.shape().as_slice();
        let b_shape = other.shape().as_slice();

        // 1D dot product (inner product)
        if a_shape.len() == 1 && b_shape.len() == 1 {
            assert_eq!(
                a_shape[0], b_shape[0],
                "Arrays must have same length for dot product"
            );

            let a_data = self.to_vec();
            let b_data = other.to_vec();
            let result: f32 =
                a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).sum();

            let buffer = Buffer::from_f32(vec![result], Device::Cpu);
            return Array::from_buffer(buffer, Shape::scalar());
        }

        // For higher dimensions, use matmul
        self.matmul(other)
    }

    /// Compute the norm of a vector or matrix.
    ///
    /// # Arguments
    ///
    /// * `ord` - Order of the norm. Common values:
    ///   - `1.0`: L1 norm (sum of absolute values)
    ///   - `2.0`: L2 norm (Euclidean norm)
    ///   - `f32::INFINITY`: L-infinity norm (maximum absolute value)
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
    /// let l2_norm = a.norm(2.0);
    /// assert_eq!(l2_norm, 5.0); // sqrt(3^2 + 4^2) = 5
    /// ```
    pub fn norm(&self, ord: f32) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();

        if ord == f32::INFINITY {
            // L-infinity norm: max absolute value
            data.iter().map(|x| x.abs()).fold(0.0, f32::max)
        } else if ord == 1.0 {
            // L1 norm: sum of absolute values
            data.iter().map(|x| x.abs()).sum()
        } else if ord == 2.0 {
            // L2 norm: Euclidean norm
            data.iter().map(|x| x * x).sum::<f32>().sqrt()
        } else {
            // General Lp norm: (sum |x|^p)^(1/p)
            data.iter()
                .map(|x| x.abs().powf(ord))
                .sum::<f32>()
                .powf(1.0 / ord)
        }
    }

    /// Compute the determinant of a square matrix.
    ///
    /// Uses LU decomposition for matrices larger than 3x3.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let det = a.det();
    /// assert_eq!(det, -2.0); // 1*4 - 2*3 = -2
    /// ```
    pub fn det(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "Determinant requires 2D array");
        assert_eq!(shape[0], shape[1], "Determinant requires square matrix");

        let n = shape[0];
        let data = self.to_vec();

        match n {
            1 => data[0],
            2 => {
                // 2x2: ad - bc
                data[0] * data[3] - data[1] * data[2]
            }
            3 => {
                // 3x3: Sarrus rule
                let a = data[0];
                let b = data[1];
                let c = data[2];
                let d = data[3];
                let e = data[4];
                let f = data[5];
                let g = data[6];
                let h = data[7];
                let i = data[8];
                a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h
            }
            _ => {
                // For larger matrices, use LU decomposition
                let (_, u, p) = self.lu_decomposition();
                let u_data = u.to_vec();

                // det(A) = det(P) * det(L) * det(U)
                // det(L) = 1 (unit diagonal)
                // det(U) = product of diagonal elements
                // det(P) = (-1)^(number of swaps)
                let mut det_u = 1.0;
                for i in 0..n {
                    det_u *= u_data[i * n + i];
                }

                // Count permutation parity
                let mut swaps = 0;
                let mut visited = vec![false; n];
                for i in 0..n {
                    if !visited[i] {
                        let mut j = i;
                        let mut cycle_len = 0;
                        while !visited[j] {
                            visited[j] = true;
                            j = p[j];
                            cycle_len += 1;
                        }
                        if cycle_len > 1 {
                            swaps += cycle_len - 1;
                        }
                    }
                }

                if swaps % 2 == 0 {
                    det_u
                } else {
                    -det_u
                }
            }
        }
    }

    /// LU decomposition with partial pivoting.
    ///
    /// Returns (L, U, P) where:
    /// - L is lower triangular with unit diagonal
    /// - U is upper triangular
    /// - P is permutation array (row swaps)
    fn lu_decomposition(&self) -> (Array, Array, Vec<usize>) {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "LU decomposition requires 2D array");
        assert_eq!(shape[0], shape[1], "LU decomposition requires square matrix");

        let n = shape[0];
        let data = self.to_vec();

        // Initialize permutation
        let mut p: Vec<usize> = (0..n).collect();
        let mut a = data.clone();

        for k in 0..n {
            // Find pivot
            let mut pivot_row = k;
            let mut max_val = a[k * n + k].abs();
            for i in (k + 1)..n {
                let val = a[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = i;
                }
            }

            // Swap rows if needed
            if pivot_row != k {
                p.swap(k, pivot_row);
                for j in 0..n {
                    a.swap(k * n + j, pivot_row * n + j);
                }
            }

            // Eliminate column
            for i in (k + 1)..n {
                let factor = a[i * n + k] / a[k * n + k];
                a[i * n + k] = factor; // Store L factor in lower triangle
                for j in (k + 1)..n {
                    a[i * n + j] -= factor * a[k * n + j];
                }
            }
        }

        // Extract L and U
        let mut l_data = vec![0.0; n * n];
        let mut u_data = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                if i > j {
                    l_data[i * n + j] = a[i * n + j];
                } else if i == j {
                    l_data[i * n + j] = 1.0;
                    u_data[i * n + j] = a[i * n + j];
                } else {
                    u_data[i * n + j] = a[i * n + j];
                }
            }
        }

        let l = Array::from_vec(l_data, Shape::new(vec![n, n]));
        let u = Array::from_vec(u_data, Shape::new(vec![n, n]));

        (l, u, p)
    }

    /// Compute the matrix inverse.
    ///
    /// Uses Gauss-Jordan elimination.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![4.0, 7.0, 2.0, 6.0], Shape::new(vec![2, 2]));
    /// let inv_a = a.inv();
    /// // Verify A * A^-1 = I
    /// let identity = a.matmul(&inv_a);
    /// let expected = vec![1.0, 0.0, 0.0, 1.0];
    /// for (i, &val) in identity.to_vec().iter().enumerate() {
    ///     assert!((val - expected[i]).abs() < 1e-5);
    /// }
    /// ```
    pub fn inv(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "Matrix inversion requires 2D array");
        assert_eq!(shape[0], shape[1], "Matrix inversion requires square matrix");

        let n = shape[0];
        let data = self.to_vec();

        // Create augmented matrix [A | I]
        let mut aug = vec![0.0; n * 2 * n];
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = data[i * n + j];
            }
            aug[i * 2 * n + n + i] = 1.0; // Identity on the right
        }

        // Gauss-Jordan elimination
        for k in 0..n {
            // Find pivot
            let mut pivot_row = k;
            let mut max_val = aug[k * 2 * n + k].abs();
            for i in (k + 1)..n {
                let val = aug[i * 2 * n + k].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = i;
                }
            }

            assert!(
                max_val > 1e-10,
                "Matrix is singular and cannot be inverted"
            );

            // Swap rows
            if pivot_row != k {
                for j in 0..(2 * n) {
                    aug.swap(k * 2 * n + j, pivot_row * 2 * n + j);
                }
            }

            // Scale pivot row
            let pivot = aug[k * 2 * n + k];
            for j in 0..(2 * n) {
                aug[k * 2 * n + j] /= pivot;
            }

            // Eliminate column
            for i in 0..n {
                if i != k {
                    let factor = aug[i * 2 * n + k];
                    for j in 0..(2 * n) {
                        aug[i * 2 * n + j] -= factor * aug[k * 2 * n + j];
                    }
                }
            }
        }

        // Extract inverse from right half
        let mut inv_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                inv_data[i * n + j] = aug[i * 2 * n + n + j];
            }
        }

        Array::from_vec(inv_data, Shape::new(vec![n, n]))
    }

    /// Solve a linear system Ax = b.
    ///
    /// Uses Gaussian elimination with partial pivoting.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// // Solve: 2x + y = 5, x + 3y = 6
    /// let a = Array::from_vec(vec![2.0, 1.0, 1.0, 3.0], Shape::new(vec![2, 2]));
    /// let b = Array::from_vec(vec![5.0, 6.0], Shape::new(vec![2]));
    /// let x = a.solve(&b);
    /// // Solution: x = [1.8, 1.4]
    /// assert!((x.to_vec()[0] - 1.8).abs() < 1e-5);
    /// assert!((x.to_vec()[1] - 1.4).abs() < 1e-5);
    /// ```
    pub fn solve(&self, b: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(b.dtype(), DType::Float32, "Only Float32 supported");

        let a_shape = self.shape().as_slice();
        let b_shape = b.shape().as_slice();

        assert_eq!(a_shape.len(), 2, "A must be 2D");
        assert_eq!(a_shape[0], a_shape[1], "A must be square");
        assert_eq!(b_shape.len(), 1, "b must be 1D");
        assert_eq!(a_shape[0], b_shape[0], "Incompatible dimensions");

        let n = a_shape[0];
        let a_data = self.to_vec();
        let b_data = b.to_vec();

        // Create augmented matrix [A | b]
        let mut aug = vec![0.0; n * (n + 1)];
        for i in 0..n {
            for j in 0..n {
                aug[i * (n + 1) + j] = a_data[i * n + j];
            }
            aug[i * (n + 1) + n] = b_data[i];
        }

        // Forward elimination with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut pivot_row = k;
            let mut max_val = aug[k * (n + 1) + k].abs();
            for i in (k + 1)..n {
                let val = aug[i * (n + 1) + k].abs();
                if val > max_val {
                    max_val = val;
                    pivot_row = i;
                }
            }

            assert!(
                max_val > 1e-10,
                "Matrix is singular, system has no unique solution"
            );

            // Swap rows
            if pivot_row != k {
                for j in 0..(n + 1) {
                    aug.swap(k * (n + 1) + j, pivot_row * (n + 1) + j);
                }
            }

            // Eliminate below
            for i in (k + 1)..n {
                let factor = aug[i * (n + 1) + k] / aug[k * (n + 1) + k];
                for j in k..(n + 1) {
                    aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i * (n + 1) + n];
            for j in (i + 1)..n {
                sum -= aug[i * (n + 1) + j] * x[j];
            }
            x[i] = sum / aug[i * (n + 1) + i];
        }

        Array::from_vec(x, Shape::new(vec![n]))
    }

    /// Compute the outer product of two 1D arrays.
    ///
    /// Given two 1D arrays a and b, returns a 2D array of shape (a.len(), b.len())
    /// where result[i, j] = a[i] * b[j].
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![4.0, 5.0], Shape::new(vec![2]));
    /// let c = a.outer(&b);
    /// assert_eq!(c.shape().as_slice(), &[3, 2]);
    /// // [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
    /// // = [[4, 5], [8, 10], [12, 15]]
    /// assert_eq!(c.to_vec(), vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    /// ```
    pub fn outer(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let a_shape = self.shape().as_slice();
        let b_shape = other.shape().as_slice();

        assert_eq!(a_shape.len(), 1, "First array must be 1D");
        assert_eq!(b_shape.len(), 1, "Second array must be 1D");

        let a_data = self.to_vec();
        let b_data = other.to_vec();
        let m = a_shape[0];
        let n = b_shape[0];

        let mut result = Vec::with_capacity(m * n);
        for &a_val in a_data.iter() {
            for &b_val in b_data.iter() {
                result.push(a_val * b_val);
            }
        }

        Array::from_vec(result, Shape::new(vec![m, n]))
    }

    /// Compute the inner product of two 1D arrays.
    ///
    /// For 1D arrays, this is the same as dot product: sum of element-wise products.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
    /// let result = a.inner(&b);
    /// assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    /// ```
    pub fn inner(&self, other: &Array) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let a_shape = self.shape().as_slice();
        let b_shape = other.shape().as_slice();

        assert_eq!(a_shape.len(), 1, "First array must be 1D");
        assert_eq!(b_shape.len(), 1, "Second array must be 1D");
        assert_eq!(
            a_shape[0], b_shape[0],
            "Arrays must have same length for inner product"
        );

        let a_data = self.to_vec();
        let b_data = other.to_vec();

        a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).sum()
    }

    /// Compute the cross product of two 3D vectors.
    ///
    /// Returns a vector perpendicular to both input vectors.
    /// Formula: a × b = [a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0]
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 0.0, 0.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![0.0, 1.0, 0.0], Shape::new(vec![3]));
    /// let c = a.cross(&b);
    /// assert_eq!(c.to_vec(), vec![0.0, 0.0, 1.0]); // i × j = k
    /// ```
    pub fn cross(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let a_shape = self.shape().as_slice();
        let b_shape = other.shape().as_slice();

        assert_eq!(a_shape.len(), 1, "First array must be 1D");
        assert_eq!(b_shape.len(), 1, "Second array must be 1D");
        assert_eq!(a_shape[0], 3, "Cross product requires 3D vectors");
        assert_eq!(b_shape[0], 3, "Cross product requires 3D vectors");

        let a = self.to_vec();
        let b = other.to_vec();

        let result = vec![
            a[1] * b[2] - a[2] * b[1],  // i component
            a[2] * b[0] - a[0] * b[2],  // j component
            a[0] * b[1] - a[1] * b[0],  // k component
        ];

        Array::from_vec(result, Shape::new(vec![3]))
    }

    /// Compute the trace of a 2D array (sum of diagonal elements).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let tr = a.trace();
    /// assert_eq!(tr, 5.0); // 1 + 4 = 5
    /// ```
    pub fn trace(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "Trace requires 2D array");
        assert_eq!(shape[0], shape[1], "Trace requires square matrix");

        let n = shape[0];
        let data = self.to_vec();

        let mut sum = 0.0;
        for i in 0..n {
            sum += data[i * n + i];
        }
        sum
    }

    /// Extract the diagonal of a 2D array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    /// let diag = a.diagonal();
    /// assert_eq!(diag.to_vec(), vec![1.0, 5.0]); // Elements at (0,0) and (1,1)
    /// ```
    pub fn diagonal(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "Diagonal requires 2D array");

        let (rows, cols) = (shape[0], shape[1]);
        let diag_len = rows.min(cols);
        let data = self.to_vec();

        let mut result = Vec::with_capacity(diag_len);
        for i in 0..diag_len {
            result.push(data[i * cols + i]);
        }

        Array::from_vec(result, Shape::new(vec![diag_len]))
    }

    /// Generate a Vandermonde matrix.
    ///
    /// Creates a matrix where each row is the input vector raised to successive powers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let v = x.vander(4);
    /// // [[1, 1, 1, 1], [1, 2, 4, 8], [1, 3, 9, 27]]
    /// ```
    pub fn vander(&self, n: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 1, "vander() only supports 1D arrays");

        let x = self.to_vec();
        let m = x.len();
        let mut result = Vec::with_capacity(m * n);

        for &val in x.iter() {
            for pow in 0..n {
                result.push(val.powi(pow as i32));
            }
        }

        Array::from_vec(result, Shape::new(vec![m, n]))
    }

    /// QR decomposition of a matrix.
    ///
    /// Decomposes matrix A into Q (orthogonal) and R (upper triangular) such that A = QR.
    /// Uses the Gram-Schmidt process.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let (q, r) = a.qr();
    /// // Q is orthogonal, R is upper triangular
    /// // Q * R ≈ A
    /// ```
    pub fn qr(&self) -> (Array, Array) {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "QR decomposition requires 2D array");

        let (m, n) = (shape[0], shape[1]);
        let data = self.to_vec();

        // Initialize Q and R
        let mut q = vec![0.0; m * n];
        let mut r = vec![0.0; n * n];

        // Modified Gram-Schmidt
        for j in 0..n {
            // Copy column j of A into v
            let mut v: Vec<f32> = (0..m).map(|i| data[i * n + j]).collect();

            // Orthogonalize against previous columns
            for i in 0..j {
                // r[i,j] = q[:,i] . v
                let mut dot = 0.0;
                for k in 0..m {
                    dot += q[k * n + i] * v[k];
                }
                r[i * n + j] = dot;

                // v = v - r[i,j] * q[:,i]
                for k in 0..m {
                    v[k] -= dot * q[k * n + i];
                }
            }

            // r[j,j] = ||v||
            let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
            r[j * n + j] = norm;

            // q[:,j] = v / norm
            if norm > 1e-10 {
                for k in 0..m {
                    q[k * n + j] = v[k] / norm;
                }
            }
        }

        let q_arr = Array::from_vec(q, Shape::new(vec![m, n]));
        let r_arr = Array::from_vec(r, Shape::new(vec![n, n]));

        (q_arr, r_arr)
    }

    /// Cholesky decomposition of a symmetric positive-definite matrix.
    ///
    /// Decomposes matrix A into L such that A = L * L^T, where L is lower triangular.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square or not positive-definite.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// // Symmetric positive-definite matrix
    /// let a = Array::from_vec(vec![4.0, 2.0, 2.0, 3.0], Shape::new(vec![2, 2]));
    /// let l = a.cholesky();
    /// // L * L^T = A
    /// ```
    pub fn cholesky(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "Cholesky decomposition requires 2D array");
        assert_eq!(shape[0], shape[1], "Cholesky decomposition requires square matrix");

        let n = shape[0];
        let data = self.to_vec();
        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                if j == i {
                    // Diagonal element
                    for k in 0..j {
                        sum += l[j * n + k] * l[j * n + k];
                    }
                    let val = data[j * n + j] - sum;
                    assert!(val > 0.0, "Matrix is not positive-definite");
                    l[j * n + j] = val.sqrt();
                } else {
                    // Off-diagonal element
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = (data[i * n + j] - sum) / l[j * n + j];
                }
            }
        }

        Array::from_vec(l, Shape::new(vec![n, n]))
    }

    /// Compute the rank of a matrix.
    ///
    /// Uses SVD-like approach (actually QR with tolerance) to estimate rank.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 2.0, 4.0], Shape::new(vec![2, 2]));
    /// let rank = a.matrix_rank();
    /// assert_eq!(rank, 1); // Rows are linearly dependent
    /// ```
    pub fn matrix_rank(&self) -> usize {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "matrix_rank requires 2D array");

        let (m, n) = (shape[0], shape[1]);
        let tolerance = 1e-10;

        // Use QR decomposition and count non-zero diagonal elements of R
        let (_, r) = self.qr();
        let r_data = r.to_vec();
        let min_dim = m.min(n);

        let mut rank = 0;
        for i in 0..min_dim {
            if r_data[i * n + i].abs() > tolerance {
                rank += 1;
            }
        }

        rank
    }

    /// Compute eigenvalues of a symmetric matrix using the power method.
    ///
    /// Returns approximate eigenvalues for symmetric matrices.
    /// For non-symmetric matrices, results may not be accurate.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![2.0, 1.0, 1.0, 2.0], Shape::new(vec![2, 2]));
    /// let eigvals = a.eigvalsh();
    /// // Eigenvalues of this symmetric matrix are 1 and 3
    /// ```
    pub fn eigvalsh(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "eigvalsh requires 2D array");
        assert_eq!(shape[0], shape[1], "eigvalsh requires square matrix");

        let n = shape[0];
        let mut eigenvalues = Vec::with_capacity(n);
        let mut a = self.clone();

        // Use deflation with power iteration
        for _ in 0..n {
            // Power iteration to find largest eigenvalue
            let mut v_data = vec![1.0; n];

            for _ in 0..100 {
                // v = A * v
                let v = Array::from_vec(v_data.clone(), Shape::new(vec![n]));
                let av = a.matmul(&v.reshape(Shape::new(vec![n, 1])))
                    .reshape(Shape::new(vec![n]));
                let av_data = av.to_vec();

                // Normalize
                let norm: f32 = av_data.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                v_data = av_data.iter().map(|x| x / norm).collect();
            }

            // Rayleigh quotient: λ = (v^T A v) / (v^T v)
            let v = Array::from_vec(v_data.clone(), Shape::new(vec![n]));
            let av = a.matmul(&v.reshape(Shape::new(vec![n, 1])))
                .reshape(Shape::new(vec![n]));
            let eigenvalue = v.inner(&av);
            eigenvalues.push(eigenvalue);

            // Deflate: A = A - λ * v * v^T
            let mut a_data = a.to_vec();
            for i in 0..n {
                for j in 0..n {
                    a_data[i * n + j] -= eigenvalue * v_data[i] * v_data[j];
                }
            }
            a = Array::from_vec(a_data, Shape::new(vec![n, n]));
        }

        Array::from_vec(eigenvalues, Shape::new(vec![n]))
    }

    /// Compute the pseudo-inverse of a matrix using the Moore-Penrose algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    /// let pinv = a.pinv();
    /// // pinv has shape [3, 2]
    /// ```
    pub fn pinv(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "pinv requires 2D array");

        let (m, n) = (shape[0], shape[1]);

        if m >= n {
            // A^+ = (A^T A)^-1 A^T
            let at = self.transpose();
            let ata = at.matmul(self);
            let ata_inv = ata.inv();
            ata_inv.matmul(&at)
        } else {
            // A^+ = A^T (A A^T)^-1
            let at = self.transpose();
            let aat = self.matmul(&at);
            let aat_inv = aat.inv();
            at.matmul(&aat_inv)
        }
    }

    /// Compute the condition number of a matrix.
    ///
    /// Uses the ratio of the largest to smallest singular value estimate.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
    /// let cond = a.cond();
    /// // Identity matrix has condition number ~1 (within numerical tolerance)
    /// assert!((cond - 1.0).abs() < 0.1);
    /// ```
    pub fn cond(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "cond requires 2D array");

        // For small matrices, use direct norm-based computation
        // cond(A) = ||A|| * ||A^-1||
        let n = shape[0];
        if n <= 4 {
            // Use Frobenius norm for simplicity
            let data = self.to_vec();
            let norm_a: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Compute inverse
            let inv = self.inv();
            let inv_data = inv.to_vec();
            let norm_inv: f32 = inv_data.iter().map(|x| x * x).sum::<f32>().sqrt();

            return norm_a * norm_inv / (n as f32); // Normalize by matrix size
        }

        // Use A^T A eigenvalues to estimate singular values
        let at = self.transpose();
        let ata = at.matmul(self);
        let eigvals = ata.eigvalsh();
        let eigvals_data = eigvals.to_vec();

        let max_eigval = eigvals_data.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
        let min_eigval = eigvals_data.iter().fold(f32::INFINITY, |a, &b| {
            if b.abs() > 1e-10 { a.min(b.abs()) } else { a }
        });

        if min_eigval < 1e-10 {
            f32::INFINITY
        } else {
            (max_eigval / min_eigval).sqrt()
        }
    }

    /// Compute singular value decomposition (SVD).
    ///
    /// Returns (U, S, Vt) where A = U @ diag(S) @ Vt.
    /// Uses power iteration to find singular values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let (u, s, vt) = a.svd();
    /// assert_eq!(s.shape().as_slice(), &[2]);
    /// ```
    pub fn svd(&self) -> (Array, Array, Array) {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "svd requires 2D array");

        let m = shape[0];
        let n = shape[1];
        let k = m.min(n);

        // Compute A^T A for right singular vectors
        let at = self.transpose();
        let ata = at.matmul(self);

        // Power iteration to get eigenvectors of A^T A (right singular vectors V)
        let mut v_data: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut s_values: Vec<f32> = Vec::with_capacity(k);
        let mut ata_data = ata.to_vec();

        for _ in 0..k {
            // Initialize random vector
            let mut v: Vec<f32> = (0..n).map(|i| ((i as f32 + 1.0) * 0.1).sin()).collect();
            let mut norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() { *x /= norm; }

            // Power iteration
            for _ in 0..50 {
                // Multiply by A^T A matrix
                let mut av = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        av[i] += ata_data[i * n + j] * v[j];
                    }
                }

                // Orthogonalize against previous vectors
                for prev in &v_data {
                    let dot: f32 = av.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                    for (a, p) in av.iter_mut().zip(prev.iter()) {
                        *a -= dot * p;
                    }
                }

                norm = av.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 { break; }
                for (v_i, av_i) in v.iter_mut().zip(av.iter()) {
                    *v_i = av_i / norm;
                }
            }

            // Singular value is sqrt of eigenvalue
            let eigenvalue = norm;
            s_values.push(eigenvalue.sqrt());
            v_data.push(v);
        }

        // Compute U = A @ V @ S^-1
        let mut u_data = vec![0.0; m * k];
        let a_data = self.to_vec();
        for col in 0..k {
            if s_values[col] > 1e-10 {
                for row in 0..m {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += a_data[row * n + j] * v_data[col][j];
                    }
                    u_data[row * k + col] = sum / s_values[col];
                }
            }
        }

        // Build output arrays
        let u = Array::from_vec(u_data, Shape::new(vec![m, k]));
        let s = Array::from_vec(s_values, Shape::new(vec![k]));
        let mut vt_data = vec![0.0; k * n];
        for i in 0..k {
            for j in 0..n {
                vt_data[i * n + j] = v_data[i][j];
            }
        }
        let vt = Array::from_vec(vt_data, Shape::new(vec![k, n]));

        (u, s, vt)
    }

    /// Solve least squares problem: minimize ||Ax - b||^2.
    ///
    /// Returns the solution x that minimizes the squared error.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0], Shape::new(vec![3, 2]));
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let x = a.lstsq(&b);
    /// assert_eq!(x.shape().as_slice(), &[2]);
    /// ```
    pub fn lstsq(&self, b: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "lstsq requires 2D matrix A");

        // Use normal equations: x = (A^T A)^-1 A^T b
        let at = self.transpose();
        let ata = at.matmul(self);
        let atb = at.matmul(b);
        ata.solve(&atb)
    }

    /// Compute eigenvalues and eigenvectors of a symmetric matrix.
    ///
    /// Returns (eigenvalues, eigenvectors) where each column of eigenvectors
    /// is an eigenvector corresponding to the eigenvalue at the same index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![2.0, 1.0, 1.0, 2.0], Shape::new(vec![2, 2]));
    /// let (vals, vecs) = a.eigh();
    /// assert_eq!(vals.shape().as_slice(), &[2]);
    /// assert_eq!(vecs.shape().as_slice(), &[2, 2]);
    /// ```
    pub fn eigh(&self) -> (Array, Array) {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "eigh requires 2D array");
        assert_eq!(shape[0], shape[1], "eigh requires square matrix");

        let n = shape[0];
        let mut a_data = self.to_vec();
        let mut eigenvectors = vec![0.0; n * n];

        // Initialize eigenvectors as identity
        for i in 0..n {
            eigenvectors[i * n + i] = 1.0;
        }

        // Jacobi eigenvalue algorithm
        for _ in 0..100 {
            // Find largest off-diagonal element
            let mut max_val = 0.0_f32;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a_data[i * n + j].abs() > max_val {
                        max_val = a_data[i * n + j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < 1e-10 { break; }

            // Compute rotation angle
            let diff = a_data[q * n + q] - a_data[p * n + p];
            let t = if diff.abs() < 1e-10 {
                1.0
            } else {
                let phi = diff / (2.0 * a_data[p * n + q]);
                1.0 / (phi.abs() + (phi * phi + 1.0).sqrt()) * phi.signum()
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Apply rotation to A
            let app = a_data[p * n + p];
            let aqq = a_data[q * n + q];
            let apq = a_data[p * n + q];

            a_data[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            a_data[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            a_data[p * n + q] = 0.0;
            a_data[q * n + p] = 0.0;

            for i in 0..n {
                if i != p && i != q {
                    let aip = a_data[i * n + p];
                    let aiq = a_data[i * n + q];
                    a_data[i * n + p] = c * aip - s * aiq;
                    a_data[p * n + i] = a_data[i * n + p];
                    a_data[i * n + q] = s * aip + c * aiq;
                    a_data[q * n + i] = a_data[i * n + q];
                }
            }

            // Update eigenvectors
            for i in 0..n {
                let vip = eigenvectors[i * n + p];
                let viq = eigenvectors[i * n + q];
                eigenvectors[i * n + p] = c * vip - s * viq;
                eigenvectors[i * n + q] = s * vip + c * viq;
            }
        }

        // Extract eigenvalues from diagonal
        let eigenvalues: Vec<f32> = (0..n).map(|i| a_data[i * n + i]).collect();

        (
            Array::from_vec(eigenvalues, Shape::new(vec![n])),
            Array::from_vec(eigenvectors, Shape::new(vec![n, n])),
        )
    }

    /// Compute eigenvalues of a general (non-symmetric) matrix.
    ///
    /// Uses QR iteration to find eigenvalues.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 0.0, 3.0], Shape::new(vec![2, 2]));
    /// let eigvals = a.eig();
    /// assert_eq!(eigvals.shape().as_slice(), &[2]);
    /// ```
    pub fn eig(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();
        assert_eq!(shape.len(), 2, "eig requires 2D array");
        assert_eq!(shape[0], shape[1], "eig requires square matrix");

        let n = shape[0];
        let mut a = self.clone();

        // QR iteration
        for _ in 0..100 {
            let (q, r) = a.qr();
            a = r.matmul(&q);
        }

        // Extract eigenvalues from diagonal
        let a_data = a.to_vec();
        let eigenvalues: Vec<f32> = (0..n).map(|i| a_data[i * n + i]).collect();

        Array::from_vec(eigenvalues, Shape::new(vec![n]))
    }

    /// Compute the tensor dot product along specified axes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let c = a.tensordot(&b, 1);
    /// assert_eq!(c.shape().as_slice(), &[2, 2]);
    /// ```
    pub fn tensordot(&self, other: &Array, axes: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let a_shape = self.shape().as_slice();
        let b_shape = other.shape().as_slice();

        // Contract last `axes` dimensions of self with first `axes` of other
        assert!(axes <= a_shape.len() && axes <= b_shape.len());

        // Reshape to 2D and use matmul
        let a_outer: usize = a_shape[..a_shape.len() - axes].iter().product();
        let a_inner: usize = a_shape[a_shape.len() - axes..].iter().product();
        let b_inner: usize = b_shape[..axes].iter().product();
        let b_outer: usize = b_shape[axes..].iter().product();

        assert_eq!(a_inner, b_inner, "Contracted dimensions must match");

        let a_2d = self.reshape(Shape::new(vec![a_outer, a_inner]));
        let b_2d = other.reshape(Shape::new(vec![b_inner, b_outer]));

        let result = a_2d.matmul(&b_2d);

        // Build output shape
        let mut out_shape = a_shape[..a_shape.len() - axes].to_vec();
        out_shape.extend_from_slice(&b_shape[axes..]);
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        result.reshape(Shape::new(out_shape))
    }

    /// Compute the Kronecker product of two arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let c = a.kron(&b);
    /// assert_eq!(c.shape().as_slice(), &[4, 4]);
    /// ```
    pub fn kron(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let a_shape = self.shape().as_slice();
        let b_shape = other.shape().as_slice();

        // For 2D arrays
        if a_shape.len() == 2 && b_shape.len() == 2 {
            let (m, n) = (a_shape[0], a_shape[1]);
            let (p, q) = (b_shape[0], b_shape[1]);
            let a_data = self.to_vec();
            let b_data = other.to_vec();

            let mut result = vec![0.0; m * p * n * q];
            for i in 0..m {
                for j in 0..n {
                    for k in 0..p {
                        for l in 0..q {
                            let out_row = i * p + k;
                            let out_col = j * q + l;
                            result[out_row * (n * q) + out_col] =
                                a_data[i * n + j] * b_data[k * q + l];
                        }
                    }
                }
            }

            Array::from_vec(result, Shape::new(vec![m * p, n * q]))
        } else {
            // 1D case
            let a_data = self.to_vec();
            let b_data = other.to_vec();
            let mut result = Vec::with_capacity(a_data.len() * b_data.len());
            for &a in &a_data {
                for &b in &b_data {
                    result.push(a * b);
                }
            }
            Array::from_vec(result, Shape::new(vec![a_data.len() * b_data.len()]))
        }
    }
}

/// Helper function for n-dimensional transpose.
fn transpose_nd(array: &Array, new_shape: Shape) -> Array {
    let old_dims = array.shape().as_slice();
    let data = array.to_vec();

    let size = array.size();
    let mut result = vec![0.0; size];

    // Compute strides for old and new layouts
    let old_strides = array.shape().default_strides();
    let new_strides = new_shape.default_strides();

    for flat_idx in 0..size {
        // Convert flat index to multi-dimensional for old layout
        let mut old_multi = vec![0; old_dims.len()];
        let mut idx = flat_idx;
        for (i, &stride) in old_strides.iter().enumerate() {
            old_multi[i] = idx / stride;
            idx %= stride;
        }

        // Reverse to get new multi-dimensional indices
        let new_multi: Vec<usize> = old_multi.iter().rev().copied().collect();

        // Convert new multi-dimensional to flat index
        let new_flat: usize = new_multi
            .iter()
            .zip(new_strides.iter())
            .map(|(idx, stride)| idx * stride)
            .sum();

        result[new_flat] = data[flat_idx];
    }

    let buffer = Buffer::from_f32(result, Device::Cpu);
    Array::from_buffer(buffer, new_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let b = a.transpose();
        assert_eq!(b.shape().as_slice(), &[3, 2]);
        assert_eq!(b.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_1d() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = a.transpose();
        assert_eq!(b.shape().as_slice(), &[3]);
        assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matmul_2d() {
        let a =
            Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b =
            Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().as_slice(), &[2, 2]);
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]]
        // = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        // = [[19, 22], [43, 50]]
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_non_square() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let b = Array::from_vec(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            Shape::new(vec![3, 2]),
        );
        let c = a.matmul(&b);
        assert_eq!(c.shape().as_slice(), &[2, 2]);
        // [[1, 2, 3], [4, 5, 6]] @ [[7, 8], [9, 10], [11, 12]]
        // = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(c.to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_vector() {
        // Matrix-vector multiplication
        let a =
            Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let v = Array::from_vec(vec![5.0, 6.0], Shape::new(vec![2]));
        let c = a.matmul(&v);
        assert_eq!(c.shape().as_slice(), &[2]);
        // [[1, 2], [3, 4]] @ [5, 6] = [1*5+2*6, 3*5+4*6] = [17, 39]
        assert_eq!(c.to_vec(), vec![17.0, 39.0]);
    }

    #[test]
    fn test_dot_1d() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
        let c = a.dot(&b);
        assert!(c.is_scalar());
        assert_eq!(c.to_vec(), vec![32.0]); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_2d() {
        let a =
            Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b =
            Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
        let c = a.dot(&b);
        // For 2D, dot is same as matmul
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    #[should_panic(expected = "Incompatible shapes")]
    fn test_matmul_incompatible() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![1, 3]));
        let b = Array::from_vec(vec![4.0, 5.0], Shape::new(vec![2, 1]));
        let _c = a.matmul(&b);
    }

    #[test]
    fn test_outer() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![4.0, 5.0], Shape::new(vec![2]));
        let c = a.outer(&b);
        assert_eq!(c.shape().as_slice(), &[3, 2]);
        // [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
        assert_eq!(c.to_vec(), vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_outer_square() {
        let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
        let c = a.outer(&b);
        assert_eq!(c.shape().as_slice(), &[2, 2]);
        // [[1*3, 1*4], [2*3, 2*4]] = [[3, 4], [6, 8]]
        assert_eq!(c.to_vec(), vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_inner() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
        let result = a.inner(&b);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_inner_zeros() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![0.0, 0.0, 0.0], Shape::new(vec![3]));
        let result = a.inner(&b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_cross_basic() {
        // i × j = k
        let i = Array::from_vec(vec![1.0, 0.0, 0.0], Shape::new(vec![3]));
        let j = Array::from_vec(vec![0.0, 1.0, 0.0], Shape::new(vec![3]));
        let k = i.cross(&j);
        assert_eq!(k.to_vec(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_cross_general() {
        let a = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![5.0, 6.0, 7.0], Shape::new(vec![3]));
        let c = a.cross(&b);
        // [3*7 - 4*6, 4*5 - 2*7, 2*6 - 3*5]
        // = [21 - 24, 20 - 14, 12 - 15]
        // = [-3, 6, -3]
        assert_eq!(c.to_vec(), vec![-3.0, 6.0, -3.0]);
    }

    #[test]
    fn test_cross_anticommutative() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
        let c1 = a.cross(&b);
        let c2 = b.cross(&a);
        // a × b = -(b × a)
        let c2_neg = c2.neg();
        assert_eq!(c1.to_vec(), c2_neg.to_vec());
    }

    #[test]
    fn test_trace() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let tr = a.trace();
        assert_eq!(tr, 5.0); // 1 + 4

        let b = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Shape::new(vec![3, 3]),
        );
        let tr_b = b.trace();
        assert_eq!(tr_b, 15.0); // 1 + 5 + 9
    }

    #[test]
    fn test_diagonal_square() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let diag = a.diagonal();
        assert_eq!(diag.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn test_diagonal_rectangular() {
        // 2x3 matrix
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let diag = a.diagonal();
        assert_eq!(diag.to_vec(), vec![1.0, 5.0]); // min(2, 3) = 2 elements

        // 3x2 matrix
        let b = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![3, 2]),
        );
        let diag_b = b.diagonal();
        assert_eq!(diag_b.to_vec(), vec![1.0, 4.0]); // min(3, 2) = 2 elements
    }
}
