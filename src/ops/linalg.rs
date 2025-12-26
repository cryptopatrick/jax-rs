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
