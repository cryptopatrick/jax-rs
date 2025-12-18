//! Linear algebra operations.

use crate::{buffer::Buffer, Array, Device, DType, Shape};

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
        assert_eq!(self.device(), Device::Cpu, "Only CPU supported for now");
        assert_eq!(other.device(), Device::Cpu, "Only CPU supported for now");

        let a_shape = self.shape().as_slice();
        let b_shape = other.shape().as_slice();

        // Handle vector-matrix and matrix-vector cases
        if a_shape.len() == 1 && b_shape.len() == 2 {
            // Vector-matrix: (N,) @ (N, M) -> (M,)
            assert_eq!(
                a_shape[0], b_shape[0],
                "Vector-matrix multiplication: incompatible shapes"
            );
            return self.reshape(Shape::new(vec![1, a_shape[0]]))
                .matmul(other)
                .reshape(Shape::new(vec![b_shape[1]]));
        }

        if a_shape.len() == 2 && b_shape.len() == 1 {
            // Matrix-vector: (M, N) @ (N,) -> (M,)
            assert_eq!(
                a_shape[1], b_shape[0],
                "Matrix-vector multiplication: incompatible shapes"
            );
            return self.matmul(&other.reshape(Shape::new(vec![b_shape[0], 1])))
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

        let a_data = self.to_vec();
        let b_data = other.to_vec();
        let mut result = vec![0.0; m * n];

        // Naive matrix multiplication O(n^3)
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
            let result: f32 = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).sum();

            let buffer = Buffer::from_f32(vec![result], Device::Cpu);
            return Array::from_buffer(buffer, Shape::scalar());
        }

        // For higher dimensions, use matmul
        self.matmul(other)
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
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
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
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
        let c = a.matmul(&b);
        assert_eq!(c.shape().as_slice(), &[2, 2]);
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]]
        // = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        // = [[19, 22], [43, 50]]
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_non_square() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let b = Array::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], Shape::new(vec![3, 2]));
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
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
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
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
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
}
