//! Core Array type for n-dimensional numeric arrays.

use crate::{buffer::Buffer, Device, DType, Shape};
use std::fmt;

/// A multidimensional numeric array.
///
/// This is the core data type of jax-rs, equivalent to `jax.Array` in JAX
/// or `torch.Tensor` in PyTorch. Unlike jax-js which uses manual reference
/// counting (`.ref` and `.dispose()`), Rust's ownership system provides
/// automatic memory management.
///
/// # Memory Model
///
/// Arrays own their data through an `Arc<Buffer>`, allowing cheap cloning
/// and zero-copy views. When the last reference to a buffer is dropped,
/// the memory is automatically freed.
///
/// # Examples
///
/// ```
/// # use jax_rs::{Array, DType, Shape};
/// // Create a 2x3 array of zeros
/// let a = Array::zeros(Shape::new(vec![2, 3]), DType::Float32);
/// assert_eq!(a.shape().as_slice(), &[2, 3]);
/// ```
#[derive(Debug, Clone)]
pub struct Array {
    /// Underlying data buffer
    buffer: Buffer,
    /// Shape of the array
    shape: Shape,
    /// Strides for indexing (in elements, not bytes)
    strides: Vec<usize>,
    /// Offset into the buffer (in elements)
    offset: usize,
}

impl Array {
    /// Create a new array filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, DType, Shape, Device, default_device};
    /// let a = Array::zeros(Shape::new(vec![2, 3]), DType::Float32);
    /// assert_eq!(a.shape().as_slice(), &[2, 3]);
    /// assert_eq!(a.dtype(), DType::Float32);
    /// ```
    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        let device = crate::default_device();
        let size = shape.size();
        let buffer = Buffer::zeros(size, dtype, device);
        let strides = shape.default_strides();
        Self {
            buffer,
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create a new array filled with ones.
    pub fn ones(shape: Shape, dtype: DType) -> Self {
        assert_eq!(dtype, DType::Float32, "Only Float32 supported for now");
        let device = crate::default_device();
        let size = shape.size();
        let buffer = Buffer::filled(1.0, size, dtype, device);
        let strides = shape.default_strides();
        Self {
            buffer,
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create a new array filled with a specific value.
    pub fn full(value: f32, shape: Shape, dtype: DType) -> Self {
        assert_eq!(dtype, DType::Float32, "Only Float32 supported for now");
        let device = crate::default_device();
        let size = shape.size();
        let buffer = Buffer::filled(value, size, dtype, device);
        let strides = shape.default_strides();
        Self {
            buffer,
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create an array from a flat Vec<f32> and shape.
    ///
    /// # Panics
    ///
    /// Panics if the shape size doesn't match the data length.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let a = Array::from_vec(data, Shape::new(vec![2, 3]));
    /// assert_eq!(a.shape().as_slice(), &[2, 3]);
    /// ```
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Self {
        assert_eq!(
            data.len(),
            shape.size(),
            "Data length must match shape size"
        );
        let device = crate::default_device();
        let buffer = Buffer::from_f32(data, device);
        let strides = shape.default_strides();
        Self {
            buffer,
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create an array from a buffer and shape (internal use).
    pub(crate) fn from_buffer(buffer: Buffer, shape: Shape) -> Self {
        let strides = shape.default_strides();
        Self {
            buffer,
            shape,
            strides,
            offset: 0,
        }
    }

    /// Get the shape of the array.
    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get the data type of the array.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }

    /// Get the device where this array lives.
    #[inline]
    pub fn device(&self) -> Device {
        self.buffer.device()
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements.
    #[inline]
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Check if this is a scalar (0-dimensional array).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    /// Copy data to a Vec (synchronous).
    ///
    /// This materializes the array and copies all data to CPU memory.
    /// For Float32 arrays only (for now).
    pub fn to_vec(&self) -> Vec<f32> {
        assert_eq!(self.dtype(), DType::Float32);
        // For now, we only support contiguous arrays
        // TODO: Handle strided/sliced arrays
        assert_eq!(self.offset, 0);
        assert_eq!(self.strides, self.shape.default_strides());
        self.buffer.to_f32_vec()
    }

    /// Reshape the array to a new shape.
    ///
    /// # Panics
    ///
    /// Panics if the total size doesn't match.
    pub fn reshape(&self, new_shape: Shape) -> Self {
        assert_eq!(
            self.shape.size(),
            new_shape.size(),
            "Cannot reshape array of size {} into shape of size {}",
            self.shape.size(),
            new_shape.size()
        );
        // For now, require contiguous data for reshape
        assert_eq!(self.offset, 0);
        assert_eq!(self.strides, self.shape.default_strides());

        Self {
            buffer: self.buffer.clone(),
            shape: new_shape.clone(),
            strides: new_shape.default_strides(),
            offset: 0,
        }
    }

    /// Remove axes of length one from the array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::zeros(Shape::new(vec![1, 3, 1, 4]), jax_rs::DType::Float32);
    /// let b = a.squeeze();
    /// assert_eq!(b.shape().as_slice(), &[3, 4]);
    /// ```
    pub fn squeeze(&self) -> Self {
        let new_dims: Vec<usize> = self
            .shape
            .as_slice()
            .iter()
            .filter(|&&dim| dim != 1)
            .copied()
            .collect();

        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        };

        self.reshape(new_shape)
    }

    /// Expand the shape of an array by inserting a new axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - Position where new axis is placed
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.expand_dims(0);
    /// assert_eq!(b.shape().as_slice(), &[1, 3]);
    /// let c = a.expand_dims(1);
    /// assert_eq!(c.shape().as_slice(), &[3, 1]);
    /// ```
    pub fn expand_dims(&self, axis: usize) -> Self {
        let mut new_dims = self.shape.as_slice().to_vec();
        assert!(
            axis <= new_dims.len(),
            "Axis {} out of bounds for array with {} dimensions",
            axis,
            new_dims.len()
        );
        new_dims.insert(axis, 1);
        self.reshape(Shape::new(new_dims))
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Array:{}{}",
            self.dtype(),
            self.shape()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_zeros() {
        let a = Array::zeros(Shape::new(vec![2, 3]), DType::Float32);
        assert_eq!(a.shape().as_slice(), &[2, 3]);
        assert_eq!(a.dtype(), DType::Float32);
        assert_eq!(a.size(), 6);
        assert_eq!(a.ndim(), 2);
        let data = a.to_vec();
        assert_eq!(data.len(), 6);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_array_ones() {
        let a = Array::ones(Shape::new(vec![3, 2]), DType::Float32);
        assert_eq!(a.shape().as_slice(), &[3, 2]);
        let data = a.to_vec();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_array_full() {
        let a = Array::full(5.0, Shape::new(vec![2, 2]), DType::Float32);
        let data = a.to_vec();
        assert_eq!(data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_array_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Array::from_vec(data.clone(), Shape::new(vec![2, 3]));
        assert_eq!(a.shape().as_slice(), &[2, 3]);
        assert_eq!(a.to_vec(), data);
    }

    #[test]
    fn test_array_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Array::from_vec(data.clone(), Shape::new(vec![2, 3]));
        let b = a.reshape(Shape::new(vec![3, 2]));
        assert_eq!(b.shape().as_slice(), &[3, 2]);
        assert_eq!(b.to_vec(), data);

        let c = a.reshape(Shape::new(vec![6]));
        assert_eq!(c.shape().as_slice(), &[6]);
    }

    #[test]
    fn test_array_display() {
        let a = Array::zeros(Shape::new(vec![2, 3]), DType::Float32);
        let s = a.to_string();
        assert!(s.contains("float32"));
        assert!(s.contains("2"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_array_clone() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = a.clone();
        assert_eq!(a.to_vec(), b.to_vec());
        assert_eq!(a.shape(), b.shape());
    }

    #[test]
    #[should_panic(expected = "Data length must match shape size")]
    fn test_array_from_vec_size_mismatch() {
        let _a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![3]));
    }

    #[test]
    #[should_panic(expected = "Cannot reshape")]
    fn test_array_reshape_size_mismatch() {
        let a = Array::zeros(Shape::new(vec![2, 3]), DType::Float32);
        let _b = a.reshape(Shape::new(vec![2, 2]));
    }
}
