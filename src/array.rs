//! Core Array type for n-dimensional numeric arrays.

use crate::{buffer::Buffer, DType, Device, Shape};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global counter for generating unique array IDs
static ARRAY_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Generate a unique ID for an array
fn next_array_id() -> usize {
    ARRAY_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

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
    /// Unique ID for tracing (pointer address)
    id: usize,
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
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create a new array filled with ones.
    pub fn ones(shape: Shape, dtype: DType) -> Self {
        let device = crate::default_device();
        let size = shape.size();
        let buffer = Buffer::filled(1.0, size, dtype, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create a new array filled with a specific value.
    pub fn full(value: f32, shape: Shape, dtype: DType) -> Self {
        let device = crate::default_device();
        let size = shape.size();
        let buffer = Buffer::filled(value, size, dtype, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
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
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<i32>.
    pub fn from_vec_i32(data: Vec<i32>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_i32(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<i8>.
    pub fn from_vec_i8(data: Vec<i8>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_i8(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<u8>.
    pub fn from_vec_u8(data: Vec<u8>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_u8(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<i16>.
    pub fn from_vec_i16(data: Vec<i16>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_i16(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<u16>.
    pub fn from_vec_u16(data: Vec<u16>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_u16(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<i64>.
    pub fn from_vec_i64(data: Vec<i64>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_i64(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<u32>.
    pub fn from_vec_u32(data: Vec<u32>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_u32(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<u64>.
    pub fn from_vec_u64(data: Vec<u64>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_u64(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<f64>.
    pub fn from_vec_f64(data: Vec<f64>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_f64(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a Vec<bool>.
    pub fn from_vec_bool(data: Vec<bool>, shape: Shape) -> Self {
        assert_eq!(data.len(), shape.size(), "Data length must match shape size");
        let device = crate::default_device();
        let buffer = Buffer::from_bool(data, device);
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Create an array from a buffer and shape (internal use).
    pub(crate) fn from_buffer(buffer: Buffer, shape: Shape) -> Self {
        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
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

    /// Get reference to the underlying buffer (internal use).
    #[inline]
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
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

    /// Get the unique ID of this array (for tracing).
    #[inline]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Check if this is a scalar (0-dimensional array).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    /// Copy data to a Vec<f32> (synchronous).
    ///
    /// This materializes the array and copies all data to CPU memory.
    /// Data is converted to f32 regardless of the array's dtype.
    pub fn to_vec(&self) -> Vec<f32> {
        // Fast path: contiguous array
        if self.offset == 0 && self.strides == self.shape.default_strides() {
            return self.buffer.to_f32_vec_converted();
        }

        // Slow path: strided/sliced array
        // Get raw buffer data and iterate through logical indices
        let raw_data = self.buffer.to_f32_vec_converted();
        let size = self.size();
        let ndim = self.ndim();

        if ndim == 0 {
            // Scalar
            return vec![raw_data[self.offset]];
        }

        let shape = self.shape.as_slice();
        let mut result = Vec::with_capacity(size);

        // Iterate through all logical indices in row-major order
        let mut indices = vec![0usize; ndim];
        for _ in 0..size {
            // Compute physical offset for current logical index
            let physical_idx: usize = self.offset
                + indices
                    .iter()
                    .zip(self.strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();

            result.push(raw_data[physical_idx]);

            // Increment indices (row-major order, last dimension increments first)
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        result
    }

    /// Copy data to a Vec<bool> (for Bool dtype arrays).
    pub fn to_bool_vec(&self) -> Vec<bool> {
        assert_eq!(self.dtype(), DType::Bool, "to_bool_vec requires Bool dtype");

        // Fast path: contiguous array
        if self.offset == 0 && self.strides == self.shape.default_strides() {
            return self.buffer.to_bool_vec();
        }

        // Slow path: strided/sliced array
        let raw_data = self.buffer.to_bool_vec();
        let size = self.size();
        let ndim = self.ndim();

        if ndim == 0 {
            return vec![raw_data[self.offset]];
        }

        let shape = self.shape.as_slice();
        let mut result = Vec::with_capacity(size);
        let mut indices = vec![0usize; ndim];

        for _ in 0..size {
            let physical_idx: usize = self.offset
                + indices
                    .iter()
                    .zip(self.strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();

            result.push(raw_data[physical_idx]);

            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        result
    }

    /// Cast array to a different dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, DType, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.5, 3.9], Shape::new(vec![3]));
    /// let b = a.astype(DType::Int32);
    /// assert_eq!(b.dtype(), DType::Int32);
    /// let data = b.to_vec();
    /// assert_eq!(data, vec![1.0, 2.0, 3.0]); // truncated to integers
    /// ```
    pub fn astype(&self, dtype: DType) -> Self {
        if self.dtype() == dtype {
            return self.clone();
        }

        // Read current data as f32
        let data = self.to_vec();

        // Create new array with target dtype, casting values
        let device = self.device();
        let shape = self.shape.clone();

        let buffer = match dtype {
            DType::Float32 => Buffer::from_f32(data, device),
            DType::Float64 => {
                let casted: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                Buffer::from_f64(casted, device)
            }
            DType::Float16 => {
                // Float16 is stored as f32 internally for now
                Buffer::from_f32_as_dtype(data, DType::Float16, device)
            }
            DType::Int8 => {
                let casted: Vec<i8> = data.iter().map(|&x| x as i8).collect();
                Buffer::from_i8(casted, device)
            }
            DType::Int16 => {
                let casted: Vec<i16> = data.iter().map(|&x| x as i16).collect();
                Buffer::from_i16(casted, device)
            }
            DType::Int32 => {
                let casted: Vec<i32> = data.iter().map(|&x| x as i32).collect();
                Buffer::from_i32(casted, device)
            }
            DType::Int64 => {
                let casted: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                Buffer::from_i64(casted, device)
            }
            DType::Uint8 => {
                let casted: Vec<u8> = data.iter().map(|&x| x as u8).collect();
                Buffer::from_u8(casted, device)
            }
            DType::Uint16 => {
                let casted: Vec<u16> = data.iter().map(|&x| x as u16).collect();
                Buffer::from_u16(casted, device)
            }
            DType::Uint32 => {
                let casted: Vec<u32> = data.iter().map(|&x| x as u32).collect();
                Buffer::from_u32(casted, device)
            }
            DType::Uint64 => {
                let casted: Vec<u64> = data.iter().map(|&x| x as u64).collect();
                Buffer::from_u64(casted, device)
            }
            DType::Bool => {
                let casted: Vec<bool> = data.iter().map(|&x| x != 0.0).collect();
                Buffer::from_bool(casted, device)
            }
        };

        let strides = shape.default_strides();
        Self { buffer, shape, strides, offset: 0, id: next_array_id() }
    }

    /// Transfer array to a different device.
    ///
    /// If the array is already on the target device, returns a clone.
    /// Otherwise, transfers the data to the new device.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use jax_rs::{Array, Device, Shape, DType};
    /// let cpu_arr = Array::zeros(Shape::new(vec![10]), DType::Float32);
    /// let gpu_arr = cpu_arr.to_device(Device::WebGpu);
    /// assert_eq!(gpu_arr.device(), Device::WebGpu);
    /// ```
    pub fn to_device(&self, device: Device) -> Array {
        if self.device() == device {
            return self.clone();
        }

        match (self.device(), device) {
            (Device::Cpu, Device::WebGpu) => {
                // Upload to GPU
                let data = self.to_vec();
                let buffer = Buffer::from_f32(data, Device::WebGpu);
                Array::from_buffer(buffer, self.shape().clone())
            }
            (Device::WebGpu, Device::Cpu) => {
                // Download from GPU
                let data = self.buffer().to_f32_vec();
                let buffer = Buffer::from_f32(data, Device::Cpu);
                Array::from_buffer(buffer, self.shape().clone())
            }
            (Device::Cpu, Device::Wasm) | (Device::Wasm, Device::Cpu) => {
                // CPU <-> WASM transfer
                let data = self.to_vec();
                let buffer = Buffer::from_f32(data, device);
                Array::from_buffer(buffer, self.shape().clone())
            }
            (Device::WebGpu, Device::Wasm) | (Device::Wasm, Device::WebGpu) => {
                // Go through CPU as intermediate
                let cpu = self.to_device(Device::Cpu);
                cpu.to_device(device)
            }
            // Same device (already handled by early return, but need exhaustive match)
            _ => self.clone()
        }
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
            id: next_array_id(),
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

    /// Remove a single dimension at the specified axis.
    ///
    /// The dimension at the given axis must be 1.
    pub fn squeeze_axis(&self, axis: usize) -> Self {
        let dims = self.shape.as_slice();
        assert!(axis < dims.len(), "Axis {} out of bounds", axis);
        assert_eq!(dims[axis], 1, "Can only squeeze axis with size 1");

        let mut new_dims = dims.to_vec();
        new_dims.remove(axis);

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
        write!(f, "Array:{}{}", self.dtype(), self.shape())
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

    #[test]
    fn test_array_zeros_all_dtypes() {
        // Test zeros with all dtypes
        let dtypes = [
            DType::Float32, DType::Float64, DType::Float16,
            DType::Int8, DType::Int16, DType::Int32, DType::Int64,
            DType::Uint8, DType::Uint16, DType::Uint32, DType::Uint64,
            DType::Bool,
        ];
        for dtype in dtypes {
            let a = Array::zeros(Shape::new(vec![2, 3]), dtype);
            assert_eq!(a.dtype(), dtype);
            assert_eq!(a.shape().as_slice(), &[2, 3]);
            let data = a.to_vec();
            assert!(data.iter().all(|&x| x == 0.0));
        }
    }

    #[test]
    fn test_array_ones_all_dtypes() {
        let dtypes = [
            DType::Float32, DType::Float64, DType::Float16,
            DType::Int8, DType::Int16, DType::Int32, DType::Int64,
            DType::Uint8, DType::Uint16, DType::Uint32, DType::Uint64,
        ];
        for dtype in dtypes {
            let a = Array::ones(Shape::new(vec![3]), dtype);
            assert_eq!(a.dtype(), dtype);
            let data = a.to_vec();
            assert!(data.iter().all(|&x| x == 1.0));
        }
    }

    #[test]
    fn test_array_from_vec_typed() {
        // Test i32
        let a = Array::from_vec_i32(vec![1, 2, 3], Shape::new(vec![3]));
        assert_eq!(a.dtype(), DType::Int32);
        assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0]);

        // Test i8
        let b = Array::from_vec_i8(vec![-1, 0, 127], Shape::new(vec![3]));
        assert_eq!(b.dtype(), DType::Int8);
        assert_eq!(b.to_vec(), vec![-1.0, 0.0, 127.0]);

        // Test u8
        let c = Array::from_vec_u8(vec![0, 128, 255], Shape::new(vec![3]));
        assert_eq!(c.dtype(), DType::Uint8);
        assert_eq!(c.to_vec(), vec![0.0, 128.0, 255.0]);

        // Test bool
        let d = Array::from_vec_bool(vec![true, false, true], Shape::new(vec![3]));
        assert_eq!(d.dtype(), DType::Bool);
        assert_eq!(d.to_vec(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_array_astype() {
        let a = Array::from_vec(vec![1.0, 2.5, 3.9], Shape::new(vec![3]));

        // Cast to Int32 (truncates)
        let b = a.astype(DType::Int32);
        assert_eq!(b.dtype(), DType::Int32);
        assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0]);

        // Cast to Bool
        let c = Array::from_vec(vec![0.0, 1.0, 5.0], Shape::new(vec![3]));
        let d = c.astype(DType::Bool);
        assert_eq!(d.dtype(), DType::Bool);
        assert_eq!(d.to_vec(), vec![0.0, 1.0, 1.0]);

        // Cast same dtype returns clone
        let e = a.astype(DType::Float32);
        assert_eq!(e.dtype(), DType::Float32);
        assert_eq!(e.to_vec(), a.to_vec());
    }

    #[test]
    fn test_array_to_bool_vec() {
        let a = Array::from_vec_bool(vec![true, false, true, false], Shape::new(vec![4]));
        let data = a.to_bool_vec();
        assert_eq!(data, vec![true, false, true, false]);
    }

    #[test]
    fn test_strided_to_vec_transposed() {
        // Create a 2x3 array and simulate a transpose by using reversed strides
        // Original: [[1, 2, 3], [4, 5, 6]] stored as [1, 2, 3, 4, 5, 6]
        // Transposed view: [[1, 4], [2, 5], [3, 6]] with shape [3, 2] and strides [1, 3]
        let buffer = Buffer::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Device::Cpu);
        let shape = Shape::new(vec![3, 2]);
        let strides = vec![1, 3]; // Transposed strides
        let arr = Array {
            buffer,
            shape,
            strides,
            offset: 0,
            id: next_array_id(),
        };

        // to_vec should return elements in row-major order of the transposed view
        // Row 0: [1, 4], Row 1: [2, 5], Row 2: [3, 6]
        let result = arr.to_vec();
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_strided_to_vec_with_offset() {
        // Create a buffer and access a slice with offset
        // Buffer: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        // View: 2x3 starting at offset 2, so [[2, 3, 4], [5, 6, 7]]
        let buffer = Buffer::from_f32(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Device::Cpu,
        );
        let shape = Shape::new(vec![2, 3]);
        let strides = vec![3, 1]; // Default strides for 2x3
        let arr = Array {
            buffer,
            shape,
            strides,
            offset: 2,
            id: next_array_id(),
        };

        let result = arr.to_vec();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_strided_to_vec_every_other() {
        // Create a view that takes every other element
        // Buffer: [0, 1, 2, 3, 4, 5, 6, 7]
        // View: shape [4], stride [2] -> [0, 2, 4, 6]
        let buffer = Buffer::from_f32(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            Device::Cpu,
        );
        let shape = Shape::new(vec![4]);
        let strides = vec![2]; // Every other element
        let arr = Array {
            buffer,
            shape,
            strides,
            offset: 0,
            id: next_array_id(),
        };

        let result = arr.to_vec();
        assert_eq!(result, vec![0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_strided_to_vec_3d() {
        // Create a 3D strided view
        // Buffer: 0..24
        // Original shape: [2, 3, 4]
        // View as transposed [4, 3, 2] with strides [1, 4, 12]
        let buffer = Buffer::from_f32((0..24).map(|x| x as f32).collect(), Device::Cpu);
        let shape = Shape::new(vec![4, 3, 2]);
        let strides = vec![1, 4, 12]; // Transposed strides
        let arr = Array {
            buffer,
            shape,
            strides,
            offset: 0,
            id: next_array_id(),
        };

        // First few elements should be:
        // [0][0][0] -> 0*1 + 0*4 + 0*12 = 0
        // [0][0][1] -> 0*1 + 0*4 + 1*12 = 12
        // [0][1][0] -> 0*1 + 1*4 + 0*12 = 4
        // [0][1][1] -> 0*1 + 1*4 + 1*12 = 16
        // [0][2][0] -> 0*1 + 2*4 + 0*12 = 8
        // [0][2][1] -> 0*1 + 2*4 + 1*12 = 20
        // [1][0][0] -> 1*1 + 0*4 + 0*12 = 1
        // etc.
        let result = arr.to_vec();
        assert_eq!(
            result,
            vec![
                0.0, 12.0, 4.0, 16.0, 8.0, 20.0, // [0][*][*]
                1.0, 13.0, 5.0, 17.0, 9.0, 21.0, // [1][*][*]
                2.0, 14.0, 6.0, 18.0, 10.0, 22.0, // [2][*][*]
                3.0, 15.0, 7.0, 19.0, 11.0, 23.0  // [3][*][*]
            ]
        );
    }
}
