//! Buffer abstraction for array data storage.

use crate::{Device, DType};
use std::sync::Arc;

/// Raw data buffer for array storage.
///
/// Buffers are reference-counted and can be shared between arrays
/// for zero-copy views. The actual data is stored on a specific device.
#[derive(Debug, Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

#[derive(Debug)]
struct BufferInner {
    /// Raw bytes of the buffer
    data: Vec<u8>,
    /// Device where this buffer lives
    device: Device,
    /// Data type of elements (for validation)
    dtype: DType,
    /// Number of elements
    len: usize,
}

impl Buffer {
    /// Create a new buffer filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements
    /// * `dtype` - Data type of elements
    /// * `device` - Target device
    pub fn zeros(len: usize, dtype: DType, device: Device) -> Self {
        let byte_len = len * dtype.byte_width();
        let data = vec![0u8; byte_len];
        Self {
            inner: Arc::new(BufferInner {
                data,
                device,
                dtype,
                len,
            }),
        }
    }

    /// Create a new buffer from f32 data.
    ///
    /// # Panics
    ///
    /// Panics if dtype is not Float32.
    pub fn from_f32(data: Vec<f32>, device: Device) -> Self {
        let dtype = DType::Float32;
        let len = data.len();
        let byte_data = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                len * dtype.byte_width(),
            )
            .to_vec()
        };
        Self {
            inner: Arc::new(BufferInner {
                data: byte_data,
                device,
                dtype,
                len,
            }),
        }
    }

    /// Create a new buffer from i32 data.
    ///
    /// # Panics
    ///
    /// Panics if dtype is not Int32.
    pub fn from_i32(data: Vec<i32>, device: Device) -> Self {
        let dtype = DType::Int32;
        let len = data.len();
        let byte_data = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                len * dtype.byte_width(),
            )
            .to_vec()
        };
        Self {
            inner: Arc::new(BufferInner {
                data: byte_data,
                device,
                dtype,
                len,
            }),
        }
    }

    /// Create a new buffer filled with a specific value.
    pub fn filled(value: f32, len: usize, dtype: DType, device: Device) -> Self {
        assert_eq!(dtype, DType::Float32, "Only Float32 supported for now");
        let data = vec![value; len];
        Self::from_f32(data, device)
    }

    /// Get the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }

    /// Get the device where this buffer lives.
    #[inline]
    pub fn device(&self) -> Device {
        self.inner.device
    }

    /// Get the dtype of elements.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    /// Read data as f32 slice (sync).
    ///
    /// # Panics
    ///
    /// Panics if dtype is not Float32.
    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.inner.dtype, DType::Float32);
        unsafe {
            std::slice::from_raw_parts(
                self.inner.data.as_ptr() as *const f32,
                self.inner.len,
            )
        }
    }

    /// Read data as i32 slice (sync).
    ///
    /// # Panics
    ///
    /// Panics if dtype is not Int32.
    pub fn as_i32_slice(&self) -> &[i32] {
        assert_eq!(self.inner.dtype, DType::Int32);
        unsafe {
            std::slice::from_raw_parts(
                self.inner.data.as_ptr() as *const i32,
                self.inner.len,
            )
        }
    }

    /// Copy data to a Vec<f32>.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.as_f32_slice().to_vec()
    }

    /// Copy data to a Vec<i32>.
    pub fn to_i32_vec(&self) -> Vec<i32> {
        self.as_i32_slice().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_zeros() {
        let buf = Buffer::zeros(10, DType::Float32, Device::Cpu);
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.device(), Device::Cpu);
        assert_eq!(buf.dtype(), DType::Float32);
        let data = buf.to_f32_vec();
        assert_eq!(data.len(), 10);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_buffer_from_f32() {
        let data = vec![1.0, 2.0, 3.0];
        let buf = Buffer::from_f32(data.clone(), Device::Cpu);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.to_f32_vec(), data);
    }

    #[test]
    fn test_buffer_filled() {
        let buf = Buffer::filled(5.0, 4, DType::Float32, Device::Cpu);
        assert_eq!(buf.len(), 4);
        let data = buf.to_f32_vec();
        assert!(data.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_buffer_clone_is_cheap() {
        let buf1 = Buffer::from_f32(vec![1.0, 2.0, 3.0], Device::Cpu);
        let buf2 = buf1.clone();
        assert_eq!(buf1.to_f32_vec(), buf2.to_f32_vec());
        assert_eq!(Arc::strong_count(&buf1.inner), 2);
    }
}
