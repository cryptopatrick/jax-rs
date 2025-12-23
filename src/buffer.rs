//! Buffer abstraction for array data storage.

use crate::{DType, Device};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Raw data buffer for array storage.
///
/// Buffers are reference-counted and can be shared between arrays
/// for zero-copy views. The actual data is stored on a specific device.
#[derive(Debug, Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

#[derive(Debug)]
enum BufferInner {
    /// CPU buffer stored as raw bytes
    Cpu {
        data: Vec<u8>,
        dtype: DType,
        len: usize,
    },
    /// WebGPU buffer
    WebGpu {
        buffer: wgpu::Buffer,
        dtype: DType,
        len: usize,
    },
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
        match device {
            Device::Cpu | Device::Wasm => {
                let byte_len = len * dtype.byte_width();
                let data = vec![0u8; byte_len];
                Self {
                    inner: Arc::new(BufferInner::Cpu { data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;

                let ctx = WebGpuContext::get();
                let byte_len = (len * dtype.byte_width()) as u64;

                let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("jax-rs GPU buffer (zeros)"),
                    size: byte_len,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
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

        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        len * dtype.byte_width(),
                    )
                    .to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu {
                        data: byte_data,
                        dtype,
                        len,
                    }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;

                let ctx = WebGpuContext::get();

                let buffer = ctx.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("jax-rs GPU buffer (f32)"),
                        contents: bytemuck::cast_slice(&data),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    },
                );

                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from i32 data.
    ///
    /// # Panics
    ///
    /// Panics if dtype is not Int32.
    #[allow(dead_code)]
    pub fn from_i32(data: Vec<i32>, device: Device) -> Self {
        let dtype = DType::Int32;
        let len = data.len();

        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        len * dtype.byte_width(),
                    )
                    .to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu {
                        data: byte_data,
                        dtype,
                        len,
                    }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;

                let ctx = WebGpuContext::get();

                let buffer = ctx.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("jax-rs GPU buffer (i32)"),
                        contents: bytemuck::cast_slice(&data),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    },
                );

                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer filled with a specific value.
    pub fn filled(
        value: f32,
        len: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        assert_eq!(dtype, DType::Float32, "Only Float32 supported for now");
        let data = vec![value; len];
        Self::from_f32(data, device)
    }

    /// Get the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        match &*self.inner {
            BufferInner::Cpu { len, .. } => *len,
            BufferInner::WebGpu { len, .. } => *len,
        }
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the device where this buffer lives.
    #[inline]
    pub fn device(&self) -> Device {
        match &*self.inner {
            BufferInner::Cpu { .. } => Device::Cpu,
            BufferInner::WebGpu { .. } => Device::WebGpu,
        }
    }

    /// Get the dtype of elements.
    #[inline]
    pub fn dtype(&self) -> DType {
        match &*self.inner {
            BufferInner::Cpu { dtype, .. } => *dtype,
            BufferInner::WebGpu { dtype, .. } => *dtype,
        }
    }

    /// Read data as f32 slice (sync, CPU only).
    ///
    /// # Panics
    ///
    /// Panics if dtype is not Float32 or if buffer is on GPU.
    pub fn as_f32_slice(&self) -> &[f32] {
        match &*self.inner {
            BufferInner::Cpu { data, dtype, len } => {
                assert_eq!(*dtype, DType::Float32);
                unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, *len)
                }
            }
            BufferInner::WebGpu { .. } => {
                panic!("Cannot get slice from GPU buffer. Use to_f32_vec() instead.");
            }
        }
    }

    /// Read data as i32 slice (sync, CPU only).
    ///
    /// # Panics
    ///
    /// Panics if dtype is not Int32 or if buffer is on GPU.
    #[allow(dead_code)]
    pub fn as_i32_slice(&self) -> &[i32] {
        match &*self.inner {
            BufferInner::Cpu { data, dtype, len } => {
                assert_eq!(*dtype, DType::Int32);
                unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const i32, *len)
                }
            }
            BufferInner::WebGpu { .. } => {
                panic!("Cannot get slice from GPU buffer. Use to_i32_vec() instead.");
            }
        }
    }

    /// Copy data to a Vec<f32> (works for both CPU and GPU).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match &*self.inner {
            BufferInner::Cpu { .. } => self.as_f32_slice().to_vec(),
            BufferInner::WebGpu { buffer, len, dtype } => {
                assert_eq!(*dtype, DType::Float32);
                // Use pollster to block on async read
                pollster::block_on(read_buffer_f32_async(buffer, *len))
            }
        }
    }

    /// Copy data to a Vec<i32> (works for both CPU and GPU).
    #[allow(dead_code)]
    pub fn to_i32_vec(&self) -> Vec<i32> {
        match &*self.inner {
            BufferInner::Cpu { .. } => self.as_i32_slice().to_vec(),
            BufferInner::WebGpu { buffer, len, dtype } => {
                assert_eq!(*dtype, DType::Int32);
                // Use pollster to block on async read
                pollster::block_on(read_buffer_i32_async(buffer, *len))
            }
        }
    }

    /// Get reference to inner GPU buffer (for internal use).
    ///
    /// # Panics
    ///
    /// Panics if buffer is not on GPU.
    #[allow(dead_code)]
    pub(crate) fn as_gpu_buffer(&self) -> &wgpu::Buffer {
        match &*self.inner {
            BufferInner::WebGpu { buffer, .. } => buffer,
            BufferInner::Cpu { .. } => {
                panic!("Buffer is on CPU, not GPU");
            }
        }
    }
}

/// Read a GPU buffer containing f32 data to CPU (async).
async fn read_buffer_f32_async(buffer: &wgpu::Buffer, len: usize) -> Vec<f32> {
    use crate::backend::webgpu::WebGpuContext;

    let ctx = WebGpuContext::get();
    let byte_len = (len * 4) as u64;

    // Create staging buffer for reading
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging buffer (f32 read)"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy GPU buffer to staging
    let mut encoder = ctx.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("buffer read encoder"),
        },
    );
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len);
    ctx.queue.submit(Some(encoder.finish()));

    // Map and read
    let slice = staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.await.unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    result
}

/// Read a GPU buffer containing i32 data to CPU (async).
async fn read_buffer_i32_async(buffer: &wgpu::Buffer, len: usize) -> Vec<i32> {
    use crate::backend::webgpu::WebGpuContext;

    let ctx = WebGpuContext::get();
    let byte_len = (len * 4) as u64;

    // Create staging buffer for reading
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging buffer (i32 read)"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy GPU buffer to staging
    let mut encoder = ctx.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("buffer read encoder"),
        },
    );
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len);
    ctx.queue.submit(Some(encoder.finish()));

    // Map and read
    let slice = staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.await.unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    result
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
