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
        // Float16 is stored as f32 internally since Rust doesn't have native f16
        if dtype == DType::Float16 {
            return Self::from_f32_as_dtype(vec![0.0f32; len], DType::Float16, device);
        }

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
        match dtype {
            DType::Float32 => {
                let data = vec![value; len];
                Self::from_f32(data, device)
            }
            DType::Float64 => {
                let data = vec![value as f64; len];
                Self::from_f64(data, device)
            }
            DType::Float16 => {
                // Store as f32 internally, convert on use
                let data = vec![value; len];
                Self::from_f32_as_dtype(data, DType::Float16, device)
            }
            DType::Int8 => {
                let data = vec![value as i8; len];
                Self::from_i8(data, device)
            }
            DType::Int16 => {
                let data = vec![value as i16; len];
                Self::from_i16(data, device)
            }
            DType::Int32 => {
                let data = vec![value as i32; len];
                Self::from_i32(data, device)
            }
            DType::Int64 => {
                let data = vec![value as i64; len];
                Self::from_i64(data, device)
            }
            DType::Uint8 => {
                let data = vec![value as u8; len];
                Self::from_u8(data, device)
            }
            DType::Uint16 => {
                let data = vec![value as u16; len];
                Self::from_u16(data, device)
            }
            DType::Uint32 => {
                let data = vec![value as u32; len];
                Self::from_u32(data, device)
            }
            DType::Uint64 => {
                let data = vec![value as u64; len];
                Self::from_u64(data, device)
            }
            DType::Bool => {
                let data = vec![value != 0.0; len];
                Self::from_bool(data, device)
            }
        }
    }

    /// Create a new buffer from i8 data.
    pub fn from_i8(data: Vec<i8>, device: Device) -> Self {
        let dtype = DType::Int8;
        let len = data.len();
        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, len).to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu { data: byte_data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (i8)"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from u8 data.
    pub fn from_u8(data: Vec<u8>, device: Device) -> Self {
        let dtype = DType::Uint8;
        let len = data.len();
        match device {
            Device::Cpu | Device::Wasm => {
                Self {
                    inner: Arc::new(BufferInner::Cpu { data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (u8)"),
                    contents: &data,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from i16 data.
    pub fn from_i16(data: Vec<i16>, device: Device) -> Self {
        let dtype = DType::Int16;
        let len = data.len();
        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, len * 2).to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu { data: byte_data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (i16)"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from u16 data.
    pub fn from_u16(data: Vec<u16>, device: Device) -> Self {
        let dtype = DType::Uint16;
        let len = data.len();
        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, len * 2).to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu { data: byte_data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (u16)"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from u32 data.
    pub fn from_u32(data: Vec<u32>, device: Device) -> Self {
        let dtype = DType::Uint32;
        let len = data.len();
        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, len * 4).to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu { data: byte_data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (u32)"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from i64 data.
    pub fn from_i64(data: Vec<i64>, device: Device) -> Self {
        let dtype = DType::Int64;
        let len = data.len();
        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, len * 8).to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu { data: byte_data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (i64)"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from u64 data.
    pub fn from_u64(data: Vec<u64>, device: Device) -> Self {
        let dtype = DType::Uint64;
        let len = data.len();
        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, len * 8).to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu { data: byte_data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (u64)"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from f64 data.
    pub fn from_f64(data: Vec<f64>, device: Device) -> Self {
        let dtype = DType::Float64;
        let len = data.len();
        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, len * 8).to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu { data: byte_data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (f64)"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from bool data.
    pub fn from_bool(data: Vec<bool>, device: Device) -> Self {
        let dtype = DType::Bool;
        let len = data.len();
        // Convert bools to u8 (0 or 1)
        let byte_data: Vec<u8> = data.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();
        match device {
            Device::Cpu | Device::Wasm => {
                Self {
                    inner: Arc::new(BufferInner::Cpu { data: byte_data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (bool)"),
                    contents: &byte_data,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
    }

    /// Create a new buffer from f32 data but store with a different dtype (for Float16).
    pub fn from_f32_as_dtype(data: Vec<f32>, dtype: DType, device: Device) -> Self {
        let len = data.len();
        match device {
            Device::Cpu | Device::Wasm => {
                let byte_data = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, len * 4).to_vec()
                };
                Self {
                    inner: Arc::new(BufferInner::Cpu { data: byte_data, dtype, len }),
                }
            }
            Device::WebGpu => {
                use crate::backend::webgpu::WebGpuContext;
                let ctx = WebGpuContext::get();
                let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jax-rs GPU buffer (f16 as f32)"),
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                });
                Self {
                    inner: Arc::new(BufferInner::WebGpu { buffer, dtype, len }),
                }
            }
        }
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

    /// Read data as f32 values, converting from the buffer's dtype.
    /// This is useful for operations that work with f32 internally.
    pub fn to_f32_vec_converted(&self) -> Vec<f32> {
        match &*self.inner {
            BufferInner::Cpu { data, dtype, len } => {
                match dtype {
                    DType::Float32 | DType::Float16 => {
                        // Float16 is stored as f32 internally
                        unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, *len).to_vec()
                        }
                    }
                    DType::Float64 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f64, *len)
                        };
                        slice.iter().map(|&x| x as f32).collect()
                    }
                    DType::Int8 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const i8, *len)
                        };
                        slice.iter().map(|&x| x as f32).collect()
                    }
                    DType::Int16 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const i16, *len)
                        };
                        slice.iter().map(|&x| x as f32).collect()
                    }
                    DType::Int32 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const i32, *len)
                        };
                        slice.iter().map(|&x| x as f32).collect()
                    }
                    DType::Int64 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const i64, *len)
                        };
                        slice.iter().map(|&x| x as f32).collect()
                    }
                    DType::Uint8 | DType::Bool => {
                        data.iter().map(|&x| x as f32).collect()
                    }
                    DType::Uint16 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const u16, *len)
                        };
                        slice.iter().map(|&x| x as f32).collect()
                    }
                    DType::Uint32 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const u32, *len)
                        };
                        slice.iter().map(|&x| x as f32).collect()
                    }
                    DType::Uint64 => {
                        let slice = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const u64, *len)
                        };
                        slice.iter().map(|&x| x as f32).collect()
                    }
                }
            }
            BufferInner::WebGpu { buffer, len, dtype } => {
                // For GPU, we only support Float32 for now
                if *dtype == DType::Float32 || *dtype == DType::Float16 {
                    pollster::block_on(read_buffer_f32_async(buffer, *len))
                } else {
                    panic!("GPU buffer dtype conversion not yet supported for {:?}", dtype);
                }
            }
        }
    }

    /// Read data as bool values.
    pub fn to_bool_vec(&self) -> Vec<bool> {
        match &*self.inner {
            BufferInner::Cpu { data, dtype, len } => {
                assert_eq!(*dtype, DType::Bool);
                data[..*len].iter().map(|&x| x != 0).collect()
            }
            BufferInner::WebGpu { .. } => {
                panic!("Bool GPU buffer read not yet supported");
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
