//! GPU operation implementations.

use crate::backend::webgpu::WebGpuContext;
use crate::backend::shaders;
use crate::buffer::Buffer;

/// Execute a binary operation on GPU.
///
/// Both input buffers and output buffer must be on GPU.
///
/// # Arguments
///
/// * `lhs` - Left-hand side buffer
/// * `rhs` - Right-hand side buffer
/// * `output` - Output buffer (must be pre-allocated)
/// * `op` - Operation string ("+", "-", "*", "/")
///
/// # Panics
///
/// Panics if any buffer is not on GPU or if sizes don't match.
pub fn gpu_binary_op(
    lhs: &Buffer,
    rhs: &Buffer,
    output: &Buffer,
    op: &str,
) {
    let ctx = WebGpuContext::get();

    // Validate all buffers are on GPU
    assert_eq!(lhs.device(), crate::Device::WebGpu, "lhs must be on GPU");
    assert_eq!(rhs.device(), crate::Device::WebGpu, "rhs must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");

    // Validate sizes match
    assert_eq!(lhs.len(), rhs.len(), "Input sizes must match");
    assert_eq!(lhs.len(), output.len(), "Output size must match inputs");

    // Get GPU buffers
    let lhs_buf = lhs.as_gpu_buffer();
    let rhs_buf = rhs.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    // Create shader module
    let shader_code = shaders::binary_op_shader(op);
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("binary_op_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout =
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("binary_op_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

    // Create pipeline
    let pipeline_layout =
        ctx.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("binary_op_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline =
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("binary_op_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("binary_op_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("binary_op_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("binary_op_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        let workgroups = (output.len() as u32 + 255) / 256;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Execute a unary operation on GPU.
///
/// Both input and output buffers must be on GPU.
///
/// # Arguments
///
/// * `input` - Input buffer
/// * `output` - Output buffer (must be pre-allocated)
/// * `func` - Function name ("sin", "cos", "sqrt", etc.)
pub fn gpu_unary_op(
    input: &Buffer,
    output: &Buffer,
    func: &str,
) {
    let ctx = WebGpuContext::get();

    // Validate buffers are on GPU
    assert_eq!(input.device(), crate::Device::WebGpu, "input must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");
    assert_eq!(input.len(), output.len(), "Sizes must match");

    let input_buf = input.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    // Create shader module
    let shader_code = shaders::unary_op_shader(func);
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("unary_op_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout =
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("unary_op_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

    // Create pipeline
    let pipeline_layout =
        ctx.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("unary_op_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline =
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("unary_op_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unary_op_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("unary_op_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("unary_op_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        let workgroups = (output.len() as u32 + 255) / 256;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Execute matrix multiplication on GPU.
///
/// Performs C = A @ B where:
/// - A is (M, K)
/// - B is (K, N)
/// - C is (M, N)
///
/// Uses tiled algorithm with 16x16 workgroups for optimal performance.
///
/// # Arguments
///
/// * `lhs` - Left matrix buffer (M x K)
/// * `rhs` - Right matrix buffer (K x N)
/// * `output` - Output buffer (M x N, must be pre-allocated)
/// * `m` - Number of rows in A and C
/// * `n` - Number of columns in B and C
/// * `k` - Number of columns in A and rows in B
///
/// # Panics
///
/// Panics if buffers are not on GPU or sizes don't match.
pub fn gpu_matmul(
    lhs: &Buffer,
    rhs: &Buffer,
    output: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) {
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    // Validate all buffers are on GPU
    assert_eq!(lhs.device(), crate::Device::WebGpu, "lhs must be on GPU");
    assert_eq!(rhs.device(), crate::Device::WebGpu, "rhs must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");

    // Validate dimensions
    assert_eq!(lhs.len(), m * k, "lhs size must be M*K");
    assert_eq!(rhs.len(), k * n, "rhs size must be K*N");
    assert_eq!(output.len(), m * n, "output size must be M*N");

    // Get GPU buffers
    let lhs_buf = lhs.as_gpu_buffer();
    let rhs_buf = rhs.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    // Create uniform buffer for dimensions
    // Note: padding for 16-byte alignment
    let dims_data: [u32; 4] = [m as u32, n as u32, k as u32, 0];
    let dims_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("matmul_dims"),
        contents: bytemuck::cast_slice(&dims_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create shader module
    let shader_code = shaders::matmul_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout =
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matmul_bind_group_layout"),
                entries: &[
                    // Matrix A (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Matrix B (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Matrix C (read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Dimensions (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

    // Create pipeline
    let pipeline_layout =
        ctx.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline =
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dims_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch with 2D workgroups (16x16 tiles)
    let workgroups_x = ((n as u32) + 15) / 16;
    let workgroups_y = ((m as u32) + 15) / 16;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_ops_compile() {
        // Just test that the module compiles
        // Actual GPU tests require WebGPU initialization
        assert!(true);
    }
}
