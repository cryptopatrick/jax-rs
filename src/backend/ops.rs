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

        let workgroups = (output.len() as u32).div_ceil(256);
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

        let workgroups = (output.len() as u32).div_ceil(256);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Execute a fused operation group on GPU.
///
/// Combines multiple element-wise operations into a single kernel dispatch,
/// reducing memory bandwidth and kernel launch overhead.
///
/// # Arguments
///
/// * `group` - The fused operation group containing operations and metadata
/// * `input_buffers` - Input buffer references (must match group.inputs)
/// * `output_buffers` - Output buffer references (must match group.outputs)
///
/// # Panics
///
/// Panics if buffer counts don't match group inputs/outputs or if buffers are not on GPU.
pub fn gpu_fused_execute(
    group: &crate::trace::FusedGroup,
    input_buffers: &[&crate::buffer::Buffer],
    output_buffers: &[&crate::buffer::Buffer],
) {
    use crate::backend::shaders::get_cached_shader;
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    // Validate
    assert_eq!(
        input_buffers.len(),
        group.inputs.len(),
        "Input buffer count mismatch"
    );
    assert_eq!(
        output_buffers.len(),
        group.outputs.len(),
        "Output buffer count mismatch"
    );

    // Generate and cache shader
    let shader_code = crate::trace::fusion::generate_fused_shader(group);
    let shader = get_cached_shader(&shader_code, &group.name, &ctx.device);

    // Create bind group layout
    let mut entries = Vec::new();

    // Input bindings
    for i in 0..input_buffers.len() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    // Output bindings
    for i in 0..output_buffers.len() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: (input_buffers.len() + i) as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    // Params binding
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: (input_buffers.len() + output_buffers.len()) as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });

    let bind_group_layout =
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{}_layout", group.name)),
                entries: &entries,
            });

    // Create pipeline
    let pipeline_layout =
        ctx.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{}_pipeline", group.name)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline =
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&group.name),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

    // Create params buffer
    let size = output_buffers[0].len() as u32;
    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fused_params"),
            contents: bytemuck::cast_slice(&[size]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Create bind group
    let mut bind_entries = Vec::new();

    for (i, buf) in input_buffers.iter().enumerate() {
        bind_entries.push(wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buf.as_gpu_buffer().as_entire_binding(),
        });
    }

    for (i, buf) in output_buffers.iter().enumerate() {
        bind_entries.push(wgpu::BindGroupEntry {
            binding: (input_buffers.len() + i) as u32,
            resource: buf.as_gpu_buffer().as_entire_binding(),
        });
    }

    bind_entries.push(wgpu::BindGroupEntry {
        binding: (input_buffers.len() + output_buffers.len()) as u32,
        resource: params_buf.as_entire_binding(),
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{}_bind_group", group.name)),
        layout: &bind_group_layout,
        entries: &bind_entries,
    });

    // Dispatch
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{}_encoder", group.name)),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&group.name),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        let workgroups = size.div_ceil(256);
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
    let workgroups_x = (n as u32).div_ceil(16);
    let workgroups_y = (m as u32).div_ceil(16);

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

/// Execute a reduction operation on GPU.
///
/// Reduces an entire array to a single scalar value.
/// Uses a two-pass algorithm for large arrays:
/// 1. First pass: Each workgroup reduces 256 elements to 1
/// 2. Second pass: Reduce workgroup results to final scalar
///
/// # Arguments
///
/// * `input` - Input buffer
/// * `output` - Output buffer (size 1 for full reduction)
/// * `op` - Operation: "sum", "max", "min", "prod"
///
/// # Panics
///
/// Panics if buffers are not on GPU.
pub fn gpu_reduce_all(
    input: &Buffer,
    output: &Buffer,
    op: &str,
) {


    let _ctx = WebGpuContext::get();

    // Validate buffers are on GPU
    assert_eq!(input.device(), crate::Device::WebGpu, "input must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");
    assert_eq!(output.len(), 1, "output must have size 1 for full reduction");

    let n = input.len();

    // If input is small enough, do single-pass reduction
    if n <= 256 {
        gpu_reduce_single_pass(input, output, op);
        return;
    }

    // Two-pass reduction for large arrays
    let num_workgroups = n.div_ceil(256);

    // Create intermediate buffer for workgroup results
    let intermediate = Buffer::zeros(num_workgroups, input.dtype(), crate::Device::WebGpu);

    // First pass: reduce to workgroup results
    gpu_reduce_pass(input, &intermediate, op, n);

    // Second pass: reduce workgroup results to final value
    gpu_reduce_single_pass(&intermediate, output, op);
}

/// Single-pass reduction (input size <= 256).
fn gpu_reduce_single_pass(
    input: &Buffer,
    output: &Buffer,
    op: &str,
) {
    gpu_reduce_pass(input, output, op, input.len());
}

/// Execute one pass of reduction.
fn gpu_reduce_pass(
    input: &Buffer,
    output: &Buffer,
    op: &str,
    n: usize,
) {
    let ctx = WebGpuContext::get();

    let input_buf = input.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    // Create shader module
    let shader_code = shaders::reduction_shader(op);
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("reduction_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout =
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("reduction_bind_group_layout"),
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
                label: Some("reduction_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

    let pipeline =
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reduction_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reduction_bind_group"),
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
    let workgroups = (n as u32).div_ceil(256);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reduction_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reduction_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Execute 2D convolution on GPU.
///
/// Performs batched 2D convolution with support for:
/// - Multiple input/output channels
/// - Configurable stride and padding
/// - Batch processing
///
/// # Arguments
///
/// * `input` - Input buffer of shape [batch, in_channels, height, width]
/// * `kernel` - Kernel buffer of shape [out_channels, in_channels, kernel_h, kernel_w]
/// * `output` - Output buffer of shape [batch, out_channels, out_h, out_w]
/// * `batch_size` - Number of samples in batch
/// * `in_channels` - Number of input channels
/// * `out_channels` - Number of output channels
/// * `input_h` - Input height
/// * `input_w` - Input width
/// * `kernel_h` - Kernel height
/// * `kernel_w` - Kernel width
/// * `stride` - Stride (same for h and w)
/// * `padding` - Padding (same for h and w)
///
/// # Panics
///
/// Panics if buffers are not on GPU or dimensions don't match.
#[allow(clippy::too_many_arguments)]
pub fn gpu_conv2d(
    input: &Buffer,
    kernel: &Buffer,
    output: &Buffer,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: usize,
    padding: usize,
) {
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    // Validate all buffers are on GPU
    assert_eq!(input.device(), crate::Device::WebGpu, "input must be on GPU");
    assert_eq!(kernel.device(), crate::Device::WebGpu, "kernel must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");

    // Compute output dimensions
    let output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
    let output_w = (input_w + 2 * padding - kernel_w) / stride + 1;

    // Validate sizes
    let expected_input_size = batch_size * in_channels * input_h * input_w;
    let expected_kernel_size = out_channels * in_channels * kernel_h * kernel_w;
    let expected_output_size = batch_size * out_channels * output_h * output_w;

    assert_eq!(input.len(), expected_input_size, "input size mismatch");
    assert_eq!(kernel.len(), expected_kernel_size, "kernel size mismatch");
    assert_eq!(output.len(), expected_output_size, "output size mismatch");

    // Get GPU buffers
    let input_buf = input.as_gpu_buffer();
    let kernel_buf = kernel.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    // Create params buffer (16 u32 values, padded to 64 bytes)
    let params_data: [u32; 16] = [
        batch_size as u32,
        in_channels as u32,
        out_channels as u32,
        input_h as u32,
        input_w as u32,
        kernel_h as u32,
        kernel_w as u32,
        output_h as u32,
        output_w as u32,
        stride as u32,  // stride_h
        stride as u32,  // stride_w
        padding as u32, // padding_h
        padding as u32, // padding_w
        0, 0, 0,        // padding for alignment
    ];

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("conv2d_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create shader module
    let shader_code = shaders::conv2d_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("conv2d_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("conv2d_bind_group_layout"),
        entries: &[
            // Input (read-only)
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
            // Kernel (read-only)
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
            // Output (read-write)
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
            // Params (uniform)
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
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("conv2d_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("conv2d_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("conv2d_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kernel_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch with 3D workgroups: (out_w, out_h, out_channels * batch_size)
    let workgroups_x = (output_w as u32).div_ceil(8);
    let workgroups_y = (output_h as u32).div_ceil(8);
    let workgroups_z = (out_channels * batch_size) as u32;

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("conv2d_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv2d_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Execute 1D convolution on GPU.
///
/// # Arguments
///
/// * `input` - Input buffer of shape [batch, in_channels, length]
/// * `kernel` - Kernel buffer of shape [out_channels, in_channels, kernel_len]
/// * `output` - Output buffer of shape [batch, out_channels, output_len]
/// * `batch_size` - Batch size
/// * `in_channels` - Number of input channels
/// * `out_channels` - Number of output channels
/// * `input_len` - Input length
/// * `kernel_len` - Kernel length
/// * `stride` - Stride for convolution
/// * `padding` - Padding to add to input
pub fn gpu_conv1d(
    input: &Buffer,
    kernel: &Buffer,
    output: &Buffer,
    batch_size: usize,
    in_channels: usize,
    out_channels: usize,
    input_len: usize,
    kernel_len: usize,
    stride: usize,
    padding: usize,
) {
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    // Validate all buffers are on GPU
    assert_eq!(input.device(), crate::Device::WebGpu, "input must be on GPU");
    assert_eq!(kernel.device(), crate::Device::WebGpu, "kernel must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");

    // Compute output dimensions
    let output_len = (input_len + 2 * padding - kernel_len) / stride + 1;

    // Validate sizes
    let expected_input_size = batch_size * in_channels * input_len;
    let expected_kernel_size = out_channels * in_channels * kernel_len;
    let expected_output_size = batch_size * out_channels * output_len;

    assert_eq!(input.len(), expected_input_size, "input size mismatch");
    assert_eq!(kernel.len(), expected_kernel_size, "kernel size mismatch");
    assert_eq!(output.len(), expected_output_size, "output size mismatch");

    // Get GPU buffers
    let input_buf = input.as_gpu_buffer();
    let kernel_buf = kernel.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    // Create params buffer (8 u32 values, padded to 32 bytes)
    let params_data: [u32; 8] = [
        batch_size as u32,
        in_channels as u32,
        out_channels as u32,
        input_len as u32,
        kernel_len as u32,
        output_len as u32,
        stride as u32,
        padding as u32,
    ];

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("conv1d_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create shader module
    let shader_code = shaders::conv1d_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("conv1d_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("conv1d_bind_group_layout"),
        entries: &[
            // Input (read-only)
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
            // Kernel (read-only)
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
            // Output (read-write)
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
            // Params (uniform)
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
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("conv1d_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("conv1d_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("conv1d_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kernel_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch with 1D workgroups
    let total_elements = batch_size * out_channels * output_len;
    let workgroups = (total_elements as u32).div_ceil(256);

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("conv1d_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv1d_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Execute 2D max pooling on GPU.
///
/// # Arguments
///
/// * `input` - Input buffer of shape [batch, channels, height, width]
/// * `output` - Output buffer of shape [batch, channels, out_h, out_w]
/// * `batch_size` - Number of samples in batch
/// * `channels` - Number of channels
/// * `input_h` - Input height
/// * `input_w` - Input width
/// * `pool_h` - Pool window height
/// * `pool_w` - Pool window width
/// * `stride` - Stride (same for h and w)
/// * `padding` - Padding (same for h and w)
#[allow(clippy::too_many_arguments)]
pub fn gpu_maxpool2d(
    input: &Buffer,
    output: &Buffer,
    batch_size: usize,
    channels: usize,
    input_h: usize,
    input_w: usize,
    pool_h: usize,
    pool_w: usize,
    stride: usize,
    padding: usize,
) {
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    // Validate buffers
    assert_eq!(input.device(), crate::Device::WebGpu, "input must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");

    // Compute output dimensions
    let output_h = (input_h + 2 * padding - pool_h) / stride + 1;
    let output_w = (input_w + 2 * padding - pool_w) / stride + 1;

    // Create params buffer
    let params_data: [u32; 16] = [
        batch_size as u32,
        channels as u32,
        input_h as u32,
        input_w as u32,
        output_h as u32,
        output_w as u32,
        pool_h as u32,
        pool_w as u32,
        stride as u32,
        stride as u32,
        padding as u32,
        padding as u32,
        0, 0, 0, 0, // padding
    ];

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("maxpool2d_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Get GPU buffers
    let input_buf = input.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    // Create shader module
    let shader_code = shaders::maxpool2d_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("maxpool2d_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout (3 bindings: input, output, params)
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("maxpool2d_bind_group_layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("maxpool2d_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("maxpool2d_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("maxpool2d_bind_group"),
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let workgroups_x = (output_w as u32).div_ceil(8);
    let workgroups_y = (output_h as u32).div_ceil(8);
    let workgroups_z = (channels * batch_size) as u32;

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("maxpool2d_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("maxpool2d_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Execute 2D average pooling on GPU.
#[allow(clippy::too_many_arguments)]
pub fn gpu_avgpool2d(
    input: &Buffer,
    output: &Buffer,
    batch_size: usize,
    channels: usize,
    input_h: usize,
    input_w: usize,
    pool_h: usize,
    pool_w: usize,
    stride: usize,
    padding: usize,
) {
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    assert_eq!(input.device(), crate::Device::WebGpu, "input must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");

    let output_h = (input_h + 2 * padding - pool_h) / stride + 1;
    let output_w = (input_w + 2 * padding - pool_w) / stride + 1;

    let params_data: [u32; 16] = [
        batch_size as u32,
        channels as u32,
        input_h as u32,
        input_w as u32,
        output_h as u32,
        output_w as u32,
        pool_h as u32,
        pool_w as u32,
        stride as u32,
        stride as u32,
        padding as u32,
        padding as u32,
        0, 0, 0, 0,
    ];

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("avgpool2d_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let input_buf = input.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    let shader_code = shaders::avgpool2d_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("avgpool2d_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("avgpool2d_bind_group_layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("avgpool2d_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("avgpool2d_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("avgpool2d_bind_group"),
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let workgroups_x = (output_w as u32).div_ceil(8);
    let workgroups_y = (output_h as u32).div_ceil(8);
    let workgroups_z = (channels * batch_size) as u32;

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("avgpool2d_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("avgpool2d_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Execute batch normalization on GPU.
///
/// # Arguments
///
/// * `input` - Input buffer of shape [batch, channels, height, width]
/// * `gamma` - Scale parameter of shape [channels]
/// * `beta` - Shift parameter of shape [channels]
/// * `output` - Output buffer of same shape as input
/// * `batch_size` - Number of samples
/// * `channels` - Number of channels
/// * `spatial_size` - height * width
/// * `epsilon` - Small constant for numerical stability
#[allow(clippy::too_many_arguments)]
pub fn gpu_batchnorm(
    input: &Buffer,
    gamma: &Buffer,
    beta: &Buffer,
    output: &Buffer,
    batch_size: usize,
    channels: usize,
    spatial_size: usize,
    epsilon: f32,
) {
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    assert_eq!(input.device(), crate::Device::WebGpu, "input must be on GPU");
    assert_eq!(gamma.device(), crate::Device::WebGpu, "gamma must be on GPU");
    assert_eq!(beta.device(), crate::Device::WebGpu, "beta must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");

    // Params: batch_size, channels, spatial_size, epsilon (as bits), padding
    let params_data: [u32; 8] = [
        batch_size as u32,
        channels as u32,
        spatial_size as u32,
        epsilon.to_bits(),
        0, 0, 0, 0,
    ];

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("batchnorm_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let input_buf = input.as_gpu_buffer();
    let gamma_buf = gamma.as_gpu_buffer();
    let beta_buf = beta.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    let shader_code = shaders::batchnorm_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("batchnorm_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("batchnorm_bind_group_layout"),
        entries: &[
            // Input
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
            // Gamma
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
            // Beta
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Params
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("batchnorm_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("batchnorm_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("batchnorm_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gamma_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: beta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // One workgroup per channel
    let workgroups = channels as u32;

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("batchnorm_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batchnorm_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Execute FFT on GPU (real input to complex output).
///
/// Converts real input to complex (adding zero imaginary parts) and then
/// performs FFT using the multi-pass butterfly algorithm.
///
/// # Arguments
///
/// * `input` - Real input buffer of size n
/// * `output` - Complex output buffer of size 2*n (interleaved real/imag)
/// * `n` - FFT size (must be power of 2)
/// * `inverse` - true for inverse FFT, false for forward FFT
pub fn gpu_fft(
    input: &Buffer,
    output: &Buffer,
    n: usize,
    inverse: bool,
) {


    let _ctx = WebGpuContext::get();

    assert_eq!(input.device(), crate::Device::WebGpu, "input must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");
    assert_eq!(input.len(), n, "input size mismatch");
    assert_eq!(output.len(), n * 2, "output size mismatch (should be 2*n for complex)");
    assert!(n.is_power_of_two(), "FFT size must be power of 2");

    // Copy real values to complex buffer (interleaved with zero imaginary)
    // For now, do this on CPU as we don't have a copy kernel
    let real_data = input.to_f32_vec();
    let mut complex_data = vec![0.0f32; n * 2];
    for i in 0..n {
        complex_data[i * 2] = real_data[i]; // real part
        complex_data[i * 2 + 1] = 0.0;      // imaginary part
    }
    let complex_input = Buffer::from_f32(complex_data, crate::Device::WebGpu);

    // Run complex FFT
    gpu_fft_complex(&complex_input, output, n, inverse);
}

/// Execute FFT on GPU (complex input to complex output).
///
/// Performs FFT using iterative Cooley-Tukey algorithm with log2(n) passes.
/// Uses ping-pong buffers to avoid read-write conflicts.
///
/// # Arguments
///
/// * `input` - Complex input buffer of size 2*n (interleaved real/imag)
/// * `output` - Complex output buffer of size 2*n
/// * `n` - FFT size (must be power of 2)
/// * `inverse` - true for inverse FFT, false for forward FFT
pub fn gpu_fft_complex(
    input: &Buffer,
    output: &Buffer,
    n: usize,
    inverse: bool,
) {
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    assert_eq!(input.device(), crate::Device::WebGpu, "input must be on GPU");
    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");
    assert_eq!(input.len(), n * 2, "input size mismatch (should be 2*n for complex)");
    assert_eq!(output.len(), n * 2, "output size mismatch (should be 2*n for complex)");
    assert!(n.is_power_of_two(), "FFT size must be power of 2");

    let num_stages = (n as f32).log2() as u32;
    let direction = if inverse { -1 } else { 1 };

    // Create ping-pong buffers
    let temp_buffer1 = Buffer::zeros(n * 2, crate::DType::Float32, crate::Device::WebGpu);
    let temp_buffer2 = Buffer::zeros(n * 2, crate::DType::Float32, crate::Device::WebGpu);

    // Create shader module
    let shader_code = shaders::fft_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fft_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fft_bind_group_layout"),
        entries: &[
            // Input buffer
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
            // Output buffer
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
            // Params
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fft_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fft_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Get GPU buffers
    let input_buf = input.as_gpu_buffer();
    let temp_buf1 = temp_buffer1.as_gpu_buffer();
    let temp_buf2 = temp_buffer2.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    // Ping-pong between buffers for each stage
    for stage in 0..num_stages {
        // Properly alternate between buffers to avoid read-write conflicts
        let (src_buf, dst_buf) = if stage == 0 {
            // First stage: input -> temp1
            (input_buf, temp_buf1)
        } else if stage == num_stages - 1 {
            // Last stage: temp -> output
            if stage % 2 == 1 {
                (temp_buf1, out_buf)
            } else {
                (temp_buf2, out_buf)
            }
        } else {
            // Middle stages: alternate between temp1 and temp2
            if stage % 2 == 1 {
                (temp_buf1, temp_buf2)
            } else {
                (temp_buf2, temp_buf1)
            }
        };

        // Create params for this stage
        let params_data: [u32; 4] = [
            n as u32,
            stage,
            direction as u32,
            0, // padding
        ];

        let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fft_params"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group for this stage
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fft_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroups = (n as u32).div_ceil(256);

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fft_encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        ctx.device.poll(wgpu::Maintain::Wait);
    }

    // Normalize for inverse FFT
    if inverse {
        // Scale output by 1/n
        // TODO: Add a normalization kernel
    }
}

/// Generate uniform random numbers on GPU using Philox PRNG.
///
/// # Arguments
///
/// * `output` - Output buffer for random values
/// * `n` - Number of random values to generate
/// * `seed` - PRNG seed (2x u64 combined into key)
/// * `offset` - Offset for counter (enables continuation)
pub fn gpu_uniform(
    output: &Buffer,
    n: usize,
    seed: [u64; 2],
    offset: u32,
) {
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");
    assert_eq!(output.len(), n, "output size mismatch");

    let out_buf = output.as_gpu_buffer();

    // Create params: seed0, seed1, offset, n
    let params_data: [u32; 4] = [
        (seed[0] & 0xFFFFFFFF) as u32,
        (seed[1] & 0xFFFFFFFF) as u32,
        offset,
        n as u32,
    ];

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("philox_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create shader
    let shader_code = shaders::philox_uniform_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("philox_uniform_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("philox_bind_group_layout"),
        entries: &[
            // Output
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Params
            wgpu::BindGroupLayoutEntry {
                binding: 1,
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
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("philox_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("philox_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("philox_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let workgroups = (n as u32).div_ceil(256);

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("philox_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("philox_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Generate normal random numbers on GPU using Box-Muller transform.
///
/// # Arguments
///
/// * `output` - Output buffer for normal random values
/// * `n` - Number of random values to generate (must be even)
/// * `seed` - PRNG seed for generating uniform values
/// * `offset` - Offset for counter
pub fn gpu_normal(
    output: &Buffer,
    n: usize,
    seed: [u64; 2],
    offset: u32,
) {
    use wgpu::util::DeviceExt;

    let ctx = WebGpuContext::get();

    assert_eq!(output.device(), crate::Device::WebGpu, "output must be on GPU");
    assert_eq!(output.len(), n, "output size mismatch");
    assert_eq!(n % 2, 0, "n must be even for Box-Muller");

    // First, generate uniform random values
    let uniform_buffer = Buffer::zeros(n, crate::DType::Float32, crate::Device::WebGpu);
    gpu_uniform(&uniform_buffer, n, seed, offset);

    let uniform_buf = uniform_buffer.as_gpu_buffer();
    let out_buf = output.as_gpu_buffer();

    // Create params
    let params_data: [u32; 4] = [n as u32, 0, 0, 0];

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("box_muller_params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create shader
    let shader_code = shaders::box_muller_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("box_muller_shader"),
        source: wgpu::ShaderSource::Wgsl(shader_code.into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("box_muller_bind_group_layout"),
        entries: &[
            // Uniform input
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
            // Output
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
            // Params
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("box_muller_pipeline_layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("box_muller_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("box_muller_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch - each thread handles 2 elements
    let workgroups = ((n / 2) as u32).div_ceil(256);

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("box_muller_encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("box_muller_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// Generate random samples from a logistic distribution on GPU.
///
/// Uses the Philox PRNG and inverse transform sampling.
///
/// # Arguments
///
/// * `output` - Output buffer (GPU)
/// * `n` - Number of samples to generate
/// * `seed` - PRNG seed (2x u64)
/// * `offset` - Counter offset
/// * `loc` - Location parameter
/// * `scale` - Scale parameter
pub fn gpu_logistic(
    output: &Buffer,
    n: usize,
    seed: [u64; 2],
    offset: u32,
    loc: f32,
    scale: f32,
) {
    use wgpu::util::DeviceExt;
    let ctx = WebGpuContext::get();

    assert_eq!(output.device(), crate::Device::WebGpu, "Output must be on GPU");

    // Create params: seed0, seed1, offset, n, loc, scale
    let params_data: [u32; 6] = [
        (seed[0] & 0xFFFFFFFF) as u32,
        (seed[1] & 0xFFFFFFFF) as u32,
        offset,
        n as u32,
        bytemuck::cast(loc),
        bytemuck::cast(scale),
    ];

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("logistic params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create shader
    let shader_source = crate::backend::shaders::logistic_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("logistic shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("logistic bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
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

    // Create pipeline layout
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("logistic pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipeline
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("logistic pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("logistic bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_gpu_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch compute shader
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("logistic encoder"),
    });

    let workgroups = (n as u32).div_ceil(256);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logistic pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
}

/// GPU implementation of exponential distribution sampling.
///
/// Uses the Philox PRNG and inverse transform sampling.
///
/// # Arguments
///
/// * `output` - Output buffer (GPU)
/// * `n` - Number of samples to generate
/// * `seed` - PRNG seed (2x u64)
/// * `offset` - Counter offset
/// * `rate` - Rate parameter (λ)
pub fn gpu_exponential(
    output: &Buffer,
    n: usize,
    seed: [u64; 2],
    offset: u32,
    rate: f32,
) {
    use wgpu::util::DeviceExt;
    let ctx = WebGpuContext::get();

    assert_eq!(output.device(), crate::Device::WebGpu, "Output must be on GPU");

    // Create params: seed0, seed1, offset, n, rate
    let params_data: [u32; 5] = [
        (seed[0] & 0xFFFFFFFF) as u32,
        (seed[1] & 0xFFFFFFFF) as u32,
        offset,
        n as u32,
        bytemuck::cast(rate),
    ];

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("exponential params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create shader
    let shader_source = crate::backend::shaders::exponential_shader();
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("exponential shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("exponential bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
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

    // Create pipeline layout
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("exponential pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipeline
    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("exponential pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("exponential bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_gpu_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch compute shader
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("exponential encoder"),
    });

    let workgroups = (n as u32).div_ceil(256);

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("exponential pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
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
