//! WGSL shader templates for GPU kernels.

/// Generate WGSL code for element-wise binary operation.
///
/// Supports operations like +, -, *, / between two arrays.
///
/// # Arguments
///
/// * `op` - The binary operator as a string ("+", "-", "*", "/")
///
/// # Examples
///
/// ```ignore
/// let shader = binary_op_shader("+");
/// // Generates WGSL shader for element-wise addition
/// ```
pub fn binary_op_shader(op: &str) -> String {
    format!(
        r#"@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {{
        return;
    }}
    output[idx] = input_a[idx] {op} input_b[idx];
}}
"#,
        op = op
    )
}

/// Generate WGSL code for element-wise unary operation.
///
/// Supports operations like sin, cos, sqrt, exp, log, etc.
///
/// # Arguments
///
/// * `func` - The unary function name (e.g., "sin", "cos", "sqrt")
///
/// # Examples
///
/// ```ignore
/// let shader = unary_op_shader("sqrt");
/// // Generates WGSL shader for element-wise square root
/// ```
pub fn unary_op_shader(func: &str) -> String {
    format!(
        r#"@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {{
        return;
    }}
    output[idx] = {func}(input[idx]);
}}
"#,
        func = func
    )
}

/// Generate WGSL code for matrix multiplication (tiled).
///
/// Uses shared memory tiling for improved performance.
/// Workgroup size is 16x16, with 16x16 tiles.
///
/// # Examples
///
/// ```ignore
/// let shader = matmul_shader();
/// // Generates optimized tiled matmul WGSL shader
/// ```
pub fn matmul_shader() -> String {
    r#"struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
    _padding: u32,
}

@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> matrix_c: array<f32>;

@group(0) @binding(3)
var<uniform> dims: Dimensions;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum = 0.0;

    // Number of tiles needed
    let num_tiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // Load tile from matrix A
        let a_col = t * TILE_SIZE + local_col;
        if (row < dims.M && a_col < dims.K) {
            tile_a[local_row][local_col] = matrix_a[row * dims.K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load tile from matrix B
        let b_row = t * TILE_SIZE + local_row;
        if (b_row < dims.K && col < dims.N) {
            tile_b[local_row][local_col] = matrix_b[b_row * dims.N + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[local_row][k] * tile_b[k][local_col];
        }

        workgroupBarrier();
    }

    // Write result
    if (row < dims.M && col < dims.N) {
        matrix_c[row * dims.N + col] = sum;
    }
}
"#
    .to_string()
}

/// Generate WGSL code for reduction operation.
///
/// Implements a parallel reduction within each workgroup.
/// Each workgroup reduces 256 elements to 1, writing to output.
/// For full array reduction, a second pass may be needed.
///
/// # Arguments
///
/// * `op` - Reduction operation: "sum", "max", "min", "prod"
///
/// # Examples
///
/// ```ignore
/// let shader = reduction_shader("sum");
/// // Generates WGSL shader for parallel sum reduction
/// ```
pub fn reduction_shader(op: &str) -> String {
    let (reduce_expr, identity) = match op {
        "sum" => ("acc + val", "0.0"),
        "max" => ("max(acc, val)", "-3.402823466e+38"), // -f32::MAX
        "min" => ("min(acc, val)", "3.402823466e+38"),  // f32::MAX
        "prod" => ("acc * val", "1.0"),
        _ => panic!("Unknown reduction operation: {}", op),
    };

    format!(
        r#"@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

const BLOCK_SIZE: u32 = 256u;

var<workgroup> shared_data: array<f32, BLOCK_SIZE>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {{
    let tid = local_id.x;
    let gid = global_id.x;
    let n = arrayLength(&input);

    // Load into shared memory with identity for out-of-bounds
    if (gid < n) {{
        shared_data[tid] = input[gid];
    }} else {{
        shared_data[tid] = {identity};
    }}

    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride = BLOCK_SIZE / 2u;
    while (stride > 0u) {{
        if (tid < stride) {{
            let val = shared_data[tid + stride];
            let acc = shared_data[tid];
            shared_data[tid] = {reduce_expr};
        }}
        stride = stride / 2u;
        workgroupBarrier();
    }}

    // First thread writes workgroup result
    if (tid == 0u) {{
        output[workgroup_id.x] = shared_data[0];
    }}
}}
"#,
        identity = identity,
        reduce_expr = reduce_expr
    )
}

/// Generate WGSL code for axis reduction operation.
///
/// Reduces along a specific axis of a multidimensional array.
/// More complex than full reduction as it preserves other dimensions.
///
/// # Arguments
///
/// * `op` - Reduction operation: "sum", "max", "min"
///
/// # Examples
///
/// ```ignore
/// let shader = axis_reduction_shader("sum");
/// // Generates WGSL shader for sum along an axis
/// ```
pub fn axis_reduction_shader(op: &str) -> String {
    let (reduce_expr, identity) = match op {
        "sum" => ("acc + val", "0.0"),
        "max" => ("max(acc, val)", "-3.402823466e+38"),
        "min" => ("min(acc, val)", "3.402823466e+38"),
        _ => panic!("Unknown reduction operation: {}", op),
    };

    format!(
        r#"struct Dimensions {{
    input_size: u32,
    axis_size: u32,
    outer_size: u32,
    _padding: u32,
}}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> dims: Dimensions;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let output_size = dims.outer_size * (dims.input_size / dims.axis_size);

    if (idx >= output_size) {{
        return;
    }}

    // Calculate which output element we're computing
    let outer_idx = idx / (dims.input_size / dims.axis_size);
    let inner_idx = idx % (dims.input_size / dims.axis_size);

    // Reduce along axis
    var acc = {identity};
    for (var i = 0u; i < dims.axis_size; i = i + 1u) {{
        let input_idx = outer_idx * dims.input_size + i * (dims.input_size / dims.axis_size) + inner_idx;
        let val = input[input_idx];
        acc = {reduce_expr};
    }}

    output[idx] = acc;
}}
"#,
        identity = identity,
        reduce_expr = reduce_expr
    )
}

/// Generate WGSL code for custom binary operation with function.
///
/// For operations that need custom logic (like pow, min, max).
///
/// # Arguments
///
/// * `expr` - The expression to compute, using `a` and `b` as variables
pub fn binary_op_custom_shader(expr: &str) -> String {
    format!(
        r#"@group(0) @binding(0)
var<storage, read> input_a: array<f32>;

@group(0) @binding(1)
var<storage, read> input_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {{
        return;
    }}
    let a = input_a[idx];
    let b = input_b[idx];
    output[idx] = {expr};
}}
"#,
        expr = expr
    )
}

/// Generate WGSL code for 2D convolution.
///
/// Performs direct 2D convolution (not im2col).
/// Input: [batch, in_channels, height, width]
/// Kernel: [out_channels, in_channels, kernel_h, kernel_w]
/// Output: [batch, out_channels, out_h, out_w]
///
/// # Examples
///
/// ```ignore
/// let shader = conv2d_shader();
/// // Generates WGSL shader for 2D convolution
/// ```
pub fn conv2d_shader() -> String {
    r#"struct ConvParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    input_h: u32,
    input_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    output_h: u32,
    output_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> kernel: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: ConvParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let out_c_batch = global_id.z;

    let batch = out_c_batch / params.out_channels;
    let out_c = out_c_batch % params.out_channels;

    if (out_x >= params.output_w || out_y >= params.output_h || batch >= params.batch_size) {
        return;
    }

    var sum: f32 = 0.0;

    // Input position (accounting for stride and padding)
    let in_y_start = i32(out_y * params.stride_h) - i32(params.padding_h);
    let in_x_start = i32(out_x * params.stride_w) - i32(params.padding_w);

    // Convolve
    for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var ky: u32 = 0u; ky < params.kernel_h; ky = ky + 1u) {
            for (var kx: u32 = 0u; kx < params.kernel_w; kx = kx + 1u) {
                let in_y = in_y_start + i32(ky);
                let in_x = in_x_start + i32(kx);

                // Check bounds
                if (in_y >= 0 && in_y < i32(params.input_h) &&
                    in_x >= 0 && in_x < i32(params.input_w)) {
                    // Input index: [batch, in_channel, y, x]
                    let input_idx = batch * params.in_channels * params.input_h * params.input_w +
                                    ic * params.input_h * params.input_w +
                                    u32(in_y) * params.input_w + u32(in_x);

                    // Kernel index: [out_channel, in_channel, ky, kx]
                    let kernel_idx = out_c * params.in_channels * params.kernel_h * params.kernel_w +
                                     ic * params.kernel_h * params.kernel_w +
                                     ky * params.kernel_w + kx;

                    sum = sum + input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }

    // Output index: [batch, out_channel, y, x]
    let output_idx = batch * params.out_channels * params.output_h * params.output_w +
                     out_c * params.output_h * params.output_w +
                     out_y * params.output_w + out_x;

    output[output_idx] = sum;
}
"#.to_string()
}

/// Generate WGSL code for 2D max pooling.
///
/// Input: [batch, channels, height, width]
/// Output: [batch, channels, out_h, out_w]
pub fn maxpool2d_shader() -> String {
    r#"struct PoolParams {
    batch_size: u32,
    channels: u32,
    input_h: u32,
    input_w: u32,
    output_h: u32,
    output_w: u32,
    pool_h: u32,
    pool_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: PoolParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let c_batch = global_id.z;

    let batch = c_batch / params.channels;
    let c = c_batch % params.channels;

    if (out_x >= params.output_w || out_y >= params.output_h || batch >= params.batch_size) {
        return;
    }

    let in_y_start = i32(out_y * params.stride_h) - i32(params.padding_h);
    let in_x_start = i32(out_x * params.stride_w) - i32(params.padding_w);

    var max_val: f32 = -3.402823466e+38; // -f32::MAX

    for (var py: u32 = 0u; py < params.pool_h; py = py + 1u) {
        for (var px: u32 = 0u; px < params.pool_w; px = px + 1u) {
            let in_y = in_y_start + i32(py);
            let in_x = in_x_start + i32(px);

            if (in_y >= 0 && in_y < i32(params.input_h) &&
                in_x >= 0 && in_x < i32(params.input_w)) {
                let input_idx = batch * params.channels * params.input_h * params.input_w +
                                c * params.input_h * params.input_w +
                                u32(in_y) * params.input_w + u32(in_x);
                max_val = max(max_val, input[input_idx]);
            }
        }
    }

    let output_idx = batch * params.channels * params.output_h * params.output_w +
                     c * params.output_h * params.output_w +
                     out_y * params.output_w + out_x;

    output[output_idx] = max_val;
}
"#.to_string()
}

/// Generate WGSL code for 2D average pooling.
pub fn avgpool2d_shader() -> String {
    r#"struct PoolParams {
    batch_size: u32,
    channels: u32,
    input_h: u32,
    input_w: u32,
    output_h: u32,
    output_w: u32,
    pool_h: u32,
    pool_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: PoolParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let c_batch = global_id.z;

    let batch = c_batch / params.channels;
    let c = c_batch % params.channels;

    if (out_x >= params.output_w || out_y >= params.output_h || batch >= params.batch_size) {
        return;
    }

    let in_y_start = i32(out_y * params.stride_h) - i32(params.padding_h);
    let in_x_start = i32(out_x * params.stride_w) - i32(params.padding_w);

    var sum: f32 = 0.0;
    var count: u32 = 0u;

    for (var py: u32 = 0u; py < params.pool_h; py = py + 1u) {
        for (var px: u32 = 0u; px < params.pool_w; px = px + 1u) {
            let in_y = in_y_start + i32(py);
            let in_x = in_x_start + i32(px);

            if (in_y >= 0 && in_y < i32(params.input_h) &&
                in_x >= 0 && in_x < i32(params.input_w)) {
                let input_idx = batch * params.channels * params.input_h * params.input_w +
                                c * params.input_h * params.input_w +
                                u32(in_y) * params.input_w + u32(in_x);
                sum = sum + input[input_idx];
                count = count + 1u;
            }
        }
    }

    let output_idx = batch * params.channels * params.output_h * params.output_w +
                     c * params.output_h * params.output_w +
                     out_y * params.output_w + out_x;

    output[output_idx] = sum / f32(count);
}
"#.to_string()
}

/// Generate WGSL code for batch normalization.
///
/// Applies batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
pub fn batchnorm_shader() -> String {
    r#"struct BNParams {
    batch_size: u32,
    channels: u32,
    spatial_size: u32,
    eps: f32,
}

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> mean: array<f32>;

@group(0) @binding(2)
var<storage, read> var_data: array<f32>;

@group(0) @binding(3)
var<storage, read> gamma: array<f32>;

@group(0) @binding(4)
var<storage, read> beta: array<f32>;

@group(0) @binding(5)
var<storage, read_write> output: array<f32>;

@group(0) @binding(6)
var<uniform> params: BNParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = params.batch_size * params.channels * params.spatial_size;

    if (idx >= total_size) {
        return;
    }

    // Determine which channel this element belongs to
    let channel = (idx / params.spatial_size) % params.channels;

    let x = input[idx];
    let m = mean[channel];
    let v = var_data[channel];
    let g = gamma[channel];
    let b = beta[channel];

    // Normalize
    let x_norm = (x - m) / sqrt(v + params.eps);

    // Scale and shift
    output[idx] = x_norm * g + b;
}
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_op_shader_generation() {
        let shader = binary_op_shader("+");
        assert!(shader.contains("input_a[idx] + input_b[idx]"));
        assert!(shader.contains("@compute @workgroup_size(256)"));
    }

    #[test]
    fn test_unary_op_shader_generation() {
        let shader = unary_op_shader("sqrt");
        assert!(shader.contains("sqrt(input[idx])"));
    }

    #[test]
    fn test_custom_binary_shader() {
        let shader = binary_op_custom_shader("pow(a, b)");
        assert!(shader.contains("pow(a, b)"));
        assert!(shader.contains("let a = input_a[idx]"));
        assert!(shader.contains("let b = input_b[idx]"));
    }

    #[test]
    fn test_matmul_shader_generation() {
        let shader = matmul_shader();
        assert!(shader.contains("matrix_a"));
        assert!(shader.contains("matrix_b"));
        assert!(shader.contains("matrix_c"));
        assert!(shader.contains("struct Dimensions"));
        assert!(shader.contains("workgroup_size(16, 16, 1)"));
        assert!(shader.contains("tile_a"));
        assert!(shader.contains("tile_b"));
    }

    #[test]
    fn test_reduction_shader_sum() {
        let shader = reduction_shader("sum");
        assert!(shader.contains("acc + val"));
        assert!(shader.contains("var<workgroup> shared_data"));
        assert!(shader.contains("workgroupBarrier"));
    }

    #[test]
    fn test_reduction_shader_max() {
        let shader = reduction_shader("max");
        assert!(shader.contains("max(acc, val)"));
        assert!(shader.contains("-3.402823466e+38")); // -f32::MAX
    }

    #[test]
    fn test_axis_reduction_shader() {
        let shader = axis_reduction_shader("sum");
        assert!(shader.contains("struct Dimensions"));
        assert!(shader.contains("axis_size"));
        assert!(shader.contains("outer_size"));
    }
}
