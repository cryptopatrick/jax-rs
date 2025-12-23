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
}
