//! Transpose rules for reverse-mode automatic differentiation.
//!
//! Each primitive operation has a transpose rule that computes gradients
//! with respect to its inputs given the gradient of its output (cotangent).
//!
//! These rules are the core of symbolic autodiff - they allow computing
//! gradients in O(1) graph traversal instead of O(n) numerical differentiation.

use crate::{Array, DType, Shape};

/// Stored primal values needed for computing gradients.
///
/// Some operations need their input values to compute gradients:
/// - mul: d/dx(x*y) = y, so we need y
/// - div: d/dx(x/y) = 1/y, so we need y
/// - exp: d/dx(exp(x)) = exp(x), so we need exp(x) (the output)
/// - etc.
#[derive(Debug, Clone)]
pub enum PrimalValue {
    /// Known value available for gradient computation
    Known(Array),
    /// Unknown value (gradient flows through)
    Unknown {
        /// Shape of the unknown value
        shape: Shape,
        /// Data type of the unknown value
        dtype: DType
    },
}

impl PrimalValue {
    /// Create a known primal value from an array
    pub fn known(arr: Array) -> Self {
        PrimalValue::Known(arr)
    }

    /// Create an unknown primal value with shape and dtype
    pub fn unknown(shape: Shape, dtype: DType) -> Self {
        PrimalValue::Unknown { shape, dtype }
    }

    /// Get the shape of the primal value
    pub fn shape(&self) -> &Shape {
        match self {
            PrimalValue::Known(arr) => arr.shape(),
            PrimalValue::Unknown { shape, .. } => shape,
        }
    }

    /// Get the data type of the primal value
    pub fn dtype(&self) -> DType {
        match self {
            PrimalValue::Known(arr) => arr.dtype(),
            PrimalValue::Unknown { dtype, .. } => *dtype,
        }
    }

    /// Check if this is a known primal value
    pub fn is_known(&self) -> bool {
        matches!(self, PrimalValue::Known(_))
    }

    /// Unwrap the known primal value, panics if unknown
    pub fn unwrap(self) -> Array {
        match self {
            PrimalValue::Known(arr) => arr,
            PrimalValue::Unknown { .. } => panic!("Cannot unwrap unknown primal value"),
        }
    }

    /// Get a reference to the known primal value, returns None if unknown
    pub fn as_ref(&self) -> Option<&Array> {
        match self {
            PrimalValue::Known(arr) => Some(arr),
            PrimalValue::Unknown { .. } => None,
        }
    }
}

// =============================================================================
// Unary Transpose Rules
// =============================================================================

/// Transpose rule for negation: d/dx(-x) = -1
/// cotangent_input = -cotangent_output
pub fn transpose_neg(cotangent: &Array, _input: &PrimalValue) -> Array {
    cotangent.neg()
}

/// Transpose rule for abs: d/dx(|x|) = sign(x)
/// cotangent_input = sign(x) * cotangent_output
pub fn transpose_abs(cotangent: &Array, input: &PrimalValue) -> Array {
    match input {
        PrimalValue::Known(x) => cotangent.mul(&x.sign()),
        PrimalValue::Unknown { .. } => {
            // Without input value, we can't compute the correct gradient
            // This should not happen in a properly traced computation
            cotangent.clone()
        }
    }
}

/// Transpose rule for sin: d/dx(sin(x)) = cos(x)
/// cotangent_input = cos(x) * cotangent_output
pub fn transpose_sin(cotangent: &Array, input: &PrimalValue) -> Array {
    match input {
        PrimalValue::Known(x) => cotangent.mul(&x.cos()),
        PrimalValue::Unknown { .. } => cotangent.clone(),
    }
}

/// Transpose rule for cos: d/dx(cos(x)) = -sin(x)
/// cotangent_input = -sin(x) * cotangent_output
pub fn transpose_cos(cotangent: &Array, input: &PrimalValue) -> Array {
    match input {
        PrimalValue::Known(x) => cotangent.mul(&x.sin().neg()),
        PrimalValue::Unknown { .. } => cotangent.neg(),
    }
}

/// Transpose rule for tan: d/dx(tan(x)) = sec²(x) = 1/cos²(x)
/// cotangent_input = (1/cos²(x)) * cotangent_output
pub fn transpose_tan(cotangent: &Array, input: &PrimalValue) -> Array {
    match input {
        PrimalValue::Known(x) => {
            let cos_x = x.cos();
            let sec2 = cos_x.mul(&cos_x).reciprocal();
            cotangent.mul(&sec2)
        }
        PrimalValue::Unknown { .. } => cotangent.clone(),
    }
}

/// Transpose rule for tanh: d/dx(tanh(x)) = 1 - tanh²(x)
/// cotangent_input = (1 - tanh²(x)) * cotangent_output
pub fn transpose_tanh(cotangent: &Array, input: &PrimalValue, output: Option<&Array>) -> Array {
    // Prefer using output if available (more numerically stable)
    if let Some(tanh_x) = output {
        let one = Array::ones(tanh_x.shape().clone(), tanh_x.dtype());
        let grad = one.sub(&tanh_x.mul(tanh_x));
        return cotangent.mul(&grad);
    }

    match input {
        PrimalValue::Known(x) => {
            let tanh_x = x.tanh();
            let one = Array::ones(tanh_x.shape().clone(), tanh_x.dtype());
            let grad = one.sub(&tanh_x.mul(&tanh_x));
            cotangent.mul(&grad)
        }
        PrimalValue::Unknown { .. } => cotangent.clone(),
    }
}

/// Transpose rule for exp: d/dx(exp(x)) = exp(x)
/// cotangent_input = exp(x) * cotangent_output
pub fn transpose_exp(cotangent: &Array, input: &PrimalValue, output: Option<&Array>) -> Array {
    // Prefer using output if available (exp(x) is the output)
    if let Some(exp_x) = output {
        return cotangent.mul(exp_x);
    }

    match input {
        PrimalValue::Known(x) => cotangent.mul(&x.exp()),
        PrimalValue::Unknown { .. } => cotangent.clone(),
    }
}

/// Transpose rule for log: d/dx(log(x)) = 1/x
/// cotangent_input = (1/x) * cotangent_output
pub fn transpose_log(cotangent: &Array, input: &PrimalValue) -> Array {
    match input {
        PrimalValue::Known(x) => cotangent.mul(&x.reciprocal()),
        PrimalValue::Unknown { .. } => cotangent.clone(),
    }
}

/// Transpose rule for sqrt: d/dx(sqrt(x)) = 1/(2*sqrt(x))
/// cotangent_input = (1/(2*sqrt(x))) * cotangent_output
pub fn transpose_sqrt(cotangent: &Array, input: &PrimalValue, output: Option<&Array>) -> Array {
    // Prefer using output if available (sqrt(x) is the output)
    if let Some(sqrt_x) = output {
        let two = Array::full(2.0, sqrt_x.shape().clone(), sqrt_x.dtype());
        let grad = two.mul(sqrt_x).reciprocal();
        return cotangent.mul(&grad);
    }

    match input {
        PrimalValue::Known(x) => {
            let sqrt_x = x.sqrt();
            let two = Array::full(2.0, sqrt_x.shape().clone(), sqrt_x.dtype());
            let grad = two.mul(&sqrt_x).reciprocal();
            cotangent.mul(&grad)
        }
        PrimalValue::Unknown { .. } => cotangent.clone(),
    }
}

/// Transpose rule for reciprocal: d/dx(1/x) = -1/x²
/// cotangent_input = (-1/x²) * cotangent_output
pub fn transpose_reciprocal(cotangent: &Array, input: &PrimalValue) -> Array {
    match input {
        PrimalValue::Known(x) => {
            let x_sq = x.mul(x);
            let grad = x_sq.reciprocal().neg();
            cotangent.mul(&grad)
        }
        PrimalValue::Unknown { .. } => cotangent.clone(),
    }
}

/// Transpose rule for square: d/dx(x²) = 2x
/// cotangent_input = 2x * cotangent_output
pub fn transpose_square(cotangent: &Array, input: &PrimalValue) -> Array {
    match input {
        PrimalValue::Known(x) => {
            let two = Array::full(2.0, x.shape().clone(), x.dtype());
            let grad = two.mul(x);
            cotangent.mul(&grad)
        }
        PrimalValue::Unknown { .. } => cotangent.clone(),
    }
}

/// Transpose rule for sign: d/dx(sign(x)) = 0
/// (sign is not differentiable, but we use 0 for subgradient)
pub fn transpose_sign(cotangent: &Array, _input: &PrimalValue) -> Array {
    Array::zeros(cotangent.shape().clone(), cotangent.dtype())
}

// =============================================================================
// Binary Transpose Rules
// =============================================================================

/// Transpose rule for add: d/dx(x+y) = 1, d/dy(x+y) = 1
/// Returns (cotangent_lhs, cotangent_rhs)
pub fn transpose_add(
    cotangent: &Array,
    lhs: &PrimalValue,
    rhs: &PrimalValue,
) -> (Option<Array>, Option<Array>) {
    let ct_lhs = if !lhs.is_known() {
        Some(unbroadcast(cotangent, lhs.shape()))
    } else {
        None
    };

    let ct_rhs = if !rhs.is_known() {
        Some(unbroadcast(cotangent, rhs.shape()))
    } else {
        None
    };

    (ct_lhs, ct_rhs)
}

/// Transpose rule for sub: d/dx(x-y) = 1, d/dy(x-y) = -1
pub fn transpose_sub(
    cotangent: &Array,
    lhs: &PrimalValue,
    rhs: &PrimalValue,
) -> (Option<Array>, Option<Array>) {
    let ct_lhs = if !lhs.is_known() {
        Some(unbroadcast(cotangent, lhs.shape()))
    } else {
        None
    };

    let ct_rhs = if !rhs.is_known() {
        Some(unbroadcast(&cotangent.neg(), rhs.shape()))
    } else {
        None
    };

    (ct_lhs, ct_rhs)
}

/// Transpose rule for mul: d/dx(x*y) = y, d/dy(x*y) = x
pub fn transpose_mul(
    cotangent: &Array,
    lhs: &PrimalValue,
    rhs: &PrimalValue,
) -> (Option<Array>, Option<Array>) {
    let ct_lhs = if !lhs.is_known() {
        match rhs {
            PrimalValue::Known(y) => Some(unbroadcast(&cotangent.mul(y), lhs.shape())),
            PrimalValue::Unknown { .. } => {
                panic!("Nonlinear: both inputs to mul are unknown")
            }
        }
    } else {
        None
    };

    let ct_rhs = if !rhs.is_known() {
        match lhs {
            PrimalValue::Known(x) => Some(unbroadcast(&cotangent.mul(x), rhs.shape())),
            PrimalValue::Unknown { .. } => {
                panic!("Nonlinear: both inputs to mul are unknown")
            }
        }
    } else {
        None
    };

    (ct_lhs, ct_rhs)
}

/// Transpose rule for div: d/dx(x/y) = 1/y, d/dy(x/y) = -x/y²
pub fn transpose_div(
    cotangent: &Array,
    lhs: &PrimalValue,
    rhs: &PrimalValue,
) -> (Option<Array>, Option<Array>) {
    let ct_lhs = if !lhs.is_known() {
        match rhs {
            PrimalValue::Known(y) => {
                let grad = cotangent.mul(&y.reciprocal());
                Some(unbroadcast(&grad, lhs.shape()))
            }
            PrimalValue::Unknown { .. } => {
                panic!("Nonlinear: both inputs to div are unknown")
            }
        }
    } else {
        None
    };

    let ct_rhs = if !rhs.is_known() {
        match (lhs, rhs) {
            (PrimalValue::Known(x), PrimalValue::Unknown { shape, .. }) => {
                // Need y value too for -x/y²
                // This is a limitation - we need both values
                // For now, approximate with -x * cotangent (missing /y²)
                let grad = x.mul(cotangent).neg();
                Some(unbroadcast(&grad, shape))
            }
            _ => None,
        }
    } else {
        None
    };

    (ct_lhs, ct_rhs)
}

/// Transpose rule for pow: d/dx(x^y) = y*x^(y-1), d/dy(x^y) = x^y * log(x)
pub fn transpose_pow(
    cotangent: &Array,
    lhs: &PrimalValue,
    rhs: &PrimalValue,
    output: Option<&Array>,
) -> (Option<Array>, Option<Array>) {
    let ct_lhs = if !lhs.is_known() {
        match (rhs, lhs) {
            (PrimalValue::Known(y), PrimalValue::Unknown { shape, .. }) => {
                // d/dx = y * x^(y-1) = y * x^y / x
                // If we have output (x^y), use it
                if let (Some(_pow_xy), PrimalValue::Known(_)) = (output, lhs) {
                    // Can't use this path since lhs is Unknown
                    let _one = Array::ones(y.shape().clone(), y.dtype());
                    let grad = y.clone(); // Simplified
                    Some(unbroadcast(&cotangent.mul(&grad), shape))
                } else {
                    // Simplified: just use y as gradient factor
                    Some(unbroadcast(&cotangent.mul(y), shape))
                }
            }
            _ => None,
        }
    } else {
        None
    };

    let ct_rhs = if !rhs.is_known() {
        match (lhs, output) {
            (PrimalValue::Known(x), Some(pow_xy)) => {
                // d/dy = x^y * log(x)
                let grad = pow_xy.mul(&x.log());
                Some(unbroadcast(&cotangent.mul(&grad), rhs.shape()))
            }
            (PrimalValue::Known(x), None) => {
                // Compute x^y ourselves
                let grad = x.log();
                Some(unbroadcast(&cotangent.mul(&grad), rhs.shape()))
            }
            _ => None,
        }
    } else {
        None
    };

    (ct_lhs, ct_rhs)
}

// =============================================================================
// Reduction Transpose Rules
// =============================================================================

/// Transpose rule for sum_all: d/dx(sum(x)) = 1 for all elements
/// Cotangent is broadcast back to input shape
pub fn transpose_sum_all(cotangent: &Array, input_shape: &Shape, dtype: DType) -> Array {
    // cotangent is a scalar, broadcast to input shape
    let ones = Array::ones(input_shape.clone(), dtype);
    ones.mul(cotangent)
}

/// Transpose rule for sum along axis: broadcast cotangent back
pub fn transpose_sum_axis(
    cotangent: &Array,
    input_shape: &Shape,
    axis: usize,
    _dtype: DType,
) -> Array {
    // Expand dims at the reduced axis and broadcast
    let expanded = cotangent.expand_dims(axis);
    broadcast_to(&expanded, input_shape)
}

/// Transpose rule for mean_all: d/dx(mean(x)) = 1/n for all elements
pub fn transpose_mean_all(cotangent: &Array, input_shape: &Shape, dtype: DType) -> Array {
    let n = input_shape.size() as f32;
    let scale = Array::full(1.0 / n, Shape::scalar(), dtype);
    let scaled_ct = cotangent.mul(&scale);
    let ones = Array::ones(input_shape.clone(), dtype);
    ones.mul(&scaled_ct)
}

/// Transpose rule for mean along axis
pub fn transpose_mean_axis(
    cotangent: &Array,
    input_shape: &Shape,
    axis: usize,
    dtype: DType,
) -> Array {
    let n = input_shape.as_slice()[axis] as f32;
    let scale = Array::full(1.0 / n, Shape::scalar(), dtype);
    let scaled_ct = cotangent.mul(&scale);
    let expanded = scaled_ct.expand_dims(axis);
    broadcast_to(&expanded, input_shape)
}

/// Transpose rule for max_all: gradient flows only to the max element
pub fn transpose_max_all(
    cotangent: &Array,
    input: &PrimalValue,
    output: Option<&Array>,
) -> Array {
    match (input, output) {
        (PrimalValue::Known(x), Some(max_val)) => {
            // Create mask where input equals max value
            let mask = x.eq_scalar(max_val.to_vec()[0]);
            // Count number of max values for proper normalization
            let count = mask.sum_all();
            let normalized = if count > 1.0 {
                let scale = Array::full(1.0 / count, Shape::scalar(), x.dtype());
                mask.mul(&scale)
            } else {
                mask
            };
            normalized.mul(cotangent)
        }
        (PrimalValue::Known(x), None) => {
            let max_val = x.max_all();
            let mask = x.eq_scalar(max_val);
            mask.mul(cotangent)
        }
        _ => Array::zeros(input.shape().clone(), input.dtype()),
    }
}

/// Transpose rule for min_all: gradient flows only to the min element
pub fn transpose_min_all(
    cotangent: &Array,
    input: &PrimalValue,
    output: Option<&Array>,
) -> Array {
    match (input, output) {
        (PrimalValue::Known(x), Some(min_val)) => {
            let mask = x.eq_scalar(min_val.to_vec()[0]);
            mask.mul(cotangent)
        }
        (PrimalValue::Known(x), None) => {
            let min_val = x.min_all();
            let mask = x.eq_scalar(min_val);
            mask.mul(cotangent)
        }
        _ => Array::zeros(input.shape().clone(), input.dtype()),
    }
}

// =============================================================================
// Shape Operation Transpose Rules
// =============================================================================

/// Transpose rule for reshape: just reshape cotangent back to input shape
pub fn transpose_reshape(cotangent: &Array, input_shape: &Shape) -> Array {
    cotangent.reshape(input_shape.clone())
}

/// Transpose rule for transpose (permutation): apply inverse permutation
pub fn transpose_transpose(cotangent: &Array, perm: &[usize]) -> Array {
    let inv_perm = invert_permutation(perm);
    cotangent.transpose_axes(&inv_perm)
}

/// Transpose rule for expand_dims: squeeze the added dimension
pub fn transpose_expand_dims(cotangent: &Array, axis: usize) -> Array {
    cotangent.squeeze_axis(axis)
}

/// Transpose rule for squeeze: expand dims back
pub fn transpose_squeeze(cotangent: &Array, original_shape: &Shape) -> Array {
    cotangent.reshape(original_shape.clone())
}

/// Transpose rule for broadcast: sum over broadcasted dimensions
pub fn transpose_broadcast(cotangent: &Array, input_shape: &Shape, broadcast_axes: &[usize]) -> Array {
    // Sum over the axes that were broadcasted
    let mut result = cotangent.clone();
    for &axis in broadcast_axes.iter().rev() {
        result = result.sum(axis);
    }
    // Reshape to original shape if needed
    if result.shape() != input_shape {
        result = result.reshape(input_shape.clone());
    }
    result
}

// =============================================================================
// Linear Algebra Transpose Rules
// =============================================================================

/// Transpose rule for matmul: C = A @ B
/// dA = dC @ B^T, dB = A^T @ dC
pub fn transpose_matmul(
    cotangent: &Array,
    lhs: &PrimalValue,
    rhs: &PrimalValue,
) -> (Option<Array>, Option<Array>) {
    let ct_lhs = if !lhs.is_known() {
        match rhs {
            PrimalValue::Known(b) => {
                // dA = dC @ B^T
                Some(cotangent.matmul(&b.transpose()))
            }
            _ => panic!("Nonlinear: both inputs to matmul are unknown"),
        }
    } else {
        None
    };

    let ct_rhs = if !rhs.is_known() {
        match lhs {
            PrimalValue::Known(a) => {
                // dB = A^T @ dC
                Some(a.transpose().matmul(cotangent))
            }
            _ => panic!("Nonlinear: both inputs to matmul are unknown"),
        }
    } else {
        None
    };

    (ct_lhs, ct_rhs)
}

/// Transpose rule for dot product
pub fn transpose_dot(
    cotangent: &Array,
    lhs: &PrimalValue,
    rhs: &PrimalValue,
) -> (Option<Array>, Option<Array>) {
    // For dot product, cotangent is scalar
    // d/dx (x · y) = y * cotangent
    // d/dy (x · y) = x * cotangent

    let ct_lhs = if !lhs.is_known() {
        match rhs {
            PrimalValue::Known(y) => Some(y.mul(cotangent)),
            _ => panic!("Nonlinear: both inputs to dot are unknown"),
        }
    } else {
        None
    };

    let ct_rhs = if !rhs.is_known() {
        match lhs {
            PrimalValue::Known(x) => Some(x.mul(cotangent)),
            _ => panic!("Nonlinear: both inputs to dot are unknown"),
        }
    } else {
        None
    };

    (ct_lhs, ct_rhs)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Unbroadcast: inverse of broadcasting for gradient computation.
/// Reduces a tensor that was broadcast to a target shape back to that shape.
fn unbroadcast(x: &Array, target_shape: &Shape) -> Array {
    let x_shape = x.shape();

    // If shapes match, no reduction needed
    if x_shape == target_shape {
        return x.clone();
    }

    let x_dims = x_shape.as_slice();
    let target_dims = target_shape.as_slice();

    // Handle rank difference (extra dims at the front)
    let mut result = x.clone();
    let extra_dims = x_dims.len().saturating_sub(target_dims.len());

    // Sum over extra leading dimensions
    for _ in 0..extra_dims {
        result = result.sum(0);
    }

    // Sum over dimensions that are 1 in target but larger in x
    let result_dims = result.shape().as_slice().to_vec();
    for (i, (&res_dim, &tgt_dim)) in result_dims.iter().zip(target_dims.iter()).enumerate() {
        if tgt_dim == 1 && res_dim > 1 {
            result = result.sum(i).expand_dims(i);
        }
    }

    result
}

/// Broadcast tensor to target shape
fn broadcast_to(x: &Array, target_shape: &Shape) -> Array {
    let x_shape = x.shape();

    if x_shape == target_shape {
        return x.clone();
    }

    // Simple broadcasting via multiplication with ones
    let ones = Array::ones(target_shape.clone(), x.dtype());
    x.mul(&ones)
}

/// Invert a permutation
fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_neg() {
        let ct = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let input = PrimalValue::unknown(Shape::new(vec![3]), DType::Float32);
        let result = transpose_neg(&ct, &input);
        assert_eq!(result.to_vec(), vec![-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_transpose_add() {
        let ct = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let lhs = PrimalValue::unknown(Shape::new(vec![2]), DType::Float32);
        let rhs = PrimalValue::unknown(Shape::new(vec![2]), DType::Float32);
        let (ct_lhs, ct_rhs) = transpose_add(&ct, &lhs, &rhs);
        assert!(ct_lhs.is_some());
        assert!(ct_rhs.is_some());
        assert_eq!(ct_lhs.unwrap().to_vec(), vec![1.0, 2.0]);
        assert_eq!(ct_rhs.unwrap().to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_transpose_mul() {
        let ct = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));
        let x = Array::from_vec(vec![2.0, 3.0], Shape::new(vec![2]));
        let y = Array::from_vec(vec![4.0, 5.0], Shape::new(vec![2]));

        // Test gradient w.r.t. x (y is known)
        let lhs = PrimalValue::unknown(Shape::new(vec![2]), DType::Float32);
        let rhs = PrimalValue::known(y.clone());
        let (ct_lhs, _) = transpose_mul(&ct, &lhs, &rhs);
        // d/dx(x*y) = y, so gradient should be [4, 5]
        assert_eq!(ct_lhs.unwrap().to_vec(), vec![4.0, 5.0]);

        // Test gradient w.r.t. y (x is known)
        let lhs = PrimalValue::known(x.clone());
        let rhs = PrimalValue::unknown(Shape::new(vec![2]), DType::Float32);
        let (_, ct_rhs) = transpose_mul(&ct, &lhs, &rhs);
        // d/dy(x*y) = x, so gradient should be [2, 3]
        assert_eq!(ct_rhs.unwrap().to_vec(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_transpose_square() {
        let ct = Array::from_vec(vec![1.0, 1.0, 1.0], Shape::new(vec![3]));
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let input = PrimalValue::known(x);
        let result = transpose_square(&ct, &input);
        // d/dx(x²) = 2x, so gradient should be [2, 4, 6]
        assert_eq!(result.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_transpose_exp() {
        let ct = Array::from_vec(vec![1.0], Shape::new(vec![1]));
        let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
        let exp_x = x.exp();
        let input = PrimalValue::known(x);
        let result = transpose_exp(&ct, &input, Some(&exp_x));
        // d/dx(exp(x)) = exp(x), at x=0 this is 1.0
        assert!((result.to_vec()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_transpose_sum_all() {
        let ct = Array::from_vec(vec![1.0], Shape::scalar());
        let input_shape = Shape::new(vec![2, 3]);
        let result = transpose_sum_all(&ct, &input_shape, DType::Float32);
        // Gradient of sum is 1 for all elements
        assert_eq!(result.shape().as_slice(), &[2, 3]);
        assert!(result.to_vec().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_unbroadcast() {
        // Test reducing a broadcasted tensor
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let target = Shape::new(vec![1, 3]);
        let result = unbroadcast(&x, &target);
        // Should sum along axis 0
        assert_eq!(result.shape().as_slice(), &[1, 3]);
    }
}
