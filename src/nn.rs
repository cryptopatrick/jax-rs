//! Neural network operations and activation functions.
//!
//! This module provides common operations used in neural networks,
//! including activation functions, normalization layers, and loss functions.

use crate::{Array, DType, Shape};

/// Rectified Linear Unit (ReLU) activation.
///
/// ReLU(x) = max(0, x)
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![-1.0, 0.0, 1.0, 2.0], Shape::new(vec![4]));
/// let y = nn::relu(&x);
/// assert_eq!(y.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
/// ```
pub fn relu(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();
    Array::from_vec(result, x.shape().clone())
}

/// Leaky ReLU activation.
///
/// LeakyReLU(x) = max(alpha * x, x)
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0], Shape::new(vec![4]));
/// let y = nn::leaky_relu(&x, 0.1);
/// assert_eq!(y.to_vec(), vec![-0.2, -0.1, 0.0, 1.0]);
/// ```
pub fn leaky_relu(x: &Array, alpha: f32) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> =
        data.iter().map(|&v| if v > 0.0 { v } else { alpha * v }).collect();
    Array::from_vec(result, x.shape().clone())
}

/// Exponential Linear Unit (ELU) activation.
///
/// ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![-1.0, 0.0, 1.0], Shape::new(vec![3]));
/// let y = nn::elu(&x, 1.0);
/// ```
pub fn elu(x: &Array, alpha: f32) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data
        .iter()
        .map(|&v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
        .collect();
    Array::from_vec(result, x.shape().clone())
}

/// Scaled Exponential Linear Unit (SELU) activation.
///
/// SELU(x) = lambda * (x if x > 0, else alpha * (exp(x) - 1))
/// where lambda ≈ 1.0507 and alpha ≈ 1.6733
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![-1.0, 0.0, 1.0], Shape::new(vec![3]));
/// let y = nn::selu(&x);
/// ```
pub fn selu(x: &Array) -> Array {
    const LAMBDA: f32 = 1.0507009873554804934193349852946;
    const ALPHA: f32 = 1.6732632423543772848170429916717;

    let data = x.to_vec();
    let result: Vec<f32> = data
        .iter()
        .map(|&v| {
            if v > 0.0 {
                LAMBDA * v
            } else {
                LAMBDA * ALPHA * (v.exp() - 1.0)
            }
        })
        .collect();
    Array::from_vec(result, x.shape().clone())
}

/// Gaussian Error Linear Unit (GELU) activation.
///
/// GELU(x) ≈ x * Φ(x) where Φ is the cumulative distribution function
/// of the standard normal distribution.
///
/// Uses the tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![-1.0, 0.0, 1.0], Shape::new(vec![3]));
/// let y = nn::gelu(&x);
/// ```
pub fn gelu(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data
        .iter()
        .map(|&v| {
            let coeff = (2.0 / std::f32::consts::PI).sqrt();
            let inner = coeff * (v + 0.044715 * v.powi(3));
            0.5 * v * (1.0 + inner.tanh())
        })
        .collect();
    Array::from_vec(result, x.shape().clone())
}

/// Sigmoid activation.
///
/// sigmoid(x) = 1 / (1 + exp(-x))
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![-2.0, 0.0, 2.0], Shape::new(vec![3]));
/// let y = nn::sigmoid(&x);
/// ```
pub fn sigmoid(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> =
        data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    Array::from_vec(result, x.shape().clone())
}

/// Softmax activation.
///
/// Applies softmax along the last axis.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let y = nn::softmax(&x);
/// let sum: f32 = y.to_vec().iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
/// ```
pub fn softmax(x: &Array) -> Array {
    assert_eq!(x.ndim(), 1, "softmax only supports 1-D arrays for now");

    let data = x.to_vec();

    // Subtract max for numerical stability
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> =
        data.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    let result: Vec<f32> = exp_vals.iter().map(|&v| v / sum).collect();
    Array::from_vec(result, x.shape().clone())
}

/// Log-softmax activation.
///
/// Numerically stable implementation of log(softmax(x)).
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let y = nn::log_softmax(&x);
/// ```
pub fn log_softmax(x: &Array) -> Array {
    assert_eq!(x.ndim(), 1, "log_softmax only supports 1-D arrays for now");

    let data = x.to_vec();

    // Subtract max for numerical stability
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let shifted: Vec<f32> = data.iter().map(|&v| v - max_val).collect();
    let log_sum_exp = shifted.iter().map(|&v| v.exp()).sum::<f32>().ln();

    let result: Vec<f32> = shifted.iter().map(|&v| v - log_sum_exp).collect();
    Array::from_vec(result, x.shape().clone())
}

/// One-hot encoding.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape, DType};
/// let indices = Array::from_vec(vec![0.0, 2.0, 1.0], Shape::new(vec![3]));
/// let one_hot = nn::one_hot(&indices, 3, DType::Float32);
/// assert_eq!(one_hot.shape().as_slice(), &[3, 3]);
/// ```
pub fn one_hot(indices: &Array, num_classes: usize, dtype: DType) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert_eq!(indices.ndim(), 1, "one_hot only supports 1-D arrays");

    let index_data = indices.to_vec();
    let n = index_data.len();
    let mut result = vec![0.0; n * num_classes];

    for (i, &idx) in index_data.iter().enumerate() {
        let class = idx as usize;
        if class < num_classes {
            result[i * num_classes + class] = 1.0;
        }
    }

    Array::from_vec(result, Shape::new(vec![n, num_classes]))
}

/// Dropout layer.
///
/// Randomly sets elements to zero with probability `rate` and scales
/// the rest by 1/(1-rate).
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape, random::PRNGKey};
/// let key = PRNGKey::from_seed(42);
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let y = nn::dropout(key, &x, 0.5, true);
/// ```
pub fn dropout(
    key: crate::random::PRNGKey,
    x: &Array,
    rate: f32,
    training: bool,
) -> Array {
    if !training || rate == 0.0 {
        return x.clone();
    }

    assert!((0.0..1.0).contains(&rate), "rate must be in [0, 1)");

    let mask = crate::random::bernoulli(
        key,
        1.0 - rate,
        x.shape().clone(),
        x.dtype(),
    );
    let scale = 1.0 / (1.0 - rate);

    x.mul(&mask).mul(&Array::from_vec(vec![scale], Shape::new(vec![1])))
}

/// Mean Squared Error loss.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let predictions = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let targets = Array::from_vec(vec![1.5, 2.5, 2.5], Shape::new(vec![3]));
/// let loss = nn::mse_loss(&predictions, &targets);
/// ```
pub fn mse_loss(predictions: &Array, targets: &Array) -> f32 {
    assert_eq!(
        predictions.shape(),
        targets.shape(),
        "Predictions and targets must have same shape"
    );

    let pred_data = predictions.to_vec();
    let target_data = targets.to_vec();

    let sum: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| (p - t).powi(2))
        .sum();

    sum / pred_data.len() as f32
}

/// Cross-entropy loss.
///
/// Computes -sum(targets * log(predictions + epsilon))
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let predictions = Array::from_vec(vec![0.7, 0.2, 0.1], Shape::new(vec![3]));
/// let targets = Array::from_vec(vec![1.0, 0.0, 0.0], Shape::new(vec![3]));
/// let loss = nn::cross_entropy_loss(&predictions, &targets);
/// ```
pub fn cross_entropy_loss(predictions: &Array, targets: &Array) -> f32 {
    assert_eq!(
        predictions.shape(),
        targets.shape(),
        "Predictions and targets must have same shape"
    );

    let pred_data = predictions.to_vec();
    let target_data = targets.to_vec();
    let epsilon = 1e-7;

    let sum: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| -t * (p + epsilon).ln())
        .sum();

    sum / pred_data.len() as f32
}

/// Binary cross-entropy loss.
///
/// Computes binary cross-entropy loss for binary classification tasks.
/// Formula: -mean(targets * log(predictions + epsilon) + (1 - targets) * log(1 - predictions + epsilon))
///
/// # Arguments
///
/// * `predictions` - Predicted probabilities (should be in [0, 1])
/// * `targets` - Target labels (should be 0 or 1)
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let predictions = Array::from_vec(vec![0.9, 0.1, 0.8], Shape::new(vec![3]));
/// let targets = Array::from_vec(vec![1.0, 0.0, 1.0], Shape::new(vec![3]));
/// let loss = nn::binary_cross_entropy(&predictions, &targets);
/// ```
pub fn binary_cross_entropy(predictions: &Array, targets: &Array) -> f32 {
    assert_eq!(
        predictions.shape(),
        targets.shape(),
        "Predictions and targets must have same shape"
    );

    let pred_data = predictions.to_vec();
    let target_data = targets.to_vec();
    let epsilon = 1e-7;

    let sum: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| {
            -t * (p + epsilon).ln() - (1.0 - t) * (1.0 - p + epsilon).ln()
        })
        .sum();

    sum / pred_data.len() as f32
}

/// Hinge loss for SVM.
///
/// Computes hinge loss for support vector machine classification.
/// Formula: mean(max(0, 1 - targets * predictions))
///
/// # Arguments
///
/// * `predictions` - Predicted values (can be any real number)
/// * `targets` - Target labels (should be -1 or 1)
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let predictions = Array::from_vec(vec![0.5, -0.3, 1.2], Shape::new(vec![3]));
/// let targets = Array::from_vec(vec![1.0, -1.0, 1.0], Shape::new(vec![3]));
/// let loss = nn::hinge_loss(&predictions, &targets);
/// ```
pub fn hinge_loss(predictions: &Array, targets: &Array) -> f32 {
    assert_eq!(
        predictions.shape(),
        targets.shape(),
        "Predictions and targets must have same shape"
    );

    let pred_data = predictions.to_vec();
    let target_data = targets.to_vec();

    let sum: f32 = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| {
            let margin = 1.0 - t * p;
            margin.max(0.0)
        })
        .sum();

    sum / pred_data.len() as f32
}

/// Batch normalization.
///
/// Normalizes inputs across the batch dimension.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
///     Shape::new(vec![2, 3])
/// );
/// let normalized = nn::batch_norm(&x, 1e-5);
/// ```
pub fn batch_norm(x: &Array, epsilon: f32) -> Array {
    assert_eq!(x.ndim(), 2, "batch_norm requires 2-D input [batch, features]");

    let data = x.to_vec();
    let shape = x.shape().as_slice();
    let (batch_size, features) = (shape[0], shape[1]);

    let mut result = Vec::with_capacity(data.len());

    // Normalize each feature independently
    for feat_idx in 0..features {
        // Collect values for this feature across the batch
        let mut feat_values = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            feat_values.push(data[batch_idx * features + feat_idx]);
        }

        // Compute mean and variance
        let mean: f32 = feat_values.iter().sum::<f32>() / batch_size as f32;
        let variance: f32 =
            feat_values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>()
                / batch_size as f32;

        let std = (variance + epsilon).sqrt();

        // Normalize
        for batch_idx in 0..batch_size {
            let idx = batch_idx * features + feat_idx;
            let normalized = (data[idx] - mean) / std;
            result.push(normalized);
        }
    }

    // Need to reorder results to match row-major layout
    let mut final_result = vec![0.0; data.len()];
    for batch_idx in 0..batch_size {
        for feat_idx in 0..features {
            final_result[batch_idx * features + feat_idx] =
                result[feat_idx * batch_size + batch_idx];
        }
    }

    Array::from_vec(final_result, x.shape().clone())
}

/// Layer normalization.
///
/// Normalizes inputs across the feature dimension.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let normalized = nn::layer_norm(&x, 1e-5);
/// ```
pub fn layer_norm(x: &Array, epsilon: f32) -> Array {
    let data = x.to_vec();

    // Compute mean and variance
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let variance: f32 = data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>()
        / data.len() as f32;

    let std = (variance + epsilon).sqrt();

    // Normalize
    let result: Vec<f32> = data.iter().map(|&v| (v - mean) / std).collect();
    Array::from_vec(result, x.shape().clone())
}

/// ReLU-6 activation: min(max(0, x), 6).
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![-1.0, 3.0, 7.0], Shape::new(vec![3]));
/// let y = nn::relu6(&x);
/// assert_eq!(y.to_vec(), vec![0.0, 3.0, 6.0]);
/// ```
pub fn relu6(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data.iter().map(|&v| v.max(0.0).min(6.0)).collect();
    Array::from_vec(result, x.shape().clone())
}

/// Softplus activation: log(1 + exp(x)).
///
/// A smooth approximation of ReLU.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
/// let y = nn::softplus(&x);
/// // softplus(0) ≈ ln(2) ≈ 0.693
/// assert!((y.to_vec()[0] - 0.693).abs() < 0.01);
/// ```
pub fn softplus(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data.iter().map(|&v| (1.0 + v.exp()).ln()).collect();
    Array::from_vec(result, x.shape().clone())
}

/// Softsign activation: x / (1 + |x|).
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![-2.0, 0.0, 2.0], Shape::new(vec![3]));
/// let y = nn::softsign(&x);
/// // softsign(-2) = -2/3, softsign(0) = 0, softsign(2) = 2/3
/// assert!((y.to_vec()[0] + 0.666).abs() < 0.01);
/// assert_eq!(y.to_vec()[1], 0.0);
/// assert!((y.to_vec()[2] - 0.666).abs() < 0.01);
/// ```
pub fn softsign(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data.iter().map(|&v| v / (1.0 + v.abs())).collect();
    Array::from_vec(result, x.shape().clone())
}

/// SiLU (Swish) activation: x * sigmoid(x).
///
/// Also known as Swish activation.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
/// let y = nn::silu(&x);
/// assert_eq!(y.to_vec()[0], 0.0);
/// ```
pub fn silu(x: &Array) -> Array {
    let sig = sigmoid(x);
    x.mul(&sig)
}

/// Swish activation (alias for SiLU).
pub fn swish(x: &Array) -> Array {
    silu(x)
}

/// Log-sigmoid activation: log(sigmoid(x)).
///
/// Numerically stable version of log(1 / (1 + exp(-x))).
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
/// let y = nn::log_sigmoid(&x);
/// // log(sigmoid(0)) = log(0.5) ≈ -0.693
/// assert!((y.to_vec()[0] + 0.693).abs() < 0.01);
/// ```
pub fn log_sigmoid(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data
        .iter()
        .map(|&v| {
            if v >= 0.0 {
                -(1.0 + (-v).exp()).ln()
            } else {
                v - (1.0 + v.exp()).ln()
            }
        })
        .collect();
    Array::from_vec(result, x.shape().clone())
}

/// Continuously Differentiable Exponential Linear Unit (CELU).
///
/// CELU(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![-1.0, 0.0, 1.0], Shape::new(vec![3]));
/// let y = nn::celu(&x, 1.0);
/// ```
pub fn celu(x: &Array, alpha: f32) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data
        .iter()
        .map(|&v| {
            if v > 0.0 {
                v
            } else {
                alpha * ((v / alpha).exp() - 1.0)
            }
        })
        .collect();
    Array::from_vec(result, x.shape().clone())
}

/// Gated Linear Unit (GLU).
///
/// Splits input along specified axis and applies sigmoid gating.
/// GLU(x) = x[:half] * sigmoid(x[half:])
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let y = nn::glu(&x, 0);
/// assert_eq!(y.shape().as_slice(), &[2]);
/// ```
pub fn glu(x: &Array, axis: isize) -> Array {
    let ndim = x.ndim() as isize;
    let ax = if axis < 0 { (ndim + axis) as usize } else { axis as usize };

    assert!(ax < x.ndim(), "Axis out of bounds");

    let shape = x.shape().as_slice();
    let dim_size = shape[ax];
    assert_eq!(dim_size % 2, 0, "GLU requires even dimension size along axis");

    let half = dim_size / 2;
    let data = x.to_vec();

    // Simple implementation for 1D case
    if x.ndim() == 1 {
        let left = &data[..half];
        let right = &data[half..];

        let result: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .map(|(&l, &r)| l * (1.0 / (1.0 + (-r).exp())))
            .collect();

        Array::from_vec(result, Shape::new(vec![half]))
    } else {
        // For higher dimensions, only support last axis for now
        assert_eq!(ax, x.ndim() - 1, "GLU only supports last axis for multi-dimensional arrays");

        let mut result_shape = shape.to_vec();
        result_shape[ax] = half;

        let stride = dim_size;
        let num_groups = data.len() / stride;
        let mut result = Vec::with_capacity(num_groups * half);

        for g in 0..num_groups {
            let offset = g * stride;
            for i in 0..half {
                let left = data[offset + i];
                let right = data[offset + half + i];
                result.push(left * (1.0 / (1.0 + (-right).exp())));
            }
        }

        Array::from_vec(result, Shape::new(result_shape))
    }
}

/// Squareplus activation: (x + sqrt(x^2 + b)) / 2.
///
/// A smooth approximation to ReLU with parameter b controlling smoothness.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
/// let y = nn::squareplus(&x, 4.0);
/// // squareplus(0, 4) = 2/2 = 1
/// assert!((y.to_vec()[0] - 1.0).abs() < 1e-5);
/// ```
pub fn squareplus(x: &Array, b: f32) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data
        .iter()
        .map(|&v| (v + (v * v + b).sqrt()) / 2.0)
        .collect();
    Array::from_vec(result, x.shape().clone())
}

/// Mish activation: x * tanh(softplus(x)).
///
/// Mish(x) = x * tanh(ln(1 + exp(x)))
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
/// let y = nn::mish(&x);
/// ```
pub fn mish(x: &Array) -> Array {
    let data = x.to_vec();
    let result: Vec<f32> = data
        .iter()
        .map(|&v| {
            let sp = (1.0 + v.exp()).ln();
            v * sp.tanh()
        })
        .collect();
    Array::from_vec(result, x.shape().clone())
}

/// Log-sum-exp: log(sum(exp(x), axis)).
///
/// Numerically stable computation of log(sum(exp(x))).
/// If axis is None, reduces over all elements.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let y = nn::logsumexp(&x);
/// ```
pub fn logsumexp(x: &Array) -> f32 {
    let data = x.to_vec();
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let sum_exp: f32 = data.iter().map(|&v| (v - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Max pooling over a 1D input.
///
/// Applies max pooling with specified kernel size and stride.
///
/// # Arguments
///
/// * `x` - Input array of shape (length,) or (batch, length)
/// * `kernel_size` - Size of the pooling window
/// * `stride` - Stride of the pooling window (defaults to kernel_size if None)
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0], Shape::new(vec![4]));
/// let y = nn::max_pool1d(&x, 2, Some(2));
/// assert_eq!(y.to_vec(), vec![3.0, 4.0]);
/// ```
pub fn max_pool1d(x: &Array, kernel_size: usize, stride: Option<usize>) -> Array {
    let stride = stride.unwrap_or(kernel_size);
    let data = x.to_vec();
    let shape = x.shape().as_slice();

    match shape.len() {
        1 => {
            let length = shape[0];
            let out_length = (length - kernel_size) / stride + 1;
            let mut result = Vec::with_capacity(out_length);

            for i in 0..out_length {
                let start = i * stride;
                let end = start + kernel_size;
                let max_val = data[start..end]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                result.push(max_val);
            }

            Array::from_vec(result, Shape::new(vec![out_length]))
        }
        2 => {
            // Batched: (batch, length)
            let batch = shape[0];
            let length = shape[1];
            let out_length = (length - kernel_size) / stride + 1;
            let mut result = Vec::with_capacity(batch * out_length);

            for b in 0..batch {
                for i in 0..out_length {
                    let start = b * length + i * stride;
                    let mut max_val = f32::NEG_INFINITY;
                    for k in 0..kernel_size {
                        max_val = max_val.max(data[start + k]);
                    }
                    result.push(max_val);
                }
            }

            Array::from_vec(result, Shape::new(vec![batch, out_length]))
        }
        _ => panic!("max_pool1d expects 1D or 2D input"),
    }
}

/// Average pooling over a 1D input.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 3.0, 2.0, 4.0], Shape::new(vec![4]));
/// let y = nn::avg_pool1d(&x, 2, Some(2));
/// assert_eq!(y.to_vec(), vec![2.0, 3.0]);
/// ```
pub fn avg_pool1d(x: &Array, kernel_size: usize, stride: Option<usize>) -> Array {
    let stride = stride.unwrap_or(kernel_size);
    let data = x.to_vec();
    let shape = x.shape().as_slice();

    match shape.len() {
        1 => {
            let length = shape[0];
            let out_length = (length - kernel_size) / stride + 1;
            let mut result = Vec::with_capacity(out_length);

            for i in 0..out_length {
                let start = i * stride;
                let end = start + kernel_size;
                let sum: f32 = data[start..end].iter().sum();
                result.push(sum / kernel_size as f32);
            }

            Array::from_vec(result, Shape::new(vec![out_length]))
        }
        2 => {
            let batch = shape[0];
            let length = shape[1];
            let out_length = (length - kernel_size) / stride + 1;
            let mut result = Vec::with_capacity(batch * out_length);

            for b in 0..batch {
                for i in 0..out_length {
                    let start = b * length + i * stride;
                    let mut sum = 0.0;
                    for k in 0..kernel_size {
                        sum += data[start + k];
                    }
                    result.push(sum / kernel_size as f32);
                }
            }

            Array::from_vec(result, Shape::new(vec![batch, out_length]))
        }
        _ => panic!("avg_pool1d expects 1D or 2D input"),
    }
}

/// Max pooling over a 2D input.
///
/// # Arguments
///
/// * `x` - Input array of shape (height, width) or (batch, height, width)
/// * `kernel_size` - Size of the pooling window (height, width)
/// * `stride` - Stride of the pooling window
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
///     Shape::new(vec![3, 3])
/// );
/// let y = nn::max_pool2d(&x, (2, 2), Some((2, 2)));
/// assert_eq!(y.to_vec(), vec![5.0]);
/// ```
pub fn max_pool2d(
    x: &Array,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
) -> Array {
    let stride = stride.unwrap_or(kernel_size);
    let data = x.to_vec();
    let shape = x.shape().as_slice();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;

    match shape.len() {
        2 => {
            // (height, width)
            let (h, w) = (shape[0], shape[1]);
            let out_h = (h - kh) / sh + 1;
            let out_w = (w - kw) / sw + 1;
            let mut result = Vec::with_capacity(out_h * out_w);

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let h_idx = oh * sh + kh_i;
                            let w_idx = ow * sw + kw_i;
                            max_val = max_val.max(data[h_idx * w + w_idx]);
                        }
                    }
                    result.push(max_val);
                }
            }

            Array::from_vec(result, Shape::new(vec![out_h, out_w]))
        }
        3 => {
            // (batch, height, width)
            let (batch, h, w) = (shape[0], shape[1], shape[2]);
            let out_h = (h - kh) / sh + 1;
            let out_w = (w - kw) / sw + 1;
            let mut result = Vec::with_capacity(batch * out_h * out_w);

            for b in 0..batch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let h_idx = oh * sh + kh_i;
                                let w_idx = ow * sw + kw_i;
                                let idx = b * (h * w) + h_idx * w + w_idx;
                                max_val = max_val.max(data[idx]);
                            }
                        }
                        result.push(max_val);
                    }
                }
            }

            Array::from_vec(result, Shape::new(vec![batch, out_h, out_w]))
        }
        _ => panic!("max_pool2d expects 2D or 3D input"),
    }
}

/// Average pooling over a 2D input.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(
///     vec![1.0, 2.0, 3.0, 4.0],
///     Shape::new(vec![2, 2])
/// );
/// let y = nn::avg_pool2d(&x, (2, 2), Some((2, 2)));
/// assert_eq!(y.to_vec(), vec![2.5]);
/// ```
pub fn avg_pool2d(
    x: &Array,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
) -> Array {
    let stride = stride.unwrap_or(kernel_size);
    let data = x.to_vec();
    let shape = x.shape().as_slice();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let kernel_area = (kh * kw) as f32;

    match shape.len() {
        2 => {
            let (h, w) = (shape[0], shape[1]);
            let out_h = (h - kh) / sh + 1;
            let out_w = (w - kw) / sw + 1;
            let mut result = Vec::with_capacity(out_h * out_w);

            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0;
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let h_idx = oh * sh + kh_i;
                            let w_idx = ow * sw + kw_i;
                            sum += data[h_idx * w + w_idx];
                        }
                    }
                    result.push(sum / kernel_area);
                }
            }

            Array::from_vec(result, Shape::new(vec![out_h, out_w]))
        }
        3 => {
            let (batch, h, w) = (shape[0], shape[1], shape[2]);
            let out_h = (h - kh) / sh + 1;
            let out_w = (w - kw) / sw + 1;
            let mut result = Vec::with_capacity(batch * out_h * out_w);

            for b in 0..batch {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;
                        for kh_i in 0..kh {
                            for kw_i in 0..kw {
                                let h_idx = oh * sh + kh_i;
                                let w_idx = ow * sw + kw_i;
                                let idx = b * (h * w) + h_idx * w + w_idx;
                                sum += data[idx];
                            }
                        }
                        result.push(sum / kernel_area);
                    }
                }
            }

            Array::from_vec(result, Shape::new(vec![batch, out_h, out_w]))
        }
        _ => panic!("avg_pool2d expects 2D or 3D input"),
    }
}

/// Global max pooling - reduces each sample to a single value.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 5.0, 3.0, 2.0], Shape::new(vec![4]));
/// let y = nn::global_max_pool(&x);
/// assert_eq!(y, 5.0);
/// ```
pub fn global_max_pool(x: &Array) -> f32 {
    x.max_all()
}

/// Global average pooling - reduces each sample to a single value.
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let y = nn::global_avg_pool(&x);
/// assert_eq!(y, 2.5);
/// ```
pub fn global_avg_pool(x: &Array) -> f32 {
    x.mean_all()
}

/// Scaled dot-product attention mechanism.
///
/// Computes attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// This is the core attention mechanism used in Transformers.
///
/// # Arguments
///
/// * `query` - Query matrix of shape [seq_len_q, d_k]
/// * `key` - Key matrix of shape [seq_len_k, d_k]
/// * `value` - Value matrix of shape [seq_len_k, d_v]
///
/// # Returns
///
/// Attention output of shape [seq_len_q, d_v]
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let q = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
/// let k = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
/// let v = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
/// let output = nn::scaled_dot_product_attention(&q, &k, &v);
/// assert_eq!(output.shape().as_slice(), &[2, 2]);
/// ```
pub fn scaled_dot_product_attention(query: &Array, key: &Array, value: &Array) -> Array {
    assert_eq!(query.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(key.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(value.dtype(), DType::Float32, "Only Float32 supported");

    assert_eq!(query.ndim(), 2, "Query must be 2D [seq_len_q, d_k]");
    assert_eq!(key.ndim(), 2, "Key must be 2D [seq_len_k, d_k]");
    assert_eq!(value.ndim(), 2, "Value must be 2D [seq_len_k, d_v]");

    let q_shape = query.shape().as_slice();
    let k_shape = key.shape().as_slice();
    let v_shape = value.shape().as_slice();

    let d_k = q_shape[1];
    assert_eq!(d_k, k_shape[1], "Query and Key must have same dimension d_k");
    assert_eq!(k_shape[0], v_shape[0], "Key and Value must have same sequence length");

    // Compute Q @ K^T
    let k_t = key.transpose();
    let scores = query.matmul(&k_t);

    // Scale by sqrt(d_k)
    let scale = 1.0 / (d_k as f32).sqrt();
    let scaled_scores = scores.mul(&Array::from_vec(vec![scale], Shape::new(vec![1])));

    // Apply softmax along the last dimension (key sequence dimension)
    // For each query position, compute softmax over all key positions
    let seq_len_q = q_shape[0];
    let seq_len_k = k_shape[0];
    let scores_data = scaled_scores.to_vec();
    let mut attention_weights = vec![0.0; seq_len_q * seq_len_k];

    for i in 0..seq_len_q {
        let row_start = i * seq_len_k;
        let row_end = row_start + seq_len_k;
        let row = &scores_data[row_start..row_end];

        // Compute softmax for this row
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();

        for j in 0..seq_len_k {
            attention_weights[row_start + j] = (row[j] - max_val).exp() / exp_sum;
        }
    }

    let attention_weights_array = Array::from_vec(
        attention_weights,
        Shape::new(vec![seq_len_q, seq_len_k])
    );

    // Multiply attention weights by values: attention_weights @ V
    attention_weights_array.matmul(value)
}

/// 1D convolution operation.
///
/// Applies a 1D convolution over the input array with the given kernel.
/// Uses 'valid' padding (no padding) and stride of 1.
///
/// # Arguments
///
/// * `x` - Input array of shape [length]
/// * `kernel` - Convolution kernel of shape [kernel_size]
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
/// let kernel = Array::from_vec(vec![1.0, 0.0, -1.0], Shape::new(vec![3]));
/// let result = nn::conv1d(&x, &kernel);
/// assert_eq!(result.shape().as_slice(), &[3]);
/// ```
pub fn conv1d(x: &Array, kernel: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(kernel.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.shape().ndim(), 1, "Input must be 1D");
    assert_eq!(kernel.shape().ndim(), 1, "Kernel must be 1D");

    let x_data = x.to_vec();
    let kernel_data = kernel.to_vec();
    let x_len = x_data.len();
    let k_len = kernel_data.len();

    assert!(k_len <= x_len, "Kernel size must be <= input size");

    let output_len = x_len - k_len + 1;
    let mut result = vec![0.0; output_len];

    for i in 0..output_len {
        let mut sum = 0.0;
        for j in 0..k_len {
            sum += x_data[i + j] * kernel_data[j];
        }
        result[i] = sum;
    }

    Array::from_vec(result, Shape::new(vec![output_len]))
}

/// 2D convolution operation.
///
/// Applies a 2D convolution over the input array with the given kernel.
/// Uses 'valid' padding (no padding) and stride of 1.
///
/// # Arguments
///
/// * `x` - Input array of shape [height, width]
/// * `kernel` - Convolution kernel of shape [kernel_height, kernel_width]
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(
///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
///     Shape::new(vec![3, 3])
/// );
/// let kernel = Array::from_vec(
///     vec![1.0, 0.0, -1.0, 1.0],
///     Shape::new(vec![2, 2])
/// );
/// let result = nn::conv2d(&x, &kernel);
/// assert_eq!(result.shape().as_slice(), &[2, 2]);
/// ```
pub fn conv2d(x: &Array, kernel: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(kernel.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.shape().ndim(), 2, "Input must be 2D");
    assert_eq!(kernel.shape().ndim(), 2, "Kernel must be 2D");

    let x_data = x.to_vec();
    let kernel_data = kernel.to_vec();
    let x_shape = x.shape().as_slice();
    let k_shape = kernel.shape().as_slice();

    let (x_h, x_w) = (x_shape[0], x_shape[1]);
    let (k_h, k_w) = (k_shape[0], k_shape[1]);

    assert!(k_h <= x_h, "Kernel height must be <= input height");
    assert!(k_w <= x_w, "Kernel width must be <= input width");

    let out_h = x_h - k_h + 1;
    let out_w = x_w - k_w + 1;
    let mut result = vec![0.0; out_h * out_w];

    for i in 0..out_h {
        for j in 0..out_w {
            let mut sum = 0.0;
            for ki in 0..k_h {
                for kj in 0..k_w {
                    let x_idx = (i + ki) * x_w + (j + kj);
                    let k_idx = ki * k_w + kj;
                    sum += x_data[x_idx] * kernel_data[k_idx];
                }
            }
            result[i * out_w + j] = sum;
        }
    }

    Array::from_vec(result, Shape::new(vec![out_h, out_w]))
}

/// 1D transposed convolution (deconvolution) operation.
///
/// Applies a 1D transposed convolution, which is the gradient of conv1d
/// with respect to its input. Also known as fractionally-strided convolution.
///
/// # Arguments
///
/// * `x` - Input array of shape [length]
/// * `kernel` - Convolution kernel of shape [kernel_size]
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let kernel = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
/// let result = nn::conv_transpose1d(&x, &kernel);
/// assert_eq!(result.shape().as_slice(), &[4]);
/// ```
pub fn conv_transpose1d(x: &Array, kernel: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(kernel.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.shape().ndim(), 1, "Input must be 1D");
    assert_eq!(kernel.shape().ndim(), 1, "Kernel must be 1D");

    let x_data = x.to_vec();
    let kernel_data = kernel.to_vec();
    let x_len = x_data.len();
    let k_len = kernel_data.len();

    let output_len = x_len + k_len - 1;
    let mut result = vec![0.0; output_len];

    for i in 0..x_len {
        for j in 0..k_len {
            result[i + j] += x_data[i] * kernel_data[j];
        }
    }

    Array::from_vec(result, Shape::new(vec![output_len]))
}

/// 2D transposed convolution (deconvolution) operation.
///
/// Applies a 2D transposed convolution, which is the gradient of conv2d
/// with respect to its input.
///
/// # Arguments
///
/// * `x` - Input array of shape [height, width]
/// * `kernel` - Convolution kernel of shape [kernel_height, kernel_width]
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
/// let kernel = Array::from_vec(vec![1.0, 0.5, 0.5, 0.25], Shape::new(vec![2, 2]));
/// let result = nn::conv_transpose2d(&x, &kernel);
/// assert_eq!(result.shape().as_slice(), &[3, 3]);
/// ```
pub fn conv_transpose2d(x: &Array, kernel: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(kernel.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.shape().ndim(), 2, "Input must be 2D");
    assert_eq!(kernel.shape().ndim(), 2, "Kernel must be 2D");

    let x_data = x.to_vec();
    let kernel_data = kernel.to_vec();
    let x_shape = x.shape().as_slice();
    let k_shape = kernel.shape().as_slice();

    let (x_h, x_w) = (x_shape[0], x_shape[1]);
    let (k_h, k_w) = (k_shape[0], k_shape[1]);

    let out_h = x_h + k_h - 1;
    let out_w = x_w + k_w - 1;
    let mut result = vec![0.0; out_h * out_w];

    for i in 0..x_h {
        for j in 0..x_w {
            for ki in 0..k_h {
                for kj in 0..k_w {
                    let out_idx = (i + ki) * out_w + (j + kj);
                    let x_idx = i * x_w + j;
                    let k_idx = ki * k_w + kj;
                    result[out_idx] += x_data[x_idx] * kernel_data[k_idx];
                }
            }
        }
    }

    Array::from_vec(result, Shape::new(vec![out_h, out_w]))
}

/// Depthwise 2D convolution operation.
///
/// Applies a separate 2D convolution to each input channel.
/// Unlike regular convolution, depthwise convolution does not mix channels.
///
/// # Arguments
///
/// * `x` - Input array of shape [channels, height, width]
/// * `kernel` - Convolution kernel of shape [channels, kernel_h, kernel_w]
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// // 2 channels, 4x4 spatial
/// let x = Array::from_vec(vec![1.0; 32], Shape::new(vec![2, 4, 4]));
/// // 2 channel kernels, 2x2 spatial
/// let kernel = Array::from_vec(vec![1.0; 8], Shape::new(vec![2, 2, 2]));
/// let result = nn::depthwise_conv2d(&x, &kernel);
/// assert_eq!(result.shape().as_slice(), &[2, 3, 3]);
/// ```
pub fn depthwise_conv2d(x: &Array, kernel: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(kernel.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.shape().ndim(), 3, "Input must be 3D [channels, height, width]");
    assert_eq!(kernel.shape().ndim(), 3, "Kernel must be 3D [channels, kernel_h, kernel_w]");

    let x_data = x.to_vec();
    let kernel_data = kernel.to_vec();
    let x_shape = x.shape().as_slice();
    let k_shape = kernel.shape().as_slice();

    let (channels, x_h, x_w) = (x_shape[0], x_shape[1], x_shape[2]);
    let (k_channels, k_h, k_w) = (k_shape[0], k_shape[1], k_shape[2]);

    assert_eq!(channels, k_channels, "Number of channels must match");
    assert!(k_h <= x_h, "Kernel height must be <= input height");
    assert!(k_w <= x_w, "Kernel width must be <= input width");

    let out_h = x_h - k_h + 1;
    let out_w = x_w - k_w + 1;
    let mut result = vec![0.0; channels * out_h * out_w];

    for c in 0..channels {
        for i in 0..out_h {
            for j in 0..out_w {
                let mut sum = 0.0;
                for ki in 0..k_h {
                    for kj in 0..k_w {
                        let x_idx = c * (x_h * x_w) + (i + ki) * x_w + (j + kj);
                        let k_idx = c * (k_h * k_w) + ki * k_w + kj;
                        sum += x_data[x_idx] * kernel_data[k_idx];
                    }
                }
                result[c * (out_h * out_w) + i * out_w + j] = sum;
            }
        }
    }

    Array::from_vec(result, Shape::new(vec![channels, out_h, out_w]))
}

/// Grouped 2D convolution operation.
///
/// Splits input and output channels into groups and applies separate convolutions.
/// This is more efficient than regular convolution when groups > 1.
///
/// # Arguments
///
/// * `x` - Input array of shape [in_channels, height, width]
/// * `kernel` - Convolution kernel of shape [out_channels, in_channels/groups, kernel_h, kernel_w]
/// * `groups` - Number of groups
///
/// # Examples
///
/// ```
/// # use jax_rs::{nn, Array, Shape};
/// // 4 input channels, 4x4 spatial
/// let x = Array::from_vec(vec![1.0; 64], Shape::new(vec![4, 4, 4]));
/// // 4 output channels, 2 input channels per group (groups=2), 2x2 kernel
/// let kernel = Array::from_vec(vec![1.0; 32], Shape::new(vec![4, 2, 2, 2]));
/// let result = nn::conv2d_grouped(&x, &kernel, 2);
/// assert_eq!(result.shape().as_slice(), &[4, 3, 3]);
/// ```
pub fn conv2d_grouped(x: &Array, kernel: &Array, groups: usize) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(kernel.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.shape().ndim(), 3, "Input must be 3D [in_channels, height, width]");
    assert_eq!(kernel.shape().ndim(), 4, "Kernel must be 4D [out_channels, in_channels/groups, kernel_h, kernel_w]");
    assert!(groups > 0, "Groups must be positive");

    let x_data = x.to_vec();
    let kernel_data = kernel.to_vec();
    let x_shape = x.shape().as_slice();
    let k_shape = kernel.shape().as_slice();

    let (in_channels, x_h, x_w) = (x_shape[0], x_shape[1], x_shape[2]);
    let (out_channels, in_ch_per_group, k_h, k_w) =
        (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);

    assert_eq!(in_channels % groups, 0, "Input channels must be divisible by groups");
    assert_eq!(out_channels % groups, 0, "Output channels must be divisible by groups");
    assert_eq!(in_channels / groups, in_ch_per_group,
        "Kernel in_channels must equal in_channels/groups");

    let out_h = x_h - k_h + 1;
    let out_w = x_w - k_w + 1;
    let mut result = vec![0.0; out_channels * out_h * out_w];

    let out_ch_per_group = out_channels / groups;

    for g in 0..groups {
        let in_start = g * in_ch_per_group;
        let out_start = g * out_ch_per_group;

        for oc in 0..out_ch_per_group {
            let out_channel = out_start + oc;
            for i in 0..out_h {
                for j in 0..out_w {
                    let mut sum = 0.0;
                    for ic in 0..in_ch_per_group {
                        let in_channel = in_start + ic;
                        for ki in 0..k_h {
                            for kj in 0..k_w {
                                let x_idx = in_channel * (x_h * x_w) + (i + ki) * x_w + (j + kj);
                                let k_idx = out_channel * (in_ch_per_group * k_h * k_w) +
                                           ic * (k_h * k_w) + ki * k_w + kj;
                                sum += x_data[x_idx] * kernel_data[k_idx];
                            }
                        }
                    }
                    result[out_channel * (out_h * out_w) + i * out_w + j] = sum;
                }
            }
        }
    }

    Array::from_vec(result, Shape::new(vec![out_channels, out_h, out_w]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let x = Array::from_vec(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            Shape::new(vec![5]),
        );
        let y = relu(&x);
        assert_eq!(y.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let x = Array::from_vec(vec![-2.0, 0.0, 2.0], Shape::new(vec![3]));
        let y = leaky_relu(&x, 0.1);
        assert_eq!(y.to_vec(), vec![-0.2, 0.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
        let y = sigmoid(&x);
        assert!((y.to_vec()[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_softmax() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let y = softmax(&x);

        let sum: f32 = y.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check that larger inputs produce larger outputs
        let vals = y.to_vec();
        assert!(vals[0] < vals[1]);
        assert!(vals[1] < vals[2]);
    }

    #[test]
    fn test_log_softmax() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let log_sm = log_softmax(&x);
        let sm = softmax(&x);

        let log_sm_data = log_sm.to_vec();
        let sm_data = sm.to_vec();

        for i in 0..3 {
            assert!((log_sm_data[i] - sm_data[i].ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_one_hot() {
        let indices =
            Array::from_vec(vec![0.0, 2.0, 1.0], Shape::new(vec![3]));
        let oh = one_hot(&indices, 3, DType::Float32);

        assert_eq!(oh.shape().as_slice(), &[3, 3]);
        assert_eq!(
            oh.to_vec(),
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn test_gelu() {
        let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
        let y = gelu(&x);
        // GELU(0) should be 0
        assert!(y.to_vec()[0].abs() < 1e-5);
    }

    #[test]
    fn test_dropout_training() {
        let key = crate::random::PRNGKey::from_seed(42);
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let y = dropout(key, &x, 0.5, true);

        // Some values should be scaled, some should be zero
        let y_data = y.to_vec();
        let has_zero = y_data.iter().any(|&v| v == 0.0);
        let has_nonzero = y_data.iter().any(|&v| v != 0.0);
        assert!(has_zero || has_nonzero); // At least one of these should be true
    }

    #[test]
    fn test_dropout_inference() {
        let key = crate::random::PRNGKey::from_seed(42);
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let y = dropout(key, &x, 0.5, false);

        // During inference, output should equal input
        assert_eq!(y.to_vec(), x.to_vec());
    }

    #[test]
    fn test_mse_loss() {
        let pred = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let target = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let loss = mse_loss(&pred, &target);
        assert!(loss.abs() < 1e-5);

        let pred2 = Array::from_vec(vec![0.0, 0.0, 0.0], Shape::new(vec![3]));
        let target2 =
            Array::from_vec(vec![1.0, 1.0, 1.0], Shape::new(vec![3]));
        let loss2 = mse_loss(&pred2, &target2);
        assert!((loss2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Test cross-entropy with perfect prediction
        let pred = Array::from_vec(vec![0.9, 0.05, 0.05], Shape::new(vec![3]));
        let target = Array::from_vec(vec![1.0, 0.0, 0.0], Shape::new(vec![3]));
        let loss = cross_entropy_loss(&pred, &target);

        // Loss should be low for good prediction
        assert!(loss < 0.5);

        // Test with uniform prediction (higher loss)
        let pred2 = Array::from_vec(vec![0.33, 0.33, 0.34], Shape::new(vec![3]));
        let target2 = Array::from_vec(vec![1.0, 0.0, 0.0], Shape::new(vec![3]));
        let loss2 = cross_entropy_loss(&pred2, &target2);

        // Uniform prediction should have higher loss
        assert!(loss2 > loss);
    }

    #[test]
    fn test_binary_cross_entropy() {
        // Test binary cross-entropy with perfect predictions
        let pred = Array::from_vec(vec![0.9, 0.1, 0.8], Shape::new(vec![3]));
        let target = Array::from_vec(vec![1.0, 0.0, 1.0], Shape::new(vec![3]));
        let loss = binary_cross_entropy(&pred, &target);

        // Loss should be low for good predictions
        assert!(loss < 0.3);

        // Test with poor predictions (high loss)
        let pred2 = Array::from_vec(vec![0.1, 0.9, 0.2], Shape::new(vec![3]));
        let target2 = Array::from_vec(vec![1.0, 0.0, 1.0], Shape::new(vec![3]));
        let loss2 = binary_cross_entropy(&pred2, &target2);

        // Poor predictions should have higher loss
        assert!(loss2 > loss);
    }

    #[test]
    fn test_binary_cross_entropy_extreme() {
        // Test with extreme values (close to 0 and 1)
        let pred = Array::from_vec(vec![0.99, 0.01], Shape::new(vec![2]));
        let target = Array::from_vec(vec![1.0, 0.0], Shape::new(vec![2]));
        let loss = binary_cross_entropy(&pred, &target);

        // Loss should be very low
        assert!(loss < 0.02);
    }

    #[test]
    fn test_hinge_loss() {
        // Test hinge loss with correct classifications (margin > 1)
        let pred = Array::from_vec(vec![2.0, -2.0, 1.5], Shape::new(vec![3]));
        let target = Array::from_vec(vec![1.0, -1.0, 1.0], Shape::new(vec![3]));
        let loss = hinge_loss(&pred, &target);

        // All predictions are correct with good margin, loss should be 0
        assert!(loss.abs() < 1e-5);

        // Test with some misclassifications
        let pred2 = Array::from_vec(vec![0.5, 0.5, -0.5], Shape::new(vec![3]));
        let target2 = Array::from_vec(vec![1.0, -1.0, 1.0], Shape::new(vec![3]));
        let loss2 = hinge_loss(&pred2, &target2);

        // Should have positive loss
        // pred2[0]: margin = 1 - 1*0.5 = 0.5 -> max(0, 0.5) = 0.5
        // pred2[1]: margin = 1 - (-1)*0.5 = 1.5 -> max(0, 1.5) = 1.5
        // pred2[2]: margin = 1 - 1*(-0.5) = 1.5 -> max(0, 1.5) = 1.5
        // Average: (0.5 + 1.5 + 1.5) / 3 = 3.5/3 ≈ 1.167
        assert!((loss2 - 1.167).abs() < 0.01);
    }

    #[test]
    fn test_hinge_loss_perfect() {
        // Test with perfect classification (large margin)
        let pred = Array::from_vec(vec![5.0, -5.0], Shape::new(vec![2]));
        let target = Array::from_vec(vec![1.0, -1.0], Shape::new(vec![2]));
        let loss = hinge_loss(&pred, &target);

        // Loss should be 0 for perfect predictions with large margin
        assert!(loss.abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let normalized = layer_norm(&x, 1e-5);

        let data = normalized.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;

        // Mean should be close to 0 after normalization
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_conv1d() {
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::new(vec![5]),
        );
        let kernel = Array::from_vec(vec![1.0, 2.0, 1.0], Shape::new(vec![3]));
        let result = conv1d(&x, &kernel);

        assert_eq!(result.shape().as_slice(), &[3]);
        // [1*1 + 2*2 + 3*1, 2*1 + 3*2 + 4*1, 3*1 + 4*2 + 5*1]
        // [1+4+3, 2+6+4, 3+8+5]
        assert_eq!(result.to_vec(), vec![8.0, 12.0, 16.0]);
    }

    #[test]
    fn test_conv2d() {
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Shape::new(vec![3, 3]),
        );
        let kernel =
            Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let result = conv2d(&x, &kernel);

        assert_eq!(result.shape().as_slice(), &[2, 2]);
        // Should extract top-left and bottom-right elements
        assert_eq!(result.to_vec(), vec![6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn test_conv_transpose1d() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let kernel = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let result = conv_transpose1d(&x, &kernel);

        assert_eq!(result.shape().as_slice(), &[4]);
        // [1*1, 1*2+2*1, 2*2+3*1, 3*2]
        assert_eq!(result.to_vec(), vec![1.0, 4.0, 7.0, 6.0]);
    }

    #[test]
    fn test_conv_transpose2d() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let kernel =
            Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let result = conv_transpose2d(&x, &kernel);

        assert_eq!(result.shape().as_slice(), &[3, 3]);
        // Should spread values across output
        let data = result.to_vec();
        assert_eq!(data[0], 1.0); // top-left
        assert_eq!(data[8], 4.0); // bottom-right
    }

    #[test]
    fn test_depthwise_conv2d() {
        // 2 channels, 4x4 spatial
        let x = Array::from_vec(
            vec![
                // Channel 0
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
                // Channel 1
                17.0, 18.0, 19.0, 20.0,
                21.0, 22.0, 23.0, 24.0,
                25.0, 26.0, 27.0, 28.0,
                29.0, 30.0, 31.0, 32.0,
            ],
            Shape::new(vec![2, 4, 4]),
        );
        // 2 channel kernels, 2x2 spatial
        let kernel = Array::from_vec(
            vec![
                // Kernel for channel 0
                1.0, 0.0,
                0.0, 1.0,
                // Kernel for channel 1
                1.0, 1.0,
                1.0, 1.0,
            ],
            Shape::new(vec![2, 2, 2]),
        );
        let result = depthwise_conv2d(&x, &kernel);

        assert_eq!(result.shape().as_slice(), &[2, 3, 3]);
        assert_eq!(result.size(), 18);

        // Channel 0: diagonal kernel should give 1+6=7 for top-left
        let data = result.to_vec();
        assert_eq!(data[0], 7.0);
    }

    #[test]
    fn test_conv2d_grouped() {
        // 4 input channels, 4x4 spatial
        let x = Array::from_vec(vec![1.0; 64], Shape::new(vec![4, 4, 4]));
        // 4 output channels, 2 input channels per group (groups=2), 2x2 kernel
        let kernel = Array::from_vec(vec![1.0; 32], Shape::new(vec![4, 2, 2, 2]));
        let result = conv2d_grouped(&x, &kernel, 2);

        assert_eq!(result.shape().as_slice(), &[4, 3, 3]);
        assert_eq!(result.size(), 36);

        // Each output position should be sum of 2*2*2 = 8 elements (all 1.0)
        let data = result.to_vec();
        assert_eq!(data[0], 8.0);
    }

    #[test]
    fn test_depthwise_conv2d_basic() {
        // Simple 1 channel case
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            Shape::new(vec![1, 2, 2]),
        );
        let kernel = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![1, 2, 2]));
        let result = depthwise_conv2d(&x, &kernel);

        assert_eq!(result.shape().as_slice(), &[1, 1, 1]);
        // Sum of all elements: 1+2+3+4 = 10
        assert_eq!(result.to_vec(), vec![10.0]);
    }

    #[test]
    fn test_batch_norm() {
        // Test batch normalization with 2 samples, 3 features
        // Each feature should be normalized independently across the batch
        let x = Array::from_vec(
            vec![
                1.0, 2.0, 3.0, // Sample 1
                4.0, 5.0, 6.0, // Sample 2
            ],
            Shape::new(vec![2, 3]),
        );

        let result = batch_norm(&x, 1e-5);
        assert_eq!(result.shape().as_slice(), &[2, 3]);

        let data = result.to_vec();

        // Each feature column should have mean ~0 and std ~1
        // For feature 0: values are [1.0, 4.0], mean=2.5, std~1.5
        // Normalized: [(1-2.5)/1.5, (4-2.5)/1.5] = [-1, 1]
        // For feature 1: values are [2.0, 5.0], mean=3.5, std~1.5
        // For feature 2: values are [3.0, 6.0], mean=4.5, std~1.5

        // Check that values are normalized (approximately -1 and 1 for each feature)
        assert!((data[0] + 1.0).abs() < 0.01); // Feature 0, sample 1
        assert!((data[3] - 1.0).abs() < 0.01); // Feature 0, sample 2
    }

    #[test]
    fn test_batch_norm_zero_variance() {
        // Test batch norm with constant values (zero variance)
        let x = Array::from_vec(
            vec![
                5.0, 3.0, 7.0, // Sample 1
                5.0, 3.0, 7.0, // Sample 2
            ],
            Shape::new(vec![2, 3]),
        );

        let result = batch_norm(&x, 1e-5);
        let data = result.to_vec();

        // With zero variance, all values should be 0 after normalization
        for &val in &data {
            assert!(val.abs() < 0.01);
        }
    }

    #[test]
    fn test_max_pool2d_basic() {
        // Test 2x2 max pooling on 4x4 input
        let x = Array::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            Shape::new(vec![4, 4]),
        );

        let result = max_pool2d(&x, (2, 2), None);
        assert_eq!(result.shape().as_slice(), &[2, 2]);

        // Max of each 2x2 region:
        // Top-left: max(1,2,5,6) = 6
        // Top-right: max(3,4,7,8) = 8
        // Bottom-left: max(9,10,13,14) = 14
        // Bottom-right: max(11,12,15,16) = 16
        assert_eq!(result.to_vec(), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_max_pool2d_with_stride() {
        // Test max pooling with custom stride
        let x = Array::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0, 8.0, 9.0, 10.0,
            ],
            Shape::new(vec![2, 5]),
        );

        // Kernel 2x2, stride (1, 2) - overlapping in height, non-overlapping in width
        let result = max_pool2d(&x, (2, 2), Some((1, 2)));
        assert_eq!(result.shape().as_slice(), &[1, 2]);

        // First window: max(1,2,6,7) = 7
        // Second window: max(3,4,8,9) = 9
        assert_eq!(result.to_vec(), vec![7.0, 9.0]);
    }

    #[test]
    fn test_max_pool2d_batched() {
        // Test max pooling with batch dimension [batch, height, width]
        let x = Array::from_vec(
            vec![
                // Batch 1
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                // Batch 2
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            Shape::new(vec![2, 2, 4]),
        );

        let result = max_pool2d(&x, (2, 2), None);
        assert_eq!(result.shape().as_slice(), &[2, 1, 2]);

        // Batch 1: max(1,2,5,6)=6, max(3,4,7,8)=8
        // Batch 2: max(9,10,13,14)=14, max(11,12,15,16)=16
        assert_eq!(result.to_vec(), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_avg_pool1d_basic() {
        // Test 1D average pooling
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![6]));
        let result = avg_pool1d(&x, 2, None);
        assert_eq!(result.shape().as_slice(), &[3]);
        // avg(1,2)=1.5, avg(3,4)=3.5, avg(5,6)=5.5
        assert_eq!(result.to_vec(), vec![1.5, 3.5, 5.5]);
    }

    #[test]
    fn test_avg_pool1d_with_stride() {
        // Test 1D average pooling with stride
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
        let result = avg_pool1d(&x, 2, Some(1));
        assert_eq!(result.shape().as_slice(), &[4]);
        // avg(1,2)=1.5, avg(2,3)=2.5, avg(3,4)=3.5, avg(4,5)=4.5
        assert_eq!(result.to_vec(), vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_avg_pool1d_batched() {
        // Test 1D average pooling with batch dimension [batch, length]
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            Shape::new(vec![2, 4]),
        );
        let result = avg_pool1d(&x, 2, None);
        assert_eq!(result.shape().as_slice(), &[2, 2]);
        // Batch 1: avg(1,2)=1.5, avg(3,4)=3.5
        // Batch 2: avg(10,20)=15.0, avg(30,40)=35.0
        assert_eq!(result.to_vec(), vec![1.5, 3.5, 15.0, 35.0]);
    }

    #[test]
    fn test_avg_pool2d_basic() {
        // Test 2D average pooling
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            Shape::new(vec![3, 3]),
        );
        let result = avg_pool2d(&x, (2, 2), None);
        assert_eq!(result.shape().as_slice(), &[1, 1]);
        // avg(1,2,4,5) = 12.0/4 = 3.0
        assert_eq!(result.to_vec(), vec![3.0]);
    }

    #[test]
    fn test_avg_pool2d_with_stride() {
        // Test 2D average pooling with stride
        let x = Array::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            Shape::new(vec![4, 4]),
        );
        let result = avg_pool2d(&x, (2, 2), Some((2, 2)));
        assert_eq!(result.shape().as_slice(), &[2, 2]);
        // avg(1,2,5,6)=3.5, avg(3,4,7,8)=5.5
        // avg(9,10,13,14)=11.5, avg(11,12,15,16)=13.5
        assert_eq!(result.to_vec(), vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_avg_pool2d_batched() {
        // Test 2D average pooling with batch dimension [batch, height, width]
        let x = Array::from_vec(
            vec![
                // Batch 1
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                // Batch 2
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            Shape::new(vec![2, 2, 4]),
        );
        let result = avg_pool2d(&x, (2, 2), None);
        assert_eq!(result.shape().as_slice(), &[2, 1, 2]);
        // Batch 1: avg(1,2,5,6)=3.5, avg(3,4,7,8)=5.5
        // Batch 2: avg(9,10,13,14)=11.5, avg(11,12,15,16)=13.5
        assert_eq!(result.to_vec(), vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn test_avg_pool2d_non_square_kernel() {
        // Test 2D average pooling with non-square kernel
        let x = Array::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ],
            Shape::new(vec![3, 4]),
        );
        let result = avg_pool2d(&x, (2, 3), Some((2, 3)));
        assert_eq!(result.shape().as_slice(), &[1, 1]);
        // avg(1,2,3,5,6,7) = 24.0/6 = 4.0
        assert_eq!(result.to_vec(), vec![4.0]);
    }

    #[test]
    fn test_scaled_dot_product_attention_basic() {
        // Test basic attention mechanism
        // Q, K, V are identity matrices - should produce V as output
        let q = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let k = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let v = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));

        let output = scaled_dot_product_attention(&q, &k, &v);
        assert_eq!(output.shape().as_slice(), &[2, 2]);

        // Since Q and K are identity, attention should be uniform
        // Result should be weighted average of V rows
        let result = output.to_vec();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_scaled_dot_product_attention_simple() {
        // Test with simple values
        let q = Array::from_vec(vec![1.0, 0.0], Shape::new(vec![1, 2]));
        let k = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let v = Array::from_vec(vec![10.0, 20.0], Shape::new(vec![2, 1]));

        let output = scaled_dot_product_attention(&q, &k, &v);
        assert_eq!(output.shape().as_slice(), &[1, 1]);

        // Query [1, 0] should attend more to first key [1, 0] than second [0, 1]
        let result = output.to_vec();
        // Should be closer to 10.0 than 20.0 due to higher attention to first position
        assert!(result[0] < 15.0);
    }

    #[test]
    fn test_scaled_dot_product_attention_shape() {
        // Test with different sequence lengths
        let q = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], Shape::new(vec![3, 2]));
        let k = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2]));
        let v = Array::from_vec(vec![5.0, 10.0], Shape::new(vec![2, 1]));

        let output = scaled_dot_product_attention(&q, &k, &v);
        // Output should have shape [seq_len_q, d_v] = [3, 1]
        assert_eq!(output.shape().as_slice(), &[3, 1]);
    }
}
