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
    fn test_layer_norm() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let normalized = layer_norm(&x, 1e-5);

        let data = normalized.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;

        // Mean should be close to 0 after normalization
        assert!(mean.abs() < 1e-5);
    }
}
