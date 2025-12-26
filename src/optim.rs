//! Optimization algorithms for training neural networks.
//!
//! This module provides common optimizers with state management:
//! - SGD (Stochastic Gradient Descent) with momentum
//! - Adam (Adaptive Moment Estimation)
//! - RMSProp (Root Mean Square Propagation)

use crate::{Array, DType, Shape};

/// SGD optimizer state containing momentum buffers.
#[derive(Debug, Clone)]
pub struct SGDState {
    /// Momentum buffers for each parameter
    pub momentum: Option<Array>,
    /// Current step number
    pub step: usize,
}

/// SGD (Stochastic Gradient Descent) optimizer with optional momentum.
///
/// Updates parameters using: `params = params - learning_rate * gradients`
/// With momentum: `velocity = momentum * velocity + gradients; params = params - lr * velocity`
///
/// # Arguments
///
/// * `learning_rate` - Learning rate (step size)
/// * `momentum` - Momentum factor (0.0 to 1.0). Set to 0.0 for standard SGD.
///
/// # Examples
///
/// ```
/// # use jax_rs::{optim, Array, Shape};
/// let params = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let grads = Array::from_vec(vec![0.1, 0.2, 0.3], Shape::new(vec![3]));
///
/// let mut state = optim::sgd_init(&params);
/// let (new_params, new_state) = optim::sgd_update(
///     0.01,      // learning_rate
///     0.9,       // momentum
///     &params,
///     &grads,
///     &state
/// );
/// ```
pub fn sgd_init(params: &Array) -> SGDState {
    SGDState {
        momentum: None,
        step: 0,
    }
}

/// Update parameters using SGD optimizer.
pub fn sgd_update(
    learning_rate: f32,
    momentum: f32,
    params: &Array,
    grads: &Array,
    state: &SGDState,
) -> (Array, SGDState) {
    assert_eq!(params.shape(), grads.shape(), "Params and grads must have same shape");

    let mut new_state = state.clone();
    new_state.step += 1;

    let new_params = if momentum == 0.0 {
        // Standard SGD without momentum
        let update = grads.mul(&Array::from_vec(vec![learning_rate], Shape::new(vec![1])));
        params.sub(&update)
    } else {
        // SGD with momentum
        let velocity = if let Some(ref prev_velocity) = state.momentum {
            // velocity = momentum * prev_velocity + grads
            let scaled_velocity = prev_velocity.mul(&Array::from_vec(vec![momentum], Shape::new(vec![1])));
            scaled_velocity.add(grads)
        } else {
            // First iteration: velocity = grads
            grads.clone()
        };

        let update = velocity.mul(&Array::from_vec(vec![learning_rate], Shape::new(vec![1])));
        new_state.momentum = Some(velocity);
        params.sub(&update)
    };

    (new_params, new_state)
}

/// Adam optimizer state containing moment estimates.
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment estimate (mean of gradients)
    pub m: Option<Array>,
    /// Second moment estimate (uncentered variance of gradients)
    pub v: Option<Array>,
    /// Current step number
    pub step: usize,
}

/// Adam (Adaptive Moment Estimation) optimizer.
///
/// Computes adaptive learning rates for each parameter using estimates of
/// first and second moments of the gradients.
///
/// # Arguments
///
/// * `learning_rate` - Learning rate (default: 0.001)
/// * `beta1` - Exponential decay rate for first moment (default: 0.9)
/// * `beta2` - Exponential decay rate for second moment (default: 0.999)
/// * `epsilon` - Small constant for numerical stability (default: 1e-8)
///
/// # Examples
///
/// ```
/// # use jax_rs::{optim, Array, Shape};
/// let params = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let grads = Array::from_vec(vec![0.1, 0.2, 0.3], Shape::new(vec![3]));
///
/// let mut state = optim::adam_init(&params);
/// let (new_params, new_state) = optim::adam_update(
///     0.001,  // learning_rate
///     0.9,    // beta1
///     0.999,  // beta2
///     1e-8,   // epsilon
///     &params,
///     &grads,
///     &state
/// );
/// ```
pub fn adam_init(params: &Array) -> AdamState {
    AdamState {
        m: None,
        v: None,
        step: 0,
    }
}

/// Update parameters using Adam optimizer.
pub fn adam_update(
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    params: &Array,
    grads: &Array,
    state: &AdamState,
) -> (Array, AdamState) {
    assert_eq!(params.shape(), grads.shape(), "Params and grads must have same shape");

    let mut new_state = state.clone();
    new_state.step += 1;
    let t = new_state.step as f32;

    // Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * grads
    let m = if let Some(ref prev_m) = state.m {
        let scaled_m = prev_m.mul(&Array::from_vec(vec![beta1], Shape::new(vec![1])));
        let scaled_grads = grads.mul(&Array::from_vec(vec![1.0 - beta1], Shape::new(vec![1])));
        scaled_m.add(&scaled_grads)
    } else {
        grads.mul(&Array::from_vec(vec![1.0 - beta1], Shape::new(vec![1])))
    };

    // Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * grads^2
    let grads_squared = grads.mul(grads);
    let v = if let Some(ref prev_v) = state.v {
        let scaled_v = prev_v.mul(&Array::from_vec(vec![beta2], Shape::new(vec![1])));
        let scaled_grads_sq = grads_squared.mul(&Array::from_vec(vec![1.0 - beta2], Shape::new(vec![1])));
        scaled_v.add(&scaled_grads_sq)
    } else {
        grads_squared.mul(&Array::from_vec(vec![1.0 - beta2], Shape::new(vec![1])))
    };

    // Compute bias-corrected first moment: m_hat = m_t / (1 - beta1^t)
    let bias_correction1 = 1.0 - beta1.powf(t);
    let m_hat = m.mul(&Array::from_vec(vec![1.0 / bias_correction1], Shape::new(vec![1])));

    // Compute bias-corrected second moment: v_hat = v_t / (1 - beta2^t)
    let bias_correction2 = 1.0 - beta2.powf(t);
    let v_hat = v.mul(&Array::from_vec(vec![1.0 / bias_correction2], Shape::new(vec![1])));

    // Update parameters: params = params - lr * m_hat / (sqrt(v_hat) + epsilon)
    let v_hat_sqrt = v_hat.sqrt();
    let v_hat_sqrt_eps = v_hat_sqrt.add(&Array::from_vec(vec![epsilon], Shape::new(vec![1])));
    let update = m_hat.div(&v_hat_sqrt_eps).mul(&Array::from_vec(vec![learning_rate], Shape::new(vec![1])));

    new_state.m = Some(m);
    new_state.v = Some(v);

    (params.sub(&update), new_state)
}

/// RMSProp optimizer state containing moving average of squared gradients.
#[derive(Debug, Clone)]
pub struct RMSPropState {
    /// Moving average of squared gradients
    pub square_avg: Option<Array>,
    /// Current step number
    pub step: usize,
}

/// RMSProp (Root Mean Square Propagation) optimizer.
///
/// Uses a moving average of squared gradients to normalize the gradient.
/// Good for handling non-stationary objectives.
///
/// # Arguments
///
/// * `learning_rate` - Learning rate (default: 0.01)
/// * `alpha` - Smoothing constant (default: 0.99)
/// * `epsilon` - Small constant for numerical stability (default: 1e-8)
///
/// # Examples
///
/// ```
/// # use jax_rs::{optim, Array, Shape};
/// let params = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let grads = Array::from_vec(vec![0.1, 0.2, 0.3], Shape::new(vec![3]));
///
/// let mut state = optim::rmsprop_init(&params);
/// let (new_params, new_state) = optim::rmsprop_update(
///     0.01,   // learning_rate
///     0.99,   // alpha
///     1e-8,   // epsilon
///     &params,
///     &grads,
///     &state
/// );
/// ```
pub fn rmsprop_init(params: &Array) -> RMSPropState {
    RMSPropState {
        square_avg: None,
        step: 0,
    }
}

/// Update parameters using RMSProp optimizer.
pub fn rmsprop_update(
    learning_rate: f32,
    alpha: f32,
    epsilon: f32,
    params: &Array,
    grads: &Array,
    state: &RMSPropState,
) -> (Array, RMSPropState) {
    assert_eq!(params.shape(), grads.shape(), "Params and grads must have same shape");

    let mut new_state = state.clone();
    new_state.step += 1;

    // Update moving average of squared gradients: square_avg = alpha * square_avg + (1 - alpha) * grads^2
    let grads_squared = grads.mul(grads);
    let square_avg = if let Some(ref prev_sq_avg) = state.square_avg {
        let scaled_sq_avg = prev_sq_avg.mul(&Array::from_vec(vec![alpha], Shape::new(vec![1])));
        let scaled_grads_sq = grads_squared.mul(&Array::from_vec(vec![1.0 - alpha], Shape::new(vec![1])));
        scaled_sq_avg.add(&scaled_grads_sq)
    } else {
        grads_squared.mul(&Array::from_vec(vec![1.0 - alpha], Shape::new(vec![1])))
    };

    // Update parameters: params = params - lr * grads / (sqrt(square_avg) + epsilon)
    let avg_sqrt = square_avg.sqrt();
    let avg_sqrt_eps = avg_sqrt.add(&Array::from_vec(vec![epsilon], Shape::new(vec![1])));
    let update = grads.div(&avg_sqrt_eps).mul(&Array::from_vec(vec![learning_rate], Shape::new(vec![1])));

    new_state.square_avg = Some(square_avg);

    (params.sub(&update), new_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_no_momentum() {
        let params = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let grads = Array::from_vec(vec![0.1, 0.2, 0.3], Shape::new(vec![3]));

        let state = sgd_init(&params);
        let (new_params, new_state) = sgd_update(0.1, 0.0, &params, &grads, &state);

        // new_params = params - 0.1 * grads
        let expected = vec![0.99, 1.98, 2.97];
        let result = new_params.to_vec();
        for i in 0..3 {
            assert!((result[i] - expected[i]).abs() < 1e-5);
        }

        assert_eq!(new_state.step, 1);
        assert!(new_state.momentum.is_none());
    }

    #[test]
    fn test_sgd_with_momentum() {
        let params = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let grads = Array::from_vec(vec![0.1, 0.2, 0.3], Shape::new(vec![3]));

        let state = sgd_init(&params);
        let (new_params, new_state) = sgd_update(0.1, 0.9, &params, &grads, &state);

        // First step: velocity = grads, new_params = params - 0.1 * grads
        let expected = vec![0.99, 1.98, 2.97];
        let result = new_params.to_vec();
        for i in 0..3 {
            assert!((result[i] - expected[i]).abs() < 1e-5);
        }

        assert!(new_state.momentum.is_some());

        // Second step with momentum
        let (new_params2, new_state2) = sgd_update(0.1, 0.9, &new_params, &grads, &new_state);

        // velocity = 0.9 * [0.1, 0.2, 0.3] + [0.1, 0.2, 0.3] = [0.19, 0.38, 0.57]
        // new_params = [0.99, 1.98, 2.97] - 0.1 * [0.19, 0.38, 0.57]
        let expected2 = vec![0.971, 1.942, 2.913];
        let result2 = new_params2.to_vec();
        for i in 0..3 {
            assert!((result2[i] - expected2[i]).abs() < 1e-4);
        }

        assert_eq!(new_state2.step, 2);
    }

    #[test]
    fn test_adam_basic() {
        let params = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let grads = Array::from_vec(vec![0.1, 0.2], Shape::new(vec![2]));

        let state = adam_init(&params);
        let (new_params, new_state) = adam_update(0.001, 0.9, 0.999, 1e-8, &params, &grads, &state);

        // After one step, parameters should have moved in the direction opposite to gradients
        let result = new_params.to_vec();
        assert!(result[0] < params.to_vec()[0]); // Decreased
        assert!(result[1] < params.to_vec()[1]); // Decreased

        assert!(new_state.m.is_some());
        assert!(new_state.v.is_some());
        assert_eq!(new_state.step, 1);
    }

    #[test]
    fn test_adam_convergence() {
        let mut params = Array::from_vec(vec![10.0], Shape::new(vec![1]));
        let mut state = adam_init(&params);

        // Gradient always points away from zero, should converge towards zero
        for _ in 0..100 {
            let grads = params.clone(); // Gradient = current value
            let (new_params, new_state) = adam_update(0.1, 0.9, 0.999, 1e-8, &params, &grads, &state);
            params = new_params;
            state = new_state;
        }

        // Should have moved significantly towards zero
        let final_val = params.to_vec()[0];
        assert!(final_val.abs() < 5.0); // Started at 10.0, should be closer to 0
    }

    #[test]
    fn test_rmsprop_basic() {
        let params = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let grads = Array::from_vec(vec![0.1, 0.2], Shape::new(vec![2]));

        let state = rmsprop_init(&params);
        let (new_params, new_state) = rmsprop_update(0.01, 0.99, 1e-8, &params, &grads, &state);

        // Parameters should have moved in the direction opposite to gradients
        let result = new_params.to_vec();
        assert!(result[0] < params.to_vec()[0]); // Decreased
        assert!(result[1] < params.to_vec()[1]); // Decreased

        assert!(new_state.square_avg.is_some());
        assert_eq!(new_state.step, 1);
    }

    #[test]
    fn test_rmsprop_adapts_to_gradient_magnitude() {
        let params = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));
        // First gradient is much larger than second
        let grads = Array::from_vec(vec![1.0, 0.1], Shape::new(vec![2]));

        let state = rmsprop_init(&params);
        let (new_params, new_state) = rmsprop_update(0.1, 0.99, 1e-8, &params, &grads, &state);

        // Second step with same gradients
        let (new_params2, _) = rmsprop_update(0.1, 0.99, 1e-8, &new_params, &grads, &new_state);

        let result = new_params2.to_vec();
        let param_change_0 = params.to_vec()[0] - result[0];
        let param_change_1 = params.to_vec()[1] - result[1];

        // Despite gradient[0] being 10x larger, the parameter changes should be more similar
        // because RMSProp normalizes by the running average of squared gradients
        let ratio = param_change_0 / param_change_1;
        assert!(ratio < 5.0); // Much less than the 10x gradient ratio
    }
}
