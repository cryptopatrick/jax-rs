//! End-to-end example: Train a simple MLP on synthetic data.
//!
//! This example demonstrates:
//! - Creating a multi-layer perceptron (MLP) network
//! - Generating synthetic training data
//! - Computing gradients with finite differences
//! - Training with stochastic gradient descent (using optim module)
//! - Monitoring loss during training

use jax_rs::{nn, optim, random, Array, DType, Shape};

/// Multi-layer perceptron forward pass.
///
/// Architecture: input -> linear -> ReLU -> linear -> output
///
/// # Arguments
/// * `params` - [W1, b1, W2, b2] where W1: [in, hidden], b1: [hidden], W2: [hidden, out], b2: [out]
/// * `x` - Input data of shape [batch, in_features]
fn mlp(params: &[Array], x: &Array) -> Array {
    // First layer: x @ W1 + b1
    let mut h = x.matmul(&params[0]);

    // Broadcast bias addition (works for any batch size)
    let b1_expanded = params[1].broadcast_to(h.shape().clone());
    h = h.add(&b1_expanded);

    // Activation
    h = nn::relu(&h);

    // Second layer: h @ W2 + b2
    let mut out = h.matmul(&params[2]);
    let b2_expanded = params[3].broadcast_to(out.shape().clone());
    out = out.add(&b2_expanded);

    out
}

/// Mean squared error loss function.
///
/// # Arguments
/// * `params` - Model parameters
/// * `x` - Input data
/// * `y` - Target labels
fn mse_loss(params: &[Array], x: &Array, y: &Array) -> f32 {
    let pred = mlp(params, x);
    let diff = pred.sub(y);
    let squared = diff.mul(&diff);
    squared.mean_all()
}

fn main() {
    println!("=== End-to-End MLP Training Example ===\n");

    let key = random::PRNGKey::from_seed(42);

    // Generate synthetic data
    println!("Generating synthetic data...");
    let n_samples = 100;
    let n_features = 10;
    let n_hidden = 64;
    let n_outputs = 1;

    let x = random::normal(key, Shape::new(vec![n_samples, n_features]), DType::Float32);
    let y = random::normal(key, Shape::new(vec![n_samples, n_outputs]), DType::Float32);

    // Initialize parameters with Xavier/He initialization
    println!("Initializing model parameters...");
    let scale1 = (2.0 / n_features as f32).sqrt();
    let w1_data = random::normal(key, Shape::new(vec![n_features, n_hidden]), DType::Float32)
        .mul(&Array::from_vec(vec![scale1], Shape::new(vec![1])));
    let b1 = Array::zeros(Shape::new(vec![n_hidden]), DType::Float32);

    let scale2 = (2.0 / n_hidden as f32).sqrt();
    let w2_data = random::normal(key, Shape::new(vec![n_hidden, n_outputs]), DType::Float32)
        .mul(&Array::from_vec(vec![scale2], Shape::new(vec![1])));
    let b2 = Array::zeros(Shape::new(vec![n_outputs]), DType::Float32);

    let mut params = vec![w1_data, b1, w2_data, b2];

    // Initialize optimizer states
    let mut states: Vec<optim::SGDState> = params.iter().map(|p| optim::sgd_init(p)).collect();

    // Training hyperparameters
    let learning_rate = 0.01;
    let momentum = 0.9;
    let n_epochs = 100;

    println!("Training MLP for {} epochs...\n", n_epochs);
    println!("Architecture: {} -> {} (ReLU) -> {}", n_features, n_hidden, n_outputs);
    println!("Optimizer: SGD with momentum={}", momentum);
    println!("Learning rate: {}\n", learning_rate);

    // Training loop
    for epoch in 0..n_epochs {
        // Compute gradients using finite differences
        // Note: This is for demonstration - production code would use automatic differentiation
        let current_loss = mse_loss(&params, &x, &y);

        // Numerical gradients with finite differences
        let epsilon = 1e-4;
        let mut grads = Vec::new();

        for (i, param) in params.iter().enumerate() {
            let param_data = param.to_vec();
            let mut grad_data = vec![0.0; param_data.len()];

            for j in 0..param_data.len() {
                // Perturb parameter
                let mut perturbed_data = param_data.clone();
                perturbed_data[j] += epsilon;
                let perturbed_param = Array::from_vec(perturbed_data, param.shape().clone());

                // Update params temporarily
                let mut params_plus = params.clone();
                params_plus[i] = perturbed_param;

                // Compute loss with perturbed parameter
                let loss_plus = mse_loss(&params_plus, &x, &y);

                // Gradient approximation
                grad_data[j] = (loss_plus - current_loss) / epsilon;
            }

            grads.push(Array::from_vec(grad_data, param.shape().clone()));
        }

        // Update parameters using SGD optimizer
        for i in 0..params.len() {
            let (new_param, new_state) = optim::sgd_update(
                learning_rate,
                momentum,
                &params[i],
                &grads[i],
                &states[i],
            );
            params[i] = new_param;
            states[i] = new_state;
        }

        // Log progress
        if epoch % 10 == 0 {
            let loss = mse_loss(&params, &x, &y);
            println!("Epoch {:3}: loss = {:.6}", epoch, loss);
        }
    }

    // Final evaluation
    let final_loss = mse_loss(&params, &x, &y);
    println!("\n=== Training Complete ===");
    println!("Final loss: {:.6}", final_loss);
    println!("\nNote: This example uses numerical gradients for demonstration.");
    println!("For production, use automatic differentiation with the grad() function.");
}
