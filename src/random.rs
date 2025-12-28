//! Random number generation with reproducible PRNG keys.
//!
//! This module provides JAX-style random number generation using explicit
//! PRNG keys for reproducibility.

use crate::{Array, DType, Shape};
use crate::buffer::Buffer;
use crate::Device;

/// PRNG key for reproducible random number generation.
///
/// Unlike traditional stateful RNGs, JAX-style keys are explicit values
/// that must be split to generate independent random streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PRNGKey {
    /// Internal state (simplified version)
    state: [u64; 2],
}

impl PRNGKey {
    /// Create a new PRNG key from a seed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::random::PRNGKey;
    /// let key = PRNGKey::from_seed(42);
    /// ```
    pub fn from_seed(seed: u64) -> Self {
        // Simple initialization - in production would use a proper hash
        Self { state: [seed, seed.wrapping_mul(0x9e3779b97f4a7c15)] }
    }

    /// Split a key into two independent keys.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::random::PRNGKey;
    /// let key = PRNGKey::from_seed(42);
    /// let (key1, key2) = key.split();
    /// ```
    pub fn split(self) -> (Self, Self) {
        let mut key1 = self;
        let mut key2 = self;

        // Mix the state differently for each key
        key1.state[0] = key1.state[0].wrapping_add(0x9e3779b97f4a7c15);
        key2.state[0] = key2.state[0].wrapping_add(0x3c6ef372fe94f82a);

        key1.state[1] = key1.state[1].rotate_left(27);
        key2.state[1] = key2.state[1].rotate_right(17);

        (key1, key2)
    }

    /// Split a key into n independent keys.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::random::PRNGKey;
    /// let key = PRNGKey::from_seed(42);
    /// let keys = key.split_n(5);
    /// assert_eq!(keys.len(), 5);
    /// ```
    pub fn split_n(mut self, n: usize) -> Vec<Self> {
        let mut keys = Vec::with_capacity(n);
        for _ in 0..n {
            let (key, next) = self.split();
            keys.push(key);
            self = next;
        }
        keys
    }

    /// Generate a random u64 using xorshift128+
    fn next_u64(&mut self) -> u64 {
        let mut s1 = self.state[0];
        let s0 = self.state[1];

        self.state[0] = s0;
        s1 ^= s1 << 23;
        s1 ^= s1 >> 17;
        s1 ^= s0;
        s1 ^= s0 >> 26;
        self.state[1] = s1;

        s1.wrapping_add(s0)
    }

    /// Generate a random f32 in [0, 1)
    fn next_f32(&mut self) -> f32 {
        let u = self.next_u64();
        // Use upper 24 bits for mantissa precision
        ((u >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Generate a random f32 in [min, max)
    fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

/// Generate uniform random values in [0, 1).
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, uniform}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = uniform(key, Shape::new(vec![3, 2]), DType::Float32);
/// assert_eq!(samples.shape().as_slice(), &[3, 2]);
/// ```
pub fn uniform(mut key: PRNGKey, shape: Shape, dtype: DType) -> Array {
    uniform_device(key, shape, dtype, Device::Cpu)
}

/// Generate uniform random values in [0, 1) on a specific device.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, uniform_device}, Shape, DType, Device};
/// let key = PRNGKey::from_seed(42);
/// let samples = uniform_device(key, Shape::new(vec![1000]), DType::Float32, Device::WebGpu);
/// ```
pub fn uniform_device(mut key: PRNGKey, shape: Shape, dtype: DType, device: Device) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");

    let size = shape.size();

    match device {
        Device::WebGpu => {
            // GPU path using Philox PRNG
            let output_buffer = Buffer::zeros(size, dtype, Device::WebGpu);
            let seed = key.state;
            crate::backend::ops::gpu_uniform(&output_buffer, size, seed, 0);
            Array::from_buffer(output_buffer, shape)
        }
        Device::Cpu | Device::Wasm => {
            // CPU path
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                data.push(key.next_f32());
            }
            Array::from_vec(data, shape)
        }
    }
}

/// Generate uniform random values in [minval, maxval).
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, uniform_range}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = uniform_range(key, Shape::new(vec![10]), -1.0, 1.0, DType::Float32);
/// ```
pub fn uniform_range(
    mut key: PRNGKey,
    shape: Shape,
    minval: f32,
    maxval: f32,
    dtype: DType,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");

    let size = shape.size();
    let mut data = Vec::with_capacity(size);

    for _ in 0..size {
        data.push(key.next_f32_range(minval, maxval));
    }

    Array::from_vec(data, shape)
}

/// Generate random integers in [minval, maxval).
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, randint}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = randint(key, Shape::new(vec![5]), 0, 10, DType::Float32);
/// ```
pub fn randint(
    mut key: PRNGKey,
    shape: Shape,
    minval: i32,
    maxval: i32,
    dtype: DType,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert!(minval < maxval, "minval must be less than maxval");

    let size = shape.size();
    let mut data = Vec::with_capacity(size);
    let range = (maxval - minval) as u64;

    for _ in 0..size {
        let r = (key.next_u64() % range) as i32 + minval;
        data.push(r as f32);
    }

    Array::from_vec(data, shape)
}

/// Generate random samples from a normal (Gaussian) distribution.
///
/// Uses Box-Muller transform to generate normal variates.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, normal}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = normal(key, Shape::new(vec![100]), DType::Float32);
/// ```
pub fn normal(mut key: PRNGKey, shape: Shape, dtype: DType) -> Array {
    normal_device(key, shape, dtype, Device::Cpu)
}

/// Generate random samples from a standard normal distribution on a specific device.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, normal_device}, Shape, DType, Device};
/// let key = PRNGKey::from_seed(42);
/// let samples = normal_device(key, Shape::new(vec![1000]), DType::Float32, Device::WebGpu);
/// ```
pub fn normal_device(mut key: PRNGKey, shape: Shape, dtype: DType, device: Device) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");

    let size = shape.size();

    match device {
        Device::WebGpu => {
            // GPU path using Philox + Box-Muller
            // Ensure even size for Box-Muller pairs
            let actual_size = if size % 2 == 0 { size } else { size + 1 };
            let output_buffer = Buffer::zeros(actual_size, dtype, Device::WebGpu);
            let seed = key.state;
            crate::backend::ops::gpu_normal(&output_buffer, actual_size, seed, 0);

            // If size was odd, take only the first 'size' elements
            if size == actual_size {
                Array::from_buffer(output_buffer, shape)
            } else {
                let all_data = Array::from_buffer(output_buffer, Shape::new(vec![actual_size]));
                let mut data_vec = all_data.to_vec();
                data_vec.truncate(size);
                Array::from_vec(data_vec, shape)
            }
        }
        Device::Cpu | Device::Wasm => {
            // CPU path with Box-Muller
            let mut data = Vec::with_capacity(size);

            // Box-Muller transform generates pairs, so generate in pairs
            let mut i = 0;
            while i < size {
                let u1 = key.next_f32();
                let u2 = key.next_f32();

                // Avoid log(0)
                let u1 = u1.max(1e-10);

                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;

                data.push(r * theta.cos());
                i += 1;

                if i < size {
                    data.push(r * theta.sin());
                    i += 1;
                }
            }

            Array::from_vec(data, shape)
        }
    }
}

/// Generate random samples from a normal distribution with given mean and stddev.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, normal_with_params}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = normal_with_params(key, Shape::new(vec![100]), 5.0, 2.0, DType::Float32);
/// ```
pub fn normal_with_params(
    key: PRNGKey,
    shape: Shape,
    mean: f32,
    stddev: f32,
    dtype: DType,
) -> Array {
    let standard = normal(key, shape, dtype);
    let data: Vec<f32> =
        standard.to_vec().iter().map(|&x| x * stddev + mean).collect();

    Array::from_vec(data, standard.shape().clone())
}

/// Generate random samples from a Bernoulli distribution.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, bernoulli}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = bernoulli(key, 0.5, Shape::new(vec![10]), DType::Float32);
/// ```
pub fn bernoulli(
    mut key: PRNGKey,
    p: f32,
    shape: Shape,
    dtype: DType,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");

    let size = shape.size();
    let mut data = Vec::with_capacity(size);

    for _ in 0..size {
        let val = if key.next_f32() < p { 1.0 } else { 0.0 };
        data.push(val);
    }

    Array::from_vec(data, shape)
}

/// Randomly permute a sequence.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, permutation}, Array, Shape};
/// let key = PRNGKey::from_seed(42);
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
/// let shuffled = permutation(key, &x);
/// ```
pub fn permutation(mut key: PRNGKey, x: &Array) -> Array {
    assert_eq!(x.ndim(), 1, "permutation only supports 1-D arrays");

    let mut data = x.to_vec();
    let n = data.len();

    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = (key.next_u64() as usize) % (i + 1);
        data.swap(i, j);
    }

    Array::from_vec(data, x.shape().clone())
}

/// Randomly sample elements from an array without replacement.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, choice}, Array, Shape};
/// let key = PRNGKey::from_seed(42);
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
/// let samples = choice(key, &x, 3, false);
/// assert_eq!(samples.size(), 3);
/// ```
pub fn choice(mut key: PRNGKey, x: &Array, n: usize, replace: bool) -> Array {
    assert_eq!(x.ndim(), 1, "choice only supports 1-D arrays");
    let size = x.size();

    if !replace {
        assert!(
            n <= size,
            "Cannot sample more elements than available without replacement"
        );
    }

    let data = x.to_vec();
    let mut result = Vec::with_capacity(n);

    if replace {
        // Sample with replacement
        for _ in 0..n {
            let idx = (key.next_u64() as usize) % size;
            result.push(data[idx]);
        }
    } else {
        // Sample without replacement - use reservoir sampling
        let mut indices: Vec<usize> = (0..size).collect();

        // Fisher-Yates shuffle first n elements
        for i in (1..n).rev() {
            let j = (key.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        for i in 0..n {
            result.push(data[indices[i]]);
        }
    }

    Array::from_vec(result, Shape::new(vec![n]))
}

/// Shuffle an array in-place (functionally, returns a new shuffled array).
///
/// This is an alias for `permutation` for API compatibility with NumPy.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, shuffle}, Array, Shape};
/// let key = PRNGKey::from_seed(42);
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
/// let shuffled = shuffle(key, &x);
/// ```
pub fn shuffle(key: PRNGKey, x: &Array) -> Array {
    permutation(key, x)
}

/// Generate random samples from an exponential distribution (CPU version).
///
/// The exponential distribution has PDF: f(x) = λ * exp(-λ * x) for x >= 0.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, exponential}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = exponential(key, 1.0, Shape::new(vec![100]), DType::Float32);
/// ```
pub fn exponential(
    key: PRNGKey,
    rate: f32,
    shape: Shape,
    dtype: DType,
) -> Array {
    exponential_device(key, rate, shape, dtype, Device::Cpu)
}

/// Generate random samples from an exponential distribution with device support.
///
/// The exponential distribution has PDF: f(x) = λ * exp(-λ * x) for x >= 0.
/// Uses inverse transform sampling: X = -ln(U) / λ
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, exponential_device}, Shape, DType, Device};
/// let key = PRNGKey::from_seed(42);
/// let samples = exponential_device(key, 1.0, Shape::new(vec![100]), DType::Float32, Device::Cpu);
/// ```
pub fn exponential_device(
    mut key: PRNGKey,
    rate: f32,
    shape: Shape,
    dtype: DType,
    device: Device,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert!(rate > 0.0, "Rate must be positive");

    let size = shape.size();

    match device {
        Device::WebGpu => {
            let output_buffer = Buffer::zeros(size, dtype, Device::WebGpu);
            let seed = key.state;
            crate::backend::ops::gpu_exponential(&output_buffer, size, seed, 0, rate);
            Array::from_buffer(output_buffer, shape)
        }
        Device::Cpu | Device::Wasm => {
            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                let u = key.next_f32().max(1e-10); // Avoid log(0)
                data.push(-u.ln() / rate);
            }
            Array::from_vec(data, shape)
        }
    }
}

/// Generate random samples from a logistic distribution (CPU version).
///
/// The logistic distribution has PDF: f(x) = exp(-(x-μ)/s) / (s * (1 + exp(-(x-μ)/s))^2)
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, logistic}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = logistic(key, Shape::new(vec![100]), DType::Float32);
/// ```
pub fn logistic(key: PRNGKey, shape: Shape, dtype: DType) -> Array {
    logistic_device(key, shape, dtype, Device::Cpu)
}

/// Generate random samples from a logistic distribution with device selection.
///
/// Uses inverse transform sampling: X = μ + s * log(U / (1 - U))
/// where U ~ Uniform(0, 1), μ = location (default 0), s = scale (default 1).
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, logistic_device}, Shape, DType, Device};
/// let key = PRNGKey::from_seed(42);
/// let samples = logistic_device(key, Shape::new(vec![100]), DType::Float32, Device::Cpu);
/// ```
pub fn logistic_device(
    mut key: PRNGKey,
    shape: Shape,
    dtype: DType,
    device: Device,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");

    let size = shape.size();

    match device {
        Device::WebGpu => {
            let output_buffer = Buffer::zeros(size, dtype, Device::WebGpu);
            let seed = key.state;
            crate::backend::ops::gpu_logistic(&output_buffer, size, seed, 0, 0.0, 1.0);
            Array::from_buffer(output_buffer, shape)
        }
        Device::Cpu | Device::Wasm => {
            // CPU implementation using inverse transform
            let mut data = Vec::with_capacity(size);

            for _ in 0..size {
                let u = key.next_f32().clamp(1e-10, 1.0 - 1e-10); // Avoid division by zero
                let x = (u / (1.0 - u)).ln();
                data.push(x);
            }

            Array::from_vec(data, shape)
        }
    }
}

/// Generate random samples from a logistic distribution with custom parameters.
///
/// # Arguments
///
/// * `key` - PRNG key
/// * `loc` - Location parameter (μ)
/// * `scale` - Scale parameter (s), must be positive
/// * `shape` - Output shape
/// * `dtype` - Data type
/// * `device` - Compute device
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, logistic_with_params}, Shape, DType, Device};
/// let key = PRNGKey::from_seed(42);
/// let samples = logistic_with_params(key, 2.0, 0.5, Shape::new(vec![100]), DType::Float32, Device::Cpu);
/// ```
pub fn logistic_with_params(
    mut key: PRNGKey,
    loc: f32,
    scale: f32,
    shape: Shape,
    dtype: DType,
    device: Device,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert!(scale > 0.0, "Scale must be positive");

    let size = shape.size();

    match device {
        Device::WebGpu => {
            let output_buffer = Buffer::zeros(size, dtype, Device::WebGpu);
            let seed = key.state;
            crate::backend::ops::gpu_logistic(&output_buffer, size, seed, 0, loc, scale);
            Array::from_buffer(output_buffer, shape)
        }
        Device::Cpu | Device::Wasm => {
            let mut data = Vec::with_capacity(size);

            for _ in 0..size {
                let u = key.next_f32().clamp(1e-10, 1.0 - 1e-10);
                let x = loc + scale * (u / (1.0 - u)).ln();
                data.push(x);
            }

            Array::from_vec(data, shape)
        }
    }
}

/// Generate random samples from a gamma distribution.
///
/// Uses the Marsaglia and Tsang method.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, gamma}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = gamma(key, 2.0, 1.0, Shape::new(vec![100]), DType::Float32);
/// ```
pub fn gamma(
    mut key: PRNGKey,
    alpha: f32,
    beta: f32,
    shape: Shape,
    dtype: DType,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert!(alpha > 0.0, "Alpha must be positive");
    assert!(beta > 0.0, "Beta must be positive");

    let size = shape.size();
    let mut data = Vec::with_capacity(size);

    // Marsaglia and Tsang method for alpha >= 1
    let d = if alpha >= 1.0 { alpha - 1.0 / 3.0 } else { alpha + 2.0 / 3.0 };
    let c = 1.0 / (9.0 * d).sqrt();

    for _ in 0..size {
        let mut x: f32;
        let mut v: f32;

        loop {
            // Generate standard normal
            let u1 = key.next_f32().max(1e-10);
            let u2 = key.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            x = r * theta.cos();

            v = 1.0 + c * x;
            if v > 0.0 {
                v = v * v * v;
                let u = key.next_f32();
                if u < 1.0 - 0.0331 * x * x * x * x {
                    break;
                }
                if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                    break;
                }
            }
        }

        let mut result = d * v;

        // Adjustment for alpha < 1
        if alpha < 1.0 {
            let u = key.next_f32().max(1e-10);
            result *= u.powf(1.0 / alpha);
        }

        data.push(result / beta);
    }

    Array::from_vec(data, shape)
}

/// Generate random samples from a beta distribution.
///
/// Uses the relationship: X ~ Beta(a, b) where X = G1 / (G1 + G2)
/// with G1 ~ Gamma(a, 1) and G2 ~ Gamma(b, 1).
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, beta}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = beta(key, 2.0, 5.0, Shape::new(vec![100]), DType::Float32);
/// ```
pub fn beta(
    key: PRNGKey,
    alpha: f32,
    beta_param: f32,
    shape: Shape,
    dtype: DType,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert!(alpha > 0.0, "Alpha must be positive");
    assert!(beta_param > 0.0, "Beta must be positive");

    let (key1, key2) = key.split();
    let g1 = gamma(key1, alpha, 1.0, shape.clone(), dtype);
    let g2 = gamma(key2, beta_param, 1.0, shape.clone(), dtype);

    let g1_data = g1.to_vec();
    let g2_data = g2.to_vec();

    let data: Vec<f32> = g1_data
        .iter()
        .zip(g2_data.iter())
        .map(|(&x, &y)| x / (x + y))
        .collect();

    Array::from_vec(data, shape)
}

/// Generate random samples from a categorical distribution.
///
/// Samples indices based on probability weights.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, categorical}, Array, Shape};
/// let key = PRNGKey::from_seed(42);
/// let logits = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let samples = categorical(key, &logits, 10);
/// ```
pub fn categorical(mut key: PRNGKey, logits: &Array, num_samples: usize) -> Array {
    assert_eq!(logits.ndim(), 1, "Logits must be 1-D");

    let logits_data = logits.to_vec();
    let n_categories = logits_data.len();

    // Convert logits to probabilities via softmax
    let max_logit = logits_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_logits: Vec<f32> = logits_data.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f32 = exp_logits.iter().sum();
    let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

    // Compute cumulative probabilities
    let mut cumprobs = Vec::with_capacity(n_categories);
    let mut cumsum = 0.0;
    for p in &probs {
        cumsum += p;
        cumprobs.push(cumsum);
    }

    // Sample
    let mut samples = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let u = key.next_f32();
        let mut idx = 0;
        for (i, &cp) in cumprobs.iter().enumerate() {
            if u <= cp {
                idx = i;
                break;
            }
            idx = i;
        }
        samples.push(idx as f32);
    }

    Array::from_vec(samples, Shape::new(vec![num_samples]))
}

/// Generate random samples from a Poisson distribution.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, poisson}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = poisson(key, 5.0, Shape::new(vec![100]), DType::Float32);
/// ```
pub fn poisson(mut key: PRNGKey, lam: f32, shape: Shape, dtype: DType) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert!(lam > 0.0, "Lambda must be positive");

    let size = shape.size();
    let mut data = Vec::with_capacity(size);

    // For small lambda, use direct method
    let l = (-lam).exp();

    for _ in 0..size {
        let mut k = 0;
        let mut p = 1.0f32;

        loop {
            p *= key.next_f32();
            if p <= l {
                break;
            }
            k += 1;
        }

        data.push(k as f32);
    }

    Array::from_vec(data, shape)
}

/// Generate random samples from a truncated normal distribution.
///
/// Samples from a normal distribution truncated to [lower, upper].
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, truncated_normal}, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let samples = truncated_normal(key, -1.0, 1.0, Shape::new(vec![100]), DType::Float32);
/// ```
pub fn truncated_normal(
    mut key: PRNGKey,
    lower: f32,
    upper: f32,
    shape: Shape,
    dtype: DType,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert!(lower < upper, "Lower must be less than upper");

    let size = shape.size();
    let mut data = Vec::with_capacity(size);

    // Use rejection sampling
    for _ in 0..size {
        loop {
            let u1 = key.next_f32().max(1e-10);
            let u2 = key.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            let z = r * theta.cos();

            if z >= lower && z <= upper {
                data.push(z);
                break;
            }
        }
    }

    Array::from_vec(data, shape)
}

/// Generate multivariate normal samples.
///
/// # Examples
///
/// ```rust,ignore
/// # use jax_rs::{random::{PRNGKey, multivariate_normal}, Array, Shape, DType};
/// let key = PRNGKey::from_seed(42);
/// let mean = Array::from_vec(vec![0.0, 0.0], Shape::new(vec![2]));
/// let cov = Array::from_vec(vec![1.0, 0.5, 0.5, 1.0], Shape::new(vec![2, 2]));
/// let samples = multivariate_normal(key, &mean, &cov, 100, DType::Float32);
/// ```
pub fn multivariate_normal(
    key: PRNGKey,
    mean: &Array,
    cov: &Array,
    num_samples: usize,
    dtype: DType,
) -> Array {
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");
    assert_eq!(mean.ndim(), 1, "Mean must be 1-D");
    assert_eq!(cov.ndim(), 2, "Covariance must be 2-D");

    let n = mean.size();
    let cov_shape = cov.shape().as_slice();
    assert_eq!(cov_shape[0], n, "Covariance must be n x n");
    assert_eq!(cov_shape[1], n, "Covariance must be n x n");

    // Compute Cholesky decomposition L where cov = L @ L^T
    let l = cov.cholesky();

    // Generate standard normal samples [num_samples, n]
    let z = normal(key, Shape::new(vec![num_samples, n]), dtype);

    // Transform: x = mean + L @ z^T
    let mean_data = mean.to_vec();
    let l_data = l.to_vec();
    let z_data = z.to_vec();

    let mut result_data = Vec::with_capacity(num_samples * n);

    for i in 0..num_samples {
        for j in 0..n {
            let mut val = mean_data[j];
            for k in 0..n {
                // L is lower triangular, so L[j][k] = 0 for k > j
                if k <= j {
                    val += l_data[j * n + k] * z_data[i * n + k];
                }
            }
            result_data.push(val);
        }
    }

    Array::from_vec(result_data, Shape::new(vec![num_samples, n]))
}

/// Generate samples from a discrete distribution.
///
/// # Examples
///
/// ```
/// # use jax_rs::{random::{PRNGKey, discrete}, Array, Shape};
/// let key = PRNGKey::from_seed(42);
/// let probs = Array::from_vec(vec![0.1, 0.3, 0.6], Shape::new(vec![3]));
/// let samples = discrete(key, &probs, 10);
/// ```
pub fn discrete(mut key: PRNGKey, probs: &Array, num_samples: usize) -> Array {
    assert_eq!(probs.ndim(), 1, "Probabilities must be 1-D");

    let probs_data = probs.to_vec();
    let n_categories = probs_data.len();

    // Normalize probabilities
    let sum: f32 = probs_data.iter().sum();
    let normalized: Vec<f32> = probs_data.iter().map(|&p| p / sum).collect();

    // Compute cumulative probabilities
    let mut cumprobs = Vec::with_capacity(n_categories);
    let mut cumsum = 0.0;
    for p in &normalized {
        cumsum += p;
        cumprobs.push(cumsum);
    }

    // Sample
    let mut samples = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
        let u = key.next_f32();
        let mut idx = 0;
        for (i, &cp) in cumprobs.iter().enumerate() {
            if u <= cp {
                idx = i;
                break;
            }
            idx = i;
        }
        samples.push(idx as f32);
    }

    Array::from_vec(samples, Shape::new(vec![num_samples]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prng_key_split() {
        let key = PRNGKey::from_seed(42);
        let (key1, key2) = key.split();

        // Keys should be different
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_prng_key_split_n() {
        let key = PRNGKey::from_seed(42);
        let keys = key.split_n(5);

        assert_eq!(keys.len(), 5);

        // All keys should be unique
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i], keys[j]);
            }
        }
    }

    #[test]
    fn test_uniform() {
        let key = PRNGKey::from_seed(42);
        let samples = uniform(key, Shape::new(vec![100]), DType::Float32);

        assert_eq!(samples.size(), 100);

        let data = samples.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_uniform_range() {
        let key = PRNGKey::from_seed(42);
        let samples = uniform_range(
            key,
            Shape::new(vec![100]),
            -5.0,
            5.0,
            DType::Float32,
        );

        let data = samples.to_vec();
        for &val in &data {
            assert!(val >= -5.0 && val < 5.0);
        }
    }

    #[test]
    fn test_randint() {
        let key = PRNGKey::from_seed(42);
        let samples =
            randint(key, Shape::new(vec![100]), 0, 10, DType::Float32);

        let data = samples.to_vec();
        for &val in &data {
            assert!(val >= 0.0 && val < 10.0);
            assert_eq!(val.fract(), 0.0); // Should be integer
        }
    }

    #[test]
    fn test_normal() {
        let key = PRNGKey::from_seed(42);
        let samples = normal(key, Shape::new(vec![1000]), DType::Float32);

        let data = samples.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                / data.len() as f32;

        // Mean should be close to 0, stddev close to 1
        assert!((mean.abs()) < 0.1, "Mean is {}", mean);
        assert!(
            (variance.sqrt() - 1.0).abs() < 0.1,
            "Stddev is {}",
            variance.sqrt()
        );
    }

    #[test]
    fn test_normal_with_params() {
        let key = PRNGKey::from_seed(42);
        let samples = normal_with_params(
            key,
            Shape::new(vec![1000]),
            5.0,
            2.0,
            DType::Float32,
        );

        let data = samples.to_vec();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;

        assert!((mean - 5.0).abs() < 0.2);
    }

    #[test]
    fn test_bernoulli() {
        let key = PRNGKey::from_seed(42);
        let samples =
            bernoulli(key, 0.5, Shape::new(vec![1000]), DType::Float32);

        let data = samples.to_vec();
        let ones = data.iter().filter(|&&x| x == 1.0).count();
        let proportion = ones as f32 / data.len() as f32;

        // Should be close to 0.5
        assert!((proportion - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_permutation() {
        let key = PRNGKey::from_seed(42);
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::new(vec![5]),
        );
        let shuffled = permutation(key, &x);

        assert_eq!(shuffled.size(), 5);

        // All original elements should be present
        let mut sorted = shuffled.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(sorted, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_choice_without_replacement() {
        let key = PRNGKey::from_seed(42);
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::new(vec![5]),
        );
        let samples = choice(key, &x, 3, false);

        assert_eq!(samples.size(), 3);

        // All samples should be from the original array
        let sample_data = samples.to_vec();
        let original_data = x.to_vec();
        for &val in &sample_data {
            assert!(original_data.contains(&val));
        }
    }

    #[test]
    fn test_choice_with_replacement() {
        let key = PRNGKey::from_seed(42);
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let samples = choice(key, &x, 10, true);

        assert_eq!(samples.size(), 10);
    }

    #[test]
    fn test_reproducibility() {
        let key = PRNGKey::from_seed(42);
        let samples1 = uniform(key, Shape::new(vec![10]), DType::Float32);

        let key = PRNGKey::from_seed(42);
        let samples2 = uniform(key, Shape::new(vec![10]), DType::Float32);

        // Same seed should produce same results
        assert_eq!(samples1.to_vec(), samples2.to_vec());
    }
}
