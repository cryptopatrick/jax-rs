//! Random number generation with reproducible PRNG keys.
//!
//! This module provides JAX-style random number generation using explicit
//! PRNG keys for reproducibility.

use crate::{Array, DType, Shape};
use std::num::Wrapping;

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
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");

    let size = shape.size();
    let mut data = Vec::with_capacity(size);

    for _ in 0..size {
        data.push(key.next_f32());
    }

    Array::from_vec(data, shape)
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
    assert_eq!(dtype, DType::Float32, "Only Float32 supported");

    let size = shape.size();
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
