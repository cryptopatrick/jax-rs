//! GPU Random Number Generation Benchmark
//!
//! Demonstrates the performance improvements from GPU-accelerated
//! random number generation using the Philox algorithm.
//!
//! Run with: cargo run --example gpu_random --release

use jax_rs::random::{PRNGKey, uniform_device, normal_device};
use jax_rs::{Device, DType, Shape};
use std::time::Instant;

fn benchmark_uniform() {
    println!("=== Uniform Distribution Benchmark ===\n");

    let key = PRNGKey::from_seed(42);
    let test_cases = vec![
        ("Small (1K)", 1_000),
        ("Medium (100K)", 100_000),
        ("Large (10M)", 10_000_000),
    ];

    for (name, size) in test_cases {
        let shape = Shape::new(vec![size]);

        // CPU benchmark
        let start = Instant::now();
        let _cpu_result = uniform_device(key, shape.clone(), DType::Float32, Device::Cpu);
        let cpu_time = start.elapsed();

        // GPU benchmark
        let start = Instant::now();
        let gpu_result = uniform_device(key, shape.clone(), DType::Float32, Device::WebGpu);
        let _ = gpu_result.to_vec(); // Force GPU completion
        let gpu_time = start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!("{} elements", name);
        println!("  CPU time: {:.3}ms", cpu_time.as_secs_f64() * 1000.0);
        println!("  GPU time: {:.3}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("  Speedup: {:.1}x\n", speedup);
    }
}

fn benchmark_normal() {
    println!("=== Normal Distribution Benchmark ===\n");

    let key = PRNGKey::from_seed(42);
    let test_cases = vec![
        ("Small (1K)", 1_000),
        ("Medium (100K)", 100_000),
        ("Large (10M)", 10_000_000),
    ];

    for (name, size) in test_cases {
        let shape = Shape::new(vec![size]);

        // CPU benchmark
        let start = Instant::now();
        let _cpu_result = normal_device(key, shape.clone(), DType::Float32, Device::Cpu);
        let cpu_time = start.elapsed();

        // GPU benchmark
        let start = Instant::now();
        let gpu_result = normal_device(key, shape.clone(), DType::Float32, Device::WebGpu);
        let _ = gpu_result.to_vec(); // Force GPU completion
        let gpu_time = start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!("{} elements", name);
        println!("  CPU time: {:.3}ms", cpu_time.as_secs_f64() * 1000.0);
        println!("  GPU time: {:.3}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("  Speedup: {:.1}x\n", speedup);
    }
}

fn test_correctness() {
    println!("=== Correctness Tests ===\n");

    let key = PRNGKey::from_seed(123);
    let shape = Shape::new(vec![10]);

    // Test uniform
    let uniform_gpu = uniform_device(key, shape.clone(), DType::Float32, Device::WebGpu);
    let uniform_values = uniform_gpu.to_vec();

    println!("Uniform samples (first 10):");
    for (i, val) in uniform_values.iter().enumerate() {
        println!("  [{i}] = {val:.6}");
    }

    // Check range
    let in_range = uniform_values.iter().all(|&x| x >= 0.0 && x < 1.0);
    println!("  All in [0, 1): {}\n", in_range);

    // Test normal
    let normal_gpu = normal_device(key, shape.clone(), DType::Float32, Device::WebGpu);
    let normal_values = normal_gpu.to_vec();

    println!("Normal samples (first 10):");
    for (i, val) in normal_values.iter().enumerate() {
        println!("  [{i}] = {val:.6}");
    }

    // Check basic statistics
    let mean: f32 = normal_values.iter().sum::<f32>() / normal_values.len() as f32;
    let variance: f32 = normal_values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / normal_values.len() as f32;
    let stddev = variance.sqrt();

    println!("  Mean: {:.6} (expected ~0.0)", mean);
    println!("  Stddev: {:.6} (expected ~1.0)\n", stddev);
}

fn main() {
    println!("\nGPU Random Number Generation Benchmark");
    println!("=======================================\n");

    // Initialize WebGPU
    println!("Initializing WebGPU...");
    pollster::block_on(async {
        jax_rs::backend::webgpu::WebGpuContext::init()
            .await
            .expect("WebGPU not available. Make sure you have a compatible GPU.");
    });
    println!("WebGPU initialized!\n");

    test_correctness();
    benchmark_uniform();
    benchmark_normal();

    println!("Benchmark complete!");
}
