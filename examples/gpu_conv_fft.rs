//! GPU Convolution and FFT Benchmark
//!
//! Demonstrates the performance improvements from GPU-accelerated
//! convolution and FFT operations.
//!
//! Run with: cargo run --example gpu_conv_fft --release

use jax_rs::{nn, ops, Array, Device, DType, Shape};
use std::time::Instant;

fn benchmark_conv2d() {
    println!("=== Conv2D Benchmark ===\n");

    // Test different image sizes
    let test_cases = vec![
        ("Small (28x28)", 1, 32, 28, 28, 64, 3),
        ("Medium (224x224)", 1, 3, 224, 224, 64, 3),
        ("Large (512x512)", 1, 3, 512, 512, 128, 3),
    ];

    for (name, batch, in_ch, h, w, out_ch, kernel_size) in test_cases {
        let input_shape = Shape::new(vec![batch, in_ch, h, w]);
        let kernel_shape = Shape::new(vec![out_ch, in_ch, kernel_size, kernel_size]);

        // CPU benchmark
        let cpu_input = Array::ones(input_shape.clone(), DType::Float32);
        let cpu_kernel = Array::ones(kernel_shape.clone(), DType::Float32);

        let start = Instant::now();
        let _cpu_result = nn::conv2d_batched(&cpu_input, &cpu_kernel, 1, 1);
        let cpu_time = start.elapsed();

        // GPU benchmark
        let gpu_input = cpu_input.to_device(Device::WebGpu);
        let gpu_kernel = cpu_kernel.to_device(Device::WebGpu);

        let start = Instant::now();
        let _gpu_result = nn::conv2d_batched(&gpu_input, &gpu_kernel, 1, 1);
        let gpu_time = start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!("{}", name);
        println!("  Input: {}x{}x{}x{}", batch, in_ch, h, w);
        println!("  Kernel: {}x{}x{}x{}", out_ch, in_ch, kernel_size, kernel_size);
        println!("  CPU time: {:.3}ms", cpu_time.as_secs_f64() * 1000.0);
        println!("  GPU time: {:.3}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("  Speedup: {:.1}x\n", speedup);
    }
}

fn benchmark_conv1d() {
    println!("=== Conv1D Benchmark ===\n");

    let test_cases = vec![
        ("Small", 100),
        ("Medium", 1000),
        ("Large", 10000),
    ];

    for (name, length) in test_cases {
        let kernel_size = 7;

        // CPU benchmark
        let cpu_input = Array::ones(Shape::new(vec![length]), DType::Float32);
        let cpu_kernel = Array::ones(Shape::new(vec![kernel_size]), DType::Float32);

        let start = Instant::now();
        let _cpu_result = nn::conv1d(&cpu_input, &cpu_kernel);
        let cpu_time = start.elapsed();

        // GPU benchmark
        let gpu_input = cpu_input.to_device(Device::WebGpu);
        let gpu_kernel = cpu_kernel.to_device(Device::WebGpu);

        let start = Instant::now();
        let _gpu_result = nn::conv1d(&gpu_input, &gpu_kernel);
        let gpu_time = start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!("{} (length={})", name, length);
        println!("  CPU time: {:.3}ms", cpu_time.as_secs_f64() * 1000.0);
        println!("  GPU time: {:.3}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("  Speedup: {:.1}x\n", speedup);
    }
}

fn benchmark_fft() {
    println!("=== FFT Benchmark ===\n");

    let test_cases = vec![
        ("Small", 64),
        ("Medium", 512),
        ("Large", 4096),
    ];

    for (name, size) in test_cases {
        // CPU benchmark
        let cpu_input = Array::ones(Shape::new(vec![size]), DType::Float32);

        let start = Instant::now();
        let _cpu_result = ops::fft::fft(&cpu_input);
        let cpu_time = start.elapsed();

        // GPU benchmark
        let gpu_input = cpu_input.to_device(Device::WebGpu);

        let start = Instant::now();
        let _gpu_result = ops::fft::fft(&gpu_input);
        let gpu_time = start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!("{} (size={})", name, size);
        println!("  CPU time: {:.3}ms", cpu_time.as_secs_f64() * 1000.0);
        println!("  GPU time: {:.3}ms", gpu_time.as_secs_f64() * 1000.0);
        println!("  Speedup: {:.1}x\n", speedup);
    }
}

fn main() {
    println!("\nGPU Convolution & FFT Performance Benchmark");
    println!("==========================================\n");

    // Initialize WebGPU
    println!("Initializing WebGPU...");
    pollster::block_on(async {
        jax_rs::backend::webgpu::WebGpuContext::init()
            .await
            .expect("WebGPU not available. Make sure you have a compatible GPU.");
    });
    println!("WebGPU initialized!\n");

    benchmark_conv2d();
    benchmark_conv1d();
    benchmark_fft();

    println!("Benchmark complete!");
}
