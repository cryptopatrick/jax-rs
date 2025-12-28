//! WebGPU matrix multiplication example.
//!
//! Run with: cargo run --example gpu_matmul --features webgpu

use jax_rs::{Array, Device, DType, Shape};
use jax_rs::backend::webgpu::WebGpuContext;

fn main() {
    // Initialize GPU
    pollster::block_on(async {
        WebGpuContext::init()
            .await
            .expect("WebGPU not available. Make sure you have a compatible GPU.");
    });

    println!("WebGPU initialized successfully!");

    // Create matrices on GPU
    let size = 1024;
    let a = Array::from_vec(
        vec![1.0; size * size],
        Shape::new(vec![size, size]),
    )
    .to_device(Device::WebGpu);

    let b = Array::from_vec(
        vec![2.0; size * size],
        Shape::new(vec![size, size]),
    )
    .to_device(Device::WebGpu);

    println!("Performing {}x{} matrix multiplication on GPU...", size, size);

    let start = std::time::Instant::now();
    let c = a.matmul(&b);
    let _ = c.to_vec(); // Force GPU completion
    let elapsed = start.elapsed();

    println!("GPU matmul completed in {:?}", elapsed);
    println!("Result shape: {:?}", c.shape().as_slice());
    println!("First element: {}", c.to_vec()[0]);
}
