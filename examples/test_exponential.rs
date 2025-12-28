use jax_rs::{Device, DType, Shape};
use jax_rs::random::{PRNGKey, exponential_device};

fn main() {
    println!("Testing exponential distribution implementation...\n");

    let rate = 1.0;

    // Test CPU implementation
    println!("=== CPU Test (rate={}) ===", rate);
    let key = PRNGKey::from_seed(42);
    let n = 1000;
    let samples_cpu = exponential_device(key.clone(), rate, Shape::new(vec![n]), DType::Float32, Device::Cpu);

    let data = samples_cpu.to_vec();
    let mean: f32 = data.iter().sum::<f32>() / n as f32;
    let variance: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;

    println!("Samples: {}", n);
    println!("Mean: {:.4} (expected: {:.4})", mean, 1.0 / rate);
    println!("Variance: {:.4} (expected: {:.4})", variance, 1.0 / (rate * rate));
    println!("First 10 samples: {:?}", &data[..10]);

    // Test GPU implementation
    #[cfg(feature = "webgpu")]
    {
        println!("\n=== GPU Test (rate={}) ===", rate);

        // Initialize WebGPU
        pollster::block_on(async {
            if let Err(e) = jax_rs::backend::webgpu::WebGpuContext::init().await {
                if !e.contains("already initialized") {
                    panic!("Failed to init WebGPU: {}", e);
                }
            }
        });

        let key_gpu = PRNGKey::from_seed(42);
        let samples_gpu = exponential_device(key_gpu, rate, Shape::new(vec![n]), DType::Float32, Device::WebGpu);

        let data_gpu = samples_gpu.to_vec();
        let mean_gpu: f32 = data_gpu.iter().sum::<f32>() / n as f32;
        let variance_gpu: f32 = data_gpu.iter().map(|x| (x - mean_gpu).powi(2)).sum::<f32>() / n as f32;

        println!("Samples: {}", n);
        println!("Mean: {:.4} (expected: {:.4})", mean_gpu, 1.0 / rate);
        println!("Variance: {:.4} (expected: {:.4})", variance_gpu, 1.0 / (rate * rate));
        println!("First 10 samples: {:?}", &data_gpu[..10]);

        // Benchmark larger sample
        println!("\n=== Performance Comparison ===");
        let n_large = 10_000_000;

        use std::time::Instant;

        let start = Instant::now();
        let key_cpu_bench = PRNGKey::from_seed(123);
        let _samples_cpu_large = exponential_device(key_cpu_bench, rate, Shape::new(vec![n_large]), DType::Float32, Device::Cpu);
        let cpu_time = start.elapsed();

        let start = Instant::now();
        let key_gpu_bench = PRNGKey::from_seed(123);
        let _samples_gpu_large = exponential_device(key_gpu_bench, rate, Shape::new(vec![n_large]), DType::Float32, Device::WebGpu);
        let gpu_time = start.elapsed();

        println!("CPU time for {} samples: {:.3}s", n_large, cpu_time.as_secs_f64());
        println!("GPU time for {} samples: {:.3}s", n_large, gpu_time.as_secs_f64());
        println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    }

    println!("\n✓ Exponential distribution test complete!");
}
