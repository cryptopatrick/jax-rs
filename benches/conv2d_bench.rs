use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jax_rs::{random, Device, DType, Shape, nn};

#[cfg(feature = "webgpu")]
use jax_rs::backend::webgpu::WebGpuContext;

/// Benchmark conv2d on CPU
/// Configuration: 1x64x256x256 input, 128 filters 3x3 (matching jax-js benchmark)
fn bench_conv2d_cpu(c: &mut Criterion) {
    // Configure for faster benchmarking (fewer samples, shorter warmup)
    let mut group = c.benchmark_group("conv2d");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(5));

    // Input: [batch=1, in_channels=64, height=256, width=256]
    let key1 = random::PRNGKey::from_seed(42);
    let input = random::normal(key1, Shape::new(vec![1, 64, 256, 256]), DType::Float32);
    // Kernel: [out_channels=128, in_channels=64, kernel_h=3, kernel_w=3]
    let key2 = random::PRNGKey::from_seed(43);
    let kernel = random::normal(key2, Shape::new(vec![128, 64, 3, 3]), DType::Float32);

    group.bench_function("cpu_1x64x256x256_128_3x3", |b| {
        b.iter(|| {
            black_box(nn::conv2d_batched(&input, &kernel, 1, 1))
        })
    });

    group.finish();
}

/// Benchmark conv2d on GPU with WebGPU
#[cfg(feature = "webgpu")]
fn bench_conv2d_gpu(c: &mut Criterion) {
    // Initialize GPU once
    if !WebGpuContext::is_initialized() {
        pollster::block_on(async {
            WebGpuContext::init().await.expect("GPU init failed");
        });
    }

    // Configure for faster benchmarking
    let mut group = c.benchmark_group("conv2d");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(5));

    // Input: [batch=1, in_channels=64, height=256, width=256]
    let key1 = random::PRNGKey::from_seed(42);
    let input = random::normal_device(key1, Shape::new(vec![1, 64, 256, 256]), DType::Float32, Device::WebGpu);
    // Kernel: [out_channels=128, in_channels=64, kernel_h=3, kernel_w=3]
    let key2 = random::PRNGKey::from_seed(43);
    let kernel = random::normal_device(key2, Shape::new(vec![128, 64, 3, 3]), DType::Float32, Device::WebGpu);

    group.bench_function("gpu_1x64x256x256_128_3x3", |b| {
        b.iter(|| {
            let result = black_box(nn::conv2d_batched(&input, &kernel, 1, 1));
            // Force GPU sync to ensure kernel completion
            let _ = result.to_vec();
        })
    });

    group.finish();
}

#[cfg(feature = "webgpu")]
criterion_group!(
    benches,
    bench_conv2d_cpu,
    bench_conv2d_gpu
);

#[cfg(not(feature = "webgpu"))]
criterion_group!(
    benches,
    bench_conv2d_cpu
);

criterion_main!(benches);
