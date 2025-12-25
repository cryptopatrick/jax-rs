use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jax_rs::{Array, Device, DType, Shape};

#[cfg(feature = "webgpu")]
use jax_rs::backend::webgpu::WebGpuContext;

fn bench_array_creation(c: &mut Criterion) {
    c.bench_function("zeros 1000", |b| {
        b.iter(|| {
            Array::zeros(black_box(Shape::new(vec![1000])), DType::Float32)
        })
    });

    c.bench_function("ones 1000", |b| {
        b.iter(|| {
            Array::ones(black_box(Shape::new(vec![1000])), DType::Float32)
        })
    });
}

fn bench_unary_ops(c: &mut Criterion) {
    let x = Array::from_vec(
        (0..1000).map(|i| i as f32).collect(),
        Shape::new(vec![1000]),
    );

    c.bench_function("neg 1000", |b| b.iter(|| black_box(&x).neg()));

    c.bench_function("sqrt 1000", |b| b.iter(|| black_box(&x).sqrt()));

    c.bench_function("exp 1000", |b| b.iter(|| black_box(&x).exp()));
}

fn bench_binary_ops(c: &mut Criterion) {
    let x = Array::from_vec(
        (0..1000).map(|i| i as f32).collect(),
        Shape::new(vec![1000]),
    );
    let y = Array::from_vec(
        (0..1000).map(|i| (i + 1) as f32).collect(),
        Shape::new(vec![1000]),
    );

    c.bench_function("add 1000", |b| {
        b.iter(|| black_box(&x).add(black_box(&y)))
    });

    c.bench_function("mul 1000", |b| {
        b.iter(|| black_box(&x).mul(black_box(&y)))
    });
}

fn bench_reductions(c: &mut Criterion) {
    let x = Array::from_vec(
        (0..1000).map(|i| i as f32).collect(),
        Shape::new(vec![1000]),
    );

    c.bench_function("sum_all 1000", |b| b.iter(|| black_box(&x).sum_all()));

    c.bench_function("mean_all 1000", |b| b.iter(|| black_box(&x).mean_all()));
}

fn bench_matmul_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_cpu");

    for size in [64, 128, 256, 512].iter() {
        let a = Array::from_vec(
            vec![1.0; size * size],
            Shape::new(vec![*size, *size]),
        );
        let b = Array::from_vec(
            vec![2.0; size * size],
            Shape::new(vec![*size, *size]),
        );

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.matmul(&b));
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "webgpu")]
fn bench_matmul_gpu(c: &mut Criterion) {
    // Initialize GPU once
    if !WebGpuContext::is_initialized() {
        pollster::block_on(async {
            WebGpuContext::init().await.expect("GPU init failed");
        });
    }

    let mut group = c.benchmark_group("matmul_gpu");

    for size in [64, 128, 256, 512, 1024].iter() {
        let a = Array::from_vec(
            vec![1.0; size * size],
            Shape::new(vec![*size, *size]),
        ).to_device(Device::WebGpu);

        let b = Array::from_vec(
            vec![2.0; size * size],
            Shape::new(vec![*size, *size]),
        ).to_device(Device::WebGpu);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |bench, _| {
                bench.iter(|| {
                    let result = black_box(a.matmul(&b));
                    // Force GPU sync
                    let _ = result.to_vec();
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "webgpu")]
criterion_group!(
    benches,
    bench_array_creation,
    bench_unary_ops,
    bench_binary_ops,
    bench_reductions,
    bench_matmul_cpu,
    bench_matmul_gpu
);

#[cfg(not(feature = "webgpu"))]
criterion_group!(
    benches,
    bench_array_creation,
    bench_unary_ops,
    bench_binary_ops,
    bench_reductions,
    bench_matmul_cpu
);

criterion_main!(benches);
