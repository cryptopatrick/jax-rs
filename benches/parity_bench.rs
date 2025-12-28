//! Performance benchmarks comparing jax-rs to jax-js.
//!
//! These benchmarks should be run alongside the jax-js benchmarks
//! to validate that jax-rs is faster than the original JavaScript implementation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use jax_rs::{Array, Device, DType, Shape};

fn bench_matmul_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_cpu");

    for size in [64, 128, 256, 512].iter() {
        let a = Array::from_vec(
            vec![1.0; size * size],
            Shape::new(vec![*size, *size]),
        );
        let b = Array::from_vec(
            vec![1.0; size * size],
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

fn bench_binary_ops_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_ops_cpu");

    let sizes = [1000, 10000, 100000];

    for size in sizes.iter() {
        let a = Array::from_vec(vec![2.0; *size], Shape::new(vec![*size]));
        let b = Array::from_vec(vec![3.0; *size], Shape::new(vec![*size]));

        group.bench_with_input(
            BenchmarkId::new("add", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.add(&b));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mul", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.mul(&b));
                });
            },
        );
    }

    group.finish();
}

fn bench_reductions_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("reductions_cpu");

    for size in [1000, 10000, 100000].iter() {
        let a = Array::from_vec(
            (0..*size).map(|i| i as f32).collect(),
            Shape::new(vec![*size]),
        );

        group.bench_with_input(
            BenchmarkId::new("sum", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.sum_all());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mean", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.mean_all());
                });
            },
        );
    }

    group.finish();
}

fn bench_unary_ops_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("unary_ops_cpu");

    for size in [1000, 10000, 100000].iter() {
        let a = Array::from_vec(
            (0..*size).map(|i| (i as f32) / 1000.0).collect(),
            Shape::new(vec![*size]),
        );

        group.bench_with_input(
            BenchmarkId::new("sin", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.sin());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("exp", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(a.exp());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_cpu,
    bench_binary_ops_cpu,
    bench_reductions_cpu,
    bench_unary_ops_cpu,
);
criterion_main!(benches);
