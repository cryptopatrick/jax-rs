use criterion::{black_box, criterion_group, criterion_main, Criterion};
use jax_rs::{Array, DType, Shape};

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

fn bench_matmul(c: &mut Criterion) {
    let a = Array::zeros(Shape::new(vec![100, 100]), DType::Float32);
    let b = Array::zeros(Shape::new(vec![100, 100]), DType::Float32);

    c.bench_function("matmul 100x100", |b_bench| {
        b_bench.iter(|| black_box(&a).matmul(black_box(&b)))
    });
}

criterion_group!(
    benches,
    bench_array_creation,
    bench_unary_ops,
    bench_binary_ops,
    bench_reductions,
    bench_matmul
);
criterion_main!(benches);
