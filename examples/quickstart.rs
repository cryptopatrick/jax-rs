//! Quickstart example demonstrating jax-rs features.
//!
//! Run with: cargo run --example quickstart

use jax_rs::{grad, jit, vmap, Array, DType, Shape};

fn main() {
    println!("=== JAX-RS Quickstart ===\n");

    // 1. Array Creation
    println!("1. Array Creation");
    let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    println!("   x = {:?}", x.to_vec());

    let zeros = Array::zeros(Shape::new(vec![2, 3]), DType::Float32);
    println!("   zeros(2, 3) = {:?}", zeros.to_vec());

    let ones = Array::ones(Shape::new(vec![2, 2]), DType::Float32);
    println!("   ones(2, 2) = {:?}\n", ones.to_vec());

    // 2. Basic Operations
    println!("2. Basic Operations");
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));

    let sum = a.add(&b);
    println!("   [1,2,3] + [4,5,6] = {:?}", sum.to_vec());

    let prod = a.mul(&b);
    println!("   [1,2,3] * [4,5,6] = {:?}\n", prod.to_vec());

    // 3. Broadcasting
    println!("3. Broadcasting");
    let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3, 1]));
    let y = Array::from_vec(vec![10.0, 20.0], Shape::new(vec![1, 2]));
    let broadcast_result = x.add(&y);
    println!(
        "   Shape {:?} + Shape {:?} = Shape {:?}",
        x.shape().as_slice(),
        y.shape().as_slice(),
        broadcast_result.shape().as_slice()
    );
    println!("   Result: {:?}\n", broadcast_result.to_vec());

    // 4. Matrix Operations
    println!("4. Matrix Operations");
    let m1 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    let m2 = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
    let matmul = m1.matmul(&m2);
    println!("   [[1,2],[3,4]] @ [[5,6],[7,8]] = {:?}\n", matmul.to_vec());

    // 5. Reductions
    println!("5. Reductions");
    let data = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::new(vec![2, 3]),
    );
    println!("   sum_all([1,2,3,4,5,6]) = {}", data.sum_all());
    println!("   mean_all([1,2,3,4,5,6]) = {}", data.mean_all());
    println!("   sum(axis=0) = {:?}", data.sum(0).to_vec());
    println!("   sum(axis=1) = {:?}\n", data.sum(1).to_vec());

    // 6. Automatic Differentiation
    println!("6. Automatic Differentiation");
    let f = |x: &Array| x.mul(x).sum_all_array();
    let df = grad(f);

    let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    let gradient = df(&x);
    println!("   f(x) = sum(x^2)");
    println!("   grad(f) at [1,2,3] ≈ {:?}\n", gradient.to_vec());

    // 7. Vectorization (vmap)
    println!("7. Vectorization (vmap)");
    let square = |x: &Array| x.mul(x);
    let vmap_square = vmap(square, 0);

    let batch = Array::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::new(vec![3, 2]),
    );
    let batch_squared = vmap_square(&batch);
    println!("   vmap(square) on shape {:?}", batch.shape().as_slice());
    println!("   Result: {:?}\n", batch_squared.to_vec());

    // 8. JIT Compilation
    println!("8. JIT Compilation");
    let jit_f = jit("my_function", |inputs: &[Array]| {
        vec![inputs[0].add(&inputs[1]).mul(&inputs[0])]
    });

    let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    let b = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));

    let result = jit_f.call(&[a.clone(), b.clone()]);
    println!("   First call (compiles): {:?}", result[0].to_vec());

    let result = jit_f.call(&[a, b]);
    println!("   Second call (cached): {:?}\n", result[0].to_vec());

    // 9. Chained Operations
    println!("9. Chained Operations");
    let x = Array::from_vec(vec![1.0, 4.0, 9.0], Shape::new(vec![3]));
    let result = x
        .sqrt()
        .mul(&Array::from_vec(vec![2.0], Shape::new(vec![1])))
        .sum_all();
    println!("   sum(sqrt([1,4,9]) * 2) = {}\n", result);

    println!("=== Quickstart Complete ===");
}
