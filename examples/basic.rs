use jax_rs::*;

fn main() {
    println!("=== jax-rs Comprehensive Example ===\n");

    // === Phase 1-2: Basic Arrays ===
    println!("1. Array Creation:");
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    let b = Array::ones(Shape::new(vec![2, 2]), DType::Float32);
    println!("   a = {}", a);
    println!("   b = {}", b);

    // === Phase 4.1: Array Creation Functions ===
    println!("\n2. Array Creation Functions:");
    let r = Array::arange(0.0, 10.0, 2.0, DType::Float32);
    println!("   arange(0, 10, 2) = {:?}", r.to_vec());

    let l = Array::linspace(0.0, 1.0, 5, true, DType::Float32);
    println!("   linspace(0, 1, 5) = {:?}", l.to_vec());

    let eye = Array::eye(3, None, DType::Float32);
    println!("   eye(3) shape = {:?}", eye.shape());

    // === Phase 3: Binary Operations ===
    println!("\n3. Binary Operations:");
    let c = a.add(&b);
    println!("   a + b = {:?}", c.to_vec());

    let d = a.mul(&b);
    println!("   a * b = {:?}", d.to_vec());

    // === Phase 3: Unary Operations ===
    println!("\n4. Unary Operations:");
    let e = a.sqrt();
    println!("   sqrt(a) = {:?}", e.to_vec());

    let f = a.sin();
    println!("   sin(a) = {:?}", f.to_vec());

    // === Phase 3: Reductions ===
    println!("\n5. Reductions:");
    println!("   sum(a) = {}", a.sum_all());
    println!("   mean(a) = {}", a.mean_all());
    println!("   max(a) = {}", a.max_all());
    println!("   sum along axis 0 = {:?}", a.sum(0).to_vec());

    // === Phase 4.2: Linear Algebra ===
    println!("\n6. Linear Algebra:");
    let m1 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    let m2 = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
    let prod = m1.matmul(&m2);
    println!("   matmul result = {:?}", prod.to_vec());

    let v1 = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    let v2 = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
    let dot_prod = v1.dot(&v2);
    println!("   dot product = {:?}", dot_prod.to_vec());

    let trans = m1.transpose();
    println!("   transpose shape = {:?}", trans.shape());

    // === Phase 4.3: Comparisons ===
    println!("\n7. Comparison Operations:");
    let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    let y = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
    println!("   x < y = {:?}", x.lt(&y).to_vec());
    println!("   x == y = {:?}", x.eq(&y).to_vec());
    println!("   x > y = {:?}", x.gt(&y).to_vec());

    // === Phase 4.4: Shape Manipulation ===
    println!("\n8. Shape Manipulation:");
    let orig = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    println!("   original shape = {:?}", orig.shape());

    let expanded = orig.expand_dims(0);
    println!("   expand_dims(0) = {:?}", expanded.shape());

    let squeezed = expanded.squeeze();
    println!("   squeeze = {:?}", squeezed.shape());

    println!("\n=== Example complete! ===");
    println!("Total operations demonstrated: 30+");
}
