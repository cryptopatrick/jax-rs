use jax_rs::*;

fn main() {
    println!("=== jax-rs Basic Example ===\n");

    // Create arrays
    println!("Creating arrays...");
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    let b = Array::ones(Shape::new(vec![2, 2]), DType::Float32);

    println!("a = {}", a);
    println!("b = {}", b);

    // Binary operations
    println!("\nBinary operations:");
    let c = a.add(&b);
    println!("a + b = {:?}", c.to_vec());

    let d = a.mul(&b);
    println!("a * b = {:?}", d.to_vec());

    // Unary operations
    println!("\nUnary operations:");
    let e = a.sqrt();
    println!("sqrt(a) = {:?}", e.to_vec());

    let f = a.exp();
    println!("exp(a) = {:?}", f.to_vec());

    // Reductions
    println!("\nReductions:");
    println!("sum(a) = {}", a.sum_all());
    println!("mean(a) = {}", a.mean_all());
    println!("max(a) = {}", a.max_all());

    // Reshape
    println!("\nReshape:");
    let g = a.reshape(Shape::new(vec![4]));
    println!("a reshaped to 1D: {:?}", g.to_vec());

    println!("\n=== Example complete! ===");
}
