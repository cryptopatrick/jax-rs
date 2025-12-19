# jax-rs

> JAX in Rust - A machine learning framework for the web, running on WebGPU & Wasm

[![CI](https://github.com/cryptopatrick/jax-rs/workflows/CI/badge.svg)](https://github.com/cryptopatrick/jax-rs/actions)
[![Crates.io](https://img.shields.io/crates/v/jax-rs.svg)](https://crates.io/crates/jax-rs)
[![Documentation](https://docs.rs/jax-rs/badge.svg)](https://docs.rs/jax-rs)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**jax-rs** is a Rust port of [jax-js](https://jax-js.com), bringing NumPy/JAX-compatible array operations, automatic differentiation, vectorization, and JIT compilation to Rust with WebGPU acceleration.

## Key Features

- ✅ **NumPy-compatible API** - Familiar array operations and broadcasting
- ✅ **Automatic Differentiation** - `grad()`, `vjp()`, `jvp()` for computing gradients
- ✅ **Vectorization** - `vmap()` for automatic batching
- ✅ **JIT Compilation** - Trace and cache computation graphs
- ✅ **Multiple Backends** - CPU (debugging), WebAssembly, WebGPU
- ✅ **Rust Memory Safety** - Automatic cleanup via `Drop`, no manual refcounting

## Quick Start

```rust
use jax_rs::{Array, Shape, DType, grad, vmap};

// Create and manipulate arrays
let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
let result = x.mul(&x).sqrt();

// Automatic differentiation
let f = |x: &Array| x.mul(x).sum_all_array();
let df = grad(f);
let gradient = df(&x);  // [2.0, 4.0, 6.0]

// Vectorization
let square = |x: &Array| x.mul(x);
let batch_square = vmap(square, 0);
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
jax-rs = "0.1"
```

## Examples

```bash
cargo run --example quickstart
cargo run --example basic
```

## Documentation

See [docs.rs/jax-rs](https://docs.rs/jax-rs) for full API documentation.

## Testing

```bash
cargo test           # Run all tests (126 tests)
cargo bench          # Run benchmarks
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.
