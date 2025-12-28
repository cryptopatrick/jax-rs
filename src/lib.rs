//! # jax-rs: JAX in Rust
//!
//! A machine learning framework for the web, running on WebGPU & Wasm.
//!
//! ## Key Features
//!
//! - **NumPy-compatible API**: Familiar array creation and manipulation
//! - **Automatic differentiation**: `grad`, `vjp`, `jvp` for computing gradients
//! - **Vectorization**: `vmap` for batching operations
//! - **JIT compilation**: Fused kernel execution for performance
//! - **Multiple backends**: CPU (debugging), WebAssembly, WebGPU
//! - **Rust memory safety**: No manual reference counting, automatic cleanup via `Drop`
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use jax_rs::{Array, DType, Shape};
//!
//! // Create arrays
//! let x = Array::zeros(Shape::new(vec![2, 3]), DType::Float32);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod array;
pub mod backend;
mod buffer;
mod device;
mod dtype;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod random;
pub mod scipy;
mod shape;
pub mod trace;

// Public exports
pub use array::Array;
pub use device::{default_device, set_default_device, Device};
pub use dtype::DType;
pub use shape::Shape;
pub use trace::{grad, jit, value_and_grad, vmap};
