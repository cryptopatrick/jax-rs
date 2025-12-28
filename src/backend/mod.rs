//! Backend implementations for different compute devices.
//!
//! This module contains device-specific implementations for:
//! - CPU: Standard Rust Vec-based operations
//! - WebGPU: GPU-accelerated compute via wgpu crate
//! - WASM: WebAssembly with SIMD (future)

pub mod ops;
pub mod shaders;
pub mod webgpu;
