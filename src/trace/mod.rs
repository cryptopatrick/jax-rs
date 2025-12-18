//! Tracing infrastructure for JIT compilation and transformations.
//!
//! This module provides the foundation for tracing array operations,
//! building an intermediate representation (IR), and enabling
//! transformations like JIT, grad, and vmap.

pub mod grad;
pub mod interpreter;
pub mod ir;
pub mod jit;
pub mod tracer;

pub use grad::{grad, JVPEngine, VJPEngine};
pub use interpreter::Interpreter;
pub use ir::{IRGraph, IRNode, Primitive};
pub use jit::jit;
pub use tracer::{
    enter_trace, exit_trace, is_tracing, trace_binary, trace_reduce, trace_unary, with_trace,
    TraceContext,
};
