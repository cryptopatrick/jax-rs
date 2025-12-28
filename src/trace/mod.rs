//! Tracing infrastructure for JIT compilation and transformations.
//!
//! This module provides the foundation for tracing array operations,
//! building an intermediate representation (IR), and enabling
//! transformations like JIT, grad, and vmap.

pub mod fusion;
pub mod grad;
pub mod interpreter;
pub mod ir;
pub mod jit;
pub mod tracer;
pub mod transpose_rules;
pub mod vmap;

pub use fusion::{analyze as analyze_fusion, optimize as optimize_fusion, FusionStats};
pub use grad::{grad, value_and_grad, JVPEngine, VJPEngine};
pub use interpreter::Interpreter;
pub use ir::{FusedGroup, IRGraph, IRNode, Primitive};
pub use jit::jit;
pub use tracer::{
    enter_trace, exit_trace, is_tracing, trace_binary, trace_reduce,
    trace_unary, with_trace, TraceContext,
};
pub use transpose_rules::PrimalValue;
pub use vmap::{vmap, vmap2, VmapConfig};
