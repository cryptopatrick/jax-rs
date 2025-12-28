//! Tracing infrastructure for capturing operations.
//!
//! This module provides the `TraceContext` that records operations
//! during execution to build an intermediate representation.

use crate::trace::{IRNode, Primitive};
use crate::{Array, DType, Shape};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

thread_local! {
    /// Global tracing context (thread-local).
    static TRACE_CONTEXT: RefCell<Option<Rc<RefCell<TraceContext>>>> = const { RefCell::new(None) };
}

/// Context for tracing operations.
///
/// Captures operations as they execute and builds an IR graph.
#[derive(Debug)]
pub struct TraceContext {
    /// Mapping from Array IDs to their IR nodes
    nodes: HashMap<usize, Arc<IRNode>>,
    /// Input nodes for this trace
    inputs: Vec<Arc<IRNode>>,
    /// Counter for generating unique IDs
    _next_id: usize,
    /// Name of the function being traced
    name: String,
}

impl TraceContext {
    /// Create a new trace context.
    pub fn new(name: String) -> Self {
        Self { nodes: HashMap::new(), inputs: Vec::new(), _next_id: 0, name }
    }

    /// Generate a unique ID for a traced value.
    fn _next_id(&mut self) -> usize {
        let id = self._next_id;
        self._next_id += 1;
        id
    }

    /// Register an input array.
    pub fn register_input(
        &mut self,
        array_id: usize,
        shape: Shape,
        dtype: DType,
    ) -> Arc<IRNode> {
        let input_id = self.inputs.len();
        let node = IRNode::input(input_id, shape, dtype);
        self.nodes.insert(array_id, node.clone());
        self.inputs.push(node.clone());
        node
    }

    /// Register a constant value.
    pub fn register_constant(
        &mut self,
        array_id: usize,
        value: f32,
        dtype: DType,
    ) -> Arc<IRNode> {
        let node = IRNode::constant(value, dtype);
        self.nodes.insert(array_id, node.clone());
        node
    }

    /// Register a unary operation.
    pub fn register_unary(
        &mut self,
        result_id: usize,
        op: Primitive,
        input: &Array,
    ) -> Arc<IRNode> {
        let input_id = Self::array_id(input);

        // Auto-register constants if not found
        let input_node = self.nodes.get(&input_id).cloned().unwrap_or_else(|| {
            let const_node = IRNode::materialized_constant(input.clone());
            self.nodes.insert(input_id, const_node.clone());
            const_node
        });

        let node = IRNode::unary(op, input_node);
        self.nodes.insert(result_id, node.clone());
        node
    }

    /// Register a binary operation.
    pub fn register_binary(
        &mut self,
        result_id: usize,
        op: Primitive,
        lhs: &Array,
        rhs: &Array,
    ) -> Arc<IRNode> {
        let lhs_id = Self::array_id(lhs);
        let rhs_id = Self::array_id(rhs);

        // Auto-register constants if not found
        let lhs_node = self.nodes.get(&lhs_id).cloned().unwrap_or_else(|| {
            let const_node = IRNode::materialized_constant(lhs.clone());
            self.nodes.insert(lhs_id, const_node.clone());
            const_node
        });
        let rhs_node = self.nodes.get(&rhs_id).cloned().unwrap_or_else(|| {
            let const_node = IRNode::materialized_constant(rhs.clone());
            self.nodes.insert(rhs_id, const_node.clone());
            const_node
        });

        let node = IRNode::binary(op, lhs_node, rhs_node);
        self.nodes.insert(result_id, node.clone());
        node
    }

    /// Register a reduction operation.
    pub fn register_reduce(
        &mut self,
        result_id: usize,
        op: Primitive,
        input: &Array,
        result_shape: Shape,
    ) -> Arc<IRNode> {
        let input_id = Self::array_id(input);
        let input_node = self
            .nodes
            .get(&input_id)
            .expect("Input array not found in trace context")
            .clone();

        let node = IRNode::reduce(op, input_node, result_shape);
        self.nodes.insert(result_id, node.clone());
        node
    }

    /// Get the IR node for an array.
    pub fn get_node(&self, array_id: usize) -> Option<Arc<IRNode>> {
        self.nodes.get(&array_id).cloned()
    }

    /// Get a unique ID for an array (using its internal ID).
    fn array_id(array: &Array) -> usize {
        array.id()
    }

    /// Finalize the trace and return the IR graph.
    pub fn finalize(self, outputs: &[Array]) -> crate::trace::IRGraph {
        let output_nodes: Vec<Arc<IRNode>> = outputs
            .iter()
            .map(|arr| {
                let id = Self::array_id(arr);
                self.nodes
                    .get(&id)
                    .expect("Output array not found in trace context")
                    .clone()
            })
            .collect();

        crate::trace::IRGraph::new(self.name, self.inputs, output_nodes)
    }
}

/// Enter tracing mode with the given context.
pub fn enter_trace(ctx: Rc<RefCell<TraceContext>>) {
    TRACE_CONTEXT.with(|trace_ctx| {
        *trace_ctx.borrow_mut() = Some(ctx);
    });
}

/// Exit tracing mode.
pub fn exit_trace() {
    TRACE_CONTEXT.with(|trace_ctx| {
        *trace_ctx.borrow_mut() = None;
    });
}

/// Check if we're currently tracing.
pub fn is_tracing() -> bool {
    TRACE_CONTEXT.with(|trace_ctx| trace_ctx.borrow().is_some())
}

/// Execute a function with tracing enabled.
pub fn with_trace<F, R>(name: String, f: F) -> (R, TraceContext)
where
    F: FnOnce() -> R,
{
    let ctx = Rc::new(RefCell::new(TraceContext::new(name)));
    enter_trace(ctx.clone());
    let result = f();
    exit_trace();

    let ctx = Rc::try_unwrap(ctx)
        .expect("TraceContext still has references")
        .into_inner();

    (result, ctx)
}

/// Register a unary operation if tracing is active.
pub fn trace_unary(result_id: usize, op: Primitive, input: &Array) {
    TRACE_CONTEXT.with(|trace_ctx| {
        if let Some(ctx) = trace_ctx.borrow().as_ref() {
            ctx.borrow_mut().register_unary(result_id, op, input);
        }
    });
}

/// Register a binary operation if tracing is active.
pub fn trace_binary(
    result_id: usize,
    op: Primitive,
    lhs: &Array,
    rhs: &Array,
) {
    TRACE_CONTEXT.with(|trace_ctx| {
        if let Some(ctx) = trace_ctx.borrow().as_ref() {
            ctx.borrow_mut().register_binary(result_id, op, lhs, rhs);
        }
    });
}

/// Register a reduction operation if tracing is active.
pub fn trace_reduce(
    result_id: usize,
    op: Primitive,
    input: &Array,
    result_shape: Shape,
) {
    TRACE_CONTEXT.with(|trace_ctx| {
        if let Some(ctx) = trace_ctx.borrow().as_ref() {
            ctx.borrow_mut().register_reduce(
                result_id,
                op,
                input,
                result_shape,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_context_creation() {
        let ctx = TraceContext::new("test".to_string());
        assert_eq!(ctx.name, "test");
        assert_eq!(ctx.inputs.len(), 0);
    }

    #[test]
    fn test_is_tracing() {
        assert!(!is_tracing());

        let ctx = Rc::new(RefCell::new(TraceContext::new("test".to_string())));
        enter_trace(ctx);
        assert!(is_tracing());

        exit_trace();
        assert!(!is_tracing());
    }

    #[test]
    fn test_with_trace() {
        assert!(!is_tracing());

        let (result, _ctx) = with_trace("test".to_string(), || {
            assert!(is_tracing());
            42
        });

        assert_eq!(result, 42);
        assert!(!is_tracing());
    }
}
