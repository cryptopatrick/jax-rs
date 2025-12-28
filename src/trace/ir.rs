//! Intermediate representation for traced operations.

use crate::{Array, DType, Shape};
use std::sync::Arc;

/// A fused operation group containing multiple fusible operations.
#[derive(Debug, Clone)]
pub struct FusedGroup {
    /// Operations in this group (in topological order)
    pub operations: Vec<Arc<IRNode>>,
    /// Input nodes to this group (from outside the group)
    pub inputs: Vec<Arc<IRNode>>,
    /// Output nodes from this group (used outside the group)
    pub outputs: Vec<Arc<IRNode>>,
    /// Human-readable name for debugging
    pub name: String,
}

/// Primitive operations that can be traced and compiled.
///
/// These correspond to the fundamental operations in the jax-rs library.
#[derive(Debug, Clone, PartialEq)]
pub enum Primitive {
    // Unary operations
    Neg,
    Abs,
    Sin,
    Cos,
    Tan,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Reciprocal,
    Square,
    Sign,

    // Binary operations
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,

    // Comparisons
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,

    // Reductions
    SumAll,
    Sum {
        axis: usize,
    },
    MeanAll,
    Mean {
        axis: usize,
    },
    MaxAll,
    MaxAxis {
        axis: usize,
    },
    MinAll,
    MinAxis {
        axis: usize,
    },
    ProdAll,
    ProdAxis {
        axis: usize,
    },

    // Linear algebra
    Matmul,
    Dot,
    Transpose,

    // Shape operations
    Reshape {
        new_shape: Vec<usize>,
    },
    Squeeze,
    ExpandDims {
        axis: usize,
    },

    // Array creation
    Zeros {
        shape: Vec<usize>,
        dtype: DType,
    },
    Ones {
        shape: Vec<usize>,
        dtype: DType,
    },
    Full {
        value: f32,
        shape: Vec<usize>,
        dtype: DType,
    },
    Arange {
        start: f32,
        stop: f32,
        step: f32,
        dtype: DType,
    },
    Linspace {
        start: f32,
        stop: f32,
        num: usize,
        endpoint: bool,
        dtype: DType,
    },
    Eye {
        n: usize,
        m: Option<usize>,
        dtype: DType,
    },
}

/// A node in the intermediate representation graph.
///
/// Each node represents either an input, a constant, or an operation.
#[derive(Debug, Clone)]
pub enum IRNode {
    /// Input to the computation (e.g., function argument)
    Input { id: usize, shape: Shape, dtype: DType },

    /// Constant value
    Constant { value: f32, dtype: DType },

    /// Materialized array constant (for arrays created during tracing)
    MaterializedConstant { value: Array },

    /// Unary operation
    Unary { op: Primitive, input: Arc<IRNode>, shape: Shape, dtype: DType },

    /// Binary operation
    Binary {
        op: Primitive,
        lhs: Arc<IRNode>,
        rhs: Arc<IRNode>,
        shape: Shape,
        dtype: DType,
    },

    /// Reduction operation
    Reduce { op: Primitive, input: Arc<IRNode>, shape: Shape, dtype: DType },

    /// Fused operation group (multiple element-wise ops in one kernel)
    FusedOp {
        group: FusedGroup,
        shape: Shape,
        dtype: DType,
    },
}

impl IRNode {
    /// Get the shape of this node's output.
    pub fn shape(&self) -> Shape {
        match self {
            IRNode::Input { shape, .. }
            | IRNode::Unary { shape, .. }
            | IRNode::Binary { shape, .. }
            | IRNode::Reduce { shape, .. }
            | IRNode::FusedOp { shape, .. } => shape.clone(),
            IRNode::Constant { .. } => Shape::scalar(),
            IRNode::MaterializedConstant { value } => value.shape().clone(),
        }
    }

    /// Get the dtype of this node's output.
    pub fn dtype(&self) -> DType {
        match self {
            IRNode::Input { dtype, .. }
            | IRNode::Constant { dtype, .. }
            | IRNode::Unary { dtype, .. }
            | IRNode::Binary { dtype, .. }
            | IRNode::Reduce { dtype, .. }
            | IRNode::FusedOp { dtype, .. } => *dtype,
            IRNode::MaterializedConstant { value } => value.dtype(),
        }
    }

    /// Create an input node.
    pub fn input(id: usize, shape: Shape, dtype: DType) -> Arc<Self> {
        Arc::new(IRNode::Input { id, shape, dtype })
    }

    /// Create a constant node.
    pub fn constant(value: f32, dtype: DType) -> Arc<Self> {
        Arc::new(IRNode::Constant { value, dtype })
    }

    /// Create a materialized constant node from an array.
    pub fn materialized_constant(value: Array) -> Arc<Self> {
        Arc::new(IRNode::MaterializedConstant { value })
    }

    /// Create a unary operation node.
    pub fn unary(op: Primitive, input: Arc<IRNode>) -> Arc<Self> {
        let shape = input.shape();
        let dtype = input.dtype();
        Arc::new(IRNode::Unary { op, input, shape, dtype })
    }

    /// Create a binary operation node.
    pub fn binary(
        op: Primitive,
        lhs: Arc<IRNode>,
        rhs: Arc<IRNode>,
    ) -> Arc<Self> {
        // For now, assume compatible shapes (broadcasting handled at runtime)
        let shape = lhs.shape();
        let dtype = lhs.dtype();
        Arc::new(IRNode::Binary { op, lhs, rhs, shape, dtype })
    }

    /// Create a reduction operation node.
    pub fn reduce(
        op: Primitive,
        input: Arc<IRNode>,
        result_shape: Shape,
    ) -> Arc<Self> {
        let dtype = input.dtype();
        Arc::new(IRNode::Reduce { op, input, shape: result_shape, dtype })
    }

    /// Create a fused operation node.
    pub fn fused_op(group: FusedGroup, shape: Shape, dtype: DType) -> Arc<Self> {
        Arc::new(IRNode::FusedOp { group, shape, dtype })
    }
}

/// A complete computation graph.
///
/// Represents a traced function as a directed acyclic graph (DAG)
/// of operations.
#[derive(Debug, Clone)]
pub struct IRGraph {
    /// Input nodes (function parameters)
    pub inputs: Vec<Arc<IRNode>>,
    /// Output nodes (function results)
    pub outputs: Vec<Arc<IRNode>>,
    /// Human-readable name for debugging
    pub name: String,
}

impl IRGraph {
    /// Create a new IR graph.
    pub fn new(
        name: String,
        inputs: Vec<Arc<IRNode>>,
        outputs: Vec<Arc<IRNode>>,
    ) -> Self {
        Self { inputs, outputs, name }
    }

    /// Get the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Get the number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_node_creation() {
        let input = IRNode::input(0, Shape::new(vec![2, 3]), DType::Float32);
        assert_eq!(input.shape().as_slice(), &[2, 3]);
        assert_eq!(input.dtype(), DType::Float32);
    }

    #[test]
    fn test_ir_unary() {
        let input = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let neg = IRNode::unary(Primitive::Neg, input);
        assert_eq!(neg.shape().as_slice(), &[3]);
        assert_eq!(neg.dtype(), DType::Float32);
    }

    #[test]
    fn test_ir_binary() {
        let input1 = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let input2 = IRNode::input(1, Shape::new(vec![3]), DType::Float32);
        let add = IRNode::binary(Primitive::Add, input1, input2);
        assert_eq!(add.shape().as_slice(), &[3]);
    }

    #[test]
    fn test_ir_graph() {
        let input1 = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let input2 = IRNode::input(1, Shape::new(vec![3]), DType::Float32);
        let add =
            IRNode::binary(Primitive::Add, input1.clone(), input2.clone());

        let graph = IRGraph::new(
            "test_add".to_string(),
            vec![input1, input2],
            vec![add],
        );

        assert_eq!(graph.num_inputs(), 2);
        assert_eq!(graph.num_outputs(), 1);
    }
}
