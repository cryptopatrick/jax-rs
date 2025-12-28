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
    /// Negation (element-wise)
    Neg,
    /// Absolute value (element-wise)
    Abs,
    /// Sine (element-wise)
    Sin,
    /// Cosine (element-wise)
    Cos,
    /// Tangent (element-wise)
    Tan,
    /// Hyperbolic tangent (element-wise)
    Tanh,
    /// Exponential (element-wise)
    Exp,
    /// Natural logarithm (element-wise)
    Log,
    /// Square root (element-wise)
    Sqrt,
    /// Reciprocal (element-wise)
    Reciprocal,
    /// Square (element-wise)
    Square,
    /// Sign (element-wise)
    Sign,

    // Binary operations
    /// Addition (element-wise)
    Add,
    /// Subtraction (element-wise)
    Sub,
    /// Multiplication (element-wise)
    Mul,
    /// Division (element-wise)
    Div,
    /// Power (element-wise)
    Pow,
    /// Minimum (element-wise)
    Min,
    /// Maximum (element-wise)
    Max,

    // Comparisons
    /// Less than (element-wise)
    Lt,
    /// Less than or equal (element-wise)
    Le,
    /// Greater than (element-wise)
    Gt,
    /// Greater than or equal (element-wise)
    Ge,
    /// Equal (element-wise)
    Eq,
    /// Not equal (element-wise)
    Ne,

    // Reductions
    /// Sum all elements
    SumAll,
    /// Sum along an axis
    Sum {
        /// Axis to reduce over
        axis: usize,
    },
    /// Mean of all elements
    MeanAll,
    /// Mean along an axis
    Mean {
        /// Axis to reduce over
        axis: usize,
    },
    /// Maximum of all elements
    MaxAll,
    /// Maximum along an axis
    MaxAxis {
        /// Axis to reduce over
        axis: usize,
    },
    /// Minimum of all elements
    MinAll,
    /// Minimum along an axis
    MinAxis {
        /// Axis to reduce over
        axis: usize,
    },
    /// Product of all elements
    ProdAll,
    /// Product along an axis
    ProdAxis {
        /// Axis to reduce over
        axis: usize,
    },

    // Linear algebra
    /// Matrix multiplication
    Matmul,
    /// Dot product
    Dot,
    /// Transpose
    Transpose,

    // Shape operations
    /// Reshape to new shape
    Reshape {
        /// Target shape
        new_shape: Vec<usize>,
    },
    /// Remove dimensions of size 1
    Squeeze,
    /// Expand dimensions at axis
    ExpandDims {
        /// Axis to insert new dimension
        axis: usize,
    },

    // Array creation
    /// Create array of zeros
    Zeros {
        /// Output shape
        shape: Vec<usize>,
        /// Output data type
        dtype: DType,
    },
    /// Create array of ones
    Ones {
        /// Output shape
        shape: Vec<usize>,
        /// Output data type
        dtype: DType,
    },
    /// Create array filled with constant value
    Full {
        /// Fill value
        value: f32,
        /// Output shape
        shape: Vec<usize>,
        /// Output data type
        dtype: DType,
    },
    /// Create array with evenly spaced values
    Arange {
        /// Start value
        start: f32,
        /// Stop value (exclusive)
        stop: f32,
        /// Step size
        step: f32,
        /// Output data type
        dtype: DType,
    },
    /// Create array with linearly spaced values
    Linspace {
        /// Start value
        start: f32,
        /// Stop value
        stop: f32,
        /// Number of values
        num: usize,
        /// Whether to include endpoint
        endpoint: bool,
        /// Output data type
        dtype: DType,
    },
    /// Create identity matrix
    Eye {
        /// Number of rows
        n: usize,
        /// Number of columns (default: n)
        m: Option<usize>,
        /// Output data type
        dtype: DType,
    },
}

/// A node in the intermediate representation graph.
///
/// Each node represents either an input, a constant, or an operation.
#[derive(Debug, Clone)]
pub enum IRNode {
    /// Input to the computation (e.g., function argument)
    Input {
        /// Unique identifier for this input
        id: usize,
        /// Shape of the input
        shape: Shape,
        /// Data type of the input
        dtype: DType
    },

    /// Constant value
    Constant {
        /// Constant value
        value: f32,
        /// Data type
        dtype: DType
    },

    /// Materialized array constant (for arrays created during tracing)
    MaterializedConstant {
        /// Array value
        value: Array
    },

    /// Unary operation
    Unary {
        /// Primitive operation
        op: Primitive,
        /// Input operand
        input: Arc<IRNode>,
        /// Output shape
        shape: Shape,
        /// Output data type
        dtype: DType
    },

    /// Binary operation
    Binary {
        /// Primitive operation
        op: Primitive,
        /// Left operand
        lhs: Arc<IRNode>,
        /// Right operand
        rhs: Arc<IRNode>,
        /// Output shape
        shape: Shape,
        /// Output data type
        dtype: DType,
    },

    /// Reduction operation
    Reduce {
        /// Primitive operation
        op: Primitive,
        /// Input operand
        input: Arc<IRNode>,
        /// Output shape
        shape: Shape,
        /// Output data type
        dtype: DType
    },

    /// Fused operation group (multiple element-wise ops in one kernel)
    FusedOp {
        /// Fused group of operations
        group: FusedGroup,
        /// Output shape
        shape: Shape,
        /// Output data type
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
