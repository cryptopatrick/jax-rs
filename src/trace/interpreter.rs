//! CPU interpreter for executing IR graphs.
//!
//! Takes a traced IR graph and executes it eagerly on the CPU.

use crate::trace::{IRGraph, IRNode, Primitive};
use crate::Array;
use std::collections::HashMap;
use std::sync::Arc;

/// Interprets and executes an IR graph on the CPU.
pub struct Interpreter {
    /// Cache of evaluated nodes (node address -> result)
    cache: HashMap<usize, Array>,
}

impl Interpreter {
    /// Create a new interpreter.
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    /// Execute an IR graph with the given inputs.
    ///
    /// Returns the output arrays.
    pub fn execute(
        &mut self,
        graph: &IRGraph,
        inputs: &[Array],
    ) -> Vec<Array> {
        // Clear cache for fresh execution
        self.cache.clear();

        // Populate cache with inputs
        for (i, (input_node, input_array)) in
            graph.inputs.iter().zip(inputs.iter()).enumerate()
        {
            // Verify input matches expected shape/dtype
            if let IRNode::Input { id, shape, dtype } = input_node.as_ref() {
                assert_eq!(*id, i, "Input ID mismatch");
                assert_eq!(input_array.shape(), shape, "Input shape mismatch");
                assert_eq!(
                    &input_array.dtype(),
                    dtype,
                    "Input dtype mismatch"
                );

                let node_addr = Arc::as_ptr(input_node) as usize;
                self.cache.insert(node_addr, input_array.clone());
            } else {
                panic!("Expected Input node in graph.inputs");
            }
        }

        // Evaluate output nodes
        graph.outputs.iter().map(|node| self.eval_node(node)).collect()
    }

    /// Evaluate a single IR node recursively.
    fn eval_node(&mut self, node: &Arc<IRNode>) -> Array {
        let node_addr = Arc::as_ptr(node) as usize;

        // Check cache
        if let Some(result) = self.cache.get(&node_addr) {
            return result.clone();
        }

        // Evaluate based on node type
        let result = match node.as_ref() {
            IRNode::Input { .. } => {
                panic!("Input node not found in cache");
            }

            IRNode::Constant { value, dtype } => {
                Array::full(*value, crate::Shape::scalar(), *dtype)
            }

            IRNode::MaterializedConstant { value } => {
                value.clone()
            }

            IRNode::Unary { op, input, .. } => {
                let input_val = self.eval_node(input);
                self.eval_unary(op, &input_val)
            }

            IRNode::Binary { op, lhs, rhs, .. } => {
                let lhs_val = self.eval_node(lhs);
                let rhs_val = self.eval_node(rhs);
                self.eval_binary(op, &lhs_val, &rhs_val)
            }

            IRNode::Reduce { op, input, .. } => {
                let input_val = self.eval_node(input);
                self.eval_reduce(op, &input_val)
            }

            IRNode::FusedOp { group, shape, dtype } => {
                self.eval_fused_op(group, shape, dtype)
            }
        };

        // Cache the result
        self.cache.insert(node_addr, result.clone());
        result
    }

    /// Evaluate a unary operation.
    fn eval_unary(&self, op: &Primitive, input: &Array) -> Array {
        match op {
            Primitive::Neg => input.neg(),
            Primitive::Abs => input.abs(),
            Primitive::Sin => input.sin(),
            Primitive::Cos => input.cos(),
            Primitive::Tan => input.tan(),
            Primitive::Tanh => input.tanh(),
            Primitive::Exp => input.exp(),
            Primitive::Log => input.log(),
            Primitive::Sqrt => input.sqrt(),
            Primitive::Reciprocal => input.reciprocal(),
            Primitive::Square => input.square(),
            _ => panic!("Unsupported unary operation: {:?}", op),
        }
    }

    /// Evaluate a binary operation.
    fn eval_binary(&self, op: &Primitive, lhs: &Array, rhs: &Array) -> Array {
        match op {
            Primitive::Add => lhs.add(rhs),
            Primitive::Sub => lhs.sub(rhs),
            Primitive::Mul => lhs.mul(rhs),
            Primitive::Div => lhs.div(rhs),
            Primitive::Pow => lhs.pow(rhs),
            Primitive::Min => lhs.minimum(rhs),
            Primitive::Max => lhs.maximum(rhs),
            Primitive::Lt => lhs.lt(rhs),
            Primitive::Le => lhs.le(rhs),
            Primitive::Gt => lhs.gt(rhs),
            Primitive::Ge => lhs.ge(rhs),
            Primitive::Eq => lhs.eq(rhs),
            Primitive::Ne => lhs.ne(rhs),
            Primitive::Matmul => lhs.matmul(rhs),
            Primitive::Dot => lhs.dot(rhs),
            _ => panic!("Unsupported binary operation: {:?}", op),
        }
    }

    /// Evaluate a reduction operation.
    fn eval_reduce(&self, op: &Primitive, input: &Array) -> Array {
        use crate::Shape;

        match op {
            Primitive::SumAll => {
                let val = input.sum_all();
                Array::from_vec(vec![val], Shape::scalar())
            }
            Primitive::Sum { axis } => input.sum(*axis),
            Primitive::MeanAll => {
                let val = input.mean_all();
                Array::from_vec(vec![val], Shape::scalar())
            }
            Primitive::Mean { axis } => input.mean(*axis),
            Primitive::MaxAll => {
                let val = input.max_all();
                Array::from_vec(vec![val], Shape::scalar())
            }
            Primitive::MaxAxis { axis } => input.max(*axis),
            Primitive::MinAll => {
                let val = input.min_all();
                Array::from_vec(vec![val], Shape::scalar())
            }
            Primitive::MinAxis { axis } => input.min(*axis),
            _ => panic!("Unsupported reduction operation: {:?}", op),
        }
    }

    /// Evaluate a fused operation group.
    fn eval_fused_op(
        &mut self,
        group: &crate::trace::FusedGroup,
        output_shape: &crate::Shape,
        output_dtype: &crate::DType,
    ) -> Array {
        // Evaluate all inputs
        let input_arrays: Vec<Array> = group
            .inputs
            .iter()
            .map(|node| self.eval_node(node))
            .collect();

        // Check if GPU execution is possible
        let all_gpu = input_arrays
            .iter()
            .all(|arr| arr.device() == crate::Device::WebGpu);

        if all_gpu {
            // GPU execution path
            use crate::backend::ops::gpu_fused_execute;

            // Allocate output
            let output = Array::zeros(output_shape.clone(), *output_dtype)
                .to_device(crate::Device::WebGpu);

            // Get buffer references
            let input_refs: Vec<&crate::buffer::Buffer> =
                input_arrays.iter().map(|arr| arr.buffer()).collect();
            let output_ref = output.buffer();

            // Execute fused kernel
            gpu_fused_execute(group, &input_refs, &[output_ref]);

            output
        } else {
            // CPU fallback: execute last operation in group
            // (Full CPU fusion could be added later)
            if let Some(final_op) = group.operations.last() {
                self.eval_node(final_op)
            } else {
                Array::zeros(output_shape.clone(), *output_dtype)
            }
        }
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape};

    #[test]
    fn test_interpreter_unary() {
        let mut interp = Interpreter::new();

        // Build graph: neg(input0)
        let input = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let neg = IRNode::unary(Primitive::Neg, input.clone());

        let graph =
            IRGraph::new("test_neg".to_string(), vec![input], vec![neg]);

        let input_data =
            Array::from_vec(vec![1.0, -2.0, 3.0], Shape::new(vec![3]));
        let outputs = interp.execute(&graph, &[input_data]);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].to_vec(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_interpreter_binary() {
        let mut interp = Interpreter::new();

        // Build graph: add(input0, input1)
        let input0 = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![3]), DType::Float32);
        let add =
            IRNode::binary(Primitive::Add, input0.clone(), input1.clone());

        let graph = IRGraph::new(
            "test_add".to_string(),
            vec![input0, input1],
            vec![add],
        );

        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
        let outputs = interp.execute(&graph, &[a, b]);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_interpreter_complex() {
        let mut interp = Interpreter::new();

        // Build graph: mul(add(input0, input1), input0)
        let input0 = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![3]), DType::Float32);
        let add =
            IRNode::binary(Primitive::Add, input0.clone(), input1.clone());
        let mul = IRNode::binary(Primitive::Mul, add, input0.clone());

        let graph = IRGraph::new(
            "test_complex".to_string(),
            vec![input0, input1],
            vec![mul],
        );

        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![3.0, 2.0, 1.0], Shape::new(vec![3]));
        let outputs = interp.execute(&graph, &[a, b]);

        assert_eq!(outputs.len(), 1);
        // (1+3)*1=4, (2+2)*2=8, (3+1)*3=12
        assert_eq!(outputs[0].to_vec(), vec![4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_interpreter_reduce() {
        let mut interp = Interpreter::new();

        // Build graph: sum_all(input0)
        let input = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let sum =
            IRNode::reduce(Primitive::SumAll, input.clone(), Shape::scalar());

        let graph =
            IRGraph::new("test_sum".to_string(), vec![input], vec![sum]);

        let input_data =
            Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let outputs = interp.execute(&graph, &[input_data]);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].to_vec(), vec![6.0]);
    }

    #[test]
    fn test_interpreter_constant() {
        let mut interp = Interpreter::new();

        // Build graph: add(input0, constant(5.0))
        let input = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let constant = IRNode::constant(5.0, DType::Float32);
        let add = IRNode::binary(Primitive::Add, input.clone(), constant);

        let graph =
            IRGraph::new("test_const".to_string(), vec![input], vec![add]);

        let input_data = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
        let outputs = interp.execute(&graph, &[input_data]);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].to_vec(), vec![6.0, 7.0]);
    }
}
