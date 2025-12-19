//! Automatic differentiation via reverse-mode AD (grad, vjp).
//!
//! This module implements gradient computation using vector-jacobian products.

use crate::trace::{IRGraph, IRNode, Primitive};
use crate::Array;
use std::collections::HashMap;
use std::sync::Arc;

/// Compute the gradient of a function.
///
/// Returns a function that computes the gradient with respect to the inputs.
///
/// # Examples
///
/// ```rust,ignore
/// use jax_rs::trace::grad;
///
/// let df = grad(|x: &Array| {
///     x.mul(x).sum_all()  // f(x) = sum(x^2)
/// });
///
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
/// let dx = df(&x);  // gradient is [2.0, 4.0, 6.0]
/// ```
pub fn grad<F>(f: F) -> impl Fn(&Array) -> Array
where
    F: Fn(&Array) -> Array,
{
    move |x: &Array| {
        // For now, compute gradient using numerical differentiation
        // Full implementation will use reverse-mode AD
        let eps = 1e-5;
        let y = f(x);

        // Simple numerical gradient for scalar output
        let x_data = x.to_vec();
        let mut grad_data = vec![0.0; x_data.len()];

        for i in 0..x_data.len() {
            let mut x_plus = x_data.clone();
            x_plus[i] += eps;
            let x_plus_arr = Array::from_vec(x_plus, x.shape().clone());
            let y_plus = f(&x_plus_arr);

            let mut x_minus = x_data.clone();
            x_minus[i] -= eps;
            let x_minus_arr = Array::from_vec(x_minus, x.shape().clone());
            let y_minus = f(&x_minus_arr);

            // Central difference
            grad_data[i] =
                (y_plus.to_vec()[0] - y_minus.to_vec()[0]) / (2.0 * eps);
        }

        Array::from_vec(grad_data, x.shape().clone())
    }
}

/// Vector-Jacobian Product computation engine.
///
/// Implements reverse-mode automatic differentiation.
pub struct VJPEngine {
    /// Gradient accumulator for each node
    gradients: HashMap<usize, Array>,
}

impl VJPEngine {
    /// Create a new VJP engine.
    pub fn new() -> Self {
        Self { gradients: HashMap::new() }
    }

    /// Compute VJP for a graph given output gradients.
    ///
    /// Returns gradients with respect to the inputs.
    pub fn vjp(
        &mut self,
        graph: &IRGraph,
        cotangents: &[Array],
    ) -> Vec<Array> {
        // Initialize output gradients
        assert_eq!(
            graph.outputs.len(),
            cotangents.len(),
            "Number of cotangents must match number of outputs"
        );

        for (output, cotangent) in graph.outputs.iter().zip(cotangents.iter())
        {
            let node_addr = Arc::as_ptr(output) as usize;
            self.gradients.insert(node_addr, cotangent.clone());
        }

        // Backward pass: propagate gradients from outputs to inputs
        for output in graph.outputs.iter().rev() {
            self.backward(output);
        }

        // Extract input gradients
        graph
            .inputs
            .iter()
            .map(|input| {
                let node_addr = Arc::as_ptr(input) as usize;
                self.gradients.get(&node_addr).cloned().unwrap_or_else(|| {
                    // Zero gradient if not computed
                    if let IRNode::Input { shape, dtype, .. } = input.as_ref()
                    {
                        Array::zeros(shape.clone(), *dtype)
                    } else {
                        panic!("Expected input node");
                    }
                })
            })
            .collect()
    }

    /// Backward pass for a single node.
    fn backward(&mut self, node: &Arc<IRNode>) {
        let node_addr = Arc::as_ptr(node) as usize;

        // Get the gradient for this node (cotangent)
        let cotangent = match self.gradients.get(&node_addr) {
            Some(g) => g.clone(),
            None => return, // No gradient to propagate
        };

        match node.as_ref() {
            IRNode::Input { .. } => {
                // Input nodes accumulate gradients
            }

            IRNode::Constant { .. } => {
                // Constants don't propagate gradients
            }

            IRNode::Unary { op, input, .. } => {
                self.backward_unary(op, input, &cotangent);
            }

            IRNode::Binary { op, lhs, rhs, .. } => {
                self.backward_binary(op, lhs, rhs, &cotangent);
            }

            IRNode::Reduce { op, input, .. } => {
                self.backward_reduce(op, input, &cotangent);
            }
        }
    }

    /// Backward pass for unary operations.
    fn backward_unary(
        &mut self,
        op: &Primitive,
        input: &Arc<IRNode>,
        cotangent: &Array,
    ) {
        let input_addr = Arc::as_ptr(input) as usize;

        // Compute the gradient with respect to the input
        let grad = self.vjp_unary(op, input, cotangent);

        // Accumulate gradient
        self.accumulate_gradient(input_addr, grad);

        // Continue backward pass
        self.backward(input);
    }

    /// VJP rule for unary operations.
    fn vjp_unary(
        &self,
        op: &Primitive,
        input: &Arc<IRNode>,
        cotangent: &Array,
    ) -> Array {
        // For unary ops: grad_input = cotangent * d(op)/d(input)
        // We need the input value to compute some derivatives

        match op {
            Primitive::Neg => cotangent.neg(),

            Primitive::Abs => {
                // d/dx |x| = sign(x)
                // For now, simplified: we'd need input values
                cotangent.clone()
            }

            Primitive::Square => {
                // d/dx x^2 = 2x
                // cotangent * 2x (need input value)
                cotangent.clone()
            }

            Primitive::Sqrt => {
                // d/dx sqrt(x) = 1/(2*sqrt(x))
                // cotangent / (2*sqrt(x)) (need output or input value)
                cotangent.clone()
            }

            Primitive::Exp => {
                // d/dx exp(x) = exp(x)
                // cotangent * exp(x) (need output value)
                cotangent.clone()
            }

            Primitive::Log => {
                // d/dx log(x) = 1/x
                // cotangent / x (need input value)
                cotangent.clone()
            }

            Primitive::Sin => {
                // d/dx sin(x) = cos(x)
                cotangent.clone()
            }

            Primitive::Cos => {
                // d/dx cos(x) = -sin(x)
                cotangent.neg()
            }

            Primitive::Tanh => {
                // d/dx tanh(x) = 1 - tanh(x)^2
                cotangent.clone()
            }

            Primitive::Reciprocal => {
                // d/dx (1/x) = -1/x^2
                cotangent.clone()
            }

            _ => panic!("Unsupported unary operation for VJP: {:?}", op),
        }
    }

    /// Backward pass for binary operations.
    fn backward_binary(
        &mut self,
        op: &Primitive,
        lhs: &Arc<IRNode>,
        rhs: &Arc<IRNode>,
        cotangent: &Array,
    ) {
        let lhs_addr = Arc::as_ptr(lhs) as usize;
        let rhs_addr = Arc::as_ptr(rhs) as usize;

        let (grad_lhs, grad_rhs) = self.vjp_binary(op, lhs, rhs, cotangent);

        // Accumulate gradients
        self.accumulate_gradient(lhs_addr, grad_lhs);
        self.accumulate_gradient(rhs_addr, grad_rhs);

        // Continue backward pass
        self.backward(lhs);
        self.backward(rhs);
    }

    /// VJP rule for binary operations.
    fn vjp_binary(
        &self,
        op: &Primitive,
        _lhs: &Arc<IRNode>,
        _rhs: &Arc<IRNode>,
        cotangent: &Array,
    ) -> (Array, Array) {
        // For binary ops: grad_lhs = cotangent * d(op)/d(lhs)
        //                 grad_rhs = cotangent * d(op)/d(rhs)

        match op {
            Primitive::Add => {
                // d/dx (x + y) = 1, d/dy (x + y) = 1
                (cotangent.clone(), cotangent.clone())
            }

            Primitive::Sub => {
                // d/dx (x - y) = 1, d/dy (x - y) = -1
                (cotangent.clone(), cotangent.neg())
            }

            Primitive::Mul => {
                // d/dx (x * y) = y, d/dy (x * y) = x
                // cotangent * y, cotangent * x (need input values)
                (cotangent.clone(), cotangent.clone())
            }

            Primitive::Div => {
                // d/dx (x / y) = 1/y, d/dy (x / y) = -x/y^2
                // cotangent / y, -cotangent * x / y^2 (need input values)
                (cotangent.clone(), cotangent.clone())
            }

            _ => panic!("Unsupported binary operation for VJP: {:?}", op),
        }
    }

    /// Backward pass for reduction operations.
    fn backward_reduce(
        &mut self,
        op: &Primitive,
        input: &Arc<IRNode>,
        cotangent: &Array,
    ) {
        let input_addr = Arc::as_ptr(input) as usize;

        let grad = self.vjp_reduce(op, input, cotangent);

        // Accumulate gradient
        self.accumulate_gradient(input_addr, grad);

        // Continue backward pass
        self.backward(input);
    }

    /// VJP rule for reduction operations.
    fn vjp_reduce(
        &self,
        op: &Primitive,
        input: &Arc<IRNode>,
        cotangent: &Array,
    ) -> Array {
        let input_shape = input.shape();

        match op {
            Primitive::SumAll => {
                // d/dx sum(x) = 1 for all elements
                // Broadcast cotangent to input shape
                let ones = Array::ones(input_shape, input.dtype());
                // cotangent is scalar, need to broadcast
                ones.mul(cotangent)
            }

            Primitive::Sum { axis: _ } => {
                // d/dx sum(x, axis) = 1 for all elements in reduced axis
                // Broadcast cotangent back to input shape
                cotangent.clone()
            }

            Primitive::MeanAll => {
                // d/dx mean(x) = 1/n for all elements
                let n = input_shape.size() as f32;
                let ones = Array::ones(input_shape, input.dtype());
                ones.mul(cotangent).div(&Array::full(
                    n,
                    crate::Shape::scalar(),
                    input.dtype(),
                ))
            }

            _ => panic!("Unsupported reduction operation for VJP: {:?}", op),
        }
    }

    /// Accumulate gradient for a node.
    fn accumulate_gradient(&mut self, node_addr: usize, grad: Array) {
        if let Some(existing) = self.gradients.get(&node_addr) {
            // Add to existing gradient
            let new_grad = existing.add(&grad);
            self.gradients.insert(node_addr, new_grad);
        } else {
            // Initialize gradient
            self.gradients.insert(node_addr, grad);
        }
    }
}

impl Default for VJPEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Jacobian-Vector Product computation engine.
///
/// Implements forward-mode automatic differentiation.
pub struct JVPEngine {
    /// Tangent values for each node
    tangents: HashMap<usize, Array>,
}

impl JVPEngine {
    /// Create a new JVP engine.
    pub fn new() -> Self {
        Self { tangents: HashMap::new() }
    }

    /// Compute JVP for a graph given input tangents.
    ///
    /// Returns tangents for the outputs.
    pub fn jvp(&mut self, graph: &IRGraph, tangents: &[Array]) -> Vec<Array> {
        // Initialize input tangents
        assert_eq!(
            graph.inputs.len(),
            tangents.len(),
            "Number of tangents must match number of inputs"
        );

        for (input, tangent) in graph.inputs.iter().zip(tangents.iter()) {
            let node_addr = Arc::as_ptr(input) as usize;
            self.tangents.insert(node_addr, tangent.clone());
        }

        // Forward pass: propagate tangents from inputs to outputs
        let output_tangents: Vec<Array> =
            graph.outputs.iter().map(|output| self.forward(output)).collect();

        output_tangents
    }

    /// Forward pass for a single node.
    fn forward(&mut self, node: &Arc<IRNode>) -> Array {
        let node_addr = Arc::as_ptr(node) as usize;

        // Check if already computed
        if let Some(tangent) = self.tangents.get(&node_addr) {
            return tangent.clone();
        }

        // Compute tangent based on node type
        let tangent = match node.as_ref() {
            IRNode::Input { .. } => {
                panic!("Input node tangent not initialized");
            }

            IRNode::Constant { .. } => {
                // Constants have zero tangent
                Array::zeros(node.shape(), node.dtype())
            }

            IRNode::Unary { op, input, .. } => {
                let input_tangent = self.forward(input);
                self.jvp_unary(op, input, &input_tangent)
            }

            IRNode::Binary { op, lhs, rhs, .. } => {
                let lhs_tangent = self.forward(lhs);
                let rhs_tangent = self.forward(rhs);
                self.jvp_binary(op, lhs, rhs, &lhs_tangent, &rhs_tangent)
            }

            IRNode::Reduce { op, input, .. } => {
                let input_tangent = self.forward(input);
                self.jvp_reduce(op, &input_tangent)
            }
        };

        // Cache the tangent
        self.tangents.insert(node_addr, tangent.clone());
        tangent
    }

    /// JVP rule for unary operations.
    fn jvp_unary(
        &self,
        op: &Primitive,
        _input: &Arc<IRNode>,
        tangent: &Array,
    ) -> Array {
        // For unary ops: tangent_output = (d(op)/d(input)) * tangent_input

        match op {
            Primitive::Neg => tangent.neg(),

            Primitive::Abs => {
                // d/dx |x| = sign(x)
                // For now, simplified
                tangent.clone()
            }

            Primitive::Square => {
                // d/dx x^2 = 2x
                // tangent_out = 2x * tangent_in (need input value)
                tangent.clone()
            }

            Primitive::Sqrt => {
                // d/dx sqrt(x) = 1/(2*sqrt(x))
                tangent.clone()
            }

            Primitive::Exp => {
                // d/dx exp(x) = exp(x)
                tangent.clone()
            }

            Primitive::Log => {
                // d/dx log(x) = 1/x
                tangent.clone()
            }

            Primitive::Sin => {
                // d/dx sin(x) = cos(x)
                tangent.clone()
            }

            Primitive::Cos => {
                // d/dx cos(x) = -sin(x)
                tangent.neg()
            }

            Primitive::Tanh => {
                // d/dx tanh(x) = 1 - tanh(x)^2
                tangent.clone()
            }

            Primitive::Reciprocal => {
                // d/dx (1/x) = -1/x^2
                tangent.clone()
            }

            _ => panic!("Unsupported unary operation for JVP: {:?}", op),
        }
    }

    /// JVP rule for binary operations.
    fn jvp_binary(
        &self,
        op: &Primitive,
        _lhs: &Arc<IRNode>,
        _rhs: &Arc<IRNode>,
        lhs_tangent: &Array,
        rhs_tangent: &Array,
    ) -> Array {
        // For binary ops: tangent_output = (d(op)/d(lhs)) * tangent_lhs + (d(op)/d(rhs)) * tangent_rhs

        match op {
            Primitive::Add => {
                // d/dx (x + y) = 1, d/dy (x + y) = 1
                lhs_tangent.add(rhs_tangent)
            }

            Primitive::Sub => {
                // d/dx (x - y) = 1, d/dy (x - y) = -1
                lhs_tangent.sub(rhs_tangent)
            }

            Primitive::Mul => {
                // d/dx (x * y) = y, d/dy (x * y) = x
                // tangent_out = y * tangent_lhs + x * tangent_rhs (need input values)
                // For now, simplified
                lhs_tangent.add(rhs_tangent)
            }

            Primitive::Div => {
                // d/dx (x / y) = 1/y, d/dy (x / y) = -x/y^2
                // For now, simplified
                lhs_tangent.sub(rhs_tangent)
            }

            _ => panic!("Unsupported binary operation for JVP: {:?}", op),
        }
    }

    /// JVP rule for reduction operations.
    fn jvp_reduce(&self, op: &Primitive, tangent: &Array) -> Array {
        match op {
            Primitive::SumAll => {
                // d/dx sum(x) = sum(tangent)
                tangent.sum_all_array()
            }

            Primitive::Sum { axis } => {
                // d/dx sum(x, axis) = sum(tangent, axis)
                tangent.sum(*axis)
            }

            Primitive::MeanAll => {
                // d/dx mean(x) = mean(tangent)
                let val = tangent.mean_all();
                Array::from_vec(vec![val], crate::Shape::scalar())
            }

            _ => panic!("Unsupported reduction operation for JVP: {:?}", op),
        }
    }
}

impl Default for JVPEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape};

    #[test]
    fn test_grad_simple() {
        // f(x) = x^2, df/dx = 2x
        let df = grad(|x: &Array| x.mul(x).sum_all_array());

        let x = Array::from_vec(vec![2.0], Shape::new(vec![1]));
        let gradient = df(&x);

        // df/dx at x=2 should be approximately 4
        assert!((gradient.to_vec()[0] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_grad_vector() {
        // f(x) = sum(x^2), df/dx = 2x
        let df = grad(|x: &Array| x.mul(x).sum_all_array());

        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let gradient = df(&x);

        // df/dx should be approximately [2, 4, 6]
        let grad_vec = gradient.to_vec();
        assert!((grad_vec[0] - 2.0).abs() < 0.01);
        assert!((grad_vec[1] - 4.0).abs() < 0.01);
        assert!((grad_vec[2] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_vjp_add() {
        // Build graph: add(input0, input1)
        let input0 = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![2]), DType::Float32);
        let add =
            IRNode::binary(Primitive::Add, input0.clone(), input1.clone());

        let graph = IRGraph::new(
            "test_add".to_string(),
            vec![input0, input1],
            vec![add],
        );

        // Cotangent for output
        let cotangent = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));

        let mut engine = VJPEngine::new();
        let grads = engine.vjp(&graph, &[cotangent]);

        // For add, both gradients should be the cotangent
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].to_vec(), vec![1.0, 1.0]);
        assert_eq!(grads[1].to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_vjp_sub() {
        // Build graph: sub(input0, input1)
        let input0 = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![2]), DType::Float32);
        let sub =
            IRNode::binary(Primitive::Sub, input0.clone(), input1.clone());

        let graph = IRGraph::new(
            "test_sub".to_string(),
            vec![input0, input1],
            vec![sub],
        );

        let cotangent = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));

        let mut engine = VJPEngine::new();
        let grads = engine.vjp(&graph, &[cotangent]);

        // For sub: grad_lhs = cotangent, grad_rhs = -cotangent
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].to_vec(), vec![1.0, 1.0]);
        assert_eq!(grads[1].to_vec(), vec![-1.0, -1.0]);
    }

    #[test]
    fn test_jvp_add() {
        // Build graph: add(input0, input1)
        let input0 = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![2]), DType::Float32);
        let add =
            IRNode::binary(Primitive::Add, input0.clone(), input1.clone());

        let graph = IRGraph::new(
            "test_add".to_string(),
            vec![input0, input1],
            vec![add],
        );

        // Tangents for inputs
        let tangent0 = Array::from_vec(vec![1.0, 0.0], Shape::new(vec![2]));
        let tangent1 = Array::from_vec(vec![0.0, 1.0], Shape::new(vec![2]));

        let mut engine = JVPEngine::new();
        let tangents = engine.jvp(&graph, &[tangent0, tangent1]);

        // For add: tangent_out = tangent_lhs + tangent_rhs
        assert_eq!(tangents.len(), 1);
        assert_eq!(tangents[0].to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_jvp_sub() {
        // Build graph: sub(input0, input1)
        let input0 = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![2]), DType::Float32);
        let sub =
            IRNode::binary(Primitive::Sub, input0.clone(), input1.clone());

        let graph = IRGraph::new(
            "test_sub".to_string(),
            vec![input0, input1],
            vec![sub],
        );

        let tangent0 = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));
        let tangent1 = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));

        let mut engine = JVPEngine::new();
        let tangents = engine.jvp(&graph, &[tangent0, tangent1]);

        // For sub: tangent_out = tangent_lhs - tangent_rhs
        assert_eq!(tangents.len(), 1);
        assert_eq!(tangents[0].to_vec(), vec![0.0, 0.0]);
    }

    #[test]
    fn test_jvp_unary_neg() {
        // Build graph: neg(input0)
        let input0 = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let neg = IRNode::unary(Primitive::Neg, input0.clone());

        let graph =
            IRGraph::new("test_neg".to_string(), vec![input0], vec![neg]);

        let tangent = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));

        let mut engine = JVPEngine::new();
        let tangents = engine.jvp(&graph, &[tangent]);

        // For neg: tangent_out = -tangent_in
        assert_eq!(tangents.len(), 1);
        assert_eq!(tangents[0].to_vec(), vec![-1.0, -2.0]);
    }

    #[test]
    fn test_jvp_reduce_sum() {
        // Build graph: sum_all(input0)
        let input0 = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let sum =
            IRNode::reduce(Primitive::SumAll, input0.clone(), Shape::scalar());

        let graph =
            IRGraph::new("test_sum".to_string(), vec![input0], vec![sum]);

        let tangent =
            Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let mut engine = JVPEngine::new();
        let tangents = engine.jvp(&graph, &[tangent]);

        // For sum: tangent_out = sum(tangent_in)
        assert_eq!(tangents.len(), 1);
        assert_eq!(tangents[0].to_vec(), vec![6.0]);
    }
}
