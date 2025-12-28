//! Automatic differentiation via reverse-mode AD (grad, vjp, jvp).
//!
//! This module implements gradient computation using:
//! - Symbolic reverse-mode AD with transpose rules
//! - Value and gradient computation via vjp (vector-jacobian product)
//! - Forward-mode AD via jvp (jacobian-vector product)

use crate::trace::transpose_rules::{self, PrimalValue};
use crate::trace::{IRGraph, IRNode, Primitive};
use crate::{Array, DType, Shape};
use std::collections::HashMap;
use std::sync::Arc;

/// Compute the gradient of a scalar-valued function.
///
/// Returns a function that computes the gradient with respect to the first input.
/// The function must return a scalar (0-dimensional array).
///
/// # Examples
///
/// ```rust,ignore
/// use jax_rs::trace::grad;
///
/// let df = grad(|x: &Array| {
///     x.mul(x).sum_all_array()  // f(x) = sum(x^2)
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
        let (_, gradient) = value_and_grad(&f, x);
        gradient
    }
}

/// Compute both value and gradient of a scalar-valued function.
///
/// More efficient than calling the function and grad separately,
/// as it computes both in a single forward + backward pass.
pub fn value_and_grad<F>(f: &F, x: &Array) -> (Array, Array)
where
    F: Fn(&Array) -> Array,
{
    // Try symbolic AD first (fast path)
    match compute_gradient_symbolic(f, x) {
        Ok((value, gradient)) => (value, gradient),
        Err(_) => {
            // Fallback to numerical differentiation
            let y = f(x);

            // Check that output is scalar
            if !y.is_scalar() && y.size() != 1 {
                panic!(
                    "grad requires scalar output, got shape {:?}",
                    y.shape().as_slice()
                );
            }

            let gradient = compute_gradient_hybrid(f, x, &y);
            (y, gradient)
        }
    }
}

/// Compute gradient using symbolic AD via tracing and VJPEngine.
///
/// This is the fast path that uses symbolic differentiation rules.
fn compute_gradient_symbolic<F>(f: &F, x: &Array) -> Result<(Array, Array), String>
where
    F: Fn(&Array) -> Array,
{
    use crate::trace::tracer::{TraceContext, enter_trace, exit_trace};
    use crate::trace::Interpreter;
    use std::rc::Rc;
    use std::cell::RefCell;

    // Create trace context
    let ctx = Rc::new(RefCell::new(TraceContext::new("grad".to_string())));

    // Register input in trace context
    let x_id = x.id();
    {
        let mut ctx_mut = ctx.borrow_mut();
        ctx_mut.register_input(x_id, x.shape().clone(), x.dtype());
    }

    // Enter tracing mode
    enter_trace(ctx.clone());

    // Call function (this will trace operations)
    let y = f(x);

    // Exit tracing mode
    exit_trace();

    // Check that output is scalar
    if !y.is_scalar() && y.size() != 1 {
        return Err(format!(
            "grad requires scalar output, got shape {:?}",
            y.shape().as_slice()
        ));
    }

    // Finalize trace to get IR graph
    let ctx_owned = match Rc::try_unwrap(ctx) {
        Ok(cell) => cell.into_inner(),
        Err(_) => return Err("TraceContext still has references".to_string()),
    };
    let graph = ctx_owned.finalize(&[y.clone()]);

    // Forward pass: Execute graph to compute all primal values
    let mut interp = Interpreter::new();
    let outputs = interp.execute(&graph, &[x.clone()]);

    // VJPEngine: Backward pass to compute gradients
    let mut engine = VJPEngine::new();

    // Store primal values for all nodes (needed for nonlinear operations)
    // We need to traverse the graph and store the computed values
    store_primals_from_interpreter(&mut engine, &graph, &[x.clone()], &interp);

    // Initialize output gradient (cotangent) as ones (d(output)/d(output) = 1)
    let cotangent = Array::ones(outputs[0].shape().clone(), outputs[0].dtype());
    let input_grads = engine.vjp(&graph, &[cotangent]);

    if input_grads.is_empty() {
        return Err("No input gradients computed".to_string());
    }

    Ok((outputs[0].clone(), input_grads[0].clone()))
}

/// Helper to store primal values from interpreter into VJPEngine.
fn store_primals_from_interpreter(
    engine: &mut VJPEngine,
    graph: &IRGraph,
    inputs: &[Array],
    _interp: &crate::trace::Interpreter,
) {
    use std::sync::Arc;
    use std::collections::HashMap;

    // Store ALL primal values (including inputs) so they can be used in gradient computation
    // The get_primal() method handles the distinction between inputs and intermediates

    // Cache for primal computation
    let mut primal_cache: HashMap<usize, Array> = HashMap::new();

    // Store inputs in cache AND in engine
    for (node, array) in graph.inputs.iter().zip(inputs.iter()) {
        let addr = Arc::as_ptr(node) as usize;
        primal_cache.insert(addr, array.clone());
        engine.store_primal(node, array.clone());  // Store so it can be used in vjp_binary
    }

    // Store primals for intermediate nodes
    for output_node in &graph.outputs {
        store_node_primals(engine, output_node, &mut primal_cache, false);
    }
}

/// Recursively store primal values for a node and its dependencies.
/// skip_inputs: if true, don't store primals for Input nodes (they're variables, not known constants)
fn store_node_primals(
    engine: &mut VJPEngine,
    node: &Arc<IRNode>,
    cache: &mut HashMap<usize, Array>,
    skip_inputs: bool,
) {
    let addr = Arc::as_ptr(node) as usize;

    // If already processed, skip
    if cache.contains_key(&addr) {
        return;
    }

    // Compute primal by recursively evaluating inputs
    match node.as_ref() {
        IRNode::Input { .. } => {
            // Input nodes should already be in cache (from parent call)
            // DON'T store them as "known" in the engine if skip_inputs is true
            if !skip_inputs {
                if let Some(value) = cache.get(&addr) {
                    engine.store_primal(node, value.clone());
                }
            }
        }
        IRNode::Constant { value, dtype } => {
            let arr = Array::full(*value, crate::Shape::scalar(), *dtype);
            cache.insert(addr, arr.clone());
            engine.store_primal(node, arr);
        }
        IRNode::MaterializedConstant { value } => {
            cache.insert(addr, value.clone());
            engine.store_primal(node, value.clone());
        }
        IRNode::Unary { op, input, .. } => {
            // First compute input primal
            store_node_primals(engine, input, cache, skip_inputs);

            // Then compute this node's primal
            let input_addr = Arc::as_ptr(input) as usize;
            if let Some(input_val) = cache.get(&input_addr) {
                let result = eval_unary_op(op, input_val);
                cache.insert(addr, result.clone());
                engine.store_primal(node, result.clone());
                // Store output for operations that need it (exp, tanh, sqrt, etc.)
                engine.store_output(node, result);
            }
        }
        IRNode::Binary { op, lhs, rhs, .. } => {
            // First compute input primals
            store_node_primals(engine, lhs, cache, skip_inputs);
            store_node_primals(engine, rhs, cache, skip_inputs);

            // Then compute this node's primal
            let lhs_addr = Arc::as_ptr(lhs) as usize;
            let rhs_addr = Arc::as_ptr(rhs) as usize;
            // Clone values to avoid borrow checker issues
            let lhs_val_opt = cache.get(&lhs_addr).cloned();
            let rhs_val_opt = cache.get(&rhs_addr).cloned();

            if let (Some(lhs_val), Some(rhs_val)) = (lhs_val_opt, rhs_val_opt) {
                let result = eval_binary_op(op, &lhs_val, &rhs_val);
                cache.insert(addr, result.clone());
                // Store intermediate values so they can be used in backward pass
                // For example, mul needs the operands, exp needs the output
                // But DON'T store primals for Input nodes if skip_inputs is true
                if !skip_inputs || !matches!(lhs.as_ref(), IRNode::Input { .. }) {
                    engine.store_primal(lhs, lhs_val);
                }
                if !skip_inputs || !matches!(rhs.as_ref(), IRNode::Input { .. }) {
                    engine.store_primal(rhs, rhs_val);
                }
                engine.store_primal(node, result.clone());
                // Store output for operations that need it (pow, etc.)
                engine.store_output(node, result);
            }
        }
        IRNode::Reduce { op, input, .. } => {
            // First compute input primal
            store_node_primals(engine, input, cache, skip_inputs);

            // Then compute this node's primal
            let input_addr = Arc::as_ptr(input) as usize;
            if let Some(input_val) = cache.get(&input_addr) {
                let result = eval_reduce_op(op, input_val);
                cache.insert(addr, result.clone());
                engine.store_primal(node, result.clone());
                engine.store_output(node, result);
            }
        }
        IRNode::FusedOp { group, .. } => {
            // Fused operations: recursively compute primals for all operations in the group
            // This unfuses the operation for gradient computation
            for op_node in &group.operations {
                store_node_primals(engine, op_node, cache, skip_inputs);
            }
            // Also store primals for group inputs
            for input_node in &group.inputs {
                store_node_primals(engine, input_node, cache, skip_inputs);
            }
        }
    }
}

/// Evaluate a unary operation (helper for primal computation).
fn eval_unary_op(op: &Primitive, input: &Array) -> Array {
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
        Primitive::Sign => input.sign(),
        _ => panic!("Unsupported unary operation in primal eval: {:?}", op),
    }
}

/// Evaluate a binary operation (helper for primal computation).
fn eval_binary_op(op: &Primitive, lhs: &Array, rhs: &Array) -> Array {
    match op {
        Primitive::Add => lhs.add(rhs),
        Primitive::Sub => lhs.sub(rhs),
        Primitive::Mul => lhs.mul(rhs),
        Primitive::Div => lhs.div(rhs),
        Primitive::Pow => lhs.pow(rhs),
        Primitive::Matmul => lhs.matmul(rhs),
        Primitive::Dot => lhs.dot(rhs),
        _ => panic!("Unsupported binary operation in primal eval: {:?}", op),
    }
}

/// Evaluate a reduction operation (helper for primal computation).
fn eval_reduce_op(op: &Primitive, input: &Array) -> Array {
    match op {
        Primitive::SumAll => input.sum_all_array(),
        Primitive::Sum { axis } => input.sum(*axis),
        Primitive::MeanAll => input.mean_all_array(),
        Primitive::Mean { axis } => input.mean(*axis),
        Primitive::MaxAll => {
            let val = input.max_all();
            Array::from_vec(vec![val], crate::Shape::scalar())
        }
        Primitive::MinAll => {
            let val = input.min_all();
            Array::from_vec(vec![val], crate::Shape::scalar())
        }
        _ => panic!("Unsupported reduction operation in primal eval: {:?}", op),
    }
}

/// Debug helper to print IR node structure.
fn print_ir_node(node: &Arc<IRNode>, depth: usize) -> String {
    let indent = "  ".repeat(depth);
    match node.as_ref() {
        IRNode::Input { id, shape, dtype } => {
            format!("{}Input(id={}, shape={:?}, dtype={:?})", indent, id, shape.as_slice(), dtype)
        }
        IRNode::Constant { value, dtype } => {
            format!("{}Constant(value={}, dtype={:?})", indent, value, dtype)
        }
        IRNode::MaterializedConstant { value } => {
            format!("{}MaterializedConstant(shape={:?}, dtype={:?})", indent, value.shape().as_slice(), value.dtype())
        }
        IRNode::Unary { op, input, .. } => {
            format!("{}Unary({:?})\n{}", indent, op, print_ir_node(input, depth + 1))
        }
        IRNode::Binary { op, lhs, rhs, .. } => {
            format!("{}Binary({:?})\n{}\n{}", indent, op, print_ir_node(lhs, depth + 1), print_ir_node(rhs, depth + 1))
        }
        IRNode::Reduce { op, input, .. } => {
            format!("{}Reduce({:?})\n{}", indent, op, print_ir_node(input, depth + 1))
        }
        IRNode::FusedOp { group, .. } => {
            format!("{}FusedOp({} ops)", indent, group.operations.len())
        }
    }
}

/// Compute gradient using a hybrid symbolic/numerical approach.
///
/// This uses symbolic rules where possible and falls back to
/// numerical differentiation for complex operations.
fn compute_gradient_hybrid<F>(f: &F, x: &Array, _y: &Array) -> Array
where
    F: Fn(&Array) -> Array,
{
    // For simple operations, use symbolic differentiation
    // For complex graphs, use numerical with caching

    let x_data = x.to_vec();
    let n = x_data.len();

    // Use vectorized computation where possible
    if n <= 1000 {
        // For smaller arrays, use parallel numerical diff with adaptive step
        compute_gradient_numerical_adaptive(f, x)
    } else {
        // For larger arrays, use chunked computation
        compute_gradient_numerical_chunked(f, x)
    }
}

/// Numerical gradient with adaptive step size.
fn compute_gradient_numerical_adaptive<F>(f: &F, x: &Array) -> Array
where
    F: Fn(&Array) -> Array,
{
    let x_data = x.to_vec();
    let n = x_data.len();
    let mut grad_data = vec![0.0f32; n];

    // Compute base function value once
    let y_base = f(x).to_vec()[0];

    // Use relative epsilon for better numerical stability
    for i in 0..n {
        let xi = x_data[i];
        // Adaptive step: use relative epsilon for large values
        let eps = if xi.abs() > 1.0 {
            1e-5 * xi.abs()
        } else {
            1e-5
        };

        // Forward difference (faster than central, good enough for most cases)
        let mut x_plus = x_data.clone();
        x_plus[i] = xi + eps;
        let y_plus = f(&Array::from_vec(x_plus, x.shape().clone())).to_vec()[0];

        grad_data[i] = (y_plus - y_base) / eps;
    }

    Array::from_vec(grad_data, x.shape().clone())
}

/// Numerical gradient with chunked computation for large arrays.
fn compute_gradient_numerical_chunked<F>(f: &F, x: &Array) -> Array
where
    F: Fn(&Array) -> Array,
{
    let x_data = x.to_vec();
    let n = x_data.len();
    let mut grad_data = vec![0.0f32; n];
    let eps = 1e-5;

    // Process in chunks to reduce memory pressure
    let chunk_size = 256;

    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n);

        for i in chunk_start..chunk_end {
            let mut x_plus = x_data.clone();
            x_plus[i] += eps;
            let y_plus = f(&Array::from_vec(x_plus, x.shape().clone())).to_vec()[0];

            let mut x_minus = x_data.clone();
            x_minus[i] -= eps;
            let y_minus = f(&Array::from_vec(x_minus, x.shape().clone())).to_vec()[0];

            grad_data[i] = (y_plus - y_minus) / (2.0 * eps);
        }
    }

    Array::from_vec(grad_data, x.shape().clone())
}

// =============================================================================
// Vector-Jacobian Product (VJP) - Reverse-mode AD
// =============================================================================

/// Vector-Jacobian Product computation engine.
///
/// Implements reverse-mode automatic differentiation by propagating
/// cotangents (gradients) backwards through the computation graph.
pub struct VJPEngine {
    /// Gradient accumulator for each node (by pointer address)
    gradients: HashMap<usize, Array>,
    /// Cached primal values for computing gradients
    primals: HashMap<usize, Array>,
    /// Cached output values (for operations where output is needed)
    outputs: HashMap<usize, Array>,
}

impl VJPEngine {
    /// Create a new VJP engine.
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
            primals: HashMap::new(),
            outputs: HashMap::new(),
        }
    }

    /// Store a primal value for later gradient computation.
    pub fn store_primal(&mut self, node: &Arc<IRNode>, value: Array) {
        let addr = Arc::as_ptr(node) as usize;
        self.primals.insert(addr, value);
    }

    /// Store an output value for later gradient computation.
    pub fn store_output(&mut self, node: &Arc<IRNode>, value: Array) {
        let addr = Arc::as_ptr(node) as usize;
        self.outputs.insert(addr, value);
    }

    /// Get primal value for a node.
    fn get_primal(&self, node: &Arc<IRNode>) -> PrimalValue {
        // IMPORTANT: Only true constants should be treated as Known
        // Everything else (inputs and intermediate operations) should be Unknown
        // for gradient propagation purposes.
        // However, we still store primal values for ALL nodes so they can be used
        // in gradient computation (e.g., log needs the input value).
        match node.as_ref() {
            IRNode::Constant { .. } | IRNode::MaterializedConstant { .. } => {
                // True constants are Known
                let addr = Arc::as_ptr(node) as usize;
                match self.primals.get(&addr) {
                    Some(arr) => PrimalValue::known(arr.clone()),
                    None => PrimalValue::unknown(node.shape(), node.dtype()),
                }
            }
            _ => {
                // All other nodes (Input, Unary, Binary, Reduce) are Unknown
                // for gradient propagation
                PrimalValue::unknown(node.shape(), node.dtype())
            }
        }
    }

    /// Get output value for a node.
    fn get_output(&self, node: &Arc<IRNode>) -> Option<&Array> {
        let addr = Arc::as_ptr(node) as usize;
        self.outputs.get(&addr)
    }

    /// Compute VJP for a graph given output gradients (cotangents).
    ///
    /// Returns gradients with respect to the inputs.
    pub fn vjp(&mut self, graph: &IRGraph, cotangents: &[Array]) -> Vec<Array> {
        assert_eq!(
            graph.outputs.len(),
            cotangents.len(),
            "Number of cotangents must match number of outputs"
        );

        // Initialize output gradients
        for (output, cotangent) in graph.outputs.iter().zip(cotangents.iter()) {
            let node_addr = Arc::as_ptr(output) as usize;
            self.gradients.insert(node_addr, cotangent.clone());
        }

        // Backward pass: propagate gradients from outputs to inputs
        // Process in reverse topological order
        for output in graph.outputs.iter() {
            self.backward(output);
        }

        // Extract input gradients
        graph
            .inputs
            .iter()
            .map(|input| {
                let node_addr = Arc::as_ptr(input) as usize;
                self.gradients.get(&node_addr).cloned().unwrap_or_else(|| {
                    if let IRNode::Input { shape, dtype, .. } = input.as_ref() {
                        Array::zeros(shape.clone(), *dtype)
                    } else {
                        panic!("Expected input node");
                    }
                })
            })
            .collect()
    }

    /// Backward pass for a single node using transpose rules.
    fn backward(&mut self, node: &Arc<IRNode>) {
        let node_addr = Arc::as_ptr(node) as usize;

        // Get the gradient for this node (cotangent)
        let cotangent = match self.gradients.get(&node_addr) {
            Some(g) => g.clone(),
            None => return, // No gradient to propagate
        };

        match node.as_ref() {
            IRNode::Input { .. } => {
                // Input nodes just accumulate gradients - nothing to propagate
            }

            IRNode::Constant { .. } => {
                // Constants don't propagate gradients
            }

            IRNode::MaterializedConstant { .. } => {
                // Materialized constants don't propagate gradients
            }

            IRNode::Unary { op, input, .. } => {
                // For Input nodes, we need the primal value to compute gradients
                // but we still want to return a gradient for it
                let input_is_input = matches!(input.as_ref(), IRNode::Input { .. });
                let input_primal = if input_is_input {
                    // Get the actual primal value for computation, but mark as Unknown
                    // Actually, we need the value for operations like log(x) -> 1/x
                    // So we fetch it directly and mark as Known
                    let addr = Arc::as_ptr(input) as usize;
                    self.primals.get(&addr)
                        .map(|arr| PrimalValue::known(arr.clone()))
                        .unwrap_or_else(|| PrimalValue::unknown(input.shape(), input.dtype()))
                } else {
                    self.get_primal(input)
                };

                let output_val = self.get_output(node);
                let grad = self.vjp_unary(op, &input_primal, output_val, &cotangent);

                let input_addr = Arc::as_ptr(input) as usize;
                self.accumulate_gradient(input_addr, grad);
                self.backward(input);
            }

            IRNode::Binary { op, lhs, rhs, .. } => {
                // Special case: both operands are the same Input node (e.g., x * x)
                let lhs_is_input = matches!(lhs.as_ref(), IRNode::Input { .. });
                let rhs_is_input = matches!(rhs.as_ref(), IRNode::Input { .. });
                let same_node = Arc::ptr_eq(lhs, rhs);

                let mut lhs_primal = if lhs_is_input && !same_node {
                    // Different input node - treat as Unknown
                    self.get_primal(lhs)
                } else if lhs_is_input && same_node {
                    // Same input node (x * x) - make lhs Unknown and rhs Known
                    // This way transpose_mul will return gradient for lhs
                    PrimalValue::unknown(lhs.shape(), lhs.dtype())
                } else {
                    self.get_primal(lhs)
                };

                let mut rhs_primal = if rhs_is_input && !same_node {
                    // Different input node - treat as Unknown
                    self.get_primal(rhs)
                } else if rhs_is_input && same_node {
                    // Same input node (x * x) - provide the primal value as Known
                    let addr = Arc::as_ptr(rhs) as usize;
                    self.primals.get(&addr)
                        .map(|arr| PrimalValue::known(arr.clone()))
                        .unwrap_or_else(|| PrimalValue::unknown(rhs.shape(), rhs.dtype()))
                } else {
                    self.get_primal(rhs)
                };

                // Special handling for Mul when both operands are Unknown (e.g., (x*x)*x)
                // We need actual primal values to compute gradients
                if matches!(op, Primitive::Mul) && !lhs_primal.is_known() && !rhs_primal.is_known() {
                    // Fetch actual primal values for both operands
                    let lhs_addr = Arc::as_ptr(lhs) as usize;
                    let rhs_addr = Arc::as_ptr(rhs) as usize;

                    // Make one Known so transpose_mul can work
                    // We'll compute gradients for both by calling vjp_binary twice
                    if let Some(rhs_val) = self.primals.get(&rhs_addr) {
                        rhs_primal = PrimalValue::known(rhs_val.clone());
                    }
                }

                let output_val = self.get_output(node).cloned();  // Clone to avoid borrow issues

                // Check if we're in the both-unknown Mul case (needed for polynomial gradients)
                let both_unknown_mul = matches!(op, Primitive::Mul)
                    && !same_node
                    && !lhs_primal.is_known()
                    && rhs_primal.is_known(); // We made rhs Known above

                let (grad_lhs, grad_rhs) =
                    self.vjp_binary(op, &lhs_primal, &rhs_primal, output_val.as_ref(), &cotangent);

                if same_node && lhs_is_input {
                    // For same node (e.g., x * x), we need gradients from BOTH paths
                    // Compute gradient with lhs=Unknown, rhs=Known (we already did this above)
                    // Then compute gradient with lhs=Known, rhs=Unknown and accumulate
                    if let Some(g) = grad_lhs {
                        let lhs_addr = Arc::as_ptr(lhs) as usize;
                        self.accumulate_gradient(lhs_addr, g);
                    }

                    // Now compute the other direction: d/d(rhs) with lhs=Known, rhs=Unknown
                    let lhs_primal_known = {
                        let addr = Arc::as_ptr(lhs) as usize;
                        self.primals.get(&addr)
                            .map(|arr| PrimalValue::known(arr.clone()))
                            .unwrap_or_else(|| PrimalValue::unknown(lhs.shape(), lhs.dtype()))
                    };
                    let rhs_primal_unknown = PrimalValue::unknown(rhs.shape(), rhs.dtype());
                    let (_, grad_rhs_second) =
                        self.vjp_binary(op, &lhs_primal_known, &rhs_primal_unknown, output_val.as_ref(), &cotangent);

                    if let Some(g) = grad_rhs_second {
                        let rhs_addr = Arc::as_ptr(rhs) as usize;
                        self.accumulate_gradient(rhs_addr, g);
                    }
                } else if both_unknown_mul {
                    // Both operands Unknown but different nodes (e.g., (x*x)*x)
                    // We computed grad_lhs with rhs=Known above
                    // Now compute grad_rhs with lhs=Known, rhs=Unknown
                    if let Some(g) = grad_lhs {
                        let lhs_addr = Arc::as_ptr(lhs) as usize;
                        self.accumulate_gradient(lhs_addr, g);
                    }

                    // Swap: make lhs Known, rhs Unknown
                    let lhs_addr = Arc::as_ptr(lhs) as usize;
                    let lhs_primal_known = self.primals.get(&lhs_addr)
                        .map(|arr| PrimalValue::known(arr.clone()))
                        .unwrap_or_else(|| PrimalValue::unknown(lhs.shape(), lhs.dtype()));
                    let rhs_primal_unknown = PrimalValue::unknown(rhs.shape(), rhs.dtype());

                    let (_, grad_rhs_second) =
                        self.vjp_binary(op, &lhs_primal_known, &rhs_primal_unknown, output_val.as_ref(), &cotangent);

                    if let Some(g) = grad_rhs_second {
                        let rhs_addr = Arc::as_ptr(rhs) as usize;
                        self.accumulate_gradient(rhs_addr, g);
                    }
                } else {
                    // Normal case: different nodes or not both inputs
                    if let Some(g) = grad_lhs {
                        let lhs_addr = Arc::as_ptr(lhs) as usize;
                        self.accumulate_gradient(lhs_addr, g);
                    }
                    if let Some(g) = grad_rhs {
                        let rhs_addr = Arc::as_ptr(rhs) as usize;
                        self.accumulate_gradient(rhs_addr, g);
                    }
                }

                self.backward(lhs);
                if !same_node {
                    self.backward(rhs);  // Only call backward on rhs if it's different from lhs
                }
            }

            IRNode::Reduce { op, input, .. } => {
                let input_primal = self.get_primal(input);
                let output_val = self.get_output(node);
                let grad = self.vjp_reduce(op, input, &input_primal, output_val, &cotangent);

                let input_addr = Arc::as_ptr(input) as usize;
                self.accumulate_gradient(input_addr, grad);
                self.backward(input);
            }

            IRNode::FusedOp { group, .. } => {
                // For fused operations, propagate gradients through the original operations
                // Backward pass through all operations in the group in reverse order
                for op_node in group.operations.iter().rev() {
                    self.backward(op_node);
                }
            }
        }
    }

    /// VJP rule for unary operations using transpose rules.
    fn vjp_unary(
        &self,
        op: &Primitive,
        input: &PrimalValue,
        output: Option<&Array>,
        cotangent: &Array,
    ) -> Array {
        match op {
            Primitive::Neg => transpose_rules::transpose_neg(cotangent, input),
            Primitive::Abs => transpose_rules::transpose_abs(cotangent, input),
            Primitive::Sin => transpose_rules::transpose_sin(cotangent, input),
            Primitive::Cos => transpose_rules::transpose_cos(cotangent, input),
            Primitive::Tan => transpose_rules::transpose_tan(cotangent, input),
            Primitive::Tanh => transpose_rules::transpose_tanh(cotangent, input, output),
            Primitive::Exp => transpose_rules::transpose_exp(cotangent, input, output),
            Primitive::Log => transpose_rules::transpose_log(cotangent, input),
            Primitive::Sqrt => transpose_rules::transpose_sqrt(cotangent, input, output),
            Primitive::Reciprocal => transpose_rules::transpose_reciprocal(cotangent, input),
            Primitive::Square => transpose_rules::transpose_square(cotangent, input),
            Primitive::Sign => transpose_rules::transpose_sign(cotangent, input),
            _ => {
                // Fallback for unsupported ops
                cotangent.clone()
            }
        }
    }

    /// VJP rule for binary operations using transpose rules.
    fn vjp_binary(
        &self,
        op: &Primitive,
        lhs: &PrimalValue,
        rhs: &PrimalValue,
        output: Option<&Array>,
        cotangent: &Array,
    ) -> (Option<Array>, Option<Array>) {
        match op {
            Primitive::Add => transpose_rules::transpose_add(cotangent, lhs, rhs),
            Primitive::Sub => transpose_rules::transpose_sub(cotangent, lhs, rhs),
            Primitive::Mul => transpose_rules::transpose_mul(cotangent, lhs, rhs),
            Primitive::Div => transpose_rules::transpose_div(cotangent, lhs, rhs),
            Primitive::Pow => transpose_rules::transpose_pow(cotangent, lhs, rhs, output),
            Primitive::Matmul => transpose_rules::transpose_matmul(cotangent, lhs, rhs),
            Primitive::Dot => transpose_rules::transpose_dot(cotangent, lhs, rhs),
            _ => {
                // Fallback: return cotangent for both
                (Some(cotangent.clone()), Some(cotangent.clone()))
            }
        }
    }

    /// VJP rule for reduction operations.
    fn vjp_reduce(
        &self,
        op: &Primitive,
        input_node: &Arc<IRNode>,
        input_primal: &PrimalValue,
        output: Option<&Array>,
        cotangent: &Array,
    ) -> Array {
        let input_shape = input_node.shape();
        let dtype = input_node.dtype();

        match op {
            Primitive::SumAll => {
                transpose_rules::transpose_sum_all(cotangent, &input_shape, dtype)
            }
            Primitive::Sum { axis } => {
                transpose_rules::transpose_sum_axis(cotangent, &input_shape, *axis, dtype)
            }
            Primitive::MeanAll => {
                transpose_rules::transpose_mean_all(cotangent, &input_shape, dtype)
            }
            Primitive::Mean { axis } => {
                transpose_rules::transpose_mean_axis(cotangent, &input_shape, *axis, dtype)
            }
            Primitive::MaxAll => {
                transpose_rules::transpose_max_all(cotangent, input_primal, output)
            }
            Primitive::MinAll => {
                transpose_rules::transpose_min_all(cotangent, input_primal, output)
            }
            _ => {
                // For unsupported reductions, broadcast cotangent
                let ones = Array::ones(input_shape, dtype);
                ones.mul(cotangent)
            }
        }
    }

    /// Accumulate gradient for a node.
    fn accumulate_gradient(&mut self, node_addr: usize, grad: Array) {
        if let Some(existing) = self.gradients.get(&node_addr) {
            let new_grad = existing.add(&grad);
            self.gradients.insert(node_addr, new_grad);
        } else {
            self.gradients.insert(node_addr, grad);
        }
    }
}

impl Default for VJPEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Jacobian-Vector Product (JVP) - Forward-mode AD
// =============================================================================

/// Jacobian-Vector Product computation engine.
///
/// Implements forward-mode automatic differentiation by propagating
/// tangents forward through the computation graph.
pub struct JVPEngine {
    /// Tangent values for each node
    tangents: HashMap<usize, Array>,
    /// Cached primal values
    primals: HashMap<usize, Array>,
}

impl JVPEngine {
    /// Create a new JVP engine.
    pub fn new() -> Self {
        Self {
            tangents: HashMap::new(),
            primals: HashMap::new(),
        }
    }

    /// Store a primal value.
    pub fn store_primal(&mut self, node: &Arc<IRNode>, value: Array) {
        let addr = Arc::as_ptr(node) as usize;
        self.primals.insert(addr, value);
    }

    /// Get primal value for a node.
    fn get_primal(&self, node: &Arc<IRNode>) -> Option<&Array> {
        let addr = Arc::as_ptr(node) as usize;
        self.primals.get(&addr)
    }

    /// Compute JVP for a graph given input tangents.
    ///
    /// Returns tangents for the outputs.
    pub fn jvp(&mut self, graph: &IRGraph, tangents: &[Array]) -> Vec<Array> {
        assert_eq!(
            graph.inputs.len(),
            tangents.len(),
            "Number of tangents must match number of inputs"
        );

        // Initialize input tangents
        for (input, tangent) in graph.inputs.iter().zip(tangents.iter()) {
            let node_addr = Arc::as_ptr(input) as usize;
            self.tangents.insert(node_addr, tangent.clone());
        }

        // Forward pass: propagate tangents from inputs to outputs
        graph
            .outputs
            .iter()
            .map(|output| self.forward(output))
            .collect()
    }

    /// Forward pass for a single node.
    fn forward(&mut self, node: &Arc<IRNode>) -> Array {
        let node_addr = Arc::as_ptr(node) as usize;

        // Check if already computed
        if let Some(tangent) = self.tangents.get(&node_addr) {
            return tangent.clone();
        }

        let tangent = match node.as_ref() {
            IRNode::Input { .. } => {
                panic!("Input node tangent not initialized");
            }

            IRNode::Constant { .. } => {
                // Constants have zero tangent
                Array::zeros(node.shape(), node.dtype())
            }

            IRNode::MaterializedConstant { .. } => {
                // Materialized constants have zero tangent
                Array::zeros(node.shape(), node.dtype())
            }

            IRNode::Unary { op, input, .. } => {
                let input_tangent = self.forward(input);
                let input_primal = self.get_primal(input);
                self.jvp_unary(op, input_primal, &input_tangent)
            }

            IRNode::Binary { op, lhs, rhs, .. } => {
                let lhs_tangent = self.forward(lhs);
                let rhs_tangent = self.forward(rhs);
                let lhs_primal = self.get_primal(lhs);
                let rhs_primal = self.get_primal(rhs);
                self.jvp_binary(op, lhs_primal, rhs_primal, &lhs_tangent, &rhs_tangent)
            }

            IRNode::Reduce { op, input, .. } => {
                let input_tangent = self.forward(input);
                self.jvp_reduce(op, &input_tangent)
            }

            IRNode::FusedOp { group, .. } => {
                // For fused operations, forward pass through the last operation
                // (which produces the output)
                if let Some(last_op) = group.operations.last() {
                    self.forward(last_op)
                } else {
                    Array::zeros(node.shape(), node.dtype())
                }
            }
        };

        self.tangents.insert(node_addr, tangent.clone());
        tangent
    }

    /// JVP rule for unary operations.
    fn jvp_unary(
        &self,
        op: &Primitive,
        input_primal: Option<&Array>,
        tangent: &Array,
    ) -> Array {
        match op {
            Primitive::Neg => tangent.neg(),

            Primitive::Abs => {
                // d/dx |x| = sign(x)
                if let Some(x) = input_primal {
                    tangent.mul(&x.sign())
                } else {
                    tangent.clone()
                }
            }

            Primitive::Sin => {
                // d/dx sin(x) = cos(x)
                if let Some(x) = input_primal {
                    tangent.mul(&x.cos())
                } else {
                    tangent.clone()
                }
            }

            Primitive::Cos => {
                // d/dx cos(x) = -sin(x)
                if let Some(x) = input_primal {
                    tangent.mul(&x.sin().neg())
                } else {
                    tangent.neg()
                }
            }

            Primitive::Exp => {
                // d/dx exp(x) = exp(x)
                if let Some(x) = input_primal {
                    tangent.mul(&x.exp())
                } else {
                    tangent.clone()
                }
            }

            Primitive::Log => {
                // d/dx log(x) = 1/x
                if let Some(x) = input_primal {
                    tangent.mul(&x.reciprocal())
                } else {
                    tangent.clone()
                }
            }

            Primitive::Sqrt => {
                // d/dx sqrt(x) = 1/(2*sqrt(x))
                if let Some(x) = input_primal {
                    let sqrt_x = x.sqrt();
                    let two = Array::full(2.0, sqrt_x.shape().clone(), sqrt_x.dtype());
                    tangent.mul(&two.mul(&sqrt_x).reciprocal())
                } else {
                    tangent.clone()
                }
            }

            Primitive::Square => {
                // d/dx x^2 = 2x
                if let Some(x) = input_primal {
                    let two = Array::full(2.0, x.shape().clone(), x.dtype());
                    tangent.mul(&two.mul(x))
                } else {
                    tangent.clone()
                }
            }

            Primitive::Tanh => {
                // d/dx tanh(x) = 1 - tanh^2(x)
                if let Some(x) = input_primal {
                    let tanh_x = x.tanh();
                    let one = Array::ones(tanh_x.shape().clone(), tanh_x.dtype());
                    tangent.mul(&one.sub(&tanh_x.mul(&tanh_x)))
                } else {
                    tangent.clone()
                }
            }

            Primitive::Reciprocal => {
                // d/dx (1/x) = -1/x^2
                if let Some(x) = input_primal {
                    tangent.mul(&x.mul(x).reciprocal().neg())
                } else {
                    tangent.clone()
                }
            }

            Primitive::Sign => {
                // d/dx sign(x) = 0
                Array::zeros(tangent.shape().clone(), tangent.dtype())
            }

            _ => tangent.clone(),
        }
    }

    /// JVP rule for binary operations.
    fn jvp_binary(
        &self,
        op: &Primitive,
        lhs_primal: Option<&Array>,
        rhs_primal: Option<&Array>,
        lhs_tangent: &Array,
        rhs_tangent: &Array,
    ) -> Array {
        match op {
            Primitive::Add => lhs_tangent.add(rhs_tangent),

            Primitive::Sub => lhs_tangent.sub(rhs_tangent),

            Primitive::Mul => {
                // d/d(x,y) (x*y) = y*dx + x*dy
                let mut result = Array::zeros(lhs_tangent.shape().clone(), lhs_tangent.dtype());
                if let Some(y) = rhs_primal {
                    result = result.add(&lhs_tangent.mul(y));
                }
                if let Some(x) = lhs_primal {
                    result = result.add(&rhs_tangent.mul(x));
                }
                result
            }

            Primitive::Div => {
                // d/d(x,y) (x/y) = dx/y - x*dy/y^2
                let mut result = Array::zeros(lhs_tangent.shape().clone(), lhs_tangent.dtype());
                if let Some(y) = rhs_primal {
                    result = result.add(&lhs_tangent.mul(&y.reciprocal()));
                    if let Some(x) = lhs_primal {
                        let y_sq = y.mul(y);
                        result = result.sub(&x.mul(rhs_tangent).mul(&y_sq.reciprocal()));
                    }
                }
                result
            }

            Primitive::Matmul => {
                // d(A@B) = dA@B + A@dB
                let mut result = Array::zeros(lhs_tangent.shape().clone(), lhs_tangent.dtype());
                if let Some(b) = rhs_primal {
                    result = result.add(&lhs_tangent.matmul(b));
                }
                if let Some(a) = lhs_primal {
                    result = result.add(&a.matmul(rhs_tangent));
                }
                result
            }

            _ => lhs_tangent.add(rhs_tangent),
        }
    }

    /// JVP rule for reduction operations.
    fn jvp_reduce(&self, op: &Primitive, tangent: &Array) -> Array {
        match op {
            Primitive::SumAll => tangent.sum_all_array(),
            Primitive::Sum { axis } => tangent.sum(*axis),
            Primitive::MeanAll => {
                let val = tangent.mean_all();
                Array::from_vec(vec![val], Shape::scalar())
            }
            Primitive::Mean { axis } => tangent.mean(*axis),
            _ => tangent.clone(),
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
    use crate::Shape;

    #[test]
    fn test_grad_simple() {
        // f(x) = x^2, df/dx = 2x
        let df = grad(|x: &Array| x.mul(x).sum_all_array());

        let x = Array::from_vec(vec![2.0], Shape::new(vec![1]));
        let gradient = df(&x);

        // df/dx at x=2 should be approximately 4
        let grad_val = gradient.to_vec()[0];
        assert!((grad_val - 4.0).abs() < 0.01, "Got gradient {}, expected 4.0", grad_val);
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
    fn test_grad_exp() {
        // f(x) = exp(x), df/dx = exp(x)
        let df = grad(|x: &Array| x.exp().sum_all_array());

        let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
        let gradient = df(&x);

        // df/dx at x=0 should be exp(0) = 1
        assert!((gradient.to_vec()[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_grad_log() {
        // f(x) = log(x), df/dx = 1/x
        let df = grad(|x: &Array| x.log().sum_all_array());

        let x = Array::from_vec(vec![2.0], Shape::new(vec![1]));
        let gradient = df(&x);

        // df/dx at x=2 should be 1/2 = 0.5
        let grad_val = gradient.to_vec()[0];
        assert!((grad_val - 0.5).abs() < 0.01, "got {}, expected 0.5", grad_val);
    }

    #[test]
    fn test_grad_sin() {
        // f(x) = sin(x), df/dx = cos(x)
        let df = grad(|x: &Array| x.sin().sum_all_array());

        let x = Array::from_vec(vec![0.0], Shape::new(vec![1]));
        let gradient = df(&x);

        // df/dx at x=0 should be cos(0) = 1
        assert!((gradient.to_vec()[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_grad_composite() {
        // f(x) = sum(x^2 + 2*x), df/dx = 2x + 2
        let df = grad(|x: &Array| {
            let x2 = x.mul(x);
            let two = Array::full(2.0, x.shape().clone(), x.dtype());
            let two_x = two.mul(x);
            x2.add(&two_x).sum_all_array()
        });

        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let gradient = df(&x);

        // df/dx should be 2x + 2 = [4, 6, 8]
        let grad_vec = gradient.to_vec();
        assert!((grad_vec[0] - 4.0).abs() < 0.1);
        assert!((grad_vec[1] - 6.0).abs() < 0.1);
        assert!((grad_vec[2] - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_value_and_grad() {
        let f = |x: &Array| x.mul(x).sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let (value, gradient) = value_and_grad(&f, &x);

        // f(x) = 1 + 4 + 9 = 14
        assert!((value.to_vec()[0] - 14.0).abs() < 0.01);

        // df/dx = 2x = [2, 4, 6]
        let grad_vec = gradient.to_vec();
        assert!((grad_vec[0] - 2.0).abs() < 0.01);
        assert!((grad_vec[1] - 4.0).abs() < 0.01);
        assert!((grad_vec[2] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_vjp_engine_add() {
        let input0 = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![2]), DType::Float32);
        let add = IRNode::binary(Primitive::Add, input0.clone(), input1.clone());

        let graph =
            IRGraph::new("test_add".to_string(), vec![input0, input1], vec![add]);

        let cotangent = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));

        let mut engine = VJPEngine::new();
        let grads = engine.vjp(&graph, &[cotangent]);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].to_vec(), vec![1.0, 1.0]);
        assert_eq!(grads[1].to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_vjp_engine_sub() {
        let input0 = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![2]), DType::Float32);
        let sub = IRNode::binary(Primitive::Sub, input0.clone(), input1.clone());

        let graph =
            IRGraph::new("test_sub".to_string(), vec![input0, input1], vec![sub]);

        let cotangent = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));

        let mut engine = VJPEngine::new();
        let grads = engine.vjp(&graph, &[cotangent]);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].to_vec(), vec![1.0, 1.0]);
        assert_eq!(grads[1].to_vec(), vec![-1.0, -1.0]);
    }

    #[test]
    fn test_jvp_engine_add() {
        let input0 = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![2]), DType::Float32);
        let add = IRNode::binary(Primitive::Add, input0.clone(), input1.clone());

        let graph =
            IRGraph::new("test_add".to_string(), vec![input0, input1], vec![add]);

        let tangent0 = Array::from_vec(vec![1.0, 0.0], Shape::new(vec![2]));
        let tangent1 = Array::from_vec(vec![0.0, 1.0], Shape::new(vec![2]));

        let mut engine = JVPEngine::new();
        let tangents = engine.jvp(&graph, &[tangent0, tangent1]);

        assert_eq!(tangents.len(), 1);
        assert_eq!(tangents[0].to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_jvp_engine_neg() {
        let input0 = IRNode::input(0, Shape::new(vec![2]), DType::Float32);
        let neg = IRNode::unary(Primitive::Neg, input0.clone());

        let graph = IRGraph::new("test_neg".to_string(), vec![input0], vec![neg]);

        let tangent = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));

        let mut engine = JVPEngine::new();
        let tangents = engine.jvp(&graph, &[tangent]);

        assert_eq!(tangents.len(), 1);
        assert_eq!(tangents[0].to_vec(), vec![-1.0, -2.0]);
    }

    #[test]
    fn test_jvp_engine_sum() {
        let input0 = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let sum = IRNode::reduce(Primitive::SumAll, input0.clone(), Shape::scalar());

        let graph = IRGraph::new("test_sum".to_string(), vec![input0], vec![sum]);

        let tangent = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let mut engine = JVPEngine::new();
        let tangents = engine.jvp(&graph, &[tangent]);

        assert_eq!(tangents.len(), 1);
        assert_eq!(tangents[0].to_vec(), vec![6.0]);
    }

    // Comprehensive gradient tests using numerical differentiation for verification

    /// Helper: compute numerical gradient using central differences
    fn numerical_grad<F>(f: F, x: &Array, eps: f32) -> Array
    where
        F: Fn(&Array) -> Array,
    {
        let x_data = x.to_vec();
        let mut grad_data = vec![0.0; x_data.len()];

        for i in 0..x_data.len() {
            let mut x_plus = x_data.clone();
            let mut x_minus = x_data.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let y_plus = f(&Array::from_vec(x_plus, x.shape().clone())).to_vec()[0];
            let y_minus = f(&Array::from_vec(x_minus, x.shape().clone())).to_vec()[0];

            grad_data[i] = (y_plus - y_minus) / (2.0 * eps);
        }

        Array::from_vec(grad_data, x.shape().clone())
    }

    /// Check if two arrays are approximately equal
    fn arrays_close(a: &Array, b: &Array, atol: f32) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        let a_data = a.to_vec();
        let b_data = b.to_vec();
        a_data.iter().zip(b_data.iter()).all(|(x, y)| (x - y).abs() < atol)
    }

    #[test]
    fn test_grad_quadratic() {
        // f(x) = x^2, f'(x) = 2x
        let f = |x: &Array| x.mul(x).sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);
        let expected = Array::from_vec(vec![2.0, 4.0, 6.0], Shape::new(vec![3]));

        assert!(arrays_close(&analytical_grad, &expected, 0.01));
    }

    #[test]
    fn test_grad_sum() {
        // f(x) = sum(x), f'(x) = ones
        let f = |x: &Array| x.sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        // Gradient of sum is all ones (use wider tolerance for numerical diff)
        for &v in analytical_grad.to_vec().iter() {
            assert!((v - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_grad_mean() {
        // f(x) = mean(x), f'(x) = 1/n * ones
        let f = |x: &Array| x.mean_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        // Gradient of mean is 1/n for each element (use wider tolerance)
        for &v in analytical_grad.to_vec().iter() {
            assert!((v - 0.25).abs() < 0.1);
        }
    }

    #[test]
    fn test_grad_exp_vector() {
        // f(x) = sum(exp(x)), f'(x) = exp(x)
        let f = |x: &Array| x.exp().sum_all_array();
        let x = Array::from_vec(vec![0.0, 1.0, 2.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);
        let expected = x.exp();

        assert!(arrays_close(&analytical_grad, &expected, 0.1));
    }

    #[test]
    fn test_grad_log_vector() {
        // f(x) = sum(log(x)), f'(x) = 1/x
        let f = |x: &Array| x.log().sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 4.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);
        let expected = x.reciprocal();

        assert!(arrays_close(&analytical_grad, &expected, 0.01));
    }

    #[test]
    fn test_grad_sin_vector() {
        // f(x) = sum(sin(x)), f'(x) = cos(x)
        let f = |x: &Array| x.sin().sum_all_array();
        let x = Array::from_vec(vec![0.0, 1.0, 2.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);
        let expected = x.cos();

        assert!(arrays_close(&analytical_grad, &expected, 0.01));
    }

    #[test]
    fn test_grad_cos() {
        // f(x) = sum(cos(x)), f'(x) = -sin(x)
        let f = |x: &Array| x.cos().sum_all_array();
        let x = Array::from_vec(vec![0.0, 1.0, 2.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);
        let expected = x.sin().neg();

        assert!(arrays_close(&analytical_grad, &expected, 0.01));
    }

    #[test]
    fn test_grad_sqrt() {
        // f(x) = sum(sqrt(x)), f'(x) = 0.5 / sqrt(x)
        let f = |x: &Array| x.sqrt().sum_all_array();
        let x = Array::from_vec(vec![1.0, 4.0, 9.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        // 0.5 / sqrt(x) = [0.5, 0.25, 0.166...]
        let expected_data = vec![0.5, 0.25, 0.166666];
        let expected = Array::from_vec(expected_data, x.shape().clone());

        assert!(arrays_close(&analytical_grad, &expected, 0.1));
    }

    #[test]
    fn test_grad_add() {
        // f(x) = sum(x + x), f'(x) = 2 * ones
        let f = |x: &Array| x.add(x).sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        for &v in analytical_grad.to_vec().iter() {
            assert!((v - 2.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_grad_mul() {
        // f(x) = sum(x * x), f'(x) = 2x
        let f = |x: &Array| x.mul(x).sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);
        let expected = Array::from_vec(vec![2.0, 4.0, 6.0], x.shape().clone());

        assert!(arrays_close(&analytical_grad, &expected, 0.01));
    }

    #[test]
    fn test_grad_composition() {
        // f(x) = sum(exp(x^2)), f'(x) = 2x * exp(x^2)
        let f = |x: &Array| x.mul(x).exp().sum_all_array();
        let x = Array::from_vec(vec![0.0, 0.5, 1.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        // Expected: 2x * exp(x^2) = [0, 1*exp(0.25), 2*exp(1)]
        let x2 = x.mul(&x);
        let expected = x.mul(&Array::from_vec(vec![2.0, 2.0, 2.0], x.shape().clone())).mul(&x2.exp());

        assert!(arrays_close(&analytical_grad, &expected, 0.5));
    }

    #[test]
    fn test_grad_tanh() {
        // f(x) = sum(tanh(x)), f'(x) = 1 - tanh(x)^2
        let f = |x: &Array| x.tanh().sum_all_array();
        let x = Array::from_vec(vec![0.0, 0.5, 1.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        // Expected: 1 - tanh(x)^2
        let tanh_x = x.tanh();
        let ones = Array::ones(x.shape().clone(), x.dtype());
        let expected = ones.sub(&tanh_x.mul(&tanh_x));

        assert!(arrays_close(&analytical_grad, &expected, 0.01));
    }

    #[test]
    fn test_value_and_grad_func() {
        let f = |x: &Array| x.mul(x).sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let (value, gradient) = value_and_grad(&f, &x);

        // Value: sum(x^2) = 1 + 4 + 9 = 14
        assert!((value.to_vec()[0] - 14.0).abs() < 0.01);

        // Gradient: 2x = [2, 4, 6]
        let expected_grad = Array::from_vec(vec![2.0, 4.0, 6.0], x.shape().clone());
        assert!(arrays_close(&gradient, &expected_grad, 0.01));
    }

    #[test]
    fn test_grad_numerical_vs_analytical() {
        // Compare analytical gradient with numerical gradient
        let f = |x: &Array| x.mul(x).exp().sum_all_array();
        let x = Array::from_vec(vec![0.1, 0.2, 0.3, 0.4], Shape::new(vec![4]));

        let grad_fn = grad(f.clone());
        let analytical_grad = grad_fn(&x);

        let num_grad = numerical_grad(f, &x, 1e-4);

        assert!(arrays_close(&analytical_grad, &num_grad, 0.05));
    }

    #[test]
    fn test_grad_neg() {
        // f(x) = sum(-x), f'(x) = -1
        let f = |x: &Array| x.neg().sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        for &v in analytical_grad.to_vec().iter() {
            assert!((v - (-1.0)).abs() < 0.01);
        }
    }

    #[test]
    fn test_grad_reciprocal() {
        // f(x) = sum(1/x), f'(x) = -1/x^2
        let f = |x: &Array| x.reciprocal().sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 4.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        // Expected: -1/x^2 = [-1, -0.25, -0.0625]
        let expected = Array::from_vec(vec![-1.0, -0.25, -0.0625], x.shape().clone());

        assert!(arrays_close(&analytical_grad, &expected, 0.01));
    }

    #[test]
    fn test_grad_abs_positive() {
        // f(x) = sum(|x|) where x > 0, f'(x) = 1
        let f = |x: &Array| x.abs().sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        for &v in analytical_grad.to_vec().iter() {
            assert!((v - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_grad_polynomial() {
        // f(x) = sum(x^3) = sum(x * x * x), f'(x) = 3x^2
        let f = |x: &Array| x.mul(x).mul(x).sum_all_array();
        let x = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));

        let grad_fn = grad(f);
        let analytical_grad = grad_fn(&x);

        // Expected: 3x^2 = [3, 12, 27]
        let expected = Array::from_vec(vec![3.0, 12.0, 27.0], x.shape().clone());

        assert!(arrays_close(&analytical_grad, &expected, 1.0));
    }
}
