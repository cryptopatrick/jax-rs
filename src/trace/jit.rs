//! JIT compilation infrastructure.
//!
//! Provides the `jit` function that traces and caches compiled versions
//! of functions for improved performance.

use crate::trace::Interpreter;
use crate::Array;
use crate::Shape;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A traced and compiled function.
///
/// This is returned by the `jit()` function and caches the traced
/// computation graph for repeated execution.
pub struct JitFunction<F> {
    /// Original function
    function: F,
    /// Cache of compiled versions (keyed by input shapes)
    cache: Arc<Mutex<HashMap<Vec<Shape>, CompiledFunction>>>,
    /// Name for debugging
    name: String,
}

/// A compiled version of a function for specific input shapes.
/// Stores the IR graph and interpreter for cached execution.
struct CompiledFunction {
    /// IR graph
    graph: crate::trace::IRGraph,
    /// Interpreter for executing the graph
    interpreter: Mutex<Interpreter>,
}

impl<F> JitFunction<F>
where
    F: Fn(&[Array]) -> Vec<Array>,
{
    /// Create a new JIT-compiled function.
    pub fn new(name: String, function: F) -> Self {
        Self { function, cache: Arc::new(Mutex::new(HashMap::new())), name }
    }

    /// Execute the function, using cached compilation if available.
    pub fn call(&self, inputs: &[Array]) -> Vec<Array> {
        let shapes: Vec<Shape> =
            inputs.iter().map(|a| a.shape().clone()).collect();

        // Check cache
        {
            let cache = self.cache.lock().unwrap();
            if let Some(compiled) = cache.get(&shapes) {
                // Cache hit - execute using the cached IR graph if available
                if compiled.graph.num_outputs() > 0 {
                    let mut interp = compiled.interpreter.lock().unwrap();
                    return interp.execute(&compiled.graph, inputs);
                } else {
                    // Placeholder graph - execute eagerly
                    return (self.function)(inputs);
                }
            }
        }

        // Cache miss - trace and compile
        // For now, we execute eagerly and store a placeholder IR
        // Full tracing integration with Array operations comes in future phases

        let results = (self.function)(inputs);

        // Store in cache with optimized IR
        // In a full implementation, we would trace the function execution
        // and build a real IR graph here
        {
            let mut cache = self.cache.lock().unwrap();

            // Build placeholder graph (full tracing comes later)
            let graph = crate::trace::IRGraph::new(
                self.name.clone(),
                vec![],
                vec![],
            );

            // Apply fusion optimization
            let (optimized_graph, fusion_groups) = crate::trace::optimize_fusion(&graph);

            // Log fusion statistics
            if !fusion_groups.is_empty() {
                let total_ops: usize = fusion_groups.iter().map(|g| g.operations.len()).sum();
                eprintln!(
                    "[JIT] {} fused {} groups with {} total ops",
                    self.name,
                    fusion_groups.len(),
                    total_ops
                );
            }

            cache.insert(
                shapes,
                CompiledFunction {
                    graph: optimized_graph,
                    interpreter: Mutex::new(Interpreter::new()),
                },
            );
        }

        results
    }
}

/// JIT-compile a function.
///
/// This traces the function on first call with given input shapes,
/// compiles it to an optimized kernel, and caches the result.
///
/// # Examples
///
/// ```rust,ignore
/// use jax_rs::jit;
///
/// let f = jit("my_function", |inputs: &[Array]| {
///     vec![inputs[0].add(&inputs[1]).mul(&inputs[0])]
/// });
///
/// let result = f.call(&[a, b]);
/// ```
pub fn jit<F>(name: &str, function: F) -> JitFunction<F>
where
    F: Fn(&[Array]) -> Vec<Array>,
{
    JitFunction::new(name.to_string(), function)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_basic() {
        let f =
            jit(
                "test_add",
                |inputs: &[Array]| vec![inputs[0].add(&inputs[1])],
            );

        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));

        let result = f.call(&[a, b]);
        assert_eq!(result[0].to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_jit_caching() {
        let f =
            jit(
                "test_mul",
                |inputs: &[Array]| vec![inputs[0].mul(&inputs[1])],
            );

        let a = Array::from_vec(vec![2.0, 3.0], Shape::new(vec![2]));
        let b = Array::from_vec(vec![4.0, 5.0], Shape::new(vec![2]));

        // First call - should trace
        let result1 = f.call(&[a.clone(), b.clone()]);
        assert_eq!(result1[0].to_vec(), vec![8.0, 15.0]);

        // Second call - should use cache
        let result2 = f.call(&[a, b]);
        assert_eq!(result2[0].to_vec(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_jit_complex() {
        let f = jit("complex", |inputs: &[Array]| {
            let x = &inputs[0];
            let y = &inputs[1];
            let z = x.add(y).mul(x).sqrt();
            vec![z]
        });

        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![3.0, 2.0, 1.0], Shape::new(vec![3]));

        let result = f.call(&[a, b]);
        // (a + b) * a -> [4, 8, 12], sqrt -> [2, 2.828..., 3.464...]
        assert!((result[0].to_vec()[0] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_jit_with_manual_ir() {
        use crate::trace::{IRNode, Primitive};
        use crate::DType;

        // Manually build an IR graph: add(input0, input1)
        let input0 = IRNode::input(0, Shape::new(vec![3]), DType::Float32);
        let input1 = IRNode::input(1, Shape::new(vec![3]), DType::Float32);
        let add =
            IRNode::binary(Primitive::Add, input0.clone(), input1.clone());

        let graph = crate::trace::IRGraph::new(
            "manual_add".to_string(),
            vec![input0, input1],
            vec![add],
        );

        // Create a compiled function with this graph
        let compiled = CompiledFunction {
            graph,
            interpreter: Mutex::new(Interpreter::new()),
        };

        // Execute the compiled function
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));

        let mut interp = compiled.interpreter.lock().unwrap();
        let results = interp.execute(&compiled.graph, &[a, b]);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].to_vec(), vec![5.0, 7.0, 9.0]);
    }
}
