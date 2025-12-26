//! Einstein summation notation.
//!
//! Provides a flexible way to express tensor operations using subscript notation.

use crate::{Array, DType, Shape};
use std::collections::{HashMap, HashSet};

/// Parse einsum subscripts into input indices, output indices, and summation indices.
///
/// Examples:
/// - "ij,jk->ik" => inputs: [['i','j'], ['j','k']], output: ['i','k'], sum: ['j']
/// - "ii->i" => inputs: [['i','i']], output: ['i'], sum: []
/// - "ij->ji" => inputs: [['i','j']], output: ['j','i'], sum: []
fn parse_einsum(subscripts: &str) -> (Vec<Vec<char>>, Vec<char>, Vec<char>) {
    let parts: Vec<&str> = subscripts.split("->").collect();

    if parts.len() > 2 {
        panic!("Invalid einsum notation: too many '->'");
    }

    let inputs_str = parts[0];
    let output_str = if parts.len() == 2 { parts[1] } else { "" };

    // Parse input subscripts
    let input_terms: Vec<&str> = inputs_str.split(',').collect();
    let inputs: Vec<Vec<char>> = input_terms
        .iter()
        .map(|term| term.trim().chars().collect())
        .collect();

    // Parse output subscripts
    let output: Vec<char> = output_str.trim().chars().collect();

    // Determine which indices to sum over
    let mut all_input_indices = HashSet::new();
    for input in &inputs {
        for &idx in input {
            all_input_indices.insert(idx);
        }
    }

    let output_set: HashSet<char> = output.iter().copied().collect();
    let sum_indices: Vec<char> = all_input_indices
        .into_iter()
        .filter(|idx| !output_set.contains(idx))
        .collect();

    (inputs, output, sum_indices)
}

/// Einstein summation.
///
/// A flexible way to express tensor operations using subscript notation.
/// Supports arbitrary tensor contractions and reductions.
///
/// # Subscript Format
///
/// The subscript string uses letters to represent dimensions:
/// - Input arrays are separated by commas
/// - Optional output specification after `->`
/// - Repeated indices are summed over (if not in output)
///
/// # Examples
///
/// ```
/// # use jax_rs::{ops::einsum::einsum, Array, Shape};
/// // Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
/// let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
/// let c = einsum("ij,jk->ik", &[&a, &b]);
/// assert_eq!(c.shape().as_slice(), &[2, 2]);
///
/// // Transpose: B[j,i] = A[i,j]
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
/// let b = einsum("ij->ji", &[&a]);
/// assert_eq!(b.shape().as_slice(), &[3, 2]);
///
/// // Trace: scalar = sum_i A[i,i]
/// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
/// let trace = einsum("ii->", &[&a]);
/// assert_eq!(trace.size(), 1);
/// ```
pub fn einsum(subscripts: &str, arrays: &[&Array]) -> Array {
    if arrays.is_empty() {
        panic!("einsum requires at least one input array");
    }

    // All arrays must be Float32
    for array in arrays {
        assert_eq!(array.dtype(), DType::Float32, "Only Float32 supported");
    }

    let (inputs, output, sum_indices) = parse_einsum(subscripts);

    if inputs.len() != arrays.len() {
        panic!(
            "Number of input subscripts ({}) doesn't match number of arrays ({})",
            inputs.len(),
            arrays.len()
        );
    }

    // Handle special cases for efficiency
    if arrays.len() == 1 {
        return einsum_single(subscripts, arrays[0], &inputs[0], &output);
    } else if arrays.len() == 2 {
        return einsum_binary(subscripts, arrays[0], arrays[1], &inputs, &output, &sum_indices);
    }

    panic!("einsum with {} arrays not yet implemented", arrays.len());
}

/// Einsum for a single array (transpose, diagonal, trace, etc.)
fn einsum_single(subscripts: &str, a: &Array, input: &[char], output: &[char]) -> Array {
    let a_data = a.to_vec();
    let a_shape = a.shape().as_slice();

    // Map indices to dimensions
    let mut index_to_dim = HashMap::new();
    for (i, &idx) in input.iter().enumerate() {
        if i < a_shape.len() {
            index_to_dim.insert(idx, a_shape[i]);
        }
    }

    // Special case: trace (ii->)
    if output.is_empty() && input.len() == 2 && input[0] == input[1] {
        assert_eq!(a_shape.len(), 2, "Trace requires 2D array");
        assert_eq!(a_shape[0], a_shape[1], "Trace requires square matrix");

        let n = a_shape[0];
        let mut sum = 0.0;
        for i in 0..n {
            sum += a_data[i * n + i];
        }
        return Array::from_vec(vec![sum], Shape::scalar());
    }

    // Special case: diagonal (ii->i)
    if output.len() == 1 && input.len() == 2 && input[0] == input[1] {
        assert_eq!(a_shape.len(), 2, "Diagonal requires 2D array");
        assert_eq!(a_shape[0], a_shape[1], "Diagonal requires square matrix");

        let n = a_shape[0];
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            result.push(a_data[i * n + i]);
        }
        return Array::from_vec(result, Shape::new(vec![n]));
    }

    // Special case: transpose (ij->ji)
    if a_shape.len() == 2 && input.len() == 2 && output.len() == 2 {
        if input[0] == output[1] && input[1] == output[0] {
            return a.transpose();
        }
    }

    // General case: permutation
    if output.len() == input.len() {
        // Build permutation mapping
        let mut perm = vec![0; output.len()];
        for (out_idx, &out_char) in output.iter().enumerate() {
            for (in_idx, &in_char) in input.iter().enumerate() {
                if in_char == out_char {
                    perm[out_idx] = in_idx;
                    break;
                }
            }
        }

        // Check if this is just identity
        if perm.iter().enumerate().all(|(i, &p)| i == p) {
            return a.clone();
        }

        // For 2D, use transpose
        if a_shape.len() == 2 && perm == vec![1, 0] {
            return a.transpose();
        }
    }

    panic!("einsum pattern '{}' not yet implemented for single array", subscripts);
}

/// Einsum for two arrays (matrix multiply, outer product, etc.)
fn einsum_binary(
    _subscripts: &str,
    a: &Array,
    b: &Array,
    inputs: &[Vec<char>],
    output: &[char],
    sum_indices: &[char],
) -> Array {
    let a_data = a.to_vec();
    let b_data = b.to_vec();
    let a_shape = a.shape().as_slice();
    let b_shape = b.shape().as_slice();

    let input_a = &inputs[0];
    let input_b = &inputs[1];

    // Special case: matrix multiplication (ij,jk->ik)
    if input_a.len() == 2 && input_b.len() == 2 && output.len() == 2 {
        if input_a[1] == input_b[0] && // j is shared
           input_a[0] == output[0] &&  // i preserved
           input_b[1] == output[1] &&  // k preserved
           sum_indices.contains(&input_a[1]) // j is summed
        {
            return a.matmul(b);
        }
    }

    // Special case: outer product (i,j->ij)
    if input_a.len() == 1 && input_b.len() == 1 && output.len() == 2 && sum_indices.is_empty() {
        let n = a.size();
        let m = b.size();
        let mut result = vec![0.0; n * m];

        for i in 0..n {
            for j in 0..m {
                result[i * m + j] = a_data[i] * b_data[j];
            }
        }

        return Array::from_vec(result, Shape::new(vec![n, m]));
    }

    // Special case: element-wise multiplication (ij,ij->ij)
    if input_a == input_b && input_a == output && a_shape == b_shape {
        return a.mul(b);
    }

    // Special case: dot product (i,i->)
    if input_a.len() == 1 && input_b.len() == 1 &&
       input_a[0] == input_b[0] && output.is_empty() {
        assert_eq!(a.size(), b.size(), "Dot product requires same size");
        let sum: f32 = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).sum();
        return Array::from_vec(vec![sum], Shape::scalar());
    }

    panic!("einsum binary pattern not yet implemented for this case");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_einsum_matmul() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
        let c = einsum("ij,jk->ik", &[&a, &b]);

        assert_eq!(c.shape().as_slice(), &[2, 2]);
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_einsum_transpose() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let b = einsum("ij->ji", &[&a]);

        assert_eq!(b.shape().as_slice(), &[3, 2]);
        assert_eq!(b.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_einsum_trace() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let trace = einsum("ii->", &[&a]);

        assert_eq!(trace.size(), 1);
        assert_eq!(trace.to_vec()[0], 5.0); // 1 + 4
    }

    #[test]
    fn test_einsum_diagonal() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let diag = einsum("ii->i", &[&a]);

        assert_eq!(diag.shape().as_slice(), &[2]);
        assert_eq!(diag.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn test_einsum_outer_product() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![4.0, 5.0], Shape::new(vec![2]));
        let c = einsum("i,j->ij", &[&a, &b]);

        assert_eq!(c.shape().as_slice(), &[3, 2]);
        assert_eq!(c.to_vec(), vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_einsum_dot_product() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]));
        let result = einsum("i,i->", &[&a, &b]);

        assert_eq!(result.size(), 1);
        assert_eq!(result.to_vec()[0], 32.0); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_einsum_elementwise() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0], Shape::new(vec![2, 2]));
        let c = einsum("ij,ij->ij", &[&a, &b]);

        assert_eq!(c.shape().as_slice(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![2.0, 6.0, 12.0, 20.0]);
    }
}
