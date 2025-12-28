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

    // For N arrays (N > 2), chain binary operations
    // Strategy: Contract arrays pairwise from left to right
    einsum_chain(arrays, &inputs, &output)
}

/// Chain einsum for N arrays by iteratively contracting pairs.
///
/// For example, "ij,jk,kl->il" with [A, B, C]:
/// 1. Contract A and B: D = einsum("ij,jk->ik", [A, B])
/// 2. Contract D and C: result = einsum("ik,kl->il", [D, C])
fn einsum_chain(arrays: &[&Array], inputs: &[Vec<char>], output: &[char]) -> Array {
    assert!(arrays.len() >= 3, "einsum_chain requires 3+ arrays");

    // Build a map of all index dimensions
    let mut index_dims: HashMap<char, usize> = HashMap::new();
    for (arr, input) in arrays.iter().zip(inputs.iter()) {
        let shape = arr.shape().as_slice();
        for (i, &idx) in input.iter().enumerate() {
            if i < shape.len() {
                if let Some(&existing) = index_dims.get(&idx) {
                    assert_eq!(existing, shape[i],
                        "Dimension mismatch for index '{}': {} vs {}", idx, existing, shape[i]);
                } else {
                    index_dims.insert(idx, shape[i]);
                }
            }
        }
    }

    // Count how many times each index appears across all inputs
    let mut index_counts: HashMap<char, usize> = HashMap::new();
    for input in inputs {
        for &idx in input {
            *index_counts.entry(idx).or_insert(0) += 1;
        }
    }

    // Determine output set
    let output_set: HashSet<char> = output.iter().copied().collect();

    // Start with the first array
    let mut result = arrays[0].clone();
    let mut result_indices: Vec<char> = inputs[0].clone();

    // Iteratively contract with remaining arrays
    for i in 1..arrays.len() {
        let next = arrays[i];
        let next_indices = &inputs[i];

        // Find indices to sum over (appear in both result and next, but not in final output,
        // unless this is not the last contraction and they appear in later arrays)
        let result_set: HashSet<char> = result_indices.iter().copied().collect();
        let next_set: HashSet<char> = next_indices.iter().copied().collect();

        // Indices in common between result and next
        let common: HashSet<char> = result_set.intersection(&next_set).copied().collect();

        // For intermediate contractions, we keep indices that appear later
        let is_last = i == arrays.len() - 1;
        let later_indices: HashSet<char> = if is_last {
            HashSet::new()
        } else {
            inputs[i+1..].iter()
                .flat_map(|inp| inp.iter().copied())
                .collect()
        };

        // Indices to sum: common indices not in output and not needed later
        let sum_indices: Vec<char> = common.iter()
            .filter(|idx| !output_set.contains(idx) && !later_indices.contains(idx))
            .copied()
            .collect();

        // New result indices: all from result + all from next, minus summed
        let mut new_indices: Vec<char> = Vec::new();
        for &idx in &result_indices {
            if !sum_indices.contains(&idx) {
                new_indices.push(idx);
            }
        }
        for &idx in next_indices.iter() {
            if !sum_indices.contains(&idx) && !new_indices.contains(&idx) {
                new_indices.push(idx);
            }
        }

        // Build subscript string for this binary operation
        let result_str: String = result_indices.iter().collect();
        let next_str: String = next_indices.iter().collect();
        let new_str: String = new_indices.iter().collect();
        let binary_subscript = format!("{},{}->{}", result_str, next_str, new_str);

        // Perform the binary einsum
        let binary_inputs = vec![result_indices.clone(), next_indices.clone()];
        result = einsum_binary(
            &binary_subscript,
            &result,
            next,
            &binary_inputs,
            &new_indices,
            &sum_indices,
        );
        result_indices = new_indices;
    }

    // If result indices don't match output, we need a final permutation/reduction
    if result_indices != output.to_vec() {
        // Check if we need to sum out remaining indices
        let result_set: HashSet<char> = result_indices.iter().copied().collect();
        let remaining_sum: Vec<char> = result_set
            .iter()
            .filter(|idx| !output_set.contains(idx))
            .copied()
            .collect();

        if !remaining_sum.is_empty() {
            // Sum over remaining indices using reshape and sum
            // For simplicity, handle the common case
            if remaining_sum.len() == result_indices.len() && output.is_empty() {
                // Sum all
                return Array::from_vec(vec![result.sum_all()], Shape::scalar());
            }
        }

        // Permute to match output order if needed
        if result_indices.len() == output.len() {
            // Build permutation
            let mut perm = vec![0usize; output.len()];
            for (out_idx, out_char) in output.iter().enumerate() {
                for (res_idx, res_char) in result_indices.iter().enumerate() {
                    if res_char == out_char {
                        perm[out_idx] = res_idx;
                        break;
                    }
                }
            }

            // Apply permutation if not identity
            if !perm.iter().enumerate().all(|(i, &p)| i == p)
                && perm.len() == 2 && perm == vec![1, 0] {
                    result = result.transpose();
                }
                // For higher dimensions, would need transpose_axes
        }
    }

    result
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
    if a_shape.len() == 2 && input.len() == 2 && output.len() == 2
        && input[0] == output[1] && input[1] == output[0] {
            return a.transpose();
        }

    // Sum reduction (e.g., "ij->" - sum all elements)
    if output.is_empty() {
        return Array::from_vec(vec![a.sum_all()], Shape::scalar());
    }

    // Axis sum reduction (e.g., "ij->i" - sum along j)
    // Find which input indices are not in output (to be summed)
    let input_set: HashSet<char> = input.iter().copied().collect();
    let output_set: HashSet<char> = output.iter().copied().collect();
    let sum_chars: Vec<char> = input_set.difference(&output_set).copied().collect();

    if !sum_chars.is_empty() && output.len() < input.len() {
        // General axis reduction
        let ndim = a_shape.len();

        // Build output shape
        let out_shape: Vec<usize> = output
            .iter()
            .map(|c| *index_to_dim.get(c).unwrap())
            .collect();
        let out_size: usize = out_shape.iter().product();
        let mut result = vec![0.0f32; out_size];

        // Compute strides for input
        let mut in_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            in_strides[i] = in_strides[i + 1] * a_shape[i + 1];
        }

        // Compute strides for output
        let mut out_strides = vec![1usize; output.len()];
        for i in (0..output.len().saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        // Iterate over all input elements
        let in_size = a.size();
        for flat in 0..in_size {
            // Convert flat index to multi-index
            let mut multi_idx = vec![0usize; ndim];
            let mut rem = flat;
            for i in 0..ndim {
                multi_idx[i] = rem / in_strides[i];
                rem %= in_strides[i];
            }

            // Compute output index
            let mut out_flat = 0usize;
            for (out_pos, &out_char) in output.iter().enumerate() {
                // Find which input dimension this corresponds to
                for (in_pos, &in_char) in input.iter().enumerate() {
                    if in_char == out_char {
                        out_flat += multi_idx[in_pos] * out_strides[out_pos];
                        break;
                    }
                }
            }

            result[out_flat] += a_data[flat];
        }

        return Array::from_vec(result, Shape::new(out_shape));
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

        // General N-dimensional permutation
        let ndim = a_shape.len();
        let out_shape: Vec<usize> = perm.iter().map(|&p| a_shape[p]).collect();
        let out_size: usize = out_shape.iter().product();
        let mut result = vec![0.0f32; out_size];

        // Compute strides
        let mut in_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            in_strides[i] = in_strides[i + 1] * a_shape[i + 1];
        }

        let mut out_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }

        // Iterate over all output positions
        for out_flat in 0..out_size {
            // Convert to output multi-index
            let mut out_idx = vec![0usize; ndim];
            let mut rem = out_flat;
            for i in 0..ndim {
                out_idx[i] = rem / out_strides[i];
                rem %= out_strides[i];
            }

            // Convert to input multi-index using permutation
            let mut in_flat = 0usize;
            for (out_pos, &p) in perm.iter().enumerate() {
                in_flat += out_idx[out_pos] * in_strides[p];
            }

            result[out_flat] = a_data[in_flat];
        }

        return Array::from_vec(result, Shape::new(out_shape));
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
    if input_a.len() == 2 && input_b.len() == 2 && output.len() == 2
        && input_a[1] == input_b[0] && // j is shared
           input_a[0] == output[0] &&  // i preserved
           input_b[1] == output[1] &&  // k preserved
           sum_indices.contains(&input_a[1]) // j is summed
        {
            return a.matmul(b);
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

    // Special case: batched matrix multiply (bij,bjk->bik or similar)
    if input_a.len() == 3 && input_b.len() == 3 && output.len() == 3 {
        // Check for batched matmul pattern: shared batch dim, contraction dim
        let batch_a = input_a[0];
        let batch_b = input_b[0];
        if batch_a == batch_b && batch_a == output[0] {
            // Check contraction: a[1] or a[2] matches b[1] or b[2]
            if input_a[2] == input_b[1] && input_a[1] == output[1] && input_b[2] == output[2] {
                // bij,bjk->bik pattern
                let batch = a_shape[0];
                let m = a_shape[1];
                let k = a_shape[2];
                let n = b_shape[2];

                assert_eq!(b_shape[0], batch, "Batch dimensions must match");
                assert_eq!(b_shape[1], k, "Contraction dimensions must match");

                let mut result = vec![0.0f32; batch * m * n];

                for b in 0..batch {
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for kk in 0..k {
                                let a_idx = b * m * k + i * k + kk;
                                let b_idx = b * k * n + kk * n + j;
                                sum += a_data[a_idx] * b_data[b_idx];
                            }
                            result[b * m * n + i * n + j] = sum;
                        }
                    }
                }

                return Array::from_vec(result, Shape::new(vec![batch, m, n]));
            }
        }
    }

    // General tensor contraction
    // Build index to dimension mapping
    let mut index_dims: HashMap<char, usize> = HashMap::new();
    for (i, &idx) in input_a.iter().enumerate() {
        if i < a_shape.len() {
            index_dims.insert(idx, a_shape[i]);
        }
    }
    for (i, &idx) in input_b.iter().enumerate() {
        if i < b_shape.len() {
            if let Some(&existing) = index_dims.get(&idx) {
                assert_eq!(existing, b_shape[i], "Dimension mismatch for index '{}'", idx);
            } else {
                index_dims.insert(idx, b_shape[i]);
            }
        }
    }

    // Compute output shape
    let out_shape: Vec<usize> = output.iter().map(|c| *index_dims.get(c).unwrap()).collect();
    let out_size: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };

    // Compute strides for inputs
    let mut a_strides = vec![1usize; a_shape.len()];
    for i in (0..a_shape.len().saturating_sub(1)).rev() {
        a_strides[i] = a_strides[i + 1] * a_shape[i + 1];
    }
    let mut b_strides = vec![1usize; b_shape.len()];
    for i in (0..b_shape.len().saturating_sub(1)).rev() {
        b_strides[i] = b_strides[i + 1] * b_shape[i + 1];
    }

    // Compute output strides
    let mut out_strides = vec![1usize; output.len()];
    for i in (0..output.len().saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Compute contraction dimensions (sum indices)
    let sum_dims: Vec<usize> = sum_indices.iter().map(|c| *index_dims.get(c).unwrap()).collect();
    let sum_size: usize = if sum_dims.is_empty() { 1 } else { sum_dims.iter().product() };

    let mut result = vec![0.0f32; out_size];

    // Iterate over all output positions
    for out_flat in 0..out_size {
        // Convert to output multi-index
        let mut out_idx: HashMap<char, usize> = HashMap::new();
        let mut rem = out_flat;
        for (pos, &c) in output.iter().enumerate() {
            let stride = if pos < out_strides.len() { out_strides[pos] } else { 1 };
            let _dim = *index_dims.get(&c).unwrap();
            out_idx.insert(c, rem / stride);
            rem %= stride;
        }

        // Sum over contraction indices
        let mut sum = 0.0f32;
        for sum_flat in 0..sum_size {
            // Compute sum indices
            let mut sum_idx: HashMap<char, usize> = HashMap::new();
            let mut rem = sum_flat;
            for (i, &c) in sum_indices.iter().enumerate() {
                let dim = sum_dims[i];
                sum_idx.insert(c, rem % dim);
                rem /= dim;
            }

            // Compute a index
            let a_flat: usize = input_a.iter().enumerate().map(|(i, &c)| {
                let idx = out_idx.get(&c).or_else(|| sum_idx.get(&c)).unwrap();
                idx * a_strides[i]
            }).sum();

            // Compute b index
            let b_flat: usize = input_b.iter().enumerate().map(|(i, &c)| {
                let idx = out_idx.get(&c).or_else(|| sum_idx.get(&c)).unwrap();
                idx * b_strides[i]
            }).sum();

            sum += a_data[a_flat] * b_data[b_flat];
        }

        result[out_flat] = sum;
    }

    Array::from_vec(result, if out_shape.is_empty() { Shape::scalar() } else { Shape::new(out_shape) })
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

    #[test]
    fn test_einsum_chain_matmul() {
        // Chain of 3 matrix multiplications: A @ B @ C
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));
        let c = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2])); // Identity

        let result = einsum("ij,jk,kl->il", &[&a, &b, &c]);

        // (A @ B) @ I = A @ B = [[19,22],[43,50]]
        assert_eq!(result.shape().as_slice(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_einsum_chain_three_matrices() {
        // Chain of 3 different matrices
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = Array::from_vec(vec![2.0, 0.0, 0.0, 2.0], Shape::new(vec![2, 2])); // 2*I
        let c = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![2, 2])); // ones

        let result = einsum("ij,jk,kl->il", &[&a, &b, &c]);

        // A @ (2*I) = [[2,4],[6,8]], then @ [[1,1],[1,1]] = [[6,6],[14,14]]
        assert_eq!(result.shape().as_slice(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![6.0, 6.0, 14.0, 14.0]);
    }

    #[test]
    fn test_einsum_chain_four_matrices() {
        // Chain of 4 matrices: A @ B @ C @ D
        let a = Array::from_vec(vec![1.0, 0.0, 0.0, 1.0], Shape::new(vec![2, 2])); // I
        let b = Array::from_vec(vec![2.0, 0.0, 0.0, 2.0], Shape::new(vec![2, 2])); // 2*I
        let c = Array::from_vec(vec![3.0, 0.0, 0.0, 3.0], Shape::new(vec![2, 2])); // 3*I
        let d = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0], Shape::new(vec![2, 2])); // ones

        let result = einsum("ij,jk,kl,lm->im", &[&a, &b, &c, &d]);

        // I @ 2*I @ 3*I @ ones = 6*I @ ones = [[6,6],[6,6]]
        assert_eq!(result.shape().as_slice(), &[2, 2]);
        assert_eq!(result.to_vec(), vec![6.0, 6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_einsum_sum_all() {
        // Sum all elements: ij->
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let result = einsum("ij->", &[&a]);
        assert_eq!(result.size(), 1);
        assert_eq!(result.to_vec()[0], 21.0); // 1+2+3+4+5+6
    }

    #[test]
    fn test_einsum_sum_axis() {
        // Sum along j: ij->i
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let result = einsum("ij->i", &[&a]);
        assert_eq!(result.shape().as_slice(), &[2]);
        assert_eq!(result.to_vec(), vec![6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_einsum_sum_axis_other() {
        // Sum along i: ij->j
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
        let result = einsum("ij->j", &[&a]);
        assert_eq!(result.shape().as_slice(), &[3]);
        assert_eq!(result.to_vec(), vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]
    }

    #[test]
    fn test_einsum_3d_permutation() {
        // Transpose 3D: ijk->kji
        // Input shape [2, 3, 4], output shape [4, 3, 2]
        // output[k][j][i] = input[i][j][k]
        let a = Array::from_vec(
            (0..24).map(|x| x as f32).collect(),
            Shape::new(vec![2, 3, 4]),
        );
        let result = einsum("ijk->kji", &[&a]);
        assert_eq!(result.shape().as_slice(), &[4, 3, 2]);

        // Output layout (row-major, shape [4,3,2]):
        // result[0] = output[0][0][0] = input[0][0][0] = 0
        // result[1] = output[0][0][1] = input[1][0][0] = 12
        // result[2] = output[0][1][0] = input[0][1][0] = 4
        // result[3] = output[0][1][1] = input[1][1][0] = 16
        // result[4] = output[0][2][0] = input[0][2][0] = 8
        // result[5] = output[0][2][1] = input[1][2][0] = 20
        // result[6] = output[1][0][0] = input[0][0][1] = 1
        assert_eq!(result.to_vec()[0], 0.0);
        assert_eq!(result.to_vec()[1], 12.0);
        assert_eq!(result.to_vec()[2], 4.0);
        assert_eq!(result.to_vec()[6], 1.0);
    }

    #[test]
    fn test_einsum_batched_matmul() {
        // Batched matrix multiply: bij,bjk->bik
        // 2 batches of 2x3 @ 3x2 matrices
        let a = Array::from_vec(
            vec![
                // Batch 0: [[1,2,3],[4,5,6]]
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                // Batch 1: [[7,8,9],[10,11,12]]
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            Shape::new(vec![2, 2, 3]),
        );
        let b = Array::from_vec(
            vec![
                // Batch 0: [[1,2],[3,4],[5,6]]
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                // Batch 1: [[1,0],[0,1],[1,0]]
                1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            ],
            Shape::new(vec![2, 3, 2]),
        );

        let result = einsum("bij,bjk->bik", &[&a, &b]);
        assert_eq!(result.shape().as_slice(), &[2, 2, 2]);

        // Batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]]
        //        = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        //        = [[22, 28], [49, 64]]
        // Batch 1: [[7,8,9],[10,11,12]] @ [[1,0],[0,1],[1,0]]
        //        = [[7+9, 8], [10+12, 11]]
        //        = [[16, 8], [22, 11]]
        assert_eq!(
            result.to_vec(),
            vec![22.0, 28.0, 49.0, 64.0, 16.0, 8.0, 22.0, 11.0]
        );
    }

    #[test]
    fn test_einsum_general_contraction() {
        // General contraction: ik,kj->ij (same as matmul but tests general path)
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
        let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));

        // Use a non-standard index pattern to force general path
        let result = einsum("ac,cb->ab", &[&a, &b]);
        assert_eq!(result.shape().as_slice(), &[2, 2]);
        // Same as matmul: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        assert_eq!(result.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }
}
