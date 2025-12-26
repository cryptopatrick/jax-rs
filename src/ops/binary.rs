//! Binary operations on arrays.

use crate::trace::{is_tracing, trace_binary, Primitive};
use crate::{buffer::Buffer, Array, DType, Device, Shape};

/// Apply a binary function element-wise to two arrays with broadcasting.
fn binary_op<F>(lhs: &Array, rhs: &Array, op: Primitive, f: F) -> Array
where
    F: Fn(f32, f32) -> f32,
{
    assert_eq!(lhs.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(rhs.dtype(), DType::Float32, "Only Float32 supported");

    // Check if shapes are broadcast-compatible
    let result_shape = lhs
        .shape()
        .broadcast_with(rhs.shape())
        .expect("Shapes are not broadcast-compatible");

    // Dispatch based on device
    let result = match (lhs.device(), rhs.device()) {
        (Device::WebGpu, Device::WebGpu) => {
            // GPU path - no broadcasting support yet, shapes must match exactly
            assert_eq!(
                lhs.shape(),
                rhs.shape(),
                "GPU operations do not support broadcasting yet"
            );

            // Map primitive to WGSL operator
            let op_str = match &op {
                Primitive::Add => "+",
                Primitive::Sub => "-",
                Primitive::Mul => "*",
                Primitive::Div => "/",
                _ => {
                    // Fallback to CPU for unsupported ops
                    return binary_op_cpu(lhs, rhs, op.clone(), f);
                }
            };

            // Create output buffer on GPU
            let output_buffer = Buffer::zeros(
                result_shape.size(),
                DType::Float32,
                Device::WebGpu,
            );

            // Execute on GPU
            crate::backend::ops::gpu_binary_op(
                lhs.buffer(),
                rhs.buffer(),
                &output_buffer,
                op_str,
            );

            Array::from_buffer(output_buffer, result_shape)
        }
        (Device::Cpu, Device::Cpu) | (Device::Wasm, Device::Wasm) => {
            // CPU path with broadcasting support
            binary_op_cpu(lhs, rhs, op.clone(), f)
        }
        _ => {
            panic!("Mixed device operations not supported. Both arrays must be on the same device.");
        }
    };

    // Register with trace context if tracing is active
    if is_tracing() {
        trace_binary(result.id(), op, lhs, rhs);
    }

    result
}

/// CPU implementation of binary operation with broadcasting support.
fn binary_op_cpu<F>(lhs: &Array, rhs: &Array, _op: Primitive, f: F) -> Array
where
    F: Fn(f32, f32) -> f32,
{
    let result_shape = lhs
        .shape()
        .broadcast_with(rhs.shape())
        .expect("Shapes are not broadcast-compatible");

    let lhs_data = lhs.to_vec();
    let rhs_data = rhs.to_vec();

    let result_data = if lhs.shape() == rhs.shape() {
        // Same shape - simple element-wise operation
        lhs_data.iter().zip(rhs_data.iter()).map(|(&a, &b)| f(a, b)).collect()
    } else {
        // Need broadcasting
        broadcast_binary(
            &lhs_data,
            lhs.shape(),
            &rhs_data,
            rhs.shape(),
            &result_shape,
            f,
        )
    };

    let buffer = Buffer::from_f32(result_data, Device::Cpu);
    Array::from_buffer(buffer, result_shape)
}

/// Helper function to perform binary operation with broadcasting.
fn broadcast_binary<F>(
    lhs_data: &[f32],
    lhs_shape: &Shape,
    rhs_data: &[f32],
    rhs_shape: &Shape,
    result_shape: &Shape,
    f: F,
) -> Vec<f32>
where
    F: Fn(f32, f32) -> f32,
{
    let size = result_shape.size();
    let mut result = Vec::with_capacity(size);

    for i in 0..size {
        let lhs_idx = broadcast_index(i, result_shape, lhs_shape);
        let rhs_idx = broadcast_index(i, result_shape, rhs_shape);
        result.push(f(lhs_data[lhs_idx], rhs_data[rhs_idx]));
    }

    result
}

/// Convert a flat index in the result array to an index in the source array,
/// accounting for broadcasting.
pub(crate) fn broadcast_index(
    flat_idx: usize,
    result_shape: &Shape,
    src_shape: &Shape,
) -> usize {
    let result_dims = result_shape.as_slice();
    let src_dims = src_shape.as_slice();

    // Convert flat index to multi-dimensional index
    let mut multi_idx = Vec::with_capacity(result_dims.len());
    let mut idx = flat_idx;
    for &dim in result_dims.iter().rev() {
        multi_idx.push(idx % dim);
        idx /= dim;
    }
    multi_idx.reverse();

    // Map to source index with broadcasting
    let offset = result_dims.len() - src_dims.len();
    let mut src_idx = 0;
    let mut stride = 1;

    for i in (0..src_dims.len()).rev() {
        let result_i = offset + i;
        let dim_idx = if src_dims[i] == 1 {
            0 // Broadcast dimension
        } else {
            multi_idx[result_i]
        };
        src_idx += dim_idx * stride;
        stride *= src_dims[i];
    }

    src_idx
}

impl Array {
    /// Add two arrays element-wise with broadcasting.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
    /// let c = a.add(&b);
    /// assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0]);
    /// ```
    pub fn add(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Add, |a, b| a + b)
    }

    /// Subtract two arrays element-wise with broadcasting.
    pub fn sub(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Sub, |a, b| a - b)
    }

    /// Multiply two arrays element-wise with broadcasting.
    pub fn mul(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Mul, |a, b| a * b)
    }

    /// Divide two arrays element-wise with broadcasting.
    pub fn div(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| a / b)
    }

    /// Raise elements to a power element-wise with broadcasting.
    pub fn pow(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Pow, |a, b| a.powf(b))
    }

    /// Element-wise minimum.
    pub fn minimum(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Min, |a, b| a.min(b))
    }

    /// Element-wise maximum.
    pub fn maximum(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Max, |a, b| a.max(b))
    }

    /// Safe division that returns 0 where division by zero would occur.
    ///
    /// Returns x / y where y != 0, and 0 where y == 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 0.0, 3.0], Shape::new(vec![3]));
    /// let c = a.divide_no_nan(&b);
    /// assert_eq!(c.to_vec(), vec![0.5, 0.0, 1.0]);
    /// ```
    pub fn divide_no_nan(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| {
            if b == 0.0 {
                0.0
            } else {
                a / b
            }
        })
    }

    /// Squared difference: (a - b)^2.
    ///
    /// Useful for computing mean squared error and similar metrics.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 2.0, 1.0], Shape::new(vec![3]));
    /// let c = a.squared_difference(&b);
    /// assert_eq!(c.to_vec(), vec![1.0, 0.0, 4.0]);
    /// ```
    pub fn squared_difference(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Sub, |a, b| {
            let diff = a - b;
            diff * diff
        })
    }

    /// Element-wise modulo operation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![5.0, 7.0, 9.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![3.0, 3.0, 3.0], Shape::new(vec![3]));
    /// let c = a.mod_op(&b);
    /// assert_eq!(c.to_vec(), vec![2.0, 1.0, 0.0]);
    /// ```
    pub fn mod_op(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| a % b)
    }

    /// Element-wise arctangent of a/b.
    ///
    /// Correctly handles signs to determine quadrant.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let y = Array::from_vec(vec![1.0, -1.0], Shape::new(vec![2]));
    /// let x = Array::from_vec(vec![1.0, 1.0], Shape::new(vec![2]));
    /// let angle = y.atan2(&x);
    /// # // We just check it compiles and runs
    /// ```
    pub fn atan2(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| a.atan2(b))
    }

    /// Element-wise hypot: sqrt(a^2 + b^2).
    ///
    /// Computes the hypotenuse in a numerically stable way.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
    /// let b = Array::from_vec(vec![4.0, 3.0], Shape::new(vec![2]));
    /// let c = a.hypot(&b);
    /// assert_eq!(c.to_vec(), vec![5.0, 5.0]);
    /// ```
    pub fn hypot(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Add, |a, b| a.hypot(b))
    }

    /// Element-wise copysign: magnitude of a with sign of b.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![-1.0, 1.0, -1.0], Shape::new(vec![3]));
    /// let c = a.copysign(&b);
    /// assert_eq!(c.to_vec(), vec![-1.0, 2.0, -3.0]);
    /// ```
    pub fn copysign(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Mul, |a, b| a.copysign(b))
    }

    /// Element-wise next representable float in direction of b.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = Array::from_vec(vec![2.0, 1.0], Shape::new(vec![2]));
    /// let c = a.next_after(&b);
    /// # // Just verify it compiles
    /// ```
    pub fn next_after(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Add, |a, b| {
            if a < b {
                // Next float towards positive infinity
                f32::from_bits(a.to_bits() + 1)
            } else if a > b {
                // Next float towards negative infinity
                f32::from_bits(a.to_bits() - 1)
            } else {
                b
            }
        })
    }

    /// Logarithm of sum of exponentials (numerically stable).
    ///
    /// Computes log(exp(x) + exp(y)) in a numerically stable way.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
    /// let c = a.logaddexp(&b);
    /// // Result: log(exp(1)+exp(2)), log(exp(2)+exp(3)), log(exp(3)+exp(4))
    /// ```
    pub fn logaddexp(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Add, |a, b| {
            let max = a.max(b);
            max + ((a - max).exp() + (b - max).exp()).ln()
        })
    }

    /// Base-2 logarithm of sum of exponentials.
    ///
    /// Computes log2(2^x + 2^y) in a numerically stable way.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
    /// let c = a.logaddexp2(&b);
    /// ```
    pub fn logaddexp2(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Add, |a, b| {
            let max = a.max(b);
            max + ((a - max).exp2() + (b - max).exp2()).log2()
        })
    }

    /// Heaviside step function.
    ///
    /// Returns 0 where x < 0, h0 where x == 0, and 1 where x > 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let x = Array::from_vec(vec![-1.0, 0.0, 1.0], Shape::new(vec![3]));
    /// let h0 = Array::from_vec(vec![0.5, 0.5, 0.5], Shape::new(vec![3]));
    /// let h = x.heaviside(&h0);
    /// assert_eq!(h.to_vec(), vec![0.0, 0.5, 1.0]);
    /// ```
    pub fn heaviside(&self, h0: &Array) -> Array {
        binary_op(self, h0, Primitive::Max, |x, h0_val| {
            if x < 0.0 {
                0.0
            } else if x == 0.0 {
                h0_val
            } else {
                1.0
            }
        })
    }

    /// Floor division (division rounding toward negative infinity).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![7.0, 7.0, -7.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![3.0, -3.0, 3.0], Shape::new(vec![3]));
    /// let c = a.floor_divide(&b);
    /// assert_eq!(c.to_vec(), vec![2.0, -3.0, -3.0]);
    /// ```
    pub fn floor_divide(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| (a / b).floor())
    }

    /// Fused multiply-add: a * b + c.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
    /// let c = Array::from_vec(vec![1.0, 1.0, 1.0], Shape::new(vec![3]));
    /// let result = a.fma(&b, &c);
    /// assert_eq!(result.to_vec(), vec![3.0, 7.0, 13.0]); // [1*2+1, 2*3+1, 3*4+1]
    /// ```
    pub fn fma(&self, b: &Array, c: &Array) -> Array {
        let product = self.mul(b);
        product.add(c)
    }

    /// Greatest common divisor element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![12.0, 15.0, 24.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![8.0, 10.0, 18.0], Shape::new(vec![3]));
    /// let c = a.gcd(&b);
    /// assert_eq!(c.to_vec(), vec![4.0, 5.0, 6.0]);
    /// ```
    pub fn gcd(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Min, |mut a, mut b| {
            a = a.abs();
            b = b.abs();
            while b > 0.5 {
                let temp = b;
                b = a % b;
                a = temp;
            }
            a
        })
    }

    /// Least common multiple element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![12.0, 15.0, 24.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![8.0, 10.0, 18.0], Shape::new(vec![3]));
    /// let c = a.lcm(&b);
    /// assert_eq!(c.to_vec(), vec![24.0, 30.0, 72.0]);
    /// ```
    pub fn lcm(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Mul, |mut a, mut b| {
            a = a.abs();
            b = b.abs();
            if a < 0.5 || b < 0.5 {
                return 0.0;
            }
            let mut gcd_val = a;
            let mut temp = b;
            while temp > 0.5 {
                let r = gcd_val % temp;
                gcd_val = temp;
                temp = r;
            }
            (a * b) / gcd_val
        })
    }

    /// Bitwise AND operation.
    /// Operates on the bit representation of Float32 values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![15.0, 31.0, 63.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![7.0, 15.0, 31.0], Shape::new(vec![3]));
    /// let c = a.bitwise_and(&b);
    /// ```
    pub fn bitwise_and(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Min, |a, b| {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            f32::from_bits(a_bits & b_bits)
        })
    }

    /// Bitwise OR operation.
    /// Operates on the bit representation of Float32 values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![8.0, 16.0, 32.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![4.0, 8.0, 16.0], Shape::new(vec![3]));
    /// let c = a.bitwise_or(&b);
    /// ```
    pub fn bitwise_or(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Max, |a, b| {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            f32::from_bits(a_bits | b_bits)
        })
    }

    /// Bitwise XOR operation.
    /// Operates on the bit representation of Float32 values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![12.0, 15.0, 18.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![10.0, 5.0, 20.0], Shape::new(vec![3]));
    /// let c = a.bitwise_xor(&b);
    /// ```
    pub fn bitwise_xor(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Add, |a, b| {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            f32::from_bits(a_bits ^ b_bits)
        })
    }

    /// Left bit shift operation.
    /// Shifts the bit representation of Float32 values left.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let c = a.left_shift(&b);
    /// ```
    pub fn left_shift(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Mul, |a, b| {
            let a_bits = a.to_bits();
            let shift = b as u32;
            f32::from_bits(a_bits << shift)
        })
    }

    /// Right bit shift operation.
    /// Shifts the bit representation of Float32 values right.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![4.0, 8.0, 16.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let c = a.right_shift(&b);
    /// ```
    pub fn right_shift(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| {
            let a_bits = a.to_bits();
            let shift = b as u32;
            f32::from_bits(a_bits >> shift)
        })
    }

    /// Element-wise maximum, ignoring NaNs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
    /// let c = a.fmax(&b);
    /// assert_eq!(c.to_vec()[0], 2.0);
    /// assert_eq!(c.to_vec()[1], 2.0);
    /// assert_eq!(c.to_vec()[2], 3.0);
    /// ```
    pub fn fmax(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Max, |a, b| {
            if a.is_nan() { b }
            else if b.is_nan() { a }
            else { a.max(b) }
        })
    }

    /// Element-wise minimum, ignoring NaNs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
    /// let c = a.fmin(&b);
    /// assert_eq!(c.to_vec()[0], 1.0);
    /// assert_eq!(c.to_vec()[1], 2.0);
    /// assert_eq!(c.to_vec()[2], 2.0);
    /// ```
    pub fn fmin(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Min, |a, b| {
            if a.is_nan() { b }
            else if b.is_nan() { a }
            else { a.min(b) }
        })
    }

    /// Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
    ///
    /// The quadrant (i.e., branch) is chosen so that arctan2(x1, x2) is
    /// the signed angle in radians between the ray ending at the origin
    /// and passing through the point (1,0), and the ray ending at the
    /// origin and passing through the point (x2, x1).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let y = Array::from_vec(vec![1.0, -1.0, 1.0, -1.0], Shape::new(vec![4]));
    /// let x = Array::from_vec(vec![1.0, 1.0, -1.0, -1.0], Shape::new(vec![4]));
    /// let angles = y.arctan2(&x);
    /// // First quadrant: pi/4, Second: -pi/4, Third: 3pi/4, Fourth: -3pi/4
    /// ```
    pub fn arctan2(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |y, x| y.atan2(x))
    }

    /// Element-wise remainder of division (fmod).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![5.0, 7.0, 10.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
    /// let c = a.fmod(&b);
    /// assert_eq!(c.to_vec(), vec![1.0, 1.0, 2.0]);
    /// ```
    pub fn fmod(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| a % b)
    }

    /// Return the next floating-point value after x1 towards x2.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0], Shape::new(vec![2]));
    /// let b = Array::from_vec(vec![2.0, 1.0], Shape::new(vec![2]));
    /// let c = a.nextafter(&b);
    /// // First element goes up slightly, second goes down
    /// ```
    pub fn nextafter(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Add, |x1, x2| {
            if x1 == x2 {
                x2
            } else if x2 > x1 {
                // Next float toward positive infinity
                let bits = x1.to_bits();
                if x1 >= 0.0 {
                    f32::from_bits(bits + 1)
                } else {
                    f32::from_bits(bits - 1)
                }
            } else {
                // Next float toward negative infinity
                let bits = x1.to_bits();
                if x1 > 0.0 {
                    f32::from_bits(bits - 1)
                } else if x1 == 0.0 {
                    f32::from_bits(1 | (1 << 31)) // Negative zero direction
                } else {
                    f32::from_bits(bits + 1)
                }
            }
        })
    }

    /// Compute the safe element-wise division, returning 0 where denominator is 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 0.0, 3.0], Shape::new(vec![3]));
    /// let c = a.safe_divide(&b);
    /// assert_eq!(c.to_vec(), vec![1.0, 0.0, 1.0]);
    /// ```
    pub fn safe_divide(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| {
            if b == 0.0 { 0.0 } else { a / b }
        })
    }

    /// Compute element-wise true division.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![5.0, 7.0, 9.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
    /// let c = a.true_divide(&b);
    /// assert_eq!(c.to_vec(), vec![2.5, 3.5, 4.5]);
    /// ```
    pub fn true_divide(&self, other: &Array) -> Array {
        self.div(other)
    }

    /// Compute element-wise remainder, with the same sign as divisor.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![7.0, -7.0, 7.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![3.0, 3.0, -3.0], Shape::new(vec![3]));
    /// let c = a.remainder(&b);
    /// // Python-style modulo: result has same sign as divisor
    /// ```
    pub fn remainder(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Div, |a, b| {
            let r = a % b;
            if (r > 0.0 && b < 0.0) || (r < 0.0 && b > 0.0) {
                r + b
            } else {
                r
            }
        })
    }

    /// Compute element-wise difference raised to a power.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 5.0, 7.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let c = a.diff_pow(&b, 2.0);  // (a - b)^2
    /// assert_eq!(c.to_vec(), vec![4.0, 9.0, 16.0]);
    /// ```
    pub fn diff_pow(&self, other: &Array, power: f32) -> Array {
        binary_op(self, other, Primitive::Sub, move |a, b| (a - b).powf(power))
    }

    /// Compute element-wise squared difference.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 5.0, 7.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let c = a.squared_diff(&b);  // (a - b)^2
    /// assert_eq!(c.to_vec(), vec![4.0, 9.0, 16.0]);
    /// ```
    pub fn squared_diff(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Sub, |a, b| {
            let d = a - b;
            d * d
        })
    }

    /// Compute element-wise average of two arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![2.0, 4.0, 6.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![4.0, 6.0, 8.0], Shape::new(vec![3]));
    /// let c = a.average_with(&b);
    /// assert_eq!(c.to_vec(), vec![3.0, 5.0, 7.0]);
    /// ```
    pub fn average_with(&self, other: &Array) -> Array {
        binary_op(self, other, Primitive::Add, |a, b| (a + b) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let c = a.add(&b);
        assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_sub() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let c = a.sub(&b);
        assert_eq!(c.to_vec(), vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul() {
        let a = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![5.0, 6.0, 7.0], Shape::new(vec![3]));
        let c = a.mul(&b);
        assert_eq!(c.to_vec(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div() {
        let a = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 4.0, 5.0], Shape::new(vec![3]));
        let c = a.div(&b);
        assert_eq!(c.to_vec(), vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_pow() {
        let a = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 2.0, 2.0], Shape::new(vec![3]));
        let c = a.pow(&b);
        assert_eq!(c.to_vec(), vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_broadcast_scalar() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![10.0], Shape::new(vec![1]));
        let c = a.add(&b);
        assert_eq!(c.to_vec(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_broadcast_2d() {
        // [2, 3] + [1, 3] -> [2, 3]
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );
        let b =
            Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![1, 3]));
        let c = a.add(&b);
        assert_eq!(c.shape().as_slice(), &[2, 3]);
        assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_row_col() {
        // [3, 1] + [1, 3] -> [3, 3]
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3, 1]));
        let b =
            Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![1, 3]));
        let c = a.add(&b);
        assert_eq!(c.shape().as_slice(), &[3, 3]);
        assert_eq!(
            c.to_vec(),
            vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0, 33.0]
        );
    }

    #[test]
    fn test_minimum_maximum() {
        let a = Array::from_vec(vec![1.0, 5.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 4.0, 6.0], Shape::new(vec![3]));

        let min_ab = a.minimum(&b);
        assert_eq!(min_ab.to_vec(), vec![1.0, 4.0, 3.0]);

        let max_ab = a.maximum(&b);
        assert_eq!(max_ab.to_vec(), vec![2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_divide_no_nan() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 0.0, 3.0], Shape::new(vec![3]));
        let c = a.divide_no_nan(&b);
        assert_eq!(c.to_vec(), vec![0.5, 0.0, 1.0]);
    }

    #[test]
    fn test_squared_difference() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 2.0, 1.0], Shape::new(vec![3]));
        let c = a.squared_difference(&b);
        assert_eq!(c.to_vec(), vec![1.0, 0.0, 4.0]);
    }

    #[test]
    fn test_mod_op() {
        let a = Array::from_vec(vec![5.0, 7.0, 9.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![3.0, 3.0, 3.0], Shape::new(vec![3]));
        let c = a.mod_op(&b);
        assert_eq!(c.to_vec(), vec![2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_atan2() {
        let y = Array::from_vec(vec![1.0, 1.0, -1.0, -1.0], Shape::new(vec![4]));
        let x = Array::from_vec(vec![1.0, -1.0, 1.0, -1.0], Shape::new(vec![4]));
        let angle = y.atan2(&x);
        let result = angle.to_vec();
        // Just verify it produces reasonable results
        assert!(result[0] > 0.0 && result[0] < 1.6); // ~π/4
        assert!(result[1] > 2.0 && result[1] < 3.2); // ~3π/4
    }

    #[test]
    fn test_hypot() {
        let a = Array::from_vec(vec![3.0, 4.0], Shape::new(vec![2]));
        let b = Array::from_vec(vec![4.0, 3.0], Shape::new(vec![2]));
        let c = a.hypot(&b);
        assert_eq!(c.to_vec(), vec![5.0, 5.0]);
    }

    #[test]
    fn test_copysign() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![-1.0, 1.0, -1.0], Shape::new(vec![3]));
        let c = a.copysign(&b);
        assert_eq!(c.to_vec(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_next_after() {
        let a = Array::from_vec(vec![1.0], Shape::new(vec![1]));
        let b = Array::from_vec(vec![2.0], Shape::new(vec![1]));
        let c = a.next_after(&b);
        // Should be slightly larger than 1.0
        assert!(c.to_vec()[0] > 1.0);
        assert!(c.to_vec()[0] < 1.0 + 1e-6);
    }

    #[test]
    fn test_broadcast_index() {
        // Test broadcast_index function
        let result_shape = Shape::new(vec![2, 3]);
        let src_shape = Shape::new(vec![1, 3]);

        // For result shape [2,3], indices 0-5 map to positions:
        // 0: [0,0] -> [0,0] in [1,3] -> flat 0
        // 1: [0,1] -> [0,1] in [1,3] -> flat 1
        // 2: [0,2] -> [0,2] in [1,3] -> flat 2
        // 3: [1,0] -> [0,0] in [1,3] -> flat 0 (broadcast)
        // 4: [1,1] -> [0,1] in [1,3] -> flat 1 (broadcast)
        // 5: [1,2] -> [0,2] in [1,3] -> flat 2 (broadcast)
        assert_eq!(broadcast_index(0, &result_shape, &src_shape), 0);
        assert_eq!(broadcast_index(1, &result_shape, &src_shape), 1);
        assert_eq!(broadcast_index(2, &result_shape, &src_shape), 2);
        assert_eq!(broadcast_index(3, &result_shape, &src_shape), 0);
        assert_eq!(broadcast_index(4, &result_shape, &src_shape), 1);
        assert_eq!(broadcast_index(5, &result_shape, &src_shape), 2);
    }
}
