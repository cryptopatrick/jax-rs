//! Unary operations on arrays.

use crate::trace::{is_tracing, trace_unary, Primitive};
use crate::{buffer::Buffer, Array, DType, Device};

#[cfg(test)]
use crate::Shape;

/// Apply a unary function element-wise to an array.
fn unary_op<F>(input: &Array, op: Primitive, f: F) -> Array
where
    F: Fn(f32) -> f32,
{
    assert_eq!(input.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(input.device(), Device::Cpu, "Only CPU supported for now");

    let data = input.to_vec();
    let result_data: Vec<f32> = data.iter().map(|&x| f(x)).collect();
    let buffer = Buffer::from_f32(result_data, Device::Cpu);

    let result = Array::from_buffer(buffer, input.shape().clone());

    // Register with trace context if tracing is active
    if is_tracing() {
        trace_unary(result.id(), op, input);
    }

    result
}

impl Array {
    /// Negate the array element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, -2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.neg();
    /// assert_eq!(b.to_vec(), vec![-1.0, 2.0, -3.0]);
    /// ```
    pub fn neg(&self) -> Array {
        unary_op(self, Primitive::Neg, |x| -x)
    }

    /// Absolute value element-wise.
    pub fn abs(&self) -> Array {
        unary_op(self, Primitive::Abs, |x| x.abs())
    }

    /// Sine element-wise.
    pub fn sin(&self) -> Array {
        unary_op(self, Primitive::Sin, |x| x.sin())
    }

    /// Cosine element-wise.
    pub fn cos(&self) -> Array {
        unary_op(self, Primitive::Cos, |x| x.cos())
    }

    /// Tangent element-wise.
    pub fn tan(&self) -> Array {
        unary_op(self, Primitive::Tan, |x| x.tan())
    }

    /// Hyperbolic tangent element-wise.
    pub fn tanh(&self) -> Array {
        unary_op(self, Primitive::Tanh, |x| x.tanh())
    }

    /// Natural exponential (e^x) element-wise.
    pub fn exp(&self) -> Array {
        unary_op(self, Primitive::Exp, |x| x.exp())
    }

    /// Natural logarithm element-wise.
    pub fn log(&self) -> Array {
        unary_op(self, Primitive::Log, |x| x.ln())
    }

    /// Square root element-wise.
    pub fn sqrt(&self) -> Array {
        unary_op(self, Primitive::Sqrt, |x| x.sqrt())
    }

    /// Reciprocal (1/x) element-wise.
    pub fn reciprocal(&self) -> Array {
        unary_op(self, Primitive::Reciprocal, |x| 1.0 / x)
    }

    /// Square (x^2) element-wise.
    pub fn square(&self) -> Array {
        unary_op(self, Primitive::Square, |x| x * x)
    }

    /// Sign function element-wise (-1, 0, or 1).
    pub fn sign(&self) -> Array {
        unary_op(self, Primitive::Sign, |x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }

    /// Hyperbolic sine element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0], Shape::new(vec![1]));
    /// let b = a.sinh();
    /// assert_eq!(b.to_vec()[0], 0.0);
    /// ```
    pub fn sinh(&self) -> Array {
        unary_op(self, Primitive::Sin, |x| x.sinh())
    }

    /// Hyperbolic cosine element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0], Shape::new(vec![1]));
    /// let b = a.cosh();
    /// assert_eq!(b.to_vec()[0], 1.0);
    /// ```
    pub fn cosh(&self) -> Array {
        unary_op(self, Primitive::Cos, |x| x.cosh())
    }

    /// Arcsine element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0], Shape::new(vec![2]));
    /// let b = a.asin();
    /// assert!((b.to_vec()[0] - 0.0).abs() < 1e-6);
    /// assert!((b.to_vec()[1] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    /// ```
    pub fn asin(&self) -> Array {
        unary_op(self, Primitive::Sin, |x| x.asin())
    }

    /// Arccosine element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0], Shape::new(vec![1]));
    /// let b = a.acos();
    /// assert!((b.to_vec()[0] - 0.0).abs() < 1e-6);
    /// ```
    pub fn acos(&self) -> Array {
        unary_op(self, Primitive::Cos, |x| x.acos())
    }

    /// Arctangent element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0], Shape::new(vec![2]));
    /// let b = a.atan();
    /// assert!((b.to_vec()[0] - 0.0).abs() < 1e-6);
    /// assert!((b.to_vec()[1] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    /// ```
    pub fn atan(&self) -> Array {
        unary_op(self, Primitive::Tan, |x| x.atan())
    }

    /// Inverse hyperbolic sine element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0], Shape::new(vec![1]));
    /// let b = a.asinh();
    /// assert!((b.to_vec()[0] - 0.0).abs() < 1e-6);
    /// ```
    pub fn asinh(&self) -> Array {
        unary_op(self, Primitive::Sin, |x| x.asinh())
    }

    /// Inverse hyperbolic cosine element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0], Shape::new(vec![1]));
    /// let b = a.acosh();
    /// assert!((b.to_vec()[0] - 0.0).abs() < 1e-6);
    /// ```
    pub fn acosh(&self) -> Array {
        unary_op(self, Primitive::Cos, |x| x.acosh())
    }

    /// Inverse hyperbolic tangent element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0], Shape::new(vec![1]));
    /// let b = a.atanh();
    /// assert!((b.to_vec()[0] - 0.0).abs() < 1e-6);
    /// ```
    pub fn atanh(&self) -> Array {
        unary_op(self, Primitive::Tanh, |x| x.atanh())
    }

    /// Ceiling function element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.2, 2.7, -0.5], Shape::new(vec![3]));
    /// let b = a.ceil();
    /// assert_eq!(b.to_vec(), vec![2.0, 3.0, 0.0]);
    /// ```
    pub fn ceil(&self) -> Array {
        unary_op(self, Primitive::Sign, |x| x.ceil())
    }

    /// Floor function element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.2, 2.7, -0.5], Shape::new(vec![3]));
    /// let b = a.floor();
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, -1.0]);
    /// ```
    pub fn floor(&self) -> Array {
        unary_op(self, Primitive::Sign, |x| x.floor())
    }

    /// Round to nearest integer element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.2, 2.7, -0.5], Shape::new(vec![3]));
    /// let b = a.round();
    /// assert_eq!(b.to_vec(), vec![1.0, 3.0, -1.0]);
    /// ```
    pub fn round(&self) -> Array {
        unary_op(self, Primitive::Sign, |x| x.round())
    }

    /// Truncate to integer element-wise (round toward zero).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.7, 2.3, -1.7], Shape::new(vec![3]));
    /// let b = a.trunc();
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, -1.0]);
    /// ```
    pub fn trunc(&self) -> Array {
        unary_op(self, Primitive::Sign, |x| x.trunc())
    }

    /// Exponential minus 1 (e^x - 1) element-wise.
    ///
    /// More accurate than exp(x) - 1 for small values of x.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0], Shape::new(vec![1]));
    /// let b = a.expm1();
    /// assert!((b.to_vec()[0] - 0.0).abs() < 1e-6);
    /// ```
    pub fn expm1(&self) -> Array {
        unary_op(self, Primitive::Exp, |x| x.exp_m1())
    }

    /// Natural logarithm of 1 + x element-wise.
    ///
    /// More accurate than log(1 + x) for small values of x.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0], Shape::new(vec![1]));
    /// let b = a.log1p();
    /// assert!((b.to_vec()[0] - 0.0).abs() < 1e-6);
    /// ```
    pub fn log1p(&self) -> Array {
        unary_op(self, Primitive::Log, |x| x.ln_1p())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_neg() {
        let a = Array::from_vec(vec![1.0, -2.0, 3.0], Shape::new(vec![3]));
        let b = a.neg();
        assert_eq!(b.to_vec(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_abs() {
        let a =
            Array::from_vec(vec![1.0, -2.0, 3.0, -4.0], Shape::new(vec![4]));
        let b = a.abs();
        assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sin_cos() {
        let a = Array::from_vec(
            vec![0.0, std::f32::consts::PI / 2.0],
            Shape::new(vec![2]),
        );
        let sin_a = a.sin();
        let cos_a = a.cos();

        assert_abs_diff_eq!(sin_a.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sin_a.to_vec()[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cos_a.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cos_a.to_vec()[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_exp_log() {
        let a = Array::from_vec(vec![0.0, 1.0, 2.0], Shape::new(vec![3]));
        let exp_a = a.exp();
        let log_exp_a = exp_a.log();

        assert_abs_diff_eq!(exp_a.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(
            exp_a.to_vec()[1],
            std::f32::consts::E,
            epsilon = 1e-6
        );

        // log(exp(x)) should equal x
        assert_abs_diff_eq!(log_exp_a.to_vec()[0], 0.0, epsilon = 1e-5);
        assert_abs_diff_eq!(log_exp_a.to_vec()[1], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(log_exp_a.to_vec()[2], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_sqrt() {
        let a = Array::from_vec(vec![0.0, 1.0, 4.0, 9.0], Shape::new(vec![4]));
        let b = a.sqrt();
        assert_eq!(b.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tanh() {
        let a = Array::from_vec(vec![0.0, 1.0], Shape::new(vec![2]));
        let b = a.tanh();
        assert_abs_diff_eq!(b.to_vec()[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(b.to_vec()[1], 0.761_594_2, epsilon = 1e-6);
    }

    #[test]
    fn test_reciprocal() {
        let a = Array::from_vec(vec![1.0, 2.0, 4.0], Shape::new(vec![3]));
        let b = a.reciprocal();
        assert_abs_diff_eq!(b.to_vec()[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(b.to_vec()[1], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(b.to_vec()[2], 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_square() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = a.square();
        assert_eq!(b.to_vec(), vec![1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_sign() {
        let a =
            Array::from_vec(vec![-2.0, -0.0, 0.0, 3.0], Shape::new(vec![4]));
        let b = a.sign();
        assert_eq!(b.to_vec(), vec![-1.0, 0.0, 0.0, 1.0]);
    }
}
