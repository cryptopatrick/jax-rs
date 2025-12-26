//! Unary operations on arrays.

use crate::trace::{is_tracing, trace_unary, Primitive};
use crate::{buffer::Buffer, Array, DType, Device};

#[cfg(test)]
use crate::Shape;

/// Stirling's approximation for lgamma for x >= 7.
fn lgamma_impl(x: f32) -> f32 {
    let x64 = x as f64;
    let c = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let tmp = x64 + 5.5;
    let tmp = tmp - (x64 + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (i, &cval) in c.iter().enumerate() {
        ser += cval / (x64 + (i + 1) as f64);
    }
    (-tmp + (2.5066282746310005 * ser / x64).ln()) as f32
}

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

    /// Safe reciprocal that returns 0 where x == 0.
    ///
    /// Returns 1/x where x != 0, and 0 where x == 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![2.0, 0.0, 4.0], Shape::new(vec![3]));
    /// let b = a.reciprocal_no_nan();
    /// assert_eq!(b.to_vec(), vec![0.5, 0.0, 0.25]);
    /// ```
    pub fn reciprocal_no_nan(&self) -> Array {
        unary_op(self, Primitive::Reciprocal, |x| {
            if x == 0.0 {
                0.0
            } else {
                1.0 / x
            }
        })
    }

    /// Convert degrees to radians.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let degrees = Array::from_vec(vec![0.0, 90.0, 180.0], Shape::new(vec![3]));
    /// let radians = degrees.deg2rad();
    /// assert!((radians.to_vec()[1] - std::f32::consts::PI / 2.0).abs() < 1e-5);
    /// ```
    pub fn deg2rad(&self) -> Array {
        unary_op(self, Primitive::Mul, |x| x * std::f32::consts::PI / 180.0)
    }

    /// Convert radians to degrees.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let radians = Array::from_vec(vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI], Shape::new(vec![3]));
    /// let degrees = radians.rad2deg();
    /// assert!((degrees.to_vec()[1] - 90.0).abs() < 1e-5);
    /// ```
    pub fn rad2deg(&self) -> Array {
        unary_op(self, Primitive::Mul, |x| x * 180.0 / std::f32::consts::PI)
    }

    /// Compute the sinc function: sin(x) / x.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let x = Array::from_vec(vec![0.0, 1.0, 2.0], Shape::new(vec![3]));
    /// let y = x.sinc();
    /// assert_eq!(y.to_vec()[0], 1.0); // sinc(0) = 1
    /// ```
    pub fn sinc(&self) -> Array {
        unary_op(self, Primitive::Sin, |x| {
            if x.abs() < 1e-10 {
                1.0
            } else {
                x.sin() / x
            }
        })
    }

    /// Compute the cube root.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![8.0, 27.0, 64.0], Shape::new(vec![3]));
    /// let b = a.cbrt();
    /// assert_eq!(b.to_vec(), vec![2.0, 3.0, 4.0]);
    /// ```
    pub fn cbrt(&self) -> Array {
        unary_op(self, Primitive::Pow, |x| x.cbrt())
    }

    /// Compute the inverse sine (arcsine) element-wise.
    ///
    /// Returns values in the range [-π/2, π/2].
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 0.5, 1.0], Shape::new(vec![3]));
    /// let b = a.arcsin();
    /// // Result: [0.0, ~0.524, ~1.571] (radians)
    /// ```
    pub fn arcsin(&self) -> Array {
        unary_op(self, Primitive::Sin, |x| x.asin())
    }

    /// Compute the inverse cosine (arccosine) element-wise.
    ///
    /// Returns values in the range [0, π].
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 0.5, 0.0], Shape::new(vec![3]));
    /// let b = a.arccos();
    /// // Result: [0.0, ~1.047, ~1.571] (radians)
    /// ```
    pub fn arccos(&self) -> Array {
        unary_op(self, Primitive::Cos, |x| x.acos())
    }

    /// Compute the inverse tangent (arctangent) element-wise.
    ///
    /// Returns values in the range [-π/2, π/2].
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0, -1.0], Shape::new(vec![3]));
    /// let b = a.arctan();
    /// // Result: [0.0, ~0.785, ~-0.785] (radians)
    /// ```
    pub fn arctan(&self) -> Array {
        unary_op(self, Primitive::Tan, |x| x.atan())
    }

    /// Compute the inverse hyperbolic sine element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0, 2.0], Shape::new(vec![3]));
    /// let b = a.arcsinh();
    /// // Result: [0.0, ~0.881, ~1.444]
    /// ```
    pub fn arcsinh(&self) -> Array {
        unary_op(self, Primitive::Sin, |x| x.asinh())
    }

    /// Compute the inverse hyperbolic cosine element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.arccosh();
    /// // Result: [0.0, ~1.317, ~1.763]
    /// ```
    pub fn arccosh(&self) -> Array {
        unary_op(self, Primitive::Cos, |x| x.acosh())
    }

    /// Compute the inverse hyperbolic tangent element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 0.5, -0.5], Shape::new(vec![3]));
    /// let b = a.arctanh();
    /// // Result: [0.0, ~0.549, ~-0.549]
    /// ```
    pub fn arctanh(&self) -> Array {
        unary_op(self, Primitive::Tan, |x| x.atanh())
    }

    /// Compute the base-10 logarithm element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 10.0, 100.0, 1000.0], Shape::new(vec![4]));
    /// let b = a.log10();
    /// assert_eq!(b.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    /// ```
    pub fn log10(&self) -> Array {
        unary_op(self, Primitive::Log, |x| x.log10())
    }

    /// Compute the base-2 logarithm element-wise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 4.0, 8.0], Shape::new(vec![4]));
    /// let b = a.log2();
    /// assert_eq!(b.to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    /// ```
    pub fn log2(&self) -> Array {
        unary_op(self, Primitive::Log, |x| x.log2())
    }

    /// Round to n decimal places.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.234, 5.678, 9.012], Shape::new(vec![3]));
    /// let b = a.around(1);
    /// // Result: [1.2, 5.7, 9.0]
    /// ```
    pub fn around(&self, decimals: i32) -> Array {
        let factor = 10_f32.powi(decimals);
        unary_op(self, Primitive::Mul, move |x| (x * factor).round() / factor)
    }

    /// Round toward zero (truncate decimal part).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.7, -2.3, 3.9], Shape::new(vec![3]));
    /// let b = a.fix();
    /// assert_eq!(b.to_vec(), vec![1.0, -2.0, 3.0]);
    /// ```
    pub fn fix(&self) -> Array {
        unary_op(self, Primitive::Abs, |x| x.trunc())
    }

    /// Check if sign bit is set (negative number).
    ///
    /// Returns 1.0 for negative numbers, 0.0 for positive.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, -2.0, 0.0, -0.0], Shape::new(vec![4]));
    /// let b = a.signbit();
    /// // Result: [0.0, 1.0, 0.0, 1.0]
    /// ```
    pub fn signbit(&self) -> Array {
        unary_op(self, Primitive::Sign, |x| if x.is_sign_negative() { 1.0 } else { 0.0 })
    }

    /// Unary positive (identity operation).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, -2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.positive();
    /// assert_eq!(b.to_vec(), vec![1.0, -2.0, 3.0]);
    /// ```
    pub fn positive(&self) -> Array {
        self.clone()
    }

    /// Unary negative (same as neg).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, -2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.negative();
    /// assert_eq!(b.to_vec(), vec![-1.0, 2.0, -3.0]);
    /// ```
    pub fn negative(&self) -> Array {
        self.neg()
    }

    /// Inverse (1/x) with safe handling of zeros.
    ///
    /// Returns infinity for zero values instead of panicking.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 4.0, 0.5], Shape::new(vec![4]));
    /// let b = a.invert();
    /// assert_eq!(b.to_vec(), vec![1.0, 0.5, 0.25, 2.0]);
    /// ```
    pub fn invert(&self) -> Array {
        self.reciprocal()
    }

    /// Convert angles from radians to degrees (alias).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, std::f32::consts::PI, std::f32::consts::PI * 2.0], Shape::new(vec![3]));
    /// let b = a.degrees();
    /// // Result: [0.0, 180.0, 360.0]
    /// ```
    pub fn degrees(&self) -> Array {
        self.rad2deg()
    }

    /// Convert angles from degrees to radians (alias).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 180.0, 360.0], Shape::new(vec![3]));
    /// let b = a.radians();
    /// // Result: [0.0, π, 2π]
    /// ```
    pub fn radians(&self) -> Array {
        self.deg2rad()
    }

    /// Return the spacing to the next representable float.
    ///
    /// For simplicity, returns a constant small value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.spacing();
    /// // Returns small epsilon values
    /// ```
    pub fn spacing(&self) -> Array {
        unary_op(self, Primitive::Abs, |x| {
            let next = f32::from_bits(x.to_bits() + 1);
            (next - x).abs()
        })
    }

    /// Return a copy of the array (alias for clone).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.copy();
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn copy(&self) -> Array {
        self.clone()
    }

    /// Return element-wise natural logarithm (alias for log).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E], Shape::new(vec![3]));
    /// let b = a.ln();
    /// // Result: [0.0, 1.0, 2.0]
    /// ```
    pub fn ln(&self) -> Array {
        self.log()
    }

    /// Return element-wise maximum with zero.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![-1.0, 0.0, 1.0, 2.0], Shape::new(vec![4]));
    /// let b = a.clip_min(0.0);
    /// assert_eq!(b.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
    /// ```
    pub fn clip_min(&self, min: f32) -> Array {
        unary_op(self, Primitive::Max, |x| x.max(min))
    }

    /// Return element-wise minimum with a maximum bound.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let b = a.clip_max(2.5);
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, 2.5, 2.5]);
    /// ```
    pub fn clip_max(&self, max: f32) -> Array {
        unary_op(self, Primitive::Min, |x| x.min(max))
    }

    /// Return the conjugate of the array (identity for real numbers).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.conj();
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn conj(&self) -> Array {
        self.clone()
    }

    /// Return the conjugate (alias for conj).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.conjugate();
    /// assert_eq!(b.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn conjugate(&self) -> Array {
        self.clone()
    }

    /// Return the angle of complex numbers (phase).
    /// For real numbers, returns 0 for positive, PI for negative.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, -1.0, 0.0], Shape::new(vec![3]));
    /// let angles = a.angle();
    /// // Positive: 0, Negative: PI, Zero: 0
    /// ```
    pub fn angle(&self) -> Array {
        unary_op(self, Primitive::Sign, |x| {
            if x > 0.0 {
                0.0
            } else if x < 0.0 {
                std::f32::consts::PI
            } else {
                0.0
            }
        })
    }

    /// Return the real part of complex numbers.
    /// For real arrays, this is the identity function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let r = a.real();
    /// assert_eq!(r.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn real(&self) -> Array {
        self.clone()
    }

    /// Return the imaginary part of complex numbers.
    /// For real arrays, returns zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let im = a.imag();
    /// assert_eq!(im.to_vec(), vec![0.0, 0.0, 0.0]);
    /// ```
    pub fn imag(&self) -> Array {
        Array::zeros(self.shape().clone(), DType::Float32)
    }

    /// Bitwise NOT operation.
    /// Inverts all bits in the bit representation of Float32 values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.bitwise_not();
    /// ```
    pub fn bitwise_not(&self) -> Array {
        unary_op(self, Primitive::Neg, |x| {
            let bits = x.to_bits();
            f32::from_bits(!bits)
        })
    }

    /// Return the reciprocal of the square root (1/sqrt(x)).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 4.0, 9.0], Shape::new(vec![3]));
    /// let b = a.rsqrt();
    /// assert!((b.to_vec()[0] - 1.0).abs() < 1e-6);
    /// assert!((b.to_vec()[1] - 0.5).abs() < 1e-6);
    /// ```
    pub fn rsqrt(&self) -> Array {
        unary_op(self, Primitive::Sqrt, |x| 1.0 / x.sqrt())
    }

    /// Return the fractional and integer parts of an array element-wise.
    /// Returns a tuple of (fractional_part, integer_part).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.5, 2.7, -3.2], Shape::new(vec![3]));
    /// let (frac, int) = a.modf();
    /// assert!((frac.to_vec()[0] - 0.5).abs() < 1e-6);
    /// assert!((int.to_vec()[0] - 1.0).abs() < 1e-6);
    /// ```
    pub fn modf(&self) -> (Array, Array) {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let frac_data: Vec<f32> = data.iter().map(|&x| x.fract()).collect();
        let int_data: Vec<f32> = data.iter().map(|&x| x.trunc()).collect();

        let frac = Array::from_vec(frac_data, self.shape().clone());
        let int = Array::from_vec(int_data, self.shape().clone());

        (frac, int)
    }

    /// Compute x * 2^exp for each element.
    /// Equivalent to ldexp function from C math library.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.ldexp(2);
    /// assert_eq!(b.to_vec(), vec![4.0, 8.0, 12.0]); // multiply by 2^2 = 4
    /// ```
    pub fn ldexp(&self, exp: i32) -> Array {
        let multiplier = 2_f32.powi(exp);
        unary_op(self, Primitive::Mul, move |x| x * multiplier)
    }

    /// Decompose x into mantissa and exponent: x = m * 2^e.
    /// Returns (mantissa, exponent) where mantissa is in [0.5, 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![4.0, 8.0, 0.5], Shape::new(vec![3]));
    /// let (mantissa, exp) = a.frexp();
    /// // 4.0 = 0.5 * 2^3, 8.0 = 0.5 * 2^4, 0.5 = 0.5 * 2^0
    /// ```
    pub fn frexp(&self) -> (Array, Array) {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let mut mantissa_data = Vec::with_capacity(data.len());
        let mut exp_data = Vec::with_capacity(data.len());

        for &x in &data {
            if x == 0.0 {
                mantissa_data.push(0.0);
                exp_data.push(0.0);
            } else {
                let bits = x.to_bits();
                let sign = (bits >> 31) & 1;
                let exponent = ((bits >> 23) & 0xFF) as i32 - 126;
                // Create mantissa in [0.5, 1.0)
                let mantissa_bits = (sign << 31) | (126 << 23) | (bits & 0x7FFFFF);
                let mantissa = f32::from_bits(mantissa_bits);
                mantissa_data.push(mantissa);
                exp_data.push(exponent as f32);
            }
        }

        let mantissa = Array::from_vec(mantissa_data, self.shape().clone());
        let exp = Array::from_vec(exp_data, self.shape().clone());

        (mantissa, exp)
    }

    /// Divide arrays element-wise with safe handling of division by zero.
    /// Returns 0 when dividing by zero instead of NaN/Inf.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = a.safe_divide_scalar(2.0);
    /// assert_eq!(b.to_vec(), vec![0.5, 1.0, 1.5]);
    /// let c = a.safe_divide_scalar(0.0);
    /// assert_eq!(c.to_vec(), vec![0.0, 0.0, 0.0]); // Returns 0 instead of Inf
    /// ```
    pub fn safe_divide_scalar(&self, divisor: f32) -> Array {
        if divisor == 0.0 {
            Array::zeros(self.shape().clone(), DType::Float32)
        } else {
            unary_op(self, Primitive::Reciprocal, move |x| x / divisor)
        }
    }

    /// Compute the modified Bessel function of the first kind, order 0.
    /// Approximation using polynomial expansion.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0, 2.0], Shape::new(vec![3]));
    /// let b = a.i0();
    /// assert!((b.to_vec()[0] - 1.0).abs() < 1e-4);  // i0(0) = 1
    /// ```
    pub fn i0(&self) -> Array {
        unary_op(self, Primitive::Exp, |x| {
            // Polynomial approximation for I0
            let ax = x.abs();
            if ax < 3.75 {
                let y = (x / 3.75).powi(2);
                1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
            } else {
                let y = 3.75 / ax;
                (ax.exp() / ax.sqrt()) * (0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))
            }
        })
    }

    /// Compute the natural logarithm of the absolute value of the gamma function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 5.0], Shape::new(vec![3]));
    /// let b = a.lgamma();
    /// assert!((b.to_vec()[0]).abs() < 1e-6);  // lgamma(1) = 0
    /// assert!((b.to_vec()[1]).abs() < 1e-6);  // lgamma(2) = 0
    /// ```
    pub fn lgamma(&self) -> Array {
        unary_op(self, Primitive::Log, |x| {
            // Stirling's approximation for larger values
            if x <= 0.0 {
                f32::INFINITY
            } else if x < 7.0 {
                // Use recurrence relation for small values
                let mut n = (7.0 - x).ceil() as i32;
                let mut y = x;
                let mut prod = 1.0;
                for _ in 0..n {
                    prod *= y;
                    y += 1.0;
                }
                lgamma_impl(y) - prod.ln()
            } else {
                lgamma_impl(x)
            }
        })
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
    fn test_reciprocal_no_nan() {
        let a = Array::from_vec(vec![2.0, 0.0, 4.0], Shape::new(vec![3]));
        let b = a.reciprocal_no_nan();
        assert_eq!(b.to_vec(), vec![0.5, 0.0, 0.25]);
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
