//! Fast Fourier Transform operations.
//!
//! Basic CPU implementation of FFT operations using the Cooley-Tukey algorithm.

use crate::{Array, DType, Shape};
use std::f32::consts::PI;

/// Complex number represented as (real, imaginary) pair
type Complex = (f32, f32);

/// Add two complex numbers
#[inline]
fn complex_add(a: Complex, b: Complex) -> Complex {
    (a.0 + b.0, a.1 + b.1)
}

/// Subtract two complex numbers
#[inline]
fn complex_sub(a: Complex, b: Complex) -> Complex {
    (a.0 - b.0, a.1 - b.1)
}

/// Multiply two complex numbers
#[inline]
fn complex_mul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Compute e^(-2πi * k / n) for FFT
#[inline]
fn twiddle_factor(k: usize, n: usize, inverse: bool) -> Complex {
    let angle = 2.0 * PI * (k as f32) / (n as f32);
    let sign = if inverse { 1.0 } else { -1.0 };
    (angle.cos(), sign * angle.sin())
}

/// Cooley-Tukey FFT algorithm (recursive, radix-2)
fn fft_recursive(input: &[Complex], inverse: bool) -> Vec<Complex> {
    let n = input.len();

    // Base case
    if n <= 1 {
        return input.to_vec();
    }

    // Check if n is power of 2
    if n & (n - 1) != 0 {
        panic!("FFT size must be a power of 2, got {}", n);
    }

    // Split into even and odd indices
    let even: Vec<Complex> = input.iter().step_by(2).copied().collect();
    let odd: Vec<Complex> = input.iter().skip(1).step_by(2).copied().collect();

    // Recursive FFT
    let fft_even = fft_recursive(&even, inverse);
    let fft_odd = fft_recursive(&odd, inverse);

    // Combine results
    let mut result = vec![(0.0, 0.0); n];
    for k in 0..n / 2 {
        let t = complex_mul(twiddle_factor(k, n, inverse), fft_odd[k]);
        result[k] = complex_add(fft_even[k], t);
        result[k + n / 2] = complex_sub(fft_even[k], t);
    }

    result
}

/// 1D Fast Fourier Transform.
///
/// Computes the discrete Fourier transform of a 1D array using the Cooley-Tukey algorithm.
/// The input size must be a power of 2.
///
/// Returns an array of complex values represented as interleaved [real, imag, real, imag, ...].
///
/// # Arguments
///
/// * `x` - Input array (real values)
///
/// # Examples
///
/// ```
/// # use jax_rs::{ops, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let result = ops::fft::fft(&x);
/// assert_eq!(result.size(), 8); // 4 complex numbers = 8 floats
/// ```
pub fn fft(x: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.ndim(), 1, "Input must be 1D");

    let data = x.to_vec();
    let n = data.len();

    // Convert real input to complex
    let complex_input: Vec<Complex> = data.iter().map(|&r| (r, 0.0)).collect();

    // Compute FFT
    let fft_result = fft_recursive(&complex_input, false);

    // Flatten to interleaved real/imag format
    let mut output = Vec::with_capacity(n * 2);
    for (real, imag) in fft_result {
        output.push(real);
        output.push(imag);
    }

    Array::from_vec(output, Shape::new(vec![n * 2]))
}

/// Inverse 1D Fast Fourier Transform.
///
/// Computes the inverse discrete Fourier transform.
/// Input should be in interleaved [real, imag, real, imag, ...] format.
///
/// # Arguments
///
/// * `x` - Input array (complex values as interleaved real/imag)
///
/// # Examples
///
/// ```
/// # use jax_rs::{ops, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let fft_x = ops::fft::fft(&x);
/// let reconstructed = ops::fft::ifft(&fft_x);
/// // Should approximately equal original (first half are real parts)
/// ```
pub fn ifft(x: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.ndim(), 1, "Input must be 1D");
    assert_eq!(x.size() % 2, 0, "Input size must be even (interleaved complex)");

    let data = x.to_vec();
    let n = data.len() / 2;

    // Convert interleaved format to complex
    let complex_input: Vec<Complex> = (0..n)
        .map(|i| (data[i * 2], data[i * 2 + 1]))
        .collect();

    // Compute inverse FFT
    let mut ifft_result = fft_recursive(&complex_input, true);

    // Normalize by 1/n
    let scale = 1.0 / (n as f32);
    for (r, i) in ifft_result.iter_mut() {
        *r *= scale;
        *i *= scale;
    }

    // Flatten to interleaved real/imag format
    let mut output = Vec::with_capacity(n * 2);
    for (real, imag) in ifft_result {
        output.push(real);
        output.push(imag);
    }

    Array::from_vec(output, Shape::new(vec![n * 2]))
}

/// Real 1D Fast Fourier Transform.
///
/// Optimized FFT for real-valued input. Only computes the positive frequencies
/// since the negative frequencies are complex conjugates.
///
/// Returns n/2 + 1 complex values (interleaved real/imag).
///
/// # Arguments
///
/// * `x` - Input array (real values)
///
/// # Examples
///
/// ```
/// # use jax_rs::{ops, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let result = ops::fft::rfft(&x);
/// assert_eq!(result.size(), 6); // (4/2 + 1) * 2 = 6 floats
/// ```
pub fn rfft(x: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.ndim(), 1, "Input must be 1D");

    let data = x.to_vec();
    let n = data.len();

    // Convert real input to complex
    let complex_input: Vec<Complex> = data.iter().map(|&r| (r, 0.0)).collect();

    // Compute FFT
    let fft_result = fft_recursive(&complex_input, false);

    // Only keep first n/2 + 1 components (Nyquist frequency)
    let m = n / 2 + 1;
    let mut output = Vec::with_capacity(m * 2);
    for i in 0..m {
        output.push(fft_result[i].0);
        output.push(fft_result[i].1);
    }

    Array::from_vec(output, Shape::new(vec![m * 2]))
}

/// 2D Fast Fourier Transform.
///
/// Computes the 2D FFT by applying 1D FFT to rows, then to columns.
/// Input must have dimensions that are powers of 2.
///
/// Returns complex values in shape [height, width * 2] (interleaved real/imag).
///
/// # Arguments
///
/// * `x` - Input 2D array (real values)
///
/// # Examples
///
/// ```
/// # use jax_rs::{ops, Array, Shape};
/// let x = Array::from_vec(
///     vec![1.0, 2.0, 3.0, 4.0,
///          5.0, 6.0, 7.0, 8.0,
///          9.0, 10.0, 11.0, 12.0,
///          13.0, 14.0, 15.0, 16.0],
///     Shape::new(vec![4, 4])
/// );
/// let result = ops::fft::fft2(&x);
/// assert_eq!(result.shape().as_slice(), &[4, 8]); // 4x4 complex = 4x8 floats
/// ```
pub fn fft2(x: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.ndim(), 2, "Input must be 2D");

    let shape = x.shape().as_slice();
    let (height, width) = (shape[0], shape[1]);
    let data = x.to_vec();

    // FFT along rows
    let mut row_fft = Vec::with_capacity(height * width * 2);
    for i in 0..height {
        let row_start = i * width;
        let row_end = row_start + width;
        let row_data = &data[row_start..row_end];

        let row_array = Array::from_vec(row_data.to_vec(), Shape::new(vec![width]));
        let row_fft_result = fft(&row_array);
        row_fft.extend_from_slice(&row_fft_result.to_vec());
    }

    // FFT along columns (on complex data)
    let mut result = Vec::with_capacity(height * width * 2);
    for j in 0..(width * 2) {
        let mut col = Vec::with_capacity(height);
        for i in 0..height {
            col.push(row_fft[i * width * 2 + j]);
        }

        // Apply FFT if this is a complete complex column (both real and imag parts)
        if j % 2 == 0 {
            // Collect complex pairs for this column
            let mut col_complex = Vec::with_capacity(height);
            for i in 0..height {
                let real = row_fft[i * width * 2 + j];
                let imag = row_fft[i * width * 2 + j + 1];
                col_complex.push((real, imag));
            }

            let col_fft = fft_recursive(&col_complex, false);

            // Interleave back
            for i in 0..height {
                if j == 0 {
                    result.resize((i + 1) * width * 2, 0.0);
                }
                result[i * width * 2 + j] = col_fft[i].0;
                result[i * width * 2 + j + 1] = col_fft[i].1;
            }
        }
    }

    Array::from_vec(result, Shape::new(vec![height, width * 2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_basic() {
        // Simple 4-point FFT
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let result = fft(&x);

        assert_eq!(result.size(), 8); // 4 complex = 8 floats

        // DC component should be sum of inputs
        let data = result.to_vec();
        assert!((data[0] - 10.0).abs() < 1e-5); // real part
        assert!(data[1].abs() < 1e-5); // imag part (should be ~0)
    }

    #[test]
    fn test_ifft_reconstruction() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let fft_x = fft(&x);
        let reconstructed = ifft(&fft_x);

        // Extract real parts (even indices)
        let data = reconstructed.to_vec();
        let real_parts: Vec<f32> = (0..4).map(|i| data[i * 2]).collect();

        // Should match original
        let original = x.to_vec();
        for i in 0..4 {
            assert!((real_parts[i] - original[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rfft() {
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let result = rfft(&x);

        // Should have n/2 + 1 = 3 complex values = 6 floats
        assert_eq!(result.size(), 6);

        let data = result.to_vec();
        // DC component
        assert!((data[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_fft2() {
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            Shape::new(vec![2, 2])
        );
        let result = fft2(&x);

        assert_eq!(result.shape().as_slice(), &[2, 4]); // 2x2 complex = 2x4 floats
    }
}
