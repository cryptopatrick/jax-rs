//! Fast Fourier Transform operations.
//!
//! Basic CPU implementation of FFT operations using the Cooley-Tukey algorithm.

use crate::{Array, DType, Shape};
use crate::buffer::Buffer;
use crate::Device;
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

    let n = x.size();
    assert!(n.is_power_of_two(), "FFT size must be power of 2");

    // Device dispatch
    match x.device() {
        Device::WebGpu => {
            // GPU path
            let output_buffer = Buffer::zeros(n * 2, DType::Float32, Device::WebGpu);
            crate::backend::ops::gpu_fft(x.buffer(), &output_buffer, n, false);
            Array::from_buffer(output_buffer, Shape::new(vec![n * 2]))
        }
        Device::Cpu | Device::Wasm => {
            // CPU path
            let data = x.to_vec();

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
    }
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

    let n = x.size() / 2;
    assert!(n.is_power_of_two(), "FFT size must be power of 2");

    // Device dispatch
    match x.device() {
        Device::WebGpu => {
            // GPU path - complex to complex inverse FFT
            let output_buffer = Buffer::zeros(n * 2, DType::Float32, Device::WebGpu);
            crate::backend::ops::gpu_fft_complex(x.buffer(), &output_buffer, n, true);
            Array::from_buffer(output_buffer, Shape::new(vec![n * 2]))
        }
        Device::Cpu | Device::Wasm => {
            // CPU path
            let data = x.to_vec();

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
    }
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

/// Inverse 2D Fast Fourier Transform.
///
/// Computes the inverse 2D FFT by applying inverse 1D FFT to rows, then to columns.
/// Input should be in shape [height, width * 2] (interleaved complex values).
///
/// # Arguments
///
/// * `x` - Input 2D array (complex values as interleaved real/imag)
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
/// let fft_result = ops::fft::fft2(&x);
/// let reconstructed = ops::fft::ifft2(&fft_result);
/// // Should approximately equal original input (extract real parts)
/// ```
pub fn ifft2(x: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.ndim(), 2, "Input must be 2D");

    let shape = x.shape().as_slice();
    let (height, width_complex) = (shape[0], shape[1]);
    assert_eq!(width_complex % 2, 0, "Width must be even (interleaved complex)");

    let width = width_complex / 2;
    let data = x.to_vec();

    // Inverse FFT along rows
    let mut row_ifft = Vec::with_capacity(height * width * 2);
    for i in 0..height {
        let row_start = i * width * 2;
        let row_end = row_start + width * 2;
        let row_data = &data[row_start..row_end];

        let row_array = Array::from_vec(row_data.to_vec(), Shape::new(vec![width * 2]));
        let row_ifft_result = ifft(&row_array);
        row_ifft.extend_from_slice(&row_ifft_result.to_vec());
    }

    // Inverse FFT along columns (on complex data)
    let mut result = Vec::with_capacity(height * width * 2);
    for j in 0..(width * 2) {
        // Apply inverse FFT if this is a complete complex column (both real and imag parts)
        if j % 2 == 0 {
            // Collect complex pairs for this column
            let mut col_complex = Vec::with_capacity(height);
            for i in 0..height {
                let real = row_ifft[i * width * 2 + j];
                let imag = row_ifft[i * width * 2 + j + 1];
                col_complex.push((real, imag));
            }

            let col_ifft = fft_recursive(&col_complex, true);

            // Normalize by 1/height
            let scale = 1.0 / (height as f32);
            let col_ifft_normalized: Vec<Complex> = col_ifft
                .iter()
                .map(|(r, i)| (r * scale, i * scale))
                .collect();

            // Interleave back
            for i in 0..height {
                if j == 0 {
                    result.resize((i + 1) * width * 2, 0.0);
                }
                result[i * width * 2 + j] = col_ifft_normalized[i].0;
                result[i * width * 2 + j + 1] = col_ifft_normalized[i].1;
            }
        }
    }

    Array::from_vec(result, Shape::new(vec![height, width * 2]))
}

/// Inverse real 1D Fast Fourier Transform.
///
/// Computes the inverse of rfft(), reconstructing real-valued output from
/// n/2 + 1 complex frequencies. Input should be in interleaved [real, imag] format.
///
/// # Arguments
///
/// * `x` - Input array (complex values from rfft)
/// * `n` - Output size (original real signal length)
///
/// # Examples
///
/// ```
/// # use jax_rs::{ops, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let rfft_result = ops::fft::rfft(&x);
/// let reconstructed = ops::fft::irfft(&rfft_result, 4);
/// // Should approximately equal original input
/// ```
pub fn irfft(x: &Array, n: usize) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(x.ndim(), 1, "Input must be 1D");
    assert_eq!(x.size() % 2, 0, "Input size must be even (interleaved complex)");

    let data = x.to_vec();
    let m = data.len() / 2; // Number of complex values
    assert_eq!(m, n / 2 + 1, "Input size {} doesn't match expected {} for output size {}", m, n / 2 + 1, n);

    // Reconstruct full spectrum by adding conjugate symmetry
    let mut full_spectrum = Vec::with_capacity(n);

    // First half: original data (except DC and Nyquist)
    for i in 0..m {
        full_spectrum.push((data[i * 2], data[i * 2 + 1]));
    }

    // Second half: conjugate symmetry (skip DC component, reverse order)
    for i in (1..n - m + 1).rev() {
        full_spectrum.push((data[i * 2], -data[i * 2 + 1]));
    }

    // Apply inverse FFT
    let mut ifft_result = fft_recursive(&full_spectrum, true);

    // Normalize by 1/n
    let scale = 1.0 / (n as f32);
    for (r, i) in ifft_result.iter_mut() {
        *r *= scale;
        *i *= scale;
    }

    // Extract real parts (imaginary parts should be ~0 for real input)
    let output: Vec<f32> = ifft_result.iter().map(|(r, _)| *r).collect();

    Array::from_vec(output, Shape::new(vec![n]))
}

/// N-dimensional real Fast Fourier Transform.
///
/// Computes the N-dimensional FFT for real-valued input.
/// Currently supports 1D and 2D arrays. For 1D, equivalent to rfft.
/// For 2D, computes FFT along all axes with real optimization on the last axis.
///
/// # Arguments
///
/// * `x` - Input array (real values)
///
/// # Examples
///
/// ```
/// # use jax_rs::{ops, Array, Shape};
/// // 1D case
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let result = ops::fft::rfftn(&x);
/// assert_eq!(result.size(), 6); // (4/2 + 1) * 2 = 6 floats
/// ```
pub fn rfftn(x: &Array) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");

    match x.ndim() {
        1 => rfft(x),
        2 => {
            let shape = x.shape().as_slice();
            let (height, width) = (shape[0], shape[1]);
            let data = x.to_vec();

            // FFT along rows (full complex FFT)
            let mut row_fft = Vec::with_capacity(height * width * 2);
            for i in 0..height {
                let row_start = i * width;
                let row_end = row_start + width;
                let row_data = &data[row_start..row_end];

                let row_array = Array::from_vec(row_data.to_vec(), Shape::new(vec![width]));
                let row_fft_result = fft(&row_array);
                row_fft.extend_from_slice(&row_fft_result.to_vec());
            }

            // Real FFT along columns (only keep positive frequencies)
            let n_freq = width / 2 + 1;
            let mut result = Vec::with_capacity(height * n_freq * 2);

            for j in 0..n_freq {
                // Collect complex column
                let mut col_complex = Vec::with_capacity(height);
                for i in 0..height {
                    let real = row_fft[i * width * 2 + j * 2];
                    let imag = row_fft[i * width * 2 + j * 2 + 1];
                    col_complex.push((real, imag));
                }

                let col_fft = fft_recursive(&col_complex, false);

                // Store results
                for i in 0..height {
                    result.push(col_fft[i].0);
                    result.push(col_fft[i].1);
                }
            }

            // Reshape to [height, n_freq * 2]
            let mut final_result = vec![0.0; height * n_freq * 2];
            for i in 0..height {
                for j in 0..n_freq {
                    let src_idx = j * height * 2 + i * 2;
                    let dst_idx = i * n_freq * 2 + j * 2;
                    final_result[dst_idx] = result[src_idx];
                    final_result[dst_idx + 1] = result[src_idx + 1];
                }
            }

            Array::from_vec(final_result, Shape::new(vec![height, n_freq * 2]))
        }
        _ => panic!("rfftn only supports 1D and 2D arrays, got {}D", x.ndim())
    }
}

/// Inverse N-dimensional real Fast Fourier Transform.
///
/// Computes the inverse of rfftn(), reconstructing real-valued output.
/// Currently supports 1D and 2D arrays.
///
/// # Arguments
///
/// * `x` - Input array (complex values from rfftn)
/// * `shape` - Output shape (original real signal dimensions)
///
/// # Examples
///
/// ```
/// # use jax_rs::{ops, Array, Shape};
/// let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
/// let rfft_result = ops::fft::rfftn(&x);
/// let reconstructed = ops::fft::irfftn(&rfft_result, &[4]);
/// // Should approximately equal original input
/// ```
pub fn irfftn(x: &Array, shape: &[usize]) -> Array {
    assert_eq!(x.dtype(), DType::Float32, "Only Float32 supported");

    match shape.len() {
        1 => irfft(x, shape[0]),
        2 => {
            let (height, width) = (shape[0], shape[1]);
            let x_shape = x.shape().as_slice();
            assert_eq!(x.ndim(), 2, "Input must be 2D for 2D inverse");

            let n_freq = x_shape[1] / 2;
            assert_eq!(n_freq, width / 2 + 1, "Input frequency dimension doesn't match output width");

            let data = x.to_vec();

            // Inverse FFT along columns first
            let mut col_ifft_results = Vec::with_capacity(height * width * 2);

            for j in 0..n_freq {
                // Collect complex column
                let mut col_complex = Vec::with_capacity(height);
                for i in 0..height {
                    let idx = i * n_freq * 2 + j * 2;
                    col_complex.push((data[idx], data[idx + 1]));
                }

                let col_ifft = fft_recursive(&col_complex, true);

                // Normalize
                let scale = 1.0 / (height as f32);
                for i in 0..height {
                    let base_idx = i * width * 2 + j * 2;
                    if j == 0 && col_ifft_results.len() < (i + 1) * width * 2 {
                        col_ifft_results.resize((i + 1) * width * 2, 0.0);
                    }
                    col_ifft_results[base_idx] = col_ifft[i].0 * scale;
                    col_ifft_results[base_idx + 1] = col_ifft[i].1 * scale;
                }
            }

            // Reconstruct full width with conjugate symmetry and inverse FFT along rows
            let mut result = Vec::with_capacity(height * width);

            for i in 0..height {
                // Extract complex row values
                let mut row_complex = Vec::with_capacity(width);

                // Positive frequencies
                for j in 0..n_freq {
                    let idx = i * width * 2 + j * 2;
                    row_complex.push((col_ifft_results[idx], col_ifft_results[idx + 1]));
                }

                // Negative frequencies (conjugate symmetry)
                for j in (1..width - n_freq + 1).rev() {
                    let idx = i * width * 2 + j * 2;
                    row_complex.push((col_ifft_results[idx], -col_ifft_results[idx + 1]));
                }

                // Inverse FFT on row
                let mut row_ifft = fft_recursive(&row_complex, true);

                // Normalize
                let scale = 1.0 / (width as f32);
                for (r, _) in row_ifft.iter_mut() {
                    *r *= scale;
                    result.push(*r);
                }
            }

            Array::from_vec(result, Shape::new(vec![height, width]))
        }
        _ => panic!("irfftn only supports 1D and 2D arrays, got shape {:?}", shape)
    }
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

    #[test]
    fn test_ifft2_reconstruction() {
        // Test that ifft2(fft2(x)) ≈ x
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0,
                 9.0, 10.0, 11.0, 12.0,
                 13.0, 14.0, 15.0, 16.0],
            Shape::new(vec![4, 4])
        );

        let fft_result = fft2(&x);
        let reconstructed = ifft2(&fft_result);

        // Extract real parts (even indices in each row)
        let data = reconstructed.to_vec();
        let original = x.to_vec();

        for i in 0..4 {
            for j in 0..4 {
                let real_idx = i * 8 + j * 2; // 8 = 4 complex * 2
                let orig_idx = i * 4 + j;
                assert!(
                    (data[real_idx] - original[orig_idx]).abs() < 1e-4,
                    "Mismatch at ({}, {}): {} vs {}",
                    i, j, data[real_idx], original[orig_idx]
                );
            }
        }
    }

    #[test]
    fn test_irfft_reconstruction() {
        // Test that irfft(rfft(x)) ≈ x
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], Shape::new(vec![8]));
        let rfft_result = rfft(&x);
        let reconstructed = irfft(&rfft_result, 8);

        let original = x.to_vec();
        let result = reconstructed.to_vec();

        for i in 0..8 {
            assert!(
                (result[i] - original[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i, result[i], original[i]
            );
        }
    }

    #[test]
    fn test_rfftn_1d() {
        // 1D case should be equivalent to rfft
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let result_rfftn = rfftn(&x);
        let result_rfft = rfft(&x);

        assert_eq!(result_rfftn.shape(), result_rfft.shape());

        let data_rfftn = result_rfftn.to_vec();
        let data_rfft = result_rfft.to_vec();

        for i in 0..data_rfftn.len() {
            assert!((data_rfftn[i] - data_rfft[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rfftn_2d() {
        // Test 2D real FFT
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0],
            Shape::new(vec![2, 4])
        );
        let result = rfftn(&x);

        // Should have shape [2, (4/2 + 1) * 2] = [2, 6]
        assert_eq!(result.shape().as_slice(), &[2, 6]);
    }

    #[test]
    fn test_irfftn_1d_reconstruction() {
        // Test that irfftn(rfftn(x)) ≈ x for 1D
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], Shape::new(vec![8]));
        let rfft_result = rfftn(&x);
        let reconstructed = irfftn(&rfft_result, &[8]);

        let original = x.to_vec();
        let result = reconstructed.to_vec();

        for i in 0..8 {
            assert!(
                (result[i] - original[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i, result[i], original[i]
            );
        }
    }

    #[test]
    fn test_irfftn_2d_reconstruction() {
        // Test that irfftn(rfftn(x)) ≈ x for 2D
        let x = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0],
            Shape::new(vec![2, 4])
        );
        let rfft_result = rfftn(&x);
        let reconstructed = irfftn(&rfft_result, &[2, 4]);

        assert_eq!(reconstructed.shape().as_slice(), &[2, 4]);

        let original = x.to_vec();
        let result = reconstructed.to_vec();

        for i in 0..original.len() {
            assert!(
                (result[i] - original[i]).abs() < 1e-3,
                "Mismatch at {}: {} vs {}",
                i, result[i], original[i]
            );
        }
    }
}
