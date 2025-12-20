//! Reduction operations on arrays.

use crate::trace::{is_tracing, trace_reduce, Primitive};
use crate::{buffer::Buffer, Array, DType, Device, Shape};

/// Reduce over all elements with a binary operation.
fn reduce_all<F>(input: &Array, init: f32, f: F) -> f32
where
    F: Fn(f32, f32) -> f32,
{
    assert_eq!(input.dtype(), DType::Float32, "Only Float32 supported");
    let data = input.to_vec();
    data.iter().fold(init, |acc, &x| f(acc, x))
}

/// Reduce along a specific axis.
fn reduce_axis<F>(
    input: &Array,
    axis: usize,
    op: Primitive,
    init: f32,
    f: F,
) -> Array
where
    F: Fn(f32, f32) -> f32,
{
    assert_eq!(input.dtype(), DType::Float32, "Only Float32 supported");
    assert_eq!(input.device(), Device::Cpu, "Only CPU supported for now");
    assert!(axis < input.ndim(), "Axis out of bounds");

    let shape = input.shape();
    let dims = shape.as_slice();

    // Result shape has the reduced axis removed
    let mut result_dims: Vec<usize> = dims.to_vec();
    result_dims.remove(axis);
    let result_shape = if result_dims.is_empty() {
        Shape::scalar()
    } else {
        Shape::new(result_dims.clone())
    };

    let input_data = input.to_vec();
    let result_size = result_shape.size();
    let mut result_data = vec![init; result_size];

    // Compute strides for input
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    // Iterate over result indices
    for (result_idx, item) in result_data.iter_mut().enumerate() {
        // Convert flat result index to multi-dimensional
        let mut result_multi = vec![0; result_dims.len()];
        let mut idx = result_idx;
        for i in (0..result_dims.len()).rev() {
            result_multi[i] = idx % result_shape.as_slice()[i];
            idx /= result_shape.as_slice()[i];
        }

        // Insert the reduced axis and iterate over it
        let mut acc = init;
        for axis_idx in 0..dims[axis] {
            let mut input_multi = Vec::with_capacity(dims.len());
            let mut result_i = 0;
            for i in 0..dims.len() {
                if i == axis {
                    input_multi.push(axis_idx);
                } else {
                    input_multi.push(result_multi[result_i]);
                    result_i += 1;
                }
            }

            // Convert multi-dimensional index to flat
            let flat_idx: usize = input_multi
                .iter()
                .zip(strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum();

            acc = f(acc, input_data[flat_idx]);
        }

        *item = acc;
    }

    let buffer = Buffer::from_f32(result_data, Device::Cpu);
    let result = Array::from_buffer(buffer, result_shape.clone());

    // Register with trace context if tracing is active
    if is_tracing() {
        trace_reduce(result.id(), op, input, result_shape);
    }

    result
}

impl Array {
    /// Sum of all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let sum = a.sum_all();
    /// assert_eq!(sum, 10.0);
    /// ```
    pub fn sum_all(&self) -> f32 {
        reduce_all(self, 0.0, |acc, x| acc + x)
    }

    /// Sum of all elements, returned as a scalar Array.
    ///
    /// This is a convenience method for autodiff that wraps `sum_all()`.
    pub fn sum_all_array(&self) -> Array {
        let val = self.sum_all();
        let result = Array::from_vec(vec![val], crate::Shape::scalar());

        // Register with trace context if tracing is active
        if is_tracing() {
            trace_reduce(
                result.id(),
                Primitive::SumAll,
                self,
                crate::Shape::scalar(),
            );
        }

        result
    }

    /// Sum along a specific axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new(vec![2, 3]));
    /// let sum_axis0 = a.sum(0);
    /// assert_eq!(sum_axis0.to_vec(), vec![5.0, 7.0, 9.0]);
    /// let sum_axis1 = a.sum(1);
    /// assert_eq!(sum_axis1.to_vec(), vec![6.0, 15.0]);
    /// ```
    pub fn sum(&self, axis: usize) -> Array {
        reduce_axis(self, axis, Primitive::Sum { axis }, 0.0, |acc, x| acc + x)
    }

    /// Mean of all elements.
    pub fn mean_all(&self) -> f32 {
        self.sum_all() / (self.size() as f32)
    }

    /// Mean along a specific axis.
    pub fn mean(&self, axis: usize) -> Array {
        reduce_axis(self, axis, Primitive::Mean { axis }, 0.0, |acc, x| {
            acc + x / (self.shape().as_slice()[axis] as f32)
        })
    }

    /// Maximum of all elements.
    pub fn max_all(&self) -> f32 {
        reduce_all(self, f32::NEG_INFINITY, |acc, x| acc.max(x))
    }

    /// Maximum along a specific axis.
    pub fn max(&self, axis: usize) -> Array {
        reduce_axis(
            self,
            axis,
            Primitive::MaxAxis { axis },
            f32::NEG_INFINITY,
            |acc, x| acc.max(x),
        )
    }

    /// Minimum of all elements.
    pub fn min_all(&self) -> f32 {
        reduce_all(self, f32::INFINITY, |acc, x| acc.min(x))
    }

    /// Minimum along a specific axis.
    pub fn min(&self, axis: usize) -> Array {
        reduce_axis(
            self,
            axis,
            Primitive::MinAxis { axis },
            f32::INFINITY,
            |acc, x| acc.min(x),
        )
    }

    /// Product of all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let prod = a.prod_all();
    /// assert_eq!(prod, 24.0);
    /// ```
    pub fn prod_all(&self) -> f32 {
        reduce_all(self, 1.0, |acc, x| acc * x)
    }

    /// Product along a specific axis.
    pub fn prod(&self, axis: usize) -> Array {
        reduce_axis(self, axis, Primitive::ProdAxis { axis }, 1.0, |acc, x| {
            acc * x
        })
    }

    /// Index of minimum element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
    /// let idx = a.argmin();
    /// assert_eq!(idx, 1);
    /// ```
    pub fn argmin(&self) -> usize {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        data.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Index of maximum element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
    /// let idx = a.argmax();
    /// assert_eq!(idx, 2);
    /// ```
    pub fn argmax(&self) -> usize {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Variance of all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let var = a.var();
    /// assert!((var - 1.25).abs() < 1e-6);
    /// ```
    pub fn var(&self) -> f32 {
        let mean = self.mean_all();
        let data = self.to_vec();
        let sum_sq_diff: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum();
        sum_sq_diff / data.len() as f32
    }

    /// Standard deviation of all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let std = a.std();
    /// assert!((std - 1.118).abs() < 0.01);
    /// ```
    pub fn std(&self) -> f32 {
        self.var().sqrt()
    }

    /// Variance along a specific axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let var_axis0 = a.var_axis(0);
    /// assert_eq!(var_axis0.shape().as_slice(), &[2]);
    /// ```
    pub fn var_axis(&self, axis: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert!(axis < self.ndim(), "Axis out of bounds");

        let mean = self.mean(axis);
        let mean_data = mean.to_vec();

        let shape = self.shape().as_slice();
        let data = self.to_vec();

        // Result shape has the reduced axis removed
        let mut result_dims: Vec<usize> = shape.to_vec();
        result_dims.remove(axis);
        let result_shape = if result_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(result_dims.clone())
        };

        let result_size = result_shape.size();
        let mut result_data = vec![0.0; result_size];

        // Compute strides for input
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Iterate over result indices
        for (result_idx, item) in result_data.iter_mut().enumerate() {
            // Convert flat result index to multi-dimensional
            let mut result_multi = vec![0; result_dims.len()];
            let mut idx = result_idx;
            for i in (0..result_dims.len()).rev() {
                result_multi[i] = idx % result_shape.as_slice()[i];
                idx /= result_shape.as_slice()[i];
            }

            let mean_val = mean_data[result_idx];
            let mut sum_sq = 0.0;

            // Iterate over the reduced axis
            for axis_idx in 0..shape[axis] {
                let mut input_multi = Vec::with_capacity(shape.len());
                let mut result_i = 0;
                for i in 0..shape.len() {
                    if i == axis {
                        input_multi.push(axis_idx);
                    } else {
                        input_multi.push(result_multi[result_i]);
                        result_i += 1;
                    }
                }

                // Convert multi-dimensional index to flat
                let flat_idx: usize = input_multi
                    .iter()
                    .zip(strides.iter())
                    .map(|(idx, stride)| idx * stride)
                    .sum();

                let diff = data[flat_idx] - mean_val;
                sum_sq += diff * diff;
            }

            *item = sum_sq / shape[axis] as f32;
        }

        Array::from_vec(result_data, result_shape)
    }

    /// Standard deviation along a specific axis.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
    /// let std_axis0 = a.std_axis(0);
    /// assert_eq!(std_axis0.shape().as_slice(), &[2]);
    /// ```
    pub fn std_axis(&self, axis: usize) -> Array {
        let var = self.var_axis(axis);
        let data = var.to_vec();
        let result: Vec<f32> = data.iter().map(|&x| x.sqrt()).collect();
        Array::from_vec(result, var.shape().clone())
    }

    /// Median of all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 3.0, 2.0], Shape::new(vec![3]));
    /// let med = a.median();
    /// assert_eq!(med, 2.0);
    /// ```
    pub fn median(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let mut data = self.to_vec();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = data.len();
        if len % 2 == 0 {
            (data[len / 2 - 1] + data[len / 2]) / 2.0
        } else {
            data[len / 2]
        }
    }

    /// Percentile of all elements.
    ///
    /// # Arguments
    ///
    /// * `q` - Percentile to compute (0-100)
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let p50 = a.percentile(50.0);
    /// assert_eq!(p50, 3.0);
    /// ```
    pub fn percentile(&self, q: f32) -> f32 {
        assert!(q >= 0.0 && q <= 100.0, "Percentile must be between 0 and 100");
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");

        let mut data = self.to_vec();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = data.len();
        if len == 1 {
            return data[0];
        }

        let index = (q / 100.0) * (len - 1) as f32;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            data[lower]
        } else {
            let weight = index - lower as f32;
            data[lower] * (1.0 - weight) + data[upper] * weight
        }
    }

    /// Cumulative sum of array elements.
    ///
    /// Returns an array of the same shape with cumulative sums.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let cumsum = a.cumsum();
    /// assert_eq!(cumsum.to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
    /// ```
    pub fn cumsum(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let mut result = Vec::with_capacity(data.len());
        let mut sum = 0.0;

        for &val in data.iter() {
            sum += val;
            result.push(sum);
        }

        Array::from_vec(result, self.shape().clone())
    }

    /// Cumulative product of array elements.
    ///
    /// Returns an array of the same shape with cumulative products.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let cumprod = a.cumprod();
    /// assert_eq!(cumprod.to_vec(), vec![1.0, 2.0, 6.0, 24.0]);
    /// ```
    pub fn cumprod(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let mut result = Vec::with_capacity(data.len());
        let mut prod = 1.0;

        for &val in data.iter() {
            prod *= val;
            result.push(prod);
        }

        Array::from_vec(result, self.shape().clone())
    }

    /// Cumulative maximum of array elements.
    ///
    /// Returns an array of the same shape with cumulative maximums.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
    /// let cummax = a.cummax();
    /// assert_eq!(cummax.to_vec(), vec![3.0, 3.0, 4.0, 4.0]);
    /// ```
    pub fn cummax(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let mut result = Vec::with_capacity(data.len());
        let mut max = f32::NEG_INFINITY;

        for &val in data.iter() {
            max = max.max(val);
            result.push(max);
        }

        Array::from_vec(result, self.shape().clone())
    }

    /// Cumulative minimum of array elements.
    ///
    /// Returns an array of the same shape with cumulative minimums.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
    /// let cummin = a.cummin();
    /// assert_eq!(cummin.to_vec(), vec![3.0, 1.0, 1.0, 1.0]);
    /// ```
    pub fn cummin(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let mut result = Vec::with_capacity(data.len());
        let mut min = f32::INFINITY;

        for &val in data.iter() {
            min = min.min(val);
            result.push(min);
        }

        Array::from_vec(result, self.shape().clone())
    }

    /// Calculate the discrete difference along the array.
    ///
    /// Computes the difference between consecutive elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 3.0, 6.0, 10.0], Shape::new(vec![4]));
    /// let diff = a.diff();
    /// assert_eq!(diff.to_vec(), vec![2.0, 3.0, 4.0]);
    /// ```
    pub fn diff(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        assert!(data.len() > 0, "Array must have at least 1 element");

        if data.len() == 1 {
            return Array::from_vec(vec![], Shape::new(vec![0]));
        }

        let mut result = Vec::with_capacity(data.len() - 1);
        for i in 1..data.len() {
            result.push(data[i] - data[i - 1]);
        }

        let len = result.len();
        Array::from_vec(result, Shape::new(vec![len]))
    }

    /// Calculate the n-th discrete difference.
    ///
    /// Recursively applies diff n times.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new(vec![5]));
    /// let diff2 = a.diff_n(2);
    /// assert_eq!(diff2.to_vec(), vec![0.0, 0.0, 0.0]);
    /// ```
    pub fn diff_n(&self, n: usize) -> Array {
        if n == 0 {
            return self.clone();
        }

        let mut result = self.diff();
        for _ in 1..n {
            result = result.diff();
        }
        result
    }

    /// Sum of array elements, ignoring NaN values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 3.0, 4.0], Shape::new(vec![4]));
    /// let sum = a.nansum();
    /// assert_eq!(sum, 8.0);
    /// ```
    pub fn nansum(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        data.iter().filter(|x| !x.is_nan()).sum()
    }

    /// Mean of array elements, ignoring NaN values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 3.0, 4.0], Shape::new(vec![4]));
    /// let mean = a.nanmean();
    /// assert_eq!(mean, 8.0 / 3.0);
    /// ```
    pub fn nanmean(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let valid: Vec<f32> = data.iter().copied().filter(|x| !x.is_nan()).collect();
        if valid.is_empty() {
            return f32::NAN;
        }
        valid.iter().sum::<f32>() / valid.len() as f32
    }

    /// Maximum of array elements, ignoring NaN values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 4.0, 2.0], Shape::new(vec![4]));
    /// let max = a.nanmax();
    /// assert_eq!(max, 4.0);
    /// ```
    pub fn nanmax(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        data.iter()
            .copied()
            .filter(|x| !x.is_nan())
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Minimum of array elements, ignoring NaN values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 4.0, 2.0], Shape::new(vec![4]));
    /// let min = a.nanmin();
    /// assert_eq!(min, 1.0);
    /// ```
    pub fn nanmin(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        data.iter()
            .copied()
            .filter(|x| !x.is_nan())
            .fold(f32::INFINITY, f32::min)
    }

    /// Standard deviation of array elements, ignoring NaN values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 3.0, 5.0], Shape::new(vec![4]));
    /// let std = a.nanstd();
    /// assert!((std - 2.0).abs() < 1e-5);
    /// ```
    pub fn nanstd(&self) -> f32 {
        self.nanvar().sqrt()
    }

    /// Variance of array elements, ignoring NaN values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 3.0, 5.0], Shape::new(vec![4]));
    /// let var = a.nanvar();
    /// assert!((var - 4.0).abs() < 1e-5);
    /// ```
    pub fn nanvar(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let valid: Vec<f32> = data.iter().copied().filter(|x| !x.is_nan()).collect();

        if valid.is_empty() || valid.len() == 1 {
            return f32::NAN;
        }

        let mean = valid.iter().sum::<f32>() / valid.len() as f32;
        let variance = valid
            .iter()
            .map(|x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f32>()
            / (valid.len() - 1) as f32; // Bessel's correction (sample variance)

        variance
    }

    /// Median of array elements, ignoring NaN values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, f32::NAN, 3.0, 5.0, 2.0], Shape::new(vec![5]));
    /// let median = a.nanmedian();
    /// assert_eq!(median, 2.5);
    /// ```
    pub fn nanmedian(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let mut valid: Vec<f32> = data.iter().copied().filter(|x| !x.is_nan()).collect();

        if valid.is_empty() {
            return f32::NAN;
        }

        valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = valid.len();

        if len % 2 == 0 {
            (valid[len / 2 - 1] + valid[len / 2]) / 2.0
        } else {
            valid[len / 2]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sum_all() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        assert_eq!(a.sum_all(), 10.0);
    }

    #[test]
    fn test_sum_axis() {
        // 2x3 array: [[1, 2, 3], [4, 5, 6]]
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        // Sum along axis 0 (collapse rows): [5, 7, 9]
        let sum_axis0 = a.sum(0);
        assert_eq!(sum_axis0.shape().as_slice(), &[3]);
        assert_eq!(sum_axis0.to_vec(), vec![5.0, 7.0, 9.0]);

        // Sum along axis 1 (collapse columns): [6, 15]
        let sum_axis1 = a.sum(1);
        assert_eq!(sum_axis1.shape().as_slice(), &[2]);
        assert_eq!(sum_axis1.to_vec(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_mean_all() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        assert_abs_diff_eq!(a.mean_all(), 2.5, epsilon = 1e-6);
    }

    #[test]
    fn test_mean_axis() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        let mean_axis0 = a.mean(0);
        assert_eq!(mean_axis0.to_vec(), vec![2.5, 3.5, 4.5]);

        let mean_axis1 = a.mean(1);
        assert_eq!(mean_axis1.to_vec(), vec![2.0, 5.0]);
    }

    #[test]
    fn test_max_all() {
        let a = Array::from_vec(vec![1.0, 5.0, 3.0, 2.0], Shape::new(vec![4]));
        assert_eq!(a.max_all(), 5.0);
    }

    #[test]
    fn test_max_axis() {
        let a = Array::from_vec(
            vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        let max_axis0 = a.max(0);
        assert_eq!(max_axis0.to_vec(), vec![2.0, 5.0, 6.0]);

        let max_axis1 = a.max(1);
        assert_eq!(max_axis1.to_vec(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_min_all() {
        let a = Array::from_vec(vec![3.0, 1.0, 5.0, 2.0], Shape::new(vec![4]));
        assert_eq!(a.min_all(), 1.0);
    }

    #[test]
    fn test_min_axis() {
        let a = Array::from_vec(
            vec![3.0, 5.0, 2.0, 1.0, 4.0, 6.0],
            Shape::new(vec![2, 3]),
        );

        let min_axis0 = a.min(0);
        assert_eq!(min_axis0.to_vec(), vec![1.0, 4.0, 2.0]);

        let min_axis1 = a.min(1);
        assert_eq!(min_axis1.to_vec(), vec![2.0, 1.0]);
    }

    #[test]
    fn test_reduce_3d() {
        // 2x2x2 array
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new(vec![2, 2, 2]),
        );

        // Sum along middle axis
        let sum_axis1 = a.sum(1);
        assert_eq!(sum_axis1.shape().as_slice(), &[2, 2]);
        assert_eq!(sum_axis1.to_vec(), vec![4.0, 6.0, 12.0, 14.0]);
    }

    #[test]
    fn test_cumsum() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let cumsum = a.cumsum();
        assert_eq!(cumsum.to_vec(), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumprod() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let cumprod = a.cumprod();
        assert_eq!(cumprod.to_vec(), vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_cummax() {
        let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
        let cummax = a.cummax();
        assert_eq!(cummax.to_vec(), vec![3.0, 3.0, 4.0, 4.0]);
    }

    #[test]
    fn test_cummin() {
        let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
        let cummin = a.cummin();
        assert_eq!(cummin.to_vec(), vec![3.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_diff() {
        let a = Array::from_vec(vec![1.0, 3.0, 6.0, 10.0], Shape::new(vec![4]));
        let diff = a.diff();
        assert_eq!(diff.to_vec(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_diff_n() {
        // Linear sequence - second derivative should be 0
        let a = Array::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            Shape::new(vec![5]),
        );
        let diff2 = a.diff_n(2);
        assert_eq!(diff2.to_vec(), vec![0.0, 0.0, 0.0]);

        // diff_n(0) should return the original array
        let diff0 = a.diff_n(0);
        assert_eq!(diff0.to_vec(), a.to_vec());
    }

    #[test]
    fn test_nansum() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, 3.0, 4.0],
            Shape::new(vec![4]),
        );
        let sum = a.nansum();
        assert_eq!(sum, 8.0);
    }

    #[test]
    fn test_nanmean() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, 3.0, 4.0],
            Shape::new(vec![4]),
        );
        let mean = a.nanmean();
        assert_abs_diff_eq!(mean, 8.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_nanmax() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, 4.0, 2.0],
            Shape::new(vec![4]),
        );
        let max = a.nanmax();
        assert_eq!(max, 4.0);
    }

    #[test]
    fn test_nanmin() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, 4.0, 2.0],
            Shape::new(vec![4]),
        );
        let min = a.nanmin();
        assert_eq!(min, 1.0);
    }

    #[test]
    fn test_nanstd() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, 3.0, 5.0],
            Shape::new(vec![4]),
        );
        let std = a.nanstd();
        assert_abs_diff_eq!(std, 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_nanvar() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, 3.0, 5.0],
            Shape::new(vec![4]),
        );
        let var = a.nanvar();
        assert_abs_diff_eq!(var, 4.0, epsilon = 1e-5);
    }

    #[test]
    fn test_nanmedian() {
        let a = Array::from_vec(
            vec![1.0, f32::NAN, 3.0, 5.0, 2.0],
            Shape::new(vec![5]),
        );
        let median = a.nanmedian();
        assert_eq!(median, 2.5);
    }
}
