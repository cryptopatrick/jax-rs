//! Sorting and searching operations.

use crate::{Array, DType, Shape};

impl Array {
    /// Sort array elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
    /// let sorted = a.sort();
    /// assert_eq!(sorted.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn sort(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let mut data = self.to_vec();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Array::from_vec(data, self.shape().clone())
    }

    /// Sort array elements in descending order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
    /// let sorted = a.sort_descending();
    /// assert_eq!(sorted.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    /// ```
    pub fn sort_descending(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let mut data = self.to_vec();
        data.sort_by(|a, b| b.partial_cmp(a).unwrap());
        Array::from_vec(data, self.shape().clone())
    }

    /// Return indices that would sort the array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
    /// let indices = a.argsort();
    /// assert_eq!(indices, vec![1, 3, 0, 2]);
    /// ```
    pub fn argsort(&self) -> Vec<usize> {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
        indices
    }

    /// Return indices that would sort the array in descending order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
    /// let indices = a.argsort_descending();
    /// assert_eq!(indices, vec![2, 0, 3, 1]);
    /// ```
    pub fn argsort_descending(&self) -> Vec<usize> {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.sort_by(|&a, &b| data[b].partial_cmp(&data[a]).unwrap());
        indices
    }

    /// Find the k smallest elements and return their indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0, 5.0], Shape::new(vec![5]));
    /// let top2 = a.top_k_smallest(2);
    /// assert_eq!(top2, vec![1, 3]);
    /// ```
    pub fn top_k_smallest(&self, k: usize) -> Vec<usize> {
        assert!(k <= self.size(), "k must be <= array size");
        let indices = self.argsort();
        indices.into_iter().take(k).collect()
    }

    /// Find the k largest elements and return their indices.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0, 5.0], Shape::new(vec![5]));
    /// let top2 = a.top_k_largest(2);
    /// assert_eq!(top2, vec![4, 2]);
    /// ```
    pub fn top_k_largest(&self, k: usize) -> Vec<usize> {
        assert!(k <= self.size(), "k must be <= array size");
        let indices = self.argsort_descending();
        indices.into_iter().take(k).collect()
    }

    /// Find indices where elements should be inserted to maintain order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 3.0, 5.0, 7.0], Shape::new(vec![4]));
    /// let idx = a.searchsorted(4.0);
    /// assert_eq!(idx, 2);
    /// ```
    pub fn searchsorted(&self, value: f32) -> usize {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();

        // Binary search
        let mut left = 0;
        let mut right = data.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if data[mid] < value {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        left
    }

    /// Find unique elements in the array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 1.0, 3.0, 2.0], Shape::new(vec![5]));
    /// let unique = a.unique();
    /// assert_eq!(unique.to_vec(), vec![1.0, 2.0, 3.0]);
    /// ```
    pub fn unique(&self) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let mut data = self.to_vec();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        data.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
        let len = data.len();
        Array::from_vec(data, Shape::new(vec![len]))
    }

    /// Count occurrences of each unique value.
    ///
    /// Returns (unique_values, counts).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0], Shape::new(vec![6]));
    /// let (values, counts) = a.unique_counts();
    /// assert_eq!(values.to_vec(), vec![1.0, 2.0, 3.0]);
    /// assert_eq!(counts, vec![3, 2, 1]);
    /// ```
    pub fn unique_counts(&self) -> (Array, Vec<usize>) {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let mut data = self.to_vec();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut unique_vals = Vec::new();
        let mut counts = Vec::new();

        if !data.is_empty() {
            let mut current = data[0];
            let mut count = 1;

            for &val in data.iter().skip(1) {
                if (val - current).abs() < 1e-7 {
                    count += 1;
                } else {
                    unique_vals.push(current);
                    counts.push(count);
                    current = val;
                    count = 1;
                }
            }
            unique_vals.push(current);
            counts.push(count);
        }

        (
            Array::from_vec(unique_vals, Shape::new(vec![counts.len()])),
            counts,
        )
    }

    /// Find the set difference of two arrays.
    ///
    /// Returns the unique values in the first array that are not in the second array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let b = Array::from_vec(vec![2.0, 4.0, 5.0], Shape::new(vec![3]));
    /// let diff = a.setdiff1d(&b);
    /// assert_eq!(diff.to_vec(), vec![1.0, 3.0]);
    /// ```
    pub fn setdiff1d(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let self_unique = self.unique();
        let other_data = other.to_vec();

        let result: Vec<f32> = self_unique
            .to_vec()
            .into_iter()
            .filter(|&val| !other_data.iter().any(|&x| (x - val).abs() < 1e-7))
            .collect();

        let len = result.len();
        Array::from_vec(result, Shape::new(vec![len]))
    }

    /// Find the union of two arrays.
    ///
    /// Returns the unique values that are in either of the two arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
    /// let union = a.union1d(&b);
    /// assert_eq!(union.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn union1d(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let mut combined = self.to_vec();
        combined.extend(other.to_vec());

        let temp = Array::from_vec(combined, Shape::new(vec![self.size() + other.size()]));
        temp.unique()
    }

    /// Find the intersection of two arrays.
    ///
    /// Returns the unique values that are in both arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
    /// let intersect = a.intersect1d(&b);
    /// assert_eq!(intersect.to_vec(), vec![2.0, 3.0]);
    /// ```
    pub fn intersect1d(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let self_unique = self.unique();
        let other_data = other.to_vec();

        let result: Vec<f32> = self_unique
            .to_vec()
            .into_iter()
            .filter(|&val| other_data.iter().any(|&x| (x - val).abs() < 1e-7))
            .collect();

        let len = result.len();
        Array::from_vec(result, Shape::new(vec![len]))
    }

    /// Find the exclusive-or of two arrays.
    ///
    /// Returns the unique values that are in exactly one of the two arrays.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
    /// let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
    /// let xor = a.setxor1d(&b);
    /// assert_eq!(xor.to_vec(), vec![1.0, 4.0]);
    /// ```
    pub fn setxor1d(&self, other: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(other.dtype(), DType::Float32, "Only Float32 supported");

        let union = self.union1d(other);
        let intersect = self.intersect1d(other);
        union.setdiff1d(&intersect)
    }

    /// Test whether each element of a 1D array is also present in a second array.
    ///
    /// Returns a boolean-like array (1.0 for true, 0.0 for false).
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
    /// let b = Array::from_vec(vec![2.0, 4.0], Shape::new(vec![2]));
    /// let result = a.in1d(&b);
    /// assert_eq!(result.to_vec(), vec![0.0, 1.0, 0.0, 1.0]);
    /// ```
    pub fn in1d(&self, test_elements: &Array) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            test_elements.dtype(),
            DType::Float32,
            "Only Float32 supported"
        );

        let data = self.to_vec();
        let test_data = test_elements.to_vec();

        let result: Vec<f32> = data
            .iter()
            .map(|&val| {
                if test_data.iter().any(|&x| (x - val).abs() < 1e-7) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        Array::from_vec(result, self.shape().clone())
    }

    /// Return the indices of the bins to which each value belongs.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let x = Array::from_vec(vec![0.2, 6.4, 3.0, 1.6], Shape::new(vec![4]));
    /// let bins = Array::from_vec(vec![0.0, 1.0, 2.5, 4.0, 10.0], Shape::new(vec![5]));
    /// let indices = x.digitize(&bins);
    /// assert_eq!(indices, vec![1, 4, 3, 2]);
    /// ```
    pub fn digitize(&self, bins: &Array) -> Vec<usize> {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(bins.dtype(), DType::Float32, "Only Float32 supported");

        let data = self.to_vec();
        let bin_edges = bins.to_vec();

        data.iter()
            .map(|&val| {
                // Find the bin index using binary search
                let mut left = 0;
                let mut right = bin_edges.len();

                while left < right {
                    let mid = left + (right - left) / 2;
                    if bin_edges[mid] <= val {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                left
            })
            .collect()
    }

    /// Compute the histogram of a dataset.
    ///
    /// Returns (hist, bin_edges) where hist contains the counts and bin_edges
    /// contains the bin boundaries.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0], Shape::new(vec![6]));
    /// let (hist, edges) = a.histogram(3, 0.0, 4.0);
    /// assert_eq!(hist, vec![3, 2, 1]);
    /// ```
    pub fn histogram(&self, bins: usize, range_min: f32, range_max: f32) -> (Vec<usize>, Vec<f32>) {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert!(bins > 0, "Number of bins must be positive");
        assert!(range_max > range_min, "range_max must be > range_min");

        let data = self.to_vec();
        let bin_width = (range_max - range_min) / bins as f32;

        // Create bin edges
        let mut bin_edges = Vec::with_capacity(bins + 1);
        for i in 0..=bins {
            bin_edges.push(range_min + i as f32 * bin_width);
        }

        // Count values in each bin
        let mut hist = vec![0; bins];
        for &val in data.iter() {
            if val >= range_min && val <= range_max {
                let bin_idx = ((val - range_min) / bin_width).floor() as usize;
                let bin_idx = bin_idx.min(bins - 1); // Handle edge case where val == range_max
                hist[bin_idx] += 1;
            }
        }

        (hist, bin_edges)
    }

    /// Count number of occurrences of each value in array of non-negative integers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 7.0], Shape::new(vec![7]));
    /// let counts = a.bincount();
    /// assert_eq!(counts, vec![1, 3, 1, 1, 0, 0, 0, 1]);
    /// ```
    pub fn bincount(&self) -> Vec<usize> {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();

        // Find the maximum value to determine array size
        let max_val = data
            .iter()
            .map(|&x| x as usize)
            .max()
            .unwrap_or(0);

        let mut counts = vec![0; max_val + 1];
        for &val in data.iter() {
            let idx = val as usize;
            counts[idx] += 1;
        }

        counts
    }

    /// Count number of occurrences with optional weights.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![0.0, 1.0, 1.0, 2.0], Shape::new(vec![4]));
    /// let weights = Array::from_vec(vec![0.3, 0.5, 0.2, 0.7], Shape::new(vec![4]));
    /// let counts = a.bincount_weighted(&weights);
    /// assert!((counts[0] - 0.3).abs() < 1e-6);
    /// assert!((counts[1] - 0.7).abs() < 1e-6);
    /// assert!((counts[2] - 0.7).abs() < 1e-6);
    /// ```
    pub fn bincount_weighted(&self, weights: &Array) -> Vec<f32> {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(weights.dtype(), DType::Float32, "Only Float32 supported");
        assert_eq!(
            self.size(),
            weights.size(),
            "Array and weights must have same size"
        );

        let data = self.to_vec();
        let weight_data = weights.to_vec();

        // Find the maximum value to determine array size
        let max_val = data
            .iter()
            .map(|&x| x as usize)
            .max()
            .unwrap_or(0);

        let mut counts = vec![0.0; max_val + 1];
        for (i, &val) in data.iter().enumerate() {
            let idx = val as usize;
            counts[idx] += weight_data[i];
        }

        counts
    }

    /// Partially sort array so that the k-th element is in sorted position.
    ///
    /// Elements smaller than the k-th element are moved before it,
    /// and elements larger are moved after it.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0], Shape::new(vec![4]));
    /// let partitioned = a.partition(2);
    /// // Element at index 2 is in correct sorted position
    /// let data = partitioned.to_vec();
    /// assert!(data[0] <= data[2] && data[1] <= data[2]);
    /// assert!(data[2] <= data[3]);
    /// ```
    pub fn partition(&self, kth: usize) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert!(kth < self.size(), "kth must be less than array size");

        let mut data = self.to_vec();

        // Use selection algorithm (quickselect-style partitioning)
        let n = data.len();
        let mut left = 0;
        let mut right = n - 1;

        while left < right {
            let pivot = data[right];
            let mut store_idx = left;

            for i in left..right {
                if data[i] < pivot {
                    data.swap(i, store_idx);
                    store_idx += 1;
                }
            }
            data.swap(store_idx, right);

            if store_idx == kth {
                break;
            } else if store_idx < kth {
                left = store_idx + 1;
            } else {
                right = store_idx.saturating_sub(1);
            }
        }

        Array::from_vec(data, self.shape().clone())
    }

    /// Return indices that would partition the array.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 4.0, 2.0, 1.0], Shape::new(vec![4]));
    /// let indices = a.argpartition(2);
    /// // Indices are such that a[indices[0]] and a[indices[1]] are smaller than a[indices[2]]
    /// ```
    pub fn argpartition(&self, kth: usize) -> Vec<usize> {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert!(kth < self.size(), "kth must be less than array size");

        let data = self.to_vec();
        let mut indices: Vec<usize> = (0..data.len()).collect();

        // Partition indices based on values
        let n = indices.len();
        let mut left = 0;
        let mut right = n - 1;

        while left < right {
            let pivot_val = data[indices[right]];
            let mut store_idx = left;

            for i in left..right {
                if data[indices[i]] < pivot_val {
                    indices.swap(i, store_idx);
                    store_idx += 1;
                }
            }
            indices.swap(store_idx, right);

            if store_idx == kth {
                break;
            } else if store_idx < kth {
                left = store_idx + 1;
            } else {
                right = store_idx.saturating_sub(1);
            }
        }

        indices
    }

    /// Perform indirect stable sort using a sequence of keys.
    ///
    /// Sort by the last key first, then by second-to-last, etc.
    /// This is equivalent to numpy's lexsort.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// // Sort by surname, then by first name
    /// let surnames = Array::from_vec(vec![1.0, 2.0, 1.0, 2.0], Shape::new(vec![4]));  // Hertz, Move, Hertz, Newton
    /// let first_names = Array::from_vec(vec![1.0, 2.0, 2.0, 1.0], Shape::new(vec![4])); // Heinrich, Never, Lansen, Isaac
    /// let indices = Array::lexsort(&[&first_names, &surnames]);
    /// // Should be sorted by surname first, then first_name
    /// ```
    pub fn lexsort(keys: &[&Array]) -> Vec<usize> {
        assert!(!keys.is_empty(), "Need at least one key");

        let n = keys[0].size();
        for key in keys {
            assert_eq!(key.size(), n, "All keys must have same length");
            assert_eq!(key.dtype(), DType::Float32, "Only Float32 supported");
        }

        let mut indices: Vec<usize> = (0..n).collect();

        // Sort by keys in reverse order (last key is primary sort key)
        indices.sort_by(|&a, &b| {
            // Compare from last key to first
            for key in keys.iter().rev() {
                let key_data = key.to_vec();
                let cmp = key_data[a].partial_cmp(&key_data[b]).unwrap();
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });

        indices
    }

    /// Return the median element without full sorting.
    ///
    /// Uses quickselect algorithm for O(n) average performance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], Shape::new(vec![5]));
    /// let med = a.median_select();
    /// assert_eq!(med, 3.0);
    /// ```
    pub fn median_select(&self) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let n = self.size();
        assert!(n > 0, "Array must not be empty");

        let partitioned = self.partition(n / 2);
        let data = partitioned.to_vec();

        if n % 2 == 1 {
            data[n / 2]
        } else {
            // For even length, also need the element before
            let left = self.partition(n / 2 - 1);
            let left_data = left.to_vec();
            (left_data[n / 2 - 1] + data[n / 2]) / 2.0
        }
    }

    /// Return the k-th smallest element using selection algorithm.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 4.0, 1.0, 5.0], Shape::new(vec![5]));
    /// let kth = a.select_kth(2); // 0-indexed, so 3rd smallest
    /// assert_eq!(kth, 3.0);
    /// ```
    pub fn select_kth(&self, k: usize) -> f32 {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        assert!(k < self.size(), "k must be less than array size");

        let partitioned = self.partition(k);
        partitioned.to_vec()[k]
    }

    /// Sort array along the last axis.
    ///
    /// For 2D arrays, sorts each row independently.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0], Shape::new(vec![2, 3]));
    /// let sorted = a.sort_axis(-1);
    /// assert_eq!(sorted.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn sort_axis(&self, _axis: i32) -> Array {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let shape = self.shape().as_slice();

        if shape.len() == 1 {
            return self.sort();
        }

        if shape.len() == 2 {
            // Sort each row
            let (rows, cols) = (shape[0], shape[1]);
            let data = self.to_vec();
            let mut result = Vec::with_capacity(data.len());

            for r in 0..rows {
                let start = r * cols;
                let mut row: Vec<f32> = data[start..start + cols].to_vec();
                row.sort_by(|a, b| a.partial_cmp(b).unwrap());
                result.extend(row);
            }

            return Array::from_vec(result, self.shape().clone());
        }

        // For higher dimensions, just sort flat
        self.sort()
    }

    /// Return indices that would sort the array in stable order.
    ///
    /// Unlike argsort, stable_argsort preserves the relative order
    /// of equal elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::{Array, Shape};
    /// let a = Array::from_vec(vec![3.0, 1.0, 1.0, 2.0], Shape::new(vec![4]));
    /// let indices = a.stable_argsort();
    /// assert_eq!(indices, vec![1, 2, 3, 0]); // The two 1.0s maintain order
    /// ```
    pub fn stable_argsort(&self) -> Vec<usize> {
        assert_eq!(self.dtype(), DType::Float32, "Only Float32 supported");
        let data = self.to_vec();
        let mut indexed: Vec<(usize, f32)> = data.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        indexed.into_iter().map(|(i, _)| i).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort() {
        let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
        let sorted = a.sort();
        assert_eq!(sorted.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sort_descending() {
        let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
        let sorted = a.sort_descending();
        assert_eq!(sorted.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_argsort() {
        let a = Array::from_vec(vec![3.0, 1.0, 4.0, 2.0], Shape::new(vec![4]));
        let indices = a.argsort();
        assert_eq!(indices, vec![1, 3, 0, 2]);
    }

    #[test]
    fn test_top_k() {
        let a = Array::from_vec(
            vec![3.0, 1.0, 4.0, 2.0, 5.0],
            Shape::new(vec![5]),
        );
        let smallest = a.top_k_smallest(2);
        assert_eq!(smallest, vec![1, 3]);

        let largest = a.top_k_largest(2);
        assert_eq!(largest, vec![4, 2]);
    }

    #[test]
    fn test_searchsorted() {
        let a = Array::from_vec(vec![1.0, 3.0, 5.0, 7.0], Shape::new(vec![4]));
        assert_eq!(a.searchsorted(4.0), 2);
        assert_eq!(a.searchsorted(0.0), 0);
        assert_eq!(a.searchsorted(10.0), 4);
        assert_eq!(a.searchsorted(5.0), 2);
    }

    #[test]
    fn test_unique() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 1.0, 3.0, 2.0],
            Shape::new(vec![5]),
        );
        let unique = a.unique();
        assert_eq!(unique.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_unique_counts() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0],
            Shape::new(vec![6]),
        );
        let (values, counts) = a.unique_counts();
        assert_eq!(values.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(counts, vec![3, 2, 1]);
    }

    #[test]
    fn test_setdiff1d() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let b = Array::from_vec(vec![2.0, 4.0, 5.0], Shape::new(vec![3]));
        let diff = a.setdiff1d(&b);
        assert_eq!(diff.to_vec(), vec![1.0, 3.0]);
    }

    #[test]
    fn test_union1d() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let union = a.union1d(&b);
        assert_eq!(union.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_intersect1d() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let intersect = a.intersect1d(&b);
        assert_eq!(intersect.to_vec(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_setxor1d() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
        let b = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]));
        let xor = a.setxor1d(&b);
        assert_eq!(xor.to_vec(), vec![1.0, 4.0]);
    }

    #[test]
    fn test_in1d() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
        let b = Array::from_vec(vec![2.0, 4.0], Shape::new(vec![2]));
        let result = a.in1d(&b);
        assert_eq!(result.to_vec(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_digitize() {
        let x = Array::from_vec(vec![0.2, 6.4, 3.0, 1.6], Shape::new(vec![4]));
        let bins =
            Array::from_vec(vec![0.0, 1.0, 2.5, 4.0, 10.0], Shape::new(vec![5]));
        let indices = x.digitize(&bins);
        assert_eq!(indices, vec![1, 4, 3, 2]);
    }

    #[test]
    fn test_histogram() {
        let a = Array::from_vec(
            vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0],
            Shape::new(vec![6]),
        );
        let (hist, edges) = a.histogram(3, 0.0, 4.0);
        assert_eq!(hist, vec![3, 2, 1]);
        assert_eq!(edges.len(), 4); // bins + 1
    }

    #[test]
    fn test_bincount() {
        let a = Array::from_vec(
            vec![0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 7.0],
            Shape::new(vec![7]),
        );
        let counts = a.bincount();
        assert_eq!(counts, vec![1, 3, 1, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn test_bincount_weighted() {
        let a = Array::from_vec(vec![0.0, 1.0, 1.0, 2.0], Shape::new(vec![4]));
        let weights =
            Array::from_vec(vec![0.3, 0.5, 0.2, 0.7], Shape::new(vec![4]));
        let counts = a.bincount_weighted(&weights);
        assert!((counts[0] - 0.3).abs() < 1e-6);
        assert!((counts[1] - 0.7).abs() < 1e-6);
        assert!((counts[2] - 0.7).abs() < 1e-6);
    }
}
