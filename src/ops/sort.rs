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
}
