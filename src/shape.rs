//! Shape and stride utilities for n-dimensional arrays.

use std::fmt;

/// Shape of an n-dimensional array.
///
/// Represented as a vector of dimensions. An empty vector represents a scalar.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Shape;
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.ndim(), 3);
    /// assert_eq!(shape.size(), 24);
    /// ```
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Create a scalar shape (empty dimensions).
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    /// Returns the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements.
    pub fn size(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// Returns a slice of the dimensions.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }

    /// Returns true if this is a scalar shape.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Get a specific dimension, or None if out of bounds.
    pub fn get(&self, index: usize) -> Option<usize> {
        self.dims.get(index).copied()
    }

    /// Compute default row-major (C-order) strides for this shape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::Shape;
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// let strides = shape.default_strides();
    /// assert_eq!(strides, vec![12, 4, 1]);
    /// ```
    pub fn default_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.ndim()];
        for i in (0..self.ndim().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }

    /// Check if two shapes are broadcast-compatible and return the result shape.
    ///
    /// Following NumPy broadcasting rules: dimensions are compatible if they are equal
    /// or one of them is 1.
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        let ndim = self.ndim().max(other.ndim());
        let mut result = Vec::with_capacity(ndim);

        for i in 0..ndim {
            let dim1 = if i < self.ndim() {
                self.dims[self.ndim() - 1 - i]
            } else {
                1
            };
            let dim2 = if i < other.ndim() {
                other.dims[other.ndim() - 1 - i]
            } else {
                1
            };

            if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
                result.push(dim1.max(dim2));
            } else {
                return None; // Incompatible shapes
            }
        }

        result.reverse();
        Some(Shape::new(result))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        if self.dims.len() == 1 {
            write!(f, ",")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.size(), 24);
        assert_eq!(shape.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_scalar_shape() {
        let shape = Shape::scalar();
        assert_eq!(shape.ndim(), 0);
        assert_eq!(shape.size(), 1);
        assert!(shape.is_scalar());
    }

    #[test]
    fn test_default_strides() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.default_strides(), vec![12, 4, 1]);

        let shape = Shape::new(vec![5]);
        assert_eq!(shape.default_strides(), vec![1]);

        let shape = Shape::scalar();
        assert_eq!(shape.default_strides(), Vec::<usize>::new());
    }

    #[test]
    fn test_broadcast() {
        let s1 = Shape::new(vec![3, 1]);
        let s2 = Shape::new(vec![1, 4]);
        assert_eq!(s1.broadcast_with(&s2), Some(Shape::new(vec![3, 4])));

        let s1 = Shape::new(vec![2, 3]);
        let s2 = Shape::new(vec![3]);
        assert_eq!(s1.broadcast_with(&s2), Some(Shape::new(vec![2, 3])));

        let s1 = Shape::new(vec![2, 3]);
        let s2 = Shape::new(vec![4]);
        assert_eq!(s1.broadcast_with(&s2), None); // Incompatible
    }

    #[test]
    fn test_display() {
        assert_eq!(Shape::new(vec![2, 3, 4]).to_string(), "(2, 3, 4)");
        assert_eq!(Shape::new(vec![5]).to_string(), "(5,)");
        assert_eq!(Shape::scalar().to_string(), "()");
    }
}
