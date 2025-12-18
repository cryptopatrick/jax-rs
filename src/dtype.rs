//! Data type definitions and utilities.

use std::fmt;

/// Numerical data type for array contents.
///
/// Corresponds to jax-js DType enum. Supports basic types
/// that can be efficiently worked with on the web.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point
    Float32,
    /// 16-bit floating point
    Float16,
    /// 32-bit signed integer
    Int32,
    /// 32-bit unsigned integer
    Uint32,
    /// Boolean (stored as 4-byte value)
    Bool,
}

impl DType {
    /// Returns the byte width of this dtype.
    #[inline]
    pub const fn byte_width(self) -> usize {
        match self {
            DType::Float32 | DType::Int32 | DType::Uint32 | DType::Bool => 4,
            DType::Float16 => 2,
        }
    }

    /// Returns true if this is a floating-point dtype.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, DType::Float32 | DType::Float16)
    }

    /// Returns true if this is an integer dtype.
    #[inline]
    pub const fn is_int(self) -> bool {
        matches!(self, DType::Int32 | DType::Uint32)
    }

    /// Returns true if this is a signed integer dtype.
    #[inline]
    pub const fn is_signed(self) -> bool {
        matches!(self, DType::Int32)
    }

    /// Promotes two dtypes according to JAX's type promotion rules.
    ///
    /// Type lattice: `bool -> uint32 -> int32 -> float16 -> float32`
    ///
    /// # Examples
    ///
    /// ```
    /// # use jax_rs::DType;
    /// assert_eq!(DType::promote(DType::Bool, DType::Int32), DType::Int32);
    /// assert_eq!(DType::promote(DType::Uint32, DType::Int32), DType::Int32);
    /// assert_eq!(DType::promote(DType::Int32, DType::Float16), DType::Float16);
    /// assert_eq!(DType::promote(DType::Float16, DType::Float32), DType::Float32);
    /// ```
    pub fn promote(dtype1: DType, dtype2: DType) -> DType {
        if dtype1 == dtype2 {
            return dtype1;
        }

        // Promotion order (higher rank = later in chain)
        let rank = |d: DType| match d {
            DType::Bool => 0,
            DType::Uint32 => 1,
            DType::Int32 => 2,
            DType::Float16 => 3,
            DType::Float32 => 4,
        };

        if rank(dtype1) > rank(dtype2) {
            dtype1
        } else {
            dtype2
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float32 => write!(f, "float32"),
            DType::Float16 => write!(f, "float16"),
            DType::Int32 => write!(f, "int32"),
            DType::Uint32 => write!(f, "uint32"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_width() {
        assert_eq!(DType::Float32.byte_width(), 4);
        assert_eq!(DType::Float16.byte_width(), 2);
        assert_eq!(DType::Int32.byte_width(), 4);
        assert_eq!(DType::Uint32.byte_width(), 4);
        assert_eq!(DType::Bool.byte_width(), 4);
    }

    #[test]
    fn test_is_float() {
        assert!(DType::Float32.is_float());
        assert!(DType::Float16.is_float());
        assert!(!DType::Int32.is_float());
        assert!(!DType::Uint32.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[test]
    fn test_type_promotion() {
        assert_eq!(DType::promote(DType::Bool, DType::Int32), DType::Int32);
        assert_eq!(DType::promote(DType::Uint32, DType::Int32), DType::Int32);
        assert_eq!(
            DType::promote(DType::Int32, DType::Float16),
            DType::Float16
        );
        assert_eq!(
            DType::promote(DType::Float16, DType::Float32),
            DType::Float32
        );
        assert_eq!(
            DType::promote(DType::Uint32, DType::Float32),
            DType::Float32
        );
        assert_eq!(DType::promote(DType::Float32, DType::Float32), DType::Float32);
    }

    #[test]
    fn test_display() {
        assert_eq!(DType::Float32.to_string(), "float32");
        assert_eq!(DType::Float16.to_string(), "float16");
        assert_eq!(DType::Int32.to_string(), "int32");
        assert_eq!(DType::Uint32.to_string(), "uint32");
        assert_eq!(DType::Bool.to_string(), "bool");
    }
}
