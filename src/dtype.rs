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
    /// 16-bit floating point (stored as u16 bits)
    Float16,
    /// 64-bit floating point
    Float64,
    /// 8-bit signed integer
    Int8,
    /// 16-bit signed integer
    Int16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 8-bit unsigned integer
    Uint8,
    /// 16-bit unsigned integer
    Uint16,
    /// 32-bit unsigned integer
    Uint32,
    /// 64-bit unsigned integer
    Uint64,
    /// Boolean (stored as 1-byte value)
    Bool,
}

impl DType {
    /// Returns the byte width of this dtype.
    #[inline]
    pub const fn byte_width(self) -> usize {
        match self {
            DType::Bool | DType::Int8 | DType::Uint8 => 1,
            DType::Float16 | DType::Int16 | DType::Uint16 => 2,
            DType::Float32 | DType::Int32 | DType::Uint32 => 4,
            DType::Float64 | DType::Int64 | DType::Uint64 => 8,
        }
    }

    /// Returns true if this is a floating-point dtype.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, DType::Float32 | DType::Float16 | DType::Float64)
    }

    /// Returns true if this is an integer dtype.
    #[inline]
    pub const fn is_int(self) -> bool {
        matches!(
            self,
            DType::Int8
                | DType::Int16
                | DType::Int32
                | DType::Int64
                | DType::Uint8
                | DType::Uint16
                | DType::Uint32
                | DType::Uint64
        )
    }

    /// Returns true if this is a signed integer dtype.
    #[inline]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        )
    }

    /// Returns true if this is an unsigned integer dtype.
    #[inline]
    pub const fn is_unsigned(self) -> bool {
        matches!(
            self,
            DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64
        )
    }

    /// Promotes two dtypes according to JAX's type promotion rules.
    ///
    /// Type lattice: `bool -> uint8 -> uint16 -> uint32 -> int8 -> int16 -> int32 -> float16 -> float32 -> float64`
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
            DType::Uint8 => 1,
            DType::Uint16 => 2,
            DType::Uint32 => 3,
            DType::Uint64 => 4,
            DType::Int8 => 5,
            DType::Int16 => 6,
            DType::Int32 => 7,
            DType::Int64 => 8,
            DType::Float16 => 9,
            DType::Float32 => 10,
            DType::Float64 => 11,
        };

        if rank(dtype1) > rank(dtype2) {
            dtype1
        } else {
            dtype2
        }
    }

    /// Cast a float32 value to this dtype (returns as f32 for storage).
    #[inline]
    pub fn cast_from_f32(self, value: f32) -> f32 {
        match self {
            DType::Float32 | DType::Float64 | DType::Float16 => value,
            DType::Int8 => (value as i8) as f32,
            DType::Int16 => (value as i16) as f32,
            DType::Int32 | DType::Int64 => (value as i32) as f32,
            DType::Uint8 => (value as u8) as f32,
            DType::Uint16 => (value as u16) as f32,
            DType::Uint32 | DType::Uint64 => (value as u32) as f32,
            DType::Bool => if value != 0.0 { 1.0 } else { 0.0 },
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Float32 => write!(f, "float32"),
            DType::Float16 => write!(f, "float16"),
            DType::Float64 => write!(f, "float64"),
            DType::Int8 => write!(f, "int8"),
            DType::Int16 => write!(f, "int16"),
            DType::Int32 => write!(f, "int32"),
            DType::Int64 => write!(f, "int64"),
            DType::Uint8 => write!(f, "uint8"),
            DType::Uint16 => write!(f, "uint16"),
            DType::Uint32 => write!(f, "uint32"),
            DType::Uint64 => write!(f, "uint64"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

impl DType {
    /// Parse a string into a DType.
    pub fn from_str(s: &str) -> Option<DType> {
        match s.to_lowercase().as_str() {
            "float32" | "f32" => Some(DType::Float32),
            "float16" | "f16" => Some(DType::Float16),
            "float64" | "f64" => Some(DType::Float64),
            "int8" | "i8" => Some(DType::Int8),
            "int16" | "i16" => Some(DType::Int16),
            "int32" | "i32" => Some(DType::Int32),
            "int64" | "i64" => Some(DType::Int64),
            "uint8" | "u8" => Some(DType::Uint8),
            "uint16" | "u16" => Some(DType::Uint16),
            "uint32" | "u32" => Some(DType::Uint32),
            "uint64" | "u64" => Some(DType::Uint64),
            "bool" => Some(DType::Bool),
            _ => None,
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
        assert_eq!(DType::Float64.byte_width(), 8);
        assert_eq!(DType::Int8.byte_width(), 1);
        assert_eq!(DType::Int16.byte_width(), 2);
        assert_eq!(DType::Int32.byte_width(), 4);
        assert_eq!(DType::Int64.byte_width(), 8);
        assert_eq!(DType::Uint8.byte_width(), 1);
        assert_eq!(DType::Uint16.byte_width(), 2);
        assert_eq!(DType::Uint32.byte_width(), 4);
        assert_eq!(DType::Uint64.byte_width(), 8);
        assert_eq!(DType::Bool.byte_width(), 1);
    }

    #[test]
    fn test_is_float() {
        assert!(DType::Float32.is_float());
        assert!(DType::Float16.is_float());
        assert!(DType::Float64.is_float());
        assert!(!DType::Int32.is_float());
        assert!(!DType::Uint32.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[test]
    fn test_is_int() {
        assert!(DType::Int8.is_int());
        assert!(DType::Int16.is_int());
        assert!(DType::Int32.is_int());
        assert!(DType::Int64.is_int());
        assert!(DType::Uint8.is_int());
        assert!(DType::Uint16.is_int());
        assert!(DType::Uint32.is_int());
        assert!(DType::Uint64.is_int());
        assert!(!DType::Float32.is_int());
        assert!(!DType::Bool.is_int());
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
        assert_eq!(
            DType::promote(DType::Float32, DType::Float32),
            DType::Float32
        );
        assert_eq!(DType::promote(DType::Uint8, DType::Uint16), DType::Uint16);
        assert_eq!(DType::promote(DType::Int8, DType::Int16), DType::Int16);
    }

    #[test]
    fn test_display() {
        assert_eq!(DType::Float32.to_string(), "float32");
        assert_eq!(DType::Float16.to_string(), "float16");
        assert_eq!(DType::Float64.to_string(), "float64");
        assert_eq!(DType::Int8.to_string(), "int8");
        assert_eq!(DType::Int16.to_string(), "int16");
        assert_eq!(DType::Int32.to_string(), "int32");
        assert_eq!(DType::Int64.to_string(), "int64");
        assert_eq!(DType::Uint8.to_string(), "uint8");
        assert_eq!(DType::Uint16.to_string(), "uint16");
        assert_eq!(DType::Uint32.to_string(), "uint32");
        assert_eq!(DType::Uint64.to_string(), "uint64");
        assert_eq!(DType::Bool.to_string(), "bool");
    }

    #[test]
    fn test_from_str() {
        assert_eq!(DType::from_str("float32"), Some(DType::Float32));
        assert_eq!(DType::from_str("f32"), Some(DType::Float32));
        assert_eq!(DType::from_str("int8"), Some(DType::Int8));
        assert_eq!(DType::from_str("i8"), Some(DType::Int8));
        assert_eq!(DType::from_str("uint16"), Some(DType::Uint16));
        assert_eq!(DType::from_str("bool"), Some(DType::Bool));
        assert_eq!(DType::from_str("unknown"), None);
    }

    #[test]
    fn test_cast_from_f32() {
        assert_eq!(DType::Int8.cast_from_f32(127.5), 127.0);
        assert_eq!(DType::Int8.cast_from_f32(-128.0), -128.0);
        assert_eq!(DType::Uint8.cast_from_f32(255.5), 255.0);
        assert_eq!(DType::Bool.cast_from_f32(0.0), 0.0);
        assert_eq!(DType::Bool.cast_from_f32(42.0), 1.0);
    }
}
