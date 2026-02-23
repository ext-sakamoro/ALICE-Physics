//! Physics Error Types
//!
//! Unified error type for the ALICE-Physics engine. Functions that can fail
//! (deserialization, body lookup, constraint validation) return
//! `Result<T, PhysicsError>` instead of raw booleans or panicking.
//!
//! Author: Moroya Sakamoto

use core::fmt;

/// Unified error type for physics operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicsError {
    /// Body index is out of range.
    InvalidBodyIndex {
        /// The invalid index that was provided
        index: usize,
        /// Current number of bodies in the world
        count: usize,
    },
    /// State deserialization failed (corrupted or incompatible data).
    DeserializationFailed,
    /// A constraint references a body that does not exist.
    InvalidConstraint {
        /// Human-readable description of the problem
        reason: &'static str,
    },
    /// A zero-length direction or normal was provided where a unit vector is required.
    ZeroLengthVector {
        /// Context describing where the zero-length vector was encountered
        context: &'static str,
    },
    /// I/O error (reading/writing scene files).
    #[cfg(feature = "std")]
    IoError {
        /// Description of the I/O error
        message: &'static str,
    },
    /// A capacity limit was exceeded (too many bodies, constraints, etc.).
    CapacityExceeded {
        /// What resource was exhausted
        resource: &'static str,
        /// The limit that was exceeded
        limit: usize,
    },
    /// Invalid configuration parameter.
    InvalidConfiguration {
        /// Description of the invalid configuration
        reason: &'static str,
    },
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBodyIndex { index, count } => {
                write!(f, "body index {index} out of range (count={count})")
            }
            Self::DeserializationFailed => write!(f, "state deserialization failed"),
            Self::InvalidConstraint { reason } => {
                write!(f, "invalid constraint: {reason}")
            }
            Self::ZeroLengthVector { context } => {
                write!(f, "zero-length vector in {context}")
            }
            #[cfg(feature = "std")]
            Self::IoError { message } => write!(f, "I/O error: {message}"),
            Self::CapacityExceeded { resource, limit } => {
                write!(f, "{resource} capacity exceeded (limit={limit})")
            }
            Self::InvalidConfiguration { reason } => {
                write!(f, "invalid configuration: {reason}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PhysicsError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn test_error_display() {
        let e = PhysicsError::InvalidBodyIndex { index: 5, count: 3 };
        let s = format!("{}", e);
        assert!(s.contains("5"), "Should contain index");
        assert!(s.contains("3"), "Should contain count");
    }

    #[test]
    fn test_error_debug() {
        let e = PhysicsError::DeserializationFailed;
        let s = format!("{:?}", e);
        assert!(s.contains("DeserializationFailed"));
    }

    #[test]
    fn test_error_variants() {
        let e1 = PhysicsError::InvalidBodyIndex { index: 0, count: 0 };
        let e2 = PhysicsError::DeserializationFailed;
        let e3 = PhysicsError::InvalidConstraint {
            reason: "body A == body B",
        };
        let e4 = PhysicsError::ZeroLengthVector {
            context: "ray direction",
        };
        assert_ne!(e1, e2);
        assert_ne!(e3, e4);
    }

    #[test]
    fn test_capacity_exceeded() {
        let e = PhysicsError::CapacityExceeded {
            resource: "bodies",
            limit: 10000,
        };
        let s = format!("{}", e);
        assert!(s.contains("bodies"));
        assert!(s.contains("10000"));
    }

    #[test]
    fn test_invalid_configuration() {
        let e = PhysicsError::InvalidConfiguration {
            reason: "substeps must be > 0",
        };
        let s = format!("{}", e);
        assert!(s.contains("substeps"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_io_error() {
        let e = PhysicsError::IoError {
            message: "file not found",
        };
        let s = format!("{}", e);
        assert!(s.contains("file not found"));
    }
}
