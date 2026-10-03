//! Audit oracles for `alice_physics::error::PhysicsError`.
//!
//! The Display text is the only specification of each message, so the exact
//! strings are written out here, field by field, and the match below has no
//! wildcard arm: adding a variant breaks compilation of this file until the
//! new variant gets a message oracle.

use alice_physics::error::PhysicsError;

fn expected_message(e: &PhysicsError) -> String {
    match e {
        PhysicsError::InvalidBodyIndex { index, count } => {
            format!("body index {index} out of range (count={count})")
        }
        PhysicsError::DeserializationFailed => "state deserialization failed".to_string(),
        PhysicsError::InvalidConstraint { reason } => format!("invalid constraint: {reason}"),
        PhysicsError::ZeroLengthVector { context } => format!("zero-length vector in {context}"),
        #[cfg(feature = "std")]
        PhysicsError::IoError { message } => format!("I/O error: {message}"),
        PhysicsError::CapacityExceeded { resource, limit } => {
            format!("{resource} capacity exceeded (limit={limit})")
        }
        PhysicsError::InvalidConfiguration { reason } => {
            format!("invalid configuration: {reason}")
        }
    }
}

fn all_variants() -> Vec<PhysicsError> {
    let mut v = vec![
        PhysicsError::InvalidBodyIndex { index: 5, count: 3 },
        PhysicsError::DeserializationFailed,
        PhysicsError::InvalidConstraint {
            reason: "body A == body B",
        },
        PhysicsError::ZeroLengthVector {
            context: "ray direction",
        },
        PhysicsError::CapacityExceeded {
            resource: "bodies",
            limit: 10_000,
        },
        PhysicsError::InvalidConfiguration {
            reason: "substeps must be > 0",
        },
    ];
    #[cfg(feature = "std")]
    v.push(PhysicsError::IoError {
        message: "file not found",
    });
    v
}

#[test]
fn display_of_every_variant_is_the_exact_documented_text() {
    for e in all_variants() {
        assert_eq!(format!("{e}"), expected_message(&e), "{e:?}");
    }
}

#[test]
fn literal_messages_for_two_representative_variants() {
    assert_eq!(
        PhysicsError::InvalidBodyIndex { index: 5, count: 3 }.to_string(),
        "body index 5 out of range (count=3)"
    );
    assert_eq!(
        PhysicsError::CapacityExceeded {
            resource: "bodies",
            limit: 10_000
        }
        .to_string(),
        "bodies capacity exceeded (limit=10000)"
    );
}

#[test]
fn display_handles_extreme_usize_fields() {
    let e = PhysicsError::InvalidBodyIndex {
        index: usize::MAX,
        count: 0,
    };
    let s = e.to_string();
    assert!(s.contains(&usize::MAX.to_string()));
    assert!(s.ends_with("(count=0)"));
}

#[test]
fn every_message_is_one_line_without_trailing_punctuation() {
    for e in all_variants() {
        let s = e.to_string();
        assert!(!s.is_empty());
        assert!(!s.contains('\n'), "{s}");
        assert!(!s.ends_with('.'), "{s}");
    }
}

#[test]
fn equality_distinguishes_variant_and_every_payload_field() {
    let vs = all_variants();
    for (i, a) in vs.iter().enumerate() {
        for (j, b) in vs.iter().enumerate() {
            assert_eq!(a == b, i == j, "{a:?} vs {b:?}");
        }
        assert_eq!(a, &a.clone());
    }
    let a = PhysicsError::InvalidBodyIndex { index: 1, count: 2 };
    assert_ne!(a, PhysicsError::InvalidBodyIndex { index: 2, count: 2 });
    assert_ne!(a, PhysicsError::InvalidBodyIndex { index: 1, count: 3 });
    let c = PhysicsError::CapacityExceeded {
        resource: "x",
        limit: 1,
    };
    assert_ne!(
        c,
        PhysicsError::CapacityExceeded {
            resource: "y",
            limit: 1
        }
    );
    assert_ne!(
        c,
        PhysicsError::CapacityExceeded {
            resource: "x",
            limit: 2
        }
    );
}

#[cfg(feature = "std")]
#[test]
fn implements_std_error_and_is_send_sync_static() {
    fn takes_error<E: std::error::Error + Send + Sync + 'static>(_: &E) {}
    for e in all_variants() {
        takes_error(&e);
        let boxed: Box<dyn std::error::Error + Send + Sync> = Box::new(e.clone());
        assert_eq!(boxed.to_string(), e.to_string());
        assert!(std::error::Error::source(&e).is_none());
    }
}

#[test]
fn debug_names_the_variant_and_shows_field_values() {
    let d = format!(
        "{:?}",
        PhysicsError::InvalidBodyIndex { index: 5, count: 3 }
    );
    assert!(
        d.contains("InvalidBodyIndex") && d.contains('5') && d.contains('3'),
        "{d}"
    );
    assert_eq!(
        format!("{:?}", PhysicsError::DeserializationFailed),
        "DeserializationFailed"
    );
}

/// The module doc says fallible functions return `Result<T, PhysicsError>`
/// "instead of raw booleans". `PhysicsWorld::deserialize_state` still returns
/// `bool`, so `DeserializationFailed` is never produced by the crate (wiring
/// finding AUD-B-S6W1-004). The signature is pinned here so the day it changes
/// this test must be revisited.
#[test]
fn deserialize_state_still_reports_failure_as_a_bool() {
    use alice_physics::{PhysicsConfig, PhysicsWorld};
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let r: bool = w.deserialize_state(&[0u8; 3]);
    assert!(!r);
}
