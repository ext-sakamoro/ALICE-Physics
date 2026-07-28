//! Bridging Distance Analysis for Support-Free Overhangs
//!
//! Phase C3 of the ALICE-Physics completeness project. Given a set of
//! **bridge edges** — horizontal spans that must be printed without support
//! by extruding filament between two anchor points — this module compares
//! each span length against the material's maximum reliable bridging
//! distance and reports the risky ones.
//!
//! # Material limits (from `filament_db`)
//!
//! | Material | Max bridge (mm) |
//! |----------|-----------------|
//! | PLA | 20 |
//! | PETG | 15 |
//! | ABS | 12 |
//! | PC | 15 |
//! | TPU | 5 |
//! | Nylon | 10 |
//! | CF-Nylon | 20 |
//! | PEEK | 25 |
//!
//! # Usage pattern
//!
//! Upstream (mesh analyser, slicer or `alice-print::bridging`) identifies
//! horizontal edges at layer boundaries and passes them in as
//! `BridgeSpan` records. This module returns per-span safety flags plus an
//! aggregate summary.

use crate::filament_db::MaterialProperties;
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Bridge span input
// ============================================================================

/// One horizontal edge that must be printed as an unsupported bridge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BridgeSpan {
    /// Start point in world coordinates (mm).
    pub start: Vec3Fix,
    /// End point in world coordinates (mm).
    pub end: Vec3Fix,
}

impl BridgeSpan {
    /// Straight-line length (mm) of the span.
    #[must_use]
    pub fn length_mm(&self) -> Fix128 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        let dz = self.end.z - self.start.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Height (Z coordinate difference) of the two endpoints.
    /// A "true" horizontal bridge has ≈ 0 height difference.
    #[must_use]
    pub fn z_delta_mm(&self) -> Fix128 {
        (self.end.z - self.start.z).abs()
    }
}

// ============================================================================
// Report
// ============================================================================

/// Per-span safety flag with details.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BridgeCheck {
    /// The span analysed.
    pub span: BridgeSpan,
    /// Measured length (mm).
    pub length_mm: Fix128,
    /// Material's maximum reliable bridge distance (mm).
    pub allowable_mm: Fix128,
    /// Safety margin = allowable / length. Values < 1 = unsafe.
    pub safety_ratio: Fix128,
    /// True iff `safety_ratio ≥ 1`.
    pub is_safe: bool,
}

/// Aggregate report from a bridge scan.
#[derive(Clone, Debug, Default)]
pub struct BridgingReport {
    /// One check per input span.
    pub checks: Vec<BridgeCheck>,
    /// Number of spans flagged as unsafe (`safety_ratio < 1`).
    pub unsafe_count: usize,
    /// Longest span seen (mm).
    pub max_length_mm: Fix128,
}

impl BridgingReport {
    /// True iff any span was flagged unsafe.
    #[inline]
    #[must_use]
    pub fn has_unsafe(&self) -> bool {
        self.unsafe_count > 0
    }

    /// Iterator over only the unsafe checks.
    pub fn unsafe_checks(&self) -> impl Iterator<Item = &BridgeCheck> {
        self.checks.iter().filter(|c| !c.is_safe)
    }
}

// ============================================================================
// Analysis
// ============================================================================

/// Analyse a single span against a material's limit.
#[must_use]
pub fn check_span(span: &BridgeSpan, m: &MaterialProperties) -> BridgeCheck {
    let length = span.length_mm();
    let allow = m.bridging_distance_mm;
    let ratio = if length.is_zero() {
        Fix128::from_int(i64::MAX >> 8)
    } else if allow.is_zero() {
        Fix128::ZERO
    } else {
        allow / length
    };
    BridgeCheck {
        span: *span,
        length_mm: length,
        allowable_mm: allow,
        safety_ratio: ratio,
        is_safe: ratio >= Fix128::ONE,
    }
}

/// Analyse all spans against a material's limit and aggregate the results.
#[must_use]
pub fn analyze_bridges(spans: &[BridgeSpan], m: &MaterialProperties) -> BridgingReport {
    let mut checks = Vec::with_capacity(spans.len());
    let mut unsafe_count = 0usize;
    let mut max_length = Fix128::ZERO;
    for span in spans {
        let c = check_span(span, m);
        if !c.is_safe {
            unsafe_count += 1;
        }
        if c.length_mm > max_length {
            max_length = c.length_mm;
        }
        checks.push(c);
    }
    BridgingReport {
        checks,
        unsafe_count,
        max_length_mm: max_length,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: i64, y: i64, z: i64) -> Vec3Fix {
        Vec3Fix::new(
            Fix128::from_int(x),
            Fix128::from_int(y),
            Fix128::from_int(z),
        )
    }

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn span_length_horizontal() {
        let s = BridgeSpan {
            start: point(0, 0, 5),
            end: point(15, 0, 5),
        };
        assert_eq!(s.length_mm(), Fix128::from_int(15));
    }

    #[test]
    fn span_length_diagonal() {
        // 3-4-5 triangle
        let s = BridgeSpan {
            start: point(0, 0, 0),
            end: point(3, 4, 0),
        };
        assert!(approx_eq(
            s.length_mm(),
            Fix128::from_int(5),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn span_z_delta_zero_for_horizontal() {
        let s = BridgeSpan {
            start: point(0, 0, 5),
            end: point(20, 0, 5),
        };
        assert_eq!(s.z_delta_mm(), Fix128::ZERO);
    }

    #[test]
    fn short_span_is_safe_for_pla() {
        // 10mm bridge, PLA allowable 20 → ratio 2, safe
        let s = BridgeSpan {
            start: point(0, 0, 0),
            end: point(10, 0, 0),
        };
        let c = check_span(&s, &MaterialProperties::pla());
        assert!(c.is_safe);
        assert_eq!(c.safety_ratio, Fix128::from_int(2));
    }

    #[test]
    fn long_span_unsafe_for_pla() {
        // 30mm bridge > PLA allowable 20
        let s = BridgeSpan {
            start: point(0, 0, 0),
            end: point(30, 0, 0),
        };
        let c = check_span(&s, &MaterialProperties::pla());
        assert!(!c.is_safe);
        assert!(c.safety_ratio < Fix128::ONE);
    }

    #[test]
    fn tpu_has_stricter_limit_than_pla() {
        let s = BridgeSpan {
            start: point(0, 0, 0),
            end: point(8, 0, 0),
        };
        let c_pla = check_span(&s, &MaterialProperties::pla());
        let c_tpu = check_span(&s, &MaterialProperties::tpu());
        // 8mm is safe for PLA (20), unsafe for TPU (5)
        assert!(c_pla.is_safe);
        assert!(!c_tpu.is_safe);
    }

    #[test]
    fn sheet_metal_zero_allowable_always_unsafe() {
        // SUS304 has bridging_distance = 0
        let s = BridgeSpan {
            start: point(0, 0, 0),
            end: point(5, 0, 0),
        };
        let c = check_span(&s, &MaterialProperties::sus304());
        assert!(!c.is_safe);
        assert_eq!(c.safety_ratio, Fix128::ZERO);
    }

    #[test]
    fn zero_length_span_reports_infinite_safety() {
        // Degenerate span — both endpoints coincide
        let s = BridgeSpan {
            start: point(0, 0, 0),
            end: point(0, 0, 0),
        };
        let c = check_span(&s, &MaterialProperties::pla());
        assert!(c.safety_ratio > Fix128::from_int(1_000_000));
    }

    #[test]
    fn analyze_bridges_empty_input() {
        let report = analyze_bridges(&[], &MaterialProperties::pla());
        assert_eq!(report.checks.len(), 0);
        assert!(!report.has_unsafe());
    }

    #[test]
    fn analyze_bridges_mixed_safety() {
        let spans = vec![
            BridgeSpan {
                start: point(0, 0, 0),
                end: point(5, 0, 0),
            }, // safe (PLA)
            BridgeSpan {
                start: point(0, 0, 0),
                end: point(25, 0, 0),
            }, // unsafe (PLA)
            BridgeSpan {
                start: point(0, 0, 0),
                end: point(10, 0, 0),
            }, // safe (PLA)
        ];
        let report = analyze_bridges(&spans, &MaterialProperties::pla());
        assert_eq!(report.checks.len(), 3);
        assert_eq!(report.unsafe_count, 1);
        assert!(report.has_unsafe());
        assert_eq!(report.max_length_mm, Fix128::from_int(25));
        assert_eq!(report.unsafe_checks().count(), 1);
    }

    #[test]
    fn peek_has_highest_allowable() {
        // PEEK: 25mm bridging — largest
        let s = BridgeSpan {
            start: point(0, 0, 0),
            end: point(23, 0, 0),
        };
        for m in [
            MaterialProperties::peek(),
            MaterialProperties::pla(),
            MaterialProperties::abs(),
        ] {
            let c = check_span(&s, &m);
            if m.name == "PEEK" || m.name == "CF-Nylon" {
                assert!(c.is_safe, "{} should handle 23mm", m.name);
            } else {
                assert!(!c.is_safe, "{} should fail on 23mm", m.name);
            }
        }
    }
}
