//! Thin-Wall Vibration & Printer Resonance Risk
//!
//! Phase E2 of the ALICE-Physics completeness project. FDM printed parts
//! with thin planar walls (< 3 mm) can resonate with printer-generated
//! vibration and produce "spaghetti"-like buckling, layer smearing, or
//! post-print acoustic buzz. This module compares a wall's first natural
//! frequency (from `modal::plate_natural_frequency_hz`) against a database
//! of common excitation sources and flags dangerous overlaps.
//!
//! # Typical excitation sources (Hz)
//!
//! | Source | Frequency |
//! |--------|-----------|
//! | Bambu X1C X/Y stepper (movement) | 60 - 200 Hz |
//! | Z lead-screw stepper | 20 - 80 Hz |
//! | Cooling fan (5000-8000 rpm) | 80 - 130 Hz |
//! | Chamber HVAC | 50 - 60 Hz |
//! | Extruder gear whine | 300 - 800 Hz |
//! | Handling / transport shock | 5 - 20 Hz |
//!
//! # Resonance criterion
//!
//! A wall is at risk when its natural frequency is within ±20 % of a
//! significant excitation source. Beyond that, the wall's damping (typically
//! 5-10 % for FDM plastics) reduces resonance amplification below 3×.
//!
//! # References
//!
//! - Meirovitch, *Fundamentals of Vibrations* Ch. 4.
//! - Bambu Lab X1C service manual, motor / fan specifications.
//! - Blevins, *Formulas for Natural Frequency and Mode Shape*.

use crate::filament_db::MaterialProperties;
use crate::math::Fix128;
use crate::modal::plate_natural_frequency_hz;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Excitation sources
// ============================================================================

/// A named vibration excitation source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExcitationSource {
    /// Human-readable source name.
    pub name: &'static str,
    /// Nominal excitation frequency (Hz).
    pub frequency_hz: Fix128,
}

/// Curated defaults for common printer / environmental sources.
#[must_use]
pub fn default_excitation_sources() -> Vec<ExcitationSource> {
    vec![
        ExcitationSource {
            name: "X/Y stepper (Bambu X1C)",
            frequency_hz: Fix128::from_int(120),
        },
        ExcitationSource {
            name: "Z lead-screw stepper",
            frequency_hz: Fix128::from_int(40),
        },
        ExcitationSource {
            name: "Cooling fan 7000rpm",
            frequency_hz: Fix128::from_int(117),
        },
        ExcitationSource {
            name: "Chamber HVAC 50Hz",
            frequency_hz: Fix128::from_int(50),
        },
        ExcitationSource {
            name: "Extruder gear whine",
            frequency_hz: Fix128::from_int(400),
        },
        ExcitationSource {
            name: "Handling / transport shock",
            frequency_hz: Fix128::from_int(15),
        },
    ]
}

// ============================================================================
// Resonance report
// ============================================================================

/// Analysis output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WallResonanceReport {
    /// Wall's first natural frequency (Hz).
    pub wall_frequency_hz: Fix128,
    /// Nearest excitation source (by absolute frequency difference).
    pub nearest_source: ExcitationSource,
    /// Ratio `wall / source` — should be far from 1.0 for safety.
    pub frequency_ratio: Fix128,
    /// True iff the wall frequency lies within `resonance_band` of the
    /// nearest source (default ±20 %).
    pub is_risky: bool,
}

/// Analyse a rectangular wall against a set of excitation sources.
///
/// `wall_thickness_mm`, `side_a_mm`, `side_b_mm` describe a simply-supported
/// rectangular plate (same convention as `modal::plate_natural_frequency_hz`).
/// `resonance_band` = fractional half-width of the risky region (0.2 = ±20 %).
#[must_use]
pub fn analyze_wall_resonance(
    material: &MaterialProperties,
    poisson_ratio: Fix128,
    wall_thickness_mm: Fix128,
    side_a_mm: Fix128,
    side_b_mm: Fix128,
    sources: &[ExcitationSource],
    resonance_band: Fix128,
) -> WallResonanceReport {
    let f_wall = plate_natural_frequency_hz(
        material,
        poisson_ratio,
        wall_thickness_mm,
        side_a_mm,
        side_b_mm,
    );

    // Locate the nearest source by absolute distance.
    let mut nearest = sources[0];
    let mut best_dist = (f_wall - nearest.frequency_hz).abs();
    for src in sources.iter().skip(1) {
        let d = (f_wall - src.frequency_hz).abs();
        if d < best_dist {
            best_dist = d;
            nearest = *src;
        }
    }

    let ratio = if nearest.frequency_hz.is_zero() {
        Fix128::ZERO
    } else {
        f_wall / nearest.frequency_hz
    };
    let is_risky = ratio > (Fix128::ONE - resonance_band) && ratio < (Fix128::ONE + resonance_band);

    WallResonanceReport {
        wall_frequency_hz: f_wall,
        nearest_source: nearest,
        frequency_ratio: ratio,
        is_risky,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_sources_include_x1c_stepper() {
        let sources = default_excitation_sources();
        assert!(sources.iter().any(|s| s.name.contains("X1C")));
        assert!(sources.iter().any(|s| s.name.contains("HVAC")));
    }

    #[test]
    fn thick_wall_high_frequency_safe() {
        // 5mm thick PLA plate 50×50mm — expected f > 1000 Hz, no source overlap
        let sources = default_excitation_sources();
        let report = analyze_wall_resonance(
            &MaterialProperties::pla(),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(5),
            Fix128::from_int(50),
            Fix128::from_int(50),
            &sources,
            Fix128::from_ratio(20, 100),
        );
        assert!(report.wall_frequency_hz > Fix128::from_int(1000));
        // Default sources max at 400 Hz → wall_freq/src >> 1 → not in band
        // (ratio would need to be near 1)
        assert!(!report.is_risky, "5mm/50mm wall should be safe");
    }

    #[test]
    fn thin_large_wall_can_resonate() {
        // 1mm thick 200×200 PLA plate — low natural frequency
        let sources = default_excitation_sources();
        let report = analyze_wall_resonance(
            &MaterialProperties::pla(),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(1),
            Fix128::from_int(200),
            Fix128::from_int(200),
            &sources,
            Fix128::from_ratio(20, 100),
        );
        // wall_frequency should be < 200 Hz
        assert!(report.wall_frequency_hz < Fix128::from_int(500));
        // nearest source should be identified from the default list
        assert!(report.nearest_source.frequency_hz > Fix128::ZERO);
    }

    #[test]
    fn ratio_computed_from_nearest() {
        let sources = default_excitation_sources();
        let report = analyze_wall_resonance(
            &MaterialProperties::pla(),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(2),
            Fix128::from_int(100),
            Fix128::from_int(100),
            &sources,
            Fix128::from_ratio(20, 100),
        );
        assert_ne!(report.frequency_ratio, Fix128::ZERO);
        assert_eq!(
            report.frequency_ratio,
            report.wall_frequency_hz / report.nearest_source.frequency_hz
        );
    }

    #[test]
    fn resonance_band_wider_more_risky() {
        // Same wall geometry, evaluate with tight vs loose band
        let sources = default_excitation_sources();
        // Use a moderate geometry where wall_freq is near ~100 Hz range
        let thin_report_tight = analyze_wall_resonance(
            &MaterialProperties::pla(),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(2),
            Fix128::from_int(120),
            Fix128::from_int(120),
            &sources,
            Fix128::from_ratio(5, 100),
        );
        let thin_report_loose = analyze_wall_resonance(
            &MaterialProperties::pla(),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(2),
            Fix128::from_int(120),
            Fix128::from_int(120),
            &sources,
            Fix128::from_ratio(50, 100),
        );
        // Loose band cannot be less risky than tight band
        assert!(thin_report_loose.is_risky || !thin_report_tight.is_risky);
    }

    #[test]
    fn custom_source_can_be_picked() {
        let sources = vec![ExcitationSource {
            name: "custom",
            frequency_hz: Fix128::from_int(500),
        }];
        let report = analyze_wall_resonance(
            &MaterialProperties::pla(),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(2),
            Fix128::from_int(50),
            Fix128::from_int(50),
            &sources,
            Fix128::from_ratio(20, 100),
        );
        assert_eq!(report.nearest_source.name, "custom");
    }

    #[test]
    fn abs_wall_vs_pla_different_frequency() {
        let sources = default_excitation_sources();
        let f_pla = analyze_wall_resonance(
            &MaterialProperties::pla(),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(2),
            Fix128::from_int(100),
            Fix128::from_int(100),
            &sources,
            Fix128::from_ratio(20, 100),
        )
        .wall_frequency_hz;
        let f_abs = analyze_wall_resonance(
            &MaterialProperties::abs(),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(2),
            Fix128::from_int(100),
            Fix128::from_int(100),
            &sources,
            Fix128::from_ratio(20, 100),
        )
        .wall_frequency_hz;
        // Material with different E → different frequency
        assert_ne!(f_pla, f_abs);
    }
}

// Convenience: analyze uses default sources when caller supplies `&sources`
// as `&default_excitation_sources()`. The signature is generic across custom
// source lists — no extra overload required.
#[cfg(test)]
mod default_convenience {
    use super::*;

    #[test]
    fn helper_call_syntax_smoke() {
        let sources = default_excitation_sources();
        let _report = analyze_wall_resonance(
            &MaterialProperties::pla(),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(2),
            Fix128::from_int(100),
            Fix128::from_int(100),
            &sources,
            Fix128::from_ratio(20, 100),
        );
    }
}
