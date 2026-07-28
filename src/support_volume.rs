//! Support Structure Volume Estimation for 3D Printing
//!
//! Phase A4 of the ALICE-Physics completeness project. Given a set of overhang
//! regions (typically produced upstream by a slicer, mesh overhang detector or
//! `alice-print::overhang`), estimates the filament volume consumed by support
//! structures, plus rough print-time / filament-length figures.
//!
//! # Model
//!
//! Support structures are modelled as three layered contributions:
//!
//! 1. **Base**: dense first N layers on the build plate (adhesion).
//! 2. **Bulk**: sparse infill filling the height between base and interface.
//! 3. **Interface**: dense layers directly under the overhang (surface quality).
//!
//! Volume for each contribution = `overhang_area × slab_height × density`
//! where `density` is the effective infill ratio (0.0–1.0).
//!
//! # References
//!
//! - Bambu Studio "Support Manual" (recommended densities & interface layers).
//! - PrusaSlicer 2.7 documentation §Support Material.
//! - Ultimaker Cura material profiles (default 15% support infill).
//! - "Additive Manufacturing Technologies" 3rd ed., Gibson/Rosen/Stucker
//!   (chapter on support strategies).

use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Configuration
// ============================================================================

/// Support-generation configuration.
#[derive(Clone, Copy, Debug)]
pub struct SupportConfig {
    /// Bulk infill density (0.0–1.0). Cura default 0.15, Bambu 0.10-0.20.
    pub bulk_density: Fix128,
    /// Density of the dense interface layers (0.0–1.0). Bambu default 1.0.
    pub interface_density: Fix128,
    /// Density of the base (first-layer) region. Default 0.9 = near solid.
    pub base_density: Fix128,
    /// Number of interface layers just below the overhang face.
    pub interface_layers: u32,
    /// Number of dense base layers on the build plate.
    pub base_layers: u32,
    /// Nominal layer height (mm). Bambu default 0.20; Prusa 0.20.
    pub layer_height_mm: Fix128,
    /// Filament diameter (mm). 1.75 for Bambu/Prusa, 2.85 for Ultimaker legacy.
    pub filament_diameter_mm: Fix128,
    /// Extrusion throughput used for coarse time estimation (mm³ per minute).
    ///
    /// A typical Bambu X1C prints at ~24 mm³/s = 1440 mm³/min. Adjust down for
    /// slow / high-quality prints (e.g. 600 for detailed 0.1mm layer).
    pub throughput_mm3_per_min: Fix128,
}

impl Default for SupportConfig {
    fn default() -> Self {
        Self {
            bulk_density: Fix128::from_ratio(15, 100), // 15%
            interface_density: Fix128::ONE,
            base_density: Fix128::from_ratio(9, 10),
            interface_layers: 2,
            base_layers: 3,
            layer_height_mm: Fix128::from_ratio(2, 10), // 0.2mm
            filament_diameter_mm: Fix128::from_ratio(175, 100), // 1.75mm
            throughput_mm3_per_min: Fix128::from_int(1440), // Bambu X1C typical
        }
    }
}

impl SupportConfig {
    /// Preset: quality-focused (lower bulk density, more interface layers).
    #[must_use]
    pub fn quality() -> Self {
        Self {
            bulk_density: Fix128::from_ratio(10, 100),
            interface_layers: 4,
            layer_height_mm: Fix128::from_ratio(1, 10),
            throughput_mm3_per_min: Fix128::from_int(600),
            ..Self::default()
        }
    }

    /// Preset: speed-focused (thicker layers, fewer interface layers).
    #[must_use]
    pub fn speed() -> Self {
        Self {
            bulk_density: Fix128::from_ratio(20, 100),
            interface_layers: 1,
            layer_height_mm: Fix128::from_ratio(28, 100), // 0.28mm
            throughput_mm3_per_min: Fix128::from_int(2400),
            ..Self::default()
        }
    }
}

// ============================================================================
// Overhang region input
// ============================================================================

/// One overhang region requiring support. Typically produced by an upstream
/// overhang detector (angle > 45° from horizontal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverhangRegion {
    /// Projected area on the build plate (mm²) that must be supported.
    pub projected_area_mm2: Fix128,
    /// Support height (mm): distance from build plate up to the overhang face.
    pub support_height_mm: Fix128,
}

// ============================================================================
// Report
// ============================================================================

/// Aggregate volume / time / filament report.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SupportVolumeReport {
    /// Volume of dense first-layer(s) contribution (mm³).
    pub base_volume_mm3: Fix128,
    /// Volume of dense interface layer(s) immediately below overhang (mm³).
    pub interface_volume_mm3: Fix128,
    /// Volume of sparse bulk infill between base and interface (mm³).
    pub bulk_volume_mm3: Fix128,
    /// Sum of the above (mm³).
    pub total_volume_mm3: Fix128,
    /// Approximate filament length required (mm), using
    /// `length = volume / (π·(d/2)²)`.
    pub filament_length_mm: Fix128,
    /// Approximate print time for the support structure alone (minutes).
    pub estimated_time_min: Fix128,
}

impl SupportVolumeReport {
    /// Filament length in metres (convenience helper for spool inventory).
    #[inline]
    #[must_use]
    pub fn filament_length_m(&self) -> Fix128 {
        self.filament_length_mm / Fix128::from_int(1000)
    }

    /// Whether the estimate is non-trivial (any positive volume).
    #[inline]
    #[must_use]
    pub fn is_nontrivial(&self) -> bool {
        !self.total_volume_mm3.is_zero()
    }
}

// ============================================================================
// Estimation
// ============================================================================

/// Volume of one overhang region's support column (mm³).
///
/// Computes:
/// - `base = area × (base_layers · layer_h) · base_density`
/// - `interface = area × (interface_layers · layer_h) · interface_density`
/// - `bulk = area × max(0, support_height − base_thickness − interface_thickness)
///           × bulk_density`
///
/// Returns `(base, interface, bulk)` in mm³.
#[must_use]
pub fn estimate_region_volume(
    overhang: &OverhangRegion,
    config: &SupportConfig,
) -> (Fix128, Fix128, Fix128) {
    let base_thickness = config.layer_height_mm * Fix128::from_int(config.base_layers as i64);
    let interface_thickness =
        config.layer_height_mm * Fix128::from_int(config.interface_layers as i64);

    let base_volume = overhang.projected_area_mm2 * base_thickness * config.base_density;
    let interface_volume =
        overhang.projected_area_mm2 * interface_thickness * config.interface_density;

    let dense_thickness = base_thickness + interface_thickness;
    let bulk_height = if overhang.support_height_mm > dense_thickness {
        overhang.support_height_mm - dense_thickness
    } else {
        Fix128::ZERO
    };
    let bulk_volume = overhang.projected_area_mm2 * bulk_height * config.bulk_density;

    (base_volume, interface_volume, bulk_volume)
}

/// Aggregate estimate across all overhang regions.
#[must_use]
pub fn estimate_support_volume(
    overhangs: &[OverhangRegion],
    config: &SupportConfig,
) -> SupportVolumeReport {
    let mut base = Fix128::ZERO;
    let mut interface = Fix128::ZERO;
    let mut bulk = Fix128::ZERO;

    for region in overhangs {
        let (b, i, bk) = estimate_region_volume(region, config);
        base = base + b;
        interface = interface + i;
        bulk = bulk + bk;
    }

    let total = base + interface + bulk;

    // Filament length: L = V / A_filament where A = π (d/2)²
    let radius = config.filament_diameter_mm.half();
    let area = Fix128::PI * radius * radius;
    let filament_length = if area.is_zero() {
        Fix128::ZERO
    } else {
        total / area
    };

    // Time = volume / throughput
    let time = if config.throughput_mm3_per_min.is_zero() {
        Fix128::ZERO
    } else {
        total / config.throughput_mm3_per_min
    };

    SupportVolumeReport {
        base_volume_mm3: base,
        interface_volume_mm3: interface,
        bulk_volume_mm3: bulk,
        total_volume_mm3: total,
        filament_length_mm: filament_length,
        estimated_time_min: time,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn default_config_uses_industry_values() {
        let cfg = SupportConfig::default();
        assert_eq!(cfg.bulk_density, Fix128::from_ratio(15, 100));
        assert_eq!(cfg.interface_density, Fix128::ONE);
        assert_eq!(cfg.interface_layers, 2);
        assert_eq!(cfg.base_layers, 3);
        assert_eq!(cfg.layer_height_mm, Fix128::from_ratio(2, 10));
        assert_eq!(cfg.filament_diameter_mm, Fix128::from_ratio(175, 100));
    }

    #[test]
    fn quality_preset_denser_and_slower() {
        let q = SupportConfig::quality();
        let d = SupportConfig::default();
        assert!(q.bulk_density < d.bulk_density);
        assert!(q.interface_layers > d.interface_layers);
        assert!(q.throughput_mm3_per_min < d.throughput_mm3_per_min);
    }

    #[test]
    fn speed_preset_faster_and_thicker() {
        let s = SupportConfig::speed();
        let d = SupportConfig::default();
        assert!(s.layer_height_mm > d.layer_height_mm);
        assert!(s.throughput_mm3_per_min > d.throughput_mm3_per_min);
    }

    #[test]
    fn single_region_volume_decomposition() {
        // 100mm² overhang, 20mm high, default config:
        // base_thick = 3 · 0.2 = 0.6mm; interface_thick = 2 · 0.2 = 0.4mm
        // dense = 1.0mm; bulk_height = 19mm
        // base = 100 · 0.6 · 0.9 = 54 mm³
        // interface = 100 · 0.4 · 1.0 = 40 mm³
        // bulk = 100 · 19 · 0.15 = 285 mm³
        let cfg = SupportConfig::default();
        let region = OverhangRegion {
            projected_area_mm2: Fix128::from_int(100),
            support_height_mm: Fix128::from_int(20),
        };
        let (b, i, bk) = estimate_region_volume(&region, &cfg);
        assert!(approx_eq(b, Fix128::from_int(54), Fix128::ONE));
        assert!(approx_eq(i, Fix128::from_int(40), Fix128::ONE));
        assert!(approx_eq(bk, Fix128::from_int(285), Fix128::ONE));
    }

    #[test]
    fn short_support_has_no_bulk() {
        // 100mm² overhang, only 0.5mm high — shorter than base+interface (1.0)
        let cfg = SupportConfig::default();
        let region = OverhangRegion {
            projected_area_mm2: Fix128::from_int(100),
            support_height_mm: Fix128::from_ratio(5, 10),
        };
        let (_, _, bulk) = estimate_region_volume(&region, &cfg);
        assert_eq!(bulk, Fix128::ZERO);
    }

    #[test]
    fn empty_input_gives_zero_report() {
        let report = estimate_support_volume(&[], &SupportConfig::default());
        assert_eq!(report.total_volume_mm3, Fix128::ZERO);
        assert_eq!(report.filament_length_mm, Fix128::ZERO);
        assert_eq!(report.estimated_time_min, Fix128::ZERO);
        assert!(!report.is_nontrivial());
    }

    #[test]
    fn aggregate_multiple_regions() {
        let cfg = SupportConfig::default();
        let regions = vec![
            OverhangRegion {
                projected_area_mm2: Fix128::from_int(50),
                support_height_mm: Fix128::from_int(10),
            },
            OverhangRegion {
                projected_area_mm2: Fix128::from_int(80),
                support_height_mm: Fix128::from_int(15),
            },
        ];
        let report = estimate_support_volume(&regions, &cfg);
        assert!(report.total_volume_mm3 > Fix128::ZERO);
        assert!(report.is_nontrivial());
        // Sum should equal individual sums
        let (b0, i0, bk0) = estimate_region_volume(&regions[0], &cfg);
        let (b1, i1, bk1) = estimate_region_volume(&regions[1], &cfg);
        assert_eq!(report.base_volume_mm3, b0 + b1);
        assert_eq!(report.interface_volume_mm3, i0 + i1);
        assert_eq!(report.bulk_volume_mm3, bk0 + bk1);
    }

    #[test]
    fn filament_length_conversion() {
        // 100mm² overhang, 100mm high, default cfg:
        // Total volume ≈ 54 + 40 + (100 · 99 · 0.15) = 94 + 1485 = 1579 mm³
        // Filament A = π · (0.875)² ≈ 2.405 mm²
        // Length = 1579 / 2.405 ≈ 656.5 mm
        let cfg = SupportConfig::default();
        let region = OverhangRegion {
            projected_area_mm2: Fix128::from_int(100),
            support_height_mm: Fix128::from_int(100),
        };
        let report = estimate_support_volume(&[region], &cfg);
        let expected_length = Fix128::from_int(656);
        assert!(
            approx_eq(
                report.filament_length_mm,
                expected_length,
                Fix128::from_int(5)
            ),
            "got {} mm, expected ~656",
            report.filament_length_mm.to_f32()
        );
    }

    #[test]
    fn filament_length_meters_scale() {
        let cfg = SupportConfig::default();
        let region = OverhangRegion {
            projected_area_mm2: Fix128::from_int(100),
            support_height_mm: Fix128::from_int(100),
        };
        let report = estimate_support_volume(&[region], &cfg);
        // ~656mm = ~0.656m
        let meters = report.filament_length_m();
        assert!(meters < Fix128::ONE);
        assert!(meters > Fix128::from_ratio(5, 10));
    }

    #[test]
    fn estimated_time_scales_with_throughput() {
        let region = OverhangRegion {
            projected_area_mm2: Fix128::from_int(200),
            support_height_mm: Fix128::from_int(50),
        };
        let fast = SupportConfig::default();
        let quality = SupportConfig::quality();
        let fast_report = estimate_support_volume(&[region], &fast);
        let quality_report = estimate_support_volume(&[region], &quality);
        // Quality preset has slower throughput → longer time
        assert!(quality_report.estimated_time_min > fast_report.estimated_time_min);
    }

    #[test]
    fn zero_area_gives_zero_volume() {
        let cfg = SupportConfig::default();
        let region = OverhangRegion {
            projected_area_mm2: Fix128::ZERO,
            support_height_mm: Fix128::from_int(50),
        };
        let (b, i, bk) = estimate_region_volume(&region, &cfg);
        assert_eq!(b, Fix128::ZERO);
        assert_eq!(i, Fix128::ZERO);
        assert_eq!(bk, Fix128::ZERO);
    }

    #[test]
    fn base_and_interface_thickness_bounded_by_layers() {
        // With base=3, interface=2, layer=0.2 → dense_total = 1.0mm
        // Confirms exactly 5 layers of dense material get accounted for.
        // Note: 0.2mm as Fix128 is not exactly representable so the
        // dense_total computed from ratios differs from 1.0 by ~ULP; bulk
        // volume will be a sub-mm³ residue, treated as effectively zero here.
        let cfg = SupportConfig::default();
        let region = OverhangRegion {
            projected_area_mm2: Fix128::from_int(100),
            support_height_mm: Fix128::ONE,
        };
        let (b, i, bk) = estimate_region_volume(&region, &cfg);
        // Bulk should be numerically negligible (< 1 mm³)
        assert!(bk < Fix128::from_ratio(1, 100));
        // base = 100 · 0.6 · 0.9 = 54
        assert!(approx_eq(
            b,
            Fix128::from_int(54),
            Fix128::from_ratio(1, 10)
        ));
        // interface = 100 · 0.4 · 1.0 = 40
        assert!(approx_eq(
            i,
            Fix128::from_int(40),
            Fix128::from_ratio(1, 10)
        ));
    }

    #[test]
    fn nontrivial_flag() {
        let empty = SupportVolumeReport::default();
        assert!(!empty.is_nontrivial());
        let nonzero = SupportVolumeReport {
            total_volume_mm3: Fix128::from_int(100),
            ..Default::default()
        };
        assert!(nonzero.is_nontrivial());
    }
}
