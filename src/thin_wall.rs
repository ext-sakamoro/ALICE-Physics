//! Thin-Wall Detection for 3D Print Safety
//!
//! Phase A2 of the ALICE-Physics completeness project. Detects regions where
//! the local wall thickness falls below a manufacturability threshold (e.g.
//! `nozzle_diameter × 2 = 0.8mm` for a 0.4mm nozzle). Uses SDF-based sphere
//! marching along the inward surface normal to measure "opposite surface
//! distance" — the classic wall-thickness definition in 3D printing.
//!
//! # Algorithm
//!
//! For each surface sample point:
//!
//! 1. Read outward normal from the SDF gradient.
//! 2. Step slightly inward (past the surface) to enter the negative-SDF region.
//! 3. Sphere-march along the inward normal, taking steps of `|SDF|` (the exact
//!    distance to the nearest surface, Lipschitz-safe by definition of SDF).
//! 4. Stop when the SDF becomes non-negative — the ray has exited through the
//!    opposite surface. Total distance travelled = local wall thickness.
//!
//! # Determinism note
//!
//! [`SdfField`] returns `f32` (SDF evaluation is inherently floating-point);
//! measured thicknesses are therefore not bit-exact across platforms. The
//! iteration structure (max_iterations, epsilon cutoffs) IS deterministic —
//! results agree to `~epsilon` across x86 / ARM / WASM.

use crate::math::{Fix128, Vec3Fix};

#[cfg(feature = "std")]
use crate::sdf_collider::SdfField;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for thin-wall detection.
#[derive(Clone, Copy, Debug)]
pub struct ThinWallConfig {
    /// Report walls thinner than this as `regions` (mm).
    ///
    /// FDM rule of thumb: `nozzle_diameter × 2` for reliable extrusion.
    /// For a 0.4mm nozzle → 0.8mm minimum wall.
    pub min_thickness_mm: Fix128,
    /// Stop ray marching after this cumulative distance (mm). Prevents infinite
    /// loops for open geometry (unbounded march) or precision starvation.
    pub max_march_distance_mm: Fix128,
    /// Safety cap on the sphere-marching iteration count.
    pub max_iterations: u32,
    /// Small offset (mm) applied along the inward normal at start of march to
    /// escape numerical noise at the surface (SDF ≈ 0).
    pub start_offset_mm: f32,
    /// Minimum step size (mm) — protects against slow convergence for near-
    /// parallel ray/surface configurations.
    pub min_step_mm: f32,
}

impl Default for ThinWallConfig {
    fn default() -> Self {
        Self {
            // 0.8mm = nozzle 0.4 × 2 (FDM industry default)
            min_thickness_mm: Fix128::from_ratio(8, 10),
            // 50mm covers typical print bed features; adjust for larger parts.
            max_march_distance_mm: Fix128::from_int(50),
            max_iterations: 128,
            start_offset_mm: 0.01,
            min_step_mm: 0.001,
        }
    }
}

impl ThinWallConfig {
    /// Construct config for a given nozzle diameter (mm).
    ///
    /// Sets `min_thickness_mm = nozzle_mm × 2` per FDM manufacturability
    /// guidance from Bambu Lab / Prusa knowledge bases.
    #[must_use]
    pub fn for_nozzle(nozzle_mm: Fix128) -> Self {
        Self {
            min_thickness_mm: nozzle_mm.double(),
            ..Self::default()
        }
    }
}

// ============================================================================
// Reports
// ============================================================================

/// A single sample point flagged as too thin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThinRegion {
    /// World-space surface point where thickness was measured.
    pub position: Vec3Fix,
    /// Outward surface normal at the sample point (from SDF gradient).
    pub outward_normal: (f32, f32, f32),
    /// Measured wall thickness at this point (mm).
    pub thickness_mm: Fix128,
}

/// Result of a thin-wall analysis pass.
#[derive(Clone, Debug, Default)]
pub struct ThinWallReport {
    /// Sample points whose thickness fell below `config.min_thickness_mm`.
    pub regions: Vec<ThinRegion>,
    /// Total number of surface samples processed.
    pub sampled_count: usize,
    /// Samples for which the opposite surface was not found within
    /// `max_march_distance_mm` (open geometry or too-large parts).
    pub unbounded_count: usize,
    /// Minimum thickness observed across all successful samples (mm).
    pub min_thickness_seen: Fix128,
    /// Maximum thickness observed across all successful samples (mm).
    pub max_thickness_seen: Fix128,
}

impl ThinWallReport {
    /// Whether any thin regions were detected (i.e. print is unsafe as-is).
    #[inline]
    #[must_use]
    pub fn has_thin_walls(&self) -> bool {
        !self.regions.is_empty()
    }

    /// Fraction of sampled points that failed thickness check.
    /// Returns 0 if `sampled_count == 0`.
    #[must_use]
    pub fn thin_fraction(&self) -> Fix128 {
        if self.sampled_count == 0 {
            return Fix128::ZERO;
        }
        Fix128::from_ratio(self.regions.len() as i64, self.sampled_count as i64)
    }
}

// ============================================================================
// Core measurement
// ============================================================================

/// Measure the wall thickness at a single surface point via sphere marching.
///
/// Returns `None` if the ray does not hit an opposite surface within
/// `config.max_march_distance_mm` or if `config.max_iterations` was exhausted.
#[cfg(feature = "std")]
#[must_use]
pub fn measure_thickness_at(
    sdf: &dyn SdfField,
    surface_point: Vec3Fix,
    outward_normal: (f32, f32, f32),
    config: &ThinWallConfig,
) -> Option<Fix128> {
    // Inward normal = -outward
    let inward = (-outward_normal.0, -outward_normal.1, -outward_normal.2);

    // Verify normal is unit-ish; degenerate inputs cannot be marched.
    let n_len_sq = inward.0 * inward.0 + inward.1 * inward.1 + inward.2 * inward.2;
    if n_len_sq < 0.5 || n_len_sq > 1.5 {
        return None;
    }

    // Convert to f32 for hot loop (SdfField is f32-native).
    let mut px = surface_point.x.to_f32();
    let mut py = surface_point.y.to_f32();
    let mut pz = surface_point.z.to_f32();

    // Step just inside the surface to leave the SDF ≈ 0 noise band.
    px += inward.0 * config.start_offset_mm;
    py += inward.1 * config.start_offset_mm;
    pz += inward.2 * config.start_offset_mm;
    let mut distance = config.start_offset_mm;

    let max_dist_f32 = config.max_march_distance_mm.to_f32();

    for _ in 0..config.max_iterations {
        let d = sdf.distance(px, py, pz);
        if d >= 0.0 {
            // Exited through the opposite surface.
            return Some(Fix128::from_f32(distance));
        }
        // Inside: |SDF| is exact distance to nearest surface, safe to advance.
        let mut step = -d;
        if step < config.min_step_mm {
            step = config.min_step_mm;
        }
        px += inward.0 * step;
        py += inward.1 * step;
        pz += inward.2 * step;
        distance += step;
        if distance > max_dist_f32 {
            return None;
        }
    }
    None
}

// ============================================================================
// Batch analysis
// ============================================================================

/// Analyse thickness for an explicit set of surface sample points.
///
/// Callers typically obtain surface points from a marching-cubes mesh, from
/// on-surface SDF root finding, or from a Poisson-disk surface sampling pass.
/// For a bounding-box grid sweep use [`analyze_thickness_grid`].
#[cfg(feature = "std")]
#[must_use]
pub fn analyze_thickness(
    sdf: &dyn SdfField,
    surface_points: &[Vec3Fix],
    config: &ThinWallConfig,
) -> ThinWallReport {
    let mut report = ThinWallReport {
        sampled_count: surface_points.len(),
        min_thickness_seen: Fix128::from_int(i64::MAX >> 1),
        ..Default::default()
    };

    for &p in surface_points {
        let normal = sdf.normal(p.x.to_f32(), p.y.to_f32(), p.z.to_f32());
        match measure_thickness_at(sdf, p, normal, config) {
            Some(thickness) => {
                if thickness < report.min_thickness_seen {
                    report.min_thickness_seen = thickness;
                }
                if thickness > report.max_thickness_seen {
                    report.max_thickness_seen = thickness;
                }
                if thickness < config.min_thickness_mm {
                    report.regions.push(ThinRegion {
                        position: p,
                        outward_normal: normal,
                        thickness_mm: thickness,
                    });
                }
            }
            None => report.unbounded_count += 1,
        }
    }

    // If nothing successful was sampled, reset min to zero for a clean report.
    if report.sampled_count == report.unbounded_count {
        report.min_thickness_seen = Fix128::ZERO;
    }

    report
}

/// Sample the SDF on a regular grid inside `[aabb_min, aabb_max]`, extract
/// approximate surface points where the SDF sign changes between adjacent
/// grid cells, then analyse thickness at each.
///
/// This is a convenience wrapper for callers without a mesh; for meshed
/// geometry use [`analyze_thickness`] directly on the vertex positions.
///
/// `grid_step_mm` controls sampling density — smaller is more thorough but
/// scales O(n³). A reasonable default is `min_thickness_mm × 0.5`.
#[cfg(feature = "std")]
#[must_use]
pub fn analyze_thickness_grid(
    sdf: &dyn SdfField,
    aabb_min: Vec3Fix,
    aabb_max: Vec3Fix,
    grid_step_mm: Fix128,
    config: &ThinWallConfig,
) -> ThinWallReport {
    let surface_points = sample_surface_points(sdf, aabb_min, aabb_max, grid_step_mm);
    analyze_thickness(sdf, &surface_points, config)
}

/// Extract approximate surface points from a grid AABB by finding sign changes
/// between adjacent cells along the X axis. Linear interpolation refines the
/// zero-crossing location. Exposed for tests and advanced callers.
#[cfg(feature = "std")]
#[must_use]
pub fn sample_surface_points(
    sdf: &dyn SdfField,
    aabb_min: Vec3Fix,
    aabb_max: Vec3Fix,
    grid_step_mm: Fix128,
) -> Vec<Vec3Fix> {
    let step = grid_step_mm.to_f32();
    if step <= 0.0 {
        return Vec::new();
    }

    let xmin = aabb_min.x.to_f32();
    let ymin = aabb_min.y.to_f32();
    let zmin = aabb_min.z.to_f32();
    let xmax = aabb_max.x.to_f32();
    let ymax = aabb_max.y.to_f32();
    let zmax = aabb_max.z.to_f32();

    let mut out = Vec::new();

    let mut y = ymin;
    while y <= ymax {
        let mut z = zmin;
        while z <= zmax {
            let mut x = xmin;
            let mut prev_d = sdf.distance(x, y, z);
            x += step;
            while x <= xmax {
                let d = sdf.distance(x, y, z);
                if (prev_d < 0.0 && d >= 0.0) || (prev_d >= 0.0 && d < 0.0) {
                    // Sign change: linear interp on the segment [x-step, x].
                    let denom = d - prev_d;
                    if denom.abs() > f32::EPSILON {
                        let t = -prev_d / denom; // in [0, 1]
                        let sx = (x - step) + t * step;
                        out.push(Vec3Fix::new(
                            Fix128::from_f32(sx),
                            Fix128::from_f32(y),
                            Fix128::from_f32(z),
                        ));
                    }
                }
                prev_d = d;
                x += step;
            }
            z += step;
        }
        y += step;
    }

    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    /// Sphere centred at origin, radius R (in mm).
    fn sphere_sdf(radius: f32) -> ClosureSdf {
        ClosureSdf::new(
            move |x, y, z| (x * x + y * y + z * z).sqrt() - radius,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt().max(f32::EPSILON);
                (x / len, y / len, z / len)
            },
        )
    }

    /// Hollow sphere: shell between R_out and R_in (SDF = max(inner_neg, outer_pos)).
    /// Wall thickness anywhere on outer surface = R_out - R_in.
    fn hollow_sphere_sdf(r_out: f32, r_in: f32) -> ClosureSdf {
        ClosureSdf::new(
            move |x, y, z| {
                let r = (x * x + y * y + z * z).sqrt();
                let outer = r - r_out; // negative inside R_out
                let inner = r_in - r; // negative outside R_in (so positive inside cavity)
                outer.max(inner)
            },
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt().max(f32::EPSILON);
                (x / len, y / len, z / len)
            },
        )
    }

    #[test]
    fn config_default_uses_industry_thresholds() {
        let cfg = ThinWallConfig::default();
        // 0.8mm = 0.4mm nozzle × 2
        assert_eq!(cfg.min_thickness_mm, Fix128::from_ratio(8, 10));
        assert!(cfg.max_iterations >= 32);
    }

    #[test]
    fn config_for_nozzle_scales() {
        let cfg = ThinWallConfig::for_nozzle(Fix128::from_ratio(6, 10));
        // 0.6mm nozzle → 1.2mm min wall. `.double()` and `from_ratio(12, 10)`
        // agree to within one ULP so compare with a small tolerance.
        let expected = Fix128::from_ratio(12, 10);
        let diff = if cfg.min_thickness_mm > expected {
            cfg.min_thickness_mm - expected
        } else {
            expected - cfg.min_thickness_mm
        };
        assert!(diff <= Fix128::from_ratio(1, 1_000_000));
    }

    #[test]
    fn thickness_solid_sphere_equals_diameter() {
        // Solid sphere R=10 → measuring inward from surface, opposite surface
        // is at ~20mm (diameter). Sphere marching should find it easily.
        let sdf = sphere_sdf(10.0);
        let cfg = ThinWallConfig::default();
        let p = Vec3Fix::new(
            Fix128::from_int(10), // on +X surface
            Fix128::ZERO,
            Fix128::ZERO,
        );
        let normal = (1.0, 0.0, 0.0);
        let t = measure_thickness_at(&sdf, p, normal, &cfg).expect("must hit opposite");
        // Expect ~20mm, allow generous tolerance from sphere-tracing convergence
        let diff = (t.to_f32() - 20.0).abs();
        assert!(diff < 0.5, "diameter ≈ 20mm, got {}", t.to_f32());
    }

    #[test]
    fn thickness_hollow_shell_matches_wall() {
        // Shell R_out=10, R_in=8 → wall = 2mm
        let sdf = hollow_sphere_sdf(10.0, 8.0);
        let cfg = ThinWallConfig::default();
        let p = Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO);
        let normal = (1.0, 0.0, 0.0);
        let t = measure_thickness_at(&sdf, p, normal, &cfg).expect("must hit inner wall");
        let diff = (t.to_f32() - 2.0).abs();
        assert!(diff < 0.2, "wall ≈ 2mm, got {}", t.to_f32());
    }

    #[test]
    fn thickness_below_threshold_is_flagged() {
        // Shell wall = 0.5mm; default threshold 0.8mm → should flag
        let sdf = hollow_sphere_sdf(5.0, 4.5);
        let cfg = ThinWallConfig::default();
        let p = Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO);
        let normal = (1.0, 0.0, 0.0);
        let t = measure_thickness_at(&sdf, p, normal, &cfg).unwrap();
        assert!(t < cfg.min_thickness_mm, "0.5mm < 0.8mm threshold");
    }

    #[test]
    fn measure_returns_none_on_degenerate_normal() {
        let sdf = sphere_sdf(10.0);
        let cfg = ThinWallConfig::default();
        let p = Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO);
        // Zero-length normal
        let t = measure_thickness_at(&sdf, p, (0.0, 0.0, 0.0), &cfg);
        assert!(t.is_none());
    }

    #[test]
    fn measure_returns_none_when_open_geometry() {
        // Half-space plane at x=0, inside is x < 0 → marching from surface
        // point (0,0,0) inward (-X) never re-enters positive SDF.
        let sdf = ClosureSdf::new(|x, _y, _z| x, |_, _, _| (1.0, 0.0, 0.0));
        let cfg = ThinWallConfig {
            max_march_distance_mm: Fix128::from_int(5),
            ..ThinWallConfig::default()
        };
        let p = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        let normal = (1.0, 0.0, 0.0);
        let t = measure_thickness_at(&sdf, p, normal, &cfg);
        assert!(t.is_none(), "open half-space should be unbounded");
    }

    #[test]
    fn batch_analyze_partitions_regions() {
        // Solid sphere R=10 with 8 sample points on the ±X, ±Y, ±Z surface.
        // All thicknesses are ~20mm → nothing should be flagged as thin.
        let sdf = sphere_sdf(10.0);
        let cfg = ThinWallConfig::default();
        let pts: Vec<Vec3Fix> = [
            (10, 0, 0),
            (-10, 0, 0),
            (0, 10, 0),
            (0, -10, 0),
            (0, 0, 10),
            (0, 0, -10),
        ]
        .into_iter()
        .map(|(x, y, z)| {
            Vec3Fix::new(
                Fix128::from_int(x),
                Fix128::from_int(y),
                Fix128::from_int(z),
            )
        })
        .collect();
        let report = analyze_thickness(&sdf, &pts, &cfg);
        assert_eq!(report.sampled_count, 6);
        assert_eq!(report.regions.len(), 0);
        assert!(!report.has_thin_walls());
    }

    #[test]
    fn batch_analyze_flags_thin_shell() {
        // Thin shell wall = 0.3mm → all 6 sample points should be flagged.
        let sdf = hollow_sphere_sdf(5.0, 4.7);
        let cfg = ThinWallConfig::default();
        let pts: Vec<Vec3Fix> = [
            (5, 0, 0),
            (-5, 0, 0),
            (0, 5, 0),
            (0, -5, 0),
            (0, 0, 5),
            (0, 0, -5),
        ]
        .into_iter()
        .map(|(x, y, z)| {
            Vec3Fix::new(
                Fix128::from_int(x),
                Fix128::from_int(y),
                Fix128::from_int(z),
            )
        })
        .collect();
        let report = analyze_thickness(&sdf, &pts, &cfg);
        assert_eq!(report.sampled_count, 6);
        assert_eq!(report.regions.len(), 6, "all 6 shell samples should flag");
        assert!(report.has_thin_walls());
        assert_eq!(report.thin_fraction(), Fix128::ONE);
    }

    #[test]
    fn thin_fraction_is_zero_when_no_samples() {
        let report = ThinWallReport::default();
        assert_eq!(report.thin_fraction(), Fix128::ZERO);
    }

    #[test]
    fn thin_fraction_half() {
        let mut report = ThinWallReport {
            sampled_count: 4,
            ..Default::default()
        };
        report.regions.push(ThinRegion {
            position: Vec3Fix::default(),
            outward_normal: (1.0, 0.0, 0.0),
            thickness_mm: Fix128::from_ratio(3, 10),
        });
        report.regions.push(ThinRegion {
            position: Vec3Fix::default(),
            outward_normal: (0.0, 1.0, 0.0),
            thickness_mm: Fix128::from_ratio(4, 10),
        });
        // 2/4 = 0.5
        assert_eq!(report.thin_fraction(), Fix128::from_ratio(1, 2));
    }

    #[test]
    fn sample_surface_points_hits_sphere() {
        let sdf = sphere_sdf(5.0);
        let pts = sample_surface_points(
            &sdf,
            Vec3Fix::new(
                Fix128::from_int(-7),
                Fix128::from_int(-7),
                Fix128::from_int(-7),
            ),
            Vec3Fix::new(
                Fix128::from_int(7),
                Fix128::from_int(7),
                Fix128::from_int(7),
            ),
            Fix128::from_int(1),
        );
        // Should find many surface crossings; each point should lie on |p|≈5
        assert!(!pts.is_empty());
        for p in &pts {
            let r = (p.x.to_f32().powi(2) + p.y.to_f32().powi(2) + p.z.to_f32().powi(2)).sqrt();
            assert!(
                (r - 5.0).abs() < 1.5,
                "surface point radius {} deviates from 5",
                r
            );
        }
    }

    #[test]
    fn grid_analyze_solid_sphere_finds_no_thin_regions() {
        let sdf = sphere_sdf(10.0);
        let cfg = ThinWallConfig::default();
        let report = analyze_thickness_grid(
            &sdf,
            Vec3Fix::new(
                Fix128::from_int(-12),
                Fix128::from_int(-12),
                Fix128::from_int(-12),
            ),
            Vec3Fix::new(
                Fix128::from_int(12),
                Fix128::from_int(12),
                Fix128::from_int(12),
            ),
            Fix128::from_int(4),
            &cfg,
        );
        assert!(report.sampled_count > 0);
        // Solid sphere → walls always thick → no flagged regions
        assert_eq!(report.regions.len(), 0);
    }
}
