//! Oracles for the production entry points of `alice_physics::thin_wall`,
//! driven by `examples/thin_wall_detection.rs`: `ThinWallConfig::for_nozzle`,
//! `measure_thickness_at`, `analyze_thickness`, `analyze_thickness_grid`,
//! `sample_surface_points`, `ThinWallReport::has_thin_walls` and
//! `ThinWallReport::thin_fraction`. Before the example existed none of the
//! seven had a caller outside the module's own `#[cfg(test)]` block —
//! `scripts/wiring_guard.py` reported all seven as `unwired`.
//!
//! # Closed forms and where they come from
//!
//! * **`for_nozzle`** (module doc): `min_thickness_mm = nozzle_mm × 2`. Checked
//!   against an independently-constructed `Fix128::from_ratio`, not against
//!   the function's own output.
//! * **`measure_thickness_at`** on a "stepped slab": the field
//!   `SDF(x, y, z) = |z| − h(x)` with `h(x) = 0.3` for `x < 0` ("thin side")
//!   and `h(x) = 0.5` for `x ≥ 0` ("thick side"). For any single query the
//!   sphere-march only ever moves along `z` (the probe's `outward_normal` is
//!   always `(0, 0, 1)`, so `inward = (0, 0, -1)` and `px`/`py` never change
//!   inside the loop), so each query sees a *constant*-`h` 1-D slab for its
//!   own `x`. A 1-D slab of half-thickness `h` has its opposite face exactly
//!   `2h` away by construction (the two parallel faces are at `z = ±h`) —
//!   this is the textbook "wall thickness = distance to the opposite
//!   surface" definition from the module's own top-of-file doc, computed by
//!   hand, not by calling `measure_thickness_at`.
//! * **`analyze_thickness`**: six hand-built points, three per side of the
//!   stepped slab above (`x ∈ {-3,-2,-1}` thin, `x ∈ {1,2,3}` thick). The
//!   expected partition (3 thin / 3 thick) is fixed by construction, not by
//!   calling the function — `thin_fraction = 3/6 = 1/2` exactly (`Fix128` is
//!   base-2 fixed point, and `(3 << 64) / 6 == (1 << 64) / 2 == 2^63` with no
//!   remainder, so the two ratios are bit-identical; see
//!   `rules/analytic-oracle-tests.md` "期待値は実装を呼ばずに...手計算").
//! * **`sample_surface_points`**: a sphere of radius 5mm scanned on a coarse
//!   `step = 6mm` grid over `[-6, 6]³`. The grid visits only `y, z ∈ {-6, 0,
//!   6}` (9 combinations); `y² + z² < 25` only for `(y, z) = (0, 0)` (every
//!   other combination has `y² + z² ∈ {36, 72}` which is `> 25`, so the
//!   scan's `x`-line never enters the sphere and no sign change is possible —
//!   enumerated by hand for all 9 pairs below, not by running the sampler).
//!   On the one line that does cross, `SDF(x, 0, 0) = |x| − 5` is *exactly*
//!   piecewise-linear in `x` (not quadratic — the `y = z = 0` cross-section
//!   of a sphere SDF degenerates to `|x| − r`), so the function's own linear
//!   interpolation is exact, not approximate: at `x = -6` (`d = 1`) to `x = 0`
//!   (`d = -5`), `t = -1/(-5-1) = 1/6`, `sx = -6 + 1/6·6 = -5` exactly; at
//!   `x = 0` (`d = -5`) to `x = 6` (`d = 1`), `t = 5/6`, `sx = 0 + 5/6·6 = 5`
//!   exactly. Expected output: exactly 2 points, at `(-5, 0, 0)` and
//!   `(5, 0, 0)`.
//! * **`analyze_thickness_grid`**: the same sphere through the full
//!   grid→sample→analyze pipeline. Both surface points are diametrically
//!   opposite on a sphere of radius 5, so the opposite-face distance at each
//!   is the diameter, `10mm` — well above any FDM nozzle threshold, so
//!   `thin_fraction = 0` and `has_thin_walls() == false`.
//!
//! # Degenerate input
//!
//! * `measure_thickness_at` with a zero-length `outward_normal`: `None`
//!   (guarded by `n_len_sq` bounds check — the module's own `#[cfg(test)]`
//!   block already covers this; re-asserted here as a production-facing
//!   check with a mutation-sensitive guard).
//! * `sample_surface_points` with `grid_step_mm = 0`: returns an empty `Vec`
//!   immediately (explicit `if step <= 0.0 { return Vec::new(); }` in the
//!   module) — not a panic, not an infinite loop.
//! * `analyze_thickness_grid` / `sample_surface_points` with an *inverted*
//!   AABB (`aabb_min` component-wise greater than `aabb_max`): `axis_steps`
//!   returns `0` immediately for that axis — empty report, `sampled_count
//!   == 0`.
//! * `analyze_thickness` with an empty point slice: `sampled_count == 0`,
//!   `thin_fraction() == 0` (explicit early return for `sampled_count == 0`),
//!   `has_thin_walls() == false`.
//!
//! ## Three root-cause fixes (2026-10-03 — previously "current
//! behaviour, not changed, design decision out of scope")
//!
//! * `ThinWallConfig::for_nozzle(nozzle_mm)` with `nozzle_mm <= 0` used to
//!   silently produce `min_thickness_mm = 0`, disabling thin-wall detection
//!   entirely with no panic and no `Err`. It now **panics** (`for_nozzle_zero_diameter_now_fails_fast`
//!   below) — a non-positive nozzle diameter has no physical meaning.
//! * `measure_thickness_at` at extreme-magnitude coordinates (`Fix128::from_int(1_000_000_000)`)
//!   used to report `Some(start_offset_mm)` — the `f32` ULP there (~64)
//!   swallowed the 0.01mm inward start offset, so the first loop iteration
//!   queried the surface itself, not a point inside it, and reported a
//!   thickness unrelated to any real opposite-face distance. It now detects
//!   "did not actually move" (bit-identical coordinates before/after the
//!   offset or any march step) and reports `None` instead
//!   (`measure_thickness_at_extreme_coordinates_does_not_panic` below).
//! * `sample_surface_points`'s grid walk used to advance with `x += step`
//!   (plain `f32` addition) with no check that `step` was representable at
//!   the current magnitude — near `x ≈ 1e9` the `f32` ULP is 64, so a
//!   `grid_step_mm` smaller than that (e.g. `1.0`) made `x += step` a no-op
//!   and `while x <= xmax` never terminated (not a panic, `catch_unwind`
//!   cannot observe it; hit empirically authoring this file, a
//!   999_999_999..1_000_000_001 AABB with `step = 1` hung indefinitely and
//!   had to be killed). The grid walk is now driven by an integer step
//!   count per axis instead of repeated `f32` addition, so it terminates
//!   unconditionally (`sample_surface_points_step_smaller_than_ulp_terminates`
//!   below reproduces the exact AABB that used to hang).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::thin_wall::{
    analyze_thickness, analyze_thickness_grid, measure_thickness_at, sample_surface_points,
    ThinWallConfig, ThinWallReport,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Tolerance for comparisons that cross the sphere-march's accumulation of
/// several `f32` steps (the module's own tests use 0.2–0.5mm tolerances for
/// the same reason; our stepped-slab scene converges in ~8 doublings from a
/// 0.01mm start offset, so a tighter tolerance is safe).
fn approx_eq_mm(a: Fix128, b_mm: f64, tol_mm: f64) -> bool {
    (a.to_f64() - b_mm).abs() <= tol_mm
}

/// Stepped slab: `SDF(x, y, z) = |z| - h(x)`, `h(x) = 0.3` for `x < 0`
/// ("thin" side, thickness 0.6mm), `h(x) = 0.5` for `x >= 0` ("thick" side,
/// thickness 1.0mm). `normal` is the constant `(0, 0, 1)` — every query point
/// used against this scene is a top face point, so this is the correct
/// outward normal for all of them (not a general-purpose SDF normal).
fn stepped_slab_sdf() -> ClosureSdf {
    ClosureSdf::new(
        |x, _y, z| {
            let h: f32 = if x < 0.0 { 0.3 } else { 0.5 };
            z.abs() - h
        },
        |_x, _y, _z| (0.0, 0.0, 1.0),
    )
}

fn half_thickness_for_x(x: f32) -> f32 {
    if x < 0.0 {
        0.3
    } else {
        0.5
    }
}

/// Uniform thin slab: half-thickness 0.3mm everywhere (thickness 0.6mm,
/// below the default 0.8mm / 0.4mm-nozzle threshold at every point).
fn uniform_thin_slab_sdf() -> ClosureSdf {
    ClosureSdf::new(|_x, _y, z| z.abs() - 0.3, |_x, _y, _z| (0.0, 0.0, 1.0))
}

/// Sphere of radius `r`, centred at the origin.
fn sphere_sdf(r: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - r,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(f32::EPSILON);
            (x / len, y / len, z / len)
        },
    )
}

// ============================================================================
// for_nozzle
// ============================================================================

#[test]
fn for_nozzle_matches_double_closed_form() {
    let cfg = ThinWallConfig::for_nozzle(Fix128::from_ratio(4, 10)); // 0.4mm
                                                                     // oracle: nozzle_mm * 2, independently constructed as 8/10.
    let expected = Fix128::from_ratio(8, 10);
    assert!(
        approx_eq_mm(cfg.min_thickness_mm, expected.to_f64(), 1e-6),
        "for_nozzle(0.4) min_thickness_mm = {}, expected 0.8",
        cfg.min_thickness_mm.to_f64()
    );
}

#[test]
#[should_panic(expected = "nozzle_mm must be positive")]
fn for_nozzle_zero_diameter_now_fails_fast() {
    // 2026-10-03 root-cause fix: a zero (or negative) nozzle
    // diameter used to silently produce `min_thickness_mm == 0`, which
    // disabled thin-wall detection entirely (`has_thin_walls()` always
    // `false`, regardless of actual geometry) with no panic and no `Err`.
    // `for_nozzle` now fails fast instead — this test is the replacement
    // for the old `for_nozzle_zero_diameter_current_behavior_is_never_thin`,
    // which asserted the silent-disable behaviour as "current, not changed".
    let _ = ThinWallConfig::for_nozzle(Fix128::ZERO);
}

// ============================================================================
// measure_thickness_at
// ============================================================================

#[test]
fn measure_thickness_at_stepped_slab_thin_side_matches_closed_form() {
    let sdf = stepped_slab_sdf();
    let cfg = ThinWallConfig::for_nozzle(Fix128::from_ratio(4, 10));
    let p = Vec3Fix::from_f32(-2.0, 0.0, 0.3);
    let t = measure_thickness_at(&sdf, p, (0.0, 0.0, 1.0), &cfg).expect("must hit opposite face");
    // oracle: thickness of a slab of half-thickness h is exactly 2h = 0.6mm.
    assert!(
        approx_eq_mm(t, 0.6, 1e-3),
        "thin side thickness = {}, expected 0.6",
        t.to_f64()
    );
}

#[test]
fn measure_thickness_at_stepped_slab_thick_side_matches_closed_form() {
    let sdf = stepped_slab_sdf();
    let cfg = ThinWallConfig::for_nozzle(Fix128::from_ratio(4, 10));
    let p = Vec3Fix::from_f32(2.0, 0.0, 0.5);
    let t = measure_thickness_at(&sdf, p, (0.0, 0.0, 1.0), &cfg).expect("must hit opposite face");
    // oracle: 2h = 1.0mm.
    assert!(
        approx_eq_mm(t, 1.0, 1e-3),
        "thick side thickness = {}, expected 1.0",
        t.to_f64()
    );
}

#[test]
fn measure_thickness_at_degenerate_zero_length_normal_returns_none() {
    let sdf = stepped_slab_sdf();
    let cfg = ThinWallConfig::default();
    let p = Vec3Fix::from_f32(-2.0, 0.0, 0.3);
    let t = measure_thickness_at(&sdf, p, (0.0, 0.0, 0.0), &cfg);
    assert!(t.is_none(), "zero-length normal must be rejected");
}

#[test]
fn measure_thickness_at_extreme_coordinates_does_not_panic() {
    let sdf = sphere_sdf(5.0);
    let cfg = ThinWallConfig::default();
    let p = Vec3Fix::from_int(1_000_000_000, 0, 0);
    let result = catch_unwind(AssertUnwindSafe(|| {
        measure_thickness_at(&sdf, p, (1.0, 0.0, 0.0), &cfg)
    }));
    assert!(
        result.is_ok(),
        "extreme-magnitude surface point must not panic: {result:?}"
    );
    // 2026-10-03 root-cause fix: at this magnitude the f32 ULP
    // (~64) swallows the 0.01mm inward start_offset_mm entirely, so `px`
    // after the initial inward step used to be bit-identical to
    // `surface_point.x.to_f32()` — the function then reported
    // `Some(start_offset_mm)` (~0.01mm), a thickness unrelated to any real
    // geometric opposite-face distance. `measure_thickness_at` now detects
    // "did not actually move" and reports `None` (cannot reliably measure)
    // instead — this replaces the old assertion that pinned the bogus
    // `Some` as "current behaviour, not changed".
    assert!(
        result.unwrap().is_none(),
        "no real movement was possible at this magnitude; must report None, not a bogus Some"
    );
}

// ============================================================================
// analyze_thickness
// ============================================================================

#[test]
fn analyze_thickness_stepped_slab_matches_hand_derived_partition() {
    let sdf = stepped_slab_sdf();
    let cfg = ThinWallConfig::for_nozzle(Fix128::from_ratio(4, 10)); // 0.8mm threshold
    let xs: [f32; 6] = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
    let points: Vec<Vec3Fix> = xs
        .iter()
        .map(|&x| Vec3Fix::from_f32(x, 0.0, half_thickness_for_x(x)))
        .collect();

    let report = analyze_thickness(&sdf, &points, &cfg);

    assert_eq!(report.sampled_count, 6);
    // oracle: 3 thin-side points (x<0, thickness 0.6 < 0.8) out of 6 total.
    assert_eq!(
        report.regions.len(),
        3,
        "exactly the 3 thin-side points must be flagged"
    );
    assert_eq!(report.unbounded_count, 0);
    assert!(report.has_thin_walls());
    // oracle: 3/6 == 1/2 exactly (Fix128 base-2 fixed point, no remainder).
    assert_eq!(report.thin_fraction(), Fix128::from_ratio(1, 2));
    assert!(approx_eq_mm(report.min_thickness_seen, 0.6, 1e-3));
    assert!(approx_eq_mm(report.max_thickness_seen, 1.0, 1e-3));

    // All three flagged regions must be exactly the thin-side x coordinates.
    let mut flagged_xs: Vec<f32> = report
        .regions
        .iter()
        .map(|r| r.position.x.to_f32())
        .collect();
    flagged_xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (got, expected) in flagged_xs.iter().zip([-3.0f32, -2.0, -1.0].iter()) {
        assert!(
            (got - expected).abs() < 1e-3,
            "got {got}, expected {expected}"
        );
    }
}

#[test]
fn analyze_thickness_uniform_thin_slab_all_flagged() {
    let sdf = uniform_thin_slab_sdf();
    let cfg = ThinWallConfig::for_nozzle(Fix128::from_ratio(4, 10)); // 0.8mm threshold
    let points: Vec<Vec3Fix> = [-3.0f32, -1.0, 1.0, 3.0]
        .iter()
        .map(|&x| Vec3Fix::from_f32(x, 0.0, 0.3))
        .collect();

    let report = analyze_thickness(&sdf, &points, &cfg);

    assert_eq!(report.sampled_count, 4);
    // oracle: thickness is 2*0.3 = 0.6mm everywhere, below the 0.8mm
    // threshold, so all 4 points are flagged.
    assert_eq!(report.regions.len(), 4);
    assert_eq!(report.thin_fraction(), Fix128::ONE);
    assert!(report.has_thin_walls());
}

#[test]
fn analyze_thickness_empty_points_is_vacuous_zero_fraction() {
    let sdf = sphere_sdf(5.0);
    let cfg = ThinWallConfig::default();
    let report = analyze_thickness(&sdf, &[], &cfg);
    assert_eq!(report.sampled_count, 0);
    assert_eq!(report.unbounded_count, 0);
    assert_eq!(report.regions.len(), 0);
    assert_eq!(report.thin_fraction(), Fix128::ZERO);
    assert!(!report.has_thin_walls());
}

// ============================================================================
// sample_surface_points / analyze_thickness_grid
// ============================================================================

#[test]
fn sample_surface_points_sphere_grid_matches_hand_enumerated_crossings() {
    let sdf = sphere_sdf(5.0);
    // oracle (by hand, see module doc comment above): y, z visit {-6,0,6};
    // y^2+z^2 < 25 only for (0,0); on that line the SDF |x|-5 is exactly
    // piecewise-linear so interpolation lands exactly on x = -5 and x = +5.
    let pts = sample_surface_points(
        &sdf,
        Vec3Fix::from_int(-6, -6, -6),
        Vec3Fix::from_int(6, 6, 6),
        Fix128::from_int(6),
    );
    assert_eq!(pts.len(), 2, "exactly 2 crossings expected, got {pts:?}");

    let mut xs: Vec<f32> = pts.iter().map(|p| p.x.to_f32()).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((xs[0] - (-5.0)).abs() < 1e-3, "got {}", xs[0]);
    assert!((xs[1] - 5.0).abs() < 1e-3, "got {}", xs[1]);

    for p in &pts {
        let (x, y, z) = p.to_f32();
        assert!(
            y.abs() < 1e-6 && z.abs() < 1e-6,
            "expected y=z=0, got ({x},{y},{z})"
        );
        // Points genuinely lie on the surface: SDF value ~ 0.
        let d = sdf.distance(x, y, z);
        assert!(d.abs() < 1e-2, "surface point sdf = {d}, expected ~0");
    }
}

#[test]
fn sample_surface_points_zero_grid_step_returns_empty() {
    let sdf = sphere_sdf(5.0);
    let pts = sample_surface_points(
        &sdf,
        Vec3Fix::from_int(-6, -6, -6),
        Vec3Fix::from_int(6, 6, 6),
        Fix128::ZERO,
    );
    assert!(pts.is_empty(), "grid_step_mm = 0 must return an empty Vec");
}

#[test]
fn sample_surface_points_inverted_aabb_returns_empty() {
    let sdf = sphere_sdf(5.0);
    // aabb_min component-wise GREATER than aabb_max.
    let pts = sample_surface_points(
        &sdf,
        Vec3Fix::from_int(6, 6, 6),
        Vec3Fix::from_int(-6, -6, -6),
        Fix128::from_int(1),
    );
    assert!(pts.is_empty(), "inverted AABB must yield no samples");
}

#[test]
fn sample_surface_points_extreme_aabb_does_not_panic() {
    // ⚠️ Extreme *coordinate magnitude*, NOT extreme *range*, and the step
    // must be large relative to the f32 ULP at that magnitude. A huge range
    // with a small step (e.g. -1e9..1e9 step 1) is a legitimately large
    // ~(2e9)^3-iteration triple loop — a performance concern, not a
    // correctness one, and out of scope here.
    let sdf = sphere_sdf(5.0);
    let result = catch_unwind(AssertUnwindSafe(|| {
        sample_surface_points(
            &sdf,
            Vec3Fix::from_int(999_999_872, -128, -128),
            Vec3Fix::from_int(1_000_000_128, 128, 128),
            Fix128::from_int(128),
        )
    }));
    assert!(
        result.is_ok(),
        "extreme-magnitude AABB must not panic: {result:?}"
    );
}

#[test]
fn sample_surface_points_step_smaller_than_ulp_terminates() {
    // ⚠️ This is the exact trap found while authoring this file and fixed
    // at the root 2026-10-03 (sibling fix): near `x ≈ 1e9` the `f32` ULP is 64 (`2^(29-23)`), so
    // `x += 1.0_f32` was a no-op and `while x <= xmax` never advanced —
    // this AABB (range 2mm, step 1mm — a 3-sample grid by any reasonable
    // count) used to hang indefinitely. `sample_surface_points` now drives
    // the grid walk off an integer step count instead of repeated `f32`
    // addition, so this terminates unconditionally. A real `catch_unwind`
    // cannot observe a hang (it only catches panics), so the actual
    // regression guard here is this test *returning at all* — CI's own
    // job timeout is the backstop if this ever regresses.
    let sdf = sphere_sdf(5.0);
    let pts = sample_surface_points(
        &sdf,
        Vec3Fix::from_int(999_999_999, 999_999_999, 999_999_999),
        Vec3Fix::from_int(1_000_000_001, 1_000_000_001, 1_000_000_001),
        Fix128::from_int(1),
    );
    // Far from the origin-centred radius-5 sphere, so no surface crossing —
    // correctness here is "terminated with a well-formed (possibly empty)
    // result", which this assertion exercises (an empty `Vec` is fine).
    assert!(
        pts.len() <= 27, // 3 steps per axis at most (0, 1, 2) => 3^3 upper bound
        "unexpectedly many samples: {}",
        pts.len()
    );
}

#[test]
fn analyze_thickness_grid_sphere_matches_hand_derived_diameter() {
    let sdf = sphere_sdf(5.0);
    let cfg = ThinWallConfig::for_nozzle(Fix128::from_ratio(4, 10)); // 0.8mm threshold
    let report = analyze_thickness_grid(
        &sdf,
        Vec3Fix::from_int(-6, -6, -6),
        Vec3Fix::from_int(6, 6, 6),
        Fix128::from_int(6),
        &cfg,
    );
    // oracle: 2 diametrically-opposite surface points, opposite-face
    // distance = diameter = 10mm, well above the 0.8mm threshold.
    assert_eq!(report.sampled_count, 2);
    assert_eq!(report.regions.len(), 0);
    assert_eq!(report.thin_fraction(), Fix128::ZERO);
    assert!(!report.has_thin_walls());
    assert!(approx_eq_mm(report.min_thickness_seen, 10.0, 0.5));
    assert!(approx_eq_mm(report.max_thickness_seen, 10.0, 0.5));
}

/// Guards against `analyze_thickness_grid` silently discarding the `config`
/// argument it is supposed to forward to `analyze_thickness` (the "新しい
/// 引数を足したら既定値でない scene を1本置く" rule: the main diameter test
/// above uses `for_nozzle(0.4)`, whose `min_thickness_mm` (0.8mm) happens to
/// equal `ThinWallConfig::default()`'s own `min_thickness_mm` field-for-field
/// — a wiring bug that drops `config` back to `default()` would be
/// observationally invisible there). Here the nozzle is deliberately huge
/// (6mm → 12mm threshold), which *differs* from the default (0.8mm) and
/// flips the expected outcome: the sphere's own 10mm diameter is thin under
/// a 12mm threshold but not under the 0.8mm default.
#[test]
fn analyze_thickness_grid_large_nozzle_flags_sphere_as_thin() {
    let sdf = sphere_sdf(5.0);
    let cfg = ThinWallConfig::for_nozzle(Fix128::from_int(6)); // 12mm threshold
    let report = analyze_thickness_grid(
        &sdf,
        Vec3Fix::from_int(-6, -6, -6),
        Vec3Fix::from_int(6, 6, 6),
        Fix128::from_int(6),
        &cfg,
    );
    assert_eq!(report.sampled_count, 2);
    // oracle: 10mm diameter < 12mm threshold -> both points flagged.
    assert_eq!(report.regions.len(), 2);
    assert_eq!(report.thin_fraction(), Fix128::ONE);
    assert!(report.has_thin_walls());
}

#[test]
fn analyze_thickness_grid_zero_extent_aabb_is_empty() {
    let sdf = sphere_sdf(5.0);
    let cfg = ThinWallConfig::default();
    let report = analyze_thickness_grid(
        &sdf,
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::from_int(0, 0, 0),
        Fix128::from_int(1),
        &cfg,
    );
    assert_eq!(report.sampled_count, 0);
    assert_eq!(report.thin_fraction(), Fix128::ZERO);
    assert!(!report.has_thin_walls());
}

// ============================================================================
// ThinWallReport::has_thin_walls / thin_fraction (direct, no SDF)
// ============================================================================

#[test]
fn thin_fraction_default_report_is_zero_and_not_thin() {
    let report = ThinWallReport::default();
    assert_eq!(report.thin_fraction(), Fix128::ZERO);
    assert!(!report.has_thin_walls());
}
