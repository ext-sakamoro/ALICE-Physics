//! Audit oracles (S2-1) for `alice_physics::thin_wall`.
//!
//! Expected values are hand-derived from hand-built signed distance fields
//! (slabs, spheres, boxes) and the module's stated definition of wall
//! thickness ("distance to the opposite surface along the inward normal").
//! `known defect` tests are `#[ignore]`d and recorded in the audit ledger;
//! they are not fixed here.

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    clippy::field_reassign_with_default,
    clippy::unnecessary_map_or,
    clippy::needless_range_loop
)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::thin_wall::{
    analyze_thickness, analyze_thickness_grid, measure_thickness_at, sample_surface_points,
    ThinWallConfig, ThinWallReport,
};

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

/// Infinite slab `|z| - h` (full thickness 2h), normal along +-z.
fn slab(h: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |_x, _y, z| z.abs() - h,
        |_x, _y, z| (0.0, 0.0, if z >= 0.0 { 1.0 } else { -1.0 }),
    )
}

fn sphere(r: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - r,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt().max(f32::EPSILON);
            (x / l, y / l, z / l)
        },
    )
}

/// Box `|x|<=hx, |y|<=hy, |z|<=hz` (exact SDF, normal = dominant face).
fn boxed(hx: f32, hy: f32, hz: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| {
            let (qx, qy, qz) = (x.abs() - hx, y.abs() - hy, z.abs() - hz);
            let o = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2) + qz.max(0.0).powi(2)).sqrt();
            o + qx.max(qy).max(qz).min(0.0)
        },
        move |x, y, z| {
            let (qx, qy, qz) = (x.abs() - hx, y.abs() - hy, z.abs() - hz);
            if qz >= qx && qz >= qy {
                (0.0, 0.0, z.signum())
            } else if qx >= qy {
                (x.signum(), 0.0, 0.0)
            } else {
                (0.0, y.signum(), 0.0)
            }
        },
    )
}

fn thick(sdf: &ClosureSdf, p: Vec3Fix, n: (f32, f32, f32), cfg: &ThinWallConfig) -> Option<f64> {
    measure_thickness_at(sdf, p, n, cfg).map(Fix128::to_f64)
}

// ---------------------------------------------------------------- config

/// Doc: default 0.8 mm = nozzle 0.4 x 2; for_nozzle(n) sets min = 2n and
/// leaves every other field at its default.
#[test]
fn for_nozzle_is_twice_the_nozzle_and_keeps_other_defaults() {
    let d = ThinWallConfig::default();
    let n = ThinWallConfig::for_nozzle(Fix128::from_ratio(4, 10));
    assert_eq!(n.min_thickness_mm, d.min_thickness_mm, "0.4 nozzle -> 0.8");
    assert_eq!(n.max_march_distance_mm, d.max_march_distance_mm);
    assert_eq!(n.max_iterations, d.max_iterations);
    assert!((n.start_offset_mm - d.start_offset_mm).abs() < f32::EPSILON);
    assert!((n.min_step_mm - d.min_step_mm).abs() < f32::EPSILON);
    for (num, den) in [(2i64, 10i64), (6, 10), (8, 10), (1, 1)] {
        let c = ThinWallConfig::for_nozzle(Fix128::from_ratio(num, den));
        let want = 2.0 * num as f64 / den as f64;
        assert!(
            (c.min_thickness_mm.to_f64() - want).abs() < 1e-12,
            "{num}/{den}"
        );
    }
    assert_eq!(d.max_march_distance_mm, Fix128::from_int(50));
}

/// Doc `# Panics`: `nozzle_mm <= 0` (both zero and negative).
#[test]
fn for_nozzle_panics_for_zero_and_negative() {
    assert!(std::panic::catch_unwind(|| ThinWallConfig::for_nozzle(Fix128::ZERO)).is_err());
    assert!(
        std::panic::catch_unwind(|| ThinWallConfig::for_nozzle(Fix128::from_ratio(-4, 10)))
            .is_err()
    );
}

// ----------------------------------------------------- measure_thickness_at

/// A slab of full thickness 2h measures 2h for any h, from either face.
#[test]
fn measure_thickness_equals_full_slab_thickness_from_either_face() {
    let cfg = ThinWallConfig::default();
    for h in [0.1f32, 0.25, 0.5, 2.0, 10.0] {
        let s = slab(h);
        let up = thick(&s, v(1.0, 2.0, f64::from(h)), (0.0, 0.0, 1.0), &cfg).unwrap();
        let dn = thick(&s, v(1.0, 2.0, -f64::from(h)), (0.0, 0.0, -1.0), &cfg).unwrap();
        let want = 2.0 * f64::from(h);
        assert!((up - want).abs() < 2e-3 * want.max(1.0), "h={h} up {up}");
        assert!((dn - want).abs() < 2e-3 * want.max(1.0), "h={h} dn {dn}");
    }
}

/// An oblique inward ray travels farther than the slab thickness: for a unit
/// normal at angle theta from z the path length is 2h / cos(theta).
#[test]
fn measure_thickness_along_oblique_normal_is_path_length() {
    let cfg = ThinWallConfig::default();
    let s = slab(1.0);
    let (nx, nz) = (0.6f32, 0.8f32);
    let got = thick(&s, v(0.0, 0.0, 1.0), (nx, 0.0, nz), &cfg).unwrap();
    let want = 2.0 / 0.8;
    assert!((got - want).abs() < 5e-3, "got {got}, want {want}");
}

/// The normal must be unit-ish: `0.5 <= |n|^2 <= 1.5` is accepted, outside is
/// rejected (None).
#[test]
fn normal_length_gate_bounds() {
    let cfg = ThinWallConfig::default();
    let s = slab(1.0);
    let p = v(0.0, 0.0, 1.0);
    assert!(
        thick(&s, p, (0.0, 0.0, 0.7), &cfg).is_none(),
        "|n|^2 = 0.49"
    );
    assert!(
        thick(&s, p, (0.0, 0.0, 0.75), &cfg).is_some(),
        "|n|^2 = 0.5625"
    );
    assert!(
        thick(&s, p, (0.0, 0.0, 1.2), &cfg).is_some(),
        "|n|^2 = 1.44"
    );
    assert!(
        thick(&s, p, (0.0, 0.0, 1.25), &cfg).is_none(),
        "|n|^2 = 1.5625"
    );
}

/// Exact edges of the unit-normal gate: `|n|^2 == 0.5` and `|n|^2 == 1.5` are
/// both accepted (closed interval). Sphere R=5 probed at (5,0,0) with
/// dyadic normals so the squared lengths are exact in f32.
#[test]
fn normal_length_gate_edges_are_inclusive() {
    let cfg = ThinWallConfig::default();
    let s = sphere(5.0);
    let p = v(5.0, 0.0, 0.0);
    assert!(
        thick(&s, p, (0.5, 0.5, 0.0), &cfg).is_some(),
        "|n|^2 == 0.5"
    );
    assert!(
        thick(&s, p, (1.0, 0.5, 0.5), &cfg).is_some(),
        "|n|^2 == 1.5"
    );
}

/// The cutoff `distance > max_march_distance_mm` is strict: a wall whose
/// travelled distance equals the budget exactly is still measured. Dyadic
/// numbers (offset 0.25, step 0.5, budget 0.75) make the equality exact.
#[test]
fn travelled_distance_equal_to_budget_is_still_measured() {
    let f = ClosureSdf::new(
        |_x, _y, z| if z > -0.75 { -0.5 } else { 0.5 },
        |_x, _y, _z| (0.0, 0.0, 1.0),
    );
    let mut cfg = ThinWallConfig::default();
    cfg.start_offset_mm = 0.25;
    cfg.max_march_distance_mm = Fix128::from_ratio(3, 4);
    let t = thick(&f, v(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), &cfg);
    assert_eq!(t, Some(0.75));
}

/// Doc: `None` if no opposite surface within `max_march_distance_mm`.
#[test]
fn max_march_distance_cutoff() {
    let s = slab(2.0);
    let p = v(0.0, 0.0, 2.0);
    let n = (0.0, 0.0, 1.0);
    let mut cfg = ThinWallConfig::default();
    cfg.max_march_distance_mm = Fix128::from_ratio(39, 10);
    assert!(
        thick(&s, p, n, &cfg).is_none(),
        "4.0 mm slab, 3.9 mm budget"
    );
    cfg.max_march_distance_mm = Fix128::from_ratio(41, 10);
    let t = thick(&s, p, n, &cfg).unwrap();
    assert!((t - 4.0).abs() < 5e-3, "{t}");
}

/// Doc: `None` if `max_iterations` was exhausted. A 4 mm slab needs about 10
/// doubling steps from the 0.01 mm start offset.
#[test]
fn max_iterations_exhaustion_returns_none() {
    let s = slab(2.0);
    let p = v(0.0, 0.0, 2.0);
    let n = (0.0, 0.0, 1.0);
    let mut cfg = ThinWallConfig::default();
    cfg.max_iterations = 3;
    assert!(thick(&s, p, n, &cfg).is_none());
    cfg.max_iterations = 64;
    assert!(thick(&s, p, n, &cfg).is_some());
}

/// `min_step_mm` keeps a slowly converging march moving: inside field with
/// |d| = 1e-4 until z = -0.5, then outside. Steps of 1e-4 would need ~5000
/// iterations; a floor of 1e-3 needs ~500 and still lands within one step.
#[test]
fn min_step_floor_lets_slow_march_finish() {
    let slow = ClosureSdf::new(
        |_x, _y, z| if z > -0.5 { -1e-4 } else { 1e-4 },
        |_x, _y, _z| (0.0, 0.0, 1.0),
    );
    let mut cfg = ThinWallConfig::default();
    cfg.max_iterations = 1000;
    let t = thick(&slow, v(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), &cfg).unwrap();
    assert!((0.5..=0.5 + 1.1e-3).contains(&t), "got {t}");
    cfg.min_step_mm = 1e-4;
    assert!(
        thick(&slow, v(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), &cfg).is_none(),
        "without a floor the 1000-iteration budget is exhausted"
    );
}

/// Configured `start_offset_mm` enters the distance: a field already outside
/// after the first offset reports the offset itself.
#[test]
fn start_offset_is_added_to_the_distance() {
    let outside_after_offset = ClosureSdf::new(
        |_x, _y, z| if z > -0.05 { -1.0 } else { 1.0 },
        |_x, _y, _z| (0.0, 0.0, 1.0),
    );
    let mut cfg = ThinWallConfig::default();
    cfg.start_offset_mm = 0.07;
    let t = thick(
        &outside_after_offset,
        v(0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        &cfg,
    )
    .unwrap();
    // first query at z = -0.07 is outside -> reports the offset (0.07)
    assert!((t - 0.07).abs() < 1e-6, "got {t}");
}

/// Fixed (AUD-A-S2W1-005): a surface point that lies just OUTSIDE the SDF zero set by
/// more than `start_offset_mm` (0.01 mm), e.g. a mesh vertex off the field by
/// 0.011 mm, makes the first march query land outside (d >= 0) and the
/// function reports `Some(0.01)` -- "exited through the opposite surface after
/// 0.01 mm" -- instead of the true 10 mm. The caller gets a silent false thin
/// region.
#[test]
fn point_slightly_outside_surface_is_not_reported_as_a_hairline_wall() {
    let cfg = ThinWallConfig::default();
    let s = sphere(5.0);
    let t = thick(&s, v(5.011, 0.0, 0.0), (1.0, 0.0, 0.0), &cfg);
    // traced onto the surface at x = 5, then the 10 mm chord to x = -5 (the
    // march exits within the minimum step of the far face)
    let t = t.expect("a point 0.011 mm off a 10 mm solid is measurable");
    assert!(
        (t - 10.0).abs() < 2e-2,
        "0.011 mm off the surface of a 10 mm solid reported {t}"
    );
    // further off (1 mm, inside the march budget): the same chord
    let t = thick(&s, v(6.0, 0.0, 0.0), (1.0, 0.0, 0.0), &cfg).unwrap();
    assert!((t - 10.0).abs() < 2e-2, "1 mm off: {t}");
}

// ------------------------------------------------------ analyze_thickness

fn mixed_field() -> ClosureSdf {
    // x < 0: slab half 0.3 (full 0.6), x >= 0: slab half 0.5 (full 1.0);
    // normals degenerate (zero) for x > 5 -> cannot be marched (unbounded).
    ClosureSdf::new(
        |x, _y, z| z.abs() - if x < 0.0 { 0.3 } else { 0.5 },
        |x, _y, _z| {
            if x > 5.0 {
                (0.0, 0.0, 0.0)
            } else {
                (0.0, 0.0, 1.0)
            }
        },
    )
}

#[test]
fn analyze_thickness_counts_extrema_unbounded_and_fraction() {
    let s = mixed_field();
    let cfg = ThinWallConfig::default(); // 0.8 mm
    let pts = [
        v(-2.0, 0.0, 0.3),
        v(-1.0, 0.0, 0.3),
        v(1.0, 0.0, 0.5),
        v(6.0, 0.0, 0.5),
    ];
    let r: ThinWallReport = analyze_thickness(&s, &pts, &cfg);
    assert_eq!(r.sampled_count, 4);
    assert_eq!(r.unbounded_count, 1);
    assert_eq!(r.regions.len(), 2);
    assert!(r.has_thin_walls());
    assert!((r.min_thickness_seen.to_f64() - 0.6).abs() < 2e-3);
    assert!((r.max_thickness_seen.to_f64() - 1.0).abs() < 2e-3);
    for reg in &r.regions {
        assert!(reg.position.x.to_f64() < 0.0);
        assert!((reg.thickness_mm.to_f64() - 0.6).abs() < 2e-3);
        assert_eq!(reg.outward_normal, (0.0, 0.0, 1.0));
    }
    // unbounded samples stay in the denominator
    assert_eq!(r.thin_fraction(), Fix128::from_ratio(1, 2));
}

/// Doc: "Minimum/Maximum thickness observed across all successful samples";
/// nothing successful -> min is reset to zero (not the i64::MAX/2 sentinel).
#[test]
fn analyze_thickness_without_successful_samples_reports_zero_extrema() {
    let s = mixed_field();
    let cfg = ThinWallConfig::default();
    let r = analyze_thickness(&s, &[v(6.0, 0.0, 0.5), v(7.0, 0.0, 0.5)], &cfg);
    assert_eq!(r.sampled_count, 2);
    assert_eq!(r.unbounded_count, 2);
    assert_eq!(r.min_thickness_seen, Fix128::ZERO);
    assert_eq!(r.max_thickness_seen, Fix128::ZERO);
    assert!(!r.has_thin_walls());
    let e = analyze_thickness(&s, &[], &cfg);
    assert_eq!(e.sampled_count, 0);
    assert_eq!(e.min_thickness_seen, Fix128::ZERO);
}

/// Doc: "Report walls thinner than `min_thickness_mm`": strict. A wall whose
/// measured thickness equals the threshold exactly is not reported.
#[test]
fn threshold_is_strict_less_than() {
    let s = slab(0.3);
    let p = v(0.0, 0.0, 0.3);
    let mut cfg = ThinWallConfig::default();
    let t = measure_thickness_at(&s, p, (0.0, 0.0, 1.0), &cfg).unwrap();
    cfg.min_thickness_mm = t;
    assert!(!analyze_thickness(&s, &[p], &cfg).has_thin_walls());
    cfg.min_thickness_mm = t + Fix128::from_ratio(1, 1000);
    assert!(analyze_thickness(&s, &[p], &cfg).has_thin_walls());
}

// ------------------------------------------------------------ the grid

/// Grid sampler on a SDF whose surface sits exactly on a grid node: exactly
/// one crossing per surface, no duplicates, at the node (x = -1 and x = +1).
#[test]
fn surface_exactly_on_a_node_yields_one_crossing_each() {
    let s = ClosureSdf::new(
        |x, _y, _z| x.abs() - 1.0,
        |x, _y, _z| (x.signum(), 0.0, 0.0),
    );
    let pts = sample_surface_points(&s, v(-2.0, 0.0, 0.0), v(2.0, 0.0, 0.0), Fix128::ONE);
    let xs: Vec<f64> = pts.iter().map(|p| p.x.to_f64()).collect();
    assert_eq!(xs, vec![-1.0, 1.0]);
}

/// R = 5 sphere, unit grid: x-lines through (y,z) with y^2+z^2 < 25 cross
/// twice. Lattice points strictly inside the circle of radius 5: 81 - 12 = 69.
/// The sphere is symmetric in the axes, so the Y and Z lines (AUD-A-S2W1-004)
/// add the same 2 * 69 each: 3 * 2 * 69 = 414. Every point lies on the grid
/// line it was found on (two of its coordinates are grid values) and within
/// 0.05 mm of the true surface (linear-interpolation error bound
/// h^2/8 * d'' / |d'| is about 0.045 mm on the most grazing lines at h = 1;
/// measured 0.013).
#[test]
fn sphere_unit_grid_crossing_count_and_accuracy() {
    let s = sphere(5.0);
    let pts = sample_surface_points(&s, v(-7.0, -7.0, -7.0), v(7.0, 7.0, 7.0), Fix128::ONE);
    assert_eq!(pts.len(), 3 * 2 * 69);
    // X lines first, then Y, then Z: block k holds the crossings of the axis-k
    // lines, whose two other coordinates are grid values
    for (k, block) in pts.chunks(2 * 69).enumerate() {
        for p in block {
            let c = [p.x.to_f64(), p.y.to_f64(), p.z.to_f64()];
            for (j, v) in c.iter().enumerate() {
                if j != k {
                    assert!(
                        (v - v.round()).abs() < 1e-9,
                        "axis {k} point {c:?} is off its line"
                    );
                }
            }
            let r = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            assert!((r - 5.0).abs() < 0.05, "point {c:?} at r={r}");
        }
    }
}

/// Grid pipeline on a solid 10 mm sphere reports the diameter for every point
/// (414 samples: the crossings of the X, Y and Z lines, see above).
#[test]
fn grid_pipeline_on_solid_sphere_measures_diameter() {
    let s = sphere(5.0);
    let r = analyze_thickness_grid(
        &s,
        v(-7.0, -7.0, -7.0),
        v(7.0, 7.0, 7.0),
        Fix128::ONE,
        &ThinWallConfig::default(),
    );
    assert_eq!(r.sampled_count, 414);
    assert_eq!(r.unbounded_count, 0);
    assert!(!r.has_thin_walls());
    assert!(r.max_thickness_seen.to_f64() <= 10.0 + 1e-3);
    assert!(r.min_thickness_seen.to_f64() > 9.9);
}

/// Fixed (AUD-A-S2W1-004): the grid scan found surface points only by sign changes
/// along X. A thin plate lying parallel to the X axis (z-thin, 0.4 mm) is
/// crossed by x-lines only at its two end faces, so the pipeline measures the
/// 20 mm plate length and reports no thin wall (measured: 158 samples, 0
/// regions, min thickness 20.0) although every point of the broad faces is a
/// 0.4 mm wall, below the 0.8 mm default threshold. The doc of
/// `analyze_thickness_grid` says "sign changes between adjacent grid cells".
#[test]
fn grid_pipeline_detects_thin_plate_parallel_to_scan_axis() {
    let s = boxed(10.0, 10.0, 0.2);
    let r = analyze_thickness_grid(
        &s,
        v(-12.0, -12.0, -1.0),
        v(12.0, 12.0, 1.0),
        Fix128::from_ratio(1, 4),
        &ThinWallConfig::default(),
    );
    assert!(r.sampled_count > 0);
    assert!(
        r.has_thin_walls(),
        "0.4 mm plate: min thickness seen {}",
        r.min_thickness_seen.to_f64()
    );
}

/// Fixed (AUD-A-S2W1-006): doc says points are extracted "inside `[aabb_min,
/// aabb_max]`", but `axis_steps` uses `ceil`, so the last sample coordinate
/// is up to one step beyond `aabb_max`. R = 5 sphere, AABB x in [-6, 4.5],
/// y in [0, 0.5], step 1: crossings are returned at x = 5 and on the line
/// y = 1, both outside the box.
#[test]
fn sampled_points_stay_inside_the_aabb() {
    let s = sphere(5.0);
    let (lo, hi) = (v(-6.0, 0.0, 0.0), v(4.5, 0.5, 0.0));
    let pts = sample_surface_points(&s, lo, hi, Fix128::ONE);
    assert!(!pts.is_empty());
    for p in &pts {
        assert!(
            p.x <= hi.x && p.y <= hi.y && p.z <= hi.z,
            "({}, {}, {}) beyond aabb_max",
            p.x.to_f64(),
            p.y.to_f64(),
            p.z.to_f64()
        );
    }
}

/// The last sample of each axis is `max` itself (AUD-A-S2W1-006): a surface
/// between the last grid point and `max` is still found. R = 4.2 sphere, AABB
/// x in [-6, 4.5], step 1: grid points end at x = 4 (inside), then 4.5
/// (outside), so the +x crossing on the line y = z = 0 is found near 4.2
#[test]
fn a_surface_between_the_last_grid_point_and_max_is_found() {
    let s = sphere(4.2);
    let (lo, hi) = (v(-6.0, 0.0, 0.0), v(4.5, 0.0, 0.0));
    let pts = sample_surface_points(&s, lo, hi, Fix128::ONE);
    let xs: Vec<f64> = pts.iter().map(|p| p.x.to_f64()).collect();
    assert_eq!(xs.len(), 2, "{xs:?}");
    assert!(
        (xs[0] + 4.2).abs() < 0.05 && (xs[1] - 4.2).abs() < 0.05,
        "{xs:?}"
    );
}

/// The inward approach of an off-surface point and the measurement share the
/// march distance budget: 5 mm outside a 10 mm solid with a 12 mm budget is
/// 15 mm of marching, so it is not measured (5 + 10 > 12); within the budget
/// (1 mm outside, 11 mm) it is
#[test]
fn the_inward_approach_counts_against_the_march_budget() {
    let mut cfg = ThinWallConfig::default();
    cfg.max_march_distance_mm = Fix128::from_int(12);
    let s = sphere(5.0);
    assert_eq!(thick(&s, v(10.0, 0.0, 0.0), (1.0, 0.0, 0.0), &cfg), None);
    let t = thick(&s, v(6.0, 0.0, 0.0), (1.0, 0.0, 0.0), &cfg).unwrap();
    assert!((t - 10.0).abs() < 2e-2, "{t}");
}
