//! Oracles for `sdf_collider::DistanceSdfQuery`: an `SdfQuery` built from a
//! distance closure alone, whose normal is the crate's central-difference
//! gradient (`FD_NORMAL_BASE_EPS`, the coordinate-scaled step, normalised,
//! `(0, 1, 0)` where the gradient vanishes).
//!
//! 1. Bit identity with the central-difference formula, restated here in
//!    f32 exactly as it stood in `ModifiedSdf::normal`,
//!    `SingleModifiedSdf::normal` and `DestructibleSdf::normal` before the
//!    three copies were folded into one helper. The four entry points
//!    (`DistanceSdfQuery` and the three wrappers) are each compared with
//!    `assert_eq!` on the `f32` triple, so changing the shared helper turns
//!    all four red.
//! 2. Closed form: on a sphere and a tilted plane the normal matches the
//!    analytic one within a bound derived from the step (rounding of the six
//!    distance evaluations and of `x ± e`, plus the `e²` truncation term).
//! 3. Through `sphere_trace_sdf_field` / `collide_point_sdf_field` /
//!    `collide_sphere_sdf_field`: the same hit / miss, `t` and depth bit for
//!    bit as a `ClosureSdfQuery` with the analytic normal (the march and the
//!    depth read only the distance), the normal within the bound of 2, and
//!    the TOI point (`pos - normal * dist`) within that bound times `dist`.
//! 4. A closure that captures non-`'static`, non-`Send` locals compiles and
//!    is called: one evaluation per distance, six per normal.
//! 5. Degenerate input: zero gradient gives `(0, 1, 0)`; a NaN or infinite
//!    distance gives a NaN normal.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use std::cell::Cell;
use std::rc::Rc;

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_ccd::{sphere_trace_sdf_field, SdfCcdConfig};
use alice_physics::sdf_collider::{
    collide_point_sdf_field, collide_sphere_sdf_field, ClosureSdf, ClosureSdfQuery,
    DistanceSdfQuery, SdfField, SdfFrame, SdfQuery, FD_NORMAL_BASE_EPS,
};
use alice_physics::sdf_destruction::{DestructibleSdf, DestructionShape};
use alice_physics::sim_modifier::{ModifiedSdf, PhysicsModifier, SingleModifiedSdf};

const RADIUS: f32 = 1.0;

fn sphere_d(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt() - RADIUS
}

fn sphere_n(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let l = (x * x + y * y + z * z).sqrt();
    (x / l, y / l, z / l)
}

/// Unit normal (2/3, -1/3, 2/3), offset 0.25.
const PLANE_N: (f32, f32, f32) = (2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0);
const PLANE_H: f32 = 0.25;

fn plane_d(x: f32, y: f32, z: f32) -> f32 {
    PLANE_N.0 * x + PLANE_N.1 * y + PLANE_N.2 * z - PLANE_H
}

/// The central-difference normal as it was written out in the three
/// wrappers before they shared a helper (do not "fix" this copy: it is the
/// reference the helper is held to).
fn reference_fd_normal(
    f: impl Fn(f32, f32, f32) -> f32,
    x: f32,
    y: f32,
    z: f32,
) -> (f32, f32, f32) {
    let scale = x.abs().max(y.abs()).max(z.abs());
    let e = 0.001_f32.max(1.0e-4 * scale);
    let dx = f(x + e, y, z) - f(x - e, y, z);
    let dy = f(x, y + e, z) - f(x, y - e, z);
    let dz = f(x, y, z + e) - f(x, y, z - e);
    let len = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
    if len < 1e-10 {
        (0.0, 1.0, 0.0)
    } else {
        (dx / len, dy / len, dz / len)
    }
}

/// Query points: near the origin (absolute step), on and off the surface,
/// and far out (step grows with the coordinate magnitude).
fn points() -> Vec<(f32, f32, f32)> {
    let mut v = vec![
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.3, 0.4, -0.5),
        (1.2, -0.7, 0.9),
        (-0.6, 0.6, 0.6),
        (2.0, 1.0, -2.0),
        (0.05, 0.02, -0.03),
        (13.0, -4.0, 7.5),
        (-120.0, 35.0, 80.0),
        (900.0, -1200.0, 50.0),
    ];
    // a deterministic spread around the unit sphere
    for i in 0..24 {
        let a = i as f32 * 0.7;
        let b = i as f32 * 0.31 - 3.0;
        let r = 0.6 + (i % 5) as f32 * 0.35;
        v.push((r * a.cos() * b.cos(), r * b.sin(), r * a.sin() * b.cos()));
    }
    v
}

fn step(x: f32, y: f32, z: f32) -> f32 {
    FD_NORMAL_BASE_EPS.max(1.0e-4 * x.abs().max(y.abs()).max(z.abs()))
}

/// Bound on `|n_fd - n_exact|` per component for a field with `|∇f| = 1`.
///
/// `mag` bounds the magnitude of the intermediate values of one distance
/// evaluation (e.g. `|p| + R` for the sphere). Each evaluation then carries
/// an absolute error of at most `3 ε mag` (a 3-term dot / sum of squares
/// plus `sqrt` and the final subtraction, each `≤ ε/2` relative, rounded
/// up), so one central difference is off by `≤ 6 ε mag`. Rounding `x ± e`
/// moves the sample by `≤ ε |x| / 2` on each side, i.e. `≤ ε (|p| + e)` in
/// the difference. Dividing by the `2 e` baseline gives the error in the
/// unnormalised gradient component; `trunc` is the central-difference
/// truncation `e² |f'''| / 6`. Normalising a vector whose norm is `1 ± δ`
/// can at most double a per-component error, and three components add
/// under the norm: factor `2 √3`.
fn fd_bound(mag: f32, p_abs: f32, e: f32, trunc: f32) -> f32 {
    let eps = f32::EPSILON;
    let per = (6.0 * eps * mag + eps * (p_abs + e)) / (2.0 * e) + trunc;
    2.0 * 3.0_f32.sqrt() * per
}

fn norm(x: f32, y: f32, z: f32) -> f32 {
    (x * x + y * y + z * z).sqrt()
}

fn max_diff(a: (f32, f32, f32), b: (f32, f32, f32)) -> f32 {
    (a.0 - b.0)
        .abs()
        .max((a.1 - b.1).abs())
        .max((a.2 - b.2).abs())
}

struct Identity;
impl PhysicsModifier for Identity {
    fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
        d
    }
    fn update(&mut self, _dt: f32) {}
    fn name(&self) -> &str {
        "identity"
    }
}

/// A modifier that changes the field: shrink everything by 0.1 plus a
/// ripple, so the wrapper's normal is not the original's.
struct Ripple;
impl PhysicsModifier for Ripple {
    fn modify_distance(&self, x: f32, y: f32, _z: f32, d: f32) -> f32 {
        d + 0.1 + 0.05 * (3.0 * x).sin() * (2.0 * y).cos()
    }
    fn update(&mut self, _dt: f32) {}
    fn name(&self) -> &str {
        "ripple"
    }
}

fn boxed_sphere() -> Box<dyn SdfField> {
    Box::new(ClosureSdf::new(sphere_d, sphere_n))
}

#[test]
fn four_entry_points_are_bit_identical_to_the_reference_formula() {
    let distance_only = DistanceSdfQuery::new(sphere_d);
    let modified_empty = ModifiedSdf::new(boxed_sphere());
    let mut modified = ModifiedSdf::new(boxed_sphere());
    modified.add_modifier(Box::new(Ripple));
    let single_identity = SingleModifiedSdf::new(boxed_sphere(), Identity);
    let single = SingleModifiedSdf::new(boxed_sphere(), Ripple);
    let destructible_empty = DestructibleSdf::new(boxed_sphere());
    let mut destructible = DestructibleSdf::new(boxed_sphere());
    destructible.apply_destruction(DestructionShape::sphere(
        Vec3Fix::from_f32(0.9, 0.2, 0.0),
        0.4,
    ));

    // mismatches per entry point, so a change to the shared formula shows
    // up on every path at once rather than at the first assert
    let names = [
        "DistanceSdfQuery",
        "ModifiedSdf",
        "SingleModifiedSdf",
        "DestructibleSdf",
    ];
    let mut bad = [0_usize; 4];
    let mut compared = 0;
    let mut distinct = 0;
    // the origin hits the zero-gradient fallback
    let mut pts = points();
    pts.push((0.0, 0.0, 0.0));
    for (x, y, z) in pts {
        let want = reference_fd_normal(sphere_d, x, y, z);
        assert_eq!(distance_only.query_distance(x, y, z), sphere_d(x, y, z));
        // with the unmodified sphere: the reference on the sphere's distance
        let plain = [
            distance_only.query_normal(x, y, z),
            modified_empty.normal(x, y, z),
            single_identity.normal(x, y, z),
            destructible_empty.normal(x, y, z),
        ];
        // with a field-changing modifier / destruction: the reference on
        // the wrapper's own distance
        let changed = [
            (
                1,
                modified.normal(x, y, z),
                reference_fd_normal(|a, b, c| modified.distance(a, b, c), x, y, z),
            ),
            (
                2,
                single.normal(x, y, z),
                reference_fd_normal(|a, b, c| single.distance(a, b, c), x, y, z),
            ),
            (
                3,
                destructible.normal(x, y, z),
                reference_fd_normal(|a, b, c| destructible.distance(a, b, c), x, y, z),
            ),
        ];
        for (i, got) in plain.iter().enumerate() {
            if *got != want {
                bad[i] += 1;
            }
        }
        for (i, got, wd) in changed {
            if got != wd {
                bad[i] += 1;
            }
            if wd != want {
                distinct += 1;
            }
        }
        compared += 1;
    }
    assert_eq!(compared, 35);
    let report: Vec<String> = names
        .iter()
        .zip(bad)
        .map(|(n, b)| format!("{n} {b}"))
        .collect();
    println!("[sdf_distance_query] mismatches: {}", report.join(", "));
    assert_eq!(
        bad,
        [0; 4],
        "mismatches vs reference: {}",
        report.join(", ")
    );
    // the modified wrappers really differ from the plain sphere somewhere
    assert!(distinct >= 34, "distinct {distinct}");
}

#[test]
fn sphere_normal_matches_closed_form_within_step_bound() {
    let q = DistanceSdfQuery::new(sphere_d);
    let mut checked = 0;
    let mut worst_ratio = 0.0_f32;
    for (x, y, z) in points() {
        let p = norm(x, y, z);
        let e = step(x, y, z);
        // |f'''| of |p| along an axis is at most 3 / |p|² (away from the
        // origin, and the samples stay at |p| - e√3 > 0 here)
        let trunc = e * e * 3.0 / (6.0 * (p - e * 3.0_f32.sqrt()).powi(2));
        let bound = fd_bound(p + e + RADIUS, p, e, trunc);
        let got = q.query_normal(x, y, z);
        let err = max_diff(got, sphere_n(x, y, z));
        assert!(err <= bound, "at {x},{y},{z}: err {err} > bound {bound}");
        worst_ratio = worst_ratio.max(err / bound);
        checked += 1;
    }
    assert_eq!(checked, 34);
    println!("[sdf_distance_query] sphere worst err / bound = {worst_ratio:.3}");
}

#[test]
fn plane_normal_matches_closed_form_within_step_bound() {
    let q = DistanceSdfQuery::new(plane_d);
    let mut checked = 0;
    for (x, y, z) in points() {
        let p = norm(x, y, z);
        let e = step(x, y, z);
        // linear field: no truncation; the dot product's terms are bounded
        // by |p| + e (|n| = 1) plus the offset
        let bound = fd_bound(p + e + PLANE_H, p, e, 0.0);
        let err = max_diff(q.query_normal(x, y, z), PLANE_N);
        assert!(err <= bound, "at {x},{y},{z}: err {err} > bound {bound}");
        checked += 1;
    }
    assert_eq!(checked, 34);
}

#[test]
fn ccd_and_contacts_match_analytic_normal_query() {
    let config = SdfCcdConfig::default();
    let analytic = ClosureSdfQuery::new(sphere_d, sphere_n);
    let fd = DistanceSdfQuery::new(sphere_d);
    let rot = QuatFix::from_axis_angle(
        Vec3Fix::from_f32(0.3, 1.0, -0.2).normalize(),
        Fix128::from_f32(0.7),
    );
    let frames = [
        SdfFrame::IDENTITY,
        SdfFrame::new(
            Vec3Fix::from_f32(0.5, -0.25, 0.75),
            rot,
            Fix128::from_f32(1.5),
        ),
    ];
    // the CCD normal goes through a rotation and Fix128 round trip on top
    // of the f32 bound; 1e-2 is the bound of the sphere oracle (≤ ~5e-3
    // for |p| ≈ 1) with that slack
    let tol = 1e-2_f32;
    let cases = [
        (
            Vec3Fix::from_f32(-5.0, 0.0, 0.0),
            Vec3Fix::from_f32(10.0, 0.0, 0.0),
            0.5_f32,
        ),
        (
            Vec3Fix::from_f32(0.3, 5.0, -0.2),
            Vec3Fix::from_f32(0.0, -10.0, 0.0),
            0.25,
        ),
        (
            Vec3Fix::from_f32(-4.0, 3.0, 1.0),
            Vec3Fix::from_f32(8.0, -6.0, -1.5),
            0.4,
        ),
        (
            Vec3Fix::from_f32(-5.0, 5.0, 0.0),
            Vec3Fix::from_f32(10.0, 0.0, 0.0),
            0.5,
        ),
    ];
    let mut hits = 0;
    let mut misses = 0;
    for frame in &frames {
        for (start, disp, r) in cases {
            let r = Fix128::from_f32(r);
            let a = sphere_trace_sdf_field(start, disp, r, &analytic, frame, &config);
            let b = sphere_trace_sdf_field(start, disp, r, &fd, frame, &config);
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.t, b.t, "toi t");
                    let d = max_diff(a.normal.to_f32(), b.normal.to_f32());
                    assert!(d <= tol, "toi normal diff {d}");
                    // `point = pos - normal * dist` with world `dist` within
                    // `tolerance` of the radius (≤ 0.5 here): off by at most
                    // `d * dist`
                    let dp = max_diff(a.point.to_f32(), b.point.to_f32());
                    assert!(dp <= d * 0.6 + 1e-6, "toi point diff {dp} (normal {d})");
                    hits += 1;
                }
                (None, None) => misses += 1,
                (a, b) => panic!("hit/miss disagree: {a:?} vs {b:?}"),
            }
        }
        // a point inside, a sphere overlapping, and a point outside
        for c in [
            Vec3Fix::from_f32(0.2, 0.3, -0.1),
            Vec3Fix::from_f32(0.9, 0.9, 0.4),
            Vec3Fix::from_f32(0.0, 3.0, 0.0),
        ] {
            let a = collide_point_sdf_field(c, &analytic, frame);
            let b = collide_point_sdf_field(c, &fd, frame);
            assert_eq!(a.is_some(), b.is_some());
            if let (Some(a), Some(b)) = (a, b) {
                assert_eq!(a.depth, b.depth);
                assert_eq!(a.point_a, b.point_a);
                assert!(max_diff(a.normal.to_f32(), b.normal.to_f32()) <= tol);
                hits += 1;
            }
            let r = Fix128::from_f32(0.6);
            let a = collide_sphere_sdf_field(c, r, &analytic, frame);
            let b = collide_sphere_sdf_field(c, r, &fd, frame);
            assert_eq!(a.is_some(), b.is_some());
            if let (Some(a), Some(b)) = (a, b) {
                assert_eq!(a.depth, b.depth);
                assert!(max_diff(a.normal.to_f32(), b.normal.to_f32()) <= tol);
                hits += 1;
            }
        }
    }
    assert!(hits >= 10 && misses >= 2, "hits {hits} misses {misses}");
}

#[test]
fn borrowed_non_send_closure_is_called() {
    let radius = 1.5_f32; // a local
    let calls = Rc::new(Cell::new(0_u32)); // neither Send nor Sync
    let counter = Rc::clone(&calls);
    let q = DistanceSdfQuery::new(|x: f32, y: f32, z: f32| {
        counter.set(counter.get() + 1);
        (x * x + y * y + z * z).sqrt() - radius
    });
    assert_eq!(q.query_distance(3.0, 0.0, 0.0), 1.5);
    assert_eq!(calls.get(), 1);
    let (nx, ny, nz) = q.query_normal(2.0, 0.0, 0.0);
    assert_eq!(calls.get(), 7, "central difference: six evaluations");
    assert!((nx - 1.0).abs() < 1e-3 && ny.abs() < 1e-3 && nz.abs() < 1e-3);
    // and it drives the borrowed-field CCD entry point
    let toi = sphere_trace_sdf_field(
        Vec3Fix::from_f32(-5.0, 0.0, 0.0),
        Vec3Fix::from_f32(10.0, 0.0, 0.0),
        Fix128::from_f32(0.5),
        &q,
        &SdfFrame::IDENTITY,
        &SdfCcdConfig::default(),
    )
    .expect("hit");
    let want = (5.0 - 1.5 - 0.5) / 10.0;
    assert!((toi.t.to_f32() - want).abs() < 1e-3);
}

#[test]
fn degenerate_inputs() {
    // zero gradient: the sphere's centre (all six samples equal by symmetry)
    let q = DistanceSdfQuery::new(sphere_d);
    assert_eq!(q.query_normal(0.0, 0.0, 0.0), (0.0, 1.0, 0.0));
    // a constant field has zero gradient everywhere
    let flat = DistanceSdfQuery::new(|_x: f32, _y: f32, _z: f32| 0.5);
    assert_eq!(flat.query_normal(3.0, -2.0, 7.0), (0.0, 1.0, 0.0));
    // NaN distance: the differences are NaN, `len < 1e-10` is false, NaN out
    let nan = DistanceSdfQuery::new(|_x: f32, _y: f32, _z: f32| f32::NAN);
    assert!(nan.query_distance(1.0, 0.0, 0.0).is_nan());
    let (a, b, c) = nan.query_normal(1.0, 0.0, 0.0);
    assert!(a.is_nan() && b.is_nan() && c.is_nan());
    // infinite distance: inf - inf = NaN, so the normal is NaN too
    let inf = DistanceSdfQuery::new(|_x: f32, _y: f32, _z: f32| f32::INFINITY);
    assert_eq!(inf.query_distance(1.0, 0.0, 0.0), f32::INFINITY);
    let (a, b, c) = inf.query_normal(1.0, 0.0, 0.0);
    assert!(a.is_nan() && b.is_nan() && c.is_nan());
    // a non-finite query point gives a non-finite step and a NaN normal
    let (a, _, _) = q.query_normal(f32::INFINITY, 0.0, 0.0);
    assert!(a.is_nan());
}
