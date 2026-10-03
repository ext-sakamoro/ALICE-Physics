//! Oracles for `sdf_ccd`: sphere tracing against closed-form time of impact.
//!
//! Geometry is chosen so every expected `t` is a hand computation:
//! plane `y = 0`, unit sphere at the origin, moving sphere radius `r`.
//! Sphere tracing stops once the gap is within `tolerance`, so a hit's `t`
//! lies in `[t* - tolerance/|disp|/cos, t*]` -- never past the true impact.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_ccd::{
    batch_sphere_trace_sdf, ray_march_sdf, sphere_trace_sdf, SdfCcdConfig,
};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn plane() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            (x / l, y / l, z / l)
        },
    )
}

fn stat(f: ClosureSdf) -> SdfCollider {
    SdfCollider::new_static(Box::new(f), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn v(x: f32, y: f32, z: f32) -> Vec3Fix {
    Vec3Fix::from_f32(x, y, z)
}

fn cfg() -> SdfCcdConfig {
    SdfCcdConfig::default()
}

const TOL: f32 = 1.0e-3;

#[test]
fn plane_toi_matches_closed_form_and_never_overshoots() {
    // start y = 5, fall 10, r = 0.5 -> contact at y = 0.5: t* = 0.45.
    let toi = sphere_trace_sdf(
        v(0.0, 5.0, 0.0),
        v(0.0, -10.0, 0.0),
        Fix128::from_f32(0.5),
        &stat(plane()),
        &cfg(),
    )
    .expect("hit");
    let t = toi.t.to_f32();
    assert!(
        (0.45 - TOL / 10.0 - 1e-6..=0.45 + 1e-6).contains(&t),
        "t = {t}"
    );
    let (nx, ny, nz) = toi.normal.to_f32();
    assert!(nx.abs() < 1e-5 && (ny - 1.0).abs() < 1e-5 && nz.abs() < 1e-5);
    // point = pos - n * dist: on the plane y = 0 (within the tolerance).
    let (_, py, _) = toi.point.to_f32();
    assert!(py.abs() <= TOL + 1e-5, "point y = {py}");
}

#[test]
fn oblique_sphere_toi_matches_quadratic() {
    // Unit sphere at origin, mover r = 0.5 along +x at y = 0.5 from x = -5:
    // centre distance 1.5 when x^2 + 0.25 = 2.25 -> x = -sqrt(2).
    let toi = sphere_trace_sdf(
        v(-5.0, 0.5, 0.0),
        v(10.0, 0.0, 0.0),
        Fix128::from_f32(0.5),
        &stat(unit_sphere()),
        &cfg(),
    )
    .expect("hit");
    let want = (5.0 - 2.0_f32.sqrt()) / 10.0; // 0.358579
    let t = toi.t.to_f32();
    assert!(
        t <= want + 1e-6 && want - t < 1.5e-3,
        "t = {t}, want {want}"
    );
    // Normal points from the sphere centre to the contact: (-sqrt(2), 0.5, 0)/1.5.
    let (nx, ny, nz) = toi.normal.to_f32();
    assert!((nx - (-2.0_f32.sqrt() / 1.5)).abs() < 2e-3, "nx {nx}");
    assert!((ny - (0.5 / 1.5)).abs() < 2e-3, "ny {ny}");
    assert!(nz.abs() < 1e-5);
}

#[test]
fn scaled_and_translated_collider_toi() {
    // Unit sphere scaled x2 and moved to (10, 0, 0): world radius 2.
    let sdf = SdfCollider::new_static(
        Box::new(unit_sphere()),
        v(10.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    )
    .with_scale(Fix128::from_int(2));
    // Mover r = 0.5 from x = 0 toward +x by 10: contact at x = 10 - 2.5 = 7.5 -> t* = 0.75.
    let toi = sphere_trace_sdf(
        Vec3Fix::ZERO,
        v(10.0, 0.0, 0.0),
        Fix128::from_f32(0.5),
        &sdf,
        &cfg(),
    )
    .expect("hit");
    let t = toi.t.to_f32();
    assert!(t <= 0.75 + 1e-6 && 0.75 - t < 2e-3, "t = {t}");
    let (nx, _, _) = toi.normal.to_f32();
    assert!((nx + 1.0).abs() < 1e-4);
}

#[test]
fn rotated_plane_reports_world_normal() {
    // Plane y = 0 rotated +90 deg about Z: local +Y -> world -X, so the plane is x = 0
    // with the free side at x < 0. Mover r = 0.5 from x = -5 toward +x.
    let q = QuatFix::from_axis_angle(
        Vec3Fix::UNIT_Z,
        Fix128::from_f32(core::f32::consts::FRAC_PI_2),
    );
    let sdf = SdfCollider::new_static(Box::new(plane()), Vec3Fix::ZERO, q);
    let toi = sphere_trace_sdf(
        v(-5.0, 0.0, 0.0),
        v(10.0, 0.0, 0.0),
        Fix128::from_f32(0.5),
        &sdf,
        &cfg(),
    )
    .expect("hit");
    // contact at x = -0.5: t* = 0.45, normal (-1, 0, 0).
    let t = toi.t.to_f32();
    assert!(t <= 0.45 + 1e-5 && 0.45 - t < 2e-3, "t = {t}");
    let (nx, ny, nz) = toi.normal.to_f32();
    assert!(
        (nx + 1.0).abs() < 1e-3 && ny.abs() < 1e-3 && nz.abs() < 1e-3,
        "n = {nx},{ny},{nz}"
    );
}

#[test]
fn misses_and_degenerate_inputs() {
    let s = stat(unit_sphere());
    let r = Fix128::from_f32(0.5);
    // Passes at y = 5: miss.
    assert!(sphere_trace_sdf(v(-5.0, 5.0, 0.0), v(10.0, 0.0, 0.0), r, &s, &cfg()).is_none());
    // Zero displacement: None, even when overlapping.
    assert!(sphere_trace_sdf(Vec3Fix::ZERO, Vec3Fix::ZERO, r, &s, &cfg()).is_none());
    // Too short to reach (contact needs 3.5): None.
    assert!(sphere_trace_sdf(v(-5.0, 0.0, 0.0), v(3.0, 0.0, 0.0), r, &s, &cfg()).is_none());
    // Moving away: None.
    assert!(sphere_trace_sdf(v(-5.0, 0.0, 0.0), v(-10.0, 0.0, 0.0), r, &s, &cfg()).is_none());
    // Already penetrating: t = 0.
    let toi = sphere_trace_sdf(
        v(0.0, 0.2, 0.0),
        v(1.0, 0.0, 0.0),
        r,
        &stat(plane()),
        &cfg(),
    )
    .expect("hit");
    assert_eq!(toi.t, Fix128::ZERO);
    // One iteration cannot reach it (conservative miss).
    let one = SdfCcdConfig {
        max_iterations: 1,
        ..cfg()
    };
    assert!(sphere_trace_sdf(v(-5.0, 0.0, 0.0), v(10.0, 0.0, 0.0), r, &s, &one).is_none());
    // A looser tolerance reports an earlier t (contact within tolerance).
    let loose = SdfCcdConfig {
        tolerance: 0.5,
        ..cfg()
    };
    let t_loose = sphere_trace_sdf(
        v(0.0, 5.0, 0.0),
        v(0.0, -10.0, 0.0),
        r,
        &stat(plane()),
        &loose,
    )
    .unwrap()
    .t
    .to_f32();
    let t_tight = sphere_trace_sdf(
        v(0.0, 5.0, 0.0),
        v(0.0, -10.0, 0.0),
        r,
        &stat(plane()),
        &cfg(),
    )
    .unwrap()
    .t
    .to_f32();
    assert!(
        t_loose < t_tight && t_loose >= 0.4 - 1e-4,
        "{t_loose} {t_tight}"
    );
}

#[test]
fn step_safety_scales_iteration_count_not_the_answer() {
    // With a smaller safety factor the hit is still found, never later than t*.
    let c = SdfCcdConfig {
        step_safety: 0.5,
        max_iterations: 200,
        ..cfg()
    };
    let t = sphere_trace_sdf(
        v(0.0, 5.0, 0.0),
        v(0.0, -10.0, 0.0),
        Fix128::from_f32(0.5),
        &stat(plane()),
        &c,
    )
    .unwrap()
    .t
    .to_f32();
    assert!(t <= 0.45 + 1e-6 && 0.45 - t < TOL / 10.0 + 1e-6, "t = {t}");
}

#[test]
fn ray_march_hits_at_closed_form_distance() {
    let s = stat(unit_sphere());
    let toi = ray_march_sdf(
        v(-5.0, 0.0, 0.0),
        Vec3Fix::UNIT_X,
        Fix128::from_int(20),
        &s,
        &cfg(),
    )
    .expect("hit");
    // surface at x = -1: distance 4 (t in distance units for a unit direction).
    let t = toi.t.to_f32();
    assert!(t <= 4.0 + 1e-6 && 4.0 - t <= TOL + 1e-5, "t = {t}");
    let (px, _, _) = toi.point.to_f32();
    assert!((px + 1.0).abs() <= TOL + 1e-4, "px {px}");
    let (nx, _, _) = toi.normal.to_f32();
    assert!((nx + 1.0).abs() < 1e-4);
    // Range shorter than 4: None. Zero direction: None. Miss: None.
    assert!(ray_march_sdf(
        v(-5.0, 0.0, 0.0),
        Vec3Fix::UNIT_X,
        Fix128::from_int(3),
        &s,
        &cfg()
    )
    .is_none());
    assert!(ray_march_sdf(
        v(-5.0, 0.0, 0.0),
        Vec3Fix::ZERO,
        Fix128::from_int(20),
        &s,
        &cfg()
    )
    .is_none());
    assert!(ray_march_sdf(
        v(-5.0, 3.0, 0.0),
        Vec3Fix::UNIT_X,
        Fix128::from_int(20),
        &s,
        &cfg()
    )
    .is_none());
    // Origin inside the surface: immediate hit at t = 0.
    let toi = ray_march_sdf(
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_X,
        Fix128::from_int(20),
        &s,
        &cfg(),
    )
    .unwrap();
    assert_eq!(toi.t, Fix128::ZERO);
}

#[test]
fn ray_march_with_a_non_unit_direction_does_not_tunnel() {
    // Same ray with the direction scaled by 2: the point must still stop on the surface x = -1.
    let s = stat(unit_sphere());
    let toi = ray_march_sdf(
        v(-5.0, 0.0, 0.0),
        v(2.0, 0.0, 0.0),
        Fix128::from_int(20),
        &s,
        &cfg(),
    );
    let toi = toi.expect("a ray aimed at the sphere must hit it");
    let (px, _, _) = toi.point.to_f32();
    assert!((px + 1.0).abs() <= TOL + 1e-4, "stopped at x = {px}");
    // t is a distance, not a multiple of |direction|: the same 4 as for a unit direction.
    let t = toi.t.to_f32();
    assert!(t <= 4.0 + 1e-5 && 4.0 - t <= TOL + 1e-5, "t = {t}");
    // max_distance is a distance too: 3 is short of the surface, 4.5 is enough.
    assert!(ray_march_sdf(
        v(-5.0, 0.0, 0.0),
        v(2.0, 0.0, 0.0),
        Fix128::from_int(3),
        &s,
        &cfg()
    )
    .is_none());
    assert!(ray_march_sdf(
        v(-5.0, 0.0, 0.0),
        v(2.0, 0.0, 0.0),
        Fix128::from_f32(4.5),
        &s,
        &cfg()
    )
    .is_some());
}

#[test]
fn world_sweep_reports_fast_bodies_earliest_first() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.set_sdf_collision_radius(Fix128::from_f32(0.5));
    w.add_sdf_collider(stat(plane())); // 0
    w.add_sdf_collider(stat(unit_sphere())); // 1
                                             // body 0: fast, falls from y = 5 at 600 m/s * (1/60) = 10: ground t = 0.45, sphere t = 0.35
    let b0 = w.add_body(
        RigidBody::new_dynamic(v(0.0, 5.0, 0.0), Fix128::ONE).with_velocity(v(0.0, -600.0, 0.0)),
    );
    // body 1: slow (below threshold 5): skipped
    let _b1 = w.add_body(
        RigidBody::new_dynamic(v(0.0, 5.0, 0.0), Fix128::ONE).with_velocity(v(0.0, -1.0, 0.0)),
    );
    // body 2: static: skipped even when fast
    let _b2 =
        w.add_body(RigidBody::new_static(v(0.0, 5.0, 0.0)).with_velocity(v(0.0, -600.0, 0.0)));
    let dt = Fix128::from_ratio(1, 60);
    let hits = w.sdf_ccd_hits(dt, &cfg());
    let keys: Vec<(usize, usize)> = hits.iter().map(|h| (h.0, h.1)).collect();
    assert_eq!(keys, vec![(b0, 1), (b0, 0)]);
    assert!((hits[0].2.t.to_f32() - 0.35).abs() < 2e-3);
    assert!((hits[1].2.t.to_f32() - 0.45).abs() < 2e-3);
    // dt = 0: no displacement, no hit.
    assert!(w.sdf_ccd_hits(Fix128::ZERO, &cfg()).is_empty());
    // A sensor is never swept.
    w.bodies[b0].is_sensor = true;
    assert!(w.sdf_ccd_hits(dt, &cfg()).is_empty());
    // An SDF attached to the body itself is skipped for that body.
    w.bodies[b0].is_sensor = false;
    let mut w2 = PhysicsWorld::new(PhysicsConfig::default());
    let id = w2.add_body(
        RigidBody::new_dynamic(v(0.0, 5.0, 0.0), Fix128::ONE).with_velocity(v(0.0, -600.0, 0.0)),
    );
    w2.add_sdf_collider(SdfCollider::new_dynamic(Box::new(plane()), id));
    assert!(w2.sdf_ccd_hits(dt, &cfg()).is_empty());
}

#[test]
fn batch_filters_and_sorts_like_the_world_sweep() {
    let cols = [stat(plane())];
    let bodies = [
        RigidBody::new_dynamic(v(0.0, 5.0, 0.0), Fix128::ONE).with_velocity(v(0.0, -10.0, 0.0)),
        RigidBody::new_dynamic(v(0.0, 2.0, 0.0), Fix128::ONE).with_velocity(v(0.0, -10.0, 0.0)),
    ];
    let disp = [v(0.0, -10.0, 0.0), v(0.0, -10.0, 0.0)];
    let hits = batch_sphere_trace_sdf(&bodies, &disp, Fix128::from_f32(0.5), &cols, &cfg());
    // body 1: contact y = 0.5 after 1.5 -> t = 0.15; body 0: 0.45. Earliest first.
    assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![1, 0]);
    assert!((hits[0].2.t.to_f32() - 0.15).abs() < 2e-3);
}

#[test]
fn contact_exactly_at_tolerance_is_a_hit_at_t_zero() {
    // Dyadic numbers: y = 1, r = 0.5 -> gap 0.5; tolerance 0.5 -> contact immediately.
    let c = SdfCcdConfig {
        tolerance: 0.5,
        ..cfg()
    };
    let toi = sphere_trace_sdf(
        v(0.0, 1.0, 0.0),
        v(0.0, -10.0, 0.0),
        Fix128::from_f32(0.5),
        &stat(plane()),
        &c,
    )
    .unwrap();
    assert_eq!(toi.t, Fix128::ZERO);
    // ray_march_sdf is strict (`dist < tolerance`, as the config doc says): a point
    // exactly `tolerance` away is not yet a hit and advances to the surface side.
    let toi = ray_march_sdf(
        v(0.0, 0.5, 0.0),
        v(0.0, -1.0, 0.0),
        Fix128::from_int(5),
        &stat(plane()),
        &c,
    )
    .unwrap();
    assert!(toi.t > Fix128::ZERO);
}

#[test]
fn ray_march_step_is_scaled_by_step_safety() {
    // Distance 4 to the sphere. With safety 0.9 the first step is 3.6 and the
    // second leaves a gap of 0.04 (4 -> 3.6 -> 3.96): two iterations are not enough.
    // A step that ignored the factor (1.0) would hit on the second iteration.
    let s = stat(unit_sphere());
    let two = SdfCcdConfig {
        max_iterations: 2,
        ..cfg()
    };
    assert!(ray_march_sdf(
        v(-5.0, 0.0, 0.0),
        Vec3Fix::UNIT_X,
        Fix128::from_int(20),
        &s,
        &two
    )
    .is_none());
    let five = SdfCcdConfig {
        max_iterations: 5,
        ..cfg()
    };
    assert!(ray_march_sdf(
        v(-5.0, 0.0, 0.0),
        Vec3Fix::UNIT_X,
        Fix128::from_int(20),
        &s,
        &five
    )
    .is_some());
}

#[test]
fn velocity_threshold_gates_on_the_bodys_speed_not_the_displacement() {
    let cols = [stat(plane())];
    let down = v(0.0, -10.0, 0.0);
    let at_threshold =
        RigidBody::new_dynamic(v(0.0, 5.0, 0.0), Fix128::ONE).with_velocity(v(0.0, -5.0, 0.0));
    let slow =
        RigidBody::new_dynamic(v(0.0, 5.0, 0.0), Fix128::ONE).with_velocity(v(0.0, -4.0, 0.0));
    let bodies = [at_threshold, slow];
    let hits = batch_sphere_trace_sdf(&bodies, &[down, down], Fix128::from_f32(0.5), &cols, &cfg());
    // speed == threshold (5) is swept; speed 4 < 5 is skipped even with a long displacement.
    assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![0]);
}
