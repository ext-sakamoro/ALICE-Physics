//! Oracles that pin the margins of world shape queries and the character
//! controller: a cast that starts overlapping a plane by one capsule end, an SDF
//! attached to an excluded body, the skin-width margins of the sweep, the ground
//! probe and the step, the slope limit at equality, the step requiring ground,
//! a capsule shorter than its diameter, and the velocity a move writes back.
//!
//! # Expected values
//!
//! Every expected value is written from the geometry by hand; the closed form is
//! in a comment next to each assertion. Nothing here calls the code under test to
//! make an expected value.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::character::{CharacterConfig, CharacterController};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;
use alice_physics::world_shape_query::WorldShapeHit;

const EXACT: f64 = 1e-12;
const MOVE_TOL: f64 = 1e-8;
const STAND: f64 = 0.91;
const R: f64 = 0.3;
const SKIN: f64 = 0.01;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn p3(p: [f64; 3]) -> Vec3Fix {
    v3(p[0], p[1], p[2])
}

fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig::default())
}

fn capsule_cast(
    w: &PhysicsWorld,
    a: [f64; 3],
    b: [f64; 3],
    r: f64,
    d: [f64; 3],
    max: f64,
) -> Option<WorldShapeHit> {
    w.cast_capsule(p3(a), p3(b), fx(r), p3(d), fx(max), &RayFilter::default())
}

#[track_caller]
fn assert_vec(got: Vec3Fix, want: [f64; 3], tol: f64, what: &str) {
    let g = f3(got);
    for k in 0..3 {
        assert!(
            (g[k] - want[k]).abs() < tol,
            "{what} {g:?} but the closed form is {want:?}"
        );
    }
}

#[test]
fn a_capsule_overlapping_a_plane_by_one_end_starts_overlapping() {
    let mut w = world();
    let p = w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    // oracle: the lower end (0, 0.2, 0) is 0.2 < 0.5 from the plane: the capsule
    // overlaps at the start, t = 0, normal −direction, point the segment midpoint.
    let h = capsule_cast(
        &w,
        [0.0, 0.2, 0.0],
        [0.0, 3.0, 0.0],
        0.5,
        [1.0, 0.0, 0.0],
        10.0,
    )
    .expect("hit");
    assert_eq!(h.target, RayTarget::StaticCollider(p));
    assert_eq!(h.t, Fix128::ZERO);
    assert_vec(h.normal, [-1.0, 0.0, 0.0], EXACT, "normal");
    assert_vec(h.point, [0.0, 1.6, 0.0], EXACT, "point");
}

#[cfg(feature = "std")]
#[test]
fn an_sdf_on_an_excluded_body_is_not_seen() {
    use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
    let mut w = world();
    let body = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.sdf_colliders.push(SdfCollider::new_dynamic(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )),
        body,
    ));
    let filter = RayFilter::default().excluding_body(body);
    // oracle: the SDF belongs to the excluded body: nothing is hit.
    assert!(w
        .cast_sphere(
            v3(-10.0, 0.0, 0.0),
            fx(0.5),
            v3(1.0, 0.0, 0.0),
            fx(100.0),
            &filter
        )
        .is_none());
    // oracle: without the filter the unit sphere field is hit near t = 8.5
    // (field tolerance 1e-3).
    let h = w
        .cast_sphere(
            v3(-10.0, 0.0, 0.0),
            fx(0.5),
            v3(1.0, 0.0, 0.0),
            fx(100.0),
            &RayFilter::default(),
        )
        .expect("hit");
    assert!((h.t.to_f64() - 8.5).abs() < 2e-3, "t = {}", h.t.to_f64());
}

#[track_caller]
fn assert_pos(got: Vec3Fix, want: [f64; 3], tol: f64) {
    let g = f3(got);
    for k in 0..3 {
        assert!(
            (g[k] - want[k]).abs() < tol,
            "position {g:?} but the closed form is {want:?}"
        );
    }
}

fn floor_plane(w: &mut PhysicsWorld) {
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
}

fn static_box(w: &mut PhysicsWorld, center: [f64; 3], half: [f64; 3]) -> usize {
    let i = w.add_body(RigidBody::new_static(v3(center[0], center[1], center[2])));
    assert!(w.set_body_shape(
        i,
        &Shape::Box {
            half_extents: v3(half[0], half[1], half[2]),
        }
    ));
    i
}

fn ctrl(p: [f64; 3], config: CharacterConfig) -> CharacterController {
    CharacterController::new(v3(p[0], p[1], p[2]), config)
}

fn settled(w: &PhysicsWorld, p: [f64; 3], config: CharacterConfig) -> CharacterController {
    let mut c = ctrl(p, config);
    w.move_character(&mut c, Vec3Fix::ZERO);
    assert!(c.grounded, "the scene's floor holds the controller");
    c
}

#[test]
fn a_capsule_shorter_than_its_diameter_is_a_sphere() {
    let mut w = world();
    floor_plane(&mut w);
    let config = CharacterConfig {
        height: fx(0.2),
        ..CharacterConfig::default()
    };
    // oracle: height 0.2 < 2r is a sphere of radius r: falling onto y = 0 it
    // stops with its centre at r + s = 0.31.
    let mut c = ctrl([0.0, 2.0, 0.0], config);
    let res = w.move_character(&mut c, v3(0.0, -5.0, 0.0));
    assert_pos(res.position, [0.0, R + SKIN, 0.0], EXACT);
}

#[test]
fn the_sweep_looks_a_skin_width_past_the_move() {
    let mut w = world();
    floor_plane(&mut w);
    // the wall face at x = 0.3 + 1.005: the capsule surface is 1.005 from it
    static_box(&mut w, [R + 1.005 + 1.0, 2.0, 0.0], [1.0, 2.0, 5.0]);
    let mut c = settled(&w, [0.0, STAND, 0.0], CharacterConfig::default());
    // oracle: a move of 1 sweeps 1 + s = 1.01 and sees the wall at 1.005: it stops
    // a skin width short, x = 1.005 − 0.01 = 0.995 (not 1, which would leave a gap
    // 0.005 below the skin width).
    let res = w.move_character(&mut c, v3(1.0, 0.0, 0.0));
    assert_pos(res.position, [0.995, STAND, 0.0], MOVE_TOL);
}

#[test]
fn the_ground_probe_reaches_a_skin_width_below() {
    let mut w = world();
    floor_plane(&mut w);
    let config = CharacterConfig {
        ground_probe_distance: Fix128::ZERO,
        ..CharacterConfig::default()
    };
    // oracle: resting a skin width above the floor, a probe of 0 + s reaches it
    // exactly (t = s): grounded.
    let y = config.height.half() + config.skin_width;
    let mut c = CharacterController::new(Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO), config);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert!(res.grounded);
}

#[test]
fn a_slope_at_exactly_the_limit_is_walkable() {
    let config = CharacterConfig::default();
    let (sin, cos) = config.max_slope_angle.sin_cos();
    // a plane through the origin whose normal makes exactly the slope limit with
    // +Y: n = (−sin a, cos a, 0), the components the limit is computed from
    let mut plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
    plane.normal = Vec3Fix::new(-sin, cos, Fix128::ZERO);
    let mut w = world();
    w.add_static_collider(StaticCollider::Plane(plane));
    // oracle: the lower end (0, y − 0.6, 0) is (y − 0.6)·cos a from the plane;
    // y = 0.6 + (r + s)/cos a leaves the skin gap, and the probe meets the plane
    // with normal n, n·Y = cos a: at most the limit, walkable, grounded.
    let y = fx(0.6) + fx(R + SKIN) / cos;
    let mut c = CharacterController::new(Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO), config);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert!(res.grounded);
}

#[test]
fn an_airborne_controller_does_not_step_up() {
    let mut w = world();
    floor_plane(&mut w);
    // a 0.2 high block (below the 0.3 step height) whose face is x = 1
    static_box(&mut w, [2.0, 0.1, 0.0], [1.0, 0.1, 5.0]);
    // oracle: a fresh controller is not grounded, so no step is tried: the capsule
    // stops at the block's top edge and stays at the standing height.
    let mut c = ctrl([0.0, STAND, 0.0], CharacterConfig::default());
    let res = w.move_character(&mut c, v3(2.0, 0.0, 0.0));
    assert!(
        (res.position.y.to_f64() - STAND).abs() < MOVE_TOL,
        "{:?}",
        f3(res.position)
    );
    assert!(res.position.x.to_f64() < 1.0, "{:?}", f3(res.position));
    // oracle: grounded first, the same move steps onto the block: centre at
    // x = 2 on the top y = 0.2, y = 0.2 + h/2 + s.
    let mut c = settled(&w, [0.0, STAND, 0.0], CharacterConfig::default());
    let res = w.move_character(&mut c, v3(2.0, 0.0, 0.0));
    assert_pos(res.position, [2.0, 0.2 + STAND, 0.0], MOVE_TOL);
}

#[test]
fn a_step_that_gains_no_more_than_a_skin_width_is_not_taken() {
    let mut w = world();
    floor_plane(&mut w);
    // a 0.4 high sliver x ∈ [1, 1.005] against a tall wall from x = 1.005
    static_box(&mut w, [1.0025, 0.2, 0.0], [0.0025, 0.2, 5.0]);
    static_box(&mut w, [2.005, 2.0, 0.0], [1.0, 2.0, 5.0]);
    let config = CharacterConfig {
        step_height: fx(0.5),
        ..CharacterConfig::default()
    };
    let mut c = settled(&w, [0.0, STAND, 0.0], config);
    // oracle: the plain move stops at the sliver face, x = 1 − r − s = 0.69; the
    // raised move only reaches the wall, x = 1.005 − r − s = 0.695, 0.005 < s
    // further: not taken.
    let res = w.move_character(&mut c, v3(2.0, 0.0, 0.0));
    assert_pos(res.position, [1.0 - R - SKIN, STAND, 0.0], MOVE_TOL);
}

#[test]
fn a_step_down_lands_up_to_two_skin_widths_below_the_rise() {
    let mut w = world();
    // left floor top y = 0 for x ≤ 1.2, right floor top y = −0.005 beyond it,
    // a 0.2 high bump x ∈ [1, 1.1] on the left floor
    static_box(&mut w, [-3.9, -0.5, 0.0], [5.1, 0.5, 5.0]);
    static_box(&mut w, [6.2, -0.505, 0.0], [5.0, 0.5, 5.0]);
    static_box(&mut w, [1.05, 0.1, 0.0], [0.05, 0.1, 5.0]);
    let mut c = settled(&w, [0.0, STAND, 0.0], CharacterConfig::default());
    // oracle: raised by 0.3 the capsule clears the bump and moves to x = 2; the
    // drop to the right floor is 0.3 + s + 0.005 ≤ 0.3 + 2s, so it lands a skin
    // width above y = −0.005: centre y = −0.005 + h/2 + s = 0.905.
    let res = w.move_character(&mut c, v3(2.0, 0.0, 0.0));
    assert_pos(res.position, [2.0, STAND - 0.005, 0.0], MOVE_TOL);
}

#[test]
fn the_controller_velocity_is_the_requested_displacement() {
    let mut w = world();
    let mut platform = RigidBody::new_kinematic(v3(0.0, -0.5, 0.0));
    platform.velocity = v3(2.0, 0.0, 0.0);
    let p = w.add_body(platform);
    assert!(w.set_body_shape(
        p,
        &Shape::Box {
            half_extents: v3(5.0, 0.5, 5.0),
        }
    ));
    let mut c = ctrl([0.0, STAND, 0.0], CharacterConfig::default());
    w.move_character_with_filter(&mut c, Vec3Fix::ZERO, &RayFilter::default());
    assert_eq!(c.platform_velocity, v3(2.0, 0.0, 0.0));
    // oracle: the move carries the platform velocity, but the controller's
    // velocity is the requested displacement alone (as `move_and_slide`).
    let res = w.move_character(&mut c, v3(1.0, 0.0, 0.0));
    assert_eq!(res.velocity, v3(1.0, 0.0, 0.0));
    assert_eq!(c.velocity, v3(1.0, 0.0, 0.0));
}

#[cfg(feature = "std")]
#[test]
fn a_capsule_cast_finds_the_nearest_of_two_sdf_bumps() {
    use alice_physics::math::QuatFix;
    use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
    // two spheres: A at (−8.75, 0, 0) radius 1 (top y = 1) and B at (0, 1, 0)
    // radius 0.1 (top y = 1.1), as one field (the minimum of the two)
    let dist_a = |x: f32, y: f32, z: f32| ((x + 8.75).powi(2) + y * y + z * z).sqrt() - 1.0;
    let dist_b = |x: f32, y: f32, z: f32| (x * x + (y - 1.0).powi(2) + z * z).sqrt() - 0.1;
    let mut w = world();
    w.sdf_colliders.push(SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            move |x, y, z| dist_a(x, y, z).min(dist_b(x, y, z)),
            move |x, y, z| {
                let (cx, cy) = if dist_a(x, y, z) < dist_b(x, y, z) {
                    (-8.75, 0.0)
                } else {
                    (0.0, 1.0)
                };
                let (dx, dy) = (x - cx, y - cy);
                let l = (dx * dx + dy * dy + z * z).sqrt();
                (dx / l, dy / l, z / l)
            },
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    // oracle: the segment x ∈ [−20, 20] at height y is nearest B's top when
    // y − 1.1 < y − 1 (A's top); falling, it is 0.5 from B at y = 1.6: t = 3.4
    // (field tolerance 1e-3 short of it at most), contact (0, 1.1, 0), +Y.
    let h = capsule_cast(
        &w,
        [-20.0, 5.0, 0.0],
        [20.0, 5.0, 0.0],
        0.5,
        [0.0, -1.0, 0.0],
        10.0,
    )
    .expect("hit");
    assert!(
        h.t.to_f64() <= 3.4 + 1e-6 && h.t.to_f64() > 3.4 - 2e-3,
        "t = {}",
        h.t.to_f64()
    );
    assert_vec(h.normal, [0.0, 1.0, 0.0], 1e-3, "normal");
}

fn torus_body(w: &mut PhysicsWorld) -> usize {
    w.add_shaped_body(
        &Shape::Torus {
            major_radius: fx(2.0),
            minor_radius: fx(0.5),
        },
        Fix128::ONE,
        Vec3Fix::ZERO,
    )
    .expect("valid shape")
}

#[test]
fn a_capsule_cast_onto_a_torus_between_ring_points_finds_the_tube() {
    let mut w = world();
    torus_body(&mut w);
    // oracle: the segment x = 1.8, z ∈ [0.55, 0.65] is nearest the ring
    // (radius 2 in XZ) at z = 0.65, horizontally 2 − √(1.8² + 0.65²) from it; the
    // capsule of radius 0.25 meets the tube (radius 0.5) when its height above
    // the ring plane is √(0.75² − off²): t = 5 − that.
    let off = 2.0 - (1.8f64 * 1.8 + 0.65 * 0.65).sqrt();
    let t = 5.0 - (0.75f64 * 0.75 - off * off).sqrt();
    let h = capsule_cast(
        &w,
        [1.8, 5.0, 0.55],
        [1.8, 5.0, 0.65],
        0.25,
        [0.0, -1.0, 0.0],
        10.0,
    )
    .expect("hit");
    assert!(
        (h.t.to_f64() - t).abs() < 1e-8,
        "t = {} but {t}",
        h.t.to_f64()
    );
}

#[test]
fn a_sphere_touching_the_inside_of_a_torus_crosses_the_hole() {
    let mut w = world();
    let b = torus_body(&mut w);
    // oracle: the sphere of radius 0.25 at (1.25, 0, 0) touches the inner wall
    // (x = 1.5); moving −X it leaves it and crosses the hole to the opposite
    // wall x = −1.5: centre at x = −1.25, t = 2.5, normal +X.
    let h = w
        .cast_sphere(
            v3(1.25, 0.0, 0.0),
            fx(0.25),
            v3(-1.0, 0.0, 0.0),
            fx(10.0),
            &RayFilter::default(),
        )
        .expect("hit");
    assert_eq!(h.target, RayTarget::Body(b));
    assert!((h.t.to_f64() - 2.5).abs() < 1e-8, "t = {}", h.t.to_f64());
    assert_vec(h.normal, [1.0, 0.0, 0.0], 1e-8, "normal");
}

#[cfg(feature = "std")]
fn sdf_world_of(
    f: impl Fn(f32, f32, f32) -> f32 + Send + Sync + 'static,
    n: impl Fn(f32, f32, f32) -> (f32, f32, f32) + Send + Sync + 'static,
) -> PhysicsWorld {
    use alice_physics::math::QuatFix;
    use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
    let mut w = world();
    w.sdf_colliders.push(SdfCollider::new_static(
        Box::new(ClosureSdf::new(f, n)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    w
}

#[cfg(feature = "std")]
#[test]
fn sdf_casts_keep_the_touching_and_stepping_rules() {
    let unit_normal = |x: f32, y: f32, z: f32| {
        let l = (x * x + y * y + z * z).sqrt();
        (x / l, y / l, z / l)
    };
    let w = sdf_world_of(|x, y, z| (x * x + y * y + z * z).sqrt() - 1.0, unit_normal);
    // oracle: touching the unit sphere field at (1.5, 0, 0) and moving away:
    // the field only grows, no hit.
    assert!(sphere_cast_at(&w, [1.5, 0.0, 0.0], [1.0, 0.0, 0.0], 10.0).is_none());

    // a field that overstates the distance 1.5 times: from (−3, 0, 0) the first
    // step, 1.5·2 − 0.5 = 2.5, lands inside the unit sphere at x = −0.5
    let w = sdf_world_of(
        |x, y, z| 1.5 * ((x * x + y * y + z * z).sqrt() - 1.0),
        unit_normal,
    );
    // oracle: a step that lands inside is reported there as a contact from
    // inside: t = 2.5, normal −direction.
    let h = sphere_cast_at(&w, [-3.0, 0.0, 0.0], [1.0, 0.0, 0.0], 10.0).expect("hit");
    assert!((h.t.to_f64() - 2.5).abs() < 1e-6, "t = {}", h.t.to_f64());
    assert_vec(h.normal, [-1.0, 0.0, 0.0], 1e-12, "normal");

    // a floor y = 0 and a wall x = 12 as one field, the sphere 0.0015 above the
    // floor: sphere tracing moves 0.0015 per step and runs out of steps
    let w = sdf_world_of(
        |x, y, _| y.min(12.0 - x),
        |x, y, _| {
            if y < 12.0 - x {
                (0.0, 1.0, 0.0)
            } else {
                (-1.0, 0.0, 0.0)
            }
        },
    );
    // oracle: out of steps, the cast still reports a contact, never after the
    // true one (the wall at t = 11.5).
    let h = sphere_cast_at(&w, [0.0, 0.5015, 0.0], [1.0, 0.0, 0.0], 20.0)
        .expect("a contact is reported");
    assert!(h.t.to_f64() <= 11.5 + 1e-3, "t = {}", h.t.to_f64());
}

#[cfg(feature = "std")]
fn sphere_cast_at(w: &PhysicsWorld, c: [f64; 3], d: [f64; 3], max: f64) -> Option<WorldShapeHit> {
    w.cast_sphere(p3(c), fx(0.5), p3(d), fx(max), &RayFilter::default())
}
