//! Oracles for the world sweeps: `PhysicsWorld::time_of_impact` (a body's
//! translation against the geometry the world collides with) and
//! `PhysicsWorld::move_sdf_character` / `PhysicsWorld::sdf_field` (an
//! `SdfCharacter` against the union of the world's SDF colliders).
//!
//! # Expected values
//!
//! Every expected value is written from the geometry by hand; the closed form is
//! in a comment next to each assertion.
//!
//! Time of impact (`ccd::TOI::t` is a fraction of the step, `0` = start, `1` =
//! end): a body of collision radius `r` at `c` moving with velocity `v` for `dt`
//! travels `|v|·dt`. It first touches
//!
//! - a sphere of radius `R` whose centre is `g + r + R` ahead on its line after
//!   travelling `g`: `t = g / (|v|·dt)`;
//! - the plane `y = 0` from height `h` moving straight down after travelling
//!   `h − r`: `t = (h − r) / (|v|·dt)`.
//!
//! The contact point is on the obstacle's surface and the normal is the
//! obstacle's surface normal there, toward the moving body (the convention of
//! `ccd::sphere_plane_toi`, and of `ccd::sphere_sphere_toi` with the obstacle as
//! sphere A and the moving body as sphere B).
//!
//! SDF character (`SdfCharacter::move_and_slide`, module doc steps 2-3): a centre
//! at signed distance `d < radius` is pushed out along the normal by
//! `radius − d + skin_width`, so it ends at distance `radius + skin_width` from a
//! plane.
//!
//! # Tolerances
//!
//! - `EXACT = 1e-12`: closed-form sphere and plane casts (`Fix128`).
//! - `SDF_T`: an SDF obstacle is sphere-traced in `f32` and stops within
//!   `SdfCcdConfig::tolerance` of the surface.
//! - `F32 = 1e-6`: the character's `f32` arithmetic.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::ccd::{sphere_sphere_toi, TOI};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;

const EXACT: f64 = 1e-12;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig::default())
}

fn ball(w: &mut PhysicsWorld, pos: Vec3Fix, r: f64) -> usize {
    w.add_body_with_radius(RigidBody::new(pos, Fix128::ONE), fx(r))
}

fn obstacle(w: &mut PhysicsWorld, pos: Vec3Fix, r: f64) -> usize {
    w.add_body_with_radius(RigidBody::new_static(pos), fx(r))
}

fn ground_plane(w: &mut PhysicsWorld) -> usize {
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )))
}

fn toi(w: &PhysicsWorld, body: usize, v: [f64; 3], dt: f64) -> Option<TOI> {
    w.time_of_impact(body, v3(v[0], v[1], v[2]), fx(dt), &RayFilter::default())
}

fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
}

fn assert_toi(hit: TOI, t: f64, point: [f64; 3], normal: [f64; 3], tol: f64) {
    assert!(
        (hit.t.to_f64() - t).abs() <= tol,
        "t = {} (want {t})",
        hit.t
    );
    assert!(
        close(f3(hit.point), point, tol),
        "point = {:?} (want {point:?})",
        f3(hit.point)
    );
    assert!(
        close(f3(hit.normal), normal, tol),
        "normal = {:?} (want {normal:?})",
        f3(hit.normal)
    );
}

// ============================================================ time of impact

#[test]
fn two_spheres_approaching_touch_at_gap_over_travel() {
    let mut w = world();
    let a = ball(&mut w, v3(0.0, 0.0, 0.0), 0.5);
    obstacle(&mut w, v3(3.0, 0.0, 0.0), 0.25);
    // oracle: gap g = 3 − 0.5 − 0.25 = 2.25, travel |v|·dt = 3·1.5 = 4.5,
    // t = 2.25 / 4.5 = 0.5. The contact is on B at 3 − 0.25 = 2.75, normal −X
    // (from B toward A).
    let hit = toi(&w, a, [3.0, 0.0, 0.0], 1.5).expect("hit");
    assert_toi(hit, 0.5, [2.75, 0.0, 0.0], [-1.0, 0.0, 0.0], EXACT);
}

#[test]
fn sphere_falling_onto_plane_touches_at_height_minus_radius_over_travel() {
    let mut w = world();
    ground_plane(&mut w);
    let a = ball(&mut w, v3(1.0, 2.0, -1.0), 0.5);
    // oracle: (h − r) / (|v|·dt) = (2 − 0.5) / (3·1) = 0.5; contact below the
    // centre on the plane, normal +Y.
    let hit = toi(&w, a, [0.0, -3.0, 0.0], 1.0).expect("hit");
    assert_toi(hit, 0.5, [1.0, 0.0, -1.0], [0.0, 1.0, 0.0], EXACT);
}

#[test]
fn nearest_of_two_obstacles_is_reported() {
    let mut w = world();
    let a = ball(&mut w, v3(0.0, 0.0, 0.0), 0.5);
    obstacle(&mut w, v3(6.0, 0.0, 0.0), 0.5);
    obstacle(&mut w, v3(3.0, 0.0, 0.0), 0.5);
    // oracle: the nearer one at gap 3 − 1 = 2 over travel 8: t = 0.25.
    let hit = toi(&w, a, [8.0, 0.0, 0.0], 1.0).expect("hit");
    assert_toi(hit, 0.25, [2.5, 0.0, 0.0], [-1.0, 0.0, 0.0], EXACT);
}

#[test]
fn passing_beside_an_obstacle_misses() {
    let mut w = world();
    let a = ball(&mut w, v3(0.0, 0.0, 0.0), 0.5);
    // oracle: lateral offset 0.8 > r + R = 0.75: the line never comes within 0.75.
    obstacle(&mut w, v3(3.0, 0.0, 0.8), 0.25);
    assert_eq!(toi(&w, a, [10.0, 0.0, 0.0], 1.0), None);
}

#[test]
fn obstacle_beyond_the_step_travel_is_not_hit() {
    let mut w = world();
    let a = ball(&mut w, v3(0.0, 0.0, 0.0), 0.5);
    obstacle(&mut w, v3(3.0, 0.0, 0.0), 0.25);
    // oracle: gap 2.25 > travel 2·1 = 2.
    assert_eq!(toi(&w, a, [2.0, 0.0, 0.0], 1.0), None);
    // oracle: travel 2.25 exactly: touches at the end of the step, t = 1.
    let hit = toi(&w, a, [2.25, 0.0, 0.0], 1.0).expect("hit at the end");
    assert_toi(hit, 1.0, [2.75, 0.0, 0.0], [-1.0, 0.0, 0.0], EXACT);
}

#[test]
fn moving_away_is_not_a_hit() {
    let mut w = world();
    let a = ball(&mut w, v3(0.0, 2.0, 0.0), 0.5);
    obstacle(&mut w, v3(3.0, 2.0, 0.0), 0.25);
    ground_plane(&mut w);
    // oracle: moving −X away from the obstacle and parallel to the plane.
    assert_eq!(toi(&w, a, [-5.0, 0.0, 0.0], 1.0), None);
    // oracle: moving up away from the plane, beside the obstacle's line.
    assert_eq!(toi(&w, a, [0.0, 5.0, 0.0], 1.0), None);
}

#[test]
fn touching_or_overlapping_at_the_start_is_time_zero() {
    let mut w = world();
    let a = ball(&mut w, v3(0.0, 0.0, 0.0), 0.5);
    obstacle(&mut w, v3(0.75, 0.0, 0.0), 0.25);
    // oracle: centres 0.75 = r + R apart, closing: touching now, t = 0, contact
    // at B's surface 0.75 − 0.25 = 0.5, normal −X.
    let hit = toi(&w, a, [1.0, 0.0, 0.0], 1.0).expect("touching");
    assert_toi(hit, 0.0, [0.5, 0.0, 0.0], [-1.0, 0.0, 0.0], EXACT);

    let mut w = world();
    let a = ball(&mut w, v3(0.0, 0.0, 0.0), 0.5);
    obstacle(&mut w, v3(0.5, 0.0, 0.0), 0.25);
    // oracle: already overlapping (0.5 < 0.75): t = 0 in either direction, as
    // `sphere_sphere_toi` reports an overlap.
    for v in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]] {
        let hit = toi(&w, a, v, 1.0).expect("overlapping");
        assert!(hit.t.is_zero(), "t = {}", hit.t);
    }
}

#[test]
fn the_moving_body_does_not_hit_itself() {
    let mut w = world();
    let a = ball(&mut w, v3(0.0, 0.0, 0.0), 0.5);
    // oracle: nothing else in the world.
    assert_eq!(toi(&w, a, [5.0, 0.0, 0.0], 1.0), None);
    // A filter that already excludes another body still excludes the mover.
    let other = obstacle(&mut w, v3(0.0, 10.0, 0.0), 0.5);
    let f = RayFilter::default().excluding_body(other);
    assert_eq!(
        w.time_of_impact(a, v3(5.0, 0.0, 0.0), Fix128::ONE, &f),
        None
    );
}

#[test]
fn the_filter_decides_what_is_seen() {
    let mut w = world();
    ground_plane(&mut w);
    let a = ball(&mut w, v3(0.0, 2.0, 0.0), 0.5);
    let no_static = RayFilter::default().with_static(false);
    // oracle: the plane is the only obstacle; without static colliders nothing.
    assert_eq!(
        w.time_of_impact(a, v3(0.0, -3.0, 0.0), Fix128::ONE, &no_static),
        None
    );
}

/// Mover centre and radius, obstacle centre and radius, velocity, `dt`.
type Pair = ([f64; 3], f64, [f64; 3], f64, [f64; 3], f64);

#[test]
fn parity_with_sphere_sphere_toi() {
    // Obstacle as sphere A (still), the mover as sphere B with the step's
    // displacement `v·dt` as its velocity.
    let cases: [Pair; 4] = [
        (
            [0.0, 0.0, 0.0],
            0.5,
            [3.0, 0.0, 0.0],
            0.25,
            [3.0, 0.0, 0.0],
            1.5,
        ),
        (
            [0.0, 0.0, 0.0],
            0.5,
            [2.0, 0.5, 0.0],
            0.5,
            [4.0, 0.0, 0.0],
            1.0,
        ),
        (
            [1.0, -2.0, 0.5],
            0.3,
            [3.0, 1.0, 2.0],
            0.7,
            [2.0, 3.0, 1.5],
            0.75,
        ),
        (
            [0.0, 0.0, 0.0],
            0.25,
            [0.0, 0.0, 5.0],
            1.0,
            [0.5, 0.25, 4.0],
            2.0,
        ),
    ];
    for (c, r, oc, or, v, dt) in cases {
        let mut w = world();
        let a = ball(&mut w, v3(c[0], c[1], c[2]), r);
        obstacle(&mut w, v3(oc[0], oc[1], oc[2]), or);
        let world_hit = toi(&w, a, v, dt);
        let disp = v3(v[0] * dt, v[1] * dt, v[2] * dt);
        let pair = sphere_sphere_toi(
            v3(oc[0], oc[1], oc[2]),
            fx(or),
            Vec3Fix::ZERO,
            v3(c[0], c[1], c[2]),
            fx(r),
            disp,
        );
        match (world_hit, pair) {
            (Some(h), Some(p)) => assert_toi(h, p.t.to_f64(), f3(p.point), f3(p.normal), 1e-9),
            (None, None) => {}
            other => panic!("disagree: {other:?}"),
        }
        assert!(pair.is_some(), "each case is a hit");
    }
}

#[test]
fn shaped_body_uses_its_bounding_sphere() {
    let mut w = world();
    ground_plane(&mut w);
    let a = ball(&mut w, v3(0.0, 3.3, 0.0), 0.1);
    assert!(w.set_body_shape(
        a,
        &Shape::Box {
            half_extents: v3(0.3, 0.4, 1.2)
        }
    ));
    // oracle: bounding radius |(0.3, 0.4, 1.2)| = 1.3, height 3.3: travel to
    // touch 3.3 − 1.3 = 2 of 4: t = 0.5.
    let hit = toi(&w, a, [0.0, -4.0, 0.0], 1.0).expect("hit");
    assert_toi(hit, 0.5, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1e-9);
}

#[test]
fn degenerate_inputs_give_none() {
    let mut w = world();
    let a = ball(&mut w, v3(0.0, 0.0, 0.0), 0.5);
    obstacle(&mut w, v3(3.0, 0.0, 0.0), 0.25);
    let no_radius = w.add_body(RigidBody::new(v3(0.0, 5.0, 0.0), Fix128::ONE));
    // invalid index
    assert_eq!(toi(&w, 99, [3.0, 0.0, 0.0], 1.0), None);
    // zero velocity: nothing is swept
    assert_eq!(toi(&w, a, [0.0, 0.0, 0.0], 1.0), None);
    // dt ≤ 0
    assert_eq!(toi(&w, a, [3.0, 0.0, 0.0], 0.0), None);
    assert_eq!(toi(&w, a, [3.0, 0.0, 0.0], -1.0), None);
    // dt < 0 is not a backward sweep: an obstacle behind is not reported either.
    obstacle(&mut w, v3(-3.0, 0.0, 0.0), 0.25);
    assert_eq!(toi(&w, a, [3.0, 0.0, 0.0], -1.0), None);
    // a body without collision geometry collides with nothing
    assert_eq!(toi(&w, no_radius, [0.0, -10.0, 0.0], 1.0), None);
    // empty world apart from the mover
    let mut e = world();
    let m = ball(&mut e, Vec3Fix::ZERO, 0.5);
    assert_eq!(toi(&e, m, [1.0, 2.0, 3.0], 1.0), None);
}

#[cfg(feature = "std")]
mod sdf {
    use super::*;
    use alice_physics::math::QuatFix;
    use alice_physics::sdf_character::SdfCharacter;
    use alice_physics::sdf_collider::{ClosureSdf, SdfCollider, SdfField};

    const F32: f32 = 1e-6;

    fn unit_sphere_field() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )
    }

    /// Solid below `y = 0`.
    fn floor_field() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    /// Solid beyond `x = 5`.
    fn wall_field() -> ClosureSdf {
        ClosureSdf::new(|x, _y, _z| 5.0 - x, |_x, _y, _z| (-1.0, 0.0, 0.0))
    }

    fn add_static(w: &mut PhysicsWorld, field: ClosureSdf, at: Vec3Fix) {
        w.sdf_colliders.push(SdfCollider::new_static(
            Box::new(field),
            at,
            QuatFix::IDENTITY,
        ));
    }

    #[test]
    fn time_of_impact_against_an_sdf_collider() {
        let mut w = world();
        add_static(&mut w, unit_sphere_field(), Vec3Fix::ZERO);
        let a = ball(&mut w, v3(-5.0, 0.0, 0.0), 0.5);
        // oracle: the centre reaches |c| = 1 + 0.5 after travelling 3.5 of 7:
        // t = 0.5, normal −X.
        let hit = toi(&w, a, [7.0, 0.0, 0.0], 1.0).expect("hit");
        let tol = f64::from(RayFilter::default().sdf.tolerance) + 1e-5;
        assert!((hit.t.to_f64() - 0.5).abs() <= tol / 7.0, "t = {}", hit.t);
        assert!(close(f3(hit.normal), [-1.0, 0.0, 0.0], 1e-4));
    }

    #[test]
    fn standing_character_rests_radius_plus_skin_above_the_ground() {
        let mut w = world();
        add_static(&mut w, floor_field(), Vec3Fix::ZERO);
        let mut ch = SdfCharacter::new([0.0, 0.3, 0.0], 0.35, 1.8);
        let out = w.move_sdf_character(&mut ch, 1.0 / 60.0, [0.0, 0.0, 0.0]);
        // oracle: d = 0.3 < r: pushed to y = r + skin = 0.35 + 1e-4.
        let want = 0.35 + ch.skin_width;
        assert!(out.converged);
        assert!(
            (ch.position[1] - want).abs() < F32,
            "y = {}",
            ch.position[1]
        );
        // oracle: standing still at rest stays there.
        for _ in 0..10 {
            w.move_sdf_character(&mut ch, 1.0 / 60.0, [0.0, 0.0, 0.0]);
        }
        assert!(
            (ch.position[1] - want).abs() < F32,
            "y = {}",
            ch.position[1]
        );
    }

    #[test]
    fn falling_character_lands_on_a_translated_ground_collider() {
        let mut w = world();
        // the floor field placed 1 up: ground at y = 1.
        add_static(&mut w, floor_field(), v3(0.0, 1.0, 0.0));
        let mut ch = SdfCharacter::new([0.0, 4.0, 0.0], 0.35, 1.8);
        let dt = 1.0 / 60.0;
        for _ in 0..240 {
            ch.apply_gravity([0.0, -9.81, 0.0], dt);
            w.move_sdf_character(&mut ch, dt, [0.0, 0.0, 0.0]);
        }
        // oracle: resting at 1 + r + skin, fall absorbed.
        let want = 1.0 + 0.35 + ch.skin_width;
        assert!(
            (ch.position[1] - want).abs() < 1e-5,
            "y = {}",
            ch.position[1]
        );
        assert!(ch.velocity[1].abs() < 1e-6, "vy = {}", ch.velocity[1]);
    }

    #[test]
    fn walking_into_a_wall_stops_radius_plus_skin_short() {
        let mut w = world();
        // floor first, wall second: the wall is only seen through the union.
        add_static(&mut w, floor_field(), Vec3Fix::ZERO);
        add_static(&mut w, wall_field(), Vec3Fix::ZERO);
        let r = 0.35_f32;
        let mut ch = SdfCharacter::new([4.0, r + 1e-4, 0.0], r, 1.8);
        let out = w.move_sdf_character(&mut ch, 1.0 / 60.0, [2.0, 0.0, 0.0]);
        // oracle: x = 6 is 1 inside the wall (d = −1): pushed back to
        // x = 5 − (r + skin).
        assert!(out.converged);
        let want = 5.0 - (r + ch.skin_width);
        assert!(
            (ch.position[0] - want).abs() < F32,
            "x = {}",
            ch.position[0]
        );
        // and still on the floor.
        assert!(
            (ch.position[1] - (r + 1e-4)).abs() < F32,
            "y = {}",
            ch.position[1]
        );
    }

    #[test]
    fn parity_with_move_and_slide_on_the_single_field() {
        let mut w = world();
        add_static(&mut w, floor_field(), Vec3Fix::ZERO);
        let direct = floor_field();
        for (start, disp) in [
            ([0.0, 0.3, 0.0], [0.5, -0.2, 0.25]),
            ([1.0, 2.0, -3.0], [0.0, -2.5, 0.0]),
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ] {
            let mut ch = SdfCharacter::new(start, 0.35, 1.8);
            let want = ch.move_and_slide(&direct, disp);
            // through the world field: the same call on the union of colliders
            assert_eq!(ch.move_and_slide(&w.sdf_field(), disp), want);
            // through the world entry point (zero velocity: displacement = control)
            let out = w.move_sdf_character(&mut ch, 1.0 / 60.0, disp);
            assert_eq!(out, want);
            assert_eq!(ch.position, want.resolved_position());
        }
    }

    #[test]
    fn parity_with_step_on_the_single_field_over_frames() {
        let mut w = world();
        add_static(&mut w, floor_field(), Vec3Fix::ZERO);
        let direct = floor_field();
        let mut a = SdfCharacter::new([0.0, 3.0, 0.0], 0.35, 1.8);
        let mut b = a;
        let dt = 1.0 / 60.0;
        for i in 0..120 {
            let control = [0.01 * (i % 7) as f32, 0.0, -0.005];
            a.apply_gravity([0.0, -9.81, 0.0], dt);
            b.apply_gravity([0.0, -9.81, 0.0], dt);
            let oa = w.move_sdf_character(&mut a, dt, control);
            let ob = b.step(&direct, dt, control);
            assert_eq!(oa, ob, "frame {i}");
            assert_eq!(a, b, "frame {i}");
        }
    }

    #[test]
    fn the_union_is_the_minimum_over_colliders() {
        let mut w = world();
        add_static(&mut w, floor_field(), Vec3Fix::ZERO);
        add_static(&mut w, wall_field(), Vec3Fix::ZERO);
        let f = w.sdf_field();
        // oracle: at (4, 3, 0) floor 3, wall 1: min 1, normal of the wall.
        assert!((f.distance(4.0, 3.0, 0.0) - 1.0).abs() < F32);
        assert_eq!(f.normal(4.0, 3.0, 0.0), (-1.0, 0.0, 0.0));
        // oracle: at (0, 0.5, 0) floor 0.5, wall 5: min 0.5, floor normal.
        assert!((f.distance(0.0, 0.5, 0.0) - 0.5).abs() < F32);
        assert_eq!(f.normal(0.0, 0.5, 0.0), (0.0, 1.0, 0.0));
    }

    #[test]
    fn a_scaled_collider_is_measured_in_world_units() {
        let mut w = world();
        w.sdf_colliders.push(
            SdfCollider::new_static(
                Box::new(unit_sphere_field()),
                Vec3Fix::ZERO,
                QuatFix::IDENTITY,
            )
            .with_scale(fx(2.0)),
        );
        let f = w.sdf_field();
        // oracle: the unit sphere scaled by 2 is a sphere of radius 2: at x = 3
        // the distance is 3 − 2 = 1 (local 3/2 − 1 = 0.5, times the scale 2).
        assert!((f.distance(3.0, 0.0, 0.0) - 1.0).abs() < F32);
        // a character of radius 0.35 dropped onto its top rests at 2 + r + skin.
        let mut ch = SdfCharacter::new([0.0, 2.2, 0.0], 0.35, 1.8);
        w.move_sdf_character(&mut ch, 1.0 / 60.0, [0.0, 0.0, 0.0]);
        let want = 2.0 + 0.35 + ch.skin_width;
        assert!(
            (ch.position[1] - want).abs() < 1e-5,
            "y = {}",
            ch.position[1]
        );
    }

    #[test]
    fn no_sdf_colliders_moves_freely() {
        let w = world();
        let mut ch = SdfCharacter::new([1.0, 2.0, 3.0], 0.35, 1.8);
        ch.velocity = [1.0, 0.0, 0.0];
        let out = w.move_sdf_character(&mut ch, 0.5, [0.0, 0.0, 0.25]);
        // oracle: nothing to resolve: position + v·dt + control.
        assert!(out.converged);
        assert_eq!(out.iterations, 0);
        assert_eq!(ch.position, [1.5, 2.0, 3.25]);
        assert_eq!(ch.velocity, [1.0, 0.0, 0.0]);
        // the empty union is infinitely far
        assert_eq!(w.sdf_field().distance(0.0, 0.0, 0.0), f32::INFINITY);
    }

    #[test]
    fn zero_or_negative_dt_is_passed_through_as_step_does() {
        let mut w = world();
        add_static(&mut w, floor_field(), Vec3Fix::ZERO);
        let direct = floor_field();
        for dt in [0.0_f32, -0.1] {
            let mut a = SdfCharacter::new([0.0, 1.0, 0.0], 0.35, 1.8);
            a.velocity = [2.0, -1.0, 0.0];
            let mut b = a;
            let oa = w.move_sdf_character(&mut a, dt, [0.1, 0.0, 0.0]);
            let ob = b.step(&direct, dt, [0.1, 0.0, 0.0]);
            assert_eq!(oa, ob);
            assert_eq!(a, b);
        }
        // oracle: dt = 0 moves by the control only.
        let mut c = SdfCharacter::new([0.0, 1.0, 0.0], 0.35, 1.8);
        c.velocity = [2.0, -1.0, 0.0];
        w.move_sdf_character(&mut c, 0.0, [0.1, 0.0, 0.0]);
        assert_eq!(c.position, [0.1, 1.0, 0.0]);
    }
}
