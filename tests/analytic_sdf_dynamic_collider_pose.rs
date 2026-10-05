//! Oracles for an SDF collider attached to a body following that body's pose.
//!
//! # What is measured
//!
//! `SdfCollider::new_dynamic` places the field's local origin at the body's
//! position and its local axes along the body's axes. The collider holds its
//! own copy of that pose; `SdfCollider::sync_to_body` (or
//! `sync_dynamic_sdf_colliders` for a set) copies the body's current pose in
//! and refreshes the cached inverse rotation. Every query after the copy has
//! a closed form in the body's frame:
//!
//! - a unit ball at body position `P`: distance `|q − P| − 1`, outward
//!   normal `(q − P) / |q − P|`, a point at `P + d·e` (`d < 1`) has depth
//!   `1 − d` along `e`;
//! - the local half-space `y < 0` on a body turned by `R`: world distance
//!   `(R⁻¹ (q − P))_y`, world normal `R ŷ`. Turned by +90° about `z`,
//!   `R ŷ = −x̂`, so the surface is the world plane `x = P_x` with the solid
//!   on the `+x` side.
//!
//! Before a sync the collider answers for the pose it last had (the world
//! origin, unrotated, for a fresh dynamic collider); that is asserted too,
//! because it is the reason the sync call is needed.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{
    collide_point_sdf, collide_sphere_sdf, detect_sdf_contacts, sync_dynamic_sdf_colliders,
    ClosureSdf, SdfCollider, SdfField, SdfFrame, SDF_STATIC,
};
use alice_physics::solver::RigidBody;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn unit_ball() -> Box<dyn SdfField> {
    Box::new(ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt().max(1e-6);
            (x / l, y / l, z / l)
        },
    ))
}

/// The local half-space `y < 0` (solid below `y = 0`).
fn half_space() -> Box<dyn SdfField> {
    Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0)))
}

fn close(a: Vec3Fix, b: [f64; 3], tol: f64) -> bool {
    let (x, y, z) = a.to_f32();
    (f64::from(x) - b[0]).abs() < tol
        && (f64::from(y) - b[1]).abs() < tol
        && (f64::from(z) - b[2]).abs() < tol
}

#[test]
fn a_ball_follows_the_translated_body() {
    let mut bodies = vec![RigidBody::new_dynamic(v3(10.0, -4.0, 2.0), Fix128::ONE)];
    let mut ball = SdfCollider::new_dynamic(unit_ball(), 0);

    // Unsynced: still the unit ball at the origin.
    assert!(collide_point_sdf(v3(0.5, 0.0, 0.0), &ball).is_some());
    assert!(collide_point_sdf(v3(10.5, -4.0, 2.0), &ball).is_none());

    assert!(ball.sync_to_body(&bodies));
    assert_eq!(ball.position, bodies[0].position);
    assert_eq!(
        ball.frame(),
        SdfFrame::new(bodies[0].position, bodies[0].rotation, Fix128::ONE)
    );

    // Point 0.5 inside along +x: depth 0.5, normal +x, surface point P + x̂.
    let c = collide_point_sdf(v3(10.5, -4.0, 2.0), &ball).expect("inside the moved ball");
    assert!(
        (c.depth.to_f64() - 0.5).abs() < 1e-6,
        "depth {}",
        c.depth.to_f64()
    );
    assert!(close(c.normal, [1.0, 0.0, 0.0], 1e-5));
    assert!(close(c.point_b, [11.0, -4.0, 2.0], 1e-5));
    // The old placement is empty now.
    assert!(collide_point_sdf(v3(0.5, 0.0, 0.0), &ball).is_none());
    // Distance |q − P| − 1 = 2 three units along −z: a sphere of radius 2.5
    // there penetrates by 0.5, normal −z.
    let c = collide_sphere_sdf(v3(10.0, -4.0, -1.0), fx(2.5), &ball).expect("reaches the ball");
    assert!(
        (c.depth.to_f64() - 0.5).abs() < 1e-6,
        "depth {}",
        c.depth.to_f64()
    );
    assert!(close(c.normal, [0.0, 0.0, -1.0], 1e-5));

    // The body moves on; the collider follows on the next sync only.
    bodies[0].position = v3(-3.0, 7.0, 0.0);
    assert!(collide_sphere_sdf(v3(-3.0, 8.25, 0.0), fx(0.5), &ball).is_none());
    assert!(ball.sync_to_body(&bodies));
    // Sphere of radius 0.5 centred 1.25 above P: penetration 0.5 − 0.25.
    let c = collide_sphere_sdf(v3(-3.0, 8.25, 0.0), fx(0.5), &ball).expect("touching");
    assert!(
        (c.depth.to_f64() - 0.25).abs() < 1e-6,
        "depth {}",
        c.depth.to_f64()
    );
    assert!(close(c.normal, [0.0, 1.0, 0.0], 1e-5));
}

#[test]
fn a_half_space_turns_with_the_rotated_body() {
    let p = [2.0, 1.0, -1.0];
    let mut body = RigidBody::new_dynamic(v3(p[0], p[1], p[2]), Fix128::ONE);
    body.rotation = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), Fix128::HALF_PI);
    let bodies = [body];
    let mut wall = SdfCollider::new_dynamic(half_space(), 0);

    // Unsynced: the solid is the world half-space y < 0.
    assert!(collide_point_sdf(v3(p[0] + 0.25, p[1], p[2]), &wall).is_none());
    assert!(collide_point_sdf(v3(0.0, -0.5, 0.0), &wall).is_some());

    assert!(wall.sync_to_body(&bodies));
    // R ŷ = −x̂: solid on the +x side of x = P_x.
    let c = collide_point_sdf(v3(p[0] + 0.25, p[1] + 5.0, p[2]), &wall).expect("inside the wall");
    assert!(
        (c.depth.to_f64() - 0.25).abs() < 1e-5,
        "depth {}",
        c.depth.to_f64()
    );
    assert!(close(c.normal, [-1.0, 0.0, 0.0], 1e-5));
    assert!(close(c.point_b, [p[0], p[1] + 5.0, p[2]], 1e-5));
    // Below the old world plane but on the −x side: outside now.
    assert!(collide_point_sdf(v3(p[0] - 0.25, -0.5, p[2]), &wall).is_none());
    // A sphere of radius 0.5 whose centre is 0.125 on the outer side.
    let c = collide_sphere_sdf(v3(p[0] - 0.125, p[1], p[2]), fx(0.5), &wall).expect("touching");
    assert!(
        (c.depth.to_f64() - 0.375).abs() < 1e-5,
        "depth {}",
        c.depth.to_f64()
    );
    assert!(close(c.normal, [-1.0, 0.0, 0.0], 1e-5));
}

#[test]
fn set_pose_places_and_update_cache_refreshes_a_direct_write() {
    let mut wall = SdfCollider::new_dynamic(half_space(), 0);
    let turn = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), Fix128::HALF_PI);

    // A direct write of `rotation` without a refresh keeps the old inverse.
    wall.rotation = turn;
    assert!(collide_point_sdf(v3(0.25, 3.0, 0.0), &wall).is_none());
    wall.update_cache();
    let c = collide_point_sdf(v3(0.25, 3.0, 0.0), &wall).expect("refreshed");
    assert!(close(c.normal, [-1.0, 0.0, 0.0], 1e-5));

    wall.set_pose(v3(0.0, 0.0, 4.0), QuatFix::IDENTITY);
    let c = collide_point_sdf(v3(9.0, -0.5, 4.0), &wall).expect("back to the y plane");
    assert!((c.depth.to_f64() - 0.5).abs() < 1e-6);
    assert!(close(c.normal, [0.0, 1.0, 0.0], 1e-6));

    // A scale written directly: a unit ball scaled by 3 has radius 3 after
    // the refresh, so a point 2 from its centre is 1 deep.
    let mut ball = SdfCollider::new_dynamic(unit_ball(), 0);
    ball.scale = Fix128::from_int(3);
    assert!(collide_point_sdf(v3(2.0, 0.0, 0.0), &ball).is_none());
    ball.update_cache();
    let c = collide_point_sdf(v3(2.0, 0.0, 0.0), &ball).expect("inside the scaled ball");
    assert!(
        (c.depth.to_f64() - 1.0).abs() < 1e-5,
        "depth {}",
        c.depth.to_f64()
    );
}

#[test]
fn the_set_sync_moves_only_dynamic_colliders_with_a_body() {
    let bodies = vec![
        RigidBody::new_dynamic(v3(0.25, 0.0, 0.0), Fix128::ONE),
        RigidBody::new_dynamic(v3(6.0, 0.0, 0.0), Fix128::ONE),
        RigidBody::new_dynamic(v3(6.0, 0.75, 0.0), Fix128::ONE),
    ];
    let mut colliders = vec![
        SdfCollider::new_static(half_space(), v3(0.0, -10.0, 0.0), QuatFix::IDENTITY),
        SdfCollider::new_dynamic(unit_ball(), 1),
        SdfCollider::new_dynamic(unit_ball(), 99),
    ];
    assert_eq!(colliders[0].body_index, SDF_STATIC);
    assert_eq!(sync_dynamic_sdf_colliders(&mut colliders, &bodies), 1);
    assert_eq!(colliders[0].position, v3(0.0, -10.0, 0.0));
    assert_eq!(colliders[1].position, v3(6.0, 0.0, 0.0));
    assert_eq!(colliders[2].position, Vec3Fix::ZERO);
    assert!(!colliders[0].sync_to_body(&bodies));
    // Degenerate sets: nothing to move, nothing moved, nothing panics.
    assert_eq!(sync_dynamic_sdf_colliders(&mut [], &bodies), 0);
    assert_eq!(sync_dynamic_sdf_colliders(&mut colliders, &[]), 0);
    assert_eq!(colliders[1].position, v3(6.0, 0.0, 0.0));

    // Body 2 (sphere 0.5, centre 0.75 above body 1) meets the ball of body 1:
    // penetration 0.5 − (0.75 − 1) = 0.75, normal +y. Body 1 skips its own
    // ball; body 0 sits inside the unsynced ball 99 at the origin.
    let contacts = detect_sdf_contacts(&bodies, &colliders, fx(0.5));
    let who: Vec<usize> = contacts.iter().map(|(i, _)| *i).collect();
    assert_eq!(who, vec![0, 2]);
    let c = &contacts[1].1;
    assert!(
        (c.depth.to_f64() - 0.75).abs() < 1e-5,
        "depth {}",
        c.depth.to_f64()
    );
    assert!(close(c.normal, [0.0, 1.0, 0.0], 1e-5));
}
