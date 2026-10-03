//! Audit oracles for `src/solver.rs`.
//!
//! Every expected value is derived by hand from the claim in the doc comment
//! (closed form / total enumeration), never by calling the implementation.
//! A test marked `#[ignore = "known defect: AUD-..."]` is red against the
//! current source and is recorded in the audit ledger; it is kept so the
//! eventual fix has an oracle waiting.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]
#![cfg(feature = "std")]

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{ContactModifier, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::{CollisionFilter, ContactEventType, SleepConfig};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn weightless() -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    }
}

fn close(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() <= tol,
        "{what}: got {g}, want {want} (tol {tol})"
    );
}

// ---------------------------------------------------------------------------
// RigidBody::new  (doc: "inverse mass computed automatically; unit-sphere inertia")
// ---------------------------------------------------------------------------

/// Unit solid sphere: I = 2/5 m r^2 with r = 1, so for m = 4 the inertia is 1.6 and
/// the inverse inertia 0.625 on every axis; inv_mass = 0.25 exactly.
#[test]
fn new_body_has_inverse_mass_and_unit_sphere_inverse_inertia() {
    let b = RigidBody::new(v3(1.0, 2.0, 3.0), fx(4.0));
    assert_eq!(b.inv_mass, Fix128::from_ratio(1, 4));
    for (axis, v) in [
        ("x", b.inv_inertia.x),
        ("y", b.inv_inertia.y),
        ("z", b.inv_inertia.z),
    ] {
        close(v, 0.625, 1e-12, &format!("inv_inertia.{axis}"));
    }
    assert_eq!(b.position, v3(1.0, 2.0, 3.0));
    assert_eq!(b.prev_position, b.position);
    assert_eq!(b.rotation, QuatFix::IDENTITY);
    assert_eq!(b.velocity, Vec3Fix::ZERO);
    assert!(b.is_dynamic() && !b.is_kinematic() && !b.is_static());
    // Documented defaults (comments in the constructor): restitution 0.5, friction 0.3,
    // gravity scale 1, no extra damping.
    assert_eq!(b.restitution, Fix128::from_ratio(5, 10));
    assert_eq!(b.friction, Fix128::from_ratio(3, 10));
    assert_eq!(b.gravity_scale, Fix128::ONE);
    assert_eq!(b.linear_damping, Fix128::ONE);
    assert_eq!(b.angular_damping, Fix128::ONE);
}

/// Zero mass: inverse mass and inverse inertia are zero (static), not NaN-like.
#[test]
fn zero_mass_body_has_zero_inverses() {
    let b = RigidBody::new(Vec3Fix::ZERO, Fix128::ZERO);
    assert_eq!(b.inv_mass, Fix128::ZERO);
    assert_eq!(b.inv_inertia, Vec3Fix::ZERO);
    assert!(b.is_static());
}

/// `mass()` is the inverse of `inv_mass`; for a non-dyadic mass the Fix128
/// round trip is accurate to far better than 1e-12.
#[test]
fn mass_round_trips_through_the_inverse() {
    for m in [1.0, 2.0, 3.0, 7.0, 0.1, 1000.0] {
        let b = RigidBody::new(Vec3Fix::ZERO, fx(m));
        close(b.mass(), m, m * 1e-12, "mass()");
    }
}

/// Doc of `mass()` says "returns infinity for static bodies"; the implementation
/// returns ZERO as the "infinite mass" sentinel (Fix128 has no infinity). This
/// test pins the actual sentinel so the doc fix cannot silently flip it.
#[test]
fn mass_of_a_static_body_is_the_zero_sentinel() {
    assert_eq!(RigidBody::new_static(Vec3Fix::ZERO).mass(), Fix128::ZERO);
    assert_eq!(RigidBody::new_kinematic(Vec3Fix::ZERO).mass(), Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// apply_impulse / apply_impulse_at
// ---------------------------------------------------------------------------

/// dv = J / m. m = 4, J = (8, -4, 2) gives dv = (2, -1, 0.5), exact (all dyadic).
#[test]
fn apply_impulse_changes_velocity_by_j_over_m() {
    let mut b = RigidBody::new(Vec3Fix::ZERO, fx(4.0));
    b.set_velocity(v3(1.0, 1.0, 1.0));
    b.apply_impulse(v3(8.0, -4.0, 2.0));
    assert_eq!(b.velocity, v3(3.0, 0.0, 1.5));
    assert_eq!(
        b.angular_velocity,
        Vec3Fix::ZERO,
        "centre-of-mass impulse does not spin"
    );
}

/// Static / kinematic / sensor bodies have infinite mass: an impulse changes nothing.
#[test]
fn impulses_do_not_move_immovable_bodies() {
    for mut b in [
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new_kinematic(Vec3Fix::ZERO),
        RigidBody::new_sensor(Vec3Fix::ZERO),
    ] {
        b.apply_impulse(v3(5.0, 5.0, 5.0));
        b.apply_impulse_at(v3(5.0, 5.0, 5.0), v3(0.0, 1.0, 0.0));
        assert_eq!(b.velocity, Vec3Fix::ZERO);
        assert_eq!(b.angular_velocity, Vec3Fix::ZERO);
    }
}

/// Isotropic sphere, m = 2 (I = 0.8, 1/I = 1.25). r = (0,1,0), J = (1,0,0):
/// r x J = (0*0-0*0, 0*1-0*0, 0*0-1*1) = (0,0,-1) so dw = (0,0,-1.25), dv = (0.5,0,0).
#[test]
fn apply_impulse_at_gives_linear_and_angular_response() {
    let mut b = RigidBody::new(Vec3Fix::ZERO, fx(2.0));
    b.apply_impulse_at(v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0));
    assert_eq!(b.velocity, v3(0.5, 0.0, 0.0));
    close(b.angular_velocity.x, 0.0, 1e-12, "wx");
    close(b.angular_velocity.y, 0.0, 1e-12, "wy");
    close(b.angular_velocity.z, -1.25, 1e-12, "wz");
}

/// A line of action through the centre of mass gives no spin.
#[test]
fn apply_impulse_through_the_centre_gives_no_spin() {
    let mut b = RigidBody::new(v3(1.0, 1.0, 1.0), fx(2.0));
    b.apply_impulse_at(v3(3.0, -2.0, 1.0), v3(4.0, -1.0, 2.0)); // r = (3,-2,1) = J
    assert_eq!(b.angular_velocity, Vec3Fix::ZERO);
}

/// The field doc says `inv_inertia` is the diagonal "in local space", and
/// `joint.rs::angular_inverse_mass` rotates the world axis into the body frame
/// before applying it. A world-frame torque on a rotated, anisotropic body must
/// therefore use the body-frame diagonal: body rotated +90 deg about z (local y
/// -> world -x, local x -> world +y), inv_inertia = (1, 2, 4) in the body frame.
/// World torque (1,0,0) is local (0,-1,0); local dw = (0,-2,0); world dw = R(0,-2,0) = (2,0,0).
fn rotated_anisotropic_body() -> RigidBody {
    let s = fx(0.5).sqrt(); // sin(45 deg) = cos(45 deg)
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = v3(1.0, 2.0, 4.0);
    b.set_rotation(QuatFix::new(Fix128::ZERO, Fix128::ZERO, s, s));
    b
}

#[test]
#[ignore = "known defect: AUD-A-S1W2-001: add_torque / apply_impulse_at multiply the world-frame torque by the body-frame inv_inertia without rotating it: torque (1,0,0) on a body rotated 90 deg about z with inv_inertia (1,2,4) gives dw.x = 1, expected 2"]
fn add_torque_uses_the_body_frame_inertia_of_a_rotated_body() {
    let mut b = rotated_anisotropic_body();
    b.add_torque(v3(1.0, 0.0, 0.0), Fix128::ONE);
    close(b.angular_velocity.x, 2.0, 1e-9, "wx");
    close(b.angular_velocity.y, 0.0, 1e-9, "wy");
    close(b.angular_velocity.z, 0.0, 1e-9, "wz");
}

#[test]
#[ignore = "known defect: AUD-A-S1W2-001: same root cause through apply_impulse_at: r = (0,1,0), J = (0,0,1) gives torque (1,0,0), dw.x = 1, expected 2"]
fn apply_impulse_at_uses_the_body_frame_inertia_of_a_rotated_body() {
    let mut b = rotated_anisotropic_body();
    // r = (0,1,0), J = (0,0,1): r x J = (1*1 - 0*0, 0, 0) = (1,0,0), the same world
    // torque as in the add_torque oracle (inv_mass = 1 so dv = (0,0,1)).
    b.apply_impulse_at(v3(0.0, 0.0, 1.0), v3(0.0, 1.0, 0.0));
    close(b.angular_velocity.x, 2.0, 1e-9, "wx");
}

// ---------------------------------------------------------------------------
// set_position / kinematic targets
// ---------------------------------------------------------------------------

/// `set_position` teleports: prev_position follows, so no velocity is derived.
#[test]
fn set_position_teleports_and_resets_prev_position() {
    let mut b = RigidBody::new(v3(1.0, 2.0, 3.0), Fix128::ONE);
    b.set_position(v3(9.0, 8.0, 7.0));
    assert_eq!(b.position, v3(9.0, 8.0, 7.0));
    assert_eq!(b.prev_position, b.position);
}

/// A kinematic body reaches its target in the step ("moved to this target during the
/// next simulation step"), and is not affected by gravity.
#[test]
fn kinematic_body_reaches_its_target_in_one_step() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let k = w.add_body(RigidBody::new_kinematic(Vec3Fix::ZERO));
    w.get_body_mut(k)
        .unwrap()
        .set_kinematic_target(v3(1.0, 2.0, 3.0), QuatFix::IDENTITY);
    w.step(dt());
    assert_eq!(w.get_body(k).unwrap().position, v3(1.0, 2.0, 3.0));
}

/// `set_kinematic_target` doc: "Velocity is automatically computed from the position
/// change." The target is 1 m away and reached within dt = 1/64 s, so the body moved at
/// 64 m/s over the frame and `velocity` after the step should read (64, 0, 0).
/// Under XPBD the body jumps in the first substep and the other 7 substeps see
/// position == target, so `update_velocities` leaves velocity 0.
#[test]
#[ignore = "known defect: AUD-A-S1W2-002: kinematic body velocity after step() is 0, not (target - position) / dt (the velocity is derived in substep 1 then overwritten to 0 by substeps 2..8)"]
fn kinematic_velocity_after_a_step_is_the_displacement_over_dt() {
    let mut w = PhysicsWorld::new(weightless());
    let k = w.add_body(RigidBody::new_kinematic(Vec3Fix::ZERO));
    w.get_body_mut(k)
        .unwrap()
        .set_kinematic_target(v3(1.0, 0.0, 0.0), QuatFix::IDENTITY);
    w.step(dt());
    close(
        w.get_body(k).unwrap().velocity.x,
        64.0,
        1e-6,
        "kinematic vx",
    );
}

// ---------------------------------------------------------------------------
// observe_body / observe_bodies
// ---------------------------------------------------------------------------

/// `observe_body` reports position / velocity / rotation / angular velocity
/// as in the body; `None` out of range; `observe_bodies` is in index order.
#[test]
fn observe_body_mirrors_the_body_and_none_out_of_range() {
    let mut w = PhysicsWorld::new(weightless());
    let a = w.add_body(RigidBody::new(v3(1.0, 2.0, 3.0), fx(2.0)));
    let b = w.add_body(RigidBody::new(v3(-5.0, 0.0, 0.0), fx(1.0)));
    {
        let ba = w.get_body_mut(a).unwrap();
        ba.set_velocity(v3(0.5, 0.0, -0.5));
        ba.set_angular_velocity(v3(0.0, 0.25, 0.0));
    }
    let o = w.observe_body(a).unwrap();
    assert_eq!(o.body_index, a);
    assert_eq!(o.position, v3(1.0, 2.0, 3.0));
    assert_eq!(o.velocity, v3(0.5, 0.0, -0.5));
    assert_eq!(o.angular_velocity, v3(0.0, 0.25, 0.0));
    assert_eq!(o.rotation, QuatFix::IDENTITY);
    assert!(!o.sleeping && !o.in_contact);
    assert!(w.observe_body(2).is_none());
    assert!(w.observe_body(usize::MAX).is_none());
    let all = w.observe_bodies();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0], o);
    assert_eq!(all[1].body_index, b);
    assert_eq!(all[1].position, v3(-5.0, 0.0, 0.0));
    assert!(PhysicsWorld::new(weightless()).observe_bodies().is_empty());
}

/// `sleeping` follows the island manager: a resting body falls asleep after
/// `frames_to_sleep` idle frames.
#[test]
fn observe_body_reports_sleeping() {
    let mut w = PhysicsWorld::new(weightless());
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: 3,
        ..SleepConfig::default()
    });
    let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    assert!(!w.observe_body(a).unwrap().sleeping);
    for _ in 0..5 {
        w.step(dt());
    }
    assert!(w.is_sleeping(a));
    assert!(w.observe_body(a).unwrap().sleeping);
}

fn touching_pair() -> (PhysicsWorld, usize, usize) {
    // Two unit-radius spheres overlapping by 0.5, weightless, the second pinned so
    // the pair is pushed apart only by the first.
    let mut w = PhysicsWorld::new(weightless());
    let a = w.add_body_with_radius(RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE), Fix128::ONE);
    let b = w.add_body_with_radius(RigidBody::new_static(v3(1.5, 0.0, 0.0)), Fix128::ONE);
    (w, a, b)
}

/// doc: `in_contact` = "the body has at least one active contact this frame (`Begin` or
/// `Persist`)". While overlapping it is true.
#[test]
fn in_contact_is_true_while_a_contact_begins() {
    let (mut w, a, b) = touching_pair();
    w.step(dt());
    assert!(
        w.contact_events()
            .iter()
            .any(|e| e.event_type == ContactEventType::Begin),
        "scene must produce a Begin event"
    );
    assert!(w.observe_body(a).unwrap().in_contact);
    assert!(w.observe_body(b).unwrap().in_contact);
}

/// On the frame the contact ends the event list holds an `End` event only, so by the
/// doc (`Begin` or `Persist`) `in_contact` is false. The implementation counts any
/// event, `End` included.
#[test]
#[ignore = "known defect: AUD-A-S1W2-003: observe_body.in_contact is true on the frame of an End event (any contact event counts), doc says Begin or Persist only"]
fn in_contact_is_false_on_the_frame_the_contact_ends() {
    let (mut w, a, _b) = touching_pair();
    w.step(dt());
    // Teleport apart: no overlap next frame.
    w.get_body_mut(a).unwrap().set_position(v3(-10.0, 0.0, 0.0));
    w.step(dt());
    let evs = w.contact_events();
    assert!(
        evs.iter().any(|e| e.event_type == ContactEventType::End),
        "scene must produce an End event, got {} events",
        evs.len()
    );
    assert!(
        evs.iter().all(|e| e.event_type == ContactEventType::End),
        "only End events this frame"
    );
    assert!(!w.observe_body(a).unwrap().in_contact);
}

// ---------------------------------------------------------------------------
// raycast
// ---------------------------------------------------------------------------

fn ray_world() -> PhysicsWorld {
    PhysicsWorld::new(weightless())
}

/// Sphere r = 1 at (5,0,0); ray from the origin along +x hits at t = 5 - 1 = 4.
#[test]
fn raycast_hits_the_near_surface_of_a_sphere() {
    let mut w = ray_world();
    let s = w.add_body_with_radius(RigidBody::new(v3(5.0, 0.0, 0.0), Fix128::ONE), Fix128::ONE);
    let (i, t) = w
        .raycast(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), fx(100.0))
        .expect("hit");
    assert_eq!(i, s);
    close(t, 4.0, 1e-9, "t");
}

/// The direction need not be a unit vector: the distance is in world units.
#[test]
fn raycast_distance_is_in_world_units_for_a_long_direction() {
    let mut w = ray_world();
    w.add_body_with_radius(RigidBody::new(v3(0.0, 0.0, 5.0), Fix128::ONE), Fix128::ONE);
    let (_, t) = w
        .raycast(Vec3Fix::ZERO, v3(0.0, 0.0, 3.0), fx(100.0))
        .expect("hit");
    close(t, 4.0, 1e-9, "t");
}

/// `max_distance` cuts the ray: the hit at t = 4 is returned for max >= 4, not for less.
#[test]
fn raycast_respects_max_distance() {
    let mut w = ray_world();
    w.add_body_with_radius(RigidBody::new(v3(5.0, 0.0, 0.0), Fix128::ONE), Fix128::ONE);
    assert!(w
        .raycast(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), fx(3.9))
        .is_none());
    assert!(w
        .raycast(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), fx(4.1))
        .is_some());
}

/// A sphere behind the origin is not hit; neither is one off to the side.
#[test]
fn raycast_ignores_spheres_behind_and_beside() {
    let mut w = ray_world();
    w.add_body_with_radius(RigidBody::new(v3(-5.0, 0.0, 0.0), Fix128::ONE), Fix128::ONE);
    w.add_body_with_radius(RigidBody::new(v3(5.0, 3.0, 0.0), Fix128::ONE), Fix128::ONE);
    assert!(w
        .raycast(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), fx(100.0))
        .is_none());
}

/// The nearest of several spheres wins, whatever the insertion order.
#[test]
fn raycast_returns_the_nearest_sphere() {
    for order in [[8.0, 3.0, 12.0], [3.0, 8.0, 12.0], [12.0, 8.0, 3.0]] {
        let mut w = ray_world();
        let mut near = usize::MAX;
        for x in order {
            let i = w.add_body_with_radius(RigidBody::new(v3(x, 0.0, 0.0), Fix128::ONE), fx(0.5));
            if x == 3.0 {
                near = i;
            }
        }
        let (i, t) = w
            .raycast(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), fx(100.0))
            .unwrap();
        assert_eq!(i, near);
        close(t, 2.5, 1e-9, "t");
    }
}

/// Origin inside the sphere: the exit point is returned (the doc comment in the
/// implementation: far intersection when the near one is behind the origin).
#[test]
fn raycast_from_inside_a_sphere_returns_the_exit_distance() {
    let mut w = ray_world();
    w.add_body_with_radius(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE), fx(2.0));
    let (_, t) = w
        .raycast(v3(0.5, 0.0, 0.0), v3(1.0, 0.0, 0.0), fx(100.0))
        .unwrap();
    close(t, 1.5, 1e-9, "t");
}

/// Zero direction and a world without collision radii answer `None`; a body without a
/// radius is not a target.
#[test]
fn raycast_degenerate_inputs_answer_none() {
    let mut w = ray_world();
    w.add_body(RigidBody::new(v3(5.0, 0.0, 0.0), Fix128::ONE)); // no radius
    assert!(w
        .raycast(Vec3Fix::ZERO, v3(1.0, 0.0, 0.0), fx(100.0))
        .is_none());
    w.add_body_with_radius(RigidBody::new(v3(5.0, 0.0, 0.0), Fix128::ONE), Fix128::ONE);
    assert!(w.raycast(Vec3Fix::ZERO, Vec3Fix::ZERO, fx(100.0)).is_none());
}

/// A diagonal ray: sphere r = 1 at (3,4,0), ray along (3,4,0)/5 from the origin hits the
/// near surface at distance |c| - r = 5 - 1 = 4.
#[test]
fn raycast_diagonal_ray() {
    let mut w = ray_world();
    w.add_body_with_radius(RigidBody::new(v3(3.0, 4.0, 0.0), Fix128::ONE), Fix128::ONE);
    let (_, t) = w
        .raycast(Vec3Fix::ZERO, v3(3.0, 4.0, 0.0), fx(100.0))
        .unwrap();
    close(t, 4.0, 1e-9, "t");
}

// ---------------------------------------------------------------------------
// combined_material
// ---------------------------------------------------------------------------

/// An index past the last body is treated as the default material.
#[test]
fn combined_material_out_of_range_is_the_default_pair() {
    let mut w = PhysicsWorld::new(weightless());
    let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let b = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let d = w.combined_material(a, b);
    assert_eq!(w.combined_material(a, 99), d);
    assert_eq!(w.combined_material(99, 100), d);
}

// ---------------------------------------------------------------------------
// remove_body and events
// ---------------------------------------------------------------------------

/// A contact pair that persists across a `remove_body` of an unrelated body must
/// produce `Persist` for the surviving pair (under its new indices) and nothing else:
/// no `Begin` for a pair that was already touching, no `End` for an index that no
/// longer exists. The event history is keyed by body index and `remove_body` does not
/// remap it.
#[test]
#[ignore = "known defect: AUD-A-S1W2-004: remove_body does not remap the event pair history; the surviving pair re-reports Begin and an End is emitted for a body index that no longer exists"]
fn remove_body_keeps_the_contact_history_of_the_survivors() {
    let mut w = PhysicsWorld::new(weightless());
    let a = w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.0, 0.0)), Fix128::ONE);
    let far = w.add_body_with_radius(RigidBody::new_static(v3(100.0, 0.0, 0.0)), Fix128::ONE);
    let c = w.add_body_with_radius(RigidBody::new(v3(1.5, 0.0, 0.0), Fix128::ONE), Fix128::ONE);
    assert_eq!((a, far, c), (0, 1, 2));
    w.step(dt());
    assert!(w
        .contact_events()
        .iter()
        .any(|e| e.event_type == ContactEventType::Begin && e.body_a == 0 && e.body_b == 2));
    w.remove_body(far); // c moves to index 1
                        // Keep c where it is so the contact continues
    w.get_body_mut(1).unwrap().set_position(v3(1.5, 0.0, 0.0));
    w.step(dt());
    let evs: Vec<_> = w.contact_events().to_vec();
    let n = w.body_count();
    assert!(
        evs.iter().all(|e| e.body_a < n && e.body_b < n),
        "event refers to a removed body index: {evs:?}"
    );
    assert!(
        evs.iter()
            .all(|e| e.event_type == ContactEventType::Persist),
        "expected only Persist for the surviving pair, got {evs:?}"
    );
}

// ---------------------------------------------------------------------------
// step: configuration edge
// ---------------------------------------------------------------------------

/// `substeps = 0` must not corrupt the world: either a clean no-op or an explicit
/// documented behaviour. Records what happens (no panic, no position change).
#[test]
fn zero_substeps_does_not_panic_or_move_a_body() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 0,
        ..PhysicsConfig::default()
    });
    let a = w.add_body(RigidBody::new(v3(0.0, 5.0, 0.0), Fix128::ONE));
    w.step(dt());
    assert_eq!(w.get_body(a).unwrap().position, v3(0.0, 5.0, 0.0));
}

// ---------------------------------------------------------------------------
// ContactModifier: "can mutate normal, depth, friction, and restitution"
// ---------------------------------------------------------------------------

struct SetMaterial {
    friction: Option<Fix128>,
    restitution: Option<Fix128>,
}

impl ContactModifier for SetMaterial {
    fn modify_contact(
        &self,
        _a: usize,
        _b: usize,
        _contact: &mut Contact,
        friction: &mut Fix128,
        restitution: &mut Fix128,
    ) -> bool {
        if let Some(f) = self.friction {
            *friction = f;
        }
        if let Some(r) = self.restitution {
            *restitution = r;
        }
        true
    }
}

/// Two unit spheres (m = 1) closing head-on at 20 m/s each (overlap 0.1 at the first
/// substep), the first also moving 5 m/s along y, with the given material on both.
fn collision(friction: f64, restitution: f64, modifier: Option<SetMaterial>) -> (Vec3Fix, Vec3Fix) {
    let mut w = PhysicsWorld::new(weightless());
    let a = w.add_body_with_radius(
        RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE).with_velocity(v3(20.0, 5.0, 0.0)),
        Fix128::ONE,
    );
    let b = w.add_body_with_radius(
        RigidBody::new(v3(2.1, 0.0, 0.0), Fix128::ONE).with_velocity(v3(-20.0, 0.0, 0.0)),
        Fix128::ONE,
    );
    let id = w
        .material_table
        .register(alice_physics::PhysicsMaterial::new(
            0,
            fx(friction),
            fx(restitution),
        ));
    w.set_body_material(a, id);
    w.set_body_material(b, id);
    if let Some(m) = modifier {
        w.add_contact_modifier(Box::new(m));
    }
    w.step(dt());
    (
        w.get_body(a).unwrap().velocity,
        w.get_body(b).unwrap().velocity,
    )
}

/// Sanity of the scene (no modifier): restitution 0 makes the bodies stick along the
/// normal (relative normal speed after the step is 0), restitution 1 makes them rebound.
#[test]
fn collision_scene_responds_to_the_material_restitution() {
    let (a0, b0) = collision(0.5, 0.0, None);
    close(a0.x, 0.0, 0.1, "a vx");
    close(b0.x, 0.0, 0.1, "b vx");
    let (a1, b1) = collision(0.5, 1.0, None);
    assert!(
        a1.x.to_f64() < -4.0 && b1.x.to_f64() > 4.0,
        "rebound {a1:?} {b1:?}"
    );
}

/// Sanity: the material friction does change the tangential relative velocity
/// (friction 0 keeps it, friction 9 removes it).
#[test]
fn collision_scene_responds_to_the_material_friction() {
    let (a_free, b_free) = collision(0.0, 1.0, None);
    let (a_grip, b_grip) = collision(9.0, 1.0, None);
    let rel = |a: Vec3Fix, b: Vec3Fix| (a.y - b.y).to_f64().abs();
    assert!(rel(a_free, b_free) > 4.0, "free {}", rel(a_free, b_free));
    assert!(rel(a_grip, b_grip) < 1.0, "grip {}", rel(a_grip, b_grip));
}

#[test]
#[ignore = "known defect: AUD-A-S1W2-005: a ContactModifier's restitution change is discarded (CPU solve keeps it in an unused local; update_velocities reads constraint.restitution); modifier restitution 1 on a restitution-0 pair still gives vx = 0"]
fn a_contact_modifier_can_change_the_restitution() {
    let (a, b) = collision(
        0.5,
        0.0,
        Some(SetMaterial {
            friction: None,
            restitution: Some(Fix128::ONE),
        }),
    );
    assert!(
        a.x.to_f64() < -4.0 && b.x.to_f64() > 4.0,
        "no rebound: {a:?} {b:?}"
    );
}

#[test]
#[ignore = "known defect: AUD-A-S1W2-005: a ContactModifier's friction change is discarded; modifier friction 0 on a friction-9 pair still removes the tangential relative velocity"]
fn a_contact_modifier_can_change_the_friction() {
    let (a, b) = collision(
        9.0,
        1.0,
        Some(SetMaterial {
            friction: Some(Fix128::ZERO),
            restitution: None,
        }),
    );
    let rel = (a.y - b.y).to_f64().abs();
    assert!(
        rel > 4.0,
        "tangential relative speed {rel}, friction 0 must keep it"
    );
}

// ---------------------------------------------------------------------------
// GPU solver bridge: Stage A filtering, write-back, joint rotation
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-solver-bridge")]
mod bridge {
    use super::*;
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::solver::ContactConstraint;
    use std::cell::RefCell;

    /// Records what `solve_contact_constraints_with_bridge` uploads, then writes back a
    /// recognisable answer: lambda = 7 + index into the received slice, and body 0 moved
    /// to x = 42.
    #[derive(Default)]
    struct Recorder {
        sent: Vec<ContactConstraint>,
        positions_sent: usize,
        inv_masses_sent: Vec<Fix128>,
        answer: RefCell<()>,
    }

    impl GpuSolverBridge for Recorder {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, c: &[ContactConstraint]) {
            self.sent = c.to_vec();
        }
        fn send_body_state(&mut self, p: &[[Fix128; 3]], i: &[Fix128]) {
            self.positions_sent = p.len();
            self.inv_masses_sent = i.to_vec();
        }
        fn dispatch_contact_solve_iteration(&mut self, _w: Fix128) {
            let _ = &self.answer;
        }
        fn recv_contact_constraints(&self, c: &mut [ContactConstraint]) {
            for (i, k) in c.iter_mut().enumerate() {
                k.cached_lambda = Fix128::from_int(7 + i as i64);
            }
        }
        fn recv_body_positions(&self, p: &mut [[Fix128; 3]]) {
            p[0][0] = Fix128::from_int(42);
        }
    }

    fn contact(depth: f64) -> Contact {
        Contact {
            depth: fx(depth),
            normal: v3(1.0, 0.0, 0.0),
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        }
    }

    fn three_body_world() -> PhysicsWorld {
        let mut w = PhysicsWorld::new(weightless());
        w.add_body(RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE));
        w.add_body(RigidBody::new(v3(1.0, 0.0, 0.0), fx(2.0)));
        w.add_body(RigidBody::new(v3(2.0, 0.0, 0.0), fx(4.0)));
        w
    }

    /// Stage A: a sensor pair is not sent; the vetoed contact is not sent; the survivors
    /// keep their order; the lambdas that come back land in the ORIGINAL slots of the
    /// survivors (index map), and positions are written back for every body.
    #[test]
    fn contact_solve_filters_sends_survivors_and_writes_back_by_original_slot() {
        let mut w = three_body_world();
        let sensor = w.add_body(RigidBody::new_sensor(v3(5.0, 0.0, 0.0)));
        // Slot 0: (0,1) vetoed by a hook. Slot 1: (1,2) survives. Slot 2: sensor pair.
        // Slot 3: (0,2) survives.
        w.add_pre_solve_hook(Box::new(|a, b, _c| !(a == 0 && b == 1)));
        for (a, b) in [(0, 1), (1, 2), (0, sensor), (0, 2)] {
            w.add_contact(ContactConstraint {
                body_a: a,
                body_b: b,
                contact: contact(0.25),
                friction: fx(0.5),
                restitution: fx(0.25),
                cached_lambda: Fix128::ZERO,
            });
        }
        let mut rec = Recorder::default();
        w.solve_contact_constraints_with_bridge(&mut rec);
        let pairs: Vec<(usize, usize)> = rec.sent.iter().map(|c| (c.body_a, c.body_b)).collect();
        assert_eq!(pairs, vec![(1, 2), (0, 2)], "sent {pairs:?}");
        assert_eq!(rec.positions_sent, 4);
        assert_eq!(
            rec.inv_masses_sent,
            vec![
                Fix128::ONE,
                Fix128::from_ratio(1, 2),
                Fix128::from_ratio(1, 4),
                Fix128::ZERO
            ]
        );
        // lambdas: survivors got 7 and 8, in the original slots 1 and 3
        let lam = |i: usize| w.contact_constraints[i].cached_lambda;
        assert_eq!(lam(0), Fix128::ZERO, "vetoed slot untouched");
        assert_eq!(lam(1), Fix128::from_int(7));
        assert_eq!(lam(2), Fix128::ZERO, "sensor slot untouched");
        assert_eq!(lam(3), Fix128::from_int(8));
        assert_eq!(w.get_body(0).unwrap().position.x, Fix128::from_int(42));
        assert_eq!(w.get_body(1).unwrap().position.x, Fix128::ONE);
    }

    /// Nothing survives Stage A: the bridge is not called at all and the world is untouched.
    #[test]
    fn contact_solve_with_no_survivor_does_not_touch_the_bridge() {
        let mut w = three_body_world();
        w.add_pre_solve_hook(Box::new(|_a, _b, _c| false));
        w.add_contact(ContactConstraint {
            body_a: 0,
            body_b: 1,
            contact: contact(0.25),
            friction: fx(0.5),
            restitution: fx(0.25),
            cached_lambda: Fix128::ZERO,
        });
        let mut rec = Recorder::default();
        w.solve_contact_constraints_with_bridge(&mut rec);
        assert!(rec.sent.is_empty());
        assert_eq!(rec.positions_sent, 0, "send_body_state must not be called");
        assert_eq!(w.get_body(0).unwrap().position, Vec3Fix::ZERO);
    }

    /// A reference joint-solve bridge: it receives exactly what the trait hands a backend
    /// (joints, positions, inverse masses, rotations), runs the crate's own CPU
    /// `solve_joints` on a reconstruction of the bodies, and returns positions, which is
    /// the only thing the trait can return.
    #[derive(Default)]
    struct JointReference {
        joints: Vec<alice_physics::joint::Joint>,
        bodies: Vec<RigidBody>,
    }

    impl GpuSolverBridge for JointReference {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_joints(&mut self, j: &[alice_physics::joint::Joint]) {
            self.joints = j.to_vec();
        }
        fn send_body_state(&mut self, p: &[[Fix128; 3]], inv_masses: &[Fix128]) {
            self.bodies = p
                .iter()
                .zip(inv_masses)
                .map(|(pos, inv_m)| {
                    // unit-mass bodies in this scene: the inverse inertia is not sent
                    let mut b = RigidBody::new(Vec3Fix::new(pos[0], pos[1], pos[2]), Fix128::ONE);
                    b.inv_mass = *inv_m;
                    b
                })
                .collect();
        }
        fn send_body_rotations(&mut self, r: &[[Fix128; 4]]) {
            for (b, q) in self.bodies.iter_mut().zip(r) {
                b.rotation = QuatFix::new(q[0], q[1], q[2], q[3]);
            }
        }
        fn dispatch_joint_solve_iteration(&mut self, dt: Fix128) {
            alice_physics::joint::solve_joints(&self.joints, &mut self.bodies, dt);
        }
        fn recv_body_positions(&self, p: &mut [[Fix128; 3]]) {
            for (out, b) in p.iter_mut().zip(&self.bodies) {
                *out = [b.position.x, b.position.y, b.position.z];
            }
        }
    }

    fn limited_hinge_world() -> PhysicsWorld {
        use alice_physics::joint::{HingeJoint, Joint};
        let mut w = PhysicsWorld::new(weightless());
        w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let b = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        w.get_body_mut(b)
            .unwrap()
            .set_rotation(QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::ONE));
        w.add_joint(Joint::Hinge(
            HingeJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z,
            )
            .with_limits(Fix128::from_ratio(-1, 2), Fix128::from_ratio(1, 2)),
        ));
        w
    }

    /// Sanity: the CPU joint solve rotates the bodies of this scene (the limit is 0.5 rad,
    /// the twist is 1 rad).
    #[test]
    fn cpu_joint_solve_corrects_a_hinge_limit_by_rotating_the_bodies() {
        let mut w = limited_hinge_world();
        let before = w.bodies[1].rotation;
        alice_physics::joint::solve_joints(&w.joints.clone(), &mut w.bodies, dt());
        assert_ne!(w.bodies[0].rotation, QuatFix::IDENTITY, "body A rotated");
        assert_ne!(w.bodies[1].rotation, before, "body B rotated");
    }

    /// `solve_joints_with_bridge` documents "byte-exact CPU parity". The CPU solve changes
    /// body rotations for an angular limit; the bridge path writes back positions only
    /// (the trait has `send_body_rotations` but no way to return them). Even a backend
    /// that runs the crate's own joint solver gives a different world.
    #[test]
    #[ignore = "known defect: AUD-A-S1W2-006: solve_joints_with_bridge writes back positions only; the rotation corrections of the CPU joint solve (joint.rs apply_angular_correction) cannot come back through GpuSolverBridge, so a hinge limit is not enforced under a bridge"]
    fn joint_solve_through_a_reference_bridge_matches_the_cpu_solve() {
        let mut cpu = limited_hinge_world();
        alice_physics::joint::solve_joints(&cpu.joints.clone(), &mut cpu.bodies, dt());

        let mut routed = limited_hinge_world();
        let mut bridge = JointReference::default();
        routed.solve_joints_with_bridge(&mut bridge, dt());

        assert_eq!(routed.bodies[0].rotation, cpu.bodies[0].rotation, "body A");
        assert_eq!(routed.bodies[1].rotation, cpu.bodies[1].rotation, "body B");
    }
}

// ---------------------------------------------------------------------------
// add_joint panic contract, wake_body bound, collider overlap, filters after removal
// ---------------------------------------------------------------------------

/// `add_joint` "Panics if either body index in the joint is out of bounds": each index
/// alone is enough, and a valid pair is accepted.
#[test]
fn add_joint_panics_when_either_index_is_out_of_bounds() {
    use alice_physics::joint::{BallJoint, Joint};
    let ball = |a: usize, b: usize| Joint::Ball(BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO));
    let mut w = PhysicsWorld::new(weightless());
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    assert_eq!(w.add_joint(ball(0, 1)), 0);
    for (a, b) in [(0, 2), (2, 0), (5, 7)] {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut w2 = PhysicsWorld::new(weightless());
            w2.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
            w2.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
            w2.add_joint(ball(a, b));
        }));
        assert!(r.is_err(), "joint ({a}, {b}) must panic");
    }
}

/// An index equal to the body count is out of range for `wake_body` and is ignored.
#[test]
fn wake_body_one_past_the_last_body_is_ignored() {
    let mut w = PhysicsWorld::new(weightless());
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let r = catch_unwind(AssertUnwindSafe(|| w.wake_body(2)));
    assert!(
        r.is_ok(),
        "wake_body(body_count) must be ignored, not panic"
    );
}

/// Spheres whose surfaces touch exactly are clear (`colliders_overlap` doc: "clear for
/// spheres"); a hair closer they overlap. Radii 1 and 2, centres 3 apart.
#[test]
fn touching_spheres_are_clear_and_a_hair_closer_overlap() {
    let mut w = PhysicsWorld::new(weightless());
    let a = w.add_body_with_radius(RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE), fx(1.0));
    let b = w.add_body_with_radius(RigidBody::new(v3(3.0, 0.0, 0.0), Fix128::ONE), fx(2.0));
    assert!(!w.colliders_overlap(a, b) && !w.colliders_overlap(b, a));
    w.get_body_mut(b).unwrap().set_position(v3(2.999, 0.0, 0.0));
    assert!(w.colliders_overlap(a, b) && w.colliders_overlap(b, a));
}

/// `remove_body` swap-removes per-body data: the filter of the body moved into the freed
/// slot is its own, not the removed one's.
#[test]
fn remove_body_keeps_each_survivors_filter() {
    let mut w = PhysicsWorld::new(weightless());
    let f0 = CollisionFilter::new(1, 0xF);
    let f1 = CollisionFilter::new(2, 0xE);
    let f2 = CollisionFilter::new(4, 0xD);
    for f in [f0, f1, f2] {
        let i = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        w.set_body_filter(i, f);
    }
    w.remove_body(0);
    assert_eq!(w.body_filter(0), f2, "last body moved into slot 0");
    assert_eq!(w.body_filter(1), f1);
    assert_eq!(w.body_filter(2), CollisionFilter::DEFAULT, "past the end");
}

// ---------------------------------------------------------------------------
// RigidBody::with_restitution / with_friction
// ---------------------------------------------------------------------------

/// `with_restitution` is documented as "set restitution (bounciness)" and the field as the
/// "Coefficient of restitution". Contacts take their restitution from the material table
/// (`add_contact_with_material` -> `combined_material`) and never read the body field, so a
/// head-on pair built with restitution 1 behaves exactly like one built with restitution 0.
#[test]
#[ignore = "known defect: AUD-A-S1W2-008: RigidBody.restitution / friction (with_restitution, with_friction, wasm setRestitution / setFriction) are never read by the 3D solver; contacts use the material table only"]
fn body_restitution_changes_the_bounce() {
    let run = |e: f64| {
        let mut w = PhysicsWorld::new(weightless());
        let a = w.add_body_with_radius(
            RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE)
                .with_velocity(v3(20.0, 0.0, 0.0))
                .with_restitution(fx(e)),
            Fix128::ONE,
        );
        w.add_body_with_radius(
            RigidBody::new(v3(2.1, 0.0, 0.0), Fix128::ONE)
                .with_velocity(v3(-20.0, 0.0, 0.0))
                .with_restitution(fx(e)),
            Fix128::ONE,
        );
        w.step(dt());
        w.get_body(a).unwrap().velocity.x.to_f64()
    };
    let (stick, bounce) = (run(0.0), run(1.0));
    assert!(
        (stick - bounce).abs() > 1.0,
        "restitution 0 -> vx {stick}, restitution 1 -> vx {bounce}: no effect"
    );
}
