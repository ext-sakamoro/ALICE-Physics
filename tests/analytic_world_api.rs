//! Closed-form oracles for the `PhysicsWorld` / `RigidBody` host API
//!
//! Every expected value below is derived by hand from the integration scheme
//! `src/solver.rs` documents, never by running the implementation:
//!
//! - `step(dt)` runs `substeps` (default 8) substeps of `h = dt / substeps`.
//!   Each substep is semi-implicit Euler: `v += g * gravity_scale * h`, then
//!   `x += v * h`, then constraints, then `v = (x - x_prev) / h`. The
//!   velocity-retention factor `damping` (default 0.99) times the body's
//!   `linear_damping` / `angular_damping` is applied **once per frame**
//!   after the substep loop.
//! - A frame of `dt = 1/64 s` therefore has `h = 1/512 s`, both exact in
//!   `Fix128`, so products such as `512 * h` or `(1/32) * h` are exact and
//!   the oracles can use `assert_eq!` on the fixed-point bits.
//! - A sphere–sphere contact of depth `d` between two unit masses moves each
//!   body by `d / 2` along the normal in the substep that sees it; the
//!   velocity derived from that displacement is `d / (2 h)`, so the bodies
//!   separate and the frame ends with each `substeps * d / 2` from where it
//!   started (`solve_contact_constraints`, XPBD contact with λ accumulated
//!   within the substep).
//! - An SDF contact pushes a body out by the penetration `r - φ(x)` in the
//!   substep that sees it (`resolve_sdf_collisions`), and the same velocity
//!   derivation carries that displacement through the remaining substeps:
//!   `y_end = y_0 + substeps * (r - y_0)` for a plane `φ = y`.
//!
//! The example `examples/world_api_tour.rs` prints the same quantities.

#![cfg(feature = "std")]

use alice_physics::collider::Contact;
use alice_physics::joint::BallJoint;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::solver::{ContactModifier, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::{
    CollisionFilter, ContactEventType, DistanceConstraint, ForceField, ForceFieldInstance, Joint,
    PhysicsMaterial,
};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

/// `dt = 1/64 s`, exact; with the default 8 substeps `h = 1/512 s`.
fn dt() -> Fix128 {
    r(1, 64)
}

/// The oracles assume the default substep count; pin it.
fn substeps() -> i64 {
    let s = PhysicsConfig::default().substeps;
    assert_eq!(s, 8, "oracles below are derived for 8 substeps");
    8
}

fn weightless() -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    }
}

/// Zero gravity and no frame damping, so velocities carry across frames
/// unchanged and multi-frame displacements stay exact.
fn weightless_undamped() -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    }
}

/// Two radius-2 unit-mass spheres at `x = 0` and `x = 3`: depth 1.
fn overlapping_pair(world: &mut PhysicsWorld) -> (usize, usize) {
    let radius = Fix128::from_int(2);
    let a = world.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), radius);
    let b = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(3, 0, 0), Fix128::ONE),
        radius,
    );
    (a, b)
}

/// After one frame the pair is at `-substeps * depth / 2` and
/// `3 + substeps * depth / 2` = `-4` and `7`.
fn separated_x() -> (Fix128, Fix128) {
    let half = r(substeps(), 2);
    (-half, Fix128::from_int(3) + half)
}

/// One frame of free fall from rest with 8 semi-implicit Euler substeps:
/// `y = -g * scale * h^2 * (1 + 2 + ... + 8) = -g * scale * 36 / 512^2`.
fn free_fall_one_frame(gravity_scale: i64) -> Fix128 {
    let s = substeps();
    let triangular = s * (s + 1) / 2;
    r(-10 * gravity_scale * triangular, 512 * 512)
}

/// The frame-end velocity of a body whose derived velocity is `v`.
fn damped(v: Fix128) -> Fix128 {
    v * PhysicsConfig::default().damping * Fix128::ONE
}

fn plane_floor() -> SdfCollider {
    SdfCollider::new_static(
        Box::new(ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
}

struct CountingModifier {
    calls: Arc<AtomicUsize>,
    halve_depth: bool,
    approve: bool,
}

impl ContactModifier for CountingModifier {
    fn modify_contact(
        &self,
        _body_a: usize,
        _body_b: usize,
        contact: &mut Contact,
        _friction: &mut Fix128,
        _restitution: &mut Fix128,
    ) -> bool {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.halve_depth {
            contact.depth = contact.depth.half();
        }
        self.approve
    }
}

// ---------------------------------------------------------------------------
// RigidBody mutators (reached through get_body_mut) and builders
// ---------------------------------------------------------------------------

/// Oracle: `add_force` documents `v += F * inv_mass * dt`; with `m = 2`,
/// `F = (4, 0, 0)`, `dt = 1/64` that is `Δv = 1/32` exactly. One weightless
/// frame then moves the body `v * dt = 1/2048` and leaves `v * 0.99`.
#[test]
fn add_force_delta_v_is_f_dt_over_m_and_one_frame_moves_v_dt() {
    let mut world = PhysicsWorld::new(weightless());
    let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2)));
    world
        .get_body_mut(idx)
        .expect("just added")
        .add_force(Vec3Fix::from_int(4, 0, 0), dt());
    let body = world.get_body(idx).expect("just added");
    assert_eq!(body.velocity, v3(r(1, 32), Fix128::ZERO, Fix128::ZERO));
    assert_eq!(body.position, Vec3Fix::ZERO, "add_force must not teleport");

    world.step(dt());
    let body = world.get_body(idx).expect("still present");
    assert_eq!(body.position, v3(r(1, 2048), Fix128::ZERO, Fix128::ZERO));
    assert_eq!(body.velocity.x, damped(r(1, 32)));
}

/// Oracle: `add_force` documents that static bodies are unaffected, and a
/// zero `dt` scales the increment to exactly zero.
#[test]
fn add_force_degenerate_static_body_and_zero_dt_change_nothing() {
    let mut world = PhysicsWorld::new(weightless());
    let wall = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let ball = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world
        .get_body_mut(wall)
        .expect("present")
        .add_force(Vec3Fix::from_int(1_000_000, 0, 0), dt());
    world
        .get_body_mut(ball)
        .expect("present")
        .add_force(Vec3Fix::from_int(1_000_000, 0, 0), Fix128::ZERO);
    assert_eq!(world.bodies[wall].velocity, Vec3Fix::ZERO);
    assert_eq!(world.bodies[ball].velocity, Vec3Fix::ZERO);
}

/// Oracle: `add_torque` documents `ω += τ * inv_inertia * dt` with the
/// unit-sphere default inertia `I = (2/5) m` (`RigidBody::new`). For `m = 1`,
/// `τ = (0, 4, 0)`, `dt = 1/64`: `Δω_y = 4 * (5/2) / 64 = 5/32`. `2/5` is
/// not dyadic, so the fixed-point reciprocal is off by a few ulp; the oracle
/// allows `2^-50` (≈ 1e-15), far below one physics step's effect.
#[test]
fn add_torque_delta_omega_is_tau_dt_over_sphere_inertia() {
    let mut world = PhysicsWorld::new(weightless());
    let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world
        .get_body_mut(idx)
        .expect("just added")
        .add_torque(Vec3Fix::from_int(0, 4, 0), dt());
    let w = world.bodies[idx].angular_velocity;
    let err = (w.y - r(5, 32)).abs();
    assert!(
        err < Fix128::from_raw(0, 1 << 14),
        "Δω_y = {} (want 5/32, err {})",
        w.y.to_f64(),
        err.to_f64()
    );
    assert_eq!((w.x, w.z), (Fix128::ZERO, Fix128::ZERO));

    let wall = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    world
        .get_body_mut(wall)
        .expect("present")
        .add_torque(Vec3Fix::from_int(0, 4, 0), dt());
    assert_eq!(world.bodies[wall].angular_velocity, Vec3Fix::ZERO);
}

/// Oracle: `set_velocity` stores the vector; `speed` is its Euclidean norm,
/// `|(3, 4, 0)| = 5` (exact: the digit-by-digit `sqrt` of 25 is 5);
/// `mass` is `1 / inv_mass`, exact for the dyadic mass 4; a static body
/// documents `mass() == 0` ("represents infinite mass").
#[test]
fn set_velocity_speed_and_mass_round_trip_exactly() {
    let mut world = PhysicsWorld::new(weightless());
    let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(4)));
    world
        .get_body_mut(idx)
        .expect("present")
        .set_velocity(Vec3Fix::from_int(3, 4, 0));
    let body = world.get_body(idx).expect("present");
    assert_eq!(body.velocity, Vec3Fix::from_int(3, 4, 0));
    assert_eq!(body.speed(), Fix128::from_int(5));
    assert_eq!(body.mass(), Fix128::from_int(4));

    let wall = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    assert_eq!(world.bodies[wall].mass(), Fix128::ZERO);
    assert_eq!(world.bodies[wall].speed(), Fix128::ZERO);
}

/// Oracle: `set_rotation` documents a direct write of `rotation` (and the
/// XPBD `prev_rotation`); `set_angular_velocity` a direct write. With zero
/// angular velocity `integrate_positions` leaves the rotation untouched, so
/// a frame later the rotation is still the written quaternion, bit for bit.
#[test]
fn set_rotation_and_set_angular_velocity_round_trip() {
    let yaw_180 = QuatFix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let mut world = PhysicsWorld::new(weightless());
    let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    {
        let body = world.get_body_mut(idx).expect("present");
        body.set_rotation(yaw_180);
        body.set_angular_velocity(Vec3Fix::from_int(0, 0, 7));
    }
    let body = world.get_body(idx).expect("present");
    assert_eq!(body.rotation, yaw_180);
    assert_eq!(body.prev_rotation, yaw_180);
    assert_eq!(body.angular_velocity, Vec3Fix::from_int(0, 0, 7));

    world
        .get_body_mut(idx)
        .expect("present")
        .set_angular_velocity(Vec3Fix::ZERO);
    world.step(dt());
    assert_eq!(world.bodies[idx].rotation, yaw_180);
    assert_eq!(world.bodies[idx].angular_velocity, Vec3Fix::ZERO);
}

/// Oracle (default config): `with_velocity((512, 0, 0))` moves the body
/// `8 * 512 * (1/512) = 8` in x during one `1/64 s` frame, while gravity
/// `-10` takes it `-360 / 512^2` in y; the frame ends with `v_x * 0.99`.
#[test]
fn with_velocity_moves_v_dt_per_frame_under_default_config() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let idx = world.add_body(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)
            .with_velocity(Vec3Fix::from_int(512, 0, 0)),
    );
    world.step(dt());
    let body = &world.bodies[idx];
    assert_eq!(body.position.x, Fix128::from_int(8));
    assert_eq!(body.position.y, free_fall_one_frame(1));
    assert_eq!(body.position.z, Fix128::ZERO);
    assert_eq!(body.velocity.x, damped(Fix128::from_int(512)));
}

/// Oracle (default config): gravity scale `s` multiplies the fall,
/// `y = -10 * s * 36 / 512^2`; `s = 0` keeps the body exactly at rest.
#[test]
fn with_gravity_scale_scales_free_fall_and_zero_rests() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let ids: Vec<usize> = [0_i64, 1, 2]
        .into_iter()
        .map(|s| {
            world.add_body(
                RigidBody::new_dynamic(Vec3Fix::from_int(10 * s, 0, 0), Fix128::ONE)
                    .with_gravity_scale(Fix128::from_int(s)),
            )
        })
        .collect();
    world.step(dt());
    for (scale, idx) in ids.iter().enumerate() {
        let body = &world.bodies[*idx];
        let expect = free_fall_one_frame(i64::try_from(scale).expect("tiny"));
        assert_eq!(body.position.y, expect, "gravity_scale {scale}");
    }
    assert_eq!(world.bodies[ids[0]].velocity, Vec3Fix::ZERO);
    assert_eq!(
        world.bodies[ids[0]].position,
        Vec3Fix::ZERO,
        "gravity_scale 0 body must not move at all"
    );
}

/// Oracle: `with_rotation` documents writing both `rotation` and
/// `prev_rotation`; with no angular velocity the rotation survives a frame.
#[test]
fn with_rotation_sets_rotation_and_prev_rotation() {
    let yaw_180 = QuatFix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_rotation(yaw_180);
    assert_eq!(body.rotation, yaw_180);
    assert_eq!(body.prev_rotation, yaw_180);

    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let idx = world.add_body(body);
    world.step(dt());
    assert_eq!(world.bodies[idx].rotation, yaw_180);
}

/// Oracle (default config): the frame-end angular velocity is
/// `ω' * damping * angular_damping`, where `ω'` is the velocity derived
/// from the integrated rotation, `ω_{k} = (2/h) sin(ω_{k-1} h / 2)` per
/// substep. Hence `angular_damping = 0` gives exactly zero, `= 2` gives
/// exactly twice the `= 1` body (doubling is exact in fixed point), and the
/// `= 1` body reads `0.99 * ω'` with `1 - 8 h^2 / 24 ≤ ω' / ω ≤ 1`, i.e.
/// within `1.3e-6` of `0.99` for `ω = 1`, `h = 1/512` (CORDIC error is far
/// smaller); the oracle allows `1e-5`.
#[test]
fn with_angular_damping_scales_frame_end_angular_velocity() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let spin = Vec3Fix::from_int(0, 1, 0);
    let ids: Vec<usize> = [Fix128::ZERO, Fix128::ONE, Fix128::from_int(2)]
        .into_iter()
        .map(|d| {
            let idx = world.add_body(
                RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_angular_damping(d),
            );
            world
                .get_body_mut(idx)
                .expect("present")
                .set_angular_velocity(spin);
            idx
        })
        .collect();
    world.step(dt());
    let w0 = world.bodies[ids[0]].angular_velocity;
    let w1 = world.bodies[ids[1]].angular_velocity;
    let w2 = world.bodies[ids[2]].angular_velocity;
    assert_eq!(w0, Vec3Fix::ZERO, "angular_damping 0 zeroes the spin");
    assert_eq!(w2, w1 + w1, "angular_damping 2 is exactly twice 1");
    let err = (w1.y - r(99, 100)).abs().to_f64();
    assert!(err < 1e-5, "ω_y = {} (want 0.99 ± 1.3e-6)", w1.y.to_f64());
    assert!(w1.y < r(99, 100), "sin(θ/2) < θ/2 so ω' < ω");
    assert_eq!((w1.x, w1.z), (Fix128::ZERO, Fix128::ZERO));
}

/// Oracle: `with_friction` documents "set friction coefficient" — a field
/// write, exact round trip. A body with the default material contributes its
/// own friction to the contact: the pair's combined friction is the Average
/// of `0.25` (body a) and `0.5` (default, body b), i.e. `0.375`; restitution
/// stays the default `(0.3 + 0.3) / 2`.
#[test]
fn with_friction_sets_the_body_field_and_contacts_average_it_with_the_other_body() {
    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = overlapping_pair(&mut world);
    world.bodies[a] = world.bodies[a].with_friction(r(1, 4));
    assert_eq!(world.bodies[a].friction, r(1, 4));
    let combined = world.combined_material(a, b);
    assert_eq!(combined.friction, (r(1, 4) + r(5, 10)).half());
    assert_eq!(combined.restitution, (r(3, 10) + r(3, 10)).half());
}

/// Oracle: the three constructors set `body_type` as named.
#[test]
fn is_dynamic_and_is_kinematic_classify_the_constructors() {
    let dynamic = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    let kinematic = RigidBody::new_kinematic(Vec3Fix::ZERO);
    let fixed = RigidBody::new_static(Vec3Fix::ZERO);
    assert_eq!(
        (dynamic.is_dynamic(), dynamic.is_kinematic()),
        (true, false)
    );
    assert_eq!(
        (kinematic.is_dynamic(), kinematic.is_kinematic()),
        (false, true)
    );
    assert_eq!((fixed.is_dynamic(), fixed.is_kinematic()), (false, false));
}

/// Oracle: a zero mass makes `inv_mass = 0` ("static/infinite mass"), so
/// `mass()` reports 0, `is_static()` is true, forces and torques are
/// ignored, while `body_type` stays `Dynamic` as constructed.
#[test]
fn zero_mass_body_is_static_for_forces_but_typed_dynamic() {
    let mut body = RigidBody::new(Vec3Fix::ZERO, Fix128::ZERO);
    assert_eq!(body.mass(), Fix128::ZERO);
    assert!(body.is_static());
    assert!(body.is_dynamic());
    body.add_force(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
    body.add_torque(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
    assert_eq!(body.velocity, Vec3Fix::ZERO);
    assert_eq!(body.angular_velocity, Vec3Fix::ZERO);
}

// ---------------------------------------------------------------------------
// Body bookkeeping
// ---------------------------------------------------------------------------

/// Oracle: `get_body` is `Some` for every index below the count and `None`
/// at and beyond it; `get_body_mut` writes through to the stored body.
#[test]
fn get_body_and_get_body_mut_follow_the_index_contract() {
    let mut world = PhysicsWorld::new(weightless());
    let idx = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(1, 2, 3),
        Fix128::ONE,
    ));
    assert_eq!(
        world.get_body(idx).map(|b| b.position),
        Some(Vec3Fix::from_int(1, 2, 3))
    );
    assert!(world.get_body(idx + 1).is_none());
    assert!(world.get_body(usize::MAX).is_none());
    assert!(world.get_body_mut(idx + 1).is_none());
    world
        .get_body_mut(idx)
        .expect("present")
        .set_position(Vec3Fix::from_int(4, 5, 6));
    assert_eq!(world.bodies[idx].position, Vec3Fix::from_int(4, 5, 6));
}

/// Oracle: `remove_body` documents swap-remove — the removed body comes
/// back, the last body fills its slot, the count drops by one, and an
/// out-of-range index returns `None`. Nothing sleeps here, so
/// `active_body_count` equals the body count throughout.
#[test]
fn remove_body_swap_removes_and_active_count_drops_by_one() {
    let mut world = PhysicsWorld::new(weightless());
    for i in 0..3 {
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(i, 0, 0),
            Fix128::ONE,
        ));
    }
    assert_eq!(world.active_body_count(), 3);

    let removed = world.remove_body(0).expect("index 0 exists");
    assert_eq!(removed.position, Vec3Fix::ZERO);
    assert_eq!(world.body_count(), 2);
    assert_eq!(world.active_body_count(), 2);
    assert_eq!(
        world.get_body(0).map(|b| b.position),
        Some(Vec3Fix::from_int(2, 0, 0)),
        "the last body moves into the freed slot"
    );
    assert!(world.get_body(2).is_none());

    assert!(world.remove_body(2).is_none(), "out of range");
    assert!(world.remove_body(usize::MAX).is_none());
    assert_eq!(world.active_body_count(), 2);

    assert!(world.remove_body(0).is_some());
    assert!(world.remove_body(0).is_some());
    assert!(
        world.remove_body(0).is_none(),
        "removing from an empty world"
    );
    assert_eq!(world.active_body_count(), 0);
}

/// Oracle: with the default `SleepConfig` a body idle for `frames_to_sleep`
/// (60) consecutive frames sleeps (`idle_frames >= 60` at the 60th frame,
/// not the 59th). `wake_body` resets the counter, so one more idle frame
/// leaves it awake, and `active_body_count` follows.
#[test]
fn wake_body_after_sleep_restores_active_body_count() {
    let frames = PhysicsWorld::new(weightless())
        .islands
        .config
        .frames_to_sleep;
    assert_eq!(frames, 60, "oracle derived for the default sleep config");

    let mut world = PhysicsWorld::new(weightless());
    let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    for _ in 0..frames - 1 {
        world.step(dt());
    }
    assert!(!world.is_sleeping(idx), "awake after 59 idle frames");
    assert_eq!(world.active_body_count(), 1);
    world.step(dt());
    assert!(world.is_sleeping(idx), "asleep at the 60th idle frame");
    assert_eq!(world.active_body_count(), 0);

    world.wake_body(idx);
    assert!(!world.is_sleeping(idx));
    assert_eq!(world.active_body_count(), 1);
    world.step(dt());
    assert!(!world.is_sleeping(idx), "wake resets idle_frames to 0");
    assert_eq!(world.active_body_count(), 1);
}

/// Oracle: `IslandManager::update_sleep` marks static bodies as sleeping
/// on every frame, so a static body leaves `active_body_count` after the
/// first step while a moving dynamic body stays counted.
#[test]
fn static_body_leaves_active_body_count_after_the_first_step() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 10, 0),
        Fix128::ONE,
    ));
    assert_eq!(
        world.active_body_count(),
        2,
        "nothing has been classified yet"
    );
    world.step(dt());
    assert_eq!(world.active_body_count(), 1);
}

// ---------------------------------------------------------------------------
// Contacts, hooks, modifiers, events
// ---------------------------------------------------------------------------

/// Oracle: the depth-1 pair produces exactly one `Begin` event for the
/// ordered pair `(0, 1)` with depth 1 and a unit normal along x (the sign
/// follows whichever order the broad phase reports the pair in — the
/// `ContactEvent::normal` doc says "A to B", `Contact::normal` says "B to
/// A", so the oracle pins magnitude and axis only), both bodies end the
/// frame at `-4` / `7`, a second drain is empty, and the next frame reports
/// exactly one `End` event with the zeroed geometry `end_frame` documents.
#[test]
fn drain_contact_events_returns_the_begin_event_once_then_the_end_event() {
    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = overlapping_pair(&mut world);
    world.step(dt());
    let (xa, xb) = separated_x();
    assert_eq!(world.bodies[a].position.x, xa);
    assert_eq!(world.bodies[b].position.x, xb);

    let events = world.drain_contact_events();
    assert_eq!(events.len(), 1);
    let e = events[0];
    assert_eq!(
        (e.body_a, e.body_b, e.event_type),
        (0, 1, ContactEventType::Begin)
    );
    assert_eq!(e.depth, Fix128::ONE);
    // Unit normal: `normalize_with_length` multiplies by the truncated
    // reciprocal `1/3`, so `|n_x|` reads `1 - 2^-64` (measured 1 ulp); the
    // bound allows 4 ulp.
    let unit_err = (e.normal.x.abs() - Fix128::ONE).abs();
    assert!(
        unit_err <= Fix128::from_raw(0, 4),
        "|n_x| off by {} ulp",
        unit_err.lo
    );
    assert_eq!((e.normal.y, e.normal.z), (Fix128::ZERO, Fix128::ZERO));
    assert!(world.drain_contact_events().is_empty(), "second drain");
    assert!(world.contact_events().is_empty());

    world.step(dt());
    let events = world.drain_contact_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, ContactEventType::End);
    assert_eq!((events[0].body_a, events[0].body_b), (0, 1));
    assert_eq!(events[0].depth, Fix128::ZERO);
    assert_eq!(events[0].normal, Vec3Fix::ZERO);
}

/// Oracle: a sensor overlap produces a trigger and no physics response, so
/// both bodies keep their positions bit for bit; the pair enters once
/// (frame 1), stays silent while it persists (frame 2), and exits once the
/// sensor's partner is moved away (frame 3).
///
/// `detect_collisions` reports every overlap to the contact-event collector
/// before branching on `is_sensor`, so the same overlap also shows up as
/// one `Begin` contact event (no constraint is created). Whether that double
/// report is intended is for the owner of `src/solver.rs`; the oracle pins
/// the count so a change is visible.
#[test]
fn with_sensor_reports_trigger_events_enter_once_then_exit() {
    let mut world = PhysicsWorld::new(weightless());
    let radius = Fix128::from_int(2);
    let zone = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_sensor(true),
        radius,
    );
    let visitor = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(3, 0, 0), Fix128::ONE),
        radius,
    );
    world.step(dt());
    assert_eq!(world.bodies[zone].position, Vec3Fix::ZERO);
    assert_eq!(world.bodies[visitor].position, Vec3Fix::from_int(3, 0, 0));
    assert_eq!(
        world.contact_events().len(),
        1,
        "the overlap is reported once as a contact event too"
    );
    assert_eq!(
        world.contact_events()[0].event_type,
        ContactEventType::Begin
    );
    assert!(
        world.contact_constraints.is_empty(),
        "no constraint for a sensor"
    );
    assert_eq!(world.trigger_events().len(), 1);
    let t = world.trigger_events()[0];
    let mut pair = [t.trigger_body, t.other_body];
    pair.sort_unstable();
    assert_eq!(pair, [zone, visitor]);
    assert!(t.entered);
    let drained = world.drain_trigger_events();
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0], t);
    assert!(world.drain_trigger_events().is_empty(), "second drain");
    assert!(world.trigger_events().is_empty());

    world.step(dt());
    assert!(
        world.trigger_events().is_empty(),
        "persisting overlap is silent"
    );

    world
        .get_body_mut(visitor)
        .expect("present")
        .set_position(Vec3Fix::from_int(10, 0, 0));
    world.step(dt());
    let exits = world.drain_trigger_events();
    assert_eq!(exits.len(), 1);
    assert!(!exits[0].entered, "leaving the zone is an exit event");
}

/// Oracle: `CollisionFilter::NONE` (layer 0, mask 0) fails
/// `can_collide` against anything, and two bodies in the same non-zero
/// group never collide; either way the pair neither moves nor reports.
#[test]
fn set_body_filter_prevents_the_contact() {
    for (name, filter_a, filter_b) in [
        ("NONE", CollisionFilter::NONE, CollisionFilter::DEFAULT),
        (
            "same group",
            CollisionFilter::DEFAULT.with_group(7),
            CollisionFilter::DEFAULT.with_group(7),
        ),
    ] {
        let mut world = PhysicsWorld::new(weightless());
        let (a, b) = overlapping_pair(&mut world);
        world.set_body_filter(a, filter_a);
        world.set_body_filter(b, filter_b);
        world.step(dt());
        assert_eq!(world.bodies[a].position, Vec3Fix::ZERO, "{name}");
        assert_eq!(
            world.bodies[b].position,
            Vec3Fix::from_int(3, 0, 0),
            "{name}"
        );
        assert!(world.contact_events().is_empty(), "{name}");
    }
}

/// Oracle: a pre-solve hook returning `false` skips the contact, so the
/// pair never separates and the solver asks again in every substep
/// (`substeps * iterations = 8` calls per frame); an approving hook is
/// asked once per frame because the first substep already separates the
/// pair. `clear_pre_solve_hooks` restores the plain `-4` / `7` frame.
#[test]
fn add_pre_solve_hook_veto_freezes_the_pair_and_clear_restores_it() {
    let iterations = i64::try_from(PhysicsConfig::default().iterations).expect("small");
    let per_frame_when_frozen = usize::try_from(substeps() * iterations).expect("small");

    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = overlapping_pair(&mut world);
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    world.add_pre_solve_hook(Box::new(move |_a, _b, _c: &Contact| {
        counter.fetch_add(1, Ordering::SeqCst);
        false
    }));
    world.step(dt());
    assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
    assert_eq!(world.bodies[b].position, Vec3Fix::from_int(3, 0, 0));
    assert_eq!(calls.load(Ordering::SeqCst), per_frame_when_frozen);
    assert_eq!(
        world.contact_events().len(),
        1,
        "the hook vetoes the response, not the detection"
    );

    world.clear_pre_solve_hooks();
    world.step(dt());
    let (xa, xb) = separated_x();
    assert_eq!(world.bodies[a].position.x, xa);
    assert_eq!(world.bodies[b].position.x, xb);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        per_frame_when_frozen,
        "cleared"
    );

    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = overlapping_pair(&mut world);
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    world.add_pre_solve_hook(Box::new(move |_a, _b, _c: &Contact| {
        counter.fetch_add(1, Ordering::SeqCst);
        true
    }));
    world.step(dt());
    assert_eq!(world.bodies[a].position.x, xa);
    assert_eq!(world.bodies[b].position.x, xb);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "approved once, then separated"
    );
}

/// Oracle: a modifier that halves the depth halves the separation
/// (`-2` / `5` instead of `-4` / `7`) and is consulted once; a vetoing
/// modifier freezes the pair and is consulted 8 times; after
/// `clear_contact_modifiers` a reset pair separates to `-4` / `7` again.
#[test]
fn add_contact_modifier_halves_or_vetoes_and_clear_restores() {
    let (xa, xb) = separated_x();
    let half = r(substeps(), 4);

    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = overlapping_pair(&mut world);
    let calls = Arc::new(AtomicUsize::new(0));
    world.add_contact_modifier(Box::new(CountingModifier {
        calls: Arc::clone(&calls),
        halve_depth: true,
        approve: true,
    }));
    world.step(dt());
    assert_eq!(world.bodies[a].position.x, -half);
    assert_eq!(world.bodies[b].position.x, Fix128::from_int(3) + half);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    world.clear_contact_modifiers();
    for (idx, x) in [(a, 0), (b, 3)] {
        let body = world.get_body_mut(idx).expect("present");
        body.set_position(Vec3Fix::from_int(x, 0, 0));
        body.set_velocity(Vec3Fix::ZERO);
    }
    world.step(dt());
    assert_eq!(world.bodies[a].position.x, xa);
    assert_eq!(world.bodies[b].position.x, xb);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "cleared modifier is silent"
    );

    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = overlapping_pair(&mut world);
    let calls = Arc::new(AtomicUsize::new(0));
    world.add_contact_modifier(Box::new(CountingModifier {
        calls: Arc::clone(&calls),
        halve_depth: false,
        approve: false,
    }));
    world.step(dt());
    assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
    assert_eq!(world.bodies[b].position, Vec3Fix::from_int(3, 0, 0));
    assert_eq!(
        calls.load(Ordering::SeqCst),
        usize::try_from(substeps()).expect("small")
    );
}

/// Oracle: bodies added without a radius are invisible to detection, so
/// the overlapping pair stays put; `set_body_collision_radius(2)` on both
/// turns the same scene into the `-4` / `7` frame, and clearing one radius
/// leaves fewer than two primitives, so nothing happens again.
#[test]
fn set_body_collision_radius_enables_contact_and_clear_disables_it() {
    let mut world = PhysicsWorld::new(weightless());
    let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(3, 0, 0),
        Fix128::ONE,
    ));
    world.step(dt());
    assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
    assert_eq!(world.bodies[b].position, Vec3Fix::from_int(3, 0, 0));

    world.set_body_collision_radius(a, Fix128::from_int(2));
    world.set_body_collision_radius(b, Fix128::from_int(2));
    world.step(dt());
    let (xa, xb) = separated_x();
    assert_eq!(world.bodies[a].position.x, xa);
    assert_eq!(world.bodies[b].position.x, xb);

    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = overlapping_pair(&mut world);
    world.clear_body_collision_radius(b);
    world.step(dt());
    assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
    assert_eq!(world.bodies[b].position, Vec3Fix::from_int(3, 0, 0));
    assert!(world.contact_events().is_empty());
}

/// Oracle: `MaterialTable` combines with the `Average` rule by default, so
/// a rubber body (`μ = 1`, `e = 1/2`) against the default material
/// (`μ = 1/2`, `e = 3/10`) gives `μ = 3/4`, `e = (3/10 + 1/2) / 2`;
/// `set_body_material` on an out-of-range index is ignored.
#[test]
fn set_body_material_changes_the_combined_material() {
    let mut world = PhysicsWorld::new(weightless());
    let (a, b) = overlapping_pair(&mut world);
    let rubber = world
        .material_table
        .register(PhysicsMaterial::new(0, Fix128::ONE, r(1, 2)));
    world.set_body_material(a, rubber);
    let combined = world.combined_material(a, b);
    assert_eq!(combined.friction, r(3, 4));
    assert_eq!(combined.restitution, (r(3, 10) + r(1, 2)).half());

    world.set_body_material(99, rubber);
    let still_default = world.combined_material(b, b);
    assert_eq!(still_default.friction, r(1, 2));
}

// ---------------------------------------------------------------------------
// Force fields
// ---------------------------------------------------------------------------

/// Oracle (zero gravity, `damping = 1`): a directional field of strength 8
/// on a 2 kg body adds `8 * (1/2) * dt = 1/16` to `v` before the substep
/// loop, so frame 1 moves `1/16 * 1/64 = 1/1024`. After
/// `remove_force_field` the body coasts: frame 2 adds another `1/1024`;
/// with the field kept it would add `2/1024`.
#[test]
fn add_force_field_accelerates_and_remove_force_field_stops_it() {
    let field = ForceFieldInstance::new(ForceField::Directional {
        direction: Vec3Fix::UNIT_X,
        strength: Fix128::from_int(8),
    });

    let mut coasting = PhysicsWorld::new(weightless_undamped());
    let idx = coasting.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2)));
    let wind = coasting.add_force_field(field.clone());
    coasting.step(dt());
    assert_eq!(coasting.bodies[idx].position.x, r(1, 1024));
    assert_eq!(coasting.bodies[idx].velocity.x, r(1, 16));
    assert!(coasting.remove_force_field(wind).is_some());
    assert!(coasting.remove_force_field(wind).is_none(), "removed twice");
    coasting.step(dt());
    assert_eq!(coasting.bodies[idx].position.x, r(2, 1024));
    assert_eq!(coasting.bodies[idx].velocity.x, r(1, 16));

    let mut pushed = PhysicsWorld::new(weightless_undamped());
    let idx = pushed.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2)));
    pushed.add_force_field(field);
    pushed.step(dt());
    pushed.step(dt());
    assert_eq!(pushed.bodies[idx].position.x, r(3, 1024));
    assert_eq!(pushed.bodies[idx].velocity.x, r(2, 16));
}

// ---------------------------------------------------------------------------
// SDF colliders
// ---------------------------------------------------------------------------

/// Oracle: a unit-mass body resting at `y_0 = 1/4` over the plane `φ = y`
/// with query radius `r` penetrates by `r - y_0`; the push-out in the
/// first substep becomes the derived velocity `(r - y_0) / h`, which carries
/// it `r - y_0` further in each of the remaining substeps:
/// `y = y_0 + 8 (r - y_0)` = `2.25` for the default `r = 1/2` and `4.25`
/// after `set_sdf_collision_radius(3/4)`. All values are dyadic, so the
/// `f32` field evaluation is exact.
#[test]
fn add_sdf_collider_pushes_out_by_substeps_times_penetration() {
    let y0 = r(1, 4);
    for (radius, set_radius) in [(r(1, 2), false), (r(3, 4), true)] {
        let mut world = PhysicsWorld::new(weightless());
        let ball = world.add_body(RigidBody::new_dynamic(
            v3(Fix128::ZERO, y0, Fix128::ZERO),
            Fix128::ONE,
        ));
        world.add_sdf_collider(plane_floor());
        if set_radius {
            world.set_sdf_collision_radius(radius);
        }
        world.step(dt());
        let expect = y0 + (radius - y0) * Fix128::from_int(substeps());
        assert_eq!(
            world.bodies[ball].position.y,
            expect,
            "radius {}",
            radius.to_f64()
        );
        assert_eq!(
            world.bodies[ball].velocity.y,
            damped((radius - y0) * Fix128::from_int(512))
        );
        assert_eq!(world.bodies[ball].position.x, Fix128::ZERO);
    }
}

/// Oracle: `remove_sdf_collider` returns the collider and leaves the body
/// unsupported, so it stays exactly at `y_0` in zero gravity; a second
/// removal of the same index is `None`; a zero query radius never
/// penetrates (`0 - y_0 < 0`), so the body stays put with the floor present.
#[test]
fn remove_sdf_collider_and_zero_radius_leave_the_body_in_place() {
    let start = v3(Fix128::ZERO, r(1, 4), Fix128::ZERO);
    let mut world = PhysicsWorld::new(weightless());
    let ball = world.add_body(RigidBody::new_dynamic(start, Fix128::ONE));
    let floor = world.add_sdf_collider(plane_floor());
    assert!(world.remove_sdf_collider(floor).is_some());
    assert!(world.remove_sdf_collider(floor).is_none(), "removed twice");
    assert!(world.remove_sdf_collider(usize::MAX).is_none());
    world.step(dt());
    assert_eq!(world.bodies[ball].position, start);

    let mut world = PhysicsWorld::new(weightless());
    let ball = world.add_body(RigidBody::new_dynamic(start, Fix128::ONE));
    world.add_sdf_collider(plane_floor());
    world.set_sdf_collision_radius(Fix128::ZERO);
    world.step(dt());
    assert_eq!(world.bodies[ball].position, start);
}

// ---------------------------------------------------------------------------
// Joints and batches
// ---------------------------------------------------------------------------

/// Oracle: `joint_count` counts additions and swap-removals; removing an
/// index twice yields `None` the second time; `add_joint` documents a
/// panic for an out-of-range body index.
#[test]
fn joint_count_and_remove_joint_track_the_joint_list() {
    let mut world = PhysicsWorld::new(weightless());
    let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(1, 0, 0),
        Fix128::ONE,
    ));
    assert_eq!(world.joint_count(), 0);
    let j = world.add_joint(Joint::Ball(BallJoint::new(
        a,
        b,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
    assert_eq!(world.joint_count(), 1);
    assert!(world.remove_joint(j).is_some());
    assert_eq!(world.joint_count(), 0);
    assert!(world.remove_joint(j).is_none(), "removed twice");
    assert!(world.remove_joint(usize::MAX).is_none());

    let out_of_range = catch_unwind(AssertUnwindSafe(|| {
        world.add_joint(Joint::Ball(BallJoint::new(
            a,
            99,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        )))
    }));
    assert!(out_of_range.is_err(), "documented `# Panics`");
    assert_eq!(world.joint_count(), 0);
}

/// Oracle: greedy edge colouring in index order. A path `0-1, 1-2, 2-3`
/// of dynamic bodies needs 2 colours (constraint 1 shares body 1 with
/// constraint 0, constraint 2 shares body 2 with constraint 1 but not body
/// with constraint 0); two disjoint constraints need 1; a static hub
/// imposes no colouring constraint, so `0-S, S-2` also needs 1; no
/// constraints need 0.
#[test]
fn num_batches_counts_the_greedy_colours() {
    fn rod(a: usize, b: usize) -> DistanceConstraint {
        DistanceConstraint {
            body_a: a,
            body_b: b,
            local_anchor_a: Vec3Fix::ZERO,
            local_anchor_b: Vec3Fix::ZERO,
            target_distance: Fix128::ONE,
            compliance: Fix128::ZERO,
            cached_lambda: Fix128::ZERO,
        }
    }
    /// (scene name, static body indices, rods as body pairs, expected colours)
    struct Case {
        name: &'static str,
        statics: &'static [usize],
        rods: &'static [(usize, usize)],
        expect: usize,
    }
    let cases = [
        Case {
            name: "path of 4",
            statics: &[],
            rods: &[(0, 1), (1, 2), (2, 3)],
            expect: 2,
        },
        Case {
            name: "two disjoint",
            statics: &[],
            rods: &[(0, 1), (2, 3)],
            expect: 1,
        },
        Case {
            name: "static hub",
            statics: &[1],
            rods: &[(0, 1), (1, 2)],
            expect: 1,
        },
        Case {
            name: "none",
            statics: &[],
            rods: &[],
            expect: 0,
        },
    ];
    for Case {
        name,
        statics,
        rods,
        expect,
    } in cases
    {
        let mut world = PhysicsWorld::new(weightless());
        for i in 0..4_i64 {
            let pos = Vec3Fix::from_int(i, 0, 0);
            let idx = usize::try_from(i).expect("small");
            world.add_body(if statics.contains(&idx) {
                RigidBody::new_static(pos)
            } else {
                RigidBody::new_dynamic(pos, Fix128::ONE)
            });
        }
        for &(a, b) in rods {
            world.add_distance_constraint(rod(a, b));
        }
        world.rebuild_batches();
        assert_eq!(world.num_batches(), expect, "{name}");
    }
}

// ---------------------------------------------------------------------------
// Degenerate inputs on the world
// ---------------------------------------------------------------------------

/// Oracle: `step` documents "non-positive dt produces no physics update",
/// so the serialized state is byte-identical before and after, even under
/// gravity with pending events.
#[test]
fn step_with_zero_or_negative_dt_changes_nothing() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    overlapping_pair(&mut world);
    world.step(dt());
    let before = world.serialize_state();
    world.step(Fix128::ZERO);
    assert_eq!(world.serialize_state(), before, "dt = 0");
    world.step(-dt());
    assert_eq!(world.serialize_state(), before, "dt < 0");
}

/// Oracle: the per-body setters guard `body_idx < len` and ignore anything
/// else, so a world that received out-of-range calls steps to the same
/// bytes as one that did not.
#[test]
fn out_of_range_body_setters_are_ignored() {
    let build = |poke: bool| {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());
        overlapping_pair(&mut world);
        if poke {
            world.set_body_collision_radius(99, Fix128::from_int(5));
            world.clear_body_collision_radius(99);
            world.set_body_filter(99, CollisionFilter::NONE);
            world.set_body_material(99, 1);
        }
        world.step(dt());
        world.serialize_state()
    };
    assert_eq!(build(true), build(false));
}

/// Oracle: the sibling per-body entry points (`set_body_filter`,
/// `set_body_collision_radius`, `set_body_material`, `remove_body`,
/// `get_body`) all treat an out-of-range index as a no-op / `None`, and
/// `wake_body` documents no panic, so the same contract is expected here:
/// no panic and no state change.
///
/// Measured 2026-10-03: `wake_body(99)` on a 2-body world panics with
/// "index out of bounds: the len is 2 but the index is 99" at
/// `src/sleeping.rs:152` (`IslandManager::find`, reached from
/// `wake_island`). Pinned red until `src/` guards the index.
#[test]
fn wake_body_out_of_range_is_ignored() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    overlapping_pair(&mut world);
    world.step(dt());
    let before = world.serialize_state();
    let result = catch_unwind(AssertUnwindSafe(|| world.wake_body(99)));
    assert!(result.is_ok(), "wake_body(99) must not panic");
    assert_eq!(world.serialize_state(), before);
}

/// Oracle: `overflow_detected` documents that a position increment leaving
/// the `Fix128` range leaves the position untouched and sets the sticky
/// flag. `v = 2^62` (via `add_force` with `F = 2^62`, `m = 1`, `dt = 1`)
/// times a substep of `4 s` (`step(32)`) is `2^64`, out of range: the
/// position stays, the flag is set; the same body with `step(1/64)` moves
/// `2^62 / 64 = 2^56` with the flag clear.
#[test]
fn huge_force_overflows_the_integrator_and_sets_the_sticky_flag() {
    let huge = Vec3Fix::new(Fix128::from_int(1 << 62), Fix128::ZERO, Fix128::ZERO);

    let mut world = PhysicsWorld::new(weightless());
    let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world
        .get_body_mut(idx)
        .expect("present")
        .add_force(huge, Fix128::ONE);
    assert_eq!(world.bodies[idx].velocity, huge);
    assert!(!world.overflow_detected());
    world.step(Fix128::from_int(32));
    assert!(world.overflow_detected(), "2^62 * 4 s is out of range");
    assert_eq!(world.bodies[idx].position, Vec3Fix::ZERO);
    world.step(dt());
    assert!(world.overflow_detected(), "sticky");

    let mut world = PhysicsWorld::new(weightless());
    let idx = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world
        .get_body_mut(idx)
        .expect("present")
        .add_force(huge, Fix128::ONE);
    world.step(dt());
    assert!(!world.overflow_detected());
    assert_eq!(world.bodies[idx].position.x, Fix128::from_int(1 << 56));
}

// ---------------------------------------------------------------------------
// GPU solver bridge (feature gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-solver-bridge")]
mod bridge {
    use super::{dt, overlapping_pair, separated_x, substeps, weightless};
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::math::{Fix128, Vec3Fix};
    use alice_physics::solver::{ContactConstraint, PhysicsConfig, PhysicsWorld};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Counts contact-solve dispatches and leaves every buffer untouched,
    /// so routed contacts are never resolved: a routed frame leaves the pair
    /// exactly where it started, which is the observable that separates
    /// "routed through the bridge" from "solved on the CPU".
    struct CountingBridge {
        dispatches: Arc<AtomicUsize>,
    }

    impl GpuSolverBridge for CountingBridge {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _fixture: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, _c: &[ContactConstraint]) {}
        fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {}
        fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {
            self.dispatches.fetch_add(1, Ordering::SeqCst);
        }
        fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
        fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
    }

    fn per_frame() -> usize {
        let iterations = i64::try_from(PhysicsConfig::default().iterations).expect("small");
        usize::try_from(substeps() * iterations).expect("small")
    }

    /// Oracle: `step_with_bridge` documents routing the contact solve of
    /// every substep through the caller's bridge, so a frame dispatches
    /// `substeps * iterations = 8` times (the pair never separates, so the
    /// contact is present in every substep) and the no-op bridge leaves
    /// both bodies at their start; a CPU `step` on the same scene gives the
    /// `-4` / `7` frame.
    #[test]
    fn step_with_bridge_dispatches_every_substep_and_a_no_op_bridge_leaves_bodies_in_place() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let mut bridge = CountingBridge {
            dispatches: Arc::clone(&dispatches),
        };
        let mut world = PhysicsWorld::new(weightless());
        let (a, b) = overlapping_pair(&mut world);
        world.step_with_bridge(&mut bridge, dt());
        assert_eq!(dispatches.load(Ordering::SeqCst), per_frame());
        assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
        assert_eq!(world.bodies[b].position, Vec3Fix::from_int(3, 0, 0));
        assert_eq!(
            world.contact_events().len(),
            1,
            "detection still runs on the CPU"
        );

        world.step(dt());
        let (xa, xb) = separated_x();
        assert_eq!(world.bodies[a].position.x, xa);
        assert_eq!(world.bodies[b].position.x, xb);
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            per_frame(),
            "CPU step did not route"
        );
    }

    /// Oracle: one `substep_with_bridge` dispatches `iterations` (1) times
    /// and, like `step_with_bridge`, ignores non-positive `dt` only at the
    /// frame level — a substep with the no-op bridge still leaves the pair
    /// in place because nothing resolves the contact.
    #[test]
    fn substep_with_bridge_dispatches_iterations_times() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let mut bridge = CountingBridge {
            dispatches: Arc::clone(&dispatches),
        };
        let mut world = PhysicsWorld::new(weightless());
        let (a, b) = overlapping_pair(&mut world);
        world.substep_with_bridge(&mut bridge, dt());
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            PhysicsConfig::default().iterations
        );
        assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
        assert_eq!(world.bodies[b].position, Vec3Fix::from_int(3, 0, 0));
    }

    /// Oracle: `set_gpu_solver_bridge(Some)` makes `gpu_solver_bridge_installed`
    /// true and routes plain `step` (8 dispatches, pair frozen);
    /// `take_gpu_solver_bridge` returns it once (`None` the second time),
    /// after which `step` is the CPU `-4` / `7` frame again.
    #[test]
    fn set_and_take_gpu_solver_bridge_lifecycle() {
        let dispatches = Arc::new(AtomicUsize::new(0));
        let mut world = PhysicsWorld::new(weightless());
        let (a, b) = overlapping_pair(&mut world);
        assert!(!world.gpu_solver_bridge_installed());
        assert!(
            world.take_gpu_solver_bridge().is_none(),
            "nothing installed yet"
        );

        world.set_gpu_solver_bridge(Some(Box::new(CountingBridge {
            dispatches: Arc::clone(&dispatches),
        })));
        assert!(world.gpu_solver_bridge_installed());
        world.step(dt());
        assert_eq!(dispatches.load(Ordering::SeqCst), per_frame());
        assert_eq!(world.bodies[a].position, Vec3Fix::ZERO);
        assert_eq!(world.bodies[b].position, Vec3Fix::from_int(3, 0, 0));

        assert!(world.take_gpu_solver_bridge().is_some());
        assert!(!world.gpu_solver_bridge_installed());
        assert!(world.take_gpu_solver_bridge().is_none(), "taken twice");
        world.step(dt());
        let (xa, xb) = separated_x();
        assert_eq!(world.bodies[a].position.x, xa);
        assert_eq!(world.bodies[b].position.x, xb);
        assert_eq!(
            dispatches.load(Ordering::SeqCst),
            per_frame(),
            "no longer routed"
        );

        world.set_gpu_solver_bridge(None);
        assert!(
            !world.gpu_solver_bridge_installed(),
            "Some(None) installs nothing"
        );
    }
}
