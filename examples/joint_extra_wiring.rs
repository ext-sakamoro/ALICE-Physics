//! Production entry point for `alice_physics::joint_extra`:
//! `ExtraJoint`, `MouseJoint::set_target`, `solve_extra_joints`,
//! `PulleyJoint::total_length`, and `WeldJoint::with_break_torque`.
//!
//! Wiring: `scripts/wiring_guard.py` lists all five as `unwired` in
//! `scripts/wiring-baseline.txt`. Nothing in `src/` / `examples/` /
//! `benches/` called any of them before this file existed (the module's
//! own `#[cfg(all(test, feature = "std"))] mod tests` does not count as a
//! production caller for the guard).
//!
//! `WeldJoint::with_break_force` is ALSO wired here even though the guard
//! does not flag it. That is a known false positive of the guard's name-only
//! matching: `src/joint.rs` defines an UNRELATED `with_break_force` (on
//! `BallJoint` / `HingeJoint` / `FixedJoint` / ... ) with a real production
//! caller in `examples/joint_limits_and_breaking.rs`, and the guard matches
//! bare identifiers across files, so that caller silently "resolves"
//! `joint_extra.rs`'s own `WeldJoint::with_break_force` too, even though it
//! had zero real callers of its own. See the project Backlog entry for the
//! name-collision backstory.
//!
//! ```bash
//! cargo run --example joint_extra_wiring --features std
//! ```
//!
//! # Closed forms (all values dyadic, exact in `Fix128` unless noted)
//!
//! * **`PulleyJoint::total_length`**: with both anchors at each body's
//!   origin and identity rotations, `world_a == body_a.position` and
//!   `world_b == body_b.position`, so `len_a` / `len_b` are plain
//!   Euclidean distances to the ground anchors. `Fix128::sqrt` is an exact
//!   `floor(sqrt(x))` digit-recurrence (see its doc comment), so a radicand
//!   that is a perfect square -- `3-4-5` / `6-8-10` here -- returns the
//!   exact integer root, not an approximation. `total_length` is then
//!   `len_a + ratio * len_b`, a single multiply-add of exact dyadic values.
//! * **`MouseJoint::set_target` + `solve_extra_joints` (Mouse)**: from rest
//!   (`velocity = 0`, so the damping term is exactly zero), one XPBD step
//!   moves the body by `direction * (stiffness * distance * dt) * dt * inv_mass`
//!   whenever that force stays under `max_force`. With `distance`, `dt`,
//!   `stiffness` and `inv_mass` all dyadic, the landed position is exact.
//! * **`WeldJoint::with_break_force` (and the `with_break_torque`
//!   name-collision backstory item)**: `compute_force` is a plain vector
//!   `.length()` (no division), so an axis-aligned separation is exact with
//!   no CORDIC involved. The break test is strict (`separation >
//!   break_force`), so `separation == break_force` must NOT break. When the
//!   weld holds (not broken) and body A is static (`inv_mass = 0`), the
//!   whole positional correction lands on body B: `w_sum = inv_mass_b`, so
//!   `inv_w_sum = 1 / inv_mass_b`, `lambda = distance / inv_mass_b`, and
//!   `correction * inv_mass_b = distance` exactly -- body B lands exactly on
//!   body A's anchor in a single solve call. The separation chosen below
//!   (`4`, along a single axis) is also a power of two, so the direction
//!   vector's `1/length` division inside `normalize_with_length` is itself
//!   exact, keeping the whole chain bit-exact (a `3-4-5` separation would
//!   still give an exact `length()`, but `1/5` is not exact in `Fix128`,
//!   so the position landing would only be approximate).
//! * **`WeldJoint::with_break_torque`**: with `local_rotation` and
//!   `body_a.rotation` both `QuatFix::IDENTITY`, `compute_torque` reduces to
//!   the magnitude of `body_b.rotation`'s `(x, y, z)` part. For a rotation
//!   of angle `pi/3` about a unit axis, `from_axis_angle` computes
//!   `sin(half_angle) = sin(pi/6)`, which is exactly `1/2` in real
//!   arithmetic -- but `sin_cos` is a 48-iteration CORDIC routine (documented
//!   `~2^-48` error per call), so the comparison below uses a `2^-40`
//!   tolerance (256x that per-call bound) rather than bit-exact equality.
//!
//! Author: Moroya Sakamoto

use alice_physics::joint_extra::{
    solve_extra_joints, solve_pulley_to_length, ExtraJoint, MouseJoint, PulleyJoint, WeldJoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// `2^-40`: 256x the documented `~2^-48` per-call CORDIC error, for the one
/// `sin_cos` call inside `QuatFix::from_axis_angle`.
fn cordic_tol() -> Fix128 {
    Fix128::from_raw(0, 1 << 24) // 2^24 / 2^64 = 2^-40
}

fn near(a: Fix128, b: Fix128, tol: Fix128) -> bool {
    (a - b).abs() < tol
}

/// Static A at the origin and a dynamic B with unit mass and unit inverse
/// inertia, both at the origin.
fn pair() -> Vec<RigidBody> {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
    vec![RigidBody::new_static(Vec3Fix::ZERO), b]
}

fn main() {
    let dt = q(1, 4);

    // ======================================================================
    // 1. ExtraJoint + PulleyJoint::total_length -- exact Pythagorean closed
    //    form (no division, so no CORDIC / rounding concerns at all).
    // ======================================================================
    let bodies = [
        RigidBody::new(Vec3Fix::from_int(3, 4, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(26, 8, 0), Fix128::ONE),
    ];
    let pulley = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,               // ground_anchor_a at the origin
        Vec3Fix::from_int(20, 0, 0), // ground_anchor_b
        q(3, 2),                     // ratio
    );
    let total = pulley.total_length(&bodies);
    // len_a = |(3,4,0) - (0,0,0)| = sqrt(9+16) = 5
    // len_b = |(26,8,0) - (20,0,0)| = |(6,8,0)| = sqrt(36+64) = 10
    // total = len_a + ratio*len_b = 5 + 1.5*10 = 20
    println!(
        "[joint_extra] PulleyJoint::total_length (3-4-5 / 6-8-10, ratio 3/2) = {} (expect 20)",
        total.to_f64()
    );
    assert_eq!(total, Fix128::from_int(20), "total_length must be exact");

    // `solve_pulley_to_length` pulls an over-long rope back toward its rest
    // length: the rope above measures 20, the rest length is 18, so the
    // measured total must shrink (a slack rope would not move at all).
    let mut taut = bodies;
    solve_pulley_to_length(&pulley, &mut taut, Fix128::from_int(18), dt);
    let after = pulley.total_length(&taut);
    println!(
        "[joint_extra] solve_pulley_to_length (rest 18) total {} -> {}",
        total.to_f64(),
        after.to_f64()
    );
    assert!(after < total, "an over-long rope must be shortened");

    // Boundary: ratio = 0 drops the second rope out of the sum entirely.
    let zero_ratio_pulley = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(20, 0, 0),
        Fix128::ZERO,
    );
    let total_zero_ratio = zero_ratio_pulley.total_length(&bodies);
    println!(
        "[joint_extra] PulleyJoint::total_length (ratio=0) = {} (expect 5, len_b dropped)",
        total_zero_ratio.to_f64()
    );
    assert_eq!(total_zero_ratio, Fix128::from_int(5));

    // Boundary: a zero-length rope (anchor already at the ground anchor).
    let coincident_bodies = [
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(26, 8, 0), Fix128::ONE),
    ];
    let zero_len_pulley = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO, // coincides with body A -> len_a = 0
        Vec3Fix::from_int(20, 0, 0),
        q(3, 2),
    );
    let total_zero_len = zero_len_pulley.total_length(&coincident_bodies);
    println!(
        "[joint_extra] PulleyJoint::total_length (len_a=0) = {} (expect 15, len_a dropped)",
        total_zero_len.to_f64()
    );
    assert_eq!(total_zero_len, Fix128::from_int(15));

    // ExtraJoint enum: both constructed variants round-trip through a match.
    let catalogue = [
        ExtraJoint::Pulley(pulley),
        ExtraJoint::Weld(WeldJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        )),
    ];
    for j in &catalogue {
        let kind = match j {
            ExtraJoint::Pulley(_) => "Pulley",
            ExtraJoint::Gear(_) => "Gear",
            ExtraJoint::Weld(_) => "Weld",
            ExtraJoint::RackAndPinion(_) => "RackAndPinion",
            ExtraJoint::Mouse(_) => "Mouse",
        };
        println!("[joint_extra] ExtraJoint variant: {kind}");
    }

    // ======================================================================
    // 2. MouseJoint::set_target + solve_extra_joints -- exact linear closed
    //    form from rest.
    // ======================================================================
    let mut mj = MouseJoint::new(
        1,             // the dynamic body in pair() -- body 0 is static (inv_mass=0)
        Vec3Fix::ZERO, // initial target == initial position (no error yet)
        Fix128::from_int(100),
        Fix128::ONE,
        Fix128::ZERO,
    );
    mj.set_target(Vec3Fix::from_int(8, 0, 0));
    assert_eq!(mj.target_position, Vec3Fix::from_int(8, 0, 0));

    let mouse_joints = [ExtraJoint::Mouse(mj)];
    let mut bodies = pair();
    solve_extra_joints(&mut bodies, &mouse_joints, dt);
    // distance=8, direction=(1,0,0), spring_force = stiffness*distance = 8,
    // damping_force = 0 (velocity is 0), clamped_force = 8 (< max_force=100),
    // impulse = direction*(8*dt) = (2,0,0), position += impulse*dt*inv_mass(1)
    // = (0.5,0,0).
    println!(
        "[joint_extra] MouseJoint::set_target(8,0,0) + solve_extra_joints -> x={} (expect 0.5)",
        bodies[1].position.x.to_f64()
    );
    assert_eq!(
        bodies[1].position,
        Vec3Fix::new(Fix128::from_ratio(1, 2), Fix128::ZERO, Fix128::ZERO)
    );

    // Boundary: set_target to the body's own position -> zero distance ->
    // solve_mouse's early return, no correction at all.
    let mut mj_zero = MouseJoint::new(
        0,
        Vec3Fix::from_int(5, 0, 0),
        Fix128::from_int(100),
        Fix128::ONE,
        Fix128::ZERO,
    );
    mj_zero.set_target(Vec3Fix::ZERO); // same as the body's own position below
    let zero_target_joints = [ExtraJoint::Mouse(mj_zero)];
    let mut zero_bodies = vec![RigidBody::new(Vec3Fix::ZERO, Fix128::ONE)];
    solve_extra_joints(&mut zero_bodies, &zero_target_joints, dt);
    println!(
        "[joint_extra] MouseJoint::set_target(0,0,0) at body position (0,0,0) -> x={} (expect 0, no correction)",
        zero_bodies[0].position.x.to_f64()
    );
    assert_eq!(zero_bodies[0].position, Vec3Fix::ZERO);

    // ======================================================================
    // 3. WeldJoint::with_break_force -- strict break boundary + the actual
    //    physics correction when the weld holds.
    // ======================================================================
    for (label, threshold, should_break) in [
        ("below (3 < 4)", 3i64, true),
        ("boundary (4 == 4)", 4, false),
        ("above (5 > 4)", 5, false),
    ] {
        let mut bodies = pair();
        bodies[1].position = Vec3Fix::from_int(4, 0, 0); // compute_force = 4 exactly
        let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_break_force(Fix128::from_int(threshold));
        let before = bodies[1].position;
        let joints = [ExtraJoint::Weld(wj)];
        solve_extra_joints(&mut bodies, &joints, dt);
        let moved = bodies[1].position != before;
        println!(
            "[joint_extra] WeldJoint::with_break_force {label}: broken={} moved={moved} (expect broken={should_break}, moved={})",
            wj.is_broken(&pair_with(before)),
            !should_break
        );
        assert_eq!(wj.is_broken(&pair_with(before)), should_break);
        // A broken weld must not move body B; a holding weld (A static,
        // inv_mass_a=0) must land B exactly on A's anchor in one solve.
        if should_break {
            assert_eq!(bodies[1].position, before, "broken weld must not move B");
        } else {
            assert_eq!(
                bodies[1].position,
                Vec3Fix::ZERO,
                "holding weld must land B exactly on A's anchor"
            );
        }
    }

    // Boundary: zero break_force threshold -- any nonzero separation breaks
    // immediately, but a joint that is already satisfied (separation == 0)
    // does not (the comparison is strict).
    let mut touching = pair();
    let wj_zero_touching = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_force(Fix128::ZERO);
    assert!(!wj_zero_touching.is_broken(&touching));
    let joints = [ExtraJoint::Weld(wj_zero_touching)];
    solve_extra_joints(&mut touching, &joints, dt);
    println!(
        "[joint_extra] WeldJoint::with_break_force(0), separation=0 -> broken={} (expect false)",
        wj_zero_touching.is_broken(&touching)
    );

    let mut separated = pair();
    separated[1].position = Vec3Fix::from_int(4, 0, 0);
    let wj_zero_separated = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_force(Fix128::ZERO);
    println!(
        "[joint_extra] WeldJoint::with_break_force(0), separation=4 -> broken={} (expect true)",
        wj_zero_separated.is_broken(&separated)
    );
    assert!(wj_zero_separated.is_broken(&separated));

    // ======================================================================
    // 4. WeldJoint::with_break_torque -- sin(pi/6) = 1/2 closed form (CORDIC
    //    tolerance), isolated from the linear path by coincident positions.
    // ======================================================================
    let angle = Fix128::PI / Fix128::from_int(3); // theta = pi/3 about +Z
    for (label, threshold, should_break) in [
        ("below (1/4 < ~0.5)", q(1, 4), true),
        ("above (3/4 > ~0.5)", q(3, 4), false),
    ] {
        let mut bodies = pair();
        bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, angle);
        let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_break_torque(threshold);
        let torque = wj.compute_torque(&bodies);
        let broken = wj.is_broken(&bodies);
        println!(
            "[joint_extra] WeldJoint::with_break_torque {label}: compute_torque={} (expect ~0.5) broken={broken} (expect {should_break})",
            torque.to_f64()
        );
        assert!(
            near(torque, q(1, 2), cordic_tol()),
            "compute_torque should match sin(pi/6)=1/2 within CORDIC tolerance, got {}",
            torque.to_f64()
        );
        assert_eq!(broken, should_break);

        let before_rot = bodies[1].rotation;
        let joints = [ExtraJoint::Weld(wj)];
        solve_extra_joints(&mut bodies, &joints, dt);
        let rotation_changed = bodies[1].rotation != before_rot;
        println!(
            "[joint_extra]   solve_extra_joints: rotation_changed={rotation_changed} (expect {})",
            !should_break
        );
        assert_eq!(
            rotation_changed, !should_break,
            "a broken weld must skip the angular correction; a holding one must apply it"
        );
    }

    // Boundary: zero break_torque threshold.
    let identity_pair = pair(); // rotation stays IDENTITY -> torque = 0
    let wj_zero_torque_identity =
        WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_break_torque(Fix128::ZERO);
    println!(
        "[joint_extra] WeldJoint::with_break_torque(0), torque=0 -> broken={} (expect false)",
        wj_zero_torque_identity.is_broken(&identity_pair)
    );
    assert!(!wj_zero_torque_identity.is_broken(&identity_pair));

    let mut rotated_pair = pair();
    rotated_pair[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, angle);
    let wj_zero_torque_rotated =
        WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_break_torque(Fix128::ZERO);
    println!(
        "[joint_extra] WeldJoint::with_break_torque(0), torque=~0.5 -> broken={} (expect true)",
        wj_zero_torque_rotated.is_broken(&rotated_pair)
    );
    assert!(wj_zero_torque_rotated.is_broken(&rotated_pair));

    println!(
        "[joint_extra] all 6 production entry points (ExtraJoint, set_target, \
         solve_extra_joints, total_length, with_break_torque, with_break_force) \
         verified against hand-derived closed forms"
    );
}

/// Rebuild a static-A / dynamic-B pair with B at the given position, for the
/// `is_broken` boundary re-checks above (the main `bodies` slice has already
/// been mutated by `solve_extra_joints` by the time those checks run).
fn pair_with(b_position: Vec3Fix) -> Vec<RigidBody> {
    let mut bodies = pair();
    bodies[1].position = b_position;
    bodies
}
