//! Substep interpolation output: snapshot capture + alpha-blended render state
//!
//! Builds a `PhysicsWorld`, captures a `WorldSnapshot` before and after a
//! physics step via `InterpolationState::capture_and_push`, then reads back
//! the alpha-blended position/rotation through `interpolate` /
//! `interpolate_all` / `interpolate_position` / `interpolate_rotation`, and
//! exercises the underlying `lerp_fix128` / `lerp_vec3` / `slerp` primitives
//! and the `BodySnapshot::from_body` / `WorldSnapshot::capture` constructors
//! directly. Every printed value is paired with the closed form it must
//! match (gravity is zero and damping is 1, so the linear motion is exact).
//!
//! ```bash
//! cargo run --example substep_interpolation --features std
//! ```

use alice_physics::interpolation::{
    lerp_fix128, lerp_vec3, slerp, BodySnapshot, InterpolationState, WorldSnapshot,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn linear_motion_interpolation() {
    println!("[interpolation] -- capture_and_push over a physics step, interpolate_all --");
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.add_body(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 0, 0), Fix128::ONE)
            .with_velocity(Vec3Fix::from_int(8, 0, 0)),
    );
    world.add_body(RigidBody::new_static(Vec3Fix::from_int(5, 5, 5)));

    let mut interp = InterpolationState::empty();
    println!(
        "[interpolation] empty(): body_count = {} (documented 0)",
        interp.body_count()
    );
    assert_eq!(interp.body_count(), 0);

    // 1st capture_and_push: current <- initial state, prev stays empty.
    interp.capture_and_push(&world);
    println!(
        "[interpolation] 1st capture_and_push: current.len() = {}, prev.is_empty() = {}",
        interp.current.len(),
        interp.prev.is_empty()
    );
    assert_eq!(interp.current.len(), 2);
    assert!(interp.prev.is_empty());

    // Step with dt = 1/4: constant velocity (8,0,0), no gravity, damping 1
    // -> position advances by exactly velocity * dt = (2, 0, 0).
    world.step(Fix128::from_ratio(1, 4));
    interp.capture_and_push(&world);
    println!(
        "[interpolation] 2nd capture_and_push: current[0].position = {} (closed form: 0 + 8*1/4 = 2)",
        interp.current.bodies[0].position
    );
    assert_eq!(
        interp.current.bodies[0].position,
        Vec3Fix::from_int(2, 0, 0)
    );

    let half = Fix128::from_ratio(1, 2);
    let mid = interp.interpolate_all(half);
    println!(
        "[interpolation] interpolate_all(alpha=1/2)[0].0 = {} (closed form midpoint: (0+2)/2 = 1)",
        mid[0].0
    );
    assert_eq!(mid[0].0, Vec3Fix::from_int(1, 0, 0));
    assert_eq!(
        mid[1].0,
        Vec3Fix::from_int(5, 5, 5),
        "static body does not move"
    );

    let at_zero = interp.interpolate_position(0, Fix128::ZERO);
    let at_one = interp.interpolate_position(0, Fix128::ONE);
    println!(
        "[interpolation] interpolate_position(alpha=0) = {} (== prev), interpolate_position(alpha=1) = {} (== current)",
        at_zero, at_one
    );
    assert_eq!(at_zero, interp.prev.bodies[0].position);
    assert_eq!(at_one, interp.current.bodies[0].position);

    // interpolate() combines position + rotation; both snapshots carry the
    // identity rotation here, so the rotation half of the pair is identity.
    let (p, q) = interp.interpolate(0, half);
    println!("[interpolation] interpolate(0, 1/2) = ({p}, {q})");
    assert_eq!(p, mid[0].0);
    assert_eq!(q, QuatFix::IDENTITY);
}

fn manual_snapshot_rotation() {
    println!("[interpolation] -- from_body / WorldSnapshot::capture / interpolate_rotation --");

    // Two one-body worlds so `WorldSnapshot::capture` is exercised directly
    // (not only through `capture_and_push`), with a non-identity rotation on
    // the second so interpolate_rotation has real NLERP work to do.
    let mut world_a = PhysicsWorld::new(PhysicsConfig::default());
    world_a.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(1, 2, 3),
        Fix128::ONE,
    ));
    let mut world_b = PhysicsWorld::new(PhysicsConfig::default());
    world_b.add_body(
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 2, 3), Fix128::ONE).with_rotation(
            QuatFix::new(
                Fix128::ZERO,
                Fix128::from_int(6),
                Fix128::ZERO,
                Fix128::from_int(7),
            ),
        ),
    );

    let snap_a = WorldSnapshot::capture(&world_a);
    let snap_b = WorldSnapshot::capture(&world_b);
    println!(
        "[interpolation] capture: a.bodies[0].rotation = {}, b.bodies[0].rotation = {}",
        snap_a.bodies[0].rotation, snap_b.bodies[0].rotation
    );
    assert_eq!(
        snap_a.bodies[0],
        BodySnapshot::from_body(&world_a.bodies[0])
    );
    assert_eq!(snap_a.bodies[0].rotation, QuatFix::IDENTITY);

    let interp = InterpolationState::new(snap_a, snap_b);
    let half = Fix128::from_ratio(1, 2);
    let q_half = interp.interpolate_rotation(0, half);

    // oracle (module's actual NLERP convention, read from `interpolation.rs`):
    //   dot(a, b) = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w = 1*7 = 7 >= 0, so
    //   there is no shortest-path flip.
    //   raw = (1-1/2)*a + (1/2)*b = (0, 3, 0, 4) exactly (t = 1/2 is dyadic,
    //   every component multiply is an exact right-shift).
    //   |raw| = sqrt(3^2 + 4^2) = sqrt(25) = 5 exactly (Pythagorean triple;
    //   `Fix128::sqrt` on an integer perfect square is exact by
    //   construction of its digit-by-digit algorithm).
    //   normalized = raw * (1/len) = (0, 3/5, 0, 4/5).
    let len = Fix128::from_int(25).sqrt();
    assert_eq!(len, Fix128::from_int(5), "sqrt(25) must be exact");
    let inv_len = Fix128::ONE / len;
    let expected = QuatFix::new(
        Fix128::ZERO,
        Fix128::from_int(3) * inv_len,
        Fix128::ZERO,
        Fix128::from_int(4) * inv_len,
    );
    println!(
        "[interpolation] interpolate_rotation(alpha=1/2) = {q_half} (closed form: {expected})"
    );
    assert_eq!(q_half, expected);
}

fn lerp_primitives() {
    println!("[interpolation] -- lerp_fix128 / lerp_vec3 / slerp primitives --");

    let a = Fix128::from_int(-4);
    let b = Fix128::from_int(12);
    let quarter = Fix128::from_ratio(1, 4);
    let r = lerp_fix128(a, b, quarter);
    println!("[interpolation] lerp_fix128(-4, 12, 1/4) = {r} (closed form: -4 + 16*1/4 = 0)");
    assert_eq!(r, Fix128::ZERO);

    let va = Vec3Fix::from_int(-4, 0, 12);
    let vb = Vec3Fix::from_int(12, 8, -4);
    let rv = lerp_vec3(va, vb, quarter);
    println!(
        "[interpolation] lerp_vec3((-4,0,12), (12,8,-4), 1/4) = {rv} (componentwise lerp_fix128)"
    );
    assert_eq!(
        rv,
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2), Fix128::from_int(8))
    );

    // slerp of two identical, already-normalized quaternions must return
    // that rotation exactly: lerp(q, q, t) = q for any t, and |q| is
    // already 1, so normalize is a no-op. This is the "angle between
    // identical rotations" edge case that a true acos-based SLERP would
    // need to special-case; NLERP does not need to.
    let q = QuatFix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    let same = slerp(q, q, Fix128::from_ratio(3, 7));
    println!("[interpolation] slerp(q, q, 3/7) = {same} (must equal q exactly)");
    assert_eq!(same, q);
}

fn main() {
    linear_motion_interpolation();
    manual_snapshot_rotation();
    lerp_primitives();
    println!("[interpolation] all closed-form checks passed");
}
