//! Analytic-solution oracle tests (1.2.0).
//!
//! The golden-hash suite (`determinism_golden.rs`) only detects *change*: a
//! simulation that was wrong from the start gets pinned as-is. The 2026-09-15
//! external review found four bugs (substep-dependent damping, XPBD lambda
//! overwrite, dead BVH broad phase, 64-iteration sqrt) of which three were
//! fixed without a single existing test failing — 1456 tests had never
//! observed the behaviour in the most ordinary use of the engine.
//!
//! Every test here compares the solver against a closed-form solution of
//! classical mechanics, on the **default configuration** wherever the
//! quantity is defined for it, and additionally asserts that *precision*
//! parameters (`substeps`, `iterations`) do not change the physical result.
//!
//! Rule (karikari-review §4 Path P / deterministic-physics-lockstep-discipline):
//! a simulation crate ships an analytic oracle for every physical law it
//! claims, and the oracle runs on `Config::default()`.

#![cfg(feature = "std")]
// The f64 libm values here are the *oracle* (closed-form references), not
// simulation state, so the det_math determinism gate does not apply.
#![allow(clippy::disallowed_methods)]

use alice_physics::joint::{solve_joints, HingeJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{BodyType, DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody};

const DT60: f64 = 1.0 / 60.0;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn cfg(gravity_y: i64, damping: Fix128, substeps: usize, iterations: usize) -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::from_int(0, gravity_y, 0),
        damping,
        substeps,
        iterations,
        ..PhysicsConfig::default()
    }
}

fn run(world: &mut PhysicsWorld, frames: usize, dt: Fix128) {
    for _ in 0..frames {
        world.step(dt);
    }
}

fn rod(a: usize, b: usize, length: Fix128, compliance: Fix128) -> DistanceConstraint {
    DistanceConstraint {
        body_a: a,
        body_b: b,
        local_anchor_a: Vec3Fix::ZERO,
        local_anchor_b: Vec3Fix::ZERO,
        target_distance: length,
        compliance,
        cached_lambda: Fix128::ZERO,
    }
}

/// Period from the mean spacing of same-direction zero crossings of `samples`.
fn period_from_zero_crossings(samples: &[f64], dt: f64) -> f64 {
    let mut crossings = Vec::new();
    for i in 1..samples.len() {
        if samples[i - 1] < 0.0 && samples[i] >= 0.0 {
            // linear interpolation of the crossing time
            let frac = -samples[i - 1] / (samples[i] - samples[i - 1]);
            crossings.push((i as f64 - 1.0 + frac) * dt);
        }
    }
    assert!(
        crossings.len() >= 3,
        "need >= 3 upward zero crossings, got {}",
        crossings.len()
    );
    (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64
}

// ---------------------------------------------------------------------------
// 1. Free fall: y(t) = -g t^2 / 2, independent of substeps
// ---------------------------------------------------------------------------

#[test]
fn free_fall_matches_analytic_and_is_substep_independent() {
    let mut ys = Vec::new();
    for substeps in [1usize, 2, 4, 8, 16] {
        let mut world = PhysicsWorld::new(cfg(-10, Fix128::ONE, substeps, 4));
        let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        run(&mut world, 60, r(1, 60));
        ys.push(world.bodies[b].position.y.to_f64());
    }
    // symplectic Euler: y_N = -g dt^2 N(N+1)/2 → -5 - g dt / 2 = -5.083 at substeps 1,
    // converging to -5 as the substep shrinks
    for y in &ys {
        assert!(
            (y + 5.0).abs() < 0.1,
            "free fall y = {y}, expected -5.0 ± 0.1 ({ys:?})"
        );
    }
    let spread =
        ys.iter().cloned().fold(f64::MIN, f64::max) - ys.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        spread < 0.1,
        "substeps changed the result by {spread} ({ys:?})"
    );
}

// ---------------------------------------------------------------------------
// 2. Default configuration actually falls (pre-1.2.0: terminal velocity 2 m/s)
// ---------------------------------------------------------------------------

#[test]
fn default_config_free_fall_reaches_analytic_within_frame_damping() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    run(&mut world, 60, r(1, 60));
    let y = world.bodies[b].position.y.to_f64();
    // Discrete closed form of the 1.2.0 scheme with s substeps and frame damping d:
    //   y_{n+1} = y_n + v_n dt + g dt^2 (s+1)/(2s)   (symplectic Euler inside the frame)
    //   v_{n+1} = (v_n + g dt) d                       (damping once per frame)
    // → y_60 = -4.1406 (undamped -5.083); pre-1.2.0 gave -1.641.
    let (g, d, dt, s) = (10.0, 0.99, DT60, 8.0);
    let mut y_ref = 0.0;
    let mut v = 0.0;
    for _ in 0..60 {
        y_ref -= v * dt + g * dt * dt * (s + 1.0) / (2.0 * s);
        v = (v + g * dt) * d;
    }
    assert!(
        (y - y_ref).abs() < 1e-6,
        "default config y = {y}, closed-form {y_ref} (pre-1.2.0 gave -1.64)"
    );
    assert!(y < -3.5, "default config fell only to {y}");
}

// ---------------------------------------------------------------------------
// 3. Projectile: x = vx t (exact), y = vy t - g t^2 / 2
// ---------------------------------------------------------------------------

#[test]
fn projectile_follows_parabola() {
    let mut world = PhysicsWorld::new(cfg(-10, Fix128::ONE, 8, 4));
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    body.velocity = Vec3Fix::from_int(3, 4, 0);
    let b = world.add_body(body);
    run(&mut world, 48, r(1, 60)); // t = 0.8 s → x = 2.4, y = 3.2 - 3.2 = 0
    let p = world.bodies[b].position;
    assert!((p.x.to_f64() - 2.4).abs() < 1e-9, "x = {}", p.x.to_f64());
    assert!((p.y.to_f64() - 0.0).abs() < 0.05, "y = {}", p.y.to_f64());
    assert_eq!(p.z, Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// 4. Frame damping: v_{n+1} = (v_n + g dt) d → v_inf = g dt d / (1 - d)
// ---------------------------------------------------------------------------

#[test]
fn terminal_velocity_under_frame_damping_matches_closed_form() {
    for substeps in [1usize, 8] {
        let mut world = PhysicsWorld::new(cfg(-10, r(99, 100), substeps, 4));
        let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        run(&mut world, 1200, r(1, 60));
        let v = -world.bodies[b].velocity.y.to_f64();
        let v_inf = 10.0 * DT60 * 0.99 / 0.01; // 16.5 m/s
        assert!(
            (v - v_inf).abs() < 0.05,
            "substeps {substeps}: terminal v = {v}, expected {v_inf} (pre-1.2.0: {})",
            10.0 * DT60 / substeps as f64 * 0.99 / 0.01
        );
    }
}

// ---------------------------------------------------------------------------
// 5. XPBD static extension mg/k, independent of iterations
// ---------------------------------------------------------------------------

#[test]
fn compliant_constraint_static_extension_is_iteration_independent() {
    let mut exts = Vec::new();
    for iterations in [1usize, 2, 4, 8, 16] {
        let mut world = PhysicsWorld::new(cfg(-10, Fix128::ONE, 4, iterations));
        let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let mut hanging = RigidBody::new_dynamic(Vec3Fix::from_int(0, -1, 0), Fix128::ONE);
        hanging.linear_damping = r(9, 10); // settle the transient
        let b = world.add_body(hanging);
        world.add_distance_constraint(rod(a, b, Fix128::ONE, r(1, 100))); // k = 100 N/m
        run(&mut world, 600, r(1, 60));
        exts.push(-world.bodies[b].position.y.to_f64() - 1.0);
    }
    for e in &exts {
        assert!(
            (e - 0.1).abs() < 0.01,
            "extension {e}, expected mg/k = 0.1 ({exts:?})"
        );
    }
}

// ---------------------------------------------------------------------------
// 6. Spring natural frequency: T = 2π sqrt(m / k)
// ---------------------------------------------------------------------------

#[test]
fn compliant_constraint_oscillates_at_natural_period() {
    let mut world = PhysicsWorld::new(cfg(0, Fix128::ONE, 8, 1));
    let a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    // rest length 1, k = 100 N/m, m = 1 → ω = 10, T = 0.6283 s; start stretched by 0.1
    let b = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(r(11, 10), Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    ));
    world.add_distance_constraint(rod(a, b, Fix128::ONE, r(1, 100)));
    let dt = r(1, 240);
    let mut xs = Vec::new();
    for _ in 0..(240 * 4) {
        world.step(dt);
        xs.push(world.bodies[b].position.x.to_f64() - 1.0);
    }
    let period = period_from_zero_crossings(&xs, 1.0 / 240.0);
    let expected = 2.0 * core::f64::consts::PI / 10.0;
    assert!(
        (period - expected).abs() / expected < 0.03,
        "spring period {period}, expected {expected}"
    );
}

// ---------------------------------------------------------------------------
// 7. Simple pendulum, small angle: T = 2π sqrt(L / g)
// ---------------------------------------------------------------------------

#[test]
fn rigid_pendulum_small_angle_period() {
    let mut world = PhysicsWorld::new(cfg(-10, Fix128::ONE, 8, 4));
    let pivot = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    // L = 1, θ0 = 0.05 rad → x0 = sin θ0, y0 = -cos θ0
    let theta0 = 0.05f64;
    let bob = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(
            Fix128::from_f64(theta0.sin()),
            Fix128::from_f64(-theta0.cos()),
            Fix128::ZERO,
        ),
        Fix128::ONE,
    ));
    world.add_distance_constraint(rod(pivot, bob, Fix128::ONE, Fix128::ZERO));
    let dt = r(1, 240);
    let mut xs = Vec::new();
    for _ in 0..(240 * 8) {
        world.step(dt);
        xs.push(world.bodies[bob].position.x.to_f64());
    }
    let period = period_from_zero_crossings(&xs, 1.0 / 240.0);
    // T = 2π sqrt(L/g) (1 + θ0²/16) = 1.9869 * 1.00016
    let expected =
        2.0 * core::f64::consts::PI * (1.0f64 / 10.0).sqrt() * (1.0 + theta0 * theta0 / 16.0);
    assert!(
        (period - expected).abs() / expected < 0.02,
        "pendulum period {period}, expected {expected}"
    );
    // the rod stays rigid: |bob| = L within the solver tolerance
    let len = world.bodies[bob].position.length().to_f64();
    assert!((len - 1.0).abs() < 1e-3, "rod length drifted to {len}");
}

// ---------------------------------------------------------------------------
// 8. Head-on collision: total momentum is conserved
// ---------------------------------------------------------------------------

#[test]
fn head_on_collision_conserves_momentum() {
    let mut world = PhysicsWorld::new(cfg(0, Fix128::ONE, 8, 4));
    let mut a = RigidBody::new_dynamic(Vec3Fix::from_int(-3, 0, 0), Fix128::ONE);
    a.velocity = Vec3Fix::from_int(4, 0, 0);
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(3, 0, 0), Fix128::from_int(2));
    b.velocity = Vec3Fix::from_int(-1, 0, 0);
    let ia = world.add_body_with_radius(a, Fix128::ONE);
    let ib = world.add_body_with_radius(b, Fix128::ONE);
    let p_before = 4.0 - 2.0; // m_a v_a + m_b v_b = 1*4 + 2*(-1)
    run(&mut world, 120, r(1, 60)); // 2 s: they meet at t ≈ 0.8 s
    let va = world.bodies[ia].velocity.x.to_f64();
    let vb = world.bodies[ib].velocity.x.to_f64();
    let p_after = 1.0 * va + 2.0 * vb;
    assert!(
        va < vb,
        "bodies did not collide / separate: va = {va}, vb = {vb}"
    );
    assert!(
        (p_after - p_before).abs() < 0.05 * p_before.abs().max(1.0),
        "momentum {p_before} → {p_after} (va = {va}, vb = {vb})"
    );
    // A collision must not create energy: separation speed <= closing speed
    // (coefficient of restitution <= 1). Closing speed is 5 m/s.
    let closing = 4.0 - (-1.0);
    let separating = vb - va;
    assert!(
        separating <= closing * 1.01,
        "collision gained energy: separation {separating} m/s > closing {closing} m/s (va = {va}, vb = {vb})"
    );
    let ke = 0.5 * va * va + 0.5 * 2.0 * vb * vb;
    let ke_before = 0.5 * 16.0 + 0.5 * 2.0 * 1.0;
    assert!(ke <= ke_before * 1.01, "kinetic energy {ke_before} → {ke}");
}

// ---------------------------------------------------------------------------
// 9. Kinematic body reaches its target exactly (bit-exact linear motion)
// ---------------------------------------------------------------------------

#[test]
fn kinematic_body_reaches_target_exactly() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let mut kin = RigidBody::new_static(Vec3Fix::ZERO);
    kin.body_type = BodyType::Kinematic;
    let k = world.add_body(kin);
    let dt = r(1, 60);
    for i in 1..=60i64 {
        let target = Vec3Fix::new(
            Fix128::from_ratio(i, 60) * Fix128::from_int(3),
            Fix128::ZERO,
            Fix128::ZERO,
        );
        world.bodies[k].set_kinematic_target(target, QuatFix::IDENTITY);
        world.step(dt);
        assert_eq!(world.bodies[k].position, target, "frame {i}");
    }
    // gravity / damping must not touch a kinematic body
    assert_eq!(world.bodies[k].position, Vec3Fix::from_int(3, 0, 0));
}

// ---------------------------------------------------------------------------
// 10. Torque-free rotation: angle = ω t, independent of substeps
// ---------------------------------------------------------------------------

#[test]
fn torque_free_rotation_angle_equals_omega_t() {
    for substeps in [1usize, 4, 16] {
        let mut world = PhysicsWorld::new(cfg(0, Fix128::ONE, substeps, 4));
        let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        body.angular_velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, r(1, 2)); // 0.5 rad/s
        let b = world.add_body(body);
        run(&mut world, 120, r(1, 60)); // 2 s → 1 rad about z
        let q = world.bodies[b].rotation;
        let sin_half = (q.x * q.x + q.y * q.y + q.z * q.z).sqrt().to_f64();
        let angle = 2.0 * sin_half.atan2(q.w.to_f64());
        assert!(
            (angle - 1.0).abs() < 1e-3,
            "substeps {substeps}: rotation angle {angle}, expected 1.0"
        );
        assert!(q.z.to_f64() > 0.0, "rotation axis flipped");
    }
}

// ---------------------------------------------------------------------------
// 11. Resting contact: a body at rest on a static body stays put and falls
//     asleep (before 1.2.0 every detected contact reset the sleep timer, so a
//     resting stack could never sleep)
// ---------------------------------------------------------------------------

#[test]
fn resting_body_on_static_support_stays_put_and_sleeps() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let ground = world.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
    // resting exactly on top: centre distance = r + r = 2
    let ball = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 2, 0), Fix128::ONE),
        Fix128::ONE,
    );
    run(&mut world, 300, r(1, 60)); // 5 s
    let y = world.bodies[ball].position.y.to_f64();
    assert!((y - 2.0).abs() < 0.05, "resting ball drifted to y = {y}");
    assert!(
        world.bodies[ball].velocity.length().to_f64() < 0.05,
        "resting ball still moving: {:?}",
        world.bodies[ball].velocity
    );
    assert!(world.is_sleeping(ball), "resting ball never fell asleep");
    assert_eq!(world.bodies[ground].position, Vec3Fix::ZERO);
}

// ---------------------------------------------------------------------------
// Joints: angular XPBD (Macklin et al. 2020 §3.3.2) — exact split by
// `w_i = n · I_i⁻¹ n`, signed twist angle
// ---------------------------------------------------------------------------

fn axis_angle_of(q: QuatFix) -> (Vec3Fix, f64) {
    let (axis, s) = Vec3Fix::new(q.x, q.y, q.z).normalize_with_length();
    let angle = 2.0 * s.to_f64().atan2(q.w.to_f64());
    (axis, angle)
}

/// A rigid hinge (compliance 0) whose axes are misaligned by θ removes the
/// whole error in one solve: the relative rotation about the correction
/// axis changes by exactly `(w_a + w_b) λ = θ`. With a static A only B
/// moves; with two dynamic bodies the rotations split as `w_a : w_b`
/// (`w_i = n · I_i⁻¹ n`, here 1 : 3), which is Macklin 2020 eq. 5–6. The
/// first-order update `(n θ/2, 1)` is exact in direction and `O(θ³)` in
/// magnitude, hence the tolerances. Before 1.2.0 both bodies received the
/// full λ with `w = |diag(I⁻¹)|`, so a unit-inertia hinge removed `1/√3`
/// of θ per step and static/dynamic pairs converged only over iterations.
#[test]
fn hinge_alignment_removes_the_full_error_in_one_solve_split_by_inertia() {
    let theta = 0.3f64;
    let tilt = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_f64(theta));
    // static A (axis x), dynamic B with axis x rotated by θ about z
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
    ];
    bodies[1].rotation = tilt;
    let hinge = Joint::Hinge(HingeJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_X,
    ));
    solve_joints(&[hinge], &mut bodies, r(1, 60));
    let axis_b = bodies[1].rotation.rotate_vec(Vec3Fix::UNIT_X);
    let residual = axis_b.cross(Vec3Fix::UNIT_X).length().to_f64();
    assert!(
        residual < theta.powi(3) / 8.0 + 1e-9,
        "static/dynamic: axis error {residual} left after one rigid solve (θ = {theta})"
    );

    // two dynamic bodies, inverse inertia about z 1 : 3 → A rotates θ/4, B by −3θ/4
    let mut bodies = vec![
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
    ];
    bodies[0].inv_inertia = Vec3Fix::from_int(1, 1, 1);
    bodies[1].inv_inertia = Vec3Fix::from_int(3, 3, 3);
    bodies[1].rotation = tilt;
    solve_joints(&[hinge], &mut bodies, r(1, 60));
    let (ax_a, ang_a) = axis_angle_of(bodies[0].rotation);
    assert!(
        (ang_a - theta / 4.0).abs() < 1e-3 && (ax_a.z.to_f64() - 1.0).abs() < 1e-9,
        "A rotated {ang_a} about {ax_a:?}, want θ/4 = {} about +z",
        theta / 4.0
    );
    let (ax_b, ang_b) = axis_angle_of(bodies[1].rotation);
    // B started at +θ and rotates by −3θ/4 → +θ/4 about z
    assert!(
        (ang_b - theta / 4.0).abs() < 1e-3 && (ax_b.z.to_f64() - 1.0).abs() < 1e-9,
        "B at {ang_b} about {ax_b:?}, want θ/4"
    );
    let axis_a = bodies[0].rotation.rotate_vec(Vec3Fix::UNIT_X);
    let axis_b = bodies[1].rotation.rotate_vec(Vec3Fix::UNIT_X);
    assert!(
        axis_a.cross(axis_b).length().to_f64() < 1e-3,
        "axes aligned"
    );
}

/// Hinge angle limits act on the *signed* relative angle: a B rotated by
/// −0.8 rad about the hinge axis with `angle_min = −0.5` is pushed *up* to
/// −0.5, and one rotated by +0.8 with `angle_max = 0.5` is pushed *down* to
/// 0.5; the limit is reached in one rigid solve. Before 1.2.0 the twist
/// angle was measured unsigned (`2·atan2(|proj|, w) ≥ 0`), so −0.8 read as
/// +0.8, hit the *max* limit and was pushed further negative.
#[test]
fn hinge_limits_are_signed_and_reached_in_one_solve() {
    for &(start, want) in &[(-0.8f64, -0.5f64), (0.8, 0.5), (-0.3, -0.3), (0.3, 0.3)] {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        ];
        bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::from_f64(start));
        let mut h = HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_X,
        );
        h.angle_min = Some(r(-1, 2));
        h.angle_max = Some(r(1, 2));
        solve_joints(&[Joint::Hinge(h)], &mut bodies, r(1, 60));
        let (axis, ang) = axis_angle_of(bodies[1].rotation);
        let signed = ang * axis.x.to_f64().signum();
        assert!(
            (signed - want).abs() < 2e-3,
            "start {start}: angle after solve {signed}, want {want}"
        );
    }
}
