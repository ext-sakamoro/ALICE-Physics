//! Joint motors driven through the world: `PhysicsWorld::add_joint_motor` /
//! `add_joint_motor_3d`, applied by `PhysicsWorld::step` every substep.
//!
//! Production entry point: `PhysicsWorld::step` (and `step_parallel` under the
//! `parallel` feature). Every expected value below is computed here, in `f64`,
//! from the closed form of the scheme, never by calling the motor code.
//!
//! # Scene
//!
//! Body 0 is static at `(0, 10, 0)`; body 1 is dynamic at the origin with mass
//! 1 and isotropic inverse inertia 1 (`I = 1`, so no gyroscopic coupling). A
//! hinge (axis `z`) or a ball joint holds body 1 by an anchor at its own
//! centre, so the joint's positional correction acts through the centre and
//! never changes the angular velocity; gravity (default config) does not
//! torque body 1 either. The joint's twist about `z` is then driven only by
//! the motor and by the per-frame damping `d = config.damping *
//! angular_damping`.
//!
//! # Closed forms
//!
//! With `n` substeps of length `h = dt / n`, a motor torque `τ` changes the
//! angular velocity by `h τ / I` per substep (`apply_motors` gives the impulse
//! `τ h`), and `step` multiplies it by `d` once per frame.
//!
//! * **Velocity motor** (`τ = kp (ω_t - ω)`, no cap reached): per substep the
//!   error `ω - ω_t` shrinks by `1 - a`, `a = h kp / I`, so per frame
//!   `ω' = d (ω_t + r (ω - ω_t))` with `r = (1 - a)^n`. Hence
//!   `ω_k = ω* + (d r)^k (ω_0 - ω*)` with the fixed point
//!   `ω* = d (1 - r) ω_t / (1 - d r)`; with `d = 1` this is the first-order
//!   response `ω_t (1 - r^k)`, whose continuous limit is
//!   `ω_t (1 - exp(-kp t / I))`.
//! * **Torque cap** (`kp |ω_t - ω| > τ_max`): the acceleration is
//!   `τ_max / I`, so per frame `ω' = d (ω + τ_max dt / I)` and
//!   `ω_k = c (1 - d^k)`, `c = d τ_max dt / (I (1 - d))`.
//! * **Position PD on a hinge** (`τ = kp (θ_t - θ) - kd ω`, `d = 1`): the
//!   substep is semi-implicit Euler, `ω' = ω + h τ / I`, `θ' = θ + h ω'`, a
//!   linear map whose `k`-th power is the discrete closed form. Its
//!   continuous limit for `kd = 2 sqrt(kp I)` (critical damping, `w = sqrt(kp
//!   / I)`) is `θ(t) = θ_t (1 - (1 + w t) e^{-w t})`.
//! * **`PdController3D` rotation target** (ball joint, rotation about one fixed
//!   unit axis): the error quaternion's vector part is `sin(φ/2)` along the
//!   axis for a remaining angle `φ`, so `τ = 2 kp sin(φ/2) - kd ω` about it;
//!   the substep map is the same semi-implicit Euler with that torque. Its
//!   linearisation is the critically damped decay above.
//!
//! The angular velocity after a substep is derived from the rotation change by
//! the exact rotation logarithm (`update_velocities`), so a turn of `h ω` gives
//! back `ω` and the substep map carries no shrink of the rate. Each test checks
//! two things: the substep map written out here in `f64` (`reference`, tight
//! tolerance), and the continuous closed form (loose tolerance covering the
//! gap between the semi-implicit substep map and the closed form, below `2e-4`
//! for every scene here). A rate derived from the rotation chord instead,
//! `2 sin(ω h / 2) / h`, falls short of `reference` by `(ω h)^2 / 24` per
//! substep and fails the tight tolerance.

// The expected values are closed forms evaluated in f64 here, outside the
// deterministic path, so the platform libm is fine for them.
#![allow(clippy::disallowed_methods)]

use alice_physics::joint::{BallJoint, HingeJoint, Joint};
use alice_physics::motor::{MotorMode, PdController};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, QuatFix, RigidBody, Vec3Fix};

const DT: (i64, i64) = (1, 60);

fn dt() -> Fix128 {
    Fix128::from_ratio(DT.0, DT.1)
}

fn dt_f() -> f64 {
    DT.0 as f64 / DT.1 as f64
}

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

/// The scene described in the module doc; returns the world and the joint index.
fn world_with(config: PhysicsConfig, hinge: bool) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(config);
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 10, 0)));
    let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = Vec3Fix::from_int(1, 1, 1);
    w.add_body(b);
    let joint = if hinge {
        Joint::Hinge(HingeJoint::new(
            0,
            1,
            Vec3Fix::from_int(0, -10, 0),
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 0, 1),
            Vec3Fix::from_int(0, 0, 1),
        ))
    } else {
        Joint::Ball(BallJoint::new(
            0,
            1,
            Vec3Fix::from_int(0, -10, 0),
            Vec3Fix::ZERO,
        ))
    };
    let j = w.add_joint(joint);
    (w, j)
}

fn hinge_world() -> (PhysicsWorld, usize) {
    world_with(PhysicsConfig::default(), true)
}

fn frame_damping(config: &PhysicsConfig) -> f64 {
    // angular_damping of a new body is 1
    config.damping.to_f64()
}

fn omega_z(w: &PhysicsWorld) -> f64 {
    w.bodies[1].angular_velocity.z.to_f64()
}

/// Twist of body 1 about `z` (body 0 is at the identity).
fn angle_z(w: &PhysicsWorld) -> f64 {
    let q = w.bodies[1].rotation;
    2.0 * q.z.to_f64().atan2(q.w.to_f64())
}

fn velocity_pd(kp: f64, max: f64, target: f64) -> PdController {
    let mut pd = PdController::new(f(kp), Fix128::ZERO, f(max));
    pd.set_velocity_target(f(target));
    pd
}

/// Substep map of a velocity motor (`τ = clamp(kp (ω_t - ω), ±max)`), frame
/// damping `d`, `n` substeps: `ω` after each of `frames` frames.
fn velocity_reference(
    kp: f64,
    max: f64,
    d: f64,
    n: usize,
    target: f64,
    w0: f64,
    frames: usize,
) -> Vec<f64> {
    let h = dt_f() / n as f64;
    let mut om = w0;
    let mut out = Vec::new();
    for _ in 0..frames {
        for _ in 0..n {
            om += h * (kp * (target - om)).clamp(-max, max);
        }
        om *= d;
        out.push(om);
    }
    out
}

/// Velocity-motor closed form: `ω_k` from `ω_0` after `k` frames.
fn velocity_closed_form(kp: f64, i: f64, d: f64, n: usize, target: f64, w0: f64, k: i32) -> f64 {
    let h = dt_f() / n as f64;
    let r = (1.0 - h * kp / i).powi(n as i32);
    let fixed = d * (1.0 - r) * target / (1.0 - d * r);
    fixed + (d * r).powi(k) * (w0 - fixed)
}

/// oracle: velocity motor first-order response, default config (d = 0.99,
/// 8 substeps): ω_k = ω* + (d r)^k (ω_0 - ω*), and the d = 1 response tracks
/// the continuous ω_t (1 - exp(-kp t / I)) to O(h).
#[test]
fn hinge_velocity_motor_follows_first_order_response() {
    let (mut w, j) = hinge_world();
    let cfg = w.config;
    let m = w.add_joint_motor(j, velocity_pd(2.0, 1000.0, 1.0));
    assert_eq!(m, 0);
    let d = frame_damping(&cfg);
    let reference = velocity_reference(2.0, 1000.0, d, cfg.substeps, 1.0, 0.0, 240);
    for k in 1..=240 {
        w.step(dt());
        let want = velocity_closed_form(2.0, 1.0, d, cfg.substeps, 1.0, 0.0, k);
        let got = omega_z(&w);
        let r = reference[k as usize - 1];
        assert!(
            (got - r).abs() < 1e-9,
            "frame {k}: ω = {got}, reference {r}"
        );
        assert!(
            (got - want).abs() < 2e-4,
            "frame {k}: ω = {got}, closed form {want}"
        );
    }
    // body 1 turns about +z (the hinge axis), not about x / y
    let wv = w.bodies[1].angular_velocity;
    assert!(wv.x.to_f64().abs() < 1e-9 && wv.y.to_f64().abs() < 1e-9);

    // no frame damping: the first-order response and its continuous limit
    let cfg1 = PhysicsConfig {
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let (mut w1, j1) = world_with(cfg1, true);
    w1.add_joint_motor(j1, velocity_pd(2.0, 1000.0, 1.0));
    for k in 1..=120 {
        w1.step(dt());
        let discrete = velocity_closed_form(2.0, 1.0, 1.0, cfg1.substeps, 1.0, 0.0, k);
        let continuous = 1.0 - (-2.0 * k as f64 * dt_f()).exp();
        let got = omega_z(&w1);
        assert!(
            (got - discrete).abs() < 2e-4,
            "frame {k}: {got} vs {discrete}"
        );
        assert!(
            (got - continuous).abs() < 2e-3,
            "frame {k}: {got} vs {continuous}"
        );
    }
}

/// oracle: with the cap reached the angular acceleration is τ_max / I:
/// ω_k = c (1 - d^k), c = d τ_max dt / (I (1 - d)).
#[test]
fn hinge_motor_torque_cap_limits_acceleration_to_max_over_inertia() {
    let (mut w, j) = hinge_world();
    let d = frame_damping(&w.config);
    let tau_max = 0.5;
    w.add_joint_motor(j, velocity_pd(1000.0, tau_max, 10.0));
    let c = d * tau_max * dt_f() / (1.0 - d);
    let n = w.config.substeps;
    let reference = velocity_reference(1000.0, tau_max, d, n, 10.0, 0.0, 60);
    for k in 1..=60 {
        w.step(dt());
        let want = c * (1.0 - d.powi(k));
        let got = omega_z(&w);
        let r = reference[k as usize - 1];
        assert!(
            (got - r).abs() < 1e-9,
            "frame {k}: ω = {got}, reference {r}"
        );
        assert!(
            (got - want).abs() < 1e-5,
            "frame {k}: ω = {got}, capped {want}"
        );
    }
    // the first frame's change is (τ_max / I) dt, times d
    let (mut w2, j2) = hinge_world();
    w2.add_joint_motor(j2, velocity_pd(1000.0, tau_max, 10.0));
    w2.step(dt());
    assert!((omega_z(&w2) - d * tau_max * dt_f()).abs() < 1e-9);
}

/// Semi-implicit Euler reference of the hinge PD substep map, `frames` frames.
fn pd_reference(kp: f64, kd: f64, i: f64, n: usize, target: f64, frames: usize) -> Vec<f64> {
    let h = dt_f() / n as f64;
    let (mut th, mut om) = (0.0_f64, 0.0_f64);
    let mut out = Vec::new();
    for _ in 0..frames {
        for _ in 0..n {
            om += h * (kp * (target - th) - kd * om) / i;
            th += h * om;
        }
        out.push(th);
    }
    out
}

/// oracle: hinge position PD, critically damped (kd = 2 sqrt(kp I)), d = 1:
/// the discrete semi-implicit Euler closed form, and the continuous
/// θ_t (1 - (1 + w t) e^{-w t}) to O(h); the angle never overshoots.
#[test]
fn hinge_position_pd_follows_critically_damped_closed_form() {
    let cfg = PhysicsConfig {
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let (mut w, j) = world_with(cfg, true);
    let (kp, kd, target) = (4.0, 4.0, 0.5);
    let mut pd = PdController::new(f(kp), f(kd), f(1000.0));
    pd.set_position_target(f(target));
    w.add_joint_motor(j, pd);
    let reference = pd_reference(kp, kd, 1.0, cfg.substeps, target, 300);
    let wn = (kp / 1.0_f64).sqrt();
    for (k, want) in reference.iter().enumerate() {
        w.step(dt());
        let got = angle_z(&w);
        assert!(
            (got - want).abs() < 1e-9,
            "frame {}: θ = {got}, discrete {want}",
            k + 1
        );
        let t = (k + 1) as f64 * dt_f();
        let cont = target * (1.0 - (1.0 + wn * t) * (-wn * t).exp());
        assert!(
            (got - cont).abs() < 2e-3,
            "frame {}: θ = {got}, continuous {cont}",
            k + 1
        );
        assert!(got <= target + 1e-9, "critically damped PD overshot: {got}");
    }
    assert!((angle_z(&w) - target).abs() < 1e-3);
}

/// Reference of the 3-axis PD about a single axis (remaining angle φ):
/// τ = 2 kp sin(φ/2) - kd ω, semi-implicit Euler, frame damping d.
fn pd3d_reference(kp: f64, kd: f64, d: f64, n: usize, target: f64, frames: usize) -> Vec<f64> {
    let h = dt_f() / n as f64;
    let (mut th, mut om) = (0.0_f64, 0.0_f64);
    let mut out = Vec::new();
    for _ in 0..frames {
        for _ in 0..n {
            let tau = 2.0 * kp * ((target - th) / 2.0).sin() - kd * om;
            om += h * tau;
            th += h * om;
        }
        om *= d;
        out.push(th);
    }
    out
}

/// oracle: PdController3D rotation target on a ball joint (default config,
/// axis (0, 0.6, 0.8)): the remaining angle follows the substep map of
/// τ = 2 kp sin(φ/2) - kd ω, and its linearised critically damped decay.
#[test]
fn ball_rotation_motor_converges_with_analytic_error_decay() {
    let (mut w, j) = world_with(PhysicsConfig::default(), false);
    let cfg = w.config;
    let d = frame_damping(&cfg);
    let (kp, kd, target) = (4.0, 4.0, 0.4);
    let m = w.add_joint_motor_3d(
        j,
        Vec3Fix::new(f(kp), f(kp), f(kp)),
        Vec3Fix::new(f(kd), f(kd), f(kd)),
        f(1000.0),
    );
    let axis = Vec3Fix::new(Fix128::ZERO, f(0.6), f(0.8));
    assert!(w.set_joint_motor_3d_rotation_target(m, QuatFix::from_axis_angle(axis, f(target))));
    let reference = pd3d_reference(kp, kd, d, cfg.substeps, target, 300);
    for (k, want) in reference.iter().enumerate() {
        w.step(dt());
        let q = w.bodies[1].rotation;
        let v = Vec3Fix::new(q.x, q.y, q.z);
        // angle about the axis, and no rotation off it
        let along = v.dot(axis).to_f64();
        let off = (v - axis * v.dot(axis)).length().to_f64();
        let got = 2.0 * along.atan2(q.w.to_f64());
        assert!(
            off < 1e-9,
            "frame {}: rotation left the axis ({off})",
            k + 1
        );
        assert!(
            (got - want).abs() < 1e-9,
            "frame {}: φ = {got}, reference {want}",
            k + 1
        );
    }
    let q = w.bodies[1].rotation;
    let v = Vec3Fix::new(q.x, q.y, q.z);
    let got = 2.0 * v.dot(axis).to_f64().atan2(q.w.to_f64());
    assert!((got - target).abs() < 5e-3, "final angle {got}");

    // linearised decay (d = 1): φ(t) = φ_t (1 - (1 + w t) e^{-w t}) to O(h + φ²)
    let cfg1 = PhysicsConfig {
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let (mut w1, j1) = world_with(cfg1, false);
    let m1 = w1.add_joint_motor_3d(
        j1,
        Vec3Fix::new(f(kp), f(kp), f(kp)),
        Vec3Fix::new(f(kd), f(kd), f(kd)),
        f(1000.0),
    );
    w1.set_joint_motor_3d_rotation_target(m1, QuatFix::from_axis_angle(axis, f(target)));
    for k in 1..=180 {
        w1.step(dt());
        let q = w1.bodies[1].rotation;
        let v = Vec3Fix::new(q.x, q.y, q.z);
        let got = 2.0 * v.dot(axis).to_f64().atan2(q.w.to_f64());
        let t = k as f64 * dt_f();
        let cont = target * (1.0 - (1.0 + 2.0 * t) * (-2.0 * t).exp());
        assert!(
            (got - cont).abs() < 5e-3,
            "frame {k}: φ = {got}, linear {cont}"
        );
    }
}

/// oracle: after disable the motor applies no torque, so ω only decays by the
/// frame damping: ω_k = ω_0 d^k exactly in the closed form.
#[test]
fn disable_stops_the_motor_torque() {
    let (mut w, j) = hinge_world();
    let d = frame_damping(&w.config);
    let m = w.add_joint_motor(j, velocity_pd(2.0, 1000.0, 1.0));
    for _ in 0..120 {
        w.step(dt());
    }
    let w0 = omega_z(&w);
    assert!(w0 > 0.5, "motor did not spin the hinge up: {w0}");
    assert!(w.disable_joint_motor(m));
    assert_eq!(
        w.joint_motor_mut(m).unwrap().controller.mode,
        MotorMode::Off
    );
    let n = w.config.substeps;
    // no torque: the reference with kp = 0 is the derivation and d only
    let reference = velocity_reference(0.0, 0.0, d, n, 0.0, w0, 60);
    for k in 1..=60 {
        w.step(dt());
        let want = w0 * d.powi(k);
        let got = omega_z(&w);
        let r = reference[k as usize - 1];
        assert!(
            (got - r).abs() < 1e-9,
            "frame {k}: ω = {got}, reference {r}"
        );
        assert!(
            (got - want).abs() < 2e-4,
            "frame {k}: ω = {got}, free decay {want}"
        );
    }
    // out of range: nothing to disable
    assert!(!w.disable_joint_motor(m + 1));
    assert!(!w.set_joint_motor_velocity_target(m + 1, Fix128::ONE));
    assert!(!w.set_joint_motor_3d_rotation_target(0, QuatFix::IDENTITY));
}

/// oracle: set_velocity_target moves the steady state to ω*(ω_t), which is
/// linear in ω_t, along the same first-order response.
#[test]
fn set_velocity_target_changes_the_steady_state() {
    let (mut w, j) = hinge_world();
    let cfg = w.config;
    let d = frame_damping(&cfg);
    let m = w.add_joint_motor(j, velocity_pd(2.0, 1000.0, 1.0));
    for _ in 0..100 {
        w.step(dt());
    }
    let w0 = omega_z(&w);
    assert!(w.set_joint_motor_velocity_target(m, f(-2.0)));
    let reference = velocity_reference(2.0, 1000.0, d, cfg.substeps, -2.0, w0, 400);
    for k in 1..=400 {
        w.step(dt());
        let want = velocity_closed_form(2.0, 1.0, d, cfg.substeps, -2.0, w0, k);
        let got = omega_z(&w);
        let r = reference[k as usize - 1];
        assert!(
            (got - r).abs() < 1e-9,
            "frame {k}: ω = {got}, reference {r}"
        );
        assert!(
            (got - want).abs() < 2e-4,
            "frame {k}: ω = {got}, closed form {want}"
        );
    }
    let fixed = velocity_closed_form(2.0, 1.0, d, cfg.substeps, -2.0, 0.0, 100_000);
    assert!(
        (omega_z(&w) - fixed).abs() < 2e-4,
        "steady state {} vs {fixed}",
        omega_z(&w)
    );
}

/// Wiring: a motor acts on the joint it was attached to and on no other, and
/// follows that joint through remove_joint / remove_body index changes.
#[test]
fn motor_drives_only_its_own_joint_and_follows_index_changes() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    // two independent hinged bodies (far apart, no contact)
    for x in [0, 20] {
        w.add_body(RigidBody::new_static(Vec3Fix::from_int(x, 10, 0)));
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(x, 0, 0), Fix128::ONE);
        b.inv_inertia = Vec3Fix::from_int(1, 1, 1);
        w.add_body(b);
    }
    let hinge = |a, b| {
        Joint::Hinge(HingeJoint::new(
            a,
            b,
            Vec3Fix::from_int(0, -10, 0),
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 0, 1),
            Vec3Fix::from_int(0, 0, 1),
        ))
    };
    let j0 = w.add_joint(hinge(0, 1));
    let j1 = w.add_joint(hinge(2, 3));
    w.add_joint_motor(j1, velocity_pd(2.0, 1000.0, 1.0));
    for _ in 0..30 {
        w.step(dt());
    }
    assert_eq!(
        w.bodies[1].angular_velocity,
        Vec3Fix::ZERO,
        "joint 0 has no motor"
    );
    assert!(w.bodies[3].angular_velocity.z.to_f64() > 0.3);

    // remove joint 0: joint 1 moves into index 0, its motor must follow
    w.remove_joint(j0);
    assert_eq!(w.joints[0].bodies(), (2, 3));
    assert_eq!(w.joint_motor_mut(0).unwrap().joint_index, 0);
    w.set_joint_motor_velocity_target(0, f(-1.0));
    for _ in 0..200 {
        w.step(dt());
    }
    assert!(w.bodies[3].angular_velocity.z.to_f64() < -0.5);
    assert_eq!(w.bodies[1].angular_velocity, Vec3Fix::ZERO);

    // remove body 3: its joint goes, and so does the motor
    w.remove_body(3);
    assert_eq!(w.joint_count(), 0);
    assert!(w.joint_motor_mut(0).is_none());
    w.step(dt());
}

/// Wiring: remove_body shifts the joints after a removed one down; the motor
/// of a kept joint follows it.
#[test]
fn remove_body_remaps_motor_of_a_shifted_joint() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    for x in [0, 20] {
        w.add_body(RigidBody::new_static(Vec3Fix::from_int(x, 10, 0)));
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(x, 0, 0), Fix128::ONE);
        b.inv_inertia = Vec3Fix::from_int(1, 1, 1);
        w.add_body(b);
    }
    let hinge = |a, b| {
        Joint::Hinge(HingeJoint::new(
            a,
            b,
            Vec3Fix::from_int(0, -10, 0),
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 0, 1),
            Vec3Fix::from_int(0, 0, 1),
        ))
    };
    w.add_joint(hinge(0, 1));
    let j1 = w.add_joint(hinge(2, 3));
    let m = w.add_joint_motor_3d(
        j1,
        Vec3Fix::from_int(4, 4, 4),
        Vec3Fix::from_int(4, 4, 4),
        Fix128::from_int(1000),
    );
    // body 1 goes with joint 0; body 3 is swapped into index 1
    w.remove_body(1);
    assert_eq!(w.joint_count(), 1);
    assert_eq!(w.joints[0].bodies(), (2, 1));
    w.set_joint_motor_3d_rotation_target(
        m,
        QuatFix::from_axis_angle(Vec3Fix::from_int(0, 0, 1), f(0.3)),
    );
    for _ in 0..240 {
        w.step(dt());
    }
    let q = w.bodies[1].rotation;
    assert!((2.0 * q.z.to_f64().atan2(q.w.to_f64()) - 0.3).abs() < 1e-2);
}

/// Wiring: a motor whose joint fell asleep wakes it once the motor has
/// something to do; a motor resting at its target lets it sleep.
#[test]
fn motor_wakes_a_sleeping_joint_and_lets_an_idle_one_sleep() {
    let (mut w, j) = hinge_world();
    let mut pd = PdController::new(f(4.0), f(4.0), f(1000.0));
    pd.set_position_target(Fix128::ZERO);
    let m = w.add_joint_motor(j, pd);
    for _ in 0..600 {
        w.step(dt());
    }
    assert!(
        w.is_sleeping(1),
        "a motor at its target must not keep the joint awake"
    );
    w.joint_motor_mut(m)
        .unwrap()
        .controller
        .set_position_target(f(0.5));
    for _ in 0..240 {
        w.step(dt());
    }
    assert!((angle_z(&w) - 0.5).abs() < 2e-2, "θ = {}", angle_z(&w));
}

/// Degenerate input: a motor naming no joint, or Off, does nothing; a motor
/// on a joint to a static body drives only the dynamic one.
#[test]
fn motor_with_no_joint_or_off_does_nothing() {
    let (mut w, _) = hinge_world();
    w.add_joint_motor(7, velocity_pd(2.0, 1000.0, 1.0));
    w.add_joint_motor_3d(9, Vec3Fix::from_int(1, 1, 1), Vec3Fix::ZERO, Fix128::ONE);
    w.set_joint_motor_3d_rotation_target(
        0,
        QuatFix::from_axis_angle(Vec3Fix::from_int(0, 0, 1), f(1.0)),
    );
    let (mut w_off, j) = hinge_world();
    w_off.add_joint_motor(j, PdController::new(f(2.0), Fix128::ZERO, f(1000.0)));
    w_off.add_joint_motor_3d(j, Vec3Fix::from_int(1, 1, 1), Vec3Fix::ZERO, Fix128::ONE);
    let (mut w_ref, _) = hinge_world();
    for _ in 0..30 {
        w.step(dt());
        w_off.step(dt());
        w_ref.step(dt());
    }
    for x in [&w, &w_off] {
        assert_eq!(
            x.bodies[1].angular_velocity,
            w_ref.bodies[1].angular_velocity
        );
        assert_eq!(x.bodies[1].position, w_ref.bodies[1].position);
        assert_eq!(x.bodies[0].position, Vec3Fix::from_int(0, 10, 0));
    }
}

/// The batched path applies the motors too (same closed form).
#[cfg(feature = "parallel")]
#[test]
fn step_parallel_applies_the_velocity_motor() {
    let (mut w, j) = hinge_world();
    let cfg = w.config;
    let d = frame_damping(&cfg);
    w.add_joint_motor(j, velocity_pd(2.0, 1000.0, 1.0));
    for k in 1..=120 {
        w.step_parallel(dt());
        let want = velocity_closed_form(2.0, 1.0, d, cfg.substeps, 1.0, 0.0, k);
        assert!((omega_z(&w) - want).abs() < 2e-4, "frame {k}");
    }
}

/// The TGS backend applies the motors once per tick with the full dt: the
/// same first-order response with one substep per frame (n = 1) and the
/// frame damping d (TGS keeps the angular velocity it integrates, so no
/// derivation shrink).
#[test]
fn tgs_backend_applies_the_velocity_motor_once_per_tick() {
    let cfg = PhysicsConfig {
        solver_backend: alice_physics::SolverBackend::Tgs,
        ..PhysicsConfig::default()
    };
    let d = frame_damping(&cfg);
    let (mut w, j) = world_with(cfg, true);
    w.add_joint_motor(j, velocity_pd(2.0, 1000.0, 1.0));
    for k in 1..=120 {
        w.step(dt());
        let want = velocity_closed_form(2.0, 1.0, d, 1, 1.0, 0.0, k);
        let got = omega_z(&w);
        assert!(
            (got - want).abs() < 1e-9,
            "frame {k}: ω = {got}, closed form {want}"
        );
    }
}

/// oracle: the 3-axis motor's torque magnitude is capped at max_torque, so
/// far from the target the rotation accelerates at τ_max / I about the error
/// axis: ω_k = c (1 - d^k), c = d τ_max dt / (I (1 - d)).
#[test]
fn ball_rotation_motor_torque_cap_limits_acceleration() {
    let (mut w, j) = world_with(PhysicsConfig::default(), false);
    let d = frame_damping(&w.config);
    let n = w.config.substeps;
    let tau_max = 0.25;
    let m = w.add_joint_motor_3d(
        j,
        Vec3Fix::from_int(1000, 1000, 1000),
        Vec3Fix::ZERO,
        f(tau_max),
    );
    let axis = Vec3Fix::new(Fix128::ZERO, f(0.6), f(0.8));
    w.set_joint_motor_3d_rotation_target(m, QuatFix::from_axis_angle(axis, f(1.0)));
    let c = d * tau_max * dt_f() / (1.0 - d);
    let reference = velocity_reference(1e9, tau_max, d, n, 1e9, 0.0, 30);
    for k in 1..=30 {
        w.step(dt());
        let wv = w.bodies[1].angular_velocity;
        let along = wv.dot(axis).to_f64();
        let off = (wv - axis * wv.dot(axis)).length().to_f64();
        let r = reference[k as usize - 1];
        assert!(off < 1e-9, "frame {k}: ω left the axis ({off})");
        assert!(
            (along - r).abs() < 1e-9,
            "frame {k}: ω = {along}, reference {r}"
        );
        assert!(
            (along - c * (1.0 - d.powi(k))).abs() < 1e-5,
            "frame {k}: ω = {along}"
        );
    }
}

/// oracle: with both bodies dynamic the motor torque is equal and opposite:
/// I_a ω_a + I_b ω_b stays 0, and the relative rate ω_b - ω_a follows the
/// velocity-motor response with the effective gain kp (1/I_a + 1/I_b).
#[test]
fn hinge_motor_between_two_dynamic_bodies_conserves_angular_momentum() {
    let cfg = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(cfg);
    let mut a = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    a.inv_inertia = Vec3Fix::new(f(0.5), f(0.5), f(0.5)); // I_a = 2
    let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = Vec3Fix::from_int(1, 1, 1); // I_b = 1
    w.add_body(a);
    w.add_body(b);
    let j = w.add_joint(Joint::Hinge(HingeJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(0, 0, 1),
        Vec3Fix::from_int(0, 0, 1),
    )));
    w.add_joint_motor(j, velocity_pd(2.0, 1000.0, 1.0));
    let d = frame_damping(&cfg);
    for k in 1..=120 {
        w.step(dt());
        let wa = w.bodies[0].angular_velocity.z.to_f64();
        let wb = w.bodies[1].angular_velocity.z.to_f64();
        assert!(
            (2.0 * wa + wb).abs() < 1e-4,
            "frame {k}: L = {}",
            2.0 * wa + wb
        );
        let want = velocity_closed_form(2.0 * 1.5, 1.0, d, cfg.substeps, 1.0, 0.0, k);
        assert!(
            ((wb - wa) - want).abs() < 2e-4,
            "frame {k}: ω_rel = {}, {want}",
            wb - wa
        );
    }
    assert!(
        w.bodies[0].angular_velocity.z.to_f64() < -0.1,
        "body A must counter-rotate"
    );
}
