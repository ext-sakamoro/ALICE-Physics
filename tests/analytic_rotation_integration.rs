//! Oracles for the rotational integration of `PhysicsWorld::step`, single-axis
//! rotation about a principal axis only (no gyroscopic coupling, no
//! precession: those are covered elsewhere).
//!
//! The physical expectation about a fixed principal axis is the closed form
//! of `I ω̇ = τ`:
//!
//! - constant torque from rest: `ω(t) = τ t / I`, angle `θ(t) = ½ (τ/I) t²`,
//!   the axis does not move
//! - free rotation: `ω(t) = ω0` (magnitude and direction), `θ(t) = ω0 t`,
//!   kinetic energy `½ I ω0²` conserved
//!
//! None of these depends on `substeps`; the angle may carry the integrator's
//! own truncation `½ (τ/I) h t` (`h = dt / s`, the same semi-implicit Euler
//! term as the linear case), the angular velocity may not drift at all.
//!
//! ## What the current step does (measured)
//!
//! Each substep rotates by `from_axis_angle(ω̂, |ω| h)` and then re-derives the
//! angular velocity from the rotation change as `2 · dq.xyz / h`, i.e.
//! `|ω| ← (2/h) sin(|ω| h / 2)`. That map loses `|ω|³ h² / 24` per substep, so
//! the angular velocity decays as `dω/dt ≈ −ω³ h / 24` with nothing acting on
//! the body: free rotation at 5 rad/s loses 2.0 % in 10 s at `s = 8`
//! (`dt = 1/64`), 13 % at `s = 1`. A constant torque (`τ/I = 1 rad/s²`) ends
//! 2.0 % / 13 % below `τ t / I` after 10 s. `add_torque` is, like `add_force`, a frame-head
//! change of the angular velocity, so the angle also carries the splitting
//! term `½ (τ/I) dt t (s − 1)/s` of the linear case.
//!
//! Every oracle reports, next to the measured value, the same quantity from an
//! f64 model of that map (`ω ← ω + (τ/I) dt` once per frame, then per substep
//! `θ += ω h`, `ω ← (2/h) sin(ω h/2)`), which reproduces the measurement to
//! rounding: the defect is this map and nothing else.
//!
//! The angular velocity is now re-derived by the exact logarithm of the
//! rotation (`θ = 2·atan2(|v|, w)`), so the two free-rotation oracles hold to
//! rounding; the measurements above are from the chord re-derivation they
//! replaced. The constant-torque oracle stays a `src gap` (`#[ignore]`d):
//! `add_torque` is applied once at the frame head, which leaves the angle
//! splitting term `½ (τ/I) dt t (s − 1)/s` (+0.078 rad at 10 s for every `s`;
//! its `ω` is already exact). Run with `--ignored` to see it fail.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The f64 values here are the oracle (closed-form references) and the f64
// model of the defect, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sleeping::SleepConfig;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

/// Frame length `dt = 1/64 s` (dyadic).
const DT_DEN: i64 = 64;
/// Simulated time 10 s.
const FRAMES: usize = 640;
/// Substep counts swept by every oracle.
const SUBSTEPS: [usize; 5] = [1, 2, 4, 8, 16];
/// Rounding-level tolerance (rad/s, rad, J).
const ROUND: f64 = 1e-9;

fn f(x: Fix128) -> f64 {
    x.to_f64()
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, DT_DEN)
}

fn v3(v: Vec3Fix) -> [f64; 3] {
    [f(v.x), f(v.y), f(v.z)]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// One dynamic unit-mass body at the origin, gravity 0 (so nothing but the
/// rotation is exercised), no global damping, sleep off.
fn world(substeps: usize, inv_inertia: Vec3Fix) -> (PhysicsWorld, usize) {
    let config = PhysicsConfig {
        substeps,
        damping: Fix128::ONE,
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::ZERO,
        angular_threshold: Fix128::ZERO,
        frames_to_sleep: u32::MAX,
    });
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    body.inv_inertia = inv_inertia;
    let id = w.add_body(body);
    (w, id)
}

/// Signed rotation angle of `q` about the unit `axis`, in `(−2π, 2π]`.
fn angle_about(q: QuatFix, axis: [f64; 3]) -> f64 {
    let along = dot([f(q.x), f(q.y), f(q.z)], axis);
    2.0 * along.atan2(f(q.w))
}

/// Unwraps successive angle samples (per-frame increments are far below π).
struct Unwrap {
    prev: f64,
    total: f64,
}

impl Unwrap {
    fn new() -> Self {
        Self {
            prev: 0.0,
            total: 0.0,
        }
    }

    fn push(&mut self, raw: f64) -> f64 {
        let two_pi = 2.0 * core::f64::consts::PI;
        let mut d = (raw - self.prev) % two_pi;
        if d > core::f64::consts::PI {
            d -= two_pi;
        } else if d < -core::f64::consts::PI {
            d += two_pi;
        }
        self.prev = raw;
        self.total += d;
        self.total
    }
}

/// f64 model of the current per-frame map: kick, then per substep rotate by
/// `ω h` and re-derive `ω = (2/h) sin(ω h/2)`. Returns `(ω, θ)` after `frames`.
fn chord_model(omega0: f64, alpha: f64, s: usize, frames: usize) -> (f64, f64) {
    let dtf = 1.0 / DT_DEN as f64;
    let h = dtf / s as f64;
    let (mut w, mut th) = (omega0, 0.0);
    for _ in 0..frames {
        w += alpha * dtf;
        for _ in 0..s {
            th += w * h;
            w = 2.0 / h * (0.5 * w * h).sin();
        }
    }
    (w, th)
}

fn report(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} case(s) off the closed form:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Constant torque `τ` about `z` from rest (default unit-sphere inertia,
/// `I = 2/5`, `τ = 2/5` so `τ/I = 1 rad/s²`), applied with `add_torque` every
/// frame.
///
/// - physical expectation: `ω(t) = (0, 0, t)` at every frame boundary
///   (to rounding), the angle `θ(t) = ½ t²` to within the truncation `½ h t`,
///   and the angular velocity stays on `z`
/// - current (measured, 10 s): `ω_z(10)` = 8.7162 (s=1), 9.2825 (2),
///   9.6183 (4), 9.8028 (8), 9.8997 (16) against 10, `θ(10)` = 47.287 /
///   48.575 / 49.296 / 49.679 / 49.876 against 50; the chord model gives the
///   same `ω` to ≤ 1e-9; the axis stays exactly on `z`
/// - difference closed form: `dω/dt = τ/I − ω³ h / 24` (chord re-derivation)
///   plus the angle splitting term `½ (τ/I) dt t (s − 1)/s`
#[test]
#[ignore = "src gap: add_torque is applied once at the frame head, so θ carries the splitting term ½ (τ/I) dt t (s − 1)/s (+0.078 rad at 10 s for every s); the frame-head external-force item is a separate pending decision"]
fn constant_torque_spins_up_as_tau_t_over_i() {
    let mut failures = Vec::new();
    let dtf = 1.0 / DT_DEN as f64;
    for s in SUBSTEPS {
        let h = dtf / s as f64;
        let (mut w, id) = world(s, {
            let inv_i = Fix128::from_ratio(5, 2);
            Vec3Fix::new(inv_i, inv_i, inv_i)
        });
        let torque = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_ratio(2, 5));
        let mut unwrap = Unwrap::new();
        let (mut worst_w, mut worst_off_axis, mut worst_theta) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut last = (0.0, 0.0);
        for n in 1..=FRAMES {
            w.bodies[id].add_torque(torque, dt());
            w.step(dt());
            let t = n as f64 * dtf;
            let om = v3(w.bodies[id].angular_velocity);
            let theta = unwrap.push(angle_about(w.bodies[id].rotation, [0.0, 0.0, 1.0]));
            worst_w = worst_w.max((om[2] - t).abs());
            worst_off_axis = worst_off_axis.max(om[0].abs().max(om[1].abs()));
            worst_theta = worst_theta.max((theta - 0.5 * t * t).abs() - 0.5 * h * t);
            last = (om[2], theta);
        }
        let (model_w, model_th) = chord_model(0.0, 1.0, s, FRAMES);
        eprintln!(
            "torque s={s}: ω_z(10) {:.12} (expected 10, chord model {model_w:.12}), θ(10) {:.9} (expected 50 ± {:.6}, model {model_th:.9}), off-axis {worst_off_axis:.3e}",
            last.0,
            last.1,
            0.5 * h * 10.0
        );
        if worst_w > ROUND || worst_off_axis > ROUND || worst_theta > ROUND {
            failures.push(format!(
                "s={s}: ω_z(10) = {:.12} rad/s (expected 10; chord model {model_w:.12}, residual {:+.3e}), max |ω_z − t| {worst_w:.3e}, θ(10) = {:.9} (expected 50 within ½ h t = {:.6}; model {model_th:.9}), off-axis {worst_off_axis:.3e}",
                last.0,
                last.0 - model_w,
                last.1,
                0.5 * h * 10.0
            ));
        }
    }
    report(&failures);
}

/// Free rotation about a principal axis keeps `ω` (magnitude and direction),
/// turns the body by `ω0 t` and keeps `½ I ω0²`.
///
/// Two bodies per substep count: an anisotropic one (`I = diag(1, 2, 3)`)
/// spinning about its `z` principal axis at 5 rad/s, and an isotropic one
/// (unit sphere, every axis principal) spinning about the oblique axis
/// `(1, 2, 2)/3` at 6 rad/s.
///
/// - physical expectation: `|ω(t)| = ω0`, `ω̂(t) = ω̂0`, `θ(t) = ω0 t`,
///   `E(t) = ½ I ω0²` at every frame boundary (to rounding) and for every `s`
/// - current (measured, 10 s): `|ω|/ω0` = 0.8685 / 0.9274 / 0.9616 / 0.9803
///   / 0.9900 for s = 1 / 2 / 4 / 8 / 16 (anisotropic, 5 rad/s; `E/E0` =
///   0.754 / 0.860 / 0.925 / 0.961 / 0.980), 0.8251 / 0.9001 / 0.9461 /
///   0.9719 / 0.9857 (isotropic, 6 rad/s); the direction stays exact (≤ 1e-12
///   off-axis); the chord model reproduces `|ω|` to ≤ 3e-9 and `θ` to ≤ 2e-8
/// - difference closed form: `dω/dt = −ω³ h / 24`
#[test]
fn free_rotation_about_a_principal_axis_conserves_omega_angle_and_energy() {
    let mut failures = Vec::new();
    let dtf = 1.0 / DT_DEN as f64;
    let cases: [(&str, [f64; 3], Vec3Fix, Vec3Fix); 2] = [
        (
            "anisotropic I=diag(1,2,3) about z, 5 rad/s",
            [1.0, 2.0, 3.0],
            Vec3Fix::new(
                Fix128::ONE,
                Fix128::from_ratio(1, 2),
                Fix128::from_ratio(1, 3),
            ),
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(5)),
        ),
        (
            "isotropic I=2/5 about (1,2,2)/3, 6 rad/s",
            [0.4, 0.4, 0.4],
            {
                let inv_i = Fix128::from_ratio(5, 2);
                Vec3Fix::new(inv_i, inv_i, inv_i)
            },
            Vec3Fix::from_int(2, 4, 4),
        ),
    ];
    for s in SUBSTEPS {
        for (name, inertia, inv_inertia, omega0_fix) in &cases {
            let (mut w, id) = world(s, *inv_inertia);
            w.bodies[id].angular_velocity = *omega0_fix;
            let om0 = v3(*omega0_fix);
            let w0 = norm(om0);
            let axis = [om0[0] / w0, om0[1] / w0, om0[2] / w0];
            let energy = |o: [f64; 3]| {
                0.5 * (inertia[0] * o[0] * o[0]
                    + inertia[1] * o[1] * o[1]
                    + inertia[2] * o[2] * o[2])
            };
            let e0 = energy(om0);
            let mut unwrap = Unwrap::new();
            let (mut worst_mag, mut worst_dir, mut worst_theta, mut worst_e) =
                (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            let mut last = (0.0, 0.0, 0.0);
            for n in 1..=FRAMES {
                w.step(dt());
                let t = n as f64 * dtf;
                let om = v3(w.bodies[id].angular_velocity);
                let mag = norm(om);
                let along = dot(om, axis);
                let perp = [
                    om[0] - along * axis[0],
                    om[1] - along * axis[1],
                    om[2] - along * axis[2],
                ];
                let theta = unwrap.push(angle_about(w.bodies[id].rotation, axis));
                worst_mag = worst_mag.max((mag - w0).abs());
                worst_dir = worst_dir.max(norm(perp));
                worst_theta = worst_theta.max((theta - w0 * t).abs());
                worst_e = worst_e.max((energy(om) - e0).abs());
                last = (mag, theta, energy(om));
            }
            let (model_w, model_th) = chord_model(w0, 0.0, s, FRAMES);
            eprintln!(
                "free s={s} {name}: |ω|/ω0 {:.9} (model {:.9}), θ − ω0 t {:+.9} (model {:+.9}), E/E0 {:.9}, off-axis {worst_dir:.3e}",
                last.0 / w0,
                model_w / w0,
                last.1 - w0 * 10.0,
                model_th - w0 * 10.0,
                last.2 / e0
            );
            if worst_mag > ROUND || worst_dir > ROUND || worst_theta > ROUND || worst_e > ROUND {
                failures.push(format!(
                    "s={s} {name}: |ω(10)| = {:.12} (expected {w0}; chord model {model_w:.12}, residual {:+.3e}), θ(10) − ω0 t = {:+.9} (model {:+.9}), E(10)/E0 = {:.9}, max off-axis {worst_dir:.3e}",
                    last.0,
                    last.0 - model_w,
                    last.1 - w0 * 10.0,
                    model_th - w0 * 10.0,
                    last.2 / e0
                ));
            }
        }
    }
    report(&failures);
}

/// The angular state after 1 s of free rotation does not depend on the
/// substep count: the five runs `s = 1, 2, 4, 8, 16` agree with each other
/// (and with `ω0`, `ω0 t`).
///
/// - physical expectation: spread of `|ω(1)|` and of `θ(1)` over `s` is 0
///   (to rounding)
/// - current (measured, 5 rad/s about `z`, `I = 2/5`): `|ω(1)|` = 4.92053 /
///   4.95980 / 4.97978 / 4.98986 / 4.99492 for s = 1 / 2 / 4 / 8 / 16, a
///   spread of 0.0744 rad/s (estimate 0.0763); `θ(1)` from 4.96057 to 4.99746
/// - difference closed form: `|ω(1)| ≈ ω0 (1 − ω0² h / 24)` per run, i.e. a
///   spread of `ω0³ (h_max − h_min)/24`
#[test]
fn free_rotation_result_does_not_depend_on_substeps() {
    let dtf = 1.0 / DT_DEN as f64;
    let mut mags = Vec::new();
    let mut thetas = Vec::new();
    for s in SUBSTEPS {
        let inv_i = Fix128::from_ratio(5, 2);
        let (mut w, id) = world(s, Vec3Fix::new(inv_i, inv_i, inv_i));
        w.bodies[id].angular_velocity = Vec3Fix::from_int(0, 0, 5);
        let mut unwrap = Unwrap::new();
        let mut theta = 0.0;
        for _ in 0..DT_DEN {
            w.step(dt());
            theta = unwrap.push(angle_about(w.bodies[id].rotation, [0.0, 0.0, 1.0]));
        }
        let mag = norm(v3(w.bodies[id].angular_velocity));
        eprintln!("sweep s={s}: |ω(1)| {mag:.9}, θ(1) {theta:.9}");
        mags.push((s, mag));
        thetas.push((s, theta));
    }
    let spread = |v: &[(usize, f64)]| {
        let lo = v.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let hi = v.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    let h_max = dtf;
    let h_min = dtf / 16.0;
    let predicted = 125.0 * (h_max - h_min) / 24.0;
    assert!(
        spread(&mags) <= ROUND && spread(&thetas) <= ROUND,
        "|ω(1)| over s = {mags:?} (spread {:.6}, chord estimate ω0³ (h_max − h_min)/24 = {predicted:.6}), θ(1) over s = {thetas:?} (spread {:.6}); expected 5 rad/s and 5 rad for every s",
        spread(&mags),
        spread(&thetas)
    );
}
