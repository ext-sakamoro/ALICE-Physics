//! Oracles for external forces integrated over the frame, measured from the
//! production entry points (`PhysicsWorld::step` with `RigidBody::add_force`
//! and the world force fields).
//!
//! The physical expectation of a constant (or position-dependent) external
//! force is the closed-form trajectory of `m ẍ = F + m g`. It does not depend
//! on `substeps`: `substeps` is a precision parameter and may only change the
//! integrator's own truncation error, which vanishes with the substep length
//! `h = dt / s`.
//!
//! ## What the current step does (measured)
//!
//! `add_force(F, dt)` and the force fields (`Phase 1` of `step`) change the
//! velocity **once, at the head of the frame** (`v += F/m · dt`), while gravity
//! is added **in every substep** (`v += g h`). The frame-boundary velocity is
//! therefore exact, but the position picks up a term that does not shrink with
//! `s`. Over `t = n dt` (constant `F`, `a = F/m + g`) the position error is
//!
//! ```text
//! x_step(t) − x_exact(t) = ½ a h t  +  ½ (F/m) dt t (s − 1)/s
//!                          └ integrator ┘   └ splitting defect ┘
//! ```
//!
//! The first term is the semi-implicit Euler truncation that a per-substep
//! force would also have; it is the tolerance of the trajectory oracles. The
//! second term is the defect: it is zero for `s = 1` and approaches
//! `½ (F/m) dt t` as `s` grows, i.e. more substeps do not help. For a body held
//! by `F = −m g` (hover) the first term is zero and the defect is
//! `n · g dt² (s − 1)/(2 s)`: +0.7292 m in 10 s at `dt = 1/60`, `s = 8`.
//!
//! A position-dependent restoring force held at its equilibrium (buoyancy,
//! a spring at its static extension `m g / k`) gets the same per-frame
//! forcing `c = g dt² (s − 1)/(2 s)` on the position, so instead of staying
//! put the body oscillates around the equilibrium with an amplitude of about
//! `c / (ω dt) = g dt (s − 1)/(2 s ω)` (`ω = √(k/m)`).
//!
//! Each oracle sweeps `s ∈ {1, 2, 4, 8, 16}` (and, where the scene has one,
//! the initial velocity or the frame length) against one closed form, so a
//! fix that only helps one setting stays red. The `s = 1` row is green today
//! (no splitting inside a single substep); the rows `s > 1` are red.
//!
//! These are `src gap` oracles: they are `#[ignore]`d until the step applies
//! external forces inside the substep loop. Run with `--ignored` to see them
//! fail with the measured values and the defect's closed form.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The f64 values here are the oracle (closed-form references), not
// simulation state, so the det_math determinism gate does not apply.
#![allow(clippy::disallowed_methods)]

use alice_physics::force::{ForceField, ForceFieldInstance};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sleeping::SleepConfig;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

/// Gravity magnitude (m/s²), the default of `PhysicsConfig`.
const G: f64 = 10.0;
/// Frame length `dt = 1/64 s` (dyadic, so `dt / s` is exact in `Fix128`).
const DT_DEN: i64 = 64;
/// Simulated time 10 s = 640 frames at `dt = 1/64`.
const FRAMES: usize = 640;
/// Substep counts swept by every oracle.
const SUBSTEPS: [usize; 5] = [1, 2, 4, 8, 16];
/// Rounding-level tolerance (m, m/s). The `Fix128` state has ~1e-19
/// resolution; 640 frames × 16 substeps accumulate far below this.
const ROUND: f64 = 1e-9;

fn f(x: Fix128) -> f64 {
    x.to_f64()
}

/// One dynamic body of mass `mass` at `(0, y0, 0)`, gravity `(0, −10, 0)`,
/// no global damping (so the closed form is the undamped one) and sleep off
/// (a zero threshold is never undercut, so the body is never parked).
fn world(substeps: usize, mass: Fix128, y0: Fix128) -> (PhysicsWorld, usize) {
    let config = PhysicsConfig {
        substeps,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::ZERO,
        angular_threshold: Fix128::ZERO,
        frames_to_sleep: u32::MAX,
    });
    let id = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, y0, Fix128::ZERO),
        mass,
    ));
    (w, id)
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, DT_DEN)
}

fn splitting_defect_hover(n: usize, s: usize) -> f64 {
    let dt = 1.0 / DT_DEN as f64;
    n as f64 * G * dt * dt * (s as f64 - 1.0) / (2.0 * s as f64)
}

fn report(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} case(s) off the closed form:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Hover with `RigidBody::add_force`: thrust `F = m g` up every frame keeps the
/// altitude constant.
///
/// - physical expectation: `y(t) = y0`, `v(t) = 0` for every `s`
/// - current (measured, 10 s, `dt = 1/64`): `y − y0` = 0 (s=1), +0.390625 (2),
///   +0.5859375 (4), +0.68359375 (8), +0.732421875 (16); frame-boundary
///   velocity 0
/// - difference closed form: `n g dt² (s − 1)/(2 s)` (the splitting defect,
///   integrator term 0 since `a = 0`)
#[test]
#[ignore = "src gap: add_force is a frame-head impulse while gravity is per substep, so a hovering body climbs n g dt^2 (s-1)/(2s)"]
fn hover_with_add_force_keeps_altitude_for_every_substep_count() {
    let mut failures = Vec::new();
    for s in SUBSTEPS {
        let mass = Fix128::from_int(2);
        let (mut w, id) = world(s, mass, Fix128::from_int(5));
        let thrust = Vec3Fix::new(Fix128::ZERO, mass * Fix128::from_int(10), Fix128::ZERO);
        let mut worst = 0.0_f64;
        let mut worst_v = 0.0_f64;
        for _ in 0..FRAMES {
            w.bodies[id].add_force(thrust, dt());
            w.step(dt());
            worst = worst.max((f(w.bodies[id].position.y) - 5.0).abs());
            worst_v = worst_v.max(f(w.bodies[id].velocity.y).abs());
        }
        let drift = f(w.bodies[id].position.y) - 5.0;
        let defect = splitting_defect_hover(FRAMES, s);
        eprintln!("hover add_force s={s}: drift {drift:+.12} (defect closed form {defect:+.12}), max |v| {worst_v:.3e}");
        if worst > ROUND || worst_v > ROUND {
            failures.push(format!(
                "s={s}: y − y0 = {drift:+.12} m after 10 s (expected 0; splitting defect n g dt²(s−1)/(2s) = {defect:+.12}, residual {:+.3e}), max |v| at frame boundaries {worst_v:.3e}",
                drift - defect
            ));
        }
    }
    report(&failures);
}

/// Hover with a world force field (`ForceField::Directional`, `Phase 1` of
/// `step`): the same expectation as the `add_force` hover, through the other
/// production entry point for external forces.
///
/// - physical expectation: `y(t) = y0` for every `s`
/// - current (measured): identical to the `add_force` hover, bit for bit
/// - difference closed form: `n g dt² (s − 1)/(2 s)`
#[test]
#[ignore = "src gap: force fields are applied once at the head of the frame while gravity is per substep, so a field-held body climbs n g dt^2 (s-1)/(2s)"]
fn hover_with_force_field_keeps_altitude_for_every_substep_count() {
    let mut failures = Vec::new();
    for s in SUBSTEPS {
        let mass = Fix128::from_int(2);
        let (mut w, id) = world(s, mass, Fix128::from_int(5));
        w.add_force_field(ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
            strength: mass * Fix128::from_int(10),
        }));
        let mut worst = 0.0_f64;
        for _ in 0..FRAMES {
            w.step(dt());
            worst = worst.max((f(w.bodies[id].position.y) - 5.0).abs());
        }
        let drift = f(w.bodies[id].position.y) - 5.0;
        let defect = splitting_defect_hover(FRAMES, s);
        eprintln!("hover field s={s}: drift {drift:+.12} (defect closed form {defect:+.12})");
        if worst > ROUND {
            failures.push(format!(
                "s={s}: y − y0 = {drift:+.12} m after 10 s (expected 0; splitting defect = {defect:+.12}, residual {:+.3e})",
                drift - defect
            ));
        }
    }
    report(&failures);
}

/// Constant force `F` (up, `F/m = 3 g`, net `a = 2 g`) from an initial
/// velocity `v0`: `y(t) = y0 + v0 t + ½ a t²`, `v(t) = v0 + a t`.
///
/// Swept over `s` and over `v0 = k/4 · g dt` (`k = 0..3`, the initial velocity
/// relative to one frame's gravity change). Every frame boundary is checked.
///
/// - physical expectation: the closed form, to within the integrator
///   truncation `½ |a| h t` (`h = dt / s`, vanishes as `s → ∞`)
/// - current (measured, 10 s): `y − y_exact = ½ a h t + ½ (F/m) dt t (s−1)/s`
///   to the last digit (residual ≤ 3e-13) for every `v0`; e.g. `s = 8`:
///   +2.24609375 m where the truncation allows 0.1953125 (s=1: +1.5625, all of
///   it truncation); frame-boundary velocity exact (≤ 3e-14)
/// - difference closed form: `½ (F/m) dt t (s − 1)/s` beyond the truncation
#[test]
#[ignore = "src gap: a frame-head force impulse adds ½ (F/m) dt t (s-1)/s to the position, which more substeps do not reduce"]
fn constant_force_follows_the_closed_form_trajectory() {
    let mut failures = Vec::new();
    let dtf = 1.0 / DT_DEN as f64;
    let f_over_m = 3.0 * G;
    let a = f_over_m - G;
    for s in SUBSTEPS {
        let h = dtf / s as f64;
        for k in 0..4_i64 {
            let mass = Fix128::from_int(3);
            let (mut w, id) = world(s, mass, Fix128::ZERO);
            // v0 = k/4 · g dt = k · 10 / 256
            let v0_fix = Fix128::from_ratio(10 * k, 4 * DT_DEN);
            w.bodies[id].velocity = Vec3Fix::new(Fix128::ZERO, v0_fix, Fix128::ZERO);
            let v0 = f(v0_fix);
            let force = Vec3Fix::new(Fix128::ZERO, mass * Fix128::from_int(30), Fix128::ZERO);
            let mut worst_excess = 0.0_f64;
            let mut worst_v = 0.0_f64;
            let mut last = (0.0, 0.0);
            for n in 1..=FRAMES {
                w.bodies[id].add_force(force, dt());
                w.step(dt());
                let t = n as f64 * dtf;
                let y = f(w.bodies[id].position.y);
                let v = f(w.bodies[id].velocity.y);
                let y_exact = v0 * t + 0.5 * a * t * t;
                let tol = 0.5 * a.abs() * h * t + ROUND;
                worst_excess = worst_excess.max((y - y_exact).abs() - tol);
                worst_v = worst_v.max((v - (v0 + a * t)).abs());
                last = (y - y_exact, t);
            }
            let (err, t) = last;
            let truncation = 0.5 * a * h * t;
            let defect = 0.5 * f_over_m * dtf * t * (s as f64 - 1.0) / s as f64;
            if k == 0 {
                eprintln!("const force s={s}: y − y_exact {err:+.12} (truncation {truncation:+.12} + defect {defect:+.12}), max |Δv| {worst_v:.3e}");
            }
            if worst_excess > 0.0 || worst_v > ROUND {
                failures.push(format!(
                    "s={s} v0={v0:.6}: y − y_exact = {err:+.12} m at t = 10 s (truncation ½ a h t = {truncation:+.12}; splitting defect ½ (F/m) dt t (s−1)/s = {defect:+.12}, residual {:+.3e}), max |Δv| {worst_v:.3e}",
                    err - truncation - defect
                ));
            }
        }
    }
    report(&failures);
}

/// The constant-force trajectory at a fixed time does not depend on how the
/// time is cut into frames: `t = 1 s` reached with `dt = 1/32, 1/64, 1/128`
/// (the coarser grids' frame boundaries are mid-frame points of the finer
/// ones) at a fixed substep length `h = 1/512`.
///
/// - physical expectation: `y(1) = ½ a` (`a = 2 g`) for every `dt`, to within
///   the truncation `½ a h` (same `h` for all three)
/// - current (measured): `y(1) − ½ a = ½ a h + ½ (F/m)(dt − h)`, i.e.
///   +0.458984375 / +0.224609375 / +0.107421875 m for `dt = 1/32 / 1/64 /
///   1/128` (truncation 0.01953125 each); the frame length, not the substep
///   length, sets the error
/// - difference closed form: `½ (F/m)(dt − h) · t`
#[test]
#[ignore = "src gap: with a frame-head force impulse the position error is set by the frame length dt, not the substep length"]
fn constant_force_position_does_not_depend_on_the_frame_length() {
    let mut failures = Vec::new();
    let f_over_m = 3.0 * G;
    let a = f_over_m - G;
    let h = 1.0 / 512.0;
    for (den, s) in [(32_i64, 16_usize), (64, 8), (128, 4)] {
        let mass = Fix128::ONE;
        let (mut w, id) = world(s, mass, Fix128::ZERO);
        let dt = Fix128::from_ratio(1, den);
        let force = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(30), Fix128::ZERO);
        for _ in 0..den {
            w.bodies[id].add_force(force, dt);
            w.step(dt);
        }
        let y = f(w.bodies[id].position.y);
        let err = y - 0.5 * a;
        let truncation = 0.5 * a * h;
        let defect = 0.5 * f_over_m * (1.0 / den as f64 - h);
        eprintln!("frame length dt=1/{den} s={s}: y(1) − ½a {err:+.12} (truncation {truncation:+.12} + defect {defect:+.12})");
        if (err.abs() - truncation.abs()) > ROUND {
            failures.push(format!(
                "dt=1/{den} s={s}: y(1) − ½ a = {err:+.12} m (truncation ½ a h = {truncation:+.12}; splitting defect ½ (F/m)(dt − h) = {defect:+.12}, residual {:+.3e})",
                err - truncation - defect
            ));
        }
    }
    report(&failures);
}

/// Buoyancy (`ForceField::Buoyancy`, force `ρ · depth` up, no drag) holding a
/// body at its equilibrium depth `d* = m g / ρ`, started there at rest.
///
/// - physical expectation: the body stays at `y = surface − d*` for every `s`
/// - current (measured, 10 s, `m = 1`, `ρ = 40`, `ω = √(ρ/m)`): an oscillation
///   around the equilibrium; max `|y − y*|` 0 (s=1), 0.006184 (2), 0.009276
///   (4), 0.010822 (8), 0.011595 (16), within 0.2 % of `g dt (s − 1)/(2 s ω)`
/// - difference closed form: per-frame position forcing `g dt² (s − 1)/(2 s)`,
///   giving an oscillation amplitude of about `g dt (s − 1)/(2 s ω)`
#[test]
#[ignore = "src gap: the buoyancy field is a frame-head impulse while gravity is per substep, so a floating body at equilibrium oscillates by ~g dt (s-1)/(2 s ω)"]
fn buoyancy_field_holds_a_body_at_its_equilibrium_depth() {
    let mut failures = Vec::new();
    let rho = 40.0_f64;
    let omega = rho.sqrt();
    let dtf = 1.0 / DT_DEN as f64;
    for s in SUBSTEPS {
        // m = 1, ρ = 40 → d* = m g / ρ = 1/4, surface at 0
        let (mut w, id) = world(s, Fix128::ONE, Fix128::from_ratio(-1, 4));
        w.add_force_field(ForceFieldInstance::new(ForceField::Buoyancy {
            surface_y: Fix128::ZERO,
            density: Fix128::from_int(40),
            drag: Fix128::ZERO,
        }));
        let mut worst = 0.0_f64;
        for _ in 0..FRAMES {
            w.step(dt());
            worst = worst.max((f(w.bodies[id].position.y) + 0.25).abs());
        }
        let amp = G * dtf * (s as f64 - 1.0) / (2.0 * s as f64 * omega);
        eprintln!("buoyancy s={s}: max |y − y*| {worst:.12} (amplitude estimate {amp:.12})");
        if worst > ROUND {
            failures.push(format!(
                "s={s}: max |y − y*| = {worst:.12} m over 10 s (expected 0; oscillation amplitude g dt (s−1)/(2 s ω) ≈ {amp:.12})"
            ));
        }
    }
    report(&failures);
}

/// A spring applied with `add_force` (`F = −k (y − y_rest)`, the caller's
/// Hooke law) holding a body at its static extension `y_rest − m g / k`,
/// started there at rest.
///
/// - physical expectation: the body stays at the static extension `m g / k`
///   below the rest point for every `s`
/// - current (measured, `m = 1`, `k = 40`): the same oscillation as the
///   buoyancy oracle, bit for bit (it is the same linear restoring force
///   through the other entry point), max `|y − y*|` ≈ `g dt (s − 1)/(2 s ω)`
/// - difference closed form: as for buoyancy, per-frame forcing
///   `g dt² (s − 1)/(2 s)`
#[test]
#[ignore = "src gap: a spring force applied by add_force is a frame-head impulse, so the static extension m g / k is not held (oscillation ~g dt (s-1)/(2 s ω))"]
fn spring_by_add_force_holds_its_static_extension() {
    let mut failures = Vec::new();
    let k_f = 40.0_f64;
    let omega = k_f.sqrt();
    let dtf = 1.0 / DT_DEN as f64;
    for s in SUBSTEPS {
        let k = Fix128::from_int(40);
        let y_rest = Fix128::from_int(2);
        // m = 1 → static extension m g / k = 1/4
        let y_star = y_rest - Fix128::from_ratio(1, 4);
        let (mut w, id) = world(s, Fix128::ONE, y_star);
        let mut worst = 0.0_f64;
        for _ in 0..FRAMES {
            let y = w.bodies[id].position.y;
            let force = Vec3Fix::new(Fix128::ZERO, -(k * (y - y_rest)), Fix128::ZERO);
            w.bodies[id].add_force(force, dt());
            w.step(dt());
            worst = worst.max((f(w.bodies[id].position.y) - f(y_star)).abs());
        }
        let amp = G * dtf * (s as f64 - 1.0) / (2.0 * s as f64 * omega);
        eprintln!("spring s={s}: max |y − y*| {worst:.12} (amplitude estimate {amp:.12})");
        if worst > ROUND {
            failures.push(format!(
                "s={s}: max |y − y*| = {worst:.12} m over 10 s (expected 0; oscillation amplitude g dt (s−1)/(2 s ω) ≈ {amp:.12})"
            ));
        }
    }
    report(&failures);
}
