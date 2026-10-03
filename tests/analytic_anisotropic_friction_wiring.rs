//! Oracles for the production entry points of
//! `alice_physics::anisotropic_friction` driven by
//! `examples/anisotropic_friction_presets.rs`:
//! `AnisotropicFriction::{tyre_asphalt, ski_snow, skate_ice}` and
//! `AnisotropicFriction::friction_force`.
//!
//! # What this file is and is not
//!
//! `examples/anisotropic_friction_presets.rs` already drives
//! `friction_force` for all three presets at the two on-axis slip
//! directions (purely longitudinal, purely transverse), where the module's
//! `v_long.is_zero()` / `v_trans.is_zero()` exact-magnitude branches apply
//! and the slip speed collapses to `|v_long|` or `|v_trans|` with no
//! `sqrt`. What that file does **not** cover, and this file does:
//!
//! * a genuinely off-axis slip direction (45 deg from the longitudinal
//!   axis) for every preset, which is the only input shape that exercises
//!   `friction_force`'s general `sqrt(v_long^2 + v_trans^2)` branch rather
//!   than either exact-magnitude shortcut,
//! * zero relative velocity, which must report zero friction force (not
//!   panic, and not divide by a zero slip speed), independent of how large
//!   the normal force is, and
//! * an extreme-magnitude relative velocity (1e15 m/s, 15 orders of
//!   magnitude past every preset's `slip_threshold_m_s`), confirming the
//!   friction-ellipse closed form still holds -- and nothing panics or
//!   silently diverges -- well short of `Fix128`'s own wrapping boundary
//!   (`src/math.rs`'s `impl Mul for Fix128` doc: `Fix128` multiplication
//!   never panics, even on overflow, but this magnitude does not reach
//!   that boundary; the closed form below is checked exactly, not just
//!   "did not panic").
//!
//! # Degenerate / extreme input summary
//!
//! * Zero relative velocity: `friction_force` returns `Vec3Fix::ZERO`
//!   exactly, for every preset and regardless of the normal force
//!   magnitude (checked at both a small and a very large normal force).
//! * Extreme-magnitude relative velocity (1e15 m/s, longitudinal axis):
//!   the friction-ellipse closed form `F = -N * mu_long_kinetic * t_long`
//!   still holds to the same tolerance as at ordinary (10 m/s) magnitude --
//!   the `v_long / slip` ratio in `friction_force` is scale-invariant on a
//!   pure axis, so nothing about the division degrades at this magnitude.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::anisotropic_friction::AnisotropicFriction;
use alice_physics::math::{Fix128, Vec3Fix};

/// Relative-error assertion against an independently hand-derived f64
/// closed form (never computed by calling `friction_force` itself).
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = ((g - want) / want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (rel err {err:.3e} > {tol:.1e})"
    );
}

/// Absolute-error assertion against an independently hand-derived f64
/// closed form, for expectations that are exactly zero (where a relative
/// error is undefined).
fn assert_abs(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (abs err {err:.3e} > {tol:.1e})"
    );
}

const FIX_TOL: f64 = 1e-9;

/// The longitudinal / transverse tangent frame used by every test below,
/// matching `src/anisotropic_friction.rs`'s own `#[cfg(test)]::axes()`
/// helper: `t_long` is the X axis, `t_trans` is the Z axis.
fn axes() -> (Vec3Fix, Vec3Fix) {
    (
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
    )
}

/// A preset's kinetic coefficients, written out again by hand from
/// `src/anisotropic_friction.rs`'s doc comments -- the independent
/// reference, not a re-read of the live struct fields. `tyre_asphalt` and
/// `ski_snow` share a 0.05 m/s `slip_threshold_m_s`; `skate_ice` uses 0.02.
/// Every scenario below uses a 10 m/s slip speed (or 1e15 m/s for the
/// extreme-magnitude case), which clears all three thresholds, so every
/// case below selects the kinetic pair, never static.
struct Preset {
    name: &'static str,
    friction: fn() -> AnisotropicFriction,
    mu_long_kinetic: f64,
    mu_trans_kinetic: f64,
}

const PRESETS: [Preset; 3] = [
    Preset {
        name: "tyre_asphalt",
        friction: AnisotropicFriction::tyre_asphalt,
        mu_long_kinetic: 0.9,
        mu_trans_kinetic: 0.7,
    },
    Preset {
        name: "ski_snow",
        friction: AnisotropicFriction::ski_snow,
        mu_long_kinetic: 0.04,
        mu_trans_kinetic: 0.75,
    },
    Preset {
        name: "skate_ice",
        friction: AnisotropicFriction::skate_ice,
        mu_long_kinetic: 0.015,
        mu_trans_kinetic: 0.7,
    },
];

/// Friction-ellipse closed form (module doc, `src/anisotropic_friction.rs`)
/// at slip angle `theta` from the longitudinal axis:
/// `Fx = -N*mu_long*cos(theta)`, `Fz = -N*mu_trans*sin(theta)`, `Fy = 0`.
/// Takes `cos_theta`/`sin_theta` directly rather than `theta` itself so
/// callers supply exact algebraic values (0 deg, 90 deg, the `sqrt(2)/2`
/// 45 deg case) without this helper calling a trigonometric method.
fn ellipse_force_f64(
    normal_force: f64,
    mu_long: f64,
    mu_trans: f64,
    cos_theta: f64,
    sin_theta: f64,
) -> (f64, f64) {
    (
        -normal_force * mu_long * cos_theta,
        -normal_force * mu_trans * sin_theta,
    )
}

/// Drives `friction_force` for one preset at 0 deg (pure longitudinal
/// slip), 45 deg (general, off-axis), and 90 deg (pure transverse slip)
/// relative to the longitudinal axis, each checked against
/// `ellipse_force_f64`. The 0 deg / 90 deg legs hit `friction_force`'s
/// exact-magnitude shortcuts (already also covered on-axis by
/// `examples/anisotropic_friction_presets.rs`); 45 deg is the only angle
/// here that reaches the general `sqrt` branch.
fn check_ellipse_at_0_45_90_degrees(preset: &Preset) {
    let (t_long, t_trans) = axes();
    let friction = (preset.friction)();
    let normal_force = Fix128::from_int(100);
    let normal_force_f64 = 100.0_f64;
    let speed = 10.0_f64;

    // 0 degrees: v_trans = 0, exact-magnitude shortcut.
    let v0 = Vec3Fix::new(Fix128::from_f64(speed), Fix128::ZERO, Fix128::ZERO);
    let f0 = friction.friction_force(normal_force, t_long, t_trans, v0);
    let (want_x0, want_z0) = ellipse_force_f64(
        normal_force_f64,
        preset.mu_long_kinetic,
        preset.mu_trans_kinetic,
        1.0,
        0.0,
    );
    assert_rel(f0.x, want_x0, FIX_TOL, &format!("{}: 0deg Fx", preset.name));
    assert_abs(f0.y, 0.0, FIX_TOL, &format!("{}: 0deg Fy", preset.name));
    assert_abs(f0.z, want_z0, FIX_TOL, &format!("{}: 0deg Fz", preset.name));

    // 45 degrees: v_long == v_trans, the only angle here that reaches the
    // general sqrt(v_long^2 + v_trans^2) branch rather than an
    // exact-magnitude shortcut.
    let s45 = 2.0_f64.sqrt() / 2.0;
    let v45 = Vec3Fix::new(
        Fix128::from_f64(speed * s45),
        Fix128::ZERO,
        Fix128::from_f64(speed * s45),
    );
    let f45 = friction.friction_force(normal_force, t_long, t_trans, v45);
    let (want_x45, want_z45) = ellipse_force_f64(
        normal_force_f64,
        preset.mu_long_kinetic,
        preset.mu_trans_kinetic,
        s45,
        s45,
    );
    assert_rel(
        f45.x,
        want_x45,
        FIX_TOL,
        &format!("{}: 45deg Fx", preset.name),
    );
    assert_abs(f45.y, 0.0, FIX_TOL, &format!("{}: 45deg Fy", preset.name));
    assert_rel(
        f45.z,
        want_z45,
        FIX_TOL,
        &format!("{}: 45deg Fz", preset.name),
    );

    // 90 degrees: v_long = 0, exact-magnitude shortcut on the other axis.
    let v90 = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_f64(speed));
    let f90 = friction.friction_force(normal_force, t_long, t_trans, v90);
    let (want_x90, want_z90) = ellipse_force_f64(
        normal_force_f64,
        preset.mu_long_kinetic,
        preset.mu_trans_kinetic,
        0.0,
        1.0,
    );
    assert_abs(
        f90.x,
        want_x90,
        FIX_TOL,
        &format!("{}: 90deg Fx", preset.name),
    );
    assert_abs(f90.y, 0.0, FIX_TOL, &format!("{}: 90deg Fy", preset.name));
    assert_rel(
        f90.z,
        want_z90,
        FIX_TOL,
        &format!("{}: 90deg Fz", preset.name),
    );
}

/// Degenerate: zero relative velocity must report zero friction force for
/// every preset -- `friction_force`'s `slip.is_zero()` early return, not a
/// division by a zero slip speed -- regardless of the normal force
/// magnitude (checked at a small and a very large normal force).
fn check_zero_velocity_yields_zero_force(preset: &Preset) {
    let (t_long, t_trans) = axes();
    let friction = (preset.friction)();
    for normal_force in [Fix128::from_int(1), Fix128::from_int(1_000_000_000)] {
        let f = friction.friction_force(normal_force, t_long, t_trans, Vec3Fix::ZERO);
        assert_eq!(
            f,
            Vec3Fix::ZERO,
            "{}: zero relative velocity at normal_force={normal_force:?} must yield exactly zero force",
            preset.name
        );
    }
}

/// Extreme magnitude: a 1e15 m/s longitudinal slip speed (15 orders of
/// magnitude past every preset's `slip_threshold_m_s`), well short of
/// `Fix128`'s own wrapping boundary (`src/math.rs`'s `impl Mul for Fix128`
/// doc: no panic even past that boundary, but this case does not reach
/// it). The exact-magnitude `v_long.is_zero() == false, v_trans.is_zero()
/// == true` branch makes the `v_long / slip` ratio exactly 1 regardless of
/// `v_long`'s scale, so the closed form `F = -N*mu_long_kinetic*t_long`
/// holds to the same `FIX_TOL` as at ordinary (10 m/s) magnitude.
fn check_extreme_magnitude_velocity_matches_closed_form(preset: &Preset) {
    let (t_long, t_trans) = axes();
    let friction = (preset.friction)();
    let normal_force = Fix128::from_int(100);
    let normal_force_f64 = 100.0_f64;
    let extreme_speed = 1.0e15_f64;

    let v = Vec3Fix::new(Fix128::from_f64(extreme_speed), Fix128::ZERO, Fix128::ZERO);
    let f = friction.friction_force(normal_force, t_long, t_trans, v);
    let want_x = -normal_force_f64 * preset.mu_long_kinetic;
    assert_rel(
        f.x,
        want_x,
        FIX_TOL,
        &format!("{}: extreme-magnitude Fx", preset.name),
    );
    assert_abs(
        f.y,
        0.0,
        FIX_TOL,
        &format!("{}: extreme-magnitude Fy", preset.name),
    );
    assert_abs(
        f.z,
        0.0,
        FIX_TOL,
        &format!("{}: extreme-magnitude Fz", preset.name),
    );
}

#[test]
fn tyre_asphalt_matches_friction_ellipse_at_0_45_90_degrees() {
    check_ellipse_at_0_45_90_degrees(&PRESETS[0]);
}

#[test]
fn tyre_asphalt_zero_velocity_yields_zero_force() {
    check_zero_velocity_yields_zero_force(&PRESETS[0]);
}

#[test]
fn tyre_asphalt_extreme_magnitude_velocity_matches_closed_form() {
    check_extreme_magnitude_velocity_matches_closed_form(&PRESETS[0]);
}

#[test]
fn ski_snow_matches_friction_ellipse_at_0_45_90_degrees() {
    check_ellipse_at_0_45_90_degrees(&PRESETS[1]);
}

#[test]
fn ski_snow_zero_velocity_yields_zero_force() {
    check_zero_velocity_yields_zero_force(&PRESETS[1]);
}

#[test]
fn ski_snow_extreme_magnitude_velocity_matches_closed_form() {
    check_extreme_magnitude_velocity_matches_closed_form(&PRESETS[1]);
}

#[test]
fn skate_ice_matches_friction_ellipse_at_0_45_90_degrees() {
    check_ellipse_at_0_45_90_degrees(&PRESETS[2]);
}

#[test]
fn skate_ice_zero_velocity_yields_zero_force() {
    check_zero_velocity_yields_zero_force(&PRESETS[2]);
}

#[test]
fn skate_ice_extreme_magnitude_velocity_matches_closed_form() {
    check_extreme_magnitude_velocity_matches_closed_form(&PRESETS[2]);
}
