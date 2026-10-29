//! Closed forms of motion under a linear damping rate `β` (1/s) written so
//! that they stay exact in the undamped limit `β → 0` (the default rate is 0).
//!
//! The textbook forms divide by `β` (`(1 − e^{−βt}) / β`, `(βt − 1 + e^{−βt}) / β²`)
//! and are `0/0` at `β = 0`. Here they are written with the entire functions
//!
//! * `φ₁(z) = (1 − e^{−z}) / z = Σ_{k≥0} (−z)^k / (k+1)!`, `φ₁(0) = 1`;
//! * `φ₂(z) = (z − 1 + e^{−z}) / z² = Σ_{k≥0} (−z)^k / (k+2)!`, `φ₂(0) = 1/2`;
//!
//! so that, for a body that starts at `v₀` and feels a constant acceleration
//! `−a` plus the damping `−β v`,
//!
//! * `v(t) = v₀ e^{−βt} − a t φ₁(βt)` (→ `v₀ − a t` as `β → 0`),
//! * `x(t) − x₀ = v₀ t φ₁(βt) − a t² φ₂(βt)` (→ `v₀ t − a t²/2`).
//!
//! **Threshold `Z = 0.1`, derived.** Below `Z` both functions are evaluated
//! by their series truncated after `z⁶`: the series alternate with terms
//! decreasing in magnitude for `z < 1`, so the truncation error is below the
//! first omitted term, `z⁷/8! ≤ 2.5e-12` for `φ₁` and `z⁷/9! ≤ 2.8e-13` for
//! `φ₂`. At and above `Z` they are evaluated directly in f64: `1 − e^{−z}`
//! and `z − 1 + e^{−z}` carry an absolute rounding error of about 2 ulp of 1
//! (`≲ 4.4e-16`); both numerators increase with `z`, so the relative error is
//! largest at `z = Z`: `4.4e-16 / (1 − e^{−0.1}) ≈ 4.6e-15` for `φ₁` and
//! `4.4e-16 / (0.1 − 1 + e^{−0.1}) ≈ 9.2e-14` for `φ₂`.
//! Both branches are therefore within `3e-12` relative of the exact value,
//! far below every tolerance these closed forms are compared with (`≥ 1e-6`).
//! There is no `β == 0` special case: `β = 0` is just `z = 0` in the series.
//!
//! Author: Moroya Sakamoto

#![allow(
    dead_code,
    reason = "shared by several oracle binaries; each one uses a subset"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 exp/ln of the damped closed forms phi1, phi2 and the engine-matched rate"
)]

use alice_physics_core::math::Fix128;

/// Below this `z` the series is used (see the module doc for the derivation).
const SERIES_BELOW: f64 = 0.1;

/// `φ₁(z) = (1 − e^{−z}) / z`, `φ₁(0) = 1`, for `z ≥ 0`.
pub fn phi1(z: f64) -> f64 {
    if z < SERIES_BELOW {
        // 1 − z/2 + z²/6 − z³/24 + z⁴/120 − z⁵/720 + z⁶/5040
        let mut sum = 0.0;
        let mut term = 1.0; // (−z)^k / (k+1)!
        for k in 0..7 {
            sum += term;
            term *= -z / f64::from(k + 2);
        }
        sum
    } else {
        (1.0 - (-z).exp()) / z
    }
}

/// `φ₂(z) = (z − 1 + e^{−z}) / z²`, `φ₂(0) = 1/2`, for `z ≥ 0`.
pub fn phi2(z: f64) -> f64 {
    if z < SERIES_BELOW {
        // 1/2 − z/6 + z²/24 − z³/120 + z⁴/720 − z⁵/5040 + z⁶/40320
        let mut sum = 0.0;
        let mut term = 0.5; // (−z)^k / (k+2)!
        for k in 0..7 {
            sum += term;
            term *= -z / f64::from(k + 3);
        }
        sum
    } else {
        (z - 1.0 + (-z).exp()) / (z * z)
    }
}

/// `v(t) = v₀ e^{−βt} − a t φ₁(βt)`: velocity under a constant deceleration
/// `a` (opposing `v₀`) and the damping rate `β`.
pub fn damped_velocity(v0: f64, a: f64, beta: f64, t: f64) -> f64 {
    v0 * (-beta * t).exp() - a * t * phi1(beta * t)
}

/// `a t² φ₂(βt)`: displacement from rest under a constant acceleration `a`
/// and the damping rate `β` (`a t²/2` at `β = 0`).
pub fn damped_displacement(a: f64, beta: f64, t: f64) -> f64 {
    a * t * t * phi2(beta * t)
}

/// `Σ_{k<n} dᵏ`, summed term by term (exact limit `n` at `d = 1`, no
/// division by `1 − d`).
pub fn geometric_sum(d: f64, n: usize) -> f64 {
    let mut sum = 0.0;
    let mut p = 1.0;
    for _ in 0..n {
        sum += p;
        p *= d;
    }
    sum
}

/// The per-step velocity factor exactly as `PhysicsWorld::step` computes it
/// for a summed rate `c` and step `dt`: `exp(−c·dt)` in `Fix128`
/// (`Fix128::exp(0)` is exactly 1: `powf_pos` with exponent 0 multiplies
/// nothing).
pub fn engine_retention(c: Fix128, dt: Fix128) -> Fix128 {
    (-(c * dt)).exp()
}

/// The continuous rate `β = −ln(d)/dt` whose `e^{−β·dt}` equals the engine's
/// per-step factor `d` ([`engine_retention`]) to f64 precision, so a
/// continuous closed form in `β` carries no `Fix128::exp` error (`≲ 1e-6`
/// relative per step) on top of its own discretisation budget. `0` at rate 0
/// (`d = 1` exactly). Needs `f64::ln` (callers allow `disallowed_methods`).
pub fn engine_beta(c: Fix128, dt: f64) -> f64 {
    -engine_retention(c, Fix128::from_f64(dt)).to_f64().ln() / dt
}

/// The pre-rate default as a rate: a velocity retention of `0.99` per step
/// of `1/60 s`, i.e. `c = −ln(0.99) · 60 ≈ 0.603 / s` (Fix128 `ln`).
pub fn old_default_rate() -> Fix128 {
    -Fix128::from_ratio(99, 100).ln() * Fix128::from_int(60)
}

/// The two rates every damping-dependent oracle runs at: 0 (the default, the
/// undamped limit) and [`old_default_rate`] (the damped branch).
pub fn both_rates() -> [Fix128; 2] {
    [Fix128::ZERO, old_default_rate()]
}
