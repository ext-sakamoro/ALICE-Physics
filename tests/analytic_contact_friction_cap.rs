//! Oracles for Coulomb friction in the XPBD velocity pass: the tangential
//! velocity change of a contact in a substep is capped by `μ` times the
//! normal impulse of that substep's position solve (Müller, Macklin,
//! Chentanez, Jeschke, Kim, "Detailed Rigid Body Simulation with Extended
//! Position Based Dynamics", SCA 2020, eq. (30)), so a sliding body
//! decelerates at `μ g` and a body on a slope above the friction angle
//! slides at the kinetic rate.
//!
//! Every scene runs through `PhysicsWorld::step` and, with the `parallel`
//! feature, `PhysicsWorld::step_parallel`, with `PhysicsConfig::default()`
//! except for the gravity direction of the slope scenes. Expected values are
//! closed forms, never obtained by running the solver:
//!
//! * frame damping: `step` multiplies every dynamic body's velocity by
//!   `d = exp(−c·dt)` once per frame of `dt` (`c = config.damping_rate`), i.e.
//!   a decay rate `β = −ln(d) / dt` per second in the continuous limit; `β`
//!   is taken from the engine's own `Fix128` factor `d` (`engine_beta`), so
//!   `Fix128::exp`'s error does not enter. Every oracle runs at `c = 0` (the
//!   default) and at `c = −ln(0.99)·60` (the old per-step retention 0.99 at
//!   `dt = 1/60` as a rate);
//! * a body that cannot rotate sliding under kinetic friction `μ`:
//!   `v(t) = (v₀ + μg/β) e^{−βt} − μg/β = v₀ e^{−βt} − μg t φ₁(βt)` while
//!   `v > 0`, with `φ₁(z) = (1 − e^{−z})/z`, `φ₁(0) = 1`, so the undamped limit
//!   is exactly `v₀ − μg t` (`tests/support/damped_limits.rs` derives the
//!   series threshold);
//! * a body on a slope of angle `θ` with `tan θ > μ`, from rest:
//!   `a = g (sin θ − μ cos θ)` and
//!   `x(t) = (a/β) t − (a/β²)(1 − e^{−βt}) = a t² φ₂(βt)`, `φ₂(0) = 1/2`, so
//!   the undamped limit is exactly `a t²/2`;
//!   holding below the friction angle is covered by
//!   `tests/analytic_contact_static_friction.rs`.
//!
//! The slope is modelled by tilting the gravity by `θ` against a level ground,
//! which is the same problem in the ground's frame.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 damped sliding velocity (v0 + mu g/beta) e^(-beta t) - mu g/beta, Mueller et al. XPBD 2020 friction cap"
)]

use alice_physics_core::collider::Contact;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{ContactModifier, PhysicsConfig, PhysicsWorld, RigidBody};

#[path = "support/damped_limits.rs"]
mod damped_limits;
use damped_limits::{both_rates, damped_displacement, damped_velocity, engine_beta};

const DT: f64 = 1.0 / 60.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// The two entry points under test.
#[derive(Clone, Copy, Debug)]
enum Path {
    Serial,
    #[cfg(feature = "parallel")]
    Batched,
}

fn paths() -> Vec<Path> {
    #[cfg(feature = "parallel")]
    {
        vec![Path::Serial, Path::Batched]
    }
    #[cfg(not(feature = "parallel"))]
    {
        vec![Path::Serial]
    }
}

/// Gravity magnitude of `PhysicsConfig::default()`.
fn g() -> f64 {
    -PhysicsConfig::default().gravity.y.to_f64()
}

/// `v₀ e^{−βt} − μg t φ₁(βt)` with `β` matched to the engine's factor for
/// the damping rate `rate`.
fn sliding_velocity(v0: f64, mu: f64, rate: Fix128, t: f64) -> f64 {
    damped_velocity(v0, mu * g(), engine_beta(rate, DT), t)
}

/// Sets every contact's friction to an absolute value.
struct AbsoluteFriction(Fix128);

impl ContactModifier for AbsoluteFriction {
    fn modify_contact(
        &self,
        _a: usize,
        _b: usize,
        _contact: &mut Contact,
        friction: &mut Fix128,
        _restitution: &mut Fix128,
    ) -> bool {
        *friction = self.0;
        true
    }
}

/// Multiplies every contact's friction by a factor.
struct ScaleFriction(Fix128);

impl ContactModifier for ScaleFriction {
    fn modify_contact(
        &self,
        _a: usize,
        _b: usize,
        _contact: &mut Contact,
        friction: &mut Fix128,
        _restitution: &mut Fix128,
    ) -> bool {
        *friction = *friction * self.0;
        true
    }
}

/// A ball of radius `0.5` that cannot rotate, touching a large static sphere
/// whose top is the plane `y = 0` (radius `1e4`: over the metres covered the
/// surface tilts by under `3e-4 rad`), both with a material of friction `mu`
/// and restitution `0`. Gravity is `|g|` tilted by `theta` towards `+x`, the
/// world damping rate is `rate`. The ball starts at `v0` along `x`; returns
/// the ball's `x` position and velocity after `frames` steps.
fn run(
    mu: f64,
    theta: f64,
    v0: f64,
    frames: usize,
    rate: Fix128,
    modifier: Option<Box<dyn ContactModifier>>,
    path: Path,
) -> (f64, f64) {
    let g = g();
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: v3(g * theta.sin(), -g * theta.cos(), 0.0),
        damping_rate: rate,
        ..PhysicsConfig::default()
    });
    let big = 1.0e4;
    let ground = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -big, 0.0)), fx(big));
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE),
        fx(0.5),
    );
    w.bodies[ball].inv_inertia = Vec3Fix::ZERO;
    w.bodies[ball].velocity = v3(v0, 0.0, 0.0);
    let id = w
        .material_table
        .register(alice_physics_core::PhysicsMaterial::new(
            0,
            fx(mu),
            Fix128::ZERO,
        ));
    w.set_body_material(ground, id);
    w.set_body_material(ball, id);
    if let Some(m) = modifier {
        w.add_contact_modifier(m);
    }
    for _ in 0..frames {
        match path {
            Path::Serial => w.step(fx(DT)),
            #[cfg(feature = "parallel")]
            Path::Batched => w.step_parallel(fx(DT)),
        }
    }
    let b = w.get_body(ball).expect("ball");
    (b.position.x.to_f64(), b.velocity.x.to_f64())
}

/// Budget for the sliding tolerance `3e-2` absolute: the friction impulse of
/// a substep is `μ` times the normal impulse `m g h` up to the surface tilt
/// (`< 3e-4` relative); per-frame rather than continuous damping changes `v`
/// by `O(β·dt)` (`1e-2`) of the damping term `μg/β (1 − e^{−βt})`
/// (`≤ 2.5` here, so `≤ 2.5e-2`). At rate 0 the damping term is 0 and the
/// budget is the friction-impulse part alone. `β` is matched to the engine's
/// `Fix128` factor (`engine_beta`), so `Fix128::exp`'s error adds nothing.
const SLIDE_TOL: f64 = 3e-2;

/// `(μ, v₀, frames)` of the sliding scenes, chosen so the ball is still
/// sliding at the end (`v > 0.5`).
const SLIDES: [(f64, f64, usize); 2] = [(0.1, 3.0, 60), (0.4, 5.0, 30)];

/// A ball sliding on level ground under the material friction `μ` follows
/// `v(t) = v₀ e^{−βt} − μg t φ₁(βt)` (`v₀ − μg t` undamped), at rate 0 and
/// at the old 0.99-per-step loss as a rate.
#[test]
fn a_sliding_ball_decelerates_at_mu_g() {
    for rate in both_rates() {
        for path in paths() {
            for (mu, v0, frames) in SLIDES {
                let want = sliding_velocity(v0, mu, rate, frames as f64 * DT);
                assert!(want > 0.5);
                let (_, v) = run(mu, 0.0, v0, frames, rate, None, path);
                assert!(
                    (v - want).abs() < SLIDE_TOL,
                    "{path:?} rate {rate:?} μ {mu}: v = {v:.5}, closed form {want:.5}"
                );
            }
        }
    }
}

/// A modifier that sets `μ = 0.1` over a material of `μ = 0.5` gives the
/// sliding closed form of `μ = 0.1`, at both rates.
#[test]
fn an_absolute_friction_modifier_sets_the_sliding_deceleration() {
    for rate in both_rates() {
        let want = sliding_velocity(3.0, 0.1, rate, 1.0);
        for path in paths() {
            let m = AbsoluteFriction(fx(0.1));
            let (_, v) = run(0.5, 0.0, 3.0, 60, rate, Some(Box::new(m)), path);
            assert!(
                (v - want).abs() < SLIDE_TOL,
                "{path:?} rate {rate:?}: v = {v:.5}, closed form {want:.5}"
            );
        }
    }
}

/// A modifier `friction *= 1/2` over a material of `μ = 0.4` gives the
/// sliding closed form of `μ_eff = 0.2`, at both rates.
#[test]
fn a_relative_friction_modifier_halves_the_sliding_deceleration() {
    for rate in both_rates() {
        let want = sliding_velocity(5.0, 0.2, rate, 0.5);
        for path in paths() {
            let m = ScaleFriction(Fix128::from_ratio(1, 2));
            let (_, v) = run(0.4, 0.0, 5.0, 30, rate, Some(Box::new(m)), path);
            assert!(
                (v - want).abs() < SLIDE_TOL,
                "{path:?} rate {rate:?}: v = {v:.5}, closed form {want:.5}"
            );
        }
    }
}

/// On a slope with `tan θ = 0.6 > μ = 0.2` the ball slides down with
/// `a = g (sin θ − μ cos θ)` from rest: `x(t) = a t² φ₂(βt)` (`a t²/2`
/// undamped) after one second, at both rates. Tolerance 2 % of
/// `x ≈ 2.0 m` (damped) / `2.1 m` (undamped): the symplectic sub-step
/// integration offsets `x` by `a t h / 2` (`≈ 2e-3`), per-frame damping by
/// `O(β·dt)` of the damping term (0 at rate 0), and the surface tilt by
/// `< 3e-4`.
#[test]
fn a_ball_on_a_slope_above_the_friction_angle_slides_at_the_kinetic_rate() {
    let g = g();
    let theta = 0.6f64.atan();
    let mu = 0.2;
    let a = g * (theta.sin() - mu * theta.cos());
    for rate in both_rates() {
        let want = damped_displacement(a, engine_beta(rate, DT), 1.0);
        for path in paths() {
            let (x, _) = run(mu, theta, 0.0, 60, rate, None, path);
            assert!(
                (x - want).abs() < 2e-2 * want,
                "{path:?} rate {rate:?}: x = {x:.5}, closed form {want:.5}"
            );
        }
    }
}
