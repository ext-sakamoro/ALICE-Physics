//! Closed-form checks of rolling friction (rolling resistance): a contact
//! whose material carries a rolling coefficient `c` (dimensionless) resists
//! the relative rolling of the pair with a torque of at most `c r λ_n` about
//! the tangential axis, `r` the lever arm to the contact point.
//!
//! Expected values from elementary mechanics (solid ball `I = 2/5 m r²`,
//! `r = 1/2`, `m = 1`, static ground, default gravity, damping off, sliding
//! friction `μ` large enough that the ball never slips):
//!
//! - rolling on level ground at `v₀`: the resistance torque `M = c r m g`
//!   with the no-slip friction gives `a = −M r / (I + m r²) = −(5/7) c g`; the
//!   ball stops after `t_s = 7 v₀ / (5 c g)` having rolled
//!   `7 v₀² / (10 c g)` and stays stopped (the torque never reverses it);
//! - on an incline of angle `θ` with `c < tan θ`: `a = (5/7) g (sin θ − c cos θ)`;
//! - on an incline with `c > tan θ`, released at rest: it stays at rest;
//! - `c = 0` is the rolling of `analytic_contact_friction_spin.rs`
//!   (`a = (5/7) g sin θ`, a level ball keeps its speed);
//! - the kinetic energy never grows;
//! - torsional friction `μ_t` (metres): a ball spinning about the normal at
//!   `ω₀` with no rolling feels the torque `μ_t m g`, so
//!   `ω(t) = ω₀ − μ_t m g t / I` and it stops at `t_s = I ω₀ / (μ_t m g)`,
//!   then stays stopped; its centre does not move.
//!
//! The tolerances are the Richardson bound `2 |x(h) − x(2h)|` of each
//! observable from the default and half the substeps (the substep is first
//! order in `h`) plus the fixed-point floor `2⁻⁴⁰`.
//!
//! Paths: `step` and, with the `parallel` feature, `step_parallel`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 rolling closed forms and energy sums"
)]

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics_core::PhysicsMaterial;

const DT: f64 = 1.0 / 60.0;
const R: f64 = 0.5;
/// `1 / I` of the solid unit-mass ball
const INV_I: f64 = 1.0 / (0.4 * R * R);
/// sliding friction high enough that the ball never slips in these scenes
const MU: f64 = 0.8;
const FLOOR: f64 = 9.094_947_017_729_282e-13; // 2⁻⁴⁰

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

#[derive(Clone, Copy, Debug)]
enum Path {
    Serial,
    #[cfg(feature = "parallel")]
    Batched,
}

fn paths() -> Vec<Path> {
    #[allow(
        unused_mut,
        reason = "v is only extended under the parallel feature; without it the mut binding is unused"
    )]
    let mut v = vec![Path::Serial];
    #[cfg(feature = "parallel")]
    v.push(Path::Batched);
    v
}

fn advance(w: &mut PhysicsWorld, path: Path) {
    match path {
        Path::Serial => w.step(fx(DT)),
        #[cfg(feature = "parallel")]
        Path::Batched => w.step_parallel(fx(DT)),
    }
}

fn g() -> f64 {
    -PhysicsConfig::default().gravity.y.to_f64()
}

fn material(c: f64) -> PhysicsMaterial {
    PhysicsMaterial::new(0, fx(MU), Fix128::ZERO).with_rolling_friction(fx(c))
}

/// a flat static box (top at `y = 0`) and the ball resting on it, rolling at
/// `v₀` along `+x` (`ω_z r = −v₀`)
fn level(c: f64, v0: f64, substeps: usize) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping_rate: Fix128::ZERO,
        substeps,
        ..PhysicsConfig::default()
    });
    let slab = w.add_body(RigidBody::new_static(v3(0.0, -0.5, 0.0)));
    assert!(w.set_body_shape(
        slab,
        &Shape::Box {
            half_extents: v3(50.0, 0.5, 50.0),
        }
    ));
    let ball = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, R, 0.0), Fix128::ONE), fx(R));
    w.bodies[ball].inv_inertia = v3(INV_I, INV_I, INV_I);
    w.bodies[ball].velocity = v3(v0, 0.0, 0.0);
    w.bodies[ball].angular_velocity = v3(0.0, 0.0, -v0 / R);
    let id = w.material_table.register(material(c));
    w.set_body_material(slab, id);
    w.set_body_material(ball, id);
    (w, ball)
}

/// static box turned by `θ` about `z` and the ball at rest on its face
fn incline(c: f64, theta: f64, substeps: usize) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping_rate: Fix128::ZERO,
        substeps,
        ..PhysicsConfig::default()
    });
    let slab = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.bodies[slab].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(theta)).normalize();
    assert!(w.set_body_shape(
        slab,
        &Shape::Box {
            half_extents: v3(20.0, 0.5, 2.0),
        }
    ));
    let up = 0.5 + R;
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(-theta.sin() * up, theta.cos() * up, 0.0), Fix128::ONE),
        fx(R),
    );
    w.bodies[ball].inv_inertia = v3(INV_I, INV_I, INV_I);
    let id = w.material_table.register(material(c));
    w.set_body_material(slab, id);
    w.set_body_material(ball, id);
    (w, ball)
}

fn kinetic(w: &PhysicsWorld, ball: usize) -> f64 {
    let b = &w.bodies[ball];
    0.5 * b.velocity.length_squared().to_f64()
        + 0.5 * b.angular_velocity.length_squared().to_f64() / INV_I
}

/// downhill displacement after `frames` on the serial path
fn incline_run(c: f64, theta: f64, substeps: usize, frames: usize) -> f64 {
    let (mut w, ball) = incline(c, theta, substeps);
    let start = w.bodies[ball].position;
    for _ in 0..frames {
        w.step(fx(DT));
    }
    downhill(&w, ball, start, theta)
}

fn downhill(w: &PhysicsWorld, ball: usize, start: Vec3Fix, theta: f64) -> f64 {
    let d = w.bodies[ball].position - start;
    -(d.x.to_f64() * theta.cos() + d.y.to_f64() * theta.sin())
}

/// What a level roll measures from `t = 1 s` (after the ball has settled into
/// its standing penetration; the first frame starts at depth 0 and removes
/// less): `(v(2) − v(1) + a, x_stop − (x(1) + v(1)² / 2a), t_stop − (1 + v(1) / a))`,
/// `t_stop` the first frame whose speed is under one frame's decrement `a Δt`
fn level_measure(c: f64, v0: f64, substeps: usize, path: Path) -> (f64, f64, f64) {
    let a = 5.0 / 7.0 * c * g();
    let (mut w, ball) = level(c, v0, substeps);
    let (mut v1, mut x1, mut v2, mut stop) = (0.0, 0.0, 0.0, None);
    for frame in 1..=480 {
        advance(&mut w, path);
        let v = w.bodies[ball].velocity.x.to_f64();
        if frame == 60 {
            (v1, x1) = (v, w.bodies[ball].position.x.to_f64());
        }
        if frame == 120 {
            v2 = v;
        }
        if stop.is_none() && v.abs() < a * DT {
            stop = Some(frame as f64 * DT);
        }
    }
    let x = w.bodies[ball].position.x.to_f64();
    (
        v2 - v1 + a,
        x - (x1 + v1 * v1 / (2.0 * a)),
        stop.expect("the ball stops") - (1.0 + v1 / a),
    )
}

/// oracle: rolling at `v₀ = 2` with `c = 0.05` (`a = (5/7) c g = 0.357`):
/// from `t = 1 s` to `2 s` the speed falls by `a`; the ball stops where
/// `x(1) + v(1)² / 2a` puts it, at `t = 1 + v(1) / a` (within one frame
/// `Δt`, the resolution of the stop), and from the next frame on it does
/// not move (`|v| ≤ 2⁻⁴⁰`, `x` constant); the kinetic energy never grows.
/// Tolerances: the Richardson bound of each measure from the default and
/// half the substeps (`2 |m(h) − m(2h)|`, the substep being first order),
/// plus `2⁻⁴⁰` (one rounding per substep of the window for the rate).
#[test]
fn a_rolling_ball_on_level_ground_slows_at_five_sevenths_of_c_g_and_stops() {
    let (c, v0) = (0.05, 2.0);
    let n = PhysicsConfig::default().substeps;
    let (r1, s1, t1) = level_measure(c, v0, n, Path::Serial);
    let (r2, s2, t2) = level_measure(c, v0, n / 2, Path::Serial);
    // plus one fixed-point rounding (2⁻⁴⁰) per substep of the 1 s window: the
    // angular impulse of each substep is a quotient rounded once
    let bound_rate = 2.0 * (r1 - r2).abs() + 60.0 * n as f64 * FLOOR;
    let bound_stop = 2.0 * (s1 - s2).abs() + FLOOR;
    let bound_t = 2.0 * (t1 - t2).abs() + DT;
    for path in paths() {
        let (rate, stop, t) = level_measure(c, v0, n, path);
        println!(
            "{path:?} level: rate err {rate:.3e} (bound {bound_rate:.2e}) stop err {stop:.3e} (bound {bound_stop:.2e}) t err {t:.3e} (bound {bound_t:.2e})"
        );
        assert!(
            rate.abs() <= bound_rate,
            "{path:?}: deceleration off by {rate}"
        );
        assert!(
            stop.abs() <= bound_stop,
            "{path:?}: stopped {stop} m from the closed form"
        );
        assert!(
            t.abs() <= bound_t,
            "{path:?}: stopped {t} s from the closed form"
        );
        // at rest from the frame after the stop, and no energy gained on the way
        let (mut w, ball) = level(c, v0, n);
        let mut e_prev = kinetic(&w, ball);
        let mut rest: Option<f64> = None;
        for frame in 1..=480 {
            advance(&mut w, path);
            let e = kinetic(&w, ball);
            assert!(
                e <= e_prev + FLOOR,
                "{path:?} frame {frame}: energy grew {e_prev} -> {e}"
            );
            e_prev = e;
            let (x, v) = (
                w.bodies[ball].position.x.to_f64(),
                w.bodies[ball].velocity.x.to_f64(),
            );
            if let Some(x_rest) = rest {
                assert!(
                    v.abs() <= FLOOR && (x - x_rest).abs() <= FLOOR,
                    "{path:?} frame {frame}: moved after stopping (v {v}, dx {})",
                    x - x_rest
                );
            } else if v.abs() <= FLOOR {
                rest = Some(x);
            }
        }
        assert!(rest.is_some(), "{path:?}: never came to rest");
    }
}

/// oracle: on an incline `θ = 0.3` with `c = 0.1` (`< tan θ`): after 1 s the
/// ball has rolled `(5/14) g (sin θ − c cos θ)`.
#[test]
fn a_ball_rolls_down_an_incline_at_five_sevenths_of_g_sin_minus_c_cos() {
    let (c, theta) = (0.1, 0.3_f64);
    let want = 5.0 / 14.0 * g() * (theta.sin() - c * theta.cos());
    let n = PhysicsConfig::default().substeps;
    let bound =
        2.0 * (incline_run(c, theta, n, 60) - incline_run(c, theta, n / 2, 60)).abs() + FLOOR;
    for path in paths() {
        let (mut w, ball) = incline(c, theta, n);
        let start = w.bodies[ball].position;
        for _ in 0..60 {
            advance(&mut w, path);
        }
        let d = downhill(&w, ball, start, theta);
        println!("{path:?} incline c={c}: {d:.6} ({want:.6}, bound {bound:.2e})");
        assert!(
            (d - want).abs() <= bound,
            "{path:?}: rolled {d}, closed form {want}"
        );
    }
}

/// oracle: on an incline `θ = 0.1` with `c = 0.15` (`> tan θ = 0.1003`,
/// the hold condition `sin θ ≤ c cos θ`), the ball released at rest stays at
/// rest: over 2.5 s it moves no more than `2⁻⁴⁰` along the face, at 4, 8, 16
/// and 32 substeps per frame and on every path. Its contrast is
/// `a_ball_rolls_down_an_incline_at_five_sevenths_of_g_sin_minus_c_cos`
/// (`c < tan θ`).
#[test]
fn a_ball_on_an_incline_flatter_than_c_stays_at_rest() {
    let (c, theta) = (0.15, 0.1_f64);
    for n in [4usize, 8, 16, 32] {
        for path in paths() {
            let (mut w, ball) = incline(c, theta, n);
            let start = w.bodies[ball].position;
            for _ in 0..150 {
                advance(&mut w, path);
            }
            let d = downhill(&w, ball, start, theta);
            println!("{path:?} n={n} held: {d:.3e}");
            assert!(
                d.abs() <= FLOOR,
                "{path:?} n={n}: crept {d} m down a slope flatter than c"
            );
        }
    }
}

/// a ball at rest on level ground spinning about the normal at `ω₀`
fn spinning(mu_t: f64, w0: f64, substeps: usize) -> (PhysicsWorld, usize) {
    let (mut w, ball) = level(0.0, 0.0, substeps);
    w.bodies[ball].angular_velocity = v3(0.0, w0, 0.0);
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(MU), Fix128::ZERO).with_torsional_friction(fx(mu_t)));
    w.set_body_material(0, id);
    w.set_body_material(ball, id);
    (w, ball)
}

/// What a spin measures from `t = 1 s`: `(ω(2) − ω(1) + α, t_stop − (1 + ω(1) / α))`
/// with `α = μ_t m g / I`, `t_stop` the first frame whose spin is under one
/// frame's decrement `α Δt`
fn spin_measure(mu_t: f64, w0: f64, substeps: usize, path: Path) -> (f64, f64) {
    let alpha = mu_t * g() * INV_I;
    let (mut w, ball) = spinning(mu_t, w0, substeps);
    let (mut w1, mut w2, mut stop) = (0.0, 0.0, None);
    for frame in 1..=480 {
        advance(&mut w, path);
        let wy = w.bodies[ball].angular_velocity.y.to_f64();
        if frame == 60 {
            w1 = wy;
        }
        if frame == 120 {
            w2 = wy;
        }
        if stop.is_none() && wy.abs() < alpha * DT {
            stop = Some(frame as f64 * DT);
        }
    }
    (
        w2 - w1 + alpha,
        stop.expect("the spin stops") - (1.0 + w1 / alpha),
    )
}

/// oracle: `μ_t = 0.01 m`, `ω₀ = 6`, `I = 0.1`, `m g = 10` (`α = μ_t m g / I
/// = 1 rad/s²`): from `t = 1 s` to `2 s` the spin falls by `α`; it stops at
/// `t = 1 + ω(1) / α` (`I ω₀ / (μ_t m g) = 6 s` from rest, within one frame
/// `Δt`) and stays stopped (`|ω| ≤ 2⁻⁴⁰`). The centre may wander only by the
/// contact point's offset from the point under the centre, which the depth
/// accuracy bounds by EPA's threshold `2⁻¹⁶` (known defect,
/// `analytic_contact_detection_small_penetration.rs`), times the angle the
/// ball turns after `t = 1 s` (`ω(1)² / 2α`). Tolerances otherwise the
/// Richardson bound from the default and half the substeps plus `2⁻⁴⁰` (one
/// rounding per substep of the window for the rate).
#[test]
fn a_spinning_ball_slows_under_torsional_friction_and_stops_at_i_omega_over_mu_t_m_g() {
    let (mu_t, w0) = (0.01, 6.0);
    let alpha = mu_t * g() * INV_I;
    let n = PhysicsConfig::default().substeps;
    let (r1, t1) = spin_measure(mu_t, w0, n, Path::Serial);
    let (r2, t2) = spin_measure(mu_t, w0, n / 2, Path::Serial);
    // plus one fixed-point rounding (2⁻⁴⁰) per substep of the 1 s window: the
    // angular impulse of each substep is a quotient rounded once
    let bound_rate = 2.0 * (r1 - r2).abs() + 60.0 * n as f64 * FLOOR;
    let bound_t = 2.0 * (t1 - t2).abs() + DT;
    for path in paths() {
        let (rate, t) = spin_measure(mu_t, w0, n, path);
        println!("{path:?} spin: rate err {rate:.3e} (bound {bound_rate:.2e}) t err {t:.3e} (bound {bound_t:.2e})");
        assert!(
            rate.abs() <= bound_rate,
            "{path:?}: spin-down off by {rate}"
        );
        assert!(
            t.abs() <= bound_t,
            "{path:?}: stopped {t} s from the closed form"
        );
        let (mut w, ball) = spinning(mu_t, w0, n);
        let (mut p1, mut w1, mut stopped) = (Vec3Fix::ZERO, 0.0, false);
        for frame in 1..=480 {
            advance(&mut w, path);
            let wy = w.bodies[ball].angular_velocity.y.to_f64();
            if frame == 60 {
                (p1, w1) = (w.bodies[ball].position, wy);
            }
            if stopped {
                assert!(
                    wy.abs() <= FLOOR,
                    "{path:?} frame {frame}: spinning again at {wy}"
                );
            } else if frame > 60 && wy.abs() < alpha * DT {
                stopped = true;
            }
        }
        let d = w.bodies[ball].position - p1;
        let wander = 2f64.powi(-16) * w1 * w1 / (2.0 * alpha);
        println!(
            "{path:?} spin: centre moved ({:.2e}, {:.2e}, {:.2e}), bound {wander:.2e}",
            d.x.to_f64(),
            d.y.to_f64(),
            d.z.to_f64()
        );
        assert!(
            d.length().to_f64() <= wander,
            "{path:?}: the centre moved {} m while spinning down",
            d.length().to_f64()
        );
    }
}

/// `(ω_y, turned angle about y)` after 2 s of the ball at rest under a steady
/// torque `τ` about the normal (applied each frame as `τ Δt`)
fn driven(mu_t: f64, tau: f64, substeps: usize, path: Path) -> (f64, f64) {
    let (mut w, ball) = spinning(mu_t, 0.0, substeps);
    for _ in 0..120 {
        w.bodies[ball].add_torque(v3(0.0, tau, 0.0), fx(DT));
        advance(&mut w, path);
    }
    let b = &w.bodies[ball];
    (
        b.angular_velocity.y.to_f64(),
        2.0 * b.rotation.y.to_f64().abs(),
    )
}

/// oracle: `add_torque` gives the frame's whole impulse `τ Δt` before the
/// step, and the contact can resist at most `μ_t λ` per substep (`λ = m g h²`,
/// so an impulse `μ_t m g h`), so static torsional friction holds the ball
/// when `τ Δt ≤ μ_t m g h`, i.e. `τ ≤ μ_t m g / n` for `n` substeps: at half
/// that (`τ = μ_t m g / 2n`) the ball does not turn at all (the angle after 2 s
/// within one rounding `2⁻⁴⁰` per substep). Above the kinetic cap
/// (`τ = 2 μ_t m g`) the ball spins up at `(τ − μ_t m g) / I`, within the
/// Richardson bound from the default and half the substeps plus one rounding
/// per substep.
#[test]
fn a_torque_below_mu_t_m_g_holds_the_ball_and_above_it_spins_it_up() {
    let mu_t = 0.01;
    let cap = mu_t * g();
    let n = PhysicsConfig::default().substeps;
    let rounding = 120.0 * n as f64 * FLOOR;
    let want = (2.0 * cap - cap) * INV_I * 2.0;
    let (w1, _) = driven(mu_t, 2.0 * cap, n, Path::Serial);
    let (w2, _) = driven(mu_t, 2.0 * cap, n / 2, Path::Serial);
    let bound = 2.0 * (w1 - w2).abs() + rounding;
    for path in paths() {
        let (wy, angle) = driven(mu_t, 0.5 * cap / n as f64, n, path);
        println!("{path:?} held torque: ω {wy:.3e} angle {angle:.3e} (bound {rounding:.2e})");
        assert!(
            angle <= rounding && wy.abs() <= rounding,
            "{path:?}: turned {angle} rad under a torque below the cap"
        );
        let (wy, _) = driven(mu_t, 2.0 * cap, n, path);
        println!("{path:?} driven: ω {wy:.6} ({want:.6}, bound {bound:.2e})");
        assert!(
            (wy - want).abs() <= bound,
            "{path:?}: spun up to {wy}, closed form {want}"
        );
    }
}
