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
//! - the kinetic energy never grows.
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

/// `(x, v_x)` after `frames` on the serial path, level ground
fn level_run(c: f64, v0: f64, substeps: usize, frames: usize) -> (f64, f64) {
    let (mut w, ball) = level(c, v0, substeps);
    for _ in 0..frames {
        w.step(fx(DT));
    }
    let b = &w.bodies[ball];
    (b.position.x.to_f64(), b.velocity.x.to_f64())
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

/// oracle: rolling at `v₀ = 2` with `c = 0.05`: at `t = 2 s`,
/// `v = v₀ − (5/7) c g t`; after `t_s = 7 v₀ / (5 c g) = 5.6 s` (`g = 10`) the
/// ball has rolled `7 v₀² / (10 c g)` and stays there through `t = 8 s`;
/// the kinetic energy never grows.
#[test]
fn a_rolling_ball_on_level_ground_slows_at_five_sevenths_of_c_g_and_stops() {
    let (c, v0) = (0.05, 2.0);
    let a = 5.0 / 7.0 * c * g();
    let want_v2 = v0 - a * 2.0;
    let want_stop = 7.0 * v0 * v0 / (10.0 * c * g());
    let n = PhysicsConfig::default().substeps;
    let (_, v_h) = level_run(c, v0, n, 120);
    let (_, v_2h) = level_run(c, v0, n / 2, 120);
    let (x_h, _) = level_run(c, v0, n, 480);
    let (x_2h, _) = level_run(c, v0, n / 2, 480);
    let bound_v = 2.0 * (v_h - v_2h).abs() + FLOOR;
    let bound_x = 2.0 * (x_h - x_2h).abs() + FLOOR;
    for path in paths() {
        let (mut w, ball) = level(c, v0, n);
        let mut e_prev = kinetic(&w, ball);
        let mut x_at_7 = 0.0;
        for frame in 1..=480 {
            advance(&mut w, path);
            let e = kinetic(&w, ball);
            assert!(
                e <= e_prev + FLOOR,
                "{path:?} frame {frame}: energy grew {e_prev} -> {e}"
            );
            e_prev = e;
            if frame == 120 {
                let v = w.bodies[ball].velocity.x.to_f64();
                println!("{path:?} t=2: v {v:.6} ({want_v2:.6}, bound {bound_v:.2e})");
                assert!(
                    (v - want_v2).abs() <= bound_v,
                    "{path:?}: v(2) {v}, closed form {want_v2}"
                );
            }
            if frame == 420 {
                x_at_7 = w.bodies[ball].position.x.to_f64();
            }
        }
        let x = w.bodies[ball].position.x.to_f64();
        println!("{path:?} stop: x {x:.6} ({want_stop:.6}, bound {bound_x:.2e})");
        assert!(
            (x - want_stop).abs() <= bound_x,
            "{path:?}: stopped at {x}, closed form {want_stop}"
        );
        assert!(
            (x - x_at_7).abs() <= FLOOR,
            "{path:?}: moved {} m after stopping",
            x - x_at_7
        );
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

/// oracle: on an incline `θ = 0.1` with `c = 0.15` (`> tan θ = 0.1003`), the
/// ball released at rest stays at rest: after 2 s it has moved no more than
/// the settling into its standing penetration (`< 1e-6 m` along the face).
#[test]
fn a_ball_on_an_incline_flatter_than_c_stays_at_rest() {
    let (c, theta) = (0.15, 0.1_f64);
    for path in paths() {
        let (mut w, ball) = incline(c, theta, PhysicsConfig::default().substeps);
        let start = w.bodies[ball].position;
        for _ in 0..120 {
            advance(&mut w, path);
        }
        let d = downhill(&w, ball, start, theta);
        println!("{path:?} held: {d:.3e}");
        assert!(
            d.abs() < 1e-6,
            "{path:?}: crept {d} m down a slope flatter than c"
        );
    }
}
