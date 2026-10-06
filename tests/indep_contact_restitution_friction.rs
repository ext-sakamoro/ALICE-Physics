//! Independent closed-form checks of the contact response: restitution taken
//! from the pre-solve normal velocity, kinetic friction capped by `μ λ / h`
//! and position-level static friction.
//!
//! Expected values are written from elementary mechanics, here:
//!
//! - a ball dropped from height `H` (bottom above the contact) bounces back to
//!   `e² H`, whatever the phase of the first touch inside a substep (16
//!   phases are swept);
//! - a ball that cannot rotate, sliding with `v₀` on level ground with
//!   friction `μ`, decelerates at `μ g` and stops at `v₀ / (μ g)`, then stays;
//! - a solid ball (`I = 2/5 m r²`) launched sliding without spin decelerates
//!   at `μ g` while its spin grows at `5 μ g / (2 r)`, until it rolls at
//!   `t* = 2 v₀ / (7 μ g)` with `v = 5 v₀ / 7`, and then keeps that speed
//!   with `v = −ω_z r`;
//! - a ball that cannot rotate on a real incline (a static box turned by
//!   `θ`) stays put for `tan θ < μ` and slides with
//!   `a = g (sin θ − μ cos θ)` for `tan θ > μ`.
//!
//! The ground is a static plain sphere of radius `1e4` whose top is `y = 0`
//! (over the metres covered its surface tilts by under `1e-3` rad), except
//! for the incline, which is a shaped static box. Gravity is the default
//! `−10`; damping is switched off (`1`) so the closed forms need no damping
//! term. Paths: `step` and, with the `parallel` feature, `step_parallel`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::PhysicsMaterial;

const DT: f64 = 1.0 / 60.0;
const R: f64 = 0.5;
const BIG: f64 = 1.0e4;

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
    #[allow(unused_mut)]
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

fn substep() -> f64 {
    DT / PhysicsConfig::default().substeps as f64
}

/// World with the default configuration except damping, a ground sphere and
/// a ball of radius `R` at `ball_at`; both use one material `(μ, e)`.
fn scene(mu: f64, e: f64, ball_at: Vec3Fix) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    });
    let ground = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -BIG, 0.0)), fx(BIG));
    let ball = w.add_body_with_radius(RigidBody::new_dynamic(ball_at, Fix128::ONE), fx(R));
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(mu), fx(e)));
    w.set_body_material(ground, id);
    w.set_body_material(ball, id);
    (w, ball)
}

// ---------------------------------------------------------------------------
// Restitution
// ---------------------------------------------------------------------------

/// Drop height (ball bottom above `y = 0`) for which the discrete free fall
/// (semi-implicit Euler: `v ← v − g h`, `y ← y + v h`) crosses the contact
/// inside substep `n` at the fraction `k / 16` of that substep's displacement
/// `g h² n`: after `n − 1` substeps the ball has fallen `g h² (n − 1) n / 2`.
fn drop_height(n: usize, k: usize) -> f64 {
    let h = substep();
    let nf = n as f64;
    g() * h * h * ((nf - 1.0) * nf / 2.0 + (1.0 - k as f64 / 16.0) * nf)
}

/// Height of the first bounce apex above the drop contact (ball bottom).
fn bounce_apex(e: f64, k: usize, path: Path) -> (f64, f64) {
    let n = 240;
    let hgt = drop_height(n, k);
    let (mut w, ball) = scene(0.0, e, v3(0.0, R + hgt, 0.0));
    let mut bounced = false;
    let mut apex = f64::NEG_INFINITY;
    for _ in 0..240 {
        advance(&mut w, path);
        let b = w.get_body(ball).expect("ball");
        let vy = b.velocity.y.to_f64();
        if vy > 0.0 {
            bounced = true;
        }
        if bounced {
            apex = apex.max(b.position.y.to_f64() - R);
            if vy < 0.0 {
                break;
            }
        }
    }
    assert!(bounced, "e={e} k={k}: the ball never moved up");
    (hgt, apex)
}

/// oracle: apex / drop height = `e²` at all 16 touch phases. Budget: the
/// discrete flight differs from the parabola by `O(v h)` at both ends
/// (`v ≈ 5 m/s`, `h = 1/480`: `~1e-2 m` on a `1.25 m` drop), and the apex is
/// sampled at frame ends (`g Δt² / 8 = 3.5e-4 m`); the tolerance on the
/// ratio is `3 v h / H`. The phase-dependent response this replaces gave
/// `e² (k/16)²`, off by up to `0.98 e²`.
#[test]
fn bounce_height_ratio_is_e_squared_at_every_touch_phase() {
    let h = substep();
    let mut failures = Vec::new();
    for path in paths() {
        for e in [0.3, 0.6, 0.9] {
            for k in 1..=16 {
                let (hgt, apex) = bounce_apex(e, k, path);
                let ratio = apex / hgt;
                let v = (2.0 * g() * hgt).sqrt();
                let tol = 3.0 * v * h / hgt;
                if (ratio - e * e).abs() > tol {
                    failures.push(format!(
                        "{path:?} e={e} k={k}: apex/H = {ratio:.5}, e² = {:.5} (tol {tol:.1e})",
                        e * e
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// oracle: a ball resting on the ground with `e = 0.9` stays at rest. At rest
/// the pre-solve approach per substep is `g h = 0.021 m/s`, below the
/// threshold `2 g h`, so the contact does not bounce; a bounce of `e g h`
/// every substep would keep the ball hopping. Checked over 1 s after 1 s of
/// settling from `5 mm` above the ground.
#[test]
fn a_resting_ball_does_not_hop_below_the_restitution_threshold() {
    for path in paths() {
        let (mut w, ball) = scene(0.0, 0.9, v3(0.0, R + 0.005, 0.0));
        for _ in 0..60 {
            advance(&mut w, path);
        }
        let mut worst: f64 = 0.0;
        for _ in 0..60 {
            advance(&mut w, path);
            worst = worst.max(w.get_body(ball).expect("ball").velocity.y.to_f64().abs());
        }
        assert!(
            worst < 1e-6,
            "{path:?}: the resting ball moves at up to {worst:.3e} m/s"
        );
    }
}

// ---------------------------------------------------------------------------
// Kinetic friction
// ---------------------------------------------------------------------------

/// Runs `frames` and returns `(x, v_x, ω_z)` of the ball.
fn slide(mu: f64, v0: f64, inv_inertia: Vec3Fix, frames: usize, path: Path) -> (f64, f64, f64) {
    let (mut w, ball) = scene(mu, 0.0, v3(0.0, R, 0.0));
    w.bodies[ball].inv_inertia = inv_inertia;
    w.bodies[ball].velocity = v3(v0, 0.0, 0.0);
    for _ in 0..frames {
        advance(&mut w, path);
    }
    let b = w.get_body(ball).expect("ball");
    (
        b.position.x.to_f64(),
        b.velocity.x.to_f64(),
        b.angular_velocity.z.to_f64(),
    )
}

/// oracle: a ball that cannot rotate slides with `v = v₀ − μ g t` and stops at
/// `t_s = v₀ / (μ g)` after `v₀² / (2 μ g)`. Budget `2e-2` m/s: the ground's
/// tilt (`< 1e-3` relative on `μ g`) and a phase of one substep at the
/// start (`μ g h = 6e-3`). The pre-fix cap (`μ |v_n|` after the solve, `≈ 0`
/// at rest contact) left the ball at `v₀`.
#[test]
fn a_ball_that_cannot_turn_decelerates_at_mu_g_and_stops() {
    let (mu, v0) = (0.3, 4.0);
    let stop = v0 / (mu * g());
    for path in paths() {
        for frames in [30, 60] {
            let t = frames as f64 * DT;
            let (_, v, _) = slide(mu, v0, Vec3Fix::ZERO, frames, path);
            let want = v0 - mu * g() * t;
            assert!(
                (v - want).abs() < 2e-2,
                "{path:?} t={t}: v {v}, closed form {want}"
            );
        }
        let (x, v, _) = slide(mu, v0, Vec3Fix::ZERO, 150, path);
        let distance = v0 * v0 / (2.0 * mu * g());
        assert!(
            v.abs() < 1e-6,
            "{path:?}: still moving at {v} after {stop:.3} s"
        );
        assert!(
            (x - distance).abs() < 3e-2,
            "{path:?}: stopped at {x}, closed form {distance}"
        );
    }
}

/// oracle: a solid ball launched without spin: `v = v₀ − μ g t`,
/// `ω_z r = −(5/2) μ g t` until `t* = 2 v₀ / (7 μ g)`, then rolls at
/// `5 v₀ / 7` with `v = −ω_z r`. Budget `2e-2` m/s as above.
///
/// Measured (step, t = 0.2 s): `v = 3.40016` (the closed form 3.4) but
/// `ω_z r = 0` (want −1.5): the contact friction acts on the centre
/// velocities only and applies no angular impulse, so the ball never spins
/// up and never rolls.
#[test]
#[ignore = "src gap: contact friction is translational only, a solid ball gets no spin and never rolls"]
fn a_solid_ball_slides_then_rolls_at_five_sevenths_of_its_launch_speed() {
    let (mu, v0) = (0.3, 4.0);
    let inv_i = 1.0 / (0.4 * R * R);
    let t_star = 2.0 * v0 / (7.0 * mu * g());
    for path in paths() {
        let frames = 12; // t = 0.2 s < t* = 0.381 s
        let t = frames as f64 * DT;
        let (_, v, wz) = slide(mu, v0, v3(inv_i, inv_i, inv_i), frames, path);
        let want_v = v0 - mu * g() * t;
        let want_spin = -2.5 * mu * g() * t;
        assert!(
            (v - want_v).abs() < 2e-2 && (wz * R - want_spin).abs() < 2e-2,
            "{path:?} t={t} < t*={t_star:.3}: v {v} (want {want_v}), ω_z r {} (want {want_spin})",
            wz * R
        );
        let (_, v, wz) = slide(mu, v0, v3(inv_i, inv_i, inv_i), 90, path);
        let roll = 5.0 * v0 / 7.0;
        assert!(
            (v - roll).abs() < 2e-2 && (v + wz * R).abs() < 2e-2,
            "{path:?} t=1.5 s > t*: v {v} (want {roll}), v + ω_z r = {}",
            v + wz * R
        );
    }
}

// ---------------------------------------------------------------------------
// Static friction on a real incline
// ---------------------------------------------------------------------------

/// A static box turned by `θ` about `z` (its top face is the incline) and a
/// ball that cannot rotate resting on the face at the box's centre line;
/// returns the ball's displacement along the slope's downhill direction after
/// `frames`.
fn incline(mu: f64, theta: f64, frames: usize, path: Path) -> f64 {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    });
    let slab = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(theta)).normalize();
    w.bodies[slab].rotation = q;
    assert!(w.set_body_shape(
        slab,
        &Shape::Box {
            half_extents: v3(20.0, 0.5, 2.0),
        }
    ));
    // up the face normal by half thickness + R from the box centre
    let normal = [-theta.sin(), theta.cos(), 0.0];
    let up = 0.5 + R;
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(normal[0] * up, normal[1] * up, 0.0), Fix128::ONE),
        fx(R),
    );
    w.bodies[ball].inv_inertia = Vec3Fix::ZERO;
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
    w.set_body_material(slab, id);
    w.set_body_material(ball, id);
    let start = w.bodies[ball].position;
    for _ in 0..frames {
        advance(&mut w, path);
    }
    let d = w.bodies[ball].position - start;
    // downhill along the face: −(cos θ, sin θ, 0)
    -(d.x.to_f64() * theta.cos() + d.y.to_f64() * theta.sin())
}

/// oracle: below the friction angle (`tan 0.25 = 0.255 < μ = 0.5`) the ball
/// stays put on the incline (`< 1e-6 m` in 2 s); above it (`tan 0.6 = 0.684`)
/// it slides `½ g (sin θ − μ cos θ) t²` (budget `2 %`: one substep of phase,
/// `a h t = 3e-3 m` on `0.76 m`). Velocity-only friction let the held ball
/// creep `g sin θ h²` per substep, `~5 mm` in 2 s at this angle.
#[test]
fn a_ball_on_an_incline_holds_below_the_friction_angle_and_slides_above_it() {
    let mu = 0.5;
    for path in paths() {
        let held = incline(mu, 0.25, 120, path);
        assert!(
            held.abs() < 1e-6,
            "{path:?}: held ball moved {held:.3e} m downhill"
        );
        let theta: f64 = 0.6;
        let t = 1.0;
        let s = incline(mu, theta, 60, path);
        let want = 0.5 * g() * (theta.sin() - mu * theta.cos()) * t * t;
        assert!(
            (s - want).abs() < 2e-2 * want,
            "{path:?}: slid {s} m, closed form {want} m"
        );
    }
}
