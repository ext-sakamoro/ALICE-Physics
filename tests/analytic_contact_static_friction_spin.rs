//! Closed-form checks of the position-level static friction's rotational
//! response: the static correction cancels the tangential displacement of the
//! two contact *material points* (not of the centres) through the generalised
//! inverse mass, so a body with inertia rolls where it used to be held.
//!
//! Expected values (solid ball `I = 2/5 m r²`, `r = 1/2`, `m = 1`, default
//! gravity `−10`, damping off; the incline is a static box turned by `θ`
//! about `z`, the ball released from rest on its face):
//!
//! - rolling without slip while `tan θ ≤ 7 μ / 2`: `a = (5/7) g sin θ`,
//!   `v = ω r`, and the mechanical energy `½ m v² + ½ I ω² + m g y` is
//!   conserved (no slip, no friction work);
//! - above the limit (`tan θ > 7 μ / 2`) the contact slips: the centre
//!   accelerates at `g (sin θ − μ cos θ)` and the spin at `α r = (5/2) μ g cos θ`;
//! - a ball that cannot turn (`inv_inertia = 0`) is unchanged: held while
//!   `tan θ < μ`, sliding at `g (sin θ − μ cos θ)` above;
//! - at rest nothing creeps: a stack of three balls and a box on a box keep
//!   their positions within a bound over 10 s, and the batched path gives the
//!   serial path's bits.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference in f64: incline closed forms and energy sums"
)]

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics_core::PhysicsMaterial;

const DT: f64 = 1.0 / 60.0;
const R: f64 = 0.5;
const INV_I: f64 = 1.0 / (0.4 * R * R);

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

fn substep() -> f64 {
    DT / PhysicsConfig::default().substeps as f64
}

/// Static box turned by `θ` about `z` (top face = the incline) and a ball on
/// the face at the box's centre line; `rotates` selects the solid-ball inertia
/// or `inv_inertia = 0`.
fn incline(mu: f64, theta: f64, rotates: bool) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping_rate: Fix128::ZERO,
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
    let normal = [-theta.sin(), theta.cos(), 0.0];
    let up = 0.5 + R;
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(normal[0] * up, normal[1] * up, 0.0), Fix128::ONE),
        fx(R),
    );
    w.bodies[ball].inv_inertia = if rotates {
        v3(INV_I, INV_I, INV_I)
    } else {
        Vec3Fix::ZERO
    };
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
    w.set_body_material(slab, id);
    w.set_body_material(ball, id);
    (w, ball)
}

/// `(downhill displacement, downhill speed, ω_z r, mechanical energy)` of the ball
fn along_slope(w: &PhysicsWorld, ball: usize, start: Vec3Fix, theta: f64) -> (f64, f64, f64, f64) {
    let b = &w.bodies[ball];
    let d = b.position - start;
    let down = -(d.x.to_f64() * theta.cos() + d.y.to_f64() * theta.sin());
    let speed = -(b.velocity.x.to_f64() * theta.cos() + b.velocity.y.to_f64() * theta.sin());
    let wr = b.angular_velocity.z.to_f64() * R;
    let energy = 0.5 * b.velocity.length_squared().to_f64()
        + 0.5 * b.angular_velocity.length_squared().to_f64() / INV_I
        + g() * b.position.y.to_f64();
    (down, speed, wr, energy)
}

/// oracle: `θ = 0.3`, `μ = 0.5` (`tan θ = 0.309 < 7μ/2 = 1.75`): rolls without
/// slip from rest, `d = (5/7) g sin θ t² / 2` after 1 s (budget 2 % + one
/// substep `a h t`), `|v − ω r| < 1e-3`, mechanical energy conserved within
/// `m g · a h t` (the phase of one substep).
#[test]
fn a_ball_below_the_slip_limit_rolls_at_five_sevenths_of_g_sin_theta() {
    let (mu, theta) = (0.5, 0.3_f64);
    let a = 5.0 / 7.0 * g() * theta.sin();
    for path in paths() {
        let (mut w, ball) = incline(mu, theta, true);
        let start = w.bodies[ball].position;
        let (.., e0) = along_slope(&w, ball, start, theta);
        for _ in 0..60 {
            advance(&mut w, path);
        }
        let (down, speed, wr, e) = along_slope(&w, ball, start, theta);
        let want = a * 0.5;
        println!(
            "{path:?} θ=0.3 t=1: d {down:.6} ({want:.6}) v {speed:.6} ω_z r {wr:.6} ΔE {:.3e}",
            e - e0
        );
        assert!(
            (down - want).abs() < 0.02 * want + a * substep()
                && (speed - wr).abs() < 1e-3
                && (e - e0).abs() < g() * a * substep(),
            "{path:?}: rolled {down:.6} (want {want:.6}), v {speed:.6}, ω r {wr:.6}, ΔE {:.3e}",
            e - e0
        );
    }
}

/// oracle: `θ = 0.3`, `μ = 0.05` (`tan θ = 0.309 > 7μ/2 = 0.175`): the
/// contact slips from the start; after 1 s `d = g (sin θ − μ cos θ) t² / 2`
/// and `ω_z r = (5/2) μ g cos θ t` (budget 2 % + one substep), and the ball
/// still slips: `v − ω r > 0`.
#[test]
fn a_ball_above_the_slip_limit_slides_and_spins_at_the_kinetic_rates() {
    let (mu, theta) = (0.05, 0.3_f64);
    let a = g() * (theta.sin() - mu * theta.cos());
    let alpha_r = 2.5 * mu * g() * theta.cos();
    for path in paths() {
        let (mut w, ball) = incline(mu, theta, true);
        let start = w.bodies[ball].position;
        for _ in 0..60 {
            advance(&mut w, path);
        }
        let (down, speed, wr, _) = along_slope(&w, ball, start, theta);
        let want = a * 0.5;
        println!("{path:?} θ=0.3 μ=0.05 t=1: d {down:.6} ({want:.6}) v {speed:.6} ω_z r {wr:.6} ({alpha_r:.6})");
        assert!(
            (down - want).abs() < 0.02 * want + a * substep()
                && (wr - alpha_r).abs() < 0.02 * alpha_r + alpha_r * substep()
                && speed - wr > 0.0,
            "{path:?}: slid {down:.6} (want {want:.6}), ω r {wr:.6} (want {alpha_r:.6}), v − ω r {:.3e}",
            speed - wr
        );
    }
}

/// oracle: a ball that cannot turn is unchanged by the rotational
/// correction: held on `θ = 0.25` with `μ = 0.5` (`tan θ = 0.255 < μ`,
/// `< 1e-6 m` in 2 s) and sliding at `g (sin θ − μ cos θ)` on `θ = 0.6`
/// (`tan θ = 0.684 > μ`, budget 2 %).
#[test]
fn a_ball_that_cannot_turn_is_held_below_mu_and_slides_above() {
    let mu = 0.5;
    for path in paths() {
        let (mut w, ball) = incline(mu, 0.25, false);
        let start = w.bodies[ball].position;
        for _ in 0..120 {
            advance(&mut w, path);
        }
        let (down, ..) = along_slope(&w, ball, start, 0.25);
        assert!(down.abs() < 1e-6, "{path:?}: held ball crept {down:.3e} m");
        let theta = 0.6_f64;
        let (mut w, ball) = incline(mu, theta, false);
        let start = w.bodies[ball].position;
        for _ in 0..120 {
            advance(&mut w, path);
        }
        let a = g() * (theta.sin() - mu * theta.cos());
        let (down, ..) = along_slope(&w, ball, start, theta);
        let want = a * 2.0;
        assert!(
            (down - want).abs() < 0.02 * want + a * substep() * 2.0,
            "{path:?}: slid {down:.6}, closed form {want:.6}"
        );
    }
}

// ---------------------------------------------------------------------------
// Rest: no creep, batched = serial
// ---------------------------------------------------------------------------

/// Three solid balls stacked on the flat ground and a unit box resting on a
/// larger static box, `μ = 0.5`.
fn resting_scene() -> (PhysicsWorld, Vec<usize>) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping_rate: Fix128::ZERO,
        ..PhysicsConfig::default()
    });
    let slab = w.add_body(RigidBody::new_static(v3(0.0, -0.5, 0.0)));
    assert!(w.set_body_shape(
        slab,
        &Shape::Box {
            half_extents: v3(50.0, 0.5, 50.0),
        }
    ));
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(0.5), Fix128::ZERO));
    w.set_body_material(slab, id);
    let mut bodies = Vec::new();
    for k in 0..3 {
        let b = w.add_body_with_radius(
            RigidBody::new_dynamic(v3(0.0, R + 2.0 * R * k as f64, 0.0), Fix128::ONE),
            fx(R),
        );
        w.bodies[b].inv_inertia = v3(INV_I, INV_I, INV_I);
        w.set_body_material(b, id);
        bodies.push(b);
    }
    let cube = w.add_body(RigidBody::new_dynamic(v3(5.0, 0.5, 0.0), Fix128::ONE));
    assert!(w.set_body_shape(
        cube,
        &Shape::Box {
            half_extents: v3(0.5, 0.5, 0.5),
        }
    ));
    // unit cube: I = m (a² + b²) / 12 = 1/6 → inv 6
    w.bodies[cube].inv_inertia = v3(6.0, 6.0, 6.0);
    w.set_body_material(cube, id);
    bodies.push(cube);
    (w, bodies)
}

/// oracle: over 10 s nothing creeps: every body settles within the first
/// second and is then exactly at rest (velocity and angular velocity bit
/// zero at 5 s and 10 s, positions and rotations unchanged between them),
/// the settle itself moves no body sideways by more than `1e-5` m or turns it
/// by more than `1e-5` (measured: `9e-7` / `1e-6`, the box contact's EPA
/// normal is vertical only to `1e-10` and that reads as slip while the
/// stack finds its rest); the batched path's state equals the serial path's
/// bit for bit.
#[test]
fn bodies_at_rest_do_not_creep_and_batched_equals_serial() {
    let mut finals: Vec<Vec<(Vec3Fix, QuatFix, Vec3Fix, Vec3Fix)>> = Vec::new();
    for path in paths() {
        let (mut w, bodies) = resting_scene();
        let starts: Vec<Vec3Fix> = bodies.iter().map(|&b| w.bodies[b].position).collect();
        for _ in 0..300 {
            advance(&mut w, path);
        }
        let at_5s: Vec<(Vec3Fix, QuatFix)> = bodies
            .iter()
            .map(|&b| (w.bodies[b].position, w.bodies[b].rotation))
            .collect();
        for &b in &bodies {
            let bd = &w.bodies[b];
            assert!(
                bd.velocity.length_squared().is_zero()
                    && bd.angular_velocity.length_squared().is_zero(),
                "{path:?}: body {b} still moving at 5 s: v {:?} ω {:?}",
                bd.velocity,
                bd.angular_velocity
            );
        }
        for _ in 0..300 {
            advance(&mut w, path);
        }
        for ((&b, &s), (p5, q5)) in bodies.iter().zip(&starts).zip(&at_5s) {
            let bd = &w.bodies[b];
            assert!(
                bd.position == *p5 && bd.rotation == *q5,
                "{path:?}: body {b} moved between 5 s and 10 s"
            );
            let dx = (bd.position.x - s.x)
                .to_f64()
                .abs()
                .max((bd.position.z - s.z).to_f64().abs());
            let turn = (bd.rotation.x.to_f64().powi(2)
                + bd.rotation.y.to_f64().powi(2)
                + bd.rotation.z.to_f64().powi(2))
            .sqrt();
            println!(
                "{path:?} body {b}: settle drift {dx:.3e} turn {turn:.3e} y {:.6}",
                bd.position.y.to_f64()
            );
            assert!(
                dx < 1e-5 && turn < 1e-5,
                "{path:?}: body {b} crept {dx:.3e} m / turned {turn:.3e} while settling"
            );
        }
        finals.push(
            bodies
                .iter()
                .map(|&b| {
                    let bd = &w.bodies[b];
                    (bd.position, bd.rotation, bd.velocity, bd.angular_velocity)
                })
                .collect(),
        );
    }
    for f in &finals[1..] {
        assert!(f == &finals[0], "batched path differs from serial");
    }
}

/// oracle: the same stack on a ground sphere of radius `1e4` (its normal is
/// the centre line, exact): over 10 s no ball moves sideways or turns at all,
/// bit for bit.
#[test]
fn a_stack_on_an_exact_ground_does_not_move_sideways_at_all() {
    for path in paths() {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            damping_rate: Fix128::ZERO,
            ..PhysicsConfig::default()
        });
        let ground = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -1.0e4, 0.0)), fx(1.0e4));
        let id = w
            .material_table
            .register(PhysicsMaterial::new(0, fx(0.5), Fix128::ZERO));
        w.set_body_material(ground, id);
        let mut balls = Vec::new();
        for k in 0..3 {
            let b = w.add_body_with_radius(
                RigidBody::new_dynamic(v3(0.0, R + 2.0 * R * k as f64, 0.0), Fix128::ONE),
                fx(R),
            );
            w.bodies[b].inv_inertia = v3(INV_I, INV_I, INV_I);
            w.set_body_material(b, id);
            balls.push(b);
        }
        for _ in 0..600 {
            advance(&mut w, path);
        }
        for &b in &balls {
            let bd = &w.bodies[b];
            assert!(
                bd.position.x.is_zero()
                    && bd.position.z.is_zero()
                    && bd.rotation.x.is_zero()
                    && bd.rotation.y.is_zero()
                    && bd.rotation.z.is_zero()
                    && bd.rotation.w == Fix128::ONE
                    && bd.angular_velocity.length_squared().is_zero(),
                "{path:?}: ball {b} moved sideways or turned: p {:?} q {:?}",
                bd.position,
                bd.rotation
            );
        }
    }
}

/// oracle: the rolling ball of the test above, started in orientations far
/// from the identity (angles 1.5–3 rad about skew axes), rolls at the same
/// `(5/7) g sin θ` and, being isotropic, moves bit for bit as the identity
/// run. Teeth: a lever arm turned back through the absolute orientation
/// instead of `ω` (`q⁻¹ r q`) makes the slip exceed the stick limit for every
/// non-identity orientation, the friction drops to its kinetic value and the
/// ball accelerates at `g sin θ − μ g cos θ` instead.
#[test]
fn a_ball_below_the_slip_limit_rolls_the_same_from_every_orientation() {
    let (mu, theta) = (0.5, 0.3_f64);
    let a = 5.0 / 7.0 * g() * theta.sin();
    let orientations: Vec<QuatFix> = (0..6_i64)
        .map(|i| {
            let axis = Vec3Fix::new(fx(1.0 + i as f64), fx(2.0 - i as f64), fx(0.5 + i as f64))
                .normalize();
            QuatFix::from_axis_angle(axis, fx(1.5 + 0.3 * i as f64)).normalize()
        })
        .collect();
    for path in paths() {
        let (mut base, ball) = incline(mu, theta, true);
        let start = base.bodies[ball].position;
        for _ in 0..60 {
            advance(&mut base, path);
        }
        let (down0, ..) = along_slope(&base, ball, start, theta);
        for q in &orientations {
            let (mut w, ball) = incline(mu, theta, true);
            w.bodies[ball].set_rotation(*q);
            for _ in 0..60 {
                advance(&mut w, path);
            }
            let (down, speed, wr, _) = along_slope(&w, ball, start, theta);
            let want = a * 0.5;
            assert!(
                (down - want).abs() < 0.02 * want + a * substep() && (speed - wr).abs() < 1e-3,
                "{path:?} from {q:?}: rolled {down:.6} (want {want:.6}, identity run {down0:.6}), v {speed:.6}, ω r {wr:.6}"
            );
            assert_eq!(
                (
                    w.bodies[ball].position,
                    w.bodies[ball].velocity,
                    w.bodies[ball].angular_velocity
                ),
                (
                    base.bodies[ball].position,
                    base.bodies[ball].velocity,
                    base.bodies[ball].angular_velocity
                ),
                "{path:?}: the rolling ball's motion depends on its orientation"
            );
        }
    }
}
