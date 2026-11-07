//! Closed-form checks of the contact friction's rotational response: the
//! velocity-level friction acts at the contact point, so a body's spin enters
//! the slip (`v + ω × r`) and the impulse turns the body through its lever
//! arm.
//!
//! Expected values are written from elementary mechanics (solid ball
//! `I = 2/5 m r²`, `r = 1/2`, `m = 1`, flat static ground, default gravity
//! `−10`, damping off):
//!
//! - launched sliding without spin at `v₀` on level ground with friction `μ`:
//!   the centre decelerates at `μ g` (the cap `μ λ_n / h` binds: every frame
//!   removes exactly `μ g Δt`) while the spin grows at `5 μ g / (2 r)`, until
//!   `t* = 2 v₀ / (7 μ g)`; from then on it rolls at `5 v₀ / 7` with
//!   `v = −ω_z r`;
//! - launched rolling (`v₀ = −ω_z r`): friction has nothing to remove, the
//!   speed stays `v₀`, the kinetic energy does not grow;
//! - released on an incline of angle `θ` with `μ > (2/7) tan θ`: it rolls
//!   without slipping at `a = (5/7) g sin θ`;
//! - dropped straight down (on a ground sphere, whose normal is exact): no
//!   tangential slip, no spin at all;
//! - a unit box on an incline: held exactly below the friction angle and
//!   sliding without any turn above it — a shaped body's contact point is the
//!   support vertex EPA reports, not the point under the contact, so the
//!   rotational response is reserved for plain spheres and a box keeps the
//!   centre form (the box's own rotational friction is a listed gap);
//! - two free balls (no gravity), one spinning, meeting head on with `e = 0`:
//!   the friction impulse of the collision substep is capped at
//!   `μ λ_n / (h w_sum)`, the pair's linear momentum is unchanged, the angular
//!   momentum about the origin changes by no more than `depth · p` (the two
//!   contact points sit `depth` apart along the normal), the kinetic energy
//!   does not grow.
//!
//! Paths: `step` and, with the `parallel` feature, `step_parallel`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference in f64: rolling closed forms and conservation sums"
)]

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics_core::PhysicsMaterial;

const DT: f64 = 1.0 / 60.0;
const R: f64 = 0.5;
/// `1 / I` of the solid unit-mass ball
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

/// Default configuration without damping; a flat static box (top at `y = 0`)
/// and a solid ball of radius `R` resting on it at `x = 0`, one material
/// `(μ, e = 0)` on both.
fn flat(mu: f64) -> (PhysicsWorld, usize) {
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
    let ball = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, R, 0.0), Fix128::ONE), fx(R));
    w.bodies[ball].inv_inertia = v3(INV_I, INV_I, INV_I);
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
    w.set_body_material(slab, id);
    w.set_body_material(ball, id);
    (w, ball)
}

/// `(x, v_x, ω_z r)` of `ball`
fn state(w: &PhysicsWorld, ball: usize) -> (f64, f64, f64) {
    let b = &w.bodies[ball];
    (
        b.position.x.to_f64(),
        b.velocity.x.to_f64(),
        b.angular_velocity.z.to_f64() * R,
    )
}

/// translational + rotational kinetic energy of the unit-mass solid ball
fn kinetic(w: &PhysicsWorld, ball: usize) -> f64 {
    let b = &w.bodies[ball];
    0.5 * b.velocity.length_squared().to_f64()
        + 0.5 * b.angular_velocity.length_squared().to_f64() / INV_I
}

// ---------------------------------------------------------------------------
// Sliding → rolling
// ---------------------------------------------------------------------------

/// oracle: launched at `v₀ = 4` with `μ = 0.3`, no spin. While sliding
/// (`t < t* = 2 v₀ / (7 μ g) = 0.381 s`): `v = v₀ − μ g t`, `ω_z r = −(5/2) μ g t`
/// (`t = 0.2 s`, budget `2e-2`); no frame removes more than the cap
/// `μ g Δt` from `v`, and frames 2 to 11 together remove `10 μ g Δt` (the cap
/// binds; budget `1e-3`, the normal multiplier settles to `g h²` over the
/// first frames and frame 1 starts from an exact touch). Rolling
/// (`t = 1.5 s`):
/// `v = 5 v₀ / 7`, `|v + ω_z r| < 1e-3`.
#[test]
fn a_launched_ball_spins_up_at_the_capped_rate_and_rolls_at_five_sevenths() {
    let (mu, v0) = (0.3, 4.0);
    let t_star = 2.0 * v0 / (7.0 * mu * g());
    for path in paths() {
        let (mut w, ball) = flat(mu);
        w.bodies[ball].velocity = v3(v0, 0.0, 0.0);
        let mut v_prev = v0;
        let mut removed = 0.0;
        for frame in 1..=90 {
            advance(&mut w, path);
            let (x, v, wr) = state(&w, ball);
            let t = frame as f64 * DT;
            // frame 1 starts from an exact touch: its first substep has no
            // penetration to push on yet, so its multiplier is short
            if frame > 1 && frame <= 11 {
                let dv = v_prev - v;
                assert!(
                    dv < mu * g() * DT + 1e-3,
                    "{path:?} t={t:.3}: frame removed {dv:.9} from v, above the cap μ g Δt = {:.9}",
                    mu * g() * DT
                );
                removed += dv;
            }
            v_prev = v;
            if frame == 12 {
                let (want_v, want_wr) = (v0 - mu * g() * t, -2.5 * mu * g() * t);
                println!(
                    "{path:?} t={t:.3}: x {x:.6} v {v:.6} ({want_v}) ω_z r {wr:.6} ({want_wr})"
                );
                assert!(
                    (v - want_v).abs() < 2e-2 && (wr - want_wr).abs() < 2e-2,
                    "{path:?} t={t:.3} < t*={t_star:.3}: v {v} (want {want_v}), ω_z r {wr} (want {want_wr})"
                );
            }
        }
        assert!(
            (removed - 10.0 * mu * g() * DT).abs() < 1e-3,
            "{path:?}: frames 2..=11 removed {removed:.6} from v, the cap gives {:.6}",
            10.0 * mu * g() * DT
        );
        let (x, v, wr) = state(&w, ball);
        let roll = 5.0 * v0 / 7.0;
        println!("{path:?} t=1.5: x {x:.6} v {v:.6} ({roll:.6}) ω_z r {wr:.6}");
        assert!(
            (v - roll).abs() < 2e-2 && (v + wr).abs() < 1e-3,
            "{path:?} t=1.5 s > t*: v {v} (want {roll}), v + ω_z r = {}",
            v + wr
        );
    }
}

/// oracle: launched rolling (`v₀ = 4`, `ω_z = −v₀ / r`) the ball has no slip
/// to remove: after 1 s `|v − v₀| < 1e-3`, `|v + ω_z r| < 1e-3`, and the
/// kinetic energy never grows by more than `1e-6` in a frame (what remains
/// is the velocity derivation's rounding of the turn, `θ² / 24` per substep).
#[test]
fn a_rolling_ball_keeps_rolling_and_gains_no_energy() {
    let v0 = 4.0;
    for path in paths() {
        let (mut w, ball) = flat(0.3);
        w.bodies[ball].velocity = v3(v0, 0.0, 0.0);
        w.bodies[ball].angular_velocity = v3(0.0, 0.0, -v0 / R);
        let mut e_prev = kinetic(&w, ball);
        let mut worst = 0.0f64;
        for _ in 0..60 {
            advance(&mut w, path);
            let e = kinetic(&w, ball);
            worst = worst.max(e - e_prev);
            e_prev = e;
        }
        let (_, v, wr) = state(&w, ball);
        println!("{path:?} t=1: v {v:.6} ω_z r {wr:.6} worst energy gain per frame {worst:.3e}");
        assert!(
            (v - v0).abs() < 1e-3 && (v + wr).abs() < 1e-3 && worst < 1e-6,
            "{path:?}: v {v} (want {v0}), v + ω_z r {}, worst energy gain {worst:.3e}",
            v + wr
        );
    }
}

/// oracle: dropped from `y = R + 0.3` onto a static ground sphere of radius
/// `1e4` (top at `y = 0`; the contact normal is the centre line, exactly
/// `+y`) with `μ = 0.5`: nothing tangential ever happens, so after 1 s the
/// ball has no angular velocity and the identity rotation, bit for bit.
/// (On the flat box the EPA normal is vertical only to about `1e-10` and
/// the impact leaves a spin of `2e-7`.)
#[test]
fn a_ball_dropped_straight_down_gets_no_spin() {
    for path in paths() {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            damping_rate: Fix128::ZERO,
            ..PhysicsConfig::default()
        });
        let ground = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -1.0e4, 0.0)), fx(1.0e4));
        let ball = w.add_body_with_radius(
            RigidBody::new_dynamic(v3(0.0, R + 0.3, 0.0), Fix128::ONE),
            fx(R),
        );
        w.bodies[ball].inv_inertia = v3(INV_I, INV_I, INV_I);
        let id = w
            .material_table
            .register(PhysicsMaterial::new(0, fx(0.5), Fix128::ZERO));
        w.set_body_material(ground, id);
        w.set_body_material(ball, id);
        let mut contacts = 0usize;
        for _ in 0..60 {
            advance(&mut w, path);
            contacts += w.contact_constraints.len();
        }
        assert!(contacts > 0, "{path:?}: the ball never touched the ground");
        let b = &w.bodies[ball];
        assert!(
            b.angular_velocity.length_squared().is_zero()
                && b.rotation.x.is_zero()
                && b.rotation.y.is_zero()
                && b.rotation.z.is_zero()
                && b.rotation.w == Fix128::ONE,
            "{path:?}: dropped ball turned: ω {:?}, q {:?}",
            b.angular_velocity,
            b.rotation
        );
    }
}

// ---------------------------------------------------------------------------
// Rolling down an incline
// ---------------------------------------------------------------------------

/// oracle: released from rest on a static box turned by `θ = 0.3` about `z`
/// with `μ = 0.5` (`> (2/7) tan θ = 0.088`, so it rolls without slipping):
/// after 1 s the displacement down the slope is `(5/7) g sin θ t² / 2` and
/// `|v + ω r| < 1e-3`. Budget `2 %` (one substep of phase, `a h t`).
///
/// Measured (step, t = 1 s): displacement `0.000000` (want 1.055): the
/// position-level static friction (`static_friction_correction`) cancels the
/// centre's tangential displacement when it is under `μ λ`, taking no lever
/// arm, so the ball is held like one that cannot turn; the velocity pass then
/// sees no slip at the contact and adds no spin.
#[test]
#[ignore = "src gap: the position-level static friction holds the centre of a ball that should roll down the incline"]
fn a_ball_rolls_down_an_incline_at_five_sevenths_of_g_sin_theta() {
    let (mu, theta) = (0.5, 0.3_f64);
    for path in paths() {
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
        w.bodies[ball].inv_inertia = v3(INV_I, INV_I, INV_I);
        let id = w
            .material_table
            .register(PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
        w.set_body_material(slab, id);
        w.set_body_material(ball, id);
        let start = w.bodies[ball].position;
        for _ in 0..60 {
            advance(&mut w, path);
        }
        let b = &w.bodies[ball];
        let d = b.position - start;
        // downhill along the face: −(cos θ, sin θ, 0)
        let down = -(d.x.to_f64() * theta.cos() + d.y.to_f64() * theta.sin());
        let speed = -(b.velocity.x.to_f64() * theta.cos() + b.velocity.y.to_f64() * theta.sin());
        let a = 5.0 / 7.0 * g() * theta.sin();
        let want = a * 0.5;
        // rolling downhill (−x-ish) turns the ball about +z: `speed = ω_z r`
        let slip = speed - b.angular_velocity.z.to_f64() * R;
        println!(
            "{path:?} incline t=1: down {down:.6} ({want:.6}) speed {speed:.6} slip {slip:.3e}"
        );
        assert!(
            (down - want).abs() < 0.02 * want + a * substep() && slip.abs() < 1e-3,
            "{path:?}: rolled {down:.6} m, closed form {want:.6}; slip {slip:.3e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Two free bodies: cap and conservation
// ---------------------------------------------------------------------------

/// oracle: no gravity; ball A at the origin moving at `+2` along `x` and
/// spinning at `ω_z = 6`, ball B at rest ahead of it with a gap chosen so that
/// the collision substep penetrates by exactly `depth = v_rel h / 2`
/// (semi-implicit Euler: the gap closes by `v_rel h` per substep); `μ = 0.6`,
/// `e = 0`. The normal multiplier of that substep is `λ = depth`, so the
/// friction impulse is capped at `p = μ λ / (h w_sum) = 0.3` and, the slip at
/// the contact (`ω r = 3 m/s`) needing more, the cap binds: A's centre gets
/// `−p` along `y`, B's `+p`, and both spins change by `I⁻¹ r p = 1.5`.
/// After 1 s: `v_a,y = −0.3`, `v_b,y = +0.3` (budget `1e-3`), `ω_a = 4.5`,
/// `ω_b = −1.5` within `Δω · depth / r` (the lever arm is read in the solved
/// pose, `depth / 2` shorter); the pair's linear momentum is unchanged (`< 1e-12`), its
/// angular momentum about the origin by at most `depth · p` (`6.3e-4`), and
/// the kinetic energy does not grow.
#[test]
fn two_free_balls_in_a_spinning_collision_are_capped_and_conserve_momentum() {
    let (mu, v_rel, spin) = (0.6, 2.0, 6.0);
    let h = substep();
    let depth = v_rel * h / 2.0;
    let p = mu * depth / (h * 2.0);
    let dw = INV_I * R * p;
    for path in paths() {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            damping_rate: Fix128::ZERO,
            gravity: Vec3Fix::ZERO,
            ..PhysicsConfig::default()
        });
        let a = w.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), fx(R));
        let gap = 5.0 * v_rel * h - depth;
        let b = w.add_body_with_radius(
            RigidBody::new_dynamic(v3(2.0 * R + gap, 0.0, 0.0), Fix128::ONE),
            fx(R),
        );
        for i in [a, b] {
            w.bodies[i].inv_inertia = v3(INV_I, INV_I, INV_I);
        }
        w.bodies[a].velocity = v3(v_rel, 0.0, 0.0);
        w.bodies[a].angular_velocity = v3(0.0, 0.0, spin);
        let id = w
            .material_table
            .register(PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
        w.set_body_material(a, id);
        w.set_body_material(b, id);
        let total = |w: &PhysicsWorld| {
            let mut lin = Vec3Fix::ZERO;
            let mut ang = Vec3Fix::ZERO;
            let mut e = 0.0;
            for i in [a, b] {
                let bd = &w.bodies[i];
                lin = lin + bd.velocity;
                ang = ang + bd.position.cross(bd.velocity) + bd.angular_velocity * fx(1.0 / INV_I);
                e += 0.5 * bd.velocity.length_squared().to_f64()
                    + 0.5 * bd.angular_velocity.length_squared().to_f64() / INV_I;
            }
            (lin, ang, e)
        };
        let (p0, l0, e0) = total(&w);
        let (mut dp, mut dl, mut e_prev, mut gain) = (0.0f64, 0.0f64, e0, 0.0f64);
        for _ in 0..60 {
            advance(&mut w, path);
            let (pn, ln, e) = total(&w);
            dp = dp.max((pn - p0).length().to_f64());
            dl = dl.max((ln - l0).length().to_f64());
            gain = gain.max(e - e_prev);
            e_prev = e;
        }
        let (va, vb) = (
            w.bodies[a].velocity.y.to_f64(),
            w.bodies[b].velocity.y.to_f64(),
        );
        let (wa, wb) = (
            w.bodies[a].angular_velocity.z.to_f64(),
            w.bodies[b].angular_velocity.z.to_f64(),
        );
        println!(
            "{path:?}: v_a,y {va:.6} v_b,y {vb:.6} ω_a {wa:.6} ω_b {wb:.6} |ΔP| {dp:.3e} |ΔL| {dl:.3e} worst gain {gain:.3e} (p {p}, Δω {dw})"
        );
        assert!(
            (va + p).abs() < 1e-3 && (vb - p).abs() < 1e-3,
            "{path:?}: centre velocities v_a,y {va}, v_b,y {vb}; the capped impulse is {p}"
        );
        // the lever arm is read in the solved pose, after the position solve
        // moved each ball out by `depth / 2`, so it is `r − depth / 2`
        let arm = dw * depth / R;
        assert!(
            (wa - (spin - dw)).abs() <= arm && (wb + dw).abs() <= arm,
            "{path:?}: spins ω_a {wa} (want {} ± {arm:.1e}), ω_b {wb} (want {} ± {arm:.1e})",
            spin - dw,
            -dw
        );
        assert!(
            dp < 1e-12 && dl <= depth * p + 1e-9 && gain <= 0.0,
            "{path:?}: |ΔP| {dp:.3e}, |ΔL| {dl:.3e} (bound {:.3e}), worst energy gain {gain:.3e}",
            depth * p
        );
    }
}

// ---------------------------------------------------------------------------
// Shaped bodies keep the centre form
// ---------------------------------------------------------------------------

/// Static box turned by `θ` about `z` and a unit cube resting on its face,
/// turned the same way, `inv_inertia = 6` (a unit cube), `μ = 0.5`.
fn cube_on_incline(theta: f64, mu: f64) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping_rate: Fix128::ZERO,
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
    let normal = [-theta.sin(), theta.cos(), 0.0];
    let cube = w.add_body(RigidBody::new_dynamic(
        v3(normal[0], normal[1], 0.0),
        Fix128::ONE,
    ));
    w.bodies[cube].rotation = q;
    assert!(w.set_body_shape(
        cube,
        &Shape::Box {
            half_extents: v3(0.5, 0.5, 0.5),
        }
    ));
    w.bodies[cube].inv_inertia = v3(6.0, 6.0, 6.0);
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
    w.set_body_material(slab, id);
    w.set_body_material(cube, id);
    (w, cube)
}

/// oracle: a unit cube on `θ = 0.25` with `μ = 0.5` (`tan θ = 0.255 < μ`) is
/// held: after 2 s it has not moved along the slope (`< 1e-9` m) and has not
/// turned at all (its rotation is still the slab's, its angular velocity
/// zero). On `θ = 0.6` (`tan θ = 0.684 > μ`) it slides at
/// `g (sin θ − μ cos θ)` (2 s, budget 2 %) and, its contact point being a
/// support vertex rather than the point under it, it must not turn either
/// (a sliding cube on a plane does not tip with `μ < 1`).
#[test]
fn a_cube_on_an_incline_is_held_below_mu_and_slides_without_turning_above() {
    for path in paths() {
        for (theta, slides) in [(0.25_f64, false), (0.6, true)] {
            let (mut w, cube) = cube_on_incline(theta, 0.5);
            let start = w.bodies[cube].position;
            let q0 = w.bodies[cube].rotation;
            for _ in 0..120 {
                advance(&mut w, path);
            }
            let b = &w.bodies[cube];
            let d = b.position - start;
            let down = -(d.x.to_f64() * theta.cos() + d.y.to_f64() * theta.sin());
            assert!(
                b.rotation == q0 && b.angular_velocity.length_squared().is_zero(),
                "{path:?} θ={theta}: the cube turned: q {:?} ω {:?}",
                b.rotation,
                b.angular_velocity
            );
            if slides {
                let a = g() * (theta.sin() - 0.5 * theta.cos());
                let want = a * 2.0;
                println!("{path:?} cube θ={theta} t=2: down {down:.6} ({want:.6})");
                assert!(
                    (down - want).abs() < 0.02 * want + a * substep() * 2.0,
                    "{path:?}: cube slid {down:.6}, closed form {want:.6}"
                );
            } else {
                println!("{path:?} cube θ={theta} t=2: down {down:.3e}");
                assert!(down.abs() < 1e-9, "{path:?}: held cube crept {down:.3e} m");
            }
        }
    }
}
