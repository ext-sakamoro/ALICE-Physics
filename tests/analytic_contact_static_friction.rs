//! Oracles for position-level static friction in the XPBD contact solve
//! (Müller, Macklin, Chentanez, Jeschke, Kim, "Detailed Rigid Body Simulation
//! with Extended Position Based Dynamics", SCA 2020, eq. (26)): a contact
//! whose tangential pull stays inside the friction cone holds the body in
//! place, not only its velocity.
//!
//! Every scene runs through `PhysicsWorld::step` and, with the `parallel`
//! feature, `PhysicsWorld::step_parallel`, with `PhysicsConfig::default()`
//! except for the gravity vector. Expected values are closed forms, never
//! obtained by running the solver:
//!
//! * a body that cannot rotate, pressed on level ground by `g_n` and pulled
//!   along it by `a_t` per unit mass, with friction `μ`: it stays at rest if
//!   `a_t < μ g_n`, otherwise it accelerates at `a = a_t − μ g_n`, and with
//!   the frame damping (`β = −ln(d) / dt`) from rest
//!   `x(t) = (a/β) t − (a/β²)(1 − e^{−βt})`;
//! * a slope of angle `θ` is the same problem with `g_n = g cos θ` and
//!   `a_t = g sin θ` (the gravity is tilted against a level ground);
//! * a horizontal push `F` on level ground is the same problem with
//!   `g_n = g` and `a_t = F / m` (applied as a constant horizontal
//!   acceleration in every substep, like the gravity);
//! * a vetoed contact has no normal and no friction: the body moves freely,
//!   `x(t)` above with `a = a_t` and `y(t)` the same with `a = −g_n`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{ContactModifier, PhysicsConfig, PhysicsWorld, RigidBody};

const DT: f64 = 1.0 / 60.0;
const FRAMES: usize = 60;

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

/// Gravity magnitude and frame-damping rate of `PhysicsConfig::default()`.
fn g_and_beta() -> (f64, f64) {
    let d = PhysicsConfig::default();
    (-d.gravity.y.to_f64(), -(d.damping.to_f64()).ln() / DT)
}

fn damped_displacement(a: f64, beta: f64, t: f64) -> f64 {
    (a / beta) * t - (a / (beta * beta)) * (1.0 - (-beta * t).exp())
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

/// What filters the scene installs.
enum Filter {
    None,
    Veto,
    Friction(f64),
}

/// A ball of radius `0.5` that cannot rotate, resting at rest on a large
/// static sphere whose top is the plane `y = 0` (radius `1e4`), both with a
/// material of friction `mu` and restitution `0`. The gravity is
/// `(a_t, −g_n, 0)`. Returns the ball's position and velocity after one
/// second.
fn run(mu: f64, a_t: f64, g_n: f64, filter: Filter, path: Path) -> (Vec3Fix, Vec3Fix) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: v3(a_t, -g_n, 0.0),
        ..PhysicsConfig::default()
    });
    let big = 1.0e4;
    let ground = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -big, 0.0)), fx(big));
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE),
        fx(0.5),
    );
    w.bodies[ball].inv_inertia = Vec3Fix::ZERO;
    let id = w
        .material_table
        .register(alice_physics::PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
    w.set_body_material(ground, id);
    w.set_body_material(ball, id);
    match filter {
        Filter::None => {}
        Filter::Veto => w.add_pre_solve_hook(Box::new(|_, _, _c: &Contact| false)),
        Filter::Friction(m) => w.add_contact_modifier(Box::new(AbsoluteFriction(fx(m)))),
    }
    for _ in 0..FRAMES {
        match path {
            Path::Serial => w.step(fx(DT)),
            #[cfg(feature = "parallel")]
            Path::Batched => w.step_parallel(fx(DT)),
        }
    }
    let b = w.get_body(ball).expect("ball");
    (b.position, b.velocity)
}

/// Error budget of a sliding run over `t` seconds, `2 (μ g_n + a) h t`: the
/// position of a substep advances with the velocity before the velocity-level
/// friction removes `μ g_n h` from it (an offset of `μ g_n h` per second), the
/// symplectic integration adds `a h / 2` per second, and the factor 2 covers
/// the per-frame (not continuous) damping of the damping term.
fn slide_tol(mu: f64, g_n: f64, a: f64, t: f64) -> f64 {
    let h = DT / PhysicsConfig::default().substeps as f64;
    2.0 * (mu * g_n + a) * h * t
}

/// Tolerance of the held position: `1e-12 m`. A held body moves only by
/// fixed-point rounding of the predicted step and its cancellation
/// (`~1e-30 m` per operation, 480 substeps), and by the surface tilt of the
/// large sphere, which is zero at the contact point `x = 0`. The creep that
/// a velocity-only friction leaves, `a_t h²` per substep, is `~6e-3 m` here.
const HOLD_TOL: f64 = 1e-12;

/// On a slope with `tan θ = 0.3 < μ = 0.5` the ball stays put.
#[test]
fn a_ball_on_a_slope_below_the_friction_angle_stays_put() {
    let (g, _) = g_and_beta();
    let theta = 0.3f64.atan();
    for path in paths() {
        let (p, v) = run(0.5, g * theta.sin(), g * theta.cos(), Filter::None, path);
        let x = p.x.to_f64();
        println!("{path:?}: slope hold x = {x:.3e}");
        assert!(
            x.abs() < HOLD_TOL && v.x.to_f64().abs() < HOLD_TOL,
            "{path:?}: x = {x:.3e}, v = {:.3e}",
            v.x.to_f64()
        );
    }
}

/// A horizontal push of `0.8 μ m g` leaves the ball at rest; a push of
/// `1.2 μ m g` makes it slide at `a = F/m − μ g` (`x(1 s) ≈ 0.41 m`,
/// tolerance [`slide_tol`] `≈ 2.5e-2 m`).
#[test]
fn a_push_below_mu_m_g_holds_and_above_it_slides() {
    let (g, beta) = g_and_beta();
    let mu = 0.5;
    for path in paths() {
        let (p, v) = run(mu, 0.8 * mu * g, g, Filter::None, path);
        assert!(
            p.x.to_f64().abs() < HOLD_TOL && v.x.to_f64().abs() < HOLD_TOL,
            "{path:?}: push 0.8 μ m g moved the ball to x = {:.3e}",
            p.x.to_f64()
        );
        let want = damped_displacement(0.2 * mu * g, beta, 1.0);
        let tol = slide_tol(mu, g, 0.2 * mu * g, 1.0);
        let (p, _) = run(mu, 1.2 * mu * g, g, Filter::None, path);
        let x = p.x.to_f64();
        assert!(
            (x - want).abs() < tol,
            "{path:?}: push 1.2 μ m g: x = {x:.5}, closed form {want:.5}"
        );
    }
}

/// The static friction uses the modifier's friction: a modifier `μ = 0.5`
/// over a material `μ = 0.1` holds the ball on the `tan θ = 0.3` slope, and
/// a modifier `μ = 0.1` over a material `μ = 0.5` lets it slide at
/// `a = g (sin θ − 0.1 cos θ)` (tolerance [`slide_tol`]).
#[test]
fn the_static_friction_uses_the_modified_friction() {
    let (g, beta) = g_and_beta();
    let theta = 0.3f64.atan();
    let (a_t, g_n) = (g * theta.sin(), g * theta.cos());
    for path in paths() {
        let (p, _) = run(0.1, a_t, g_n, Filter::Friction(0.5), path);
        assert!(
            p.x.to_f64().abs() < HOLD_TOL,
            "{path:?}: modifier μ 0.5: x = {:.3e}",
            p.x.to_f64()
        );
        let want = damped_displacement(a_t - 0.1 * g_n, beta, 1.0);
        let tol = slide_tol(0.1, g_n, a_t - 0.1 * g_n, 1.0);
        let (p, _) = run(0.5, a_t, g_n, Filter::Friction(0.1), path);
        let x = p.x.to_f64();
        assert!(
            (x - want).abs() < tol,
            "{path:?}: modifier μ 0.1: x = {x:.5}, closed form {want:.5}"
        );
    }
}

/// A vetoed contact gets neither a normal nor a static friction response:
/// on the `tan θ = 0.3` slope with `μ = 0.5` the ball moves freely,
/// `x(t)` with `a = g sin θ` and `y(t)` with `a = −g cos θ` (tolerance
/// `2e-2 m`: the sub-step integration offset `a t h / 2 ≤ 1e-2`).
#[test]
fn a_vetoed_contact_gets_no_static_friction() {
    let (g, beta) = g_and_beta();
    let theta = 0.3f64.atan();
    let (a_t, g_n) = (g * theta.sin(), g * theta.cos());
    let want_x = damped_displacement(a_t, beta, 1.0);
    let want_y = 0.5 + damped_displacement(-g_n, beta, 1.0);
    for path in paths() {
        let (p, _) = run(0.5, a_t, g_n, Filter::Veto, path);
        let (x, y) = (p.x.to_f64(), p.y.to_f64());
        assert!(
            (x - want_x).abs() < 2e-2 && (y - want_y).abs() < 2e-2,
            "{path:?}: ({x:.5}, {y:.5}), free motion ({want_x:.5}, {want_y:.5})"
        );
    }
}

/// The bridge path: a bridge that performs the normal contact projection
/// (`λ = depth`, separation split by inverse mass, the reference XPBD normal
/// step written out here) leaves the static friction to the CPU, and the
/// ball on the `tan θ = 0.3 < μ = 0.5` slope stays put under
/// `step_with_bridge` as under `step`.
#[cfg(feature = "gpu-solver-bridge")]
mod bridge {
    use super::*;
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::solver::ContactConstraint;

    #[derive(Default)]
    struct NormalProjection {
        constraints: Vec<ContactConstraint>,
        positions: Vec<[Fix128; 3]>,
        inv_masses: Vec<Fix128>,
    }

    impl GpuSolverBridge for NormalProjection {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, c: &[ContactConstraint]) {
            self.constraints = c.to_vec();
        }
        fn send_body_state(&mut self, p: &[[Fix128; 3]], i: &[Fix128]) {
            self.positions = p.to_vec();
            self.inv_masses = i.to_vec();
        }
        fn dispatch_contact_solve_iteration(&mut self, _w: Fix128) {
            for c in &mut self.constraints {
                let d = c.contact.depth - c.cached_lambda;
                let (wa, wb) = (self.inv_masses[c.body_a], self.inv_masses[c.body_b]);
                let w = wa + wb;
                if d <= Fix128::ZERO || w.is_zero() {
                    continue;
                }
                c.cached_lambda = c.contact.depth;
                let n = c.contact.normal;
                let (ka, kb) = (d * wa / w, d * wb / w);
                let pa = &mut self.positions[c.body_a];
                pa[0] = pa[0] + n.x * ka;
                pa[1] = pa[1] + n.y * ka;
                pa[2] = pa[2] + n.z * ka;
                let pb = &mut self.positions[c.body_b];
                pb[0] = pb[0] - n.x * kb;
                pb[1] = pb[1] - n.y * kb;
                pb[2] = pb[2] - n.z * kb;
            }
        }
        fn recv_contact_constraints(&self, c: &mut [ContactConstraint]) {
            c.copy_from_slice(&self.constraints);
        }
        fn recv_body_positions(&self, p: &mut [[Fix128; 3]]) {
            p.copy_from_slice(&self.positions);
        }
    }

    #[test]
    fn a_ball_on_a_slope_below_the_friction_angle_stays_put_through_the_bridge() {
        let (g, _) = g_and_beta();
        let theta = 0.3f64.atan();
        let mut w = PhysicsWorld::new(PhysicsConfig {
            gravity: v3(g * theta.sin(), -g * theta.cos(), 0.0),
            ..PhysicsConfig::default()
        });
        let big = 1.0e4;
        let ground = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -big, 0.0)), fx(big));
        let ball = w.add_body_with_radius(
            RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE),
            fx(0.5),
        );
        w.bodies[ball].inv_inertia = Vec3Fix::ZERO;
        let id = w
            .material_table
            .register(alice_physics::PhysicsMaterial::new(
                0,
                fx(0.5),
                Fix128::ZERO,
            ));
        w.set_body_material(ground, id);
        w.set_body_material(ball, id);
        let mut bridge = NormalProjection::default();
        for _ in 0..FRAMES {
            w.step_with_bridge(&mut bridge, fx(DT));
        }
        let b = w.get_body(ball).expect("ball");
        let (x, y) = (b.position.x.to_f64(), b.position.y.to_f64());
        assert!(
            (y - 0.5).abs() < 1e-3,
            "the bridge must hold the ball on the ground, y = {y:.6}"
        );
        assert!(x.abs() < HOLD_TOL, "bridge: x = {x:.3e}");
    }
}
