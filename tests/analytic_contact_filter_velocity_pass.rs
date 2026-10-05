//! Oracles for the contact filters (pre-solve hooks and contact modifiers) on
//! the XPBD backend: what a filter decides for a contact holds in the velocity
//! pass (restitution and friction) as well as in the position pass, in the
//! serial path (`step`) and the batched path (`step_parallel`).
//!
//! Every scene runs through `PhysicsWorld::step` and, with the `parallel`
//! feature, `PhysicsWorld::step_parallel`. Expected values are closed forms
//! written from the scene's geometry, never obtained by running the solver:
//!
//! * frame damping: each `step` of `dt` multiplies every dynamic body's
//!   velocity by `d = config.damping` once, so a body with no contact moving
//!   at `v₀` along `x` has `v_n = v₀ dⁿ` and
//!   `x_n = x₀ + v₀ dt (1 − dⁿ) / (1 − d)` after `n` steps;
//! * two equal spheres of radius `r` closing head-on at `±v` first overlap in
//!   the substep (length `h = dt / substeps`) whose predicted gap is negative,
//!   by `p = −gap`. The position pass removes that overlap, so the relative
//!   normal velocity derived from the substep is `2v − p/h`, and restitution
//!   `e` reverses it: each body leaves at `∓ e (v − p / (2h))`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{ContactModifier, PhysicsConfig, PhysicsWorld, RigidBody};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

fn advance(w: &mut PhysicsWorld, path: Path, steps: usize) {
    for _ in 0..steps {
        match path {
            Path::Serial => w.step(fx(DT)),
            #[cfg(feature = "parallel")]
            Path::Batched => w.step_parallel(fx(DT)),
        }
    }
}

// ---------------------------------------------------------------------------
// Head-on pair
// ---------------------------------------------------------------------------

const R: f64 = 0.5;
const X0: f64 = 1.0;
const V0: f64 = 1.0;

/// Two unit-mass spheres of radius `R` at `x = ∓X0` closing at `±V0` (default
/// gravity acts on both alike and does not touch the `x` axis). An optional
/// pre-solve hook counts its calls and accepts or rejects every contact.
fn head_on(
    config: PhysicsConfig,
    hook: Option<(bool, Arc<AtomicUsize>)>,
) -> (PhysicsWorld, usize, usize) {
    let mut w = PhysicsWorld::new(config);
    let a = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(-X0, 0.0, 0.0), Fix128::ONE).with_velocity(v3(V0, 0.0, 0.0)),
        fx(R),
    );
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(X0, 0.0, 0.0), Fix128::ONE).with_velocity(v3(-V0, 0.0, 0.0)),
        fx(R),
    );
    if let Some((accept, calls)) = hook {
        w.add_pre_solve_hook(Box::new(move |_, _, _c: &Contact| {
            calls.fetch_add(1, Ordering::SeqCst);
            accept
        }));
    }
    (w, a, b)
}

/// Free motion of body `a` after `n` steps: `(x_n, v_n)`.
fn free(d: f64, n: usize) -> (f64, f64) {
    let dn = d.powi(n as i32);
    (-X0 + V0 * DT * (1.0 - dn) / (1.0 - d), V0 * dn)
}

/// Predicted gap of the free pair at the end of substep `s` (1-based) of
/// frame `k` (0-based).
fn free_gap(d: f64, substeps: usize, k: usize, s: usize) -> f64 {
    let h = DT / substeps as f64;
    let (x_start, v) = free(d, k);
    let xa = x_start + v * s as f64 * h;
    -2.0 * xa - 2.0 * R
}

/// The vetoed pair moves as two free bodies: positions and velocities equal
/// the frame-damped free motion, and the spheres pass through each other.
/// The hook was called (the scene does have the contact it removes).
///
/// Tolerance `1e-9`: the only error is fixed-point rounding of the per-substep
/// position update and the velocity re-derivation (`~1e-30` per operation).
#[test]
fn a_vetoed_head_on_pair_moves_as_free_bodies() {
    let config = PhysicsConfig::default();
    let d = config.damping.to_f64();
    let n = 120;
    let (x_want, v_want) = free(d, n);
    assert!(
        x_want > 0.0,
        "the scene must carry a past b: x_a = {x_want}"
    );
    for path in paths() {
        let calls = Arc::new(AtomicUsize::new(0));
        let (mut w, a, b) = head_on(config, Some((false, Arc::clone(&calls))));
        advance(&mut w, path, n);
        let ba = w.get_body(a).expect("a");
        let bb = w.get_body(b).expect("b");
        assert!(calls.load(Ordering::SeqCst) > 0, "{path:?}: no contact");
        for (got, want, what) in [
            (ba.position.x.to_f64(), x_want, "x_a"),
            (bb.position.x.to_f64(), -x_want, "x_b"),
            (ba.velocity.x.to_f64(), v_want, "v_a"),
            (bb.velocity.x.to_f64(), -v_want, "v_b"),
        ] {
            assert!(
                (got - want).abs() < 1e-9,
                "{path:?}: {what} = {got:.12}, free motion {want:.12}"
            );
        }
    }
}

/// Control: without the veto the pair bounces once with restitution `e` on
/// the derived approach velocity, then moves freely:
/// `v_a = −e (v_k − p/(2h)) d^{n−k}` after `n` steps when the impact is in
/// frame `k`. `e = 0.3` is the default material; a modifier that sets
/// `e = 0.8` must give the bounce of `e = 0.8` (the velocity pass reads the
/// modified restitution). Tolerance `1e-9` as above.
#[test]
fn an_accepted_head_on_pair_bounces_with_the_restitution() {
    let config = PhysicsConfig::default();
    let d = config.damping.to_f64();
    let substeps = config.substeps;
    let h = DT / substeps as f64;
    let (k, s) = (0..200)
        .flat_map(|k| (1..=substeps).map(move |s| (k, s)))
        .find(|&(k, s)| free_gap(d, substeps, k, s) < 0.0)
        .expect("impact");
    let p = -free_gap(d, substeps, k, s);
    let n = 60;
    assert!(k < n, "impact frame {k} must be inside the run");
    for (e, modified) in [(0.3, false), (0.8, true)] {
        let leave = -e * (free(d, k).1 - p / (2.0 * h));
        let v_want = leave * d.powi((n - k) as i32);
        for path in paths() {
            let (mut w, a, b) = head_on(config, None);
            if modified {
                w.add_contact_modifier(Box::new(Absolute {
                    friction: fx(0.5),
                    restitution: fx(e),
                }));
            }
            advance(&mut w, path, n);
            let va = w.get_body(a).expect("a").velocity.x.to_f64();
            let vb = w.get_body(b).expect("b").velocity.x.to_f64();
            assert!(
                (va - v_want).abs() < 1e-9 && (vb + v_want).abs() < 1e-9,
                "{path:?}, e = {e}: v_a = {va:.12}, v_b = {vb:.12}, closed form ±{v_want:.12}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding ball
// ---------------------------------------------------------------------------

/// Sets every contact's friction and restitution to absolute values.
struct Absolute {
    friction: Fix128,
    restitution: Fix128,
}

impl ContactModifier for Absolute {
    fn modify_contact(
        &self,
        _a: usize,
        _b: usize,
        _contact: &mut Contact,
        friction: &mut Fix128,
        restitution: &mut Fix128,
    ) -> bool {
        *friction = self.friction;
        *restitution = self.restitution;
        true
    }
}

/// A ball of radius `0.5` that cannot rotate, sliding at 3 m/s along `x` on a
/// large static sphere whose top is the plane `y = 0` (radius `1e4`), with a
/// material of friction `mu` and restitution `0` on both bodies. Returns the
/// ball's `x` velocity after one second.
fn slide(
    iterations: usize,
    mu: f64,
    modifier: Option<Box<dyn ContactModifier>>,
    path: Path,
) -> Fix128 {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        iterations,
        ..PhysicsConfig::default()
    });
    let big = 1.0e4;
    let ground = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -big, 0.0)), fx(big));
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 0.49, 0.0), Fix128::ONE),
        fx(0.5),
    );
    w.bodies[ball].inv_inertia = Vec3Fix::ZERO;
    w.bodies[ball].velocity = v3(3.0, 0.0, 0.0);
    let id = w
        .material_table
        .register(alice_physics::PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
    w.set_body_material(ground, id);
    w.set_body_material(ball, id);
    if let Some(m) = modifier {
        w.add_contact_modifier(m);
    }
    advance(&mut w, path, 60);
    w.get_body(ball).expect("ball").velocity.x
}

/// A modifier that sets `μ = 0.1, e = 0` gives the same velocity pass as a
/// material with `μ = 0.1, e = 0`: the friction the velocity pass applies is
/// the modified value, not the material's `μ = 0.5`.
#[test]
fn an_absolute_friction_modifier_equals_the_material_with_that_friction() {
    for path in paths() {
        let m = Absolute {
            friction: fx(0.1),
            restitution: Fix128::ZERO,
        };
        let modified = slide(1, 0.5, Some(Box::new(m)), path);
        let reference = slide(1, 0.1, None, path);
        assert_eq!(modified, reference, "{path:?}");
        assert_ne!(slide(1, 0.5, None, path), reference, "{path:?}: no teeth");
    }
}
