//! Oracles for how often the contact filters (pre-solve hooks and contact
//! modifiers) run: once per contact per substep, ahead of the solver
//! iterations, in both the serial path (`step`) and the batched path
//! (`step_parallel`), so a relative modifier does not compound with
//! `iterations` in either path.
//!
//! Every scene runs through `PhysicsWorld::step` and, with the `parallel`
//! feature, `PhysicsWorld::step_parallel`. Expected values are closed forms
//! written from the scene's geometry, never obtained by running the solver:
//!
//! * frame damping: each `step` of `dt` multiplies every dynamic body's
//!   velocity by `d = config.damping` once, so a body with no contact moving
//!   at `v₀` along `x` has `v_n = v₀ dⁿ` and
//!   `x_n = x₀ + v₀ dt (1 − dⁿ) / (1 − d)` after `n` steps;
//! * a hook that rejects every contact leaves the pair free, so the number of
//!   substeps in which it overlaps follows from that free motion;
//! * a modifier `friction *= 1/2` applied once gives `μ_eff = μ/2`, the same
//!   as a material with `μ/2` from the start.
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

/// Number of substeps in the first `n` steps in which the free pair overlaps.
fn overlapping_substeps(d: f64, substeps: usize, n: usize) -> usize {
    (0..n)
        .flat_map(|k| (1..=substeps).map(move |s| (k, s)))
        .filter(|&(k, s)| free_gap(d, substeps, k, s) < 0.0)
        .count()
}

/// A hook runs once per contact per substep in both paths, whatever the
/// iteration count: with a rejecting hook the pair moves freely, so the call
/// count is the number of substeps in which the free pair overlaps.
#[test]
fn a_hook_runs_once_per_substep_in_both_paths() {
    for iterations in [1usize, 4] {
        let config = PhysicsConfig {
            iterations,
            ..PhysicsConfig::default()
        };
        let d = config.damping.to_f64();
        let n = 120;
        let want = overlapping_substeps(d, config.substeps, n);
        assert!(want > 0);
        for path in paths() {
            let calls = Arc::new(AtomicUsize::new(0));
            let (mut w, _, _) = head_on(config, Some((false, Arc::clone(&calls))));
            advance(&mut w, path, n);
            assert_eq!(
                calls.load(Ordering::SeqCst),
                want,
                "{path:?}, iterations {iterations}: hook calls"
            );
        }
    }
}

/// An accepting hook leaves the bounce unchanged (bit-identical to no hook)
/// in both paths and is called once per overlapping substep: the pair
/// overlaps only in the impact substep, so once.
#[test]
fn an_accepting_hook_changes_nothing() {
    for iterations in [1usize, 4] {
        let config = PhysicsConfig {
            iterations,
            ..PhysicsConfig::default()
        };
        for path in paths() {
            let calls = Arc::new(AtomicUsize::new(0));
            let (mut hooked, a, _) = head_on(config, Some((true, Arc::clone(&calls))));
            let (mut plain, _, _) = head_on(config, None);
            advance(&mut hooked, path, 60);
            advance(&mut plain, path, 60);
            assert_eq!(
                hooked.get_body(a).expect("a").velocity,
                plain.get_body(a).expect("a").velocity,
                "{path:?}, iterations {iterations}"
            );
            assert_eq!(
                calls.load(Ordering::SeqCst),
                1,
                "{path:?}, iterations {iterations}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Sliding ball
// ---------------------------------------------------------------------------

/// Multiplies every contact's friction by `factor` and counts its calls.
struct ScaleFriction {
    factor: Fix128,
    calls: Arc<AtomicUsize>,
}

impl ContactModifier for ScaleFriction {
    fn modify_contact(
        &self,
        _a: usize,
        _b: usize,
        _contact: &mut Contact,
        friction: &mut Fix128,
        _restitution: &mut Fix128,
    ) -> bool {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *friction = *friction * self.factor;
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

/// A relative modifier `friction *= 1/2` applied once per substep gives the
/// effective friction `μ_eff = μ/2` whatever the iteration count, so the run
/// is bit-identical to a material with `μ/2` in both paths, and the modifier
/// is called once per substep in which the ball touches the ground.
#[test]
fn a_relative_friction_modifier_does_not_compound_in_either_path() {
    for iterations in [1usize, 4] {
        let mut counts = Vec::new();
        for path in paths() {
            let calls = Arc::new(AtomicUsize::new(0));
            let m = ScaleFriction {
                factor: Fix128::from_ratio(1, 2),
                calls: Arc::clone(&calls),
            };
            let modified = slide(iterations, 0.4, Some(Box::new(m)), path);
            let reference = slide(iterations, 0.2, None, path);
            assert_eq!(modified, reference, "{path:?}, iterations {iterations}");
            assert_ne!(
                slide(iterations, 0.4, None, path),
                reference,
                "{path:?}: no teeth"
            );
            counts.push(calls.load(Ordering::SeqCst));
        }
        assert!(counts[0] > 0);
        assert!(
            counts.iter().all(|&c| c == counts[0]),
            "iterations {iterations}: modifier calls per path {counts:?}"
        );
    }
}
