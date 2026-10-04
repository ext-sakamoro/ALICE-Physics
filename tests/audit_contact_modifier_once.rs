//! Oracles for AUD-A-S34-001: a `ContactModifier` runs once per substep, not
//! once per solver iteration.
//!
//! Expected values are closed forms, not values copied from the solver:
//! - the modifier call count is `substeps x contacts`;
//! - the friction a solver pass receives is `mu / 2` on every iteration, so
//!   the velocities after a step with `friction *= 0.5` equal those of a world
//!   whose material has `mu / 2` from the start, for every `iterations`.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]
#![cfg(feature = "std")]

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{ContactModifier, PhysicsConfig, PhysicsWorld, RigidBody};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn config(substeps: usize, iterations: usize) -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps,
        iterations,
        ..PhysicsConfig::default()
    }
}

/// Multiplies the friction by `factor` and counts its calls.
struct ScaleFriction {
    factor: Fix128,
    calls: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Fix128>>>,
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
        self.seen.lock().unwrap().push(*friction);
        *friction = *friction * self.factor;
        true
    }
}

fn scale_friction(factor: Fix128) -> (ScaleFriction, Arc<AtomicUsize>, Arc<Mutex<Vec<Fix128>>>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    (
        ScaleFriction {
            factor,
            calls: Arc::clone(&calls),
            seen: Arc::clone(&seen),
        },
        calls,
        seen,
    )
}

/// Two unit spheres closing head-on at 20 m/s each (overlap 0.1 at the first
/// substep), the first also moving 5 m/s along y, restitution 1, friction `mu`
/// on both materials. Returns the two velocities after one step.
fn collision(iterations: usize, mu: f64, modifier: Option<ScaleFriction>) -> (Vec3Fix, Vec3Fix) {
    let mut w = PhysicsWorld::new(config(1, iterations));
    let a = w.add_body_with_radius(
        RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE).with_velocity(v3(20.0, 5.0, 0.0)),
        Fix128::ONE,
    );
    let b = w.add_body_with_radius(
        RigidBody::new(v3(2.1, 0.0, 0.0), Fix128::ONE).with_velocity(v3(-20.0, 0.0, 0.0)),
        Fix128::ONE,
    );
    let id = w
        .material_table
        .register(alice_physics::PhysicsMaterial::new(0, fx(mu), Fix128::ONE));
    w.set_body_material(a, id);
    w.set_body_material(b, id);
    if let Some(m) = modifier {
        w.add_contact_modifier(Box::new(m));
    }
    w.step(dt());
    (
        w.get_body(a).unwrap().velocity,
        w.get_body(b).unwrap().velocity,
    )
}

/// Closed form: with `friction *= 1/2` applied once, the effective friction is
/// `mu / 2` whatever the iteration count, so the result is bit-identical to a
/// material that has `mu / 2` from the start.
#[test]
fn a_relative_friction_modifier_does_not_compound_with_iterations() {
    for iterations in [1usize, 2, 4, 8] {
        let (m, calls, seen) = scale_friction(Fix128::from_ratio(1, 2));
        let modified = collision(iterations, 0.2, Some(m));
        let reference = collision(iterations, 0.1, None);
        assert_eq!(
            modified, reference,
            "iterations {iterations}: modifier run must equal the mu/2 material"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "iterations {iterations}: one substep, one contact, one call"
        );
        assert_eq!(
            *seen.lock().unwrap(),
            vec![fx(0.2)],
            "iterations {iterations}: the modifier sees the material friction"
        );
    }
}

/// The scene has teeth: mu = 0.2 and mu = 0.1 give different velocities, so
/// the bit-equality above can fail.
#[test]
fn the_scene_is_sensitive_to_friction() {
    let hi = collision(4, 0.2, None);
    let lo = collision(4, 0.1, None);
    assert_ne!(hi, lo);
}

/// Closed form: a modifier is called `substeps x contacts` times per step.
/// Three bodies in a row (a-b and b-c overlap, a-c do not) at rest. The
/// modifier sets the depth to zero, so nothing separates and both contacts
/// exist in every substep.
#[test]
fn a_modifier_is_called_once_per_substep_per_contact() {
    struct ZeroDepth(Arc<AtomicUsize>);
    impl ContactModifier for ZeroDepth {
        fn modify_contact(
            &self,
            _a: usize,
            _b: usize,
            contact: &mut Contact,
            _f: &mut Fix128,
            _r: &mut Fix128,
        ) -> bool {
            self.0.fetch_add(1, Ordering::SeqCst);
            contact.depth = Fix128::ZERO;
            true
        }
    }
    for (substeps, iterations) in [(1usize, 4usize), (3, 4), (3, 1), (2, 8)] {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut w = PhysicsWorld::new(config(substeps, iterations));
        for x in [-1.5, 0.0, 1.5] {
            w.add_body_with_radius(RigidBody::new(v3(x, 0.0, 0.0), Fix128::ONE), Fix128::ONE);
        }
        w.add_contact_modifier(Box::new(ZeroDepth(Arc::clone(&calls))));
        w.step(dt());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            substeps * 2,
            "substeps {substeps}, iterations {iterations}"
        );
    }
}

/// An absolute assignment is unaffected by the move (guards the existing
/// semantics): friction := 0 gives the same result for every iteration count
/// as a material with friction 0.
#[test]
fn an_absolute_friction_assignment_still_applies() {
    for iterations in [1usize, 4] {
        let (m, _, _) = scale_friction(Fix128::ZERO);
        assert_eq!(
            collision(iterations, 0.9, Some(m)),
            collision(iterations, 0.0, None),
            "iterations {iterations}"
        );
    }
}

#[cfg(feature = "gpu-solver-bridge")]
mod bridge {
    use super::*;
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::solver::ContactConstraint;

    /// Leaves every buffer untouched and records the friction of each
    /// constraint it is sent, one entry per dispatch.
    struct Recording {
        frictions: Arc<Mutex<Vec<Fix128>>>,
        pending: Vec<Fix128>,
    }

    impl GpuSolverBridge for Recording {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, c: &[ContactConstraint]) {
            self.pending = c.iter().map(|k| k.friction).collect();
        }
        fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {}
        fn dispatch_contact_solve_iteration(&mut self, _w: Fix128) {
            self.frictions.lock().unwrap().extend(self.pending.iter());
        }
        fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
        fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
    }

    fn world(iterations: usize) -> PhysicsWorld {
        let mut w = PhysicsWorld::new(config(1, iterations));
        let a = w.add_body_with_radius(
            RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE).with_velocity(v3(20.0, 5.0, 0.0)),
            Fix128::ONE,
        );
        let b = w.add_body_with_radius(
            RigidBody::new(v3(2.1, 0.0, 0.0), Fix128::ONE).with_velocity(v3(-20.0, 0.0, 0.0)),
            Fix128::ONE,
        );
        let id = w
            .material_table
            .register(alice_physics::PhysicsMaterial::new(0, fx(0.2), Fix128::ONE));
        w.set_body_material(a, id);
        w.set_body_material(b, id);
        w
    }

    /// Closed form: every iteration's dispatch receives `mu / 2 = 0.1`, and
    /// the modifier is called once per substep (not once per iteration).
    #[test]
    fn step_with_bridge_sends_the_halved_friction_on_every_iteration() {
        for iterations in [1usize, 2, 4, 8] {
            let frictions = Arc::new(Mutex::new(Vec::new()));
            let mut bridge = Recording {
                frictions: Arc::clone(&frictions),
                pending: Vec::new(),
            };
            let (m, calls, seen) = scale_friction(Fix128::from_ratio(1, 2));
            let mut w = world(iterations);
            w.add_contact_modifier(Box::new(m));
            w.step_with_bridge(&mut bridge, dt());
            assert_eq!(
                *frictions.lock().unwrap(),
                vec![fx(0.1); iterations],
                "iterations {iterations}"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1, "iterations {iterations}");
            assert_eq!(*seen.lock().unwrap(), vec![fx(0.2)]);
        }
    }

    /// The bridge installed on the world takes the same route through `step`.
    #[test]
    fn an_installed_bridge_sees_the_halved_friction_on_every_iteration() {
        for iterations in [1usize, 4] {
            let frictions = Arc::new(Mutex::new(Vec::new()));
            let bridge = Recording {
                frictions: Arc::clone(&frictions),
                pending: Vec::new(),
            };
            let (m, calls, _) = scale_friction(Fix128::from_ratio(1, 2));
            let mut w = world(iterations);
            w.add_contact_modifier(Box::new(m));
            w.set_gpu_solver_bridge(Some(Box::new(bridge)));
            w.step(dt());
            assert_eq!(
                *frictions.lock().unwrap(),
                vec![fx(0.1); iterations],
                "iterations {iterations}"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1, "iterations {iterations}");
        }
    }
}
