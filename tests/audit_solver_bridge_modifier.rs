//! Audit oracles for solver: the contact modifier result on the bridge path
//! (AUD-A-S1W2-005)
//!
//! `solve_contact_constraints_with_bridge` runs the contact modifiers on the CPU
//! (Stage A) and `update_velocities` later reads friction and restitution from
//! `PhysicsWorld::contact_constraints`, not from the slice sent to the bridge. The
//! modifier result therefore has to be stored back into the constraint, as the
//! sequential and parallel paths do.
//!
//! Two levels are measured:
//! - state: after one bridge solve the stored friction / restitution equal the
//!   values the modifier assigned
//! - behaviour: under `step_with_bridge`, a modifier that assigns a value gives
//!   bit-identical velocities to a world whose material carries that value from
//!   the start, and the material value itself changes the result (so the
//!   equivalence is not vacuous)
//!
//! The modifiers here assign absolute values, so the result does not depend on how
//! many times per substep the modifier runs.

#![cfg(feature = "gpu-solver-bridge")]

use alice_physics::collider::Contact;
use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{
    ContactConstraint, ContactModifier, PhysicsConfig, PhysicsWorld, RigidBody,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// Returns what it receives: lambdas and positions come back unchanged.
#[derive(Default)]
struct PassThrough {
    calls: usize,
}

impl GpuSolverBridge for PassThrough {
    fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
    fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
    fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
    fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
        Ok(())
    }
    fn send_contact_constraints(&mut self, _c: &[ContactConstraint]) {}
    fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {}
    fn dispatch_contact_solve_iteration(&mut self, _w: Fix128) {
        self.calls += 1;
    }
    fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
    fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
}

struct SetMaterial {
    friction: Option<Fix128>,
    restitution: Option<Fix128>,
}

impl ContactModifier for SetMaterial {
    fn modify_contact(
        &self,
        _a: usize,
        _b: usize,
        _contact: &mut Contact,
        friction: &mut Fix128,
        restitution: &mut Fix128,
    ) -> bool {
        if let Some(f) = self.friction {
            *friction = f;
        }
        if let Some(r) = self.restitution {
            *restitution = r;
        }
        true
    }
}

fn weightless() -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    }
}

// ------------------------------------------------------------------ state

#[test]
fn bridge_solve_stores_the_modified_friction_and_restitution_in_the_constraint() {
    let mut w = PhysicsWorld::new(weightless());
    w.add_body(RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE));
    w.add_body(RigidBody::new(v3(1.0, 0.0, 0.0), Fix128::ONE));
    w.add_contact(ContactConstraint {
        body_a: 0,
        body_b: 1,
        contact: Contact {
            depth: fx(0.25),
            normal: v3(1.0, 0.0, 0.0),
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        },
        friction: fx(0.5),
        restitution: fx(0.25),
        cached_lambda: Fix128::ZERO,
    });
    w.add_contact_modifier(Box::new(SetMaterial {
        friction: Some(Fix128::from_ratio(1, 8)),
        restitution: Some(Fix128::from_ratio(3, 4)),
    }));
    let mut bridge = PassThrough::default();
    w.solve_contact_constraints_with_bridge(&mut bridge);
    assert_eq!(bridge.calls, 1, "the contact must reach the bridge");
    assert_eq!(w.contact_constraints[0].friction, Fix128::from_ratio(1, 8));
    assert_eq!(w.contact_constraints[0].restitution, Fix128::from_ratio(3, 4));
}

#[test]
fn bridge_solve_keeps_the_constraint_values_without_a_modifier() {
    let mut w = PhysicsWorld::new(weightless());
    w.add_body(RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE));
    w.add_body(RigidBody::new(v3(1.0, 0.0, 0.0), Fix128::ONE));
    w.add_contact(ContactConstraint {
        body_a: 0,
        body_b: 1,
        contact: Contact {
            depth: fx(0.25),
            normal: v3(1.0, 0.0, 0.0),
            point_a: Vec3Fix::ZERO,
            point_b: Vec3Fix::ZERO,
        },
        friction: fx(0.5),
        restitution: fx(0.25),
        cached_lambda: Fix128::ZERO,
    });
    let mut bridge = PassThrough::default();
    w.solve_contact_constraints_with_bridge(&mut bridge);
    assert_eq!(w.contact_constraints[0].friction, fx(0.5));
    assert_eq!(w.contact_constraints[0].restitution, fx(0.25));
}

// -------------------------------------------------------------- behaviour

/// Two unit spheres (m = 1) closing head-on at 20 m/s each (overlap 0.1 at the
/// first substep), the first also moving 5 m/s along y, stepped once through the
/// bridge. Returns the velocities of both bodies.
fn bridged_collision(
    friction: f64,
    restitution: f64,
    modifier: Option<SetMaterial>,
) -> (Vec3Fix, Vec3Fix) {
    let mut w = PhysicsWorld::new(weightless());
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
        .register(alice_physics::PhysicsMaterial::new(
            0,
            fx(friction),
            fx(restitution),
        ));
    w.set_body_material(a, id);
    w.set_body_material(b, id);
    if let Some(m) = modifier {
        w.add_contact_modifier(Box::new(m));
    }
    let mut bridge = PassThrough::default();
    w.step_with_bridge(&mut bridge, Fix128::from_ratio(1, 60));
    assert!(bridge.calls > 0, "the contact must reach the bridge");
    (
        w.get_body(a).unwrap().velocity,
        w.get_body(b).unwrap().velocity,
    )
}

#[test]
fn bridged_scene_responds_to_the_material_restitution_and_friction() {
    // without this the equivalences below could hold because nothing is read
    assert_ne!(
        bridged_collision(0.5, 0.0, None),
        bridged_collision(0.5, 1.0, None),
        "restitution must change the bridged step"
    );
    assert_ne!(
        bridged_collision(0.0, 1.0, None),
        bridged_collision(9.0, 1.0, None),
        "friction must change the bridged step"
    );
}

#[test]
fn bridged_modifier_restitution_equals_the_material_restitution() {
    let via_modifier = bridged_collision(
        0.5,
        0.0,
        Some(SetMaterial {
            friction: None,
            restitution: Some(Fix128::ONE),
        }),
    );
    assert_eq!(via_modifier, bridged_collision(0.5, 1.0, None));
}

#[test]
fn bridged_modifier_friction_equals_the_material_friction() {
    let via_modifier = bridged_collision(
        9.0,
        1.0,
        Some(SetMaterial {
            friction: Some(Fix128::ZERO),
            restitution: None,
        }),
    );
    assert_eq!(via_modifier, bridged_collision(0.0, 1.0, None));
}
