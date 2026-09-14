//! Graph coloring soundness for the `parallel` constraint solver (v1.0.1).
//!
//! Pre-1.0.1 the per-body color mask was a single `u64`; once a body took
//! part in 64 colors every further constraint on it was assigned color 64
//! *without* recording occupancy, so batch 64 could hold many constraints
//! sharing one body and the parallel solver handed out aliasing `&mut`
//! references. This suite pins the observable consequences through the
//! public API (`rebuild_batches` / `num_batches` / `step`).

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{DistanceConstraint, PhysicsWorld, RigidBody, SolverConfig};

fn hub_world(hub: RigidBody, spokes: usize) -> PhysicsWorld {
    let mut world = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    });
    let hub_idx = world.add_body(hub);
    for i in 0..spokes {
        let spoke = world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(i as i64 + 1, 0, 0),
            Fix128::ONE,
        ));
        world.add_distance_constraint(DistanceConstraint::new(
            hub_idx,
            spoke,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::ONE,
        ));
    }
    world
}

#[test]
fn dynamic_hub_needs_one_batch_per_constraint_beyond_64() {
    for spokes in [64usize, 65, 70, 200] {
        let mut world = hub_world(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), spokes);
        world.rebuild_batches();
        assert_eq!(world.num_batches(), spokes, "spokes = {spokes}");
    }
}

#[test]
fn static_hub_collapses_to_a_single_batch() {
    let mut world = hub_world(RigidBody::new_static(Vec3Fix::ZERO), 200);
    world.rebuild_batches();
    assert_eq!(world.num_batches(), 1);
}

#[test]
fn stepping_static_hub_scene_converges_and_stays_finite() {
    // 70 spokes on a static hub, target distance 1 from initial distance
    // i+1: every spoke must be pulled onto the unit sphere around the hub.
    let mut world = hub_world(RigidBody::new_static(Vec3Fix::ZERO), 70);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..90 {
        world.step(dt);
    }
    let hub = world.bodies[0].position;
    for body in &world.bodies[1..] {
        let d = (body.position - hub).length().to_f32();
        assert!(
            (d - 1.0).abs() < 0.05,
            "spoke did not converge to the hub sphere: distance = {d}"
        );
    }
}
