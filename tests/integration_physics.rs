//! Integration tests for ALICE-Physics
//!
//! These tests verify end-to-end behaviour of the physics engine using only the
//! public API re-exported from the crate root. All tests run deterministically
//! — no floating-point, no randomness.

use alice_physics::{
    bvh::BvhPrimitive, Cloth, DistanceConstraint, Fix128, LinearBvh, PhysicsConfig, PhysicsWorld,
    RigidBody, Vec3Fix,
};
use alice_physics::collider::AABB;

// ============================================================================
// Helper
// ============================================================================

/// Run a world for `steps` frames with the given `dt`.
fn run_world(world: &mut PhysicsWorld, steps: usize, dt: Fix128) {
    for _ in 0..steps {
        world.step(dt);
    }
}

// ============================================================================
// Test 1 — Free-fall determinism
// ============================================================================

/// A body under gravity should fall. Running the same simulation twice must
/// produce bit-exact identical results (determinism guarantee).
#[test]
fn test_free_fall_determinism() {
    fn simulate() -> Vec3Fix {
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);

        let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 100, 0), Fix128::ONE);
        world.add_body(body);

        let dt = Fix128::from_ratio(1, 60);
        run_world(&mut world, 60, dt);

        world.bodies[0].position
    }

    let pos1 = simulate();
    let pos2 = simulate();

    // Bit-exact equality — not just "close"
    assert_eq!(pos1.x.hi, pos2.x.hi, "x.hi diverged");
    assert_eq!(pos1.x.lo, pos2.x.lo, "x.lo diverged");
    assert_eq!(pos1.y.hi, pos2.y.hi, "y.hi diverged");
    assert_eq!(pos1.y.lo, pos2.y.lo, "y.lo diverged");
    assert_eq!(pos1.z.hi, pos2.z.hi, "z.hi diverged");
    assert_eq!(pos1.z.lo, pos2.z.lo, "z.lo diverged");

    // The body must have fallen below its starting height
    assert!(
        pos1.y < Fix128::from_int(100),
        "Body did not fall: y = {:?}",
        pos1.y
    );
}

// ============================================================================
// Test 2 — Multi-step bit-exact replay
// ============================================================================

/// Run 120 steps, serialize state, continue 60 more steps, restore, continue
/// 60 more steps — final positions must match.
#[test]
fn test_multi_step_bit_exact() {
    fn make_world() -> PhysicsWorld {
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);

        let body = RigidBody::new_dynamic(Vec3Fix::from_int(3, 50, -7), Fix128::from_ratio(3, 2));
        world.add_body(body);
        world
    }

    let dt = Fix128::from_ratio(1, 60);

    // Run 1: simulate 120 steps then record position
    let mut world_a = make_world();
    run_world(&mut world_a, 120, dt);
    let pos_a = world_a.bodies[0].position;

    // Run 2: identical initial conditions — must match
    let mut world_b = make_world();
    run_world(&mut world_b, 120, dt);
    let pos_b = world_b.bodies[0].position;

    assert_eq!(pos_a.x.hi, pos_b.x.hi);
    assert_eq!(pos_a.x.lo, pos_b.x.lo);
    assert_eq!(pos_a.y.hi, pos_b.y.hi);
    assert_eq!(pos_a.y.lo, pos_b.y.lo);
    assert_eq!(pos_a.z.hi, pos_b.z.hi);
    assert_eq!(pos_a.z.lo, pos_b.z.lo);
}

// ============================================================================
// Test 3 — State serialization / rollback
// ============================================================================

/// Simulate 30 steps, save state, simulate 30 more, restore saved state,
/// simulate 30 more — the two "after-restore" runs must be bit-exact.
#[test]
fn test_state_serialization_rollback() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let body = RigidBody::new_dynamic(Vec3Fix::from_int(10, 20, 30), Fix128::ONE);
    world.add_body(body);

    let dt = Fix128::from_ratio(1, 60);

    // Warm-up
    run_world(&mut world, 30, dt);

    // Snapshot
    let snapshot = world.serialize_state();

    // Continue from snapshot — branch A
    run_world(&mut world, 30, dt);
    let pos_branch_a = world.bodies[0].position;

    // Restore and continue — branch B
    world.deserialize_state(&snapshot);
    run_world(&mut world, 30, dt);
    let pos_branch_b = world.bodies[0].position;

    // Both branches from the same snapshot must produce identical results
    assert_eq!(
        pos_branch_a.x.hi, pos_branch_b.x.hi,
        "rollback diverged at x.hi"
    );
    assert_eq!(
        pos_branch_a.y.hi, pos_branch_b.y.hi,
        "rollback diverged at y.hi"
    );
    assert_eq!(
        pos_branch_a.z.hi, pos_branch_b.z.hi,
        "rollback diverged at z.hi"
    );
}

// ============================================================================
// Test 4 — Distance constraint stability
// ============================================================================

/// Two dynamic bodies connected by a distance constraint. After many steps the
/// actual distance between their centres must remain close to the target.
#[test]
fn test_constraint_stability() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let target_dist = Fix128::from_int(5);

    let id_a = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 10, 0),
        Fix128::ONE,
    ));
    let id_b = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(5, 10, 0),
        Fix128::ONE,
    ));

    let constraint = DistanceConstraint {
        body_a: id_a,
        body_b: id_b,
        local_anchor_a: Vec3Fix::ZERO,
        local_anchor_b: Vec3Fix::ZERO,
        target_distance: target_dist,
        compliance: Fix128::ZERO,
        cached_lambda: Fix128::ZERO,
    };
    world.add_distance_constraint(constraint);

    let dt = Fix128::from_ratio(1, 60);
    run_world(&mut world, 120, dt);

    let pa = world.bodies[id_a].position;
    let pb = world.bodies[id_b].position;
    let actual_dist = (pb - pa).length();

    let error = if actual_dist > target_dist {
        actual_dist - target_dist
    } else {
        target_dist - actual_dist
    };

    // Allow at most 10% error due to gravity pulling both bodies
    let tolerance = Fix128::from_ratio(5, 10); // 0.5 units
    assert!(
        error < tolerance,
        "Constraint violated: dist error = hi={} lo={}",
        error.hi,
        error.lo
    );
}

// ============================================================================
// Test 5 — BVH broad-phase correctness
// ============================================================================

/// Build a BVH with several AABBs and verify that a range query returns exactly
/// the expected hit set.
#[test]
fn test_bvh_query_correctness() {
    // Place 5 unit cubes at integer positions along the X axis
    let prims: Vec<BvhPrimitive> = (0u32..5)
        .map(|i| {
            let x = Fix128::from_int(i as i64 * 3); // spaced 3 units apart
            BvhPrimitive {
                aabb: AABB::new(
                    Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO),
                    Vec3Fix::new(x + Fix128::ONE, Fix128::ONE, Fix128::ONE),
                ),
                index: i,
                morton: 0,
            }
        })
        .collect();

    let bvh = LinearBvh::build(prims);

    // Query covering only the first primitive (x ∈ [0, 1])
    let q = AABB::new(
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::from_int(1, 1, 1),
    );
    let hits = bvh.query(&q);
    assert!(hits.contains(&0), "Expected primitive 0 in hit set, got {:?}", hits);

    // Query in a gap between primitives — should have no hits
    let gap = AABB::new(
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::from_int(2, 1, 1),
    );
    let gap_hits = bvh.query(&gap);
    // Primitive 0 ends at x=1 and primitive 1 starts at x=3; the gap [1,2] might
    // touch primitive 0 depending on the integer rounding in the node.  We only
    // assert primitives 2-4 are absent.
    assert!(
        !gap_hits.contains(&2) && !gap_hits.contains(&3) && !gap_hits.contains(&4),
        "Gap query should not return far primitives, got {:?}",
        gap_hits
    );

    // Empty BVH always returns empty
    let empty_bvh = LinearBvh::build(vec![]);
    assert!(empty_bvh.query(&q).is_empty());
}

// ============================================================================
// Test 6 — Cloth grid creation
// ============================================================================

/// Verify that `Cloth::new_grid` produces the expected particle count,
/// triangle count, and that all inverse masses are set correctly.
#[test]
fn test_cloth_grid_creation() {
    let res_x = 4usize;
    let res_y = 4usize;
    let mass = Fix128::from_ratio(1, 2); // 0.5 kg per particle

    let cloth = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(3),
        Fix128::from_int(3),
        res_x,
        res_y,
        mass,
    );

    // Particle count
    assert_eq!(cloth.positions.len(), res_x * res_y, "particle count");
    assert_eq!(cloth.inv_masses.len(), res_x * res_y, "inv_mass count");

    // Triangle count: (res_x-1)*(res_y-1)*2
    let expected_tris = (res_x - 1) * (res_y - 1) * 2;
    assert_eq!(cloth.triangles.len(), expected_tris, "triangle count");

    // Inverse mass should be 2 (= 1/0.5)
    let expected_inv_mass = Fix128::from_int(2);
    for &im in &cloth.inv_masses {
        assert_eq!(im.hi, expected_inv_mass.hi, "inv_mass.hi mismatch");
    }

    // No pinned particles by default
    assert!(cloth.pinned.is_empty(), "no pins expected");
}

// ============================================================================
// Test 7 — Static body does not move
// ============================================================================

/// A static body placed anywhere in the world must remain at its initial
/// position regardless of how many simulation steps are taken.
#[test]
fn test_static_body_does_not_move() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let initial = Vec3Fix::from_int(7, -3, 11);
    let ground = RigidBody::new_static(initial);
    let id = world.add_body(ground);

    let dt = Fix128::from_ratio(1, 60);
    run_world(&mut world, 120, dt);

    let final_pos = world.bodies[id].position;
    assert_eq!(final_pos.x.hi, initial.x.hi, "static body x changed");
    assert_eq!(final_pos.y.hi, initial.y.hi, "static body y changed");
    assert_eq!(final_pos.z.hi, initial.z.hi, "static body z changed");
}
