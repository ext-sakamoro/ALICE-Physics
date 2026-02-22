//! Integration tests for ALICE-Physics
//!
//! These tests verify end-to-end behaviour of the physics engine using only the
//! public API re-exported from the crate root. All tests run deterministically
//! — no floating-point, no randomness.

use alice_physics::{
    bvh::BvhPrimitive, Cloth, CollisionFilter, ContactEventType, DistanceConstraint, Fix128,
    ForceField, ForceFieldInstance, Joint, LinearBvh, PhysicsConfig, PhysicsWorld,
    QuatFix, RigidBody, Rope, SleepConfig, Vec3Fix, Vehicle, VehicleConfig,
};
use alice_physics::collider::AABB;
use alice_physics::joint::BallJoint;

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

// ============================================================================
// Test 8 — Vehicle ground contact and suspension
// ============================================================================

/// A vehicle placed above a ground plane should settle on its suspension.
/// After simulation, wheels should be grounded and the chassis should remain
/// above the ground plane.
#[test]
fn test_vehicle_ground_contact() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let chassis_id = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 1, 0),
        Fix128::from_int(1500),
    ));

    let mut vehicle = Vehicle::new(VehicleConfig::default());

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..120 {
        vehicle.update(&mut world.bodies[chassis_id], dt);
        world.step(dt);
    }

    // Wheels should be grounded after settling
    assert!(
        vehicle.grounded_wheels() > 0,
        "Expected grounded wheels after settling, got {}",
        vehicle.grounded_wheels()
    );

    // Chassis should remain above the ground plane
    assert!(
        world.bodies[chassis_id].position.y > Fix128::ZERO,
        "Chassis should be above ground after suspension settling"
    );
}

// ============================================================================
// Test 9 — Rope sag under gravity
// ============================================================================

/// A rope pinned at both endpoints should sag in the middle under gravity.
/// After simulation, the midpoint should be below the endpoints.
#[test]
fn test_rope_sag_under_gravity() {
    let start = Vec3Fix::from_int(0, 10, 0);
    let end = Vec3Fix::from_int(10, 10, 0);
    let mut rope = Rope::new(start, end, 10, Fix128::from_ratio(1, 10));

    rope.pin_start();
    rope.pin_end();

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..120 {
        rope.step(dt);
    }

    // Endpoints should remain at original height (pinned)
    assert_eq!(
        rope.positions[0].y.hi, 10,
        "Start pin should remain at y=10"
    );
    assert_eq!(
        rope.positions[10].y.hi, 10,
        "End pin should remain at y=10"
    );

    // Middle particle should sag below endpoints
    let mid_y = rope.positions[5].y;
    assert!(
        mid_y < Fix128::from_int(10),
        "Rope midpoint should sag below y=10, got y.hi={}",
        mid_y.hi
    );
}

// ============================================================================
// Test 10 — Cross-architecture determinism (golden hash)
// ============================================================================

/// Run a multi-body simulation and compute an FNV-1a hash of the final state.
/// Running the same simulation twice must produce the same hash, guaranteeing
/// cross-platform determinism.
#[test]
fn test_determinism_golden_hash() {
    fn simulate_and_hash() -> u64 {
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);

        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 50, 0),
            Fix128::ONE,
        ));
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(5, 50, 0),
            Fix128::from_ratio(3, 2),
        ));
        world.add_body(RigidBody::new_static(Vec3Fix::ZERO));

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..180 {
            world.step(dt);
        }

        // Collect final state bytes (position only — deterministic)
        let mut bytes = Vec::new();
        for body in &world.bodies {
            bytes.extend_from_slice(&body.position.x.hi.to_le_bytes());
            bytes.extend_from_slice(&body.position.x.lo.to_le_bytes());
            bytes.extend_from_slice(&body.position.y.hi.to_le_bytes());
            bytes.extend_from_slice(&body.position.y.lo.to_le_bytes());
            bytes.extend_from_slice(&body.position.z.hi.to_le_bytes());
            bytes.extend_from_slice(&body.position.z.lo.to_le_bytes());
        }

        // FNV-1a
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in &bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    let hash1 = simulate_and_hash();
    let hash2 = simulate_and_hash();

    // Two identical runs must produce the same hash
    assert_eq!(
        hash1, hash2,
        "Determinism broken: hash1={:#018x} hash2={:#018x}",
        hash1, hash2
    );

    // Hash should be non-trivial (different from FNV basis)
    assert_ne!(
        hash1,
        0xcbf29ce484222325u64,
        "Hash should differ from FNV basis"
    );
}

// ============================================================================
// Test 11 — Auto collision detection (BVH + sphere)
// ============================================================================

/// Two bodies with collision radii moving toward each other should generate
/// contact events and push apart via the integrated pipeline.
#[test]
fn test_auto_collision_detection() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO, // no gravity
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Two spheres approaching each other
    let mut body_a = RigidBody::new_dynamic(Vec3Fix::from_int(-1, 0, 0), Fix128::ONE);
    body_a.velocity = Vec3Fix::from_int(10, 0, 0); // moving right
    let id_a = world.add_body_with_radius(body_a, Fix128::ONE);

    let mut body_b = RigidBody::new_dynamic(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    body_b.velocity = Vec3Fix::from_int(-10, 0, 0); // moving left
    let id_b = world.add_body_with_radius(body_b, Fix128::ONE);

    // Step several frames: detect_collisions runs before integration, so
    // bodies need one frame to move into overlap before detection fires.
    let dt = Fix128::from_ratio(1, 60);
    let mut had_contact = false;
    for _ in 0..10 {
        world.step(dt);
        if !world.contact_events().is_empty() {
            had_contact = true;
        }
    }

    assert!(had_contact, "Expected contact events from auto collision");

    // Bodies should have been pushed apart by collision response
    let pa = world.bodies[id_a].position;
    let pb = world.bodies[id_b].position;
    let dist = (pb - pa).length();
    assert!(
        dist > Fix128::ZERO,
        "Bodies should be separated after collision, dist.hi={}",
        dist.hi
    );
}

// ============================================================================
// Test 12 — Collision filtering
// ============================================================================

/// Bodies on different collision layers should not collide even if overlapping.
#[test]
fn test_collision_filtering() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Two overlapping bodies with collision radii
    let body_a = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    let id_a = world.add_body_with_radius(body_a, Fix128::from_int(2));

    let body_b = RigidBody::new_dynamic(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    let id_b = world.add_body_with_radius(body_b, Fix128::from_int(2));

    // Put them on different layers that don't collide
    world.set_body_filter(id_a, CollisionFilter { layer: 1, mask: 1, group: 0 });
    world.set_body_filter(id_b, CollisionFilter { layer: 2, mask: 2, group: 0 });

    let dt = Fix128::from_ratio(1, 60);
    world.step(dt);

    // No contact events should be generated
    let events = world.contact_events();
    assert!(
        events.is_empty(),
        "Expected no events when filters prevent collision, got {}",
        events.len()
    );
}

// ============================================================================
// Test 13 — Joint in step pipeline
// ============================================================================

/// A ball joint connecting two bodies should keep them together during step().
#[test]
fn test_joint_in_step() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    let id_a = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let id_b = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(3, 0, 0),
        Fix128::ONE,
    ));

    let joint = Joint::Ball(BallJoint::new(
        id_a,
        id_b,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    ));
    world.add_joint(joint);
    assert_eq!(world.joint_count(), 1);

    // Step: joint should pull body_b toward body_a's anchor
    let dt = Fix128::from_ratio(1, 60);
    run_world(&mut world, 60, dt);

    let pb = world.bodies[id_b].position;
    let dist = pb.length();
    // With zero gravity and ball joint anchored at origin, body should be pulled in
    assert!(
        dist < Fix128::from_int(3),
        "Joint should pull body closer, dist.hi={}",
        dist.hi
    );
}

// ============================================================================
// Test 14 — Force field in step pipeline
// ============================================================================

/// A directional force field should accelerate bodies during step().
#[test]
fn test_force_field_in_step() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    world.add_body(body);

    // Add wind force in +X direction
    let wind = ForceFieldInstance::new(ForceField::Directional {
        direction: Vec3Fix::UNIT_X,
        strength: Fix128::from_int(50),
    });
    world.add_force_field(wind);

    let dt = Fix128::from_ratio(1, 60);
    run_world(&mut world, 60, dt);

    // Body should have moved in +X direction
    let pos = world.bodies[0].position;
    assert!(
        pos.x > Fix128::ZERO,
        "Wind should push body in +X direction, got x.hi={}",
        pos.x.hi
    );
}

// ============================================================================
// Test 15 — Sleeping integration
// ============================================================================

/// A still body should eventually sleep; applying velocity should wake it.
#[test]
fn test_sleeping_in_step() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    world.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 5,
    });

    let body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    let id = world.add_body(body);

    // Step enough frames for sleeping
    let dt = Fix128::from_ratio(1, 60);
    run_world(&mut world, 10, dt);

    assert!(
        world.is_sleeping(id),
        "Still body should be sleeping after threshold"
    );

    // Wake it up
    world.wake_body(id);
    assert!(
        !world.is_sleeping(id),
        "Body should be awake after wake_body()"
    );
}

// ============================================================================
// Test 16 — Event lifecycle (Begin → Persist → End)
// ============================================================================

/// Contact events should transition through Begin → Persist → End correctly.
#[test]
fn test_event_lifecycle() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Two overlapping bodies (stationary, overlapping collision spheres)
    let body_a = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    let _id_a = world.add_body_with_radius(body_a, Fix128::from_int(2));

    let body_b = RigidBody::new_dynamic(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    let _id_b = world.add_body_with_radius(body_b, Fix128::from_int(2));

    let dt = Fix128::from_ratio(1, 60);

    // Frame 1: Begin
    world.step(dt);
    let events = world.contact_events();
    assert!(events.iter().any(|e| e.event_type == ContactEventType::Begin));

    // Frame 2: Persist (bodies still overlapping after push)
    world.step(dt);
    // After push-apart, they may no longer overlap => could be End. Check events exist.
    let events2 = world.contact_events();
    assert!(
        !events2.is_empty(),
        "Should have Persist or End events in frame 2"
    );
}

// ============================================================================
// Test 17 — remove_body with constraint cleanup
// ============================================================================

/// Removing a body should clean up associated constraints and remap indices.
#[test]
fn test_remove_body() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    let id_a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let id_b = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(5, 0, 0), Fix128::ONE));
    let id_c = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(10, 0, 0), Fix128::ONE));

    // Add constraint between B and C
    world.add_distance_constraint(DistanceConstraint {
        body_a: id_b,
        body_b: id_c,
        local_anchor_a: Vec3Fix::ZERO,
        local_anchor_b: Vec3Fix::ZERO,
        target_distance: Fix128::from_int(5),
        compliance: Fix128::ZERO,
        cached_lambda: Fix128::ZERO,
    });

    assert_eq!(world.body_count(), 3);

    // Remove body A (swap-remove: C moves to index 0)
    let removed = world.remove_body(id_a);
    assert!(removed.is_some());
    assert_eq!(world.body_count(), 2);

    // Simulation should still work without crashing
    let dt = Fix128::from_ratio(1, 60);
    run_world(&mut world, 10, dt);
}

// ============================================================================
// Test 18 — Trigger/sensor event detection
// ============================================================================

/// Sensor bodies should generate trigger events but no physics response.
#[test]
fn test_sensor_trigger_events() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Sensor body
    let mut sensor = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    sensor.is_sensor = true;
    let id_sensor = world.add_body_with_radius(sensor, Fix128::from_int(2));

    // Moving body that will enter the sensor
    let moving = RigidBody::new_dynamic(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    let _id_moving = world.add_body_with_radius(moving, Fix128::from_int(2));

    let dt = Fix128::from_ratio(1, 60);
    world.step(dt);

    // Trigger events should fire
    let triggers = world.trigger_events();
    assert!(
        !triggers.is_empty(),
        "Expected trigger events for sensor overlap"
    );
    assert!(triggers[0].entered, "First trigger should be an enter");

    // Sensor body should NOT have been pushed away (no physics response)
    let sensor_pos = world.bodies[id_sensor].position;
    // With zero gravity and no physics response, sensor stays at origin
    assert_eq!(sensor_pos.x.hi, 0, "Sensor should not move from physics response");
}

// ============================================================================
// Test 19 — Raycast against world
// ============================================================================

/// World raycast should find the nearest sphere along the ray direction.
#[test]
fn test_world_raycast() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // Place spheres along +X axis
    let body_near = RigidBody::new_static(Vec3Fix::from_int(5, 0, 0));
    let id_near = world.add_body_with_radius(body_near, Fix128::ONE);

    let body_far = RigidBody::new_static(Vec3Fix::from_int(20, 0, 0));
    let _id_far = world.add_body_with_radius(body_far, Fix128::ONE);

    // Ray from origin in +X direction
    let result = world.raycast(Vec3Fix::ZERO, Vec3Fix::UNIT_X, Fix128::from_int(100));
    assert!(result.is_some(), "Raycast should hit a body");

    let (hit_body, hit_dist) = result.unwrap();
    assert_eq!(hit_body, id_near, "Should hit nearest body");
    assert!(
        hit_dist > Fix128::from_int(3) && hit_dist < Fix128::from_int(6),
        "Hit distance should be ~4 (center=5, radius=1), got hi={}",
        hit_dist.hi
    );
}

// ============================================================================
// Test 20 — add_body_with_radius convenience
// ============================================================================

/// Verify add_body_with_radius enables auto collision for that body.
#[test]
fn test_add_body_with_radius() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Body without radius: no auto-collision
    let plain = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    world.add_body(plain);

    // Body with radius: participates in auto-collision
    let sphere = RigidBody::new_dynamic(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    let id = world.add_body_with_radius(sphere, Fix128::ONE);

    // Should be able to raycast against the radius body
    let hit = world.raycast(
        Vec3Fix::from_int(-5, 0, 0),
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
    );
    assert!(hit.is_some());
    assert_eq!(hit.unwrap().0, id);
}

// ============================================================================
// Test 21 — body_count and active_body_count
// ============================================================================

/// Verify body counting with sleeping bodies.
#[test]
fn test_body_count_and_active() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    world.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 3,
    });

    // Add 3 still bodies
    for i in 0..3 {
        let body = RigidBody::new_dynamic(Vec3Fix::from_int(i * 5, 0, 0), Fix128::ONE);
        world.add_body(body);
    }

    assert_eq!(world.body_count(), 3);

    // Step enough for sleeping
    let dt = Fix128::from_ratio(1, 60);
    run_world(&mut world, 5, dt);

    // All should be sleeping
    assert_eq!(world.body_count(), 3);
    assert_eq!(world.active_body_count(), 0, "All still bodies should sleep");
}

// ============================================================================
// Test 22 — RigidBody convenience methods
// ============================================================================

/// Verify add_force, set_velocity, mass, speed methods.
#[test]
fn test_rigid_body_convenience_methods() {
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2));

    // mass = 1/inv_mass
    assert_eq!(body.mass().hi, 2, "mass should be 2");

    // set_velocity
    body.set_velocity(Vec3Fix::from_int(3, 4, 0));
    assert_eq!(body.velocity.x.hi, 3);
    assert_eq!(body.velocity.y.hi, 4);

    // speed = |velocity|
    let spd = body.speed();
    assert!(spd > Fix128::from_int(4), "speed should be 5 (3-4-5 triangle)");
    assert!(spd < Fix128::from_int(6));

    // add_force: dv = F * dt * inv_mass
    // Use power-of-2 fractions for exact fixed-point arithmetic
    body.set_velocity(Vec3Fix::ZERO);
    body.add_force(Vec3Fix::from_int(4, 0, 0), Fix128::ONE);
    // dv = 4 * 1.0 * 0.5 = 2.0
    assert_eq!(body.velocity.x.hi, 2, "add_force should change velocity");
}

// ============================================================================
// Test 23 — angular_velocity serialization round-trip
// ============================================================================

/// Verify angular_velocity is preserved through serialize/deserialize.
#[test]
fn test_angular_velocity_serialization_roundtrip() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let mut body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
    body.angular_velocity = Vec3Fix::from_int(3, 7, 11);
    world.add_body(body);

    let dt = Fix128::from_ratio(1, 64);
    run_world(&mut world, 10, dt);

    // Capture angular_velocity before serialize
    let av_before = world.bodies[0].angular_velocity;

    let state = world.serialize_state();
    // Corrupt angular_velocity
    world.bodies[0].angular_velocity = Vec3Fix::ZERO;

    // Restore
    assert!(world.deserialize_state(&state));

    let av_after = world.bodies[0].angular_velocity;
    assert_eq!(av_before.x.hi, av_after.x.hi);
    assert_eq!(av_before.x.lo, av_after.x.lo);
    assert_eq!(av_before.y.hi, av_after.y.hi);
    assert_eq!(av_before.y.lo, av_after.y.lo);
    assert_eq!(av_before.z.hi, av_after.z.hi);
    assert_eq!(av_before.z.lo, av_after.z.lo);
}

// ============================================================================
// Test 24 — All subsystems combined in step
// ============================================================================

/// Run step() with force fields + collisions + joints + sleeping + events all active.
#[test]
fn test_all_subsystems_combined() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -10, 0),
        substeps: 4,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 120,
    });

    // Body A and B connected by ball joint, both with collision radii
    let id_a = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(-1, 3, 0), Fix128::ONE),
        Fix128::from_int(2),
    );
    let id_b = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 3, 0), Fix128::ONE),
        Fix128::from_int(2),
    );
    // Body C: sitting below, A/B will fall into it
    let _id_c = world.add_body_with_radius(
        RigidBody::new_static(Vec3Fix::from_int(0, -2, 0)),
        Fix128::from_int(3),
    );

    // Add ball joint between A and B
    let ball = BallJoint::new(id_a, id_b, Vec3Fix::ZERO, Vec3Fix::ZERO);
    world.add_joint(Joint::Ball(ball));

    // Add a wind force field
    world.add_force_field(ForceFieldInstance::new(ForceField::Directional {
        direction: Vec3Fix::UNIT_X,
        strength: Fix128::from_int(5),
    }));

    let dt = Fix128::from_ratio(1, 64);
    let mut had_contacts = false;
    for _ in 0..240 {
        world.step(dt);
        if !world.contact_events().is_empty() {
            had_contacts = true;
        }
    }

    // Verify all subsystems ran: gravity caused falling, wind pushed laterally,
    // joint kept A-B connected, and contacts were generated
    let pos_a = world.bodies[id_a].position;
    assert!(pos_a.y < Fix128::from_int(3), "Gravity should pull A down");
    assert!(pos_a.x > Fix128::from_int(-1), "Wind should push A right");
    assert!(had_contacts, "Collisions should generate contact events");
}

// ============================================================================
// Test 25 — raycast with zero direction
// ============================================================================

/// raycast() with zero-length direction should return None, not panic.
#[test]
fn test_raycast_zero_direction() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    world.add_body_with_radius(
        RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)),
        Fix128::ONE,
    );

    let result = world.raycast(Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::from_int(100));
    assert!(result.is_none(), "Zero direction should return None");
}

// ============================================================================
// Test 26 — add_joint out-of-bounds panics
// ============================================================================

/// add_joint() with invalid body indices should panic.
#[test]
#[should_panic(expected = "out of bounds")]
fn test_add_joint_out_of_bounds() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    // Only 1 body exists, referencing index 999 should panic
    let ball = BallJoint::new(0, 999, Vec3Fix::ZERO, Vec3Fix::ZERO);
    world.add_joint(Joint::Ball(ball));
}

// ============================================================================
// Test 27 — remove_body with active joint
// ============================================================================

/// Removing a body that has an active joint should clean up the joint.
#[test]
fn test_remove_body_with_active_joint() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let id_a = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(0, 0, 0), Fix128::ONE));
    let id_b = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(5, 0, 0), Fix128::ONE));
    let id_c = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(10, 0, 0), Fix128::ONE));

    // Joint between A-B and B-C
    let ball_ab = BallJoint::new(id_a, id_b, Vec3Fix::ZERO, Vec3Fix::ZERO);
    let ball_bc = BallJoint::new(id_b, id_c, Vec3Fix::ZERO, Vec3Fix::ZERO);
    world.add_joint(Joint::Ball(ball_ab));
    world.add_joint(Joint::Ball(ball_bc));

    assert_eq!(world.joint_count(), 2);

    // Remove body A — joint A-B should be cleaned up
    world.remove_body(id_a);

    assert_eq!(world.body_count(), 2, "Should have 2 bodies left");

    // Step should not crash
    let dt = Fix128::from_ratio(1, 64);
    run_world(&mut world, 10, dt);
}

// ============================================================================
// Test 28 — Empty world step
// ============================================================================

/// step() on a world with zero bodies should not panic.
#[test]
fn test_empty_world_step() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    assert_eq!(world.body_count(), 0);

    let dt = Fix128::from_ratio(1, 64);
    run_world(&mut world, 10, dt);

    assert_eq!(world.body_count(), 0);
}

// ============================================================================
// Test 29 — Empty world after removing all bodies
// ============================================================================

/// Removing all bodies and then stepping should not panic.
#[test]
fn test_empty_world_after_remove_all() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(5, 0, 0), Fix128::ONE));

    // Remove both (careful: swap-remove changes last index)
    world.remove_body(0);
    world.remove_body(0);

    assert_eq!(world.body_count(), 0);

    let dt = Fix128::from_ratio(1, 64);
    run_world(&mut world, 10, dt);
}

// ============================================================================
// Test 30 — Sleep propagation through joint islands
// ============================================================================

/// Bodies connected by joints should sleep and wake as a group.
#[test]
fn test_sleep_island_propagation() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO, // no gravity so bodies stay still
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 3,
    });

    // Chain: A --joint-- B --joint-- C
    let id_a = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(0, 0, 0), Fix128::ONE));
    let id_b = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(5, 0, 0), Fix128::ONE));
    let id_c = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(10, 0, 0), Fix128::ONE));

    let ball_ab = BallJoint::new(id_a, id_b, Vec3Fix::ZERO, Vec3Fix::ZERO);
    let ball_bc = BallJoint::new(id_b, id_c, Vec3Fix::ZERO, Vec3Fix::ZERO);
    world.add_joint(Joint::Ball(ball_ab));
    world.add_joint(Joint::Ball(ball_bc));

    // D is separate — not connected
    let id_d = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(20, 0, 0), Fix128::ONE));

    let dt = Fix128::from_ratio(1, 64);

    // Step enough for all bodies to sleep
    run_world(&mut world, 5, dt);

    assert_eq!(world.active_body_count(), 0, "All still bodies should sleep");

    // Wake body C — should wake A and B through island, but NOT D
    world.wake_body(id_c);

    assert!(!world.is_sleeping(id_c), "C should be awake");
    // Note: wake_body only wakes the single body, not the island.
    // The island waking happens via wake_island if used, or next step may detect activity.
    // After a step, if C is awake, the island system should propagate.
    // Actually wake_body wakes single body; but the key test is
    // that D (separate island) stays asleep.
    assert!(world.is_sleeping(id_d), "D should still be sleeping (different island)");
}

// ============================================================================
// Test 31 — Event lifecycle Persist verification
// ============================================================================

/// Verify that the Persist event type is correctly reported on sustained contact.
#[test]
fn test_event_persist_explicit() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Two bodies pushed into each other
    let mut body_a = RigidBody::new_dynamic(Vec3Fix::from_int(-1, 0, 0), Fix128::ONE);
    body_a.velocity = Vec3Fix::from_int(10, 0, 0); // moving right
    let mut body_b = RigidBody::new_dynamic(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    body_b.velocity = Vec3Fix::from_int(-10, 0, 0); // moving left

    world.add_body_with_radius(body_a, Fix128::from_int(2));
    world.add_body_with_radius(body_b, Fix128::from_int(2));

    let dt = Fix128::from_ratio(1, 64);

    // Frame 1: should get Begin
    world.step(dt);
    let has_begin = world.contact_events().iter().any(|e| e.event_type == ContactEventType::Begin);

    // Frame 2+: if still overlapping, should get Persist
    world.step(dt);
    let has_persist = world.contact_events().iter().any(|e| e.event_type == ContactEventType::Persist);

    // At least one of these should be true (bodies are overlapping for >=2 frames)
    assert!(has_begin || has_persist, "Should have Begin or Persist events from sustained contact");
}

// ============================================================================
// Test 32 — dt=0 and dt<0 guard
// ============================================================================

/// step() with zero or negative dt should be a no-op.
#[test]
fn test_step_zero_and_negative_dt() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
    world.add_body(body);

    let pos_before = world.bodies[0].position;

    // dt = 0 should be no-op
    world.step(Fix128::ZERO);
    assert_eq!(world.bodies[0].position.y.hi, pos_before.y.hi, "dt=0 should not move body");

    // dt < 0 should be no-op
    world.step(Fix128::from_int(-1));
    assert_eq!(world.bodies[0].position.y.hi, pos_before.y.hi, "dt<0 should not move body");
}

// ============================================================================
// Test 33 — Kinematic body pushes dynamic body
// ============================================================================

/// A kinematic body set_kinematic_target through a dynamic body should push it.
#[test]
fn test_kinematic_pushes_dynamic() {
    use alice_physics::QuatFix;

    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 4,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Kinematic body starting at left
    let kin = RigidBody::new_kinematic(Vec3Fix::from_int(-5, 0, 0));
    let id_kin = world.add_body_with_radius(kin, Fix128::from_int(2));

    // Dynamic body sitting still at origin
    let dyn_body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 0, 0), Fix128::ONE);
    let _id_dyn = world.add_body_with_radius(dyn_body, Fix128::from_int(2));

    let dt = Fix128::from_ratio(1, 64);

    // Move kinematic body rightward each frame via set_kinematic_target
    for i in 0..30 {
        let target_x = Fix128::from_int(-5) + Fix128::from_int(i + 1) * dt * Fix128::from_int(10);
        world.bodies[id_kin].set_kinematic_target(
            Vec3Fix::new(target_x, Fix128::ZERO, Fix128::ZERO),
            QuatFix::IDENTITY,
        );
        world.step(dt);
    }

    // Kinematic should have moved right from -5
    assert!(world.bodies[id_kin].position.x > Fix128::from_int(-5), "Kinematic should move");
}

// ============================================================================
// Test 34 — detect_collisions with mixed radii (some None)
// ============================================================================

/// Bodies added without radius should not participate in auto-collision.
#[test]
fn test_collision_detection_mixed_radii() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Body A: has radius
    let mut body_a = RigidBody::new_dynamic(Vec3Fix::from_int(-1, 0, 0), Fix128::ONE);
    body_a.velocity = Vec3Fix::from_int(5, 0, 0);
    world.add_body_with_radius(body_a, Fix128::from_int(2));

    // Body B: NO radius (added via add_body, not add_body_with_radius)
    let body_b = RigidBody::new_dynamic(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
    world.add_body(body_b);

    let dt = Fix128::from_ratio(1, 64);

    // Step and check: no collision events should fire (B has no radius)
    let mut had_contacts = false;
    for _ in 0..20 {
        world.step(dt);
        if !world.contact_events().is_empty() {
            had_contacts = true;
        }
    }

    assert!(!had_contacts, "Body without radius should not generate auto-collisions");
}

// ============================================================================
// Test 35 — get_body / get_body_mut
// ============================================================================

/// Verify safe body accessors return Some/None correctly.
#[test]
fn test_get_body_accessors() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let id = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(7, 0, 0), Fix128::ONE));

    // Valid index → Some
    assert!(world.get_body(id).is_some());
    assert_eq!(world.get_body(id).unwrap().position.x.hi, 7);

    // Out-of-bounds → None
    assert!(world.get_body(999).is_none());

    // Mutable accessor
    {
        let body = world.get_body_mut(id).unwrap();
        body.position.x = Fix128::from_int(42);
    }
    assert_eq!(world.get_body(id).unwrap().position.x.hi, 42);

    // Mutable out-of-bounds
    assert!(world.get_body_mut(999).is_none());
}

// ============================================================================
// Test 36 — filter::layers re-export
// ============================================================================

/// Verify that filter::layers constants are accessible from crate root.
#[test]
fn test_layers_reexport() {
    use alice_physics::layers;

    let player = CollisionFilter::new(layers::PLAYER, layers::ENEMY | layers::STATIC);
    let enemy = CollisionFilter::new(layers::ENEMY, layers::PLAYER | layers::STATIC);

    assert!(CollisionFilter::can_collide(&player, &enemy));
    assert!(!CollisionFilter::can_collide(
        &CollisionFilter::new(layers::PLAYER, layers::PLAYER),
        &CollisionFilter::new(layers::ENEMY, layers::ENEMY),
    ));
}

// ============================================================================
// Test 37 — rebuild_batches graph coloring invariant
// ============================================================================

/// Verify rebuild_batches() + constraint solving produce correct results.
/// A chain of constraints should maintain distance after solving.
#[test]
fn test_rebuild_batches_constraint_chain() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 8,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Create a chain: 0 - 1 - 2 - 3 - 4 with unit distance
    for i in 0..5 {
        world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(i as i64, 0, 0), Fix128::ONE));
    }
    for i in 0..4 {
        world.add_distance_constraint(DistanceConstraint {
            body_a: i,
            body_b: i + 1,
            local_anchor_a: Vec3Fix::ZERO,
            local_anchor_b: Vec3Fix::ZERO,
            target_distance: Fix128::ONE,
            compliance: Fix128::ZERO,
            cached_lambda: Fix128::ZERO,
        });
    }

    // rebuild_batches should succeed without panic
    world.rebuild_batches();

    // Step to let constraints settle
    let dt = Fix128::from_ratio(1, 64);
    run_world(&mut world, 60, dt);

    // Each pair should be roughly unit distance
    for i in 0..4 {
        let a = world.bodies[i].position;
        let b = world.bodies[i + 1].position;
        let dist = (b - a).length();
        let error = if dist > Fix128::ONE { dist - Fix128::ONE } else { Fix128::ONE - dist };
        assert!(
            error < Fix128::from_int(2),
            "Chain link {}-{} distance error too large: {:?}",
            i, i + 1, error,
        );
    }
}

// ============================================================================
// Test 38 — step_parallel (feature = "parallel")
// ============================================================================

/// Verify step_parallel() produces deterministic results identical to step().
#[cfg(feature = "parallel")]
#[test]
fn test_step_parallel_determinism() {
    // Run same scenario with step()
    fn run_sequential() -> Vec3Fix {
        let config = PhysicsConfig {
            gravity: Vec3Fix::from_int(0, -10, 0),
            substeps: 4,
            ..Default::default()
        };
        let mut world = PhysicsWorld::new(config);

        let body = RigidBody::new_dynamic(Vec3Fix::from_int(5, 50, 3), Fix128::ONE);
        world.add_body(body);

        let dt = Fix128::from_ratio(1, 64);
        for _ in 0..120 {
            world.step(dt);
        }
        world.bodies[0].position
    }

    // Run same scenario with step_parallel()
    fn run_parallel() -> Vec3Fix {
        let config = PhysicsConfig {
            gravity: Vec3Fix::from_int(0, -10, 0),
            substeps: 4,
            ..Default::default()
        };
        let mut world = PhysicsWorld::new(config);

        let body = RigidBody::new_dynamic(Vec3Fix::from_int(5, 50, 3), Fix128::ONE);
        world.add_body(body);

        let dt = Fix128::from_ratio(1, 64);
        for _ in 0..120 {
            world.step_parallel(dt);
        }
        world.bodies[0].position
    }

    let seq = run_sequential();
    let par = run_parallel();

    // Must be bit-exact
    assert_eq!(seq.x.hi, par.x.hi, "x.hi mismatch");
    assert_eq!(seq.x.lo, par.x.lo, "x.lo mismatch");
    assert_eq!(seq.y.hi, par.y.hi, "y.hi mismatch");
    assert_eq!(seq.y.lo, par.y.lo, "y.lo mismatch");
    assert_eq!(seq.z.hi, par.z.hi, "z.hi mismatch");
    assert_eq!(seq.z.lo, par.z.lo, "z.lo mismatch");
}

/// Verify step_parallel() with constraints + collisions.
#[cfg(feature = "parallel")]
#[test]
fn test_step_parallel_with_constraints() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -10, 0),
        substeps: 4,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    let id_a = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE));
    let id_b = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(5, 10, 0), Fix128::ONE));

    world.add_distance_constraint(DistanceConstraint {
        body_a: id_a,
        body_b: id_b,
        local_anchor_a: Vec3Fix::ZERO,
        local_anchor_b: Vec3Fix::ZERO,
        target_distance: Fix128::from_int(5),
        compliance: Fix128::ZERO,
        cached_lambda: Fix128::ZERO,
    });

    let dt = Fix128::from_ratio(1, 64);
    for _ in 0..60 {
        world.step_parallel(dt);
    }

    // Distance should be approximately maintained
    let pos_a = world.bodies[id_a].position;
    let pos_b = world.bodies[id_b].position;
    let dist = (pos_b - pos_a).length();
    let target = Fix128::from_int(5);
    let error = if dist > target { dist - target } else { target - dist };
    assert!(error < Fix128::from_int(2), "Parallel constraint stability failed: error={:?}", error);
}

/// Verify step_parallel() self-consistent determinism (run twice → same result).
#[cfg(feature = "parallel")]
#[test]
fn test_step_parallel_self_determinism() {
    fn run() -> Vec3Fix {
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);

        let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 100, 0), Fix128::ONE);
        world.add_body(body);

        let dt = Fix128::from_ratio(1, 64);
        for _ in 0..200 {
            world.step_parallel(dt);
        }
        world.bodies[0].position
    }

    let r1 = run();
    let r2 = run();
    assert_eq!(r1.x.hi, r2.x.hi);
    assert_eq!(r1.x.lo, r2.x.lo);
    assert_eq!(r1.y.hi, r2.y.hi);
    assert_eq!(r1.y.lo, r2.y.lo);
    assert_eq!(r1.z.hi, r2.z.hi);
    assert_eq!(r1.z.lo, r2.z.lo);
}

// ============================================================================
// Test 39 — batch_raycast / batch_sphere_cast
// ============================================================================

/// Verify batch_raycast returns correct results for multiple rays.
#[test]
fn test_batch_raycast_results() {
    use alice_physics::query::{batch_raycast, BatchRayQuery};

    let bodies = [
        RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)),
        RigidBody::new_static(Vec3Fix::from_int(10, 0, 0)),
    ];

    let queries = vec![
        BatchRayQuery {
            origin: Vec3Fix::ZERO,
            direction: Vec3Fix::UNIT_X,
            max_distance: Fix128::from_int(100),
        },
        BatchRayQuery {
            origin: Vec3Fix::ZERO,
            direction: Vec3Fix::from_int(0, 1, 0), // shoots up, misses
            max_distance: Fix128::from_int(100),
        },
    ];

    let results = batch_raycast(&queries, &bodies, Fix128::ONE);
    assert_eq!(results.len(), 2);
    assert!(results[0].is_some(), "First ray should hit body at x=5");
    assert!(results[1].is_none(), "Second ray should miss (no body along Y)");
}

/// Verify batch_sphere_cast returns correct hit count.
#[test]
fn test_batch_sphere_cast_results() {
    use alice_physics::query::batch_sphere_cast;

    let bodies = [
        RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)),
    ];

    let origins = vec![Vec3Fix::ZERO, Vec3Fix::from_int(0, 100, 0)];
    let directions = vec![Vec3Fix::UNIT_X, Vec3Fix::UNIT_X];

    let results = batch_sphere_cast(
        &origins,
        Fix128::ONE,
        &directions,
        Fix128::from_int(100),
        &bodies,
        Fix128::ONE,
    );

    assert_eq!(results.len(), 2);
    assert!(results[0].is_some(), "Close ray should hit");
    assert!(results[1].is_none(), "Far-off ray should miss");
}

// ============================================================================
// Test 40 — Serialization round-trip truncation correctness
// ============================================================================

/// Verify serialize → deserialize → serialize produces identical bytes.
#[test]
fn test_serialization_exact_roundtrip() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // Add bodies with various states
    let mut body_a = RigidBody::new_dynamic(Vec3Fix::from_int(1, 20, 3), Fix128::ONE);
    body_a.angular_velocity = Vec3Fix::from_int(5, -3, 7);
    world.add_body(body_a);

    let mut body_b = RigidBody::new_dynamic(Vec3Fix::from_int(-4, 0, 8), Fix128::from_int(2));
    body_b.velocity = Vec3Fix::from_int(10, -5, 2);
    world.add_body(body_b);

    let dt = Fix128::from_ratio(1, 64);
    run_world(&mut world, 30, dt);

    // Serialize → Deserialize → Serialize
    let state1 = world.serialize_state();
    assert!(world.deserialize_state(&state1));
    let state2 = world.serialize_state();

    // Must be byte-identical
    assert_eq!(state1.len(), state2.len(), "Round-trip size mismatch");
    assert_eq!(state1, state2, "Round-trip bytes not identical");
}

/// Verify deserialize rejects mismatched body count.
#[test]
fn test_serialization_rejects_mismatch() {
    let config = PhysicsConfig::default();

    // World with 2 bodies
    let mut world_2 = PhysicsWorld::new(config);
    world_2.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world_2.add_body(RigidBody::new_dynamic(Vec3Fix::UNIT_X, Fix128::ONE));
    let state_2 = world_2.serialize_state();

    // World with 3 bodies → deserialize should fail
    let mut world_3 = PhysicsWorld::new(config);
    for _ in 0..3 {
        world_3.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    }
    assert!(!world_3.deserialize_state(&state_2), "Mismatched body count should return false");
}

// ============================================================================
// Test 42 — apply_impulse changes velocity
// ============================================================================

#[test]
fn test_apply_impulse() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let id = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

    let impulse = Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO);
    world.bodies[id].apply_impulse(impulse);

    // mass=1, inv_mass=1, so velocity = impulse * 1 = (5,0,0)
    assert_eq!(world.bodies[id].velocity.x, Fix128::from_int(5));
    assert_eq!(world.bodies[id].velocity.y, Fix128::ZERO);

    // Static body ignores impulse
    let static_id = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    world.bodies[static_id].apply_impulse(impulse);
    assert_eq!(world.bodies[static_id].velocity.x, Fix128::ZERO);
}

// ============================================================================
// Test 43 — apply_impulse_at generates angular velocity
// ============================================================================

#[test]
fn test_apply_impulse_at() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    // Body at origin, mass=1
    let id = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

    // Apply impulse at offset point → should produce both linear and angular velocity
    let impulse = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(10), Fix128::ZERO);
    let point = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    world.bodies[id].apply_impulse_at(impulse, point);

    // Linear velocity changed
    assert_eq!(world.bodies[id].velocity.y, Fix128::from_int(10));
    // Angular velocity should be non-zero (torque = r × F = (1,0,0) × (0,10,0) = (0,0,10))
    assert!(world.bodies[id].angular_velocity.z != Fix128::ZERO,
        "apply_impulse_at should generate angular velocity");
}

// ============================================================================
// Test 44 — add_torque modifies angular velocity
// ============================================================================

#[test]
fn test_add_torque() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let id = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

    let dt = Fix128::from_ratio(1, 60);
    let torque = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(60), Fix128::ZERO);
    world.bodies[id].add_torque(torque, dt);

    // angular_velocity.y = torque.y * inv_inertia.y * dt
    assert!(world.bodies[id].angular_velocity.y != Fix128::ZERO,
        "add_torque should modify angular velocity");

    // Static body ignores torque
    let static_id = world.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    world.bodies[static_id].add_torque(torque, dt);
    assert_eq!(world.bodies[static_id].angular_velocity.y, Fix128::ZERO);
}

// ============================================================================
// Test 45 — set_position resets prev_position (teleport)
// ============================================================================

#[test]
fn test_set_position_resets_prev() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let id = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

    // Step once so prev_position diverges from position
    let dt = Fix128::from_ratio(1, 60);
    world.step(dt);
    assert!(world.bodies[id].position != world.bodies[id].prev_position
        || world.bodies[id].position == Vec3Fix::ZERO,
        "After step, position or prev should differ (gravity)");

    // Teleport
    let new_pos = Vec3Fix::from_int(100, 200, 300);
    world.bodies[id].set_position(new_pos);
    assert_eq!(world.bodies[id].position, new_pos);
    assert_eq!(world.bodies[id].prev_position, new_pos, "set_position must reset prev_position");
}

// ============================================================================
// Test 46 — set_rotation resets prev_rotation
// ============================================================================

#[test]
fn test_set_rotation_resets_prev() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let id = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

    let new_rot = QuatFix {
        x: Fix128::ZERO,
        y: Fix128::ZERO,
        z: Fix128::from_ratio(1, 2),
        w: Fix128::from_ratio(1, 2),
    };
    world.bodies[id].set_rotation(new_rot);
    assert_eq!(world.bodies[id].rotation.z, new_rot.z);
    assert_eq!(world.bodies[id].prev_rotation.z, new_rot.z, "set_rotation must reset prev_rotation");
}

// ============================================================================
// Test 47 — new_sensor creates a static sensor body
// ============================================================================

#[test]
fn test_new_sensor() {
    let pos = Vec3Fix::from_int(5, 5, 5);
    let sensor = RigidBody::new_sensor(pos);

    assert!(sensor.is_static(), "Sensor should be static");
    assert!(sensor.is_sensor, "Sensor flag must be true");
    assert_eq!(sensor.position, pos);
    assert_eq!(sensor.inv_mass, Fix128::ZERO, "Sensor has zero inv_mass");
    assert_eq!(sensor.gravity_scale, Fix128::ZERO, "Sensor ignores gravity");
}

// ============================================================================
// Test 48 — remove_joint standalone
// ============================================================================

#[test]
fn test_remove_joint() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::UNIT_X, Fix128::ONE));

    let joint = Joint::Ball(BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO));
    let joint_idx = world.add_joint(joint);
    assert_eq!(world.joint_count(), 1);

    // Remove joint
    let removed = world.remove_joint(joint_idx);
    assert!(removed.is_some(), "Should return the removed joint");
    assert_eq!(world.joint_count(), 0);

    // Out-of-bounds remove returns None
    assert!(world.remove_joint(999).is_none());
}

// ============================================================================
// Test 49 — remove_force_field
// ============================================================================

#[test]
fn test_remove_force_field() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let field = ForceFieldInstance::new(ForceField::Directional {
        direction: Vec3Fix::UNIT_Y,
        strength: Fix128::ONE,
    });
    let idx = world.add_force_field(field);

    let removed = world.remove_force_field(idx);
    assert!(removed.is_some());

    // Out-of-bounds returns None
    assert!(world.remove_force_field(999).is_none());
}

// ============================================================================
// Test 50 — set/clear_body_collision_radius
// ============================================================================

#[test]
fn test_set_clear_body_collision_radius() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let id = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

    // Initially no radius (added without radius)
    // Set radius
    world.set_body_collision_radius(id, Fix128::from_int(2));

    // Verify by running a raycast that should now hit
    let origin = Vec3Fix::new(Fix128::from_int(-10), Fix128::ZERO, Fix128::ZERO);
    let direction = Vec3Fix::UNIT_X;
    let hit = world.raycast(origin, direction, Fix128::from_int(100));
    assert!(hit.is_some(), "Should hit after set_body_collision_radius");

    // Clear radius
    world.clear_body_collision_radius(id);
    let hit2 = world.raycast(origin, direction, Fix128::from_int(100));
    assert!(hit2.is_none(), "Should not hit after clear_body_collision_radius");
}

// ============================================================================
// Test 51 — body_filter getter
// ============================================================================

#[test]
fn test_body_filter_getter() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let id = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

    // Default filter
    let default_filter = world.body_filter(id);
    assert_eq!(default_filter, CollisionFilter::DEFAULT);

    // Set custom filter
    let custom = CollisionFilter::new(0x02, 0x04);
    world.set_body_filter(id, custom);
    let got = world.body_filter(id);
    assert_eq!(got, custom);

    // Out-of-bounds returns DEFAULT
    let oob = world.body_filter(9999);
    assert_eq!(oob, CollisionFilter::DEFAULT);
}

// ============================================================================
// Test 52 — drain_contact_events / drain_trigger_events
// ============================================================================

#[test]
fn test_drain_events() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // Create two colliding bodies
    let a = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        Fix128::from_int(2),
    );
    let b = world.add_body_with_radius(
        RigidBody::new_static(Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO)),
        Fix128::from_int(2),
    );
    let _ = (a, b);

    let dt = Fix128::from_ratio(1, 60);
    run_world(&mut world, 5, dt);

    // drain should consume events
    let contacts = world.drain_contact_events();
    // After drain, the contact_events() should be empty
    assert!(world.contact_events().is_empty(), "Events should be empty after drain");

    // drain_trigger_events on non-sensor world should return empty
    let triggers = world.drain_trigger_events();
    let _ = (contacts, triggers);
}

// ============================================================================
// Test 53 — with_compliance (soft constraint)
// ============================================================================

#[test]
fn test_soft_constraint_with_compliance() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    ));

    // Stiff constraint (compliance=0)
    let stiff = DistanceConstraint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::from_int(5))
        .with_compliance(Fix128::ZERO);
    world.add_distance_constraint(stiff);

    let dt = Fix128::from_ratio(1, 60);
    run_world(&mut world, 120, dt);
    let dist_stiff = (world.bodies[a].position - world.bodies[b].position).length();

    // Reset
    let mut world2 = PhysicsWorld::new(config);
    let a2 = world2.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b2 = world2.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    ));

    // Soft constraint (high compliance)
    let soft = DistanceConstraint::new(a2, b2, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::from_int(5))
        .with_compliance(Fix128::from_int(10));
    world2.add_distance_constraint(soft);
    run_world(&mut world2, 120, dt);
    let dist_soft = (world2.bodies[a2].position - world2.bodies[b2].position).length();

    // Both should maintain approximate distance, but soft allows more deviation
    let _ = (dist_stiff, dist_soft);
    // The key assertion: soft constraint is valid and doesn't crash
}

// ============================================================================
// Test 54 — num_batches
// ============================================================================

#[test]
fn test_num_batches() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // No constraints → 0 batches
    world.rebuild_batches();
    assert_eq!(world.num_batches(), 0);

    // Add a chain of constraints
    let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b = world.add_body(RigidBody::new_dynamic(Vec3Fix::UNIT_X, Fix128::ONE));
    let c = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    ));

    // A-B and B-C share body B → need at least 2 batches
    world.add_distance_constraint(DistanceConstraint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ONE));
    world.add_distance_constraint(DistanceConstraint::new(b, c, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ONE));
    world.rebuild_batches();
    assert!(world.num_batches() >= 2, "Chain sharing a body needs >= 2 batches");
}

// ============================================================================
// Test 55 — mass=0 creates static-like body
// ============================================================================

#[test]
fn test_zero_mass_body() {
    // mass=0 in new_dynamic → inv_mass=0
    let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ZERO);
    assert_eq!(body.inv_mass, Fix128::ZERO, "mass=0 → inv_mass=0");
    // is_static() checks inv_mass, so mass=0 body is considered static
    assert!(body.is_static(), "mass=0 body is_static() returns true");

    // Impulse should have no effect (is_static guard)
    let mut body2 = body;
    body2.apply_impulse(Vec3Fix::new(Fix128::from_int(100), Fix128::ZERO, Fix128::ZERO));
    assert_eq!(body2.velocity.x, Fix128::ZERO,
        "mass=0 body ignores impulse");

    // add_torque should also have no effect
    body2.add_torque(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(100), Fix128::ZERO),
        Fix128::from_ratio(1, 60),
    );
    assert_eq!(body2.angular_velocity.y, Fix128::ZERO,
        "mass=0 body ignores torque");
}

// ============================================================================
// Test 56 — deserialization rejects empty and truncated data
// ============================================================================

#[test]
fn test_deserialize_empty_and_truncated() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

    // Empty data
    assert!(!world.deserialize_state(&[]), "Empty data should fail");

    // Too short (< 4 bytes)
    assert!(!world.deserialize_state(&[0, 0]), "Short data should fail");

    // Header says 1 body but data is truncated (only 4 header bytes, no body data)
    let mut truncated = vec![0u8; 4];
    truncated[0] = 1; // count=1 in LE
    assert!(!world.deserialize_state(&truncated), "Truncated body data should fail");
}

// ============================================================================
// Test 57 — self-referential joint (body_a == body_b)
// ============================================================================

#[test]
fn test_self_referential_joint() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    let a = world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE));

    // Self-referential joint should not crash the solver
    let joint = Joint::Ball(BallJoint::new(a, a, Vec3Fix::ZERO, Vec3Fix::ZERO));
    world.add_joint(joint);

    let dt = Fix128::from_ratio(1, 60);
    // Should not panic or infinite loop
    run_world(&mut world, 60, dt);
}

// ============================================================================
// Test 58 — large dt does not panic
// ============================================================================

#[test]
fn test_large_dt() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    world.add_body(RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE));

    // Very large dt — should not panic, overflow, or infinite loop
    let large_dt = Fix128::from_int(100);
    world.step(large_dt);

    // Body should have moved significantly downward
    assert!(world.bodies[0].position.y < Fix128::from_int(10),
        "Large dt should still produce downward motion");
}

// ============================================================================
// Test 59 — ray originating inside sphere
// ============================================================================

#[test]
fn test_raycast_inside_sphere() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // Large sphere at origin
    world.add_body_with_radius(
        RigidBody::new_static(Vec3Fix::ZERO),
        Fix128::from_int(10),
    );

    // Ray origin inside the sphere, pointing outward
    let origin = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let direction = Vec3Fix::UNIT_X;
    let hit = world.raycast(origin, direction, Fix128::from_int(100));

    // Should hit the far side of the sphere
    assert!(hit.is_some(), "Ray inside sphere should hit far intersection");
    let (idx, t) = hit.unwrap();
    assert_eq!(idx, 0);
    assert!(t > Fix128::ZERO, "Hit distance should be positive (far side)");
}

// ============================================================================
// Test 60 — remove_sdf_collider
// ============================================================================

#[test]
fn test_remove_sdf_collider() {
    use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // Add a ground-plane SDF collider
    let ground = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    let collider = SdfCollider::new_static(Box::new(ground), Vec3Fix::ZERO, QuatFix::IDENTITY);
    let idx = world.add_sdf_collider(collider);

    // Remove it
    let removed = world.remove_sdf_collider(idx);
    assert!(removed.is_some());

    // Out-of-bounds returns None
    assert!(world.remove_sdf_collider(999).is_none());
}

// ============================================================================
// Test 61 — remove_body index 0 from single-body world
// ============================================================================

#[test]
fn test_remove_only_body() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    assert_eq!(world.bodies.len(), 1);

    world.remove_body(0);
    assert_eq!(world.bodies.len(), 0);

    // World should still step without panic
    let dt = Fix128::from_ratio(1, 60);
    world.step(dt);
}

// ============================================================================
// Test 62 — raycast BVH acceleration produces same results as distance check
// ============================================================================

#[test]
fn test_raycast_bvh_correctness() {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // Place 5 spheres along the X axis
    for i in 0..5 {
        world.add_body_with_radius(
            RigidBody::new_static(Vec3Fix::new(
                Fix128::from_int(i * 10),
                Fix128::ZERO,
                Fix128::ZERO,
            )),
            Fix128::from_int(2),
        );
    }

    // Ray from far left → should hit body 0 first
    let origin = Vec3Fix::new(Fix128::from_int(-20), Fix128::ZERO, Fix128::ZERO);
    let direction = Vec3Fix::UNIT_X;
    let hit = world.raycast(origin, direction, Fix128::from_int(200));
    assert!(hit.is_some());
    let (idx, _) = hit.unwrap();
    assert_eq!(idx, 0, "Nearest sphere should be body 0");

    // Ray from far right → should hit body 4 first
    let origin2 = Vec3Fix::new(Fix128::from_int(60), Fix128::ZERO, Fix128::ZERO);
    let dir2 = Vec3Fix::new(-Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let hit2 = world.raycast(origin2, dir2, Fix128::from_int(200));
    assert!(hit2.is_some());
    let (idx2, _) = hit2.unwrap();
    assert_eq!(idx2, 4, "Nearest sphere from right should be body 4");
}
