//! ALICE-Physics: Deterministic 128-bit Fixed-Point Physics Engine
//!
//! A high-precision physics engine designed for deterministic simulation
//! across different platforms and hardware. Uses 128-bit fixed-point arithmetic
//! to ensure bit-exact results regardless of CPU or compiler.
//!
//! # Features
//!
//! - **Deterministic**: Bit-exact results across all platforms (no floating-point)
//! - **128-bit Fixed-Point**: I64F64 format with CORDIC trigonometry
//! - **XPBD Solver**: Extended Position Based Dynamics for stable constraints
//! - **GJK/EPA**: Robust collision detection for convex shapes
//! - **Linear BVH**: Morton code-based spatial acceleration
//! - **Rollback Support**: State serialization for netcode
//!
//! # Example
//!
//! ```rust
//! use alice_physics::{PhysicsWorld, PhysicsConfig, RigidBody, Fix128, Vec3Fix};
//!
//! // Create physics world
//! let config = PhysicsConfig::default();
//! let mut world = PhysicsWorld::new(config);
//!
//! // Add a dynamic body
//! let body = RigidBody::new_dynamic(
//!     Vec3Fix::from_int(0, 10, 0),  // position
//!     Fix128::ONE,                   // mass = 1
//! );
//! let body_id = world.add_body(body);
//!
//! // Step simulation
//! let dt = Fix128::from_ratio(1, 60);  // 1/60 second
//! world.step(dt);
//! ```
//!
//! # Modules
//!
//! - [`math`]: Fixed-point math primitives (Fix128, Vec3Fix, QuatFix, CORDIC)
//! - [`collider`]: Collision shapes and GJK/EPA detection
//! - [`solver`]: XPBD physics solver and rigid body dynamics
//! - [`bvh`]: Linear BVH for broad-phase collision detection
//! - [`filter`]: Collision filtering with layer/mask bitmasks
//! - [`rng`]: Deterministic pseudo-random number generator (PCG-XSH-RR)
//! - [`event`]: Contact and trigger event tracking
//! - [`joint`]: Joint constraints (Ball, Hinge, Fixed, Slider, Spring)
//! - [`raycast`]: Ray and shape casting queries
//! - [`ccd`]: Continuous collision detection (TOI, conservative advancement)
//! - [`sleeping`]: Sleep/wake and island management
//! - [`trimesh`]: Triangle mesh collision with BVH acceleration
//! - [`heightfield`]: Height field terrain collision
//! - [`motor`]: PD controllers and joint motors
//! - [`articulation`]: Articulated bodies (ragdolls, robotic arms)
//! - [`force`]: Custom force fields (wind, gravity wells, buoyancy, vortex)
//!
//! # Determinism
//!
//! This engine guarantees bit-exact results by:
//!
//! 1. Using 128-bit fixed-point instead of IEEE 754 floating-point
//! 2. CORDIC algorithm for trigonometry (no FPU instructions)
//! 3. Fixed iteration counts in all algorithms
//! 4. Deterministic sorting (stable sort with explicit comparators)
//!
//! This makes it suitable for:
//! - Lockstep multiplayer games
//! - Rollback netcode
//! - Replay systems
//! - Distributed physics simulation

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod math;
pub mod collider;
pub mod solver;
pub mod bvh;
pub mod sdf_collider;
pub mod filter;
pub mod rng;
pub mod event;
pub mod joint;
pub mod raycast;
pub mod ccd;
pub mod sleeping;
pub mod trimesh;
pub mod heightfield;
pub mod motor;
pub mod articulation;
pub mod force;
#[cfg(feature = "neural")]
pub mod neural;

// Re-export commonly used types
pub use math::{Fix128, Vec3Fix, QuatFix, Mat3Fix};
pub use collider::{AABB, Sphere, Capsule, ConvexHull, CollisionResult};
pub use solver::{PhysicsWorld, PhysicsConfig, RigidBody, DistanceConstraint, ContactConstraint};
pub use bvh::{LinearBvh, BvhNode, BvhPrimitive};
pub use sdf_collider::{SdfField, SdfCollider, ClosureSdf};
pub use filter::CollisionFilter;
pub use rng::DeterministicRng;
pub use event::{EventCollector, ContactEvent, ContactEventType};
pub use joint::{Joint, BallJoint, HingeJoint, FixedJoint, SliderJoint, SpringJoint};
pub use raycast::{Ray, RayHit};
pub use ccd::CcdConfig;
pub use sleeping::{SleepState, SleepData, SleepConfig, IslandManager};
pub use trimesh::{Triangle, TriMesh};
pub use heightfield::HeightField;
pub use motor::{PdController, JointMotor, MotorMode};
pub use articulation::ArticulatedBody;
pub use force::{ForceField, ForceFieldInstance};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::math::{Fix128, Vec3Fix, QuatFix, Mat3Fix};
    pub use crate::collider::{AABB, Sphere, Capsule, ConvexHull, CollisionResult, Support};
    pub use crate::solver::{PhysicsWorld, PhysicsConfig, RigidBody, DistanceConstraint, ContactConstraint};
    pub use crate::bvh::{LinearBvh, BvhNode, BvhPrimitive};
    pub use crate::sdf_collider::{SdfField, SdfCollider, ClosureSdf};
    pub use crate::filter::CollisionFilter;
    pub use crate::rng::DeterministicRng;
    pub use crate::event::{EventCollector, ContactEvent, ContactEventType};
    pub use crate::joint::{Joint, BallJoint, HingeJoint, FixedJoint, SliderJoint, SpringJoint};
    pub use crate::raycast::{Ray, RayHit};
    pub use crate::ccd::CcdConfig;
    pub use crate::sleeping::{SleepState, SleepData, SleepConfig, IslandManager};
    pub use crate::trimesh::{Triangle, TriMesh};
    pub use crate::heightfield::HeightField;
    pub use crate::motor::{PdController, JointMotor, MotorMode};
    pub use crate::articulation::ArticulatedBody;
    pub use crate::force::{ForceField, ForceFieldInstance};
    #[cfg(feature = "neural")]
    pub use crate::neural::{
        DeterministicNetwork, RagdollController, ControllerConfig,
        ControllerOutput, FixedTernaryWeight, Activation,
        fix128_ternary_matvec, fix128_relu, fix128_hard_tanh,
        fix128_tanh_approx, fix128_leaky_relu,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_simulation() {
        // Create world with default config
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);

        // Add a falling body
        let mut body = RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 100, 0),
            Fix128::ONE,
        );
        body.restitution = Fix128::from_ratio(5, 10); // 0.5 bounce

        let body_id = world.add_body(body);

        // Add ground (static body)
        let ground = RigidBody::new_static(Vec3Fix::ZERO);
        world.add_body(ground);

        // Step simulation
        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            world.step(dt);
        }

        // Body should have fallen
        let pos = world.bodies[body_id].position;
        assert!(pos.y < Fix128::from_int(100), "Body should have fallen");
    }

    #[test]
    fn test_determinism() {
        // Run simulation twice with same initial conditions
        fn run_simulation() -> Vec3Fix {
            let config = PhysicsConfig::default();
            let mut world = PhysicsWorld::new(config);

            let body = RigidBody::new_dynamic(
                Vec3Fix::from_int(5, 50, 3),
                Fix128::from_ratio(3, 2), // mass = 1.5
            );
            world.add_body(body);

            let dt = Fix128::from_ratio(1, 60);
            for _ in 0..120 {
                world.step(dt);
            }

            world.bodies[0].position
        }

        let result1 = run_simulation();
        let result2 = run_simulation();

        // Results must be bit-exact
        assert_eq!(result1.x.hi, result2.x.hi);
        assert_eq!(result1.x.lo, result2.x.lo);
        assert_eq!(result1.y.hi, result2.y.hi);
        assert_eq!(result1.y.lo, result2.y.lo);
        assert_eq!(result1.z.hi, result2.z.hi);
        assert_eq!(result1.z.lo, result2.z.lo);
    }

    #[test]
    fn test_state_serialization() {
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);

        // Add body and simulate
        let body = RigidBody::new_dynamic(
            Vec3Fix::from_int(10, 20, 30),
            Fix128::ONE,
        );
        world.add_body(body);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..30 {
            world.step(dt);
        }

        // Serialize state
        let state = world.serialize_state();

        // Continue simulation
        for _ in 0..30 {
            world.step(dt);
        }
        let pos_after = world.bodies[0].position;

        // Restore state
        world.deserialize_state(&state);

        // Continue simulation from restored state
        for _ in 0..30 {
            world.step(dt);
        }
        let pos_restored = world.bodies[0].position;

        // Results must be identical
        assert_eq!(pos_after.x.hi, pos_restored.x.hi);
        assert_eq!(pos_after.y.hi, pos_restored.y.hi);
        assert_eq!(pos_after.z.hi, pos_restored.z.hi);
    }

    #[test]
    fn test_distance_constraint() {
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);

        // Two bodies connected by distance constraint
        let body_a = RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 10, 0),
            Fix128::ONE,
        );
        let body_b = RigidBody::new_dynamic(
            Vec3Fix::from_int(5, 10, 0),
            Fix128::ONE,
        );
        let id_a = world.add_body(body_a);
        let id_b = world.add_body(body_b);

        // Target distance = 5 units
        let constraint = DistanceConstraint {
            body_a: id_a,
            body_b: id_b,
            local_anchor_a: Vec3Fix::ZERO,
            local_anchor_b: Vec3Fix::ZERO,
            target_distance: Fix128::from_int(5),
            compliance: Fix128::ZERO,
        };
        world.add_distance_constraint(constraint);

        // Simulate
        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            world.step(dt);
        }

        // Distance should remain approximately 5
        let pos_a = world.bodies[id_a].position;
        let pos_b = world.bodies[id_b].position;
        let dist = (pos_b - pos_a).length();

        let target = Fix128::from_int(5);
        let error = if dist > target {
            dist - target
        } else {
            target - dist
        };

        // Allow 10% error due to gravity
        assert!(error < Fix128::ONE, "Distance constraint violated: error = {:?}", error);
    }
}
