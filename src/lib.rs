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
//! - [`sdf_manifold`]: Multi-point contact manifold from SDF surfaces
//! - [`sdf_ccd`]: Sphere tracing continuous collision detection for SDF
//! - [`sdf_force`]: SDF-driven force fields (attract, repel, contain, flow)
//! - [`sdf_destruction`]: Real-time CSG boolean destruction
//! - [`rope`]: XPBD distance chain rope and cable simulation
//! - [`cloth`]: XPBD triangle mesh cloth simulation
//! - [`fluid`]: Position-Based Fluids (PBF) with spatial hash grid
//! - [`deformable`]: FEM-XPBD deformable body simulation
//! - [`sdf_adaptive`]: Adaptive SDF evaluation with distance-based LOD
//! - [`convex_decompose`]: Convex decomposition from SDF voxel grid
//! - [`gpu_sdf`]: GPU compute shader interface for batch SDF evaluation
//! - [`fluid_netcode`]: Deterministic fluid netcode with delta compression
//! - [`vehicle`]: Vehicle physics (wheel, suspension, engine, steering)
//! - [`animation_blend`]: Ragdoll animation blending with SLERP
//! - [`audio_physics`]: Physics-based audio parameter generation
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

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "analytics")]
pub mod analytics_bridge;
pub mod animation_blend;
pub mod articulation;
pub mod audio_physics;
pub mod box_collider;
pub mod bvh;
pub mod ccd;
pub mod character;
pub mod cloth;
pub mod collider;
pub mod compound;
pub mod contact_cache;
pub mod convex_decompose;
pub mod cylinder;
#[cfg(feature = "replay")]
pub mod db_bridge;
pub mod debug_render;
pub mod deformable;
pub mod dynamic_bvh;
pub mod erosion;
pub mod event;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod filter;
pub mod fluid;
#[cfg(feature = "std")]
pub mod fluid_netcode;
pub mod force;
pub mod fracture;
#[cfg(feature = "std")]
pub mod gpu_sdf;
pub mod heightfield;
pub mod interpolation;
pub mod joint;
pub mod material;
pub mod math;
pub mod motor;
pub mod netcode;
#[cfg(feature = "neural")]
pub mod neural;
pub mod phase_change;
pub mod pressure;
pub mod profiling;
#[cfg(feature = "python")]
mod python;
pub mod query;
pub mod raycast;
#[cfg(feature = "replay")]
pub mod replay;
pub mod rng;
pub mod rope;
pub mod sdf_adaptive;
pub mod sdf_ccd;
pub mod sdf_collider;
pub mod sdf_destruction;
pub mod sdf_force;
pub mod sdf_manifold;
pub mod sim_field;
pub mod sim_modifier;
pub mod sleeping;
pub mod solver;
pub mod thermal;
pub mod trimesh;
pub mod vehicle;
#[cfg(feature = "wasm")]
mod wasm;

// Re-export commonly used types
pub use animation_blend::{AnimationBlender, AnimationClip, BlendMode, SkeletonPose};
pub use articulation::{ArticulatedBody, FeatherstoneSolver};
pub use audio_physics::{AudioConfig, AudioEvent, AudioGenerator, AudioMaterial};
pub use box_collider::OrientedBox;
pub use bvh::{BvhNode, BvhPrimitive, LinearBvh};
pub use ccd::{speculative_contact, CcdConfig};
pub use character::{CharacterConfig, CharacterController, MoveResult, PushImpulse};
pub use cloth::{Cloth, ClothConfig};
pub use collider::{Capsule, CollisionResult, ConvexHull, ScaledShape, Sphere, AABB};
pub use compound::{CompoundShape, ShapeRef};
pub use contact_cache::{BodyPairKey, ContactCache, ContactManifold};
pub use convex_decompose::{DecomposeConfig, DecompositionResult};
pub use cylinder::Cylinder;
pub use debug_render::{debug_draw_world, DebugColor, DebugDrawData, DebugDrawFlags};
pub use deformable::{DeformableBody, DeformableConfig};
pub use dynamic_bvh::DynamicAabbTree;
pub use erosion::{ErosionConfig, ErosionModifier, ErosionType};
pub use event::{ContactEvent, ContactEventType, EventCollector};
pub use filter::CollisionFilter;
pub use fluid::{Fluid, FluidConfig};
#[cfg(feature = "std")]
pub use fluid_netcode::{FluidDelta, FluidSnapshot};
pub use force::{ForceField, ForceFieldInstance};
pub use fracture::{Crack, FractureConfig, FractureModifier};
#[cfg(feature = "std")]
pub use gpu_sdf::{GpuDispatchConfig, GpuSdfBatch, GpuSdfQuery, GpuSdfResult};
pub use heightfield::HeightField;
pub use interpolation::{BodySnapshot, InterpolationState, WorldSnapshot};
pub use joint::solve_joints_breakable;
pub use joint::{
    BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint, HingeJoint, Joint, SliderJoint,
    SpringJoint,
};
pub use material::{CombineRule, CombinedMaterial, MaterialId, MaterialTable, PhysicsMaterial};
pub use math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};
pub use motor::{JointMotor, MotorMode, PdController};
pub use netcode::{
    DeterministicSimulation, FrameInput, InputApplicator, NetcodeConfig, SimulationChecksum,
    SimulationSnapshot,
};
pub use phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
pub use pressure::{PressureConfig, PressureModifier};
pub use profiling::{PhysicsProfiler, ProfileEntry, StepStats};
pub use query::{
    batch_raycast, batch_sphere_cast, capsule_cast, overlap_aabb, overlap_sphere, sphere_cast,
    BatchRayQuery, OverlapResult, ShapeCastHit,
};
pub use raycast::{
    raycast_all_aabbs, raycast_all_spheres, raycast_any_aabbs, raycast_any_spheres, Ray, RayHit,
};
pub use rng::DeterministicRng;
pub use rope::{Rope, RopeConfig};
pub use sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
pub use sdf_ccd::SdfCcdConfig;
pub use sdf_collider::{ClosureSdf, SdfCollider, SdfField};
pub use sdf_destruction::{DestructibleSdf, DestructionShape};
pub use sdf_force::{SdfForceField, SdfForceType};
pub use sdf_manifold::{ManifoldConfig, SdfManifold};
pub use sim_field::{ScalarField3D, VectorField3D};
pub use sim_modifier::{ModifiedSdf, PhysicsModifier, SingleModifiedSdf};
pub use sleeping::{IslandManager, SleepConfig, SleepData, SleepState};
#[cfg(feature = "std")]
pub use solver::ContactModifier;
pub use solver::{
    BodyType, ContactConstraint, DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody,
};
pub use thermal::{HeatSource, ThermalConfig, ThermalModifier};
pub use trimesh::{TriMesh, Triangle};
pub use vehicle::{Vehicle, VehicleConfig};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::animation_blend::{AnimationBlender, AnimationClip, BlendMode, SkeletonPose};
    pub use crate::articulation::{ArticulatedBody, FeatherstoneSolver};
    pub use crate::audio_physics::{AudioConfig, AudioEvent, AudioGenerator, AudioMaterial};
    pub use crate::box_collider::OrientedBox;
    pub use crate::bvh::{BvhNode, BvhPrimitive, LinearBvh};
    pub use crate::ccd::{speculative_contact, CcdConfig};
    pub use crate::character::{CharacterConfig, CharacterController, MoveResult, PushImpulse};
    pub use crate::cloth::{Cloth, ClothConfig};
    pub use crate::collider::{
        Capsule, CollisionResult, ConvexHull, ScaledShape, Sphere, Support, AABB,
    };
    pub use crate::compound::{CompoundShape, ShapeRef};
    pub use crate::contact_cache::{BodyPairKey, ContactCache, ContactManifold};
    pub use crate::convex_decompose::{DecomposeConfig, DecompositionResult};
    pub use crate::cylinder::Cylinder;
    pub use crate::debug_render::{debug_draw_world, DebugColor, DebugDrawData, DebugDrawFlags};
    pub use crate::deformable::{DeformableBody, DeformableConfig};
    pub use crate::dynamic_bvh::DynamicAabbTree;
    pub use crate::erosion::{ErosionConfig, ErosionModifier, ErosionType};
    pub use crate::event::{ContactEvent, ContactEventType, EventCollector};
    pub use crate::filter::CollisionFilter;
    pub use crate::fluid::{Fluid, FluidConfig};
    #[cfg(feature = "std")]
    pub use crate::fluid_netcode::{FluidDelta, FluidSnapshot};
    pub use crate::force::{ForceField, ForceFieldInstance};
    pub use crate::fracture::{Crack, FractureConfig, FractureModifier};
    #[cfg(feature = "std")]
    pub use crate::gpu_sdf::{GpuDispatchConfig, GpuSdfBatch, GpuSdfQuery, GpuSdfResult};
    pub use crate::heightfield::HeightField;
    pub use crate::interpolation::{BodySnapshot, InterpolationState, WorldSnapshot};
    pub use crate::joint::solve_joints_breakable;
    pub use crate::joint::{
        BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint, HingeJoint, Joint, SliderJoint,
        SpringJoint,
    };
    pub use crate::material::{
        CombineRule, CombinedMaterial, MaterialId, MaterialTable, PhysicsMaterial,
    };
    pub use crate::math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};
    pub use crate::motor::{JointMotor, MotorMode, PdController};
    pub use crate::netcode::{
        DeterministicSimulation, FrameInput, InputApplicator, NetcodeConfig, SimulationChecksum,
        SimulationSnapshot,
    };
    #[cfg(feature = "neural")]
    pub use crate::neural::{
        fix128_hard_tanh, fix128_leaky_relu, fix128_relu, fix128_tanh_approx,
        fix128_ternary_matvec, Activation, ControllerConfig, ControllerOutput,
        DeterministicNetwork, FixedTernaryWeight, RagdollController,
    };
    pub use crate::phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
    pub use crate::pressure::{PressureConfig, PressureModifier};
    pub use crate::profiling::{PhysicsProfiler, ProfileEntry, StepStats};
    pub use crate::query::{
        batch_raycast, batch_sphere_cast, capsule_cast, overlap_aabb, overlap_sphere, sphere_cast,
        BatchRayQuery, OverlapResult, ShapeCastHit,
    };
    pub use crate::raycast::{
        raycast_all_aabbs, raycast_all_spheres, raycast_any_aabbs, raycast_any_spheres, Ray, RayHit,
    };
    pub use crate::rng::DeterministicRng;
    pub use crate::rope::{Rope, RopeConfig};
    pub use crate::sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
    pub use crate::sdf_ccd::SdfCcdConfig;
    pub use crate::sdf_collider::{ClosureSdf, SdfCollider, SdfField};
    pub use crate::sdf_destruction::{DestructibleSdf, DestructionShape};
    pub use crate::sdf_force::{SdfForceField, SdfForceType};
    pub use crate::sdf_manifold::{ManifoldConfig, SdfManifold};
    pub use crate::sim_field::{ScalarField3D, VectorField3D};
    pub use crate::sim_modifier::{ModifiedSdf, PhysicsModifier, SingleModifiedSdf};
    pub use crate::sleeping::{IslandManager, SleepConfig, SleepData, SleepState};
    #[cfg(feature = "std")]
    pub use crate::solver::ContactModifier;
    pub use crate::solver::{
        BodyType, ContactConstraint, DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody,
    };
    pub use crate::thermal::{HeatSource, ThermalConfig, ThermalModifier};
    pub use crate::trimesh::{TriMesh, Triangle};
    pub use crate::vehicle::{Vehicle, VehicleConfig};
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
        let mut body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 100, 0), Fix128::ONE);
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
        let body = RigidBody::new_dynamic(Vec3Fix::from_int(10, 20, 30), Fix128::ONE);
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
        let body_a = RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
        let body_b = RigidBody::new_dynamic(Vec3Fix::from_int(5, 10, 0), Fix128::ONE);
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
        assert!(
            error < Fix128::ONE,
            "Distance constraint violated: error = {:?}",
            error
        );
    }
}
