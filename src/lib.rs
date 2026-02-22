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
//! - [`sketch`]: Probabilistic sketches (HyperLogLog, DDSketch, Count-Min)
//! - [`anomaly`]: Streaming anomaly detection (MAD, EWMA, Z-score)
//! - [`privacy`]: Local differential privacy (Laplace, RAPPOR)
//! - [`pipeline`]: Lock-free metric aggregation pipeline
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

#[cfg(all(feature = "wasm", feature = "ffi"))]
compile_error!("Features `wasm` and `ffi` are mutually exclusive. Enable only one.");

#[cfg(all(feature = "wasm", not(feature = "std")))]
compile_error!("Feature `wasm` requires `std`.");

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "analytics")]
pub mod analytics_bridge;
#[cfg(feature = "std")]
pub mod anomaly;
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
#[cfg(feature = "std")]
pub mod convex_decompose;
pub mod cylinder;
#[cfg(feature = "replay")]
pub mod db_bridge;
pub mod debug_render;
pub mod deformable;
pub mod dynamic_bvh;
pub mod error;
#[cfg(feature = "std")]
pub mod erosion;
pub mod event;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod filter;
pub mod fluid;
#[cfg(feature = "std")]
pub mod fluid_netcode;
pub mod force;
#[cfg(feature = "std")]
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
#[cfg(feature = "std")]
pub mod phase_change;
#[cfg(feature = "std")]
pub mod pipeline;
#[cfg(feature = "std")]
pub mod pressure;
#[cfg(feature = "std")]
pub mod privacy;
pub mod profiling;
#[cfg(feature = "python")]
mod python;
pub mod query;
pub mod raycast;
#[cfg(feature = "replay")]
pub mod replay;
pub mod rng;
pub mod rope;
#[cfg(feature = "std")]
pub mod sdf_adaptive;
pub mod sdf_ccd;
pub mod sdf_collider;
#[cfg(feature = "std")]
pub mod sdf_destruction;
pub mod sdf_force;
pub mod sdf_manifold;
#[cfg(feature = "std")]
pub mod sim_field;
#[cfg(feature = "std")]
pub mod sim_modifier;

#[cfg(feature = "std")]
pub mod sketch;
pub mod sleeping;
pub mod solver;
pub mod spatial;
#[cfg(feature = "std")]
pub mod thermal;
pub mod trimesh;
pub mod vehicle;
#[cfg(feature = "wasm")]
mod wasm;

// Re-export commonly used types
#[cfg(feature = "std")]
pub use anomaly::{AnomalyCallback, AnomalyEvent, CompositeDetector, EwmaDetector, MadDetector, StreamingMedian, ZScoreDetector};
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
#[cfg(feature = "std")]
pub use convex_decompose::{DecomposeConfig, DecompositionResult};
pub use cylinder::Cylinder;
pub use debug_render::{debug_draw_world, DebugColor, DebugDrawData, DebugDrawFlags};
pub use deformable::{DeformableBody, DeformableConfig};
pub use dynamic_bvh::DynamicAabbTree;
pub use error::PhysicsError;
#[cfg(feature = "std")]
pub use erosion::{ErosionConfig, ErosionModifier, ErosionType};
pub use event::{ContactEvent, ContactEventType, EventCollector};
pub use filter::CollisionFilter;
pub use fluid::{Fluid, FluidConfig};
#[cfg(feature = "std")]
pub use fluid_netcode::{FluidDelta, FluidSnapshot};
pub use force::{ForceField, ForceFieldInstance};
#[cfg(feature = "std")]
pub use fracture::{Crack, FractureConfig, FractureModifier};
#[cfg(feature = "std")]
pub use gpu_sdf::{
    batch_size, GpuDispatchConfig, GpuSdfBatch, GpuSdfInstancedBatch, GpuSdfMultiDispatch,
    GpuSdfQuery, GpuSdfResult,
};
pub use heightfield::HeightField;
pub use interpolation::{BodySnapshot, InterpolationState, WorldSnapshot};
pub use joint::solve_joints_breakable;
pub use joint::{
    BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint, HingeJoint, Joint, SliderJoint,
    SpringJoint,
};
pub use material::{CombineRule, CombinedMaterial, MaterialId, MaterialTable, PhysicsMaterial};
pub use math::{simd_width, Fix128, Mat3Fix, QuatFix, Vec3Fix, SIMD_WIDTH};
pub use motor::{JointMotor, MotorMode, PdController};
pub use netcode::{
    DeterministicSimulation, FrameInput, InputApplicator, NetcodeConfig, SimulationChecksum,
    SimulationSnapshot,
};
#[cfg(feature = "std")]
pub use phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
#[cfg(feature = "std")]
pub use pipeline::{MetricEvent, MetricPipeline, MetricRegistry, MetricSnapshot, MetricType, RingBuffer};
#[cfg(feature = "std")]
pub use pressure::{PressureConfig, PressureModifier};
#[cfg(feature = "std")]
pub use privacy::{LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse, Rappor, XorShift64};
pub use profiling::{PhysicsProfiler, ProfileEntry, StepStats};
pub use query::{
    batch_raycast, batch_sphere_cast, capsule_cast, overlap_aabb, overlap_aabb_bvh,
    overlap_sphere, overlap_sphere_bvh, sphere_cast, BatchRayQuery, OverlapResult, ShapeCastHit,
};
pub use raycast::{
    raycast_all_aabbs, raycast_all_spheres, raycast_any_aabbs, raycast_any_spheres, Ray, RayHit,
};
pub use rng::DeterministicRng;
pub use rope::{Rope, RopeConfig};
#[cfg(feature = "std")]
pub use sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
pub use sdf_ccd::SdfCcdConfig;
pub use sdf_collider::{ClosureSdf, SdfCollider, SdfField};
#[cfg(feature = "std")]
pub use sdf_destruction::{DestructibleSdf, DestructionShape};
pub use sdf_force::{SdfForceField, SdfForceType};
pub use sdf_manifold::{ManifoldConfig, SdfManifold};
#[cfg(feature = "std")]
pub use sim_field::{ScalarField3D, VectorField3D};
#[cfg(feature = "std")]
pub use sim_modifier::{ModifiedSdf, PhysicsModifier, SingleModifiedSdf};
#[cfg(feature = "std")]
pub use sketch::{CountMinSketch, DDSketch, FnvHasher, HeavyHitters, HyperLogLog, Mergeable};
pub use sleeping::{IslandManager, SleepConfig, SleepData, SleepState};
pub use spatial::SpatialGrid;
#[cfg(feature = "std")]
pub use solver::ContactModifier;
pub use solver::{
    BodyType, ContactConstraint, DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody,
};
#[cfg(feature = "std")]
pub use thermal::{HeatSource, ThermalConfig, ThermalModifier};
pub use trimesh::{TriMesh, Triangle};
pub use vehicle::{Vehicle, VehicleConfig};

/// Prelude module for convenient imports
pub mod prelude {
    #[cfg(feature = "std")]
    pub use crate::anomaly::{AnomalyCallback, AnomalyEvent, CompositeDetector, EwmaDetector, MadDetector, StreamingMedian, ZScoreDetector};
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
    #[cfg(feature = "std")]
    pub use crate::convex_decompose::{DecomposeConfig, DecompositionResult};
    pub use crate::cylinder::Cylinder;
    pub use crate::debug_render::{debug_draw_world, DebugColor, DebugDrawData, DebugDrawFlags};
    pub use crate::deformable::{DeformableBody, DeformableConfig};
    pub use crate::dynamic_bvh::DynamicAabbTree;
    pub use crate::error::PhysicsError;
    #[cfg(feature = "std")]
    pub use crate::erosion::{ErosionConfig, ErosionModifier, ErosionType};
    pub use crate::event::{ContactEvent, ContactEventType, EventCollector};
    pub use crate::filter::CollisionFilter;
    pub use crate::fluid::{Fluid, FluidConfig};
    #[cfg(feature = "std")]
    pub use crate::fluid_netcode::{FluidDelta, FluidSnapshot};
    pub use crate::force::{ForceField, ForceFieldInstance};
    #[cfg(feature = "std")]
    pub use crate::fracture::{Crack, FractureConfig, FractureModifier};
    #[cfg(feature = "std")]
    pub use crate::gpu_sdf::{
        batch_size, GpuDispatchConfig, GpuSdfBatch, GpuSdfInstancedBatch, GpuSdfMultiDispatch,
        GpuSdfQuery, GpuSdfResult,
    };
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
    pub use crate::math::{simd_width, Fix128, Mat3Fix, QuatFix, Vec3Fix, SIMD_WIDTH};
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
    #[cfg(feature = "std")]
    pub use crate::phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
    #[cfg(feature = "std")]
    pub use crate::pipeline::{MetricEvent, MetricPipeline, MetricRegistry, MetricSnapshot, MetricType, RingBuffer};
    #[cfg(feature = "std")]
    pub use crate::pressure::{PressureConfig, PressureModifier};
    #[cfg(feature = "std")]
    pub use crate::privacy::{LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse, Rappor, XorShift64};
    pub use crate::profiling::{PhysicsProfiler, ProfileEntry, StepStats};
    pub use crate::query::{
        batch_raycast, batch_sphere_cast, capsule_cast, overlap_aabb, overlap_aabb_bvh,
        overlap_sphere, overlap_sphere_bvh, sphere_cast, BatchRayQuery, OverlapResult,
        ShapeCastHit,
    };
    pub use crate::raycast::{
        raycast_all_aabbs, raycast_all_spheres, raycast_any_aabbs, raycast_any_spheres, Ray, RayHit,
    };
    pub use crate::rng::DeterministicRng;
    pub use crate::rope::{Rope, RopeConfig};
    #[cfg(feature = "std")]
    pub use crate::sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
    pub use crate::sdf_ccd::SdfCcdConfig;
    pub use crate::sdf_collider::{ClosureSdf, SdfCollider, SdfField};
    #[cfg(feature = "std")]
    pub use crate::sdf_destruction::{DestructibleSdf, DestructionShape};
    pub use crate::sdf_force::{SdfForceField, SdfForceType};
    pub use crate::sdf_manifold::{ManifoldConfig, SdfManifold};
    #[cfg(feature = "std")]
    pub use crate::sim_field::{ScalarField3D, VectorField3D};
    #[cfg(feature = "std")]
    pub use crate::sim_modifier::{ModifiedSdf, PhysicsModifier, SingleModifiedSdf};
    #[cfg(feature = "std")]
    pub use crate::sketch::{CountMinSketch, DDSketch, FnvHasher, HeavyHitters, HyperLogLog, Mergeable};
    pub use crate::sleeping::{IslandManager, SleepConfig, SleepData, SleepState};
    pub use crate::spatial::SpatialGrid;
    #[cfg(feature = "std")]
    pub use crate::solver::ContactModifier;
    pub use crate::solver::{
        BodyType, ContactConstraint, DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody,
    };
    #[cfg(feature = "std")]
    pub use crate::thermal::{HeatSource, ThermalConfig, ThermalModifier};
    pub use crate::trimesh::{TriMesh, Triangle};
    pub use crate::vehicle::{Vehicle, VehicleConfig};
}

#[cfg(test)]
mod compile_smoke_tests {
    //! Verify that core types from the prelude and key modules are accessible.
    //! These tests catch accidental breakage of public re-exports.

    use super::*;

    #[test]
    fn test_prelude_types_accessible() {
        let _ = Fix128::ZERO;
        let _ = Vec3Fix::ZERO;
        let _ = QuatFix::IDENTITY;
        let _ = Mat3Fix::IDENTITY;
        let _ = PhysicsConfig::default();
        let _ = CollisionFilter::default();
        let _ = SleepConfig::default();
    }

    #[test]
    fn test_error_type_accessible() {
        let e = PhysicsError::InvalidBodyIndex { index: 0, count: 0 };
        let _ = format!("{}", e);
    }

    #[test]
    fn test_spatial_grid_accessible() {
        let mut grid = SpatialGrid::new(Fix128::ONE, 16);
        grid.insert(0, Vec3Fix::ZERO);
        let mut neighbors = Vec::new();
        grid.query_neighbors_into(Vec3Fix::ZERO, Fix128::ONE, &mut neighbors);
        assert!(neighbors.contains(&0));
    }

    #[test]
    fn test_wasm_ffi_mutual_exclusion() {
        // The compile_error! macros in lib.rs guarantee that `wasm` + `ffi`
        // cannot be enabled together. This test documents the invariant.
        // If this crate compiles at all, the exclusion is enforced.
        assert!(true);
    }
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
            cached_lambda: Fix128::ZERO,
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
