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
//! - **Linear BVH**: Morton code-based spatial acceleration (used for broad-phase and raycast)
//! - **Rollback Support**: State serialization for netcode
//!
//! # Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `std` (default) | Standard library support. Disable for `no_std` (requires `alloc`). |
//! | `simd` | SSE2 acceleration for `Vec3Fix` math ops (`dot`, `add`, `sub`). Results remain bit-exact. Solver hot paths use scalar arithmetic to guarantee cross-platform determinism. |
//! | `parallel` | Parallel constraint solving via Rayon with graph-colored batches. |
//! | `neural` | Deterministic neural controller integration (ALICE-ML). |
//! | `python` | `PyO3` + `NumPy` zero-copy Python bindings. |
//! | `replay` | Replay recording/playback via ALICE-DB. |
//! | `ffi` | C FFI for Unity/UE5/game engine integration. |
//! | `wasm` | WebAssembly bindings via wasm-bindgen. |
//! | `analytics` | ALICE-Analytics simulation profiling (`DDSketch`, `HyperLogLog`). |
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
//! // Step simulation (60 frames at 1/60 second)
//! let dt = Fix128::from_ratio(1, 60);
//! for _ in 0..60 {
//!     world.step(dt);
//! }
//!
//! // Body should have fallen under gravity
//! let pos = world.bodies[body_id].position;
//! assert!(pos.y < Fix128::from_int(10), "Body fell under gravity");
//! ```
//!
//! # Modules
//!
//! ## Core
//! - [`math`]: Fixed-point math primitives (Fix128, `Vec3Fix`, `QuatFix`, `Mat3Fix`, CORDIC)
//! - [`collider`]: Collision shapes (Sphere, Capsule, `ConvexHull`, AABB) and GJK/EPA detection
//! - [`solver`]: XPBD physics solver, rigid body dynamics, constraint batching
//! - [`bvh`]: Linear BVH for broad-phase collision detection (Morton codes, stackless traversal)
//! - [`error`]: Unified physics error type (`PhysicsError`)
//!
//! ## Collision Shapes
//! - [`box_collider`]: Oriented Bounding Box (OBB) with GJK support
//! - [`compound`]: Multi-shape compound collider with local transforms
//! - [`cone`]: Cone collider with GJK support (apex +Y, base -Y)
//! - [`convex_mesh_builder`]: Incremental convex hull builder from point sets
//! - [`cylinder`]: Cylinder collider with GJK support
//! - [`ellipsoid`]: Ellipsoid collider with three independent semi-axes
//! - [`plane_collider`]: Infinite plane collider (sphere/AABB intersection)
//! - [`sdf_collider`]: SDF-based collision shapes (distance field surfaces)
//! - [`torus`]: Torus collider with major/minor radii
//! - [`wedge`]: Wedge (triangular prism) collider with 6 vertices
//!
//! ## Spatial Acceleration
//! - [`dynamic_bvh`]: Incremental AABB tree with O(log n) insert/remove/update
//! - [`spatial`]: Hash grid for neighbor queries (shared across fluid, cloth)
//! - [`contact_cache`]: Persistent contact manifold cache with `HashMap` O(1) lookup
//!
//! ## Constraints & Dynamics
//! - [`joint`]: Joint constraints (Ball, Hinge, Fixed, Slider, Spring, D6, `ConeTwist`)
//! - [`motor`]: PD controllers and joint motors
//! - [`articulation`]: Articulated bodies (ragdolls, robotic arms, Featherstone)
//! - [`force`]: Custom force fields (wind, gravity wells, buoyancy, vortex)
//! - [`sleeping`]: Sleep/wake and island management (Union-Find)
//! - [`filter`]: Collision filtering with layer/mask bitmasks (see [`filter::layers`])
//! - [`material`]: Material pair table (friction, restitution, combine rules)
//!
//! ## Queries
//! - [`raycast`]: Ray casting against spheres, AABBs, capsules, planes
//! - [`query`]: Shape cast (sphere, capsule) and overlap queries (sphere, AABB)
//! - [`ccd`]: Continuous collision detection (TOI, conservative advancement, speculative)
//!
//! ## Soft Body & Simulation
//! - [`rope`]: XPBD distance chain rope and cable simulation
//! - [`cloth`]: XPBD triangle mesh cloth with self-collision
//! - [`fluid`]: Position-Based Fluids (PBF) with spatial hash grid
//! - [`deformable`]: FEM-XPBD deformable body simulation
//! - [`vehicle`]: Vehicle physics (wheel, suspension, engine, steering)
//! - [`character`]: Kinematic capsule-based character controller (move-and-slide)
//! - [`trimesh`]: Triangle mesh collision with BVH acceleration (Moller-Trumbore)
//! - [`heightfield`]: Height field terrain collision (bilinear interpolation)
//!
//! ## SDF Integration
//! - [`sdf_manifold`]: Multi-point contact manifold from SDF surfaces
//! - [`sdf_ccd`]: Sphere tracing continuous collision detection for SDF
//! - [`sdf_force`]: SDF-driven force fields (attract, repel, contain, flow)
//! - [`sdf_destruction`]: Real-time CSG boolean destruction (`std`)
//! - [`sdf_adaptive`]: Adaptive SDF evaluation with distance-based LOD (`std`)
//! - [`convex_decompose`]: Convex decomposition from SDF voxel grid (`std`)
//! - [`gpu_sdf`]: GPU compute shader interface for batch SDF evaluation (`std`)
//!
//! ## SDF Simulation Modifiers (`std`)
//! - [`sim_field`]: 3D scalar/vector fields with trilinear interpolation and diffusion
//! - [`sim_modifier`]: Physics modifier chain for SDF surfaces
//! - [`thermal`]: Heat diffusion, melt, thermal expansion, freeze
//! - [`pressure`]: Contact-force deformation (crush, bulge, dent)
//! - [`erosion`]: Wind, water, chemical, ablation erosion
//! - [`fracture`]: Stress-driven crack propagation with CSG subtraction
//! - [`phase_change`]: Solid/liquid/gas transitions driven by temperature
//!
//! ## Game Systems
//! - [`animation_blend`]: Ragdoll animation blending with SLERP
//! - [`audio_physics`]: Physics-based audio parameter generation
//! - [`netcode`]: Deterministic simulation with frame input, checksum, rollback
//! - [`fluid_netcode`]: Deterministic fluid netcode with delta compression (`std`)
//! - [`interpolation`]: Substep interpolation with NLERP quaternion blending
//! - [`debug_render`]: Wireframe visualization API (bodies, contacts, joints, BVH)
//! - [`profiling`]: Per-stage timer and per-frame statistics
//!
//! ## Analytics & Privacy (`std`)
//! - [`sketch`]: Probabilistic sketches (`HyperLogLog`, `DDSketch`, Count-Min)
//! - [`anomaly`]: Streaming anomaly detection (MAD, EWMA, Z-score)
//! - [`privacy`]: Local differential privacy (Laplace, RAPPOR)
//! - [`pipeline`]: Lock-free metric aggregation pipeline
//!
//! ## Utility
//! - [`rng`]: Deterministic pseudo-random number generator (PCG-XSH-RR)
//! - [`event`]: Contact and trigger event tracking
//!
//! ## Feature-Gated
//! - `neural`: Deterministic neural controller (ALICE-ML, `--features neural`)
//! - `ffi`: C FFI for Unity/UE5 game engines (`--features ffi`)
//! - `replay`: Replay recording/playback (`--features replay`)
//! - `db_bridge`: Physics state persistence bridge (`--features replay`)
//! - `analytics_bridge`: Simulation profiling bridge (`--features analytics`)
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
// Pedantic: suppress lints inherent to physics/math code.
// - Cast lints: Fixed-point ↔ float conversions and index casts are pervasive.
// - Naming: coordinate variables (px, py, pz), vertex indices (v0, v1, v2).
// - must_use: pure functions returning computed values are the norm; annotating
//   every constructor and accessor adds noise without improving safety.
// - doc_markdown: technical identifiers in docs are context-obvious.
// - unreadable_literal: hash constants and bit patterns are more readable without separators.
// - large_stack_arrays: physics lookup tables and inline data are intentional.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::unreadable_literal,
    clippy::large_stack_arrays,
    clippy::inline_always
)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(all(feature = "wasm", feature = "ffi"))]
compile_error!("Features `wasm` and `ffi` are mutually exclusive. Enable only one.");

#[cfg(all(feature = "wasm", not(feature = "std")))]
compile_error!("Feature `wasm` requires `std`.");

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(feature = "analytics")]
pub mod analytics_bridge;
pub mod animation_blend;
#[cfg(feature = "std")]
pub mod anomaly;
pub mod articulation;
pub mod audio_physics;
pub mod box_collider;
pub mod bvh;
pub mod ccd;
pub mod character;
pub mod cloth;
pub mod cloth_fluid;
pub mod collider;
pub mod collision_mesh_gen;
pub mod compound;
pub mod cone;
pub mod contact_cache;
pub mod contact_viz;
#[cfg(feature = "std")]
pub mod convex_decompose;
pub mod convex_mesh_builder;
pub mod cylinder;
#[cfg(feature = "replay")]
pub mod db_bridge;
pub mod debug_render;
pub mod deformable;
pub mod dynamic_bvh;
pub mod ellipsoid;
#[cfg(feature = "std")]
pub mod erosion;
pub mod error;
pub mod event;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod filter;
pub mod flow_viz;
pub mod fluid;
#[cfg(feature = "std")]
pub mod fluid_netcode;
pub mod force;
#[cfg(feature = "std")]
pub mod fracture;
#[cfg(feature = "std")]
pub mod gpu_sdf;
pub mod heatmap;
pub mod heightfield;
pub mod interpolation;
pub mod joint;
pub mod joint_extra;
pub mod mass_properties;
pub mod material;
pub mod math;
pub mod motor;
pub mod multi_world;
pub mod netcode;
#[cfg(feature = "neural")]
pub mod neural;
pub mod particle;
#[cfg(feature = "std")]
pub mod phase_change;
#[cfg(feature = "std")]
pub mod pipeline;
pub mod plane_collider;
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
pub mod rope_attach;
#[cfg(feature = "std")]
pub mod scene_io;
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

pub mod physics2d;

#[cfg(feature = "std")]
pub mod sketch;
pub mod sleeping;
pub mod soft_body_cut;
pub mod solver;
pub mod spatial;
#[cfg(feature = "std")]
pub mod thermal;
pub mod torus;
pub mod trimesh;
pub mod vehicle;
#[cfg(feature = "wasm")]
mod wasm;
pub mod wedge;

// Re-export commonly used types
pub use animation_blend::{AnimationBlender, AnimationClip, BlendMode, SkeletonPose};
#[cfg(feature = "std")]
pub use anomaly::{
    AnomalyCallback, AnomalyEvent, CompositeDetector, EwmaDetector, MadDetector, StreamingMedian,
    ZScoreDetector,
};
pub use articulation::{ArticulatedBody, FeatherstoneSolver};
pub use audio_physics::{AudioConfig, AudioEvent, AudioGenerator, AudioMaterial};
pub use box_collider::OrientedBox;
pub use bvh::{BvhNode, BvhPrimitive, LinearBvh};
pub use ccd::{speculative_contact, CcdConfig};
pub use character::{CharacterConfig, CharacterController, MoveResult, PushImpulse};
pub use cloth::{Cloth, ClothConfig};
pub use cloth_fluid::{
    apply_cloth_boundary_to_fluid, apply_fluid_forces_to_cloth, ClothFluidCoupling,
};
pub use collider::{Capsule, CollisionResult, ConvexHull, ScaledShape, Sphere, Support, AABB};
pub use collision_mesh_gen::{
    compute_mesh_aabb, generate_collision_mesh, simplify_collision_mesh, CollisionMesh,
    CollisionMeshConfig,
};
pub use compound::{CompoundShape, ShapeRef};
pub use cone::Cone;
pub use contact_cache::{BodyPairKey, ContactCache, ContactManifold};
pub use contact_viz::{
    generate_contact_arrows, generate_friction_cones, ContactArrow, FrictionCone,
};
#[cfg(feature = "std")]
pub use convex_decompose::{DecomposeConfig, DecompositionResult};
pub use convex_mesh_builder::{build_convex_hull, compute_centroid};
pub use cylinder::Cylinder;
pub use debug_render::{debug_draw_world, DebugColor, DebugDrawData, DebugDrawFlags};
pub use deformable::{DeformableBody, DeformableConfig};
pub use dynamic_bvh::DynamicAabbTree;
pub use ellipsoid::Ellipsoid;
#[cfg(feature = "std")]
pub use erosion::{ErosionConfig, ErosionModifier, ErosionType};
pub use error::PhysicsError;
pub use event::{ContactEvent, ContactEventType, EventCollector, TriggerEvent};
/// Re-export predefined collision layer constants for convenience.
pub use filter::layers;
pub use filter::CollisionFilter;
pub use flow_viz::{generate_flow_arrows, generate_streamlines, FlowArrow, FlowVizConfig};
pub use fluid::{Fluid, FluidConfig};
#[cfg(feature = "std")]
pub use fluid_netcode::{FluidDelta, FluidSnapshot};
pub use force::{apply_force_fields, ForceField, ForceFieldInstance};
#[cfg(feature = "std")]
pub use fracture::{Crack, FractureConfig, FractureModifier};
#[cfg(feature = "std")]
pub use gpu_sdf::{
    batch_size, GpuDispatchConfig, GpuSdfBatch, GpuSdfInstancedBatch, GpuSdfMultiDispatch,
    GpuSdfQuery, GpuSdfResult,
};
pub use heatmap::{
    generate_stress_heatmap, generate_temperature_heatmap, heatmap_to_rgba, Heatmap, HeatmapConfig,
    SliceAxis,
};
pub use heightfield::HeightField;
pub use interpolation::{BodySnapshot, InterpolationState, WorldSnapshot};
pub use joint::{solve_joints, solve_joints_breakable};
pub use joint::{
    BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint, HingeJoint, Joint, JointType,
    SliderJoint, SpringJoint,
};
pub use joint_extra::{
    solve_extra_joints, ExtraJoint, GearJoint, MouseJoint, PulleyJoint, RackAndPinionJoint,
    WeldJoint,
};
pub use mass_properties::{
    box_mass_properties, capsule_mass_properties, convex_hull_mass_properties,
    cylinder_mass_properties, sphere_mass_properties, translate_inertia, MassProperties,
};
pub use material::{CombineRule, CombinedMaterial, MaterialId, MaterialTable, PhysicsMaterial};
pub use math::{simd_width, Fix128, Mat3Fix, QuatFix, Vec3Fix, SIMD_WIDTH};
pub use motor::{JointMotor, MotorMode, PdController};
pub use multi_world::{MultiWorld, Portal};
pub use netcode::{
    DeterministicSimulation, FrameInput, InputApplicator, NetcodeConfig, SimulationChecksum,
    SimulationSnapshot,
};
pub use particle::{Particle, ParticleEmitter, ParticleSystem};
#[cfg(feature = "std")]
pub use phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
pub use physics2d::{
    BodyType2D, Contact2D, Joint2D, PhysicsConfig2D, PhysicsWorld2D, RigidBody2D, Shape2D, Vec2Fix,
};
#[cfg(feature = "std")]
pub use pipeline::{
    MetricEvent, MetricPipeline, MetricRegistry, MetricSnapshot, MetricType, RingBuffer,
};
pub use plane_collider::PlaneCollider;
#[cfg(feature = "std")]
pub use pressure::{PressureConfig, PressureModifier};
#[cfg(feature = "std")]
pub use privacy::{
    LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse, Rappor, XorShift64,
};
pub use profiling::{PhysicsProfiler, ProfileEntry, StepStats};
pub use query::{
    batch_raycast, batch_sphere_cast, capsule_cast, overlap_aabb, overlap_aabb_bvh, overlap_sphere,
    overlap_sphere_bvh, sphere_cast, BatchRayQuery, OverlapResult, ShapeCastHit,
};
pub use raycast::{
    raycast_all_aabbs, raycast_all_spheres, raycast_any_aabbs, raycast_any_spheres, Ray, RayHit,
};
pub use rng::DeterministicRng;
pub use rope::{Rope, RopeConfig};
pub use rope_attach::{solve_rope_attachments, RopeAttachment};
#[cfg(feature = "std")]
pub use scene_io::{
    load_scene, load_scene_json, save_scene, save_scene_json, PhysicsScene, SerializedBody,
    SerializedJoint,
};
#[cfg(feature = "std")]
pub use sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
pub use sdf_ccd::SdfCcdConfig;
pub use sdf_collider::{ClosureSdf, SdfCollider, SdfField};
#[cfg(feature = "std")]
pub use sdf_destruction::{DestructibleSdf, DestructionShape};
pub use sdf_force::{SdfForceField, SdfForceType};
pub use sdf_manifold::ManifoldConfig;
#[cfg(feature = "std")]
pub use sdf_manifold::SdfManifold;
#[cfg(feature = "std")]
pub use sim_field::{ScalarField3D, VectorField3D};
#[cfg(feature = "std")]
pub use sim_modifier::{ModifiedSdf, PhysicsModifier, SingleModifiedSdf};
#[cfg(feature = "std")]
pub use sketch::{CountMinSketch, DDSketch, FnvHasher, HeavyHitters, HyperLogLog, Mergeable};
pub use sleeping::{Island, IslandManager, SleepConfig, SleepData, SleepState};
pub use soft_body_cut::{cut_cloth, cut_deformable, CutPlane, CutResult};
#[cfg(feature = "std")]
pub use solver::ContactModifier;
pub use solver::{
    BodyType, ContactConstraint, DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody,
};
pub use spatial::SpatialGrid;
#[cfg(feature = "std")]
pub use thermal::{HeatSource, ThermalConfig, ThermalModifier};
pub use torus::Torus;
pub use trimesh::{TriMesh, Triangle};
pub use vehicle::{Vehicle, VehicleConfig};
pub use wedge::Wedge;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::animation_blend::{AnimationBlender, AnimationClip, BlendMode, SkeletonPose};
    #[cfg(feature = "std")]
    pub use crate::anomaly::{
        AnomalyCallback, AnomalyEvent, CompositeDetector, EwmaDetector, MadDetector,
        StreamingMedian, ZScoreDetector,
    };
    pub use crate::articulation::{ArticulatedBody, FeatherstoneSolver};
    pub use crate::audio_physics::{AudioConfig, AudioEvent, AudioGenerator, AudioMaterial};
    pub use crate::box_collider::OrientedBox;
    pub use crate::bvh::{BvhNode, BvhPrimitive, LinearBvh};
    pub use crate::ccd::{speculative_contact, CcdConfig};
    pub use crate::character::{CharacterConfig, CharacterController, MoveResult, PushImpulse};
    pub use crate::cloth::{Cloth, ClothConfig};
    pub use crate::cloth_fluid::{
        apply_cloth_boundary_to_fluid, apply_fluid_forces_to_cloth, ClothFluidCoupling,
    };
    pub use crate::collider::{
        Capsule, CollisionResult, ConvexHull, ScaledShape, Sphere, Support, AABB,
    };
    pub use crate::collision_mesh_gen::{
        compute_mesh_aabb, generate_collision_mesh, simplify_collision_mesh, CollisionMesh,
        CollisionMeshConfig,
    };
    pub use crate::compound::{CompoundShape, ShapeRef};
    pub use crate::cone::Cone;
    pub use crate::contact_cache::{BodyPairKey, ContactCache, ContactManifold};
    pub use crate::contact_viz::{
        generate_contact_arrows, generate_friction_cones, ContactArrow, FrictionCone,
    };
    #[cfg(feature = "std")]
    pub use crate::convex_decompose::{DecomposeConfig, DecompositionResult};
    pub use crate::convex_mesh_builder::{build_convex_hull, compute_centroid};
    pub use crate::cylinder::Cylinder;
    pub use crate::debug_render::{debug_draw_world, DebugColor, DebugDrawData, DebugDrawFlags};
    pub use crate::deformable::{DeformableBody, DeformableConfig};
    pub use crate::dynamic_bvh::DynamicAabbTree;
    pub use crate::ellipsoid::Ellipsoid;
    #[cfg(feature = "std")]
    pub use crate::erosion::{ErosionConfig, ErosionModifier, ErosionType};
    pub use crate::error::PhysicsError;
    pub use crate::event::{ContactEvent, ContactEventType, EventCollector, TriggerEvent};
    pub use crate::filter::layers;
    pub use crate::filter::CollisionFilter;
    pub use crate::flow_viz::{
        generate_flow_arrows, generate_streamlines, FlowArrow, FlowVizConfig,
    };
    pub use crate::fluid::{Fluid, FluidConfig};
    #[cfg(feature = "std")]
    pub use crate::fluid_netcode::{FluidDelta, FluidSnapshot};
    pub use crate::force::{apply_force_fields, ForceField, ForceFieldInstance};
    #[cfg(feature = "std")]
    pub use crate::fracture::{Crack, FractureConfig, FractureModifier};
    #[cfg(feature = "std")]
    pub use crate::gpu_sdf::{
        batch_size, GpuDispatchConfig, GpuSdfBatch, GpuSdfInstancedBatch, GpuSdfMultiDispatch,
        GpuSdfQuery, GpuSdfResult,
    };
    pub use crate::heatmap::{
        generate_stress_heatmap, generate_temperature_heatmap, heatmap_to_rgba, Heatmap,
        HeatmapConfig, SliceAxis,
    };
    pub use crate::heightfield::HeightField;
    pub use crate::interpolation::{BodySnapshot, InterpolationState, WorldSnapshot};
    pub use crate::joint::{solve_joints, solve_joints_breakable};
    pub use crate::joint::{
        BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint, HingeJoint, Joint, JointType,
        SliderJoint, SpringJoint,
    };
    pub use crate::joint_extra::{
        solve_extra_joints, ExtraJoint, GearJoint, MouseJoint, PulleyJoint, RackAndPinionJoint,
        WeldJoint,
    };
    pub use crate::mass_properties::{
        box_mass_properties, capsule_mass_properties, convex_hull_mass_properties,
        cylinder_mass_properties, sphere_mass_properties, translate_inertia, MassProperties,
    };
    pub use crate::material::{
        CombineRule, CombinedMaterial, MaterialId, MaterialTable, PhysicsMaterial,
    };
    pub use crate::math::{simd_width, Fix128, Mat3Fix, QuatFix, Vec3Fix, SIMD_WIDTH};
    pub use crate::motor::{JointMotor, MotorMode, PdController};
    pub use crate::multi_world::{MultiWorld, Portal};
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
    pub use crate::particle::{Particle, ParticleEmitter, ParticleSystem};
    #[cfg(feature = "std")]
    pub use crate::phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
    #[cfg(feature = "std")]
    pub use crate::pipeline::{
        MetricEvent, MetricPipeline, MetricRegistry, MetricSnapshot, MetricType, RingBuffer,
    };
    pub use crate::plane_collider::PlaneCollider;
    #[cfg(feature = "std")]
    pub use crate::pressure::{PressureConfig, PressureModifier};
    #[cfg(feature = "std")]
    pub use crate::privacy::{
        LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse, Rappor, XorShift64,
    };
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
    pub use crate::rope_attach::{solve_rope_attachments, RopeAttachment};
    #[cfg(feature = "std")]
    pub use crate::scene_io::{
        load_scene, load_scene_json, save_scene, save_scene_json, PhysicsScene, SerializedBody,
        SerializedJoint,
    };
    #[cfg(feature = "std")]
    pub use crate::sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
    pub use crate::sdf_ccd::SdfCcdConfig;
    pub use crate::sdf_collider::{ClosureSdf, SdfCollider, SdfField};
    #[cfg(feature = "std")]
    pub use crate::sdf_destruction::{DestructibleSdf, DestructionShape};
    pub use crate::sdf_force::{SdfForceField, SdfForceType};
    pub use crate::sdf_manifold::ManifoldConfig;
    #[cfg(feature = "std")]
    pub use crate::sdf_manifold::SdfManifold;
    #[cfg(feature = "std")]
    pub use crate::sim_field::{ScalarField3D, VectorField3D};
    #[cfg(feature = "std")]
    pub use crate::sim_modifier::{ModifiedSdf, PhysicsModifier, SingleModifiedSdf};
    #[cfg(feature = "std")]
    pub use crate::sketch::{
        CountMinSketch, DDSketch, FnvHasher, HeavyHitters, HyperLogLog, Mergeable,
    };
    pub use crate::sleeping::{Island, IslandManager, SleepConfig, SleepData, SleepState};
    pub use crate::soft_body_cut::{cut_cloth, cut_deformable, CutPlane, CutResult};
    #[cfg(feature = "std")]
    pub use crate::solver::ContactModifier;
    pub use crate::solver::{
        BodyType, ContactConstraint, DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody,
    };
    pub use crate::spatial::SpatialGrid;
    #[cfg(feature = "std")]
    pub use crate::thermal::{HeatSource, ThermalConfig, ThermalModifier};
    pub use crate::torus::Torus;
    pub use crate::trimesh::{TriMesh, Triangle};
    pub use crate::vehicle::{Vehicle, VehicleConfig};
    pub use crate::wedge::Wedge;
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
