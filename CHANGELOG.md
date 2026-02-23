# Changelog

All notable changes to ALICE-Physics will be documented in this file.

## [0.6.0] - 2026-02-23

### Added

- **Per-body damping**: `linear_damping` and `angular_damping` fields on `RigidBody` with builder methods
- **CCD extensions**: `capsule_plane_toi`, `sphere_capsule_toi`, `aabb_plane_toi` functions
- **PhysicsWorld Debug**: manual `Debug` impl showing body/constraint/joint counts
- **RigidBody builder pattern**: `with_restitution`, `with_friction`, `with_gravity_scale`, `with_sensor`, `with_velocity`, `with_rotation`
- **RigidBody Default**: creates 1kg dynamic body at origin
- **Display impls**: `Fix128`, `Vec3Fix`, `QuatFix` now implement `Display`
- **From/Into conversions**: `Fix128` from `i64`/`i32`, `Vec3Fix` ↔ `[Fix128; 3]`, `QuatFix` ↔ `[Fix128; 4]`
- **Error variants**: `PhysicsError::IoError` (std), `CapacityExceeded`, `InvalidConfiguration`
- **HeightField derives**: `Clone`, `Debug`, `PartialEq`
- **PartialEq**: added to 90+ public types across all modules
- **Debug impls**: manual `Debug` for 10 container structs (Cloth, Fluid, Rope, Vehicle, DeformableBody, ParticleSystem, PhysicsWorld2D, MultiWorld, CharacterController, ContactCache)
- **scene_io PartialEq**: `PhysicsScene`, `SerializedBody`, `SerializedJoint`, `PhysicsConfig`
- **SECURITY.md**: security policy and vulnerability reporting guidelines
- **Examples**: `basic_physics` (world setup, builder pattern, simulation loop) and `cloth_simulation` (grid cloth, wind, normals)
- Explanation comments on ignored doc tests (`sdf_collider.rs`)
- 10 new integration tests (stress test, damping, builder, Display, From/Into, CCD, errors, HeightField)

### Changed

- `unwrap()` replaced with `expect()` in `query.rs`, `netcode.rs`, and `contact_cache.rs` for better error messages
- `query.rs`: branchless `map_or` replaces `is_none() || .unwrap()` pattern
- `scene_io.rs`: `scene_to_json` refactored to use inner function with `?` operator (removes 15 `unwrap()` calls)
- `no_std` compatibility: 24 test modules gated with `#[cfg(all(test, feature = "std"))]`

## [0.5.0] - 2026-02-23

### Added

- **Collision shapes** (6 new): `cone`, `ellipsoid`, `torus`, `plane_collider`, `wedge`, `convex_mesh_builder`
  - Cone with apex/base GJK support, AABB, volume, inertia
  - Ellipsoid with 3-axis radii and anisotropic support function
  - Torus with Minkowski sum decomposition support
  - Infinite plane (Hessian normal form) with sphere/AABB intersection
  - Wedge (triangular prism) with 6-vertex support
  - Incremental convex hull builder with full algorithm (tetrahedron init, visible face removal, horizon patching)
- **Joint types** (5 new in `joint_extra`): `PulleyJoint`, `GearJoint`, `WeldJoint` (breakable), `RackAndPinionJoint`, `MouseJoint`
- **Force variants** (2 new): `Explosion` (radial impulse with falloff), `Magnetic` (simplified dipole, 1/r^3)
- **2D physics subsystem** (`physics2d`): complete 2D physics engine
  - `Vec2Fix`, `Shape2D` (Circle/Polygon/Capsule/Edge), `RigidBody2D`, `PhysicsWorld2D`
  - SAT collision detection with Voronoi region classification
  - XPBD 2D solver with substeps
  - Joint2D: Revolute, Distance, Weld, Mouse
  - Inertia computation for all 2D shapes
- **Mass properties** (`mass_properties`): sphere, box, cylinder, capsule, convex hull inertia computation with parallel axis theorem
- **Collision mesh generation** (`collision_mesh_gen`): marching cubes SDF-to-mesh with edge-collapse simplification
- **Scene I/O** (`scene_io`, std-only): binary format (APHYS magic, LE encoding) with raw Fix128 serialization + JSON export
- **Cloth-fluid coupling** (`cloth_fluid`): drag, buoyancy, boundary repulsion
- **Rope attachment** (`rope_attach`): rigid body rope attachment with compliance and break force
- **Soft body cutting** (`soft_body_cut`): plane-based cutting of deformable bodies and cloth
- **Visualization**: `heatmap` (stress/temperature with viridis colormap), `flow_viz` (fluid arrows and streamlines), `contact_viz` (contact force arrows and friction cones)
- **Multi-world** (`multi_world`): multiple independent physics worlds with body transfer
- **Particle system** (`particle`): general-purpose emitters, lifetime, force field integration
- 19 new source files, ~10,800 lines of new code
- 303 new tests (397 → 700 total)

## [0.4.0] - 2026-02-22

### Added

- Analytics & Privacy layer: `sketch`, `anomaly`, `privacy`, `pipeline` modules
- `sketch`: HyperLogLog, DDSketch, Count-Min Sketch, Heavy Hitters, FnvHasher
- `anomaly`: MAD, EWMA, Z-score composite detector with streaming median
- `privacy`: Laplace noise, RAPPOR, randomized response, privacy budget tracking
- `pipeline`: Lock-free ring buffer metric aggregation with registry
- `analytics_bridge` module for ALICE-Analytics integration (`--features analytics`)
- SDF simulation modifiers: `sim_field`, `sim_modifier`, `thermal`, `pressure`, `erosion`, `fracture`, `phase_change`
- `netcode` module: `DeterministicSimulation`, `FrameInput`, `SimulationChecksum`, `InputApplicator`
- `fluid_netcode` module: delta-compressed deterministic fluid snapshots
- `character` module: kinematic capsule-based character controller with stair stepping
- `interpolation` module: substep interpolation with NLERP quaternion blending
- `debug_render` module: wireframe visualization API (bodies, contacts, joints, BVH, forces)
- `profiling` module: per-stage timer and per-frame statistics
- `material` module: per-pair friction/restitution table with combine rules
- `dynamic_bvh` module: incremental AABB tree with AVL balancing
- `contact_cache` module: HashMap O(1) manifold lookup with warm starting
- `box_collider` module: OBB with GJK support
- `compound` module: multi-shape compound collider
- `cylinder` module: cylinder collider with GJK support
- `error` module: unified `PhysicsError` type
- Python batch APIs: `add_bodies_batch`, `set_velocities_batch`, `apply_impulses_batch`, `states()`
- Feature flags documentation table in crate-level docs
- 52 new integration tests (10 → 62 total)
- CI: integration tests, no_std build verification, parallel tests, SIMD build check

### Changed

- `PhysicsWorld::raycast()` now uses BVH broad-phase for O(log n) candidate pruning
- `raycast()` handles ray-inside-sphere case (far-side intersection fallback)
- Constraint graph coloring uses `u64` bitmask instead of `Vec<usize>` (O(1) free-color lookup)
- `dot_simd_sse2` cleaned up: removed dead SIMD register loads, simplified to scalar carry chain
- `python.rs`: 4 `unwrap()` replaced with descriptive `expect()` messages
- CI workflow: `cargo test` now runs integration + doc tests (was `--lib` only)
- Clippy: strict `-W clippy::all` enforced in CI
- MSRV: 1.70.0 (replaced `is_multiple_of`, `div_ceil`, `is_none_or` with compatible alternatives)

### Fixed

- MSRV compatibility: removed usage of `is_multiple_of` (1.87), `div_ceil` (1.73), `is_none_or` (1.82)

## [0.3.0] - 2026-02-22

### Added

- `spatial` module: shared `SpatialGrid` extracted from `fluid.rs` for reuse across modules
- `error` module: unified `PhysicsError` type with 4 variants (`InvalidBodyIndex`, `DeserializationFailed`, `InvalidConstraint`, `ZeroLengthVector`)
- `query::overlap_sphere_bvh()`: BVH-accelerated sphere overlap query (O(log n + k))
- `query::overlap_aabb_bvh()`: BVH-accelerated AABB overlap query (O(log n + k))
- `Vec3Fix::normalize_with_length()`: returns unit vector and original length in one call
- `Vec3Fix::try_normalize()`: returns `Option<Vec3Fix>` for zero-length safety
- SIMD-accelerated `dot_simd()`, `length_squared_simd()`, `cross_simd()` (x86_64 SSE2)
- `select_fix128()`, `select_vec3()`: branchless selection primitives
- `simd_width()` / `SIMD_WIDTH`: compile-time SIMD lane width detection
- Integration tests: vehicle ground contact, rope sag, determinism golden hash, BVH query, cloth grid, constraint stability, state serialization
- Benchmark suite (criterion): physics step, math ops, BVH query
- FFI struct field documentation for `AliceVec3`, `AliceQuat`, `AlicePhysicsConfig`, `AliceBodyInfo`
- Compile smoke tests for public re-exports and feature flag invariants

### Changed

- `SpatialGrid::query_neighbors_into` now takes `&self` instead of `&mut self`
- `cloth::EdgeConstraint` removed unused `inv_rest_length` field
- `fluid.rs` now delegates to shared `spatial::SpatialGrid` instead of inline implementation

### Fixed

- Dead code warning from `inv_rest_length` in cloth edge constraints
- Missing doc comments on FFI struct fields (fixes warnings with `--features ffi`)

## [0.2.0] - 2026-02-20

### Added

- Cloth simulation (`cloth` module): XPBD triangle mesh with distance + bending constraints, self-collision, SDF collision
- Rope simulation (`rope` module): XPBD distance chain with SDF collision
- Vehicle physics (`vehicle` module): wheel, suspension, engine, steering
- Fluid simulation (`fluid` module): Position-Based Fluids with spatial hash
- Deformable bodies (`deformable` module): FEM-XPBD tetrahedral mesh
- Animation blending (`animation_blend` module): ragdoll SLERP blending
- Audio physics (`audio_physics` module): physics-driven audio parameters
- SDF ecosystem: `sdf_manifold`, `sdf_ccd`, `sdf_force`, `sdf_destruction`, `sdf_adaptive`
- Convex decomposition from SDF voxel grids
- GPU compute shader interface for batch SDF evaluation
- Fluid netcode with delta compression
- Character controller
- Physics profiler

## [0.1.0] - 2026-02-15

### Added

- Core physics engine with deterministic 128-bit fixed-point arithmetic (I64F64)
- XPBD solver with substep integration
- GJK/EPA collision detection for convex shapes
- Linear BVH (Morton code) for broad-phase
- Joint system: Ball, Hinge, Fixed, Slider, Spring, ConeTwist, D6
- Collision filtering with layer/mask bitmasks
- Deterministic RNG (PCG-XSH-RR)
- Contact and trigger event tracking
- Ray and shape casting
- Continuous collision detection (TOI, conservative advancement)
- Sleep/wake and island management
- Triangle mesh collision with BVH acceleration
- Height field terrain collision
- PD controllers and joint motors
- Articulated bodies (Featherstone solver)
- Custom force fields (wind, gravity wells, buoyancy, vortex)
- State serialization for rollback netcode
- FFI bindings for Unity/UE5 (`--features ffi`)
- WASM bindings for browsers (`--features wasm`)
- Python bindings via PyO3 (`--features python`)
