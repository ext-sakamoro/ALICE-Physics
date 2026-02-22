# Changelog

All notable changes to ALICE-Physics will be documented in this file.

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
