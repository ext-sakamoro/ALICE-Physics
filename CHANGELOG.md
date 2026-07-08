# Changelog

All notable changes to ALICE-Physics will be documented in this file.

## [0.11.0] - 2026-07-08

### Added — Field-based GPU solver bridge with automatic contact-solve routing on `step` / `substep`

Introduces `PhysicsWorld::set_gpu_solver_bridge(Option<Box<dyn GpuSolverBridge + Send + Sync>>)` (with `take_gpu_solver_bridge()` / `gpu_solver_bridge_installed()` accessors) so callers can install a GPU solver bridge on the world once and have every subsequent `step` / `substep` transparently route contact-solve through the bridge. Complements the v0.10.0 explicit helper methods (`solve_contact_constraints_with_bridge` / `substep_with_bridge` / `step_with_bridge`), which continue to work unchanged with an externally-owned bridge — the helpers ignore the installed bridge and use the borrowed one passed in, giving callers a clean escape hatch for hot-swap harnesses, game-engine wrappers, or per-frame bridge choice.

Internally, `PhysicsWorld::solve_contact_constraints` gained a short-circuit at the top: if a bridge is installed, `Option::take()` moves it out of `self`, calls `self.solve_contact_constraints_with_bridge(bridge.as_mut())`, then reinstalls the bridge — the field is unchanged from the caller's perspective, and `step` / `substep` inherit the auto-routing for free without needing to know about the bridge.

- **`PhysicsWorld::set_gpu_solver_bridge(bridge)`** — install (`Some(_)`) or clear (`None`) the world's GPU solver bridge.
- **`PhysicsWorld::take_gpu_solver_bridge() -> Option<Box<dyn ...>>`** — remove and return the currently installed bridge. Subsequent `step` / `substep` calls revert to the CPU contact solver.
- **`PhysicsWorld::gpu_solver_bridge_installed() -> bool`** — read-only check of the installed state (`const fn`, no allocation).

The trait object is `Box<dyn GpuSolverBridge + Send + Sync>`. Implementers must satisfy `Send + Sync`, which is satisfied by ALICE-TRT v3.0.0+ (`TrtSolverAdapter` refactored to `Arc<GpuDevice>` alongside this release — see the ALICE-TRT v3.0.0 CHANGELOG entry for the coordinated migration).

Three tests added to `mod tests`: `physics_world_step_auto_routes_through_bridge_when_attached` (counter-based routing verification via `Arc<AtomicUsize>` inside a `RecordingBridge` no-op implementer, asserts `send_contact_constraints` / `send_body_state` / `dispatch_contact_solve_iteration` all fire `config.iterations` times per `substep`), `physics_world_step_falls_back_to_cpu_when_bridge_detached` (asserts CPU-driven position drift on a one-contact fixture when no bridge is installed), and `physics_world_bridge_attach_detach_lifecycle` (asserts install → `take` returns `Some` → subsequent `take` returns `None` and `gpu_solver_bridge_installed` mirrors the state).

Total lib tests: **721 → 724** (all pass, zero regression against v0.10.0 baseline).

### Fixed — `_with_bridge` methods now correctly gated behind `gpu-solver-bridge` feature

The v0.10.0 helper methods (`solve_contact_constraints_with_bridge` / `step_with_bridge` / `substep_with_bridge`) were declared with `#[cfg(feature = "std")]` but reference the `crate::gpu_bridge` module, which is itself behind `#[cfg(feature = "gpu-solver-bridge")]`. Default builds (`std` on, `gpu-solver-bridge` off) failed with `E0433: could not find gpu_bridge in the crate root` — the pre-existing v0.10.0 CI has been red on `Test (ubuntu-latest / macos-latest / windows-latest)`, `Clippy`, and `Doc` since 2026-07-07 for this reason. Retagging the three helper methods with `#[cfg(feature = "gpu-solver-bridge")]` restores the correct gate: they only compile when the module they depend on is available. Default `cargo build --lib`, `cargo build --lib --no-default-features`, and `cargo build --lib --features std,gpu-solver-bridge` all now build clean locally; CI verification follows this push.

## [0.10.0] - 2026-07-07

### Added — `PhysicsWorld` helper methods for bridge-routed contact solve

Adds three public methods on `PhysicsWorld` that route the PGS contact-solve stage through a caller-supplied `&mut dyn GpuSolverBridge` while keeping the surrounding pipeline (integrate + distance + joint + velocity update) on the CPU. Zero storage of the bridge inside `PhysicsWorld` — the bridge is passed by mutable reference per call, so no lifetime or `Send + Sync` bound is imposed on the world. This is the pragmatic realisation of the v0.9.0 `GpuSolverBridge` contact-solve trait extension for callers that want end-to-end GPU offload of the numeric hot path without the `PhysicsWorld` storing a boxed trait object.

- **`PhysicsWorld::solve_contact_constraints_with_bridge<B: GpuSolverBridge + ?Sized>(&mut self, bridge: &mut B)`** — run one PGS contact-solve iteration through the bridge. Applies Stage A (sensor filter, pre-solve hooks, contact modifiers) on the CPU upfront to produce a filtered constraint list; uploads that list plus per-body positions + inverse masses via `bridge.send_contact_constraints` and `bridge.send_body_state`; dispatches one iteration via `bridge.dispatch_contact_solve_iteration`; reads back the updated `cached_lambda` and positions via `bridge.recv_contact_constraints` and `bridge.recv_body_positions`; writes those results back into `self.contact_constraints[slot].cached_lambda` (via an index map from the filtered list back to the original slot) and `self.bodies[i].position`. Byte-exact CPU parity is preserved because Stage A closures are pure functions of their inputs (don't depend on live body state), so batched upfront application is equivalent to the CPU's per-constraint interleaved application.
- **`PhysicsWorld::substep_with_bridge<B: GpuSolverBridge + ?Sized>(&mut self, bridge: &mut B, dt: Fix128)`** — run one substep with the contact-solve stage routed through the bridge. Integrate + distance + joint + velocity-update stay on the CPU; only the inner PGS contact-solve iterations use the bridge. Semantically equivalent to the private `substep(dt)` with `solve_contact_constraints(dt)` replaced by `solve_contact_constraints_with_bridge(bridge)` inside the inner iteration loop.
- **`PhysicsWorld::step_with_bridge<B: GpuSolverBridge + ?Sized>(&mut self, bridge: &mut B, dt: Fix128)`** — run one full simulation step with bridge-routed contact solve. Mirrors `step(dt)` phase-for-phase: phase 0 event lifecycle + clear contacts, phase 0.5 island rebuild, phase 1 force fields, phase 2 collision detection, phase 3 substep loop (via `substep_with_bridge`), phase 4 sleep update, phase 5 event end. Enables the shortest opt-in path — attach a bridge, call `step_with_bridge` once per frame, get GPU-accelerated contact solve with byte-exact CPU parity.

### Design rationale — why not `Option<Box<dyn ...>>` field on `PhysicsWorld`?

Storing the bridge as an `Option<Box<dyn GpuSolverBridge + Send + Sync>>` field on `PhysicsWorld` was considered and deferred. Two complications:

1. **Lifetime propagation**: ALICE-TRT's `TrtSolverAdapter<'a>` borrows `&'a GpuDevice`. Storing it in `Box<dyn ... + 'static>` requires either dropping the lifetime (breaking change to `TrtSolverAdapter::new` — bump `alice-trt` to a MAJOR version) or propagating the lifetime to `PhysicsWorld<'a>` (breaking change to every downstream consumer that constructs a world).
2. **`Send + Sync` bound analysis**: wgpu resources are `Send + Sync` from wgpu 0.7+, so the trait object bound is satisfiable. But verifying the bound across the Metal / Vulkan / DX12 CI matrix requires additional test infrastructure.

The helper-method approach used in this release accepts the bridge by `&mut B` per call, so both complications vanish — the borrow lives only for the duration of the method, and no bound is imposed. The trade-off is that callers must invoke `step_with_bridge` explicitly instead of the world auto-routing through the bridge on `step`. This is aligned with the ALICE-* API design principle of "explicit control over implicit magic" and matches how the parallel-vs-sequential distance-constraint dispatch is exposed on `TrtSolverAdapter` via a `set_parallel_dispatch(bool)` toggle rather than auto-detection.

A future MAJOR release (e.g., `alice-trt` v3.0.0 with `TrtSolverAdapter` refactored to own `Arc<GpuDevice>`) can revisit the field-based auto-routing.

### Backwards compatibility

100% source-compatible with v0.9.x. The existing `step(dt)` and every existing `PhysicsWorld` API is unchanged. The new methods are additive and gated on `#[cfg(feature = "std")]` (they use the pre-solve hook and contact modifier collections which are `std`-only). Callers not using the bridge helpers see no difference from v0.9.x.

## [0.9.0] - 2026-07-07

### Added — `GpuSolverBridge` contact solve pipeline extension (v0.9 opt-in)

Extends the `GpuSolverBridge` trait in `src/gpu_bridge.rs` with five new methods covering the Phase 3 contact solve pipeline stage: `send_contact_constraints`, `send_body_state`, `dispatch_contact_solve_iteration`, `recv_contact_constraints`, and `recv_body_positions`. Every new method ships with a `panic!("...not implemented by this GpuSolverBridge backend")` default implementation, so pre-v0.9 backends compile unchanged and continue to work for the integrate + distance pipeline they already implement; they only panic if a caller tries to route contact solve through them, which is fail-fast behaviour that surfaces the missing capability immediately (silent no-op defaults were considered and rejected per the ALICE-* silent-Ok(()) prohibition rule).

The trait extension unblocks `ALICE-TRT::TrtSolverAdapter` v2.7.0, which implements all five new methods on top of its existing v2.6.0 `dispatch_fix128_pgs_contact_solve` standalone kernel and exposes byte-exact GPU PGS contact solve through the trait-object interface. Callers can drive a full v2.2 → v2.6 broad-phase → narrow-phase → solve pipeline via a single `&mut dyn GpuSolverBridge` handle.

Deeper wire-through — where `PhysicsWorld::solve_contact_constraints` automatically routes through an attached `GpuSolverBridge` implementation — is scheduled for a follow-up release (requires an `Option<Box<dyn GpuSolverBridge + Send + Sync>>` field on `PhysicsWorld` plus the lifetime + `Send + Sync` bounds analysis for the trait-object storage). The v0.9.0 extension provides the surface; the wire-through is additive on top.

- **`send_contact_constraints(&mut self, constraints: &[ContactConstraint])`** — upload the constraint list. Element indices must line up with the caller's `PhysicsWorld::contact_constraints` slot ordering so `recv_contact_constraints` can write updated `cached_lambda` warm-start values back in place.
- **`send_body_state(&mut self, positions: &[[Fix128; 3]], inv_masses: &[Fix128])`** — upload the per-body state that contact solve reads. `positions[i]` is the position of body id `i`; `inv_masses[i] == Fix128::ZERO` marks body `i` as static.
- **`dispatch_contact_solve_iteration(&mut self, warm_start_factor: Fix128)`** — run one sequential Gauss-Seidel PGS iteration against the uploaded state. Updates cached_lambda in place and applies position corrections. Callers loop for multiple iterations, mirroring the CPU `for _ in 0..config.iterations { ... }` pattern.
- **`recv_contact_constraints(&self, constraints: &mut [ContactConstraint])`** — read back the post-solve `cached_lambda`; other constraint fields (body indices, contact geometry, friction, restitution) are left untouched.
- **`recv_body_positions(&self, positions: &mut [[Fix128; 3]])`** — read back the post-solve body positions.

Also updates the trait's doc-comment to describe both pipeline stages (integrate + distance, contact solve) and replaces the "Skeleton" section that predates the v0.7.x rollout.

### Backwards compatibility

100% source-compatible with v0.8.x. Existing `GpuSolverBridge` implementations (including the `StubBridge` test type and any external backend) compile unchanged. The new methods only affect callers that opt in to the contact solve pipeline stage.

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
- **Feature Combinations**: documented tested feature matrix in README (11 combinations + compatibility matrix)
- 10 new integration tests (stress test, damping, builder, Display, From/Into, CCD, errors, HeightField)

### Changed

- **PyO3 upgrade**: 0.22 → 0.27 (Python 3.14 support), numpy 0.22 → 0.27
- `python.rs`: `into_pyarray_bound()` → `into_pyarray()`, `py.allow_threads()` → `py.detach()`
- **Release workflow**: added static library builds (.a/.lib) and Python wheel (maturin) builds
- **README_JP.md**: full update from v0.4.0 to v0.6.0

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
