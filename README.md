# ALICE-Physics

**Deterministic 128-bit Fixed-Point Physics Engine** - v0.4.0

English | [日本語](README_JP.md)

A high-precision physics engine designed for deterministic simulation across different platforms and hardware. Uses 128-bit fixed-point arithmetic to ensure bit-exact results regardless of CPU, compiler, or operating system.

## Features

| Feature | Description |
|---------|-------------|
| **128-bit Fixed-Point** | I64F64 format (64-bit integer + 64-bit fraction) for extreme precision |
| **CORDIC Trigonometry** | Deterministic sin/cos/atan without FPU instructions |
| **XPBD Solver** | Extended Position Based Dynamics for stable constraint solving |
| **GJK/EPA Collision** | Robust collision detection for convex shapes |
| **Stackless BVH** | Morton code-based spatial acceleration with escape pointers |
| **Constraint Batching** | Graph-colored parallel constraint solving |
| **Rollback Support** | Complete state serialization for netcode |
| **Neural Controller** | Deterministic AI via ALICE-ML ternary weights + Fix128 inference |
| **7 Joint Types** | Ball, Hinge, Fixed, Slider, Spring, D6, Cone-Twist with angle limits, motors, and breakable constraints |
| **Raycasting** | Ray and shape casting against spheres, AABBs, capsules, planes |
| **Shape Cast / Overlap** | Sphere cast, capsule cast, overlap sphere, overlap AABB queries |
| **CCD** | Continuous collision detection (TOI, conservative advancement) |
| **Sleep/Islands** | Automatic sleep with Union-Find island management |
| **Triangle Mesh** | BVH-accelerated triangle mesh collision (Moller-Trumbore) |
| **Height Field** | Grid terrain with bilinear interpolation |
| **Articulated Bodies** | Multi-joint chains, ragdolls, robotic arms with FK propagation |
| **Force Fields** | Wind, gravity wells, drag, buoyancy, vortex |
| **PD Controllers** | 1D/3D proportional-derivative joint motors |
| **Collision Filtering** | Layer/mask bitmask system with collision groups |
| **Trigger/Sensor** | Sensor bodies that detect overlap without physics response |
| **Character Controller** | Kinematic capsule-based move-and-slide with stair stepping and SDF terrain |
| **Rope** | XPBD distance chain rope and cable simulation |
| **Cloth** | XPBD triangle mesh cloth with self-collision (spatial hash grid) |
| **Fluid** | Position-Based Fluids (PBF) with spatial hash grid |
| **Deformable** | FEM-XPBD deformable body (tetrahedral mesh) |
| **Vehicle** | Wheel, suspension, engine, steering, gear shifting |
| **Animation Blend** | Ragdoll-to-animation blending with SLERP |
| **Audio Physics** | Physics-based audio parameter generation (impact, friction, rolling) |
| **SDF Manifold** | Multi-point contact manifold from SDF surfaces |
| **SDF CCD** | Sphere tracing continuous collision detection for SDF |
| **SDF Force Fields** | SDF-driven force fields (attract, repel, contain, flow) |
| **SDF Destruction** | Real-time CSG boolean destruction |
| **SDF Adaptive** | Adaptive SDF evaluation with distance-based LOD |
| **Convex Decompose** | Convex decomposition from SDF voxel grid |
| **GPU SDF** | GPU compute shader interface for batch SDF evaluation |
| **Fluid Netcode** | Deterministic fluid netcode with delta compression |
| **Simulation Fields** | 3D scalar/vector fields with trilinear interpolation and diffusion |
| **Thermal** | Heat diffusion, melt, thermal expansion, freeze |
| **Pressure** | Contact force accumulation, crush, bulge, dent deformation |
| **Erosion** | Wind, water, chemical corrosion, ablation |
| **Fracture** | Stress-driven crack propagation with CSG subtraction |
| **Phase Change** | Solid/liquid/gas transitions driven by temperature |
| **Deterministic RNG** | PCG-XSH-RR pseudo-random number generator |
| **Contact Events** | Begin/Persist/End contact and trigger event tracking |
| **Box/OBB Collider** | Oriented Bounding Box with GJK support and fast AABB |
| **Compound Shape** | Multi-shape compound collider with local transforms |
| **Contact Cache** | Persistent manifold cache with HashMap O(1) lookup and warm starting |
| **Dynamic AABB Tree** | Incremental BVH with O(log n) insert/remove/update, AVL balancing |
| **D6 Joint** | 6-DOF configurable joint with per-axis lock/free/limit |
| **Cone-Twist Joint** | Ball joint with cone swing limit and twist limit |
| **Material Table** | Per-pair friction/restitution with combine rules (Avg, Min, Max, Mul) |
| **Scaled Shape** | Uniform scale wrapper for any Support-implementing shape |
| **Speculative CCD** | Speculative contacts for fast-moving bodies without time rewinding |
| **Featherstone** | O(n) forward dynamics for articulated bodies |
| **Debug Render** | Wireframe visualization API (bodies, contacts, joints, BVH, forces) |
| **Profiling** | Per-stage timer and per-frame statistics API |
| **Substep Interpolation** | WorldSnapshot with NLERP quaternion blending for smooth rendering |
| **Probabilistic Sketches** | HyperLogLog, DDSketch, Count-Min Sketch, Heavy Hitters |
| **Streaming Anomaly Detection** | MAD, EWMA, Z-score composite detector |
| **Local Differential Privacy** | Laplace noise, RAPPOR, randomized response |
| **Metric Pipeline** | Lock-free ring buffer metric aggregation with registry |
| **no_std Compatible** | Works on embedded systems and WebAssembly |

## Optimizations ("黒焦げ" Edition) — 100/100

ALICE-Physics achieves a **perfect 100/100 optimization score** across 6 layers:

### Optimization Scorecard

| Layer | Score | Key Techniques |
|-------|-------|----------------|
| **L1: Memory Layout** | 15/15 | `#[repr(C, align(64))]` on RigidBody/Cloth/Fluid, hot/cold field reorder |
| **L2: Execution** | 20/20 | `#[inline(always)]` on GJK/BVH/constraints, branchless select, precomputed grid_half |
| **L3: Compute** | 20/20 | Warm-start `cached_lambda`, reciprocal precomputation (`inv_rest_length`, `inv_rest_density`) |
| **L4: GPU & Throughput** | 15/15 | `SIMD_WIDTH` const + `simd_width()`, `GpuSdfInstancedBatch`/`GpuSdfMultiDispatch`, `batch_size()` |
| **L5: Build Profile** | 10/10 | `opt-level=3`, `lto="fat"`, `codegen-units=1`, `panic="abort"`, `strip=true` |
| **L6: Code Quality** | 20/20 | 432 tests (358 unit + 62 integration + 12 doc), clippy 0 warnings |
| **Total** | **100/100** | |

### L1: Memory Layout (15/15)

All hot data structures use 64-byte cache-line alignment with hot/cold field separation:

```rust
#[repr(C, align(64))]
pub struct RigidBody {
    // HOT fields (accessed every substep) — first cache line
    pub position: Vec3Fix,
    pub velocity: Vec3Fix,
    pub inv_mass: Fix128,
    pub inv_inertia: Vec3Fix,
    pub prev_position: Vec3Fix,
    // COLD fields (accessed occasionally)
    pub rotation: QuatFix,
    pub angular_velocity: Vec3Fix,
    pub restitution: Fix128,
    pub friction: Fix128,
    // ...
}
```

- **RigidBody** (`solver.rs`): `#[repr(C, align(64))]` + hot/cold field reorder
- **Cloth** (`cloth.rs`): `#[repr(C, align(64))]` + precomputed `inv_rest_length` per edge constraint
- **Fluid** (`fluid.rs`): `#[repr(C, align(64))]` + precomputed `inv_rest_density` and `grid_half`

### L2: Execution Model (20/20)

Aggressive inlining and branchless execution across all hot paths:

- **GJK/BVH**: All support functions and traversal marked `#[inline(always)]` (35 annotations total)
- **Constraint solving**: `#[inline(always)]` on distance/contact constraint kernels
- **Fluid spatial grid**: `grid_half` precomputed at construction, eliminates repeated division in `hash()` hot path
- **Branchless primitives** (`math.rs`): `select_fix128()`, `select_vec3()` — CMOV-equivalent using bitwise masks, zero pipeline flushes

### L3: Compute Strategy (20/20)

Division elimination and warm-starting for faster convergence:

- **Distance constraint warm-start**: `cached_lambda` field stores the Lagrange multiplier from the previous substep, biasing the initial guess for dramatically faster convergence
- **Cloth reciprocal precomputation**: `inv_rest_length` computed once at construction, constraint solving uses multiplication instead of division
- **Fluid reciprocal precomputation**: `inv_rest_density` eliminates 2+ divisions per particle per iteration in incompressibility constraint
- **Contact manifold warm-start**: Per-contact `lambda_n`, `lambda_t1`, `lambda_t2` accumulated across frames via `apply_warm_start()`
- **Contact modifier parallel pre-pass**: `pre_process_contacts()` runs sequentially before Rayon parallel dispatch, applying pre-solve hooks and contact modifiers safely without data races

### L4: GPU & Throughput (15/15)

SIMD-width-aware batching for optimal GPU and CPU utilization:

```rust
// Compile-time CPU feature detection
pub const SIMD_WIDTH: usize = simd_width();
// AVX2=8, SSE=4, NEON=4, scalar=1

#[inline(always)]
pub fn batch_size() -> usize { crate::math::SIMD_WIDTH }
```

- **`SIMD_WIDTH` constant** (`math.rs`): Compile-time CPU feature detection (AVX2=8, SSE=4, NEON=4, scalar=1)
- **`GpuSdfInstancedBatch`** (`gpu_sdf.rs`): Groups SDF queries by ID to minimize GPU kernel launches
- **`GpuSdfMultiDispatch`** (`gpu_sdf.rs`): Manages multiple GPU batches with `total_queries()` and `total_dispatches()` statistics
- **`batch_size()`** (`gpu_sdf.rs`): Returns `SIMD_WIDTH` for register-width-aligned data streams

### L5: Build Profile (10/10)

```toml
[profile.release]
opt-level = 3          # Maximum optimization
lto = "fat"            # Link-time optimization (all crates unified)
codegen-units = 1      # Single code generation unit (maximum optimization)
panic = "abort"        # No unwinding tables
strip = true           # Strip symbols
```

### L6: Code Quality (20/20)

- **358 unit tests** across 63 modules
- **62 integration tests** (end-to-end physics scenarios)
- **12 doc tests** with runnable examples (sketch, anomaly, privacy, pipeline, solver)
- **Total: 432 passing tests**, clippy: 0 warnings

---

### Additional Optimizations

#### Stackless BVH Traversal

Traditional BVH traversal uses a stack to track nodes to visit. Our implementation uses **escape pointers** embedded in each node:

```
┌──────────────────────────────────────────────────────┐
│  BvhNode Layout (32 bytes, cache-aligned)            │
├──────────────────────────────────────────────────────┤
│  aabb_min[3]        (12 bytes)  - Bounding box min   │
│  first_child/prim   (4 bytes)   - Child or prim idx  │
│  aabb_max[3]        (12 bytes)  - Bounding box max   │
│  prim_count_escape  (4 bytes)   - [count:8|escape:24]│
└──────────────────────────────────────────────────────┘

Traversal: single index variable, no stack allocation
  if (hit)  → descend to first_child
  if (miss) → jump to escape_idx (skip entire subtree)
```

**Benefits:**
- Zero heap allocation during queries
- Single register for traversal state
- Better branch prediction
- i32 AABB comparison (no Fix128 reconstruction)

#### SIMD Acceleration (optional)

Enable with `--features simd`:

```rust
// x86_64 with SSE2
impl Fix128 {
    pub unsafe fn add_simd(self, rhs: Self) -> Self;
    pub unsafe fn sub_simd(self, rhs: Self) -> Self;
}

impl Vec3Fix {
    pub fn dot_simd(self, rhs: Self) -> Fix128;
    pub fn cross_simd(self, rhs: Self) -> Self;
    pub fn dot_batch_4(a: [Self; 4], b: [Self; 4]) -> [Fix128; 4];
}
```

#### Constraint Batching (optional)

Enable with `--features parallel`:

Constraints are grouped by **graph coloring** - constraints with no shared bodies are placed in the same "color" and can be solved independently:

```rust
// Rebuild constraint batches (greedy graph coloring)
world.rebuild_batches();

// Step with batched constraint solving
world.step_parallel(dt);

// Check number of color batches
println!("Batches: {}", world.num_batches());
```

**Benefits:**
- Zero heap allocation in constraint loop (index-based iteration)
- Rayon-ready parallel solving
- Reduced lock contention
- Better cache utilization

#### HashMap Contact Cache

Contact manifold lookup via `BodyPairKey → usize` HashMap for O(1) access:

```rust
// O(1) manifold lookup (was O(n) linear scan)
pub fn find(&self, a: usize, b: usize) -> Option<&ContactManifold> {
    let key = BodyPairKey::new(a, b);
    self.pair_index.get(&key).map(|&i| &self.manifolds[i])
}
```

Falls back to linear scan for `no_std` environments.

#### Rayon Parallel Integration

Enable with `--features parallel`:

Position integration and velocity updates run in parallel via `par_iter_mut()`:

```rust
// Parallel position integration (gravity + damping + Euler)
bodies.par_iter_mut().for_each(|body| {
    body.velocity = body.velocity + gravity * dt;
    body.velocity = body.velocity * damping;
    body.position = body.position + body.velocity * dt;
});
```

#### Python Batch APIs

Zero-copy NumPy batch operations with GIL release:

```python
# Batch body creation from (N,4) array [x, y, z, mass]
world.add_bodies_batch(np.array([[0,10,0,1.0], [5,10,0,2.0]]))

# Batch velocity update with GIL release
world.set_velocities_batch(velocities_array)  # (N,3)

# Combined state output as (N,10) array [px,py,pz,vx,vy,vz,qx,qy,qz,qw]
states = world.states()
```

## Why Deterministic Physics?

Traditional physics engines using IEEE 754 floating-point can produce different results across:
- Different CPU architectures (x86 vs ARM)
- Different compilers (GCC vs Clang vs MSVC)
- Different optimization levels (-O0 vs -O3)
- Different instruction sets (SSE vs AVX)

ALICE-Physics guarantees **bit-exact results** everywhere, enabling:

- **Lockstep Multiplayer**: All clients compute identical simulation
- **Rollback Netcode**: Replay inputs deterministically
- **Replay Systems**: Perfect reproduction of game sessions
- **Distributed Simulation**: Parallel computation with consistent results

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ALICE-Physics v0.4.0                                │
│              63 modules, 358 unit tests, 62 integration, 12 doc tests         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Core Layer                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  math    │ │ collider │ │  solver  │ │   bvh    │ │sdf_colldr│          │
│  │ Fix128   │ │ AABB     │ │ RigidBody│ │ Morton   │ │ SdfField │          │
│  │ Vec3Fix  │ │ Sphere   │ │ XPBD     │ │ Stackless│ │ Gradient │          │
│  │ QuatFix  │ │ Capsule  │ │ Sensor   │ │ Zero-    │ │ Early-out│          │
│  │ CORDIC   │ │ GJK/EPA  │ │ Rollback │ │  alloc   │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                              │
│  AAA Engine Layer (v0.4.0)                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │box_colldr│ │ compound │ │cont_cache│ │dynamic_bv│ │ material │          │
│  │ OBB      │ │ Multi-   │ │ HashMap  │ │ Incr BVH │ │ Pair Tbl │          │
│  │ GJK Supp │ │ Shape    │ │ O(1) Get │ │ AVL Bal  │ │ Combine  │          │
│  │ Inertia  │ │ Local Tx │ │ Warm Str │ │ O(log n) │ │ Friction │          │
│  │ Corners  │ │ AABB Mrg │ │ 4-point  │ │ Fat AABB │ │ Restit   │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                    │
│  │dbg_rendr │ │profiling │ │ interp   │                                    │
│  │ Wireframe│ │ Timers   │ │ Snapshot │                                    │
│  │ Contacts │ │ Per-Stage│ │ NLERP    │                                    │
│  │ Joints   │ │ Stats    │ │ Blend    │                                    │
│  │ BVH/AABB │ │ History  │ │ Alpha    │                                    │
│  └──────────┘ └──────────┘ └──────────┘                                    │
│                                                                              │
│  Constraint & Dynamics Layer                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  joint   │ │  motor   │ │articulatn│ │  force   │ │ sleeping │          │
│  │ Ball     │ │ PD 1D/3D │ │ Ragdoll  │ │ Wind     │ │ Islands  │          │
│  │ Hinge    │ │ Position │ │ FK Chain │ │ Gravity  │ │ Union-   │          │
│  │ Fixed    │ │ Velocity │ │ Robotic  │ │ Buoyancy │ │  Find    │          │
│  │ Slider   │ │ Max Torq │ │ Feather- │ │ Drag     │ │ Auto     │          │
│  │ Spring   │ │          │ │  stone   │ │ Vortex   │ │  Sleep   │          │
│  │ D6       │ │          │ │ 12-body  │ │          │ │          │          │
│  │ ConeTwst │ │          │ │          │ │          │ │          │          │
│  │ Breakable│ │          │ │          │ │          │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                              │
│  Query & Collision Layer                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ raycast  │ │   ccd    │ │ trimesh  │ │heightfld │ │  filter  │          │
│  │ Sphere   │ │ TOI      │ │ Triangle │ │ Bilinear │ │ Layer    │          │
│  │ AABB     │ │ Conserv. │ │ BVH-accel│ │ Normal   │ │ Mask     │          │
│  │ Capsule  │ │ Advance  │ │ Moller-  │ │ Sphere   │ │ Group    │          │
│  │ Plane    │ │ Swept    │ │ Trumbore │ │ Collide  │ │ Bidirect │          │
│  │ Sweep    │ │ Specultv │ │ Closest  │ │ Signed   │ │          │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                                                  │
│  │  query   │ │character │                                                  │
│  │ SphCast  │ │ Move&Sld │                                                  │
│  │ CapCast  │ │ Stair    │                                                  │
│  │ Overlap  │ │ Ground   │                                                  │
│  │ AABB Ovr │ │ SDF Terr │                                                  │
│  └──────────┘ └──────────┘                                                  │
│                                                                              │
│  Soft Body & Simulation Layer                                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  rope    │ │  cloth   │ │  fluid   │ │deformable│ │ vehicle  │          │
│  │ XPBD     │ │ XPBD     │ │ PBF      │ │ FEM-XPBD │ │ Wheel    │          │
│  │ Distance │ │ Triangle │ │ SPH Hash │ │ Tetrahedr│ │ Suspensn │          │
│  │ Chain    │ │ Self-Col │ │ Density  │ │ Volume   │ │ Engine   │          │
│  │ Cable    │ │ SpatHash │ │ Viscosty │ │ Neo-Hook │ │ Steering │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                              │
│  SDF Advanced Layer                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │sdf_mnfld │ │ sdf_ccd  │ │sdf_force │ │sdf_destr │ │sdf_adapt │          │
│  │ Manifold │ │ SphTrace │ │ Attract  │ │ CSG Bool │ │ LOD      │          │
│  │ N-point  │ │ March    │ │ Repel    │ │ Subtract │ │ Distance │          │
│  │ Contact  │ │ TOI      │ │ Contain  │ │ Real-time│ │ Adaptive │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                                                  │
│  │cvx_decomp│ │ gpu_sdf  │                                                  │
│  │ Voxel    │ │ Compute  │                                                  │
│  │ Flood    │ │ Batch    │                                                  │
│  │ Convex   │ │ Shader   │                                                  │
│  └──────────┘ └──────────┘                                                  │
│                                                                              │
│  SDF Simulation Modifiers Layer                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │sim_field │ │sim_modif │ │ thermal  │ │ pressure │ │ erosion  │          │
│  │ Scalar3D │ │ Modifier │ │ Heat Eq  │ │ Crush    │ │ Wind     │          │
│  │ Vector3D │ │ Chain    │ │ Melt     │ │ Bulge    │ │ Water    │          │
│  │ Trilin   │ │ Modified │ │ Freeze   │ │ Dent     │ │ Chemical │          │
│  │ Diffuse  │ │ SDF      │ │ Expand   │ │ Yield    │ │ Ablation │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│  ┌──────────┐ ┌──────────┐                                                  │
│  │ fracture │ │phase_chg │                                                  │
│  │ Stress   │ │ Solid    │                                                  │
│  │ Crack    │ │ Liquid   │                                                  │
│  │ CSG Sub  │ │ Gas      │                                                  │
│  │ Voronoi  │ │ Latent H │                                                  │
│  └──────────┘ └──────────┘                                                  │
│                                                                              │
│  Game Systems Layer                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                                    │
│  │anim_blnd │ │audio_phys│ │ netcode  │                                    │
│  │ SLERP    │ │ Impact   │ │ FrameInp │                                    │
│  │ Ragdoll  │ │ Friction │ │ Checksum │                                    │
│  │ Blend    │ │ Rolling  │ │ Snapshot │                                    │
│  │ IK Mix   │ │ Material │ │ Rollback │                                    │
│  └──────────┘ └──────────┘ └──────────┘                                    │
│                                                                              │
│  Analytics & Privacy Layer (v0.4.0)                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                       │
│  │  sketch  │ │ anomaly  │ │ privacy  │ │ pipeline │                       │
│  │ HyperLog │ │ MAD      │ │ Laplace  │ │ MetricPi │                       │
│  │ DDSketch │ │ EWMA     │ │ RAPPOR   │ │ RingBuf  │                       │
│  │ CountMin │ │ Z-score  │ │ RandResp │ │ Registry │                       │
│  │ HeavyHit │ │ Composit │ │ XorShift │ │ Snapshot │                       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                       │
│                                                                              │
│  Utility Layer                                                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────────────────────────┐│
│  │   rng    │ │  event   │ │fluid_net │ │      neural (ALICE-ML × Phys)   ││
│  │ PCG-XSH  │ │ Begin    │ │ Delta    │ │ Ternary {-1,0,+1} → Fix128     ││
│  │ Fix128   │ │ Persist  │ │ Compress │ │ Deterministic AI                ││
│  │ Direction│ │ End      │ │ Snapshot │ │ Ragdoll Controller              ││
│  └──────────┘ └──────────┘ └──────────┘ └─────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Simulation

```rust
use alice_physics::prelude::*;

fn main() {
    // Create physics world with default config
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // Add a dynamic body (falling sphere)
    let sphere = RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 100, 0),  // position: (0, 100, 0)
        Fix128::ONE,                    // mass: 1.0
    );
    let sphere_id = world.add_body(sphere);

    // Add static ground
    let ground = RigidBody::new_static(Vec3Fix::ZERO);
    world.add_body(ground);

    // Simulate at 60 FPS
    let dt = Fix128::from_ratio(1, 60);  // 1/60 second

    for frame in 0..300 {  // 5 seconds
        world.step(dt);

        let pos = world.bodies[sphere_id].position;
        println!("Frame {}: y = {}", frame, pos.y.hi);
    }
}
```

### Distance Constraint (Rope/Chain)

```rust
use alice_physics::prelude::*;

fn create_rope(world: &mut PhysicsWorld, segments: usize) {
    let mut prev_id = None;

    for i in 0..segments {
        let body = RigidBody::new_dynamic(
            Vec3Fix::from_int(i as i64 * 2, 50, 0),
            Fix128::ONE,
        );
        let id = world.add_body(body);

        if let Some(prev) = prev_id {
            let constraint = DistanceConstraint {
                body_a: prev,
                body_b: id,
                local_anchor_a: Vec3Fix::ZERO,
                local_anchor_b: Vec3Fix::ZERO,
                target_distance: Fix128::from_int(2),
                compliance: Fix128::from_ratio(1, 1000),  // Soft constraint
            };
            world.add_distance_constraint(constraint);
        }

        prev_id = Some(id);
    }

    // Fix first segment in place
    world.bodies[0].inv_mass = Fix128::ZERO;
}
```

### Rollback Netcode

```rust
use alice_physics::prelude::*;

struct GameState {
    physics: PhysicsWorld,
    frame: u64,
    input_buffer: Vec<PlayerInput>,
}

impl GameState {
    fn save_snapshot(&self) -> Vec<u8> {
        self.physics.serialize_state()
    }

    fn load_snapshot(&mut self, data: &[u8]) {
        self.physics.deserialize_state(data);
    }

    fn rollback_and_resimulate(&mut self, to_frame: u64, new_input: PlayerInput) {
        // Load snapshot from frame
        let snapshot = self.get_snapshot(to_frame);
        self.load_snapshot(&snapshot);

        // Replay with corrected input
        self.input_buffer[to_frame as usize] = new_input;

        for frame in to_frame..self.frame {
            let input = &self.input_buffer[frame as usize];
            self.apply_input(input);
            self.physics.step(Fix128::from_ratio(1, 60));
        }
    }
}
```

### BVH Broad-Phase Collision

```rust
use alice_physics::bvh::{LinearBvh, BvhPrimitive};

// Build BVH from primitives
let primitives: Vec<BvhPrimitive> = bodies.iter().enumerate()
    .map(|(i, body)| BvhPrimitive {
        aabb: body.compute_aabb(),
        index: i as u32,
        morton: 0,  // Computed during build
    })
    .collect();

let bvh = LinearBvh::build(primitives);

// Query with zero heap allocation (callback version)
bvh.query_callback(&query_aabb, |prim_idx| {
    // Handle potential collision with primitive
});

// Or collect results
let hits = bvh.query(&query_aabb);

// Get BVH statistics
let stats = bvh.stats();
println!("Nodes: {}, Leaves: {}", stats.node_count, stats.leaf_count);
```

## Modules

### `math` - Fixed-Point Primitives

| Type | Description |
|------|-------------|
| `Fix128` | 128-bit fixed-point number (I64F64) |
| `Vec3Fix` | 3D vector with Fix128 components |
| `QuatFix` | Quaternion for rotations |
| `Mat3Fix` | 3x3 matrix for inertia tensors |

**Constants:**
- `Fix128::ZERO`, `Fix128::ONE`, `Fix128::NEG_ONE`
- `Fix128::PI`, `Fix128::HALF_PI`, `Fix128::TWO_PI`

**CORDIC functions (deterministic, no FPU):**
- `Fix128::sin()`, `Fix128::cos()`, `Fix128::sin_cos()`
- `Fix128::atan()`, `Fix128::atan2()`
- `Fix128::sqrt()` (Newton-Raphson, 64 iterations)

**Utility:**
- `Fix128::from_ratio(num, denom)` - Create from fraction
- `Fix128::half()`, `Fix128::double()` - Exact bit shifts
- `Fix128::abs()`, `Fix128::floor()`, `Fix128::ceil()`

### `collider` - Collision Detection

| Shape | Description |
|-------|-------------|
| `AABB` | Axis-Aligned Bounding Box |
| `Sphere` | Sphere collider |
| `Capsule` | Capsule (cylinder + hemispheres) |
| `ConvexHull` | Arbitrary convex polyhedron |
| `CollisionResult` | Contact information |

**Algorithms:**
- **GJK**: Gilbert-Johnson-Keerthi for intersection test (64 iterations max)
- **EPA**: Expanding Polytope Algorithm for penetration depth (64 iterations max)

### `solver` - XPBD Physics

**RigidBody fields:**
| Field | Type | Description |
|-------|------|-------------|
| `position` | `Vec3Fix` | Center of mass position |
| `rotation` | `QuatFix` | Orientation quaternion |
| `velocity` | `Vec3Fix` | Linear velocity |
| `angular_velocity` | `Vec3Fix` | Angular velocity |
| `inv_mass` | `Fix128` | Inverse mass (0 = static) |
| `inv_inertia` | `Vec3Fix` | Inverse inertia tensor (diagonal) |
| `restitution` | `Fix128` | Bounciness (0-1) |
| `friction` | `Fix128` | Friction coefficient |
| `is_sensor` | `bool` | Sensor mode: detects overlap but no physics response |

**Constraints:**
- `DistanceConstraint`: Fixed distance between anchor points
- `ContactConstraint`: Collision response with friction/restitution

**Methods:**
- `RigidBody::new(position, mass)` - Create dynamic body
- `RigidBody::new_dynamic(position, mass)` - Alias for new
- `RigidBody::new_static(position)` - Create immovable body
- `RigidBody::new_sensor(position)` - Create sensor/trigger body (no physics response)

### `bvh` - Spatial Acceleration

| Type | Description |
|------|-------------|
| `LinearBvh` | Flat array BVH with stackless traversal |
| `BvhNode` | 32-byte cache-aligned node with escape pointer |
| `BvhPrimitive` | Primitive entry for construction |
| `BvhStats` | Tree statistics |

**Features:**
- **Morton Codes**: Z-order curve for spatial locality
- **Escape Pointers**: Stackless traversal (zero allocation)
- **i32 AABB**: Fast integer comparison without Fix128 reconstruction

### `joint` - 7 Joint Types + Breakable Constraints

| Type | Description |
|------|-------------|
| `BallJoint` | Spherical joint (3 rotational DOF) |
| `HingeJoint` | Revolute joint (1 rotational DOF) with angle limits |
| `FixedJoint` | Weld joint (0 DOF) |
| `SliderJoint` | Prismatic joint (1 translational DOF) with limits |
| `SpringJoint` | Damped spring constraint |
| `D6Joint` | 6-DOF configurable joint with per-axis lock/free/limit |
| `ConeTwistJoint` | Ball joint with cone swing limit and twist limit |

All joint types support **breakable constraints** via `with_break_force(max_force)`. When the constraint force exceeds the threshold, the joint breaks and is removed from simulation.

```rust
use alice_physics::joint::*;
// Hinge with angle limits
let hinge = HingeJoint::new(body_a, body_b, anchor_a, anchor_b, axis_a, axis_b)
    .with_limits(-Fix128::HALF_PI, Fix128::HALF_PI);

// D6 joint (lock X translation, free Y rotation, limit Z rotation)
let d6 = D6Joint::new(body_a, body_b, anchor_a, anchor_b)
    .with_axis(Axis::LinearX, D6Mode::Locked)
    .with_axis(Axis::AngularY, D6Mode::Free)
    .with_axis(Axis::AngularZ, D6Mode::Limited(-Fix128::HALF_PI, Fix128::HALF_PI));

// Cone-twist joint (shoulder-like with cone + twist limits)
let cone = ConeTwistJoint::new(body_a, body_b, anchor_a, anchor_b, twist_axis)
    .with_cone_limit(Fix128::from_ratio(45, 1))  // 45° cone
    .with_twist_limit(Fix128::from_ratio(30, 1)); // 30° twist

// Breakable ball joint (breaks at force > 100)
let ball = BallJoint::new(body_a, body_b, anchor_a, anchor_b)
    .with_break_force(Fix128::from_int(100));

// Solve with breakable support — returns indices of broken joints
let broken = solve_joints_breakable(&joints, &mut bodies, dt);
```

### `raycast` - Ray & Shape Casting

| Function | Description |
|----------|-------------|
| `ray_sphere` | Ray vs sphere intersection |
| `ray_aabb` | Ray vs AABB (slab method) |
| `ray_capsule` | Ray vs capsule |
| `ray_plane` | Ray vs infinite plane |
| `sweep_sphere` | Moving sphere vs sphere (Minkowski expansion) |

### `ccd` - Continuous Collision Detection

| Function | Description |
|----------|-------------|
| `sphere_sphere_toi` | Sphere-sphere TOI via quadratic |
| `sphere_plane_toi` | Sphere-plane TOI |
| `conservative_advancement` | TOI via iterative safe stepping |
| `swept_aabb` | Swept AABB bounding volume |
| `speculative_contact` | Speculative contact for CCD integration with solver |
| `needs_ccd` | Velocity threshold check for CCD activation |

### `sleeping` - Sleep & Island System

| Type | Description |
|------|-------------|
| `IslandManager` | Union-Find island management |
| `SleepData` | Per-body sleep tracking |
| `SleepConfig` | Velocity thresholds, idle frame count |

### `trimesh` - Triangle Mesh Collision

| Type | Description |
|------|-------------|
| `Triangle` | Single triangle with Moller-Trumbore intersection |
| `TriMesh` | BVH-accelerated triangle mesh |

**Features:**
- Ray-triangle intersection (Moller-Trumbore)
- Closest point on triangle
- Sphere-mesh collision detection

### `heightfield` - Height Field Terrain

| Type | Description |
|------|-------------|
| `HeightField` | Grid-based terrain with bilinear interpolation |

**Features:**
- `sample_height(world_x, world_z)` with bilinear interpolation
- `sample_normal(world_x, world_z)` from central differences
- `collide_sphere()` for sphere-terrain collision
- `signed_distance(point)` for point-terrain distance

### `filter` - Collision Filtering

| Type | Description |
|------|-------------|
| `CollisionFilter` | Layer/mask bitmask with collision groups |
| `layers` module | Predefined layers (DEFAULT, STATIC, PLAYER, ENEMY, etc.) |

### `force` - Force Fields

| Variant | Description |
|---------|-------------|
| `Directional` | Uniform wind/gravity (direction + strength) |
| `Point` | Radial gravity well with inverse-square falloff |
| `Drag` | Velocity-proportional drag |
| `Buoyancy` | Plane-based buoyancy with density |
| `Vortex` | Spinning vortex field (center + axis + strength) |

### `motor` - PD Controllers

| Type | Description |
|------|-------------|
| `PdController` | 1D proportional-derivative controller |
| `PdController3D` | 3D PD controller |
| `JointMotor` | Joint motor with position/velocity/torque modes |

### `articulation` - Articulated Bodies

| Type | Description |
|------|-------------|
| `ArticulatedBody` | Multi-joint chain with FK propagation |
| `Link` | Joint + body pair in a chain |

**Presets:**
- `build_ragdoll()` — 12-body humanoid (head, torso, arms, legs)

### `rng` - Deterministic Random

| Type | Description |
|------|-------------|
| `DeterministicRng` | PCG-XSH-RR 32-bit generator |

**Methods:**
- `next_fix128()` — Random Fix128 in [0, 1)
- `range_fix128(min, max)` — Random in [min, max)
- `next_direction()` — Random unit Vec3Fix

### `event` - Contact Events

| Type | Description |
|------|-------------|
| `EventCollector` | Tracks contact begin/persist/end per pair |
| `ContactEvent` | Body pair + event type + contact point |
| `ContactEventType` | Begin, Persist, End |

### `query` - Shape Cast & Overlap Queries

| Function | Description |
|----------|-------------|
| `sphere_cast` | Sweep a sphere along a direction (Minkowski sum approach) |
| `capsule_cast` | Sweep a capsule along a direction (3-point sphere cast) |
| `overlap_sphere` | Find all bodies overlapping a sphere |
| `overlap_aabb` | Find all bodies within an AABB (point test) |
| `overlap_aabb_expanded` | Find all bodies within an AABB expanded by body radius |

### `character` - Character Controller

| Type | Description |
|------|-------------|
| `CharacterController` | Kinematic capsule-based character with move-and-slide |
| `CharacterConfig` | Radius, height, max slope, step height, skin width |
| `MoveResult` | Movement result with ground state |

**Features:**
- Collide-and-slide algorithm with configurable max slides
- Ground detection via downward raycast + SDF
- Stair stepping (automatic step-up for small obstacles)
- SDF terrain integration (push out of SDF surfaces)

### `rope` - XPBD Rope & Cable

| Type | Description |
|------|-------------|
| `Rope` | XPBD distance-chain rope simulation |
| `RopeConfig` | Segment count, length, stiffness, damping |

### `cloth` - XPBD Cloth Simulation

| Type | Description |
|------|-------------|
| `Cloth` | XPBD triangle mesh cloth with stretch/shear/bend constraints |
| `ClothConfig` | Stiffness, damping, self-collision, gravity |

**Features:**
- Distance constraints (stretch resistance)
- Bending constraints
- Self-collision via spatial hash grid

### `fluid` - Position-Based Fluids

| Type | Description |
|------|-------------|
| `Fluid` | Position-Based Fluids (PBF) simulation |
| `FluidConfig` | Particle count, rest density, viscosity, kernel radius |

**Features:**
- SPH density estimation with spatial hash grid
- Incompressibility constraint solving
- Viscosity (XSPH)
- Surface tension

### `deformable` - FEM-XPBD Deformable Bodies

| Type | Description |
|------|-------------|
| `DeformableBody` | FEM tetrahedral mesh deformable body |
| `DeformableConfig` | Young's modulus, Poisson's ratio, damping |

### `vehicle` - Vehicle Physics

| Type | Description |
|------|-------------|
| `Vehicle` | Complete vehicle simulation |
| `VehicleConfig` | Wheel count, suspension, engine, steering parameters |

**Features:**
- Wheel contact with ground/terrain
- Spring-damper suspension
- Engine torque with gear shifting
- Ackermann steering geometry

### `animation_blend` - Animation Blending

| Type | Description |
|------|-------------|
| `AnimationBlender` | Ragdoll-to-animation pose blending |
| `BlendMode` | Lerp, Slerp, Additive |
| `SkeletonPose` | Joint transforms for a skeleton |
| `AnimationClip` | Keyframed animation data |

### `audio_physics` - Physics-Based Audio

| Type | Description |
|------|-------------|
| `AudioGenerator` | Generates audio parameters from physics contacts |
| `AudioConfig` | Material properties for audio synthesis |
| `AudioEvent` | Impact, friction, rolling audio events |
| `AudioMaterial` | Surface material with audio properties |

### `sdf_manifold` - SDF Contact Manifold

Multi-point contact generation from SDF surfaces for more stable collision response.

### `sdf_ccd` - SDF Continuous Collision Detection

Sphere tracing-based CCD for bodies moving against SDF surfaces.

### `sdf_force` - SDF Force Fields

| Variant | Description |
|---------|-------------|
| `Attract` | Pull bodies toward SDF surface |
| `Repel` | Push bodies away from SDF surface |
| `Contain` | Keep bodies inside SDF boundary |
| `Flow` | Flow field following SDF gradient |

### `sdf_destruction` - SDF Boolean Destruction

Real-time CSG boolean operations for destructible environments.

| Type | Description |
|------|-------------|
| `DestructibleSdf` | SDF that accumulates destruction shapes |
| `DestructionShape` | Sphere, box, or custom SDF subtraction |

### `sdf_adaptive` - Adaptive SDF Evaluation

Distance-based LOD for SDF evaluation — faraway queries use coarser sampling.

### `convex_decompose` - Convex Decomposition

Generate convex hull decomposition from an SDF voxel grid for use with GJK/EPA.

### `gpu_sdf` - GPU Compute Shader

Batch SDF evaluation on GPU via compute shaders. Requires `--features std`.

### `fluid_netcode` - Fluid Netcode

Deterministic fluid simulation with delta-compressed snapshots for network sync. Requires `--features std`.

### `sim_field` - 3D Simulation Fields

| Type | Description |
|------|-------------|
| `ScalarField3D` | Uniform grid scalar field with trilinear interpolation |
| `VectorField3D` | Uniform grid vector field with trilinear interpolation |

**Features:**
- Trilinear interpolation for smooth sampling
- Diffusion (heat equation solver)
- Decay over time
- Point splatting for localized effects

### `sim_modifier` - SDF Simulation Modifiers

| Type | Description |
|------|-------------|
| `PhysicsModifier` | Trait for physics-driven SDF modification |
| `ModifiedSdf` | Chain of modifiers that alter SDF distance |
| `SingleModifiedSdf` | Single-modifier convenience wrapper |

### `thermal` - Thermal Simulation

| Type | Description |
|------|-------------|
| `ThermalModifier` | Heat diffusion SDF modifier |
| `ThermalConfig` | Conductivity, melt/freeze thresholds |
| `HeatSource` | Point/area heat source |

**Effects:** Melt (surface recedes), thermal expansion, freeze (crystalline growth)

### `pressure` - Pressure Simulation

| Type | Description |
|------|-------------|
| `PressureModifier` | Contact-force-driven deformation |
| `PressureConfig` | Yield threshold, elastic spring-back |

**Effects:** Crush (high pressure), bulge (internal pressure), permanent dents

### `erosion` - Erosion Simulation

| Type | Description |
|------|-------------|
| `ErosionModifier` | Surface material removal from forces |
| `ErosionConfig` | Hardness, erosion rate |
| `ErosionType` | Wind, Water, Chemical, Ablation |

### `fracture` - Fracture Simulation

| Type | Description |
|------|-------------|
| `FractureModifier` | Stress-driven crack propagation |
| `FractureConfig` | Fracture toughness, crack speed |
| `Crack` | Individual crack with position and direction |

**Effects:** Stress accumulation, crack seeds, Voronoi-like fragmentation via CSG subtraction

### `phase_change` - Phase Change Simulation

| Type | Description |
|------|-------------|
| `PhaseChangeModifier` | Temperature-driven state transitions |
| `PhaseChangeConfig` | Melt/boil temperatures, latent heat |
| `Phase` | Solid, Liquid, Gas |

**Effects:** Melting (solid→liquid, flows downward), vaporization (liquid→gas, expands), solidification, condensation

### `box_collider` - Oriented Bounding Box

| Type | Description |
|------|-------------|
| `OrientedBox` | OBB with center, half-extents, and rotation quaternion |

**Features:**
- GJK `Support` trait implementation for collision detection
- Fast world-space AABB computation from OBB
- Corner vertices, volume, surface area, inertia tensor

### `compound` - Compound Shape

| Type | Description |
|------|-------------|
| `CompoundShape` | Multi-shape collider with local transforms |
| `ChildShape` | Individual shape + local offset/rotation |

**Features:**
- Merges child AABBs for broadphase
- Per-child GJK support (transforms direction to child local space)
- Mass properties from children

### `contact_cache` - Persistent Contact Cache

| Type | Description |
|------|-------------|
| `ContactCache` | Frame-persistent contact manifold storage |
| `ContactManifold` | Up to 4 contact points with warm-started lambdas |

**Features:**
- HashMap O(1) manifold lookup by body pair
- Warm starting (carry lambda across frames for stable stacking)
- Automatic manifold pruning at end of frame

### `dynamic_bvh` - Dynamic AABB Tree

| Type | Description |
|------|-------------|
| `DynamicBvh` | Incremental BVH with insert/remove/update |

**Features:**
- O(log n) insert, remove, and update operations
- AVL-style tree balancing (rotation on height imbalance)
- Fat AABBs with margin to reduce update frequency
- AABB query and ray query

### `material` - Material Pair Table

| Type | Description |
|------|-------------|
| `MaterialTable` | Per-pair friction/restitution override table |
| `MaterialProperties` | Friction, restitution, combine rule |
| `CombineRule` | Average, Min, Max, Multiply |

### `debug_render` - Debug Visualization

| Type | Description |
|------|-------------|
| `DebugRenderer` | Wireframe rendering of physics state |
| `DebugLine` | Line segment with color |
| `DebugDrawFlags` | Bitmask for bodies, contacts, joints, BVH, forces |

### `profiling` - Performance Profiling

| Type | Description |
|------|-------------|
| `PhysicsProfiler` | Per-stage timing and statistics |
| `StageTimer` | Individual stage measurement |
| `FrameStats` | Broadphase/narrowphase/solver/total times |

### `interpolation` - Substep Interpolation

| Type | Description |
|------|-------------|
| `WorldSnapshot` | Captured physics state for interpolation |
| `interpolate_snapshots` | NLERP blending between two snapshots |

**Features:**
- Capture position + rotation snapshots
- NLERP quaternion interpolation (faster than SLERP, visually smooth)
- Alpha-based blending for fixed-timestep rendering

### `pipeline` - Metric Aggregation Pipeline

| Type | Description |
|------|-------------|
| `MetricPipeline` | Lock-free ring buffer metric aggregation engine |
| `MetricRegistry` | Named metric registration and lookup |
| `MetricSnapshot` | Point-in-time metric state capture |
| `MetricEvent` | Timestamped metric event with type and value |
| `MetricType` | Counter, Gauge, Histogram metric kinds |
| `RingBuffer` | Fixed-capacity circular buffer for time-series data |

**Features:**
- Lock-free ring buffer for high-throughput metric collection
- Named metric registry with typed metric support
- Snapshot capture for periodic reporting
- Integrates with `sketch` module for probabilistic aggregation

### `sketch` - Probabilistic Sketches

| Type | Description |
|------|-------------|
| `HyperLogLog` | Cardinality estimation with configurable precision |
| `DDSketch` | Distribution-aware quantile sketch (P50/P99/P99.9) |
| `CountMinSketch` | Frequency estimation with configurable width/depth |
| `HeavyHitters` | Top-k frequent item tracking via Count-Min Sketch |
| `FnvHasher` | FNV-1a hash for deterministic, allocation-free hashing |
| `Mergeable` | Trait for sketch merging across distributed workers |

**Size variants:**
- `HyperLogLog10` (1 KB), `HyperLogLog12` (4 KB), `HyperLogLog14` (16 KB), `HyperLogLog16` (64 KB)
- `DDSketch128`, `DDSketch256`, `DDSketch512`
- `CountMinSketch256x4`, `CountMinSketch512x4`, `CountMinSketch1024x8`

### `anomaly` - Streaming Anomaly Detection

| Type | Description |
|------|-------------|
| `MadDetector` | Median Absolute Deviation detector |
| `EwmaDetector` | Exponentially Weighted Moving Average detector |
| `ZScoreDetector` | Standard Z-score detector with running statistics |
| `CompositeDetector` | Combines multiple detectors with voting |
| `StreamingMedian` | O(log n) streaming median via sorted-insert buffer |
| `AnomalyEvent` | Timestamped anomaly report with severity |
| `AnomalyCallback` | Callback trait for anomaly notifications |

**Features:**
- Streaming operation with constant memory per detector
- Composite detector with configurable voting threshold
- Callback-driven anomaly notification

### `privacy` - Local Differential Privacy

| Type | Description |
|------|-------------|
| `LaplaceNoise` | Calibrated Laplace mechanism for numeric queries |
| `Rappor` | RAPPOR protocol for categorical data collection |
| `RandomizedResponse` | Binary randomized response with tunable epsilon |
| `PrivacyBudget` | Epsilon budget tracker with composition accounting |
| `PrivateAggregator` | Privacy-preserving aggregation over multiple queries |
| `XorShift64` | Deterministic PRNG for reproducible noise generation |

**Features:**
- Laplace noise calibrated to sensitivity/epsilon
- RAPPOR with Bloom filter encoding for categorical data
- Composable privacy budget tracking

## SDF Collider (ALICE-SDF Integration)

ALICE-Physics can use [ALICE-SDF](../ALICE-SDF) distance fields as collision shapes. Instead of approximating complex shapes with convex hulls (GJK/EPA), the solver samples the SDF directly — giving mathematically exact surfaces at O(1) cost per query.

### How It Works

```
Body (sphere)                    SdfCollider
  ┌───┐                         ┌──────────────────────┐
  │ ● │──world_to_local(pos)──▶│ SdfField::distance() │─── >0 → no hit (early-out)
  └───┘                         │ SdfField::normal()   │─── ≤0 → contact + resolve
                                │ cached inv_rotation   │
                                │ cached scale_f32      │
                                └──────────────────────┘
```

### Key Optimizations

| Optimization | Description | Benefit |
|-------------|-------------|---------|
| **Early-out** | Call `distance()` (1 eval) first, only compute `normal()` (4 evals) on collision | 80% fewer evals for non-colliding bodies |
| **Cached invariants** | Pre-computed `inv_rotation`, `scale_f32`, `inv_scale_f32` | No per-query recomputation |
| **Rayon parallel** | `par_iter_mut` over bodies with `--features parallel` | Linear speedup with cores |
| **4-eval combined** | Tetrahedral gradient gives distance + normal from 4 evals | 1 eval saved vs naive 1+4 |

### Usage

```rust
use alice_physics::prelude::*;
use alice_physics::sdf_collider::SdfCollider;
use alice_sdf::physics_bridge::CompiledSdfField;
use alice_sdf::prelude::*;

// 1. Create SDF shape in ALICE-SDF
let terrain = SdfNode::plane(0.0, 1.0, 0.0, 0.0)  // ground plane
    .union(SdfNode::sphere(2.0).translate(0.0, -1.5, 0.0));  // hill

let field = CompiledSdfField::new(terrain);

// 2. Create physics world
let mut world = PhysicsWorld::new(PhysicsConfig::default());

// 3. Register SDF as static collider
let collider = SdfCollider::new_static(
    Box::new(field),
    Vec3Fix::ZERO,                    // position
    QuatFix::IDENTITY,                // rotation
);
world.add_sdf_collider(collider);

// 4. Add dynamic bodies — they will collide with the SDF surface
let ball = RigidBody::new_dynamic(
    Vec3Fix::from_int(0, 10, 0),     // start above terrain
    Fix128::ONE,                      // mass
);
world.add_body(ball);

// 5. Simulate — SDF collisions resolved automatically in step()
let dt = Fix128::from_ratio(1, 60);
for _ in 0..300 {
    world.step(dt);
}
```

### SdfCollider API

| Method | Description |
|--------|-------------|
| `SdfCollider::new_static(field, position, rotation)` | Static SDF collider |
| `SdfCollider::new_dynamic(field, position, rotation, body_index)` | Attached to a body |
| `.with_scale(scale)` | Set uniform scale (updates cached invariants) |
| `.update_cache()` | Recompute cached `inv_rotation`, `scale_f32` after manual changes |
| `collide_point_sdf(point, sdf)` | Point vs SDF contact test |
| `collide_sphere_sdf(center, radius, sdf)` | Sphere vs SDF contact test |

### Requirements

Enable the `physics` feature in ALICE-SDF:

```toml
[dependencies]
alice-sdf = { path = "../ALICE-SDF", features = ["physics"] }
```

See the [ALICE-SDF README](../ALICE-SDF/README.md#physics-bridge-alice-physics-integration) for `CompiledSdfField` details.

## Deterministic Neural Controller (ALICE-ML Integration)

ALICE-Physics integrates with [ALICE-ML](../ALICE-ML) to provide **bit-exact deterministic AI** for game characters. By combining 1.58-bit ternary weights {-1, 0, +1} with 128-bit fixed-point arithmetic, neural inference reduces to pure addition/subtraction — no floating-point, no rounding, no platform-dependent behavior.

This is the "holy grail" for networked fighting games and action games: all clients compute identical AI behavior without synchronization.

### How It Works

```
Ternary Weight {-1, 0, +1}:
  +1 → Fix128 addition
  -1 → Fix128 subtraction
   0 → skip (free sparsity)

Result: Zero floating-point multiplication in the entire inference pipeline.
```

### Ragdoll Controller Example

```rust
use alice_physics::prelude::*;
use alice_ml::{TernaryWeight, quantize_to_ternary};

// 1. Quantize pre-trained weights to ternary
let (w1, _) = quantize_to_ternary(&trained_weights_l1, hidden_size, input_size);
let (w2, _) = quantize_to_ternary(&trained_weights_l2, output_size, hidden_size);

// 2. Convert to fixed-point (one-time)
let ftw1 = FixedTernaryWeight::from_ternary_weight(w1);
let ftw2 = FixedTernaryWeight::from_ternary_weight(w2);

// 3. Build deterministic network
let network = DeterministicNetwork::new(
    vec![ftw1, ftw2],
    vec![Activation::ReLU, Activation::HardTanh],
);

// 4. Create ragdoll controller
let config = ControllerConfig {
    max_torque: Fix128::from_int(100),
    num_joints: 8,    // 8 joints × 3 axes = 24 outputs
    num_bodies: 9,    // 9 body parts × 13 features = 117 inputs
    features_per_body: 13,  // pos(3) + vel(3) + rot(4) + angvel(3)
};
let mut controller = RagdollController::new(network, config);

// 5. In physics loop — deterministic across ALL clients
for frame in 0..3600 {
    let output = controller.compute(&world.bodies);
    for (joint_idx, torque) in output.torques.iter().enumerate() {
        world.bodies[joint_idx].apply_impulse(*torque);
    }
    world.step(dt);
}
```

### Neural Controller API

| Type | Description |
|------|-------------|
| `FixedTernaryWeight` | TernaryWeight wrapper with Fix128 scale |
| `DeterministicNetwork` | Multi-layer network (zero-allocation forward pass) |
| `RagdollController` | Body states → joint torques |
| `ControllerConfig` | max_torque, num_joints, num_bodies |
| `ControllerOutput` | Vec3Fix torque per joint |
| `Activation` | ReLU, HardTanh, TanhApprox, None |

| Function | Description |
|----------|-------------|
| `fix128_ternary_matvec()` | Core kernel — pure add/sub, zero multiplication |
| `fix128_relu()` | max(0, x) via sign-bit comparison |
| `fix128_hard_tanh()` | clamp(x, -1, 1) for bounded output |
| `fix128_tanh_approx()` | Padé rational x(27+x²)/(27+9x²) |
| `fix128_leaky_relu()` | Leaky ReLU with Fix128 alpha |

### Requirements

```toml
[dependencies]
alice-physics = { path = "../ALICE-Physics", features = ["neural"] }
```

## Configuration

```rust
// SolverConfig / PhysicsConfig
let config = PhysicsConfig {
    substeps: 8,       // XPBD substeps per frame (more = stable but slower)
    iterations: 4,     // Constraint iterations per substep
    gravity: Vec3Fix::new(
        Fix128::ZERO,
        Fix128::from_int(-10),  // -10 m/s²
        Fix128::ZERO,
    ),
    damping: Fix128::from_ratio(99, 100),  // 0.99 velocity retention
};

// Or use defaults
let config = PhysicsConfig::default();
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Fix128 add/sub | O(1) | ~2-3 cycles |
| Fix128 mul | O(1) | ~10 cycles (128-bit multiply) |
| Fix128 div | O(1) | ~40 cycles (128-bit division) |
| CORDIC sin/cos | O(48) | 48 iterations, deterministic |
| GJK intersection | O(64) | Max 64 iterations |
| EPA penetration | O(64) | Max 64 iterations |
| BVH build | O(n log n) | Morton code sort |
| BVH query | O(log n) | Stackless traversal |

## Building

**MSRV:** 1.70.0 (Minimum Supported Rust Version)

```bash
# Standard build
cargo build --release

# no_std build (for embedded/WASM)
cargo build --release --no-default-features

# Run tests
cargo test

# Run all feature combinations
cargo test --features simd
cargo test --features parallel
cargo test --features neural
cargo test --features "simd,parallel"

# Run benchmarks (requires nightly or criterion)
cargo bench
```

## Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `simd` | No | SIMD-accelerated Fix128/Vec3Fix operations (x86_64) |
| `parallel` | No | Constraint batching with Rayon (graph-colored parallel solving) |
| `neural` | No | Deterministic neural controller via ALICE-ML ternary inference |
| `python` | No | Python bindings (PyO3 + NumPy zero-copy) |
| `replay` | No | Replay recording/playback via ALICE-DB |
| `ffi` | No | C FFI for Unity, Unreal Engine, and other game engines |
| `wasm` | No | WebAssembly bindings via wasm-bindgen |
| `analytics` | No | Simulation profiling via ALICE-Analytics |

> **Note:** `wasm` and `ffi` are mutually exclusive — enabling both causes a compile error.

```bash
# Enable SIMD optimizations
cargo build --release --features simd

# Enable parallel constraint solving
cargo build --release --features parallel

# Enable both
cargo build --release --features "simd,parallel"

# Enable neural controller (requires ALICE-ML)
cargo build --release --features neural

# Build C shared library for game engines
cargo build --release --features ffi
```

## Game Engine Integration (C FFI / Unity / UE5)

ALICE-Physics provides a C FFI layer for integration with Unity, Unreal Engine, and any language that can call C functions.

### Building the Shared Library

```bash
# macOS (.dylib)
cargo build --release --features ffi

# Linux (.so)
cargo build --release --features ffi

# Windows (.dll)
cargo build --release --features ffi
```

The output library is in `target/release/` (`libalice_physics.dylib`, `libalice_physics.so`, or `alice_physics.dll`).

### C API

The C header is at `include/alice_physics.h`. All types use `f64` at the FFI boundary and convert to `Fix128` internally.

```c
#include "alice_physics.h"

// Create world
AlicePhysicsWorld* world = alice_physics_world_create();

// Add bodies
AliceVec3 pos = {0.0, 10.0, 0.0};
uint32_t body = alice_physics_body_add_dynamic(world, pos, 1.0);

// Step simulation
alice_physics_world_step(world, 1.0 / 60.0);

// Read back position
AliceVec3 out_pos;
alice_physics_body_get_position(world, body, &out_pos);

// State serialization (rollback netcode)
uint32_t len;
uint8_t* state = alice_physics_state_serialize(world, &len);
alice_physics_state_deserialize(world, state, len);
alice_physics_state_free(state, len);

// Cleanup
alice_physics_world_destroy(world);
```

### Unity C# Bindings

Copy `bindings/AlicePhysics.cs` and the native library to your Unity project:

```
Assets/
  Plugins/
    macOS/    libalice_physics.dylib
    Win64/    alice_physics.dll
    Linux/    libalice_physics.so
  Scripts/
    AlicePhysics.cs
```

```csharp
using AlicePhysics;

var world = new AlicePhysicsWorld();
uint body = world.AddDynamicBody(new Vector3(0, 10, 0), 1.0);
world.Step(1.0 / 60.0);
Vector3 pos = world.GetBodyPosition(body);

// Rollback netcode
byte[] state = world.SerializeState();
world.DeserializeState(state);

world.Dispose();
```

### Unreal Engine 5 Plugin

Copy the `unreal-plugin/` contents to `Plugins/AlicePhysics/` in your UE5 project and place the native library in `ThirdParty/AlicePhysics/lib/<Platform>/`.

The plugin provides `UAlicePhysicsWorldComponent` with full Blueprint support:

- `AddDynamicBody`, `AddStaticBody` — body creation
- `GetBodyPosition`, `GetBodyRotation`, `GetBodyVelocity` — state queries
- `ApplyImpulse`, `ApplyImpulseAt` — force application
- `SerializeState`, `DeserializeState` — rollback netcode
- Automatic coordinate system conversion (UE5 Z-up cm → ALICE Y-up m)

### Release Workflow

Tag a version to trigger automatic cross-platform builds:

```bash
git tag v0.4.0
git push origin v0.4.0
```

GitHub Actions builds for macOS (ARM + Intel), Windows, and Linux, then packages:
- **UE5 Plugin ZIP** — includes native library + plugin source + header
- **Unity Package ZIP** — includes native library + C# bindings + header

## Comparison with Floating-Point Engines

| Aspect | ALICE-Physics | Float Engines |
|--------|---------------|---------------|
| Determinism | Guaranteed | Platform-dependent |
| Precision | 64-bit fraction | 23-bit (f32) / 52-bit (f64) |
| Speed | Slower (2-5x) | Faster |
| Rollback | Trivial | Requires care |
| Embedded | no_std | Requires FPU |
| Range | ±9.2×10^18 | ±3.4×10^38 (f32) |

## Deterministic Netcode

ALICE-Physics includes a frame-based deterministic netcode foundation. Because the engine guarantees bit-exact results, **only player inputs need to be synchronized** — state sync is unnecessary.

### Bandwidth Savings

| Approach | Per-Frame (10 bodies, 2 players) |
|----------|-------------------------------|
| State sync (traditional) | ~1,600 bytes |
| **Input sync (ALICE)** | **~40 bytes** |
| Savings | **97.5%** |

### Core Types

| Type | Description |
|------|-------------|
| `FrameInput` | 20-byte serializable player input (movement, actions, aim) |
| `SimulationChecksum` | XOR rolling hash of physics state (WyHash-style avalanche) |
| `SimulationSnapshot` | Full state capture for rollback |
| `DeterministicSimulation` | PhysicsWorld wrapper with frame counter, checksum history, snapshot ring buffer |
| `InputApplicator` | Trait for game-specific input → physics force mapping |

### Usage

```rust
use alice_physics::prelude::*;

// Both clients create identical simulation
let mut sim = DeterministicSimulation::new(NetcodeConfig::default());

// Add bodies and assign to players
let body0 = sim.add_body(RigidBody::new_dynamic(
    Vec3Fix::from_int(0, 10, 0), Fix128::ONE,
));
sim.assign_player_body(0, body0);

// Each frame: collect inputs, advance, compare checksums
let inputs = vec![
    FrameInput::new(0).with_movement(Vec3Fix::from_int(1, 0, 0)),
    FrameInput::new(1).with_movement(Vec3Fix::from_int(0, 0, -1)),
];
let checksum = sim.advance_frame(&inputs);

// Save snapshot for rollback
sim.save_snapshot();

// Verify remote client's checksum
assert_eq!(sim.verify_checksum(1, checksum), Some(true));
```

## ALICE-Sync Integration (Game Engine Pipeline)

ALICE-Physics integrates with [ALICE-Sync](../ALICE-Sync) for complete multiplayer game networking. Enable `physics` feature in ALICE-Sync:

```toml
[dependencies]
alice-sync = { path = "../ALICE-Sync", features = ["physics"] }
```

### How It Works

```
Player Input ──► InputFrame (i16, 24B) ──► FrameInput (Fix128) ──► PhysicsWorld
                     ALICE-Sync                 bridge                 step()
                                                  │
PhysicsWorld ──► SimulationChecksum ──► WorldHash ──► Desync Verify
                   from_world()           bridge        ALICE-Sync
```

ALICE-Sync's `PhysicsRollbackSession` wraps both `RollbackSession` (input sync) and `DeterministicSimulation` (physics) into a single game loop driver:

1. Submit local input → predict remote inputs
2. Convert InputFrame (i16) → FrameInput (Fix128)
3. Save snapshot → step physics → record checksum
4. On mismatch: auto rollback + re-simulate

### Bandwidth

| Approach | Per-Frame (4 players, 60fps) |
|----------|----------------------------|
| State sync | ~960 KB/s |
| **Input sync (ALICE)** | **5.6 KB/s** |
| Savings | **99.4%** |

See [ALICE-Sync README](../ALICE-Sync/README.md#game-engine-pipeline-alice-physics-bridge) for full API.

## Python Bindings (PyO3 + NumPy Zero-Copy)

Install with:

```bash
pip install maturin
maturin develop --release --features python
```

### Optimization Layers

| Layer | Technique | Effect |
|-------|-----------|--------|
| L1 | GIL Release (`py.allow_threads`) | Parallel physics stepping |
| L2 | Zero-Copy NumPy (`into_pyarray_bound`) | No memcpy for bulk positions/velocities |
| L3 | Batch API (`step_n`, `positions`, `states`) | FFI amortization |
| L4 | Batch Mutation (`add_bodies_batch`, `set_velocities_batch`, `apply_impulses_batch`) | Bulk operations with GIL release |
| L5 | Rust backend (Fix128, XPBD, BVH) | Hardware-speed simulation |

### Python API

```python
import alice_physics

# Basic physics world
world = alice_physics.PhysicsWorld()
body0 = world.add_dynamic_body(0.0, 10.0, 0.0, mass=1.0)
ground = world.add_static_body(0.0, 0.0, 0.0)

# Step with GIL release (other Python threads can run)
world.step(1.0 / 60.0)

# Batch step for training loops
world.step_n(1.0 / 60.0, steps=300)

# Get all positions as NumPy (N, 3) float64 array (zero-copy)
positions = world.positions()  # shape: (N, 3)
velocities = world.velocities()  # shape: (N, 3)

# Deterministic netcode simulation
sim = alice_physics.DeterministicSimulation(player_count=2, fps=60)
body = sim.add_body(0.0, 10.0, 0.0, mass=1.0)
sim.assign_player(0, body)

# Advance with player inputs: (player_id, move_x, move_y, move_z, actions)
checksum = sim.advance_frame([(0, 1.0, 0.0, 0.0, 0), (1, 0.0, 0.0, -1.0, 0)])

# Snapshot for rollback
frame = sim.save_snapshot()
sim.load_snapshot(frame)

# Serialization
state = world.serialize_state()  # NumPy uint8 array
world.deserialize_state(state.tolist())

# Frame input encoding (20 bytes, network-ready)
data = alice_physics.encode_frame_input(player_id=0, move_x=1.0, actions=0x3)
player_id, mx, my, mz, actions, ax, ay, az = alice_physics.decode_frame_input(data)

# === Batch APIs (v0.4.0) ===

import numpy as np

# Batch body creation from (N,4) array [x, y, z, mass]
ids = world.add_bodies_batch(np.array([
    [0.0, 10.0, 0.0, 1.0],
    [5.0, 10.0, 0.0, 2.0],
    [10.0, 10.0, 0.0, 0.5],
]))

# Batch velocity update (N,3) with GIL release
world.set_velocities_batch(np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 5.0, 0.0],
]))

# Batch impulse (M,4) [body_id, ix, iy, iz] with GIL release
world.apply_impulses_batch(np.array([
    [0.0, 100.0, 0.0, 0.0],
    [2.0, 0.0, 50.0, 0.0],
]))

# Combined state output (N,10) [px,py,pz, vx,vy,vz, qx,qy,qz,qw]
states = world.states()  # shape: (N, 10), zero-copy NumPy
```

## Replay Recording (ALICE-DB Integration)

ALICE-Physics can record and replay simulation trajectories via [ALICE-DB](../ALICE-DB). Enable with `--features replay`.

```toml
[dependencies]
alice-physics = { path = "../ALICE-Physics", features = ["replay"] }
```

### Recording

```rust
use alice_physics::replay::ReplayRecorder;

// Create recorder (path, body_count)
let mut recorder = ReplayRecorder::new("./replay_data", 3)?;

// In game loop: record every frame
for _ in 0..300 {
    world.step(dt);
    recorder.record_frame(&world)?;
}

recorder.flush()?;
recorder.close()?;
```

### Playback

```rust
use alice_physics::replay::ReplayPlayer;

let player = ReplayPlayer::open("./replay_data", 3)?;

// Random access: get position at any frame
let pos = player.get_position(frame, body_id)?;
// Returns Option<(f32, f32, f32)>

// Range query: scan positions over frame range
let trajectory = player.scan_positions(0, 299, body_id)?;
// Returns Vec<(u64, f32, f32, f32)>

player.close()?;
```

### Storage Format

Each body stores 6 channels (pos_x/y/z, vel_x/y/z) in ALICE-DB. ALICE-DB's model-based compression automatically fits trajectories:
- Constant velocity → linear model (2 coefficients)
- Projectile motion → quadratic model (3 coefficients)
- Complex motion → Fourier model

## Test Results

```
v0.4.0 Test Summary:
  - 358 unit tests across 63 modules
  - 62 integration tests (end-to-end physics scenarios)
  - 12 doc tests (runnable examples)
  - Total: 432 passing tests (435 with --features parallel)
  - Clippy: 0 warnings
  - CI: Format + Clippy + Test (macOS/Ubuntu/Windows)
  - All feature combinations pass (default, parallel, simd, no_std)
```

## Cross-Crate Bridges

ALICE-Physics connects to other ALICE ecosystem crates via feature-gated bridge modules:

| Bridge | Feature | Target Crate | Description |
|--------|---------|--------------|-------------|
| Physics Visualization | `view` | [ALICE-View](../ALICE-View) | Real-time physics debug overlay rendering |
| GPU Physics Controller | `trt` | [ALICE-TRT](../ALICE-TRT) | GPU ternary inference for physics control policies |
| Physics State Streaming | `asp` | [ALICE-Streaming-Protocol](../ALICE-Streaming-Protocol) | Physics body state delta encoding as ASP D-packets |
| `db_bridge` | `replay` | [ALICE-DB](../ALICE-DB) | Physics state snapshot persistence (energy, bodies, contacts) |
| `analytics_bridge` | `analytics` | [ALICE-Analytics](../ALICE-Analytics) | Simulation profiling (step time, collision pairs, energy drift) |

### Recent Performance Improvements

| Module | Change | Impact |
|--------|--------|--------|
| `fluid.rs` | Buffer reuse in neighbor queries | ~99% allocation reduction in SPH simulation |
| `animation_blend.rs` | `clone_from()` replaces `clone()` | Eliminates 60 allocs/sec during blend |

### Cargo Profile

Standardized `[profile.bench]` added for consistent benchmarking across ALICE crates.

## Complete API Reference

### Core Types

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `math` | `Fix128` | 128-bit fixed-point number (I64F64) |
| `math` | `Vec3Fix` | 3D vector with Fix128 components |
| `math` | `QuatFix` | Quaternion for rotations |
| `math` | `Mat3Fix` | 3x3 matrix for inertia tensors |
| `math` | `SIMD_WIDTH` | Compile-time SIMD lane count (AVX2=8, NEON=4, scalar=1) |
| `math` | `simd_width()` | Returns SIMD width at compile time |
| `math` | `select_fix128()` | Branchless conditional select |
| `math` | `select_vec3()` | Branchless Vec3 conditional select |

### Collision Detection

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `collider` | `AABB` | Axis-Aligned Bounding Box |
| `collider` | `Sphere` | Sphere collider |
| `collider` | `Capsule` | Capsule (cylinder + hemispheres) |
| `collider` | `ConvexHull` | Arbitrary convex polyhedron |
| `collider` | `ScaledShape` | Uniform scale wrapper for any Support shape |
| `collider` | `CollisionResult` | Contact point, normal, depth |
| `collider` | `Support` trait | GJK support function interface |
| `box_collider` | `OrientedBox` | OBB with center, half-extents, rotation |
| `compound` | `CompoundShape` | Multi-shape compound collider |
| `compound` | `ShapeRef` | Shape reference within compound |
| `cylinder` | `Cylinder` | Cylinder collider with GJK support |
| `filter` | `CollisionFilter` | Layer/mask bitmask collision groups |

### Solver & Dynamics

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `solver` | `PhysicsWorld` | Main simulation world |
| `solver` | `PhysicsConfig` | Substeps, iterations, gravity, damping |
| `solver` | `RigidBody` | Dynamic/static/sensor rigid body |
| `solver` | `BodyType` | Dynamic, Static, Kinematic |
| `solver` | `DistanceConstraint` | Fixed distance between anchor points |
| `solver` | `ContactConstraint` | Collision response with friction/restitution |
| `solver` | `ContactModifier` | Trait for custom contact modification (std) |
| `joint` | `BallJoint` | Spherical joint (3 rotational DOF) |
| `joint` | `HingeJoint` | Revolute joint with angle limits |
| `joint` | `FixedJoint` | Weld joint (0 DOF) |
| `joint` | `SliderJoint` | Prismatic joint with limits |
| `joint` | `SpringJoint` | Damped spring constraint |
| `joint` | `D6Joint` | 6-DOF configurable joint |
| `joint` | `ConeTwistJoint` | Cone swing + twist limit joint |
| `joint` | `solve_joints_breakable()` | Solve with break-force support |
| `motor` | `PdController` | 1D proportional-derivative controller |
| `motor` | `JointMotor` | Joint motor (position/velocity/torque modes) |
| `force` | `ForceField` | Wind, gravity well, drag, buoyancy, vortex |
| `force` | `ForceFieldInstance` | Active force field in simulation |
| `material` | `MaterialTable` | Per-pair friction/restitution override |
| `material` | `PhysicsMaterial` | Material properties |
| `material` | `CombineRule` | Average, Min, Max, Multiply |

### Spatial Acceleration

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `bvh` | `LinearBvh` | Flat array BVH with stackless traversal |
| `bvh` | `BvhNode` | 32-byte cache-aligned node |
| `bvh` | `BvhPrimitive` | Primitive entry for BVH construction |
| `dynamic_bvh` | `DynamicAabbTree` | Incremental BVH with O(log n) insert/remove |
| `spatial` | `SpatialGrid` | Hash grid for neighbor queries |

### Queries

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `raycast` | `Ray` | Ray origin + direction |
| `raycast` | `RayHit` | Hit result (t, normal, index) |
| `raycast` | `ray_sphere()`, `ray_aabb()`, `ray_capsule()`, `ray_plane()` | Shape-specific ray tests |
| `raycast` | `raycast_all_spheres()`, `raycast_all_aabbs()` | Batch raycasts |
| `query` | `sphere_cast()` | Sweep sphere along direction |
| `query` | `capsule_cast()` | Sweep capsule along direction |
| `query` | `overlap_sphere()` | Find bodies in sphere |
| `query` | `overlap_aabb()` | Find bodies in AABB |
| `query` | `batch_raycast()`, `batch_sphere_cast()` | Batch queries |
| `ccd` | `sphere_sphere_toi()` | Time of impact (sphere-sphere) |
| `ccd` | `conservative_advancement()` | Iterative safe stepping TOI |
| `ccd` | `speculative_contact()` | Speculative CCD contacts |

### Soft Body & Simulation

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `rope` | `Rope` | XPBD distance-chain rope |
| `rope` | `RopeConfig` | Segment count, stiffness, damping |
| `cloth` | `Cloth` | XPBD triangle mesh cloth |
| `cloth` | `ClothConfig` | Stiffness, self-collision, gravity |
| `fluid` | `Fluid` | Position-Based Fluids (PBF) |
| `fluid` | `FluidConfig` | Rest density, viscosity, kernel radius |
| `deformable` | `DeformableBody` | FEM tetrahedral mesh |
| `deformable` | `DeformableConfig` | Young's modulus, Poisson's ratio |
| `vehicle` | `Vehicle` | Vehicle simulation |
| `vehicle` | `VehicleConfig` | Wheels, suspension, engine, steering |
| `character` | `CharacterController` | Kinematic capsule move-and-slide |
| `character` | `CharacterConfig` | Radius, height, max slope, step height |

### SDF Integration

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `sdf_collider` | `SdfCollider` | SDF as collision shape |
| `sdf_collider` | `SdfField` trait | Distance + normal evaluation interface |
| `sdf_collider` | `ClosureSdf` | Closure-based SDF implementation |
| `sdf_manifold` | `SdfManifold` | Multi-point contact from SDF surface |
| `sdf_manifold` | `ManifoldConfig` | Contact point count, spread |
| `sdf_ccd` | `SdfCcdConfig` | Sphere tracing CCD for SDF |
| `sdf_force` | `SdfForceField` | SDF-driven force field |
| `sdf_force` | `SdfForceType` | Attract, Repel, Contain, Flow |
| `sdf_destruction` | `DestructibleSdf` | CSG boolean destruction (std) |
| `sdf_destruction` | `DestructionShape` | Sphere/box subtraction shape |
| `sdf_adaptive` | `AdaptiveSdfEvaluator` | Distance-based LOD evaluation (std) |
| `convex_decompose` | `DecomposeConfig` | Voxel grid decomposition settings (std) |
| `gpu_sdf` | `GpuSdfBatch` | GPU compute SDF batch (std) |
| `gpu_sdf` | `GpuSdfInstancedBatch` | Instanced GPU SDF batch |
| `gpu_sdf` | `GpuSdfMultiDispatch` | Multi-batch GPU dispatch |

### SDF Simulation Modifiers

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `sim_field` | `ScalarField3D` | 3D scalar field with trilinear interpolation (std) |
| `sim_field` | `VectorField3D` | 3D vector field with trilinear interpolation (std) |
| `sim_modifier` | `PhysicsModifier` trait | SDF modifier interface (std) |
| `sim_modifier` | `ModifiedSdf` | Modifier chain (std) |
| `thermal` | `ThermalModifier` | Heat diffusion, melt, freeze (std) |
| `thermal` | `ThermalConfig` | Conductivity, melt/freeze thresholds |
| `pressure` | `PressureModifier` | Contact-force deformation (std) |
| `pressure` | `PressureConfig` | Yield threshold, spring-back |
| `erosion` | `ErosionModifier` | Wind/water/chemical erosion (std) |
| `erosion` | `ErosionType` | Wind, Water, Chemical, Ablation |
| `fracture` | `FractureModifier` | Stress-driven crack propagation (std) |
| `fracture` | `FractureConfig` | Toughness, crack speed |
| `phase_change` | `PhaseChangeModifier` | Solid/liquid/gas transitions (std) |
| `phase_change` | `Phase` | Solid, Liquid, Gas |

### Game Systems

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `animation_blend` | `AnimationBlender` | Ragdoll-to-animation blending |
| `animation_blend` | `BlendMode` | Lerp, Slerp, Additive |
| `animation_blend` | `SkeletonPose` | Joint transforms for skeleton |
| `audio_physics` | `AudioGenerator` | Physics-based audio parameters |
| `audio_physics` | `AudioEvent` | Impact, friction, rolling events |
| `audio_physics` | `AudioMaterial` | Surface material with audio properties |
| `netcode` | `DeterministicSimulation` | PhysicsWorld + frame counter + checksum |
| `netcode` | `FrameInput` | 20-byte serializable player input |
| `netcode` | `SimulationChecksum` | XOR rolling hash of physics state |
| `netcode` | `SimulationSnapshot` | Full state capture for rollback |
| `netcode` | `InputApplicator` trait | Game-specific input → force mapping |
| `fluid_netcode` | `FluidSnapshot` | Deterministic fluid state (std) |
| `fluid_netcode` | `FluidDelta` | Delta-compressed fluid update (std) |

### Analytics & Privacy

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `sketch` | `HyperLogLog` | Cardinality estimation (std) |
| `sketch` | `DDSketch` | Quantile sketch (P50/P99/P99.9) (std) |
| `sketch` | `CountMinSketch` | Frequency estimation (std) |
| `sketch` | `HeavyHitters` | Top-k frequent items (std) |
| `sketch` | `FnvHasher` | FNV-1a deterministic hasher (std) |
| `sketch` | `Mergeable` trait | Distributed sketch merging (std) |
| `anomaly` | `MadDetector` | Median Absolute Deviation (std) |
| `anomaly` | `EwmaDetector` | Exponentially Weighted Moving Average (std) |
| `anomaly` | `ZScoreDetector` | Standard Z-score detector (std) |
| `anomaly` | `CompositeDetector` | Multi-detector voting (std) |
| `anomaly` | `StreamingMedian` | O(log n) streaming median (std) |
| `privacy` | `LaplaceNoise` | Calibrated Laplace mechanism (std) |
| `privacy` | `Rappor` | RAPPOR protocol for categorical data (std) |
| `privacy` | `RandomizedResponse` | Binary randomized response (std) |
| `privacy` | `PrivacyBudget` | Epsilon budget tracker (std) |
| `privacy` | `PrivateAggregator` | Privacy-preserving aggregation (std) |
| `privacy` | `XorShift64` | Deterministic PRNG (std) |
| `pipeline` | `MetricPipeline` | Lock-free ring buffer aggregation (std) |
| `pipeline` | `MetricRegistry` | Named metric registration (std) |
| `pipeline` | `MetricEvent` | Timestamped metric event (std) |
| `pipeline` | `MetricType` | Counter, Gauge, Histogram (std) |
| `pipeline` | `RingBuffer` | Fixed-capacity circular buffer (std) |

### Utility

| Module | Type / Function | Description |
|--------|----------------|-------------|
| `rng` | `DeterministicRng` | PCG-XSH-RR 32-bit generator |
| `event` | `EventCollector` | Contact begin/persist/end tracking |
| `event` | `ContactEvent` | Body pair + event type + contact point |
| `sleeping` | `IslandManager` | Union-Find island management |
| `sleeping` | `SleepConfig` | Velocity thresholds, idle frame count |
| `trimesh` | `TriMesh` | BVH-accelerated triangle mesh |
| `trimesh` | `Triangle` | Single triangle (Moller-Trumbore) |
| `heightfield` | `HeightField` | Grid terrain with bilinear interpolation |
| `articulation` | `ArticulatedBody` | Multi-joint chain with FK propagation |
| `articulation` | `FeatherstoneSolver` | O(n) forward dynamics |
| `contact_cache` | `ContactCache` | HashMap O(1) manifold lookup |
| `contact_cache` | `ContactManifold` | Up to 4 contact points |
| `interpolation` | `WorldSnapshot` | Physics state for interpolation |
| `debug_render` | `DebugDrawData` | Wireframe visualization |
| `debug_render` | `DebugDrawFlags` | Bitmask for what to draw |
| `profiling` | `PhysicsProfiler` | Per-stage timing |
| `profiling` | `StepStats` | Broadphase/narrowphase/solver times |
| `error` | `PhysicsError` | Typed physics error enum |

### Feature-Gated Modules

| Module | Feature | Description |
|--------|---------|-------------|
| `neural` | `neural` | Deterministic ternary neural controller (ALICE-ML) |
| `python` | `python` | PyO3 + NumPy zero-copy Python bindings |
| `ffi` | `ffi` | C FFI for Unity/UE5 game engines |
| `wasm` | `wasm` | WebAssembly bindings (wasm-bindgen) |
| `replay` | `replay` | Replay recording/playback via ALICE-DB |
| `db_bridge` | `replay` | Physics state persistence bridge |
| `analytics_bridge` | `analytics` | Simulation profiling bridge (ALICE-Analytics) |

### Python API (PyO3)

| Class / Function | Description |
|-----------------|-------------|
| `PhysicsWorld` | Python wrapper for physics simulation |
| `PhysicsWorld.add_dynamic_body(x, y, z, mass)` | Add dynamic body |
| `PhysicsWorld.add_static_body(x, y, z)` | Add static body |
| `PhysicsWorld.step(dt)` | Step with GIL release |
| `PhysicsWorld.step_n(dt, steps)` | Batch step for training |
| `PhysicsWorld.positions()` | NumPy (N,3) zero-copy |
| `PhysicsWorld.velocities()` | NumPy (N,3) zero-copy |
| `PhysicsWorld.states()` | NumPy (N,10) combined state |
| `PhysicsWorld.add_bodies_batch(array)` | Batch body creation from (N,4) |
| `PhysicsWorld.set_velocities_batch(array)` | Batch velocity update (N,3) |
| `PhysicsWorld.apply_impulses_batch(array)` | Batch impulse (M,4) |
| `PhysicsWorld.serialize_state()` | State to NumPy uint8 |
| `PhysicsWorld.deserialize_state(data)` | Restore from bytes |
| `DeterministicSimulation` | Netcode simulation wrapper |
| `DeterministicSimulation.advance_frame(inputs)` | Advance with player inputs |
| `DeterministicSimulation.save_snapshot()` | Save for rollback |
| `DeterministicSimulation.load_snapshot(frame)` | Restore snapshot |
| `encode_frame_input(...)` | Encode 20-byte input |
| `decode_frame_input(data)` | Decode 20-byte input |

### C FFI API

| Function | Description |
|----------|-------------|
| `alice_physics_world_create()` | Create physics world |
| `alice_physics_world_destroy(world)` | Destroy physics world |
| `alice_physics_world_step(world, dt)` | Step simulation |
| `alice_physics_body_add_dynamic(world, pos, mass)` | Add dynamic body |
| `alice_physics_body_add_static(world, pos)` | Add static body |
| `alice_physics_body_get_position(world, id, out)` | Get body position |
| `alice_physics_body_apply_impulse(world, id, impulse)` | Apply impulse |
| `alice_physics_state_serialize(world, len)` | Serialize state |
| `alice_physics_state_deserialize(world, data, len)` | Deserialize state |
| `alice_physics_state_free(data, len)` | Free serialized buffer |

## License

AGPL-3.0 - See [LICENSE](LICENSE) for details.

Copyright (C) 2024-2026 Moroya Sakamoto

## Acknowledgments

- XPBD: Müller et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
- GJK/EPA: Ericson, "Real-Time Collision Detection"
- Morton Codes: Morton, "A Computer Oriented Geodetic Data Base"
- CORDIC: Volder, "The CORDIC Trigonometric Computing Technique"
