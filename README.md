# ALICE-Physics

**Deterministic 128-bit Fixed-Point Physics Engine** - v0.3.0

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
| **5 Joint Types** | Ball, Hinge, Fixed, Slider, Spring with angle limits and motors |
| **Raycasting** | Ray and shape casting against spheres, AABBs, capsules, planes |
| **CCD** | Continuous collision detection (TOI, conservative advancement) |
| **Sleep/Islands** | Automatic sleep with Union-Find island management |
| **Triangle Mesh** | BVH-accelerated triangle mesh collision (Moller-Trumbore) |
| **Height Field** | Grid terrain with bilinear interpolation |
| **Articulated Bodies** | Multi-joint chains, ragdolls, robotic arms with FK propagation |
| **Force Fields** | Wind, gravity wells, drag, buoyancy, vortex |
| **PD Controllers** | 1D/3D proportional-derivative joint motors |
| **Collision Filtering** | Layer/mask bitmask system with collision groups |
| **Deterministic RNG** | PCG-XSH-RR pseudo-random number generator |
| **Contact Events** | Begin/Persist/End contact and trigger event tracking |
| **no_std Compatible** | Works on embedded systems and WebAssembly |

## Optimizations ("黒焦げ" Edition)

ALICE-Physics includes several performance optimizations:

### 1. Stackless BVH Traversal

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

### 2. SIMD Acceleration (optional)

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

### 3. Constraint Batching (optional)

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
┌──────────────────────────────────────────────────────────────────────┐
│                       ALICE-Physics v0.3.0                           │
├──────────────────────────────────────────────────────────────────────┤
│  Core Layer                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  math    │ │ collider │ │  solver  │ │   bvh    │ │sdf_colldr│  │
│  │ Fix128   │ │ AABB     │ │ RigidBody│ │ Morton   │ │ SdfField │  │
│  │ Vec3Fix  │ │ Sphere   │ │ XPBD     │ │ Stackless│ │ Gradient │  │
│  │ QuatFix  │ │ Capsule  │ │ Batching │ │ Zero-    │ │ Early-out│  │
│  │ CORDIC   │ │ GJK/EPA  │ │ Rollback │ │  alloc   │ │          │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                      │
│  Constraint & Dynamics Layer                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  joint   │ │  motor   │ │articulatn│ │  force   │ │ sleeping │  │
│  │ Ball     │ │ PD 1D/3D │ │ Ragdoll  │ │ Wind     │ │ Islands  │  │
│  │ Hinge    │ │ Position │ │ FK Chain │ │ Gravity  │ │ Union-   │  │
│  │ Fixed    │ │ Velocity │ │ Robotic  │ │ Buoyancy │ │  Find    │  │
│  │ Slider   │ │ Max Torq │ │ 12-body  │ │ Drag     │ │ Auto     │  │
│  │ Spring   │ │          │ │          │ │ Vortex   │ │  Sleep   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                      │
│  Query & Collision Layer                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ raycast  │ │   ccd    │ │ trimesh  │ │heightfld │ │  filter  │  │
│  │ Sphere   │ │ TOI      │ │ Triangle │ │ Bilinear │ │ Layer    │  │
│  │ AABB     │ │ Conserv. │ │ BVH-accel│ │ Normal   │ │ Mask     │  │
│  │ Capsule  │ │ Advance  │ │ Moller-  │ │ Sphere   │ │ Group    │  │
│  │ Plane    │ │ Swept    │ │ Trumbore │ │ Collide  │ │ Bidirect │  │
│  │ Sweep    │ │ AABB     │ │ Closest  │ │ Signed   │ │          │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│                                                                      │
│  Utility Layer                                                       │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────────────────────────┐  │
│  │   rng    │ │  event   │ │      neural (ALICE-ML × Physics)    │  │
│  │ PCG-XSH  │ │ Begin    │ │ Ternary {-1,0,+1} → Fix128 Add/Sub │  │
│  │ Fix128   │ │ Persist  │ │ Deterministic AI                    │  │
│  │ Direction│ │ End      │ │                                     │  │
│  └──────────┘ └──────────┘ └─────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
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

**Constraints:**
- `DistanceConstraint`: Fixed distance between anchor points
- `ContactConstraint`: Collision response with friction/restitution

**Methods:**
- `RigidBody::new(position, mass)` - Create dynamic body
- `RigidBody::new_dynamic(position, mass)` - Alias for new
- `RigidBody::new_static(position)` - Create immovable body

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

### `joint` - 5 Joint Types

| Type | Description |
|------|-------------|
| `BallJoint` | Spherical joint (3 rotational DOF) |
| `HingeJoint` | Revolute joint (1 rotational DOF) with angle limits |
| `FixedJoint` | Weld joint (0 DOF) |
| `SliderJoint` | Prismatic joint (1 translational DOF) with limits |
| `SpringJoint` | Damped spring constraint |

```rust
use alice_physics::joint::*;
let hinge = HingeJoint::new(body_a, body_b, anchor_a, anchor_b, axis_a, axis_b)
    .with_limits(-Fix128::HALF_PI, Fix128::HALF_PI);  // radians
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
| `time_of_impact_spheres` | Sphere-sphere TOI via quadratic |
| `conservative_advancement` | TOI via iterative safe stepping |
| `swept_aabb` | Swept AABB bounding volume |

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

```bash
# Enable SIMD optimizations
cargo build --release --features simd

# Enable parallel constraint solving
cargo build --release --features parallel

# Enable both
cargo build --release --features "simd,parallel"

# Enable neural controller (requires ALICE-ML)
cargo build --release --features neural
```

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
| L2 | Zero-Copy NumPy (`into_pyarray`) | No memcpy for bulk positions/velocities |
| L3 | Batch API (`step_n`, `positions`) | FFI amortization |
| L4 | Rust backend (Fix128, XPBD, BVH) | Hardware-speed simulation |

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
v0.3.0 Test Summary:
  - 121 unit tests across 19 modules (including netcode, replay)
  - 1 doc test
  - All feature combinations pass
  - Zero warnings
```

## License

AGPL-3.0 - See [LICENSE](LICENSE) for details.

Copyright (C) 2024 Moroya Sakamoto

## Acknowledgments

- XPBD: Müller et al., "XPBD: Position-Based Simulation of Compliant Constrained Dynamics"
- GJK/EPA: Ericson, "Real-Time Collision Detection"
- Morton Codes: Morton, "A Computer Oriented Geodetic Data Base"
- CORDIC: Volder, "The CORDIC Trigonometric Computing Technique"
