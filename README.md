# ALICE-Physics

**Deterministic 128-bit Fixed-Point Physics Engine** - Optimized Edition v0.2.0

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
| **no_std Compatible** | Works on embedded systems and WebAssembly |

## Optimizations ("黒焦げ" Edition)

ALICE-Physics v0.2.0 includes several performance optimizations:

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
┌─────────────────────────────────────────────────────────────────┐
│                    ALICE-Physics v0.2.0                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    math     │  │  collider   │  │   solver    │             │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤             │
│  │ Fix128      │  │ AABB        │  │ RigidBody   │             │
│  │ Vec3Fix     │  │ Sphere      │  │ Distance    │             │
│  │ QuatFix     │  │ Capsule     │  │ Contact     │             │
│  │ Mat3Fix     │  │ ConvexHull  │  │ Batching    │             │
│  │ CORDIC      │  │ GJK/EPA     │  │ XPBD        │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    bvh (Stackless)                          ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  Morton Codes → Linear BVH → Escape Pointers → Zero-alloc  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
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
cargo test --features "simd,parallel"
```

## Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `simd` | No | SIMD-accelerated Fix128/Vec3Fix operations (x86_64) |
| `parallel` | No | Constraint batching with Rayon (graph-colored parallel solving) |

```bash
# Enable SIMD optimizations
cargo build --release --features simd

# Enable parallel constraint solving
cargo build --release --features parallel

# Enable both
cargo build --release --features "simd,parallel"
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

## Test Results

```
v0.2.0 Test Summary:
  - 29 unit tests
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
