# Ecosystem Contracts — alice-physics 1.0

Documents the API contracts between `alice-physics` and its partner crates in
the ALICE ecosystem. These contracts are **frozen for 1.0** — any change to
the listed items requires a semver-major bump on `alice-physics`.

**Scope**: partner-facing surface only. Internal-only API is out of scope
(see `PUB_AUDIT_FINAL.md` for the internal-surface audit).

## Partner overview

| Partner | Contract type | Primary integration point | Frozen at |
|---------|---------------|---------------------------|-----------|
| **[ALICE-TRT](https://github.com/ext-sakamoro/ALICE-TRT)** | Trait impl | `GpuSolverBridge` | `v0.12.0` (extended, current) |
| **[ALICE-SDF](https://github.com/Project-ALICE/ALICE-SDF)** | Trait impl | `SdfField` | `v0.14` (current) |
| **[ALICE-Bamboo](https://github.com/ext-sakamoro/ALICE-Bamboo)** | Concrete-type consumption | `beam_stress`, `filament_db`, `warp_risk`, `thermal_stress`, `layer_adhesion` modules | `v0.14` (current) |
| **[ALICE-Anima](https://github.com/ext-sakamoro/ALICE-Anima)** | Concrete-type consumption | `PhysicsWorld`, `RigidBody`, math prelude | `v0.14` (current) |
| **[ALICE-Kinematics](https://github.com/Project-ALICE/ALICE-Kinematics)** | Reserved (future) | 8-byte Intent-packet bridge (L1 Physical Intent) | Post-1.0 (deferred) |

## 1. ALICE-TRT contract — `GpuSolverBridge`

**File**: `src/gpu_bridge.rs`
**Cargo feature required**: `gpu-solver-bridge` (opt-in, no runtime cost when off).

### Frozen trait signature

```rust
pub trait GpuSolverBridge {
    fn send_bodies(&mut self, bodies: &[RigidBody]);
    fn send_contacts(&mut self, contacts: &[ContactConstraint]);
    fn send_distance_constraints(&mut self, joints: &[DistanceConstraint]);
    fn dispatch_pgs_contact_solve(&mut self, ...);
    fn dispatch_contact_solve_iteration(&mut self, ...);
    fn recv_bodies(&mut self, bodies: &mut [RigidBody]);

    // v0.12.0 extension methods (5 new methods added at physics v0.12.0):
    fn send_joints(&mut self, joints: &[Joint]);
    fn send_body_rotations(&mut self, rotations: &[QuatFix]);
    fn dispatch_joint_solve_iteration(&mut self, ...);
    // ...
}
```

### Frozen types
- `RigidBody` (module `solver`, prelude)
- `ContactConstraint` (module `solver`, prelude)
- `DistanceConstraint` (module `solver`, prelude)
- `QuatFix` (module `math`, prelude)
- `Joint` (module `joint`, prelude)

### Migration notes
- ALICE-TRT already implements this trait at `TrtSolverAdapter` in `physics_bridge.rs`.
- The `v0.12.0` extension methods are additions that do not break `v0.11.x` callers when they omit the new impl methods (default impls or `!()` sentinels — check the actual signatures).

### Version pinning
```toml
# ALICE-TRT Cargo.toml
[dependencies]
alice-physics = { version = "1", features = ["gpu-solver-bridge"] }
```

## 2. ALICE-SDF contract — `SdfField`

**File**: `src/sdf_collider.rs`

### Frozen trait signature

```rust
pub trait SdfField: Send + Sync {
    fn sample(&self, point: Vec3Fix) -> Fix128;
    fn sample_batch(&self, points: &[Vec3Fix], out: &mut [Fix128]) {
        // default impl provided
    }
    // ...
}
```

### Frozen types
- `Vec3Fix` (module `math`, prelude)
- `Fix128` (module `math`, prelude)

### Migration notes
- ALICE-SDF implements `SdfField for CompiledSdfField` in `physics_bridge.rs`.
- The `Send + Sync` bound is preserved.
- Batch sampling default impl is stable.

### Version pinning
```toml
# ALICE-SDF Cargo.toml
[dependencies]
alice-physics = "1"
```

## 3. ALICE-Bamboo contract — concrete types (7 modules)

**Files**: `src/beam_stress.rs`, `src/filament_db.rs`, `src/warp_risk.rs`, `src/thermal_stress.rs`, `src/layer_adhesion.rs`, `src/math.rs`

### Frozen types (Bamboo `src/safety.rs` import list)

| From `alice-physics` | Bamboo usage |
|----------------------|--------------|
| `beam_stress::BeamAnalysis` | Beam stress analysis wrapper |
| `beam_stress::CrossSection` (enum) | Rectangular / Circular / Hollow variants |
| `beam_stress::LoadCase` (enum) | CantileverEndPoint / SimplySupported etc. |
| `filament_db::FilamentDb` | Material property lookup by filament name |
| `filament_db::MaterialProperties` (struct) | Youngs modulus / yield / UTS / etc. |
| `layer_adhesion::EffectiveStrength` | Anisotropic Z/XY strength ratio |
| `layer_adhesion::PrintOrientation` | Vertical / Horizontal / Diagonal |
| `math::Fix128` | Universal deterministic scalar |
| `thermal_stress::analyze_thermal_stress` (fn) | Thermal stress computation |
| `thermal_stress::ThermalStressReport` | Analysis result |
| `warp_risk::WarpRiskCategory` (enum) | Warp risk classification |
| `warp_risk::{compute_warp_risk, WarpRiskConfig, WarpRiskReport}` | Warp risk analysis |

### Migration notes
- All items are prelude-exported or fully-qualified with stable module paths.
- No trait implementations required — Bamboo uses concrete types directly.
- `#[non_exhaustive]` was **not** applied to `CrossSection` / `LoadCase` / `MaterialProperties` / `WarpRiskCategory` in the audit (these are prelude-exported enums / structs that Bamboo consumes as pattern-match targets — adding `#[non_exhaustive]` would break existing pattern matches). Freeze commitment: these types will not gain new required variants / fields in 1.x.

### Version pinning
```toml
# ALICE-Bamboo Cargo.toml
[dependencies]
alice-physics = "1"
```

## 4. ALICE-Anima contract — concrete types (physics_sim.rs entry)

**Files**: `src/solver.rs`, `src/math.rs`

### Frozen types (Anima `anima-core/src/physics_sim.rs` import list)

| From `alice-physics` | Anima usage |
|----------------------|--------------|
| `PhysicsWorld` (struct) | Owned by `Anima::physics_sim` module |
| `PhysicsConfig` (type alias for `SolverConfig`) | Passed to `PhysicsWorld::new_with_config` |
| `RigidBody` (struct) | Sphere-based habitat body representation |
| `Fix128`, `Vec3Fix` (math prelude) | Universal deterministic types |

### Frozen methods on `PhysicsWorld`
- `new(config: PhysicsConfig) -> Self`
- `add_body(&mut self, body: RigidBody) -> usize`
- `step(&mut self, dt: Fix128)`
- `config` field (pub, exposed as `world.config.gravity` etc. by Anima)
- Body access via numeric IDs (e.g. `bodies_mut()` etc. — check the specific method Anima uses).

### Migration notes
- Anima uses `alice_physics::PhysicsWorld` as the physics kernel of its `physics_sim` module (Landing sub-phase 1b-β).
- `PhysicsWorld` is not `#[non_exhaustive]` (would break Anima's pattern of directly setting `world.config.gravity`).

### Version pinning
```toml
# ALICE-Anima Cargo.toml
[dependencies]
alice-physics = "1"
```

## 5. ALICE-Kinematics contract — reserved (post-1.0)

**Status**: no current alice-physics dependency (`rg 'alice_physics::' ~/ALICE-Kinematics` returns 0 hits).

**Planned integration** (per L1 Physical Intent roadmap, [[project_alice_lol_ir_roadmap]]):
- 8-byte Intent packet bridge: `IntentNode::Physical` → `PhysicsWorld` action.
- Requires new API on alice-physics: `PhysicsWorld::apply_intent(&mut self, packet: PhysicalIntent)`.
- Scheduled for **post-1.0** (target: 1.1 or 1.2 semver-minor).

The alice-physics 1.0 stable contract with ALICE-Kinematics is:
- **Nothing is frozen yet** — the integration point is a future addition.
- When Kinematics adopts alice-physics as its physics backend, the API additions will be non-breaking (semver-minor).

## Freeze semantics

### What "frozen" means for alice-physics 1.x

- **Trait signatures** (`GpuSolverBridge`, `SdfField`) — no method removal, no return-type changes, no argument-type changes, no bound tightening. Adding new methods with default impls is allowed (semver-minor).
- **Struct fields** listed in the contract — no removal, no visibility reduction, no type changes. Adding new fields to non-`#[non_exhaustive]` structs is NOT allowed until 2.0. Adding new fields to `#[non_exhaustive]` structs is allowed (semver-minor).
- **Enum variants** (`CrossSection`, `LoadCase`, `WarpRiskCategory`, etc.) — no removal, no reordering that changes discriminants. Adding new variants requires `#[non_exhaustive]` upgrade (currently NOT `#[non_exhaustive]` per Bamboo compatibility) — held for 2.0.
- **Free functions** (`analyze_thermal_stress`, `compute_warp_risk`) — no signature changes.
- **Type aliases** (`PhysicsConfig = SolverConfig`) — no rebinding.

### What can change in 1.x (semver-minor)
- New trait methods with default impls.
- New pub items (types, functions, constants).
- Performance improvements (algorithm swap-outs) that preserve behaviour.
- New feature flags.
- New `#[non_exhaustive]` structs with additive fields.
- New enum variants ONLY if the enum was already `#[non_exhaustive]` (currently: `BucklingRegime`, but Bamboo doesn't depend on it).

### What requires 2.0 (semver-major)
- Removing any listed item.
- Changing any listed signature.
- Adding a new required trait method (no default).
- Adding a required field to non-`#[non_exhaustive]` structs.
- Renaming any listed item.

## CI enforcement

- `cargo semver-checks` — currently `continue-on-error: true`; will be promoted to hard-gate at the 1.0 release event (see `docs/ROADMAP.md` Item C).
- `docs/PUBLIC_API_SNAPSHOT.txt` — enforced by `public-api-diff` CI job. Contract-listed items are within this snapshot; any accidental change surfaces as a snapshot diff.
- `cargo public-api` output is regenerated on Mac aarch64 to match `macos-latest` CI runner (SIMD-item drift avoidance).

## Partner responsibilities

Each partner crate is expected to:

1. **Pin `alice-physics = "1"`** in their `Cargo.toml` (patch and minor updates auto-adopted).
2. **Run `cargo test`** on each `alice-physics` semver-minor update; report regressions.
3. **Coordinate breaking changes** — if a partner needs a currently-frozen item modified, open an issue to schedule the change for 2.0.

## Related docs

- [`docs/MIGRATION_0.x_TO_1.0.md`](./MIGRATION_0.x_TO_1.0.md) — downstream 0.x → 1.0 migration.
- [`docs/PUB_AUDIT_FINAL.md`](./PUB_AUDIT_FINAL.md) — audit campaign summary.
- [`docs/PUBLIC_API_SNAPSHOT.txt`](./PUBLIC_API_SNAPSHOT.txt) — full 1.0-candidate surface (19,406 items).
- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 milestone tracking.
