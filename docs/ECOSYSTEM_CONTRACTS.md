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
| **[ALICE-SDF](https://github.com/ext-sakamoro/ALICE-SDF)** | Trait impl | `SdfField` | `v0.14` (current) |
| **ALICE-Bamboo** | Concrete-type consumption | `beam_stress`, `filament_db`, `warp_risk`, `thermal_stress`, `layer_adhesion` modules | `v0.14` (current) |
| **[ALICE-Kinematics](https://github.com/ext-sakamoro/ALICE-Kinematics)** | Reserved (future) | 8-byte Intent-packet bridge (L1 Physical Intent) | Post-1.0 (deferred) |

## 1. ALICE-TRT contract — `GpuSolverBridge`

**File**: `src/gpu_bridge.rs` (module `gpu_bridge`, not in the prelude)
**Cargo feature required**: `gpu-solver-bridge` (opt-in, no runtime cost when off).

### Frozen trait signature

The block below is the complete method list, compared with `src/` by
`scripts/docs_lint.py` (a default body is written `{ ... }`).

```rust
pub trait GpuSolverBridge {
    fn send_island(&mut self, positions: &[[Fix128; 3]], velocities: &[[Fix128; 3]]);
    fn dispatch_iterations(&mut self, iters: u32, dt: Fix128);
    fn recv_island(&self, positions: &mut [[Fix128; 3]], velocities: &mut [[Fix128; 3]]);
    fn assert_bit_exact_vs_cpu(&self, fixture: &DiffFixture) -> Result<(), GpuDivergence>;

    // contact-solve pipeline (added at v0.9.0; the defaults panic)
    fn send_contact_constraints(&mut self, _constraints: &[ContactConstraint]) { ... }
    fn send_body_state(&mut self, _positions: &[[Fix128; 3]], _inv_masses: &[Fix128]) { ... }
    fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) { ... }
    fn recv_contact_constraints(&self, _constraints: &mut [ContactConstraint]) { ... }
    fn recv_body_positions(&self, _positions: &mut [[Fix128; 3]]) { ... }

    // joint-solve pipeline (added at v0.12.0; the defaults panic)
    fn send_joints(&mut self, _joints: &[Joint]) { ... }
    fn send_body_rotations(&mut self, _rotations: &[[Fix128; 4]]) { ... }
    fn dispatch_joint_solve_iteration(&mut self, _dt: Fix128) { ... }
}
```

The four methods without a default are required. Every other method has a
default implementation that panics with "not implemented by this
GpuSolverBridge backend", so a backend that implements only the integrate +
distance stage still compiles and fails fast when a caller routes contact or
joint solve through it.

### Frozen types
- `Fix128` (module `math`, prelude)
- `ContactConstraint` (module `solver`, prelude)
- `Joint` (module `joint`, prelude)
- `DiffFixture`, `GpuDivergence` (module `gpu_bridge`)

Body state crosses the boundary as plain arrays (`[Fix128; 3]` per position or
velocity, `[Fix128; 4]` per rotation in `[x, y, z, w]` order), not as
`RigidBody` or `QuatFix`.

### Migration notes
- ALICE-TRT implements this trait for `TrtSolverAdapter` in `src/physics_bridge.rs`.

### Version pinning
```toml
# ALICE-TRT Cargo.toml: a path dependency, enabled by its `physics-solver` feature
[dependencies]
alice-physics = { path = "../ALICE-Physics", optional = true, default-features = false, features = ["std"] }

[features]
physics-solver = ["physics", "fix128-arithmetic", "alice-physics/gpu-solver-bridge"]
```

## 2. ALICE-SDF contract — `SdfField`

**File**: `src/sdf_collider.rs` (re-exported at the crate root and in the prelude)

### Frozen trait signature

Compared with `src/` by `scripts/docs_lint.py`.

```rust
pub trait SdfField: Send + Sync {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32;
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32);
    fn distance_and_normal(&self, x: f32, y: f32, z: f32) -> (f32, (f32, f32, f32)) { ... }
}
```

- `distance` is signed: positive outside, zero on the surface, negative inside.
- `normal` is the outward unit gradient.
- `distance_and_normal` defaults to `(self.distance(x, y, z), self.normal(x, y, z))`.
  An override must return the same distance as `distance`: callers such as
  `sdf_adaptive::AdaptiveSdfEvaluator` cache the distance it returns while
  `collide_point_sdf` takes contact depth from `distance`.

### Frozen types
- `f32` coordinates and distances. SDF evaluation is floating point; the
  `Fix128` ↔ `f32` conversion happens inside `SdfCollider`, not in the trait.

Every `SdfField` also implements `SdfQuery` (blanket impl), the borrowed,
non-`Send` query trait taken by `sdf_ccd::sphere_trace_sdf_field`.

### Migration notes
- ALICE-SDF implements `SdfField for CompiledSdfField` in `src/physics_bridge.rs`
  (feature `physics`) and does not override `distance_and_normal`.
- The `Send + Sync` bound is preserved (a collider owns its field as `Box<dyn SdfField>`).

### Version pinning
```toml
# ALICE-SDF Cargo.toml: from crates.io, behind its `physics` feature
[dependencies]
alice-physics = { version = "1.1", optional = true }

[features]
physics = ["dep:alice-physics"]
```

### Downstream test

`.github/workflows/downstream.yml` runs `scripts/downstream_check.sh` on every
change to `src/`, `Cargo.toml` or `rust-toolchain.toml`: ALICE-SDF (with this
checkout patched in for the crates.io dependency) and ALICE-LOL (path
dependency) are built and their `physics`-feature tests run against the changed
crate, and so is ALICE-TRT (path dependency, `physics-solver` feature), the
implementer of `GpuSolverBridge`. The version requirement of each on
alice-physics must accept this crate's version: a downstream that still
requires an older major fails with that requirement rather than testing the
published release.

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
alice-physics = "2"
```

## 4. ALICE-Kinematics contract — reserved (post-1.0)

**Status**: ALICE-Kinematics has no alice-physics dependency.

**Planned integration** (per the L1 Physical Intent roadmap):
- 8-byte Intent packet bridge: `IntentNode::Physical` → `PhysicsWorld` action.
- Requires new API on alice-physics: `PhysicsWorld::apply_intent(&mut self, packet: PhysicalIntent)`.
- Scheduled for **post-1.0** (target: 1.1 or 1.2 semver-minor).

The alice-physics 1.0 stable contract with ALICE-Kinematics is:
- **Nothing is frozen yet** — the integration point is a future addition.
- When Kinematics adopts alice-physics as its physics backend, the API additions will be non-breaking (semver-minor).

## Freeze semantics

### What "frozen" means for alice-physics 1.x

- **Trait signatures** (`GpuSolverBridge` and `SdfField`, exactly as listed in sections 1 and 2) — no method removal, no return-type changes, no argument-type changes, no bound tightening. Adding new methods with default impls is allowed (semver-minor).
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

- `scripts/docs_lint.py` compares each `pub trait` / `pub unsafe trait` written in a `rust` block of this document with the trait in `src/` (bounds and `unsafe`, the set of associated fns, types and consts, each signature including `const` / `async` / `unsafe` / `extern "ABI"`, associated type bounds and const types, and which items have a default); a mismatch fails CI.
- `.github/workflows/downstream.yml` builds ALICE-SDF, ALICE-LOL and ALICE-TRT against each change and runs their tests that reach alice-physics (see section 2); a downstream whose version requirement does not accept this crate's version fails with that requirement.
- `cargo semver-checks` — currently `continue-on-error: true`; will be promoted to hard-gate at the 1.0 release event (see `docs/ROADMAP.md` Item C).
- `docs/PUBLIC_API_SNAPSHOT.txt` — enforced by `public-api-diff` CI job. Contract-listed items are within this snapshot; any accidental change surfaces as a snapshot diff.
- `cargo public-api` output is regenerated on Mac aarch64 to match `macos-latest` CI runner (SIMD-item drift avoidance).

## Partner responsibilities

Each partner crate is expected to:

1. **Pin `alice-physics = "2"`** in their `Cargo.toml` (patch and minor updates auto-adopted).
2. **Run `cargo test`** on each `alice-physics` semver-minor update; report regressions.
3. **Coordinate breaking changes** — if a partner needs a currently-frozen item modified, open an issue to schedule the change for 2.0.

## Related docs

- [`docs/MIGRATION_0.x_TO_1.0.md`](./MIGRATION_0.x_TO_1.0.md) — downstream 0.x → 1.0 migration.
- [`docs/PUB_AUDIT_FINAL.md`](./PUB_AUDIT_FINAL.md) — audit campaign summary.
- [`docs/PUBLIC_API_SNAPSHOT.txt`](./PUBLIC_API_SNAPSHOT.txt) — full 1.0-candidate surface (19,406 items).
- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 milestone tracking.
