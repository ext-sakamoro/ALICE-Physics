# Migration Guide — alice-physics 0.x → 1.0

This guide catalogues the API changes between the `0.14.x` preview line and the upcoming stable `1.0` release, together with concrete migration steps for downstream crates.

**Target audience**: consumers of `alice-physics` (currently `ALICE-Bamboo`, `ALICE-Anima`, `ALICE-LOL`, `Yoin`, `ALICE-Kinematics`, `text-to-print-ios`, plus any future crates.io users).

**Version scope**:
- **From**: any `0.14.0-preview.X` (`0.14.0-preview.4` published to crates.io first, current `0.14.0-preview.8`+).
- **To**: `1.0.0` stable — expected via `1.0.0-rc.1` → `1.0.0-rc.2` → `1.0.0`.

## Table of contents

1. [Overview](#overview)
2. [Removed API surface](#removed-api-surface) (Iterations 2–4 + Final)
3. [Reserved-internal API (`pub` → `pub(crate)`)](#reserved-internal-api-pub--pubcrate) (Iterations 4–5)
4. [`#[non_exhaustive]` on prelude-exported structs](#non_exhaustive-on-prelude-exported-structs) (Final)
5. [Feature flag surface](#feature-flag-surface)
6. [Cargo.toml migration steps](#cargotoml-migration-steps)
7. [Post-1.0 stability guarantees](#post-10-stability-guarantees)

## Overview

alice-physics 1.0 tightens the public API surface based on the audit campaign documented in `PUB_AUDIT_ITERATION_{1..6}.md` + `PUB_AUDIT_FINAL.md`. The net effect for downstream code:

- **~795 items** removed from the public API surface baseline (`20,201 → 19,406`).
- The core physics API used by `ALICE-Bamboo` and other production consumers is **unchanged**: `Fix128`, `Vec3Fix`, `QuatFix`, `Mat3Fix`, `RigidBody`, `PhysicsWorld`, `PhysicsConfig`, `BeamAnalysis`, `CrossSection`, `LoadCase`, `Grid3d`, `MacGrid`, `LinearBvh`, `SpatialGrid`, `SnCurve`, `FindleyParameters`, `predict_strain`, etc.
- Removed items are **all previously unused by any downstream crate** at the time of audit (surveys returned 0 refs across 6 sibling repos).
- Structs that expose internal types via public fields are now `#[non_exhaustive]`, forward-compatible with future field additions.

If your project only uses the prelude (`use alice_physics::prelude::*;`) and never reaches into private submodules, **you likely need no changes**.

## Removed API surface

The following items were part of the `0.14.0-preview.X` public surface but are `pub(crate)` (crate-internal) in 1.0. If your code imported any of these, it will need to be adjusted.

### `dynamic_bvh` (Iteration 2)

| Item | Status | Migration |
|------|--------|-----------|
| `dynamic_bvh::NULL_NODE` | removed | Internal sentinel — should not be needed externally. Use `Option<u32>` in your own code. |
| `dynamic_bvh::DynamicNode` | removed | Interact with `DynamicAabbTree` via its `u32` proxy ID API (`insert`, `remove`, `update`, `query`, etc.) — the node layout is a private detail. |

### `contact_cache` (Iteration 2)

| Item | Status | Migration |
|------|--------|-----------|
| `contact_cache::MAX_MANIFOLD_POINTS` | removed | Internal cap (4 points). Do not depend on the exact number. |
| `contact_cache::tangent_frame` | removed | If you need an orthonormal tangent frame, implement locally (see `sdf_manifold::build_tangent_frame` for a reference impl). |

### `math` (Iteration 3)

| Item | Status | Migration |
|------|--------|-----------|
| `math::pack_pair` (Fix128 method, SIMD only) | removed | Internal SIMD helper. Fix128 arithmetic operators (`+`, `-`, `*`, `/`) already SIMD-accelerated on x86_64 where beneficial. |
| `math::select_fix128` | removed | Use `if condition { a } else { b }` — the branchless intent is achievable via LLVM optimisation. |
| `math::select_vec3` | removed | Same as above. |

### `bvh` (Iteration 3)

| Item | Status | Migration |
|------|--------|-----------|
| `bvh::morton_code` | removed | Internal helper for Morton-coded BVH build. Use `LinearBvh::build` as the public entry. |
| `bvh::point_to_morton` | removed | Same as above. |
| `bvh::ESCAPE_NONE` | removed | Internal traversal sentinel. |
| `BvhNode::MAX_PRIMS_PER_LEAF` | removed | Internal cap (255). |
| `bvh::BroadphaseHybrid` | removed | Skeleton stability-stub with zero adoption. Use `LinearBvh` + a separate hash grid, or wait for a future stable broadphase API. |

### `eulerian_grid` (Iteration 4)

| Item | Status | Migration |
|------|--------|-----------|
| `eulerian_grid::project_pressure_red_black_gs` | removed | Use `project_pressure` (which delegates to the red-black GS variant internally). |
| `eulerian_grid::project_pressure_jacobi` | removed | Session 3 legacy benchmark — no runtime API. |
| `eulerian_grid::project_pressure_bicgstab` | removed | Advanced iterative variant with unit-test-only usage. If needed post-1.0, request via a semver-minor bump. |
| `eulerian_grid::BicgstabStats` | removed | Return type of the removed BiCGStab variant. |
| `eulerian_grid::p2g_trilinear`, `p2g_nearest` | removed | Zero external usage. Implement locally or request via post-1.0 semver-minor. |

### `multiphase` (Iteration 4)

| Item | Status | Migration |
|------|--------|-----------|
| `multiphase::advect_vof_uniform` | removed | Zero usage. If needed, request re-exposure post-1.0. |
| `multiphase::advect_vof_uniform_semi_lagrangian` | removed | Same. |
| `multiphase::total_volume_vof` | removed | Implement locally: `field.data.iter().sum() * field.dx.pow(3)`. |
| `multiphase::reinitialize_level_set` | removed | Use `interface_capture::fast_sweeping_reinit` (the primary path). |

### `interface_capture` (Iteration 4)

| Item | Status | Migration |
|------|--------|-----------|
| `interface_capture::plic_normal` | removed | Zero usage. If needed, request post-1.0 exposure. |
| `interface_capture::plic_plane_offset` | removed | Same. |
| `interface_capture::truncated_cube_volume` | removed | Same. |

### `turbulence` (Iteration 4)

| Item | Status | Migration |
|------|--------|-----------|
| `turbulence::KE_C_MU`, `KE_SIGMA_K`, `KE_SIGMA_EPS`, `KE_C1_EPS`, `KE_C2_EPS`, `KW_BETA_STAR`, `KW_BETA` (k-ε / k-ω constants) | removed | Zero usage. Standard values (Launder & Spalding 1974, Wilcox 1988) are publicly available. |
| `turbulence::VON_KARMAN`, `LOG_LAW_B`, `Y_PLUS_TRANSITION` (wall-function constants) | removed | Same. |
| `turbulence::KEpsilonState`, `KOmegaState` (RANS state types) + methods | removed | k-ε / k-ω transport is not wired to `cfd_solver` yet. The `SMAGORINSKY_*` path remains available. |
| `turbulence::dynamic_smagorinsky_cs`, `y_plus`, `u_plus`, `wall_k_epsilon` | removed | Session 3 wall-function helpers with zero adoption. |

`SMAGORINSKY_CS`, `smagorinsky_eddy_viscosity`, and `strain_rate_magnitude` remain `pub` (used by `cfd_solver`).

### `beam_stress` (Iteration 5)

| Item | Status | Migration |
|------|--------|-----------|
| `beam_stress::euler_critical_load_n` | removed | Use `buckling::critical_stress_mpa` (primary path for Euler + Johnson regimes) followed by `stress × area`. |

### `plastic` (Iteration 5)

| Item | Status | Migration |
|------|--------|-----------|
| `plastic::StressTensor` + `uniaxial_x`, `hydrostatic`, `von_mises` | removed | Zero usage. If needed, implement locally (see the source for the analytic formulas). |
| `plastic::PlasticStep` (return type of `radial_return_1d`) | removed | See below — `radial_return_1d` is also `pub(crate)`; use `structural_solver`'s composite API instead. |
| `plastic::PlasticModel::with_hardening` | removed | The `hardening_type` field is public; set it directly or via `PlasticModel::from_fdm_material` (which defaults to `Isotropic`). |
| `plastic::current_yield_mpa` | removed | Zero usage. |
| `plastic::NortonCreep::petg_room_temp`, `strain_rate_per_s` | removed | Use `NortonCreep::pla_room_temp` or construct directly with `NortonCreep { a, n }`. |
| `plastic::radial_return_1d` | removed | Access via `structural_solver::step_stress` (or equivalent orchestrator) rather than calling directly. |

### `buckling` (Iteration 5)

| Item | Status | Migration |
|------|--------|-----------|
| `buckling::radius_of_gyration_mm`, `slenderness_ratio`, `transition_slenderness`, `critical_stress_mpa` | removed | Use `buckling::analyze_column` (composite public API) instead of the individual helpers. |
| `buckling::plate_buckling_mpa`, `snap_through_load_n` | removed | Zero usage. |

### `fatigue` (Iteration 5)

| Item | Status | Migration |
|------|--------|-----------|
| `fatigue::SnCurve::steel_sus304`, `aluminum_a5052` | removed | Use `SnCurve::from_fdm_material(&material)` or construct directly with `SnCurve { ultimate_tensile_mpa, endurance_stress_mpa, endurance_cycles, fatigue_exponent_m }`. |
| `fatigue::INFINITE_LIFE` (u64 sentinel) | removed | Compare `cycles` to `u64::MAX` if you need to detect the sentinel — but `miner_damage` handles it internally. |
| `fatigue::cycles_to_failure`, `stress_at_cycles` | removed | Use `miner_damage(&spectrum, &curve)` for the standard damage-accumulation flow. |
| `fatigue::FatigueReport` + `analyze_spectrum` | removed | Compute damage via `miner_damage` and wrap in your own report struct if needed. |

### `creep_longterm` (Iteration 5)

| Item | Status | Migration |
|------|--------|-----------|
| `creep_longterm::FindleyParameters::petg_25c_moderate`, `strain_at` | removed | Use `FindleyParameters::pla_25c_moderate` or construct directly. |
| `creep_longterm::WlfConstants` + `universal` | removed | Internal to `predict_strain`. |
| `creep_longterm::CREEP_FROZEN_AT` | removed | Internal sentinel. |
| `creep_longterm::wlf_shift_factor`, `effective_time_at_temp` | removed | Internal helpers. Use `predict_strain(&params, &material, t_hours, temp_c)` as the single public entry. |

### `solver_tgs*` family (Final iteration)

The entire six-module `solver_tgs` family is now `pub(crate)`. This affects:

- `solver_tgs` module (`BodyLike`, `ContactLike`, `JointLike`, `TgsConfig`, `AdaptiveSubStepConfig`, `HasVelocity`, `TgsHooks`, `tgs_step`, `build_islands`, `dispatch_islands`, `par_dispatch_islands`, `CachedImpulse`, `ImpulseCache`, `Island`, `UnionFind`, `BodyRef`, `ContactRef`, `DistanceRef`, ...).
- `solver_tgs_hooks` module (`SimpleBodyState`, `SimpleContact`, `PgsConfig`, `PgsHooks`).
- `solver_tgs_hooks_6dof` module (`Body6DofState`, `Contact6Dof`, `Pgs6DofConfig`, `Pgs6DofHooks`).
- `solver_tgs_hooks_6dof_oriented` module (all).
- `solver_tgs_hooks_6dof_scoped` module (all).
- `solver_tgs_hooks_6dof_oriented_scoped` module (all).

**Migration**:
- If your project uses `PhysicsWorld::step` (the primary solver entry), you are unaffected — the internal TGS solver in `crate::solver` is unchanged.
- If your project implements custom `TgsHooks` on top of your own body types, you were an unregistered user of an internal extension mechanism. Request re-exposure via a semver-minor bump post-1.0 with a concrete use case.

## Reserved-internal API (`pub` → `pub(crate)`)

Some `pub` items were reserved as extension points but had zero adoption. They are now `pub(crate)` and, where relevant, wrapped in `#[allow(dead_code)]` with "Integration status" doc notes. See the per-iteration audit docs for the full item list.

## `#[non_exhaustive]` on prelude-exported structs

The following prelude-exported structs are now `#[non_exhaustive]`:

- `contact_cache::ContactManifold`
- `contact_cache::CachedContactPoint`
- `debug_render::DebugDrawData`
- `debug_render::DebugLine`
- `debug_render::DebugPoint`
- `scene_io::PhysicsScene`
- `scene_io::PhysicsConfig`

**Effect on downstream code**:
- You **cannot** construct these via a struct literal (`ContactManifold { pair, points, normal, .. }`). Use factory / builder methods provided by the crate.
- You **cannot** exhaustively pattern-match them. Use `{ .. }` wildcard patterns:
  ```rust
  // Before
  match manifold {
      ContactManifold { pair, points, .. } => { /* ... */ }
  }
  // After — same code still works because `..` was already there.

  // Before
  let ContactManifold { pair, points, normal, friction, restitution } = manifold;
  // After — this now fails to compile. Rewrite as:
  let pair = manifold.pair;
  let points = &manifold.points;
  // ...
  ```
- Public field **read/write** access continues to work: `manifold.points.iter()`, `data.lines.push(...)`, etc.
- Future field additions on these structs are **non-breaking** — they can land in a semver-minor bump.

## Feature flag surface

The feature flag set is unchanged between 0.14 and 1.0:

| Feature | Purpose | Default | Notes |
|---------|---------|---------|-------|
| `std` | Enable `std`-dependent modules | ✅ | Turn off for `no_std` embedded targets. |
| `simd` | AVX2 SIMD on x86_64 | ❌ | No-op on non-x86_64. |
| `parallel` | Rayon-based parallel constraint solving | ❌ |  |
| `python` | PyO3 + NumPy bindings | ❌ |  |
| `ffi` | C FFI for Unity/UE5 | ❌ | Mutually exclusive with `wasm`. |
| `wasm` | WebAssembly bindings | ❌ | Mutually exclusive with `ffi`. |
| `gpu-solver-bridge` | `GpuSolverBridge` trait for external GPU offload | ❌ |  |
| `neural` | `alice-ml` deterministic neural controller bridge | ❌ | Restored in `0.14.0-preview.8` (J-4). |
| `replay` | `alice-db` state snapshot / playback bridge | ❌ | Restored in `0.14.0-preview.8`. |
| `analytics` | `alice-analytics` physics telemetry bridge | ❌ | Restored in `0.14.0-preview.8`. |

## Cargo.toml migration steps

```toml
[dependencies]
# Before
alice-physics = "0.14.0-preview.4"

# After (once 1.0.0 is published)
alice-physics = "1"
```

Downstream crates should:

1. **Update the version bound** to `"1"` (allowing patch and minor updates automatically).
2. **Run `cargo build`** to surface any broken imports of now-removed items.
3. **Fix broken imports** using the per-module tables above. Most fixes are one-line adjustments (switch to a public alternative, or remove the reference).
4. **Test the prelude-imported code path** — if you only use `use alice_physics::prelude::*;`, expect zero changes.
5. **Check `#[non_exhaustive]` sensitive code paths**:
   - `grep -R 'ContactManifold {' src/`
   - `grep -R 'DebugDrawData {' src/`
   - `grep -R 'PhysicsScene {' src/`
   - `grep -R 'PhysicsConfig {' src/`
   - `grep -R 'DebugLine {' src/`
   - `grep -R 'DebugPoint {' src/`
   - `grep -R 'CachedContactPoint {' src/`
   - Any struct-literal construction of these types outside the crate now fails; rewrite to use factory methods (e.g. `ContactManifold::new`, `PhysicsScene::default()`).

## Post-1.0 stability guarantees

Once 1.0 lands, alice-physics commits to:

- **No breaking changes to items in the current 1.0 public surface** without a `2.0` major bump.
- **`cargo semver-checks` CI hard-gate** — PRs that break semver are blocked (previously `continue-on-error: true` during the audit campaign).
- **6-environment determinism CI** — bit-exact golden test suite validates cross-platform (Item E, in-flight).
- **Ecosystem trait freeze** — `TrtSolver`, `SdfField`, `BambooReport`, `AnimaState`, `KinematicsChain` trait signatures frozen (Item H, in-flight).
- **Public API snapshot enforced** — `docs/PUBLIC_API_SNAPSHOT.txt` (currently 19,406 items) is CI-diffed on every PR.

## Reporting bugs / requesting re-exposure

If your project relied on a removed item and cannot find a suitable replacement:

- Open an issue at https://github.com/ext-sakamoro/ALICE-Physics/issues with:
  - The item you were using (module + name).
  - The concrete use case (what physics behaviour you need).
  - Whether a semver-minor re-exposure post-1.0 would meet your need.

Re-exposing a `pub(crate)` item to `pub` is a non-breaking change (semver-minor), so we can accommodate reasonable requests without waiting for `2.0`.

## Related docs

- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 milestone tracking.
- [`docs/PUB_AUDIT_FINAL.md`](./PUB_AUDIT_FINAL.md) — audit campaign summary + rationale.
- [`docs/PUB_AUDIT_ITERATION_1.md`](./PUB_AUDIT_ITERATION_1.md) – [`docs/PUB_AUDIT_ITERATION_6.md`](./PUB_AUDIT_ITERATION_6.md) — per-iteration audit findings.
- [`docs/PUBLIC_API_SNAPSHOT.txt`](./PUBLIC_API_SNAPSHOT.txt) — current 1.0-candidate public API surface (19,406 items).
