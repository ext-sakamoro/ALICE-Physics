# Public API Audit — Iteration 4 (P3 CFD Internals)

Ties into v1.0 roadmap **Item B** (Public API surface freeze). This fourth
audit pass targets the P3 module category flagged in
`PUB_AUDIT_ITERATION_2.md`: "CFD / fluid internals" (`eulerian_grid`,
`multiphase`, `interface_capture`, `turbulence`).

**Date**: 2026-09-13
**Base**: alice-physics post-`2d51f6c` (Iteration 3 landed + CI green)
**Method**: identical to Iterations 1 – 3.

## Findings summary

### Modules surveyed (4 total)

| Module | pub items | prelude | example | downstream | internal cross-module | Verdict |
|--------|----------:|---------|---------|-----------:|----------------------|---------|
| `eulerian_grid` | 16 | 0 | 0 | 0 | cfd_solver (MacGrid + 8 helpers) | **6 items → `pub(crate)`** |
| `multiphase` | 14 | 0 | 2 (`Grid3d`) | 0 | cfd_solver + surface_tension_csf + interface_capture | **4 items → `pub(crate)`** |
| `interface_capture` | 4 | 0 | 0 | 0 | cfd_solver (fast_sweeping_reinit only) | **3 items → `pub(crate)`** |
| `turbulence` | 24 | 0 | 0 | 0 | cfd_solver (3 items only: SMAGORINSKY_CS, smagorinsky_eddy_viscosity, strain_rate_magnitude) | **18 items → `pub(crate)`** |

**Total pub items surveyed: 58 across 4 modules.**
**Applied `pub → pub(crate)` reductions: 31 items (4 commits + 1 dead_code cleanup commit).**

## Scope decision

Iteration 4 focuses on **clear dead pub** — items with zero usage anywhere. Cross-module internal helpers (used by `cfd_solver.rs`, `surface_tension_csf.rs`) are kept `pub` in this iteration; their downgrade requires a separate design call about whether they represent intentional numerical primitives that advanced downstream CFD users might compose.

## Applied reductions

### `eulerian_grid` (commit `1d72044`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `project_pressure_red_black_gs` (line 198) | `pub` | `pub(crate)` | Internal delegate reached via `project_pressure` public wrapper; zero external refs |
| `project_pressure_jacobi` (line 299) | `pub` | `pub(crate)` | Session 3 I9 legacy benchmark; only unit-test usage |
| `project_pressure_bicgstab` (line 428) | `pub` | `pub(crate)` | Only unit-test usage |
| `BicgstabStats` (line 611) + 3 fields | `pub` | `pub(crate)` | Return type of `project_pressure_bicgstab`; downgraded together |
| `p2g_trilinear` (line 1007) | `pub` | `pub(crate)` | Zero usage anywhere |
| `p2g_nearest` (line 1018) | `pub` | `pub(crate)` | Zero usage anywhere (comment reference only) |

**Snapshot delta**: 21 lines removed (BicgstabStats struct + 3 fields + auto-impls + 5 fn signatures).

### `multiphase` (commit `72da7f2`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `advect_vof_uniform` (line 122) | `pub` | `pub(crate)` | Zero usage |
| `advect_vof_uniform_semi_lagrangian` (line 267) | `pub` | `pub(crate)` | Zero usage |
| `total_volume_vof` (line 315) | `pub` | `pub(crate)` | Zero usage |
| `reinitialize_level_set` (line 356) | `pub` | `pub(crate)` | Zero usage (only doc-comment reference from `interface_capture`); superseded by `interface_capture::fast_sweeping_reinit` |

**Snapshot delta**: 4 lines removed.

### `interface_capture` (commit `0612a51`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `plic_normal` (line 194) | `pub` | `pub(crate)` | PLIC helper; zero usage anywhere |
| `plic_plane_offset` (line 223) | `pub` | `pub(crate)` | PLIC helper; zero usage anywhere |
| `truncated_cube_volume` (line 297) | `pub` | `pub(crate)` | Deterministic Monte-Carlo-free helper; zero usage |

**Snapshot delta**: 3 lines removed.

### `turbulence` (commit `32acdce`)

Bulk reduction of the reserved RANS / wall-function API family. Only `SMAGORINSKY_CS`, `smagorinsky_eddy_viscosity`, and `strain_rate_magnitude` remain `pub` (wired into `cfd_solver.rs` Smagorinsky path).

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| 10 constants (`KE_C_MU`, `KE_SIGMA_K`, `KE_SIGMA_EPS`, `KE_C1_EPS`, `KE_C2_EPS`, `KW_BETA_STAR`, `KW_BETA`, `VON_KARMAN`, `LOG_LAW_B`, `Y_PLUS_TRANSITION`) | `pub` | `pub(crate)` | RANS + wall-function constants; zero external usage |
| `KEpsilonState` struct + 2 fields + `impl` block | `pub` | `pub(crate)` | k-ε RANS state; zero external usage |
| `KOmegaState` struct + 2 fields + `impl` block | `pub` | `pub(crate)` | k-ω RANS state; zero external usage |
| `KEpsilonState::eddy_viscosity`, `advance_k`, `advance_epsilon` | `pub` | `pub(crate)` | k-ε method surface |
| `KOmegaState::eddy_viscosity`, `from_k_epsilon` | `pub` | `pub(crate)` | k-ω method surface |
| `dynamic_smagorinsky_cs` | `pub` | `pub(crate)` | Session 3 I10 estimator; zero external usage |
| `y_plus`, `u_plus`, `wall_k_epsilon` | `pub` | `pub(crate)` | Session 3 I8 wall functions; zero external usage |

**Snapshot delta**: 63 lines removed (structs + fields + method fns + trait auto-impls + constants).

### Dead-code lint side effect

When `pub` → `pub(crate)` is applied to items that are only referenced from unit tests within the same module, the `dead_code` lint kicks in (public items are exempt from the lint, crate-internal items aren't). To keep the lint output clean without hiding genuine future dead code:

- **`turbulence.rs`**: module-level `#![allow(dead_code)]` with "Integration status" doc note (all downgraded items are reserved RANS/wall-function API awaiting `cfd_solver` integration).
- **`interface_capture.rs`**: item-level `#[allow(dead_code)]` on 3 PLIC helpers + "Integration status" doc note.
- **`eulerian_grid.rs`**: module-level `#![allow(dead_code)]` with "Integration status" doc note (Jacobi/BiCGStab pressure variants + P2G scatter operators reserved).
- **`multiphase.rs`**: module-level `#![allow(dead_code)]` with "Integration status" doc note (VOF advection variants + legacy reinit reserved; primary API `Grid3d` + `curvature_at` + `trilinear_*` remain active).

Trade-off: module-level `#![allow(dead_code)]` suppresses future dead-code detection in these modules. Alternative would be ~15 item-level annotations; the module-level attribute is preferred here because the modules are semantically "reserved API awaiting integration" as a whole.

### Combined snapshot impact

- **Before**: 20,155 public API items (`docs/PUBLIC_API_SNAPSHOT.txt`, commit `2d51f6c`)
- **After**: 20,064 public API items (this iteration)
- **Delta**: −91 items (all clean removals, zero additions)

Regenerated via CI-canonical command: `cargo +nightly public-api --features "std,simd,parallel,ffi,gpu-solver-bridge" --simplified > docs/PUBLIC_API_SNAPSHOT.txt` on Mac aarch64 (matches CI runner platform).

## Deferred (kept pub for now)

### `eulerian_grid` cross-module internal helpers

The following items remain `pub` because they are actively used by `cfd_solver.rs` but represent numerical primitives that advanced downstream CFD users might reasonably compose:

- `MacGrid` (leaks via `CfdSolver.grid` public field) + 8 methods (`new`, `u`, `v`, `w`, `pressure`, `cell_velocity`, `divergence`, `dx` etc.)
- `project_pressure` (public wrapper over `project_pressure_red_black_gs`)
- `g2p_velocity`
- `sample_u_range`, `sample_u_trilinear`, `sample_v_range`, `sample_v_trilinear`, `sample_w_range`, `sample_w_trilinear`

### `multiphase` cross-module internal helpers

- `Grid3d` (leaks via `CfdSolver.level_set` / `CfdSolver.temperature` public fields + 2 examples: `bfecc_advection_demo`, `buckmaster_alpoge_boussinesq_r2`)
- `trilinear_range`, `trilinear_sample` (used by cfd_solver)
- `initialize_level_set_sphere` (used by cfd_solver + surface_tension_csf)
- `curvature_at` (used by surface_tension_csf)

### `interface_capture` cross-module internal

- `fast_sweeping_reinit` (used by cfd_solver)

### `turbulence` cross-module internal

- `SMAGORINSKY_CS` const (used by cfd_solver)
- `smagorinsky_eddy_viscosity` fn (used by cfd_solver)
- `strain_rate_magnitude` fn (used by cfd_solver)

**Decision**: these could be `pub(crate)` in a follow-up iteration (Iter 4b or v1.0-rc.1), but require a design call about whether alice-physics commits to exposing CFD numerical primitives publicly.

## Interpretation

Iterations 1–4 progress on Item B:

| Iteration | Modules | Pub items surveyed | Downgraded |
|-----------|---------|-------------------:|-----------:|
| 1 | 7 (recently-added feature modules) | 53 | 0 |
| 2 | 9 (P1 solver internals) | 127 | 4 |
| 3 | 3 (P2 math / BVH / spatial) | 108 | 8 |
| 4 | 4 (P3 CFD internals) | 58 | 31 |
| **Total** | **23** | **346** | **43** |

**Yield trend**: 0% → 3% → 7% → 53% — Iteration 4 is a large outlier. Why? The `turbulence` module alone contained an entire reserved RANS + wall-function subsystem (18 items) that was pub-exposed at design time but never wired to the CFD solver. When a module ships as "future extension point" and no downstream ever picks it up, dead pub items accumulate. Bulk downgrade in a single iteration is efficient.

The pattern suggests remaining iterations (P4 structural, P5 I/O) may or may not find similar bulk-reduction opportunities depending on whether they contain analogous "future extension" subsystems.

## Downstream survey command (reproducible)

```bash
rg -n 'alice_physics::(eulerian_grid|multiphase|interface_capture|turbulence)::' \
    ~/ALICE-Bamboo ~/ALICE-Anima ~/Yoin ~/ALICE-LOL ~/ALICE-Kinematics ~/text-to-print-ios \
    2>/dev/null | grep -v '/target/'
```

Result: **0 matches** for internal modules, 2 example refs to `multiphase::Grid3d` (protected by prelude leak via `CfdSolver.level_set/temperature`).

## Revised audit strategy (Iteration 5+)

| Priority | Module category | Candidate modules |
|----------|-----------------|-------------------|
| **Iter 5** | Structural internals (P4) | `beam_stress`, `plastic`, `buckling`, `fatigue`, `creep_longterm` |
| **Iter 6** | I/O + serialization (P5) | `scene_io`, `collision_mesh_gen`, `debug_render`, `heatmap` |
| **Iter 4b (optional)** | CFD cross-module helpers | `project_pressure`, `g2p_velocity`, `sample_*` (eulerian_grid), `trilinear_range/sample`, `initialize_level_set_sphere`, `curvature_at` (multiphase), `fast_sweeping_reinit` (interface_capture), `SMAGORINSKY_CS`/`smagorinsky_eddy_viscosity`/`strain_rate_magnitude` (turbulence) |
| **User decision required (before v1.0)** | `solver_tgs*` extension mechanism | Options A/B/C from Iter 2 |
| **v1.0-rc.1 design task** | `CachedContactPoint` field visibility | `#[non_exhaustive]` or accessor-based API |

## Roadmap wiring

- v1.0 Item B progress: Iterations 1 + 2 + 3 + 4 complete (23 modules, 346 pub items surveyed, 43 downgrades).
- Snapshot integrity: `docs/PUBLIC_API_SNAPSHOT.txt` regenerated on Mac aarch64 (matches CI runner platform).
- CI enforcement: `.github/workflows/security-audit.yml` `public-api-diff` job continues to enforce diff-must-be-empty against the new 20,064-item baseline.

## Related

- [`docs/PUB_AUDIT_ITERATION_1.md`](./PUB_AUDIT_ITERATION_1.md) — priority-modules survey (0 changes)
- [`docs/PUB_AUDIT_ITERATION_2.md`](./PUB_AUDIT_ITERATION_2.md) — P1 solver internals (4 changes)
- [`docs/PUB_AUDIT_ITERATION_3.md`](./PUB_AUDIT_ITERATION_3.md) — P2 math / BVH / spatial (8 changes)
- [`docs/PUBLIC_API_SNAPSHOT.txt`](./PUBLIC_API_SNAPSHOT.txt) — 20,064-item baseline post-Iteration 4
- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 Item B tracking
