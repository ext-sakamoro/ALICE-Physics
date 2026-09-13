# Public API Audit — Iteration 5 (P4 Structural Internals)

Ties into v1.0 roadmap **Item B** (Public API surface freeze). This fifth
audit pass targets the P4 module category flagged in
`PUB_AUDIT_ITERATION_2.md`: "structural internals" (`beam_stress`,
`plastic`, `buckling`, `fatigue`, `creep_longterm`).

**Date**: 2026-09-13
**Base**: alice-physics post-`eaf245f` (Iteration 4 landed + CI green)
**Method**: identical to Iterations 1 – 4.

## Findings summary

### Modules surveyed (5 total)

| Module | pub items | prelude | example | downstream | internal cross-module | Verdict |
|--------|----------:|---------|---------|-----------:|----------------------|---------|
| `beam_stress` | 18 | 0 | 2 (`CrossSection`, `LoadCase`) | 3 (Bamboo: `BeamAnalysis`, `CrossSection`, `LoadCase`) | buckling, print_pipeline_solver, modal | **1 item → `pub(crate)`** |
| `plastic` | 17 | 0 | 0 | 0 | structural_solver (`radial_return_1d`, `NortonCreep`, `PlasticModel`, `PlasticState`) | **9 items → `pub(crate)`** |
| `buckling` | 9 | 0 | 0 | 0 | structural_solver (`analyze_column`, `BucklingRegime`, `ColumnBucklingReport`) | **6 items → `pub(crate)`** |
| `fatigue` | 11 | 0 | 0 | 0 | structural_solver (`SnCurve`, `miner_damage`, `SpectrumEntry`) | **7 items → `pub(crate)`** |
| `creep_longterm` | 10 | 0 | 0 | 0 | structural_solver (`FindleyParameters`, `predict_strain`) | **7 items → `pub(crate)`** |

**Total pub items surveyed: 65 across 5 modules.**
**Applied `pub → pub(crate)` reductions: 30 items (5 commits + module-level dead_code cleanup).**

## Applied reductions

### `beam_stress` (commit `4374397`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `euler_critical_load_n` | `pub` | `pub(crate)` | Standalone Euler helper; `buckling::critical_stress_mpa` is the primary path. `#[allow(dead_code)]` added |

**Snapshot delta**: 1 fn signature removed.

**Kept pub** (all 17 remaining items): `CrossSection` (+ 4 methods), `LoadCase` (+ 3 methods), `ColumnEndCondition` (+ `k_factor`), `BeamAnalysis` (+ 4 methods), `BeamReport` — used downstream (Bamboo) + examples + cross-module.

### `plastic` (commit `7a0e7f5`)

Bulk reduction of the reserved stress-tensor + plastic-step + hardening helper subsystem:

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `StressTensor` struct + 6 fields + `uniaxial_x`, `hydrostatic`, `von_mises` methods | `pub` | `pub(crate)` | 3D stress helper; zero external use |
| `PlasticStep` struct + 3 fields | `pub` | `pub(crate)` | Incremental update result (not leaked via return types) |
| `PlasticModel::with_hardening` | `pub` | `pub(crate)` | Unused builder method |
| `current_yield_mpa` | `pub` | `pub(crate)` | Unused |
| `NortonCreep::petg_room_temp`, `strain_rate_per_s` | `pub` | `pub(crate)` | Unused factory / accessor |

Module-level `#![allow(dead_code)]` + "Integration status" doc note.

**Kept pub**: `HardeningType` (leaks via `PlasticModel.hardening_type` pub field), `PlasticModel` + `from_fdm_material` (used by structural_solver), `PlasticState`, `NortonCreep` + `pla_room_temp` + `integrate`, `radial_return_1d`.

### `buckling` (commit `4052faa`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `radius_of_gyration_mm`, `slenderness_ratio`, `transition_slenderness`, `critical_stress_mpa` | `pub` | `pub(crate)` | Internal helpers called by `analyze_column` |
| `plate_buckling_mpa`, `snap_through_load_n` | `pub` | `pub(crate)` | Zero usage anywhere |

Module-level `#![allow(dead_code)]` + "Integration status" doc note.

**Kept pub**: `analyze_column`, `ColumnBucklingReport`, `BucklingRegime` (leaks via `ColumnBucklingReport.regime` pub field).

### `fatigue` (commit `cd87aef`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `SnCurve::steel_sus304`, `aluminum_a5052` | `pub` | `pub(crate)` | Unused factories |
| `INFINITE_LIFE` const | `pub` | `pub(crate)` | Internal sentinel |
| `cycles_to_failure` | `pub` | `pub(crate)` | Internal helper called by `miner_damage` |
| `stress_at_cycles` | `pub` | `pub(crate)` | Unused (Basquin inverse, future API) |
| `FatigueReport` struct + 3 fields + `analyze_spectrum` | `pub` | `pub(crate)` | Unused convenience wrapper + return type |

Module-level `#![allow(dead_code)]` + "Integration status" doc note.

**Kept pub**: `SnCurve`, `SnCurve::from_fdm_material`, `SpectrumEntry`, `miner_damage` (used by structural_solver).

### `creep_longterm` (commit `19d017f`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `FindleyParameters::petg_25c_moderate` | `pub` | `pub(crate)` | Unused factory |
| `FindleyParameters::strain_at` | `pub` | `pub(crate)` | Internal helper called by `predict_strain` |
| `WlfConstants` struct + 2 fields + `universal` | `pub` | `pub(crate)` | Internal WLF subsystem |
| `CREEP_FROZEN_AT` const | `pub` | `pub(crate)` | Unused sentinel |
| `wlf_shift_factor`, `effective_time_at_temp` | `pub` | `pub(crate)` | Internal helpers called by `predict_strain` |

Module-level `#![allow(dead_code)]` + "Integration status" doc note.

**Kept pub**: `FindleyParameters` (+ `pla_25c_moderate`), `predict_strain` (used by structural_solver).

### Combined snapshot impact (Iter 5 + Iter 6)

- **Before**: 20,064 public API items (`docs/PUBLIC_API_SNAPSHOT.txt`, commit `eaf245f`)
- **After**: 19,956 public API items (this iteration)
- **Delta**: −108 items (Iter 5 = 30 items → many auto-impl entries; Iter 6 = 0 items but same regeneration captures cumulative deltas across intermediate commits)

Regenerated via CI-canonical command on Mac aarch64.

## Related

- [`docs/PUB_AUDIT_ITERATION_1.md`](./PUB_AUDIT_ITERATION_1.md) — priority-modules survey (0 changes)
- [`docs/PUB_AUDIT_ITERATION_2.md`](./PUB_AUDIT_ITERATION_2.md) — P1 solver internals (4 changes)
- [`docs/PUB_AUDIT_ITERATION_3.md`](./PUB_AUDIT_ITERATION_3.md) — P2 math / BVH / spatial (8 changes)
- [`docs/PUB_AUDIT_ITERATION_4.md`](./PUB_AUDIT_ITERATION_4.md) — P3 CFD internals (31 changes)
- [`docs/PUB_AUDIT_ITERATION_6.md`](./PUB_AUDIT_ITERATION_6.md) — P5 I/O + serialization (0 changes)
- [`docs/PUBLIC_API_SNAPSHOT.txt`](./PUBLIC_API_SNAPSHOT.txt) — 19,956-item baseline
- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 Item B tracking
