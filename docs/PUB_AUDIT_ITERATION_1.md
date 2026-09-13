# Public API Audit — Iteration 1 (Priority Modules)

Ties into v1.0 roadmap **Item B** (Public API surface freeze). This first
audit pass surveys the modules previously flagged as "priority" in
`docs/ROADMAP.md` — the ones that landed most recently and were most likely
to have accidental `pub` exposure of internal helpers.

**Date**: 2026-09-13
**Base**: alice-physics v0.14.0-preview.7 (commit `f1b4209`)
**Method**: `grep -nE "^pub (fn|struct|enum|const)|^\s+pub (fn|struct|enum|const)"` per module, followed by manual review of each pub item's use-case fit.

## Findings

### Priority modules surveyed (7 total)

| Module | `pub` count | Verdict |
|--------|------------:|---------|
| `netcode_prediction` | 4 | ✅ **Keep as-is** — `PredictedInput`, `Snapshot`, `PredictionBuffer` (+ methods) and `reconcile` fn form a minimal, documented public surface. Every item is directly consumable by downstream game code (Yoin / SBR / ALICE-Bamboo). No accidental pub exposure. |
| `character_state` | 6 | ✅ **Keep as-is** — `CharacterState` enum + `CharacterStateContext` struct + `transition` fn form the intended FSM interface. All items documented, no impl helpers leaked. |
| `character` | 11 | ✅ **Keep as-is** — `CharacterConfig`, `PushImpulse`, `MoveResult`, `CharacterController` struct + methods (constructors + `move_and_slide` + `compute_push_impulses` + `apply_gravity`) form the canonical character controller API. All items match the public documentation contract. |
| `sdf_character` | 5 | ✅ **Keep as-is** — `MoveOutcome`, `SdfCharacter` struct + `new`, `move_and_slide`, `is_grounded` methods. Minimal, clean SDF-boundary character controller. |
| `sdf_sph` | 16 | 🟡 **Keep as-is (justified)** — includes helper fns `poly6`, `spiky_grad`, `viscosity_lap` (canonical Müller 2003 SPH kernels). These are documented as extension points for callers building custom kernels on top of the same primitives, so keeping them public is deliberate. |
| `sdf_wind_field` | 3 | ✅ **Keep as-is** — `SdfWindField` + `new` + `sample`. Minimal. |
| `sdf_fem_mesh` | 8 | ✅ **Keep as-is** — `SdfTetMesh` + `generate_marching_tets` + `refine_by_max_edge_length` etc. All items land after v0.14.0-preview.1 / preview.2 and are documented builder / refinement API. |

**Total pub items surveyed: 53 across 7 modules.**
**Recommended pub → pub(crate) changes: 0.**

## Interpretation

The "priority modules" flagged in ROADMAP were the recently-added Session 4
modules (v0.13.0) and v0.14.0 preview additions. These modules were designed
with a public API contract in mind and reviewed carefully at land time; they
do not have significant accidental `pub` exposure.

The v1.0 Item B (public API surface freeze) work therefore needs a broader
audit target. The 20,201 items in `docs/PUBLIC_API_SNAPSHOT.txt` are
concentrated in older / infrastructure modules where "just make it work"
pressure historically dominated over API hygiene.

## Revised audit strategy (Iteration 2+)

Target older / heavier modules for the next audit rounds:

| Priority | Module category | Candidate modules |
|----------|-----------------|-------------------|
| **P1** | Internal solver internals | `solver_tgs*` (multiple sub-files), `constraint`, `contact_cache`, `dynamic_bvh` |
| **P2** | Math / low-level primitives | `math` (Fix128 helpers), `bvh` (internal BVH nodes), `broadphase*` |
| **P3** | CFD / fluid internals | `eulerian_grid` (helper fns), `multiphase`, `interface_capture`, `turbulence` |
| **P4** | Structural internals | `beam_stress`, `plastic`, `buckling`, `fatigue`, `creep_longterm` |
| **P5** | I/O + serialization | `scene_io`, `collision_mesh_gen`, `debug_render`, `heatmap` |

Approach for each audit target:

1. Read module top-level and enumerate all `pub` items.
2. For each item, ask:
   - Is this item documented in the crate-level prelude?
   - Does at least one example in `examples/` use it?
   - Does downstream `ALICE-Bamboo` / `ALICE-Anima` / `SBR` code reference it?
3. If **none of the three**, mark as `pub(crate)` candidate.
4. Apply changes in small commits (one module per commit) so bisection is
   easy if downstream breaks.
5. Regenerate `docs/PUBLIC_API_SNAPSHOT.txt` after each audit round.
6. The CI job (`public-api-diff` in `security-audit.yml`) enforces that
   the snapshot stays consistent with reality.

## Downstream survey (deferred)

Before applying `pub(crate)` reductions in Iteration 2+, run:

```bash
# Search downstream repos for actual usage patterns
rg 'alice_physics::' ~/ALICE-Bamboo ~/ALICE-Anima ~/Yoin 2>/dev/null | grep -v '/target/'
```

Items referenced by downstream code must **not** be reduced without a
deprecation cycle (`#[deprecated]` alias for one minor version). Items with
zero downstream references are the safest reductions.

## Roadmap wiring

Iteration 1 completes the surveyed priority list without downstream-visible
changes. The v0.14.0 stable ship gate for Item B is **now dependent on
Iteration 2+** targeting the P1-P5 module categories above.

See [`ROADMAP.md`](ROADMAP.md) § v0.14.0 stable — 継続開発 — the "B" line
should now read "Iteration 2 P1 solver internals audit" as the next
concrete work item.
