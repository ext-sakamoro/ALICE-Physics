# Public API Audit — Iteration 6 (P5 I/O + Serialization)

Ties into v1.0 roadmap **Item B** (Public API surface freeze). This sixth
audit pass targets the P5 module category flagged in
`PUB_AUDIT_ITERATION_2.md`: "I/O + serialization" (`scene_io`,
`collision_mesh_gen`, `debug_render`, `heatmap`).

**Date**: 2026-09-13
**Base**: alice-physics post-Iter 5 (5 module downgrades landed)
**Method**: identical to Iterations 1 – 5.

## Findings summary

### Modules surveyed (4 total)

| Module | pub items | prelude | example | downstream | internal cross-module | Verdict |
|--------|----------:|---------|---------|-----------:|----------------------|---------|
| `scene_io` | 8 | 7 items (`load_scene`, `load_scene_json`, `save_scene`, `save_scene_json`, `PhysicsScene`, `SerializedBody`, `SerializedJoint`) | 0 | 0 | 0 | **Keep all pub** |
| `collision_mesh_gen` | 5 | 5 items (all) | 0 | 0 | 0 | **Keep all pub** |
| `debug_render` | ~22 (types + methods + constants) | 4 items (`debug_draw_world`, `DebugColor`, `DebugDrawData`, `DebugDrawFlags`) | 0 | 0 | 0 | **Keep all pub** |
| `heatmap` | 6 | 6 items (all) | 0 | 0 | 0 | **Keep all pub** |

**Total pub items surveyed: ~41 across 4 modules.**
**Applied `pub → pub(crate)` reductions: 0 items.**

## Interpretation

P5 I/O + serialization is a "clean public surface" module category — every pub item in these 4 modules is:

1. **Explicitly re-exported in the prelude** — the intended public API commitment.
2. **Or leaked via a prelude-exported struct's public field** — `DebugDrawData.lines: Vec<DebugLine>` and `.points: Vec<DebugPoint>` leak `DebugLine`/`DebugPoint`; `PhysicsScene.config: PhysicsConfig` leaks `scene_io::PhysicsConfig`.

Neither category admits mechanical downgrade without changing the prelude API commitment. Downgrading the "leaked via field" cases (`DebugLine`, `DebugPoint`, `PhysicsConfig`) would require also demoting the fields on prelude-exported structs, which is a **design decision** not a mechanical audit outcome — these fields are debug-inspection APIs (users read `.lines`/`.points` to render debug data, or `.config` to inspect scene settings).

This is analogous to the `CachedContactPoint` deferral in Iterations 2 and 3: the field visibility question belongs in the `#[non_exhaustive]` design task for v1.0-rc.1.

Iteration 6 confirms that the P5 category is API-clean and needs no changes.

## Combined progress across Iterations 1 – 6

| Iteration | Modules | Pub items surveyed | Downgraded |
|-----------|---------|-------------------:|-----------:|
| 1 | 7 (recently-added feature modules) | 53 | 0 |
| 2 | 9 (P1 solver internals) | 127 | 4 |
| 3 | 3 (P2 math / BVH / spatial) | 108 | 8 |
| 4 | 4 (P3 CFD internals) | 58 | 31 |
| 5 | 5 (P4 structural internals) | 65 | 30 |
| 6 | 4 (P5 I/O + serialization) | ~41 | 0 |
| **Total** | **32** | **~452** | **73** |

**Snapshot progression**: 20,201 (Iter 1 baseline) → 19,956 (post-Iter 6) — **−245 items** across 6 iterations.

## Remaining deferred design decisions

The mechanical audit is now essentially complete. The remaining items require user-level or v1.0-rc.1 design decisions rather than surveys:

1. **`solver_tgs*` extension mechanism** (Iter 2 deferred) — 60+ items across 6 modules. Options:
   - **A**: Feature-gate under `tgs-hooks-extension` (default off) — preserves extension mechanism, opt-in users accept API instability.
   - **B**: Keep `pub`, add rustdoc "unstable extension" caveat — zero code change, honest signalling but committed at v1.0.
   - **C**: `pub(crate)` all items — cleanest v1.0 surface, easiest to re-add if demand emerges (zero current users).

2. **`CachedContactPoint` field visibility** (Iter 2/3 deferred) — v1.0-rc.1 `#[non_exhaustive]` treatment or accessor-based API. Same pattern applies to `DebugLine`/`DebugPoint` (Iter 6 finding) and `scene_io::PhysicsConfig` (Iter 6 finding).

3. **CFD cross-module helpers** (Iter 4 deferred) — Optional Iter 4b downgrade of `MacGrid`/`Grid3d`/`project_pressure`/`g2p_velocity`/`sample_*`/`trilinear_*`/`fast_sweeping_reinit`/`SMAGORINSKY_CS` + 2 fns. Same design call: expose CFD numerical primitives publicly, or hide behind higher-level `CfdSolver` API?

## v1.0 Item B status

**Iterations 1 – 6 complete**. The remaining Item B work is:

- User decision on `solver_tgs*` A/B/C (necessary before v1.0)
- v1.0-rc.1 design task for field-visibility items (`CachedContactPoint`, `DebugLine`/`DebugPoint`, `scene_io::PhysicsConfig`)
- Optional Iter 4b for CFD cross-module helpers

The mechanical "obvious internal leaks" cleanup phase of Item B is done.

## Related

- [`docs/PUB_AUDIT_ITERATION_1.md`](./PUB_AUDIT_ITERATION_1.md) — priority-modules (0 changes)
- [`docs/PUB_AUDIT_ITERATION_2.md`](./PUB_AUDIT_ITERATION_2.md) — P1 solver internals (4 changes)
- [`docs/PUB_AUDIT_ITERATION_3.md`](./PUB_AUDIT_ITERATION_3.md) — P2 math / BVH / spatial (8 changes)
- [`docs/PUB_AUDIT_ITERATION_4.md`](./PUB_AUDIT_ITERATION_4.md) — P3 CFD internals (31 changes)
- [`docs/PUB_AUDIT_ITERATION_5.md`](./PUB_AUDIT_ITERATION_5.md) — P4 structural internals (30 changes)
- [`docs/PUBLIC_API_SNAPSHOT.txt`](./PUBLIC_API_SNAPSHOT.txt) — 19,956-item baseline post-Iteration 6
- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 Item B tracking
