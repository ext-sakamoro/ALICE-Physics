# Public API Audit — Iteration 2 (P1 Solver Internals)

Ties into v1.0 roadmap **Item B** (Public API surface freeze). This second
audit pass targets the P1 module category flagged in `PUB_AUDIT_ITERATION_1.md`:
"internal solver internals" (`solver_tgs*`, `contact_cache`, `dynamic_bvh`).

**Date**: 2026-09-13
**Base**: alice-physics v0.14.0-preview.7 (commit `f1b4209`)
**Method**: identical to Iteration 1 — per-module `grep -nE "^pub (fn|struct|enum|const|trait|type)"`, cross-module usage grep in `src/`, downstream survey across `~/ALICE-Bamboo` `~/ALICE-Anima` `~/Yoin` `~/ALICE-LOL` `~/ALICE-Kinematics` `~/text-to-print-ios`, and example usage grep in `examples/`.

## Roadmap adjustment vs Iteration 1 plan

Iteration 1 listed `constraint` as a target under P1. **No such module exists** in the crate — the actual solver internals live in `solver.rs` plus the `solver_tgs*` family. This audit covers the real 9 modules; `docs/ROADMAP.md` Iteration 3+ targets have been updated accordingly.

## Findings summary

### Modules surveyed (9 total)

| Module | pub items | prelude | example | downstream | internal cross-module | Verdict |
|--------|----------:|---------|---------|-----------:|----------------------|---------|
| `solver` | 33 | 7 items | ✅ heavy (`basic_physics`, `ragdoll_demo`) | 0 | 20+ src files (RigidBody, PhysicsWorld, PhysicsConfig heavily used) | **Keep all pub** |
| `contact_cache` | 22 | 3 items (`BodyPairKey`, `ContactCache`, `ContactManifold`) | 0 | 0 | `solver.rs` uses `ContactCache` + `BodyPairKey` | **2 items → `pub(crate)`** (see below) |
| `dynamic_bvh` | 15 | 1 item (`DynamicAabbTree`) | 0 | 0 | 0 (only `lib.rs` prelude re-export) | **2 items → `pub(crate)`** (see below) |
| `solver_tgs` | 32 | 0 | 0 | 0 | 0 (only 1 doc-link in `ccd.rs`) | **Defer — architectural decision needed** |
| `solver_tgs_hooks` | 4 | 0 | 0 | 0 | 0 | **Defer** |
| `solver_tgs_hooks_6dof` | 5 | 0 | 0 | 0 | 0 | **Defer** |
| `solver_tgs_hooks_6dof_oriented` | 10 | 0 | 0 | 0 | 0 | **Defer** |
| `solver_tgs_hooks_6dof_scoped` | 3 fn | 0 | 0 | 0 | 0 | **Defer** |
| `solver_tgs_hooks_6dof_oriented_scoped` | 3 fn | 0 | 0 | 0 | 0 | **Defer** |

**Total pub items surveyed: 127 across 9 modules.**
**Applied `pub → pub(crate)` reductions: 4 items (2 modules, 2 commits).**

## Applied reductions

### `dynamic_bvh` (commit `b32f6d9`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `NULL_NODE: u32` const | `pub` | `pub(crate)` | Internal sentinel, 20+ intra-module uses, zero external refs |
| `DynamicNode` struct (+ 7 pub fields) | `pub` | `pub(crate)` | Internal tree node layout; callers only see the `u32` proxy IDs and `AABB` returned by `DynamicAabbTree` methods, never `DynamicNode` itself |

**Snapshot delta**: 20 lines removed (struct + fields + auto-impls) + 1 line removed (const) = 21 items from `dynamic_bvh` namespace.

### `contact_cache` (commit `65db351`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `MAX_MANIFOLD_POINTS: usize` const | `pub` | `pub(crate)` | Internal cap (currently 4), 4 intra-module uses, zero external refs; `ContactManifold.points` doc `` [`MAX_MANIFOLD_POINTS`] `` reference converted to plain "up to 4 per manifold" text |
| `tangent_frame` fn | `pub` | `pub(crate)` | Internal helper, 1 intra-module use; `sdf_manifold.rs` maintains its own private `build_tangent_frame` duplicate — consolidation is out of scope for this audit iteration |

**Snapshot delta**: 2 lines removed (1 const + 1 fn).

### Combined snapshot impact

- **Before**: 20,201 public API items (`docs/PUBLIC_API_SNAPSHOT.txt`, commit `f1b4209`)
- **After**: 20,179 public API items (this iteration, commit TBD)
- **Delta**: −22 items (all clean removals, zero additions)

Regenerated via CI-canonical command: `cargo +nightly public-api --features "std,simd,parallel,ffi,gpu-solver-bridge" --simplified > docs/PUBLIC_API_SNAPSHOT.txt` on Mac aarch64 (matches CI runner platform).

## Deferred items

### `contact_cache::CachedContactPoint`

`ContactManifold.points: Vec<CachedContactPoint>` is a `pub` field of a prelude-exported struct, so `CachedContactPoint` leaks even without direct re-export. Downgrading it to `pub(crate)` requires also demoting `ContactManifold.points` to `pub(crate)`, which changes the visibility of a prelude-exported struct's field.

- Downstream survey: 0 hits — technically safe.
- Ergonomic impact: any user reading `manifold.points[i].lambda_n` (warm-start impulse inspection) would break.
- **Recommendation**: revisit in Iteration 3 alongside a broader `ContactManifold` field-visibility audit.

### `solver_tgs*` family (6 modules, 60+ pub items)

These modules form a **generic TGS integrator extension mechanism**:
- `TgsHooks`, `BodyLike`, `ContactLike`, `JointLike`, `HasVelocity`, `ContactModifier` — user-implementable traits.
- `tgs_step<H: TgsHooks>`, `solve_island_isolated`, `solve_oriented_islands_parallel` — public entry-point fns.
- `TgsConfig`, `Pgs6DofConfig`, `Pgs6DofOrientedConfig` — solver configuration.
- `BodyRef`, `ContactRef`, `DistanceRef`, `SimpleBodyState`, `Body6DofState`, `Body6DofOrientedState` — adapter view types.

The surface is fully documented (rustdoc extension-point language) but has **zero actual users**:
- Zero downstream refs across all sibling repos.
- Zero example refs in `examples/`.
- Zero intra-crate refs (only 1 doc-link `[crate::solver_tgs::adaptive_substeps_for_ccd]` in `ccd.rs` line 309, which is not a compile-time dependency).

`solver.rs` implements a specialised, non-generic TGS solver directly for `PhysicsWorld`; the `solver_tgs*` family exists as a parallel generic path that no one currently drives.

**Three architectural options** for v1.0:

| Option | Effect | Trade-off |
|--------|--------|-----------|
| **A. Feature-gate under `tgs-hooks-extension` (default off)** | Preserves extension mechanism, hides from default surface, allows internal tests | Adds a feature flag; opt-in users accept API instability pre-v1.0 |
| **B. Keep `pub`, add rustdoc "unstable extension" caveat** | Zero code change, honest signalling | v1.0 stability freeze still applies unless we explicitly exempt |
| **C. `pub(crate)` all items (remove extension mechanism)** | Cleanest v1.0 surface, easiest to re-add if demand emerges | Breaking change to any hypothetical current user (survey says 0) |

**Decision deferred to user** — not a mechanical audit outcome; requires strategic call on whether ALICE-Physics commits to a generic TGS-hooks API in v1.0.

Applied strategy: leave all 60+ items as `pub` in this iteration; produce this findings doc so the choice is explicit at Iteration 3 planning time.

## Interpretation

Iteration 1's methodology (3-question test: prelude / example / downstream) applied to P1 solver internals yields two distinct outcomes:

1. **Implementation-detail internals** (const sentinels, private node structs, helper fns not documented as extension points) — clean `pub(crate)` candidates once verified they don't leak via existing prelude-exported types.

2. **Documented extension mechanisms with zero users** — the 3-question test flags them but doesn't tell us whether to (a) hide them, (b) commit to them, or (c) gate them. This is a strategic decision, not a mechanical one.

Iteration 2 delivers (1) and surfaces (2) for explicit user decision.

## Downstream survey command (reproducible)

```bash
rg -n 'alice_physics::(solver|solver_tgs|solver_tgs_hooks|contact_cache|dynamic_bvh)' \
    ~/ALICE-Bamboo ~/ALICE-Anima ~/Yoin ~/ALICE-LOL ~/ALICE-Kinematics ~/text-to-print-ios \
    2>/dev/null | grep -v '/target/'
```

Result: **0 matches** for the P1 module-internal namespaces. `alice_physics::solver::{RigidBody, PhysicsWorld, PhysicsConfig}` is used but that's via the prelude, and `solver` module is already justified.

## Revised audit strategy (Iteration 3+)

| Priority | Module category | Candidate modules |
|----------|-----------------|-------------------|
| **Iter 3** | Math / low-level primitives (P2 from Iter 1) | `math` (Fix128 helpers), `bvh` (internal BVH nodes), `broadphase*` |
| **Iter 3** | `contact_cache` field-visibility completion | `ContactManifold.points` (pub field visibility decision) |
| **Iter 4** | CFD / fluid internals (P3) | `eulerian_grid` helper fns, `multiphase`, `interface_capture`, `turbulence` |
| **Iter 5** | Structural internals (P4) | `beam_stress`, `plastic`, `buckling`, `fatigue`, `creep_longterm` |
| **Iter 6** | I/O + serialization (P5) | `scene_io`, `collision_mesh_gen`, `debug_render`, `heatmap` |
| **User decision required (before v1.0)** | `solver_tgs*` extension mechanism | Options A/B/C above |

## Roadmap wiring

- v1.0 Item B progress: Iterations 1 + 2 complete, 5 iterations planned (Iter 3–6 + a user-decision cycle for `solver_tgs*`).
- Snapshot integrity: `docs/PUBLIC_API_SNAPSHOT.txt` regenerated on Mac aarch64 (matches CI runner platform per `feedback_ci_public_api_platform_drift`).
- CI enforcement: `.github/workflows/security-audit.yml` `public-api-diff` job continues to enforce diff-must-be-empty against this new baseline.

## Related

- [`docs/PUB_AUDIT_ITERATION_1.md`](./PUB_AUDIT_ITERATION_1.md) — priority-modules survey (0 changes)
- [`docs/PUBLIC_API_SNAPSHOT.txt`](./PUBLIC_API_SNAPSHOT.txt) — 20,179-item baseline post-Iteration 2
- [`docs/PUBLIC_API_SNAPSHOT.md`](./PUBLIC_API_SNAPSHOT.md) — regenerate + diff + roadmap wiring guide
- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 Item B tracking
