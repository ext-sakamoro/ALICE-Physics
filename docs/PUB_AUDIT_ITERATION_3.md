# Public API Audit — Iteration 3 (P2 Math / BVH / Spatial)

Ties into v1.0 roadmap **Item B** (Public API surface freeze). This third
audit pass targets the P2 module category flagged in `PUB_AUDIT_ITERATION_2.md`:
"math / low-level primitives" (`math`, `bvh`, `broadphase*`).

**Date**: 2026-09-13
**Base**: alice-physics post-`06b0619` (CI drift cleanup landed)
**Method**: identical to Iterations 1 + 2 — per-module `grep -nE "^pub (fn|struct|enum|const|trait|type)"`, cross-module usage grep in `src/`, downstream survey across `~/ALICE-Bamboo` `~/ALICE-Anima` `~/Yoin` `~/ALICE-LOL` `~/ALICE-Kinematics` `~/text-to-print-ios`, and example usage grep in `examples/`.

## Roadmap adjustment vs Iteration 2 plan

Iteration 2 listed `broadphase*` as a target under P2. **No such module exists** in the crate as a top-level `pub mod` — the actual broadphase-adjacent code lives in `bvh` (Linear BVH + `BroadphaseHybrid` skeleton) and `spatial` (hash grid). This audit covers the real 3 modules; `docs/ROADMAP.md` Iteration 4+ targets have been updated accordingly.

## Findings summary

### Modules surveyed (3 total)

| Module | pub items | prelude | example | downstream | internal cross-module | Verdict |
|--------|----------:|---------|---------|-----------:|----------------------|---------|
| `math` | 74 | 6 items (`Fix128`, `Vec3Fix`, `QuatFix`, `Mat3Fix`, `simd_width`, `SIMD_WIDTH`) | 9+ examples heavy usage | 4 refs (Bamboo + LOL, `Fix128` only) | `select_vec3` → `solver.rs` | **3 items → `pub(crate)`** |
| `bvh` | 27 | 3 items (`BvhNode`, `BvhPrimitive`, `LinearBvh`) | 0 | 0 | `LinearBvh` + `BvhPrimitive` → `solver.rs` + `trimesh.rs` + `query.rs` | **5 items → `pub(crate)`** |
| `spatial` | 7 | 1 item (`SpatialGrid`) | 0 | 0 | `SpatialGrid` → `bvh.rs` + `fluid.rs` | **Keep all pub** |

**Total pub items surveyed: 108 across 3 modules.**
**Applied `pub → pub(crate)` reductions: 8 items (2 modules, 2 commits).**

## Applied reductions

### `math` (commit `dc32236`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `Fix128::pack_pair` (line 361) | `pub` | `pub(crate)` | SIMD helper on `Fix128`, declared but **zero use anywhere** (not even inside `math.rs`). Only visible in x86_64 snapshot due to `#[cfg(all(feature = "simd", target_arch = "x86_64"))]` gate |
| `select_fix128` fn (line 1457) | `pub` | `pub(crate)` | Branchless select, **completely unused** anywhere in the crate |
| `select_vec3` fn (line 1476) | `pub` | `pub(crate)` | Branchless select for `Vec3Fix`; used 4× inside `solver.rs` XPBD constraint resolution, but zero external / example / test refs |

**Snapshot delta**: 2 lines removed from Mac aarch64 snapshot (`select_fix128` + `select_vec3`). `pack_pair` is x86_64-only gated so it's absent from Mac aarch64 baseline — the downgrade shows up only on x86_64 snapshot regeneration.

### `bvh` (commit `3f0d12c`)

| Item | Before | After | Rationale |
|------|--------|-------|-----------|
| `morton_code` fn (line 39) | `pub` | `pub(crate)` | Morton code helper, zero use even inside `bvh.rs` (leaf dead code) |
| `point_to_morton` fn (line 49) | `pub` | `pub(crate)` | `bvh.rs`-internal helper, 1 intra-module use only |
| `ESCAPE_NONE` const (line 100) | `pub` | `pub(crate)` | BVH traversal sentinel, zero external use |
| `BvhNode::MAX_PRIMS_PER_LEAF` const (line 132) | `pub` | `pub(crate)` | Inherent const on `BvhNode` (prelude type), zero external use; `leaf` fn's rustdoc reference to `Self::MAX_PRIMS_PER_LEAF` converted to plain "255" text |
| `BroadphaseHybrid` struct + impl (line 748+) | `pub` | `pub(crate)` | Skeleton stability-stub API; only exercised by unit tests within `bvh.rs`. `#[allow(dead_code)]` added on struct + impl block as intentional stub |

**Snapshot delta**: 22 lines removed (BroadphaseHybrid struct + 2 fields + 5 impl methods + 7 auto-impls + MAX_PRIMS_PER_LEAF ×3 occurrences + ESCAPE_NONE + morton_code + point_to_morton).

**Not downgraded**:
- `BvhStats` — return type of `pub fn LinearBvh::stats(&self) -> BvhStats`. Since `LinearBvh` is prelude-exported, `BvhStats` leaks via its public method signature and cannot be reduced without also changing `stats()`'s return type. **Keep pub**.

### `spatial` (no changes)

All 7 pub items are either in the prelude (`SpatialGrid` type) or are inherent methods of a prelude-exported type (`SpatialGrid::new`, `hash`, `insert`, `build`, `query_neighbors_into`, `clear`). Additionally, `SpatialGrid` is used across 2 internal modules (`bvh.rs` for `BroadphaseHybrid`'s dynamic grid slot, `fluid.rs` for SPH neighbor search). Keep all pub — this is a well-scoped public API.

### Combined snapshot impact

- **Before**: 20,179 public API items (`docs/PUBLIC_API_SNAPSHOT.txt`, commit `1afae8a`)
- **After**: 20,155 public API items (this iteration)
- **Delta**: −24 items (all clean removals, zero additions)

Regenerated via CI-canonical command: `cargo +nightly public-api --features "std,simd,parallel,ffi,gpu-solver-bridge" --simplified > docs/PUBLIC_API_SNAPSHOT.txt` on Mac aarch64 (matches CI runner platform).

## Iteration 2 deferred item resolution

### `contact_cache::CachedContactPoint` field visibility

**Decision: keep both `CachedContactPoint` and `ContactManifold.points` field as `pub`, defer to v1.0-rc.1 for `#[non_exhaustive]` treatment.**

Rationale:
- `ContactManifold.points: Vec<CachedContactPoint>` is the primary observability surface for warm-start impulse inspection (`lambda_n`, `lambda_t1`, `lambda_t2`, `age` fields exposed for debugging / analytics).
- Zero downstream usage today (`rg` result), so technically safe to downgrade.
- However, downgrading is not a mechanical audit outcome — it's an API design decision about whether alice-physics commits to warm-start observability as a public feature or hides it as internal implementation.
- Recommended path for v1.0-rc.1: mark both `ContactManifold` and `CachedContactPoint` with `#[non_exhaustive]` so future field additions don't break downstream, while keeping the observability surface documented and stable.
- Alternatively, add explicit accessor methods (`points_iter()`, `warm_start_impulses(idx)`) and reduce field visibility. This is a v1.0-rc.1 design task.

## Interpretation

Iterations 1–3 progress on Item B:

| Iteration | Modules | Pub items surveyed | Downgraded |
|-----------|---------|-------------------:|-----------:|
| 1 | 7 (recently-added feature modules) | 53 | 0 |
| 2 | 9 (P1 solver internals) | 127 | 4 |
| 3 | 3 (P2 math / BVH / spatial) | 108 | 8 |
| **Total** | **19** | **288** | **12** |

Pattern emerging: **most pub reductions come from module-internal helpers (const sentinels, private structs, helper fns) that leak into the surface only because the module itself is `pub`.** Prelude-exported types + their inherent methods are almost always intentionally public.

The v1.0 Item B work is well-scoped and near-complete for the "surface leak" category. Remaining strategic decisions:
- **`solver_tgs*`** (60+ items, Iter 2 deferred) — architectural decision A/B/C (user)
- **`CachedContactPoint` field visibility** (Iter 2/3 deferred) — v1.0-rc.1 `#[non_exhaustive]` treatment
- **P3–P5 audit** (Iter 4–6) — CFD internals, structural internals, I/O

Iteration 3's low-yield/high-signal ratio (108 items → 8 downgrades, 7.4%) suggests remaining iterations will find increasingly few "obvious internal leaks" as the crate's public API matures.

## Downstream survey command (reproducible)

```bash
rg -n 'alice_physics::(math|bvh|spatial)::[a-zA-Z_]' \
    ~/ALICE-Bamboo ~/ALICE-Anima ~/Yoin ~/ALICE-LOL ~/ALICE-Kinematics ~/text-to-print-ios \
    2>/dev/null | grep -v '/target/'
```

Result: **4 matches**, all `alice_physics::math::Fix128` — protects `Fix128` core API. Zero refs to `Vec3Fix`, `QuatFix`, `Mat3Fix`, or any `bvh` / `spatial` item.

## Revised audit strategy (Iteration 4+)

| Priority | Module category | Candidate modules |
|----------|-----------------|-------------------|
| **Iter 4** | CFD / fluid internals (P3) | `eulerian_grid` helper fns, `multiphase`, `interface_capture`, `turbulence` |
| **Iter 5** | Structural internals (P4) | `beam_stress`, `plastic`, `buckling`, `fatigue`, `creep_longterm` |
| **Iter 6** | I/O + serialization (P5) | `scene_io`, `collision_mesh_gen`, `debug_render`, `heatmap` |
| **User decision required (before v1.0)** | `solver_tgs*` extension mechanism | Options A/B/C from Iter 2 |
| **v1.0-rc.1 design task** | `CachedContactPoint` field visibility | `#[non_exhaustive]` or accessor-based API |

## Roadmap wiring

- v1.0 Item B progress: Iterations 1 + 2 + 3 complete (19 modules, 288 pub items, 12 downgrades).
- Snapshot integrity: `docs/PUBLIC_API_SNAPSHOT.txt` regenerated on Mac aarch64 (matches CI runner platform).
- CI enforcement: `.github/workflows/security-audit.yml` `public-api-diff` job continues to enforce diff-must-be-empty against this new 20,155-item baseline.

## Related

- [`docs/PUB_AUDIT_ITERATION_1.md`](./PUB_AUDIT_ITERATION_1.md) — priority-modules survey (0 changes)
- [`docs/PUB_AUDIT_ITERATION_2.md`](./PUB_AUDIT_ITERATION_2.md) — P1 solver internals (4 changes, `solver_tgs*` deferred)
- [`docs/PUBLIC_API_SNAPSHOT.txt`](./PUBLIC_API_SNAPSHOT.txt) — 20,155-item baseline post-Iteration 3
- [`docs/PUBLIC_API_SNAPSHOT.md`](./PUBLIC_API_SNAPSHOT.md) — regenerate + diff + roadmap wiring guide
- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 Item B tracking
