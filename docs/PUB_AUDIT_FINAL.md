# Public API Audit — Final Iteration (v1.0 Item B Closure)

Ties into v1.0 roadmap **Item B** (Public API surface freeze). This final
audit iteration closes out the two design decisions deferred through
Iterations 2–6: the `solver_tgs*` extension mechanism and the
prelude-exported field-visibility hedges.

**Date**: 2026-09-13
**Base**: alice-physics `f1bea34` (J-4 bridge restoration, v0.14.0-preview.8)
**Method**: user-authorised recommendation execution (Options C + `#[non_exhaustive]`).

## Executed decisions

### 1. `solver_tgs*` extension mechanism — Option C (pub(crate) 全撤去)

Commit `ee3efb0`. Reduces 6 modules × 60+ items × auto-impl entries from the public surface.

**Rationale**:
- Zero downstream adoption of the extension mechanism (survey across `ALICE-Bamboo`, `ALICE-Anima`, `Yoin`, `ALICE-LOL`, `ALICE-Kinematics`, `text-to-print-ios` returned 0 refs at every audit iteration).
- Zero intra-crate compile-time dependency (only a rustdoc reference in `ccd.rs`).
- Zero example usage.
- Committing a 60+ item generic hooks-based TGS integrator to v1.0 API stability without a single caller commits us to non-breaking evolution of an untested API.
- If concrete demand emerges post-v1.0, re-exposure via a semver-minor bump is straightforward (adding `pub` items is non-breaking).

**Change scope**:
- 6 `pub mod` → `pub(crate) mod` in `src/lib.rs`.
- 60+ pub items (fn / struct / enum / const / trait / type + struct fields) → `pub(crate)` in all 6 module files.
- Module-level `#![allow(dead_code)]` + `#![allow(rustdoc::broken_intra_doc_links)]` + "Visibility" doc note in each.
- `src/ccd.rs` intra-doc link `[crate::solver_tgs::adaptive_substeps_for_ccd]` converted to plain-text reference.
- `src/plastic.rs::radial_return_1d` demoted `pub` → `pub(crate)` (fixed `private_interfaces` warning: it returned `PlasticStep` which was Iter 5-downgraded to `pub(crate)` while `radial_return_1d` remained `pub`).

**Snapshot delta**: ~550 items removed from Mac aarch64 baseline (struct + fields + methods + auto-impls for full 6-module family).

### 2. Prelude-leaked struct field-visibility hedge — `#[non_exhaustive]` on 7 structs

Commit `427d378`. Adds forward-compat marker to structs that leak internal types via public fields.

**Target structs**:
- `contact_cache::ContactManifold` (leaks `CachedContactPoint` via `points: Vec<CachedContactPoint>`)
- `contact_cache::CachedContactPoint` (fields carry warm-start impulse magnitudes)
- `debug_render::DebugDrawData` (leaks `DebugLine` / `DebugPoint` via `lines: Vec<DebugLine>` / `points: Vec<DebugPoint>`)
- `debug_render::DebugLine` (leaked field type)
- `debug_render::DebugPoint` (leaked field type)
- `scene_io::PhysicsScene` (leaks `PhysicsConfig` via `config: PhysicsConfig`)
- `scene_io::PhysicsConfig` (leaked field type)

**Effect**:
- External code cannot construct via struct literal (`ContactManifold { pair, points, .. }` etc.) — must use factory / builder methods.
- External code cannot exhaustively match — must use `{ .. }` wildcard.
- Public field read/write access is preserved (`manifold.points.iter()` still works).
- Future field additions become non-breaking (semver-minor).

**Trade-off**:
- v1.0 keeps observability API (users can still read `.points`, `.lines`, `.config`).
- Accessor-based API migration (`points_iter()`, `warm_start_impulse(idx)`, etc.) is a v2.0-scope task if demand emerges.

## v1.0 Item B — Progress across all iterations

| Iteration | Modules | Pub items | Downgraded | Snapshot Δ |
|-----------|--------:|----------:|-----------:|-----------:|
| 1 (pre-session) | 7 | 53 | 0 | 0 |
| 2 (P1 solver) | 9 | 127 | 4 | −22 |
| 3 (P2 math/BVH) | 3 | 108 | 8 | −24 |
| 4 (P3 CFD) | 4 | 58 | 31 | −91 |
| 5 (P4 structural) | 5 | 65 | 30 | −108 |
| 6 (P5 I/O) | 4 | ~41 | 0 | 0 |
| Final (solver_tgs* Option C + `#[non_exhaustive]`) | 6 + 7 structs | 60+ items + attr | 60+ | ~−550 |
| **Total** | **38** | **~512** | **133+** | **~−795** |

**Snapshot progression**: 20,201 (Iter 1 baseline) → **19,406** (post-Final) — cumulative reduction of ~795 items across the audit campaign.

## v1.0 Item B — Status: COMPLETE

The public API surface freeze has landed all mechanical audit work AND both deferred architectural decisions. What remains is:

- **Item C**: `cargo-semver-checks` hard-gate (remove `continue-on-error: true`) — unblocked, workflow YAML 1-line change.
- **Item E**: Determinism CI 6-environment matrix — estimated 2 weeks.
- **Item H**: Ecosystem contract freeze (5-partner trait) — estimated 1 week.
- **Item I**: Migration guide (0.x → 1.0) — estimated 3–5 days.
- **Item J**: crates.io publish continuation (preview.9+, then v1.0-rc.1) — user-triggered.

Optional post-v1.0 work:
- **Iter 4b**: CFD cross-module helpers pub(crate) (design call — should CFD numerical primitives be public composable API?).
- **Accessor migration** for `#[non_exhaustive]` structs (v2.0-scope).

## Related

- [`docs/PUB_AUDIT_ITERATION_1.md`](./PUB_AUDIT_ITERATION_1.md)
- [`docs/PUB_AUDIT_ITERATION_2.md`](./PUB_AUDIT_ITERATION_2.md)
- [`docs/PUB_AUDIT_ITERATION_3.md`](./PUB_AUDIT_ITERATION_3.md)
- [`docs/PUB_AUDIT_ITERATION_4.md`](./PUB_AUDIT_ITERATION_4.md)
- [`docs/PUB_AUDIT_ITERATION_5.md`](./PUB_AUDIT_ITERATION_5.md)
- [`docs/PUB_AUDIT_ITERATION_6.md`](./PUB_AUDIT_ITERATION_6.md)
- [`docs/PUBLIC_API_SNAPSHOT.txt`](./PUBLIC_API_SNAPSHOT.txt) — 19,406-item baseline post-Final
- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 tracking
