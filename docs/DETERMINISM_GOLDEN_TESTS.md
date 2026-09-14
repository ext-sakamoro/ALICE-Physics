# Determinism Golden Tests — v1.0 Item E

Framework for verifying **bit-exact cross-platform reproducibility** of
alice-physics simulations. Each golden test runs a fixed scenario for a
fixed step count, hashes the resulting body state, and compares against
a hard-coded golden hash recorded on Mac aarch64.

**Location**: `tests/determinism_golden.rs`.
**Roadmap Item**: E (Determinism CI 6-environment matrix).

## Design principles

1. **Zero-tolerance for bit drift**: two runs on any supported platform
   must produce byte-identical body state. Any divergence is a
   determinism bug, not "acceptable numerical variance".
2. **Fix128-only arithmetic**: scenarios avoid any code path that could
   accidentally reach `f32`/`f64` (SIMD paths are `#[cfg]`-gated and
   opt-in via the `simd` feature; default-feature tests use scalar-only
   fixed-point).
3. **Stable serialisation**: body state is written as
   `(pos.x.hi, pos.x.lo, pos.y.hi, ...)` — two `i64/u64` limbs per
   Fix128, little-endian bytes. No trailing padding, no version prefix.
4. **SHA-256 hash**: platform-independent (audited implementation via
   `sha2` dev-dep), collision-resistant beyond any practical fixture
   count.

## Fixtures (Phase 1 + Phase 2)

### Phase 1 fixtures (rigid body core)

| Scenario | Steps | dt | Bodies | Exercises |
|----------|------:|----|-------:|-----------|
| `freefall` | 60 | 1/60 s | 2 dynamic | gravity + Fix128 integration |
| `kinematic_drift` | 120 | 1/60 s | 1 dynamic | pure velocity integration, no gravity |
| `cascade` | 200 | 1/60 s | 5 dynamic + 1 static floor | contact resolution, damping, restitution |

### Phase 2 fixtures (extended subsystems)

| Scenario | Steps | dt | Subsystem | Exercises |
|----------|------:|----|-----------|-----------|
| `joint_pendulum` | 240 | 1/60 s | solver + joints | `DistanceConstraint` iteration + gravity |
| `cloth_drape` | 120 | 1/60 s | cloth (XPBD) | soft-body constraint iteration with pinned particles |
| `fluid_step` | 30 | 1/100 s | CFD (`CfdSolver` + `MacGrid`) | advection + diffusion + pressure projection |
| `sdf_ccd_glance` | 90 | 1/60 s | SDF collider + speculative CCD | `f32`-based SDF distance + normal, glancing pass |
| `trimesh_probe` | (5x5x5 samples) | — | trimesh collision | `TriMesh::collide_sphere` at 125 lattice points |

Total: **8 scenarios** + 1 meta test.

**Fixture hashes** (Mac aarch64, alice-physics `0.14.0-preview.8`):

| Scenario | SHA-256 |
|----------|---------|
| `freefall` | `49598cff32da429d198e5b281f9d46d89cb5dc6a430d368942b0e5249c779905` |
| `kinematic_drift` | `c79a3ee897fe95bde1bb5660ceb49552aacec0741ec4b8a2f5d46fd6c62ae099` |
| `cascade` | `0a9fb401dcc5fbf12ecd8ef36bd03caed7fb271b68d4035b764dbe24f6a47854` |
| `joint_pendulum` | `d1d51ea466dd4c55c7e9c220afca66004e7401fb53a4a2781e40be3e4ba9ef8f` |
| `cloth_drape` | `0df965eec802104c96168bf4eaf6e4096373343b39efe41bbfcdb3b3ef9d340c` |
| `fluid_step` | `20f4ba26edef79d64321fdd19c306e80d9e464707e86ba3a036769ff7b1cd3b7` |
| `sdf_ccd_glance` | `822813e3a278e960c30acb54ece430b0c51ddda325aa4ebb72009b26a6f62a4e` |
| `trimesh_probe` | `85fbff506a4cd8492163697b0d9cdb2252786d5171b0ff0c125c24a84dec485a` |

**Verified bit-exact identical hashes** on both Mac aarch64 (native) and `wasm32-wasip1` (via wasmtime) at Phase 2 landing time.

## CI matrix (Phase 2 complete)

Extended `.github/workflows/ci.yml` to **6 platform coverage**:

### Native test matrix (5 platforms, single `test` job)

- `macos-latest` (aarch64-apple-darwin) — primary dev platform, golden hash baseline.
- `macos-15-intel` (x86_64-apple-darwin) — Mac Intel coverage.
- `ubuntu-latest` (x86_64-unknown-linux-gnu) — Linux x86.
- `ubuntu-24.04-arm` (aarch64-unknown-linux-gnu) — Linux ARM (GA runner since 2025-01).
- `windows-latest` (x86_64-pc-windows-msvc) — Windows x86.

### WASM job (1 platform, separate `wasm-test` job)

- `wasm32-wasip1` target compiled on `ubuntu-latest`, executed via `wasmtime`
  (installed via official `install.sh`).
- Uses the exact same `tests/determinism_golden.rs` test file — WASI `std`
  shim provides stdio + libtest support without code changes.

Every PR runs all 6 platforms; a hash mismatch on any platform fails CI.

### Determinism guarantees confirmed

- **Fix128 arithmetic**: pure integer (`i64` + `u64` limbs) — bit-exact by
  language spec across every target.
- **IEEE 754 `f32` basic ops** (used by `ClosureSdf`): `+`, `-`, `*`, `/`,
  `sqrt` are spec-required bit-exact on every architecture that Rust
  supports (Rust reference §Behavior considered undefined).
  **Transcendentals** (`sin`, `cos`, `exp`, `ln`, `powf`, `cbrt`, `hypot`, …)
  are *not* IEEE-specified and differ between platform `libm`s. Since
  v1.1.0 every such call in the crate goes through `src/det_math.rs`
  (integer range reduction + fixed-order polynomials over the exact basic
  ops), and `clippy.toml` `disallowed-methods` rejects the `f32` / `f64`
  `libm` methods at CI. A user closure passed to `ClosureSdf` is outside
  that gate: call `det_math::*` inside it to keep the contact it feeds
  platform-independent.
- **`f32` / `f64` field modules** are pinned by
  `tests/determinism_golden_f32.rs` (13 scenarios, 29 modules) with the same
  record-on-aarch64 / verify-on-6-platforms workflow as this file.
- **No SIMD gates** in the default-feature test path (SIMD is opt-in via
  the `simd` feature; determinism_golden test builds without it).

## Adding a new fixture

1. Write a new `#[test] fn determinism_<name>()` in `tests/determinism_golden.rs`.
2. Set `const GOLDEN_<NAME>: &str = "TBD";` (placeholder).
3. Run `cargo test --test determinism_golden -- --nocapture` locally
   on Mac aarch64.
4. Copy the failure's "actual" hex into the const.
5. Re-run to confirm green.
6. Commit both the test and the fixture in the same PR.

## Regenerating an existing fixture (intentional simulation drift)

If a simulation change is intentional (e.g. algorithm improvement,
solver-parameter tuning, bugfix):

1. Update the algorithm.
2. Run `cargo test --test determinism_golden -- --nocapture`.
3. Copy the new "actual" hex into the corresponding `GOLDEN_*` const.
4. Verify the change is expected (peer review).
5. **Note the drift in the PR description and CHANGELOG** — pre-1.0 this
   is a semver-minor event; post-1.0 it requires v2.0 major bump unless
   it's a documented bugfix.

## Diagnosing a cross-platform failure

If a fixture matches on Mac aarch64 but fails on Linux x86 / Windows
(or WASM once Phase 2 lands):

1. **Identify the divergence step** — the failing scenario's step count
   is fixed; bisect by running the scenario with a smaller step count
   and hashing intermediate state to find where the two platforms
   diverge first.
2. **Common causes**:
   - Errant `f32` / `f64` operation slipping into a nominally Fix128
     code path (grep for `as f32`, `as f64`, `.to_f32()`, `.to_f64()`
     in the hot path).
   - SIMD gate that produces different bit patterns (grep for
     `#[cfg(target_arch)]` in `math/`, `bvh/`, `contact_cache/`).
   - Undefined behaviour (e.g. integer overflow with `-C
     overflow-checks=off`).
   - Order-dependent iteration over a `HashMap` — should use
     `BTreeMap` or explicit sorted iteration.
3. **Fix at the source, not at the fixture** — do NOT regenerate the
   fixture to "match" the broken platform.

## Post-1.0 stability

- Golden fixtures are frozen for 1.x. Any drift is a **breaking change**
  triggering a v2.0 major bump.
- Bugfix drifts are the sole exception; must be documented in
  `CHANGELOG.md` and `MIGRATION_1.x_TO_2.0.md`.
- The v1.0 release event will finalise the Phase 1 fixture hashes as
  the semver-locked reference.

## Related

- [`docs/ROADMAP.md`](./ROADMAP.md) — v1.0 Item E tracking.
- [`docs/ECOSYSTEM_CONTRACTS.md`](./ECOSYSTEM_CONTRACTS.md) — partner-facing API freeze (Item H).
- [`docs/MIGRATION_0.x_TO_1.0.md`](./MIGRATION_0.x_TO_1.0.md) — downstream migration guide (Item I).
- `tests/determinism_golden.rs` — the tests themselves.
