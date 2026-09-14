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

## Phase 1 fixtures (this iteration)

| Scenario | Steps | dt | Bodies | Constraints | Exercises |
|----------|------:|----|-------:|-------------|-----------|
| `freefall` | 60 | 1/60 s | 2 dynamic | none | gravity + Fix128 integration |
| `kinematic_drift` | 120 | 1/60 s | 1 dynamic | none | pure velocity integration, no gravity |
| `cascade` | 200 | 1/60 s | 5 dynamic + 1 static floor | contact | contact resolution, damping, restitution |

Total: 3 scenarios + 1 meta test (`hashing_helper_is_deterministic`).

**Fixture hashes** (Mac aarch64, alice-physics `0.14.0-preview.8`):
- `GOLDEN_FREEFALL       = 49598cff32da429d198e5b281f9d46d89cb5dc6a430d368942b0e5249c779905`
- `GOLDEN_KINEMATIC_DRIFT = c79a3ee897fe95bde1bb5660ceb49552aacec0741ec4b8a2f5d46fd6c62ae099`
- `GOLDEN_CASCADE        = 0a9fb401dcc5fbf12ecd8ef36bd03caed7fb271b68d4035b764dbe24f6a47854`

## CI matrix (Phase 1 → Phase 2)

### Phase 1 (this iteration, 4 platforms)

Extended `.github/workflows/ci.yml` test matrix from 3 → 4:

- `macos-latest` (aarch64-apple-darwin) — primary dev platform, golden hash source.
- `macos-15-intel` (x86_64-apple-darwin) — Mac Intel coverage.
- `ubuntu-latest` (x86_64-unknown-linux-gnu) — Linux x86 coverage.
- `windows-latest` (x86_64-pc-windows-msvc) — Windows x86 coverage.

Every PR runs all 4 platforms; a hash mismatch on any platform fails CI.

### Phase 2 (deferred, 2 more platforms)

- **Linux ARM** (`ubuntu-24.04-arm` runner, GitHub Actions GA since 2025).
  - Trivially added to matrix; deferred to Phase 2 for isolated verification.
- **WASM** (`wasm32-unknown-unknown` target + `wasmtime` runner).
  - Requires:
    - Cross-compile step (`cargo build --target wasm32-unknown-unknown`).
    - WASM runner container (`wasmtime` or `wasmer`).
    - Test harness that produces host-side hash from WASM execution
      (via WASI `stdout` capture or shared-memory export).
  - Deferred to Phase 2; higher risk (potential SIMD-related drift under
    wasm32 relaxed-SIMD if that feature ever enters ALICE-Physics).

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
