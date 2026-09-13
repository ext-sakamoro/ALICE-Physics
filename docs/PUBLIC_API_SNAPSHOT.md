# Public API Snapshot

This directory contains a snapshot of alice-physics' public API surface generated
by [cargo-public-api](https://crates.io/crates/cargo-public-api). It exists to
support the v1.0 roadmap Items **B** (public API surface freeze) and **C**
(cargo-semver-checks / cargo-public-api CI integration).

## Files

- [`PUBLIC_API_SNAPSHOT.txt`](PUBLIC_API_SNAPSHOT.txt) — text listing of every
  public item (module / struct / enum / fn / const / trait / type alias / impl)
  exposed by the crate under the recommended native feature set
  (`std,simd,parallel,ffi,gpu-solver-bridge`).

## Baseline

The current snapshot was generated at **v0.14.0-preview.5** (commit `7d5d214`,
2026-09-13) and totals **20,201 items**. This is the pre-freeze baseline; the
v1.0 roadmap Item B (public API surface freeze) will progressively move items
from `pub` to `pub(crate)` and mark others `#[deprecated]` before v1.0 stable.

## Regenerating

```bash
# Requires nightly toolchain for rustdoc JSON output
cargo +nightly public-api \
  --features "std,simd,parallel,ffi,gpu-solver-bridge" \
  --simplified \
  > docs/PUBLIC_API_SNAPSHOT.txt
```

`--simplified` removes disambiguating type qualifiers where they are unambiguous,
producing a shorter and more human-readable diff.

## Diff between versions

```bash
cargo +nightly public-api diff \
  --features "std,simd,parallel,ffi,gpu-solver-bridge" \
  0.14.0-preview.4 0.14.0-preview.5
```

Once v0.14.0 stable ships, the snapshot at that commit becomes the reference for
Item **C** (cargo-semver-checks CI gate). PRs that change the public API surface
will diff against this file so intentional additions vs accidental leaks stay
visible in code review.

## Roadmap wiring

See [`ROADMAP.md`](ROADMAP.md) — Item B (Public API surface freeze) and Item C
(cargo-semver-checks / cargo-public-api CI). The snapshot is the input side of
both items; the CI integration (Item C) will consume this file to detect
breakage.

## Warnings (documentation)

`cargo public-api` currently reports 8 pre-existing rustdoc warnings during
compilation (broken intra-doc links to `Joint::Ball`, `GpuSolverBridge`, and
private items linked from public documentation). These are documentation-only
issues that do not affect the public API surface listing. Fixing them is
tracked as a v0.14.0 stable follow-up.
