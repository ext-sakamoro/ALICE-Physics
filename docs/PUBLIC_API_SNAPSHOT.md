# Public API Snapshot

This directory contains a snapshot of alice-physics' public API surface generated
by [cargo-public-api](https://crates.io/crates/cargo-public-api). It exists to
support the v1.0 roadmap Items **B** (public API surface freeze) and **C**
(cargo-semver-checks / cargo-public-api CI integration).

## Files

- [`PUBLIC_API_SNAPSHOT.txt`](PUBLIC_API_SNAPSHOT.txt) — text listing of every
  public item (module / struct / enum / fn / const / trait / type alias / impl)
  exposed by the crate under the recommended native feature set
  (`std,simd,parallel,ffi,gpu-solver-bridge`), for `aarch64-unknown-linux-gnu`.
- [`PUBLIC_API_SNAPSHOT.x86_64.txt`](PUBLIC_API_SNAPSHOT.x86_64.txt) — the same for
  `x86_64-unknown-linux-gnu`. It also lists the items under
  `#[cfg(target_arch = "x86_64")]` (the SIMD methods `Fix128::add_simd` / `sub_simd`,
  `Vec3Fix::cross_simd` / `dot_batch_4`), which the aarch64 file does not have.

Both are generated with `--target`, so any host compares both (rustdoc's JSON needs the
target's standard library, not a linker). `scripts/public_api_arch_check.py` checks that
both exist and that every architecture-only `pub` item of `src/` is in its file.

## Baseline

The current snapshot was generated at **v0.14.0-preview.5** (commit `7d5d214`,
2026-09-13) and totals **20,201 items**. This is the pre-freeze baseline; the
v1.0 roadmap Item B (public API surface freeze) will progressively move items
from `pub` to `pub(crate)` and mark others `#[deprecated]` before v1.0 stable.

## Regenerating

```bash
# the nightly and cargo-public-api versions security-audit.yml pins (another nightly prints
# another surface), with both targets:
#   rustup toolchain install nightly-2026-09-26
#   rustup target add --toolchain nightly-2026-09-26 aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu
#   cargo install cargo-public-api --locked --version 0.52.0
python3 scripts/ledgers.py --write   # every ledger, both snapshots among them
# or one snapshot: cargo +nightly-2026-09-26 public-api --target <target> \
#   --features "std,simd,parallel,ffi,gpu-solver-bridge" --simplified > <its file>
```

`--simplified` removes disambiguating type qualifiers where they are unambiguous,
producing a shorter and more human-readable diff.

## Diff between versions

```bash
cargo +nightly-2026-09-26 public-api diff \
  --features "std,simd,parallel,ffi,gpu-solver-bridge" \
  0.14.0-preview.4 0.14.0-preview.5
```

Once v0.14.0 stable ships, the snapshot at that commit becomes the reference for
Item **C** (cargo-semver-checks CI gate). PRs that change the public API surface
will diff against this file so intentional additions vs accidental leaks stay
visible in code review.

## Roadmap wiring

This file serves Item B (Public API surface freeze) and Item C
(cargo-semver-checks / cargo-public-api CI). The snapshot is the input side of
both items; the CI integration (Item C) will consume this file to detect
breakage.

## Warnings (documentation)

`cargo public-api` currently reports 8 pre-existing rustdoc warnings during
compilation (broken intra-doc links to `Joint::Ball`, `GpuSolverBridge`, and
private items linked from public documentation). These are documentation-only
issues that do not affect the public API surface listing. Fixing them is
tracked as a v0.14.0 stable follow-up.
