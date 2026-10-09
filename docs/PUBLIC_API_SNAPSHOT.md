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

## How it is kept current

The snapshot is the public API of version **2.0.0** plus the unreleased changes
on main. `scripts/land.py` regenerates it when it lands a commit, and both the
`public-api-diff` job of `.github/workflows/security-audit.yml` and
`scripts/preflight.sh` fail when the file differs from what the crate exports.
The version in the first sentence of this section is checked against
`Cargo.toml` by `scripts/version_sync.py` (`claim` in `scripts/version-sync.toml`),
so a release that bumps the crate version also has to update it here.

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
  <older-version> <newer-version>
```

A change to the public API surface shows up as a diff of
`PUBLIC_API_SNAPSHOT.txt` in the commit that makes it, so intentional additions
and accidental leaks stay visible in review. `cargo semver-checks` (the
`semver-checks` job of the same workflow) compares each commit with the one
before it.

## Warnings (documentation)

rustdoc warnings printed while `cargo public-api` builds the documentation do
not change the listing.
