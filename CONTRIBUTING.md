# Contributing to ALICE-Physics

## Prerequisites

- Rust 1.70.0 or later (MSRV)
- `cargo fmt` and `cargo clippy` installed

## Development Workflow

```bash
# Build
cargo build

# Run all tests (unit + integration + doc)
cargo test

# Run with optional features
cargo test --features parallel
cargo test --features simd

# Verify no_std compatibility
cargo build --lib --no-default-features

# Lint
cargo clippy -- -W clippy::all

# Format
cargo fmt

# Benchmarks
cargo bench
```

## Code Style

- Run `cargo fmt` before committing
- All clippy warnings must be resolved (`-W clippy::all`)
- MSRV is 1.70.0 — do not use APIs from newer Rust editions
- Maintain `no_std` compatibility for modules not gated behind `#[cfg(feature = "std")]`
- `wasm` and `ffi` features are mutually exclusive

## Determinism

This engine guarantees bit-exact results across all platforms. When contributing:

- Never use `f32` or `f64` in simulation paths — use `Fix128`
- No floating-point trigonometry — use CORDIC functions (`Fix128::sin`, `Fix128::cos`)
- Use `DeterministicRng` instead of `rand` or system RNG
- Ensure fixed iteration counts in all algorithms
- Use stable sort with explicit comparators

## Testing

- Unit tests go in `#[cfg(test)] mod tests` within each source file
- Integration tests go in `tests/integration_physics.rs`
- Doc tests with ```` ```rust ```` blocks are encouraged for public APIs
- Aim for at least one test per public function

## Dependencies

`Cargo.lock` is committed. CI builds every job from it, and the oracle ledger
(`docs/oracle-status.md`) records the versions it holds, so a dependency that
publishes a new release changes nothing until the lock is updated.

- Update a dependency with `cargo update -p <crate>` (or edit `Cargo.toml` and
  run `cargo update`) and commit `Cargo.toml` and `Cargo.lock` together. CI
  rejects a `Cargo.toml` whose lock is out of date (`cargo metadata --locked`).
- Dependabot opens weekly pull requests that update the lock. Their changes
  are re-committed locally with the project author rather than merged on
  GitHub, and land with `scripts/land.py`, which regenerates the ledger.
- The lock only fixes this repository's builds. A project that depends on
  `alice-physics` resolves its own versions: Cargo ignores the lock file of a
  dependency, including the copy inside the published package.
- The lock is generated with `resolver = "3"`, which picks versions that support
  `rust-version`, so the MSRV job builds from the same lock.

## License

By contributing, you agree that your contributions will be licensed under AGPL-3.0.
