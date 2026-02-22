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

## License

By contributing, you agree that your contributions will be licensed under AGPL-3.0.
