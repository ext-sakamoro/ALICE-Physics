# alice-physics

The API of [`alice-physics-core`](https://crates.io/crates/alice-physics-core) under the
`alice_physics` paths.

The deterministic physics engine (128-bit fixed-point rigid bodies with bit-exact, replayable
results, typed observation and rollback, FEM, CFD and 3D-print analysis modules) lives in
`alice-physics-core`. This crate re-exports every public item of it at the same path and
forwards every feature under the same name, so code written against `alice_physics` builds
unchanged:

```sh
cargo add alice-physics
```

```rust
use alice_physics::prelude::*;
```

It defines nothing of its own: `alice_physics::PHYSICS_SEMANTICS_ID` is the engine's
identifier, and the types are the engine's types.

The C ABI libraries, the Python extension and the WebAssembly package are built from this
crate, so their file and module names are `alice_physics`:

```sh
cargo rustc -p alice-physics --lib --release --features ffi --crate-type cdylib
cargo rustc -p alice-physics --lib --release --features ffi --crate-type staticlib
maturin build --release -m crates/alice-physics/Cargo.toml --features python
```

Documentation, features and the determinism contract: see the
[repository README](https://github.com/ext-sakamoro/ALICE-Physics).

## License

`LicenseRef-Proprietary`. The text is [LICENSE-Proprietary](LICENSE-Proprietary).

Enquiries: https://extoria.co.jp/
