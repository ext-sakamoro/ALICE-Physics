//! How long the `Fix128` operations and a cloth step take, natively and in WebAssembly.
//!
//! The same program runs on both, so the two columns compare directly:
//!
//! ```text
//! cargo run --release --example fix128_timing
//! cargo build --release --example fix128_timing --target wasm32-wasip1
//! wasmtime target/wasm32-wasip1/release/examples/fix128_timing.wasm
//! ```
//!
//! Inputs are the sizes a cloth solver sees (distances near 1), and each operation folds its result
//! into the next input, so the compiler cannot remove the work. The final line of each row prints the
//! folded value: it must not change when an operation is made faster.

use alice_physics_core::cloth::Cloth;
use alice_physics_core::math::{Fix128, Vec3Fix};
use std::time::Instant;

fn row(name: &str, ops: u64, f: impl FnOnce() -> Fix128) {
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: the time per operation that this program prints"
    )]
    let start = Instant::now();
    let folded = f();
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: the time per operation that this program prints"
    )]
    let ns = start.elapsed().as_nanos() as f64 / ops as f64;
    println!(
        "{name:<12} {ns:>10.1} ns/op   folded {:016x}{:016x}",
        folded.hi as u64, folded.lo
    );
}

fn main() {
    const N: u64 = 200_000;
    let start = Fix128::from_ratio(5, 4);
    let step = Fix128::from_raw(0, 0x0000_1234_5678_9ABC);

    row("sqrt", N, || {
        let mut x = start;
        for _ in 0..N {
            x = x.sqrt() + start + step;
        }
        x
    });
    row("div", N, || {
        let mut x = start;
        for _ in 0..N {
            x = (x / (start + step)) + start;
        }
        x
    });
    row("mul", N, || {
        let mut x = start;
        for _ in 0..N {
            x = (x * Fix128::from_ratio(3, 4)) + start;
        }
        x
    });
    row("atan2", N / 10, || {
        let mut x = start;
        for _ in 0..N / 10 {
            x = Fix128::atan2(x, start) + start;
        }
        x
    });

    const STEPS: u64 = 120;
    let mut cloth = Cloth::new_grid(
        Vec3Fix::from_int(0, 2, 0),
        Fix128::ONE,
        Fix128::ONE,
        16,
        16,
        Fix128::from_ratio(1, 100),
    );
    cloth.pin_top_row(16);
    let dt = Fix128::from_ratio(1, 60);
    row("cloth step", STEPS, || {
        for _ in 0..STEPS {
            cloth.step(dt);
        }
        cloth.positions[255].y
    });
}
