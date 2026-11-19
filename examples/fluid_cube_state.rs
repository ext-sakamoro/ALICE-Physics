//! Writes a stored fluid state of the cube scene of `tests/support/fluid_cube_fixture.rs` (cube
//! 2 h above the floor): frame 124, read by `tests/analytic_fluid_newton_fixture.rs`, or frame
//! 840, read by `tests/analytic_fluid_sdf_body.rs`, stepped from the lattice in frames of
//! 1/60 s; positions and velocities in the form described there.
//!
//! `cargo run --release --example fluid_cube_state -- 124 tests/fixtures/fluid_newton_cube_f124.bin`
//! `cargo run --release --example fluid_cube_state -- 840 tests/fixtures/fluid_cube_f840.bin`
//!
//! Prints the byte count and the SHA-256 of what it wrote.
//!
//! Author: Moroya Sakamoto

#[path = "../tests/support/fluid_cube_fixture.rs"]
mod fixture;

use sha2::{Digest, Sha256};

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: fluid_cube_state <124 | 840> <output file>";
    let frame: usize = args.next().and_then(|s| s.parse().ok()).expect(usage);
    let out = args.next().expect(usage);
    let stored = [fixture::NEWTON, fixture::SETTLED]
        .into_iter()
        .find(|s| s.frame == frame)
        .expect(usage);
    let sc = stored.run();
    let bytes = stored.encode(&sc.fluid);
    std::fs::write(&out, &bytes).expect("write the state");
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!(
        "{} particles, frame {frame}, {} bytes, sha256 {sha}",
        sc.n,
        bytes.len()
    );
}
