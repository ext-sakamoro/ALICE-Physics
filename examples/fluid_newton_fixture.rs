//! Writes the stored fluid state read by `tests/analytic_fluid_newton_fixture.rs`: the cube scene
//! of `tests/support/fluid_cube_fixture.rs` stepped 124 frames of 1/60 s from the lattice, its
//! positions and velocities in the form described there.
//!
//! `cargo run --example fluid_newton_fixture -- tests/fixtures/fluid_newton_cube_f124.bin`
//!
//! Prints the byte count and the SHA-256 of what it wrote.
//!
//! Author: Moroya Sakamoto

#[path = "../tests/support/fluid_cube_fixture.rs"]
mod fixture;

use sha2::{Digest, Sha256};

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: fluid_newton_fixture <output file>");
    let sc = fixture::run_to_fixture_frame();
    let bytes = fixture::encode(&sc.fluid);
    std::fs::write(&out, &bytes).expect("write the fixture");
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!(
        "{} particles, frame {}, {} bytes, sha256 {sha}",
        sc.n,
        fixture::FRAME,
        bytes.len()
    );
}
