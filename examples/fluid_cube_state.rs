//! Writes a stored fluid state of the cube scene of `tests/support/fluid_cube_fixture.rs` (cube
//! 2 h above the floor): frame 124, read by `tests/analytic_fluid_newton_fixture.rs`, or frame
//! 840, read by `tests/analytic_fluid_sdf_body.rs`, stepped from the lattice in frames of
//! 1/60 s; positions and velocities in the form described there.
//!
//! `cargo run --release --example fluid_cube_state -- 124 tests/fixtures/fluid_newton_cube_f124.bin`
//! `cargo run --release --example fluid_cube_state -- 840 tests/fixtures/fluid_cube_f840.bin`
//!
//! Prints the byte count and the SHA-256 of what it wrote. Without arguments it writes nothing:
//! it reads both stored files back into the scene and checks that their stored form is the same
//! bytes again (the header, the particle count and every coordinate; the stepping itself is what
//! the two tests above check).
//!
//! Author: Moroya Sakamoto

#[path = "../tests/support/fluid_cube_fixture.rs"]
mod fixture;

use sha2::{Digest, Sha256};

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: fluid_cube_state [<124 | 840> <output file>]";
    let Some(frame) = args.next() else {
        for stored in [fixture::NEWTON, fixture::SETTLED] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(stored.file);
            let kept = std::fs::read(&path).expect("read the stored state");
            let sc = stored.restore(&kept);
            let again = stored.encode(&sc.fluid);
            assert!(again == kept, "{} is not its own stored form", stored.file);
            println!(
                "{}: {} particles, frame {}, {} bytes, sha256 {}",
                stored.file,
                sc.n,
                stored.frame,
                kept.len(),
                sha256(&kept)
            );
        }
        return;
    };
    let frame: usize = frame.parse().expect(usage);
    let out = args.next().expect(usage);
    let stored = [fixture::NEWTON, fixture::SETTLED]
        .into_iter()
        .find(|s| s.frame == frame)
        .expect(usage);
    let sc = stored.run();
    let bytes = stored.encode(&sc.fluid);
    std::fs::write(&out, &bytes).expect("write the state");
    println!(
        "{} particles, frame {frame}, {} bytes, sha256 {}",
        sc.n,
        bytes.len(),
        sha256(&bytes)
    );
}
