//! Newton 3 between the fluid and the solids, to rounding, on one frame from a stored state: the
//! cube scene of `tests/analytic_fluid_sdf_body.rs` (cube 2 h above the floor, 3008 particles)
//! at frame 124 of its own run, read from `tests/fixtures/fluid_newton_cube_f124.bin` and stepped
//! once.
//!
//! The state is chosen because a grid built once per substep, instead of on every constraint
//! iteration, finds a pair near a cell edge from one side only there: stepped once with that
//! grid this state gives a largest |sum| of 9.18e5 times the rounding bound, against 2.7e-3 with
//! the grid of every iteration (measured 2026-11-19 with the grid switched in a measurement
//! build, frames 0 to 899 of the run: frames 124 and 128 leak, 128 at 2.2 times the bound). This
//! one-frame test is the per-push detector of that leak named in the fluid MODEL "Newton 3".
//!
//! The file is an input state, written by the example `fluid_cube_state` from the code of
//! the commit that adds it; the ignored test below writes it again and compares the SHA-256.
//!
//! oracle: `|sum of the solids' forces + the boundary force| <= (n · iterations · m / dt^2 +
//! bodies + 1) · 2^-64` N per component (the fluid MODEL "Newton 3").

#[path = "support/fluid_cube_fixture.rs"]
mod fixture;

use alice_physics_core::math::Fix128;
use sha2::{Digest, Sha256};

const STATE: &[u8] = include_bytes!("fixtures/fluid_newton_cube_f124.bin");
const STATE_SHA256: &str = "3ed9ccea89b9bbe330de353e8a06afb73a784910259e36c3e2a133c63845669a";

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn one_frame_from_the_stored_state_keeps_newton_3_within_the_rounding_bound() {
    assert_eq!(sha256(STATE), STATE_SHA256, "the stored state");
    let mut sc = fixture::NEWTON.restore(STATE);
    sc.fluid.step(Fix128::from_ratio(1, 60));
    let bf = sc.fluid.body_forces();
    assert_eq!(bf.len(), 2, "both solids report a force");
    let sum = fixture::largest(bf[0] + bf[1] + sc.fluid.boundary_force());
    let bound = fixture::newton_bound(&sc.fluid, 1.0 / 60.0);
    eprintln!(
        "stored state, one frame: Newton 3 |sum| {sum} N, bound {bound} N, ratio {}",
        sum / bound
    );
    assert!(sum > 0.0, "the step leaves a rounding residue to compare");
    assert!(
        sum <= bound,
        "Newton 3 {sum} N over the rounding bound {bound} N"
    );
}

#[test]
#[ignore = "runtime: about 3.5 min in release (124 frames of 3008 particles); run by run_ignored.py"]
fn the_example_writes_the_stored_state_again() {
    let sc = fixture::NEWTON.run();
    let bytes = fixture::NEWTON.encode(&sc.fluid);
    assert_eq!(bytes.len(), STATE.len(), "length");
    assert_eq!(sha256(&bytes), STATE_SHA256, "the regenerated state");
}
