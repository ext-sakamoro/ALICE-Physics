//! `FrameInput` wire round trip: decoding an encoded input gives back the
//! input that was encoded, every bit of it.
//!
//! A peer that applies its own input and sends the encoded form to the others
//! stays in lockstep only if the others decode the same value. The oracle is
//! the input itself (`decode(encode(x)) == x`); the inputs carry fractional
//! movement and aim, since an integer-only input cannot tell a lossless
//! encoding from one that keeps the integer part.

use alice_physics::netcode::FrameInput;
use alice_physics::{Fix128, Vec3Fix};

fn fractional_input() -> FrameInput {
    FrameInput::new(2)
        .with_movement(Vec3Fix::new(
            Fix128::from_ratio(1, 3),
            Fix128::from_ratio(-5, 2),
            Fix128::from_raw(0, 1),
        ))
        .with_actions(0x8000_0005)
        .with_aim(Vec3Fix::new(
            Fix128::from_ratio(-2, 7),
            Fix128::from_raw(-1, u64::MAX),
            Fix128::from_int(3),
        ))
}

#[test]
fn a_fractional_input_survives_the_wire() {
    let x = fractional_input();
    assert_eq!(FrameInput::from_bytes(&x.to_bytes()), x);
}

#[test]
fn an_integer_input_survives_the_wire() {
    let x = FrameInput::new(0)
        .with_movement(Vec3Fix::from_int(1, 0, -1))
        .with_actions(0x1)
        .with_aim(Vec3Fix::from_int(0, 1, 0));
    assert_eq!(FrameInput::from_bytes(&x.to_bytes()), x);
}
