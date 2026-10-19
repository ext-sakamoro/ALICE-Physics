//! `FrameInput` wire round trip: decoding an encoded input gives back the
//! input that was encoded, every bit of it, and what is not the current form
//! is refused instead of read as a value.
//!
//! A peer that applies its own input and sends the encoded form to the others
//! stays in lockstep only if the others decode the same value. The oracle is
//! the input itself (`from_wire(to_wire(x)) == x`); the inputs carry
//! fractional movement and aim, since an integer-only input cannot tell a
//! lossless encoding from one that keeps the integer part (the form before
//! 4.0.0 kept `hi` as an `i16`).

use alice_physics_core::netcode::FrameInput;
use alice_physics_core::{Fix128, Vec3Fix, WireError};

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
    assert_eq!(FrameInput::from_wire(&x.to_wire()), Ok(x));
}

#[test]
fn an_integer_input_survives_the_wire() {
    let x = FrameInput::new(0)
        .with_movement(Vec3Fix::from_int(1, 0, -1))
        .with_actions(0x1)
        .with_aim(Vec3Fix::from_int(0, 1, 0));
    assert_eq!(FrameInput::from_wire(&x.to_wire()), Ok(x));
}

#[test]
fn an_input_beyond_the_old_i16_range_survives_the_wire() {
    let x = FrameInput::new(255).with_movement(Vec3Fix::new(
        Fix128::from_raw(i64::MAX, u64::MAX),
        Fix128::from_raw(i64::MIN, 0),
        Fix128::from_int(40_000),
    ));
    assert_eq!(FrameInput::from_wire(&x.to_wire()), Ok(x));
}

/// The layout written in the `to_wire` doc, byte by byte: version first, then
/// `hi` before `lo` of each component.
#[test]
fn the_layout_is_the_documented_one() {
    let w = fractional_input().to_wire();
    assert_eq!(w.len(), FrameInput::WIRE_BYTES);
    assert_eq!(w.len(), 102);
    assert_eq!(w[0], FrameInput::WIRE_VERSION);
    assert_eq!(w[1], 2);
    // movement.x = 1/3: hi 0, lo = floor(2^64 / 3)
    assert_eq!(&w[2..10], &0i64.to_le_bytes());
    assert_eq!(&w[10..18], &0x5555_5555_5555_5555u64.to_le_bytes());
    // movement.z = 2^-64: hi 0, lo 1
    assert_eq!(&w[34..42], &0i64.to_le_bytes());
    assert_eq!(&w[42..50], &1u64.to_le_bytes());
    assert_eq!(&w[50..54], &0x8000_0005u32.to_le_bytes());
    // aim.y = -2^-64: hi -1, lo u64::MAX
    assert_eq!(&w[70..78], &(-1i64).to_le_bytes());
    assert_eq!(&w[78..86], &u64::MAX.to_le_bytes());
}

#[test]
fn the_legacy_twenty_byte_form_is_refused() {
    assert_eq!(FrameInput::from_wire(&[0u8; 20]), Err(WireError::Legacy));
}

#[test]
fn another_length_is_refused() {
    let w = fractional_input().to_wire();
    assert_eq!(
        FrameInput::from_wire(&w[..101]),
        Err(WireError::Length { found: 101 })
    );
    let mut long = w.to_vec();
    long.push(0);
    assert_eq!(
        FrameInput::from_wire(&long),
        Err(WireError::Length { found: 103 })
    );
    assert_eq!(
        FrameInput::from_wire(&[]),
        Err(WireError::Length { found: 0 })
    );
}

#[test]
fn another_version_is_refused() {
    let mut w = fractional_input().to_wire();
    w[0] = FrameInput::WIRE_VERSION.wrapping_add(1);
    assert_eq!(
        FrameInput::from_wire(&w),
        Err(WireError::Version {
            found: FrameInput::WIRE_VERSION.wrapping_add(1)
        })
    );
    w[0] = 0;
    assert_eq!(
        FrameInput::from_wire(&w),
        Err(WireError::Version { found: 0 })
    );
}
