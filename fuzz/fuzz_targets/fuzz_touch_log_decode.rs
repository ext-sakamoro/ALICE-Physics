//! Fuzz target: the strict byte form of a touch log (`InputLog::<Touch>::from_bytes_bounded`)
//!
//! A touch log comes from another device, so its bytes are untrusted. Any byte string, with any bounds, must
//! decode without panicking or allocating from the untrusted entry count, and whatever it accepts must be in the
//! one exact form: written back with `to_bytes` it gives the same bytes, and it stays within the bounds it was
//! read under (every step below `max_step`, every grabbed body below `max_body`).

#![no_main]

use alice_physics::input_log::InputLog;
use alice_physics::touch::Touch;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    max_step: u64,
    max_body: u32,
    bytes: Vec<u8>,
}

fuzz_target!(|input: Input| {
    let Ok(log) =
        InputLog::<Touch>::from_bytes_bounded(&input.bytes, input.max_step, input.max_body)
    else {
        return;
    };
    assert_eq!(
        log.to_bytes(),
        input.bytes,
        "an accepted log is not in its exact form"
    );
    for (step, touch) in log.entries() {
        assert!(
            *step < input.max_step,
            "step {step} at or past {}",
            input.max_step
        );
        if let Touch::Grab { body, .. } = touch {
            assert!(
                *body < input.max_body,
                "body {body} at or past {}",
                input.max_body
            );
        }
    }
});
