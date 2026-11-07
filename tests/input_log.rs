//! Oracles for `input_log`: a log replayed from the same first state reaches the same `state_digest` as the run it
//! came from, a code carries it whole, and what is not a code is refused without a panic.

use alice_physics_core::input_log::{InputLog, LogError, LogInput};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Input {
    Blow(i16, i16),
    Rest,
}

impl LogInput for Input {
    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Self::Blow(x, z) => {
                out.push(1);
                out.extend_from_slice(&x.to_le_bytes());
                out.extend_from_slice(&z.to_le_bytes());
            }
            Self::Rest => out.push(2),
        }
    }
    fn read(bytes: &mut &[u8]) -> Option<Self> {
        let (&tag, rest) = bytes.split_first()?;
        *bytes = rest;
        match tag {
            1 => {
                if bytes.len() < 4 {
                    return None;
                }
                let x = i16::from_le_bytes([bytes[0], bytes[1]]);
                let z = i16::from_le_bytes([bytes[2], bytes[3]]);
                *bytes = &bytes[4..];
                Some(Self::Blow(x, z))
            }
            2 => Some(Self::Rest),
            _ => None,
        }
    }
}

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn dt() -> Fix128 {
    fx(1.0 / 60.0)
}

fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    for k in 0..8 {
        w.add_body(RigidBody::new(
            Vec3Fix::new(fx(f64::from(k)), fx(1.0), fx(0.0)),
            Fix128::ONE,
        ));
    }
    w
}

fn apply(w: &mut PhysicsWorld, i: &Input) {
    if let Input::Blow(x, z) = *i {
        w.apply_radial_impulse(
            Vec3Fix::new(fx(f64::from(x) * 0.1), fx(0.0), fx(f64::from(z) * 0.1)),
            fx(6.0),
            fx(5.0),
        );
    }
}

fn plan() -> Vec<(u64, Input)> {
    vec![
        (0, Input::Rest),
        (5, Input::Blow(10, 3)),
        (5, Input::Blow(-4, 7)),
        (40, Input::Rest),
        (300, Input::Blow(30, -30)),
    ]
}

/// The run itself, input by input, as it happened
fn live() -> ([u8; 32], InputLog<Input>) {
    let mut w = world();
    let mut log = InputLog::new();
    let inputs = plan();
    let mut next = 0;
    for step in 0..360u64 {
        while next < inputs.len() && inputs[next].0 == step {
            apply(&mut w, &inputs[next].1);
            log.record(step, inputs[next].1).expect("in order");
            next += 1;
        }
        w.step(dt());
    }
    (w.state_digest(), log)
}

#[test]
fn a_replay_reaches_the_digest_of_the_run_it_came_from() {
    let (want, log) = live();
    let mut w = world();
    log.replay(&mut w, 360, dt(), apply);
    assert_eq!(w.state_digest(), want);
}

#[test]
fn without_the_inputs_the_digest_is_another() {
    let (want, _) = live();
    let mut w = world();
    InputLog::<Input>::new().replay(&mut w, 360, dt(), apply);
    assert_ne!(w.state_digest(), want);
    // and one input changed
    let mut other = InputLog::new();
    other.record(5, Input::Blow(11, 3)).expect("in order");
    let mut w = world();
    other.replay(&mut w, 360, dt(), apply);
    assert_ne!(w.state_digest(), want);
}

#[test]
fn an_input_after_the_last_step_is_given_after_it() {
    let mut log = InputLog::new();
    log.record(10, Input::Blow(1, 0)).expect("in order");
    let mut w = world();
    let mut given = 0;
    log.replay(&mut w, 3, dt(), |_, _| given += 1);
    assert_eq!(given, 1);
    // the last step was the third: the input did not run a step more
    let mut plain = world();
    for _ in 0..3 {
        plain.step(dt());
    }
    assert_eq!(w.state_digest(), plain.state_digest());
}

#[test]
fn a_step_before_the_last_is_refused() {
    let mut log = InputLog::new();
    log.record(5, Input::Rest).expect("first");
    log.record(5, Input::Rest).expect("same step");
    assert_eq!(log.record(4, Input::Rest), Err(LogError::Backwards));
    assert_eq!(log.entries().len(), 2);
}

#[test]
fn a_code_carries_the_log_and_the_digest_and_replays_to_the_same_bits() {
    let (want, log) = live();
    let code = log.to_code(&want);
    assert!(
        code.bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
        "{code}"
    );
    let (head, back) = InputLog::<Input>::from_code(&code).expect("a code");
    assert_eq!(head, [want[0], want[1], want[2], want[3]]);
    assert_eq!(back, log);
    let mut w = world();
    back.replay(&mut w, 360, dt(), apply);
    assert_eq!(w.state_digest(), want);
}

#[test]
fn an_empty_log_has_a_code_too() {
    let log = InputLog::<Input>::new();
    let code = log.to_code(&[7; 32]);
    let (head, back) = InputLog::<Input>::from_code(&code).expect("a code");
    assert_eq!(head, [7; 4]);
    assert_eq!(back.entries().len(), 0);
}

#[test]
fn large_steps_survive_a_code() {
    let mut log = InputLog::new();
    log.record(u64::from(u32::MAX) + 5, Input::Rest)
        .expect("in order");
    log.record(u64::MAX, Input::Blow(-1, 1)).expect("in order");
    let (_, back) = InputLog::<Input>::from_code(&log.to_code(&[0; 32])).expect("a code");
    assert_eq!(back, log);
}

#[test]
fn what_is_not_a_code_is_refused() {
    let (want, log) = live();
    let good = log.to_code(&want);
    for bad in [
        "",
        "-",
        "abc",
        "not a code",
        "zzzzzzzz-!!!!",
        "aaaaaaa-AAAA",
        "gggggggg-AAAA",
        "aaaaaaaa",
    ] {
        assert_eq!(
            InputLog::<Input>::from_code(bad),
            Err(LogError::NotACode),
            "{bad:?}"
        );
    }
    // cut off anywhere it must be either refused or a shorter log, and never panic
    for k in 0..good.len() {
        let _ = InputLog::<Input>::from_code(&good[..k]);
    }
    // bytes that say a step far past the end of a u64
    let huge = format!("{}-{}", "00000000", "_".repeat(40));
    assert_eq!(InputLog::<Input>::from_code(&huge), Err(LogError::NotACode));
    // the steps of a code can only go forward, so one that overflows is refused
    let mut log = InputLog::new();
    log.record(u64::MAX, Input::Rest).expect("in order");
    log.record(u64::MAX, Input::Rest).expect("in order");
    let (_, back) = InputLog::<Input>::from_code(&log.to_code(&[0; 32])).expect("a code");
    assert_eq!(back.entries().len(), 2);
}

#[test]
fn bytes_are_an_input_and_a_code_carries_them() {
    let mut log = InputLog::<Vec<u8>>::new();
    log.record(3, vec![1, 2, 3]).expect("in order");
    log.record(9, Vec::new()).expect("in order");
    log.record(9, vec![0xAB; 300]).expect("in order");
    let (_, back) = InputLog::<Vec<u8>>::from_code(&log.to_code(&[0; 32])).expect("a code");
    assert_eq!(back, log);
    // a length past the end of the bytes
    let code = format!("00000000-{}", letters(&[0, 200, 1, 1, 2]));
    assert_eq!(
        InputLog::<Vec<u8>>::from_code(&code),
        Err(LogError::NotACode)
    );
    // a length that does not fit
    let mut long = vec![0];
    long.extend_from_slice(&[0xff; 12]);
    let code = format!("00000000-{}", letters(&long));
    assert_eq!(
        InputLog::<Vec<u8>>::from_code(&code),
        Err(LogError::NotACode)
    );
}

/// Raw bytes as the letters of a code, written apart from the code's own writer
fn letters(bytes: &[u8]) -> String {
    const L: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for c in bytes.chunks(3) {
        let n = c.iter().fold(0u32, |n, b| (n << 8) | u32::from(*b)) << (8 * (3 - c.len()));
        for k in 0..=c.len() {
            out.push(char::from(L[((n >> (18 - 6 * k)) & 63) as usize]));
        }
    }
    out
}

#[test]
fn steps_that_run_past_the_end_are_refused() {
    let mut b = vec![0xff; 9];
    b.push(0x01); // u64::MAX
    b.push(2);
    b.push(1); // one more
    b.push(2);
    let code = format!("00000000-{}", letters(&b));
    assert_eq!(InputLog::<Input>::from_code(&code), Err(LogError::NotACode));
    // the same bytes without the last gap are a log, so it is the sum that is refused
    let ok = format!("00000000-{}", letters(&b[..11]));
    assert_eq!(
        InputLog::<Input>::from_code(&ok)
            .expect("a code")
            .1
            .entries()
            .len(),
        1
    );
}

#[test]
fn a_letter_alone_at_the_end_is_refused() {
    assert_eq!(
        InputLog::<Input>::from_code("00000000-A"),
        Err(LogError::NotACode)
    );
    let log = InputLog::<Input>::new();
    assert!(InputLog::<Input>::from_code(&log.to_code(&[0; 32])).is_ok());
}
