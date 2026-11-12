//! Layered touch playback on soft bodies: a live recording and its replay from the encoded logs agree frame for
//! frame; a higher layer takes a body over from the layers below and leaves every other body bit-equal to the
//! lower layers alone; the active-body budget settles the earliest free body exactly to rest; the history digest
//! tells apart two logs that end at the same rest; the strict byte form of a touch log; degenerate scenes; and one
//! fixed scene's history digest pinned for every target (run natively and under wasmtime).
//!
//! Oracles come from outside the playback: the override reference is built by hand on two hands of
//! `TouchBodies` from logs edited as the rule says (a lower span holding the body lets go at the grab, its later
//! spans on the body are dropped), with the lower layer's touches given first, so it does not run the override
//! code; the budget and rest cases compare particles against the rest shape bit for bit.
//!
//! Author: Moroya Sakamoto

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "step counts and body indices here are small (under 400), so the casts are exact"
)]

use alice_physics_core::cloth::Cloth;
use alice_physics_core::input_log::{InputLog, LogDecodeError, TOUCH_LOG_MAX_BYTES};
use alice_physics_core::touch::{Hand, Touch};
use alice_physics_core::touch_layers::{
    history_digest, replay, replay_with, LayerError, LayeredPlayback, PlaybackConfig, TouchBodies,
};
use alice_physics_core::{Fix128, Vec3Fix};

const STEPS: u64 = 300;
const BUDGET: usize = 5;

/// Particles per side of a sheet
const RES: usize = 4;

/// Body `i`'s rest centre: unit squares in the plane `y = 0`, 1.2 m apart along x
fn rest_centre(i: usize) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_ratio(6 * i as i64, 5) + Fix128::from_ratio(1, 2),
        Fix128::ZERO,
        Fix128::from_ratio(1, 2),
    )
}

/// `n` soft sheets: 4 × 4 particles of 10 g on a unit square, no gravity, held to their rest shape by a tether
/// of period 0.5 s and damping ratio 0.42
fn sheets(n: usize) -> Vec<Cloth> {
    let mass = Fix128::from_ratio(1, 100);
    let omega = Fix128::from_ratio(12_566, 1000);
    let stiffness = mass * omega * omega;
    let damping = Fix128::from_ratio(84, 100) * mass * omega;
    let half = Fix128::from_ratio(1, 2);
    (0..n)
        .map(|i| {
            let origin = rest_centre(i) - Vec3Fix::new(half, Fix128::ZERO, half);
            let mut c = Cloth::new_grid(origin, Fix128::ONE, Fix128::ONE, RES, RES, mass);
            c.config.gravity = Vec3Fix::ZERO;
            c.config.damping = Fix128::ONE;
            c.config.iterations = 4;
            c.config.substeps = 1;
            c.set_rest_tether(stiffness, damping);
            c
        })
        .collect()
}

fn hand() -> Hand {
    Hand::new(Fix128::from_ratio(4, 10))
}

fn config(steps: u64, budget: usize) -> PlaybackConfig {
    PlaybackConfig::new(
        steps,
        Fix128::from_ratio(1, 60),
        budget,
        Fix128::from_ratio(1, 65_536),
    )
    .expect("config")
}

/// A point `(dx, dz)` hundredths of a metre from body `g`'s rest centre
fn at(g: usize, dx: i64, dz: i64) -> Vec3Fix {
    rest_centre(g)
        + Vec3Fix::new(
            Fix128::from_ratio(dx, 100),
            Fix128::ZERO,
            Fix128::from_ratio(dz, 100),
        )
}

fn log(entries: &[(u64, Touch)]) -> InputLog<Touch> {
    let mut l = InputLog::new();
    for (s, t) in entries {
        l.record(*s, *t).expect("in order");
    }
    l
}

/// Every body's particles
type Frame = Vec<Vec<Vec3Fix>>;

fn snapshot(b: &TouchBodies) -> Frame {
    (0..b.len())
        .map(|i| b.cloth(i).expect("body").positions.clone())
        .collect()
}

fn rest_of(b: &TouchBodies, i: usize) -> Vec<Vec3Fix> {
    b.cloth(i).expect("body").rest_positions().to_vec()
}

/// Record live: `touches(step)` before each step on top of `lower`; the recording and every frame's particles
fn record(
    n: usize,
    lower: Vec<InputLog<Touch>>,
    cfg: PlaybackConfig,
    touches: impl Fn(u64) -> Vec<Touch>,
) -> (
    alice_physics_core::touch_layers::Recording,
    Vec<Frame>,
    Vec<Vec<usize>>,
) {
    let mut p = LayeredPlayback::new(sheets(n), &hand(), lower, cfg).expect("playback");
    let mut frames = Vec::new();
    let mut actives = Vec::new();
    while p.step_now() < STEPS {
        for t in touches(p.step_now()) {
            p.touch(&t).expect("touch");
        }
        p.advance().expect("step");
        frames.push(snapshot(p.bodies()));
        actives.push(p.bodies().active().to_vec());
    }
    (p.finish().0, frames, actives)
}

/// A smoothstep in hundredths: `k` of `n`
fn ease(k: i64, n: i64) -> i64 {
    k * k * (3 * n - 2 * k) * 100 / (n * n * n)
}

/// A lower layer: one span on each body in turn (0 at 10–70, 1 at 80–140, 2 at 150–210, 3 at 220–280), each a
/// grab near a corner and a pull out and back
fn lower_layer() -> InputLog<Touch> {
    let mut l = InputLog::new();
    for g in 0..4u32 {
        let s0 = 10 + 70 * u64::from(g);
        let gi = g as usize;
        l.record(s0, Touch::grab(g, at(gi, -40, -40)))
            .expect("in order");
        for s in s0 + 1..s0 + 60 {
            let k = (s - s0) as i64;
            let out = if k <= 30 {
                ease(k, 30)
            } else {
                100 - ease(k - 30, 29)
            };
            l.record(
                s,
                Touch::drag(at(gi, -40 + 30 * out / 100, -40 + 20 * out / 100)),
            )
            .expect("in order");
        }
        l.record(s0 + 60, Touch::Release).expect("in order");
    }
    l
}

/// The top layer of the override scene: takes body 1 at step 100 (inside the lower span on it, 80–140) and pulls
/// it sideways and back; at 120 moves to body 2 (before the lower span on it, 150–210) for a short press, released
/// at 135
fn top_layer(step: u64) -> Vec<Touch> {
    match step {
        100 => vec![Touch::grab(1, at(1, -10, 10))],
        101..=119 => {
            let k = (step - 100) as i64;
            let dx = if k <= 10 {
                20 * ease(k, 10) / 100
            } else {
                20 - 20 * ease(k - 10, 9) / 100
            };
            vec![Touch::drag(at(1, -10 + dx, 10))]
        }
        120 => vec![Touch::grab(2, at(2, -10, 10))],
        121..=134 => vec![Touch::drag(at(2, -10, 10 - (step as i64 - 120)))],
        135 => vec![Touch::Release],
        _ => Vec::new(),
    }
}

/// `log` with body `body`'s spans edited as a takeover at `from` does: a span holding the body at `from` lets go
/// at `from`, and every touch of its spans from `from` on is dropped
fn taken_over(log: &InputLog<Touch>, body: u32, from: u64) -> InputLog<Touch> {
    let mut out = InputLog::new();
    let mut on = None;
    let mut released = false;
    for &(step, t) in log.entries() {
        if let Touch::Grab { body: b, .. } = t {
            on = Some(b);
        }
        let mine = on == Some(body);
        if mine && step >= from {
            if !released && !matches!(t, Touch::Grab { .. }) {
                out.record(from, Touch::Release).expect("in order");
            }
            released = true;
        } else {
            out.record(step, t).expect("in order");
        }
        if t == Touch::Release {
            on = None;
        }
    }
    out
}

/// Drive `logs` on bodies directly, one hand each, each step's touches given hand 0 first
fn by_hand(n: usize, logs: &[InputLog<Touch>]) -> Vec<Frame> {
    let mut b =
        TouchBodies::new(sheets(n), &hand(), logs.len(), config(STEPS, BUDGET)).expect("bodies");
    let mut frames = Vec::new();
    for step in 0..STEPS {
        for (h, l) in logs.iter().enumerate() {
            for (_, t) in l.entries().iter().filter(|(s, _)| *s == step) {
                b.input(h, t);
            }
        }
        b.step();
        frames.push(snapshot(&b));
    }
    frames
}

/// A recording over 12 bodies under the lower layer, encoded, decoded and replayed, goes through the same states
#[test]
fn a_live_recording_and_its_replay_from_the_encoded_logs_agree_frame_for_frame() {
    let n = 12;
    let lower = lower_layer();
    let touches = |s: u64| -> Vec<Touch> {
        match s {
            30 => vec![Touch::grab(7, at(7, 30, -30))],
            31..=89 => {
                let k = (s - 30) as i64;
                vec![Touch::drag(at(
                    7,
                    30 - 60 * ease(k, 59) / 100,
                    -30 + 40 * ease(k, 59) / 100,
                ))]
            }
            90 | 200 => vec![Touch::Release],
            170 => vec![Touch::grab(2, at(2, 0, 0))],
            171..=199 => vec![Touch::drag(at(2, 0, -((s - 170) as i64) * 50 / 30))],
            _ => Vec::new(),
        }
    };
    let (rec, _, _) = record(n, vec![lower.clone()], config(STEPS, BUDGET), touches);
    assert_eq!(rec.frames.len() as u64, STEPS);
    assert_eq!(rec.history, history_digest(&rec.frames));
    let lower_read =
        InputLog::from_bytes_bounded(&lower.to_bytes(), STEPS, n as u32).expect("lower reads");
    let top_read =
        InputLog::from_bytes_bounded(&rec.top.to_bytes(), STEPS, n as u32).expect("top reads");
    assert_eq!(lower_read, lower);
    assert_eq!(top_read, rec.top);
    let again = replay(
        &sheets(n),
        &hand(),
        &[lower_read, top_read],
        config(STEPS, BUDGET),
    )
    .expect("replay");
    assert_eq!(
        again.frames, rec.frames,
        "a frame differs between the recording and its replay"
    );
    assert_eq!(again.history, rec.history);
    // the touches moved bodies, so the frames are not all the same
    assert!(rec.frames.windows(2).any(|w| w[0] != w[1]));
}

/// The override rule, body by body and frame by frame (4 bodies, under the budget of 5, so the budget never couples
/// one body to another): before the top layer's grab a body is bit-equal to the lower layer alone; from the grab on
/// it is bit-equal to the hand-built reference; bodies the top layer never touches are bit-equal to the lower layer
/// alone at every frame. A rule that sums the layers (the lower hand keeps holding and dragging) fails this.
#[test]
fn a_higher_layer_takes_a_body_over_and_leaves_the_others_bit_equal() {
    let n = 4;
    let lower = lower_layer();
    let (_, only, _) = record(n, vec![lower.clone()], config(STEPS, BUDGET), |_| {
        Vec::new()
    });
    let (rec, real, _) = record(n, vec![lower.clone()], config(STEPS, BUDGET), top_layer);
    let edited = taken_over(&taken_over(&lower, 1, 100), 2, 120);
    let reference = by_hand(n, &[edited, rec.top.clone()]);
    for f in 0..STEPS as usize {
        for (g, from) in [(0, None), (1, Some(100)), (2, Some(120)), (3, None)] {
            let want = match from {
                Some(s) if f as u64 >= s => &reference[f][g],
                _ => &only[f][g],
            };
            assert_eq!(&real[f][g], want, "body {g} at frame {f}");
        }
    }
    // not vacuous: the takeover changed bodies 1 and 2, and the lower layer moves 0 and 3
    for g in [1, 2] {
        assert!(
            (0..STEPS as usize).any(|f| only[f][g] != real[f][g]),
            "body {g} never differs from the lower layer alone"
        );
    }
    for g in [0, 3] {
        assert!(
            (0..STEPS as usize).any(|f| only[f][g] != only[0][g]),
            "body {g} never moves in the lower layer"
        );
    }
}

/// A lower layer whose next grab falls on a body a higher layer has taken: the grab and its span are skipped, but
/// the grab still lets go of the body the lower hand held (a grab always does), so that body springs back from that
/// step. The reference is the lower layer edited by hand (a release in place of the skipped grab, the span dropped)
#[test]
fn a_skipped_grab_still_lets_go_of_what_the_lower_hand_held() {
    let n = 3;
    let mut lower = InputLog::new();
    lower
        .record(10, Touch::grab(0, at(0, 0, 0)))
        .expect("in order");
    for s in 11..80u64 {
        let k = (s - 10).min(30) as i64;
        lower.record(s, Touch::drag(at(0, k, 0))).expect("in order");
    }
    lower
        .record(80, Touch::grab(1, at(1, 0, 0)))
        .expect("in order");
    for s in 81..120u64 {
        lower
            .record(s, Touch::drag(at(1, 0, (s - 80) as i64)))
            .expect("in order");
    }
    lower.record(120, Touch::Release).expect("in order");
    let top = |s: u64| -> Vec<Touch> {
        match s {
            50 => vec![Touch::grab(1, at(1, 10, 10))],
            51..=69 => vec![Touch::drag(at(1, 10, 10 + (s as i64 - 50)))],
            70 => vec![Touch::Release],
            _ => Vec::new(),
        }
    };
    let (rec, real, _) = record(n, vec![lower.clone()], config(STEPS, BUDGET), top);
    let mut edited = InputLog::new();
    for &(s, t) in lower.entries() {
        if s < 80 {
            edited.record(s, t).expect("in order");
        } else if s == 80 {
            edited.record(s, Touch::Release).expect("in order");
        }
    }
    let reference = by_hand(n, &[edited, rec.top.clone()]);
    for f in 0..STEPS as usize {
        for g in 0..n {
            assert_eq!(real[f][g], reference[f][g], "body {g} at frame {f}");
        }
    }
    // body 0 was out and comes back after step 80
    let rest = sheets(1);
    assert_ne!(real[79][0], rest[0].rest_positions());
    assert_eq!(
        real[STEPS as usize - 1][0],
        rest[0].rest_positions(),
        "body 0 was never let go"
    );
}

/// A higher-layer grab that the budget refuses (one moving body, and the lower hand holds it) takes nothing, so it
/// owns nothing: the lower layer's later grab of that body is played, and every frame is bit-equal to the lower
/// layer alone (the reference has no top layer, so it does not run the override code)
#[test]
fn a_refused_higher_grab_does_not_take_the_body_from_the_lower_layers() {
    let n = 2;
    let budget = 1;
    let mut lower = InputLog::new();
    lower
        .record(10, Touch::grab(0, at(0, 0, 0)))
        .expect("in order");
    for s in 11..30u64 {
        lower
            .record(s, Touch::drag(at(0, (s - 10) as i64, 0)))
            .expect("in order");
    }
    lower.record(30, Touch::Release).expect("in order");
    lower
        .record(40, Touch::grab(1, at(1, 0, 0)))
        .expect("in order");
    for s in 41..80u64 {
        lower
            .record(s, Touch::drag(at(1, 0, (s - 40) as i64)))
            .expect("in order");
    }
    lower.record(80, Touch::Release).expect("in order");
    let top = |s: u64| -> Vec<Touch> {
        match s {
            20 => vec![Touch::grab(1, at(1, 10, 10))],
            21..=24 => vec![Touch::drag(at(1, 10, 10 + (s as i64 - 20)))],
            25 => vec![Touch::Release],
            _ => Vec::new(),
        }
    };
    let (_, real, actives) = record(n, vec![lower.clone()], config(STEPS, budget), top);
    assert_eq!(actives[22], vec![0], "the refused grab made body 1 move");
    let mut reference = Vec::new();
    replay_with(
        &sheets(n),
        &hand(),
        &[lower],
        config(STEPS, budget),
        |_, b| {
            reference.push(snapshot(b));
        },
    )
    .expect("replay");
    for f in 0..STEPS as usize {
        for g in 0..n {
            assert_eq!(real[f][g], reference[f][g], "body {g} at frame {f}");
        }
    }
    let rest = sheets(n);
    assert_ne!(
        real[60][1],
        rest[1].rest_positions(),
        "the lower layer's grab of body 1 was skipped"
    );
}

/// A sixth body grabbed while five move: the one that started first (and is not held) is put exactly at rest at
/// that grab, the sixth moves, and the recording replays to the same frames
#[test]
fn a_grab_past_the_budget_settles_the_earliest_free_body_exactly_to_rest() {
    let n = 12;
    let touches = |s: u64| -> Vec<Touch> {
        let g = (s / 3) as usize;
        if g >= 6 {
            return Vec::new();
        }
        match s % 3 {
            0 => vec![Touch::grab(g as u32, at(g, 30, 30))],
            1 => vec![Touch::drag(at(g, 45, 45))],
            _ => vec![Touch::Release],
        }
    };
    let (rec, frames, actives) = record(n, Vec::new(), config(STEPS, BUDGET), touches);
    // after the fifth body's three steps all five still move, body 0 not at rest
    assert_eq!(
        actives[14],
        vec![0, 1, 2, 3, 4],
        "all five still moving before the sixth"
    );
    let rest = sheets(n);
    assert_ne!(
        frames[14][0],
        rest[0].rest_positions(),
        "body 0 had already come to rest"
    );
    // the sixth grab (step 15) put body 0 exactly at rest and moved body 5
    assert_eq!(actives[15], vec![1, 2, 3, 4, 5]);
    assert_eq!(
        frames[15][0],
        rest[0].rest_positions(),
        "body 0 is not exactly at rest"
    );
    assert_ne!(
        frames[15][5],
        rest[5].rest_positions(),
        "the sixth body did not move"
    );
    let again = replay(
        &sheets(n),
        &hand(),
        std::slice::from_ref(&rec.top),
        config(STEPS, BUDGET),
    )
    .expect("replay");
    assert_eq!(again.frames, rec.frames);
}

/// No more than the budget ever move through a busy second (12 bodies touched in turn), and the scene reaches it
#[test]
fn no_more_than_the_budget_ever_move() {
    let n = 12;
    let touches = |s: u64| -> Vec<Touch> {
        if s >= 60 {
            return Vec::new();
        }
        let g = (s / 5) as usize % n;
        match s % 5 {
            0 => vec![Touch::grab(g as u32, at(g, 20, 20))],
            1..=3 => vec![Touch::drag(at(g, 40, 10 * (s % 5) as i64))],
            _ => vec![Touch::Release],
        }
    };
    let (_, _, actives) = record(n, Vec::new(), config(STEPS, BUDGET), touches);
    let most = actives.iter().map(Vec::len).max().unwrap_or(0);
    assert_eq!(
        most, BUDGET,
        "{most} bodies moved at once (budget {BUDGET})"
    );
}

/// Let go after a pull, a body springs back (it overshoots through rest: damping ratio 0.42), and once within the
/// tolerance it is put exactly at rest, its velocities zero, and stops being stepped
#[test]
fn a_released_body_comes_back_to_rest_and_is_put_exactly_there() {
    let n = 2;
    let touches = |s: u64| -> Vec<Touch> {
        match s {
            0 => vec![Touch::grab(1, at(1, 0, 0))],
            1..=20 => vec![Touch::drag(at(1, 2 * s as i64, 0))],
            21 => vec![Touch::Release],
            _ => Vec::new(),
        }
    };
    let mut p = LayeredPlayback::new(sheets(n), &hand(), Vec::new(), config(STEPS, BUDGET))
        .expect("playback");
    let rest_x = rest_centre(1).x;
    let mut crossed = false;
    let mut settled_at = None;
    while p.step_now() < STEPS {
        for t in touches(p.step_now()) {
            p.touch(&t).expect("touch");
        }
        p.advance().expect("step");
        let b = p.bodies();
        if p.step_now() > 22 && b.centre(1).expect("centre").x < rest_x {
            crossed = true;
        }
        if settled_at.is_none() && b.active().is_empty() && p.step_now() > 22 {
            settled_at = Some(p.step_now());
            let c = b.cloth(1).expect("body");
            assert_eq!(c.positions, rest_of(b, 1), "not exactly at rest");
            assert_eq!(c.prev_positions, rest_of(b, 1));
            assert!(c.velocities.iter().all(|v| *v == Vec3Fix::ZERO));
        }
    }
    assert!(crossed, "the body never went past rest (no spring back)");
    let s = settled_at.expect("the body never came to rest within 5 s");
    assert!(s > 40, "settled too soon, at step {s}");
}

/// Two logs that differ in the middle and both end at rest: the same last frame, different histories
#[test]
fn the_history_digest_tells_apart_two_logs_that_end_at_the_same_rest() {
    let n = 3;
    let a = log(&[
        (5, Touch::grab(1, at(1, 0, 0))),
        (20, Touch::drag(at(1, 20, 0))),
        (40, Touch::Release),
    ]);
    let b = log(&[
        (5, Touch::grab(1, at(1, 0, 0))),
        (20, Touch::drag(at(1, 0, 20))),
        (40, Touch::Release),
    ]);
    let ra = replay(&sheets(n), &hand(), &[a], config(STEPS, BUDGET)).expect("a");
    let rb = replay(&sheets(n), &hand(), &[b], config(STEPS, BUDGET)).expect("b");
    assert_eq!(
        ra.frames.last(),
        rb.frames.last(),
        "the two do not end at the same rest"
    );
    assert_ne!(ra.frames[30], rb.frames[30], "the two never differed");
    assert_ne!(
        ra.history, rb.history,
        "the history digest does not tell them apart"
    );
}

/// The byte form of a touch log is read only as it is written
#[test]
fn a_touch_log_is_read_only_in_its_exact_form() {
    let l = log(&[
        (0, Touch::grab(3, at(3, 10, -10))),
        (1, Touch::drag(at(3, 20, -5))),
        (299, Touch::Release),
    ]);
    let bytes = l.to_bytes();
    assert_eq!(InputLog::from_bytes_bounded(&bytes, 300, 4), Ok(l.clone()));
    // every shorter prefix is refused, and cutting the last byte is a truncation
    for k in 0..bytes.len() {
        assert!(
            InputLog::from_bytes_bounded(&bytes[..k], 300, 4).is_err(),
            "prefix {k} read"
        );
    }
    assert_eq!(
        InputLog::from_bytes_bounded(&bytes[..bytes.len() - 1], 300, 4),
        Err(LogDecodeError::Truncated)
    );
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(
        InputLog::from_bytes_bounded(&trailing, 300, 4),
        Err(LogDecodeError::Trailing)
    );
    // the step bound: 299 < 300 reads, under a bound of 299 it does not
    assert_eq!(
        InputLog::from_bytes_bounded(&bytes, 299, 4),
        Err(LogDecodeError::StepOutOfRange)
    );
    // the body bound: body 3 needs at least 4
    assert_eq!(
        InputLog::from_bytes_bounded(&bytes, 300, 3),
        Err(LogDecodeError::BodyOutOfRange)
    );
    // the byte budget
    assert_eq!(
        InputLog::from_bytes_bounded(&vec![0; TOUCH_LOG_MAX_BYTES + 1], 300, 4),
        Err(LogDecodeError::TooLarge)
    );
    // a step that overflows: a release at step 1, then a delta of 2⁶⁴ − 1
    let mut over = vec![2, 1, 3];
    over.extend_from_slice(&[
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 3,
    ]);
    assert_eq!(
        InputLog::from_bytes_bounded(&over, u64::MAX, 4),
        Err(LogDecodeError::Overflow)
    );
    // a varint past 64 bits
    let long = [
        0x80u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00,
    ];
    assert_eq!(
        InputLog::from_bytes_bounded(&long, 300, 4),
        Err(LogDecodeError::Overflow)
    );
    // a padded count (0 written as two bytes) reads as the same numbers but is not the one form
    assert_eq!(
        InputLog::from_bytes_bounded(&[0x80, 0x00], 300, 4),
        Err(LogDecodeError::Malformed)
    );
    // an unknown touch tag
    assert_eq!(
        InputLog::from_bytes_bounded(&[1, 0, 9], 300, 4),
        Err(LogDecodeError::Malformed)
    );
    // a count larger than the bytes can hold
    assert_eq!(
        InputLog::from_bytes_bounded(&[5, 0, 3], 300, 4),
        Err(LogDecodeError::Truncated)
    );
}

/// Degenerate scenes, each with its stated result and no panic
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one case after another, each a few lines, read top to bottom"
)]
fn degenerate_scenes_give_their_stated_results() {
    // config: a step not above zero, a negative tolerance
    let dt = Fix128::from_ratio(1, 60);
    assert_eq!(
        PlaybackConfig::new(1, Fix128::ZERO, 1, Fix128::ZERO),
        Err(LayerError::BadStep)
    );
    assert_eq!(
        PlaybackConfig::new(1, dt, 1, -Fix128::ONE),
        Err(LayerError::BadTolerance)
    );
    assert!(PlaybackConfig::new(0, dt, 0, Fix128::ZERO).is_ok());

    // 0 bodies: a grab is refused, every frame is the same
    let mut p =
        LayeredPlayback::new(Vec::new(), &hand(), Vec::new(), config(10, BUDGET)).expect("empty");
    assert!(p.bodies().is_empty());
    assert_eq!(
        p.touch(&Touch::grab(0, Vec3Fix::ZERO)),
        Err(LayerError::NoSuchBody)
    );
    let (rec, _) = p.finish();
    assert_eq!(rec.frames.len(), 10);
    assert!(rec.frames.windows(2).all(|w| w[0] == w[1]));
    assert!(rec.top.entries().is_empty(), "a refused grab was recorded");

    // a budget of 0: every grab of a body at rest takes nothing, nothing moves
    let pull = log(&[
        (0, Touch::grab(0, at(0, 0, 0))),
        (1, Touch::drag(at(0, 30, 0))),
        (5, Touch::Release),
    ]);
    let still = replay(&sheets(2), &hand(), &[], config(20, 0)).expect("still");
    let zero = replay(
        &sheets(2),
        &hand(),
        std::slice::from_ref(&pull),
        config(20, 0),
    )
    .expect("budget 0");
    assert_eq!(
        zero.frames, still.frames,
        "a body moved under a budget of 0"
    );

    // a budget of 1, one layer: grabbing another body lets go of the first, which is settled exactly
    let mut p =
        LayeredPlayback::new(sheets(2), &hand(), Vec::new(), config(20, 1)).expect("budget 1");
    p.touch(&Touch::grab(0, at(0, 0, 0))).expect("grab");
    p.advance().expect("step");
    p.touch(&Touch::drag(at(0, 30, 0))).expect("drag");
    p.advance().expect("step");
    p.touch(&Touch::grab(1, at(1, 0, 0))).expect("grab");
    p.advance().expect("step");
    assert_eq!(p.bodies().active(), &[1]);
    assert_eq!(
        p.bodies().cloth(0).expect("body").positions,
        rest_of(p.bodies(), 0)
    );
    // a budget of 1, two layers: the lower hand holds body 0, so the top grab of body 1 takes nothing
    let hold = log(&[
        (0, Touch::grab(0, at(0, 0, 0))),
        (1, Touch::drag(at(0, 30, 0))),
    ]);
    let mut p = LayeredPlayback::new(sheets(2), &hand(), vec![hold], config(20, 1)).expect("held");
    p.advance().expect("step");
    p.advance().expect("step");
    p.touch(&Touch::grab(1, at(1, 0, 0))).expect("recorded");
    p.advance().expect("step");
    p.touch(&Touch::drag(at(1, 30, 0))).expect("recorded");
    p.advance().expect("step");
    assert_eq!(
        p.bodies().active(),
        &[0],
        "a held body was settled or the budget exceeded"
    );
    assert_eq!(
        p.bodies().cloth(1).expect("body").positions,
        rest_of(p.bodies(), 1)
    );

    // a lower layer's grab of a body that does not exist takes nothing
    let ghost = log(&[
        (0, Touch::grab(9, Vec3Fix::ZERO)),
        (1, Touch::drag(at(0, 30, 0))),
    ]);
    let r = replay(
        &sheets(2),
        &hand(),
        &[ghost, InputLog::new()],
        config(20, BUDGET),
    )
    .expect("ghost");
    let none = replay(
        &sheets(2),
        &hand(),
        &[InputLog::new(), InputLog::new()],
        config(20, BUDGET),
    )
    .expect("base");
    assert_eq!(r.frames, none.frames);
    // a top-layer one is refused when replayed too
    let top_ghost = log(&[(0, Touch::grab(9, Vec3Fix::ZERO))]);
    assert_eq!(
        replay(&sheets(2), &hand(), &[top_ghost], config(20, BUDGET)),
        Err(LayerError::NoSuchBody)
    );

    // empty logs: every frame is the first state
    let first = TouchBodies::new(sheets(3), &hand(), 1, config(20, BUDGET))
        .expect("bodies")
        .digest();
    let r = replay(&sheets(3), &hand(), &[], config(20, BUDGET)).expect("empty");
    assert_eq!(r.frames, vec![first; 20]);

    // a touch at or after the last step
    let late = log(&[(20, Touch::Release)]);
    assert_eq!(
        replay(
            &sheets(1),
            &hand(),
            std::slice::from_ref(&late),
            config(20, BUDGET)
        ),
        Err(LayerError::TooLong)
    );
    assert_eq!(
        replay(
            &sheets(1),
            &hand(),
            &[late, InputLog::new()],
            config(20, BUDGET)
        ),
        Err(LayerError::TooLong)
    );
    let mut p =
        LayeredPlayback::new(sheets(1), &hand(), Vec::new(), config(0, BUDGET)).expect("no steps");
    assert_eq!(p.touch(&Touch::Release), Err(LayerError::TooLong));
    assert_eq!(p.advance(), Err(LayerError::TooLong));
    assert!(p.finish().0.frames.is_empty());

    // a hand that holds, a body without a rest shape
    let mut busy = hand();
    let mut cs = sheets(1);
    busy.apply(&mut cs, &Touch::grab(0, at(0, 0, 0)));
    assert_eq!(
        TouchBodies::new(sheets(1), &busy, 1, config(1, 1)).err(),
        Some(LayerError::HandHolds)
    );
    let mut tetherless = sheets(2);
    tetherless[1] = Cloth::new_grid(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE, 2, 2, Fix128::ONE);
    assert_eq!(
        TouchBodies::new(tetherless, &hand(), 1, config(1, 1)).err(),
        Some(LayerError::NoRestShape(1))
    );

    // a bound of 0 steps: only the empty log reads
    assert_eq!(
        InputLog::from_bytes_bounded(&[0], 0, 0),
        Ok(InputLog::new())
    );
    assert_eq!(
        InputLog::from_bytes_bounded(&log(&[(0, Touch::Release)]).to_bytes(), 0, 0),
        Err(LogDecodeError::StepOutOfRange)
    );
}

/// The history digest of one fixed scene (8 bodies, two pulls), recorded natively and checked on every target
/// (`CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime cargo test --release --target wasm32-wasip1 --test
/// analytic_touch_layers`). A change of the physics or the scene moves it, and every target is then checked again.
const PINNED: &str = "76874bcf0ab80e87f02fd76b8f7bbd6104891359afddc30f9897c1395d444edb";

#[test]
fn a_fixed_scene_reaches_the_pinned_history_digest() {
    let touches = |s: u64| -> Vec<Touch> {
        match s {
            5 => vec![Touch::grab(1, at(1, -40, -40))],
            6..=40 => {
                let k = (s - 5) as i64;
                vec![Touch::drag(at(1, -40 + 2 * k, -40 + k))]
            }
            41 | 121 => vec![Touch::Release],
            90 => vec![Touch::grab(6, at(6, 0, 0))],
            91..=120 => vec![Touch::drag(at(6, 0, -((s - 90) as i64)))],
            _ => Vec::new(),
        }
    };
    let (rec, _, _) = record(8, Vec::new(), config(STEPS, BUDGET), touches);
    let hex: String = rec.history.iter().fold(String::new(), |mut h, b| {
        use std::fmt::Write as _;
        let _ = write!(h, "{b:02x}");
        h
    });
    println!("touch-layers pinned scene history digest: {hex}");
    assert_eq!(
        hex, PINNED,
        "this target's playback differs from the pinned one"
    );
}
