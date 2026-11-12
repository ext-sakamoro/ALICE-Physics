//! Limits on a layered touch playback: a shake faster than the rate is found (its reversals counted against the
//! closed-form count of a triangle wave) and weakened until it passes; a pull beyond the distance is weakened; a safe
//! log passes unchanged; weakening changes only the top layer (a lower layer beyond the limits cannot be fixed, and
//! a lower layer on the same body within one second stays at full strength in what passes); a jitter under the
//! swing is not a reversal; limits out of range are refused.
//!
//! Author: Moroya Sakamoto

#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    reason = "step counts and body indices here are small (under 400), so the casts are exact"
)]

use alice_physics_core::cloth::Cloth;
use alice_physics_core::input_log::InputLog;
use alice_physics_core::touch::{Hand, Touch};
use alice_physics_core::touch_layers::{replay, LayerError, PlaybackConfig};
use alice_physics_core::touch_limits::{check, within_limits, TouchLimits, Violation};
use alice_physics_core::{Fix128, Vec3Fix};

const STEPS: u64 = 300;
const RATE: u64 = 60;

fn rest_centre(i: usize) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_ratio(6 * i as i64, 5) + Fix128::from_ratio(1, 2),
        Fix128::ZERO,
        Fix128::from_ratio(1, 2),
    )
}

/// `n` soft sheets as in `tests/analytic_touch_layers.rs`: 4 × 4 particles of 10 g, tether period 0.5 s, ζ 0.42
fn sheets(n: usize) -> Vec<Cloth> {
    let mass = Fix128::from_ratio(1, 100);
    let omega = Fix128::from_ratio(12_566, 1000);
    let stiffness = mass * omega * omega;
    let damping = Fix128::from_ratio(84, 100) * mass * omega;
    let half = Fix128::from_ratio(1, 2);
    (0..n)
        .map(|i| {
            let origin = rest_centre(i) - Vec3Fix::new(half, Fix128::ZERO, half);
            let mut c = Cloth::new_grid(origin, Fix128::ONE, Fix128::ONE, 4, 4, mass);
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

fn config() -> PlaybackConfig {
    PlaybackConfig::new(
        STEPS,
        Fix128::from_ratio(1, 60),
        5,
        Fix128::from_ratio(1, 65_536),
    )
    .expect("config")
}

/// At most 6 reversals a second (3 Hz), 0.6 m from rest, swings under 1/32 m not counted
fn limits() -> TouchLimits {
    TouchLimits::new(
        6,
        Fix128::from_ratio(6, 10),
        Fix128::from_ratio(1, 32),
        RATE,
    )
    .expect("limits")
}

fn at(g: usize, dx: i64, dz: i64) -> Vec3Fix {
    rest_centre(g)
        + Vec3Fix::new(
            Fix128::from_ratio(dx, 100),
            Fix128::ZERO,
            Fix128::from_ratio(dz, 100),
        )
}

/// A triangle-wave shake of body `g` along x from `from` to `to` (released there unless `to` is the last step, so
/// that no spring back after a release adds a turn): `period` steps a cycle (even, the same number of steps each way),
/// between `−amp/2` and `+amp/2` hundredths of a metre around the grab point
fn shake(g: u32, from: u64, to: u64, period: u64, amp: i64) -> InputLog<Touch> {
    let mut l = InputLog::new();
    let gi = g as usize;
    l.record(from, Touch::grab(g, at(gi, 0, 0)))
        .expect("in order");
    let p = period as i64;
    for s in from + 1..to {
        let ph = ((s - from) % period) as i64;
        let tri = if ph < p / 2 { ph } else { p - ph };
        l.record(s, Touch::drag(at(gi, amp * 2 * tri / p - amp / 2, 0)))
            .expect("in order");
    }
    if to < STEPS {
        l.record(to, Touch::Release).expect("in order");
    }
    l
}

/// `l` with every drag brought toward its span's grab point by `scale` (written here from the rule, not taken from
/// the module)
fn scaled(l: &InputLog<Touch>, scale: Fix128) -> InputLog<Touch> {
    let mut out = InputLog::new();
    let mut a = Vec3Fix::ZERO;
    for (s, t) in l.entries() {
        let t = match *t {
            Touch::Grab { at, .. } => {
                a = at;
                *t
            }
            Touch::Drag { to } => Touch::drag(a + (to - a) * scale),
            Touch::Release => *t,
        };
        out.record(*s, t).expect("in order");
    }
    out
}

const LADDER: [(i64, i64); 7] = [(3, 4), (1, 2), (3, 8), (1, 4), (1, 8), (1, 16), (1, 32)];

/// The rung of `scale` on the ladder, and the scale of the rung before it (1 for the first)
fn rung(scale: Fix128) -> (usize, Fix128) {
    let k = LADDER
        .iter()
        .position(|&(a, b)| Fix128::from_ratio(a, b) == scale)
        .expect("a rung of the ladder");
    let before = if k == 0 {
        Fix128::ONE
    } else {
        Fix128::from_ratio(LADDER[k - 1].0, LADDER[k - 1].1)
    };
    (k, before)
}

/// Body `g` grabbed `ax` hundredths from its centre at `from`, pulled to `bx` with a smoothstep by `hold`, let go at
/// `release`
fn pull(g: u32, from: u64, hold: u64, release: u64, ax: i64, bx: i64) -> InputLog<Touch> {
    let mut out = InputLog::new();
    let gi = g as usize;
    out.record(from, Touch::grab(g, at(gi, ax, 0)))
        .expect("in order");
    let n = (hold - from) as i64;
    for s in from + 1..=hold {
        let k = (s - from) as i64;
        let eased = k * k * (3 * n - 2 * k) * 100 / (n * n * n);
        out.record(s, Touch::drag(at(gi, ax + (bx - ax) * eased / 100, 0)))
            .expect("in order");
    }
    out.record(release, Touch::Release).expect("in order");
    out
}

/// A shake of 3.75 Hz over the whole playback (16 steps a cycle: a turn every 8 steps, so 8 turns lie within any 60
/// steps and 9 never do) is reported with 8 reversals, weakened at the first rung of the ladder that passes (the rung
/// before it does not), and what is sent replays the same twice
#[test]
fn a_shake_faster_than_the_limit_is_found_and_weakened_until_it_passes() {
    let cs = sheets(6);
    let top = shake(1, 0, STEPS, 16, 36);
    let found = check(
        &cs,
        &hand(),
        std::slice::from_ref(&top),
        config(),
        &limits(),
    )
    .expect("check");
    assert_eq!(
        found,
        vec![Violation::TooFast {
            body: 1,
            reversals: 8
        }]
    );
    let (sent, scale) = within_limits(
        &cs,
        &hand(),
        std::slice::from_ref(&top),
        config(),
        &limits(),
    )
    .expect("check")
    .expect("weakened");
    assert_eq!(
        sent,
        scaled(&top, scale),
        "the drags were not scaled toward their grab point"
    );
    assert!(check(
        &cs,
        &hand(),
        std::slice::from_ref(&sent),
        config(),
        &limits()
    )
    .expect("check")
    .is_empty());
    let (_, before) = rung(scale);
    assert!(scale < Fix128::ONE);
    assert!(
        !check(&cs, &hand(), &[scaled(&top, before)], config(), &limits())
            .expect("check")
            .is_empty(),
        "the rung before {scale:?} passes too"
    );
    let a = replay(&cs, &hand(), std::slice::from_ref(&sent), config()).expect("replay");
    let b = replay(&cs, &hand(), std::slice::from_ref(&sent), config()).expect("replay");
    assert_eq!(a.history, b.history);
}

/// A pull to 1.2 m (twice the bound) is reported and weakened until it passes
#[test]
fn a_pull_beyond_the_distance_is_weakened_until_it_passes() {
    let cs = sheets(6);
    let top = pull(3, 0, 60, 60, 0, 120);
    let found = check(
        &cs,
        &hand(),
        std::slice::from_ref(&top),
        config(),
        &limits(),
    )
    .expect("check");
    assert_eq!(found, vec![Violation::TooFar { body: 3 }]);
    let (sent, scale) = within_limits(
        &cs,
        &hand(),
        std::slice::from_ref(&top),
        config(),
        &limits(),
    )
    .expect("check")
    .expect("weakened");
    assert!(scale < Fix128::ONE);
    assert_eq!(sent, scaled(&top, scale));
    assert!(check(&cs, &hand(), &[sent], config(), &limits())
        .expect("check")
        .is_empty());
    let (_, before) = rung(scale);
    assert!(
        !check(&cs, &hand(), &[scaled(&top, before)], config(), &limits())
            .expect("check")
            .is_empty()
    );
}

/// A gentle pull (0.25 m once) passes as it is, at scale 1
#[test]
fn a_safe_log_passes_unchanged_at_scale_one() {
    let cs = sheets(6);
    let top = pull(2, 0, 60, 60, 0, 25);
    assert!(check(
        &cs,
        &hand(),
        std::slice::from_ref(&top),
        config(),
        &limits()
    )
    .expect("check")
    .is_empty());
    assert_eq!(
        within_limits(
            &cs,
            &hand(),
            std::slice::from_ref(&top),
            config(),
            &limits()
        ),
        Ok(Some((top, Fix128::ONE)))
    );
}

/// A lower layer beyond the limits on its own (a 4 Hz shake): weakening the top layer cannot fix it
#[test]
fn a_lower_layer_beyond_the_limits_cannot_be_fixed_by_weakening_the_top() {
    let cs = sheets(3);
    let lower = shake(0, 0, 120, 15, 36);
    let top = pull(1, 150, 151, 151, 0, 1);
    assert!(!check(
        &cs,
        &hand(),
        &[lower.clone(), top.clone()],
        config(),
        &limits()
    )
    .expect("check")
    .is_empty());
    assert_eq!(
        within_limits(&cs, &hand(), &[lower, top], config(), &limits()),
        Ok(None)
    );
}

/// A lower layer pulls body 0 to 0.4 m (within the bound alone) and lets go at step 60; the top layer takes the
/// body at step 62 while it is still out and pulls it farther. Only the top layer is weakened: the scale returned
/// passes with the lower layer at full strength (1/4 here), and the larger scale at which weakening both layers
/// would pass (1/2) does not pass with the lower layer at full strength
#[test]
fn weakening_leaves_a_lower_layer_on_the_same_body_at_full_strength() {
    let cs = sheets(1);
    let lower = pull(0, 0, 30, 60, 0, 40);
    let top = pull(0, 62, 82, 102, 35, 80);
    assert!(check(
        &cs,
        &hand(),
        &[lower.clone(), InputLog::new()],
        config(),
        &limits()
    )
    .expect("check")
    .is_empty());
    assert!(!check(
        &cs,
        &hand(),
        &[lower.clone(), top.clone()],
        config(),
        &limits()
    )
    .expect("check")
    .is_empty());
    let (sent, scale) = within_limits(
        &cs,
        &hand(),
        &[lower.clone(), top.clone()],
        config(),
        &limits(),
    )
    .expect("check")
    .expect("weakened");
    assert_eq!(scale, Fix128::from_ratio(1, 4));
    assert!(
        check(&cs, &hand(), &[lower.clone(), sent], config(), &limits())
            .expect("check")
            .is_empty()
    );
    // what weakening both would accept: half on both passes, half on the top alone does not
    let half = Fix128::from_ratio(1, 2);
    let weak = |l: &InputLog<Touch>| scaled(l, half);
    assert!(check(
        &cs,
        &hand(),
        &[weak(&lower), weak(&top)],
        config(),
        &limits()
    )
    .expect("check")
    .is_empty());
    assert!(
        !check(&cs, &hand(), &[lower, weak(&top)], config(), &limits())
            .expect("check")
            .is_empty()
    );
}

/// A shake of 8 Hz whose centre swings stay under the swing threshold is no reversal at all; the same shake counted
/// with no threshold is far over the rate
#[test]
fn a_jitter_under_the_swing_is_not_a_reversal() {
    let cs = sheets(2);
    let jitter = shake(1, 0, STEPS, 8, 2);
    assert!(check(
        &cs,
        &hand(),
        std::slice::from_ref(&jitter),
        config(),
        &limits()
    )
    .expect("check")
    .is_empty());
    let no_threshold =
        TouchLimits::new(6, Fix128::from_ratio(6, 10), Fix128::ZERO, RATE).expect("limits");
    let found = check(&cs, &hand(), &[jitter], config(), &no_threshold).expect("check");
    assert!(
        matches!(found.as_slice(), [Violation::TooFast { body: 1, reversals }] if *reversals > 6),
        "{found:?}"
    );
}

/// A shake of 26 steps a cycle over the whole playback: a turn of the drag every 13 steps, so 5 turns lie within
/// any 60 steps (a span of 52) and 6 never do (65), with 8 steps of room either way for the body's turns to lag the
/// drag's. At a limit of 5 it passes; at 4 it is reported with 5; counted in windows of 120 steps it has 10 (a span
/// of 117) and never 11 (130)
#[test]
fn reversals_are_counted_in_any_one_second_window() {
    let cs = sheets(2);
    let top = shake(1, 0, STEPS, 26, 36);
    let at_most = |n: usize, window: u64| {
        TouchLimits::new(
            n,
            Fix128::from_ratio(6, 10),
            Fix128::from_ratio(1, 32),
            window,
        )
        .expect("limits")
    };
    let run = |l: &TouchLimits| {
        check(&cs, &hand(), std::slice::from_ref(&top), config(), l).expect("check")
    };
    assert!(run(&at_most(5, RATE)).is_empty());
    assert_eq!(
        run(&at_most(4, RATE)),
        vec![Violation::TooFast {
            body: 1,
            reversals: 5
        }]
    );
    assert_eq!(
        run(&at_most(9, 2 * RATE)),
        vec![Violation::TooFast {
            body: 1,
            reversals: 10
        }]
    );
}

/// Limits out of range are refused; a playback with no bodies or no touches has nothing to report
#[test]
fn limits_out_of_range_are_refused() {
    let d = Fix128::from_ratio(6, 10);
    let s = Fix128::from_ratio(1, 32);
    assert_eq!(TouchLimits::new(6, d, s, 0), Err(LayerError::BadLimits));
    assert_eq!(TouchLimits::new(6, -d, s, RATE), Err(LayerError::BadLimits));
    assert_eq!(TouchLimits::new(6, d, -s, RATE), Err(LayerError::BadLimits));
    assert!(TouchLimits::new(0, Fix128::ZERO, Fix128::ZERO, 1).is_ok());
    assert_eq!(
        check(&[], &hand(), &[], config(), &limits()),
        Ok(Vec::new())
    );
    assert_eq!(
        within_limits(&sheets(2), &hand(), &[], config(), &limits()),
        Ok(Some((InputLog::new(), Fix128::ONE)))
    );
}
