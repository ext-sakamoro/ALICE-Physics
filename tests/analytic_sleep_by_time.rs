//! Oracle: a body falls asleep after an idle **time**, not after an idle step
//! count.
//!
//! # What is measured
//!
//! `SleepConfig::frames_to_sleep` (default 60, "1 second at 60fps") is compared
//! with a count of idle steps, whatever `dt` is. A simulation stepped at
//! `dt = 1/3000 s` therefore puts an idle body to sleep after 0.02 s, fifty
//! times too early, and one stepped at `dt = 1/6 s` after 10 s. The closed form
//! of the intended behaviour is that a body idle from `t = 0` sleeps at the
//! first step whose elapsed time `Σ dt` reaches 1 s, and no earlier.
//!
//! The rule: a body sleeps at the first step whose accumulated idle time
//! reaches `SleepConfig::time_to_sleep` within half a step
//! (`idle + dt / 2 >= time_to_sleep`), so at `dt = 1/60` the default 1 s is
//! still exactly 60 steps (the control), at `dt = 1/3000` it is 3000 and at
//! `dt = 1/6` it is 6; a sequence of mixed steps sleeps when their sum reaches
//! 1 s, wherever the step boundaries fall.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::sleeping::SleepConfig;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::PlaneCollider;
use sha2::{Digest as _, Sha256};

/// A world with one idle body (no gravity, zero velocity) stepped at `dt`:
/// the first step after which it is asleep (1-based), or `None` within `max`.
fn steps_until_asleep(dt: Fix128, max: usize) -> Option<usize> {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    let i = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        Fix128::ONE,
    );
    for n in 1..=max {
        w.step(dt);
        if w.is_sleeping(i) {
            return Some(n);
        }
    }
    None
}

/// Control: at `dt = 1/60` the default (60 frames) is 1 s either way.
#[test]
fn at_one_sixtieth_an_idle_body_sleeps_after_sixty_steps() {
    assert_eq!(steps_until_asleep(Fix128::from_ratio(1, 60), 200), Some(60));
}

/// An idle body sleeps when its idle time reaches 1 s: 3000 steps at
/// `dt = 1/3000`, 6 steps at `dt = 1/6`; never after 60 steps of 1/3000 s
/// (0.02 s) nor before 6 steps of 1/6 s.
#[test]
fn an_idle_body_sleeps_after_one_second_whatever_the_step() {
    let fine = steps_until_asleep(Fix128::from_ratio(1, 3000), 4000);
    assert_eq!(fine, Some(3000), "dt = 1/3000: want asleep at 1 s");
    let coarse = steps_until_asleep(Fix128::from_ratio(1, 6), 100);
    assert_eq!(coarse, Some(6), "dt = 1/6: want asleep at 1 s");
}

/// Mixed steps: 30 steps of 1/60 s (0.5 s), then steps of 1/120 s; the body
/// sleeps at the step whose sum reaches 1 s within half a step: the 60th of
/// the fine ones (0.5 + 60/120 = 1 s), not the 59th (0.9917 + 0.0042 < 1).
#[test]
fn a_sequence_of_mixed_steps_sleeps_when_their_sum_reaches_one_second() {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    let i = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        Fix128::ONE,
    );
    for _ in 0..30 {
        w.step(Fix128::from_ratio(1, 60));
    }
    assert!(!w.is_sleeping(i), "0.5 s idle: awake");
    let mut fine = 0;
    while !w.is_sleeping(i) {
        w.step(Fix128::from_ratio(1, 120));
        fine += 1;
        assert!(fine <= 120, "never slept");
    }
    assert_eq!(fine, 60, "asleep when the sum of the steps reaches 1 s");
    // a step of 1/2 s after 0.7 s idle reaches 1 s within half a step: asleep
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    let j = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        Fix128::ONE,
    );
    for _ in 0..42 {
        w.step(Fix128::from_ratio(1, 60));
    }
    assert!(!w.is_sleeping(j), "0.7 s idle: awake");
    w.step(Fix128::from_ratio(1, 2));
    assert!(
        w.is_sleeping(j),
        "0.7 + 0.5 s: asleep (1 s reached within half a step)"
    );
}

/// Closed form of the rule for a body idle from `t = 0` at a fixed step `p/q` s and a rule
/// `T = a/b` s: the first `k` with `k·dt + dt/2 >= T`, i.e. the smallest integer `k` with
/// `(2k + 1)·p·b >= 2·a·q`, computed here in integers, apart from the solver.
fn closed_form_steps(p: i64, q: i64, a: i64, b: i64) -> usize {
    let needed = 2 * a * q;
    let mut k = 0i64;
    while (2 * k + 1) * p * b < needed {
        k += 1;
    }
    usize::try_from(k).expect("positive")
}

fn steps_until_asleep_with(dt: Fix128, time_to_sleep: Fix128, max: usize) -> Option<usize> {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        time_to_sleep,
        ..SleepConfig::default()
    });
    let i = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE),
        Fix128::ONE,
    );
    for n in 1..=max {
        w.step(dt);
        if w.is_sleeping(i) {
            return Some(n);
        }
    }
    None
}

/// oracle: at ordinary and odd steps (1/120, 1/30, 1/64, 1/70, 3/10, 1/60 s) and for two
/// rules (1 s and 1/2 s) the body sleeps at exactly the closed-form step, not one earlier.
#[test]
fn the_sleep_step_is_the_closed_form_at_every_step_size() {
    for (p, q) in [(1i64, 120i64), (1, 30), (1, 64), (1, 70), (3, 10), (1, 60)] {
        for (a, b) in [(1i64, 1i64), (1, 2)] {
            let want = closed_form_steps(p, q, a, b);
            let got = steps_until_asleep_with(
                Fix128::from_ratio(p, q),
                Fix128::from_ratio(a, b),
                want + 50,
            );
            assert_eq!(got, Some(want), "dt = {p}/{q} s, T = {a}/{b} s");
        }
    }
    // the closed form itself: 1/120 → 120, 1/30 → 30, 3/10 → 3 (0.9 + 0.15 >= 1);
    // for T = 1/2: 1/120 → 60, 3/10 → 2 (0.6 + 0.15 >= 0.5 while 0.3 + 0.15 < 0.5)
    assert_eq!(closed_form_steps(1, 120, 1, 1), 120);
    assert_eq!(closed_form_steps(1, 30, 1, 1), 30);
    assert_eq!(closed_form_steps(3, 10, 1, 1), 3);
    assert_eq!(closed_form_steps(1, 120, 1, 2), 60);
    assert_eq!(closed_form_steps(3, 10, 1, 2), 2);
}

/// SHA-256 of `serialize_state` after the pinned scene; every CI target agrees. Pinned in
/// `PHYSICS_SEMANTICS_PINS` as "sleep by time": it moves only with the sleep rule, the
/// solver or the blob format, each an announced change.
const SLEEP_BY_TIME_SHA256: &str =
    "cceb5bd2e478cc97095b26732e754178eb4097dafe10fb4f9dd44a8c6653fd5a";

/// The pinned scene: four spheres dropped on a plane at `dt = 1/30` s with a 1/2 s rule,
/// 400 steps (13.3 s): they land at different times and sleep at different steps.
fn pinned_scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_sleep_config(SleepConfig {
        time_to_sleep: Fix128::from_ratio(1, 2),
        ..SleepConfig::default()
    });
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        Fix128::ZERO,
    )));
    for k in 0..4i64 {
        let x = Fix128::from_int(3 * k);
        let y = Fix128::from_int(2 + 3 * k);
        w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::new(x, y, Fix128::ZERO), Fix128::ONE),
            Fix128::ONE,
        );
    }
    for _ in 0..400 {
        w.step(Fix128::from_ratio(1, 30));
    }
    w
}

/// golden (determinism): the pinned scene's blob hashes to the frozen value; its bodies
/// are all asleep at the end.
#[test]
fn sleep_by_time_scene_has_its_pinned_sha256() {
    let w = pinned_scene();
    assert!(
        (0..4).all(|i| w.is_sleeping(i)),
        "every sphere sleeps within 13 s"
    );
    let hex: String = Sha256::digest(w.serialize_state())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("sleep-by-time scene sha256 {hex}");
    assert_eq!(hex, SLEEP_BY_TIME_SHA256);
}
