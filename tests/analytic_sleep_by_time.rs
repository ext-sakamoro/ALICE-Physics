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
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

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
