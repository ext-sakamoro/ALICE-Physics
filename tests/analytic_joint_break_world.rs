//! Joint breaking inside `PhysicsWorld::step` (AUD-A-S1W6-009).
//!
//! The force a joint breaks on is the reaction force its joint solve
//! transmitted in the substep, `|Σ λ · n| / h²` (Macklin, Müller, Chentanez
//! 2016, eq. 10 and 18), judged after the solve.
//!
//! Closed form, one joint: a body of mass `m` hung from a static anchor by a
//! rigid ball joint, starting at rest on the anchor, is moved by one gravity
//! substep of length `h` to `g · h²` below it; the solve pulls it back with
//! `λ = g h² / (1/m)`, a reaction force of `m · g`. With `h = 1/64`, `g = 8` and
//! `m = 2` every quantity is dyadic, so the force is exactly 16 N, and 16 N
//! again on every later substep (the solve leaves the body at rest).
//!
//! Chain (static S, then A and B of 2 kg each): gravity moves A and B by the
//! same amount, so the lower joint A–B has no gap before the solve; its load
//! reaches it through the upper joint during the solve. The lower joint
//! carries `m · g = 16 N` and the upper `2 m · g = 32 N` once the chain hangs at
//! rest (a single Gauss–Seidel sweep transmits less in the first substep and
//! overshoots before settling, measured below).
//!
//! A joint whose force exceeds `break_force` (strictly) is removed after the
//! solve that measured it (its correction for that substep is applied) and
//! reported as a `JointBreakEvent`; the body then falls freely.

use alice_physics::solver::SolverBackend;
use alice_physics::{
    BallJoint, Fix128, Joint, JointBreakEvent, PhysicsConfig as SolverConfig, PhysicsWorld,
    RigidBody, Vec3Fix,
};

fn world(backend: SolverBackend) -> PhysicsWorld {
    let config = SolverConfig {
        substeps: 1,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-8), Fix128::ZERO),
        damping: Fix128::ONE,
        solver_backend: backend,
        ..SolverConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
    w
}

fn h() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn hung(backend: SolverBackend, break_force: Fix128) -> PhysicsWorld {
    let mut w = world(backend);
    w.add_joint(Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(break_force),
    ));
    w
}

#[test]
fn xpbd_joint_below_its_load_breaks_and_reports_m_g() {
    let mut w = hung(SolverBackend::Xpbd, Fix128::from_int(15));
    w.step(h());
    assert!(w.joints.is_empty(), "16 N > 15 N: the joint is removed");
    let events: Vec<JointBreakEvent> = w.events.joint_break_events().to_vec();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].index, 0);
    assert_eq!(events[0].force, Fix128::from_int(16), "m g exactly");
    assert_eq!(
        events[0].joint,
        Joint::Ball(
            BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
                .with_break_force(Fix128::from_int(15))
        )
    );
    // broken after the solve that measured 16 N: that substep's correction was
    // applied, so the body is still on the anchor after the first step
    assert_eq!(w.bodies[1].position.y, Fix128::ZERO);
    // and it keeps falling: after 64 steps (1 s) it is far below the anchor
    for _ in 0..63 {
        w.step(h());
    }
    assert!(
        w.bodies[1].position.y < Fix128::from_int(-3),
        "{:?}",
        w.bodies[1].position.y
    );
    assert!(
        w.events.joint_break_events().is_empty(),
        "events are per step"
    );
}

#[test]
fn xpbd_joint_at_or_above_its_load_holds_for_a_second() {
    for limit in [16, 17] {
        let mut w = hung(SolverBackend::Xpbd, Fix128::from_int(limit));
        for step in 0..64 {
            w.step(h());
            assert_eq!(
                w.joints.len(),
                1,
                "break_force {limit} N holds at step {step} (strict >)"
            );
            assert!(w.events.joint_break_events().is_empty());
        }
        assert_eq!(w.bodies[1].position, Vec3Fix::ZERO, "held on the anchor");
    }
}

#[test]
fn unbreakable_joint_is_never_removed() {
    let mut w = world(SolverBackend::Xpbd);
    w.add_joint(Joint::Ball(BallJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
    for _ in 0..64 {
        w.step(h());
    }
    assert_eq!(w.joints.len(), 1);
}

#[test]
fn tgs_backend_breaks_through_the_same_check() {
    let mut weak = hung(SolverBackend::Tgs, Fix128::from_int(15));
    weak.step(h());
    assert!(
        weak.joints.is_empty(),
        "the TGS joint projection also breaks"
    );
    let force = weak.events.joint_break_events()[0].force.to_f64();
    assert!(
        (force - 16.0).abs() < 1e-9,
        "m g on the TGS path too: {force}"
    );

    let mut strong = hung(SolverBackend::Tgs, Fix128::from_int(17));
    for _ in 0..64 {
        strong.step(h());
    }
    assert_eq!(strong.joints.len(), 1);
}

#[test]
fn every_overloaded_joint_breaks_in_the_same_substep_and_the_motor_goes_with_it() {
    let mut w = world(SolverBackend::Xpbd);
    w.add_body(RigidBody::new(
        Vec3Fix::from_int(3, 0, 0),
        Fix128::from_int(2),
    ));
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(3, 0, 0)));
    let weak = |a: usize, b: usize| {
        Joint::Ball(
            BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(Fix128::ONE),
        )
    };
    w.add_joint(weak(0, 1)); // 16 N > 1 N
    w.add_joint(Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_compliance(Fix128::ONE),
    )); // unbreakable
    w.add_joint(weak(3, 2)); // 16 N > 1 N
    let motor = w.add_joint_motor(
        2,
        alice_physics::motor::PdController::new(Fix128::ONE, Fix128::ZERO, Fix128::ONE),
    );
    w.step(h());
    // both weak joints go, from the highest index down; the unbreakable one stays
    let indices: Vec<usize> = w
        .events
        .joint_break_events()
        .iter()
        .map(|e| e.index)
        .collect();
    assert_eq!(indices, vec![2, 0]);
    assert_eq!(w.joints.len(), 1);
    assert_eq!(w.joints[0].break_force(), None);
    assert!(
        !w.disable_joint_motor(motor),
        "the motor on the broken joint is removed"
    );
}

#[test]
fn a_sleeping_body_left_hanging_by_a_broken_joint_wakes_and_falls() {
    // hang the body on an unbreakable joint until its island sleeps
    let mut w = hung(SolverBackend::Xpbd, Fix128::from_int(1_000_000));
    let mut slept = false;
    for _ in 0..2_000 {
        w.step(h());
        if w.is_sleeping(1) {
            slept = true;
            break;
        }
    }
    assert!(slept, "a body held at rest on its anchor falls asleep");
    // lower the threshold and pull the body off the anchor by writing the pub fields
    // directly (neither write wakes anything): the joint now carries far more than 1 N
    w.joints[0] = Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(Fix128::ONE),
    );
    w.bodies[1].position = Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(-1, 64), Fix128::ZERO);
    w.step(h());
    assert!(w.joints.is_empty(), "the joint breaks");
    assert!(
        !w.is_sleeping(1),
        "the body it held is awake (the solve that broke the joint moved it)"
    );
    for _ in 0..63 {
        w.step(h());
    }
    // the breaking substep's solve pulled the body back onto the anchor, so it
    // starts at 1/64 m per 1/64 s upward; free fall still takes it far down
    assert!(
        w.bodies[1].position.y < Fix128::from_int(-1),
        "and falls instead of staying frozen in the air: {:?}",
        w.bodies[1].position.y
    );
}

#[test]
fn a_joint_breaking_in_the_middle_swaps_the_last_joint_into_its_index() {
    // joints 0..=3 on four bodies hung from the anchor; only joint 1 is weak
    let mut w = world(SolverBackend::Xpbd);
    for _ in 0..3 {
        w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
    }
    let strong = Fix128::from_int(1_000);
    let joint = |b: usize, limit: Fix128| {
        Joint::Ball(BallJoint::new(0, b, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(limit))
    };
    w.add_joint(joint(1, strong));
    w.add_joint(joint(2, Fix128::ONE)); // 16 N > 1 N
    w.add_joint(joint(3, strong));
    w.add_joint(joint(4, strong));
    let motor_on_last = w.add_joint_motor(
        3,
        alice_physics::motor::PdController::new(Fix128::ONE, Fix128::ZERO, Fix128::ONE),
    );
    w.step(h());
    let events = w.events.joint_break_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].index, 1, "reported at the index it had");
    assert_eq!(events[0].joint.bodies(), (0, 2));
    // swap-remove: the last joint (to body 4) now sits at index 1
    let order: Vec<usize> = w.joints.iter().map(|j| j.bodies().1).collect();
    assert_eq!(order, vec![1, 4, 3]);
    // its motor followed it to index 1 (still there, so it can be disabled)
    assert!(w.disable_joint_motor(motor_on_last));
    // the bodies left on unbroken joints are still held on the anchor
    assert_eq!(w.bodies[1].position, Vec3Fix::ZERO);
    assert_eq!(w.bodies[4].position, Vec3Fix::ZERO);
}

fn chain(
    backend: SolverBackend,
    substeps: usize,
    upper: Fix128,
    lower: Fix128,
    sleep: bool,
) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        substeps,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-8), Fix128::ZERO),
        damping: Fix128::ONE,
        solver_backend: backend,
        ..SolverConfig::default()
    });
    if !sleep {
        w.set_sleep_config(alice_physics::sleeping::SleepConfig {
            frames_to_sleep: u32::MAX,
            ..alice_physics::sleeping::SleepConfig::default()
        });
    }
    w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
    let ball = |a: usize, b: usize, limit: Fix128| {
        Joint::Ball(BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(limit))
    };
    w.add_joint(ball(0, 1, upper));
    w.add_joint(ball(1, 2, lower));
    w
}

const CONFIGS: [(SolverBackend, usize); 4] = [
    (SolverBackend::Xpbd, 1),
    (SolverBackend::Xpbd, 4),
    (SolverBackend::Tgs, 1),
    (SolverBackend::Tgs, 4),
];

/// First substep of the chain, one Gauss–Seidel sweep: the upper joint pulls A
/// back by `g h²` (λ = m g h², 16 N); then the lower joint closes the `g h²` gap
/// to B with `w = 2/m` (λ = m g h² / 2, 8 N). Both are exact in Fix128.
#[test]
fn the_lower_joint_of_a_chain_carries_load_passed_down_in_the_same_sweep() {
    for (backend, substeps) in CONFIGS {
        // thresholds just under the first-sweep forces: both break in substep 1
        let mut w = chain(
            backend,
            substeps,
            Fix128::from_int(15),
            Fix128::from_int(7),
            true,
        );
        w.step(h());
        let mut forces: Vec<(usize, Fix128)> = w
            .events
            .joint_break_events()
            .iter()
            .map(|e| (e.joint.bodies().1, e.force))
            .collect();
        forces.sort();
        assert_eq!(
            forces,
            vec![(1, Fix128::from_int(16)), (2, Fix128::from_int(8))],
            "{backend:?} substeps {substeps}: the lower joint had no gap before the solve and still carries 8 N"
        );
    }
}

/// Once the chain hangs at rest (sleep off so no wake-up replays the start), the
/// lower joint carries m g = 16 N and the upper 2 m g = 32 N.
#[test]
fn a_hanging_chain_settles_to_m_g_below_and_two_m_g_above() {
    let huge = Fix128::from_int(1_000_000);
    for (backend, substeps) in CONFIGS {
        for (joint, load) in [(1usize, 16i64), (0, 32)] {
            for (offset, breaks) in [(-1i64, true), (1, false)] {
                let mut w = chain(backend, substeps, huge, huge, false);
                for _ in 0..64 {
                    w.step(h());
                }
                // the transient has settled: set the threshold 0.1 N below / above the load
                let limit = Fix128::from_int(load) + Fix128::from_ratio(offset, 10);
                let j = &mut w.joints[joint];
                *j = match *j {
                    Joint::Ball(b) => Joint::Ball(b.with_break_force(limit)),
                    _ => unreachable!(),
                };
                for _ in 0..64 {
                    w.step(h());
                }
                let broke = w.joints.len() < 2;
                assert_eq!(
                    broke, breaks,
                    "{backend:?} substeps {substeps}: joint {joint} at {load} N {offset:+} × 0.1 N"
                );
            }
        }
    }
}

/// From rest a single sweep per substep overshoots before settling (measured:
/// lower 8, 16, 20, 20, 18 … N, upper 16, 32, 40, 40, 36 … N on every
/// backend and substep count). A threshold under the static load breaks; one
/// above the overshoot holds.
#[test]
fn a_chain_breaks_under_its_static_load_and_holds_above_its_start_up_peak() {
    let huge = Fix128::from_int(1_000_000);
    for (backend, substeps) in CONFIGS {
        for (upper, lower, broken_bodies) in [
            (huge, Fix128::from_int(15), vec![2usize]), // lower under m g
            (Fix128::from_int(31), huge, vec![1]),      // upper under 2 m g
        ] {
            let mut w = chain(backend, substeps, upper, lower, true);
            let mut seen = Vec::new();
            for _ in 0..256 {
                w.step(h());
                seen.extend(
                    w.events
                        .joint_break_events()
                        .iter()
                        .map(|e| e.joint.bodies().1),
                );
            }
            assert_eq!(seen, broken_bodies, "{backend:?} substeps {substeps}");
        }
        let mut w = chain(
            backend,
            substeps,
            Fix128::from_int(41),
            Fix128::from_int(21),
            true,
        );
        for _ in 0..256 {
            w.step(h());
        }
        assert_eq!(
            w.joints.len(),
            2,
            "{backend:?} substeps {substeps}: above the peak holds"
        );
    }
}

/// After a break the islands no longer join the two bodies within the same
/// step (the sleep update at the end of the step and a snapshot taken after it
/// see the remaining joints only), while the unbroken joint still joins its pair.
#[test]
fn a_broken_joint_no_longer_joins_its_bodies_in_the_islands() {
    let huge = Fix128::from_int(1_000_000);
    let mut w = chain(SolverBackend::Xpbd, 1, huge, Fix128::from_int(7), true);
    assert_eq!(
        w.islands.find(1),
        w.islands.find(2),
        "joined before the break"
    );
    w.step(h());
    assert_eq!(w.joints.len(), 1, "8 N > 7 N: the lower joint broke");
    assert_ne!(
        w.islands.find(1),
        w.islands.find(2),
        "the broken joint no longer joins A and B"
    );
    assert_eq!(
        w.islands.find(0),
        w.islands.find(1),
        "the upper joint still joins S and A"
    );
}
