//! Oracles for the wiring of `event::EventCollector::has_events` and
//! `sleeping::IslandManager::build_islands` (`examples/world_events_and_islands.rs`).
//!
//! Expected values are hand derived:
//! * `has_events() == !contact_queue.is_empty() || !trigger_queue.is_empty()`
//! * islands are the connected components of the union graph; they are listed
//!   in order of their smallest body index, bodies ascending inside each;
//!   `all_sleeping` is the AND of per-body sleeping flags.
#![allow(clippy::disallowed_methods)]

use alice_physics::event::{ContactEventType, EventCollector};
use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sleeping::{IslandManager, SleepConfig};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn report(e: &mut EventCollector, a: usize, b: usize) {
    e.report_contact(
        a,
        b,
        Vec3Fix::UNIT_Y,
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ZERO,
    );
}

#[test]
fn has_events_is_the_or_of_both_queues() {
    let mut e = EventCollector::new();
    assert!(!e.has_events());
    e.begin_frame();
    report(&mut e, 3, 1);
    assert!(e.has_events());
    e.drain_contact_events();
    assert!(!e.has_events());
    e.report_trigger(0, 2);
    assert!(e.has_events());
    assert!(e.contact_events().is_empty());
    e.drain_trigger_events();
    assert!(!e.has_events());
}

#[test]
fn has_events_through_world_step_begin_then_end() {
    let cfg = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut w = PhysicsWorld::new(cfg);
    let r = Fix128::ONE;
    w.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), r);
    w.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ),
        r,
    );
    assert!(!w.events.has_events());
    w.step(Fix128::from_ratio(1, 60));
    assert!(w.events.has_events());
    assert_eq!(w.events.contact_events().len(), 1);
    assert_eq!(
        w.events.contact_events()[0].event_type,
        ContactEventType::Begin
    );
    // far apart: a step with no overlap and no previous pair has no events
    let mut far = PhysicsWorld::new(cfg);
    far.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), r);
    far.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(100, 0, 0), Fix128::ONE),
        r,
    );
    far.step(Fix128::from_ratio(1, 60));
    assert!(!far.events.has_events());
}

fn components(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    // independent DFS oracle
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for s in 0..n {
        if seen[s] {
            continue;
        }
        let mut stack = vec![s];
        let mut comp = Vec::new();
        seen[s] = true;
        while let Some(u) = stack.pop() {
            comp.push(u);
            for &(a, b) in edges {
                for (x, y) in [(a, b), (b, a)] {
                    if x == u && !seen[y] {
                        seen[y] = true;
                        stack.push(y);
                    }
                }
            }
        }
        comp.sort_unstable();
        out.push(comp);
    }
    out
}

#[test]
fn build_islands_equals_connected_components() {
    let cases: &[(usize, &[(usize, usize)])] = &[
        (0, &[]),
        (1, &[]),
        (5, &[(0, 1), (2, 3)]),
        (6, &[(5, 0), (0, 3), (3, 5), (1, 2)]),
        (7, &[(6, 5), (5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]),
        (4, &[(0, 0), (1, 1)]),
    ];
    for &(n, edges) in cases {
        let mut m = IslandManager::new(n, SleepConfig::default());
        for &(a, b) in edges {
            m.union(a, b);
        }
        let got: Vec<Vec<usize>> = m.build_islands().into_iter().map(|i| i.bodies).collect();
        assert_eq!(got, components(n, edges), "n={n} edges={edges:?}");
    }
}

#[test]
fn island_all_sleeping_is_and_of_members() {
    let mut m = IslandManager::new(4, SleepConfig::default());
    m.union(0, 1);
    m.union(2, 3);
    m.sleep_data[0].state = alice_physics::sleeping::SleepState::Sleeping;
    m.sleep_data[1].state = alice_physics::sleeping::SleepState::Sleeping;
    m.sleep_data[2].state = alice_physics::sleeping::SleepState::Sleeping;
    let isl = m.build_islands();
    assert_eq!(isl.len(), 2);
    assert!(isl[0].all_sleeping, "{{0,1}} both asleep");
    assert!(!isl[1].all_sleeping, "{{2,3}}: body 3 awake");
}

#[test]
fn world_joints_define_islands() {
    let cfg = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut w = PhysicsWorld::new(cfg);
    for i in 0..4 {
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(i, 0, 0),
            Fix128::ONE,
        ));
    }
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(10, 0, 0)));
    for (a, b) in [(0usize, 1usize), (2, 3)] {
        w.add_joint(Joint::Ball(BallJoint::new(
            a,
            b,
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::ZERO,
        )));
    }
    w.step(Fix128::from_ratio(1, 60));
    let isl = w.islands.build_islands();
    let bodies: Vec<_> = isl.iter().map(|i| i.bodies.clone()).collect();
    assert_eq!(bodies, vec![vec![0, 1], vec![2, 3], vec![4]]);
    let sleeping: Vec<_> = isl.iter().map(|i| i.all_sleeping).collect();
    assert_eq!(
        sleeping,
        vec![false, false, true],
        "static body 4 is asleep"
    );
}

#[test]
fn trigger_exit_keeps_the_roles_of_the_enter_event() {
    let mut e = EventCollector::new();
    e.begin_frame();
    e.report_trigger(5, 2);
    e.end_frame();
    let enter = e.trigger_events()[0];
    assert_eq!(
        (enter.trigger_body, enter.other_body, enter.entered),
        (5, 2, true)
    );
    e.begin_frame();
    e.end_frame();
    let exit = e.trigger_events()[0];
    assert!(!exit.entered);
    assert_eq!(
        (exit.trigger_body, exit.other_body),
        (5, 2),
        "exit must name the same trigger body"
    );
}

fn kinds(e: &EventCollector) -> Vec<(usize, usize, ContactEventType)> {
    e.contact_events()
        .iter()
        .map(|c| (c.body_a, c.body_b, c.event_type))
        .collect()
}

#[test]
fn contact_lifecycle_begin_persist_end_then_silence() {
    let mut e = EventCollector::new();
    // frame 1: reported in reversed order, normalized to (1, 3); duplicate ignored
    e.begin_frame();
    report(&mut e, 3, 1);
    report(&mut e, 1, 3);
    e.end_frame();
    assert_eq!(kinds(&e), vec![(1, 3, ContactEventType::Begin)]);
    // frame 2: persists
    e.begin_frame();
    report(&mut e, 1, 3);
    e.end_frame();
    assert_eq!(kinds(&e), vec![(1, 3, ContactEventType::Persist)]);
    // frame 3: gone -> exactly one End
    e.begin_frame();
    e.end_frame();
    assert_eq!(kinds(&e), vec![(1, 3, ContactEventType::End)]);
    // frame 4: nothing remembered -> silence
    e.begin_frame();
    e.end_frame();
    assert!(kinds(&e).is_empty());
    assert!(!e.has_events());
    // frame 5: comes back -> Begin again, not Persist
    e.begin_frame();
    report(&mut e, 1, 3);
    e.end_frame();
    assert_eq!(kinds(&e), vec![(1, 3, ContactEventType::Begin)]);
}

#[test]
fn trigger_enter_once_per_frame_and_not_while_overlapping() {
    let mut e = EventCollector::new();
    e.begin_frame();
    e.report_trigger(4, 7);
    e.report_trigger(7, 4); // same pair, same frame
    e.end_frame();
    assert_eq!(e.trigger_events().len(), 1);
    // still overlapping next frame: no enter, no exit
    e.begin_frame();
    e.report_trigger(7, 4);
    e.end_frame();
    assert!(e.trigger_events().is_empty());
    // leaves, then stays gone: exit once
    e.begin_frame();
    e.end_frame();
    assert_eq!(e.trigger_events().len(), 1);
    e.begin_frame();
    e.end_frame();
    assert!(e.trigger_events().is_empty());
}

#[test]
fn trigger_roles_are_stable_for_all_report_orders() {
    // enter order x persist-frame order: 4 combinations; the exit event must
    // carry exactly the roles of the enter event
    for (t, o) in [(5usize, 2usize), (2, 5)] {
        for (pt, po) in [(5usize, 2usize), (2, 5)] {
            let mut e = EventCollector::new();
            e.begin_frame();
            e.report_trigger(t, o);
            e.end_frame();
            let enter = e.trigger_events()[0];
            assert_eq!(
                (enter.trigger_body, enter.other_body, enter.entered),
                (t, o, true)
            );
            e.begin_frame();
            e.report_trigger(pt, po);
            e.end_frame();
            assert!(e.trigger_events().is_empty(), "persist frame emits nothing");
            e.begin_frame();
            e.end_frame();
            let exit = e.trigger_events()[0];
            assert_eq!(
                (exit.trigger_body, exit.other_body, exit.entered),
                (t, o, false),
                "enter=({t},{o}) persist=({pt},{po})"
            );
        }
    }
}
