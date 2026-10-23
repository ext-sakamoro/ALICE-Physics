//! Oracle: the overflow flag does not depend on the broad-phase kind.
//!
//! # What is measured
//!
//! `Broadphase` promises that every kind steps a world to bit-identical results. The
//! world's state includes the sticky overflow flag (`PhysicsWorld::overflow_detected`,
//! one byte of `serialize_state`). The narrow-phase sets that flag when the sum of a
//! candidate pair's collision radii leaves the `Fix128` range, and it does so before it
//! looks at the pair's boxes. `Broadphase::Bvh` hands the narrow-phase a superset of the
//! overlapping pairs (its leaf boxes are quantised to `i32`), `Broadphase::Hybrid` only
//! the pairs whose boxes overlap. A pair whose radius sum overflows while its boxes do
//! not meet could therefore set the flag under one kind and not under another.
//!
//! The oracle sweeps two bodies along x over extreme positions and radii (no velocity,
//! no gravity, so nothing but the contact path can raise the flag), steps once under
//! each kind and requires the same `serialize_state` bytes. It also requires that the
//! sweep raised the flag at least once, so the comparison is not between worlds that
//! never reach the overflow path.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::sleeping::SleepState;
use alice_physics_core::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};

const KINDS: [Broadphase; 3] = [Broadphase::Bvh, Broadphase::DynamicTree, Broadphase::Hybrid];

fn x_at(x: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO)
}

/// Two still bodies on the x axis with the given collision radii, stepped once.
fn stepped(kind: Broadphase, xa: Fix128, xb: Fix128, ra: Fix128, rb: Fix128) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
    w.add_body_with_radius(RigidBody::new(x_at(xa), Fix128::ONE), ra);
    w.add_body_with_radius(RigidBody::new(x_at(xb), Fix128::ONE), rb);
    w.step(Fix128::from_ratio(1, 60));
    w
}

/// The raw 128-bit value of `v`.
fn raw(v: Fix128) -> i128 {
    (i128::from(v.hi) << 64) | i128::from(v.lo)
}

/// Whether a body at `x` with collision radius `r` has a box that leaves the `Fix128`
/// range: the sphere box `x ± r`, or the dynamic tree's proxy, which adds a margin of
/// one half (`DynamicAabbTree`'s default) on each side. Either wraps into an inverted box.
fn box_wraps(x: Fix128, r: Fix128) -> bool {
    let half = raw(Fix128::from_ratio(1, 2));
    let hi = raw(x).checked_add(raw(r)).and_then(|v| v.checked_add(half));
    let lo = raw(x).checked_sub(raw(r)).and_then(|v| v.checked_sub(half));
    hi.is_none() || lo.is_none()
}

fn positions() -> Vec<Fix128> {
    let max = Fix128::from_raw(i64::MAX, u64::MAX);
    let mut p = vec![Fix128::ZERO];
    for e in [31u32, 40, 62] {
        let v = Fix128::from_int(1i64 << e);
        p.push(v);
        p.push(-v);
    }
    p.push(max);
    p.push(-max);
    p
}

fn radii() -> Vec<Fix128> {
    vec![
        Fix128::from_int(1i64 << 62),
        Fix128::from_int((1i64 << 62) + (1i64 << 61)),
        Fix128::from_raw(i64::MAX, u64::MAX),
    ]
}

#[test]
fn the_overflow_flag_and_state_do_not_depend_on_the_broadphase_kind() {
    let ps = positions();
    let rs = radii();
    let mut compared = 0usize;
    let mut flagged = 0usize;
    let mut differing = Vec::new();
    let mut differing_wrapped = 0usize;
    let mut first_unwrapped: Option<String> = None;
    for (i, &xa) in ps.iter().enumerate() {
        for &xb in &ps[i + 1..] {
            for &ra in &rs {
                for &rb in &rs {
                    let worlds: Vec<PhysicsWorld> =
                        KINDS.iter().map(|&k| stepped(k, xa, xb, ra, rb)).collect();
                    let flags: Vec<bool> =
                        worlds.iter().map(PhysicsWorld::overflow_detected).collect();
                    let states: Vec<Vec<u8>> =
                        worlds.iter().map(PhysicsWorld::serialize_state).collect();
                    compared += 1;
                    if flags.iter().any(|&f| f) {
                        flagged += 1;
                    }
                    if states.windows(2).any(|w| w[0] != w[1]) {
                        if box_wraps(xa, ra) || box_wraps(xb, rb) {
                            differing_wrapped += 1;
                        } else if first_unwrapped.is_none() {
                            first_unwrapped = Some(format!(
                                "xa={xa:?} xb={xb:?} ra={ra:?} rb={rb:?} flags={flags:?}"
                            ));
                        }
                        differing.push(format!("xa={xa:?} xb={xb:?} ra={ra:?} rb={rb:?} flags (Bvh, DynamicTree, Hybrid)={flags:?}"));
                    }
                }
            }
        }
    }
    assert!(compared > 0, "the sweep compared nothing");
    assert!(
        flagged > 0,
        "no configuration reached the overflow path ({compared} compared)"
    );
    assert!(
        differing.is_empty(),
        "{} of {compared} configurations step to different states by kind ({differing_wrapped} of them with a sphere or proxy box that leaves the Fix128 range; {flagged} raised the flag), first: {}; first without a wrapped box: {}",
        differing.len(),
        differing.first().map_or("", String::as_str),
        first_unwrapped.as_deref().unwrap_or("none")
    );
}

/// A body at `xa` carrying an ellipsoid collider with radii `(r, r, r)` (so the
/// collider's own world box is `xa ± r`, computed by the collider) and a plain sphere
/// body at `xb`, stepped once.
fn stepped_with_collider(kind: Broadphase, xa: Fix128, r: Fix128, xb: Fix128) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
    let a = w.add_body_with_radius(RigidBody::new(x_at(xa), Fix128::ONE), Fix128::ONE);
    assert!(w.set_body_shape(
        a,
        &Shape::Ellipsoid {
            radii: Vec3Fix::new(r, r, r),
        }
    ));
    w.add_body_with_radius(RigidBody::new(x_at(xb), Fix128::ONE), Fix128::ONE);
    w.step(Fix128::from_ratio(1, 60));
    w
}

/// The same comparison for a body whose collider box (computed by the collider, not
/// the sphere cube) leaves the `Fix128` range: that box comes out inverted and has to
/// be replaced, or the kinds see different candidates again.
#[test]
fn a_collider_box_that_leaves_the_range_steps_the_same_under_every_kind() {
    let ps = positions();
    let rs = radii();
    let mut compared = 0usize;
    let mut collider_wraps = 0usize;
    let mut differing: Vec<String> = Vec::new();
    let mut overlapping = 0usize;
    let mut missed: Vec<String> = Vec::new();
    for &xa in &ps {
        for &r in &rs {
            for &xb in &ps {
                if xa == xb {
                    continue;
                }
                let worlds: Vec<PhysicsWorld> = KINDS
                    .iter()
                    .map(|&k| stepped_with_collider(k, xa, r, xb))
                    .collect();
                let states: Vec<Vec<u8>> =
                    worlds.iter().map(PhysicsWorld::serialize_state).collect();
                compared += 1;
                if raw(xa).checked_add(raw(r)).is_none() || raw(xa).checked_sub(raw(r)).is_none() {
                    collider_wraps += 1;
                }
                // The ellipsoid spans x in [xa - r, xa + r] (cut at the ends of the range)
                // and the sphere [xb - 1, xb + 1]: when they overlap, every kind must hand
                // the pair to the narrow-phase. A collider box that came out inverted would
                // be empty and drop the pair under every kind alike, which the state
                // comparison above cannot see.
                let one = raw(Fix128::ONE);
                let (a_lo, a_hi) = (
                    raw(xa).saturating_sub(raw(r)),
                    raw(xa).saturating_add(raw(r)),
                );
                let (b_lo, b_hi) = (raw(xb).saturating_sub(one), raw(xb).saturating_add(one));
                if a_lo <= b_hi && b_lo <= a_hi {
                    overlapping += 1;
                    for (k, w) in KINDS.iter().zip(&worlds) {
                        if w.stage_work().broadphase_pairs == 0 {
                            missed.extend([format!("{k:?}: xa={xa:?} r={r:?} xb={xb:?}")]);
                        }
                    }
                }
                if states.windows(2).any(|w| w[0] != w[1]) {
                    let flags: Vec<bool> =
                        worlds.iter().map(PhysicsWorld::overflow_detected).collect();
                    differing.extend([format!("xa={xa:?} r={r:?} xb={xb:?} flags={flags:?}")]);
                }
            }
        }
    }
    assert!(compared > 0, "the sweep compared nothing");
    assert!(
        collider_wraps > 0,
        "no configuration has a collider box that leaves the range ({compared} compared)"
    );
    assert!(
        overlapping > 0,
        "no configuration has the two bodies overlapping"
    );
    assert!(
        missed.is_empty(),
        "{} overlapping pairs missing from the broad-phase candidates ({overlapping} overlapping configurations), first: {}",
        missed.len(),
        missed.first().map_or("", String::as_str)
    );
    assert!(
        differing.is_empty(),
        "{} of {compared} configurations step to different states by kind ({collider_wraps} with a collider box that leaves the range), first: {}",
        differing.len(),
        differing.first().map_or("", String::as_str)
    );
}

/// Two bodies asleep at rest whose sphere boxes leave the `Fix128` range, stepped
/// three times under `kind`, with the sleep skip on or off.
///
/// With the skip on (only `Broadphase::Bvh` skips) both bodies are parked from the
/// first step: neither is staged, and a pair of two parked bodies never reaches the
/// narrow-phase, so the only place their boxes are built is the insert into the tree
/// of parked bodies. That insert has to raise the overflow flag as staging does.
fn parked_pair(kind: Broadphase, skip: bool, xb: Fix128, r: Fix128, steps: usize) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
    w.set_sleep_skip(skip);
    w.add_body_with_radius(RigidBody::new_dynamic(x_at(Fix128::ZERO), Fix128::ONE), r);
    w.add_body_with_radius(RigidBody::new_dynamic(x_at(xb), Fix128::ONE), r);
    for i in 0..2 {
        w.islands.sleep_data[i].state = SleepState::Sleeping;
        w.islands.sleep_data[i].idle_frames = 100;
    }
    for _ in 0..steps {
        w.step(Fix128::from_ratio(1, 60));
    }
    w
}

/// Whether the sphere box `x ± r` itself leaves the `Fix128` range (the box the
/// insert into the tree of parked bodies builds; the tree's own margin stops at the
/// ends of the range without raising the flag)
fn sphere_box_leaves(x: Fix128, r: Fix128) -> bool {
    raw(x).checked_add(raw(r)).is_none() || raw(x).checked_sub(raw(r)).is_none()
}

/// The parked pairs of [`parked_pair`] compared with the staged Bvh world (sleep skip
/// off, flag asserted raised) under the Bvh with the skip on, the dynamic tree and the
/// hybrid broad-phase: same overflow flag and same `serialize_state`. `boxes_leave`
/// selects the configurations where a sphere box leaves the `Fix128` range (`true`)
/// or where only the sum of the two radii does (`false`). Returns (compared,
/// differing descriptions).
fn compare_parked(boxes_leave: bool) -> (usize, Vec<String>) {
    let mut compared = 0usize;
    let mut differing: Vec<String> = Vec::new();
    for xb in [Fix128::from_int(1 << 40), Fix128::from_int(-(1 << 62))] {
        for r in radii() {
            if (sphere_box_leaves(Fix128::ZERO, r) || sphere_box_leaves(xb, r)) != boxes_leave {
                continue;
            }
            // the skip really parks both bodies on the first step (two inserts into the tree)
            let first = parked_pair(Broadphase::Bvh, true, xb, r, 1);
            assert!(
                first.stage_work().tree_inserts >= 2,
                "xb={xb:?} r={r:?}: the bodies were not parked"
            );
            let reference = parked_pair(Broadphase::Bvh, false, xb, r, 3);
            assert!(
                reference.overflow_detected(),
                "xb={xb:?} r={r:?}: a radius sum or a staged box leaves the range, so the flag is raised"
            );
            let reference_state = reference.serialize_state();
            for (kind, skip) in [
                (Broadphase::Bvh, true),
                (Broadphase::DynamicTree, false),
                (Broadphase::Hybrid, false),
            ] {
                let w = parked_pair(kind, skip, xb, r, 3);
                compared += 1;
                if w.overflow_detected() != reference.overflow_detected()
                    || w.serialize_state() != reference_state
                {
                    differing.extend([format!(
                        "{kind:?} skip={skip} xb={xb:?} r={r:?}: flag {}",
                        w.overflow_detected()
                    )]);
                }
            }
        }
    }
    (compared, differing)
}

/// Oracle: the staged world (sleep skip off), whose boxes are built at staging and
/// raise the flag there when they leave the range. With the skip on, the parked
/// bodies' boxes are built only at the insert into the tree of parked bodies, which
/// must raise the flag the same way.
#[test]
fn parked_bodies_whose_boxes_leave_the_range_raise_the_flag_like_staged_ones() {
    let (compared, differing) = compare_parked(true);
    assert!(
        compared > 0,
        "no configuration has a box that leaves the range"
    );
    assert!(
        differing.is_empty(),
        "{} of {compared} worlds differ from the staged Bvh world, first: {}",
        differing.len(),
        differing.first().map_or("", String::as_str)
    );
}

/// Oracle: the staged world (sleep skip off), where the pair reaches the narrow-phase
/// and the sum of the two radii (here past the range while both boxes stay inside it)
/// raises the flag. With the skip on, a pair of two parked bodies never reaches the
/// narrow-phase, so the sum is never formed and the flag stays down: the sleep skip is
/// not bit-identical here (AUD-A-S6W1-011).
#[test]
#[ignore = "known defect: AUD-A-S6W1-011: sleep skip radius sum: a pair of two parked bodies never reaches the narrow-phase, so a radius sum past the Fix128 range raises the overflow flag with the sleep skip off but not with it on"]
fn parked_bodies_whose_radius_sum_leaves_the_range_raise_the_flag_like_staged_ones() {
    let (compared, differing) = compare_parked(false);
    assert!(
        compared > 0,
        "no configuration has only the radius sum past the range"
    );
    assert!(
        differing.is_empty(),
        "{} of {compared} worlds differ from the staged Bvh world, first: {}",
        differing.len(),
        differing.first().map_or("", String::as_str)
    );
}

/// The two parked bodies of [`parked_pair`] with the largest radius, so the second
/// body's sphere box (`2^40 + r`) leaves the range, set back to a state taken before
/// their first step (overflow flag down) and stepped once more. `via_snapshot` selects `snapshot_world` / `restore_world`
/// instead of `serialize_state` / `deserialize_state`.
fn stepped_after_rollback(kind: Broadphase, skip: bool, via_snapshot: bool) -> PhysicsWorld {
    let xb = Fix128::from_int(1 << 40);
    let r = Fix128::from_raw(i64::MAX, u64::MAX);
    assert!(sphere_box_leaves(xb, r), "the second box leaves the range");
    let mut w = parked_pair(kind, skip, xb, r, 0);
    let state = w.serialize_state();
    let snapshot = w.snapshot_world();
    assert!(
        !w.overflow_detected(),
        "the flag is down before the first step"
    );
    for _ in 0..3 {
        w.step(Fix128::from_ratio(1, 60));
    }
    assert!(
        w.overflow_detected(),
        "{kind:?} skip={skip}: the box past the range raised the flag"
    );
    if via_snapshot {
        w.restore_world(&snapshot).expect("the snapshot restores");
    } else {
        assert!(w.deserialize_state(&state), "the state loads");
    }
    assert!(!w.overflow_detected(), "the loaded state has the flag down");
    w.step(Fix128::from_ratio(1, 60));
    w
}

/// Oracle: the flag after a rollback follows the loaded state the same way in every
/// world: the first step after it raises the flag again, as the first step did, because
/// the box past the range is built again. With the sleep skip on, the parked bodies'
/// tree is not part of the loaded state, so the loading has to send them through the
/// insert (and its box check) again.
#[test]
fn a_rollback_below_the_flag_raises_it_again_through_the_box_on_the_next_step() {
    let mut compared = 0usize;
    let mut wrong: Vec<String> = Vec::new();
    for via_snapshot in [false, true] {
        let reference = stepped_after_rollback(Broadphase::Bvh, false, via_snapshot);
        let reference_state = reference.serialize_state();
        for (kind, skip) in [
            (Broadphase::Bvh, false),
            (Broadphase::Bvh, true),
            (Broadphase::DynamicTree, false),
            (Broadphase::DynamicTree, true),
            (Broadphase::Hybrid, false),
            (Broadphase::Hybrid, true),
        ] {
            let w = stepped_after_rollback(kind, skip, via_snapshot);
            compared += 1;
            if !w.overflow_detected() || w.serialize_state() != reference_state {
                wrong.extend([format!(
                    "{kind:?} skip={skip} via_snapshot={via_snapshot}: flag {}",
                    w.overflow_detected()
                )]);
            }
        }
    }
    assert_eq!(
        compared, 12,
        "2 ways to roll back x 3 kinds x skip on / off"
    );
    assert!(
        wrong.is_empty(),
        "{} of {compared} worlds differ after the rollback, first: {}",
        wrong.len(),
        wrong.first().map_or("", String::as_str)
    );
}
