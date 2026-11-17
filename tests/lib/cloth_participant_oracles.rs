//! Lib tests of `src/cloth_participant.rs`: the step rule it declares, the
//! payload it writes and refuses, and one step of the frame's width matching the
//! world's own cloth step bit for bit.
//!
//! Included from `src/cloth_participant.rs` as a `#[cfg(test)]` module, so they run
//! with `cargo test --lib`, the test set the mutation run uses. The wider oracles
//! (settle across substeps, rewind and fork, no interaction with bodies) live in
//! `tests/cloth_resident_compare.rs`.

use super::*;
use crate::cloth::{ClothConfig, OutlineDesc};
use crate::math::Vec3Fix;
use crate::solver::{PhysicsWorld, SolverConfig};
use crate::world_participant::RegisterError;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// A 0.3 m square sheet, thrown at 1 m/s out of its plane, stepping `m` times a frame.
fn sheet(m: usize) -> Cloth {
    let p = |x: i64, y: i64| [r(x, 10), r(y, 10)];
    let mut c = Cloth::from_outline(&OutlineDesc {
        origin: Vec3Fix::ZERO,
        u: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        v: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        contours: vec![vec![p(0, 0), p(3, 0), p(3, 3), p(0, 3)]],
        spacing: r(1, 10),
        mass_per_particle: Fix128::ONE,
        config: ClothConfig {
            iterations: 4,
            substeps: m,
            gravity: Vec3Fix::ZERO,
            damping: r(985, 1000),
            ..ClothConfig::default()
        },
        tether: None,
    })
    .expect("sheet");
    c.set_sleep(false);
    for v in &mut c.velocities {
        *v = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
    }
    c
}

fn world(substeps: usize) -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps,
        ..SolverConfig::default()
    })
}

#[test]
fn new_takes_the_cloth_substeps_as_its_steps_per_frame() {
    let p = ClothParticipant::new(sheet(3));
    assert_eq!(p.step_rule(), StepRule::PerFrame(3));
    // each call is one step of the cloth: the world places the steps
    assert_eq!(p.cloth.config.substeps, 1);
    assert_eq!(p.kind(), CLOTH_PARTICIPANT_KIND);
    assert_eq!(CLOTH_PARTICIPANT_KIND, ParticipantKind::new(0x434c_5448));
    // a cloth of 0 substeps still steps once a frame
    assert_eq!(
        ClothParticipant::new(sheet(0)).step_rule(),
        StepRule::PerFrame(1)
    );
}

#[test]
fn one_step_per_frame_is_the_world_cloth_bit_for_bit() {
    let mut a = world(4);
    a.add_participant(Box::new(ClothParticipant::new(sheet(1))))
        .expect("register");
    let mut b = world(4);
    b.add_cloth(sheet(1));
    for frame in 0..60 {
        a.step(r(1, 60));
        b.step(r(1, 60));
        let mut want = Vec::new();
        want.extend_from_slice(&CLOTH_STATE_VERSION.to_le_bytes());
        b.cloths[0].write_rollback(&mut want);
        assert_eq!(
            a.participant_state(0).expect("payload"),
            want,
            "frame {frame}"
        );
    }
}

#[test]
fn more_steps_per_frame_than_substeps_is_refused() {
    let mut w = world(2);
    let err = w
        .add_participant(Box::new(ClothParticipant::new(sheet(3))))
        .unwrap_err();
    assert_eq!(
        err,
        RegisterError::StepsPerFrameAboveSubsteps {
            steps: 3,
            substeps: 2
        }
    );
}

#[test]
fn the_payload_round_trips_and_bad_payloads_are_refused() {
    let mut p = ClothParticipant::new(sheet(1));
    let mut start = Vec::new();
    p.write_state(&mut start);
    assert_eq!(&start[..4], &CLOTH_STATE_VERSION.to_le_bytes());
    assert_eq!(p.check_state(&start), Ok(()));

    // move the cloth, then read the start back: the bytes come back exactly
    p.cloth.step_by_its_own(r(1, 60));
    let mut moved = Vec::new();
    p.write_state(&mut moved);
    assert_ne!(moved, start, "the step moved the cloth");
    p.read_state(&start);
    let mut back = Vec::new();
    p.write_state(&mut back);
    assert_eq!(back, start);

    let mut wrong_version = start.clone();
    wrong_version[0] ^= 1;
    assert_eq!(p.check_state(&wrong_version), Err(StateError::InvalidValue));
    assert_eq!(p.check_state(&start[..4]), Err(StateError::InvalidValue));
    let mut trailing = start.clone();
    trailing.push(0);
    assert_eq!(
        p.check_state(&trailing),
        Err(StateError::Length {
            expected: start.len(),
            found: start.len() + 1
        })
    );
    // a refused payload leaves the cloth as it was
    let mut after = Vec::new();
    p.write_state(&mut after);
    assert_eq!(after, start);
}

#[test]
fn a_step_of_no_width_is_a_fault() {
    let mut p = ClothParticipant::new(sheet(1));
    let mut before = Vec::new();
    p.write_state(&mut before);
    let bodies: [crate::solver::RigidBody; 0] = [];
    let mut forces = crate::world_participant::ForceAccumulator::new(0);
    let mut ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, r(1, 60)).expect("ctx");
    assert_eq!(
        p.substep(&mut ctx, Fix128::ZERO),
        Err(ParticipantFault::InvalidState)
    );
    assert_eq!(
        p.substep(&mut ctx, -r(1, 60)),
        Err(ParticipantFault::InvalidState)
    );
    let mut after = Vec::new();
    p.write_state(&mut after);
    assert_eq!(after, before, "a refused step changes nothing");
    assert_eq!(p.substep(&mut ctx, r(1, 60)), Ok(()));
}
