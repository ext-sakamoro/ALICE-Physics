//! A participant that panics inside a step leaves the world faulted: the panic is not swallowed into a
//! world that keeps stepping on a half-updated state. After the panic is caught (by the host, or by the
//! C entry point), the world reports a fault, the next step is refused and changes nothing, and a
//! snapshot taken then carries the fault, so a world restored from it refuses to step too.
//!
//! oracle: the documented fault contract of `PhysicsWorld::try_step` (a faulted world returns
//! `StepError::Faulted` and is not touched), applied to a fault the step could not report by `Err`.

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, StepError,
    SubstepCtx,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Pushes body 0 every substep and panics on substep `panic_at` (counted from 1), after it has already
/// pushed and counted that substep: the panic comes from inside a step, with the participant's own
/// state already changed.
struct PanicsMidStep {
    substeps: u32,
    panic_at: u32,
}

impl Participant for PanicsMidStep {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0x5041_4e49)
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _h: Fix128) -> Result<(), ParticipantFault> {
        ctx.add_force(0, Vec3Fix::from_int(1, 0, 0))
            .map_err(|_| ParticipantFault::InvalidState)?;
        self.substeps += 1;
        assert!(
            self.substeps != self.panic_at,
            "participant panics inside the step"
        );
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.substeps.to_le_bytes());
        out.extend_from_slice(&self.panic_at.to_le_bytes());
    }
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() == 8 {
            Ok(())
        } else {
            Err(StateError::InvalidValue)
        }
    }
    fn read_state(&mut self, bytes: &[u8]) {
        self.substeps = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        self.panic_at = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    }
}

fn world(panic_at: u32) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    w.add_participant(Box::new(PanicsMidStep {
        substeps: 0,
        panic_at,
    }))
    .expect("register");
    w
}

/// Position and velocity of every body, as bits.
fn bodies(w: &PhysicsWorld) -> Vec<(Vec3Fix, Vec3Fix)> {
    w.bodies.iter().map(|b| (b.position, b.velocity)).collect()
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// Checks the world after the panic was caught; `step_again` is the entry point under test.
fn after_the_panic(w: &mut PhysicsWorld, step_again: &dyn Fn(&mut PhysicsWorld) -> bool) {
    assert!(
        w.fault().is_some(),
        "a participant panicked inside the step and the world records no fault"
    );

    let before = bodies(w);
    assert!(!step_again(w), "a step after the panic is refused");
    assert_eq!(bodies(w), before, "a refused step changes no body");
    assert!(
        matches!(w.try_step(dt()), Err(StepError::Faulted(_))),
        "try_step reports the recorded fault"
    );
    assert_eq!(bodies(w), before);

    // the snapshot carries the fault: a world restored from it refuses to step
    let blob = w.snapshot_world();
    let mut restored = world(u32::MAX);
    restored.restore_world(&blob).expect("restore the snapshot");
    assert!(
        restored.fault().is_some(),
        "the fault is part of the snapshot"
    );
    let before = bodies(&restored);
    assert!(matches!(
        restored.try_step(dt()),
        Err(StepError::Faulted(_))
    ));
    assert_eq!(bodies(&restored), before);
}

#[test]
fn a_participant_panic_inside_a_step_leaves_the_world_faulted() {
    let mut compared = 0;
    // the panic on the first substep and on a later one (the body was already pushed that step)
    for panic_at in [1, 3] {
        let mut w = world(panic_at);
        let caught = catch_unwind(AssertUnwindSafe(|| w.step(dt())));
        assert!(
            caught.is_err(),
            "panic_at {panic_at}: the participant panicked"
        );
        after_the_panic(&mut w, &|w| w.try_step(dt()).is_ok());
        compared += 1;
    }
    assert_eq!(compared, 2);
}

/// The same through the C entry point, which catches the panic itself and returns 0.
#[cfg(feature = "ffi")]
#[test]
fn a_participant_panic_through_the_c_step_leaves_the_world_faulted() {
    use alice_physics_core::ffi::alice_physics_world_step;
    let mut compared = 0;
    for panic_at in [1, 3] {
        let mut w = world(panic_at);
        let ok = unsafe { alice_physics_world_step(&mut w, 1.0 / 60.0) };
        assert_eq!(ok, 0, "panic_at {panic_at}: the C step reports the panic");
        after_the_panic(
            &mut w,
            &|w| unsafe { alice_physics_world_step(w, 1.0 / 60.0) } == 1,
        );
        compared += 1;
    }
    assert_eq!(compared, 2);
}
