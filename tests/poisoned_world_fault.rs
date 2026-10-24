//! A fault that exists only as the poisoned participant lock (a panic in a participant's `observe`,
//! where the world records no field) is the same fault everywhere the world reports it: the snapshot
//! carries it, a world restored from that snapshot reports it, and `clear_fault` forgets it.
//!
//! oracle: the documented contract of `PhysicsWorld::fault` (a panic in `observe` / `write_state` is
//! `WorldFault::Panicked` from the moment of the panic, read from the lock) and of `clear_fault`
//! (steps run again); the snapshot is documented to carry the fault.

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
    WorldFault,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Panics in its observation when `panic_on_observe`; its state is a 4-byte counter.
struct Watcher {
    panic_on_observe: bool,
    substeps: u32,
}

impl Participant for Watcher {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0x5741_5443)
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        self.substeps += 1;
        Ok(())
    }
    fn observe(&self, out: &mut ObservationSink) {
        assert!(
            !self.panic_on_observe,
            "participant panics in its observation"
        );
        out.push(0, Fix128::from_int(i64::from(self.substeps)));
    }
    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.substeps.to_le_bytes());
    }
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() == 4 {
            Ok(())
        } else {
            Err(StateError::InvalidValue)
        }
    }
    fn read_state(&mut self, bytes: &[u8]) {
        self.substeps = u32::from_le_bytes(bytes.try_into().expect("4 bytes"));
    }
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn world(panic_on_observe: bool) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    w.add_participant(Box::new(Watcher {
        panic_on_observe,
        substeps: 0,
    }))
    .expect("register");
    w
}

/// A world whose only fault is the poisoned lock: one step, then a panic in `observe`.
fn poisoned() -> PhysicsWorld {
    let mut w = world(true);
    w.try_step(dt()).expect("a healthy step");
    assert!(w.fault().is_none(), "no fault before the panic");
    let caught = catch_unwind(AssertUnwindSafe(|| drop(w.observe_participant(0))));
    assert!(caught.is_err(), "the participant panicked");
    assert_eq!(w.fault(), Some(WorldFault::Panicked));
    w
}

/// The snapshot of a world faulted only by the poisoned lock carries the fault: a world restored
/// from it reports `Panicked` and refuses to step, as the original does.
#[test]
fn the_snapshot_carries_a_fault_held_only_by_the_lock() {
    let w = poisoned();
    let blob = w.snapshot_world();
    let mut restored = world(false);
    restored.restore_world(&blob).expect("restores");
    assert_eq!(restored.fault(), Some(WorldFault::Panicked));
    assert!(
        restored.try_step(dt()).is_err(),
        "the restored world refuses to step"
    );
}

/// `clear_fault` forgets a fault held only by the poisoned lock: the world reports no fault and
/// steps again.
#[test]
fn clear_fault_forgets_a_fault_held_only_by_the_lock() {
    let mut w = poisoned();
    w.clear_fault();
    assert_eq!(w.fault(), None, "the fault is forgotten");
    w.try_step(dt()).expect("the world steps again");
}
