//! `alice_physics_world_fault(world)` tells a C host which fault the world holds: 0 none,
//! 1 a participant fault (an error the participant returned, or a panic in its substep), 5 a panic
//! anywhere else (here the rigid step), 255 a null world. Codes 2 to 4 (rigid overflow, force and
//! field out of range) are the existing `WorldFault` variants and are not produced here.
//!
//! oracle: the code table of the C header for `alice_physics_world_fault`.

#![cfg(feature = "ffi")]

use alice_physics_core::ffi::{alice_physics_world_fault, alice_physics_world_step};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

struct Fails {
    panic: bool,
}

impl Participant for Fails {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0x434f_4445)
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        assert!(!self.panic, "participant panics in its substep");
        Err(ParticipantFault::InvalidState)
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _: &[u8]) {}
}

fn ball() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.add_body(RigidBody::new(Vec3Fix::from_int(0, 1, 0), Fix128::ONE));
    w
}

#[test]
fn the_c_fault_code_names_the_fault() {
    let mut compared = 0;
    unsafe {
        assert_eq!(
            alice_physics_world_fault(std::ptr::null()),
            255,
            "null world"
        );
        compared += 1;

        let mut w = ball();
        assert_eq!(alice_physics_world_fault(&w), 0, "a healthy world");
        alice_physics_world_step(&mut w, 1.0 / 60.0);
        assert_eq!(
            alice_physics_world_fault(&w),
            0,
            "a healthy world after a step"
        );
        compared += 1;

        for (what, panic) in [("participant error", false), ("participant panic", true)] {
            let mut w = ball();
            w.add_participant(Box::new(Fails { panic }))
                .expect("register");
            alice_physics_world_step(&mut w, 1.0 / 60.0);
            assert_eq!(alice_physics_world_fault(&w), 1, "{what}");
            compared += 1;
        }

        let mut w = ball();
        w.add_sdf_collider(SdfCollider::new_static(
            Box::new(ClosureSdf::new(
                |_x, _y, _z| panic!("the collider's distance panics inside the rigid step"),
                |_x, _y, _z| (0.0, 1.0, 0.0),
            )),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));
        alice_physics_world_step(&mut w, 1.0 / 60.0);
        assert_eq!(
            alice_physics_world_fault(&w),
            5,
            "a panic in the rigid step"
        );
        compared += 1;
    }
    assert_eq!(compared, 5);
}
