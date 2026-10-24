//! A faulted world refuses every change and keeps answering reads, through Rust and through the C
//! entry points, and a panic anywhere a participant runs under the world's lock (its substep, its
//! observation, its state writer) faults the world like a participant error does.
//!
//! oracle: the documented fault contract of `PhysicsWorld::try_step` (a faulted world is not touched)
//! extended to every entry that changes the world; "not touched" is checked as the world snapshot's
//! bytes before and after the call.

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, StepError,
    SubstepCtx,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Fails (returns `Err`) or panics at chosen points; otherwise pushes nothing.
#[derive(Default)]
struct Faulty {
    err_on_substep: bool,
    panic_on_substep: Option<u32>,
    panic_on_observe: bool,
    panic_on_write: bool,
    substeps: u32,
}

impl Participant for Faulty {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0x4641_554c)
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        self.substeps += 1;
        assert!(
            self.panic_on_substep != Some(self.substeps),
            "participant panics in its substep"
        );
        if self.err_on_substep {
            return Err(ParticipantFault::InvalidState);
        }
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
        assert!(
            !self.panic_on_write,
            "participant panics in its state writer"
        );
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
        self.substeps = u32::from_le_bytes(bytes.try_into().unwrap());
    }
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn world_with(p: Faulty) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    w.add_body(RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::ONE));
    w.add_participant(Box::new(p)).expect("register");
    w
}

fn positions(w: &PhysicsWorld) -> Vec<Vec3Fix> {
    w.bodies.iter().map(|b| b.position).collect()
}

/// A panic in the observation or the state writer, under the world's lock, faults the world.
#[test]
fn a_panic_in_a_participant_read_faults_the_world() {
    let mut compared = 0;
    for (what, p) in [
        (
            "observe",
            Faulty {
                panic_on_observe: true,
                ..Faulty::default()
            },
        ),
        (
            "write_state",
            Faulty {
                panic_on_write: true,
                ..Faulty::default()
            },
        ),
    ] {
        let mut w = world_with(p);
        let caught = catch_unwind(AssertUnwindSafe(|| match what {
            "observe" => drop(w.observe_participant(0)),
            _ => drop(w.snapshot_world()),
        }));
        assert!(caught.is_err(), "{what}: the participant panicked");
        assert!(
            w.fault().is_some(),
            "{what}: the world records no fault after the panic"
        );
        let before = positions(&w);
        assert!(
            matches!(w.try_step(dt()), Err(StepError::Faulted(_))),
            "{what}"
        );
        assert_eq!(
            positions(&w),
            before,
            "{what}: a refused step changes no body"
        );
        compared += 1;
    }
    assert_eq!(compared, 2);
}

/// After a participant panic, clearing the fault lets the world step again.
#[test]
fn clearing_the_fault_after_a_participant_panic_steps_again() {
    let mut w = world_with(Faulty {
        panic_on_substep: Some(1),
        ..Faulty::default()
    });
    assert!(catch_unwind(AssertUnwindSafe(|| w.step(dt()))).is_err());
    assert!(w.fault().is_some(), "the panic is recorded");
    w.clear_fault();
    assert!(w.fault().is_none());
    let before = positions(&w);
    assert_eq!(
        w.try_step(dt()),
        Ok(()),
        "the world steps again after clear_fault"
    );
    assert_ne!(positions(&w), before, "the step moved the bodies (gravity)");
}

#[cfg(feature = "ffi")]
mod c_api {
    use super::*;
    use alice_physics_core::ffi::*;

    fn v(x: f64, y: f64, z: f64) -> AliceVec3 {
        AliceVec3 { x, y, z }
    }

    /// A world with two bodies, a static plane and a joint, faulted by a participant error.
    fn faulted() -> PhysicsWorld {
        let mut w = world_with(Faulty {
            err_on_substep: true,
            ..Faulty::default()
        });
        unsafe {
            assert_ne!(
                alice_physics_static_add_plane(&mut w, v(0.0, 1.0, 0.0), -5.0),
                u32::MAX
            );
            assert_ne!(
                alice_physics_joint_add_ball(&mut w, 0, 1, v(1.0, 0.0, 0.0), v(-1.0, 0.0, 0.0)),
                u32::MAX
            );
        }
        assert!(matches!(w.try_step(dt()), Err(StepError::FaultRaised(_))));
        assert!(w.fault().is_some());
        w
    }

    /// A participant error during the C step is reported as a failure (0), not as success.
    #[test]
    fn the_c_step_reports_a_raised_fault() {
        let mut w = world_with(Faulty {
            err_on_substep: true,
            ..Faulty::default()
        });
        let ok = unsafe { alice_physics_world_step(&mut w, 1.0 / 60.0) };
        assert!(w.fault().is_some());
        assert_eq!(
            ok, 0,
            "the step raised a fault and the C step reports success"
        );
    }

    /// Every C entry that changes the world is refused on a faulted world (its documented failure
    /// value) and leaves the world's snapshot bytes as they were.
    #[test]
    fn c_mutations_on_a_faulted_world_are_refused_and_change_nothing() {
        let mut w = faulted();
        let before = w.snapshot_world();
        let wp: *mut PhysicsWorld = &mut w;
        let vel = [1.0_f64; 6];
        let heights = [0.0_f64; 4];
        let verts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0_f64];
        let idx = [0_u32, 1, 2];
        let shape = AlicePhysicsShape {
            kind: 0,
            a: 0.5,
            b: 0.5,
            c: 0.5,
        };
        let mut refused: Vec<(&str, bool)> = Vec::new();
        unsafe {
            refused.push((
                "world_step_n",
                alice_physics_world_step_n(wp, 1.0 / 60.0, 2) == 0,
            ));
            refused.push((
                "set_velocities_batch",
                alice_physics_world_set_velocities_batch(wp, vel.as_ptr(), 2) == 0,
            ));
            refused.push((
                "apply_impulses_batch",
                alice_physics_body_apply_impulses_batch(wp, vel.as_ptr(), 2) == 0,
            ));
            refused.push((
                "body_add_dynamic",
                alice_physics_body_add_dynamic(wp, v(0.0, 3.0, 0.0), 1.0) == u32::MAX,
            ));
            refused.push((
                "body_add_static",
                alice_physics_body_add_static(wp, v(0.0, 3.0, 0.0)) == u32::MAX,
            ));
            refused.push((
                "body_add_sensor",
                alice_physics_body_add_sensor(wp, v(0.0, 3.0, 0.0)) == u32::MAX,
            ));
            refused.push((
                "body_add_shaped",
                alice_physics_body_add_shaped(wp, shape, 1.0, v(0.0, 3.0, 0.0)) == u32::MAX,
            ));
            refused.push((
                "body_set_position",
                alice_physics_body_set_position(wp, 0, v(5.0, 5.0, 5.0)) == 0,
            ));
            refused.push((
                "body_set_velocity",
                alice_physics_body_set_velocity(wp, 0, v(1.0, 0.0, 0.0)) == 0,
            ));
            refused.push((
                "body_set_restitution",
                alice_physics_body_set_restitution(wp, 0, 0.9) == 0,
            ));
            refused.push((
                "body_set_friction",
                alice_physics_body_set_friction(wp, 0, 0.9) == 0,
            ));
            refused.push((
                "body_set_collision_radius",
                alice_physics_body_set_collision_radius(wp, 0, 0.25) == 0,
            ));
            refused.push((
                "body_clear_collision_radius",
                alice_physics_body_clear_collision_radius(wp, 0) == 0,
            ));
            refused.push((
                "body_set_shape",
                alice_physics_body_set_shape(wp, 0, shape) == 0,
            ));
            refused.push((
                "body_apply_impulse",
                alice_physics_body_apply_impulse(wp, 0, v(1.0, 0.0, 0.0)) == 0,
            ));
            refused.push((
                "body_apply_impulse_at",
                alice_physics_body_apply_impulse_at(wp, 0, v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0)) == 0,
            ));
            alice_physics_world_set_gravity(wp, 0.0, 9.8, 0.0);
            refused.push(("world_set_gravity", (*wp).snapshot_world() == before));
            alice_physics_world_set_substeps(wp, 3);
            refused.push(("world_set_substeps", (*wp).snapshot_world() == before));
            refused.push((
                "state_deserialize",
                alice_physics_state_deserialize(wp, [0_u8; 4].as_ptr(), 4) == 0,
            ));
            refused.push((
                "static_add_plane",
                alice_physics_static_add_plane(wp, v(1.0, 0.0, 0.0), 0.0) == u32::MAX,
            ));
            refused.push((
                "static_add_heightfield",
                alice_physics_static_add_heightfield(
                    wp,
                    heights.as_ptr(),
                    2,
                    2,
                    1.0,
                    v(0.0, 0.0, 0.0),
                ) == u32::MAX,
            ));
            refused.push((
                "static_add_trimesh",
                alice_physics_static_add_trimesh(wp, verts.as_ptr(), 3, idx.as_ptr(), 3)
                    == u32::MAX,
            ));
            refused.push(("static_remove", alice_physics_static_remove(wp, 0) == 0));
            refused.push((
                "joint_add_ball",
                alice_physics_joint_add_ball(wp, 0, 1, v(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0))
                    == u32::MAX,
            ));
            refused.push(("joint_remove", alice_physics_joint_remove(wp, 0) == 0));
        }
        let changed = w.snapshot_world() != before;
        let not_refused: Vec<&str> = refused
            .iter()
            .filter(|(_, r)| !r)
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(refused.len(), 25);
        assert!(
            not_refused.is_empty() && !changed,
            "a faulted world accepted changes through the C API: not refused {not_refused:?}, snapshot changed: {changed}"
        );
    }

    /// Reads keep answering on a faulted world.
    #[test]
    fn c_reads_answer_on_a_faulted_world() {
        let w = faulted();
        let wp: *const PhysicsWorld = &w;
        unsafe {
            assert_eq!(alice_physics_world_body_count(wp), 2);
            assert_eq!(alice_physics_static_count(wp), 1);
            assert_eq!(alice_physics_joint_count(wp), 1);
            let mut out = v(0.0, 0.0, 0.0);
            assert_eq!(alice_physics_body_get_position(wp, 0, &mut out), 1);
            assert_eq!(alice_physics_body_get_velocity(wp, 0, &mut out), 1);
            let mut buf = [0.0_f64; 6];
            assert_eq!(
                alice_physics_world_get_positions_batch(wp, buf.as_mut_ptr(), 6),
                1
            );
            assert_eq!(
                alice_physics_world_get_velocities_batch(wp, buf.as_mut_ptr(), 6),
                1
            );
            let mut len = 0_u32;
            let data = alice_physics_state_serialize(wp, &mut len);
            assert!(
                !data.is_null() && len > 0,
                "state_serialize answers on a faulted world"
            );
            alice_physics_state_free(data, len);
        }
    }
}
