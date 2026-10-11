//! A joint that names a body past the last body, or the same body twice.
//!
//! `PhysicsWorld::add_joint` refuses both (assert); `joints` is a public field,
//! so such a joint can still be pushed directly or restored from a snapshot.
//! Every step checks all joints at its start: `try_step` returns
//! `StepError::Joint` with the world unchanged (nothing ran: no joint broke, no
//! body moved), and `step` panics with that error's message. A snapshot holding
//! a self-joint is refused by the restore (`WorldSnapshotError::SelfJoint`);
//! one naming a missing body was already refused (`DanglingIndex`).

use alice_physics::world_participant::StepError;
use alice_physics::{BallJoint, Fix128, Joint, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn world() -> PhysicsWorld {
    let config = PhysicsConfig {
        substeps: 1,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-8), Fix128::ZERO),
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
    w
}

fn joint(a: usize, b: usize) -> Joint {
    Joint::Ball(BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO))
}

/// A joint holding the 2 kg body under `g = 8` (16 N) with a 15 N limit: it
/// breaks in the first step that runs.
fn breaking() -> Joint {
    Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(Fix128::from_int(15)),
    )
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn states(w: &PhysicsWorld) -> Vec<(Vec3Fix, Vec3Fix)> {
    w.bodies.iter().map(|b| (b.position, b.velocity)).collect()
}

#[test]
#[should_panic(expected = "Joint body indices (0, 7) out of bounds (body count = 2)")]
fn add_joint_refuses_an_index_past_the_last_body() {
    world().add_joint(joint(0, 7));
}

#[test]
#[should_panic(expected = "Joint joins body 1 to itself")]
fn add_joint_refuses_a_self_joint() {
    world().add_joint(joint(1, 1));
}

/// `try_step` refuses each invalid joint with its index and bodies, and the
/// world (bodies, joints) is exactly as before.
#[test]
fn try_step_refuses_every_invalid_joint_and_leaves_the_world_unchanged() {
    for (a, b) in [(0, 7), (7, 0), (7, 7), (2, 2), (1, 1), (0, 0)] {
        let mut w = world();
        w.add_joint(breaking());
        w.joints.push(joint(a, b));
        let before = states(&w);
        assert_eq!(
            w.try_step(dt()),
            Err(StepError::Joint {
                index: 1,
                bodies: (a, b),
                body_count: 2
            }),
            "joint ({a}, {b})"
        );
        assert_eq!(states(&w), before, "joint ({a}, {b}): a body moved");
        assert_eq!(
            w.joints.len(),
            2,
            "joint ({a}, {b}): the breaking joint broke, so the step ran"
        );
    }
}

#[test]
fn a_valid_world_still_steps_and_breaks_its_overloaded_joint() {
    let mut w = world();
    w.add_joint(breaking());
    assert_eq!(w.try_step(dt()), Ok(()));
    assert!(w.joints.is_empty(), "the 15 N joint breaks under 16 N");
}

#[test]
#[should_panic(
    expected = "step refused: joint 0 names bodies (0, 7), out of bounds (body count = 2)"
)]
fn step_panics_on_a_dangling_joint_with_the_error_message() {
    let mut w = world();
    w.joints.push(joint(0, 7));
    w.step(dt());
}

#[test]
#[should_panic(expected = "step refused: joint 1 joins body 7 to itself (body count = 2)")]
fn step_panics_on_a_self_joint_with_the_error_message() {
    let mut w = world();
    w.add_joint(breaking());
    w.joints.push(joint(7, 7));
    w.step(dt());
}

#[test]
fn the_error_message_names_the_joint() {
    let e = StepError::Joint {
        index: 3,
        bodies: (1, 1),
        body_count: 2,
    };
    assert_eq!(
        e.to_string(),
        "step refused: joint 3 joins body 1 to itself (body count = 2)"
    );
}

#[test]
fn a_non_positive_dt_runs_nothing_and_checks_nothing() {
    let mut w = world();
    w.joints.push(joint(1, 1));
    assert_eq!(w.try_step(Fix128::ZERO), Ok(()));
}

#[cfg(feature = "parallel")]
#[test]
#[should_panic(
    expected = "step refused: joint 0 names bodies (7, 1), out of bounds (body count = 2)"
)]
fn step_parallel_panics_on_a_dangling_joint_with_the_error_message() {
    let mut w = world();
    w.joints.push(joint(7, 1));
    w.step_parallel(dt());
}

#[cfg(feature = "parallel")]
#[test]
fn try_step_parallel_refuses_an_invalid_joint_unchanged() {
    let mut w = world();
    w.add_joint(breaking());
    w.joints.push(joint(1, 1));
    let before = states(&w);
    assert_eq!(
        w.try_step_parallel(dt()),
        Err(StepError::Joint {
            index: 1,
            bodies: (1, 1),
            body_count: 2
        })
    );
    assert_eq!(states(&w), before);
    assert_eq!(w.joints.len(), 2);
}

/// A snapshot written from a world whose `joints` holds a self-joint (pushed
/// directly) is refused by the restore, and the target world is untouched.
#[test]
fn a_snapshot_holding_a_self_joint_is_refused_by_the_restore() {
    let mut source = world();
    source.joints.push(joint(1, 1));
    let blob = source.snapshot_world();

    let mut target = world();
    target.add_joint(breaking());
    let before = (states(&target), target.joints.len());
    assert_eq!(
        target.restore_world(&blob),
        Err(alice_physics::WorldSnapshotError::SelfJoint { index: 0, body: 1 })
    );
    assert_eq!((states(&target), target.joints.len()), before);
}

#[cfg(feature = "gpu-solver-bridge")]
mod bridge {
    use super::*;
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};

    /// A bridge that does nothing: the joint check runs before any bridge call.
    struct Null;

    impl GpuSolverBridge for Null {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        // the joint methods panic by default (a backend without joint support);
        // this one accepts the joints and solves nothing
        fn send_joints(&mut self, _j: &[Joint]) {}
        fn send_body_state(&mut self, _p: &[[Fix128; 3]], _m: &[Fix128]) {}
        fn send_body_rotations(&mut self, _r: &[[Fix128; 4]]) {}
        fn dispatch_joint_solve_iteration(&mut self, _dt: Fix128) {}
        fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
    }

    #[test]
    #[should_panic(expected = "step refused: joint 0 joins body 1 to itself (body count = 2)")]
    fn step_with_bridge_panics_on_a_self_joint() {
        let mut w = world();
        w.joints.push(joint(1, 1));
        w.step_with_bridge(&mut Null, dt());
    }

    #[test]
    #[should_panic(
        expected = "step refused: joint 0 names bodies (0, 7), out of bounds (body count = 2)"
    )]
    fn substep_with_bridge_checks_the_joints_at_its_start() {
        let mut w = world();
        w.joints.push(joint(0, 7));
        w.substep_with_bridge(&mut Null, dt());
    }

    #[test]
    fn step_with_bridge_still_steps_a_valid_world() {
        // the do-nothing bridge solves no joint, so only gravity acts here: the
        // point is that a world with valid joints is not refused
        let mut w = world();
        w.add_joint(breaking());
        let before = w.bodies[1].velocity;
        w.step_with_bridge(&mut Null, dt());
        assert_ne!(w.bodies[1].velocity, before, "the valid world did not step");
    }
}
