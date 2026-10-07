//! Audit oracles for solver: a joint whose two ends are the same body on the
//! GPU bridge path
//!
//! The CPU joint solve (`joint::solve_joints` / `solve_joints_breakable`) skips a
//! joint with `body_a == body_b`: a body cannot move relative to itself, so the
//! joint constrains nothing and the body stays a free body. A backend behind
//! `GpuSolverBridge` walks the uploaded joint list top to bottom and has no such
//! rule, so `solve_joints_with_bridge` must not upload those joints at all.
//!
//! Measured through `PhysicsWorld::step` with the bridge installed, against a
//! reference backend that runs the crate's own CPU `solve_joints` on what it
//! receives:
//! - the joint lists the backend receives are exactly the lists of a world built
//!   without the self joints (same joints, same order), and nothing is uploaded
//!   when every joint is a self joint
//! - body position, velocity, rotation and angular velocity are bit-identical to
//!   that world
//!
//! The reference backend skips self joints itself (it calls `solve_joints`), so
//! the body-state comparison alone would also hold for an upload that still
//! contains them; the received lists are what distinguish the two.

#![cfg(feature = "gpu-solver-bridge")]

use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
use alice_physics::joint::{
    BallJoint, ConeTwistJoint, FixedJoint, HingeJoint, Joint, SliderJoint, SpringJoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{ContactConstraint, PhysicsConfig, PhysicsWorld, RigidBody};
use std::sync::{Arc, Mutex};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

const STEPS: usize = 8;

type Log = Arc<Mutex<Vec<Vec<Joint>>>>;

/// Records every joint list it receives and solves it with the crate's CPU
/// `solve_joints` on a reconstruction of the bodies; returns positions, the only
/// joint result the trait carries back.
struct JointReference {
    log: Log,
    joints: Vec<Joint>,
    bodies: Vec<RigidBody>,
}

impl JointReference {
    fn new(log: Log) -> Self {
        Self {
            log,
            joints: Vec::new(),
            bodies: Vec::new(),
        }
    }
}

impl GpuSolverBridge for JointReference {
    fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
    fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
    fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
    fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
        Ok(())
    }
    fn send_contact_constraints(&mut self, _c: &[ContactConstraint]) {}
    fn dispatch_contact_solve_iteration(&mut self, _w: Fix128) {}
    fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
    fn send_joints(&mut self, j: &[Joint]) {
        self.log.lock().unwrap().push(j.to_vec());
        self.joints = j.to_vec();
    }
    fn send_body_state(&mut self, p: &[[Fix128; 3]], inv_masses: &[Fix128]) {
        self.bodies = p
            .iter()
            .zip(inv_masses)
            .map(|(pos, inv_m)| {
                let mut b = RigidBody::new(Vec3Fix::new(pos[0], pos[1], pos[2]), Fix128::ONE);
                b.inv_mass = *inv_m;
                b
            })
            .collect();
    }
    fn send_body_rotations(&mut self, r: &[[Fix128; 4]]) {
        for (b, q) in self.bodies.iter_mut().zip(r) {
            b.rotation = QuatFix::new(q[0], q[1], q[2], q[3]);
        }
    }
    fn dispatch_joint_solve_iteration(&mut self, dt: Fix128) {
        alice_physics::joint::solve_joints(&self.joints, &mut self.bodies, dt);
    }
    fn recv_body_positions(&self, p: &mut [[Fix128; 3]]) {
        for (out, b) in p.iter_mut().zip(&self.bodies) {
            *out = [b.position.x, b.position.y, b.position.z];
        }
    }
}

/// Five free bodies, spread apart (no colliders, no contacts), each with its own
/// velocity and spin, under gravity.
fn bodies_world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    for i in 0..5 {
        let k = i as f64;
        let mut b = RigidBody::new(v3(3.0 * k, 0.5 * k, -k), Fix128::ONE);
        b.velocity = v3(1.0 - 0.5 * k, 0.25 * k, 0.75);
        b.angular_velocity = v3(0.1 * k, -0.2, 0.3);
        w.add_body(b);
    }
    w
}

fn ball(a: usize, b: usize) -> Joint {
    Joint::Ball(BallJoint::new(a, b, v3(0.5, 0.0, 0.0), v3(-0.5, 0.0, 0.0)))
}

fn fixed(a: usize, b: usize) -> Joint {
    Joint::Fixed(FixedJoint::new(
        a,
        b,
        v3(0.0, 0.5, 0.0),
        v3(0.0, -0.5, 0.0),
        QuatFix::IDENTITY,
    ))
}

fn hinge(a: usize, b: usize) -> Joint {
    Joint::Hinge(HingeJoint::new(
        a,
        b,
        v3(0.0, 0.0, 0.5),
        v3(0.0, 0.0, -0.5),
        Vec3Fix::UNIT_Z,
        Vec3Fix::UNIT_Z,
    ))
}

fn slider(a: usize, b: usize) -> Joint {
    Joint::Slider(SliderJoint::new(
        a,
        b,
        Vec3Fix::UNIT_X,
        v3(0.25, 0.0, 0.0),
        v3(-0.25, 0.0, 0.0),
    ))
}

fn spring(a: usize, b: usize) -> Joint {
    Joint::Spring(SpringJoint::new(
        a,
        b,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::ONE,
        fx(50.0),
        fx(0.5),
    ))
}

fn cone_twist(a: usize, b: usize) -> Joint {
    Joint::ConeTwist(ConeTwistJoint::new(
        a,
        b,
        v3(0.0, 0.5, 0.0),
        v3(0.0, -0.5, 0.0),
        Vec3Fix::UNIT_Y,
        Vec3Fix::UNIT_Y,
    ))
}

type BodyState = (Vec3Fix, Vec3Fix, QuatFix, Vec3Fix);

/// Steps `joints` on `bodies_world()` with a `JointReference` installed; returns
/// the per-body state and every joint list the backend received.
fn run(joints: &[Joint]) -> (Vec<BodyState>, Vec<Vec<Joint>>) {
    let mut w = bodies_world();
    for j in joints {
        w.add_joint(*j);
    }
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    w.set_gpu_solver_bridge(Some(Box::new(JointReference::new(Arc::clone(&log)))));
    for _ in 0..STEPS {
        w.step(dt());
    }
    let state = w
        .bodies
        .iter()
        .map(|b| (b.position, b.velocity, b.rotation, b.angular_velocity))
        .collect();
    let received = log.lock().unwrap().clone();
    (state, received)
}

/// One self joint: no upload at all, and the world matches a world with no joint.
#[test]
fn a_single_self_joint_is_not_uploaded_and_leaves_the_body_free() {
    let (free_state, free_log) = run(&[]);
    assert!(free_log.is_empty(), "no joint, no upload");

    let (state, received) = run(&[ball(2, 2)]);
    assert_eq!(
        received, free_log,
        "a self joint alone must not reach the backend"
    );
    assert_eq!(state, free_state, "the body stays a free body");
}

/// Self joints of several kinds mixed in among ordinary joints: the backend
/// receives exactly the ordinary joints in their order, and the world matches a
/// world built with the ordinary joints only.
#[test]
fn self_joints_mixed_with_ordinary_joints_are_dropped_in_order() {
    // the ordinary joints form a chain 0-1-2-3, so their order changes the result
    let ordinary = [ball(0, 1), fixed(1, 2), hinge(2, 3), slider(3, 4)];
    let mixed = [
        fixed(1, 1),
        ordinary[0],
        hinge(3, 3),
        ordinary[1],
        spring(0, 0),
        ordinary[2],
        ball(4, 4),
        cone_twist(2, 2),
        ordinary[3],
        slider(4, 4),
    ];

    let (expected_state, expected_log) = run(&ordinary);
    assert!(!expected_log.is_empty(), "the ordinary joints are uploaded");
    assert!(
        expected_log.iter().all(|l| l.as_slice() == ordinary),
        "the ordinary world uploads its joints unchanged"
    );

    let (state, received) = run(&mixed);
    assert_eq!(received, expected_log, "only the ordinary joints, in order");
    assert_eq!(state, expected_state);
}

/// Every joint a self joint, one of each kind: nothing is uploaded and the world
/// matches a world with no joint.
#[test]
fn a_world_of_self_joints_only_uploads_nothing() {
    let (free_state, _) = run(&[]);
    let all_self = [
        ball(0, 0),
        fixed(1, 1),
        hinge(2, 2),
        slider(3, 3),
        spring(4, 4),
        cone_twist(0, 0),
    ];
    let (state, received) = run(&all_self);
    assert_eq!(received.len(), 0, "nothing reaches the backend");
    assert_eq!(state, free_state);
}

/// Non-vacuity: the ordinary chain moves the bodies away from the free motion,
/// and its order changes the result, so the comparisons above can fail.
#[test]
fn the_ordinary_chain_changes_the_result_and_its_order_matters() {
    let (free_state, _) = run(&[]);
    let forward = [ball(0, 1), fixed(1, 2), hinge(2, 3), slider(3, 4)];
    let mut backward = forward;
    backward.reverse();
    let (fwd, _) = run(&forward);
    let (bwd, _) = run(&backward);
    assert_ne!(fwd, free_state, "the joints act");
    assert_ne!(fwd, bwd, "the order of the joints changes the result");
}
