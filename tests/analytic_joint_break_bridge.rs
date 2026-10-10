//! Joint breaking on the GPU-bridge paths (AUD-A-S1W6-009).
//!
//! Same closed form as `tests/analytic_joint_break_world.rs`: a 2 kg body hung
//! by a rigid ball joint from a static anchor carries `m · g = 16 N` after one
//! gravity substep (`h = 1/64`, `g = 8`). The joint is removed before the bridge
//! joint solve once that force exceeds `break_force`, both through
//! `step_with_bridge` (which calls `substep_with_bridge`) and through a bridge
//! installed with `set_gpu_solver_bridge` and plain `step`. The reference
//! bridge solves the joints it receives with the CPU `solve_joints`.

#![cfg(feature = "gpu-solver-bridge")]

use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{ContactConstraint, PhysicsConfig, PhysicsWorld, RigidBody};

#[derive(Default)]
struct JointReference {
    joints: Vec<Joint>,
    bodies: Vec<RigidBody>,
    sends: usize,
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
        self.sends += 1;
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

fn h() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn hung(limit: i64) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 1,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-8), Fix128::ZERO),
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    });
    w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
    w.add_joint(Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
            .with_break_force(Fix128::from_int(limit)),
    ));
    w
}

#[test]
fn step_with_bridge_breaks_a_joint_below_its_load() {
    let mut w = hung(15);
    let mut bridge = JointReference::default();
    w.step_with_bridge(&mut bridge, h());
    assert!(
        w.joints.is_empty(),
        "16 N > 15 N: removed on the step_with_bridge path"
    );
    let events = w.events.joint_break_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].force, Fix128::from_int(16), "m g exactly");
    // judged after the bridge solve: the bridge solved it in this substep, and
    // is sent no joint from the next one on
    assert_eq!((bridge.sends, bridge.joints.len()), (1, 1));
    w.step_with_bridge(&mut bridge, h());
    assert_eq!(bridge.sends, 1, "the broken joint is not sent again");
}

#[test]
fn step_with_bridge_keeps_a_joint_at_or_above_its_load() {
    for limit in [16, 17] {
        let mut w = hung(limit);
        let mut bridge = JointReference::default();
        for _ in 0..32 {
            w.step_with_bridge(&mut bridge, h());
        }
        assert_eq!(w.joints.len(), 1, "break_force {limit} N holds");
        assert_eq!(bridge.joints.len(), 1);
    }
}

#[test]
fn an_installed_bridge_breaks_through_plain_step() {
    let mut w = hung(15);
    w.set_gpu_solver_bridge(Some(Box::new(JointReference::default())));
    w.step(h());
    assert!(w.joints.is_empty());
    assert_eq!(w.events.joint_break_events()[0].force, Fix128::from_int(16));
}
