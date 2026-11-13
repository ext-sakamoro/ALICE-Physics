//! The GPU-bridge contact contract against a bridge that solves the linear
//! contact law.
//!
//! `step_with_bridge` hands a bridge the contact constraints, the body
//! positions and the inverse masses, and nothing else. Since the contact rework
//! the CPU solves a contact as an XPBD point constraint (Müller et al. 2020
//! §3.5): the correction acts at the contact point, through the inverse inertia,
//! softened by the contact compliance `α / h²`, with the start overlap pushed out
//! rigidly first. The body state of the contract is the positions and the
//! inverse masses: no rotations and no inverse inertia. A bridge that projects
//! positions along the normal with the inverse masses (the law the solver had
//! before the rework, and the one a GPU backend that predates it runs) gives
//! other bits than `step` once persisting contacts reach the solve (measured
//! here on a chain of spheres), so such a bridge must not report
//! [`alice_physics_core::PHYSICS_SEMANTICS_ID`]; the world then refuses it with a
//! `PeerMismatch` (ALICE-Law G7) instead of running another contact law under
//! this identifier.
//!
//! The stub here claims the identifier on purpose, to show the mismatch the
//! claim would hide: a chain of six overlapping spheres, one step at 60 Hz,
//! `world.step` against `step_with_bridge` with the linear stub.

#![cfg(feature = "gpu-solver-bridge")]

use alice_physics_core::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
use alice_physics_core::joint::Joint;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{ContactConstraint, PhysicsConfig, PhysicsWorld, RigidBody};

/// A bridge that solves the contacts it is sent with the linear law: per
/// constraint with positive depth, the accumulated multiplier grows to the
/// depth and the positions move apart along the normal by that increment, split
/// by the inverse masses. Joints and islands are handed back unchanged.
#[derive(Default)]
struct LinearContactBridge {
    constraints: Vec<ContactConstraint>,
    positions: Vec<[Fix128; 3]>,
    inv_masses: Vec<Fix128>,
    sent: usize,
    solves: usize,
}

impl GpuSolverBridge for LinearContactBridge {
    /// The false claim this file is about: the law it computes is not the one
    /// this identifier names.
    fn semantics_id(&self) -> [u8; 32] {
        alice_physics_core::PHYSICS_SEMANTICS_ID
    }

    fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
    fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
    fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
    fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
        Ok(())
    }
    fn send_contact_constraints(&mut self, c: &[ContactConstraint]) {
        self.sent += c.len();
        self.constraints = c.to_vec();
    }
    fn send_body_state(&mut self, positions: &[[Fix128; 3]], inv_masses: &[Fix128]) {
        self.positions = positions.to_vec();
        self.inv_masses = inv_masses.to_vec();
    }
    fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {
        for c in &mut self.constraints {
            if c.contact.depth <= Fix128::ZERO {
                continue;
            }
            let (wa, wb) = (self.inv_masses[c.body_a], self.inv_masses[c.body_b]);
            let w = wa + wb;
            if w.is_zero() {
                continue;
            }
            let dlambda = c.contact.depth - c.cached_lambda;
            if dlambda <= Fix128::ZERO {
                continue;
            }
            c.cached_lambda = c.cached_lambda + dlambda;
            let correction = c.contact.normal * dlambda;
            let ca = correction * (wa / w);
            let cb = correction * (wb / w);
            let pa = &mut self.positions[c.body_a];
            let a = Vec3Fix::new(pa[0], pa[1], pa[2]) + ca;
            *pa = [a.x, a.y, a.z];
            let pb = &mut self.positions[c.body_b];
            let b = Vec3Fix::new(pb[0], pb[1], pb[2]) - cb;
            *pb = [b.x, b.y, b.z];
            self.solves += 1;
        }
    }
    fn recv_contact_constraints(&self, c: &mut [ContactConstraint]) {
        c.clone_from_slice(&self.constraints);
    }
    fn recv_body_positions(&self, positions: &mut [[Fix128; 3]]) {
        positions.copy_from_slice(&self.positions);
    }
    fn send_joints(&mut self, _joints: &[Joint]) {}
    fn send_body_rotations(&mut self, _rotations: &[[Fix128; 4]]) {}
    fn dispatch_joint_solve_iteration(&mut self, _dt: Fix128) {}
}

/// Six 1 kg spheres of radius 1.1 m with centres 2 m apart on x, under the
/// default configuration (gravity on y): each neighbour pair overlaps by 0.2 m.
/// The start overlap is pushed out rigidly; the contacts the next substeps find
/// persist and reach the contact solve, two per inner body.
fn sphere_chain() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let radius = Fix128::from_int(1) + Fix128::from_ratio(1, 10);
    for i in 0..6i64 {
        let body = RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_int(i * 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        );
        world.add_body_with_radius(body, radius);
    }
    world
}

/// A bridge that solves the linear contact law does not reproduce `step` on a
/// contact, so the contract (positions and inverse masses) cannot carry the
/// point-constraint contact, and no bridge computing from it may report this
/// build's identifier. Turns red if the CPU contact solve becomes the linear
/// law again (the two would then agree), or if the bridge path stops routing the
/// contact solve through the bridge (`sent` / `solves` stay 0).
///
/// Scope: a single head-on pair is not enough to show it; on a pair whose
/// contact does not persist into the solve the two paths agree, so the scene is
/// the chain, where the persisting contacts reach the solve.
#[test]
fn a_bridge_solving_the_linear_contact_law_does_not_reproduce_step() {
    let dt = Fix128::from_ratio(1, 60);

    let mut cpu = sphere_chain();
    cpu.step(dt);
    let mut bridged = sphere_chain();
    let mut bridge = LinearContactBridge::default();
    bridged
        .step_with_bridge(&mut bridge, dt)
        .expect("the stub claims this build's identifier");
    assert!(bridge.sent > 0, "the contact solve went through the bridge");
    assert!(bridge.solves > 0, "the stub moved bodies on a contact");

    let state = |w: &PhysicsWorld| -> Vec<(Vec3Fix, Vec3Fix)> {
        w.bodies.iter().map(|b| (b.position, b.velocity)).collect()
    };
    assert_ne!(
        state(&cpu),
        state(&bridged),
        "the linear law reproduced the point-constraint contact"
    );
}

/// The refusal the identifier exists for: a bridge reporting any other
/// identifier is refused before anything runs, with both identifiers named.
#[test]
fn a_bridge_reporting_another_identifier_is_refused_and_nothing_runs() {
    struct OtherLaw(LinearContactBridge);
    impl GpuSolverBridge for OtherLaw {
        fn semantics_id(&self) -> [u8; 32] {
            [0x5a; 32]
        }
        fn send_island(&mut self, p: &[[Fix128; 3]], v: &[[Fix128; 3]]) {
            self.0.send_island(p, v);
        }
        fn dispatch_iterations(&mut self, iters: u32, dt: Fix128) {
            self.0.dispatch_iterations(iters, dt);
        }
        fn recv_island(&self, p: &mut [[Fix128; 3]], v: &mut [[Fix128; 3]]) {
            self.0.recv_island(p, v);
        }
        fn assert_bit_exact_vs_cpu(&self, f: &DiffFixture) -> Result<(), GpuDivergence> {
            self.0.assert_bit_exact_vs_cpu(f)
        }
        fn send_contact_constraints(&mut self, c: &[ContactConstraint]) {
            self.0.send_contact_constraints(c);
        }
        fn send_body_state(&mut self, p: &[[Fix128; 3]], m: &[Fix128]) {
            self.0.send_body_state(p, m);
        }
        fn dispatch_contact_solve_iteration(&mut self, w: Fix128) {
            self.0.dispatch_contact_solve_iteration(w);
        }
        fn recv_contact_constraints(&self, c: &mut [ContactConstraint]) {
            self.0.recv_contact_constraints(c);
        }
        fn recv_body_positions(&self, p: &mut [[Fix128; 3]]) {
            self.0.recv_body_positions(p);
        }
        fn send_joints(&mut self, j: &[Joint]) {
            self.0.send_joints(j);
        }
        fn send_body_rotations(&mut self, r: &[[Fix128; 4]]) {
            self.0.send_body_rotations(r);
        }
        fn dispatch_joint_solve_iteration(&mut self, dt: Fix128) {
            self.0.dispatch_joint_solve_iteration(dt);
        }
    }

    let mut world = sphere_chain();
    let before: Vec<_> = world
        .bodies
        .iter()
        .map(|b| (b.position, b.velocity))
        .collect();
    let mut bridge = OtherLaw(LinearContactBridge::default());
    let err = world
        .step_with_bridge(&mut bridge, Fix128::from_ratio(1, 60))
        .expect_err("another identifier is refused");
    assert_eq!(err.bridge, "step_with_bridge");
    assert_eq!(err.peer, "gpu-solver-bridge");
    assert_eq!(err.found, [0x5a; 32]);
    assert_eq!(err.expected, alice_physics_core::PHYSICS_SEMANTICS_ID);
    assert_eq!(bridge.0.sent, 0, "nothing was sent to the refused bridge");
    let after: Vec<_> = world
        .bodies
        .iter()
        .map(|b| (b.position, b.velocity))
        .collect();
    assert_eq!(before, after);
}
