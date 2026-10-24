//! A panic anywhere inside a step, not only inside a participant, leaves the world faulted: the rigid
//! step (here a static SDF collider whose distance panics), the C batch step `alice_physics_world_step_n`
//! (a C entry other than the single step), and the GPU bridge paths (a bridge whose upload panics).
//! After the panic is caught, the world reports a fault, the next step is refused and changes nothing.
//!
//! oracle: the documented fault contract of `PhysicsWorld::try_step` (a faulted world is not touched),
//! applied to a panic that does not come from a participant.

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::sdf_collider::SdfCollider;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::world_participant::StepError;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// A ball resting just above a static SDF floor whose distance function panics: the rigid step's
/// collision query reaches it on the first step.
fn world_over_a_panicking_floor() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.add_body(RigidBody::new(Vec3Fix::from_int(0, 1, 0), Fix128::ONE));
    w.add_sdf_collider(SdfCollider::new(
        |_x, _y, _z| panic!("the collider's distance panics inside the rigid step"),
        |_x, _y, _z| (0.0, 1.0, 0.0),
    ));
    w
}

fn positions(w: &PhysicsWorld) -> Vec<Vec3Fix> {
    w.bodies.iter().map(|b| b.position).collect()
}

fn refuses_further_steps(w: &mut PhysicsWorld, what: &str) {
    assert!(
        w.fault().is_some(),
        "{what}: a panic in the rigid step and the world records no fault"
    );
    let before = positions(w);
    assert!(
        matches!(
            catch_unwind(AssertUnwindSafe(|| w.try_step(dt()))),
            Ok(Err(StepError::Faulted(_)))
        ),
        "{what}: the next step is refused (not run into the same panic)"
    );
    assert_eq!(
        positions(w),
        before,
        "{what}: a refused step changes no body"
    );
}

#[test]
fn a_panic_in_the_rigid_step_faults_the_world() {
    let mut w = world_over_a_panicking_floor();
    let caught = catch_unwind(AssertUnwindSafe(|| w.try_step(dt())));
    assert!(
        !matches!(caught, Ok(Ok(()))),
        "the collider panicked inside the step; the step did not report success"
    );
    refuses_further_steps(&mut w, "rigid step");
}

#[cfg(feature = "ffi")]
#[test]
fn a_panic_inside_the_c_batch_step_faults_the_world() {
    use alice_physics_core::ffi::alice_physics_world_step_n;
    let mut w = world_over_a_panicking_floor();
    let ok = unsafe { alice_physics_world_step_n(&mut w, 1.0 / 60.0, 3) };
    assert_eq!(ok, 0, "the C batch step reports the panic");
    refuses_further_steps(&mut w, "C batch step");
}

#[cfg(feature = "gpu-solver-bridge")]
mod bridge {
    use super::*;
    use alice_physics_core::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics_core::joint::Joint;
    use alice_physics_core::solver::ContactConstraint;

    /// A bridge that passes the island through and panics on its first upload.
    #[derive(Default)]
    struct PanicsOnUpload {
        positions: Vec<[Fix128; 3]>,
        velocities: Vec<[Fix128; 3]>,
        constraints: Vec<ContactConstraint>,
        body_positions: Vec<[Fix128; 3]>,
    }

    impl GpuSolverBridge for PanicsOnUpload {
        fn semantics_id(&self) -> [u8; 32] {
            alice_physics_core::PHYSICS_SEMANTICS_ID
        }
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {
            panic!("the bridge panics on its upload");
        }
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, p: &mut [[Fix128; 3]], v: &mut [[Fix128; 3]]) {
            p.copy_from_slice(&self.positions);
            v.copy_from_slice(&self.velocities);
        }
        fn assert_bit_exact_vs_cpu(&self, _fixture: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, c: &[ContactConstraint]) {
            self.constraints = c.to_vec();
        }
        fn send_body_state(&mut self, positions: &[[Fix128; 3]], _inv_masses: &[Fix128]) {
            self.body_positions = positions.to_vec();
        }
        fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {}
        fn recv_contact_constraints(&self, c: &mut [ContactConstraint]) {
            c.clone_from_slice(&self.constraints);
        }
        fn recv_body_positions(&self, positions: &mut [[Fix128; 3]]) {
            positions.copy_from_slice(&self.body_positions);
        }
        fn send_joints(&mut self, _joints: &[Joint]) {}
        fn send_body_rotations(&mut self, _rotations: &[[Fix128; 4]]) {}
        fn dispatch_joint_solve_iteration(&mut self, _dt: Fix128) {}
    }

    fn falling_ball() -> PhysicsWorld {
        let mut w = PhysicsWorld::new(SolverConfig::default());
        w.add_body(RigidBody::new(Vec3Fix::from_int(0, 10, 0), Fix128::ONE));
        w
    }

    #[test]
    fn a_bridge_panic_faults_the_world_and_the_bridge_paths_refuse_it() {
        let mut compared = 0;
        for path in ["step_with_bridge", "substep_with_bridge"] {
            let mut w = falling_ball();
            let mut bridge = PanicsOnUpload::default();
            let caught = catch_unwind(AssertUnwindSafe(|| match path {
                "step_with_bridge" => drop(w.step_with_bridge(&mut bridge, dt())),
                _ => drop(w.substep_with_bridge(&mut bridge, dt())),
            }));
            assert!(caught.is_err(), "{path}: the bridge panicked");
            assert!(
                w.fault().is_some(),
                "{path}: a bridge panic and the world records no fault"
            );
            let before = positions(&w);
            let again = catch_unwind(AssertUnwindSafe(|| match path {
                "step_with_bridge" => drop(w.step_with_bridge(&mut bridge, dt())),
                _ => drop(w.substep_with_bridge(&mut bridge, dt())),
            }));
            assert!(
                again.is_ok(),
                "{path}: a faulted world refuses the bridge step instead of running it"
            );
            assert_eq!(
                positions(&w),
                before,
                "{path}: a refused bridge step changes no body"
            );
            compared += 1;
        }
        assert_eq!(compared, 2);
    }
}
