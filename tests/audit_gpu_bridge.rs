//! Audit oracles for `alice_physics::gpu_bridge` (trait defaults and host routing).
#![cfg(feature = "gpu-solver-bridge")]
#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Contact;
use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{ContactConstraint, PhysicsConfig, PhysicsWorld, RigidBody};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

/// Implements only the four required methods; every contact / joint method keeps the default.
struct Minimal;
impl GpuSolverBridge for Minimal {
    fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
    fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
    fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
    fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
        Ok(())
    }
}

fn panic_text(f: impl FnOnce()) -> Option<String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(prev);
    match r {
        Ok(()) => None,
        Err(e) => Some(
            e.downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_default(),
        ),
    }
}

/// Every default method documents "Default: panics" with a message naming itself and
/// "not implemented by this GpuSolverBridge backend".
#[test]
fn every_default_method_panics_naming_itself() {
    // serialise: the panic hook is process global
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut b = Minimal;
    let cases: Vec<(&str, Option<String>)> = vec![
        ("send_contact_constraints", panic_text(b_send_cc)),
        (
            "send_body_state",
            panic_text(|| Minimal.send_body_state(&[], &[])),
        ),
        (
            "dispatch_contact_solve_iteration",
            panic_text(|| Minimal.dispatch_contact_solve_iteration(Fix128::ONE)),
        ),
        (
            "recv_contact_constraints",
            panic_text(|| Minimal.recv_contact_constraints(&mut [])),
        ),
        (
            "recv_body_positions",
            panic_text(|| Minimal.recv_body_positions(&mut [])),
        ),
        ("send_joints", panic_text(|| Minimal.send_joints(&[]))),
        (
            "send_body_rotations",
            panic_text(|| Minimal.send_body_rotations(&[])),
        ),
        (
            "dispatch_joint_solve_iteration",
            panic_text(|| Minimal.dispatch_joint_solve_iteration(Fix128::ONE)),
        ),
    ];
    let _ = &mut b;
    for (name, msg) in cases {
        let msg = msg.unwrap_or_else(|| panic!("{name} did not panic"));
        assert!(
            msg.contains(name),
            "{name}: message {msg:?} does not name the method"
        );
        assert!(
            msg.contains("not implemented by this GpuSolverBridge backend"),
            "{name}: {msg:?}"
        );
    }
}

fn b_send_cc() {
    Minimal.send_contact_constraints(&[]);
}

/// The required methods work through a trait object (object safety) and the diff gate
/// returns the structured divergence it is documented to carry.
struct Diverging;
impl GpuSolverBridge for Diverging {
    fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
    fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
    fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
    fn assert_bit_exact_vs_cpu(&self, f: &DiffFixture) -> Result<(), GpuDivergence> {
        if f.tolerance == Fix128::ZERO {
            Err(GpuDivergence {
                axis: "body[3].position.y",
                cpu_hi: 1,
                cpu_lo: 2,
                gpu_hi: 1,
                gpu_lo: 3,
            })
        } else {
            Ok(())
        }
    }
}

#[test]
fn trait_object_and_divergence_report_round_trip() {
    let mut b: Box<dyn GpuSolverBridge> = Box::new(Diverging);
    b.send_island(&[[Fix128::ONE; 3]], &[[Fix128::ZERO; 3]]);
    let strict = DiffFixture {
        description: "strict",
        tolerance: Fix128::ZERO,
    };
    let e = b.assert_bit_exact_vs_cpu(&strict).unwrap_err();
    assert_eq!(
        (e.axis, e.cpu_hi, e.cpu_lo, e.gpu_hi, e.gpu_lo),
        ("body[3].position.y", 1, 2, 1, 3)
    );
    let loose = DiffFixture {
        description: "loose",
        tolerance: Fix128::from_ratio(1, 1000),
    };
    assert!(b.assert_bit_exact_vs_cpu(&loose).is_ok());
    assert_eq!(strict.clone().description, "strict");
}

struct Recorder {
    seen_len: Arc<Mutex<Vec<usize>>>,
    seen_first_bodies: Arc<Mutex<Vec<(usize, usize)>>>,
}
impl GpuSolverBridge for Recorder {
    fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
    fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
    fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
    fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
        Ok(())
    }
    fn send_contact_constraints(&mut self, c: &[ContactConstraint]) {
        self.seen_len.lock().unwrap().push(c.len());
        if let Some(f) = c.first() {
            self.seen_first_bodies
                .lock()
                .unwrap()
                .push((f.body_a, f.body_b));
        }
    }
    fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {}
    fn dispatch_contact_solve_iteration(&mut self, _w: Fix128) {}
    fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
    fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
}

fn contact() -> Contact {
    Contact {
        depth: Fix128::from_ratio(1, 10),
        normal: Vec3Fix::UNIT_X,
        point_a: Vec3Fix::ZERO,
        point_b: Vec3Fix::ZERO,
    }
}

/// `send_contact_constraints` doc: "Element indices must line up with the caller's
/// `PhysicsWorld::contact_constraints` slot ordering". With a sensor body in slot 0 the host
/// skips that slot and sends a compacted list, so the bridge cannot index by slot.
#[test]
#[ignore = "known defect: AUD-A-S2W3-003: send_contact_constraints doc says indices line up with contact_constraints slots, host sends a sensor/hook-filtered compacted list (2 slots -> 1 sent)"]
fn contact_upload_keeps_slot_alignment_with_a_sensor_in_slot_zero() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let a = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(1, 0, 0),
        Fix128::ONE,
    ));
    let c = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(2, 0, 0),
        Fix128::ONE,
    ));
    w.bodies[a].is_sensor = true;
    w.contact_constraints
        .push(ContactConstraint::new(a, b, contact()));
    w.contact_constraints
        .push(ContactConstraint::new(b, c, contact()));
    let lens = Arc::new(Mutex::new(vec![]));
    let firsts = Arc::new(Mutex::new(vec![]));
    let mut rec = Recorder {
        seen_len: Arc::clone(&lens),
        seen_first_bodies: Arc::clone(&firsts),
    };
    w.solve_contact_constraints_with_bridge(&mut rec);
    assert_eq!(
        *lens.lock().unwrap(),
        vec![2],
        "uploaded list must have one entry per slot"
    );
    assert_eq!(firsts.lock().unwrap()[0], (a, b), "slot 0 must be (a, b)");
}

/// Companion (green): the sensor is skipped from the upload and the non-sensor constraint's
/// `cached_lambda` is written back to its own slot (slot 1), not slot 0.
#[test]
fn filtered_upload_writes_lambda_back_to_the_original_slot() {
    struct Lam;
    impl GpuSolverBridge for Lam {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _i: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _f: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, _c: &[ContactConstraint]) {}
        fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {}
        fn dispatch_contact_solve_iteration(&mut self, _w: Fix128) {}
        fn recv_contact_constraints(&self, c: &mut [ContactConstraint]) {
            for (i, k) in c.iter_mut().enumerate() {
                k.cached_lambda = Fix128::from_int(7 + i as i64);
            }
        }
        fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
    }
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let a = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(1, 0, 0),
        Fix128::ONE,
    ));
    let c = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(2, 0, 0),
        Fix128::ONE,
    ));
    w.bodies[a].is_sensor = true;
    w.contact_constraints
        .push(ContactConstraint::new(a, b, contact()));
    w.contact_constraints
        .push(ContactConstraint::new(b, c, contact()));
    w.solve_contact_constraints_with_bridge(&mut Lam);
    assert_eq!(w.contact_constraints[0].cached_lambda, Fix128::ZERO);
    assert_eq!(w.contact_constraints[1].cached_lambda, Fix128::from_int(7));
}
