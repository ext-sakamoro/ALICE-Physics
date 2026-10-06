//! The world side of the participant contract, checked against closed forms
//! and against worlds built without participants.
//!
//! Oracles:
//!
//! * **Constant force, semi-implicit Euler.** A participant stages a constant
//!   force `F` on a free body of mass `m` in every substep. The world applies
//!   it before the substep body as `v += (F/m)·h`, and the substep then moves
//!   the body by `x += v·h` (no gravity, no damping, no contact). After `k`
//!   substeps of width `h`, with `a = F/m` and `t = k·h`:
//!
//!   ```text
//!   v_k = v0 + k·a·h                         = v0 + a·t
//!   x_k = x0 + Σ_{j=1..k} v_j·h
//!       = x0 + k·v0·h + a·h²·k(k+1)/2         = x0 + v0·t + a·t²/2 + a·t·h/2
//!   ```
//!
//!   The last term is the first-order error of the scheme against the
//!   continuous `x0 + v0·t + F·t²/(2m)`; it vanishes as `h → 0`. Every value
//!   below is a dyadic rational (`h = 1/256`, `a = 3/2`), so the closed form is
//!   exact in [`Fix128`] and is compared with `assert_eq!`.
//! * **Zero participants** are compared byte for byte with worlds that never
//!   registered anything, and the version 2 blob of such a world with the
//!   version 1 fixture written before the format moved.
//! * **Faults** are compared with a world without the faulty participant.

#![cfg(feature = "std")]

use std::sync::{Arc, Mutex};

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{
    PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, WorldSnapshotError,
};
use alice_physics::world_participant::{
    FieldLayout, FieldMode, ObservationSink, Observed, Participant, ParticipantFault,
    ParticipantKind, Port, PortId, RegisterError, StateError, StepError, StepRule, SubstepCtx,
    WorldFault,
};

const KIND_PUSH: ParticipantKind = ParticipantKind::new(0x5055_5348);

/// Stages `force` on `body` in every substep; records the `h` it was given.
struct Push {
    body: usize,
    force: Vec3Fix,
    rule: StepRule,
    seen_h: Arc<Mutex<Vec<Fix128>>>,
    fail: bool,
}

impl Push {
    fn new(body: usize, force: Vec3Fix) -> Self {
        Self {
            body,
            force,
            rule: StepRule::FollowSubstep,
            seen_h: Arc::new(Mutex::new(Vec::new())),
            fail: false,
        }
    }
}

impl Participant for Push {
    fn kind(&self) -> ParticipantKind {
        KIND_PUSH
    }
    fn step_rule(&self) -> StepRule {
        self.rule
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        self.seen_h.lock().expect("lock").push(h);
        ctx.add_force(self.body, self.force)
            .map_err(|_| ParticipantFault::InvalidState)?;
        if self.fail {
            return Err(ParticipantFault::InvalidState);
        }
        Ok(())
    }
    fn observe(&self, out: &mut ObservationSink) {
        out.push(0, self.force.x);
    }
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(StateError::Length {
                expected: 0,
                found: bytes.len(),
            })
        }
    }
    fn read_state(&mut self, _: &[u8]) {}
}

/// Writes `value` into a [`FieldMode::Sum`] field in every substep.
struct SumWriter {
    field: PortId,
    value: Fix128,
    ports: Vec<Port>,
}

impl SumWriter {
    fn new(field: PortId, value: Fix128) -> Self {
        Self {
            field,
            value,
            ports: vec![Port::writes(field)],
        }
    }
}

impl Participant for SumWriter {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0x5355_4d57)
    }
    fn ports(&self) -> &[Port] {
        &self.ports
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        let staged = ctx
            .stage_field(self.field)
            .map_err(|_| ParticipantFault::InvalidState)?;
        staged.fill(self.value);
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(StateError::InvalidValue)
        }
    }
    fn read_state(&mut self, _: &[u8]) {}
}

fn dyadic_dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn free_config(backend: SolverBackend) -> PhysicsConfig {
    PhysicsConfig {
        substeps: 4,
        iterations: 4,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        solver_backend: backend,
        ..Default::default()
    }
}

const MASS: (i64, i64) = (2, 1);

/// A free body of mass 2 at the origin moving at `v0 = 1/2` along x, and a
/// participant pushing it with `F = 3` along x.
fn free_body_world(backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(free_config(backend));
    let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_ratio(MASS.0, MASS.1));
    b.velocity = Vec3Fix::new(Fix128::from_ratio(1, 2), Fix128::ZERO, Fix128::ZERO);
    w.add_body(b);
    w.add_participant(Box::new(Push::new(0, Vec3Fix::from_int(3, 0, 0))))
        .expect("register");
    w
}

/// `(x_k, v_k)` of the closed form in the module documentation.
fn closed_form(k: i64) -> (Fix128, Fix128) {
    let v0 = Fix128::from_ratio(1, 2);
    let a = Fix128::from_ratio(3, 2);
    let h = Fix128::from_ratio(1, 256);
    let kk = Fix128::from_int(k);
    let v = v0 + kk * a * h;
    let x = kk * v0 * h + a * h * h * Fix128::from_int(k * (k + 1) / 2);
    (x, v)
}

const FRAMES: i64 = 10;

fn assert_closed_form(w: &PhysicsWorld, path: &str) {
    let (x, v) = closed_form(FRAMES * 4);
    let b = &w.bodies[0];
    assert_eq!(b.position.x, x, "{path}: x");
    assert_eq!(b.velocity.x, v, "{path}: v");
    assert_eq!(b.position.y, Fix128::ZERO, "{path}: y");
    assert_eq!(b.position.z, Fix128::ZERO, "{path}: z");
}

#[test]
fn a_constant_force_follows_the_semi_implicit_euler_closed_form() {
    let mut w = free_body_world(SolverBackend::Xpbd);
    for _ in 0..FRAMES {
        w.try_step(dyadic_dt()).expect("step");
    }
    assert_closed_form(&w, "xpbd");
    // the closed form is the continuous one plus a·t·h/2
    let t = Fix128::from_ratio(FRAMES * 4, 256);
    let a = Fix128::from_ratio(3, 2);
    let continuous = Fix128::from_ratio(1, 2) * t + a * t * t / Fix128::from_int(2);
    let h = Fix128::from_ratio(1, 256);
    assert_eq!(
        w.bodies[0].position.x - continuous,
        a * t * h / Fix128::from_int(2)
    );
}

#[test]
fn the_step_wrapper_gives_the_checked_step() {
    let mut a = free_body_world(SolverBackend::Xpbd);
    let mut b = free_body_world(SolverBackend::Xpbd);
    for _ in 0..FRAMES {
        a.step(dyadic_dt());
        b.try_step(dyadic_dt()).expect("step");
    }
    assert_eq!(a.snapshot_world(), b.snapshot_world());
}

#[test]
fn tgs_with_a_participant_follows_the_same_closed_form() {
    let mut w = free_body_world(SolverBackend::Tgs);
    for _ in 0..FRAMES {
        w.try_step(dyadic_dt()).expect("step");
    }
    assert_closed_form(&w, "tgs");
}

#[cfg(feature = "parallel")]
#[test]
fn the_parallel_path_follows_the_same_closed_form() {
    let mut w = free_body_world(SolverBackend::Xpbd);
    for _ in 0..FRAMES {
        w.try_step_parallel(dyadic_dt()).expect("step");
    }
    assert_closed_form(&w, "parallel");
}

#[cfg(feature = "gpu-solver-bridge")]
mod bridge {
    use super::*;
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::solver::ContactConstraint;

    /// A bridge that solves nothing (the scene has no contact).
    struct Idle;

    impl GpuSolverBridge for Idle {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _fixture: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, _c: &[ContactConstraint]) {}
        fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {}
        fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {}
        fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
        fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
    }

    #[test]
    fn the_bridge_path_follows_the_same_closed_form() {
        let mut w = free_body_world(SolverBackend::Xpbd);
        for _ in 0..FRAMES {
            w.step_with_bridge(&mut Idle, dyadic_dt());
        }
        assert_closed_form(&w, "bridge");
    }
}

/// Every path hands the participant `h = dt / substeps`, divided in
/// [`Fix128`]: with 3 substeps `1/3` is not a binary fraction, so a width
/// taken from an `f32` reciprocal differs in the low bits.
#[test]
fn every_path_hands_participants_dt_over_substeps() {
    let dt = Fix128::from_ratio(1, 60);
    let expected = dt / Fix128::from_int(3);
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            substeps: 3,
            solver_backend: backend,
            ..Default::default()
        });
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let push = Push::new(0, Vec3Fix::ZERO);
        let seen = Arc::clone(&push.seen_h);
        w.add_participant(Box::new(push)).expect("register");
        w.try_step(dt).expect("step");
        assert_eq!(
            *seen.lock().expect("lock"),
            vec![expected; 3],
            "{backend:?}"
        );
    }
}

// ── Zero participants ────────────────────────────────────────────────────

fn stacked(backend: SolverBackend) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 4,
        iterations: 4,
        solver_backend: backend,
        ..Default::default()
    });
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    for i in 0..3 {
        w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 1 + 2 * i, 0), Fix128::ONE),
            Fix128::ONE,
        );
    }
    w
}

/// With nothing registered the checked step is the step, on every backend,
/// byte for byte in the whole snapshot.
#[test]
fn zero_participants_are_the_unchecked_step_byte_for_byte() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let mut a = stacked(backend);
        let mut b = stacked(backend);
        for _ in 0..90 {
            a.step(Fix128::from_ratio(1, 60));
            b.try_step(Fix128::from_ratio(1, 60)).expect("step");
        }
        assert_eq!(a.snapshot_world(), b.snapshot_world(), "{backend:?}");
        assert_eq!(b.fault(), None);
    }
}

/// The version 2 blob of a world without participants is the version 1
/// payload followed by an empty `participants` section, no fault and an empty
/// `fields` section; only the version, the length and the checksum differ.
#[test]
fn a_version_2_blob_extends_the_version_1_payload() {
    let v1: &[u8] = include_bytes!("fixtures/world_snapshot_v1_stacked.bin");
    let w = PhysicsWorld::from_world_snapshot(v1).expect("v1 blob");
    let v2 = w.snapshot_world();
    assert_eq!(&v2[0..4], &v1[0..4]);
    assert_eq!(&v2[4..6], &2u16.to_le_bytes());
    let len = |b: &[u8]| u64::from_le_bytes(b[8..16].try_into().expect("8 bytes")) as usize;
    let (l1, l2) = (len(v1), len(&v2));
    assert_eq!(l2, l1 + 8 + 1 + 8);
    assert_eq!(&v2[16..16 + l1], &v1[16..16 + l1]);
    assert_eq!(&v2[16 + l1..16 + l2], &[0u8; 17][..]);
}

/// The version 1 fixture steps on as the world it was taken from: a world
/// stepped once and then 60 more times equals the fixture restored and
/// stepped 60 times.
#[test]
fn a_version_1_blob_restores_the_world_it_was_taken_from() {
    let v1: &[u8] = include_bytes!("fixtures/world_snapshot_v1_stacked.bin");
    let config = PhysicsConfig {
        substeps: 4,
        iterations: 4,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        ..Default::default()
    };
    let mut original = PhysicsWorld::new(config);
    original.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    for i in 0..3 {
        original.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 1 + 2 * i, 0), Fix128::ONE),
            Fix128::ONE,
        );
    }
    original.step(Fix128::from_ratio(1, 60));
    let mut restored = PhysicsWorld::from_world_snapshot(v1).expect("v1 blob");
    assert_eq!(restored.snapshot_world(), original.snapshot_world());
    for _ in 0..60 {
        original.step(Fix128::from_ratio(1, 60));
        restored.step(Fix128::from_ratio(1, 60));
    }
    assert_eq!(restored.snapshot_world(), original.snapshot_world());
}

// ── Waking ───────────────────────────────────────────────────────────────

/// A body that is sleeping but not parked (sleep skip off) follows the same
/// rule as a parked one: woken above the threshold, untouched below it.
#[test]
fn a_sleeping_body_without_the_skip_wakes_only_above_the_threshold() {
    let build = || {
        let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        w.set_sleep_skip(false);
        for _ in 0..120 {
            w.step(dyadic_dt());
        }
        assert!(w.is_sleeping(0), "fixture: the body must be asleep");
        w
    };
    // h = 1/256; threshold 0.01: 2 N on 1 kg gives Δv = 1/128 < 0.01
    let mut below = build();
    let before = below.bodies[0].position;
    below
        .add_participant(Box::new(Push::new(0, Vec3Fix::from_int(2, 0, 0))))
        .expect("register");
    below.try_step(dyadic_dt()).expect("step");
    assert!(below.is_sleeping(0));
    assert_eq!(below.bodies[0].position, before);
    // 3 N gives Δv = 3/256 > 0.01
    let mut above = build();
    above
        .add_participant(Box::new(Push::new(0, Vec3Fix::from_int(3, 0, 0))))
        .expect("register");
    above.try_step(dyadic_dt()).expect("step");
    assert!(!above.is_sleeping(0));
    assert!(above.bodies[0].velocity.x > Fix128::ZERO);
}

/// A parked body woken by a participant force moves exactly as the same body
/// does with the sleep skip off (the skip must not change the result): woken
/// and unparked, it is integrated in the substep that woke it.
#[test]
fn a_woken_parked_body_moves_as_without_the_sleep_skip() {
    let build = |skip: bool| {
        let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        w.set_sleep_skip(skip);
        for _ in 0..120 {
            w.step(dyadic_dt());
        }
        assert!(w.is_sleeping(0), "fixture: the body must be asleep");
        w.add_participant(Box::new(Push::new(0, Vec3Fix::from_int(1000, 0, 0))))
            .expect("register");
        w.try_step(dyadic_dt()).expect("step");
        w
    };
    let parked = build(true);
    let awake = build(false);
    assert!(
        parked.bodies[0].position.x > Fix128::ZERO,
        "the woken body did not move"
    );
    assert_eq!(parked.bodies[0].position, awake.bodies[0].position);
    assert_eq!(parked.bodies[0].velocity, awake.bodies[0].velocity);
}

// ── Faults ───────────────────────────────────────────────────────────────

/// The fault is sticky from the first one: two participants failing in the
/// same substep record the one that ran first.
#[test]
fn the_first_fault_is_kept() {
    let mut w = free_body_world(SolverBackend::Xpbd);
    for _ in 0..2 {
        let mut p = Push::new(0, Vec3Fix::ZERO);
        p.fail = true;
        w.add_participant(Box::new(p)).expect("register");
    }
    let first = WorldFault::Participant {
        index: 1,
        kind: KIND_PUSH,
        fault: ParticipantFault::InvalidState,
    };
    assert_eq!(w.try_step(dyadic_dt()), Err(StepError::FaultRaised(first)));
    assert_eq!(w.fault(), Some(first));
}

/// `F·inv_mass·h` past the range of [`Fix128`]: the force is not applied (the
/// body moves as in a world without the participant) and the fault names the
/// body.
#[test]
fn an_overflowing_force_is_a_fault_and_leaves_the_body_alone() {
    let build = || {
        let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
        let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_ratio(1, 1024));
        b.velocity = Vec3Fix::from_int(1, 0, 0);
        w.add_body(b);
        w
    };
    let mut w = build();
    let huge = Fix128::from_int(1 << 62);
    w.add_participant(Box::new(Push::new(
        0,
        Vec3Fix::new(huge, Fix128::ZERO, Fix128::ZERO),
    )))
    .expect("register");
    let mut reference = build();
    let fault = WorldFault::ForceOutOfRange { body: 0 };
    assert_eq!(w.try_step(dyadic_dt()), Err(StepError::FaultRaised(fault)));
    reference.step(dyadic_dt());
    assert_eq!(w.bodies[0].position, reference.bodies[0].position);
    assert_eq!(w.bodies[0].velocity, reference.bodies[0].velocity);
    assert_eq!(w.fault(), Some(fault));
    assert_eq!(w.try_step(dyadic_dt()), Err(StepError::Faulted(fault)));
}

/// The scene of `tests/wm01_overflow_is_not_silent.rs` (`g = -2³⁰`,
/// `dt = 2²⁰`) raises the world's overflow flag in its first step.
fn washing_world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-(1 << 30)), Fix128::ZERO),
        ..Default::default()
    });
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    w
}

fn washing_dt() -> Fix128 {
    Fix128::from_int(1 << 20)
}

#[test]
fn the_overflow_flag_going_up_is_a_rigid_overflow_fault() {
    let mut w = washing_world();
    w.add_participant(Box::new(Push::new(0, Vec3Fix::ZERO)))
        .expect("register");
    assert!(!w.overflow_detected());
    assert_eq!(
        w.try_step(washing_dt()),
        Err(StepError::FaultRaised(WorldFault::RigidOverflow))
    );
    assert!(w.overflow_detected());
}

/// Only the rising edge counts: a flag already up before the participants
/// arrived is not a new fault, and a world without participants never
/// records one (it keeps stepping, as it always did).
#[test]
fn a_flag_already_up_and_a_world_without_participants_record_no_fault() {
    let mut w = washing_world();
    w.step(washing_dt());
    assert!(w.overflow_detected());
    assert_eq!(w.fault(), None, "a world without participants");
    w.add_participant(Box::new(Push::new(0, Vec3Fix::ZERO)))
        .expect("register");
    assert_eq!(w.try_step(washing_dt()), Ok(()));
    assert_eq!(w.fault(), None);
    assert_eq!(
        w.observe_body_checked(0),
        Some(Observed::Undecided),
        "the flag still makes the body undecided"
    );
}

/// A sum field whose committed sum leaves the range is a field fault naming
/// the writer that pushed it out (the second one, in run order).
#[test]
fn a_sum_field_out_of_range_is_a_field_fault() {
    let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let heat = PortId::new(7);
    w.declare_field(heat, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum)
        .expect("declare");
    let big = Fix128::from_int(1 << 62);
    w.add_participant(Box::new(SumWriter::new(heat, big)))
        .expect("register");
    w.add_participant(Box::new(SumWriter::new(heat, big)))
        .expect("register");
    let fault = WorldFault::FieldOutOfRange {
        field: heat,
        index: 1,
    };
    assert_eq!(w.try_step(dyadic_dt()), Err(StepError::FaultRaised(fault)));
}

/// Every fault code survives a snapshot: the restored world is faulted the
/// same way and writes the same blob.
#[test]
fn every_fault_kind_round_trips_through_the_snapshot() {
    let mut worlds: Vec<(PhysicsWorld, Box<dyn Fn() -> PhysicsWorld>)> = Vec::new();

    // participant
    let build: Box<dyn Fn() -> PhysicsWorld> = Box::new(|| {
        let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let mut p = Push::new(0, Vec3Fix::ZERO);
        p.fail = true;
        w.add_participant(Box::new(p)).expect("register");
        w
    });
    worlds.push((build(), build));
    // rigid overflow
    let build: Box<dyn Fn() -> PhysicsWorld> = Box::new(|| {
        let mut w = washing_world();
        w.add_participant(Box::new(Push::new(0, Vec3Fix::ZERO)))
            .expect("register");
        w
    });
    worlds.push((build(), build));
    // force out of range
    let build: Box<dyn Fn() -> PhysicsWorld> = Box::new(|| {
        let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::ZERO,
            Fix128::from_ratio(1, 1024),
        ));
        w.add_participant(Box::new(Push::new(
            0,
            Vec3Fix::new(Fix128::from_int(1 << 62), Fix128::ZERO, Fix128::ZERO),
        )))
        .expect("register");
        w
    });
    worlds.push((build(), build));
    // field out of range
    let build: Box<dyn Fn() -> PhysicsWorld> = Box::new(|| {
        let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let heat = PortId::new(7);
        w.declare_field(heat, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum)
            .expect("declare");
        for _ in 0..2 {
            w.add_participant(Box::new(SumWriter::new(heat, Fix128::from_int(1 << 62))))
                .expect("register");
        }
        w
    });
    worlds.push((build(), build));

    let mut kinds = Vec::new();
    for (mut w, build) in worlds {
        let dt = if w.config.gravity.y < Fix128::from_int(-1000) {
            washing_dt()
        } else {
            dyadic_dt()
        };
        assert!(matches!(w.try_step(dt), Err(StepError::FaultRaised(_))));
        let fault = w.fault().expect("faulted");
        let blob = w.snapshot_world();
        let mut restored = build();
        restored.restore_world(&blob).expect("restore");
        assert_eq!(restored.fault(), Some(fault));
        assert_eq!(restored.snapshot_world(), blob);
        kinds.push(fault);
    }
    assert!(matches!(kinds[0], WorldFault::Participant { .. }));
    assert_eq!(kinds[1], WorldFault::RigidOverflow);
    assert_eq!(kinds[2], WorldFault::ForceOutOfRange { body: 0 });
    assert!(matches!(kinds[3], WorldFault::FieldOutOfRange { .. }));
}

/// An unknown fault code in the blob is refused as an invalid value.
#[test]
fn an_unknown_fault_code_is_refused() {
    let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let mut blob = w.snapshot_world();
    // the fault code sits before the empty fields section (8 bytes) and the
    // checksum (8 bytes)
    let at = blob.len() - 8 - 8 - 1;
    assert_eq!(blob[at], 0);
    blob[at] = 9;
    let body_end = blob.len() - 8;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &blob[..body_end] {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0001_0000_01b3);
    }
    blob[body_end..].copy_from_slice(&hash.to_le_bytes());
    let before = w.snapshot_world();
    assert_eq!(
        w.restore_world(&blob),
        Err(WorldSnapshotError::InvalidValue { section: "fault" })
    );
    assert_eq!(w.snapshot_world(), before);
}

// ── Degenerate inputs ────────────────────────────────────────────────────

#[test]
fn a_non_positive_dt_changes_nothing() {
    let mut w = free_body_world(SolverBackend::Xpbd);
    let before = w.snapshot_world();
    assert_eq!(w.try_step(Fix128::ZERO), Ok(()));
    assert_eq!(w.try_step(-dyadic_dt()), Ok(()));
    assert_eq!(w.snapshot_world(), before);
}

/// With 0 substeps the substep width is 0, which no step rule follows: the
/// step is refused and changes nothing.
#[test]
fn zero_substeps_are_refused_with_participants() {
    let mut w = free_body_world(SolverBackend::Xpbd);
    w.config.substeps = 0;
    let before = w.snapshot_world();
    assert_eq!(
        w.try_step(dyadic_dt()),
        Err(StepError::Rule {
            index: 0,
            error: RegisterError::NonPositiveStep
        })
    );
    assert_eq!(w.snapshot_world(), before);
}

#[test]
fn fixed_step_rules_are_checked_at_registration_and_at_every_step() {
    let mut w = free_body_world(SolverBackend::Xpbd);
    let mut zero = Push::new(0, Vec3Fix::ZERO);
    zero.rule = StepRule::Fixed(Fix128::ZERO);
    assert_eq!(
        w.add_participant(Box::new(zero)),
        Err(RegisterError::NonPositiveStep)
    );
    let mut third = Push::new(0, Vec3Fix::ZERO);
    // h = 1/256 is not a multiple of 1/3
    third.rule = StepRule::Fixed(Fix128::from_ratio(1, 3));
    assert_eq!(w.add_participant(Box::new(third)), Ok(1));
    let before = w.snapshot_world();
    assert!(matches!(
        w.try_step(dyadic_dt()),
        Err(StepError::Rule {
            index: 1,
            error: RegisterError::StepRuleMismatch { .. }
        })
    ));
    assert_eq!(w.snapshot_world(), before);
}

/// Bodies added after the participants: the forces follow the new body list;
/// a per-body field that no longer has one sample per body refuses the step.
#[test]
fn bodies_added_after_registration() {
    let mut w = free_body_world(SolverBackend::Xpbd);
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(5, 0, 0),
        Fix128::ONE,
    ));
    w.try_step(dyadic_dt()).expect("step");
    assert_eq!(w.bodies[1].velocity, Vec3Fix::ZERO);

    let mut w = free_body_world(SolverBackend::Xpbd);
    let heat = PortId::new(9);
    w.declare_field(heat, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum)
        .expect("declare");
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(5, 0, 0),
        Fix128::ONE,
    ));
    let before = w.snapshot_world();
    assert_eq!(
        w.try_step(dyadic_dt()),
        Err(StepError::BodyCount {
            field: heat,
            samples: 1,
            bodies: 2
        })
    );
    assert_eq!(w.snapshot_world(), before);
}

/// A field declared after the participants that would make their ports
/// invalid is refused and the world keeps its fields.
#[test]
fn a_field_that_breaks_registered_ports_is_refused() {
    let mut w = free_body_world(SolverBackend::Xpbd);
    let a = PortId::new(3);
    w.add_participant(Box::new(SumWriter::new(a, Fix128::ONE)))
        .expect("register");
    w.add_participant(Box::new(SumWriter::new(a, Fix128::ONE)))
        .expect("register");
    let err = w
        .declare_field(a, FieldLayout::PerBody { bodies: 1 }, FieldMode::Replace)
        .expect_err("two writers of a replace field");
    assert!(matches!(
        err,
        alice_physics::solver::DeclareFieldError::Ports(RegisterError::Field(_))
    ));
    assert!(w.fields().ids().is_empty());
    w.declare_field(a, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum)
        .expect("a sum field takes both writers");
    assert_eq!(w.fields().ids(), vec![a]);
}

#[test]
fn observing_out_of_range_indices_gives_none() {
    let w = free_body_world(SolverBackend::Xpbd);
    assert!(w.observe_participant(1).is_none());
    assert!(w.participant_state(1).is_none());
    assert!(w.observe_body_checked(1).is_none());
    assert_eq!(w.participant_count(), 1);
    match w.observe_participant(0) {
        Some(Observed::Exact(sink)) => {
            assert_eq!(sink.values(), &[(0, Fix128::from_int(3))]);
        }
        other => panic!("expected an exact observation, got {other:?}"),
    }
}

/// `reset_world` drops the participants, the fault and the fields.
#[test]
fn reset_world_drops_participants_fault_and_fields() {
    let mut w = washing_world();
    w.add_participant(Box::new(Push::new(0, Vec3Fix::ZERO)))
        .expect("register");
    w.declare_field(
        PortId::new(1),
        FieldLayout::PerBody { bodies: 1 },
        FieldMode::Sum,
    )
    .expect("declare");
    assert!(w.try_step(washing_dt()).is_err());
    w.reset_world();
    assert_eq!(w.participant_count(), 0);
    assert_eq!(w.fault(), None);
    assert!(w.fields().ids().is_empty());
}
