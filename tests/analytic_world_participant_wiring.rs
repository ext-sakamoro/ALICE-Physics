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

use alice_physics::joint::{BallJoint, Joint};
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
    pub(super) struct Idle;

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

/// Every path hands the participant the substep width its own rigid solve
/// uses. XPBD, the parallel path and the bridge divide in [`Fix128`],
/// `h = dt / substeps`; TGS uses its own width `dt · from_f32(1 / substeps)`
/// (the split `tgs_step` makes). With 3 substeps `1/3` is not a binary
/// fraction, so the two widths differ in the low bits; handing TGS's
/// participants its own width is what keeps a participant that stages
/// nothing from changing the TGS bodies.
#[test]
fn every_path_hands_participants_the_width_its_solve_uses() {
    let dt = Fix128::from_ratio(1, 60);
    let divided = dt / Fix128::from_int(3);
    let tgs = dt * Fix128::from_f32(1.0 / 3.0);
    assert_ne!(divided, tgs, "fixture: the two widths must differ");
    for (backend, expected) in [(SolverBackend::Xpbd, divided), (SolverBackend::Tgs, tgs)] {
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

/// A [`StepRule::Fixed`] step is checked against the width the participants
/// are handed: under TGS with 3 substeps a fixed step equal to TGS's own
/// width runs, one equal to `dt / 3` (which differs from it in the low bits)
/// is refused unchanged.
#[test]
fn a_fixed_step_rule_is_checked_against_the_tgs_width() {
    let dt = Fix128::from_ratio(1, 60);
    let tgs = dt * Fix128::from_f32(1.0 / 3.0);
    let divided = dt / Fix128::from_int(3);
    let build = |step: Fix128| {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            substeps: 3,
            solver_backend: SolverBackend::Tgs,
            ..Default::default()
        });
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let mut push = Push::new(0, Vec3Fix::ZERO);
        push.rule = StepRule::Fixed(step);
        w.add_participant(Box::new(push)).expect("register");
        w
    };
    let mut accepted = build(tgs);
    assert_eq!(accepted.try_step(dt), Ok(()));
    let mut refused = build(divided);
    let before = refused.snapshot_world();
    assert!(matches!(
        refused.try_step(dt),
        Err(StepError::Rule { index: 0, .. })
    ));
    assert_eq!(refused.snapshot_world(), before);
}

/// The parallel path and the bridge run the XPBD substep loop whatever the
/// backend, so a world configured for TGS hands its participants
/// `h = dt / substeps` there, and a [`StepRule::Fixed`] step is checked
/// against that width: with 3 substeps a fixed step equal to `dt / 3` runs
/// and is handed `dt / 3` in each substep, one equal to TGS's own width
/// (which differs from `dt / 3` in the low bits) is refused, the world
/// unchanged and the participant not called. `step_path` steps once and
/// returns whether the step ran, or `None` on a path that does not report it.
#[cfg(any(feature = "parallel", feature = "gpu-solver-bridge"))]
fn a_tgs_configured_world_checks_the_width_the_path_hands_out(
    step_path: fn(&mut PhysicsWorld, Fix128) -> Option<bool>,
) {
    let dt = Fix128::from_ratio(1, 60);
    let tgs = dt * Fix128::from_f32(1.0 / 3.0);
    let divided = dt / Fix128::from_int(3);
    assert_ne!(divided, tgs, "fixture: the two widths must differ");
    let build = |step: Fix128| {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            substeps: 3,
            solver_backend: SolverBackend::Tgs,
            ..Default::default()
        });
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        let mut push = Push::new(0, Vec3Fix::ZERO);
        push.rule = StepRule::Fixed(step);
        let seen = Arc::clone(&push.seen_h);
        w.add_participant(Box::new(push)).expect("register");
        (w, seen)
    };
    let (mut accepted, seen) = build(divided);
    let before = accepted.snapshot_world();
    assert_ne!(step_path(&mut accepted, dt), Some(false));
    assert_eq!(*seen.lock().expect("lock"), vec![divided; 3]);
    assert_ne!(
        accepted.snapshot_world(),
        before,
        "fixture: the step must move the body"
    );
    let (mut refused, seen) = build(tgs);
    let before = refused.snapshot_world();
    assert_ne!(step_path(&mut refused, dt), Some(true));
    assert!(seen.lock().expect("lock").is_empty());
    assert_eq!(refused.snapshot_world(), before);
}

/// See [`a_tgs_configured_world_checks_the_width_the_path_hands_out`].
#[cfg(feature = "parallel")]
#[test]
fn a_tgs_configured_world_on_the_parallel_path_checks_the_width_it_hands_out() {
    a_tgs_configured_world_checks_the_width_the_path_hands_out(|w, dt| {
        match w.try_step_parallel(dt) {
            Ok(()) => Some(true),
            Err(StepError::Rule { index: 0, .. }) => Some(false),
            other => panic!("unexpected {other:?}"),
        }
    });
}

/// See [`a_tgs_configured_world_checks_the_width_the_path_hands_out`]. The
/// bridge reports nothing: a refused step is one that does nothing.
#[cfg(feature = "gpu-solver-bridge")]
#[test]
fn a_tgs_configured_world_on_the_bridge_checks_the_width_it_hands_out() {
    a_tgs_configured_world_checks_the_width_the_path_hands_out(|w, dt| {
        w.step_with_bridge(&mut bridge::Idle, dt);
        None
    });
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

// ── Torque ───────────────────────────────────────────────────────────────

const KIND_TWIST: ParticipantKind = ParticipantKind::new(0x5457_5354);

/// Stages `torque` on body 0 in every substep and counts its calls (the
/// count is its state, so a restore that reads it is visible).
struct Twist {
    torque: Vec3Fix,
    calls: u64,
}

impl Twist {
    fn new(torque: Vec3Fix) -> Self {
        Self { torque, calls: 0 }
    }
}

impl Participant for Twist {
    fn kind(&self) -> ParticipantKind {
        KIND_TWIST
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        ctx.add_torque(0, self.torque)
            .map_err(|_| ParticipantFault::InvalidState)?;
        self.calls += 1;
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.calls.to_le_bytes());
    }
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() == 8 {
            Ok(())
        } else {
            Err(StateError::Length {
                expected: 8,
                found: bytes.len(),
            })
        }
    }
    fn read_state(&mut self, bytes: &[u8]) {
        let mut b = [0_u8; 8];
        b.copy_from_slice(bytes);
        self.calls = u64::from_le_bytes(b);
    }
}

/// One substep of a torque `τ = 3` about x on a free body at rest with the
/// identity rotation gives `ω = I⁻¹·τ·h` (oracle: the body's own diagonal
/// `inv_inertia` and `h = 1/64`, `substeps = 1`), and nothing about y or z.
/// XPBD derives ω again from the orientation change at the end of the
/// substep (`q += ½·h·ω⊗q`, normalized), which agrees with the applied ω to
/// second order in `|ω|·h`; measured relative difference `5.6e-12`, so the
/// tolerance is `1e-9` relative.
#[test]
fn a_torque_changes_the_angular_velocity_by_inv_inertia_tau_h() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..Default::default()
    });
    let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    b.angular_damping = Fix128::ONE;
    w.add_body(b);
    let inv_i = w.bodies[0].inv_inertia.x;
    assert!(inv_i > Fix128::ZERO, "fixture: a body that can turn");
    w.add_participant(Box::new(Twist::new(Vec3Fix::from_int(3, 0, 0))))
        .expect("register");
    let h = Fix128::from_ratio(1, 64);
    w.try_step(h).expect("step");
    let expected = Fix128::from_int(3) * inv_i * h;
    let got = w.bodies[0].angular_velocity.x;
    let rel = ((got - expected).to_f64() / expected.to_f64()).abs();
    assert!(
        rel < 1e-9,
        "ω.x {} vs I⁻¹τh {} (relative {rel:e})",
        got.to_f64(),
        expected.to_f64()
    );
    assert_eq!(w.bodies[0].angular_velocity.y, Fix128::ZERO);
    assert_eq!(w.bodies[0].angular_velocity.z, Fix128::ZERO);
}

/// A light body (mass `2⁻²⁰`, so `inv_inertia ≈ 2.6e6`) under `τ = 2⁵⁰`:
/// `I⁻¹·τ ≈ 2.9e21` is past the range of [`Fix128`] (`2⁶³`) before it is
/// scaled by `h`. The torque is not applied (the body turns as in a world
/// without the participant) and the fault names the body.
#[test]
fn an_out_of_range_inv_inertia_torque_is_a_fault_and_leaves_the_body_alone() {
    let build = || {
        let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::ZERO,
            Fix128::from_ratio(1, 1 << 20),
        ));
        w
    };
    let mut w = build();
    let inv_i = w.bodies[0].inv_inertia.x;
    assert!(
        inv_i.checked_mul(Fix128::from_int(1 << 50)).is_none(),
        "fixture: I⁻¹τ must leave the range, inv_inertia {}",
        inv_i.to_f64()
    );
    w.add_participant(Box::new(Twist::new(Vec3Fix::new(
        Fix128::from_int(1 << 50),
        Fix128::ZERO,
        Fix128::ZERO,
    ))))
    .expect("register");
    let mut reference = build();
    let fault = WorldFault::ForceOutOfRange { body: 0 };
    assert_eq!(w.try_step(dyadic_dt()), Err(StepError::FaultRaised(fault)));
    reference.step(dyadic_dt());
    assert_eq!(
        w.bodies[0].angular_velocity,
        reference.bodies[0].angular_velocity
    );
    assert_eq!(w.bodies[0].rotation, reference.bodies[0].rotation);
    assert_eq!(w.bodies[0].velocity, reference.bodies[0].velocity);
    assert_eq!(w.fault(), Some(fault));
}

/// A parked body (mass 1/4, so `inv_mass = 4`, and `inv_inertia` set to 4)
/// under a torque or a force of `2⁶²`: `I⁻¹·τ` and `F·inv_mass` are `2⁶⁴`,
/// past the range of [`Fix128`]; a wrapping product would be exactly 0 and
/// leave the body parked. A change too large to represent wakes the body
/// ([`alice_physics::world_participant::wakes_parked_body`]) and applying it
/// is then the fault of the awake case, with the body unchanged.
fn parked_light_body() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(free_config(SolverBackend::Xpbd));
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::ZERO,
        Fix128::from_ratio(1, 4),
    ));
    for _ in 0..120 {
        w.step(dyadic_dt());
    }
    assert!(w.is_sleeping(0), "fixture: the body must be asleep");
    w.bodies[0].inv_inertia = Vec3Fix::from_int(4, 4, 4);
    w
}

#[test]
fn an_out_of_range_inv_inertia_torque_wakes_a_parked_body_and_is_a_fault() {
    let mut w = parked_light_body();
    let before = (w.bodies[0].rotation, w.bodies[0].angular_velocity);
    w.add_participant(Box::new(Twist::new(Vec3Fix::new(
        Fix128::from_int(1 << 62),
        Fix128::ZERO,
        Fix128::ZERO,
    ))))
    .expect("register");
    let fault = WorldFault::ForceOutOfRange { body: 0 };
    assert_eq!(w.try_step(dyadic_dt()), Err(StepError::FaultRaised(fault)));
    assert!(
        !w.is_sleeping(0),
        "an unrepresentable change wakes the body"
    );
    assert_eq!((w.bodies[0].rotation, w.bodies[0].angular_velocity), before);
}

#[test]
fn an_out_of_range_force_wakes_a_parked_body_and_is_a_fault() {
    let mut w = parked_light_body();
    let before = (w.bodies[0].position, w.bodies[0].velocity);
    w.add_participant(Box::new(Push::new(
        0,
        Vec3Fix::new(Fix128::from_int(1 << 62), Fix128::ZERO, Fix128::ZERO),
    )))
    .expect("register");
    let fault = WorldFault::ForceOutOfRange { body: 0 };
    assert_eq!(w.try_step(dyadic_dt()), Err(StepError::FaultRaised(fault)));
    assert!(
        !w.is_sleeping(0),
        "an unrepresentable change wakes the body"
    );
    assert_eq!((w.bodies[0].position, w.bodies[0].velocity), before);
}

/// `I⁻¹·τ·h` with a negative component just inside the range edge: with
/// `inv_inertia = 1`, `h = 1/256` and `τ.x = −3037000499.5·256`, the change
/// is `Δω.x = −3037000499.5`, whose square `≈ 9.2233720339e18` is below
/// `2⁶³ ≈ 9.2233720369e18`. Every product on the way is in range, so the
/// torque is applied and no fault is recorded.
fn edge_torque() -> Vec3Fix {
    // −3037000499.5 · 256 = −777472127872
    Vec3Fix::new(
        Fix128::from_int(-777_472_127_872),
        Fix128::ZERO,
        Fix128::ZERO,
    )
}

fn edge_body() -> RigidBody {
    let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = Vec3Fix::from_int(1, 1, 1);
    b
}

#[test]
fn a_torque_with_a_negative_component_near_the_range_edge_is_applied() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 1,
        ..free_config(SolverBackend::Xpbd)
    });
    w.add_body(edge_body());
    w.add_participant(Box::new(Twist::new(edge_torque())))
        .expect("register");
    // h = 1/256 with substeps 1
    let r = w.try_step(Fix128::from_ratio(1, 256));
    assert_ne!(
        r,
        Err(StepError::FaultRaised(WorldFault::ForceOutOfRange {
            body: 0
        })),
        "an in-range I⁻¹τh was reported out of range"
    );
    assert_ne!(
        w.fault(),
        Some(WorldFault::ForceOutOfRange { body: 0 }),
        "an in-range I⁻¹τh was reported out of range"
    );
}

/// The same edge change on a parked body whose angular sleep threshold is
/// `3037000499.75` (square `≈ 9.2233720354e18`, still below `2⁶³`):
/// `|Δω|² < threshold²`, so the body stays parked and the torque has no
/// effect.
#[test]
fn a_change_near_the_range_edge_below_the_threshold_leaves_a_parked_body_parked() {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 1,
        ..free_config(SolverBackend::Xpbd)
    });
    w.add_body(edge_body());
    for _ in 0..120 {
        w.step(Fix128::from_ratio(1, 256));
    }
    assert!(w.is_sleeping(0), "fixture: the body must be asleep");
    w.set_sleep_config(alice_physics::sleeping::SleepConfig {
        angular_threshold: Fix128::from_ratio(12_148_001_999, 4),
        ..Default::default()
    });
    let before = (w.bodies[0].rotation, w.bodies[0].angular_velocity);
    w.add_participant(Box::new(Twist::new(edge_torque())))
        .expect("register");
    w.try_step(Fix128::from_ratio(1, 256)).expect("step");
    assert!(
        w.is_sleeping(0),
        "a change below the threshold woke the body"
    );
    assert_eq!((w.bodies[0].rotation, w.bodies[0].angular_velocity), before);
}

// ── Rigid trajectory with a participant that stages nothing ─────────────

fn rigid_motion(w: &PhysicsWorld) -> Vec<(Vec3Fix, Vec3Fix, Vec3Fix, Vec3Fix, Fix128)> {
    w.bodies
        .iter()
        .map(|b| {
            (
                b.position,
                b.velocity,
                b.angular_velocity,
                Vec3Fix::new(b.rotation.x, b.rotation.y, b.rotation.z),
                b.rotation.w,
            )
        })
        .collect()
}

/// A stack of three spheres on a static one, spinning, under gravity
/// (contacts in every frame), and with `joints` a fourth sphere hanging from
/// the static body by a ball joint.
fn quiet_scene(backend: SolverBackend, substeps: usize, joints: bool) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps,
        iterations: 4,
        solver_backend: backend,
        ..Default::default()
    });
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    for i in 0..3 {
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(i % 2, 1 + 2 * i, 0), Fix128::ONE);
        b.angular_velocity = Vec3Fix::new(Fix128::from_ratio(1, 3), Fix128::ZERO, Fix128::ONE);
        w.add_body_with_radius(b, Fix128::ONE);
    }
    if joints {
        let hanging = w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(5, -1, 0), Fix128::ONE),
            Fix128::ONE,
        );
        w.add_joint(Joint::Ball(BallJoint::new(
            0,
            hanging,
            Vec3Fix::from_int(3, 0, 0),
            Vec3Fix::from_int(-2, 0, 0),
        )));
    }
    w
}

#[derive(Clone, Copy, Debug)]
enum Path {
    Step,
    #[cfg(feature = "parallel")]
    Parallel,
    #[cfg(feature = "gpu-solver-bridge")]
    Bridge,
}

/// One frame of `w` on `path`, checked when `checked` (a world with
/// participants) and plain otherwise.
fn frame(w: &mut PhysicsWorld, path: Path, checked: bool, dt: Fix128) {
    match path {
        Path::Step => {
            if checked {
                w.try_step(dt).expect("step");
            } else {
                w.step(dt);
            }
        }
        #[cfg(feature = "parallel")]
        Path::Parallel => {
            if checked {
                w.try_step_parallel(dt).expect("step");
            } else {
                w.step_parallel(dt);
            }
        }
        #[cfg(feature = "gpu-solver-bridge")]
        Path::Bridge => w.step_with_bridge(&mut bridge::Idle, dt),
    }
}

/// Registering a participant that stages nothing (no force, no torque, no
/// field) leaves every body bit for bit as in the same world without it,
/// for 1, 2, 3 and 8 substeps, on XPBD and TGS (with and without joints),
/// the parallel path and the bridge (without joints: the idle bridge does not
/// solve them). The oracle is the world without the
/// participant, compared with `assert_eq!` after every frame. On TGS this
/// needs the participants' substep loop to split `dt` as the solve without
/// participants does and to copy the bodies out and back in every substep.
#[test]
fn a_participant_that_stages_nothing_leaves_every_path_bit_identical() {
    let cases = [
        (SolverBackend::Xpbd, Path::Step),
        (SolverBackend::Tgs, Path::Step),
        #[cfg(feature = "parallel")]
        (SolverBackend::Xpbd, Path::Parallel),
        #[cfg(feature = "gpu-solver-bridge")]
        (SolverBackend::Xpbd, Path::Bridge),
    ];
    let dt = Fix128::from_ratio(1, 60);
    for (backend, path) in cases {
        for substeps in [1, 2, 3, 8] {
            // the idle bridge does not take joints (`send_joints` panics)
            #[cfg(feature = "gpu-solver-bridge")]
            let joint_cases: &[bool] = if matches!(path, Path::Bridge) {
                &[false]
            } else {
                &[false, true]
            };
            #[cfg(not(feature = "gpu-solver-bridge"))]
            let joint_cases: &[bool] = &[false, true];
            for &joints in joint_cases {
                let mut a = quiet_scene(backend, substeps, joints);
                let mut b = quiet_scene(backend, substeps, joints);
                b.add_participant(Box::new(Push::new(1, Vec3Fix::ZERO)))
                    .expect("register");
                let start = rigid_motion(&a);
                for f in 0..60 {
                    frame(&mut a, path, false, dt);
                    frame(&mut b, path, true, dt);
                    assert_eq!(
                        rigid_motion(&a),
                        rigid_motion(&b),
                        "{backend:?} {path:?} substeps {substeps} joints {joints} frame {f}"
                    );
                }
                assert_ne!(rigid_motion(&a), start, "fixture: the scene must move");
                assert_eq!(b.fault(), None);
            }
        }
    }
}

// ── Restore order ────────────────────────────────────────────────────────

/// A restore whose participants pass their check but whose fields are
/// refused (the blob has a field section the target does not declare)
/// leaves the participants' state as it was: the participants are read only
/// after every section has been checked.
#[test]
fn a_restore_refused_on_the_fields_leaves_the_participant_state_unchanged() {
    let build = |declare: bool| {
        let mut w = PhysicsWorld::new(PhysicsConfig::default());
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        if declare {
            w.declare_field(
                PortId::new(4),
                FieldLayout::PerBody { bodies: 1 },
                FieldMode::Sum,
            )
            .expect("declare");
        }
        w.add_participant(Box::new(Twist::new(Vec3Fix::ZERO)))
            .expect("register");
        w
    };
    let mut source = build(true);
    for _ in 0..3 {
        source.try_step(Fix128::from_ratio(1, 60)).expect("step");
    }
    let blob = source.snapshot_world();
    let mut target = build(false);
    let before = target.participant_state(0).expect("participant 0");
    assert_ne!(
        source.participant_state(0).expect("participant 0"),
        before,
        "fixture: the blob must hold a different participant state"
    );
    assert!(matches!(
        target.restore_world(&blob),
        Err(WorldSnapshotError::FieldState(_))
    ));
    assert_eq!(target.participant_state(0).expect("participant 0"), before);
}
