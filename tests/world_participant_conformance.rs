//! Conformance tests of the participant contract ([`alice_physics::world_participant`]).
//!
//! Two layers:
//!
//! * **participant layer**: the types of the contract on their
//!   own, and [`check_participant_contract`], which any participant type can be
//!   run through (fail leaves it unchanged, `write_state` → `check_state` →
//!   `read_state` restores it bit for bit, a short payload is rejected).
//! * **world layer**: what `PhysicsWorld` must do with registered
//!   participants, through the receivers in the `world` module below (one
//!   per world entry point).
//!
//! Oracles: call order and bit identity are compared against a second world
//! built independently (a world without participants, a world stepped
//! without a snapshot in between), never against values the world under test
//! computed for itself.

#![cfg(feature = "std")]

use std::sync::{Arc, Mutex};

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sleeping::SleepConfig;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::world_participant::{
    check_field_ports, deposit_bodies, execution_order, remap_conserving, run_substep,
    wakes_parked_body, AccumulateError, ExchangeError, FieldAccessError, FieldBoard, FieldError,
    FieldLayout, FieldMode, FieldPortError, FieldStage, ForceAccumulator, ObservationSink,
    Observed, OrderError, Participant, ParticipantFault, ParticipantKind, ParticipantMismatch,
    ParticipantPlan, Port, PortAccess, PortId, RegisterError, RemapError, StateError, StepError,
    StepRule, SubstepCtx, SubstepTime, Verdict, WorldFault,
};

// ============================================================================
// Test participant
// ============================================================================

const KIND_TETHER: ParticipantKind = ParticipantKind::new(0x5445_5448);
const KIND_OTHER: ParticipantKind = ParticipantKind::new(0x4f54_4852);

/// A damped point oscillator `x'' = -k x` (symplectic Euler) that pulls body
/// `target` towards its own position with force `c (x - y_body)` along y.
///
/// State: `x`, `v`, `calls` (24 + 8 bytes = 40 bytes in the payload, all
/// little endian). `fail_on_call = Some(n)` makes the `n`-th call (0-based)
/// stage its force and then return `Err` without committing its state, so a
/// world that does not drop staged forces of a failed call is caught.
struct Tether {
    kind: ParticipantKind,
    id: u32,
    target: usize,
    x: Fix128,
    v: Fix128,
    calls: u64,
    fail_on_call: Option<u64>,
    log: Option<CallLog>,
    ports: Vec<Port>,
}

/// `(participant id, substep index)` per call, shared across participants.
type CallLog = Arc<Mutex<Vec<(u32, usize)>>>;

const TETHER_STATE_LEN: usize = 16 + 16 + 8;

impl Tether {
    fn new(id: u32, target: usize) -> Self {
        Self {
            kind: KIND_TETHER,
            id,
            target,
            x: Fix128::from_int(3),
            v: Fix128::ZERO,
            calls: 0,
            fail_on_call: None,
            log: None,
            ports: Vec::new(),
        }
    }

    fn with_ports(mut self, ports: &[Port]) -> Self {
        self.ports = ports.to_vec();
        self
    }

    fn with_kind(mut self, kind: ParticipantKind) -> Self {
        self.kind = kind;
        self
    }

    fn failing_on(mut self, call: u64) -> Self {
        self.fail_on_call = Some(call);
        self
    }

    fn logging(mut self, log: &CallLog) -> Self {
        self.log = Some(Arc::clone(log));
        self
    }

    fn state_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_state(&mut out);
        out
    }
}

fn put_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn get_fix(b: &[u8]) -> Fix128 {
    let hi = i64::from_le_bytes(b[0..8].try_into().expect("8 bytes"));
    let lo = u64::from_le_bytes(b[8..16].try_into().expect("8 bytes"));
    Fix128 { hi, lo }
}

impl Participant for Tether {
    fn kind(&self) -> ParticipantKind {
        self.kind
    }

    fn ports(&self) -> &[Port] {
        &self.ports
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        if let Some(log) = &self.log {
            log.lock()
                .expect("log lock")
                .push((self.id, ctx.substep_index()));
        }
        let k = Fix128::from_int(4);
        let c = Fix128::from_int(2);
        let v = self.v - k * self.x * h;
        let x = self.x + v * h;
        let body_y = ctx
            .bodies()
            .get(self.target)
            .ok_or(ParticipantFault::InvalidState)?
            .position
            .y;
        ctx.add_force(
            self.target,
            Vec3Fix::new(Fix128::ZERO, c * (x - body_y), Fix128::ZERO),
        )
        .map_err(|_| ParticipantFault::InvalidState)?;
        if self.fail_on_call == Some(self.calls) {
            return Err(ParticipantFault::OutOfRange);
        }
        self.x = x;
        self.v = v;
        self.calls += 1;
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        out.push(0, self.x);
        out.push(1, self.v);
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        put_fix(out, self.x);
        put_fix(out, self.v);
        out.extend_from_slice(&self.calls.to_le_bytes());
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() == TETHER_STATE_LEN {
            Ok(())
        } else {
            Err(StateError::Length {
                expected: TETHER_STATE_LEN,
                found: bytes.len(),
            })
        }
    }

    fn read_state(&mut self, bytes: &[u8]) {
        self.x = get_fix(&bytes[0..16]);
        self.v = get_fix(&bytes[16..32]);
        self.calls = u64::from_le_bytes(bytes[32..40].try_into().expect("8 bytes"));
    }
}

// ============================================================================
// Participant-layer harness (usable for every participant type)
// ============================================================================

fn bodies_for_harness() -> Vec<RigidBody> {
    vec![
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 1, 0), Fix128::ONE),
        RigidBody::new_dynamic(Vec3Fix::from_int(2, 5, 0), Fix128::from_int(3)),
    ]
}

/// Run `p` through the participant-side rules of the contract:
/// 1. `steps` successful substeps, then a state round trip into `fresh`
///    (`write_state` → `check_state` → `read_state`) gives identical bytes;
/// 2. a payload one byte short is rejected by `check_state` and `check_state`
///    does not change the participant;
/// 3. if `p` fails on some call, the failing call leaves `p` byte-identical.
fn check_participant_contract<P: Participant>(mut p: P, mut fresh: P, steps: usize) {
    let bodies = bodies_for_harness();
    let h = Fix128::from_ratio(1, 240);
    for i in 0..steps {
        let before = state_of(&p);
        let mut forces = ForceAccumulator::new(bodies.len());
        let mut ctx =
            SubstepCtx::new(&bodies, &mut forces, i % 4, 4, h).expect("one slot per body");
        if p.substep(&mut ctx, h).is_err() {
            assert_eq!(
                state_of(&p),
                before,
                "a participant that returned Err changed its own state (call {i})"
            );
            break;
        }
    }
    let blob = state_of(&p);
    assert!(
        !blob.is_empty(),
        "write_state wrote nothing: the round trip compares nothing"
    );
    fresh
        .check_state(&blob)
        .expect("check_state rejected a payload write_state produced");
    fresh.read_state(&blob);
    assert_eq!(
        state_of(&fresh),
        blob,
        "read_state did not restore the state bit for bit"
    );

    let before = state_of(&fresh);
    assert!(
        fresh.check_state(&blob[..blob.len() - 1]).is_err(),
        "check_state accepted a payload one byte short"
    );
    assert_eq!(
        state_of(&fresh),
        before,
        "check_state changed the participant"
    );
}

fn state_of<P: Participant>(p: &P) -> Vec<u8> {
    let mut out = Vec::new();
    p.write_state(&mut out);
    out
}

#[test]
fn tether_satisfies_the_participant_contract() {
    check_participant_contract(Tether::new(0, 0), Tether::new(0, 0), 50);
}

#[test]
fn a_failing_call_leaves_the_tether_unchanged() {
    check_participant_contract(Tether::new(0, 0).failing_on(7), Tether::new(0, 0), 50);
}

/// The harness has teeth: a participant that commits state before failing is
/// caught by rule 3.
#[test]
#[should_panic(expected = "returned Err changed its own state")]
fn harness_catches_a_participant_that_changes_state_on_err() {
    struct Leaky(Tether);
    impl Participant for Leaky {
        fn kind(&self) -> ParticipantKind {
            self.0.kind()
        }
        fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
            self.0.calls += 1;
            if self.0.calls == 3 {
                return Err(ParticipantFault::OutOfRange);
            }
            self.0.substep(ctx, h)
        }
        fn observe(&self, out: &mut ObservationSink) {
            self.0.observe(out);
        }
        fn write_state(&self, out: &mut Vec<u8>) {
            self.0.write_state(out);
        }
        fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
            self.0.check_state(bytes)
        }
        fn read_state(&mut self, bytes: &[u8]) {
            self.0.read_state(bytes);
        }
    }
    check_participant_contract(Leaky(Tether::new(0, 0)), Leaky(Tether::new(0, 0)), 10);
}

// ============================================================================
// Contract types on their own (run today)
// ============================================================================

/// (f) A participant cannot write a body: `SubstepCtx` hands out
/// `&[RigidBody]` only (a write does not compile, see the `compile_fail`
/// example in the module documentation), and staging a force leaves the
/// bodies it was given untouched.
#[test]
fn staging_a_force_does_not_touch_the_bodies() {
    let bodies = bodies_for_harness();
    let copy = bodies.clone();
    let mut forces = ForceAccumulator::new(bodies.len());
    let h = Fix128::from_ratio(1, 60);
    {
        let mut ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, h).expect("slots");
        let mut p = Tether::new(0, 1);
        p.substep(&mut ctx, h).expect("tether substep");
        let read_only: &[RigidBody] = ctx.bodies();
        assert_eq!(read_only.len(), 2);
    }
    for (a, b) in bodies.iter().zip(&copy) {
        assert_eq!((a.position, a.velocity), (b.position, b.velocity));
    }
    // The force went to the accumulator, slot 1 only.
    assert_eq!(forces.force(0), Some(Vec3Fix::ZERO));
    assert_ne!(forces.force(1), Some(Vec3Fix::ZERO));
}

#[test]
fn ctx_needs_one_slot_per_body() {
    let bodies = bodies_for_harness();
    let mut forces = ForceAccumulator::new(1);
    let err = SubstepCtx::new(&bodies, &mut forces, 0, 1, Fix128::ONE).unwrap_err();
    assert_eq!(
        err,
        AccumulateError::LengthMismatch {
            slots: 1,
            bodies: 2
        }
    );
}

#[test]
fn accumulator_rejects_a_missing_body_and_adds_nothing() {
    let mut acc = ForceAccumulator::new(0);
    assert_eq!(
        acc.add_force(0, Vec3Fix::from_int(1, 0, 0)),
        Err(AccumulateError::BodyOutOfRange { index: 0, len: 0 })
    );
    let mut acc = ForceAccumulator::new(2);
    assert_eq!(
        acc.add_torque(2, Vec3Fix::from_int(1, 0, 0)),
        Err(AccumulateError::BodyOutOfRange { index: 2, len: 2 })
    );
    assert_eq!(
        acc,
        ForceAccumulator::new(2),
        "a rejected add changed a slot"
    );
}

/// Wrapping addition is associative and commutative, so the merged total does
/// not depend on the order of the staged forces, even across the wrap.
#[test]
fn accumulator_total_does_not_depend_on_order_even_across_the_wrap() {
    let big = Fix128 {
        hi: i64::MAX,
        lo: u64::MAX,
    };
    let forces = [
        Vec3Fix::new(big, Fix128::ONE, Fix128::from_int(-7)),
        Vec3Fix::new(big, Fix128::from_ratio(1, 3), Fix128::from_int(5)),
        Vec3Fix::new(Fix128::from_int(-9), Fix128::from_ratio(-2, 7), big),
    ];
    let mut forward = ForceAccumulator::new(1);
    for f in forces {
        forward.add_force(0, f).expect("slot 0");
    }
    let mut backward = ForceAccumulator::new(1);
    for f in forces.iter().rev() {
        backward.add_force(0, *f).expect("slot 0");
    }
    assert_eq!(forward, backward);

    let mut merged = ForceAccumulator::new(1);
    let mut one = ForceAccumulator::new(1);
    one.add_force(0, forces[0]).expect("slot 0");
    let mut two = ForceAccumulator::new(1);
    two.add_force(0, forces[1]).expect("slot 0");
    two.add_force(0, forces[2]).expect("slot 0");
    merged.merge(&two).expect("same length");
    merged.merge(&one).expect("same length");
    assert_eq!(merged, forward);
    assert!(merged.merge(&ForceAccumulator::new(2)).is_err());
    assert_eq!(merged, forward, "a rejected merge changed a slot");
}

#[test]
fn step_rule_fixed_needs_an_integer_ratio() {
    let h = Fix128::from_ratio(1, 60);
    assert_eq!(StepRule::FollowSubstep.steps_per_substep(h), Ok(1));
    assert_eq!(StepRule::Subcycle.steps_per_substep(h), Ok(1));
    assert_eq!(StepRule::Fixed(h).steps_per_substep(h), Ok(1));
    assert_eq!(
        StepRule::Fixed(h / Fix128::from_int(3)).steps_per_substep(h),
        Ok(3)
    );
    let quarter = Fix128::from_ratio(1, 4);
    assert_eq!(
        StepRule::Fixed(quarter).steps_per_substep(Fix128::ONE),
        Ok(4)
    );
    // 1 / (2/3) = 1.5
    let two_thirds = Fix128::from_ratio(2, 3);
    assert_eq!(
        StepRule::Fixed(two_thirds).steps_per_substep(Fix128::ONE),
        Err(RegisterError::StepRuleMismatch {
            substep: Fix128::ONE,
            step: two_thirds
        })
    );
    // h = 7 + 2^-64, Δt = 7: h / Δt = 1 + 2^-64 / 7 truncates to exactly 1,
    // but 1 · 7 ≠ h, so the quotient alone must not be trusted
    let h_off = Fix128 { hi: 7, lo: 1 };
    assert_eq!(
        StepRule::Fixed(Fix128::from_int(7)).steps_per_substep(h_off),
        Err(RegisterError::StepRuleMismatch {
            substep: h_off,
            step: Fix128::from_int(7)
        })
    );
    // own step longer than the substep: n = 0
    assert!(StepRule::Fixed(Fix128::from_int(2))
        .steps_per_substep(Fix128::ONE)
        .is_err());
}

/// Degenerate inputs: zero / negative widths give an explicit error, no
/// division by zero.
#[test]
fn step_rule_rejects_non_positive_widths() {
    let h = Fix128::from_ratio(1, 60);
    assert_eq!(
        StepRule::Fixed(Fix128::ZERO).steps_per_substep(h),
        Err(RegisterError::NonPositiveStep)
    );
    assert_eq!(
        StepRule::Fixed(Fix128::from_int(-1)).steps_per_substep(h),
        Err(RegisterError::NonPositiveStep)
    );
    assert_eq!(
        StepRule::FollowSubstep.steps_per_substep(Fix128::ZERO),
        Err(RegisterError::NonPositiveStep)
    );
    assert_eq!(
        StepRule::Fixed(h).steps_per_substep(-h),
        Err(RegisterError::NonPositiveStep)
    );
}

#[test]
fn mismatch_classification_covers_count_kind_and_order() {
    let a = KIND_TETHER;
    let b = KIND_OTHER;
    let c = ParticipantKind::new(3);
    assert_eq!(ParticipantMismatch::classify(&[a, b], &[a, b]), None);
    assert_eq!(ParticipantMismatch::classify(&[], &[]), None);
    assert_eq!(
        ParticipantMismatch::classify(&[a], &[a, b]),
        Some(ParticipantMismatch::Count {
            snapshot: 1,
            world: 2
        })
    );
    assert_eq!(
        ParticipantMismatch::classify(&[a, b], &[a, c]),
        Some(ParticipantMismatch::Kind {
            index: 1,
            snapshot: b,
            world: c
        })
    );
    assert_eq!(
        ParticipantMismatch::classify(&[a, b], &[b, a]),
        Some(ParticipantMismatch::Order {
            index: 0,
            snapshot: a,
            world: b
        })
    );
    // same multiset size, one duplicated kind: kind, not order
    assert_eq!(
        ParticipantMismatch::classify(&[a, b], &[a, a]),
        Some(ParticipantMismatch::Kind {
            index: 1,
            snapshot: b,
            world: a
        })
    );
}

#[test]
fn observed_gives_three_verdicts() {
    let exact = Observed::Exact(Fix128::from_int(2));
    assert_eq!(exact.verdict(|v| *v > Fix128::ONE), Verdict::Holds);
    assert_eq!(exact.verdict(|v| *v < Fix128::ONE), Verdict::Violated);
    let undecided: Observed<Fix128> = Observed::Undecided;
    assert_eq!(undecided.verdict(|_| true), Verdict::Undecided);
}

#[test]
fn fault_tags_round_trip_and_unknown_tags_are_refused() {
    for f in [ParticipantFault::OutOfRange, ParticipantFault::InvalidState] {
        assert_eq!(ParticipantFault::from_tag(f.tag()), Some(f));
    }
    assert_eq!(ParticipantFault::from_tag(0), None);
    assert_eq!(ParticipantFault::from_tag(255), None);
}

// ============================================================================
// Execution order from declared ports (pure, no world needed)
// ============================================================================

const PORT_A: PortId = PortId::new(1);
const PORT_B: PortId = PortId::new(2);
const PORT_C: PortId = PortId::new(3);

/// A participant that keeps the default `ports`.
struct Silent;

impl Participant for Silent {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0)
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _: &[u8]) {}
}

#[test]
fn ports_are_typed_and_default_to_none() {
    assert!(Silent.ports().is_empty());
    let r = Port::reads(PORT_A);
    let w = Port::writes(PORT_A);
    assert_eq!((r.id(), r.access()), (PORT_A, PortAccess::Read));
    assert_eq!((w.id(), w.access()), (PORT_A, PortAccess::Write));
    assert_ne!(r, w);
    assert_eq!(PortId::new(7).get(), 7);
}

/// Oracle: with nothing declared the order is the registration order (the
/// behaviour before ports existed).
#[test]
fn execution_order_without_ports_is_registration_order() {
    assert_eq!(execution_order(&[]), Ok(vec![]));
    assert_eq!(execution_order(&[&[], &[], &[], &[]]), Ok(vec![0, 1, 2, 3]));
    // a participant reading and writing its own port is not ordered against itself
    let own: &[Port] = &[Port::reads(PORT_A), Port::writes(PORT_A)];
    assert_eq!(execution_order(&[own, &[]]), Ok(vec![0, 1]));
    // reads without a writer, writes without a reader: no constraint
    assert_eq!(
        execution_order(&[&[Port::reads(PORT_A)], &[Port::writes(PORT_B)]]),
        Ok(vec![0, 1])
    );
}

/// Oracle: a writer runs before its readers, whatever the registration order.
#[test]
fn execution_order_runs_writers_before_readers() {
    assert_eq!(
        execution_order(&[&[Port::reads(PORT_A)], &[Port::writes(PORT_A)]]),
        Ok(vec![1, 0])
    );
    // chain 2 → 1 → 0 registered backwards
    assert_eq!(
        execution_order(&[
            &[Port::reads(PORT_B)],
            &[Port::reads(PORT_A), Port::writes(PORT_B)],
            &[Port::writes(PORT_A)],
        ]),
        Ok(vec![2, 1, 0])
    );
    // a reader of two ports waits for both writers
    assert_eq!(
        execution_order(&[
            &[Port::reads(PORT_A), Port::reads(PORT_B)],
            &[Port::writes(PORT_A)],
            &[Port::writes(PORT_B)],
        ]),
        Ok(vec![1, 2, 0])
    );
}

/// Oracle: among participants free to run, registration order decides; the
/// same declarations always give the same order.
#[test]
fn execution_order_keeps_registration_order_among_equals() {
    // 1 and 3 are free; 2 must precede 0. Free ones are taken lowest first:
    // 1, 2, then 0 (now free, lower than 3), 3.
    let decl: [&[Port]; 4] = [&[Port::reads(PORT_A)], &[], &[Port::writes(PORT_A)], &[]];
    assert_eq!(execution_order(&decl), Ok(vec![1, 2, 0, 3]));
    assert_eq!(execution_order(&decl), execution_order(&decl));
    // two writers of one port keep their registration order, both before the reader
    assert_eq!(
        execution_order(&[
            &[Port::reads(PORT_A)],
            &[Port::writes(PORT_A)],
            &[Port::writes(PORT_A)],
        ]),
        Ok(vec![1, 2, 0])
    );
    // two independent chains interleave by registration order
    assert_eq!(
        execution_order(&[
            &[Port::writes(PORT_A)],
            &[Port::writes(PORT_B)],
            &[Port::reads(PORT_A)],
            &[Port::reads(PORT_B)],
        ]),
        Ok(vec![0, 1, 2, 3])
    );
}

/// Oracle: a loop is refused, naming exactly the participants on it; nothing
/// is ordered silently.
#[test]
fn execution_order_refuses_a_cycle() {
    // 0 ⇄ 1 through A and B; 2 only reads from the loop; 3 is free
    let err = execution_order(&[
        &[Port::reads(PORT_A), Port::writes(PORT_B)],
        &[Port::reads(PORT_B), Port::writes(PORT_A)],
        &[Port::reads(PORT_A)],
        &[],
    ]);
    assert_eq!(
        err,
        Err(OrderError::Cycle {
            members: vec![0, 1]
        })
    );
    // three-way loop registered after a free participant and a writer feeding it
    let err = execution_order(&[
        &[],
        &[Port::writes(PORT_C)],
        &[
            Port::reads(PORT_C),
            Port::reads(PORT_A),
            Port::writes(PORT_B),
        ],
        &[Port::reads(PORT_B), Port::writes(PORT_C)],
        &[Port::reads(PORT_C), Port::writes(PORT_A)],
    ]);
    assert_eq!(
        err,
        Err(OrderError::Cycle {
            members: vec![2, 3, 4]
        })
    );
}

// ============================================================================
// Shared fields (pure: board, staging, commit, registration checks, remap)
// ============================================================================

const TEMP: PortId = PortId::new(10);
const HEAT: PortId = PortId::new(11);
const LINK: PortId = PortId::new(12);

const KIND_WRITER: ParticipantKind = ParticipantKind::new(0x5752_4954);
const KIND_READER: ParticipantKind = ParticipantKind::new(0x5245_4144);

/// Stages `value(substep index)` into every sample of `field`, optionally
/// pushes body 0 along x with 1 N, and returns `Err` after staging when
/// `fail` is set (so a world that keeps staged writes of a failed call is
/// caught).
struct Writer {
    ports: Vec<Port>,
    field: PortId,
    value: fn(usize) -> Fix128,
    push: bool,
    fail: bool,
}

impl Writer {
    fn new(field: PortId, value: fn(usize) -> Fix128) -> Self {
        Self {
            ports: vec![Port::writes(field)],
            field,
            value,
            push: false,
            fail: false,
        }
    }

    fn also(mut self, port: Port) -> Self {
        self.ports.push(port);
        self
    }
}

impl Participant for Writer {
    fn kind(&self) -> ParticipantKind {
        KIND_WRITER
    }
    fn ports(&self) -> &[Port] {
        &self.ports
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        let v = (self.value)(ctx.substep_index());
        ctx.stage_field(self.field)
            .map_err(|_| ParticipantFault::InvalidState)?
            .fill(v);
        if self.push {
            ctx.add_force(0, Vec3Fix::from_int(1, 0, 0))
                .map_err(|_| ParticipantFault::InvalidState)?;
        }
        if self.fail {
            return Err(ParticipantFault::OutOfRange);
        }
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _: &[u8]) {}
}

/// Logs the committed value of `field` it sees in every substep.
struct Reader {
    ports: Vec<Port>,
    field: PortId,
    log: Arc<Mutex<Vec<Vec<Fix128>>>>,
}

impl Participant for Reader {
    fn kind(&self) -> ParticipantKind {
        KIND_READER
    }
    fn ports(&self) -> &[Port] {
        &self.ports
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        let seen = ctx
            .field(self.field)
            .map_err(|_| ParticipantFault::InvalidState)?
            .to_vec();
        self.log.lock().expect("log").push(seen);
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _: &[u8]) {}
}

fn grid(origin: (i64, i64, i64), cell: Fix128, dims: [usize; 3]) -> FieldLayout {
    FieldLayout::Grid {
        origin: Vec3Fix::from_int(origin.0, origin.1, origin.2),
        cell,
        dims,
    }
}

fn h_sub() -> Fix128 {
    Fix128::from_ratio(1, 240)
}

/// Runs `substeps` substeps of `ps` over `bodies` and `board`; returns the
/// faults of each substep and the summed forces.
fn drive(
    ps: &mut [Box<dyn Participant>],
    bodies: &[RigidBody],
    board: &mut FieldBoard,
    substeps: usize,
) -> (Vec<Vec<WorldFault>>, ForceAccumulator, Vec<bool>) {
    let plan = ParticipantPlan::new(ps, board).expect("plan");
    let mut frozen = vec![false; ps.len()];
    let mut forces = ForceAccumulator::new(bodies.len());
    let mut faults = Vec::new();
    for i in 0..substeps {
        let time = SubstepTime {
            index: i,
            count: substeps,
            h: h_sub(),
        };
        faults.push(
            run_substep(ps, &plan, &mut frozen, bodies, board, &mut forces, time)
                .expect("inputs fit"),
        );
    }
    (faults, forces, frozen)
}

fn one_value(i: usize) -> Fix128 {
    Fix128::from_int(i as i64 + 1)
}

/// Oracle (closed form): a reader sees the value committed before the
/// substep began. The writer stores `k + 1` in substep `k`; the reader, run
/// after it, must log `0, 1, 2` (the start values), never `1, 2, 3`.
#[test]
fn field_reads_see_the_value_committed_before_the_substep() {
    let mut board = FieldBoard::new();
    board
        .declare(TEMP, FieldLayout::PerBody { bodies: 1 }, FieldMode::Replace)
        .expect("declare");
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Writer::new(TEMP, one_value)),
        Box::new(Reader {
            ports: vec![Port::reads_committed(TEMP)],
            field: TEMP,
            log: Arc::clone(&log),
        }),
    ];
    let bodies = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    let (faults, _, _) = drive(&mut ps, &bodies, &mut board, 3);
    assert!(faults.iter().all(Vec::is_empty));
    let seen: Vec<Fix128> = log.lock().expect("log").iter().map(|v| v[0]).collect();
    assert_eq!(
        seen,
        vec![Fix128::ZERO, Fix128::ONE, Fix128::from_int(2)],
        "a read saw a value staged in the same substep"
    );
    assert_eq!(board.value(TEMP), Some(&[Fix128::from_int(3)][..]));
}

/// Two participants that each read what the other writes (the thermal ⇄
/// structure shape) are accepted through committed reads, while the same loop
/// through plain reads is refused.
#[test]
fn a_loop_through_committed_reads_is_accepted() {
    let mut board = FieldBoard::new();
    for id in [TEMP, HEAT] {
        board
            .declare(id, FieldLayout::PerBody { bodies: 1 }, FieldMode::Replace)
            .expect("declare");
    }
    let a: &[Port] = &[Port::reads_committed(HEAT), Port::writes(TEMP)];
    let b: &[Port] = &[Port::reads_committed(TEMP), Port::writes(HEAT)];
    assert_eq!(check_field_ports(&board, &[a, b]), Ok(()));
    assert_eq!(execution_order(&[a, b]), Ok(vec![0, 1]));
    let a_plain: &[Port] = &[Port::reads(LINK), Port::writes(TEMP)];
    let b_plain: &[Port] = &[Port::reads(PORT_A), Port::writes(LINK)];
    let c_plain: &[Port] = &[Port::reads(TEMP)];
    assert!(execution_order(&[a_plain, b_plain]).is_ok());
    assert_eq!(
        check_field_ports(&board, &[a_plain, b_plain, c_plain]),
        Err(FieldPortError::ReadOfField {
            participant: 2,
            field: TEMP
        })
    );
}

const BIG: Fix128 = Fix128 { hi: 1 << 62, lo: 0 };

fn plus_big(_: usize) -> Fix128 {
    BIG
}

fn minus_big(_: usize) -> Fix128 {
    -BIG
}

fn seven(_: usize) -> Fix128 {
    Fix128::from_int(7)
}

/// Oracle: staged writes are committed in execution order. Participant 0
/// reads a port participant 2 writes, so the run order is `1, 2, 0`, not the
/// registration order. The partial sums are checked in that order
/// (B = 2⁶², the range ends at 2⁶³):
///
/// * increments `+B, −B, +B` by registration index: run order gives
///   `−B, 0, +B`, all in range, value `B`; the reverse of the run order
///   (`0, 2, 1`) would overflow at participant 2.
/// * increments `−B, +B, +B`: run order gives `+B, +2B` = out of range at
///   participant 2; no field changes (the `Replace` field participant 2 wrote
///   in the same substep keeps its value). The reverse run order and the
///   registration order would both stay in range.
#[test]
fn staged_field_writes_commit_in_execution_order() {
    let bodies = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    let make_board = || {
        let mut board = FieldBoard::new();
        board
            .declare(HEAT, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum)
            .expect("declare");
        board
            .declare(TEMP, FieldLayout::PerBody { bodies: 1 }, FieldMode::Replace)
            .expect("declare");
        board
    };
    let participants = |inc: [fn(usize) -> Fix128; 3]| -> Vec<Box<dyn Participant>> {
        vec![
            Box::new(Writer::new(HEAT, inc[0]).also(Port::reads(LINK))),
            Box::new(Writer::new(HEAT, inc[1])),
            Box::new(Writer {
                ports: vec![Port::writes(HEAT), Port::writes(LINK)],
                ..Writer::new(HEAT, inc[2])
            }),
        ]
    };
    let mut ps = participants([plus_big, minus_big, plus_big]);
    let mut board = make_board();
    assert_eq!(
        ParticipantPlan::new(&ps, &board).expect("plan").order(),
        &[1, 2, 0]
    );
    let (faults, _, _) = drive(&mut ps, &bodies, &mut board, 1);
    assert_eq!(faults, vec![vec![]], "an in-range commit was refused");
    assert_eq!(board.value(HEAT), Some(&[BIG][..]));

    // out of range at participant 2 in run order; nothing commits
    struct Both(Writer);
    impl Participant for Both {
        fn kind(&self) -> ParticipantKind {
            self.0.kind()
        }
        fn ports(&self) -> &[Port] {
            &self.0.ports
        }
        fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
            ctx.stage_field(TEMP)
                .map_err(|_| ParticipantFault::InvalidState)?
                .fill(Fix128::from_int(5));
            self.0.substep(ctx, h)
        }
        fn observe(&self, _: &mut ObservationSink) {}
        fn write_state(&self, _: &mut Vec<u8>) {}
        fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
            Ok(())
        }
        fn read_state(&mut self, _: &[u8]) {}
    }
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Writer::new(HEAT, minus_big).also(Port::reads(LINK))),
        Box::new(Writer::new(HEAT, plus_big)),
        Box::new(Both(Writer {
            ports: vec![Port::writes(HEAT), Port::writes(LINK), Port::writes(TEMP)],
            ..Writer::new(HEAT, plus_big)
        })),
    ];
    let mut board = make_board();
    board.set(HEAT, &[Fix128::from_int(-3)]).expect("set");
    board.set(TEMP, &[Fix128::from_int(2)]).expect("set");
    let before = board.clone();
    let (faults, _, frozen) = drive(&mut ps, &bodies, &mut board, 1);
    assert_eq!(
        faults,
        vec![vec![WorldFault::FieldOutOfRange {
            field: HEAT,
            index: 2
        }]]
    );
    assert_eq!(board, before, "a refused commit changed a field");
    assert_eq!(frozen, vec![false; 3], "a commit fault froze a participant");
}

/// A participant that returns `Err` after staging: its field writes and its
/// forces are dropped, it is frozen and reported, and the others commit. The
/// oracle is the board and forces of the same scene without the failing
/// participant's contribution, written out in closed form.
#[test]
fn a_failed_participant_s_staged_writes_are_dropped() {
    let bodies = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    let mut board = FieldBoard::new();
    board
        .declare(HEAT, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum)
        .expect("declare");
    board
        .declare(TEMP, FieldLayout::PerBody { bodies: 1 }, FieldMode::Replace)
        .expect("declare");
    board.set(TEMP, &[Fix128::from_int(20)]).expect("set");
    let failing = Writer {
        ports: vec![Port::writes(HEAT), Port::writes(TEMP)],
        push: true,
        fail: true,
        ..Writer::new(TEMP, seven)
    };
    let ok = Writer {
        push: true,
        ..Writer::new(HEAT, seven)
    };
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(failing), Box::new(ok)];
    let (faults, forces, frozen) = drive(&mut ps, &bodies, &mut board, 2);
    assert_eq!(
        faults,
        vec![
            vec![WorldFault::Participant {
                index: 0,
                kind: KIND_WRITER,
                fault: ParticipantFault::OutOfRange
            }],
            vec![]
        ]
    );
    assert_eq!(frozen, vec![true, false]);
    assert_eq!(
        board.value(TEMP),
        Some(&[Fix128::from_int(20)][..]),
        "a failed participant's staged write was committed"
    );
    assert_eq!(board.value(HEAT), Some(&[Fix128::from_int(7)][..]));
    // two substeps of the surviving 1 N push only
    assert_eq!(forces.force(0), Some(Vec3Fix::from_int(2, 0, 0)));
}

/// Sum fields start every substep at zero: with nobody writing, the value
/// returns to zero (a load, not a running total); a Replace field keeps its
/// value.
#[test]
fn a_sum_field_without_writers_commits_zero_and_replace_keeps() {
    let bodies = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    let mut board = FieldBoard::new();
    board
        .declare(HEAT, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum)
        .expect("declare");
    board
        .declare(TEMP, FieldLayout::PerBody { bodies: 1 }, FieldMode::Replace)
        .expect("declare");
    board.set(HEAT, &[Fix128::from_int(9)]).expect("set");
    board.set(TEMP, &[Fix128::from_int(4)]).expect("set");
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(Silent)];
    let (faults, _, _) = drive(&mut ps, &bodies, &mut board, 1);
    assert_eq!(faults, vec![vec![]]);
    assert_eq!(board.value(HEAT), Some(&[Fix128::ZERO][..]));
    assert_eq!(board.value(TEMP), Some(&[Fix128::from_int(4)][..]));
}

#[test]
fn field_access_must_be_declared() {
    let bodies = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    let mut board = FieldBoard::new();
    board
        .declare(TEMP, FieldLayout::PerBody { bodies: 1 }, FieldMode::Replace)
        .expect("declare");
    let mut forces = ForceAccumulator::new(1);
    let ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, Fix128::ONE).expect("slots");
    assert_eq!(ctx.field(TEMP), Err(FieldAccessError::Unknown(TEMP)));
    let mut stage = FieldStage::new();
    let ports = [Port::writes(TEMP)];
    let mut ctx = ctx.with_fields(&board, &mut stage, &ports);
    assert_eq!(
        ctx.field(TEMP),
        Err(FieldAccessError::NotDeclared {
            field: TEMP,
            access: PortAccess::ReadCommitted
        })
    );
    assert_eq!(ctx.field(HEAT), Err(FieldAccessError::Unknown(HEAT)));
    assert_eq!(
        ctx.stage_field(HEAT).map(|b| b.len()),
        Err(FieldAccessError::Unknown(HEAT))
    );
    ctx.stage_field(TEMP).expect("declared write")[0] = Fix128::ONE;
    assert_eq!(stage.staged(TEMP), Some(&[Fix128::ONE][..]));
    assert_eq!(
        board.value(TEMP),
        Some(&[Fix128::ZERO][..]),
        "staging wrote the board"
    );

    let mut forces = ForceAccumulator::new(1);
    let mut stage = FieldStage::new();
    let read_only = [Port::reads_committed(TEMP)];
    let mut ctx = SubstepCtx::new(&bodies, &mut forces, 0, 1, Fix128::ONE)
        .expect("slots")
        .with_fields(&board, &mut stage, &read_only);
    assert_eq!(ctx.field(TEMP), Ok(&[Fix128::ZERO][..]));
    assert_eq!(
        ctx.stage_field(TEMP).map(|b| b.len()),
        Err(FieldAccessError::NotDeclared {
            field: TEMP,
            access: PortAccess::Write
        })
    );
    assert!(stage.is_empty());
}

#[test]
fn registration_checks_fields_against_ports() {
    let mut board = FieldBoard::new();
    board
        .declare(TEMP, FieldLayout::PerBody { bodies: 2 }, FieldMode::Replace)
        .expect("declare");
    board
        .declare(HEAT, FieldLayout::PerBody { bodies: 2 }, FieldMode::Sum)
        .expect("declare");
    let w_t: &[Port] = &[Port::writes(TEMP)];
    let w_t_twice: &[Port] = &[Port::writes(TEMP), Port::writes(TEMP)];
    let w_h: &[Port] = &[Port::writes(HEAT)];
    assert_eq!(
        check_field_ports(&board, &[w_t_twice, w_h, w_h, w_h]),
        Ok(())
    );
    assert_eq!(
        check_field_ports(&board, &[w_h, w_t, &[], w_t]),
        Err(FieldPortError::SecondWriter {
            field: TEMP,
            first: 1,
            second: 3
        })
    );
    assert_eq!(
        check_field_ports(&board, &[&[Port::reads_committed(LINK)]]),
        Err(FieldPortError::UnknownField {
            participant: 0,
            field: LINK
        })
    );
    // a participant-held port (not a field) keeps its plain read / write
    assert_eq!(
        check_field_ports(&board, &[&[Port::writes(LINK)], &[Port::reads(LINK)]]),
        Ok(())
    );
    // through the plan, as the world registers
    let ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Writer::new(TEMP, seven)),
        Box::new(Writer::new(TEMP, seven)),
    ];
    assert_eq!(
        ParticipantPlan::new(&ps, &board),
        Err(RegisterError::Field(FieldPortError::SecondWriter {
            field: TEMP,
            first: 0,
            second: 1
        }))
    );
    let looped: Vec<Box<dyn Participant>> = vec![
        Box::new(Tether::new(0, 0).with_ports(&[Port::reads(PORT_A), Port::writes(PORT_B)])),
        Box::new(Tether::new(1, 0).with_ports(&[Port::reads(PORT_B), Port::writes(PORT_A)])),
    ];
    assert_eq!(
        ParticipantPlan::new(&looped, &board),
        Err(RegisterError::Order(OrderError::Cycle {
            members: vec![0, 1]
        }))
    );
}

#[test]
fn declaring_and_setting_fields_check_their_inputs() {
    let mut board = FieldBoard::new();
    let quarter = Fix128::from_ratio(1, 4);
    assert_eq!(
        board.declare(
            TEMP,
            grid((0, 0, 0), Fix128::ZERO, [2, 2, 2]),
            FieldMode::Sum
        ),
        Err(FieldError::InvalidLayout)
    );
    assert_eq!(
        board.declare(TEMP, grid((0, 0, 0), quarter, [2, 0, 2]), FieldMode::Sum),
        Err(FieldError::InvalidLayout)
    );
    assert_eq!(
        board.declare(
            TEMP,
            grid((0, 0, 0), quarter, [usize::MAX, 2, 1]),
            FieldMode::Sum
        ),
        Err(FieldError::InvalidLayout)
    );
    assert_eq!(board.ids(), vec![]);
    board
        .declare(HEAT, grid((0, 0, 0), quarter, [2, 3, 1]), FieldMode::Sum)
        .expect("declare");
    board
        .declare(TEMP, FieldLayout::PerBody { bodies: 0 }, FieldMode::Replace)
        .expect("declare");
    assert_eq!(
        board.declare(TEMP, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum),
        Err(FieldError::Duplicate(TEMP))
    );
    assert_eq!(board.ids(), vec![TEMP, HEAT]);
    assert_eq!(board.mode(HEAT), Some(FieldMode::Sum));
    assert_eq!(board.layout(TEMP), Some(FieldLayout::PerBody { bodies: 0 }));
    assert_eq!(board.value(HEAT).map(<[Fix128]>::len), Some(6));
    let before = board.clone();
    assert_eq!(
        board.set(HEAT, &[Fix128::ONE; 5]),
        Err(FieldError::Length {
            expected: 6,
            found: 5
        })
    );
    assert_eq!(board.set(LINK, &[]), Err(FieldError::Unknown(LINK)));
    assert_eq!(board, before);
}

/// Inputs that do not fit are refused before any participant runs.
#[test]
fn run_substep_refuses_inputs_that_do_not_fit() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let bodies = vec![RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE)];
    let mut board = FieldBoard::new();
    board
        .declare(TEMP, FieldLayout::PerBody { bodies: 2 }, FieldMode::Replace)
        .expect("declare");
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(Tether::new(0, 0).logging(&log))];
    let plan = ParticipantPlan::new(&ps, &board).expect("plan");
    let empty_plan = ParticipantPlan::new(&[], &board).expect("plan");
    let time = SubstepTime {
        index: 0,
        count: 1,
        h: h_sub(),
    };
    let mut forces = ForceAccumulator::new(1);
    let mut frozen = vec![false];
    assert_eq!(
        run_substep(
            &mut ps,
            &empty_plan,
            &mut frozen,
            &bodies,
            &mut board,
            &mut forces,
            time
        ),
        Err(ExchangeError::Plan {
            participants: 1,
            planned: 0
        })
    );
    assert_eq!(
        run_substep(
            &mut ps,
            &plan,
            &mut [],
            &bodies,
            &mut board,
            &mut forces,
            time
        ),
        Err(ExchangeError::Frozen {
            flags: 0,
            participants: 1
        })
    );
    assert_eq!(
        run_substep(
            &mut ps,
            &plan,
            &mut frozen,
            &bodies,
            &mut board,
            &mut ForceAccumulator::new(2),
            time
        ),
        Err(ExchangeError::Accumulate(AccumulateError::LengthMismatch {
            slots: 2,
            bodies: 1
        }))
    );
    assert_eq!(
        run_substep(
            &mut ps,
            &plan,
            &mut frozen,
            &bodies,
            &mut board,
            &mut forces,
            time
        ),
        Err(ExchangeError::BodyCount {
            field: TEMP,
            samples: 2,
            bodies: 1
        })
    );
    assert!(log.lock().expect("log").is_empty(), "a participant ran");
}

/// The committed values round-trip through the snapshot bytes into a board
/// declared the same way; a board with another layout or a short blob is
/// refused without being changed.
#[test]
fn field_values_round_trip_through_snapshot_bytes() {
    let quarter = Fix128::from_ratio(1, 4);
    let declare = |cells: usize| {
        let mut b = FieldBoard::new();
        b.declare(
            HEAT,
            grid((-1, 0, 2), quarter, [cells, 1, 1]),
            FieldMode::Sum,
        )
        .expect("declare");
        b.declare(TEMP, FieldLayout::PerBody { bodies: 2 }, FieldMode::Replace)
            .expect("declare");
        b
    };
    let mut a = declare(3);
    a.set(
        HEAT,
        &[Fix128::from_ratio(1, 3), -BIG, Fix128 { hi: 0, lo: 1 }],
    )
    .expect("set");
    a.set(TEMP, &[Fix128::from_int(300), Fix128::from_ratio(-7, 9)])
        .expect("set");
    let mut blob = Vec::new();
    a.write_values(&mut blob);
    assert_eq!(
        blob.len(),
        8 + (4 + 1 + 1 + 8 + 2 * 16) + (4 + 1 + 1 + 64 + 24 + 3 * 16)
    );
    let mut b = declare(3);
    b.check_values(&blob).expect("same declarations");
    b.read_values(&blob);
    assert_eq!(b, a, "read_values did not restore every value");

    let other = declare(4);
    let before = other.clone();
    assert!(other.check_values(&blob).is_err());
    assert_eq!(other, before);
    let mut swapped = FieldBoard::new();
    swapped
        .declare(
            HEAT,
            grid((-1, 0, 2), quarter, [3, 1, 1]),
            FieldMode::Replace,
        )
        .expect("declare");
    swapped
        .declare(TEMP, FieldLayout::PerBody { bodies: 2 }, FieldMode::Replace)
        .expect("declare");
    assert_eq!(swapped.check_values(&blob), Err(StateError::InvalidValue));
    assert_eq!(
        b.check_values(&blob[..blob.len() - 1]),
        Err(StateError::Length {
            expected: blob.len(),
            found: blob.len() - 1
        })
    );
    let mut empty = Vec::new();
    FieldBoard::new().write_values(&mut empty);
    assert_eq!(empty, 0u64.to_le_bytes());
    assert_eq!(FieldBoard::new().check_values(&empty), Ok(()));
}

/// A deterministic, uneven set of amounts (thirds, sevenths, negatives, one
/// ulp) so that no exact division hides a lost remainder.
fn amounts(n: usize) -> Vec<Fix128> {
    (0..n)
        .map(|i| {
            let i = i as i64;
            Fix128::from_ratio(i * 37 % 11 - 5, 3 + i % 7)
                + Fix128 {
                    hi: 0,
                    lo: i as u64,
                }
        })
        .collect()
}

fn total(v: &[Fix128]) -> Fix128 {
    v.iter().fold(Fix128::ZERO, |a, b| a + *b)
}

/// Oracle: coarsening a 4×2×6 grid of cell 1/4 onto a 2×1×3 grid of cell 1/2
/// gives each coarse cell the sum of its 8 fine cells (reference loop below,
/// written independently of the library), and the totals are equal.
#[test]
fn coarsening_sums_children_and_keeps_the_total() {
    let fine = grid((1, -2, 0), Fix128::from_ratio(1, 4), [4, 2, 6]);
    let coarse = grid((1, -2, 0), Fix128::from_ratio(1, 2), [2, 1, 3]);
    let v = amounts(48);
    let out = remap_conserving(&fine, &v, &coarse).expect("nested");
    let mut expected = vec![Fix128::ZERO; 6];
    for (f, value) in v.iter().enumerate() {
        let (x, y, z) = (f % 4, (f / 4) % 2, f / 8);
        expected[(z / 2) * 2 + (x / 2)] = expected[(z / 2) * 2 + (x / 2)] + *value;
        assert_eq!(y / 2, 0);
    }
    assert_eq!(out, expected);
    assert_eq!(total(&out), total(&v));
}

/// Oracle: refining keeps the total exactly and each coarse cell's children
/// add up to it, for amounts that `r³` does not divide; an amount that it
/// does divide (8 · 3/4) is split evenly (closed form 3/4 each).
#[test]
fn refining_splits_amounts_and_keeps_the_total_exactly() {
    let coarse = grid((0, 0, 0), Fix128::from_ratio(1, 2), [2, 1, 3]);
    let fine = grid((0, 0, 0), Fix128::from_ratio(1, 4), [4, 2, 6]);
    let v = amounts(6);
    let out = remap_conserving(&coarse, &v, &fine).expect("nested");
    assert_eq!(total(&out), total(&v), "refining changed the total");
    for (c, q) in v.iter().enumerate() {
        let (cx, cz) = (c % 2, c / 2);
        let mut sum = Fix128::ZERO;
        for z in 2 * cz..2 * cz + 2 {
            for y in 0..2 {
                for x in 2 * cx..2 * cx + 2 {
                    sum = sum + out[(z * 2 + y) * 4 + x];
                }
            }
        }
        assert_eq!(sum, *q, "children of coarse cell {c} do not add up to it");
    }
    let even = remap_conserving(&coarse, &[Fix128::from_int(6); 6], &fine).expect("nested");
    assert!(even.iter().all(|x| *x == Fix128::from_ratio(3, 4)));
    // ratio 3: 1/27 splits are not exact in binary
    let third = Fix128::from_ratio(1, 3);
    let c3 = grid((0, 0, 0), Fix128::from_int(3) * third, [1, 1, 1]);
    let f3 = grid((0, 0, 0), third, [3, 3, 3]);
    let q = [Fix128::from_ratio(-10, 7)];
    let split = remap_conserving(&c3, &q, &f3).expect("nested");
    assert_eq!(split.len(), 27);
    assert_eq!(total(&split), q[0]);
    let back = remap_conserving(&f3, &split, &c3).expect("nested");
    assert_eq!(back, q.to_vec(), "refine then coarsen is not the identity");
    // same grid: copy
    assert_eq!(remap_conserving(&c3, &q, &c3), Ok(q.to_vec()));
}

#[test]
fn remap_refuses_grids_that_are_not_nested() {
    let q = Fix128::from_ratio(1, 4);
    let a = grid((0, 0, 0), q, [4, 4, 4]);
    let v = amounts(64);
    assert_eq!(
        remap_conserving(
            &a,
            &v,
            &grid((1, 0, 0), Fix128::from_ratio(1, 2), [2, 2, 2])
        ),
        Err(RemapError::NotNested)
    );
    assert_eq!(
        remap_conserving(
            &a,
            &v,
            &grid((0, 0, 0), Fix128::from_ratio(3, 8), [2, 2, 2])
        ),
        Err(RemapError::NotNested)
    );
    assert_eq!(
        remap_conserving(
            &a,
            &v,
            &grid((0, 0, 0), Fix128::from_ratio(1, 2), [2, 2, 3])
        ),
        Err(RemapError::NotNested)
    );
    assert_eq!(
        remap_conserving(&a, &v[..63], &a),
        Err(RemapError::Length {
            expected: 64,
            found: 63
        })
    );
    assert_eq!(
        remap_conserving(&FieldLayout::PerBody { bodies: 64 }, &v, &a),
        Err(RemapError::InvalidLayout)
    );
    assert_eq!(
        remap_conserving(&a, &v, &grid((0, 0, 0), Fix128::ZERO, [1, 1, 1])),
        Err(RemapError::InvalidLayout)
    );
    let near_max = Fix128 {
        hi: i64::MAX,
        lo: 0,
    };
    let one = grid((0, 0, 0), Fix128::from_ratio(1, 2), [1, 1, 1]);
    assert_eq!(
        remap_conserving(&grid((0, 0, 0), q, [2, 2, 2]), &[near_max; 8], &one),
        Err(RemapError::OutOfRange)
    );
}

/// Oracle: amounts per body land in the cell holding the body and keep their
/// total; a body outside the grid is refused, never dropped.
#[test]
fn depositing_bodies_keeps_the_total() {
    let layout = grid((-2, 0, 0), Fix128::ONE, [4, 1, 1]);
    let at = |x: Fix128| {
        RigidBody::new_dynamic(
            Vec3Fix::new(x, Fix128::from_ratio(1, 2), Fix128::ZERO),
            Fix128::ONE,
        )
    };
    let bodies = vec![
        at(Fix128::from_ratio(-3, 2)),  // cell 0
        at(Fix128::ZERO),               // cell 2 (lower face belongs to the cell)
        at(Fix128::from_ratio(1, 3)),   // cell 2
        at(Fix128::from_ratio(19, 10)), // cell 3
    ];
    let q = [
        Fix128::from_ratio(1, 3),
        Fix128::from_int(2),
        Fix128::from_ratio(-5, 7),
        Fix128 { hi: 0, lo: 1 },
    ];
    let out = deposit_bodies(&bodies, &q, &layout).expect("inside");
    assert_eq!(out, vec![q[0], Fix128::ZERO, q[1] + q[2], q[3]]);
    assert_eq!(total(&out), total(&q));
    let outside = vec![at(Fix128::from_int(2))];
    assert_eq!(
        deposit_bodies(&outside, &[Fix128::ONE], &layout),
        Err(RemapError::OutsideGrid { body: 0 })
    );
    let below = vec![at(Fix128::from_ratio(-21, 10))];
    assert_eq!(
        deposit_bodies(&below, &[Fix128::ONE], &layout),
        Err(RemapError::OutsideGrid { body: 0 })
    );
    assert_eq!(
        deposit_bodies(&bodies, &q[..3], &layout),
        Err(RemapError::Length {
            expected: 4,
            found: 3
        })
    );
}

// ============================================================================
// Waking a parked body (pure)
// ============================================================================

fn sleep_cfg(t: Fix128) -> SleepConfig {
    SleepConfig {
        linear_threshold: t,
        angular_threshold: t,
        ..SleepConfig::default()
    }
}

/// Oracle (exact in binary): a participant force wakes a parked body when the
/// change the world applies, `Δv = (F·inv_mass)·h` or `Δω = (I⁻¹τ)·h`, is
/// non-zero in any component, whatever the sleep thresholds. With mass 1,
/// `I⁻¹ = 1` and `h = 1/4`: a force of 4 raw units gives `Δv` = one raw unit
/// and wakes; 1 or 3 raw units give `Δv = 0` (the product rounds down) and do
/// not; −1 raw unit gives `Δv` = −1 raw unit (rounded toward −∞) and wakes.
/// The former boundary `Δv = t = 2⁻⁷` wakes under every threshold. The same
/// for torque.
#[test]
fn a_parked_body_wakes_on_any_non_zero_applied_change() {
    let h = Fix128::from_ratio(1, 4);
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    body.inv_inertia = Vec3Fix::from_int(1, 1, 1);
    let raw = |k: i64| Fix128 {
        hi: if k < 0 { -1 } else { 0 },
        lo: k as u64,
    };
    let x = |f: Fix128| Vec3Fix::new(f, Fix128::ZERO, Fix128::ZERO);
    let z = Vec3Fix::ZERO;
    for cfg in [
        sleep_cfg(Fix128::from_ratio(1, 128)),
        SleepConfig::default(),
        sleep_cfg(Fix128::from_int(1000)),
    ] {
        assert!(
            wakes_parked_body(&body, x(raw(4)), z, h, &cfg),
            "Δv = 1 raw"
        );
        assert!(
            !wakes_parked_body(&body, x(raw(1)), z, h, &cfg),
            "Δv rounds to 0"
        );
        assert!(
            !wakes_parked_body(&body, x(raw(3)), z, h, &cfg),
            "Δv rounds to 0"
        );
        assert!(
            wakes_parked_body(&body, x(raw(-1)), z, h, &cfg),
            "Δv = −1 raw"
        );
        assert!(
            wakes_parked_body(&body, z, x(raw(4)), h, &cfg),
            "Δω = 1 raw"
        );
        assert!(
            !wakes_parked_body(&body, z, x(raw(3)), h, &cfg),
            "Δω rounds to 0"
        );
        let former = Fix128::from_ratio(1, 32);
        assert!(wakes_parked_body(&body, x(former), z, h, &cfg));
        assert!(wakes_parked_body(&body, z, x(former), h, &cfg));
        assert!(!wakes_parked_body(&body, z, z, h, &cfg));
        assert!(!wakes_parked_body(
            &RigidBody::new_static(Vec3Fix::ZERO),
            x(Fix128::from_int(1000)),
            z,
            h,
            &cfg
        ));
        // a change too large to represent wakes (the world reports it as a fault)
        let huge = Fix128 {
            hi: i64::MAX,
            lo: 0,
        };
        assert!(wakes_parked_body(
            &body,
            x(huge),
            z,
            Fix128::from_int(4),
            &cfg
        ));
        // the world tests' inputs
        assert!(wakes_parked_body(
            &body,
            Vec3Fix::from_int(1000, 0, 0),
            z,
            h_sub(),
            &cfg
        ));
        assert!(wakes_parked_body(
            &body,
            x(Fix128 { hi: 0, lo: 1 << 34 }),
            z,
            h_sub(),
            &cfg
        ));
    }
}

// ============================================================================
// World layer: receivers for entry points that do not exist yet
// ============================================================================

/// Receivers for the world entry points, one per entry point the tests
/// use.
mod world {
    use super::*;
    use alice_physics::solver::WorldSnapshotError;

    pub fn add_participant(
        world: &mut PhysicsWorld,
        p: Box<dyn Participant>,
    ) -> Result<usize, RegisterError> {
        world.add_participant(p)
    }

    pub fn participant_kinds(world: &PhysicsWorld) -> Vec<ParticipantKind> {
        world.participant_kinds()
    }

    pub fn participant_state(world: &PhysicsWorld, index: usize) -> Vec<u8> {
        world
            .participant_state(index)
            .expect("participant index in range")
    }

    pub fn try_step(world: &mut PhysicsWorld, dt: Fix128) -> Result<(), StepError> {
        world.try_step(dt)
    }

    #[cfg(feature = "parallel")]
    pub fn try_step_parallel(world: &mut PhysicsWorld, dt: Fix128) -> Result<(), StepError> {
        world.try_step_parallel(dt)
    }

    pub fn fault(world: &PhysicsWorld) -> Option<WorldFault> {
        world.fault()
    }

    pub fn clear_fault(world: &mut PhysicsWorld) {
        world.clear_fault();
    }

    pub fn observe_participant(
        world: &PhysicsWorld,
        index: usize,
    ) -> Option<Observed<ObservationSink>> {
        world.observe_participant(index)
    }

    pub fn observe_body_checked(
        world: &PhysicsWorld,
        index: usize,
    ) -> Option<Observed<alice_physics::solver::BodyObservation>> {
        world.observe_body_checked(index)
    }

    pub fn declare_field(
        world: &mut PhysicsWorld,
        id: PortId,
        layout: FieldLayout,
        mode: FieldMode,
    ) -> Result<(), FieldError> {
        match world.declare_field(id, layout, mode) {
            Ok(()) => Ok(()),
            Err(alice_physics::solver::DeclareFieldError::Field(e)) => Err(e),
            Err(e) => panic!("declaring a field refused by the ports: {e:?}"),
        }
    }

    pub fn fields(world: &PhysicsWorld) -> &FieldBoard {
        world.fields()
    }

    pub fn set_field(world: &mut PhysicsWorld, id: PortId, v: &[Fix128]) -> Result<(), FieldError> {
        world.set_field(id, v)
    }

    /// The field state error carried by a restore error.
    pub fn field_state_error_of(e: &WorldSnapshotError) -> Option<StateError> {
        match e {
            WorldSnapshotError::FieldState(s) => Some(*s),
            _ => None,
        }
    }

    /// The participant mismatch carried by a restore error.
    pub fn mismatch_of(e: &WorldSnapshotError) -> Option<ParticipantMismatch> {
        match e {
            WorldSnapshotError::ParticipantMismatch(m) => Some(*m),
            _ => None,
        }
    }

    /// The participant state error carried by a restore error.
    pub fn state_error_of(e: &WorldSnapshotError) -> Option<(usize, StateError)> {
        match e {
            WorldSnapshotError::ParticipantState { index, error } => Some((*index, *error)),
            _ => None,
        }
    }
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn config() -> PhysicsConfig {
    PhysicsConfig {
        substeps: 4,
        iterations: 4,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        ..Default::default()
    }
}

/// A floor and three stacked bodies, so contacts share bodies (the case where
/// solver order matters).
fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(config());
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    for i in 0..3 {
        w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 1 + 2 * i, 0), Fix128::ONE),
            Fix128::ONE,
        );
    }
    w
}

fn with_tethers(mut w: PhysicsWorld) -> PhysicsWorld {
    world::add_participant(&mut w, Box::new(Tether::new(0, 1))).expect("register");
    world::add_participant(&mut w, Box::new(Tether::new(1, 3).with_kind(KIND_OTHER)))
        .expect("register");
    w
}

fn motion(w: &PhysicsWorld) -> Vec<(Vec3Fix, Vec3Fix)> {
    w.bodies.iter().map(|b| (b.position, b.velocity)).collect()
}

/// (a) Every substep calls the participants once each, in registration order.
#[test]
fn substeps_call_participants_in_registration_order() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut w = scene();
    world::add_participant(&mut w, Box::new(Tether::new(7, 1).logging(&log))).expect("register");
    world::add_participant(&mut w, Box::new(Tether::new(3, 2).logging(&log))).expect("register");
    world::try_step(&mut w, dt()).expect("step");
    world::try_step(&mut w, dt()).expect("step");
    let expected: Vec<(u32, usize)> = (0..2)
        .flat_map(|_| (0..4).flat_map(|s| [(7, s), (3, s)]))
        .collect();
    assert_eq!(*log.lock().expect("log"), expected);
}

/// (b) A participant's `Err`: it stays as it was before the failing call, its
/// staged force never reaches the bodies (the bodies match a world without
/// it), the step reports `FaultRaised`, and it is not called again.
#[test]
fn failing_participant_leaves_itself_and_the_world_unchanged() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut w = scene();
    let failing = Tether::new(9, 1).failing_on(0).logging(&log);
    let before = failing.state_bytes();
    world::add_participant(&mut w, Box::new(failing)).expect("register");
    let mut reference = scene();

    let expected = WorldFault::Participant {
        index: 0,
        kind: KIND_TETHER,
        fault: ParticipantFault::OutOfRange,
    };
    assert_eq!(
        world::try_step(&mut w, dt()),
        Err(StepError::FaultRaised(expected))
    );
    reference.step(dt());

    assert_eq!(
        world::participant_state(&w, 0),
        before,
        "the failed participant changed"
    );
    assert_eq!(
        motion(&w),
        motion(&reference),
        "the failed participant's force reached a body"
    );
    assert_eq!(
        log.lock().expect("log").len(),
        1,
        "a failed participant was called again"
    );
    assert_eq!(world::fault(&w), Some(expected));
}

/// (c) snapshot → restore → N steps is bit-identical to N steps of the
/// original, participant state included (sequential path).
#[test]
fn snapshot_restore_is_bit_identical_with_participants() {
    let mut a = with_tethers(scene());
    for _ in 0..30 {
        world::try_step(&mut a, dt()).expect("step");
    }
    let blob = a.snapshot_world();
    let mut b = with_tethers(scene());
    b.restore_world(&blob).expect("restore");
    assert_eq!(
        b.snapshot_world(),
        blob,
        "restore did not reproduce the blob"
    );
    for _ in 0..60 {
        world::try_step(&mut a, dt()).expect("step");
        world::try_step(&mut b, dt()).expect("step");
    }
    assert_eq!(a.snapshot_world(), b.snapshot_world());
    for i in 0..2 {
        assert_eq!(
            world::participant_state(&a, i),
            world::participant_state(&b, i)
        );
    }
}

/// (c) the same on the parallel path, and the parallel path equals the
/// sequential one (both bit identity rules are v1).
#[cfg(feature = "parallel")]
#[test]
#[ignore = "src gap: step_parallel stores batch bookkeeping in the snapshot that step does not (single pipeline, separate work)"]
fn snapshot_restore_is_bit_identical_with_participants_in_parallel() {
    let mut a = with_tethers(scene());
    for _ in 0..30 {
        world::try_step_parallel(&mut a, dt()).expect("step");
    }
    let blob = a.snapshot_world();
    let mut b = with_tethers(scene());
    b.restore_world(&blob).expect("restore");
    let mut serial = with_tethers(scene());
    serial.restore_world(&blob).expect("restore");
    for _ in 0..60 {
        world::try_step_parallel(&mut a, dt()).expect("step");
        world::try_step_parallel(&mut b, dt()).expect("step");
        world::try_step(&mut serial, dt()).expect("step");
    }
    assert_eq!(a.snapshot_world(), b.snapshot_world());
    assert_eq!(
        a.snapshot_world(),
        serial.snapshot_world(),
        "parallel differs from sequential"
    );
}

fn restore_rejects(
    target: &mut PhysicsWorld,
    blob: &[u8],
) -> alice_physics::solver::WorldSnapshotError {
    let before = target.snapshot_world();
    let err = target
        .restore_world(blob)
        .expect_err("restore must be refused");
    assert_eq!(
        target.snapshot_world(),
        before,
        "a refused restore changed the target"
    );
    err
}

/// (d-1) different number of participants.
#[test]
fn restore_with_another_participant_count_is_refused() {
    let mut src = with_tethers(scene());
    world::try_step(&mut src, dt()).expect("step");
    let blob = src.snapshot_world();
    let mut target = scene();
    world::add_participant(&mut target, Box::new(Tether::new(0, 1))).expect("register");
    let err = restore_rejects(&mut target, &blob);
    assert_eq!(
        world::mismatch_of(&err),
        Some(ParticipantMismatch::Count {
            snapshot: 2,
            world: 1
        })
    );
}

/// (d-2) same count, another kind.
#[test]
fn restore_with_another_participant_kind_is_refused() {
    let mut src = with_tethers(scene());
    world::try_step(&mut src, dt()).expect("step");
    let blob = src.snapshot_world();
    let mut target = scene();
    let third = ParticipantKind::new(3);
    world::add_participant(&mut target, Box::new(Tether::new(0, 1))).expect("register");
    world::add_participant(&mut target, Box::new(Tether::new(1, 3).with_kind(third)))
        .expect("register");
    let err = restore_rejects(&mut target, &blob);
    assert_eq!(
        world::mismatch_of(&err),
        Some(ParticipantMismatch::Kind {
            index: 1,
            snapshot: KIND_OTHER,
            world: third
        })
    );
}

/// (d-3) same kinds, another order.
#[test]
fn restore_with_another_participant_order_is_refused() {
    let mut src = with_tethers(scene());
    world::try_step(&mut src, dt()).expect("step");
    let blob = src.snapshot_world();
    let mut target = scene();
    world::add_participant(
        &mut target,
        Box::new(Tether::new(1, 3).with_kind(KIND_OTHER)),
    )
    .expect("register");
    world::add_participant(&mut target, Box::new(Tether::new(0, 1))).expect("register");
    let err = restore_rejects(&mut target, &blob);
    assert_eq!(
        world::mismatch_of(&err),
        Some(ParticipantMismatch::Order {
            index: 0,
            snapshot: KIND_TETHER,
            world: KIND_OTHER
        })
    );
}

/// (d-4) every `check_state` runs before any `read_state`: a payload the
/// second participant refuses leaves the first one untouched too.
#[test]
fn restore_refused_by_the_second_participant_leaves_the_first_unchanged() {
    struct Picky(Tether);
    impl Participant for Picky {
        fn kind(&self) -> ParticipantKind {
            KIND_OTHER
        }
        fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
            self.0.substep(ctx, h)
        }
        fn observe(&self, out: &mut ObservationSink) {
            self.0.observe(out);
        }
        fn write_state(&self, out: &mut Vec<u8>) {
            self.0.write_state(out);
        }
        fn check_state(&self, _bytes: &[u8]) -> Result<(), StateError> {
            Err(StateError::InvalidValue)
        }
        fn read_state(&mut self, bytes: &[u8]) {
            self.0.read_state(bytes);
        }
    }
    let mut src = with_tethers(scene());
    world::try_step(&mut src, dt()).expect("step");
    let blob = src.snapshot_world();
    let mut target = scene();
    world::add_participant(&mut target, Box::new(Tether::new(0, 1))).expect("register");
    world::add_participant(&mut target, Box::new(Picky(Tether::new(1, 3)))).expect("register");
    let first_before = world::participant_state(&target, 0);
    let err = restore_rejects(&mut target, &blob);
    assert_eq!(
        world::state_error_of(&err),
        Some((1, StateError::InvalidValue))
    );
    assert_eq!(world::participant_state(&target, 0), first_before);
}

/// (e) While a fault is recorded: the next step is refused and does not touch
/// the world, observations are undecided; an explicit clear lets steps run
/// again.
#[test]
fn a_recorded_fault_refuses_the_next_step_and_observations_are_undecided() {
    let mut w = scene();
    world::add_participant(&mut w, Box::new(Tether::new(0, 1).failing_on(1))).expect("register");
    assert!(matches!(
        world::try_step(&mut w, dt()),
        Err(StepError::FaultRaised(_))
    ));
    let fault = world::fault(&w).expect("fault recorded");

    let before = w.snapshot_world();
    assert_eq!(
        world::try_step(&mut w, dt()),
        Err(StepError::Faulted(fault))
    );
    assert_eq!(
        w.snapshot_world(),
        before,
        "a refused step changed the world"
    );

    assert_eq!(world::observe_participant(&w, 0), Some(Observed::Undecided));
    assert_eq!(
        world::observe_body_checked(&w, 1),
        Some(Observed::Undecided)
    );
    assert_eq!(world::observe_body_checked(&w, 99), None);

    // the fault is part of the snapshot: a restored branch is still faulted
    let mut branch = scene();
    world::add_participant(&mut branch, Box::new(Tether::new(0, 1))).expect("register");
    branch.restore_world(&before).expect("restore");
    assert_eq!(world::fault(&branch), Some(fault));

    world::clear_fault(&mut w);
    assert_eq!(world::fault(&w), None);
    // the step runs again (it is not refused); the participant is unchanged
    // by its failure, so it fails again and the observation is undecided
    assert!(!matches!(
        world::try_step(&mut w, dt()),
        Err(StepError::Faulted(_))
    ));
    assert_eq!(
        world::observe_body_checked(&w, 1),
        Some(Observed::Undecided)
    );
}

/// (g) A version-1 blob (a byte fixture written while `snapshot_world` wrote
/// version 1, no `participants` section) reads as zero participants: it restores into a
/// world without participants, and is refused by one with participants.
#[test]
fn version_1_blob_reads_as_zero_participants() {
    // `scene()` stepped once by `dt()`, written by `snapshot_world` while the
    // format was version 1 (the writer now emits version 2).
    let v1: &[u8] = include_bytes!("fixtures/world_snapshot_v1_stacked.bin");
    assert_eq!(
        &v1[4..6],
        &1u16.to_le_bytes(),
        "fixture must be a version-1 blob"
    );
    let restored = PhysicsWorld::from_world_snapshot(v1).expect("v1 blob");
    assert!(world::participant_kinds(&restored).is_empty());
    let mut target = scene();
    world::add_participant(&mut target, Box::new(Tether::new(0, 1))).expect("register");
    let err = restore_rejects(&mut target, v1);
    assert_eq!(
        world::mismatch_of(&err),
        Some(ParticipantMismatch::Count {
            snapshot: 0,
            world: 1
        })
    );
}

/// A participant that pushes body `body` with a constant force every substep.
struct Push {
    body: usize,
    force: Vec3Fix,
}

const KIND_PUSH: ParticipantKind = ParticipantKind::new(0x5055_5348);

impl Participant for Push {
    fn kind(&self) -> ParticipantKind {
        KIND_PUSH
    }
    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        ctx.add_force(self.body, self.force)
            .map_err(|_| ParticipantFault::InvalidState)
    }
    fn observe(&self, _: &mut ObservationSink) {}
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

/// Ports change the order the world calls participants in: the reader
/// registered first runs after the writer registered second, in every
/// substep.
#[test]
fn substeps_call_participants_in_port_order() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut w = scene();
    let reader = Tether::new(7, 1)
        .logging(&log)
        .with_ports(&[Port::reads(PORT_A)]);
    let writer = Tether::new(3, 2)
        .logging(&log)
        .with_ports(&[Port::writes(PORT_A)]);
    world::add_participant(&mut w, Box::new(reader)).expect("register");
    world::add_participant(&mut w, Box::new(writer)).expect("register");
    world::try_step(&mut w, dt()).expect("step");
    let expected: Vec<(u32, usize)> = (0..4).flat_map(|s| [(3, s), (7, s)]).collect();
    assert_eq!(*log.lock().expect("log"), expected);
}

/// A participant that would close a loop of ports is refused at
/// registration and the world keeps the participants it had.
#[test]
fn registering_a_port_cycle_is_refused_and_changes_nothing() {
    let mut w = scene();
    let a = Tether::new(0, 1).with_ports(&[Port::reads(PORT_A), Port::writes(PORT_B)]);
    let b = Tether::new(1, 2)
        .with_kind(KIND_OTHER)
        .with_ports(&[Port::reads(PORT_B), Port::writes(PORT_A)]);
    assert_eq!(world::add_participant(&mut w, Box::new(a)), Ok(0));
    let before = w.snapshot_world();
    assert_eq!(
        world::add_participant(&mut w, Box::new(b)),
        Err(RegisterError::Order(OrderError::Cycle {
            members: vec![0, 1]
        }))
    );
    assert_eq!(world::participant_kinds(&w), vec![KIND_TETHER]);
    assert_eq!(w.snapshot_world(), before);
}

/// A world with one dynamic body at rest in zero gravity, stepped until the
/// body is parked (sleeping, skipped by the step).
fn parked_scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 4,
        iterations: 4,
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    });
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    w.set_sleep_skip(true);
    for _ in 0..120 {
        w.step(dt());
    }
    assert!(w.is_sleeping(0), "fixture: the body must be parked");
    w
}

/// A parked body is woken by a participant force that changes it
/// ([`wakes_parked_body`]). The force (1000 N on 1 kg, a velocity change of
/// about 4 m/s per substep) is far from zero.
#[test]
fn a_force_wakes_a_parked_body() {
    let mut w = parked_scene();
    let push = Push {
        body: 0,
        force: Vec3Fix::from_int(1000, 0, 0),
    };
    world::add_participant(&mut w, Box::new(push)).expect("register");
    world::try_step(&mut w, dt()).expect("step");
    assert!(!w.is_sleeping(0), "the body stayed parked");
    assert!(
        w.bodies[0].velocity.x > Fix128::ZERO,
        "the force had no effect"
    );
}

/// A force far below the sleep threshold still wakes a parked body when its
/// change is not zero: 2⁻³⁰ N on 1 kg over `h = 1/240` changes the velocity
/// by about `4e-12` m/s (threshold 0.01 m/s).
#[test]
fn a_force_below_the_sleep_threshold_wakes_a_parked_body() {
    let mut w = parked_scene();
    let tiny = Fix128 { hi: 0, lo: 1 << 34 };
    let push = Push {
        body: 0,
        force: Vec3Fix::new(tiny, Fix128::ZERO, Fix128::ZERO),
    };
    world::add_participant(&mut w, Box::new(push)).expect("register");
    world::try_step(&mut w, dt()).expect("step");
    assert!(!w.is_sleeping(0), "a non-zero change left the body asleep");
}

/// A force whose change rounds to zero (one raw unit on 1 kg over
/// `h = 1/240`) leaves a parked body parked, and the body does not move.
#[test]
fn a_force_whose_change_rounds_to_zero_leaves_a_parked_body_parked() {
    let mut w = parked_scene();
    let before = motion(&w);
    let push = Push {
        body: 0,
        force: Vec3Fix::new(Fix128 { hi: 0, lo: 1 }, Fix128::ZERO, Fix128::ZERO),
    };
    world::add_participant(&mut w, Box::new(push)).expect("register");
    for _ in 0..10 {
        world::try_step(&mut w, dt()).expect("step");
    }
    assert!(w.is_sleeping(0), "a zero change woke the body");
    assert_eq!(motion(&w), before, "the force moved a parked body");
}

/// A world with one dynamic body at rest in zero gravity, the sleep thresholds
/// at `2⁻⁷`, stepped with `dt = 1/64` (substep `h = 1/256`, exact in binary)
/// until the body is parked.
fn parked_scene_at_binary_threshold() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 4,
        iterations: 4,
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    });
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    w.set_sleep_config(sleep_cfg(Fix128::from_ratio(1, 128)));
    w.set_sleep_skip(true);
    for _ in 0..120 {
        w.step(Fix128::from_ratio(1, 64));
    }
    assert!(w.is_sleeping(0), "fixture: the body must be parked");
    w
}

/// At the sleep threshold, in the world: 2 N on 1 kg over `h = 1/256`
/// changes the velocity by exactly `2⁻⁷`, the threshold; the change is not
/// zero, so the body wakes and moves (the threshold plays no part).
#[test]
fn a_force_exactly_at_the_sleep_threshold_wakes_a_parked_body() {
    let mut w = parked_scene_at_binary_threshold();
    let push = Push {
        body: 0,
        force: Vec3Fix::from_int(2, 0, 0),
    };
    world::add_participant(&mut w, Box::new(push)).expect("register");
    world::try_step(&mut w, Fix128::from_ratio(1, 64)).expect("step");
    assert!(!w.is_sleeping(0), "the body stayed parked");
    assert!(
        w.bodies[0].velocity.x > Fix128::ZERO,
        "the force had no effect"
    );
}

/// Just above the sleep threshold: `2 + 2⁻⁵⁰` N gives `Δv = 2⁻⁷ + 2⁻⁵⁸` and
/// wakes the body.
#[test]
fn a_force_just_above_the_wake_threshold_wakes_a_parked_body() {
    let mut w = parked_scene_at_binary_threshold();
    let f = Fix128::from_int(2) + Fix128 { hi: 0, lo: 1 << 14 };
    let push = Push {
        body: 0,
        force: Vec3Fix::new(f, Fix128::ZERO, Fix128::ZERO),
    };
    world::add_participant(&mut w, Box::new(push)).expect("register");
    world::try_step(&mut w, Fix128::from_ratio(1, 64)).expect("step");
    assert!(!w.is_sleeping(0), "the body stayed parked");
    assert!(
        w.bodies[0].velocity.x > Fix128::ZERO,
        "the force had no effect"
    );
}

/// Fields in the world: the board the world owns is committed once per
/// substep. With a writer storing `k + 1` in substep `k` (k = 0..4 per frame)
/// and a reader registered first, the reader sees `0, 1, 2, 3` and the
/// committed value after the frame is 4 (closed form).
#[test]
fn fields_are_committed_once_per_substep_in_the_world() {
    let mut w = scene();
    world::declare_field(
        &mut w,
        TEMP,
        FieldLayout::PerBody { bodies: 4 },
        FieldMode::Replace,
    )
    .expect("declare");
    let log = Arc::new(Mutex::new(Vec::new()));
    world::add_participant(
        &mut w,
        Box::new(Reader {
            ports: vec![Port::reads_committed(TEMP)],
            field: TEMP,
            log: Arc::clone(&log),
        }),
    )
    .expect("register");
    world::add_participant(&mut w, Box::new(Writer::new(TEMP, one_value))).expect("register");
    world::try_step(&mut w, dt()).expect("step");
    let seen: Vec<Fix128> = log.lock().expect("log").iter().map(|v| v[0]).collect();
    assert_eq!(seen, (0..4).map(Fix128::from_int).collect::<Vec<_>>());
    assert_eq!(
        world::fields(&w).value(TEMP),
        Some(&[Fix128::from_int(4); 4][..])
    );
}

/// The committed field values are in the snapshot once (the world's section,
/// not a participant payload): a restore into a world declared the same way
/// reproduces them, a world with another layout refuses the blob unchanged.
#[test]
fn snapshot_carries_field_values_once() {
    let build = |bodies: usize| {
        let mut w = scene();
        world::declare_field(
            &mut w,
            HEAT,
            FieldLayout::PerBody { bodies },
            FieldMode::Sum,
        )
        .expect("declare");
        w
    };
    let mut a = build(4);
    world::set_field(&mut a, HEAT, &[Fix128::from_ratio(1, 3); 4]).expect("set");
    world::add_participant(&mut a, Box::new(Writer::new(HEAT, seven))).expect("register");
    let mut b = build(4);
    world::add_participant(&mut b, Box::new(Writer::new(HEAT, seven))).expect("register");
    world::try_step(&mut a, dt()).expect("step");
    let blob = a.snapshot_world();
    b.restore_world(&blob).expect("restore");
    assert_eq!(world::fields(&b), world::fields(&a));
    assert_eq!(b.snapshot_world(), blob);

    let mut other = scene();
    world::declare_field(
        &mut other,
        HEAT,
        FieldLayout::PerBody { bodies: 4 },
        FieldMode::Replace,
    )
    .expect("declare");
    world::add_participant(&mut other, Box::new(Writer::new(HEAT, seven))).expect("register");
    let err = restore_rejects(&mut other, &blob);
    assert_eq!(
        world::field_state_error_of(&err),
        Some(StateError::InvalidValue)
    );
}

/// (h) With no participant registered, the checked step is the existing step
/// bit for bit (the container changes nothing), on the scenes of
/// `tests/determinism_golden.rs` (cascade: golden hash copied below) and on the
/// stacked scene here.
#[test]
fn zero_participants_match_the_existing_step_and_golden() {
    let mut a = scene();
    let mut b = scene();
    for _ in 0..120 {
        a.step(dt());
        world::try_step(&mut b, dt()).expect("step");
    }
    assert_eq!(a.snapshot_world(), b.snapshot_world());

    // determinism_golden::determinism_cascade, with try_step
    let config = PhysicsConfig {
        substeps: 4,
        iterations: 8,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        damping: Fix128::from_ratio(99, 100),
        ..Default::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    for i in 0..5 {
        w.add_body(
            RigidBody::new_dynamic(Vec3Fix::from_int(i, 5 + i * 2, 0), Fix128::ONE)
                .with_restitution(Fix128::from_ratio(6, 10))
                .with_linear_damping(Fix128::from_ratio(98, 100)),
        );
    }
    for _ in 0..200 {
        world::try_step(&mut w, dt()).expect("step");
    }
    assert_eq!(cascade_hash(&w), GOLDEN_CASCADE);
}

/// Re-recorded when XPBD began keeping the predicted velocity of bodies the
/// solve did not move (`v = v_pred + Δx_corr / h`).
const GOLDEN_CASCADE: &str = "95d1f0805b7b5b2cae4030ba7bd74749cfe7bc60869d1478eeeac67fab27d381";

fn cascade_hash(w: &PhysicsWorld) -> String {
    use sha2::{Digest, Sha256};
    let mut bytes = Vec::new();
    for b in &w.bodies {
        for f in [
            b.position.x,
            b.position.y,
            b.position.z,
            b.velocity.x,
            b.velocity.y,
            b.velocity.z,
        ] {
            put_fix(&mut bytes, f);
        }
    }
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The golden copied above is the live one: the existing `step` still gives
/// it, so (h) compares against the current fixture, not a stale copy.
#[test]
fn copied_cascade_golden_is_current() {
    let config = PhysicsConfig {
        substeps: 4,
        iterations: 8,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        damping: Fix128::from_ratio(99, 100),
        ..Default::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    for i in 0..5 {
        w.add_body(
            RigidBody::new_dynamic(Vec3Fix::from_int(i, 5 + i * 2, 0), Fix128::ONE)
                .with_restitution(Fix128::from_ratio(6, 10))
                .with_linear_damping(Fix128::from_ratio(98, 100)),
        );
    }
    for _ in 0..200 {
        w.step(dt());
    }
    assert_eq!(cascade_hash(&w), GOLDEN_CASCADE);
}
