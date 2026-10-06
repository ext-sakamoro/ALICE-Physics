//! Conformance tests of the participant contract ([`alice_physics::world_participant`]).
//!
//! Two layers:
//!
//! * **participant layer** (runs today): the types of the contract on their
//!   own, and [`check_participant_contract`], which any participant type can be
//!   run through (fail leaves it unchanged, `write_state` → `check_state` →
//!   `read_state` restores it bit for bit, a short payload is rejected).
//! * **world layer** (`#[ignore = "src gap: WORLD-V1-S1 ..."]`): what
//!   `PhysicsWorld` must do with registered participants. The world side is
//!   not written yet, so these tests call the receivers in the `world` module
//!   below, which stop with `todo!` naming the missing entry point. When the
//!   entry point lands, replace the receiver body with the call and remove
//!   the `ignore`.
//!
//! Oracles: call order and bit identity are compared against a second world
//! built independently (a world without participants, a world stepped
//! without a snapshot in between), never against values the world under test
//! computed for itself.

#![cfg(feature = "std")]

use std::sync::{Arc, Mutex};

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::world_participant::{
    execution_order, AccumulateError, ForceAccumulator, ObservationSink, Observed, OrderError,
    Participant, ParticipantFault, ParticipantKind, ParticipantMismatch, Port, PortAccess, PortId,
    RegisterError, StateError, StepError, StepRule, SubstepCtx, Verdict, WorldFault,
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
// World layer: receivers for entry points that do not exist yet
// ============================================================================

/// Receivers with the signatures the world side will have. Each stops with
/// `todo!` naming the entry point; replace the body with the call when it
/// lands.
mod world {
    use super::*;

    pub fn add_participant(
        _world: &mut PhysicsWorld,
        _p: Box<dyn Participant>,
    ) -> Result<usize, RegisterError> {
        todo!("src gap: WORLD-V1-S1 PhysicsWorld::add_participant")
    }

    pub fn participant_kinds(_world: &PhysicsWorld) -> Vec<ParticipantKind> {
        todo!("src gap: WORLD-V1-S1 PhysicsWorld::participant_kinds")
    }

    pub fn participant_state(_world: &PhysicsWorld, _index: usize) -> Vec<u8> {
        todo!("src gap: WORLD-V1-S1 PhysicsWorld::participant_state")
    }

    pub fn try_step(_world: &mut PhysicsWorld, _dt: Fix128) -> Result<(), StepError> {
        todo!("src gap: WORLD-V1-S1 PhysicsWorld::try_step")
    }

    #[cfg(feature = "parallel")]
    pub fn try_step_parallel(_world: &mut PhysicsWorld, _dt: Fix128) -> Result<(), StepError> {
        todo!("src gap: WORLD-V1-S1 PhysicsWorld::try_step_parallel")
    }

    pub fn fault(_world: &PhysicsWorld) -> Option<WorldFault> {
        todo!("src gap: WORLD-V1-S1 PhysicsWorld::fault")
    }

    pub fn clear_fault(_world: &mut PhysicsWorld) {
        todo!("src gap: WORLD-V1-S1 PhysicsWorld::clear_fault")
    }

    pub fn observe_participant(
        _world: &PhysicsWorld,
        _index: usize,
    ) -> Option<Observed<ObservationSink>> {
        todo!("src gap: WORLD-V1-S1 PhysicsWorld::observe_participant")
    }

    pub fn observe_body_checked(
        _world: &PhysicsWorld,
        _index: usize,
    ) -> Option<Observed<alice_physics::solver::BodyObservation>> {
        todo!("src gap: WORLD-V1-S1 PhysicsWorld::observe_body_checked")
    }

    /// The participant mismatch carried by a restore error.
    pub fn mismatch_of(
        _e: &alice_physics::solver::WorldSnapshotError,
    ) -> Option<ParticipantMismatch> {
        todo!("src gap: WORLD-V1-S1 WorldSnapshotError::ParticipantMismatch")
    }

    /// The participant state error carried by a restore error.
    pub fn state_error_of(
        _e: &alice_physics::solver::WorldSnapshotError,
    ) -> Option<(usize, StateError)> {
        todo!("src gap: WORLD-V1-S1 WorldSnapshotError::ParticipantState")
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
#[ignore = "src gap: WORLD-V1-S1 PhysicsWorld::add_participant / try_step"]
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
#[ignore = "src gap: WORLD-V1-S1 PhysicsWorld::add_participant / try_step / fault"]
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
#[ignore = "src gap: WORLD-V1-S1 snapshot section `participants` (format v2)"]
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
#[ignore = "src gap: WORLD-V1-S1 PhysicsWorld::try_step_parallel (single pipeline with step)"]
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
#[ignore = "src gap: WORLD-V1-S1 restore checks the participant count"]
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
#[ignore = "src gap: WORLD-V1-S1 restore checks the participant kinds"]
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
#[ignore = "src gap: WORLD-V1-S1 restore checks the participant order"]
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
#[ignore = "src gap: WORLD-V1-S1 restore checks every participant payload before reading any"]
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
#[ignore = "src gap: WORLD-V1-S1 sticky fault, observe_participant / observe_body_checked, clear_fault"]
fn a_recorded_fault_refuses_the_next_step_and_observations_are_undecided() {
    let mut w = scene();
    world::add_participant(&mut w, Box::new(Tether::new(0, 1).failing_on(5))).expect("register");
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
    assert!(world::try_step(&mut w, dt()).is_ok());
    assert!(matches!(
        world::observe_body_checked(&w, 1),
        Some(Observed::Exact(_))
    ));
}

/// (g) A version-1 blob (what `snapshot_world` writes today, no
/// `participants` section) reads as zero participants: it restores into a
/// world without participants, and is refused by one with participants.
#[test]
#[ignore = "src gap: WORLD-V1-S1 snapshot format v2 reads v1 as zero participants"]
fn version_1_blob_reads_as_zero_participants() {
    let mut src = scene();
    src.step(dt());
    let v1 = src.snapshot_world();
    assert_eq!(
        &v1[4..6],
        &1u16.to_le_bytes(),
        "fixture must be a version-1 blob"
    );
    // The fixture is captured before the version moves. The commit that makes
    // the writer emit version 2 replaces it with these bytes fixed in the
    // test, the way a golden hash is updated: the old and the new bytes are
    // shown side by side in that commit.
    let restored = PhysicsWorld::from_world_snapshot(&v1).expect("v1 blob");
    assert!(world::participant_kinds(&restored).is_empty());
    let mut target = scene();
    world::add_participant(&mut target, Box::new(Tether::new(0, 1))).expect("register");
    let err = restore_rejects(&mut target, &v1);
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
#[ignore = "src gap: WORLD-V1-S1 PhysicsWorld::add_participant runs participants in execution_order"]
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
#[ignore = "src gap: WORLD-V1-S1 PhysicsWorld::add_participant refuses a cycle of ports"]
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

/// A parked body is woken by a participant force above the wake threshold.
/// The force (1000 N on 1 kg, a velocity change of about 4 m/s per substep)
/// is far above any threshold of the form discussed in the design notes
/// (fraction of the sleep velocity threshold per substep, or of `m·|g|`).
#[test]
#[ignore = "src gap: WORLD-V1-S1 a parked body wakes above the force threshold"]
fn a_force_above_the_wake_threshold_wakes_a_parked_body() {
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

/// A parked body is not woken by a participant force below the threshold,
/// and the force has no effect on it (the sleep contract is unchanged). The
/// force (2⁻³⁰ N on 1 kg) changes the velocity by far less than the sleep
/// threshold of 0.01 m/s.
#[test]
#[ignore = "src gap: WORLD-V1-S1 a parked body stays parked below the force threshold"]
fn a_force_below_the_wake_threshold_leaves_a_parked_body_parked() {
    let mut w = parked_scene();
    let before = motion(&w);
    let tiny = Fix128 { hi: 0, lo: 1 << 34 };
    let push = Push {
        body: 0,
        force: Vec3Fix::new(tiny, Fix128::ZERO, Fix128::ZERO),
    };
    world::add_participant(&mut w, Box::new(push)).expect("register");
    for _ in 0..10 {
        world::try_step(&mut w, dt()).expect("step");
    }
    assert!(
        w.is_sleeping(0),
        "a force below the threshold woke the body"
    );
    assert_eq!(motion(&w), before, "the force moved a parked body");
}

/// (h) With no participant registered, the checked step is the existing step
/// bit for bit (the container changes nothing), on the scenes of
/// `tests/determinism_golden.rs` (cascade: golden hash copied below) and on the
/// stacked scene here.
#[test]
#[ignore = "src gap: WORLD-V1-S1 PhysicsWorld::try_step"]
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

const GOLDEN_CASCADE: &str = "f442b544c0eab74a004567061ab7b573d6892158b2c31480b6d133e2e5970be2";

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
