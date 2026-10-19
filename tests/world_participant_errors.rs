//! The participant errors as standard errors: `RegisterError` and
//! `StepError` print a message and implement `std::error::Error`, so they
//! pass through `?` into `Box<dyn Error>`; `StepError::Rule` names its
//! `RegisterError` as the source.
//!
//! Every message below is the stated text (the exact string is the oracle),
//! and the errors in the world tests are the ones the world returns, not
//! values built by hand.

use std::error::Error;

use alice_physics_core::world_participant::FieldPortError;
use alice_physics_core::world_participant::{
    ObservationSink, OrderError, Participant, ParticipantFault, ParticipantKind, PortId,
    RegisterError, StateError, StepError, StepRule, SubstepCtx, WorldFault,
};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

/// A participant with no state: a fixed step and an optional failure.
struct Probe {
    rule: StepRule,
    fail: bool,
}

impl Participant for Probe {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(u32::from_be_bytes(*b"PRBE"))
    }
    fn step_rule(&self) -> StepRule {
        self.rule
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        if self.fail {
            Err(ParticipantFault::OutOfRange)
        } else {
            Ok(())
        }
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, _: &[u8]) -> Result<(), StateError> {
        Ok(())
    }
    fn read_state(&mut self, _: &[u8]) {}
}

fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 4,
        ..PhysicsConfig::default()
    });
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    w
}

fn register(w: &mut PhysicsWorld, rule: StepRule, fail: bool) -> Result<usize, Box<dyn Error>> {
    Ok(w.add_participant(Box::new(Probe { rule, fail }))?)
}

fn step(w: &mut PhysicsWorld, dt: Fix128) -> Result<(), Box<dyn Error>> {
    w.try_step(dt)?;
    Ok(())
}

/// oracle (stated text): every `RegisterError` variant prints its message.
#[test]
fn register_error_messages() {
    let cases: Vec<(RegisterError, &str)> = vec![
        (
            RegisterError::NonPositiveStep,
            "the world substep or the participant's fixed step is not positive",
        ),
        (
            RegisterError::StepRuleMismatch {
                substep: Fix128::from_ratio(1, 4),
                step: Fix128::from_ratio(1, 3),
            },
            "the fixed step 0.3333 does not divide the substep 0.2500",
        ),
        (
            RegisterError::Order(OrderError::Cycle {
                members: vec![0, 2],
            }),
            "the declared ports form a loop through participants [0, 2]",
        ),
        (
            RegisterError::Field(FieldPortError::SecondWriter {
                field: PortId::new(7),
                first: 1,
                second: 3,
            }),
            "replace field 7 is written by participants 1 and 3",
        ),
        (
            RegisterError::Field(FieldPortError::ReadOfField {
                participant: 2,
                field: PortId::new(7),
            }),
            "participant 2 reads field 7 with a plain read (a field is read committed)",
        ),
        (
            RegisterError::Field(FieldPortError::UnknownField {
                participant: 2,
                field: PortId::new(9),
            }),
            "participant 2 reads port 9 as committed, but no field has that id",
        ),
    ];
    for (e, want) in cases {
        assert_eq!(e.to_string(), want, "{e:?}");
        assert!(e.source().is_none(), "{e:?} has no source");
    }
}

/// oracle (stated text): every `StepError` variant prints its message, with
/// the fault it carries; the kind prints as its four ASCII letters, or as hex
/// when they are not printable.
#[test]
fn step_error_messages() {
    let prbe = ParticipantKind::new(u32::from_be_bytes(*b"PRBE"));
    let participant = WorldFault::Participant {
        index: 1,
        kind: prbe,
        fault: ParticipantFault::OutOfRange,
    };
    let cases: Vec<(StepError, &str)> = vec![
        (
            StepError::Faulted(participant),
            "a fault is already recorded, the world was not stepped: participant 1 (PRBE) failed: a value left its range",
        ),
        (
            StepError::FaultRaised(WorldFault::Participant {
                index: 0,
                kind: ParticipantKind::new(7),
                fault: ParticipantFault::InvalidState,
            }),
            "a participant failed during this step: participant 0 (0x00000007) failed: its state is inconsistent",
        ),
        (
            StepError::FaultRaised(WorldFault::RigidOverflow),
            "a participant failed during this step: the rigid integration left the range of Fix128",
        ),
        (
            StepError::FaultRaised(WorldFault::ForceOutOfRange { body: 4 }),
            "a participant failed during this step: the participant force on body 4 left the range of Fix128",
        ),
        (
            StepError::FaultRaised(WorldFault::FieldOutOfRange { field: PortId::new(5), index: 2 }),
            "a participant failed during this step: committing field 5 left the range of Fix128 at participant 2",
        ),
        (
            StepError::Rule { index: 3, error: RegisterError::NonPositiveStep },
            "participant 3 cannot follow this frame's substep: the world substep or the participant's fixed step is not positive",
        ),
        (
            StepError::BodyCount { field: PortId::new(5), samples: 2, bodies: 3 },
            "per-body field 5 has 2 samples but the world has 3 bodies",
        ),
    ];
    for (e, want) in cases {
        assert_eq!(e.to_string(), want, "{e:?}");
    }
}

/// oracle (source chain): `StepError::Rule` names its `RegisterError` as the
/// source; the other variants have none.
#[test]
fn only_the_rule_error_has_a_source() {
    let rule = StepError::Rule {
        index: 0,
        error: RegisterError::NonPositiveStep,
    };
    let source = rule.source().expect("source");
    assert_eq!(
        source.to_string(),
        RegisterError::NonPositiveStep.to_string()
    );
    assert!(StepError::FaultRaised(WorldFault::RigidOverflow)
        .source()
        .is_none());
    assert!(StepError::BodyCount {
        field: PortId::new(1),
        samples: 1,
        bodies: 2
    }
    .source()
    .is_none());
}

/// oracle (through `?`, from the world): registering a zero fixed step is
/// refused with `NonPositiveStep`; a fixed step of `1/3` under a substep of
/// `1/240` (`dt = 1/60`, 4 substeps) is refused at the step with `Rule`,
/// whose source is the mismatch; a failing participant raises a fault and the
/// next step reports it already recorded. Each error reaches the caller as a
/// `Box<dyn Error>` with its message and downcasts back to its type.
#[test]
fn world_errors_pass_through_question_mark() {
    let dt = Fix128::from_ratio(1, 60);

    let mut w = world();
    let e = register(&mut w, StepRule::Fixed(Fix128::ZERO), false).expect_err("zero step");
    assert_eq!(e.to_string(), RegisterError::NonPositiveStep.to_string());
    assert_eq!(
        e.downcast_ref::<RegisterError>(),
        Some(&RegisterError::NonPositiveStep)
    );

    let mut w = world();
    register(&mut w, StepRule::Fixed(Fix128::from_ratio(1, 3)), false).expect("registered");
    let e = step(&mut w, dt).expect_err("rule mismatch");
    let got = e.downcast_ref::<StepError>().expect("a StepError");
    assert!(matches!(got, StepError::Rule { index: 0, .. }), "{got:?}");
    assert!(
        e.to_string()
            .starts_with("participant 0 cannot follow this frame's substep: the fixed step"),
        "{e}"
    );
    assert!(e.source().is_some(), "the rule error names its source");

    let mut w = world();
    register(&mut w, StepRule::FollowSubstep, true).expect("registered");
    let e = step(&mut w, dt).expect_err("fault");
    assert!(
        matches!(
            e.downcast_ref::<StepError>(),
            Some(StepError::FaultRaised(_))
        ),
        "{e:?}"
    );
    assert!(
        e.to_string().ends_with("failed: a value left its range"),
        "{e}"
    );
    let e = step(&mut w, dt).expect_err("already faulted");
    assert!(
        e.to_string().starts_with("a fault is already recorded"),
        "{e}"
    );
}
