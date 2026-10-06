//! Independent checks of the execution order of participants within a
//! substep, with the seven law participants (THRM, PHAS, PRES, FRAC, EROS,
//! CRWD, MDVV) registered between probe participants that declare ports.
//!
//! The rule under test (module doc of `world_participant`): a writer of a
//! port runs before its readers, registration order otherwise; a loop of
//! ports is refused, never ordered. The law participants declare no ports,
//! so they keep their registration rank among the participants free to run.
//!
//! Two observables encode the order: a shared call log, and the index a
//! [`FieldMode::Sum`] commit names when its partial sums, taken in execution
//! order, leave the `Fix128` range (the sums do not commute with the range
//! check, so another order names another participant or none).
//!
//! Scenes differ from `tests/world_participant_conformance.rs` (two
//! tethers, one port): the dependency runs against registration order across
//! law participants, a chain of three, two writers of one port with a free
//! participant between them, a loop of three with a dependant, and the
//! missing-dependency cases of the registration checks.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};

use alice_physics::crowd_force::{
    CrowdParticipant, InteractionParams, NeighborSearch, Pedestrian, SocialForce,
};
use alice_physics::erosion::{ErosionConfig, ErosionModifier};
use alice_physics::fracture::{FractureConfig, FractureModifier};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::molecular_dynamics::{MdParticipant, PeriodicBox, VelocityVerlet};
use alice_physics::pair_potential::{LennardJones, ShiftMode, Truncated};
use alice_physics::phase_change::{PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::physics2d::Vec2Fix;
use alice_physics::pressure::{PressureConfig, PressureModifier};
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::solver::RigidBody;
use alice_physics::thermal::{ThermalConfig, ThermalModifier};
use alice_physics::world_participant::{
    execution_order, run_substep, FieldBoard, FieldLayout, FieldMode, FieldPortError,
    ForceAccumulator, ObservationSink, OrderError, Participant, ParticipantFault, ParticipantKind,
    ParticipantPlan, Port, PortId, RegisterError, StateError, SubstepCtx, SubstepTime, WorldFault,
};

// ============================================================================
// Probe participant
// ============================================================================

type Log = Arc<Mutex<Vec<u32>>>;

/// Logs its tag on every call, optionally stages a value on a sum field;
/// its state is the number of calls.
struct Probe {
    tag: u32,
    ports: Vec<Port>,
    log: Log,
    stage: Option<(PortId, Fix128)>,
    calls: u64,
}

impl Probe {
    fn new(tag: u32, ports: &[Port], log: &Log) -> Self {
        Self {
            tag,
            ports: ports.to_vec(),
            log: Arc::clone(log),
            stage: None,
            calls: 0,
        }
    }

    fn staging(mut self, field: PortId, value: Fix128) -> Self {
        self.stage = Some((field, value));
        self
    }
}

impl Participant for Probe {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0x77_0000 + self.tag)
    }

    fn ports(&self) -> &[Port] {
        &self.ports
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, _h: Fix128) -> Result<(), ParticipantFault> {
        self.log.lock().expect("log").push(self.tag);
        if let Some((field, value)) = self.stage {
            ctx.stage_field(field).expect("declared write")[0] = value;
        }
        self.calls += 1;
        Ok(())
    }

    fn observe(&self, _out: &mut ObservationSink) {}

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
        let mut b = [0u8; 8];
        b.copy_from_slice(bytes);
        self.calls = u64::from_le_bytes(b);
    }
}

// ============================================================================
// Law participants (small scenes; their content is not under test here)
// ============================================================================

const LO: (f32, f32, f32) = (-1.5, -0.5, -2.0);
const HI: (f32, f32, f32) = (1.5, 2.5, 1.0);

fn thermal() -> ThermalModifier {
    let mut m = ThermalModifier::new(ThermalConfig::default(), 4, LO, HI);
    m.add_heat_point(0.5, 1.0, -0.5, 900.0, 0.8);
    m
}

fn phase_change() -> PhaseChangeModifier {
    let mut m = PhaseChangeModifier::new(PhaseChangeConfig::default(), 4, LO, HI);
    m.apply_heat_at(-0.5, 0.5, 0.0, 700.0, 1.0);
    m
}

fn pressure() -> PressureModifier {
    let mut m = PressureModifier::new(PressureConfig::default(), 4, LO, HI);
    m.apply_impact(0.0, 1.0, -1.0, 6.0, 0.9);
    m
}

fn fracture() -> FractureModifier {
    let mut m = FractureModifier::new(FractureConfig::default(), 4, LO, HI);
    m.apply_stress_at(0.2, 1.2, -0.4, 500.0, 1.0);
    m
}

fn erosion() -> ErosionModifier {
    let mut m = ErosionModifier::new(ErosionConfig::default(), 4, LO, HI);
    m.set_exposure_at(0.0, 2.0, 0.5, 1.0, 1.0);
    m
}

fn crowd() -> CrowdParticipant {
    let params = InteractionParams {
        strength_n: Fix128::from_int(1000),
        range_m: Fix128::from_ratio(1, 8),
        body_stiffness: Fix128::from_int(50_000),
        sliding_friction: Fix128::from_int(100_000),
    };
    let model = SocialForce::new(params, params, Fix128::ONE, Fix128::from_int(2)).expect("model");
    let p = |x: i64, d: i64| Pedestrian {
        position: Vec2Fix::new(Fix128::from_int(x), Fix128::ZERO),
        velocity: Vec2Fix::ZERO,
        radius_m: Fix128::from_ratio(1, 4),
        mass_kg: Fix128::from_int(60),
        desired_speed_m_s: Fix128::ONE,
        desired_direction: Vec2Fix::new(Fix128::from_int(d), Fix128::ZERO),
        relaxation_time_s: Fix128::from_ratio(1, 2),
    };
    CrowdParticipant::new(
        model,
        vec![p(-1, 1), p(1, -1)],
        Vec::new(),
        NeighborSearch::CellList,
        None,
    )
    .expect("crowd")
}

fn md() -> MdParticipant<LennardJones> {
    let pot = Truncated::new(
        LennardJones::new(Fix128::ONE, Fix128::ONE).expect("lj"),
        Fix128::from_int(2),
        ShiftMode::None,
    )
    .expect("truncated");
    let s = VelocityVerlet::new(
        pot,
        PeriodicBox::cubic(Fix128::from_int(5)).expect("box"),
        vec![Vec3Fix::from_int(1, 1, 1), Vec3Fix::from_int(2, 1, 1)],
        vec![Vec3Fix::ZERO, Vec3Fix::ZERO],
        vec![Fix128::ONE, Fix128::ONE],
    )
    .expect("system");
    MdParticipant::new(s, Fix128::ONE).expect("participant")
}

// ============================================================================
// Rig
// ============================================================================

fn h() -> Fix128 {
    Fix128::from_ratio(1, 200)
}

/// Runs `substeps` substeps of `ps` over `board`; returns all faults.
fn run(
    ps: &mut [Box<dyn Participant>],
    board: &mut FieldBoard,
    substeps: usize,
) -> (ParticipantPlan, Vec<WorldFault>) {
    let plan = ParticipantPlan::new(ps, board).expect("plan");
    let bodies = vec![RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 3, 0),
        Fix128::ONE,
    )];
    let mut frozen = vec![false; ps.len()];
    let mut faults = Vec::new();
    for i in 0..substeps {
        let mut forces = ForceAccumulator::new(bodies.len());
        let time = SubstepTime {
            index: i,
            count: substeps,
            h: h(),
        };
        faults.extend(
            run_substep(ps, &plan, &mut frozen, &bodies, board, &mut forces, time)
                .expect("inputs fit"),
        );
    }
    (plan, faults)
}

fn bytes(p: &dyn Participant) -> Vec<u8> {
    let mut out = Vec::new();
    p.write_state(&mut out);
    out
}

fn repeat(per_substep: &[u32], n: usize) -> Vec<u32> {
    (0..n).flat_map(|_| per_substep.iter().copied()).collect()
}

const X: PortId = PortId::new(0x10);
const Y: PortId = PortId::new(0x11);
const Z: PortId = PortId::new(0x12);
const S: PortId = PortId::new(0x20);

// ============================================================================
// Orders
// ============================================================================

/// The reader of `X` is registered first, its writer third; between them a
/// law participant without ports. Free participants run by registration
/// rank, so the law participant goes first, then the writer, then the
/// reader, then the dynamics.
#[test]
fn a_dependency_against_registration_order_moves_the_reader_behind_its_writer() {
    let log = Log::default();
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Probe::new(10, &[Port::reads(X)], &log)),
        Box::new(thermal()),
        Box::new(Probe::new(20, &[Port::writes(X)], &log)),
        Box::new(md()),
    ];
    let mut board = FieldBoard::new();
    let (plan, faults) = run(&mut ps, &mut board, 3);
    assert!(faults.is_empty());
    assert_eq!(plan.order(), &[1, 2, 0, 3]);
    assert_eq!(*log.lock().expect("log"), repeat(&[20, 10], 3));
    // the law participant ran three times exactly, as if alone
    let mut t = thermal();
    for _ in 0..3 {
        t.update(h().to_f32());
    }
    assert_eq!(bytes(ps[1].as_ref()), bytes(&t));
}

/// A chain `C → B → A` (C writes Z read by B, B writes Y read by A),
/// registered as A, crowd, B, C, phase change.
#[test]
fn a_chain_of_three_runs_from_its_head_with_free_participants_by_rank() {
    let log = Log::default();
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Probe::new(1, &[Port::reads(Y)], &log)),
        Box::new(crowd()),
        Box::new(Probe::new(2, &[Port::reads(Z), Port::writes(Y)], &log)),
        Box::new(Probe::new(3, &[Port::writes(Z)], &log)),
        Box::new(phase_change()),
    ];
    let mut board = FieldBoard::new();
    let (plan, faults) = run(&mut ps, &mut board, 4);
    assert!(faults.is_empty());
    assert_eq!(plan.order(), &[1, 3, 2, 0, 4]);
    assert_eq!(*log.lock().expect("log"), repeat(&[3, 2, 1], 4));
}

/// Two writers of `X` (indices 1 and 3) are not ordered against each other
/// and a free participant between them keeps its rank; the reader (index 0)
/// waits for both. A participant that reads and writes the same port is not
/// ordered against itself.
#[test]
fn two_writers_of_one_port_keep_registration_order_and_the_reader_waits_for_both() {
    let log = Log::default();
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Probe::new(7, &[Port::reads(X)], &log)),
        Box::new(Probe::new(5, &[Port::writes(X)], &log)),
        Box::new(Probe::new(6, &[Port::reads(Y), Port::writes(Y)], &log)),
        Box::new(Probe::new(4, &[Port::writes(X)], &log)),
        Box::new(pressure()),
    ];
    let mut board = FieldBoard::new();
    let (plan, faults) = run(&mut ps, &mut board, 2);
    assert!(faults.is_empty());
    assert_eq!(plan.order(), &[1, 2, 3, 0, 4]);
    assert_eq!(*log.lock().expect("log"), repeat(&[5, 6, 4, 7], 2));
}

/// The seven law participants and a writer/reader pair registered last but
/// in reverse: only the pair is reordered.
#[test]
fn the_seven_laws_keep_registration_order_around_a_reordered_pair() {
    let log = Log::default();
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(fracture()),
        Box::new(erosion()),
        Box::new(md()),
        Box::new(Probe::new(9, &[Port::reads(X)], &log)),
        Box::new(thermal()),
        Box::new(crowd()),
        Box::new(Probe::new(8, &[Port::writes(X)], &log)),
        Box::new(pressure()),
        Box::new(phase_change()),
    ];
    let mut board = FieldBoard::new();
    let (plan, faults) = run(&mut ps, &mut board, 2);
    assert!(faults.is_empty());
    assert_eq!(plan.order(), &[0, 1, 2, 4, 5, 6, 3, 7, 8]);
    assert_eq!(*log.lock().expect("log"), repeat(&[8, 9], 2));
}

// ============================================================================
// An order-dependent result: the first writer that leaves the range
// ============================================================================

fn sum_board() -> FieldBoard {
    let mut b = FieldBoard::new();
    b.declare(S, FieldLayout::PerBody { bodies: 1 }, FieldMode::Sum)
        .expect("declare");
    b
}

/// Three writers of the sum field `S` stage `+¾·2⁶³`, `+¾·2⁶³` and
/// `−¾·2⁶³`. Taken in execution order the partial sums leave the range at
/// the second positive term, and the commit names the participant that
/// added it; in another order the sum stays in range or names another one.
fn sum_scene(order_ports: bool) -> (Vec<WorldFault>, Vec<usize>, FieldBoard) {
    let log = Log::default();
    let big = Fix128::from_raw(3 << 61, 0);
    let ports = |extra: &[Port]| {
        let mut p = vec![Port::writes(S)];
        if order_ports {
            p.extend_from_slice(extra);
        }
        p
    };
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Probe::new(1, &ports(&[Port::reads(X)]), &log).staging(S, big)),
        Box::new(erosion()),
        Box::new(Probe::new(2, &ports(&[Port::writes(X)]), &log).staging(S, big)),
        Box::new(Probe::new(3, &ports(&[]), &log).staging(S, Fix128::ZERO - big)),
    ];
    let mut board = sum_board();
    let (plan, faults) = run(&mut ps, &mut board, 1);
    (faults, plan.order().to_vec(), board)
}

#[test]
fn the_sum_commit_names_the_writer_that_leaves_the_range_in_execution_order() {
    // with X: probe 2 (index 2) runs before probe 1 (index 0)
    let (faults, order, board) = sum_scene(true);
    assert_eq!(order, vec![1, 2, 0, 3]);
    assert_eq!(
        faults,
        vec![WorldFault::FieldOutOfRange { field: S, index: 0 }]
    );
    assert_eq!(
        board.value(S),
        Some(&[Fix128::ZERO][..]),
        "no field changed"
    );

    // without X: registration order, the second positive term is index 2
    let (faults, order, board) = sum_scene(false);
    assert_eq!(order, vec![0, 1, 2, 3]);
    assert_eq!(
        faults,
        vec![WorldFault::FieldOutOfRange { field: S, index: 2 }]
    );
    assert_eq!(board.value(S), Some(&[Fix128::ZERO][..]));
}

/// The negative term placed between the two positive ones by a dependency
/// keeps every partial sum in range: no fault, and the committed value is
/// the exact sum `¾·2⁶³`.
#[test]
fn moving_the_negative_writer_forward_keeps_the_sum_in_range() {
    let log = Log::default();
    let big = Fix128::from_raw(3 << 61, 0);
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Probe::new(1, &[Port::writes(S), Port::reads(X)], &log).staging(S, big)),
        Box::new(Probe::new(2, &[Port::writes(S), Port::reads(X)], &log).staging(S, big)),
        Box::new(
            Probe::new(3, &[Port::writes(S), Port::writes(X)], &log).staging(S, Fix128::ZERO - big),
        ),
    ];
    let mut board = sum_board();
    let (plan, faults) = run(&mut ps, &mut board, 1);
    assert_eq!(plan.order(), &[2, 0, 1]);
    assert!(faults.is_empty(), "{faults:?}");
    assert_eq!(board.value(S), Some(&[big][..]));
}

// ============================================================================
// Loops and missing dependencies
// ============================================================================

/// A loop of three (A writes X read by B, B writes Y read by C, C writes Z
/// read by A) and a participant that only depends on the loop, between law
/// participants: the plan is refused with exactly the loop members.
#[test]
fn a_loop_of_three_is_refused_with_its_members_and_not_its_dependant() {
    let log = Log::default();
    let ps: Vec<Box<dyn Participant>> = vec![
        Box::new(crowd()),
        Box::new(Probe::new(1, &[Port::reads(Z), Port::writes(X)], &log)),
        Box::new(md()),
        Box::new(Probe::new(2, &[Port::reads(X), Port::writes(Y)], &log)),
        Box::new(Probe::new(4, &[Port::reads(Y)], &log)),
        Box::new(Probe::new(3, &[Port::reads(Y), Port::writes(Z)], &log)),
    ];
    let err = ParticipantPlan::new(&ps, &FieldBoard::new()).expect_err("a loop");
    assert_eq!(
        err,
        RegisterError::Order(OrderError::Cycle {
            members: vec![1, 3, 5]
        })
    );
    // the same through execution_order directly
    let ports: Vec<&[Port]> = ps.iter().map(|p| p.ports()).collect();
    assert_eq!(
        execution_order(&ports),
        Err(OrderError::Cycle {
            members: vec![1, 3, 5]
        })
    );
    assert!(log.lock().expect("log").is_empty(), "nothing ran");
}

/// Two separate loops: every member of both is listed, ascending.
#[test]
fn two_loops_list_every_member() {
    let p = |a: &[Port]| a.to_vec();
    let ports = [
        p(&[Port::reads(X), Port::writes(Y)]),
        p(&[Port::reads(Z), Port::writes(S)]),
        p(&[Port::reads(Y), Port::writes(X)]),
        p(&[]),
        p(&[Port::reads(S), Port::writes(Z)]),
    ];
    let views: Vec<&[Port]> = ports.iter().map(Vec::as_slice).collect();
    assert_eq!(
        execution_order(&views),
        Err(OrderError::Cycle {
            members: vec![0, 1, 2, 4]
        })
    );
}

/// A read of a port nobody writes is not an error and adds no constraint.
#[test]
fn a_read_without_a_writer_keeps_registration_order() {
    let log = Log::default();
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Probe::new(1, &[Port::reads(Z)], &log)),
        Box::new(fracture()),
        Box::new(Probe::new(2, &[Port::reads(Z), Port::reads(Y)], &log)),
    ];
    let mut board = FieldBoard::new();
    let (plan, faults) = run(&mut ps, &mut board, 2);
    assert!(faults.is_empty());
    assert_eq!(plan.order(), &[0, 1, 2]);
    assert_eq!(*log.lock().expect("log"), repeat(&[1, 2], 2));
}

/// A committed read of a field that is not on the board, a plain read of a
/// field and a second writer of a replace field are refused at
/// registration, naming the participant.
#[test]
fn missing_or_misdeclared_fields_are_refused_at_registration() {
    let log = Log::default();
    let ps: Vec<Box<dyn Participant>> = vec![
        Box::new(thermal()),
        Box::new(Probe::new(1, &[Port::reads_committed(S)], &log)),
    ];
    assert_eq!(
        ParticipantPlan::new(&ps, &FieldBoard::new()),
        Err(RegisterError::Field(FieldPortError::UnknownField {
            participant: 1,
            field: S,
        }))
    );

    let ps: Vec<Box<dyn Participant>> = vec![
        Box::new(md()),
        Box::new(crowd()),
        Box::new(Probe::new(1, &[Port::reads(S)], &log)),
    ];
    assert_eq!(
        ParticipantPlan::new(&ps, &sum_board()),
        Err(RegisterError::Field(FieldPortError::ReadOfField {
            participant: 2,
            field: S,
        }))
    );

    let mut replace = FieldBoard::new();
    replace
        .declare(S, FieldLayout::PerBody { bodies: 1 }, FieldMode::Replace)
        .expect("declare");
    let ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Probe::new(1, &[Port::writes(S)], &log)),
        Box::new(erosion()),
        Box::new(Probe::new(2, &[Port::writes(S)], &log)),
    ];
    assert_eq!(
        ParticipantPlan::new(&ps, &replace),
        Err(RegisterError::Field(FieldPortError::SecondWriter {
            field: S,
            first: 0,
            second: 2,
        }))
    );
}

/// A frozen (faulted) participant is skipped and the others keep their
/// order: probe 2 is frozen before the run.
#[test]
fn a_frozen_participant_is_skipped_and_the_order_of_the_rest_is_kept() {
    let log = Log::default();
    let mut ps: Vec<Box<dyn Participant>> = vec![
        Box::new(Probe::new(1, &[Port::reads(X)], &log)),
        Box::new(Probe::new(2, &[Port::writes(X)], &log)),
        Box::new(Probe::new(3, &[], &log)),
    ];
    let board = FieldBoard::new();
    let plan = ParticipantPlan::new(&ps, &board).expect("plan");
    let mut board = board;
    let bodies: Vec<RigidBody> = Vec::new();
    let mut frozen = vec![false, true, false];
    for i in 0..3 {
        let mut forces = ForceAccumulator::new(0);
        let time = SubstepTime {
            index: i,
            count: 3,
            h: h(),
        };
        let faults = run_substep(
            &mut ps,
            &plan,
            &mut frozen,
            &bodies,
            &mut board,
            &mut forces,
            time,
        )
        .expect("inputs fit");
        assert!(faults.is_empty());
    }
    assert_eq!(*log.lock().expect("log"), repeat(&[1, 3], 3));
    assert_eq!(bytes(ps[1].as_ref()), 0u64.to_le_bytes().to_vec());
}
