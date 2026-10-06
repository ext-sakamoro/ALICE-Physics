//! Independent checks of the seven laws that take part in the substep loop
//! as participants: `ThermalModifier` (THRM), `PhaseChangeModifier` (PHAS),
//! `PressureModifier` (PRES), `FractureModifier` (FRAC), `ErosionModifier`
//! (EROS), `CrowdParticipant` (CRWD) and `MdParticipant` (MDVV).
//!
//! The laws themselves are not re-derived here. What is checked is the
//! participant layer around them, with scenes that the existing tests
//! (`tests/participant_sdf_modifiers.rs`, `tests/participant_crowd_md.rs`)
//! do not use: asymmetric grid bounds and other resolutions, a varying
//! substep width within a run, all seven participants registered together
//! next to rigid bodies, a diagonal corridor with direct neighbour search
//! and no speed cap, Morse and screened Coulomb particles in a non-cubic box.
//!
//! * bit identity: `n` substeps through [`run_substep`] equal `n` direct calls
//!   of the law (`PhysicsModifier::update(h.to_f32())`, `SocialForce::step`,
//!   `VelocityVerlet::step`) on a twin built by the same constructor, field by
//!   field and as state bytes;
//! * snapshot / restore: `N` substeps, snapshot, `M` more, restore, the same
//!   `M` again give the same bytes after every substep, also when restored
//!   into participants built with other contents;
//! * out of range: a substep the law refuses is reported as the documented
//!   fault, the participant keeps the bytes it had before the step and the
//!   other participants continue as if it were absent.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 builds scene inputs only, never state that is compared.
#![allow(clippy::disallowed_methods)]

use alice_physics::crowd_force::{
    CrowdParticipant, InteractionParams, NeighborSearch, Pedestrian, SocialForce, WallSegment,
    CROWD_PARTICIPANT_KIND,
};
use alice_physics::erosion::{ErosionConfig, ErosionModifier, ErosionType};
use alice_physics::fracture::{Crack, FractureConfig, FractureModifier};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::molecular_dynamics::{
    MdParticipant, PeriodicBox, VelocityVerlet, MD_PARTICIPANT_KIND,
};
use alice_physics::pair_potential::{Morse, ShiftMode, Truncated, Yukawa};
use alice_physics::phase_change::{PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::physics2d::Vec2Fix;
use alice_physics::pressure::{PressureConfig, PressureModifier};
use alice_physics::sim_field::ScalarField3D;
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::solver::RigidBody;
use alice_physics::thermal::{HeatSource, ThermalConfig, ThermalModifier};
use alice_physics::world_participant::{
    run_substep, FieldBoard, ForceAccumulator, Participant, ParticipantFault, ParticipantKind,
    ParticipantPlan, SubstepTime, WorldFault,
};

// ============================================================================
// Harness: the part of the world step that runs participants
// ============================================================================

/// Participants, plan, frozen flags and three rigid bodies; one call of
/// [`Rig::substep`] is what the world does once per substep.
struct Rig {
    ps: Vec<Box<dyn Participant>>,
    plan: ParticipantPlan,
    frozen: Vec<bool>,
    bodies: Vec<RigidBody>,
    board: FieldBoard,
    calls: usize,
}

impl Rig {
    fn new(ps: Vec<Box<dyn Participant>>) -> Self {
        let board = FieldBoard::new();
        let plan = ParticipantPlan::new(&ps, &board).expect("no ports, no fields");
        let frozen = vec![false; ps.len()];
        Self {
            ps,
            plan,
            frozen,
            bodies: vec![
                RigidBody::new_dynamic(Vec3Fix::from_int(-4, 2, 1), Fix128::from_int(2)),
                RigidBody::new_dynamic(Vec3Fix::from_int(3, -1, 7), Fix128::ONE),
                RigidBody::new_static(Vec3Fix::from_int(0, -9, 0)),
            ],
            board,
            calls: 0,
        }
    }

    /// One substep of width `h`; the participants must stage no force.
    fn substep(&mut self, h: Fix128) -> Vec<WorldFault> {
        let mut forces = ForceAccumulator::new(self.bodies.len());
        let time = SubstepTime {
            index: self.calls % 3,
            count: 3,
            h,
        };
        self.calls += 1;
        let faults = run_substep(
            &mut self.ps,
            &self.plan,
            &mut self.frozen,
            &self.bodies,
            &mut self.board,
            &mut forces,
            time,
        )
        .expect("inputs fit");
        for b in 0..self.bodies.len() {
            assert_eq!(forces.force(b), Some(Vec3Fix::ZERO), "force on body {b}");
            assert_eq!(forces.torque(b), Some(Vec3Fix::ZERO), "torque on body {b}");
        }
        faults
    }

    fn states(&self) -> Vec<Vec<u8>> {
        self.ps.iter().map(|p| bytes(p.as_ref())).collect()
    }

    /// Check every payload first, then read every one (a restore never reads
    /// a payload before all of them are accepted).
    fn restore(&mut self, snap: &[Vec<u8>]) {
        assert_eq!(snap.len(), self.ps.len());
        for (p, b) in self.ps.iter().zip(snap) {
            p.check_state(b).expect("own payload accepted");
        }
        for (p, b) in self.ps.iter_mut().zip(snap) {
            p.read_state(b);
        }
    }

    fn kinds(&self) -> Vec<ParticipantKind> {
        self.ps.iter().map(|p| p.kind()).collect()
    }
}

fn bytes(p: &dyn Participant) -> Vec<u8> {
    let mut out = Vec::new();
    p.write_state(&mut out);
    out
}

/// A varying substep width: the world may change `dt` between frames.
fn widths(n: usize) -> Vec<Fix128> {
    let cycle = [
        Fix128::from_ratio(1, 90),
        Fix128::from_ratio(1, 250),
        Fix128::from_ratio(3, 400),
        Fix128::from_ratio(1, 1000),
        Fix128::from_ratio(1, 60),
        Fix128::from_ratio(7, 1500),
    ];
    (0..n).map(|i| cycle[i % cycle.len()]).collect()
}

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Every public field of a grid, as bits.
fn grid(f: &ScalarField3D) -> (Vec<u32>, [usize; 3], [u32; 6]) {
    (
        f32_bits(&f.data),
        [f.nx, f.ny, f.nz],
        [
            f.min.0.to_bits(),
            f.min.1.to_bits(),
            f.min.2.to_bits(),
            f.max.0.to_bits(),
            f.max.1.to_bits(),
            f.max.2.to_bits(),
        ],
    )
}

// ============================================================================
// Scenes (none of them is used by the existing participant tests)
// ============================================================================

const LO: (f32, f32, f32) = (-1.0, -3.0, -0.5);
const HI: (f32, f32, f32) = (3.0, 1.0, 2.5);

fn thermal() -> ThermalModifier {
    let cfg = ThermalConfig {
        diffusion_rate: 0.7,
        cooling_rate: 0.04,
        melt_temperature: 35.0,
        melt_rate: 0.6,
        freeze_temperature: 12.0,
        freeze_rate: 0.25,
        ..ThermalConfig::default()
    };
    let mut m = ThermalModifier::new(cfg, 6, LO, HI);
    m.add_heat_point(2.0, -2.0, 1.0, 2500.0, 1.2);
    m.add_heat_point(-0.5, 0.5, 0.0, 700.0, 0.6);
    m.heat_sources.push(HeatSource::Volume {
        min: (1.0, -1.0, 0.0),
        max: (3.0, 1.0, 2.5),
        power: 120.0,
    });
    m.apply_heat_at(0.0, -2.5, 2.0, 80.0, 0.9);
    m
}

fn phase_change() -> PhaseChangeModifier {
    let cfg = PhaseChangeConfig {
        melt_temperature: 30.0,
        boil_temperature: 85.0,
        diffusion_rate: 0.6,
        cooling_rate: 0.02,
        ..PhaseChangeConfig::default()
    };
    let mut m = PhaseChangeModifier::new(cfg, 7, LO, HI);
    m.apply_heat_at(2.5, 0.5, 2.0, 1500.0, 1.4);
    m.apply_heat_at(-0.8, -2.8, -0.3, 300.0, 0.8);
    m
}

fn pressure() -> PressureModifier {
    let cfg = PressureConfig {
        diffusion_rate: 0.4,
        decay_rate: 0.15,
        yield_threshold: 3.0,
        internal_pressure: 0.7,
        ..PressureConfig::default()
    };
    let mut m = PressureModifier::new(cfg, 6, LO, HI);
    m.apply_pressure_at(2.0, 0.0, 1.0, 250.0, 1.1);
    m.apply_impact(-0.5, -2.0, 2.0, 9.0, 0.7);
    m.apply_impact(1.0, -1.0, 0.0, 3.0, 1.6);
    m
}

fn fracture() -> FractureModifier {
    let cfg = FractureConfig {
        fracture_toughness: 12.0,
        max_cracks: 6,
        propagation_speed: 1.5,
        ..FractureConfig::default()
    };
    let mut m = FractureModifier::new(cfg, 6, LO, HI);
    m.apply_stress_at(1.5, -1.5, 1.0, 420.0, 1.3);
    m.apply_stress_at(-0.5, 0.5, 2.0, 90.0, 0.6);
    m.cracks.push(Crack {
        start: (1.0, -1.0, 1.0),
        end: (1.2, -1.1, 1.0),
        direction: (0.8, -0.6, 0.0),
        length: 0.22,
        active: true,
    });
    m
}

fn erosion() -> ErosionModifier {
    let cfg = ErosionConfig {
        erosion_type: ErosionType::Water,
        rate: 0.3,
        flow_direction: (0.0, -1.0, 0.3),
        flow_speed: 2.0,
        ..ErosionConfig::default()
    };
    let mut m = ErosionModifier::new(cfg, 7, LO, HI);
    m.set_exposure_at(2.0, 0.5, 1.5, 0.9, 1.2);
    m.set_exposure_at(-0.5, -2.5, 0.0, 0.4, 0.7);
    m
}

fn crowd_params() -> InteractionParams {
    InteractionParams {
        strength_n: Fix128::from_int(1500),
        range_m: Fix128::from_ratio(1, 10),
        body_stiffness: Fix128::from_int(90_000),
        sliding_friction: Fix128::from_int(150_000),
    }
}

fn crowd_model() -> SocialForce {
    SocialForce::new(
        crowd_params(),
        crowd_params(),
        Fix128::from_ratio(3, 10),
        Fix128::from_int(3),
    )
    .expect("valid model")
}

/// A corridor at 45° (walls `y = x ± 2`).
fn crowd_walls() -> Vec<WallSegment> {
    let p = |x: i64, y: i64| Vec2Fix::new(Fix128::from_int(x), Fix128::from_int(y));
    vec![
        WallSegment {
            start: p(-8, -6),
            end: p(8, 10),
        },
        WallSegment {
            start: p(-8, -10),
            end: p(8, 6),
        },
    ]
}

fn ped(x: f64, y: f64, dir: (i64, i64), m: i64, v0: f64) -> Pedestrian {
    Pedestrian {
        position: Vec2Fix::new(fx(x), fx(y)),
        velocity: Vec2Fix::new(fx(0.1 * x.signum()), Fix128::ZERO),
        radius_m: fx(0.25),
        mass_kg: Fix128::from_int(m),
        desired_speed_m_s: fx(v0),
        desired_direction: Vec2Fix::new(Fix128::from_int(dir.0), Fix128::from_int(dir.1)),
        relaxation_time_s: fx(0.4),
    }
}

fn crowd_people() -> Vec<Pedestrian> {
    vec![
        ped(-3.0, -2.6, (1, 1), 70, 1.4),
        ped(-3.4, -3.5, (1, 1), 85, 1.1),
        ped(-2.0, -1.0, (1, 1), 60, 1.6),
        ped(-4.1, -4.4, (1, 1), 75, 1.2),
        ped(3.0, 3.2, (-1, -1), 90, 1.3),
        ped(2.6, 2.1, (-1, -1), 65, 1.5),
        ped(3.9, 4.6, (-1, -1), 80, 1.0),
        ped(0.1, 0.2, (0, 0), 72, 0.0),
    ]
}

fn crowd() -> CrowdParticipant {
    CrowdParticipant::new(
        crowd_model(),
        crowd_people(),
        crowd_walls(),
        NeighborSearch::Direct,
        None,
    )
    .expect("valid crowd")
}

fn md_box() -> PeriodicBox {
    PeriodicBox::new(Vec3Fix::new(fx(5.0), fx(6.5), fx(4.25))).expect("box")
}

fn morse() -> Truncated<Morse> {
    Truncated::new(
        Morse::new(fx(1.5), fx(2.0), fx(1.1)).expect("morse"),
        fx(2.0),
        ShiftMode::EnergyShift,
    )
    .expect("truncated")
}

fn md_system() -> VelocityVerlet<Morse> {
    VelocityVerlet::new(
        morse(),
        md_box(),
        vec![
            Vec3Fix::new(fx(0.5), fx(0.5), fx(0.5)),
            Vec3Fix::new(fx(1.7), fx(0.6), fx(0.4)),
            Vec3Fix::new(fx(1.1), fx(1.5), fx(0.6)),
            Vec3Fix::new(fx(3.6), fx(4.9), fx(3.7)),
            Vec3Fix::new(fx(4.7), fx(5.8), fx(0.2)),
        ],
        vec![
            Vec3Fix::new(fx(0.3), fx(-0.1), fx(0.0)),
            Vec3Fix::new(fx(-0.2), fx(0.25), fx(0.1)),
            Vec3Fix::new(fx(0.0), fx(-0.3), fx(-0.15)),
            Vec3Fix::new(fx(0.4), fx(0.0), fx(0.2)),
            Vec3Fix::new(fx(-0.35), fx(0.1), fx(-0.05)),
        ],
        vec![fx(1.0), fx(2.0), fx(0.5), fx(3.0), fx(1.25)],
    )
    .expect("system")
}

fn md() -> MdParticipant<Morse> {
    MdParticipant::new(md_system(), fx(0.75)).expect("participant")
}

/// The seven, registered together in an order that is not the order of the
/// kind table.
fn all_seven() -> Vec<Box<dyn Participant>> {
    vec![
        Box::new(md()),
        Box::new(erosion()),
        Box::new(thermal()),
        Box::new(crowd()),
        Box::new(fracture()),
        Box::new(pressure()),
        Box::new(phase_change()),
    ]
}

fn code(tag: &[u8; 4]) -> ParticipantKind {
    ParticipantKind::new(u32::from_be_bytes(*tag))
}

#[test]
fn the_rig_holds_the_seven_documented_kinds() {
    let rig = Rig::new(all_seven());
    assert_eq!(
        rig.kinds(),
        vec![
            code(b"MDVV"),
            code(b"EROS"),
            code(b"THRM"),
            code(b"CRWD"),
            code(b"FRAC"),
            code(b"PRES"),
            code(b"PHAS"),
        ]
    );
    assert_eq!(rig.plan.order(), &[0, 1, 2, 3, 4, 5, 6]);
}

// ============================================================================
// 1. Bit identity: participant path against direct calls of the law
// ============================================================================

const RUN: usize = 24;

/// Drives the seven together and hands back the rig after `n` substeps of
/// [`widths`].
fn driven(n: usize) -> Rig {
    let mut rig = Rig::new(all_seven());
    for h in widths(n) {
        let faults = rig.substep(h);
        assert!(faults.is_empty(), "unexpected faults {faults:?}");
    }
    rig
}

#[test]
fn thermal_through_the_rig_equals_direct_updates() {
    let rig = driven(RUN);
    let mut direct = thermal();
    for h in widths(RUN) {
        direct.update(h.to_f32());
    }
    let mut back = ThermalModifier::new(ThermalConfig::default(), 1, LO, LO);
    let b = bytes(rig.ps[2].as_ref());
    back.check_state(&b).expect("own payload");
    back.read_state(&b);
    assert_eq!(grid(&back.temperature), grid(&direct.temperature));
    assert_eq!(grid(&back.melt_accumulator), grid(&direct.melt_accumulator));
    assert_eq!(back.heat_sources, direct.heat_sources);
    assert_eq!(back.config, direct.config);
    assert_eq!(back.enabled, direct.enabled);
    assert_eq!(b, bytes(&direct));
    assert_ne!(b, bytes(&thermal()), "the run must move the state");
}

#[test]
fn phase_change_through_the_rig_equals_direct_updates() {
    let rig = driven(RUN);
    let mut direct = phase_change();
    for h in widths(RUN) {
        direct.update(h.to_f32());
    }
    let mut back = PhaseChangeModifier::new(PhaseChangeConfig::default(), 1, LO, LO);
    let b = bytes(rig.ps[6].as_ref());
    back.check_state(&b).expect("own payload");
    back.read_state(&b);
    assert_eq!(grid(&back.temperature), grid(&direct.temperature));
    assert_eq!(grid(&back.phase), grid(&direct.phase));
    assert_eq!(grid(&back.latent_heat), grid(&direct.latent_heat));
    assert_eq!(grid(&back.sdf_offset), grid(&direct.sdf_offset));
    assert_eq!(back.config, direct.config);
    assert_eq!(b, bytes(&direct));
    assert_ne!(b, bytes(&phase_change()), "the run must move the state");
}

#[test]
fn pressure_through_the_rig_equals_direct_updates() {
    let rig = driven(RUN);
    let mut direct = pressure();
    for h in widths(RUN) {
        direct.update(h.to_f32());
    }
    let mut back = PressureModifier::new(PressureConfig::default(), 1, LO, LO);
    let b = bytes(rig.ps[5].as_ref());
    back.check_state(&b).expect("own payload");
    back.read_state(&b);
    assert_eq!(grid(&back.pressure), grid(&direct.pressure));
    assert_eq!(grid(&back.deformation), grid(&direct.deformation));
    assert_eq!(back.config, direct.config);
    assert_eq!(b, bytes(&direct));
    assert_ne!(b, bytes(&pressure()), "the run must move the state");
}

#[test]
fn fracture_through_the_rig_equals_direct_updates() {
    let rig = driven(RUN);
    let mut direct = fracture();
    for h in widths(RUN) {
        direct.update(h.to_f32());
    }
    let mut back = FractureModifier::new(FractureConfig::default(), 1, LO, LO);
    let b = bytes(rig.ps[4].as_ref());
    back.check_state(&b).expect("own payload");
    back.read_state(&b);
    assert_eq!(grid(&back.stress), grid(&direct.stress));
    assert_eq!(back.cracks.len(), direct.cracks.len());
    for (a, d) in back.cracks.iter().zip(&direct.cracks) {
        let bits = |c: &Crack| {
            [
                c.start.0,
                c.start.1,
                c.start.2,
                c.end.0,
                c.end.1,
                c.end.2,
                c.direction.0,
                c.direction.1,
                c.direction.2,
                c.length,
            ]
            .map(f32::to_bits)
        };
        assert_eq!(bits(a), bits(d));
        assert_eq!(a.active, d.active);
    }
    assert_eq!(b, bytes(&direct));
    assert_ne!(b, bytes(&fracture()), "the run must move the state");
}

#[test]
fn erosion_through_the_rig_equals_direct_updates() {
    let rig = driven(RUN);
    let mut direct = erosion();
    for h in widths(RUN) {
        direct.update(h.to_f32());
    }
    let mut back = ErosionModifier::new(ErosionConfig::default(), 1, LO, LO);
    let b = bytes(rig.ps[1].as_ref());
    back.check_state(&b).expect("own payload");
    back.read_state(&b);
    assert_eq!(grid(&back.erosion_depth), grid(&direct.erosion_depth));
    assert_eq!(grid(&back.exposure), grid(&direct.exposure));
    assert_eq!(b, bytes(&direct));
    assert_ne!(b, bytes(&erosion()), "the run must move the state");
}

#[test]
fn crowd_through_the_rig_equals_direct_social_force_steps() {
    let rig = driven(RUN);
    let mut people = crowd_people();
    let model = crowd_model();
    let walls = crowd_walls();
    for h in widths(RUN) {
        model
            .step(&mut people, &walls, h, NeighborSearch::Direct, None)
            .expect("valid step");
    }
    let mut back = crowd();
    let b = bytes(rig.ps[3].as_ref());
    back.check_state(&b).expect("own payload");
    back.read_state(&b);
    assert_eq!(back.pedestrians(), people.as_slice());
    let mut reference = crowd();
    reference.pedestrians_mut().copy_from_slice(&people);
    assert_eq!(b, bytes(&reference));
    assert_ne!(b, bytes(&crowd()), "the run must move the state");
}

#[test]
fn md_through_the_rig_equals_direct_verlet_steps() {
    let rig = driven(RUN);
    let mut system = md_system();
    for h in widths(RUN) {
        system.step(h).expect("valid step");
    }
    let mut back = md();
    let b = bytes(rig.ps[0].as_ref());
    back.check_state(&b).expect("own payload");
    back.read_state(&b);
    let s = back.system();
    assert_eq!(s.positions(), system.positions());
    assert_eq!(s.velocities(), system.velocities());
    assert_eq!(s.forces(), system.forces());
    assert_eq!(s.masses(), system.masses());
    assert_eq!(s.potential_energy(), system.potential_energy());
    let reference = MdParticipant::new(system, fx(0.75)).expect("participant");
    assert_eq!(b, bytes(&reference));
    assert_ne!(b, bytes(&md()), "the run must move the state");
}

// ============================================================================
// 2. Snapshot / restore
// ============================================================================

/// Bytes of every participant after each of the substeps `hs`.
fn trace(rig: &mut Rig, hs: &[Fix128]) -> Vec<Vec<Vec<u8>>> {
    hs.iter()
        .map(|&h| {
            assert!(rig.substep(h).is_empty());
            rig.states()
        })
        .collect()
}

fn column(t: &[Vec<Vec<u8>>], i: usize) -> Vec<Vec<u8>> {
    t.iter().map(|s| s[i].clone()).collect()
}

#[test]
fn snapshot_then_restore_replays_the_same_substeps_per_participant() {
    let hs = widths(40);
    let (first, second) = hs.split_at(13);
    let mut rig = Rig::new(all_seven());
    trace(&mut rig, first);
    let snap = rig.states();
    let a = trace(&mut rig, second);
    rig.restore(&snap);
    assert_eq!(rig.states(), snap, "restore gives the snapshot bytes back");
    let b = trace(&mut rig, second);
    for (i, kind) in rig.kinds().into_iter().enumerate() {
        let (ca, cb) = (column(&a, i), column(&b, i));
        assert_eq!(ca, cb, "participant {i} ({kind:?}) diverged after restore");
        assert_ne!(ca[0], ca[ca.len() - 1], "participant {i} did not move");
    }
}

/// Participants built with other contents (other resolution and
/// configuration for the modifiers, other pedestrians for the crowd, other
/// particles for the dynamics) take the snapshot and continue like the
/// original world. The dynamics keeps the same potential: its type and
/// parameters are documented as not part of the payload.
fn others() -> Vec<Box<dyn Participant>> {
    let other_md = {
        let s = VelocityVerlet::new(
            morse(),
            md_box(),
            vec![Vec3Fix::new(fx(2.0), fx(2.0), fx(2.0))],
            vec![Vec3Fix::ZERO],
            vec![fx(9.0)],
        )
        .expect("system");
        MdParticipant::new(s, fx(0.75)).expect("participant")
    };
    let other_crowd = CrowdParticipant::new(
        crowd_model(),
        vec![ped(0.0, 0.0, (1, 0), 50, 0.5)],
        crowd_walls(),
        NeighborSearch::Direct,
        None,
    )
    .expect("crowd");
    let tiny = (0.0, 0.0, 0.0);
    vec![
        Box::new(other_md),
        Box::new(ErosionModifier::new(ErosionConfig::default(), 3, tiny, HI)),
        Box::new(ThermalModifier::new(ThermalConfig::default(), 2, tiny, HI)),
        Box::new(other_crowd),
        Box::new(FractureModifier::new(
            FractureConfig::default(),
            4,
            tiny,
            HI,
        )),
        Box::new(PressureModifier::new(
            PressureConfig::default(),
            2,
            tiny,
            HI,
        )),
        Box::new(PhaseChangeModifier::new(
            PhaseChangeConfig::default(),
            3,
            tiny,
            HI,
        )),
    ]
}

#[test]
fn a_snapshot_restored_into_a_fresh_world_continues_bit_identically() {
    let hs = widths(37);
    let (first, second) = hs.split_at(11);
    let mut rig = Rig::new(all_seven());
    trace(&mut rig, first);
    let snap = rig.states();
    let a = trace(&mut rig, second);

    let mut fresh = Rig::new(others());
    assert_ne!(fresh.states(), snap, "the fresh world must start elsewhere");
    fresh.restore(&snap);
    assert_eq!(fresh.states(), snap);
    let b = trace(&mut fresh, second);
    for i in 0..7 {
        assert_eq!(
            column(&a, i),
            column(&b, i),
            "participant {i} differs in the fresh world"
        );
    }
}

/// The restore is not merely a copy of bytes: a participant restored from a
/// snapshot taken at substep `N` and the participant that never stopped give
/// the same observations too.
#[test]
fn observations_after_a_restore_equal_those_of_the_uninterrupted_run() {
    let hs = widths(20);
    let (first, second) = hs.split_at(9);
    let mut rig = Rig::new(all_seven());
    trace(&mut rig, first);
    let snap = rig.states();
    trace(&mut rig, second);
    let observed = |r: &Rig| -> Vec<Vec<(u32, Fix128)>> {
        r.ps.iter()
            .map(|p| {
                let mut s = alice_physics::world_participant::ObservationSink::new();
                p.observe(&mut s);
                s.values().to_vec()
            })
            .collect()
    };
    let a = observed(&rig);
    let mut fresh = Rig::new(others());
    fresh.restore(&snap);
    trace(&mut fresh, second);
    assert_eq!(observed(&fresh), a);
}

// ============================================================================
// 4. Out of range
// ============================================================================

/// Runs the seven where participant `bad` is made to fail before substep
/// `at`; returns the faults and the bytes of `bad` just before that substep
/// and at the end, and the bytes of every other participant at the end.
fn run_with_a_bad_participant(
    bad: usize,
    break_it: impl FnOnce(&mut Box<dyn Participant>),
    at: usize,
) -> (Vec<WorldFault>, Vec<u8>, Vec<u8>, Vec<Vec<u8>>) {
    let hs = widths(18);
    let mut rig = Rig::new(all_seven());
    let mut faults = Vec::new();
    let mut before = Vec::new();
    let mut break_it = Some(break_it);
    for (k, &h) in hs.iter().enumerate() {
        if k == at {
            (break_it.take().expect("once"))(&mut rig.ps[bad]);
            before = bytes(rig.ps[bad].as_ref());
        }
        faults.extend(rig.substep(h));
    }
    let after = bytes(rig.ps[bad].as_ref());
    (faults, before, after, rig.states())
}

/// The other six end where an uninterrupted run of the seven ends.
fn others_unaffected(bad: usize, end: &[Vec<u8>]) {
    let clean = driven(18).states();
    for i in (0..7).filter(|&i| i != bad) {
        assert_eq!(end[i], clean[i], "participant {i} was affected by {bad}");
    }
}

fn pedestrian_change(f: impl Fn(&mut Pedestrian)) -> impl FnOnce(&mut Box<dyn Participant>) {
    move |p| {
        let blob = bytes(p.as_ref());
        let mut c = crowd();
        c.read_state(&blob);
        f(&mut c.pedestrians_mut()[5]);
        *p = Box::new(c);
    }
}

#[test]
fn a_crowd_with_a_negative_mass_faults_and_keeps_its_bytes() {
    let (faults, before, after, end) =
        run_with_a_bad_participant(3, pedestrian_change(|p| p.mass_kg = fx(-80.0)), 7);
    assert_eq!(
        faults,
        vec![WorldFault::Participant {
            index: 3,
            kind: CROWD_PARTICIPANT_KIND,
            fault: ParticipantFault::InvalidState,
        }]
    );
    assert_eq!(after, before, "the crowd changed although it faulted");
    others_unaffected(3, &end);
}

#[test]
fn a_crowd_with_a_negative_radius_or_speed_faults_and_keeps_its_bytes() {
    for f in [
        (|p: &mut Pedestrian| p.radius_m = fx(-0.01)) as fn(&mut Pedestrian),
        |p: &mut Pedestrian| p.desired_speed_m_s = fx(-1.0),
        |p: &mut Pedestrian| p.relaxation_time_s = fx(-0.4),
    ] {
        let (faults, before, after, end) = run_with_a_bad_participant(3, pedestrian_change(f), 0);
        assert_eq!(faults.len(), 1, "{faults:?}");
        assert!(matches!(
            faults[0],
            WorldFault::Participant {
                index: 3,
                fault: ParticipantFault::InvalidState,
                ..
            }
        ));
        assert_eq!(after, before);
        others_unaffected(3, &end);
    }
}

/// Two screened Coulomb particles that are brought to the separation
/// `final_gap` by one drift of width `h` (the half kicks of the force at the
/// start separation `0.8` enter the drift, so the initial velocity is
/// chosen after that force is known), next to a spectator particle.
fn yukawa_approach(h: Fix128, final_gap: Fix128) -> MdParticipant<Yukawa> {
    let gap = fx(0.8);
    let pot = Truncated::new(
        // k q q = 8.99e9 · 1e-8 ≈ 89.9
        Yukawa::new(fx(1e-4), fx(1e-4), fx(1.5)).expect("yukawa"),
        fx(2.0),
        ShiftMode::ForceShift,
    )
    .expect("truncated");
    let make = |v: Fix128| {
        VelocityVerlet::new(
            pot,
            PeriodicBox::cubic(Fix128::from_int(5)).expect("box"),
            vec![
                Vec3Fix::new(fx(1.0), fx(2.0), fx(2.0)),
                Vec3Fix::new(fx(1.0) + gap, fx(2.0), fx(2.0)),
                Vec3Fix::new(fx(4.0), fx(4.0), fx(4.0)),
            ],
            vec![
                Vec3Fix::new(v, Fix128::ZERO, Fix128::ZERO),
                Vec3Fix::ZERO,
                Vec3Fix::new(fx(0.1), fx(0.2), fx(0.3)),
            ],
            vec![Fix128::ONE; 3],
        )
        .expect("system")
    };
    let kick = make(Fix128::ZERO).forces()[0].x * h.half();
    // particle 1 receives the opposite half kick and moves away by it
    let v = (gap - final_gap) / h - kick - kick;
    MdParticipant::new(make(v), Fix128::ONE).expect("participant")
}

/// The pair comes to `r = 2⁻³⁰` in one drift: `k q q/r² ≈ 2^66.5` leaves the
/// `Fix128` range, the potential reports `Overflow` and the participant
/// `OutOfRange`; the state before the step is kept, the participant is not
/// called again and the two other participants continue as if alone.
#[test]
fn an_md_yukawa_overflow_is_out_of_range_and_keeps_the_bytes() {
    let h = Fix128::from_ratio(1, 64);
    let p = yukawa_approach(h, Fix128::from_raw(0, 1 << 34));
    let before = bytes(&p);
    let mut rig = Rig::new(vec![Box::new(thermal()), Box::new(p), Box::new(crowd())]);
    let mut faults = Vec::new();
    for _ in 0..4 {
        faults.extend(rig.substep(h));
    }
    assert_eq!(
        faults,
        vec![WorldFault::Participant {
            index: 1,
            kind: MD_PARTICIPANT_KIND,
            fault: ParticipantFault::OutOfRange,
        }]
    );
    assert_eq!(bytes(rig.ps[1].as_ref()), before);
    assert!(rig.frozen[1] && !rig.frozen[0] && !rig.frozen[2]);
    // the thermal and crowd participants went on as if alone
    let mut t = thermal();
    let mut c = crowd();
    for _ in 0..4 {
        t.update(h.to_f32());
    }
    let model = crowd_model();
    let mut people = crowd_people();
    for _ in 0..4 {
        model
            .step(&mut people, &crowd_walls(), h, NeighborSearch::Direct, None)
            .expect("step");
    }
    c.pedestrians_mut().copy_from_slice(&people);
    assert_eq!(bytes(rig.ps[0].as_ref()), bytes(&t));
    assert_eq!(bytes(rig.ps[2].as_ref()), bytes(&c));
}

/// The pair comes to `r = 2⁻⁴⁰`, where `k q q/r²` is out of range by far
/// more than at `2⁻³⁰`; `|d|² = 2⁻⁸⁰` rounds to zero, which the participant
/// reports as out of range (not as a coincident pair, `InvalidState`).
#[test]
fn an_md_pair_closer_than_the_length_resolution_is_out_of_range() {
    let h = Fix128::from_ratio(1, 64);
    let p = yukawa_approach(h, Fix128::from_raw(0, 1 << 24));
    let before = bytes(&p);
    let mut rig = Rig::new(vec![Box::new(p)]);
    let faults = rig.substep(h);
    assert_eq!(bytes(rig.ps[0].as_ref()), before);
    assert_eq!(
        faults,
        vec![WorldFault::Participant {
            index: 0,
            kind: MD_PARTICIPANT_KIND,
            fault: ParticipantFault::OutOfRange,
        }]
    );
}

// ============================================================================
// Values out of the range: a fault, never a wrapped value
// ============================================================================

/// A velocity-Verlet kick `v + F·h/(2m)` whose product leaves the `Fix128`
/// range is a fault, not a wrapped velocity.
#[test]
fn an_md_kick_out_of_range_is_a_fault() {
    // two Morse particles at r = 1.1 (the minimum) would feel no force; put
    // them at r = 0.6 where the Morse force is 2·D·a·e(e−1) ≈ 28 (e = e¹),
    // and give particle 0 a mass of 2^-62 so that h/(2m) = 2^56·h
    let system = VelocityVerlet::new(
        morse(),
        md_box(),
        vec![
            Vec3Fix::new(fx(1.0), fx(1.0), fx(1.0)),
            Vec3Fix::new(fx(1.6), fx(1.0), fx(1.0)),
        ],
        vec![Vec3Fix::ZERO, Vec3Fix::ZERO],
        vec![Fix128::from_raw(0, 4), Fix128::ONE],
    )
    .expect("system");
    let p = MdParticipant::new(system, Fix128::ONE).expect("participant");
    let before = bytes(&p);
    let mut rig = Rig::new(vec![Box::new(p)]);
    let faults = rig.substep(Fix128::ONE);
    assert_eq!(
        faults,
        vec![WorldFault::Participant {
            index: 0,
            kind: MD_PARTICIPANT_KIND,
            fault: ParticipantFault::OutOfRange,
        }],
        "a kick of |F|·h/(2m) ≈ 28·2^61 was accepted, v0 after the step = {:?}",
        {
            let mut back = MdParticipant::new(md_system(), Fix128::ONE).expect("participant");
            back.read_state(&bytes(rig.ps[0].as_ref()));
            back.system().velocities()[0]
        }
    );
    assert_eq!(bytes(rig.ps[0].as_ref()), before);
}

/// A pedestrian whose driving term `m (v0 ê − v)/τ` leaves the `Fix128`
/// range is a fault, not a wrapped velocity.
#[test]
fn a_crowd_velocity_out_of_range_is_a_fault() {
    let mut p = crowd_people()[0];
    p.velocity = Vec2Fix::new(Fix128::from_int(1 << 61), Fix128::ZERO);
    p.desired_direction = Vec2Fix::new(Fix128::ONE, Fix128::ZERO);
    p.relaxation_time_s = Fix128::from_raw(0, 1 << 40);
    let c = CrowdParticipant::new(
        crowd_model(),
        vec![p],
        Vec::new(),
        NeighborSearch::Direct,
        None,
    )
    .expect("valid crowd");
    let before = bytes(&c);
    let mut rig = Rig::new(vec![Box::new(c)]);
    let faults = rig.substep(Fix128::from_ratio(1, 100));
    assert_eq!(
        faults,
        vec![WorldFault::Participant {
            index: 0,
            kind: CROWD_PARTICIPANT_KIND,
            fault: ParticipantFault::OutOfRange,
        }],
        "a driving term m·v/τ ≈ 70·2^85 was accepted, v after the step = {:?}",
        {
            let mut back = crowd();
            back.read_state(&bytes(rig.ps[0].as_ref()));
            back.pedestrians()[0].velocity
        }
    );
    assert_eq!(bytes(rig.ps[0].as_ref()), before);
}
