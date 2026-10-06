//! The five SDF modifiers (`thermal`, `phase_change`, `pressure`, `fracture`,
//! `erosion`) as world participants ([`alice_physics::world_participant`]).
//!
//! What is checked:
//!
//! * (a) each modifier passes the participant contract harness (copied
//!   unchanged from `tests/world_participant_conformance.rs`);
//! * (b) `n` substeps of width `h` through [`run_substep`] leave the same state
//!   bytes as `n` direct calls of `PhysicsModifier::update(h.to_f32())`;
//! * (c) closed forms through the participant path: a thermal point source
//!   (smoothstep splat, diffusion and cooling off), explicit diffusion of a
//!   hot node against an independent reference stencil, and the
//!   enthalpy-method plateau of `phase_change`;
//! * (d) a state round trip in the middle of a run, then continuing, gives
//!   the bytes of an uninterrupted run, also into a participant built with
//!   another resolution and configuration;
//! * (e) `check_state` refuses truncated, extended, wrong-version and
//!   invalid payloads and leaves the participant unchanged;
//! * the allocated kinds are distinct from each other and from the kinds the
//!   conformance tests and the contract example use.
//!
//! Oracles are computed by the test (direct `update` calls, closed forms, a
//! stencil written here), never read back from the participant under test.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::erosion::{ErosionConfig, ErosionModifier, ErosionType};
use alice_physics::fracture::{Crack, FractureConfig, FractureModifier};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::pressure::{PressureConfig, PressureModifier};
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::solver::RigidBody;
use alice_physics::thermal::{ThermalConfig, ThermalModifier};
use alice_physics::world_participant::{
    run_substep, FieldBoard, ForceAccumulator, ObservationSink, Participant, ParticipantKind,
    ParticipantPlan, RegisterError, StateError, StepRule, SubstepCtx, SubstepTime,
};

// ============================================================================
// Participant-layer harness (copied unchanged from
// tests/world_participant_conformance.rs)
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

// ============================================================================
// Scenes: every modifier with non-trivial state
// ============================================================================

const LO: (f32, f32, f32) = (-2.0, -2.0, -2.0);
const HI: (f32, f32, f32) = (2.0, 2.0, 2.0);

fn h_sub() -> Fix128 {
    Fix128::from_ratio(1, 240)
}

fn thermal() -> ThermalModifier {
    let cfg = ThermalConfig {
        melt_temperature: 60.0,
        droop_strength: 0.2,
        ..ThermalConfig::default()
    };
    let mut m = ThermalModifier::new(cfg, 5, LO, HI);
    m.add_heat_point(0.0, 0.0, 0.0, 4000.0, 1.5);
    m.heat_sources
        .push(alice_physics::thermal::HeatSource::Volume {
            min: (-2.0, -2.0, -2.0),
            max: (0.0, 0.0, 0.0),
            power: 300.0,
        });
    m.apply_heat_at(1.0, 0.0, 0.0, 50.0, 1.0);
    m
}

fn phase_change() -> PhaseChangeModifier {
    let mut m = PhaseChangeModifier::new(PhaseChangeConfig::default(), 5, LO, HI);
    m.apply_heat_at(0.0, 0.0, 0.0, 900.0, 1.8);
    m
}

fn pressure() -> PressureModifier {
    let cfg = PressureConfig {
        internal_pressure: 2.0,
        ..PressureConfig::default()
    };
    let mut m = PressureModifier::new(cfg, 5, LO, HI);
    m.apply_pressure_at(0.0, 0.0, 0.0, 400.0, 1.5);
    m.apply_impact(1.0, 1.0, 0.0, 5.0, 1.0);
    m
}

fn fracture() -> FractureModifier {
    let cfg = FractureConfig {
        fracture_toughness: 20.0,
        max_cracks: 4,
        ..FractureConfig::default()
    };
    let mut m = FractureModifier::new(cfg, 5, LO, HI);
    m.apply_stress_at(0.0, 0.0, 0.0, 300.0, 1.5);
    m.cracks.push(Crack {
        start: (0.5, 0.5, 0.5),
        end: (0.6, 0.5, 0.5),
        direction: (1.0, 0.0, 0.0),
        length: 0.1,
        active: false,
    });
    m
}

fn erosion() -> ErosionModifier {
    let cfg = ErosionConfig {
        erosion_type: ErosionType::Ablation,
        flow_speed: 3.0,
        ..ErosionConfig::default()
    };
    let mut m = ErosionModifier::new(cfg, 5, LO, HI);
    m.set_exposure_at(0.0, 0.0, 0.0, 1.0, 1.5);
    m
}

/// Calls `f` once per modifier type with a scene and an untouched copy built
/// with another resolution and configuration.
macro_rules! for_each_modifier {
    ($f:ident) => {
        $f(
            "thermal",
            thermal(),
            ThermalModifier::new(ThermalConfig::default(), 2, LO, LO),
        );
        $f(
            "phase_change",
            phase_change(),
            PhaseChangeModifier::new(PhaseChangeConfig::default(), 2, LO, LO),
        );
        $f(
            "pressure",
            pressure(),
            PressureModifier::new(PressureConfig::default(), 2, LO, LO),
        );
        $f(
            "fracture",
            fracture(),
            FractureModifier::new(FractureConfig::default(), 2, LO, LO),
        );
        $f(
            "erosion",
            erosion(),
            ErosionModifier::new(ErosionConfig::default(), 2, LO, LO),
        );
    };
}

/// Runs `substeps` substeps of `p` through [`run_substep`] (no bodies, no
/// fields), returning it. Every substep must succeed.
fn drive<P: Participant + 'static>(p: P, substeps: usize) -> Box<dyn Participant> {
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(p)];
    let mut board = FieldBoard::new();
    let plan = ParticipantPlan::new(&ps, &board).expect("no ports");
    let mut frozen = vec![false];
    let bodies: Vec<RigidBody> = Vec::new();
    let mut forces = ForceAccumulator::new(0);
    for i in 0..substeps {
        let time = SubstepTime {
            index: i,
            count: substeps,
            h: h_sub(),
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
        assert!(faults.is_empty(), "substep {i} faulted: {faults:?}");
    }
    assert_eq!(forces.len(), 0);
    ps.pop().expect("one participant")
}

fn bytes_of(p: &dyn Participant) -> Vec<u8> {
    let mut out = Vec::new();
    p.write_state(&mut out);
    out
}

fn observe(p: &dyn Participant) -> Vec<(u32, Fix128)> {
    let mut sink = ObservationSink::new();
    p.observe(&mut sink);
    sink.values().to_vec()
}

// ============================================================================
// (a) contract
// ============================================================================

#[test]
fn every_modifier_satisfies_the_participant_contract() {
    fn one<P: Participant>(_: &str, p: P, fresh: P) {
        check_participant_contract(p, fresh, 40);
    }
    for_each_modifier!(one);
}

#[test]
fn an_empty_grid_satisfies_the_contract_and_reports_no_maximum() {
    let empty = || ThermalModifier::new(ThermalConfig::default(), 0, LO, HI);
    check_participant_contract(empty(), empty(), 5);
    let p = drive(empty(), 3);
    let obs = observe(p.as_ref());
    assert_eq!(
        obs,
        vec![(1, Fix128::ZERO)],
        "no cells: no maximum, zero melt"
    );
}

// ============================================================================
// (b) run_substep equals direct update
// ============================================================================

#[test]
fn run_substep_equals_direct_update_calls() {
    fn one<P: Participant + PhysicsModifier + Twin + 'static>(name: &str, p: P, _: P) {
        let mut direct = state_twin(&p);
        let n = 30;
        for _ in 0..n {
            direct.update(h_sub().to_f32());
        }
        let driven = drive(p, n);
        assert_eq!(
            bytes_of(driven.as_ref()),
            state_of(&direct),
            "{name}: {n} substeps through run_substep differ from {n} update calls"
        );
        assert!(
            state_of(&direct).len() > 100,
            "{name}: the state is too small to compare anything"
        );
    }
    for_each_modifier!(one);
}

/// The scene advanced by some steps changes its bytes (the comparisons in (b)
/// and (d) compare a moving state).
#[test]
fn every_scene_changes_state_when_driven() {
    fn one<P: Participant + 'static>(name: &str, p: P, _: P) {
        let before = state_of(&p);
        let after = bytes_of(drive(p, 5).as_ref());
        assert_ne!(before, after, "{name}: five substeps changed nothing");
    }
    for_each_modifier!(one);
}

/// A copy of `p` made through its own state bytes (the modifiers are not
/// `Clone`); the copy starts from a default-configured participant of the
/// same type so the bytes carry everything.
fn state_twin<P: Participant + PhysicsModifier + Twin>(p: &P) -> P {
    let mut t = P::blank();
    let b = state_of(p);
    t.check_state(&b).expect("own bytes");
    t.read_state(&b);
    t
}

trait Twin {
    fn blank() -> Self;
}
impl Twin for ThermalModifier {
    fn blank() -> Self {
        Self::new(ThermalConfig::default(), 1, LO, HI)
    }
}
impl Twin for PhaseChangeModifier {
    fn blank() -> Self {
        Self::new(PhaseChangeConfig::default(), 1, LO, HI)
    }
}
impl Twin for PressureModifier {
    fn blank() -> Self {
        Self::new(PressureConfig::default(), 1, LO, HI)
    }
}
impl Twin for FractureModifier {
    fn blank() -> Self {
        Self::new(FractureConfig::default(), 1, LO, HI)
    }
}
impl Twin for ErosionModifier {
    fn blank() -> Self {
        Self::new(ErosionConfig::default(), 1, LO, HI)
    }
}

/// (b) without going through the state bytes: two scenes built the same way,
/// one driven, one updated directly, compared by their public fields.
#[test]
fn run_substep_equals_direct_update_on_public_fields() {
    let n = 25;
    let dt = h_sub().to_f32();

    let mut t = thermal();
    for _ in 0..n {
        t.update(dt);
    }
    let mut tp = thermal();
    let driven = drive_back(&mut tp, n);
    assert_eq!(driven, state_of(&t));
    assert_eq!(bits(&tp.temperature.data), bits(&t.temperature.data));
    assert_eq!(
        bits(&tp.melt_accumulator.data),
        bits(&t.melt_accumulator.data)
    );

    let mut c = phase_change();
    for _ in 0..n {
        c.update(dt);
    }
    let mut cp = phase_change();
    drive_back(&mut cp, n);
    assert_eq!(bits(&cp.temperature.data), bits(&c.temperature.data));
    assert_eq!(bits(&cp.phase.data), bits(&c.phase.data));
    assert_eq!(bits(&cp.latent_heat.data), bits(&c.latent_heat.data));
    assert_eq!(bits(&cp.sdf_offset.data), bits(&c.sdf_offset.data));

    let mut f = fracture();
    for _ in 0..n {
        f.update(dt);
    }
    let mut fp = fracture();
    drive_back(&mut fp, n);
    assert_eq!(fp.cracks, f.cracks);
    assert!(f.cracks.len() > 1, "the scene must grow a crack");
}

/// Drives `p` in place (through a boxed copy) and writes the result back.
fn drive_back<P: Participant + Twin + 'static>(p: &mut P, n: usize) -> Vec<u8> {
    let start = std::mem::replace(p, P::blank());
    let driven = drive(start, n);
    let bytes = bytes_of(driven.as_ref());
    p.check_state(&bytes).expect("own bytes");
    p.read_state(&bytes);
    bytes
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ============================================================================
// (c) closed forms through the participant path
// ============================================================================

/// Smoothstep splat weight `w(d) = s²(3 − 2s)`, `s = 1 − d/r` (the
/// documented weight of `ScalarField3D::splat`; the same closed form as
/// `tests/analytic_thermal_wiring.rs`).
fn smoothstep_weight(d: f64, r: f64) -> f64 {
    if d >= r {
        return 0.0;
    }
    let s = 1.0 - d / r;
    s * s * (3.0 - 2.0 * s)
}

/// Point source, diffusion / cooling / melt off: each substep adds
/// `power · h · w(d)` at a node at distance `d`, so after `n` substeps
/// `T = T_amb + n · power · h · w(d)` (f32 rounding aside).
#[test]
fn a_thermal_point_source_heats_nodes_by_the_smoothstep_closed_form() {
    let cfg = ThermalConfig {
        diffusion_rate: 0.0,
        ambient_temperature: 20.0,
        cooling_rate: 0.0,
        melt_temperature: 1.0e9,
        melt_rate: 0.0,
        droop_strength: 0.0,
        expansion_coefficient: 0.0,
        freeze_temperature: -1.0e9,
        freeze_rate: 0.0,
    };
    // 9³ nodes on (-4, 4)³: node spacing exactly 1
    let mut m = ThermalModifier::new(cfg, 9, (-4.0, -4.0, -4.0), (4.0, 4.0, 4.0));
    let (power, radius) = (240.0_f32, 2.5_f32);
    m.add_heat_point(0.0, 0.0, 0.0, power, radius);
    let n = 48;
    let mut back = ThermalModifier::new(cfg, 1, LO, HI);
    let bytes = bytes_of(drive(m, n).as_ref());
    back.check_state(&bytes).expect("own bytes");
    back.read_state(&bytes);

    let h = 1.0 / 240.0;
    let mut checked = 0;
    for (x, y, z) in [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (2, 0, 0),
        (1, 1, 1),
        (3, 0, 0),
    ] {
        let d = f64::from(x * x + y * y + z * z).sqrt();
        let expected = 20.0 + n as f64 * f64::from(power) * h * smoothstep_weight(d, 2.5);
        let got = f64::from(back.temperature.get(
            (x + 4) as usize,
            (y + 4) as usize,
            (z + 4) as usize,
        ));
        assert!(
            (got - expected).abs() <= 1e-4 * expected.abs(),
            "node ({x},{y},{z}): got {got}, closed form {expected}"
        );
        checked += 1;
    }
    assert_eq!(checked, 6);
    // the node outside the radius is untouched
    assert_eq!(back.temperature.get(7, 4, 4), 20.0);
}

/// Explicit diffusion (7-point stencil, forward Euler, mirrored boundary —
/// the scheme `ScalarField3D::diffuse` documents) of one hot node, against
/// the same stencil written here in f64.
#[test]
fn thermal_diffusion_follows_the_explicit_stencil() {
    let cfg = ThermalConfig {
        diffusion_rate: 0.5,
        ambient_temperature: 0.0,
        cooling_rate: 0.0,
        melt_temperature: 1.0e9,
        melt_rate: 0.0,
        droop_strength: 0.0,
        expansion_coefficient: 0.0,
        freeze_temperature: -1.0e9,
        freeze_rate: 0.0,
    };
    let r = 7usize;
    let mut m = ThermalModifier::new(cfg, r, (-3.0, -3.0, -3.0), (3.0, 3.0, 3.0));
    let c = m.temperature.index(3, 3, 3);
    m.temperature.data[c] = 1000.0;

    let n = 60;
    let mut back = ThermalModifier::new(cfg, 1, LO, HI);
    let bytes = bytes_of(drive(m, n).as_ref());
    back.check_state(&bytes).expect("own bytes");
    back.read_state(&bytes);

    // reference: f64, spacing 1, rate·h = 0.5/240 (stable: ≤ 1/6)
    let k = 0.5 / 240.0;
    let idx = |x: usize, y: usize, z: usize| (z * r + y) * r + x;
    let mut t = vec![0.0_f64; r * r * r];
    t[idx(3, 3, 3)] = 1000.0;
    for _ in 0..n {
        let mut next = t.clone();
        for z in 0..r {
            for y in 0..r {
                for x in 0..r {
                    let v = t[idx(x, y, z)];
                    let at = |xx: Option<usize>, yy: Option<usize>, zz: Option<usize>| match (
                        xx, yy, zz,
                    ) {
                        (Some(a), Some(b), Some(cc)) if a < r && b < r && cc < r => {
                            t[idx(a, b, cc)]
                        }
                        _ => v,
                    };
                    let lap = at(x.checked_sub(1), Some(y), Some(z))
                        + at(Some(x + 1), Some(y), Some(z))
                        + at(Some(x), y.checked_sub(1), Some(z))
                        + at(Some(x), Some(y + 1), Some(z))
                        + at(Some(x), Some(y), z.checked_sub(1))
                        + at(Some(x), Some(y), Some(z + 1))
                        - 6.0 * v;
                    next[idx(x, y, z)] = v + k * lap;
                }
            }
        }
        t = next;
    }
    let total_ref: f64 = t.iter().sum();
    let total: f64 = back.temperature.data.iter().map(|&v| f64::from(v)).sum();
    assert!(
        (total_ref - 1000.0).abs() < 1e-9,
        "the stencil conserves heat"
    );
    assert!(
        (total - 1000.0).abs() < 1e-2,
        "participant lost heat: {total}"
    );
    for (i, (&got, &want)) in back.temperature.data.iter().zip(&t).enumerate() {
        assert!(
            (f64::from(got) - want).abs() <= 1e-3,
            "cell {i}: got {got}, stencil {want}"
        );
    }
    assert!(t[idx(3, 3, 3)] < 900.0, "heat spread from the hot node");
}

const TM: f32 = 100.0;
const TB: f32 = 300.0;
const LF: f32 = 10.0;
const LV: f32 = 20.0;

/// Enthalpy method with heat capacity 1 (closed form from
/// `tests/analytic_phase_change_wiring.rs`, Voller & Cross 1981): a cell of
/// total enthalpy `H = T + latent` settles to
/// `H ≤ Tm` solid at `H`; `Tm < H < Tm+Lf` solid on the plateau `Tm`;
/// `Tm+Lf ≤ H ≤ Tb+Lf` liquid at `H−Lf`; `Tb+Lf < H < Tb+Lf+Lv` liquid on
/// `Tb`; above gas at `H−Lf−Lv`.
fn enthalpy_closed_form(h: f32) -> (f32, Phase) {
    if h <= TM {
        (h, Phase::Solid)
    } else if h < TM + LF {
        (TM, Phase::Solid)
    } else if h <= TB + LF {
        (h - LF, Phase::Liquid)
    } else if h < TB + LF + LV {
        (TB, Phase::Liquid)
    } else {
        (h - LF - LV, Phase::Gas)
    }
}

#[test]
fn phase_change_settles_on_the_enthalpy_closed_form_through_the_participant() {
    let cfg = PhaseChangeConfig {
        melt_temperature: TM,
        boil_temperature: TB,
        latent_heat_fusion: LF,
        latent_heat_vaporization: LV,
        diffusion_rate: 0.0,
        ambient_temperature: 20.0,
        cooling_rate: 0.0,
        ..PhaseChangeConfig::default()
    };
    for start in [50.0_f32, 105.0, 200.0, 315.0, 400.0] {
        let mut m = PhaseChangeModifier::new(cfg, 3, LO, HI);
        m.temperature.data.fill(start);
        let mut back = PhaseChangeModifier::new(cfg, 1, LO, HI);
        let bytes = bytes_of(drive(m, 6).as_ref());
        back.check_state(&bytes).expect("own bytes");
        back.read_state(&bytes);
        let (t, phase) = enthalpy_closed_form(start);
        for i in 0..back.temperature.data.len() {
            let got_t = back.temperature.data[i];
            let got_h = got_t + back.latent_heat.data[i];
            assert_eq!(got_h, start, "start {start}: enthalpy not conserved");
            assert_eq!(got_t, t, "start {start}: temperature");
        }
        assert_eq!(back.phase_at(0.0, 0.0, 0.0), phase, "start {start}: phase");
    }
}

// ============================================================================
// (d) round trip mid-run
// ============================================================================

#[test]
fn a_round_trip_mid_run_continues_bit_identically() {
    fn one<P: Participant + Rebuild + 'static>(name: &str, p: P, mut fresh: P) {
        let (k, n) = (7, 20);
        let uninterrupted = bytes_of(drive(P::rebuild(name), n).as_ref());

        let first = bytes_of(drive(p, k).as_ref());
        let before = state_of(&fresh);
        assert_ne!(
            before, first,
            "{name}: fresh participant already equals the snapshot"
        );
        fresh.check_state(&first).expect("own bytes");
        fresh.read_state(&first);
        assert_eq!(state_of(&fresh), first, "{name}: restore");
        let resumed = bytes_of(drive(fresh, n - k).as_ref());
        assert_eq!(
            resumed,
            uninterrupted,
            "{name}: restore after {k} substeps then {} more differs from {n} in one run",
            n - k
        );
    }
    for_each_modifier!(one);
}

trait Rebuild {
    fn rebuild(name: &str) -> Self;
}
impl Rebuild for ThermalModifier {
    fn rebuild(_: &str) -> Self {
        thermal()
    }
}
impl Rebuild for PhaseChangeModifier {
    fn rebuild(_: &str) -> Self {
        phase_change()
    }
}
impl Rebuild for PressureModifier {
    fn rebuild(_: &str) -> Self {
        pressure()
    }
}
impl Rebuild for FractureModifier {
    fn rebuild(_: &str) -> Self {
        fracture()
    }
}
impl Rebuild for ErosionModifier {
    fn rebuild(_: &str) -> Self {
        erosion()
    }
}

/// `enabled` is part of the state: a disabled scene restored into an enabled
/// participant stays disabled (driving it changes nothing).
#[test]
fn a_disabled_modifier_stays_disabled_after_a_round_trip() {
    fn one<P: Participant + Toggle + 'static>(name: &str, mut p: P, mut fresh: P) {
        p.disable();
        let b = state_of(&p);
        fresh.check_state(&b).expect("own bytes");
        fresh.read_state(&b);
        let after = bytes_of(drive(fresh, 5).as_ref());
        assert_eq!(after, b, "{name}: the restored participant is not disabled");
    }
    for_each_modifier!(one);
}

trait Toggle {
    fn disable(&mut self);
}
impl Toggle for ThermalModifier {
    fn disable(&mut self) {
        self.enabled = false;
    }
}
impl Toggle for PhaseChangeModifier {
    fn disable(&mut self) {
        self.enabled = false;
    }
}
impl Toggle for PressureModifier {
    fn disable(&mut self) {
        self.enabled = false;
    }
}
impl Toggle for FractureModifier {
    fn disable(&mut self) {
        self.enabled = false;
    }
}
impl Toggle for ErosionModifier {
    fn disable(&mut self) {
        self.enabled = false;
    }
}

/// The configuration is part of the state: a restored participant follows
/// the snapshot's configuration, not the one it was built with.
#[test]
fn the_configuration_travels_in_the_state() {
    let src = thermal();
    let mut dst = ThermalModifier::new(ThermalConfig::default(), 2, LO, LO);
    let b = state_of(&src);
    dst.check_state(&b).expect("own bytes");
    dst.read_state(&b);
    assert_eq!(dst.config, src.config);
    assert_eq!(dst.heat_sources, src.heat_sources);
    assert_eq!(dst.temperature.nx, 5);
    assert_eq!(dst.temperature.min, LO);
    assert_eq!(dst.temperature.max, HI);

    let src = erosion();
    let mut dst = ErosionModifier::new(ErosionConfig::default(), 2, LO, LO);
    let b = state_of(&src);
    dst.check_state(&b).expect("own bytes");
    dst.read_state(&b);
    assert_eq!(dst.config, src.config);
    assert_eq!(dst.config.erosion_type, ErosionType::Ablation);

    let mut src = pressure();
    src.enabled = false;
    let mut dst = pressure();
    let b = state_of(&src);
    dst.check_state(&b).expect("own bytes");
    dst.read_state(&b);
    assert!(!dst.enabled);
}

// ============================================================================
// (e) check_state refusals
// ============================================================================

fn refuses<P: Participant>(name: &str, p: &P, bytes: &[u8], what: &str) -> StateError {
    let before = state_of(p);
    let err = match p.check_state(bytes) {
        Err(e) => e,
        Ok(()) => panic!("{name}: check_state accepted a payload with {what}"),
    };
    assert_eq!(
        state_of(p),
        before,
        "{name}: check_state changed the participant"
    );
    err
}

#[test]
fn check_state_refuses_bad_payloads_and_changes_nothing() {
    fn one<P: Participant>(name: &str, p: P, other: P) {
        let good = state_of(&p);
        for cut in 0..good.len() {
            let e = refuses(name, &other, &good[..cut], "a truncated payload");
            assert!(
                matches!(e, StateError::Length { found, .. } if found == cut),
                "{name}: cut {cut} gave {e:?}"
            );
        }
        let mut longer = good.clone();
        longer.push(0);
        let e = refuses(name, &other, &longer, "a trailing byte");
        assert_eq!(
            e,
            StateError::Length {
                expected: good.len(),
                found: good.len() + 1
            }
        );
        for version in [0_u32, 2, u32::MAX] {
            let mut wrong = good.clone();
            wrong[..4].copy_from_slice(&version.to_le_bytes());
            let e = refuses(name, &other, &wrong, "another version");
            assert_eq!(e, StateError::InvalidValue, "{name}: version {version}");
        }
        assert_eq!(u32::from_le_bytes(good[..4].try_into().unwrap()), 1);
    }
    for_each_modifier!(one);
}

/// Field offsets of the version-1 payloads (documented in `sim_modifier`).
const THERMAL_ENABLED_AT: usize = 4 + 9 * 4;
const EROSION_TYPE_AT: usize = 4;

#[test]
fn check_state_refuses_invalid_values() {
    let p = thermal();
    let good = state_of(&p);
    let mut bad = good.clone();
    assert!(bad[THERMAL_ENABLED_AT] <= 1);
    bad[THERMAL_ENABLED_AT] = 2;
    assert_eq!(
        refuses("thermal", &p, &bad, "enabled = 2"),
        StateError::InvalidValue
    );

    let e = erosion();
    let mut bad = state_of(&e);
    assert_eq!(bad[EROSION_TYPE_AT], 3, "Ablation is tag 3");
    bad[EROSION_TYPE_AT] = 4;
    assert_eq!(
        refuses("erosion", &e, &bad, "erosion type 4"),
        StateError::InvalidValue
    );

    // the heat source tags sit right after the two fields: find the first
    // one by rewriting the count to 0 and checking the length it then asks for
    let mut lone = ThermalModifier::new(ThermalConfig::default(), 1, LO, HI);
    lone.heat_sources
        .push(alice_physics::thermal::HeatSource::Point {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            power: 1.0,
            radius: 1.0,
        });
    let mut bad = state_of(&lone);
    let tag_at = bad.len() - 5 * 4 - 1;
    assert_eq!(bad[tag_at], 0, "Point is tag 0");
    bad[tag_at] = 7;
    assert_eq!(
        refuses("thermal", &lone, &bad, "heat source tag 7"),
        StateError::InvalidValue
    );

    // a grid size whose cell count overflows
    let mut bad = state_of(&lone);
    let nx_at = THERMAL_ENABLED_AT + 1;
    bad[nx_at..nx_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(matches!(
        refuses("thermal", &lone, &bad, "a cell count past usize"),
        StateError::InvalidValue | StateError::Length { .. }
    ));
}

// ============================================================================
// Step rule: non-positive widths never reach the participant
// ============================================================================

#[test]
fn the_step_rule_refuses_non_positive_widths_before_any_substep() {
    fn one<P: Participant>(name: &str, p: P, _: P) {
        assert_eq!(p.step_rule(), StepRule::FollowSubstep, "{name}");
        assert_eq!(p.step_rule().steps_per_substep(h_sub()), Ok(1));
        for h in [
            Fix128::ZERO,
            Fix128::from_int(-1),
            Fix128::from_ratio(-1, 240),
        ] {
            assert_eq!(
                p.step_rule().steps_per_substep(h),
                Err(RegisterError::NonPositiveStep),
                "{name}: h = {h:?}"
            );
        }
        assert!(p.ports().is_empty(), "{name}: no ports in this version");
    }
    for_each_modifier!(one);
}

// ============================================================================
// Observations
// ============================================================================

#[test]
fn observations_follow_the_documented_channels() {
    let t = thermal();
    let obs = observe(&t);
    let max_t = t.temperature.data.iter().copied().fold(f32::MIN, f32::max);
    let melt: Fix128 = t
        .melt_accumulator
        .data
        .iter()
        .fold(Fix128::ZERO, |a, &v| a + Fix128::from_f32(v));
    assert_eq!(obs, vec![(0, Fix128::from_f32(max_t)), (1, melt)]);

    let c = phase_change();
    let driven = drive(phase_change(), 40);
    let obs = observe(driven.as_ref());
    assert_eq!(obs.iter().map(|o| o.0).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert_eq!(observe(&c)[1], (1, Fix128::ZERO), "nothing melted yet");

    let f = fracture();
    let obs = observe(&f);
    assert_eq!(obs[1], (1, Fix128::from_int(1)), "one crack");
    assert_eq!(obs[2], (2, Fix128::ZERO), "no growing crack");

    for (name, p) in [
        ("pressure", Box::new(pressure()) as Box<dyn Participant>),
        ("erosion", Box::new(erosion())),
    ] {
        let obs = observe(p.as_ref());
        assert_eq!(
            obs.iter().map(|o| o.0).collect::<Vec<_>>(),
            vec![0, 1],
            "{name}"
        );
    }
}

// ============================================================================
// Kinds
// ============================================================================

const fn fourcc(code: &[u8; 4]) -> ParticipantKind {
    ParticipantKind::new(u32::from_be_bytes(*code))
}

/// Every kind allocated in the crate, with its type. Extend this list when a
/// participant type is added (it mirrors the table in the
/// `world_participant` module documentation).
fn in_crate_kinds() -> Vec<(&'static str, ParticipantKind)> {
    vec![
        ("ThermalModifier", ThermalModifier::PARTICIPANT_KIND),
        ("PhaseChangeModifier", PhaseChangeModifier::PARTICIPANT_KIND),
        ("PressureModifier", PressureModifier::PARTICIPANT_KIND),
        ("FractureModifier", FractureModifier::PARTICIPANT_KIND),
        ("ErosionModifier", ErosionModifier::PARTICIPANT_KIND),
        // reserved for the crowd and molecular dynamics participants
        ("crowd", fourcc(b"CRWD")),
        ("molecular dynamics", fourcc(b"MDVV")),
    ]
}

/// Kinds used by `tests/world_participant_conformance.rs`.
fn conformance_test_kinds() -> Vec<ParticipantKind> {
    vec![
        fourcc(b"TETH"),
        fourcc(b"OTHR"),
        fourcc(b"WRIT"),
        fourcc(b"READ"),
        fourcc(b"PUSH"),
        ParticipantKind::new(0),
        ParticipantKind::new(3),
    ]
}

/// Kinds used by `examples/world_participant_contract.rs`.
fn contract_example_kinds() -> Vec<ParticipantKind> {
    [1, 2, 4, 5].into_iter().map(ParticipantKind::new).collect()
}

#[test]
fn allocated_kinds_are_the_documented_codes() {
    assert_eq!(ThermalModifier::PARTICIPANT_KIND, fourcc(b"THRM"));
    assert_eq!(PhaseChangeModifier::PARTICIPANT_KIND, fourcc(b"PHAS"));
    assert_eq!(PressureModifier::PARTICIPANT_KIND, fourcc(b"PRES"));
    assert_eq!(FractureModifier::PARTICIPANT_KIND, fourcc(b"FRAC"));
    assert_eq!(ErosionModifier::PARTICIPANT_KIND, fourcc(b"EROS"));
    assert_eq!(ThermalModifier::PARTICIPANT_KIND.get(), 0x5448_524d);
    // the values in the table of the `world_participant` module documentation
    assert_eq!(PhaseChangeModifier::PARTICIPANT_KIND.get(), 0x5048_4153);
    assert_eq!(PressureModifier::PARTICIPANT_KIND.get(), 0x5052_4553);
    assert_eq!(FractureModifier::PARTICIPANT_KIND.get(), 0x4652_4143);
    assert_eq!(ErosionModifier::PARTICIPANT_KIND.get(), 0x4552_4f53);
    let p: Box<dyn Participant> = Box::new(thermal());
    assert_eq!(p.kind(), ThermalModifier::PARTICIPANT_KIND);
    let p: Box<dyn Participant> = Box::new(erosion());
    assert_eq!(p.kind(), ErosionModifier::PARTICIPANT_KIND);
}

#[test]
fn in_crate_kinds_are_distinct_and_clear_of_test_and_example_kinds() {
    let crate_kinds = in_crate_kinds();
    for (i, (a, ka)) in crate_kinds.iter().enumerate() {
        assert!(
            ka.get() >= 0x0100_0000,
            "{a}: values below 0x0100_0000 are left for tests and user code"
        );
        for (b, kb) in &crate_kinds[i + 1..] {
            assert_ne!(ka, kb, "{a} and {b} share a kind");
        }
        for k in conformance_test_kinds()
            .into_iter()
            .chain(contract_example_kinds())
        {
            assert_ne!(*ka, k, "{a} collides with a test or example kind");
        }
    }
    // every kind used by the conformance tests and the contract example is
    // distinct from every other, across both sources
    let mut all: Vec<ParticipantKind> = conformance_test_kinds()
        .into_iter()
        .chain(contract_example_kinds())
        .collect();
    let n = all.len();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), n, "a test or example kind is used twice");
}
