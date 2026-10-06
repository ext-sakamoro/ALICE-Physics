//! The social-force crowd ([`CrowdParticipant`]) and velocity-Verlet molecular
//! dynamics ([`MdParticipant`]) as World v1 participants.
//!
//! The world side of the participant contract is not wired yet, so every test
//! drives the participants through [`run_substep`], the function the world
//! calls once per substep.
//!
//! Oracles:
//! * contract: the participant-side rules ([`check_participant_contract`],
//!   copied unchanged from `tests/world_participant_conformance.rs`);
//! * equivalence: `n` substeps through [`run_substep`] against `n` direct
//!   calls of the module's own step (an independent path), bit for bit;
//! * closed forms: the discrete relaxation of a lone pedestrian, and two
//!   Lennard-Jones particles at rest at the potential minimum `r = 2^{1/6} σ`
//!   plus the velocity-Verlet energy bound of a small oscillation about it;
//! * snapshot: a run interrupted by `write_state` / `read_state` against an
//!   uninterrupted run, bit for bit.

#![cfg(feature = "std")]
// f64 `powf` / `powi` / `exp` compute closed-form references, not state.
#![allow(clippy::disallowed_methods)]

use alice_physics::crowd_force::{
    CrowdParticipant, InteractionParams, NeighborSearch, Pedestrian, SocialForce, WallSegment,
    CROWD_OBS_COUNT, CROWD_OBS_MEAN_SPEED, CROWD_PARTICIPANT_KIND,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::molecular_dynamics::{
    MdParticipant, PeriodicBox, VelocityVerlet, MD_OBS_KINETIC, MD_OBS_POTENTIAL,
    MD_OBS_TEMPERATURE, MD_OBS_TOTAL, MD_PARTICIPANT_KIND,
};
use alice_physics::pair_potential::{LennardJones, ShiftMode, Truncated};
use alice_physics::physics2d::Vec2Fix;
use alice_physics::solver::RigidBody;
use alice_physics::world_participant::{
    run_substep, FieldBoard, ForceAccumulator, ObservationSink, Participant, ParticipantFault,
    ParticipantKind, ParticipantPlan, StateError, SubstepCtx, SubstepTime, WorldFault,
};

// ============================================================================
// Harness (copied unchanged from tests/world_participant_conformance.rs)
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
// Fixtures
// ============================================================================

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

/// Helbing–Farkas–Vicsek 2000 values (`A = 2000 N`, `B = 0.08 m`,
/// `k = 1.2·10⁵`, `κ = 2.4·10⁵`) for pedestrians and walls.
fn hfv_params() -> InteractionParams {
    InteractionParams {
        strength_n: Fix128::from_int(2000),
        range_m: Fix128::from_ratio(2, 25),
        body_stiffness: Fix128::from_int(120_000),
        sliding_friction: Fix128::from_int(240_000),
    }
}

fn model() -> SocialForce {
    SocialForce::new(
        hfv_params(),
        hfv_params(),
        Fix128::from_ratio(1, 2),
        Fix128::from_int(2),
    )
    .expect("valid model")
}

fn pedestrian(x: f64, y: f64, dir: (i64, i64)) -> Pedestrian {
    Pedestrian {
        position: Vec2Fix::new(fx(x), fx(y)),
        velocity: Vec2Fix::new(Fix128::ZERO, Fix128::ZERO),
        radius_m: Fix128::from_ratio(3, 10),
        mass_kg: Fix128::from_int(80),
        desired_speed_m_s: Fix128::from_ratio(13, 10),
        desired_direction: Vec2Fix::new(Fix128::from_int(dir.0), Fix128::from_int(dir.1)),
        relaxation_time_s: Fix128::from_ratio(1, 2),
    }
}

/// A corridor 4 m wide with two groups walking towards each other.
fn corridor_walls() -> Vec<WallSegment> {
    vec![
        WallSegment {
            start: Vec2Fix::new(Fix128::from_int(-10), Fix128::from_int(-2)),
            end: Vec2Fix::new(Fix128::from_int(10), Fix128::from_int(-2)),
        },
        WallSegment {
            start: Vec2Fix::new(Fix128::from_int(-10), Fix128::from_int(2)),
            end: Vec2Fix::new(Fix128::from_int(10), Fix128::from_int(2)),
        },
    ]
}

fn corridor_crowd() -> Vec<Pedestrian> {
    vec![
        pedestrian(-3.0, 0.4, (1, 0)),
        pedestrian(-3.5, -0.5, (1, 0)),
        pedestrian(-2.4, -1.1, (1, 0)),
        pedestrian(3.0, 0.1, (-1, 0)),
        pedestrian(2.6, 1.0, (-1, 0)),
        pedestrian(3.4, -0.8, (-1, 0)),
    ]
}

fn crowd() -> CrowdParticipant {
    CrowdParticipant::new(
        model(),
        corridor_crowd(),
        corridor_walls(),
        NeighborSearch::CellList,
        Some(Fix128::from_ratio(13, 10)),
    )
    .expect("valid crowd")
}

/// Lennard-Jones in reduced units (`ε = σ = 1`), cut at `2.5 σ`.
fn lj() -> Truncated<LennardJones> {
    lj_shifted(ShiftMode::ForceShift)
}

fn lj_shifted(mode: ShiftMode) -> Truncated<LennardJones> {
    Truncated::new(
        LennardJones::new(Fix128::ONE, Fix128::ONE).expect("lj"),
        fx(2.5),
        mode,
    )
    .expect("truncated")
}

/// 8 particles of a perturbed simple cubic lattice in a box of side 6.
fn md_system() -> VelocityVerlet<LennardJones> {
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut k = 0i64;
    for ix in 0..2 {
        for iy in 0..2 {
            for iz in 0..2 {
                k += 1;
                positions.push(Vec3Fix::new(
                    fx(1.5 + 3.0 * f64::from(ix) + 0.03 * (k % 3) as f64),
                    fx(1.5 + 3.0 * f64::from(iy) - 0.02 * (k % 5) as f64),
                    fx(1.5 + 1.15 * f64::from(iz)),
                ));
                velocities.push(Vec3Fix::new(
                    fx(0.1 * ((k * 7 % 5) - 2) as f64),
                    fx(0.1 * ((k * 3 % 5) - 2) as f64),
                    fx(0.05 * ((k % 3) - 1) as f64),
                ));
            }
        }
    }
    VelocityVerlet::new(
        lj(),
        PeriodicBox::cubic(Fix128::from_int(6)).expect("box"),
        positions,
        velocities,
        vec![Fix128::ONE; 8],
    )
    .expect("system")
}

fn md() -> MdParticipant<LennardJones> {
    MdParticipant::new(md_system(), Fix128::ONE).expect("participant")
}

fn h_sub() -> Fix128 {
    Fix128::from_ratio(1, 256)
}

/// `substeps` substeps of `ps` through [`run_substep`] with width `h`,
/// starting at substep index `start` of a frame of 4.
fn drive(
    ps: &mut [Box<dyn Participant>],
    start: usize,
    substeps: usize,
    h: Fix128,
) -> Vec<WorldFault> {
    let bodies = bodies_for_harness();
    let mut board = FieldBoard::new();
    let plan = ParticipantPlan::new(ps, &board).expect("plan");
    let mut frozen = vec![false; ps.len()];
    let mut faults = Vec::new();
    for i in start..start + substeps {
        let mut forces = ForceAccumulator::new(bodies.len());
        let time = SubstepTime {
            index: i % 4,
            count: 4,
            h,
        };
        faults.extend(
            run_substep(
                ps,
                &plan,
                &mut frozen,
                &bodies,
                &mut board,
                &mut forces,
                time,
            )
            .expect("inputs fit"),
        );
        for b in 0..bodies.len() {
            assert_eq!(forces.force(b), Some(Vec3Fix::ZERO), "no forces on bodies");
        }
    }
    faults
}

fn boxed_state(p: &dyn Participant) -> Vec<u8> {
    let mut out = Vec::new();
    p.write_state(&mut out);
    out
}

// ============================================================================
// Kinds
// ============================================================================

fn code(tag: &[u8; 4]) -> ParticipantKind {
    ParticipantKind::new(u32::from_be_bytes(*tag))
}

#[test]
fn kinds_are_the_documented_codes_and_distinct() {
    assert_eq!(CROWD_PARTICIPANT_KIND, code(b"CRWD"));
    assert_eq!(MD_PARTICIPANT_KIND, code(b"MDVV"));
    assert_eq!(crowd().kind(), CROWD_PARTICIPANT_KIND);
    assert_eq!(md().kind(), MD_PARTICIPANT_KIND);
    assert_ne!(CROWD_PARTICIPANT_KIND, MD_PARTICIPANT_KIND);
    for taken in [
        b"THRM", b"PHAS", b"PRES", b"FRAC", b"EROS", b"TETH", b"OTHR", b"WRIT", b"READ", b"PUSH",
    ] {
        assert_ne!(CROWD_PARTICIPANT_KIND, code(taken), "{taken:?}");
        assert_ne!(MD_PARTICIPANT_KIND, code(taken), "{taken:?}");
    }
}

// ============================================================================
// (a) contract
// ============================================================================

#[test]
fn crowd_satisfies_the_participant_contract() {
    check_participant_contract(crowd(), crowd(), 200);
}

#[test]
fn md_satisfies_the_participant_contract() {
    check_participant_contract(md(), md(), 200);
}

#[test]
fn a_failing_crowd_call_passes_the_contract() {
    let mut p = crowd();
    p.pedestrians_mut()[2].relaxation_time_s = Fix128::ZERO;
    check_participant_contract(p, crowd(), 10);
}

// ============================================================================
// (b) run_substep == the module's own step
// ============================================================================

#[test]
fn crowd_through_run_substep_matches_social_force_step() {
    let n = 400;
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(crowd())];
    assert!(drive(&mut ps, 0, n, h_sub()).is_empty());

    let mut peds = corridor_crowd();
    let walls = corridor_walls();
    let m = model();
    for _ in 0..n {
        m.step(
            &mut peds,
            &walls,
            h_sub(),
            NeighborSearch::CellList,
            Some(Fix128::from_ratio(13, 10)),
        )
        .expect("step");
    }
    let direct = CrowdParticipant::new(
        model(),
        peds.clone(),
        corridor_walls(),
        NeighborSearch::CellList,
        Some(Fix128::from_ratio(13, 10)),
    )
    .expect("crowd");
    assert_eq!(boxed_state(ps[0].as_ref()), state_of(&direct));
    // the crowd moved, so the comparison is not between two initial states
    assert_ne!(peds, corridor_crowd());
}

#[test]
fn md_through_run_substep_matches_velocity_verlet_step() {
    let n = 400;
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(md())];
    assert!(drive(&mut ps, 0, n, h_sub()).is_empty());

    let mut sys = md_system();
    for _ in 0..n {
        sys.step(h_sub()).expect("step");
    }
    assert_ne!(sys.positions(), md_system().positions());
    let direct = MdParticipant::new(sys, Fix128::ONE).expect("participant");
    assert_eq!(boxed_state(ps[0].as_ref()), state_of(&direct));
}

// ============================================================================
// (c) closed forms
// ============================================================================

/// A lone pedestrian (no neighbour, no wall) under the driving term only.
///
/// Continuous law: `v(t) = v_d + (v_0 − v_d) e^{−t/τ}` with `v_d = v0 ê`.
/// The module's semi-implicit Euler `v ← v + h (v_d − v)/τ` gives
/// `v_{n+1} − v_d = (1 − h/τ)(v_n − v_d)`, so the discrete closed form is
///
/// ```text
/// v_n = v_d + (v_0 − v_d)(1 − h/τ)^n
/// ```
///
/// and, with `x = h/τ`, `0 ≤ e^{−x} − (1 − x) ≤ x²/2` and
/// `|aⁿ − bⁿ| ≤ n |a − b|` for `a, b ∈ [0, 1]`, it differs from the
/// continuous law by at most `n x²/2 · |v_0 − v_d|`.
#[test]
fn a_lone_pedestrian_relaxes_to_its_desired_velocity() {
    let tau = 0.5;
    let v0 = 1.3;
    let h = h_sub(); // 2⁻⁸, so x = h/τ = 2⁻⁷ exactly
    let mut p0 = pedestrian(0.0, 0.0, (1, 0));
    p0.velocity = Vec2Fix::new(fx(-0.4), fx(0.25));
    let start = p0.velocity;
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(
        CrowdParticipant::new(model(), vec![p0], Vec::new(), NeighborSearch::Direct, None)
            .expect("crowd"),
    )];
    let x = h.to_f64() / tau;
    let mut last = 0usize;
    for n in [1usize, 10, 64, 200, 600] {
        drive(&mut ps, last, n - last, h);
        last = n;
        let mut obs = ObservationSink::new();
        ps[0].observe(&mut obs);
        let mut out = Vec::new();
        ps[0].write_state(&mut out);
        let fresh =
            &mut CrowdParticipant::new(model(), vec![p0], Vec::new(), NeighborSearch::Direct, None)
                .expect("crowd");
        fresh.read_state(&out);
        let v = fresh.pedestrians()[0].velocity;
        let decay = (1.0 - x).powi(n as i32);
        let want = [
            v0 + (start.x.to_f64() - v0) * decay,
            start.y.to_f64() * decay,
        ];
        let got = [v.x.to_f64(), v.y.to_f64()];
        for k in 0..2 {
            assert!(
                (got[k] - want[k]).abs() < 1e-12,
                "n = {n}, axis {k}: {} vs discrete closed form {}",
                got[k],
                want[k]
            );
            let cont = [
                v0 + (start.x.to_f64() - v0) * (-(n as f64) * x).exp(),
                start.y.to_f64() * (-(n as f64) * x).exp(),
            ];
            let dv = [(start.x.to_f64() - v0).abs(), start.y.to_f64().abs()];
            assert!(
                (got[k] - cont[k]).abs() <= n as f64 * x * x / 2.0 * dv[k] + 1e-12,
                "n = {n}, axis {k}: beyond the first-order bound of the continuous law"
            );
        }
        // observation: one pedestrian, speed |v|
        let speed = (got[0] * got[0] + got[1] * got[1]).sqrt();
        assert_eq!(obs.values()[0], (CROWD_OBS_COUNT, Fix128::ONE));
        assert_eq!(obs.values()[1].0, CROWD_OBS_MEAN_SPEED);
        assert!((obs.values()[1].1.to_f64() - speed).abs() < 1e-12);
    }
}

fn lj_pair(separation: Fix128, v_rel: Fix128) -> MdParticipant<LennardJones> {
    let c = Fix128::from_int(5);
    let half = separation.half();
    let vh = v_rel.half();
    // energy shift: the force is the plain Lennard-Jones force, so its zero
    // stays at 2^{1/6} σ (a force shift would move it)
    let sys = VelocityVerlet::new(
        lj_shifted(ShiftMode::EnergyShift),
        PeriodicBox::cubic(Fix128::from_int(10)).expect("box"),
        vec![Vec3Fix::new(c - half, c, c), Vec3Fix::new(c + half, c, c)],
        vec![
            Vec3Fix::new(Fix128::ZERO - vh, Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(vh, Fix128::ZERO, Fix128::ZERO),
        ],
        vec![Fix128::ONE; 2],
    )
    .expect("pair");
    MdParticipant::new(sys, Fix128::ONE).expect("participant")
}

fn md_observation(p: &dyn Participant) -> Vec<(u32, Fix128)> {
    let mut obs = ObservationSink::new();
    p.observe(&mut obs);
    obs.values().to_vec()
}

fn channel(obs: &[(u32, Fix128)], c: u32) -> f64 {
    obs.iter()
        .find(|(k, _)| *k == c)
        .map(|(_, v)| v.to_f64())
        .expect("channel reported")
}

/// `r_min = 2^{1/6} σ` is where `F(r) = 24ε(2σ¹²/r¹³ − σ⁶/r⁷) = 0`. Fix128
/// places the pair within one `f64` rounding of `r_min` (`|δr| ≲ 2⁻⁵²`), so
/// the residual force is `|U''(r_min) δr| ≈ 57 · 2⁻⁵² < 1.3·10⁻¹⁴`; over
/// `t = 4` the pair stays at rest within `10⁻¹²` (the restoring force keeps the
/// motion an oscillation of amplitude `|δr|`).
#[test]
fn two_lj_particles_at_the_minimum_stay_at_rest() {
    let r_min = fx(2f64.powf(1.0 / 6.0));
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(lj_pair(r_min, Fix128::ZERO))];
    let obs0 = md_observation(ps[0].as_ref());
    // EnergyShift: U(r_min) = −ε − U_LJ(r_c)
    let rc = 2.5f64;
    let u_lj = |r: f64| 4.0 * (r.powi(-12) - r.powi(-6));
    let r = 2f64.powf(1.0 / 6.0);
    let u_want = u_lj(r) - u_lj(rc);
    assert!((channel(&obs0, MD_OBS_POTENTIAL) - u_want).abs() < 1e-12);
    assert_eq!(channel(&obs0, MD_OBS_KINETIC), 0.0);

    drive(&mut ps, 0, 1024, h_sub());
    let obs = md_observation(ps[0].as_ref());
    assert!(
        channel(&obs, MD_OBS_KINETIC) < 1e-24,
        "kinetic {}",
        channel(&obs, MD_OBS_KINETIC)
    );
    assert!((channel(&obs, MD_OBS_POTENTIAL) - u_want).abs() < 1e-12);
}

/// A small oscillation about `r_min`. Linearised, the relative coordinate is
/// a harmonic oscillator of `ω² = U''(r_min)/μ` (`U'' = 72 ε/(2^{1/3} σ²) = 36·2^{2/3} ε/σ²`;
/// the energy shift adds no curvature; `μ = m/2`). For
/// `x'' = −ω²x` velocity Verlet conserves `J = ω²x² + v²/(1 − (ωh)²/4)`
/// exactly, so `E = (μ/2)(v² + ω²x²)` stays in `[(1 − (ωh)²/4) J μ/2, J μ/2]`:
/// its spread is at most `(ωh)²/4 · E_osc`. The test allows twice that for
/// the anharmonic part (amplitude 1% of `r_min`) plus the Fix128 rounding.
#[test]
fn a_small_lj_oscillation_conserves_energy_within_the_verlet_bound() {
    let r_min = fx(2f64.powf(1.0 / 6.0));
    let v_rel = fx(0.1);
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(lj_pair(r_min, v_rel))];
    let e0 = channel(&md_observation(ps[0].as_ref()), MD_OBS_TOTAL);
    let mu = 0.5;
    let omega = (36.0 * 2f64.powf(2.0 / 3.0) / mu).sqrt();
    let wh = omega * h_sub().to_f64();
    let e_osc = 0.5 * mu * 0.1 * 0.1;
    let bound = 2.0 * wh * wh / 4.0 * e_osc + 1e-15;
    let mut worst = 0f64;
    for k in 0..256 {
        drive(&mut ps, k * 8, 8, h_sub());
        let obs = md_observation(ps[0].as_ref());
        let e = channel(&obs, MD_OBS_TOTAL);
        assert_eq!(
            obs.iter()
                .find(|(c, _)| *c == MD_OBS_TOTAL)
                .map(|(_, v)| *v),
            Some(
                obs.iter()
                    .find(|(c, _)| *c == MD_OBS_KINETIC)
                    .map(|(_, v)| *v)
                    .expect("K")
                    + obs
                        .iter()
                        .find(|(c, _)| *c == MD_OBS_POTENTIAL)
                        .map(|(_, v)| *v)
                        .expect("U")
            ),
            "total = kinetic + potential"
        );
        // T = 2K / (k_B (3N − 3)) with N = 2, k_B = 1
        assert!(
            (channel(&obs, MD_OBS_TEMPERATURE) - channel(&obs, MD_OBS_KINETIC) / 1.5).abs() < 1e-15
        );
        worst = worst.max((e - e0).abs());
    }
    assert!(
        worst <= bound,
        "energy spread {worst} above the Verlet bound {bound}"
    );
    // the oscillation is resolved: the spread is not trivially zero
    assert!(worst > 0.0);
}

// ============================================================================
// (d) snapshot mid-run
// ============================================================================

fn interrupted_equals_uninterrupted(make: &dyn Fn() -> Box<dyn Participant>) {
    let (k, m) = (150, 250);
    let mut straight: Vec<Box<dyn Participant>> = vec![make()];
    drive(&mut straight, 0, k + m, h_sub());

    let mut first: Vec<Box<dyn Participant>> = vec![make()];
    drive(&mut first, 0, k, h_sub());
    let blob = boxed_state(first[0].as_ref());
    let mut second: Vec<Box<dyn Participant>> = vec![make()];
    second[0].check_state(&blob).expect("own payload");
    second[0].read_state(&blob);
    drive(&mut second, k, m, h_sub());

    assert_eq!(
        boxed_state(second[0].as_ref()),
        boxed_state(straight[0].as_ref())
    );
    assert_ne!(
        boxed_state(straight[0].as_ref()),
        boxed_state(make().as_ref())
    );
}

#[test]
fn crowd_snapshot_mid_run_continues_bit_identically() {
    interrupted_equals_uninterrupted(&|| Box::new(crowd()));
}

#[test]
fn md_snapshot_mid_run_continues_bit_identically() {
    interrupted_equals_uninterrupted(&|| Box::new(md()));
}

// ============================================================================
// (e) check_state refuses bad payloads and changes nothing
// ============================================================================

fn refuses<P: Participant>(p: &P, bad: &[u8], what: &str) -> StateError {
    let before = state_of(p);
    let e = p.check_state(bad).expect_err(what);
    assert_eq!(
        state_of(p),
        before,
        "{what}: check_state changed the participant"
    );
    e
}

fn flipped(blob: &[u8], at: usize) -> Vec<u8> {
    let mut b = blob.to_vec();
    b[at] ^= 0x01;
    b
}

#[test]
fn crowd_check_state_refuses_truncated_and_corrupted_payloads() {
    let p = crowd();
    let blob = state_of(&p);
    for cut in [0, 3, 4, 11, 12, 19, blob.len() - 16, blob.len() - 1] {
        assert!(
            matches!(
                refuses(&p, &blob[..cut], "truncated"),
                StateError::Length { .. }
            ),
            "cut {cut}"
        );
    }
    let mut long = blob.clone();
    long.push(0);
    assert!(matches!(
        refuses(&p, &long, "one byte long"),
        StateError::Length { .. }
    ));
    assert_eq!(
        refuses(&p, &flipped(&blob, 0), "version"),
        StateError::InvalidValue
    );
    assert_eq!(
        refuses(&p, &flipped(&blob, 5), "digest"),
        StateError::InvalidValue
    );
    // pedestrian count says one more than the bytes hold
    assert!(matches!(
        refuses(&p, &flipped(&blob, 12), "count"),
        StateError::Length { .. }
    ));

    // a payload of a crowd with other walls or another model
    let mut other_walls = corridor_walls();
    other_walls[1].end.x = Fix128::from_int(11);
    let w = CrowdParticipant::new(
        model(),
        corridor_crowd(),
        other_walls,
        NeighborSearch::CellList,
        Some(Fix128::from_ratio(13, 10)),
    )
    .expect("crowd");
    assert_eq!(
        refuses(&p, &state_of(&w), "walls"),
        StateError::InvalidValue
    );
    let other_model =
        SocialForce::new(hfv_params(), hfv_params(), Fix128::ONE, Fix128::from_int(2))
            .expect("model");
    let m = CrowdParticipant::new(
        other_model,
        corridor_crowd(),
        corridor_walls(),
        NeighborSearch::CellList,
        Some(Fix128::from_ratio(13, 10)),
    )
    .expect("crowd");
    assert_eq!(
        refuses(&p, &state_of(&m), "model"),
        StateError::InvalidValue
    );
    let s = CrowdParticipant::new(
        model(),
        corridor_crowd(),
        corridor_walls(),
        NeighborSearch::Direct,
        Some(Fix128::from_ratio(13, 10)),
    )
    .expect("crowd");
    assert_eq!(
        refuses(&p, &state_of(&s), "search"),
        StateError::InvalidValue
    );
    let c = CrowdParticipant::new(
        model(),
        corridor_crowd(),
        corridor_walls(),
        NeighborSearch::CellList,
        None,
    )
    .expect("crowd");
    assert_eq!(
        refuses(&p, &state_of(&c), "speed cap"),
        StateError::InvalidValue
    );

    // a payload with another number of pedestrians is state, and accepted
    let fewer = CrowdParticipant::new(
        model(),
        corridor_crowd()[..2].to_vec(),
        corridor_walls(),
        NeighborSearch::CellList,
        Some(Fix128::from_ratio(13, 10)),
    )
    .expect("crowd");
    p.check_state(&state_of(&fewer))
        .expect("pedestrian count is state");
}

#[test]
fn md_check_state_refuses_truncated_and_corrupted_payloads() {
    let p = md();
    let blob = state_of(&p);
    for cut in [0, 3, 4, 11, 12, blob.len() - 16, blob.len() - 1] {
        assert!(
            matches!(
                refuses(&p, &blob[..cut], "truncated"),
                StateError::Length { .. }
            ),
            "cut {cut}"
        );
    }
    let mut long = blob.clone();
    long.push(0);
    assert!(matches!(
        refuses(&p, &long, "one byte long"),
        StateError::Length { .. }
    ));
    assert_eq!(
        refuses(&p, &flipped(&blob, 0), "version"),
        StateError::InvalidValue
    );
    assert_eq!(
        refuses(&p, &flipped(&blob, 5), "digest"),
        StateError::InvalidValue
    );
    // box length x carried in the header (bytes 12..28)
    assert_eq!(
        refuses(&p, &flipped(&blob, 20), "box"),
        StateError::InvalidValue
    );
    // cutoff carried in the header (bytes 60..76)
    assert_eq!(
        refuses(&p, &flipped(&blob, 68), "cutoff"),
        StateError::InvalidValue
    );

    // a system in another box
    let mut sys_box = md_system().positions().to_vec();
    sys_box.truncate(2);
    let other = VelocityVerlet::new(
        lj(),
        PeriodicBox::cubic(Fix128::from_int(7)).expect("box"),
        sys_box,
        vec![Vec3Fix::ZERO; 2],
        vec![Fix128::ONE; 2],
    )
    .expect("system");
    let q = MdParticipant::new(other, Fix128::ONE).expect("participant");
    assert_eq!(
        refuses(&p, &state_of(&q), "other box"),
        StateError::InvalidValue
    );
    // another k_B
    let r = MdParticipant::new(md_system(), Fix128::from_int(2)).expect("participant");
    assert_eq!(
        refuses(&p, &state_of(&r), "other k_B"),
        StateError::InvalidValue
    );

    // a non-positive mass in an otherwise valid payload
    let header = 4 + 8 + 48 + 16 + 8;
    let mut bad_mass = blob.clone();
    for b in &mut bad_mass[header..header + 16] {
        *b = 0;
    }
    assert_eq!(
        refuses(&p, &bad_mass, "zero mass"),
        StateError::InvalidValue
    );
}

// ============================================================================
// (f) a failing substep leaves the participant unchanged
// ============================================================================

/// `τ ≤ 0` for a pedestrian set after construction: `SocialForce::step`
/// refuses it with the crowd unchanged; the participant reports
/// `InvalidState`, `run_substep` freezes it and reports the fault once.
#[test]
fn a_crowd_with_an_invalid_pedestrian_faults_and_is_unchanged() {
    let mut p = crowd();
    drive_one(&mut p, 20);
    p.pedestrians_mut()[4].relaxation_time_s = Fix128::ZERO;
    let before = state_of(&p);
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(p)];
    let faults = drive(&mut ps, 0, 5, h_sub());
    assert_eq!(
        faults,
        vec![WorldFault::Participant {
            index: 0,
            kind: CROWD_PARTICIPANT_KIND,
            fault: ParticipantFault::InvalidState,
        }]
    );
    assert_eq!(boxed_state(ps[0].as_ref()), before);
}

fn drive_one<P: Participant + 'static + Clone>(p: &mut P, n: usize) {
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(p.clone())];
    drive(&mut ps, 0, n, h_sub());
    let blob = boxed_state(ps[0].as_ref());
    p.read_state(&blob);
}

/// Two particles fired at each other: within one step the separation falls
/// to `≈ 0.01 σ`, where `(σ/r)¹²` leaves the Fix128 range. The potential
/// reports `Overflow`, `VelocityVerlet::step` keeps its state and the
/// participant reports `OutOfRange`.
#[test]
fn an_md_overflow_faults_and_is_unchanged() {
    // r = 1, v_rel = −(1 − 0.01)/h (+ the repulsive half kick, ≈ 24·h/2 per particle)
    let h = h_sub();
    let v = Fix128::ZERO - (Fix128::ONE - fx(0.01)) / h - fx(24.0) * h;
    let p = lj_pair(Fix128::ONE, v);
    let before = state_of(&p);
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(p)];
    let faults = drive(&mut ps, 0, 3, h);
    assert_eq!(
        faults,
        vec![WorldFault::Participant {
            index: 0,
            kind: MD_PARTICIPANT_KIND,
            fault: ParticipantFault::OutOfRange,
        }]
    );
    assert_eq!(boxed_state(ps[0].as_ref()), before);
}

/// Coincident particles after a drift: the potential reports a non-positive
/// distance, the participant `InvalidState`.
#[test]
fn md_coincident_particles_fault_as_invalid_state() {
    let h = h_sub();
    // F(1) = 24 (repulsive); each half kick changes v_rel by +24·h; the
    // drift uses the velocity after the first half kick
    let v = Fix128::ZERO - Fix128::ONE / h - fx(24.0) * h;
    let p = lj_pair(Fix128::ONE, v);
    let mut ps: Vec<Box<dyn Participant>> = vec![Box::new(p)];
    let faults = drive(&mut ps, 0, 1, h);
    assert_eq!(faults.len(), 1);
    assert!(matches!(
        faults[0],
        WorldFault::Participant {
            fault: ParticipantFault::InvalidState | ParticipantFault::OutOfRange,
            ..
        }
    ));
}

#[test]
fn an_empty_crowd_reports_count_zero_and_no_mean_speed() {
    let p = CrowdParticipant::new(
        model(),
        Vec::new(),
        Vec::new(),
        NeighborSearch::Direct,
        None,
    )
    .expect("crowd");
    let mut obs = ObservationSink::new();
    p.observe(&mut obs);
    assert_eq!(obs.values(), &[(CROWD_OBS_COUNT, Fix128::ZERO)]);
}

#[test]
fn a_single_md_particle_reports_no_temperature() {
    let sys = VelocityVerlet::new(
        lj(),
        PeriodicBox::cubic(Fix128::from_int(6)).expect("box"),
        vec![Vec3Fix::from_int(1, 1, 1)],
        vec![Vec3Fix::from_int(1, 0, 0)],
        vec![Fix128::from_int(2)],
    )
    .expect("system");
    let p = MdParticipant::new(sys, Fix128::ONE).expect("participant");
    let obs = md_observation(&p);
    assert_eq!(
        obs.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
        vec![MD_OBS_KINETIC, MD_OBS_POTENTIAL, MD_OBS_TOTAL]
    );
    assert_eq!(channel(&obs, MD_OBS_KINETIC), 1.0);
}

#[test]
fn constructors_refuse_invalid_configuration() {
    assert!(MdParticipant::new(md_system(), Fix128::ZERO).is_err());
    assert!(CrowdParticipant::new(
        model(),
        corridor_crowd(),
        Vec::new(),
        NeighborSearch::Direct,
        Some(Fix128::ZERO)
    )
    .is_err());
    let mut bad = corridor_crowd();
    bad[0].mass_kg = Fix128::ZERO;
    assert!(CrowdParticipant::new(model(), bad, Vec::new(), NeighborSearch::Direct, None).is_err());
}
