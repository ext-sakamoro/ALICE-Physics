//! Oracles for the range-checked steps `molecular_dynamics::VelocityVerlet::try_step`
//! and `crowd_force::SocialForce::try_step`.
//!
//! # Claims
//!
//! - In range, `try_step` is bit-identical to `step`: the same state after
//!   every step (`assert_eq!` on every `Fix128`), over physical scenes and
//!   over scenes whose values sit in the window `|x| ∈ [3037000499, 2³¹·⁵)`
//!   on the negative side, where `x²` is just below `2⁶³` and the integer
//!   part of the floor product alone already exceeds it.
//! - Out of range, `try_step` returns `Err` and the state is unchanged
//!   (every field equal to the one before the call), while `step` returns
//!   `Ok` with a wrapped value (its behaviour is not changed).
//! - An MD pair at `d ≠ 0` whose `|d|²` rounds to 0 is
//!   `MdStepError::PairBelowResolution` under `try_step` (`step` keeps
//!   reporting it as a coincident pair).
//!
//! Author: Moroya Sakamoto

use alice_physics::crowd_force::{
    CrowdForceError, CrowdStepError, InteractionParams, NeighborSearch, Pedestrian, SocialForce,
    WallSegment,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::molecular_dynamics::{MdError, MdStepError, PeriodicBox, VelocityVerlet};
use alice_physics::pair_potential::{
    LennardJones, Morse, PairPotential, PairPotentialError, ShiftMode, Truncated,
};
use alice_physics::physics2d::Vec2Fix;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

/// `−3037000499.75`: `x² ≈ 9.2233720340e18 < 2⁶³`, in the window where the
/// floor product of the integer parts alone is above `2⁶³`.
fn window_neg() -> Fix128 {
    Fix128::from_raw(-3_037_000_500, 1 << 62)
}

/// Deterministic numbers in `[−1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

// ============================================================================
// Molecular dynamics
// ============================================================================

type MdState = (
    Vec<Vec3Fix>,
    Vec<Vec3Fix>,
    Vec<Vec3Fix>,
    Vec<Fix128>,
    Fix128,
);

fn md_state<P: PairPotential>(s: &VelocityVerlet<P>) -> MdState {
    (
        s.positions().to_vec(),
        s.velocities().to_vec(),
        s.forces().to_vec(),
        s.masses().to_vec(),
        s.potential_energy(),
    )
}

/// Steps one copy with `step` and one with `try_step` and compares the
/// states after every step.
fn md_bit_identical<P: PairPotential + Clone>(system: VelocityVerlet<P>, dt: Fix128, steps: usize) {
    let mut a = system.clone();
    let mut b = system;
    for k in 0..steps {
        a.step(dt).expect("step in range");
        b.try_step(dt).expect("try_step in range");
        assert_eq!(md_state(&a), md_state(&b), "state differs after step {k}");
    }
}

fn lj() -> Truncated<LennardJones> {
    Truncated::new(
        LennardJones::new(Fix128::ONE, Fix128::ONE).expect("lj"),
        fx(2.5),
        ShiftMode::EnergyShift,
    )
    .expect("truncated")
}

/// A jittered 4×4×4 lattice of spacing 1.5 in a box of 6, LJ, random
/// velocities of either sign, three masses.
fn lj_lattice() -> VelocityVerlet<LennardJones> {
    let mut rng = Lcg(7);
    let mut x = Vec::new();
    let mut v = Vec::new();
    let mut m = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let c = |n: i32, r: f64| fx(0.75 + 1.5 * f64::from(n) + 0.1 * r);
                x.push(Vec3Fix::new(
                    c(i, rng.next()),
                    c(j, rng.next()),
                    c(k, rng.next()),
                ));
                v.push(Vec3Fix::new(
                    fx(0.5 * rng.next()),
                    fx(0.5 * rng.next()),
                    fx(0.5 * rng.next()),
                ));
                m.push(fx([1.0, 2.0, 0.75][(i + j + k) as usize % 3]));
            }
        }
    }
    VelocityVerlet::new(lj(), PeriodicBox::cubic(fx(6.0)).expect("box"), x, v, m).expect("system")
}

#[test]
fn md_try_step_equals_step_on_an_lj_lattice() {
    md_bit_identical(lj_lattice(), Fix128::from_ratio(1, 512), 200);
}

#[test]
fn md_try_step_equals_step_on_a_morse_cluster_crossing_the_box_edge() {
    let morse = Truncated::new(
        Morse::new(fx(1.5), fx(2.0), fx(1.1)).expect("morse"),
        fx(2.0),
        ShiftMode::ForceShift,
    )
    .expect("truncated");
    let system = VelocityVerlet::new(
        morse,
        PeriodicBox::new(Vec3Fix::new(fx(5.0), fx(6.5), fx(4.25))).expect("box"),
        vec![
            Vec3Fix::new(fx(4.9), fx(0.1), fx(4.2)),
            Vec3Fix::new(fx(0.8), fx(6.3), fx(0.1)),
            Vec3Fix::new(fx(0.2), fx(1.0), fx(3.9)),
            Vec3Fix::new(fx(2.5), fx(3.2), fx(2.0)),
        ],
        vec![
            Vec3Fix::new(fx(0.7), fx(-0.4), fx(0.9)),
            Vec3Fix::new(fx(-0.6), fx(0.5), fx(-0.8)),
            Vec3Fix::new(fx(-0.9), fx(-0.2), fx(0.3)),
            Vec3Fix::new(fx(0.1), fx(0.6), fx(-0.5)),
        ],
        vec![fx(1.0), fx(0.5), fx(2.0), fx(1.25)],
    )
    .expect("system");
    md_bit_identical(system, Fix128::from_ratio(1, 128), 400);
}

/// Box `2³³`, cutoff `2³¹`: particle 1 is `3037000499.75` from particle 0
/// along `−x` (minimum image keeps the sign), so `|d|²` is just below `2⁶³`
/// and outside the cutoff; particle 1 also moves at `v_x = −3037000499.75`.
/// Particles 0 and 2 interact.
#[test]
fn md_try_step_equals_step_in_the_negative_square_window() {
    let pot = Truncated::new(
        LennardJones::new(Fix128::ONE, Fix128::ONE).expect("lj"),
        Fix128::from_int(1 << 31),
        ShiftMode::None,
    )
    .expect("truncated");
    let q = fx(0.25);
    let system = VelocityVerlet::new(
        pot,
        PeriodicBox::cubic(Fix128::from_int(1 << 33)).expect("box"),
        vec![
            Vec3Fix::new(q, q, q),
            Vec3Fix::new(q - window_neg(), q, q),
            Vec3Fix::new(fx(1.4), fx(0.9), fx(0.6)),
        ],
        vec![
            Vec3Fix::new(fx(0.1), Fix128::ZERO, fx(-0.2)),
            Vec3Fix::new(window_neg(), Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(fx(-0.3), fx(0.2), Fix128::ZERO),
        ],
        vec![Fix128::ONE, fx(2.0), fx(0.5)],
    )
    .expect("system");
    let d = system
        .periodic_box()
        .minimum_image(system.positions()[0] - system.positions()[1]);
    assert!(
        d.x.is_negative() && d.x.hi == -3_037_000_500,
        "d.x = {:?}",
        d.x
    );
    // one step at 2⁻⁴⁰ moves particle 1 by about 0.003, staying in the window
    md_bit_identical(system, Fix128::from_raw(0, 1 << 24), 4);
}

fn assert_md_unchanged<P: PairPotential>(s: &VelocityVerlet<P>, before: &MdState) {
    assert_eq!(&md_state(s), before, "a failed try_step changed the state");
}

/// Morse pair at `r = 0.6` (force ≈ 28) with particle 0 of mass `2⁻⁶²`: the
/// kick `F h/(2m) ≈ 28·2⁶¹` is out of range.
#[test]
fn md_kick_out_of_range_is_overflow_and_leaves_the_state() {
    let morse = Truncated::new(
        Morse::new(fx(1.5), fx(2.0), fx(1.1)).expect("morse"),
        fx(2.0),
        ShiftMode::EnergyShift,
    )
    .expect("truncated");
    let system = VelocityVerlet::new(
        morse,
        PeriodicBox::new(Vec3Fix::new(fx(5.0), fx(6.5), fx(4.25))).expect("box"),
        vec![
            Vec3Fix::new(fx(1.0), fx(1.0), fx(1.0)),
            Vec3Fix::new(fx(1.6), fx(1.0), fx(1.0)),
        ],
        vec![Vec3Fix::ZERO, Vec3Fix::ZERO],
        vec![Fix128::from_raw(0, 4), Fix128::ONE],
    )
    .expect("system");
    let mut checked = system.clone();
    let before = md_state(&checked);
    assert_eq!(checked.try_step(Fix128::ONE), Err(MdStepError::Overflow));
    assert_md_unchanged(&checked, &before);
    // `step` is unchanged: it accepts the step and wraps
    let mut plain = system;
    assert_eq!(plain.step(Fix128::ONE), Ok(()));
}

/// Free particle at `v = 2⁶²` with `h = 4`: the drift `v h = 2⁶⁴` is out of
/// range and wraps to exactly 0, so only the drift check can see it.
#[test]
fn md_drift_out_of_range_is_overflow_and_leaves_the_state() {
    let mut s = VelocityVerlet::new(
        lj(),
        PeriodicBox::cubic(fx(6.0)).expect("box"),
        vec![
            Vec3Fix::new(fx(1.0), fx(1.0), fx(1.0)),
            Vec3Fix::new(fx(4.0), fx(4.0), fx(4.0)),
        ],
        vec![
            Vec3Fix::new(Fix128::ZERO, Fix128::from_int(1 << 62), Fix128::ZERO),
            Vec3Fix::ZERO,
        ],
        vec![Fix128::ONE, Fix128::ONE],
    )
    .expect("system");
    let mut plain = s.clone();
    let before = md_state(&s);
    assert_eq!(s.try_step(fx(4.0)), Err(MdStepError::Overflow));
    assert_md_unchanged(&s, &before);
    assert_eq!(plain.step(fx(4.0)), Ok(()));
    // the wrapped drift is 0: `step` leaves the particle where it was
    assert_eq!(plain.positions()[0], before.0[0]);
}

/// Mass `2⁻⁶⁴` with `h = 4`: the kick factor `h/(2m) = 2⁶⁵` is out of range.
#[test]
fn md_kick_factor_out_of_range_is_overflow_and_leaves_the_state() {
    let mut s = VelocityVerlet::new(
        lj(),
        PeriodicBox::cubic(fx(6.0)).expect("box"),
        vec![Vec3Fix::new(fx(1.0), fx(1.0), fx(1.0))],
        vec![Vec3Fix::ZERO],
        vec![Fix128::from_raw(0, 1)],
    )
    .expect("system");
    let before = md_state(&s);
    assert_eq!(s.try_step(fx(4.0)), Err(MdStepError::Overflow));
    assert_md_unchanged(&s, &before);
}

/// Two particles whose drift brings them to a gap of `2⁻⁴⁰` (the velocity
/// is solved for that gap, as in the participant oracle).
fn md_pair_approaching(h: Fix128, final_gap: Fix128) -> VelocityVerlet<LennardJones> {
    let gap = fx(1.2);
    let make = |v: Fix128| {
        VelocityVerlet::new(
            lj(),
            PeriodicBox::cubic(fx(6.0)).expect("box"),
            vec![
                Vec3Fix::new(fx(1.0), fx(2.0), fx(2.0)),
                Vec3Fix::new(fx(1.0) + gap, fx(2.0), fx(2.0)),
            ],
            vec![Vec3Fix::new(v, Fix128::ZERO, Fix128::ZERO), Vec3Fix::ZERO],
            vec![Fix128::ONE; 2],
        )
        .expect("system")
    };
    let kick = make(Fix128::ZERO).forces()[0].x * h.half();
    let v = (gap - final_gap) / h - kick - kick;
    make(v)
}

#[test]
fn md_pair_below_the_length_resolution_is_refused_and_leaves_the_state() {
    let h = Fix128::from_ratio(1, 64);
    let gap = Fix128::from_raw(0, 1 << 24);
    let mut s = md_pair_approaching(h, gap);
    let mut plain = s.clone();
    let before = md_state(&s);
    assert_eq!(
        s.try_step(h),
        Err(MdStepError::PairBelowResolution { i: 0, j: 1 })
    );
    assert_md_unchanged(&s, &before);
    // `step` keeps its report: the rounded distance 0 is a coincident pair
    assert_eq!(
        plain.step(h),
        Err(MdError::Potential {
            i: 0,
            j: 1,
            error: PairPotentialError::NonPositiveDistance
        })
    );
    assert_eq!(md_state(&plain), before);
}

#[test]
fn md_try_step_reports_the_errors_of_step() {
    let mut s = lj_lattice();
    let before = md_state(&s);
    assert_eq!(
        s.try_step(Fix128::ZERO),
        Err(MdStepError::Md(MdError::NonPositiveTimestep))
    );
    assert_md_unchanged(&s, &before);
}

// ============================================================================
// Crowd
// ============================================================================

fn params() -> InteractionParams {
    InteractionParams {
        strength_n: Fix128::from_int(1500),
        range_m: Fix128::from_ratio(1, 10),
        body_stiffness: Fix128::from_int(90_000),
        sliding_friction: Fix128::from_int(150_000),
    }
}

fn model() -> SocialForce {
    SocialForce::new(
        params(),
        params(),
        Fix128::from_ratio(3, 10),
        Fix128::from_int(3),
    )
    .expect("valid model")
}

fn walls() -> Vec<WallSegment> {
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
        WallSegment {
            start: p(0, -3),
            end: p(0, -3),
        },
    ]
}

fn ped(x: f64, y: f64, dir: (i64, i64), m: i64, v0: f64) -> Pedestrian {
    Pedestrian {
        position: Vec2Fix::new(fx(x), fx(y)),
        velocity: Vec2Fix::new(fx(0.1 * x.signum()), fx(-0.05)),
        radius_m: fx(0.25),
        mass_kg: Fix128::from_int(m),
        desired_speed_m_s: fx(v0),
        desired_direction: Vec2Fix::new(Fix128::from_int(dir.0), Fix128::from_int(dir.1)),
        relaxation_time_s: fx(0.4),
    }
}

/// Two groups walking at each other in a corridor (contacts and wall
/// contacts happen), one standing pedestrian.
fn people() -> Vec<Pedestrian> {
    vec![
        ped(-3.0, -2.6, (1, 1), 70, 1.4),
        ped(-3.4, -3.5, (1, 1), 85, 1.1),
        ped(-2.0, -1.0, (1, 1), 60, 1.6),
        ped(-4.1, -4.4, (1, 1), 75, 1.2),
        ped(3.0, 3.2, (-1, -1), 90, 1.3),
        ped(2.6, 2.1, (-1, -1), 65, 1.5),
        ped(3.9, 4.6, (-1, -1), 80, 1.0),
        ped(0.1, 0.2, (0, 0), 72, 0.0),
        ped(0.3, 0.3, (1, -2), 66, 1.2),
    ]
}

fn crowd_bit_identical(
    start: &[Pedestrian],
    walls: &[WallSegment],
    h: Fix128,
    search: NeighborSearch,
    cap: Option<Fix128>,
    steps: usize,
) {
    let m = model();
    let mut a = start.to_vec();
    let mut b = start.to_vec();
    for k in 0..steps {
        m.step(&mut a, walls, h, search, cap)
            .expect("step in range");
        m.try_step(&mut b, walls, h, search, cap)
            .expect("try_step in range");
        assert_eq!(
            a, b,
            "crowd differs after step {k} ({search:?}, cap {cap:?})"
        );
    }
}

#[test]
fn crowd_try_step_equals_step_in_a_corridor() {
    let h = Fix128::from_ratio(1, 200);
    for search in [NeighborSearch::Direct, NeighborSearch::CellList] {
        for cap in [None, Some(Fix128::from_ratio(13, 10))] {
            crowd_bit_identical(&people(), &walls(), h, search, cap, 600);
        }
    }
}

/// Pedestrian 0 sits `3037000499.75` m from pedestrian 1 and from the point
/// wall along `−x` (`3037000499.35` m from pedestrian 2), walks at `v_x = −3037000499.75` and wants to go along a
/// desired direction of that length: every squared length on its side is
/// just below `2⁶³` with a negative component.
#[test]
fn crowd_try_step_equals_step_in_the_negative_square_window() {
    let far = Pedestrian {
        position: Vec2Fix::new(window_neg(), Fix128::ZERO),
        velocity: Vec2Fix::new(window_neg(), Fix128::ZERO),
        radius_m: fx(0.3),
        mass_kg: Fix128::from_int(80),
        desired_speed_m_s: fx(1.3),
        desired_direction: Vec2Fix::new(window_neg(), Fix128::ZERO),
        relaxation_time_s: Fix128::from_int(1 << 20),
    };
    let near = Pedestrian {
        position: Vec2Fix::ZERO,
        ..ped(0.0, 0.0, (1, 0), 70, 1.0)
    };
    let close = ped(-0.4, 0.1, (1, 0), 65, 1.2);
    let point_wall = vec![WallSegment {
        start: Vec2Fix::ZERO,
        end: Vec2Fix::ZERO,
    }];
    let crowd = [far, near, close];
    let h = Fix128::from_raw(0, 1 << 20);
    for search in [NeighborSearch::Direct, NeighborSearch::CellList] {
        for cap in [None, Some(Fix128::from_int(1 << 40))] {
            crowd_bit_identical(&crowd, &point_wall, h, search, cap, 1);
        }
    }
    // the cap 1.3·v0 does bind at this speed: the scaled velocity is compared too
    crowd_bit_identical(
        &crowd,
        &point_wall,
        h,
        NeighborSearch::Direct,
        Some(Fix128::from_ratio(13, 10)),
        3,
    );
}

fn crowd_overflow_leaves_the_crowd(
    start: Vec<Pedestrian>,
    walls: &[WallSegment],
    h: Fix128,
    search: NeighborSearch,
) {
    let m = model();
    let mut checked = start.clone();
    assert_eq!(
        m.try_step(&mut checked, walls, h, search, None),
        Err(CrowdStepError::Overflow)
    );
    assert_eq!(checked, start, "a failed try_step changed the crowd");
    // `step` is unchanged: it accepts the step and wraps
    let mut plain = start;
    assert_eq!(m.step(&mut plain, walls, h, search, None), Ok(()));
}

/// `v = 2⁶¹`, `τ = 2⁻²⁴`: the driving force `m v/τ ≈ 70·2⁸⁵` is out of range.
#[test]
fn crowd_driving_force_out_of_range_is_overflow() {
    let mut p = people()[0];
    p.velocity = Vec2Fix::new(Fix128::from_int(1 << 61), Fix128::ZERO);
    p.desired_direction = Vec2Fix::new(Fix128::ONE, Fix128::ZERO);
    p.relaxation_time_s = Fix128::from_raw(0, 1 << 40);
    for search in [NeighborSearch::Direct, NeighborSearch::CellList] {
        crowd_overflow_leaves_the_crowd(vec![p], &[], Fix128::from_ratio(1, 100), search);
    }
}

/// Two overlapping pedestrians (body force ≈ 9000 N), one of mass `2⁻⁶⁰`:
/// `F h/m ≈ 9000·2⁵³` is out of range.
#[test]
fn crowd_velocity_update_out_of_range_is_overflow() {
    let mut a = ped(0.0, 0.0, (1, 0), 70, 1.0);
    let b = ped(0.4, 0.0, (-1, 0), 70, 1.0);
    a.mass_kg = Fix128::from_raw(0, 1 << 4);
    for search in [NeighborSearch::Direct, NeighborSearch::CellList] {
        crowd_overflow_leaves_the_crowd(vec![b, a], &[], Fix128::from_ratio(1, 100), search);
    }
}

/// `x = 2⁶² + 2²³`, `v = 2⁶²` (driving `−m v/τ = −2²²` with `τ = 2⁴⁰`),
/// `h = 1`: the position update `x + v h = 2⁶³ + 2²²` is out of range.
#[test]
fn crowd_position_update_out_of_range_is_overflow() {
    let big = Fix128::from_int(1 << 62);
    let p = Pedestrian {
        position: Vec2Fix::new(Fix128::ZERO, big + Fix128::from_int(1 << 23)),
        velocity: Vec2Fix::new(Fix128::ZERO, big),
        radius_m: fx(0.25),
        mass_kg: Fix128::ONE,
        desired_speed_m_s: Fix128::ZERO,
        desired_direction: Vec2Fix::ZERO,
        relaxation_time_s: Fix128::from_int(1 << 40),
    };
    crowd_overflow_leaves_the_crowd(vec![p], &[], Fix128::ONE, NeighborSearch::Direct);
}

/// Two pedestrians in contact (overlap `g = 0.1`) moving tangentially in
/// opposite directions at `2⁶⁰` m/s: the sliding friction
/// `κ g Δv_t = 1.5e4 · 2⁶¹ = 1875 · 2⁶⁴` is out of range and wraps to
/// exactly 0, so only the friction check can see it (the driving term
/// `m v/τ = 70 · 2²⁰` with `τ = 2⁴⁰` is in range).
#[test]
fn crowd_contact_force_out_of_range_is_overflow() {
    let mut a = ped(0.0, 0.0, (1, 0), 70, 1.0);
    let mut b = ped(0.4, 0.0, (-1, 0), 70, 1.0);
    a.velocity = Vec2Fix::new(Fix128::ZERO, Fix128::from_int(1 << 60));
    b.velocity = Vec2Fix::new(Fix128::ZERO, -Fix128::from_int(1 << 60));
    a.relaxation_time_s = Fix128::from_int(1 << 40);
    b.relaxation_time_s = Fix128::from_int(1 << 40);
    crowd_overflow_leaves_the_crowd(
        vec![a, b],
        &[],
        Fix128::from_raw(0, 1),
        NeighborSearch::Direct,
    );
}

#[test]
fn crowd_try_step_reports_the_errors_of_step() {
    let m = model();
    let mut crowd = people();
    crowd[3].mass_kg = Fix128::ZERO;
    let before = crowd.clone();
    assert_eq!(
        m.try_step(&mut crowd, &walls(), fx(0.01), NeighborSearch::Direct, None),
        Err(CrowdStepError::Crowd(CrowdForceError::NonPositiveMass {
            index: 3
        }))
    );
    assert_eq!(crowd, before);
    assert_eq!(
        m.try_step(
            &mut crowd,
            &walls(),
            Fix128::ZERO,
            NeighborSearch::Direct,
            None
        ),
        Err(CrowdStepError::Crowd(CrowdForceError::NonPositiveTimeStep))
    );
    assert_eq!(crowd, before);
}
