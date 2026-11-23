//! Oracles for the water film (`water_film`): still water stands level at the height its
//! volume fixes, volume is conserved to the last quantum, a groove holds water up to its rim
//! and no lower and fills in the time the rain rate gives, films and deeper flow carry the
//! lubrication flux, a free overfall drains at the weir rate, inertial flows are refused, a
//! rewound run and a rerun are bit identical, and the order the faces are visited does not
//! change the state.
//!
//! Every expected value here comes from a closed form or a ledger written independently of
//! the stepping code; none is taken from a run of the film.

use alice_physics_core::sdf_collider::SdfField;
use alice_physics_core::water_film::{
    isqrt_u128, join_step, metres_to_quanta, quanta_to_metres, simultaneous_transfer, FilmBed,
    FilmBuildError, FilmConfig, FilmEdge, FilmScheme, FluxConstants, WaterFilm, DEPTH_QUANTUM_BITS,
    MAX_FACE_DEPTH, MAX_LUBRICATION_DIVISOR,
};
use alice_physics_core::world_participant::{Participant, ParticipantFault, StateError};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix};
use sha2::{Digest, Sha256};

const CELL_MM: i64 = 10;
/// One depth quantum in metres (`2⁻³²`, exact in f64).
const QUANTUM_M: f64 = 1.0 / 4_294_967_296.0;
const RIM_MM: i64 = 0;
const FLOOR_MM: i64 = -30;
const BEVEL_MM: i64 = -15;

fn mm(v: i64) -> Fix128 {
    Fix128::from_ratio(v, 1000)
}

/// A groove along x in the middle row of a 3-row strip: columns `first..=last` are the floor
/// (30 mm below the rim), the column on each side a 45° bevel (half way), everything else
/// plateau at the rim height.
fn groove_bed(nx: usize, first: usize, last: usize, groove_only: bool) -> FilmBed {
    let nz = if groove_only { 1 } else { 3 };
    let row = if groove_only { 0 } else { 1 };
    let mut h = vec![mm(RIM_MM); nx * nz];
    for i in 0..nx {
        let v = if i >= first && i <= last {
            FLOOR_MM
        } else if i + 1 == first || i == last + 1 {
            BEVEL_MM
        } else {
            RIM_MM
        };
        h[i + row * nx] = mm(v);
    }
    FilmBed::from_heights(nx, nz, mm(CELL_MM), (Fix128::ZERO, Fix128::ZERO), &h, 1).expect("bed")
}

/// Still water in row `row` up to `level` (quanta): every cell whose bed is below it holds
/// `level − b` (a level pool, inside the model at any depth).
fn fill_level(f: &mut WaterFilm, row: usize, level: i64) {
    let nx = f.bed().nx();
    for i in 0..nx {
        let b = f.bed().heights()[i + row * nx];
        if b < level {
            f.set_initial_depth(i, row, level - b);
        }
    }
}

/// Thin films (`depth`, below the deepest film) on the plateau rows 0 and 2.
fn wet_plateau(f: &mut WaterFilm, depth: i64) {
    for k in [0usize, 2] {
        for i in 0..f.bed().nx() {
            f.set_initial_depth(i, k, depth);
        }
    }
}

fn film(bed: FilmBed, scheme: FilmScheme, edge: FilmEdge) -> WaterFilm {
    WaterFilm::new(FilmConfig::water(scheme, edge), bed).expect("film")
}

/// The level `η*` (quanta) a volume `v` (quanta over one cell) stands at over `beds`
/// (cells of equal area): the largest `η` with `Σ max(η − b, 0) ≤ v`, by bisection on
/// integers. oracle: hydrostatics, a still free surface is level.
fn level_for(beds: &[i64], v: i64) -> i64 {
    let fill = |eta: i64| -> i128 { beds.iter().map(|b| i128::from((eta - b).max(0))).sum() };
    let (mut lo, mut hi) = (*beds.iter().min().expect("beds"), i64::MAX / 4);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fill(mid) <= i128::from(v) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Substep (at `Δt_max`) until a whole cycle of the four face sets moves nothing; returns the
/// substeps taken. Panics if the film never comes to rest.
fn settle(f: &mut WaterFilm, cap: u64) -> u64 {
    let dt = f.config().max_substep;
    let mut quiet = 0;
    for n in 0..cap {
        f.step_once(dt).expect("substep");
        if f.last_moved() == 0 {
            quiet += 1;
            if quiet == 4 {
                return n + 1;
            }
        } else {
            quiet = 0;
        }
    }
    panic!("the film did not come to rest in {cap} substeps");
}

/// `Σ (η² − b²)` over the cells, in quanta² (the potential energy over `g·l²/2`).
fn potential(f: &WaterFilm) -> i128 {
    f.bed()
        .heights()
        .iter()
        .zip(f.depths())
        .map(|(b, d)| {
            let (b, eta) = (i128::from(*b), i128::from(b + d));
            eta * eta - b * b
        })
        .sum()
}

fn state(f: &WaterFilm) -> Vec<u8> {
    let mut out = Vec::new();
    f.write_state(&mut out);
    out
}

/// Water piled 40 mm deep at one end of a closed groove collapses as a dam break (an
/// inertial flow deeper than the film): the model does not carry it, so the advance is
/// refused, names the piled cells, and changes nothing. A still surface levelling to the
/// closed form is checked where the model holds it (a groove filled by rain).
#[test]
fn piled_deep_water_is_refused() {
    let mut f = film(
        groove_bed(12, 3, 8, true),
        FilmScheme::DiffusionWave,
        FilmEdge::Closed,
    );
    let pile = metres_to_quanta(mm(40));
    for i in 3..6 {
        assert!(f.set_initial_depth(i, 0, pile));
    }
    let before = state(&f);
    assert_eq!(f.step_once(f.substep()), Err(ParticipantFault::OutOfRange));
    assert!(
        f.last_band_cells().contains(&5),
        "band {:?}",
        f.last_band_cells()
    );
    assert_eq!(state(&f), before);
}

/// oracle: without a source the potential `Σ g(η² − b²)/2` never rises: every substep of a
/// closed groove and of an open strip draining over its edges lowers it or leaves it.
#[test]
fn the_potential_never_rises_without_a_source() {
    let mut compared = 0;
    for edge in [FilmEdge::Closed, FilmEdge::Open] {
        let mut f = film(groove_bed(14, 4, 9, false), FilmScheme::DiffusionWave, edge);
        // a still pool in the groove, plateau films running into it, and one 0.9 mm cell
        fill_level(&mut f, 1, metres_to_quanta(mm(-10)));
        wet_plateau(&mut f, metres_to_quanta(Fix128::from_ratio(8, 10_000)));
        f.set_initial_depth(0, 0, metres_to_quanta(Fix128::from_ratio(9, 10_000)));
        let dt = f.config().max_substep;
        let mut last = potential(&f);
        let mut fell = 0;
        for n in 0..20_000 {
            f.step_once(dt).expect("substep");
            let p = potential(&f);
            assert!(
                p <= last,
                "{edge:?} substep {n}: potential rose by {}",
                p - last
            );
            fell += usize::from(p < last);
            last = p;
            compared += 1;
        }
        assert!(fell > 100, "{edge:?}: the potential fell only {fell} times");
    }
    // a deep flat pool (four wet neighbours per cell) with a bump of 8 quanta and a bump of
    // 0.15 mm, both within the join step: the pool takes them in and stands level
    let bed = FilmBed::from_heights(
        9,
        9,
        mm(CELL_MM),
        (Fix128::ZERO, Fix128::ZERO),
        &[Fix128::ZERO; 81],
        1,
    )
    .expect("bed");
    let mut f = film(bed, FilmScheme::DiffusionWave, FilmEdge::Closed);
    let base = metres_to_quanta(mm(20));
    for k in 0..9 {
        for i in 0..9 {
            f.set_initial_depth(i, k, base);
        }
    }
    f.set_initial_depth(4, 4, base + 8);
    // within the join step (1 % of 20 mm): a deeper bump would be a wave, outside the model
    f.set_initial_depth(
        2,
        6,
        base + metres_to_quanta(Fix128::from_ratio(15, 100_000)),
    );
    let dt = f.config().max_substep;
    let mut last = potential(&f);
    let mut fell = 0;
    for n in 0..2_000 {
        f.step_once(dt).expect("substep");
        let p = potential(&f);
        assert!(
            p <= last,
            "pool substep {n}: potential rose by {}",
            p - last
        );
        fell += usize::from(p < last);
        last = p;
        compared += 1;
    }
    // the deep pool is one component of faces deeper than the film: it levels at once
    assert!(fell >= 1, "pool: the potential never fell");
    let s: Vec<i64> = f
        .bed()
        .heights()
        .iter()
        .zip(f.depths())
        .map(|(b, d)| b + d)
        .collect();
    let spread = s.iter().max().unwrap_or(&0) - s.iter().min().unwrap_or(&0);
    assert!(
        spread <= 1,
        "pool: surface spread {spread} quanta after levelling"
    );
    assert_eq!(compared, 42_000);
}

/// oracle: no cell gives more than it holds. A 0.9 mm film on a pedestal 10 cm above four
/// lower neighbours spills on all four sides at once (each side alone could carry more than a
/// quarter of the column in one substep); every depth stays at or above zero and the volume
/// is exact at every substep.
#[test]
fn a_column_spilling_four_ways_never_goes_negative() {
    let mut h = vec![Fix128::ZERO; 25];
    h[12] = mm(100);
    let bed =
        FilmBed::from_heights(5, 5, mm(CELL_MM), (Fix128::ZERO, Fix128::ZERO), &h, 1).expect("bed");
    let mut f = film(bed, FilmScheme::DiffusionWave, FilmEdge::Closed);
    let start = metres_to_quanta(Fix128::from_ratio(9, 10_000));
    f.set_initial_depth(2, 2, start);
    let dt = f.config().max_substep;
    let mut compared = 0;
    for n in 0..500 {
        f.step_once(dt).expect("substep");
        assert!(
            f.depths().iter().all(|d| *d >= 0),
            "substep {n}: a depth below zero"
        );
        assert_eq!(f.total(), f.ledger().expected(), "substep {n}");
        compared += 1;
    }
    assert!(f.depths()[12] < start, "the column did not spill");
    assert_eq!(compared, 500);
}

/// oracle: conservation as integers. Through rain, evaporation, point deposits and open
/// edges, the water held equals initial + rain + deposits − evaporated − outflow to the last
/// quantum, at every advance, and no depth is ever negative.
#[test]
fn volume_is_conserved_to_the_last_quantum() {
    for scheme in [FilmScheme::DiffusionWave] {
        let mut cfg = FilmConfig::water(scheme, FilmEdge::Open);
        cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(200));
        cfg.evaporation = FilmConfig::rain_mm_per_hour(Fix128::from_int(70));
        let mut f = WaterFilm::new(cfg, groove_bed(14, 4, 9, false)).expect("film");
        fill_level(&mut f, 1, metres_to_quanta(mm(-10)));
        wet_plateau(&mut f, metres_to_quanta(Fix128::from_ratio(8, 10_000)));
        let mut compared = 0;
        for n in 0..600 {
            if n % 7 == 0 {
                let (x, z) = f.bed().centre(n % 14, 1);
                f.deposit(x, z, Fix128::from_ratio(1, 100_000_000));
            }
            f.advance(Fix128::from_ratio(1, 20)).expect("advance");
            assert_eq!(f.total(), f.ledger().expected(), "{scheme:?} after {n}");
            assert!(
                f.depths().iter().all(|d| *d >= 0),
                "{scheme:?} negative depth"
            );
            compared += 1;
        }
        let l = f.ledger();
        assert!(
            l.rain > 0 && l.evaporated > 0 && l.deposits > 0 && l.outflow > 0,
            "{l:?}"
        );
        assert_eq!(compared, 600);
    }
}

/// oracle: a basin holds water up to its rim. (1) Below the rim nothing leaves the groove:
/// the plateau stays dry to the quantum and nothing crosses the open edges. (2) Filled over
/// the rim, the groove spills until its surface stands at the rim and never below it.
#[test]
fn a_groove_overflows_at_its_rim_height() {
    for scheme in [FilmScheme::DiffusionWave] {
        let rim = metres_to_quanta(mm(RIM_MM));
        // (1) below the rim
        let mut f = film(groove_bed(14, 4, 9, false), scheme, FilmEdge::Open);
        fill_level(&mut f, 1, rim - metres_to_quanta(mm(5)));
        settle(&mut f, 10_000_000);
        assert_eq!(f.ledger().outflow, 0, "{scheme:?}");
        let nx = f.bed().nx();
        for k in [0usize, 2] {
            for i in 0..nx {
                assert_eq!(f.depths()[i + k * nx], 0, "{scheme:?} plateau ({i},{k})");
            }
        }
        // (2) rising past the rim: the groove starts level 2 mm below the rim and rain raises it
        // past the rim; after the rain stops it comes to rest no lower than the rim (less the
        // rest band of `L` quanta) and the water that left over the edges is what came in
        // above the capacity, less the films left on the plateau.
        let mut cfg = FilmConfig::water(scheme, FilmEdge::Open);
        cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(200));
        let mut f = WaterFilm::new(cfg, groove_bed(14, 4, 9, false)).expect("film");
        let start = rim - metres_to_quanta(mm(2));
        for i in 0..nx {
            let b = f.bed().heights()[i + nx];
            if b < start {
                f.set_initial_depth(i, 1, start - b);
            }
        }
        // 200 mm/h for 60 s: 3.3 mm, past the rim
        for _ in 0..1200 {
            f.advance(Fix128::from_ratio(1, 20)).expect("rain");
        }
        let mut cfg = *f.config();
        cfg.rain = Fix128::ZERO;
        let mut dry = WaterFilm::new(cfg, f.bed().clone()).expect("film");
        // the configuration (rain) differs, so carry the water over cell by cell
        for k in 0..f.bed().nz() {
            for i in 0..nx {
                dry.set_initial_depth(i, k, f.depths()[i + k * nx]);
            }
        }
        settle(&mut dry, 10_000_000);
        let groove: Vec<usize> = (0..nx)
            .filter(|i| f.bed().heights()[i + nx] < rim)
            .collect();
        let band = 7 * groove.len() as i64;
        let mut compared = 0;
        for &i in &groove {
            let eta = dry.bed().heights()[i + nx] + dry.depths()[i + nx];
            assert!(
                eta >= rim - band,
                "{scheme:?} cell {i}: {} quanta below the rim",
                rim - eta
            );
            assert!(
                eta - rim < metres_to_quanta(Fix128::from_ratio(1, 10_000)),
                "{scheme:?} cell {i}: {} m over the rim at rest",
                quanta_to_metres(eta - rim).to_f64()
            );
            compared += 1;
        }
        assert!(f.ledger().outflow > 0, "{scheme:?}: nothing spilled");
        assert_eq!(compared, 8);
    }
}

/// oracle: the fill time. Rain at rate `R` over a closed groove of `N` cells fills the
/// capacity below the rim, `V = Σ (rim − b)`, at `t = V / (R·N)` (same cell area on both
/// sides). The first advance at which the groove holds `V` ends within one advance of `t`.
/// The rate is 200 mm/h (in range: a filling pool stays within its level tolerance); the same formula gives the site's 4 and 20 mm/h
/// groove times (also checked against the hand value).
#[test]
fn a_groove_fills_in_the_closed_form_time() {
    let bed = groove_bed(10, 2, 7, true);
    let rim = metres_to_quanta(mm(RIM_MM));
    let cap_q: i64 = bed.heights().iter().map(|b| rim - b).sum();
    let n = bed.nx() as i64;
    let cap_m = quanta_to_metres(cap_q).to_f64();
    // the site's rate: a 30 mm groove of 6 floor + 2 bevel + 2 rim cells at 4 mm/h
    let r4 = FilmConfig::rain_mm_per_hour(Fix128::from_int(4)).to_f64();
    let t4 = cap_m / (r4 * n as f64);
    let hand = (6.0 * 0.030 + 2.0 * 0.015) / (4.0e-3 / 3600.0 * 10.0);
    assert!((t4 - hand).abs() / hand < 1e-6, "{t4} s vs {hand} s");
    for scheme in [FilmScheme::DiffusionWave] {
        let mut cfg = FilmConfig::water(scheme, FilmEdge::Closed);
        // 200 mm/h: a filling pool 1 mm deep stays within its 1 % level tolerance (at
        // 1 800 mm/h it is refused as outside the model)
        cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(200));
        let r = cfg.rain.to_f64();
        let t_closed = cap_m / (r * n as f64);
        let mut f = WaterFilm::new(cfg, bed.clone()).expect("film");
        let h = Fix128::from_ratio(1, 4);
        let mut t = 0.0;
        let mut filled_at = None;
        for _ in 0..2000 {
            if f.advance(h).is_err() {
                panic!("{scheme:?}: refused, band cells {:?}", f.last_band_cells());
            }
            t += 0.25;
            if f.total() >= cap_q {
                filled_at = Some(t);
                break;
            }
        }
        let at = filled_at.expect("filled");
        assert!(
            (at - t_closed).abs() <= 0.25,
            "{scheme:?}: filled at {at} s, closed form {t_closed} s"
        );
        assert_eq!(f.ledger().outflow, 0);
        assert_eq!(f.total(), f.ledger().rain);
    }
}

fn busy_film(scheme: FilmScheme) -> WaterFilm {
    let mut cfg = FilmConfig::water(scheme, FilmEdge::Open);
    cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(200));
    cfg.evaporation = FilmConfig::rain_mm_per_hour(Fix128::from_int(70));
    let mut f = WaterFilm::new(cfg, groove_bed(14, 4, 9, false)).expect("film");
    // a pool 20 mm deep in the groove, films draining into it from the plateau
    fill_level(&mut f, 1, metres_to_quanta(mm(-10)));
    wet_plateau(&mut f, metres_to_quanta(Fix128::from_ratio(8, 10_000)));
    f
}

/// determinism: two runs give the same bytes; a run stopped half way, written, read into a
/// fresh film and continued gives the same bytes as the run that never stopped (the substep
/// count that picks the face set, the carries and the ledger all travel in the payload).
#[test]
fn rewind_and_rerun_are_bit_identical() {
    for scheme in [FilmScheme::DiffusionWave] {
        let h = Fix128::from_ratio(1, 20);
        let run = |steps: usize| {
            let mut f = busy_film(scheme);
            for _ in 0..steps {
                f.advance(h).expect("advance");
            }
            f
        };
        let whole = state(&run(300));
        assert_eq!(whole, state(&run(300)), "{scheme:?} rerun");
        let half = run(150);
        let saved = state(&half);
        let mut fresh = busy_film(scheme);
        assert_eq!(fresh.check_state(&saved), Ok(()));
        fresh.read_state(&saved);
        // the substep count of the next advance comes from the state, so it matches too
        assert_eq!(fresh.subcycles(h), half.subcycles(h), "{scheme:?}");
        for _ in 0..150 {
            fresh.advance(h).expect("advance");
        }
        assert_eq!(state(&fresh), whole, "{scheme:?} restore");
    }
}

/// determinism: every face reads the state at the start of the substep and the moves are
/// summed, so visiting the faces backwards gives the same bytes. The scene moves water every
/// substep, which the counter asserts.
#[test]
fn face_order_does_not_change_the_state() {
    for scheme in [FilmScheme::DiffusionWave] {
        let mut a = busy_film(scheme);
        let mut b = busy_film(scheme);
        b.set_reverse_faces(true);
        let h = Fix128::from_ratio(1, 20);
        let mut moved = 0u64;
        for _ in 0..300 {
            a.advance(h).expect("a");
            b.advance(h).expect("b");
            assert_eq!(state(&a), state(&b), "{scheme:?} after {}", a.steps());
            moved += a.last_moved();
        }
        assert!(moved > 0, "{scheme:?}: no water moved");
    }
}

/// oracle: a payload of another length, version, bed or bed generation is refused, and a
/// negative depth too; nothing is read.
#[test]
fn a_tampered_payload_is_refused() {
    let f = busy_film(FilmScheme::DiffusionWave);
    let good = state(&f);
    assert_eq!(f.check_state(&good), Ok(()));
    let mut v = good.clone();
    v[0] ^= 1;
    assert_eq!(f.check_state(&v), Err(StateError::InvalidValue));
    let mut d = good.clone();
    d[4] ^= 1;
    assert_eq!(f.check_state(&d), Err(StateError::InvalidValue));
    assert!(matches!(
        f.check_state(&good[..good.len() - 1]),
        Err(StateError::Length { .. })
    ));
    let mut neg = good.clone();
    let at = 112; // first cell's depth
    neg[at..at + 8].copy_from_slice(&(-1i64).to_le_bytes());
    assert_eq!(f.check_state(&neg), Err(StateError::InvalidValue));
    // a negative deposit since the last advance (after the depth and the dry age)
    let mut neg_fed = good.clone();
    neg_fed[at + 12..at + 20].copy_from_slice(&(-1i64).to_le_bytes());
    assert_eq!(f.check_state(&neg_fed), Err(StateError::InvalidValue));
    // a rebuilt bed (generation 2) refuses the payload written over generation 1
    let mut g = busy_film(FilmScheme::DiffusionWave);
    let mut bed2 = groove_bed(14, 4, 9, false);
    bed2 = FilmBed::from_heights(
        bed2.nx(),
        bed2.nz(),
        bed2.cell(),
        (Fix128::ZERO, Fix128::ZERO),
        &bed2
            .heights()
            .iter()
            .map(|q| quanta_to_metres(*q))
            .collect::<Vec<_>>(),
        2,
    )
    .expect("bed2");
    g.set_bed(bed2).expect("rebuild");
    assert_eq!(g.check_state(&good), Err(StateError::InvalidValue));
    // and a bed that does not raise the generation is refused
    assert_eq!(
        g.set_bed(groove_bed(14, 4, 9, false)),
        Err(FilmBuildError::BadGrid)
    );
}

/// degenerate inputs: each refused with the stated error, nothing changed.
#[test]
fn degenerate_inputs_are_refused() {
    let one = [Fix128::ZERO];
    let o = (Fix128::ZERO, Fix128::ZERO);
    assert_eq!(
        FilmBed::from_heights(0, 1, mm(10), o, &[], 1),
        Err(FilmBuildError::BadGrid)
    );
    assert_eq!(
        FilmBed::from_heights(2, 1, mm(10), o, &one, 1),
        Err(FilmBuildError::BadGrid)
    );
    assert_eq!(
        FilmBed::from_heights(1, 1, Fix128::from_ratio(1, 10_000), o, &one, 1),
        Err(FilmBuildError::CellOutOfRange)
    );
    assert_eq!(
        FilmBed::from_heights(1, 1, Fix128::from_int(2), o, &one, 1),
        Err(FilmBuildError::CellOutOfRange)
    );
    let bed = FilmBed::from_heights(1, 1, mm(10), o, &one, 1).expect("bed");
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.viscosity = Fix128::ZERO;
    assert_eq!(
        WaterFilm::new(cfg, bed.clone()).err(),
        Some(FilmBuildError::BadConfig)
    );
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.rain = -Fix128::ONE;
    assert_eq!(
        WaterFilm::new(cfg, bed.clone()).err(),
        Some(FilmBuildError::BadConfig)
    );

    let mut f = film(
        groove_bed(6, 2, 3, true),
        FilmScheme::DiffusionWave,
        FilmEdge::Closed,
    );
    assert!(!f.set_initial_depth(6, 0, 1));
    assert!(!f.set_initial_depth(0, 0, -1));
    assert_eq!(f.deposit(-Fix128::ONE, Fix128::ZERO, mm(1)), 0);
    assert_eq!(f.deposit(Fix128::from_int(5), Fix128::ZERO, mm(1)), 0);
    assert_eq!(f.deposit(mm(5), mm(5), -mm(1)), 0);
    let before = state(&f);
    assert_eq!(f.advance(Fix128::ZERO), Err(ParticipantFault::OutOfRange));
    assert_eq!(f.advance(-Fix128::ONE), Err(ParticipantFault::OutOfRange));
    // a step needing more substeps than allowed: deep water, a long step, a cap of 1
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.max_subcycles = 1;
    let mut g = WaterFilm::new(cfg, groove_bed(6, 2, 3, true)).expect("film");
    g.set_initial_depth(2, 0, metres_to_quanta(mm(30)));
    let g_before = state(&g);
    assert_eq!(g.advance(Fix128::ONE), Err(ParticipantFault::OutOfRange));
    assert_eq!(state(&g), g_before);
    assert_eq!(state(&f), before);
}

/// The film registered in a `PhysicsWorld` steps with the world (subcycling on its own) and
/// the world's snapshot restores it: a world rewound to step 64 and stepped again ends with
/// the same snapshot as the world that ran straight through.
#[test]
fn the_film_steps_and_rewinds_inside_a_world() {
    let make = || {
        let mut w = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3Fix::new(
                Fix128::ZERO,
                -Fix128::from_ratio(980_665, 100_000),
                Fix128::ZERO,
            ),
            ..PhysicsConfig::default()
        });
        w.add_participant(Box::new(busy_film(FilmScheme::DiffusionWave)))
            .expect("register");
        w
    };
    let dt = Fix128::from_ratio(1, 60);
    let mut straight = make();
    for _ in 0..128 {
        straight.try_step(dt).expect("step");
    }
    let mut rewound = make();
    for _ in 0..64 {
        rewound.try_step(dt).expect("step");
    }
    let snap = rewound.snapshot_world();
    for _ in 0..10 {
        rewound.try_step(dt).expect("step");
    }
    rewound.restore_world(&snap).expect("restore");
    for _ in 0..64 {
        rewound.try_step(dt).expect("step");
    }
    assert_eq!(rewound.snapshot_world(), straight.snapshot_world());
    let state = straight.participant_state(0).expect("state");
    let held = i64::from_le_bytes(state[112..120].try_into().expect("8 bytes"));
    assert!(held >= 0);
    let digest: String = Sha256::digest(straight.snapshot_world())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(digest.len(), 64);
    // one quantum is 2^-32 m
    assert_eq!(
        quanta_to_metres(1),
        Fix128::from_raw(0, 1u64 << (64 - DEPTH_QUANTUM_BITS))
    );
}

/// oracle: the Nusselt film. A steady inflow `q` per unit width fed at the top of a plane of
/// slope `tan α` runs down as a film of uniform depth `h = (3 ν q / (g tan α))^(1/3)` (the
/// model's flux `g h³ ∇η / (3ν)` with `∇η = tan α`; for the small angle here `tan α` and
/// `sin α` differ by 0.1 %). Every face of a uniform steady film reads the same state, so the
/// only error is the truncation of each transfer to whole quanta: at most one quantum in
/// `q·Δt/l`, a relative flux error `ε = 2⁻³² l / (q Δt)` and, since `h ∝ q^(1/3)`, a depth
/// error `ε/3`. Cells away from the inlet and the outlet are within `ε` of it.
#[test]
fn a_steady_film_on_an_incline_has_the_nusselt_depth() {
    // a channel one cell wide between walls 0.1 m high (open edges drain every side, so the
    // walls keep the water in the channel), closed at the top by a wall cell, open at the foot
    let nx = 42;
    let tan: f64 = 0.05;
    let cell = mm(CELL_MM);
    let wall = Fix128::from_ratio(1, 10);
    let mut h = vec![wall; nx * 3];
    for i in 1..nx {
        h[i + nx] = -Fix128::from_ratio(5, 100) * cell * Fix128::from_int(i as i64);
    }
    let bed = FilmBed::from_heights(nx, 3, cell, (Fix128::ZERO, Fix128::ZERO), &h, 1).expect("bed");
    let mut f = film(bed, FilmScheme::DiffusionWave, FilmEdge::Open);
    // q = 2e-5 m²/s per unit width, fed into the top channel cell as volume q·Δt·l per advance
    let q = Fix128::from_ratio(2, 100_000);
    let dt = Fix128::from_ratio(1, 64);
    let (x0, z0) = f.bed().centre(1, 1);
    for _ in 0..(64 * 60) {
        f.deposit(x0, z0, q * dt * cell);
        f.advance(dt).expect("advance");
    }
    // compare the flux the depth carries, `g h³ tan α / (3ν)`, with the inflow: a relative
    // flux error within `ε` (equivalently a depth error within `ε/3`)
    let q_f = 2.0e-5;
    let eps = QUANTUM_M * 0.01 / (q_f * f.config().max_substep.to_f64());
    let mut compared = 0;
    for i in 10..30 {
        let d = f.depth_m(i, 1).to_f64();
        let carried = 9.80665 * d * d * d * tan / (3.0 * 1.0e-6);
        assert!(
            (carried - q_f).abs() / q_f <= eps,
            "cell {i}: depth {d:.6e} m carries {carried:.6e} m²/s, inflow {q_f:.1e}, bound {eps:.2e}"
        );
        compared += 1;
    }
    assert_eq!(compared, 20);
}

/// oracle: evaporation. A layer of mean depth `h̄` on a flat closed bed under evaporation `E`
/// and no rain is gone at `t = h̄ / E` (to one substep), and every cell is dry (depth at or
/// below the wet threshold) at `(h̄ − threshold) / E` (to one substep). One cell starts at a
/// third of the others: on a flat bed it is refilled by its neighbours faster than it
/// evaporates, so it dries with them (at the mean), never early and never below zero.
#[test]
fn a_wet_cell_dries_in_the_closed_form_time() {
    let bed = FilmBed::from_heights(
        4,
        3,
        mm(CELL_MM),
        (Fix128::ZERO, Fix128::ZERO),
        &[Fix128::ZERO; 12],
        1,
    )
    .expect("bed");
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.evaporation = FilmConfig::rain_mm_per_hour(Fix128::from_int(360));
    let threshold = cfg.wet_threshold;
    let mut f = WaterFilm::new(cfg, bed).expect("film");
    let h0 = metres_to_quanta(mm(1));
    for k in 0..3 {
        for i in 0..4 {
            f.set_initial_depth(i, k, h0);
        }
    }
    // one shallower cell: it must dry first and never go below zero
    f.set_initial_depth(3, 2, h0 / 3);
    let dt = f.config().max_substep;
    let e = f.config().evaporation.to_f64();
    let q = QUANTUM_M;
    let mean = f.total() as f64 / 12.0;
    let t_gone = mean * q / e;
    let t_dry = (mean - threshold as f64) * q / e;
    let (mut gone_at, mut dry_at) = (None, None);
    for n in 1..200_000u64 {
        f.step_once(dt).expect("substep");
        assert!(
            f.depths().iter().all(|d| *d >= 0),
            "substep {n}: a depth below zero"
        );
        let t = n as f64 * dt.to_f64();
        if dry_at.is_none() && f.dry_ages().iter().all(|a| *a > 0) {
            dry_at = Some(t);
        }
        if f.total() == 0 {
            gone_at = Some(t);
            break;
        }
    }
    let step = dt.to_f64();
    let gone = gone_at.expect("gone");
    let dry = dry_at.expect("dry");
    assert!(
        (gone - t_gone).abs() <= step,
        "gone at {gone} s, closed form {t_gone} s"
    );
    assert!(
        (dry - t_dry).abs() <= step,
        "dry at {dry} s, closed form {t_dry} s"
    );
    assert_eq!(f.ledger().evaporated, f.ledger().initial);
}

/// A tilted plane `y = a·x + c` as an SDF (exact distance).
struct TiltedPlane {
    a: f32,
    c: f32,
}

impl SdfField for TiltedPlane {
    fn distance(&self, x: f32, y: f32, _z: f32) -> f32 {
        (y - self.a * x - self.c) / (1.0 + self.a * self.a).sqrt()
    }
    fn normal(&self, _x: f32, _y: f32, _z: f32) -> (f32, f32, f32) {
        let n = (1.0 + self.a * self.a).sqrt();
        (-self.a / n, 1.0 / n, 0.0)
    }
}

/// oracle: the bed from an SDF is the bed from the same surface as a height function: the
/// sampled heights agree to within the f32 resolution at that height (the bisection works in
/// f32), and both agree with the closed form `a·x + c` at the cell centres.
#[test]
fn an_sdf_bed_and_a_height_bed_agree() {
    let plane = TiltedPlane { a: -0.25, c: 0.02 };
    let cell = mm(CELL_MM);
    let o = (Fix128::ZERO, Fix128::ZERO);
    let from_sdf = FilmBed::from_sdf(16, 2, cell, o, 1, &plane, (-1.0, 1.0)).expect("sdf bed");
    let from_fn =
        FilmBed::from_height_fn(16, 2, cell, o, 1, |x, _| plane.a * x + plane.c).expect("fn bed");
    let mut compared = 0;
    for k in 0..2 {
        for i in 0..16 {
            let c = i + k * 16;
            let (x, _) = from_fn.centre(i, k);
            let exact = metres_to_quanta(Fix128::from_f64(-0.25 * x.to_f64() + 0.02));
            // one f32 step at |y| (and at the bisection bounds' scale, 1 m), in quanta, plus one
            let ulp = (f32::EPSILON as f64 * (1.0 / QUANTUM_M)).ceil() as i64 + 1;
            let (s, h) = (from_sdf.heights()[c], from_fn.heights()[c]);
            assert!(
                (s - h).abs() <= ulp,
                "cell {c}: sdf {s} vs fn {h} quanta (bound {ulp})"
            );
            assert!(
                (h - exact).abs() <= ulp,
                "cell {c}: fn {h} vs exact {exact} quanta"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 32);
    // a column with no surface between the bounds is refused
    assert_eq!(
        FilmBed::from_sdf(2, 1, cell, o, 1, &plane, (0.5, 1.0)).err(),
        Some(FilmBuildError::NoSurface)
    );
}

/// The substep count is `⌈h / Δt⌉`, `Δt = min(Δt_max, 3 ν l² / (8 g h_film³))`, whatever the water: dry, deep, flowing, raining
/// (a count read from the moving water changes `Δt` in phase with the motion).
#[test]
fn the_substep_count_depends_on_declared_inputs_only() {
    let mut f = busy_film(FilmScheme::DiffusionWave);
    // the substep from declared inputs: min(max_substep, 3 ν l² / (8 g h_film³))
    let dt_max = f.substep();
    let mut compared = 0;
    for _ in 0..200 {
        for h in [
            Fix128::from_ratio(1, 20),
            Fix128::from_ratio(1, 300),
            Fix128::from_ratio(7, 3),
        ] {
            let want = (h / dt_max).ceil().hi.max(1) as u32;
            assert_eq!(f.subcycles(h), Some(want), "h = {}", h.to_f64());
            compared += 1;
        }
        f.advance(Fix128::from_ratio(1, 20)).expect("advance");
    }
    assert_eq!(compared, 600);
}

/// A mixed scene: an uneven bed with a trough, scattered thin films (up to 0.78 mm, below the
/// deepest film), rain, evaporation and point deposits, stepped 400 times; returns the SHA-256
/// of the state bytes.
fn mixed_scene(edge: FilmEdge, rev: bool) -> String {
    let (nx, nz) = (23usize, 17usize);
    let h: Vec<Fix128> = (0..nx * nz)
        .map(|c| {
            let (i, k) = (c % nx, c / nx);
            mm(((i * 7 + k * 13) % 11) as i64 - 5)
                - if (5..9).contains(&k) && (3..19).contains(&i) {
                    mm(30)
                } else {
                    Fix128::ZERO
                }
        })
        .collect();
    let bed = FilmBed::from_heights(nx, nz, mm(10), (Fix128::ZERO, Fix128::ZERO), &h, 1).unwrap();
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, edge);
    cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(200));
    cfg.evaporation = FilmConfig::rain_mm_per_hour(Fix128::from_int(70));
    // this scene pins bytes, not accuracy: its small pools on a bumpy bed may stand level
    // whatever flow they carry
    cfg.pool_level_tolerance = Fix128::ONE;
    let mut f = WaterFilm::new(cfg, bed).unwrap();
    f.set_reverse_faces(rev);
    for c in (0..nx * nz).step_by(5) {
        f.set_initial_depth(
            c % nx,
            c / nx,
            metres_to_quanta(Fix128::from_ratio(((c * 3) % 40) as i64, 50_000)),
        );
    }
    for n in 0..400 {
        if n % 9 == 0 {
            let (x, z) = f.bed().centre(n % nx, (n / 3) % nz);
            f.deposit(x, z, Fix128::from_ratio(1, 100_000_000));
        }
        f.advance(Fix128::from_ratio(1, 20)).unwrap();
    }
    let mut s = Vec::new();
    f.write_state(&mut s);
    Sha256::digest(&s)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// golden (determinism): the state of the mixed scene, closed and open, visited forwards and
/// backwards. Re-pinned when the step gained pools (faces deeper than the film levelled
/// exactly) and the derived substep, and the scene moved inside the model's range (thin
/// initial films, 200 mm/h), and again when pools joined only across steps within the join
/// step and deep spills over a step became out of range, and again when pools carrying
/// throughflow stepped at local substeps and free overfalls became weirs: the bytes moved,
/// every validation oracle passes.
#[test]
fn the_mixed_scene_digest_is_pinned() {
    const CLOSED: &str = "3dc2f732c2eeb8e62f321146c34cfa7a260c8a02ff3ce8b18589a38d7c41e471";
    const OPEN: &str = "cb143677a13a1c92ac9adc4d032bcbc70929fd19929f7fab09a918324bcc7c98";
    for rev in [false, true] {
        assert_eq!(
            mixed_scene(FilmEdge::Closed, rev),
            CLOSED,
            "closed, reverse {rev}"
        );
        assert_eq!(
            mixed_scene(FilmEdge::Open, rev),
            OPEN,
            "open, reverse {rev}"
        );
    }
}

/// The integer square root is the floor of the square root at both ends of its range and
/// across perfect squares.
#[test]
fn the_integer_square_root_is_exact() {
    let mut compared = 0;
    for r in [
        0u128,
        1,
        2,
        3,
        1 << 20,
        (1 << 51) - 1,
        1 << 51,
        (1u128 << 64) - 1,
    ] {
        for n in [r * r, r * r + r, (r * r + 2 * r).max(r * r)] {
            assert_eq!(isqrt_u128(n), r, "√{n}");
            compared += 1;
        }
        if r > 0 {
            assert_eq!(isqrt_u128(r * r - 1), r - 1, "√({r}² − 1)");
        }
    }
    assert_eq!(isqrt_u128(u128::MAX), (1u128 << 64) - 1);
    assert!(compared >= 24);
}

/// The flux law's bounds: a face of [`MAX_FACE_DEPTH`] is accepted and one quantum more is
/// refused; at that depth with the largest surface difference the products saturate and the
/// transfer is exactly the `⌊h_f/4⌋` cap; a substep so short that `D_l` passes
/// [`MAX_LUBRICATION_DIVISOR`] is refused, and a film stepping it reports `OutOfRange` and
/// does not change.
#[test]
fn the_flux_law_holds_at_its_bounds() {
    let g = Fix128::from_ratio(980_665, 100_000);
    let nu = Fix128::from_ratio(1, 1_000_000);
    let k = FluxConstants::new(mm(CELL_MM), Fix128::from_ratio(1, 256), g, nu).expect("constants");
    let h = MAX_FACE_DEPTH as i64;
    // high side: bed 0, depth h; low side: a dry bed far below
    let t = simultaneous_transfer((0, h), (-(1i64 << 40), 0), k).expect("at the bound");
    assert_eq!(t, h / 4, "saturated products give the cap");
    assert_eq!(
        simultaneous_transfer((0, h + 1), (-(1i64 << 40), 0), k),
        None
    );
    // D_l = 3 ν l² 2⁹⁶ / (g Δt) passes 2⁹⁸ when 3 ν l² / (g Δt) > 4: l = 1 m, Δt = 2⁻²⁶ s
    let tiny = Fix128::from_raw(0, 1u64 << 38);
    assert_eq!(FluxConstants::new(Fix128::ONE, tiny, g, nu), None);
    let at =
        FluxConstants::new(Fix128::ONE, Fix128::from_ratio(1, 1_000_000), g, nu).expect("1 µs");
    assert!(at.lubrication <= MAX_LUBRICATION_DIVISOR);
    let bed = FilmBed::from_heights(
        2,
        1,
        Fix128::ONE,
        (Fix128::ZERO, Fix128::ZERO),
        &[Fix128::ZERO; 2],
        1,
    )
    .expect("bed");
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.max_substep = tiny;
    cfg.max_subcycles = 4;
    let mut f = WaterFilm::new(cfg, bed).expect("film");
    f.set_initial_depth(0, 0, metres_to_quanta(mm(10)));
    let before = state(&f);
    assert_eq!(f.advance(tiny), Err(ParticipantFault::OutOfRange));
    assert_eq!(state(&f), before);
}

/// oracle: critical outflow over a free edge. A single cell holding a film 0.9 mm deep
/// (thinner than the deepest film; a deeper spill over an edge is outside the model), open
/// on all four edges, empties
/// over them at the critical flux `q = h √(g h)` per unit width (the lubrication flux of a
/// fall the depth of the cell is far larger), so `dh/dt = −4 √g h^(3/2) / l` and
/// `h(t) = (h₀^(−1/2) + 2 √g t / l)^(−2)`. The step is explicit, so the depth follows the
/// closed form to a relative error of the order of one substep's change, `4 √(g h₀) Δt / l`
/// (the bound used), measured at the time the closed form reaches 0.7 mm.
#[test]
fn a_cell_drains_over_its_edges_at_the_critical_rate() {
    let bed = FilmBed::from_heights(
        1,
        1,
        mm(CELL_MM),
        (Fix128::ZERO, Fix128::ZERO),
        &[Fix128::ZERO],
        1,
    )
    .expect("bed");
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Open);
    cfg.max_substep = Fix128::from_ratio(1, 4096);
    let mut f = WaterFilm::new(cfg, bed).expect("film");
    let h0 = 0.0009;
    f.set_initial_depth(0, 0, metres_to_quanta(Fix128::from_ratio(9, 10_000)));
    let (g, l) = (9.80665_f64, 0.01_f64);
    let sqrt_g = 3.131_557_121_916_5_f64; // √9.80665
    let h0_inv_sqrt = 33.333_333_333_333_336_f64; // 1/√0.0009
                                                  // the time the closed form reaches 0.7 mm (below 0.62 mm the lubrication flux of the
                                                  // fall becomes the smaller one and the drain is no longer critical)
    let h1_inv_sqrt = 37.796_447_300_922_72_f64; // 1/√0.0007
    let t_half = (h1_inv_sqrt - h0_inv_sqrt) * l / (2.0 * sqrt_g);
    let dt = f.config().max_substep.to_f64();
    let steps = (t_half / dt).round() as u64;
    for _ in 0..steps {
        f.step_once(f.config().max_substep).expect("substep");
    }
    let t = steps as f64 * dt;
    let s = h0_inv_sqrt + 2.0 * sqrt_g * t / l;
    let expected = 1.0 / (s * s);
    let got = f.depth_m(0, 0).to_f64();
    let bound = 4.0 * (g * h0).sqrt_approx() * dt / l;
    assert!(
        (got - expected).abs() / expected <= bound,
        "depth {got:.6e} m at {t:.5} s, closed form {expected:.6e} m, bound {bound:.3e}"
    );
    assert!(f.ledger().outflow > 0);
}

trait SqrtApprox {
    fn sqrt_approx(self) -> f64;
}
impl SqrtApprox for f64 {
    // Newton iterations: a bound only, no libm
    fn sqrt_approx(self) -> f64 {
        let mut x = self.max(1.0e-300);
        for _ in 0..60 {
            x = 0.5 * (x + self / x);
        }
        x
    }
}

/// The band is stepped, not levelled: water 3 mm deep flowing down a channel at slope
/// 1/1000 (Re ≈ 88, Fr ≈ 0.17: lubrication is the right closure) is deeper than the declared
/// deepest film of 1.5 mm, so its faces would pool; a level pool cannot carry the flow (it
/// needs a surface fall of about 1.2 mm along the 1.2 m channel, against 1 % of the 3 mm
/// depth), so its cells step on their own at local substeps and the surface keeps a gradient
/// to carry the flow: over the middle 20 cells it falls far more than the rest band of a
/// levelled pool (7 quanta per cell).
#[test]
fn a_flowing_channel_deeper_than_the_film_is_stepped_not_levelled() {
    let (nx, nz) = (60usize, 3usize);
    let cell = mm(20);
    let wall = Fix128::from_ratio(1, 10);
    let mut h = vec![wall; nx * nz];
    for i in 0..nx {
        h[i + nx] = -Fix128::from_ratio(1, 1000) * cell * Fix128::from_int(i as i64);
    }
    let bed =
        FilmBed::from_heights(nx, nz, cell, (Fix128::ZERO, Fix128::ZERO), &h, 1).expect("bed");
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Open);
    cfg.film_depth_max = Fix128::from_ratio(15, 10_000);
    let mut f = WaterFilm::new(cfg, bed).expect("film");
    for i in 0..nx {
        f.set_initial_depth(i, 1, metres_to_quanta(mm(3)));
    }
    let q = Fix128::from_ratio(88, 1_000_000) * cell;
    let step = f.substep();
    let (x0, z0) = f.bed().centre(1, 1);
    let mut most = 0;
    for _ in 0..((10.0 / step.to_f64()) as usize) {
        f.deposit(x0, z0, q * step);
        f.advance(step).expect("advance");
        most = most.max(f.last_local_substeps());
    }
    assert!(most > 1, "the channel never took local substeps");
    let eta = |i: usize| f.bed().heights()[i + nx] + f.depths()[i + nx];
    let fall = quanta_to_metres(eta(20) - eta(40)).to_f64();
    // a levelled pool stands within its rest band (7 quanta per cell); the stepped channel
    // keeps a fall far above it (its uniform-flow depth is checked against the closed form in
    // flow_in_the_band_has_the_lubrication_depth)
    let rest = 7 * 20;
    println!(
        "flowing channel: surface fell {fall:.3e} m over 20 cells (rest band {:.1e} m), local substeps up to {most}",
        quanta_to_metres(rest).to_f64()
    );
    assert!(
        eta(20) - eta(40) > 100 * rest,
        "the channel stood level: fall {fall:.3e} m"
    );
}

/// A closed groove filling under 200 mm/h of rain (no water enters or leaves the domain, a
/// pool exchanges water only with its own bevels) is never refused and ends level: once the
/// rain stops, every wet groove cell stands within the rest band of the closed-form level.
#[test]
fn a_filling_groove_is_accepted_and_levels() {
    let bed = groove_bed(12, 3, 8, true);
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(200));
    let mut f = WaterFilm::new(cfg, bed.clone()).expect("film");
    let step = Fix128::from_ratio(1, 6);
    for n in 0..(6 * 120) {
        if f.advance(step).is_err() {
            panic!("refused at step {n}: band cells {:?}", f.last_band_cells());
        }
    }
    let mut dry_cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    dry_cfg.rain = Fix128::ZERO;
    let mut g = WaterFilm::new(dry_cfg, bed).expect("film");
    for i in 0..12 {
        g.set_initial_depth(i, 0, f.depths()[i]);
    }
    settle(&mut g, 10_000_000);
    let beds = g.bed().heights().to_vec();
    // the level of the water below it (the films stalled above it are their own residue)
    let first = level_for(&beds, g.total());
    let below: Vec<usize> = (0..12).filter(|c| beds[*c] < first).collect();
    let pool_beds: Vec<i64> = below.iter().map(|c| beds[*c]).collect();
    let eta = level_for(&pool_beds, below.iter().map(|c| g.depths()[*c]).sum());
    // above the level only thin films remain, each stopped where its lubrication transfer
    // rounds to zero: on every face between two such cells h_f³ |Δη| < D_l with
    // D_l = 3 ν l² 2⁹⁶ / (g Δt) (the film rest set)
    let dt = g.config().max_substep.to_f64();
    let d_l = 3.0e-6 * 1.0e-4 / (QUANTUM_M * QUANTUM_M * QUANTUM_M) / (9.80665 * dt);
    let pool: Vec<usize> = (0..12).filter(|c| beds[*c] < eta).collect();
    assert!(pool.len() >= 6, "only {} cells below the level", pool.len());
    for &c in &pool {
        let off = (beds[c] + g.depths()[c] - eta).abs();
        assert!(
            off <= 7 * pool.len() as i64,
            "cell {c}: {off} quanta from the level"
        );
    }
    let mut films = 0;
    for c in 0..11 {
        let (a, b) = (c, c + 1);
        if beds[a] < eta || beds[b] < eta {
            continue;
        }
        let (ea, eb) = (beds[a] + g.depths()[a], beds[b] + g.depths()[b]);
        let hf = (ea.max(eb) - beds[a].max(beds[b])).max(0) as f64;
        let e = (ea - eb).abs() as f64;
        assert!(
            hf * hf * hf * e < d_l,
            "face {a}|{b}: film still moving at rest"
        );
        films += 1;
    }
    assert!(films >= 2, "compared {films} film faces");
}

/// oracle: steady flow along one straight stroke (floor 11.8 cm, 1 cm walls 3 cm deep,
/// slope 1/200, 1.9e-6 m³/s: depth ≈ 1 mm, Re ≈ 16, Fr ≈ 0.17, drawdown length h/S ≈ 0.2 m
/// against a 1.2 m channel). At the substep derived from the deepest film (1.5 mm) no face is
/// capped and the depth at the middle equals the lubrication closed form over the sampled
/// section, `Σ_columns g h_c³ S l / (3ν) = Q`, to the transfer rounding (flux error
/// `ε = 2⁻³² l / (q Δt)`, depth `ε/3`), at 2 cm and at 1 cm cells. The continuous section is
/// reported (the 2 cm grid samples the 11.8 cm floor as 10 cm).
fn stroke_has_the_lubrication_depth(cells_cm: &[i64]) {
    let (w, wall, depth, slope, q_total) = (0.118_f64, 0.01_f64, 0.03_f64, 0.005_f64, 1.9e-6_f64);
    for &cm in cells_cm {
        let l = cm as f64 / 100.0;
        let (nx, nz) = ((1.2 / l) as usize, (0.16 / l) as usize);
        let cell = Fix128::from_ratio(cm, 100);
        let bed = FilmBed::from_height_fn(
            nx,
            nz,
            cell,
            (Fix128::ZERO, Fix128::from_ratio(-8, 100)),
            1,
            move |x, z| {
                let (a, x) = (f64::from(z).abs(), f64::from(x));
                let cut = if x < 0.04 {
                    0.0
                } else if a <= w / 2.0 {
                    depth
                } else if a <= w / 2.0 + wall {
                    depth * (1.0 - (a - w / 2.0) / wall)
                } else {
                    0.0
                };
                (-(slope * x) - cut) as f32
            },
        )
        .expect("bed");
        let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Open);
        // the deepest film 1.5 mm: the depth plus the feed's bump each step (0.26 mm at 1 cm)
        cfg.film_depth_max = Fix128::from_ratio(15, 10_000);
        let mut f = WaterFilm::new(cfg, bed).expect("film");
        let im = (0.6 / l) as usize;
        let beds: Vec<f64> = (0..nz)
            .map(|k| quanta_to_metres(f.bed().heights()[im + k * nx]).to_f64())
            .collect();
        let kc = nz / 2;
        let floor_bed = beds[kc];
        let carried = |hf: f64| -> f64 {
            beds.iter()
                .map(|b| {
                    let h = (hf - (b - floor_bed)).max(0.0);
                    9.80665 * h * h * h * slope / 3.0e-6 * l
                })
                .sum()
        };
        let (mut lo, mut hi) = (0.0, 0.01);
        for _ in 0..80 {
            let m = 0.5 * (lo + hi);
            if carried(m) < q_total {
                lo = m;
            } else {
                hi = m;
            }
        }
        let h_closed = 0.5 * (lo + hi);
        // start at the closed-form depth everywhere in the channel, then let it run
        for i in 0..nx {
            for k in 0..nz {
                let b = quanta_to_metres(f.bed().heights()[i + k * nx]).to_f64();
                let surface = floor_bed + h_closed - slope * (i as f64 - im as f64) * l;
                if surface > b {
                    f.set_initial_depth(i, k, metres_to_quanta(Fix128::from_f64(surface - b)));
                }
            }
        }
        let i0 = (0.06 / l) as usize;
        let floor_cells: Vec<usize> = (0..nz)
            .filter(|&k| f.bed().centre(i0, k).1.to_f64().abs() <= w / 2.0)
            .collect();
        let step = Fix128::from_ratio(1, 6);
        let per = Fix128::from_f64(q_total / floor_cells.len() as f64) * step;
        let mut capped = 0u64;
        for _ in 0..(6 * 40) {
            for &k in &floor_cells {
                let (x, z) = f.bed().centre(i0, k);
                f.deposit(x, z, per);
            }
            f.advance(step).expect("advance");
            capped += u64::from(f.last_cap_faces());
        }
        assert_eq!(capped, 0, "{cm} cm: lubrication faces were capped");
        assert!(
            f.last_pooled_cells().is_empty(),
            "{cm} cm: the stroke pooled"
        );
        let h = f.depth_m(im, kc).to_f64();
        let q_face = q_total / (floor_cells.len() as f64 * l);
        let eps = QUANTUM_M * l / (q_face * f.substep().to_f64());
        let rel = (h - h_closed) / h_closed;
        println!("{cm} cm: depth {h:.5e} m, sampled-section closed form {h_closed:.5e} m ({rel:+.2e}), bound {:.2e}, substep {:.2e} s", eps, f.substep().to_f64());
        assert!(
            rel.abs() <= eps,
            "{cm} cm: depth {h:.5e} vs {h_closed:.5e} ({rel:+.3e}, bound {eps:.2e})"
        );
    }
}

/// [`stroke_has_the_lubrication_depth`] at 2 cm cells; at 1 cm (the same check, 4 times the
/// cells) in `flow_along_a_stroke_has_the_lubrication_depth_at_one_centimetre`, run weekly.
#[test]
fn flow_along_a_stroke_has_the_lubrication_depth() {
    stroke_has_the_lubrication_depth(&[2]);
}

#[test]
#[ignore = "runtime: the 1 cm stroke, 120 x 16 cells over 40 s, about 28 s without \
            optimisation; the 2 cm stroke is the lib test; run by run_ignored.py"]
fn flow_along_a_stroke_has_the_lubrication_depth_at_one_centimetre() {
    stroke_has_the_lubrication_depth(&[1]);
}

/// The sill scene: one row of 1 cm cells, closed: a left basin (10 cells, floor −30 mm), a
/// sill (bed −10 mm) and a right basin (10 cells, floor −15 mm).
fn sill_scene(rain_mm_h: i64) -> WaterFilm {
    let beds: Vec<Fix128> = (0..21)
        .map(|i| match i {
            0..=9 => mm(-30),
            10 => mm(-10),
            _ => mm(-15),
        })
        .collect();
    let bed = FilmBed::from_heights(21, 1, mm(CELL_MM), (Fix128::ZERO, Fix128::ZERO), &beds, 1)
        .expect("bed");
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(rain_mm_h));
    WaterFilm::new(cfg, bed).expect("film")
}

/// Water standing 15 mm over the sill spills into the dry right basin as a free overfall (a
/// weir, inside the model), but the water it drops then runs over the basin's dry flat floor
/// as a dam-break front, an inertial flow outside the model: the advance is refused, names
/// the cells at the front, and changes nothing.
#[test]
fn a_dam_break_front_below_a_spill_is_refused() {
    let mut f = sill_scene(0);
    let level = metres_to_quanta(mm(5));
    for i in 0..=10 {
        f.set_initial_depth(i, 0, level - f.bed().heights()[i]);
    }
    let before = state(&f);
    assert_eq!(
        f.advance(Fix128::from_ratio(1, 20)),
        Err(ParticipantFault::OutOfRange)
    );
    assert!(
        f.last_band_cells().contains(&11) && f.last_band_cells().contains(&12),
        "band {:?}",
        f.last_band_cells()
    );
    assert_eq!(state(&f), before);
}

/// oracle: two basins joined over a sill by rain (200 mm/h). The left basin starts 3.5 mm
/// below the sill, the right one dry, so the right basin is 3.5 mm deep (past the band where
/// a pool carrying inflow is refused) when the left one starts to spill.
/// (1) From then the left basin's rain crosses the sill as a film (its depth over the sill
/// stays below the deepest film), so the right basin reaches the sill when the rain on all
/// 21 cells has filled the rest of its 50 mm·cell deficit:
/// `t = x/r + (0.05 − 10x) / (21 r)` with `x` = 3.5 mm, later by at most the water the films
/// over the left basin and the sill hold (`11 h_film / (21 r)`).
/// (2) The two sides join into one pool only when the step across the sill is within the
/// join step, so at the join the left level moves by at most that step (besides the rain).
/// (3) After the join the surface is the level of the whole volume over all 21 beds.
#[test]
fn two_basins_join_over_a_sill_without_a_jump() {
    let mut f = sill_scene(200);
    let sill = f.bed().heights()[10];
    let x = metres_to_quanta(Fix128::from_ratio(35, 10_000));
    for i in 0..=9 {
        f.set_initial_depth(i, 0, sill - x - f.bed().heights()[i]);
    }
    let cfg = *f.config();
    let r = cfg.rain.to_f64();
    let dt = f.substep().to_f64();
    let rain_q = (r * dt / QUANTUM_M).ceil() as i64;
    let h_film = cfg.film_depth_max.to_f64();
    let xm = quanta_to_metres(x).to_f64();
    let t_fill = xm / r + (0.05 - 10.0 * xm) / (21.0 * r);
    let late = 11.0 * h_film / (21.0 * r);
    let surface = |f: &WaterFilm, c: usize| f.bed().heights()[c] + f.depths()[c];
    let (mut reached, mut joined) = (None, None);
    let mut n = 0u64;
    while joined.is_none() && n < 2_000_000 {
        let before = (surface(&f, 0), f.depths()[10], f.depths()[11]);
        f.step_once(f.substep()).expect("substep");
        n += 1;
        let t = n as f64 * dt;
        if reached.is_none() && surface(&f, 20) >= sill {
            reached = Some(t);
        }
        let pooled = f.last_pooled_cells();
        if [0usize, 9, 10, 11, 20].iter().all(|c| pooled.contains(c))
            && (surface(&f, 0) - surface(&f, 20)).abs() <= 1
        {
            let j = join_step(cfg.pool_join_step, before.1, before.2);
            let jump = (surface(&f, 0) - before.0).abs();
            assert!(
                jump <= j + rain_q,
                "join at t {t:.2} s: left level moved {jump} quanta, join step {j} + rain {rain_q}"
            );
            joined = Some(t);
            // (3) one level over all 21 beds
            let all: Vec<i64> = f.bed().heights().to_vec();
            let eta = level_for(&all, f.total());
            for c in 0..21 {
                let off = (surface(&f, c) - eta).abs();
                assert!(
                    off <= 1,
                    "cell {c}: {off} quanta from the level after the join"
                );
            }
            println!(
                "join at {t:.2} s: left level moved {jump} quanta (join step {j}, rain {rain_q})"
            );
        }
    }
    let reached = reached.expect("the right basin never reached the sill");
    assert!(joined.is_some(), "the basins never joined");
    assert!(
        reached >= t_fill - dt && reached <= t_fill + late + dt,
        "right basin at the sill at {reached:.3} s, closed form {t_fill:.3} s (+ at most {late:.3} s)"
    );
    println!("right basin at the sill at {reached:.3} s (closed form {t_fill:.3} s, + at most {late:.3} s)");
}

/// The join step, at its edges. Two deep cells side by side (beds 10 mm apart, depths 20 mm
/// and 30 mm, so the shallower side is 20 mm) whose surfaces differ by `s`: with the default
/// fraction (1 %) the join step is `⌊0.01 · 20 mm⌋` in quanta; at `s` equal to it the two
/// join and stand level, one quantum more is a step outside the model and the advance is
/// refused. With the fraction 0 the step is the floor of 7 quanta.
#[test]
fn the_join_step_is_one_percent_of_the_shallower_side() {
    let run = |fraction: Fix128, s: i64| {
        let beds = [Fix128::ZERO, mm(-10)];
        let bed = FilmBed::from_heights(2, 1, mm(CELL_MM), (Fix128::ZERO, Fix128::ZERO), &beds, 1)
            .expect("bed");
        let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
        cfg.pool_join_step = fraction;
        let mut f = WaterFilm::new(cfg, bed).expect("film");
        let d = metres_to_quanta(mm(20));
        f.set_initial_depth(0, 0, d);
        let (b0, b1) = (f.bed().heights()[0], f.bed().heights()[1]);
        f.set_initial_depth(1, 0, b0 + d + s - b1);
        let r = f.step_once(f.substep());
        let level = r.is_ok() && {
            let e: Vec<i64> = (0..2)
                .map(|c| f.bed().heights()[c] + f.depths()[c])
                .collect();
            (e[0] - e[1]).abs() <= 1 && f.last_pooled_cells() == [0, 1]
        };
        (r, level)
    };
    let j = metres_to_quanta(mm(20)) / 100;
    let mut compared = 0;
    for (fraction, step) in [(Fix128::from_ratio(1, 100), j), (Fix128::ZERO, 7)] {
        let (r, level) = run(fraction, step);
        assert!(
            r.is_ok() && level,
            "step {step}: {r:?}, joined and level {level}"
        );
        let (r, _) = run(fraction, step + 1);
        assert_eq!(r, Err(ParticipantFault::OutOfRange), "step {}", step + 1);
        compared += 2;
    }
    assert_eq!(compared, 4);
}

/// A row of cells between two dry rims 50 mm high (rows 0 and 2), open edges.
fn walled_row(nx: usize, l_mm: i64, beds: &[Fix128], cfg: FilmConfig) -> WaterFilm {
    let mut h = vec![mm(50); nx * 3];
    h[nx..2 * nx].copy_from_slice(beds);
    let bed =
        FilmBed::from_heights(nx, 3, mm(l_mm), (Fix128::ZERO, Fix128::ZERO), &h, 1).expect("bed");
    WaterFilm::new(cfg, bed).expect("film")
}

/// oracle: a free overfall. A basin (10 cells, floor −30 mm) behind a sill (bed −10 mm) at an
/// open edge, walled on its other sides, starts level at +5 mm (head `h₀` = 15 mm over the
/// sill). The sill spills over the edge as a broad-crested weir, `q = (2/3)^{3/2} √g h^{3/2}`
/// per unit width, so the head of the pool (area `11 l²`, the sill included) follows
/// `h(t) = (h₀^{-1/2} + (2/3)^{3/2} √g t / (22 l))⁻²`, to the explicit step's error
/// `(2/3)^{3/2} √(g h₀) Δt_local / (11 l)` (one local substep's relative change). Three
/// cases: the sill at the open edge with 1 cm cells at the film substep (one local substep),
/// and with 2 cm cells and a substep of 15 ms (the weir moves more than `|Δη|/8` per substep
/// and needs local substeps); and the sill over an interior drop into a sump 1 m deep.
#[test]
fn a_free_overfall_drains_at_the_weir_rate() {
    for (l_mm, max_substep, t_end, local_needed, sump) in [
        (CELL_MM, Fix128::from_ratio(1, 256), 0.125, false, false),
        (20, Fix128::from_ratio(1, 64), 0.5, true, false),
        (CELL_MM, Fix128::from_ratio(1, 256), 0.125, false, true),
    ] {
        let mut beds = vec![mm(50)];
        beds.extend(core::iter::repeat_n(mm(-30), 10));
        beds.push(mm(-10));
        if sump {
            beds.push(mm(-1000));
        }
        let nx = beds.len();
        let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Open);
        cfg.max_substep = max_substep;
        let mut f = walled_row(nx, l_mm, &beds, cfg);
        let level = metres_to_quanta(mm(5));
        for i in 1..12 {
            f.set_initial_depth(i, 1, level - f.bed().heights()[nx + i]);
        }
        let sill = f.bed().heights()[nx + 11];
        let (g, l, h0) = (9.80665_f64, l_mm as f64 / 1000.0, 0.015_f64);
        let cw = (8.0_f64 / 27.0).sqrt();
        let dt = f.substep().to_f64();
        let n = (t_end / dt).round() as u64;
        let mut most = 0;
        for _ in 0..n {
            f.step_once(f.substep()).expect("substep");
            most = most.max(f.last_local_substeps());
        }
        let t = n as f64 * dt;
        let head = quanta_to_metres(f.bed().heights()[nx + 1] + f.depths()[nx + 1] - sill).to_f64();
        let s = 1.0 / h0.sqrt() + cw * g.sqrt() * t / (22.0 * l);
        let closed = 1.0 / (s * s);
        let bound = cw * (g * h0).sqrt() * dt / f64::from(most) / (11.0 * l);
        let rel = (head - closed) / closed;
        println!("overfall at {l_mm} mm (sump {sump}): head {head:.5e} m, closed form {closed:.5e} m ({rel:+.2e}, bound {bound:.2e}), local substeps up to {most}");
        assert_eq!(most > 1, local_needed, "{l_mm} mm: local substeps {most}");
        assert!(
            rel.abs() <= bound,
            "{l_mm} mm: head {head:.5e} vs {closed:.5e} ({rel:+.3e}, bound {bound:.2e})"
        );
        assert!(f.ledger().outflow > 0);
        assert_eq!(f.total(), f.ledger().expected());
    }
}

/// oracle: uniform flow inside the band. A channel at slope `S` (one walled row of 2 cm
/// cells), fed `q` per unit width at its head and spilling over an open edge at its foot,
/// starts at the lubrication depth `h = (3 ν q / (g S))^{1/3}`. The depth is deeper than the
/// film, so its faces step at local substeps; at its middle the depth stays at `h` to the
/// transfer rounding at the local substep, `ε = 2⁻³² l / (q Δt_local)` (depth `ε/3`), plus
/// `10⁻⁴` for the drawdown from the brink, read 3 to 4.5 backwater lengths `h/S` away.
fn band_has_the_lubrication_depth(channels: &[(i64, i64, i64, usize, usize)]) {
    for &(s_num, s_den, h_um, nx, at) in channels {
        let slope = s_num as f64 / s_den as f64;
        let mut beds = vec![mm(50)];
        beds.extend((1..nx).map(|i| -Fix128::from_ratio(s_num * 20 * i as i64, s_den * 1000)));
        let mut f = walled_row(
            nx,
            20,
            &beds,
            FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Open),
        );
        let h = h_um as f64 * 1e-6;
        let q = 9.80665 * h * h * h * slope / 3.0e-6;
        for i in 1..nx {
            f.set_initial_depth(i, 1, metres_to_quanta(Fix128::from_f64(h)));
        }
        // fed once per substep (a whole 1/6 s of feed at once would be a 2 mm surge at the
        // head, an inertial front outside the model)
        let step = f.substep();
        let (x0, z0) = f.bed().centre(1, 1);
        let mut most = 0;
        for _ in 0..((20.0 / step.to_f64()) as usize) {
            f.deposit(x0, z0, Fix128::from_f64(q * 0.02) * step);
            if f.advance(step).is_err() {
                let c = f.last_band_cells().to_vec();
                let d: Vec<f64> = c
                    .iter()
                    .map(|c| quanta_to_metres(f.depths()[*c]).to_f64() * 1e3)
                    .collect();
                panic!("S {slope}: refused, cells {c:?} depths mm {d:?}");
            }
            most = most.max(f.last_local_substeps());
        }
        let got = f.depth_m(at, 1).to_f64();
        let local = f.substep().to_f64() / f64::from(most);
        let eps = QUANTUM_M * 0.02 / (q * local) / 3.0 + 1.0e-4;
        let rel = (got - h) / h;
        println!("band channel S {slope}: depth {got:.6e} m vs {h:.6e} m ({rel:+.2e}, bound {eps:.2e}), local substeps up to {most}");
        assert!(most > 1, "S {slope}: no local substeps");
        assert!(
            rel.abs() <= eps,
            "S {slope}: depth {got:.6e} vs {h:.6e} ({rel:+.3e})"
        );
    }
}

/// [`band_has_the_lubrication_depth`] on the channel at slope 1/200 (151 cells, the depth read
/// 1.5 m from the brink, 3 backwater lengths `h/S` of 0.5 m); the channel at 1/1000 (601
/// cells, read 9 m from the brink, 3 lengths of 3 m) in
/// `flow_in_the_band_has_the_lubrication_depth_on_the_gentle_slope`, run weekly. The channel
/// is long against the drawdown from the brink, which decays over `h/S`.
#[test]
fn flow_in_the_band_has_the_lubrication_depth() {
    band_has_the_lubrication_depth(&[(1, 200, 2_500, 151, 75)]);
}

#[test]
#[ignore = "runtime: the 601-cell channel at slope 1/1000, about 10 s without optimisation; \
            the 1/200 channel is the lib test; run by run_ignored.py"]
fn flow_in_the_band_has_the_lubrication_depth_on_the_gentle_slope() {
    band_has_the_lubrication_depth(&[(1, 1_000, 3_000, 601, 150)]);
}

/// oracle: a source inside a pool is inflow. A flat channel (40 cells of 1 cm, walled) fed
/// `q` per unit width at its head and spilling over an open edge at its foot carries a
/// lubrication gradient: `q = g h³ (dη/dx)/(3ν)`, so `h(x)⁴ = h_b⁴ + 12 ν q x / g` with `x`
/// from the brink. Its deeper part is a pool (steps between cells within the join step)
/// whose only inflow is the water deposited into it: counted as throughflow it is stepped on
/// its own and keeps the gradient, at 0.3 m and 0.4 m from the brink within 1 % of the
/// scheme's steady profile (which samples the gradient at the upstream face depth and lies
/// within 3 % of the closed form); not counted, it would be levelled flat.
#[test]
fn a_pool_fed_from_inside_keeps_its_gradient() {
    let nx = 42;
    let mut beds = vec![mm(50)];
    beds.extend(core::iter::repeat_n(Fix128::ZERO, nx - 1));
    let mut f = walled_row(
        nx,
        CELL_MM,
        &beds,
        FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Open),
    );
    let (q, g, nu, l) = (5.0e-6_f64, 9.80665_f64, 1.0e-6_f64, 0.01_f64);
    let profile = |x: f64| (12.0 * nu * q * x / g).sqrt().sqrt();
    for i in 1..nx {
        let x = (nx - i) as f64 * l - l / 2.0;
        f.set_initial_depth(i, 1, metres_to_quanta(Fix128::from_f64(profile(x))));
    }
    let step = Fix128::from_ratio(1, 6);
    let (x0, z0) = f.bed().centre(1, 1);
    let mut freed = 0;
    for _ in 0..(6 * 40) {
        f.deposit(x0, z0, Fix128::from_f64(q * l) * step);
        f.advance(step).expect("advance");
        freed = freed.max(f.last_local_substeps());
    }
    // the scheme's own steady profile: from the brink (the critical film, q = h √(g h)),
    // each face upstream carries q at its upstream depth, q = g h_up³ (h_up − h_dn) / (3 ν l);
    // it differs from the continuous closed form by the face sampling (reported)
    let brink = {
        let (mut lo, mut hi) = (0.0_f64, 0.01_f64);
        for _ in 0..200 {
            let m = 0.5 * (lo + hi);
            if m * m * m < q * q / g {
                lo = m;
            } else {
                hi = m;
            }
        }
        0.5 * (lo + hi)
    };
    let mut discrete = vec![brink];
    for _ in 0..nx {
        let dn = discrete[discrete.len() - 1];
        let (mut lo, mut hi) = (dn, dn + 0.01);
        for _ in 0..200 {
            let m = 0.5 * (lo + hi);
            if g * m * m * m * (m - dn) / (3.0 * nu * l) < q {
                lo = m;
            } else {
                hi = m;
            }
        }
        discrete.push(0.5 * (lo + hi));
    }
    let mut compared = 0;
    for x_cm in [30usize, 40] {
        let i = nx - x_cm;
        let x = x_cm as f64 * l - l / 2.0;
        let (got, want, scheme) = (f.depth_m(i, 1).to_f64(), profile(x), discrete[x_cm - 1]);
        let sampling = (scheme - want).abs() / want;
        let rel = (got - scheme) / scheme;
        println!("fed pool at {x:.3} m: depth {got:.5e} m, scheme's steady profile {scheme:.5e} m ({rel:+.2e}), closed form {want:.5e} m (sampling {sampling:.2e})");
        assert!(
            rel.abs() <= 0.01,
            "x {x}: depth {got:.5e} vs the steady profile {scheme:.5e} ({rel:+.3e})"
        );
        assert!(
            sampling <= 0.03,
            "x {x}: the steady profile is {sampling:.3e} from the closed form"
        );
        compared += 1;
    }
    assert_eq!(compared, 2);
    assert!(freed >= 1);
}

/// The site's letter: a groove 3 cm deep (12 × 3 cells of 1 cm) cut in flat ground (24 × 9),
/// closed, under 20 mm/h of rain for 120 s. The ground's runoff (five times the groove's own
/// rain) fills the groove through the band (a pool 1–5 mm deep carrying inflow), which is
/// stepped at local substeps: no advance is refused, the groove stands level, the volume
/// ledger holds, and the groove holds at least its own rain.
#[test]
fn a_letter_fills_from_its_runoff() {
    let (nx, nz) = (24, 9);
    let groove = |x: f32, z: f32| -> f32 {
        if (0.06..0.18).contains(&x) && (0.03..0.06).contains(&z) {
            -0.03
        } else {
            0.0
        }
    };
    let bed = FilmBed::from_height_fn(nx, nz, mm(CELL_MM), (Fix128::ZERO, Fix128::ZERO), 1, groove)
        .expect("bed");
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(20));
    let mut f = WaterFilm::new(cfg, bed).expect("film");
    let step = Fix128::from_ratio(1, 6);
    let mut most = 0;
    for n in 0..(6 * 120) {
        if f.advance(step).is_err() {
            panic!("refused at step {n}: cells {:?}", f.last_band_cells());
        }
        most = most.max(f.last_local_substeps());
        assert_eq!(f.total(), f.ledger().expected());
    }
    let cells: Vec<usize> = (0..nx * nz).filter(|c| f.bed().heights()[*c] < 0).collect();
    assert_eq!(cells.len(), 36);
    let s: Vec<i64> = cells
        .iter()
        .map(|c| f.bed().heights()[*c] + f.depths()[*c])
        .collect();
    let spread = s.iter().max().unwrap_or(&0) - s.iter().min().unwrap_or(&0);
    let held: i64 = cells.iter().map(|c| f.depths()[*c]).sum();
    // the groove carries its runoff Q (the ground's rain) to the far end at most: a
    // lubrication gradient across its length L = 12 cm, ΔH = 3 ν Q L² / (g h̄³ A)
    let r = cfg.rain.to_f64();
    let q_in = 180.0 * r * 1.0e-4;
    let h_bar = quanta_to_metres(held / 36).to_f64();
    let dh = 3.0e-6 * q_in * 0.12 * 0.12 / (9.80665 * h_bar * h_bar * h_bar * 36.0e-4);
    let bound = (dh / QUANTUM_M).ceil() as i64 + 7 * 12;
    assert!(
        spread <= bound,
        "groove spread {spread} quanta, gradient bound {bound}"
    );
    let rain_per_cell = f.ledger().rain / (nx * nz) as i64;
    assert!(
        held >= 36 * rain_per_cell,
        "the groove holds less than its own rain"
    );
    println!(
        "letter: groove {:.3} mm deep, spread {spread} quanta (bound {bound}), ground holds {:.3} mm·cell, local substeps up to {most}",
        quanta_to_metres(held / 36).to_f64() * 1e3,
        quanta_to_metres(f.total() - held).to_f64() * 1e3
    );
}

/// The scheme's steady profile on a flat row draining to a brink at its last cell: `q[j]` is
/// the flux (per unit width) across the face downstream of cell `j`; the brink cell carries
/// `q` over its edge at the critical film depth `(q²/g)^{1/3}`, and each face upstream at
/// its upstream depth, `q = g h_up³ (h_up − h_dn) / (3 ν l)` (bisection).
fn steady_row(q: &[f64], l: f64) -> Vec<f64> {
    let (g, nu) = (9.80665_f64, 1.0e-6_f64);
    let bisect = |f: &dyn Fn(f64) -> bool, lo: f64| {
        let (mut lo, mut hi) = (lo, lo + 0.01);
        for _ in 0..200 {
            let m = 0.5 * (lo + hi);
            if f(m) {
                lo = m;
            } else {
                hi = m;
            }
        }
        0.5 * (lo + hi)
    };
    let n = q.len();
    let mut h = vec![0.0; n];
    let qb = q[n - 1];
    h[n - 1] = bisect(&|m| m * m * m < qb * qb / g, 0.0);
    for j in (0..n - 1).rev() {
        let (dn, qj) = (h[j + 1], q[j]);
        h[j] = bisect(&|m| g * m * m * m * (m - dn) / (3.0 * nu * l) < qj, dn);
    }
    h
}

/// oracle: rain inside a pool is inflow. A flat channel 0.8 m long (one row of 1 cm cells,
/// closed at its head) drains over a drop into a deep sump at its foot, under 30 mm/h of
/// rain: the flux grows with the rain caught upstream, `q = r (L − x)` with `x` from the
/// brink and `L` = 0.8 m, so `h(x)⁴ = h_b⁴ + 12 ν r (L x − x²/2) / g`. Its deeper part (over 1 mm) is a pool whose only inflow is
/// the rain on it: counted as throughflow it steps on its own and keeps its gradient (within
/// 1 % of the scheme's steady profile at 0.6 m and 0.8 m from the brink, which lies within
/// 3 % of the closed form); not counted, it would be levelled flat.
#[test]
fn a_pool_fed_by_rain_keeps_its_gradient() {
    let n = 80;
    let mut beds = vec![Fix128::ZERO; n];
    beds.push(mm(-1000));
    let bed = FilmBed::from_heights(
        n + 1,
        1,
        mm(CELL_MM),
        (Fix128::ZERO, Fix128::ZERO),
        &beds,
        1,
    )
    .expect("bed");
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(30));
    let r = cfg.rain.to_f64();
    let mut f = WaterFilm::new(cfg, bed).expect("film");
    let l = 0.01_f64;
    let q: Vec<f64> = (0..n).map(|j| r * l * (j + 1) as f64).collect();
    let steady = steady_row(&q, l);
    for (j, h) in steady.iter().enumerate() {
        f.set_initial_depth(j, 0, metres_to_quanta(Fix128::from_f64(*h)));
    }
    let step = Fix128::from_ratio(1, 6);
    let mut most = 0;
    for _ in 0..(6 * 40) {
        f.advance(step).expect("advance");
        most = most.max(f.last_local_substeps());
    }
    let mut compared = 0;
    for x_cm in [60usize, 80] {
        let j = n - x_cm;
        let x = x_cm as f64 * l - l / 2.0;
        let want = (12.0e-6 * r * (0.8 * x - x * x / 2.0) / 9.80665)
            .sqrt()
            .sqrt();
        let got = f.depth_m(j, 0).to_f64();
        let scheme = steady[j];
        let sampling = (scheme - want).abs() / want;
        let rel = (got - scheme) / scheme;
        println!("rain-fed pool at {x:.3} m: depth {got:.5e} m, scheme's steady profile {scheme:.5e} m ({rel:+.2e}), closed form {want:.5e} m (sampling {sampling:.2e}), local substeps up to {most}");
        assert!(
            got > 1.0e-3,
            "x {x}: {got:.3e} m is not deeper than the film"
        );
        assert!(
            rel.abs() <= 0.01,
            "x {x}: depth {got:.5e} vs the steady profile {scheme:.5e} ({rel:+.3e})"
        );
        assert!(
            sampling <= 0.03,
            "x {x}: the steady profile is {sampling:.3e} from the closed form"
        );
        compared += 1;
    }
    assert_eq!(compared, 2);
}

/// oracle: a new bed keeps the water. Replacing the bed (every third cell 3 µm lower, the next
/// 2 µm higher: within the join step, so the pool stays a still pool) keeps every cell's
/// depth, so no water is made or lost: the total and the ledger are unchanged, and stepping
/// on the new bed keeps the ledger to the last quantum. (A bed change of millimetres under a
/// deep pool leaves its surface in steps, an inertial flow the model refuses.)
#[test]
fn a_new_bed_keeps_every_cells_water() {
    let mut f = busy_film(FilmScheme::DiffusionWave);
    for _ in 0..30 {
        f.advance(Fix128::from_ratio(1, 6)).expect("advance");
    }
    let (depths, total, ledger) = (f.depths().to_vec(), f.total(), f.ledger());
    let old = f.bed().clone();
    let heights: Vec<Fix128> = old
        .heights()
        .iter()
        .enumerate()
        .map(|(c, q)| {
            let m = quanta_to_metres(*q);
            match c % 3 {
                0 => m - Fix128::from_ratio(3, 1_000_000),
                1 => m + Fix128::from_ratio(2, 1_000_000),
                _ => m,
            }
        })
        .collect();
    let new = FilmBed::from_heights(
        old.nx(),
        old.nz(),
        old.cell(),
        (Fix128::ZERO, Fix128::ZERO),
        &heights,
        2,
    )
    .expect("bed");
    assert_ne!(new.heights(), old.heights());
    f.set_bed(new).expect("set_bed");
    assert_eq!(f.depths(), depths.as_slice());
    assert_eq!(f.total(), total);
    assert_eq!(f.ledger(), ledger);
    for _ in 0..30 {
        f.advance(Fix128::from_ratio(1, 6))
            .expect("advance on the new bed");
        assert_eq!(f.total(), f.ledger().expected());
    }
}

/// A face deeper than the flux law accepts (4 m) is refused before its depth is cubed: the
/// advance returns out of range and changes nothing.
#[test]
fn a_face_deeper_than_the_law_accepts_is_refused() {
    let bed = FilmBed::from_heights(
        2,
        1,
        mm(CELL_MM),
        (Fix128::ZERO, Fix128::ZERO),
        &[Fix128::ZERO; 2],
        1,
    )
    .expect("bed");
    let mut f = film(bed, FilmScheme::DiffusionWave, FilmEdge::Closed);
    assert!(f.set_initial_depth(0, 0, 1i64 << 44));
    let before = state(&f);
    assert_eq!(
        f.advance(Fix128::from_ratio(1, 256)),
        Err(ParticipantFault::OutOfRange)
    );
    assert_eq!(state(&f), before);
}

/// A pool deeper than the law accepts: two cells side by side, both 2⁴⁴ quanta (4 096 m)
/// deep and level, join as one still pool, so their face is levelled and never stepped by
/// the flux law. The depth limit still holds there (a pool's band sums `h̄³` in fixed point,
/// which wraps far above it): the advance is refused and changes nothing. Both cells at the
/// limit itself (the depth a face accepts) step.
#[test]
fn a_pool_deeper_than_the_law_accepts_is_refused() {
    let bed = FilmBed::from_heights(
        2,
        1,
        mm(CELL_MM),
        (Fix128::ZERO, Fix128::ZERO),
        &[Fix128::ZERO; 2],
        1,
    )
    .expect("bed");
    let mut f = film(bed, FilmScheme::DiffusionWave, FilmEdge::Closed);
    assert!(f.set_initial_depth(0, 0, 1i64 << 44));
    assert!(f.set_initial_depth(1, 0, 1i64 << 44));
    let before = state(&f);
    assert_eq!(
        f.advance(Fix128::from_ratio(1, 256)),
        Err(ParticipantFault::OutOfRange)
    );
    assert_eq!(state(&f), before);
    let at_limit = MAX_FACE_DEPTH as i64;
    let bed = f.bed().clone();
    let mut g = film(bed, FilmScheme::DiffusionWave, FilmEdge::Closed);
    assert!(g.set_initial_depth(0, 0, at_limit));
    assert!(g.set_initial_depth(1, 0, at_limit));
    assert!(g.advance(Fix128::from_ratio(1, 256)).is_ok());
}

/// A drowned sill: water 28 mm deep (surface −2 mm) beside a sill 20 mm higher than its bed
/// holding water whose surface (−8 mm) stands over the sill's crest. The face is deeper than
/// the film, its step (6 mm) is above the join step, the lower side is submerged and the flux
/// is at critical: inertial, outside the model. The advance is refused, names both cells, and
/// changes nothing.
#[test]
fn a_drowned_sill_is_refused() {
    let beds = [mm(50), mm(-30), mm(-10), mm(50)];
    let mut f = walled_row(
        4,
        CELL_MM,
        &beds,
        FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed),
    );
    let (b1, b2) = (f.bed().heights()[5], f.bed().heights()[6]);
    f.set_initial_depth(1, 1, metres_to_quanta(mm(-2)) - b1);
    f.set_initial_depth(2, 1, metres_to_quanta(mm(-8)) - b2);
    let before = state(&f);
    assert_eq!(
        f.advance(Fix128::from_ratio(1, 20)),
        Err(ParticipantFault::OutOfRange)
    );
    assert!(
        f.last_band_cells().contains(&5) && f.last_band_cells().contains(&6),
        "band {:?}",
        f.last_band_cells()
    );
    assert_eq!(state(&f), before);
    // and a later advance that is accepted clears the report
    f.set_initial_depth(1, 1, metres_to_quanta(mm(-11)) - b1);
    f.set_initial_depth(2, 1, 0);
    f.advance(Fix128::from_ratio(1, 20))
        .expect("still water below the crest");
    assert!(
        f.last_band_cells().is_empty(),
        "stale report {:?}",
        f.last_band_cells()
    );
}

/// A deposit is a rate, not a burst: a flat pool 3 mm deep (19 cells of 1 cm, walled) fed
/// 5·10⁻⁶ m²/s at its head drains over a crest 0.6 mm below its surface into a sump. Its
/// throughflow is the smaller of what enters and what leaves, here the feed: carrying it needs
/// a fall of `ΔH = 3 ν q L / (g h̄³)` ≈ 11 µm, within its 1 % level tolerance (30 µm), so it
/// stays a still pool, levelled. While it drains faster than it is fed, counting each 1/6 s
/// deposit as inflow in every one of the advance's substeps (43 times its rate) would make
/// the throughflow the outflow over the crest, beyond the tolerance, and free the pool.
#[test]
fn a_gently_fed_pool_stays_level() {
    let nx = 22;
    let mut beds = vec![mm(50)];
    beds.extend(core::iter::repeat_n(Fix128::ZERO, nx - 3));
    // a crest 2.4 mm high before a drop into a sump: the pool drains over it towards the
    // film that carries the feed alone (about 0.2 mm deep)
    beds.push(Fix128::from_ratio(24, 10_000));
    beds.push(mm(-1000));
    let mut f = walled_row(
        nx,
        CELL_MM,
        &beds,
        FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed),
    );
    for i in 1..nx - 2 {
        f.set_initial_depth(i, 1, metres_to_quanta(mm(3)));
    }
    f.set_initial_depth(nx - 2, 1, metres_to_quanta(Fix128::from_ratio(6, 10_000)));
    let (q, l) = (5.0e-6_f64, 0.01_f64);
    let step = Fix128::from_ratio(1, 6);
    let (x0, z0) = f.bed().centre(1, 1);
    let mut levelled = 0;
    for _ in 0..30 {
        f.deposit(x0, z0, Fix128::from_f64(q * l) * step);
        f.advance(step).expect("advance");
        let interior = (3..nx - 3).map(|i| i + nx);
        if interior.clone().all(|c| f.last_pooled_cells().contains(&c)) {
            levelled += 1;
        }
    }
    assert_eq!(
        levelled,
        30,
        "the pool was freed in {} of 30 advances",
        30 - levelled
    );
}

/// The local substep count of a supercritical deep face, against its closed form. Two cells
/// on one floor, 5 mm and 4 mm deep (walled): the face is deeper than the film, its step
/// (1 mm) is above the join step, the lower side is submerged and the beds are level, so it
/// steps by the film flux, which here is the critical one (the lubrication flux of a 1 mm
/// step at 5 mm depth is far larger). It needs `⌈8 t_c / E⌉` local substeps for the critical
/// transfer `t_c = ⌊⌊√(H³)⌋ · N_c / 2⁶⁴⌋` never to meet the `⌊E/8⌋` cap (the count decides the
/// state's bits, so it is pinned here).
#[test]
fn a_supercritical_deep_face_takes_its_critical_count() {
    let beds = [mm(50), Fix128::ZERO, Fix128::ZERO, mm(50)];
    let mut f = walled_row(
        4,
        CELL_MM,
        &beds,
        FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed),
    );
    let (h, e) = (metres_to_quanta(mm(5)), metres_to_quanta(mm(1)));
    f.set_initial_depth(1, 1, h);
    f.set_initial_depth(2, 1, h - e);
    let k = FluxConstants::new(
        mm(CELL_MM),
        f.substep(),
        Fix128::from_ratio(980_665, 100_000),
        Fix128::from_ratio(1, 1_000_000),
    )
    .expect("constants");
    let h3 = u128::from(h.unsigned_abs()).pow(3);
    let critical = (isqrt_u128(h3) * k.critical) >> 64;
    let lubrication = h3 * u128::from(e.unsigned_abs()) / k.lubrication;
    assert!(critical < lubrication, "the face is not critical-bound");
    let want = (8 * critical).div_ceil(u128::from(e.unsigned_abs()));
    f.step_once(f.substep()).expect("substep");
    assert_eq!(u128::from(f.last_local_substeps()), want);
    assert!(want > 1, "the scene needs no local substeps");
}
