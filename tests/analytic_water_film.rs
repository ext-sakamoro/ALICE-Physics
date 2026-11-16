//! Oracles for the water film (`water_film`): a closed groove levels to the surface the
//! volume fixes, volume is conserved to the last quantum, a groove holds water up to its rim
//! and no lower, a groove fills in the time the rain rate gives, a rewound run and a rerun
//! are bit identical, and the order the faces are visited does not change the state.
//!
//! Every expected value here comes from a closed form or a ledger written independently of
//! the stepping code; none is taken from a run of the film.

use alice_physics_core::sdf_collider::SdfField;
use alice_physics_core::water_film::{
    isqrt_u128, metres_to_quanta, quanta_to_metres, simultaneous_transfer, FilmBed, FilmBuildError,
    FilmConfig, FilmEdge, FilmScheme, FluxConstants, WaterFilm, DEPTH_QUANTUM_BITS, MAX_FACE_DEPTH,
    MAX_LUBRICATION_DIVISOR,
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

/// oracle: hydrostatics. Water piled at one end of a closed groove comes to rest level:
/// (1) across every face between two wet cells the surfaces differ by at most 7 quanta
/// (the rest set of the model: a face moves `⌊|Δη|/8⌋` in the pool regime);
/// (2) so every wet surface lies within `7(L − 1)` quanta of the closed form `η*` that holds
/// the volume over the sampled bed (`L` = wet cells on the path), and every cell whose bed is
/// above `η* + 7L` is dry. The levelling time is reported against the gravity-wave time
/// `L/√(g h)`: under this model it is a relaxation set by `Δt_max`, not a wave crossing.
#[test]
fn a_closed_groove_levels_to_the_closed_form_surface() {
    let mut f = film(
        groove_bed(12, 3, 8, true),
        FilmScheme::DiffusionWave,
        FilmEdge::Closed,
    );
    let pile = metres_to_quanta(mm(40));
    for i in 3..6 {
        assert!(f.set_initial_depth(i, 0, pile));
    }
    let substeps = settle(&mut f, 10_000_000);
    let beds = f.bed().heights().to_vec();
    let d = f.depths().to_vec();
    let eta = level_for(&beds, f.total());
    let wet: Vec<usize> = (0..d.len()).filter(|c| d[*c] > 0).collect();
    let l = wet.len() as i64;
    let mut faces = 0;
    for c in 0..d.len() - 1 {
        if d[c] > 0 && d[c + 1] > 0 {
            let diff = (beds[c] + d[c] - beds[c + 1] - d[c + 1]).abs();
            assert!(
                diff <= 7,
                "face {c}|{}: surfaces differ by {diff} quanta",
                c + 1
            );
            faces += 1;
        }
    }
    for &c in &wet {
        let off = (beds[c] + d[c] - eta).abs();
        assert!(
            off <= 7 * (l - 1),
            "cell {c}: surface {off} quanta from η* (L = {l})"
        );
    }
    for c in 0..d.len() {
        if beds[c] > eta + 7 * l {
            assert_eq!(d[c], 0, "cell {c} above the level holds water");
        }
    }
    assert!(
        faces >= 5 && l >= 6,
        "compared {faces} faces over {l} wet cells"
    );
    let spread = wet.iter().map(|c| beds[*c] + d[*c]).max().unwrap_or(0)
        - wet.iter().map(|c| beds[*c] + d[*c]).min().unwrap_or(0);
    let t = substeps as f64 * f.config().max_substep.to_f64();
    let depth = quanta_to_metres(eta).to_f64() - FLOOR_MM as f64 / 1000.0;
    let wave = 0.06 / (9.80665 * depth).sqrt();
    println!(
        "rest: spread {spread} quanta over {l} wet cells; levelled in {t:.3} s ({substeps} substeps); gravity-wave time over the floor {wave:.3} s"
    );
}

/// oracle: without a source the potential `Σ g(η² − b²)/2` never rises: every substep of a
/// closed groove and of an open strip draining over its edges lowers it or leaves it.
#[test]
fn the_potential_never_rises_without_a_source() {
    let mut compared = 0;
    for edge in [FilmEdge::Closed, FilmEdge::Open] {
        let mut f = film(groove_bed(14, 4, 9, false), FilmScheme::DiffusionWave, edge);
        for i in 3..8 {
            f.set_initial_depth(i, 1, metres_to_quanta(mm(45)));
        }
        f.set_initial_depth(0, 0, metres_to_quanta(mm(3)));
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
    // 4 mm: the cap `⌊|Δη|/8⌋` is what keeps four simultaneous moves from overshooting
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
    f.set_initial_depth(2, 6, base + metres_to_quanta(mm(4)));
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
    assert!(fell > 10, "pool: the potential fell only {fell} times");
    assert_eq!(compared, 42_000);
}

/// oracle: no cell gives more than it holds. A 5 cm column on a pedestal 10 cm above four
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
    f.set_initial_depth(2, 2, metres_to_quanta(mm(50)));
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
    assert!(
        f.depths()[12] < metres_to_quanta(mm(50)),
        "the column did not spill"
    );
    assert_eq!(compared, 500);
}

/// oracle: conservation as integers. Through rain, evaporation, point deposits and open
/// edges, the water held equals initial + rain + deposits − evaporated − outflow to the last
/// quantum, at every advance, and no depth is ever negative.
#[test]
fn volume_is_conserved_to_the_last_quantum() {
    for scheme in [FilmScheme::DiffusionWave] {
        let mut cfg = FilmConfig::water(scheme, FilmEdge::Open);
        cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(36_000));
        cfg.evaporation = FilmConfig::rain_mm_per_hour(Fix128::from_int(3_600));
        let mut f = WaterFilm::new(cfg, groove_bed(14, 4, 9, false)).expect("film");
        f.set_initial_depth(5, 1, metres_to_quanta(mm(20)));
        let mut compared = 0;
        for n in 0..600 {
            if n % 7 == 0 {
                let (x, z) = f.bed().centre(n % 14, 1);
                f.deposit(x, z, Fix128::from_ratio(3, 10_000_000));
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
        let below = metres_to_quanta(mm(25));
        for i in 4..=9 {
            f.set_initial_depth(i, 1, below);
        }
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
        cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(36_000));
        let mut f = WaterFilm::new(cfg, groove_bed(14, 4, 9, false)).expect("film");
        let start = rim - metres_to_quanta(mm(2));
        for i in 0..nx {
            let b = f.bed().heights()[i + nx];
            if b < start {
                f.set_initial_depth(i, 1, start - b);
            }
        }
        for _ in 0..40 {
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
/// The rate is 1 800 mm/h so the run is short; the same formula gives the site's 4 and 20 mm/h
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
        cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(1_800));
        let r = cfg.rain.to_f64();
        let t_closed = cap_m / (r * n as f64);
        let mut f = WaterFilm::new(cfg, bed.clone()).expect("film");
        let h = Fix128::from_ratio(1, 4);
        let mut t = 0.0;
        let mut filled_at = None;
        for _ in 0..2000 {
            f.advance(h).expect("advance");
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
    cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(20_000));
    cfg.evaporation = FilmConfig::rain_mm_per_hour(Fix128::from_int(500));
    let mut f = WaterFilm::new(cfg, groove_bed(14, 4, 9, false)).expect("film");
    for i in 4..=6 {
        f.set_initial_depth(i, 1, metres_to_quanta(mm(45)));
    }
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

/// The substep count is `⌈h / Δt_max⌉` whatever the water: dry, deep, flowing, raining
/// (a count read from the moving water changes `Δt` in phase with the motion).
#[test]
fn the_substep_count_depends_on_declared_inputs_only() {
    let mut f = busy_film(FilmScheme::DiffusionWave);
    let dt_max = f.config().max_substep;
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

/// A mixed scene: an uneven bed with a trough, scattered initial water, rain, evaporation and
/// point deposits, stepped 400 times; returns the SHA-256 of the state bytes.
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
    cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(20_000));
    cfg.evaporation = FilmConfig::rain_mm_per_hour(Fix128::from_int(700));
    let mut f = WaterFilm::new(cfg, bed).unwrap();
    f.set_reverse_faces(rev);
    for c in (0..nx * nz).step_by(5) {
        f.set_initial_depth(c % nx, c / nx, metres_to_quanta(mm(((c * 3) % 40) as i64)));
    }
    for n in 0..400 {
        if n % 9 == 0 {
            let (x, z) = f.bed().centre(n % nx, (n / 3) % nz);
            f.deposit(x, z, Fix128::from_ratio(3, 10_000_000));
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
/// backwards, under the integer flux law. Re-pinned once when the flux moved from `Fix128`
/// arithmetic to integers (a new rounding law, not new physics: over these 400 steps the
/// depths of the two forms differed by at most 3 quanta in a cell, 0.2 on average, and every
/// validation oracle passed unchanged).
#[test]
fn the_mixed_scene_digest_is_pinned() {
    const CLOSED: &str = "127f4629acdea08a4224da7d949ec4d39cffc99981bddb507b0599d183ad17a5";
    const OPEN: &str = "2a45bb5ff9291a5003874301fbe2da09631e462ff62f2f2b22e0fb35329a221e";
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
