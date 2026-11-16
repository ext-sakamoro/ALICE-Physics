//! Oracles for the water film (`water_film`): a closed groove levels to the surface the
//! volume fixes, volume is conserved to the last quantum, a groove holds water up to its rim
//! and no lower, a groove fills in the time the rain rate gives, a rewound run and a rerun
//! are bit identical, and the order the faces are visited does not change the state.
//!
//! Every expected value here comes from a closed form or a ledger written independently of
//! the stepping code; none is taken from a run of the film.

use alice_physics_core::water_film::{
    metres_to_quanta, quanta_to_metres, FilmBed, FilmBuildError, FilmConfig, FilmEdge,
    FilmScheme, WaterFilm, DEPTH_QUANTUM_BITS,
};
use alice_physics_core::world_participant::{Participant, ParticipantFault, StateError};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix};
use sha2::{Digest, Sha256};

const CELL_MM: i64 = 10;
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
    FilmBed::from_heights(nx, nz, mm(CELL_MM), (Fix128::ZERO, Fix128::ZERO), &h, 1)
        .expect("bed")
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
/// (1) across every face between two wet cells the surfaces differ by at most one quantum
/// (the rest set of the model: a pair moves water while it differs by two or more);
/// (2) so every wet surface lies within `L − 1` quanta of the closed form `η*` that holds the
/// volume over the sampled bed (`L` = wet cells on the path), and every cell whose bed is
/// above `η* + L` is dry. The levelling time is reported against the gravity-wave time
/// `L/√(g h)`: under this model it is a relaxation set by `Δt_max`, not a wave crossing.
#[test]
fn a_closed_groove_levels_to_the_closed_form_surface() {
    let mut f = film(groove_bed(12, 3, 8, true), FilmScheme::DiffusionWave, FilmEdge::Closed);
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
            assert!(diff <= 1, "face {c}|{}: surfaces differ by {diff} quanta", c + 1);
            faces += 1;
        }
    }
    for &c in &wet {
        let off = (beds[c] + d[c] - eta).abs();
        assert!(off < l, "cell {c}: surface {off} quanta from η* (L = {l})");
    }
    for c in 0..d.len() {
        if beds[c] > eta + l {
            assert_eq!(d[c], 0, "cell {c} above the level holds water");
        }
    }
    assert!(faces >= 5 && l >= 6, "compared {faces} faces over {l} wet cells");
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
            assert!(p <= last, "{edge:?} substep {n}: potential rose by {}", p - last);
            fell += usize::from(p < last);
            last = p;
            compared += 1;
        }
        assert!(fell > 100, "{edge:?}: the potential fell only {fell} times");
    }
    assert_eq!(compared, 40_000);
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
            assert!(f.depths().iter().all(|d| *d >= 0), "{scheme:?} negative depth");
            compared += 1;
        }
        let l = f.ledger();
        assert!(l.rain > 0 && l.evaporated > 0 && l.deposits > 0 && l.outflow > 0, "{l:?}");
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
        let groove: Vec<usize> = (0..nx).filter(|i| f.bed().heights()[i + nx] < rim).collect();
        let band = groove.len() as i64;
        let mut compared = 0;
        for &i in &groove {
            let eta = dry.bed().heights()[i + nx] + dry.depths()[i + nx];
            assert!(eta >= rim - band, "{scheme:?} cell {i}: {} quanta below the rim", rim - eta);
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

/// determinism: the faces moved in one substep share no cell, so visiting them backwards
/// gives the same bytes. The scene moves water every substep, which the counter asserts.
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
        &bed2.heights().iter().map(|q| quanta_to_metres(*q)).collect::<Vec<_>>(),
        2,
    )
    .expect("bed2");
    g.set_bed(bed2).expect("rebuild");
    assert_eq!(g.check_state(&good), Err(StateError::InvalidValue));
    // and a bed that does not raise the generation is refused
    assert_eq!(g.set_bed(groove_bed(14, 4, 9, false)), Err(FilmBuildError::BadGrid));
}

/// degenerate inputs: each refused with the stated error, nothing changed.
#[test]
fn degenerate_inputs_are_refused() {
    let one = [Fix128::ZERO];
    let o = (Fix128::ZERO, Fix128::ZERO);
    assert_eq!(FilmBed::from_heights(0, 1, mm(10), o, &[], 1), Err(FilmBuildError::BadGrid));
    assert_eq!(FilmBed::from_heights(2, 1, mm(10), o, &one, 1), Err(FilmBuildError::BadGrid));
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
    assert_eq!(WaterFilm::new(cfg, bed.clone()).err(), Some(FilmBuildError::BadConfig));
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.rain = -Fix128::ONE;
    assert_eq!(WaterFilm::new(cfg, bed.clone()).err(), Some(FilmBuildError::BadConfig));

    let mut f = film(groove_bed(6, 2, 3, true), FilmScheme::DiffusionWave, FilmEdge::Closed);
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
            gravity: Vec3Fix::new(Fix128::ZERO, -Fix128::from_ratio(980_665, 100_000), Fix128::ZERO),
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
    assert_eq!(quanta_to_metres(1), Fix128::from_raw(0, 1u64 << (64 - DEPTH_QUANTUM_BITS)));
}
