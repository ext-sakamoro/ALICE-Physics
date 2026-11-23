//! Oracles for `smoke_plume`: buoyant heat and soot carried as integer quanta across the
//! faces of a MAC grid whose velocity the CFD solver steps.
//!
//! Expected values come from outside the module: the source ledger from the declared rate
//! times the elapsed time, the drain of one cell from the upwind recurrence written here in
//! integers, the burner rate from the Arrhenius closed form in floating point, and the
//! projection from the divergence summed here over the faces.

#![cfg(feature = "std")]

use alice_physics_core::det_math;
use alice_physics_core::eulerian_grid::MacGrid;
use alice_physics_core::smoke_fire::ArrheniusReaction;
use alice_physics_core::smoke_plume::{
    from_quanta, CellBox, QuantaField, SmokeBuildError, SmokeConfig, SmokePlume, SmokeSource,
    SourceKind, HEAT_QUANTUM_BITS, MAX_CELL_QUANTA, SMOKE_OBS_HEAT, SOOT_QUANTUM_BITS,
};
use alice_physics_core::world_participant::{Observed, Participant, ParticipantFault, StateError};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix};

const STEP: (i64, i64) = (1, 128);

fn config() -> SmokeConfig {
    let mut c = SmokeConfig::air();
    c.step_max = Fix128::from_ratio(STEP.0, STEP.1);
    c
}

fn step() -> Fix128 {
    Fix128::from_ratio(STEP.0, STEP.1)
}

/// 8 × 16 × 8 cells of 1 cm, heat and soot diffusing, a 2 × 1 × 2 source at the floor's centre heating at 50 K/s and
/// making 10⁻³ kg/(m³·s) of soot, and a warm, sooty puff above it.
fn plume() -> SmokePlume {
    // soot diffuses too (the default is 0), so every oracle on this scene covers both fluxes
    let mut c = config();
    c.soot_diffusivity_m2_s = Fix128::from_ratio(1, 10_000);
    let mut p = SmokePlume::new(8, 16, 8, Fix128::from_ratio(1, 100), c).unwrap();
    p.add_source(SmokeSource {
        cells: CellBox {
            lo: [3, 1, 3],
            hi: [5, 2, 5],
        },
        kind: SourceKind::Rates {
            heat_k_per_s: Fix128::from_int(50),
            soot_kg_m3_s: Fix128::from_ratio(1, 1000),
        },
    })
    .unwrap();
    p.set_initial(4, 6, 4, 3 << HEAT_QUANTUM_BITS, 1 << 20)
        .unwrap();
    p
}

/// [`plume`] with its top open, after each of `windows` windows of 64 steps (half a second).
fn open_plume_windows(windows: usize) -> Vec<SmokePlume> {
    let mut p = plume();
    p.set_open_top().unwrap();
    (0..windows)
        .map(|_| {
            p.advance(step() * Fix128::from_int(64)).unwrap();
            p.clone()
        })
        .collect()
}

fn state(p: &SmokePlume) -> Vec<u8> {
    let mut b = Vec::new();
    p.write_state(&mut b);
    b
}

/// Heat and soot held equal the ledger to the last quantum, and the ledger's source total
/// equals the declared rate times the elapsed time: 50 K/s over steps of 1/128 s is
/// 25/64 K = 25 · 2²⁶ quanta per cell per step exactly.
#[test]
fn heat_and_soot_are_conserved_to_the_last_quantum() {
    let mut p = plume();
    let steps = 48i64;
    let mut moved = 0u64;
    for _ in 0..steps {
        p.advance(step()).unwrap();
        moved += p.last_moved();
        let l = p.ledger();
        assert_eq!(p.heat().total(), Some(l.heat_expected()));
        assert_eq!(p.soot().total(), Some(l.soot_expected()));
        assert!(p.heat().quanta().iter().all(|q| *q >= 0));
    }
    let l = p.ledger();
    assert_eq!(l.heat_initial, 3 << HEAT_QUANTUM_BITS);
    assert_eq!(l.heat_added, 4 * steps * 25 * (1 << 26));
    // soot: 10⁻³ kg/(m³·s) · 1/128 s per step, floored over the whole run per cell
    let per_cell = (steps as i128 * (1i128 << SOOT_QUANTUM_BITS)) / (1000 * 128);
    assert_eq!(i128::from(l.soot_added), 4 * per_cell);
    // the scene moved heat: the plume rose above the source
    assert!(moved > 0);
    let f = p.heat();
    let above: i64 = (8..16)
        .flat_map(|j| (0..8).flat_map(move |k| (0..8).map(move |i| (i, j, k))))
        .map(|(i, j, k)| f.quanta()[f.index(i, j, k)])
        .sum();
    assert!(above > 0, "no heat reached the upper half");
}

/// After every step the walls carry no flow and the divergence left by the projection,
/// as the fraction of a cell's volume it would compress in one step, stays below 10⁻⁶.
#[test]
fn the_projection_leaves_a_bounded_divergence_and_closed_walls() {
    let mut p = plume();
    let dt = step();
    let mut worst = 0f64;
    let mut peak_speed = 0f64;
    for _ in 0..32 {
        p.advance(dt).unwrap();
        let g = p.velocity();
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        for k in 0..nz {
            for j in 0..ny {
                assert!(g.u[(nx + 1) * (j + ny * k)].is_zero());
                assert!(g.u[nx + (nx + 1) * (j + ny * k)].is_zero());
            }
        }
        for k in 0..nz {
            for i in 0..nx {
                assert!(g.v[i + nx * (ny + 1) * k].is_zero());
                assert!(g.v[i + nx * (ny + (ny + 1) * k)].is_zero());
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                assert!(g.w[i + nx * j].is_zero());
                assert!(g.w[i + nx * (j + ny * nz)].is_zero());
            }
        }
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let d = (g.u[i + 1 + (nx + 1) * (j + ny * k)]
                        - g.u[i + (nx + 1) * (j + ny * k)])
                        + (g.v[i + nx * (j + 1 + (ny + 1) * k)] - g.v[i + nx * (j + (ny + 1) * k)])
                        + (g.w[i + nx * (j + ny * (k + 1))] - g.w[i + nx * (j + ny * k)]);
                    worst = worst.max((d * dt / g.dx).to_f64().abs());
                    let (_, vc, _) = g.cell_velocity(i, j, k);
                    peak_speed = peak_speed.max(vc.to_f64().abs());
                }
            }
        }
    }
    assert!(peak_speed > 1e-3, "the scene did not move: {peak_speed}");
    assert!(worst < 1e-6, "divergence per step {worst:e}");
}

/// A state written mid-run and read into a fresh plume reruns to the same bits.
#[test]
fn rewind_and_rerun_are_bit_identical() {
    let mut a = plume();
    for _ in 0..12 {
        a.advance(step()).unwrap();
    }
    let saved = state(&a);
    for _ in 0..12 {
        a.advance(step()).unwrap();
    }
    let mut b = plume();
    b.check_state(&saved).unwrap();
    b.read_state(&saved);
    assert_eq!(state(&b), saved);
    for _ in 0..12 {
        b.advance(step()).unwrap();
    }
    assert_eq!(state(&a), state(&b));
}

/// A twin that visits the cells in reverse order holds the same bits.
#[test]
fn cell_order_does_not_change_the_state() {
    let mut a = plume();
    let mut b = plume();
    b.set_reverse_cells(true);
    for _ in 0..24 {
        a.advance(step()).unwrap();
        b.advance(step()).unwrap();
    }
    assert_eq!(state(&a), state(&b));
    assert!(a.last_moved() > 0);
}

/// Payloads with a negative cell, a carry of a whole quantum, another digest or another
/// length are refused and change nothing.
#[test]
fn a_tampered_payload_is_refused() {
    let mut p = plume();
    p.advance(step()).unwrap();
    let good = state(&p);
    assert_eq!(p.check_state(&good), Ok(()));
    let header = 4 + 8 + 8 + 8 + 8 * 4 + 4 + 4 + 8 + 4;
    let mut neg = good.clone();
    neg[header..header + 8].copy_from_slice(&(-1i64).to_le_bytes());
    assert_eq!(p.check_state(&neg), Err(StateError::InvalidValue));
    let mut over = good.clone();
    over[header..header + 8].copy_from_slice(&(MAX_CELL_QUANTA + 1).to_le_bytes());
    assert_eq!(p.check_state(&over), Err(StateError::InvalidValue));
    let mut whole = good.clone();
    whole[header + 8..header + 16].copy_from_slice(&1i64.to_le_bytes());
    assert_eq!(p.check_state(&whole), Err(StateError::InvalidValue));
    let mut soot_carry = good.clone();
    soot_carry[header + 48..header + 56].copy_from_slice(&1i64.to_le_bytes());
    assert_eq!(p.check_state(&soot_carry), Err(StateError::InvalidValue));
    let mut heat_dcarry = good.clone();
    heat_dcarry[header + 24..header + 32].copy_from_slice(&1i64.to_le_bytes());
    assert_eq!(p.check_state(&heat_dcarry), Err(StateError::InvalidValue));
    let mut digest = good.clone();
    digest[4] ^= 1;
    assert_eq!(p.check_state(&digest), Err(StateError::InvalidValue));
    assert!(matches!(
        p.check_state(&good[..good.len() - 1]),
        Err(StateError::Length { .. })
    ));
    // a plume with another source has another digest
    let mut other = plume();
    other
        .add_source(SmokeSource {
            cells: CellBox {
                lo: [0, 0, 0],
                hi: [1, 1, 1],
            },
            kind: SourceKind::Rates {
                heat_k_per_s: Fix128::ONE,
                soot_kg_m3_s: Fix128::ZERO,
            },
        })
        .unwrap();
    assert!(other.check_state(&good).is_err());
    assert_eq!(state(&p), good);
}

/// A field of spikes (alternate cells empty and holding 10⁹ quanta) through a pseudo-random
/// velocity scaled so the busiest cell sends out exactly half of a cell per substep: no cell
/// gives more than it holds and none goes below zero, and the total is unchanged.
#[test]
fn no_cell_goes_negative_at_the_largest_courant_number() {
    let n = 8;
    let mut g = MacGrid::new(n, n, n, Fix128::ONE);
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        Fix128::from_int((s % 2001) as i64 - 1000)
    };
    for x in g.u.iter_mut() {
        *x = next();
    }
    for x in g.v.iter_mut() {
        *x = next();
    }
    for x in g.w.iter_mut() {
        *x = next();
    }
    g.set_closed_box_walls();
    let m = QuantaField::max_out_courant(&g, Fix128::ONE);
    let dt = Fix128::ONE / (m * Fix128::from_int(2));
    let c = QuantaField::max_out_courant(&g, dt);
    assert!(c <= Fix128::from_ratio(1, 2) && c > Fix128::from_ratio(499, 1000));
    let mut f = QuantaField::new(n, n, n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                if (i + j + k) % 2 == 0 {
                    f.add(f.index(i, j, k), 1_000_000_000).unwrap();
                }
            }
        }
    }
    let total = f.total();
    for _ in 0..200 {
        let r = f.step(&g, dt, false);
        assert_eq!(r.overrun_cells, 0);
        assert!(r.min >= 0);
    }
    assert_eq!(f.total(), total);
}

/// One cell in a row, with flow out of it along +x at Courant number `c = 3/16` (exact in binary) and the
/// cell downstream draining at `1/2` (so it holds less than the first, the limiter's slope
/// stays zero and the face value is the cell's own): it sends
/// `⌊c q + r⌋` each substep and keeps the fraction `r` (the upwind recurrence, written here
/// in integers).
#[test]
fn an_isolated_cell_drains_at_the_upwind_rate() {
    let (a, b) = (3i128, 16i128);
    let (mut g, mut f) = row(16);
    for (i, x) in g.u.iter_mut().enumerate() {
        *x = if i <= 5 {
            Fix128::from_ratio(3, 16)
        } else {
            Fix128::from_ratio(1, 2)
        };
    }
    g.set_closed_box_walls();
    let c0 = f.index(4, 0, 0);
    f.add(c0, 1_000_003).unwrap();
    let (mut q, mut r) = (1_000_003i128, 0i128);
    for _ in 0..40 {
        f.step(&g, Fix128::ONE, false);
        let out = (a * q + r) / b;
        r = (a * q + r) % b;
        q -= out;
        assert_eq!(i128::from(f.quanta()[c0]), q);
        assert!(f.quanta()[f.index(5, 0, 0)] <= f.quanta()[c0]);
    }
}

/// A cell holding one quantum with flow out at Courant number 1/64 sends nothing for 63
/// substeps (a face moves whole quanta) and the quantum on the 64th: the carried fraction
/// makes it drain instead of sticking.
#[test]
fn a_cell_below_one_face_quantum_still_drains() {
    let (mut g, mut f) = row(16);
    set_row_velocity(&mut g, Fix128::from_ratio(1, 64));
    let c0 = f.index(4, 0, 0);
    f.add(c0, 1).unwrap();
    for _ in 0..63 {
        f.step(&g, Fix128::ONE, false);
        assert_eq!(f.quanta()[c0], 1);
    }
    f.step(&g, Fix128::ONE, false);
    assert_eq!(f.quanta()[c0], 0);
    assert_eq!(f.quanta()[f.index(5, 0, 0)], 1);
}

/// A puff of 5 · 10⁶ quanta carried along a row at Courant number 0.3 for 200 substeps:
/// the exact upwind tail more than 30 cells behind it holds below 10⁻⁶ quanta, so every cell
/// there must hold at most one quantum (a cell that keeps what it cannot send in whole
/// quanta would hold up to `⌈1/c⌉ − 1 = 3`).
#[test]
fn the_soot_left_behind_a_passing_puff_is_bounded() {
    let (mut g, mut f) = row(128);
    set_row_velocity(&mut g, Fix128::from_ratio(3, 10));
    for i in 2..7 {
        f.add(f.index(i, 0, 0), 1_000_000).unwrap();
    }
    let total = f.total();
    for _ in 0..200 {
        f.step(&g, Fix128::ONE, false);
    }
    assert_eq!(f.total(), total);
    let tail: Vec<i64> = (0..30).map(|i| f.quanta()[f.index(i, 0, 0)]).collect();
    assert!(tail.iter().all(|q| *q <= 1), "tail {tail:?}");
    // the puff has moved on: its peak is past cell 50
    let peak = (0..128)
        .max_by_key(|&i| f.quanta()[f.index(i, 0, 0)])
        .unwrap();
    assert!(peak > 50, "peak at {peak}");
}

fn row(n: usize) -> (MacGrid, QuantaField) {
    (
        MacGrid::new(n, 1, 1, Fix128::ONE),
        QuantaField::new(n, 1, 1),
    )
}

fn set_row_velocity(g: &mut MacGrid, c: Fix128) {
    for x in g.u.iter_mut() {
        *x = c;
    }
    g.set_closed_box_walls();
}

/// Bad grids, configs, sources, cells and steps are refused, and a refused advance changes
/// nothing.
#[test]
fn degenerate_inputs_are_refused() {
    let dx = Fix128::from_ratio(1, 100);
    for (nx, ny, nz) in [(2, 8, 8), (8, 6, 8), (8, 8, 512), (0, 8, 8)] {
        assert_eq!(
            SmokePlume::new(nx, ny, nz, dx, config()).unwrap_err(),
            SmokeBuildError::BadGrid
        );
    }
    for d in [
        Fix128::ZERO,
        Fix128::from_ratio(1, 10_000),
        Fix128::from_int(2),
    ] {
        assert_eq!(
            SmokePlume::new(8, 8, 8, d, config()).unwrap_err(),
            SmokeBuildError::BadGrid
        );
    }
    let mut c = config();
    c.step_max = Fix128::ZERO;
    assert_eq!(
        SmokePlume::new(8, 8, 8, dx, c).unwrap_err(),
        SmokeBuildError::BadConfig
    );
    let mut c = config();
    c.viscosity_pas = Fix128::ONE; // ν dt / l² far past 1/6
    assert_eq!(
        SmokePlume::new(8, 8, 8, dx, c).unwrap_err(),
        SmokeBuildError::BadConfig
    );
    let mut c = config();
    c.multigrid_cycles = 0;
    assert_eq!(
        SmokePlume::new(8, 8, 8, dx, c).unwrap_err(),
        SmokeBuildError::BadConfig
    );

    let mut p = plume();
    let rates = SourceKind::Rates {
        heat_k_per_s: Fix128::ONE,
        soot_kg_m3_s: Fix128::ZERO,
    };
    for cells in [
        CellBox {
            lo: [1, 1, 1],
            hi: [1, 2, 2],
        },
        CellBox {
            lo: [0, 0, 0],
            hi: [9, 1, 1],
        },
    ] {
        assert_eq!(
            p.add_source(SmokeSource { cells, kind: rates }),
            Err(SmokeBuildError::BadSource)
        );
    }
    let negative = SourceKind::Rates {
        heat_k_per_s: -Fix128::ONE,
        soot_kg_m3_s: Fix128::ZERO,
    };
    assert_eq!(
        p.add_source(SmokeSource {
            cells: CellBox {
                lo: [0, 0, 0],
                hi: [1, 1, 1]
            },
            kind: negative,
        }),
        Err(SmokeBuildError::BadSource)
    );
    assert_eq!(
        p.set_initial(8, 0, 0, 1, 1),
        Err(SmokeBuildError::CellOutOfRange)
    );
    assert_eq!(
        p.set_initial(0, 0, 0, -1, 0),
        Err(SmokeBuildError::CellOutOfRange)
    );

    let before = state(&p);
    assert_eq!(p.advance(Fix128::ZERO), Err(ParticipantFault::OutOfRange));
    assert_eq!(p.advance(-step()), Err(ParticipantFault::OutOfRange));
    // more velocity steps than the config allows
    assert_eq!(
        p.advance(Fix128::from_int(100)),
        Err(ParticipantFault::OutOfRange)
    );
    // a cell past its bound: a source that would take a full cell past 2⁶⁰ quanta
    let mut hot = SmokePlume::new(4, 4, 4, dx, config()).unwrap();
    hot.set_initial(0, 0, 0, MAX_CELL_QUANTA, 0).unwrap();
    hot.add_source(SmokeSource {
        cells: CellBox {
            lo: [0, 0, 0],
            hi: [1, 1, 1],
        },
        kind: rates,
    })
    .unwrap();
    let hot_before = state(&hot);
    assert_eq!(hot.advance(step()), Err(ParticipantFault::OutOfRange));
    assert_eq!(state(&hot), hot_before);
    // a field refuses to take a cell past the bound, and keeps the cell
    let mut f = QuantaField::new(4, 1, 1);
    assert_eq!(f.add(0, MAX_CELL_QUANTA), Some(()));
    assert_eq!(f.add(0, 1), None);
    assert_eq!(f.add(1, -1), None);
    assert_eq!(f.quanta()[..2], [MAX_CELL_QUANTA, 0]);
    // an initial amount past the bound, and a ledger past the integer range (eight full cells
    // hold 2⁶³ quanta), are refused and change nothing
    let mut full = SmokePlume::new(4, 4, 4, dx, config()).unwrap();
    assert_eq!(
        full.set_initial(0, 0, 0, MAX_CELL_QUANTA + 1, 0),
        Err(SmokeBuildError::CellOutOfRange)
    );
    for i in 0..4 {
        for k in 0..2 {
            let r = full.set_initial(i, 0, k, MAX_CELL_QUANTA, 0);
            if (i, k) == (3, 1) {
                assert_eq!(r, Err(SmokeBuildError::CellOutOfRange));
            } else {
                r.unwrap();
            }
        }
    }
    assert_eq!(full.ledger().heat_initial, 7 * MAX_CELL_QUANTA);
    assert_eq!(full.heat().quanta()[full.heat().index(3, 0, 1)], 0);
    assert_eq!(state(&p), before);
}

/// The plume as a world participant: the world steps it and reports the heat it holds,
/// `Σ ΔT · V`, equal to the ledger.
#[test]
fn the_plume_steps_inside_a_world() {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(
            Fix128::ZERO,
            -Fix128::from_ratio(980_665, 100_000),
            Fix128::ZERO,
        ),
        ..PhysicsConfig::default()
    });
    let at = world.add_participant(Box::new(plume())).unwrap();
    for _ in 0..8 {
        world.try_step(Fix128::from_ratio(1, 64)).unwrap();
    }
    // 8 world steps of 1/64 s are 16 plume steps of 1/128 s
    let added = 4 * 16 * 25 * (1i64 << 26);
    let dx = Fix128::from_ratio(1, 100);
    let expected =
        from_quanta((3 << HEAT_QUANTUM_BITS) + added, HEAT_QUANTUM_BITS) * (dx * dx * dx);
    let Some(Observed::Exact(sink)) = world.observe_participant(at) else {
        panic!("no observation");
    };
    let heat = sink
        .values()
        .iter()
        .find(|(ch, _)| *ch == SMOKE_OBS_HEAT)
        .map(|(_, v)| *v)
        .unwrap();
    assert_eq!(heat, expected);
}

/// A burner cell heats at `A e^{−E_a/(R T)} ρ_F ρ_O Δh_c / (ρ c_p)`: the first step's heat
/// against that closed form in floating point, to the quantum and the float's error.
#[test]
fn a_burner_heats_at_the_arrhenius_rate() {
    let mut p = SmokePlume::new(4, 4, 4, Fix128::from_ratio(1, 100), config()).unwrap();
    let reaction = ArrheniusReaction::pla_air();
    p.add_source(SmokeSource {
        cells: CellBox {
            lo: [1, 1, 1],
            hi: [2, 2, 2],
        },
        kind: SourceKind::Burner {
            reaction,
            fuel_kg_m3: Fix128::from_ratio(1, 10),
            oxidizer_kg_m3: Fix128::from_ratio(23, 100),
        },
    })
    .unwrap();
    // start the cell hot so the reaction runs: 900 K
    p.set_initial(1, 1, 1, 600 << HEAT_QUANTUM_BITS, 0).unwrap();
    p.advance(step()).unwrap();
    let t: f64 = 900.0;
    let r = 5.0e8 * det_math::exp64(-150_000.0 / (8.314 * t)) * 0.1 * 0.23;
    let k_per_s = r * 18.0e6 / (1.2 * 1005.0);
    let expect = k_per_s / 128.0 * (1u64 << HEAT_QUANTUM_BITS) as f64;
    let got = p.ledger().heat_added as f64;
    assert!(expect > 1e6, "the burner barely runs: {expect}");
    // the cell takes whole quanta: the floor of the closed form, to the float's error
    assert!(
        (got - expect.floor()).abs() <= 1e-9 * expect,
        "{got} vs {expect}"
    );
    let soot = r * 0.02 / 128.0 * (1u64 << SOOT_QUANTUM_BITS) as f64;
    let got_soot = p.ledger().soot_added as f64;
    assert!(
        (got_soot - soot.floor()).abs() <= 1e-9 * soot,
        "{got_soot} vs {soot}"
    );
}

/// A hot column (300 K above the air) rises fast enough that a cell sends out more than half
/// of itself in one velocity step: each step takes `⌈2 max Σ_out c⌉` transport substeps,
/// read here from the velocity the step left, and at least one step needs two or more.
#[test]
fn the_transport_substeps_keep_half_a_cell_per_substep() {
    let mut p = SmokePlume::new(8, 16, 8, Fix128::from_ratio(1, 100), config()).unwrap();
    for j in 1..5 {
        for (i, k) in [(3, 3), (4, 3), (3, 4), (4, 4)] {
            p.set_initial(i, j, k, 300 << HEAT_QUANTUM_BITS, 0).unwrap();
        }
    }
    let mut most = 0;
    for _ in 0..40 {
        p.advance(step()).unwrap();
        let c = QuantaField::max_out_courant(p.velocity(), step());
        let expect = (c * Fix128::from_int(2)).ceil().hi.max(1);
        assert_eq!(i64::from(p.last_transport_substeps()), expect);
        most = most.max(p.last_transport_substeps());
        assert_eq!(p.heat().total(), Some(p.ledger().heat_expected()));
    }
    assert!(most >= 2, "the scene never needed a second substep: {most}");
}

/// A soot puff in still air spreads as the diffusion equation says: the explicit scheme's
/// second moment grows by exactly `2 d l²` per axis per substep (`d = κ Δt_s / l²`), so
/// after one velocity step `σ² = 2 κ Δt` per axis while no quantum has reached a wall (the
/// stencil reaches one cell per substep, the walls are 7.5 cells away), and the puff stays
/// centred. Two diffusivities: one inside the explicit limit (`κ Δt / l² = 0.078`, one
/// substep per step, four steps) and one 4.7 times past it (`0.78`, which needs five).
#[test]
fn a_puff_spreads_with_variance_two_kappa_t() {
    for (kappa, steps, substeps) in [((1, 1000), 4, 1), ((1, 100), 1, 5)] {
        let mut c = config();
        c.soot_diffusivity_m2_s = Fix128::from_ratio(kappa.0, kappa.1);
        let mut p = SmokePlume::new(16, 16, 16, Fix128::from_ratio(1, 100), c).unwrap();
        p.set_initial(8, 8, 8, 0, 1 << 50).unwrap();
        for _ in 0..steps {
            p.advance(step()).unwrap();
            assert_eq!(p.last_diffusion_substeps(), substeps);
        }
        assert_eq!(p.soot().total(), Some(1 << 50));
        let f = p.soot();
        let l = 0.01f64;
        let (mut m0, mut m1, mut m2) = ([0f64; 3], [0f64; 3], [0f64; 3]);
        for k in 0..16 {
            for j in 0..16 {
                for i in 0..16 {
                    let q = f.quanta()[f.index(i, j, k)] as f64;
                    for (a, x) in [i, j, k].into_iter().enumerate() {
                        let x = (x as f64 - 8.0) * l;
                        m0[a] += q;
                        m1[a] += q * x;
                        m2[a] += q * x * x;
                    }
                }
            }
        }
        let t = steps as f64 / 128.0;
        let expect = 2.0 * (kappa.0 as f64 / kappa.1 as f64) * t;
        for a in 0..3 {
            let mean = m1[a] / m0[a];
            let var = m2[a] / m0[a] - mean * mean;
            assert!(mean.abs() < 1e-9, "axis {a}: centre moved {mean:e} m");
            assert!(
                ((var - expect) / expect).abs() < 1e-9,
                "axis {a}: {var:e} vs {expect:e}"
            );
        }
        // nothing reached the walls
        for k in 0..16 {
            for j in 0..16 {
                assert_eq!(f.quanta()[f.index(0, j, k)], 0);
                assert_eq!(f.quanta()[f.index(15, j, k)], 0);
            }
        }
    }
}

/// One quantum alone with two empty neighbours, at diffusion number 1/64: it sends nothing
/// for 31 substeps (a face moves whole quanta) and the quantum on the 32nd, when its carried
/// fraction `2 · 32/64` reaches one, so a gradient below one quantum per face does not
/// freeze.
#[test]
fn a_gradient_below_one_quantum_still_diffuses() {
    let mut f = QuantaField::new(16, 1, 1);
    let c0 = f.index(4, 0, 0);
    f.add(c0, 1).unwrap();
    let d = Fix128::from_ratio(1, 64);
    for _ in 0..31 {
        f.diffuse(d, false);
        assert_eq!(f.quanta()[c0], 1);
    }
    f.diffuse(d, false);
    assert_eq!(f.quanta()[c0], 0);
    assert_eq!(f.total(), Some(1));
    // the tie goes to the earlier direction (−x)
    assert_eq!(f.quanta()[f.index(3, 0, 0)], 1);
}

#[path = "support/smoke_laminar_scene.rs"]
mod laminar_scene;

/// The push-time guard of the laminar plume oracle (`examples/smoke_laminar_plume.rs`, run
/// weekly): the same scene at 10 mm for 3 velocity steps. The heat held equals the declared
/// rate times the source's cells times the elapsed time, floored to a quantum per cell at every
/// step (oracle: each step adds ⌊100 K/s × 1/120 s⌋ in quanta to each of the 8 cells, 3 steps);
/// the air over the source rises
/// (buoyancy); and the state's SHA-256 is pinned, so a change in the scene or the wiring is
/// seen at push time without the long run.
#[test]
fn the_laminar_plume_scene_starts_rising_with_its_declared_heat() {
    use sha2::{Digest, Sha256};
    let mut p = laminar_scene::scene(10, 200, 100);
    for _ in 0..3 {
        p.advance(laminar_scene::step(10)).unwrap();
    }
    // floored per step, not once over the run: 3·⌊(100 · 2³²) / 120⌋, one quantum below 2.5 K
    let per_cell = 3 * ((100_i128 << HEAT_QUANTUM_BITS) / 120);
    assert_eq!(p.heat().total(), Some((8 * per_cell) as i64));
    assert_eq!(p.ledger().heat_expected(), (8 * per_cell) as i64);
    // the source spans cells 15..17 on x and z and 1..3 on y: the air just above its top rises
    let w = p.velocity().cell_velocity(16, 3, 16).1;
    assert!(
        w > Fix128::ZERO,
        "the air over the source must rise, got {}",
        w.to_f64()
    );
    let mut state = Vec::new();
    p.write_state(&mut state);
    let digest: String = Sha256::digest(&state)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest,
        "7221c60fb6bdf5341a6e9bc5bd662833b48bd66ab9cc239638ca566d1d2e7c02"
    );
}

/// A column of `n` cells of 1 m with walls, and the velocity `c` (m/s, so the Courant
/// number at a step of 1 s) on every interior y face and on the top face; the floor face
/// stays a wall.
fn column(n: usize, c: Fix128, top: Fix128) -> (MacGrid, QuantaField) {
    let mut g = MacGrid::new(1, n, 1, Fix128::ONE);
    g.set_closed_box_walls();
    for j in 1..n {
        g.v[j] = c;
    }
    g.v[n] = top;
    (g, QuantaField::new(1, n, 1))
}

/// The open top is upwind: the top cell sends `⌊c q⌋` out at Courant number `c`, whatever
/// the cell below holds (oracle: `⌊1000 / 4⌋ = 250`). A slope across the top face would
/// change it: a zero ghost outside with 2000 below gives the MUSCL face value
/// `1000 + 3/8 · minmod(−1000, −1000) = 625`, i.e. 156 out; a linear ghost with 500 below
/// gives `1000 + 3/8 · 500 = 1187`, i.e. 296 out. A closed top or a top face pointing in
/// lets nothing out, and the quanta in the column plus the quanta out are the quanta it
/// started with.
#[test]
fn the_open_top_sends_the_upwind_share_out() {
    let quarter = Fix128::from_ratio(1, 4);
    for below in [2000i64, 500] {
        let (g, mut f) = column(4, quarter, quarter);
        f.add(f.index(0, 3, 0), 1000).unwrap();
        f.add(f.index(0, 2, 0), below).unwrap();
        let (_, left) = f.step_open_top(&g, Fix128::ONE, false, true);
        assert_eq!(left, 250, "below {below}");
        assert_eq!(f.total().unwrap() + left, 1000 + below);
        // the same column closed: the top face's velocity is not used
        let (g, mut f) = column(4, quarter, quarter);
        f.add(f.index(0, 3, 0), 1000).unwrap();
        f.add(f.index(0, 2, 0), below).unwrap();
        let (_, left) = f.step_open_top(&g, Fix128::ONE, false, false);
        assert_eq!(left, 0);
        assert_eq!(f.total(), Some(1000 + below));
    }
    // air coming in through the top brings no excess and takes none out
    let (g, mut f) = column(4, quarter, Fix128::ZERO - quarter);
    f.add(f.index(0, 3, 0), 1000).unwrap();
    let (_, left) = f.step_open_top(&g, Fix128::ONE, false, true);
    assert_eq!(left, 0);
    assert_eq!(f.total(), Some(1000));
    // and the quanta leave whole, with the cell's carried fraction: c = 1/64 on one quantum
    let (g, mut f) = column(4, Fix128::ZERO, Fix128::from_ratio(1, 64));
    f.add(f.index(0, 3, 0), 1).unwrap();
    let mut out = 0;
    for n in 1..=64 {
        out += f.step_open_top(&g, Fix128::ONE, false, true).1;
        assert_eq!(out, i64::from(n == 64), "after {n}");
    }
    assert_eq!(f.total(), Some(0));
}

/// The plume of [`plume`] with its top open: heat and soot leave through the top, and what
/// the cells hold equals what they were given less what left, to the last quantum, at
/// every step; the same plume closed keeps everything.
#[test]
fn an_open_top_lets_heat_out_and_the_ledger_still_closes() {
    let mut open = plume();
    open.set_open_top().unwrap();
    assert!(open.open_top());
    let mut closed = plume();
    for n in 0..384 {
        open.advance(step()).unwrap();
        assert_eq!(
            open.heat().total(),
            Some(open.heat_held_expected()),
            "step {n}"
        );
        assert_eq!(
            open.soot().total(),
            Some(open.soot_held_expected()),
            "step {n}"
        );
        if n % 64 == 63 {
            closed.advance(step() * Fix128::from_int(64)).unwrap();
        }
    }
    let out = open.outflow();
    assert!(
        out.heat_left > 0 && out.soot_left > 0,
        "nothing left: {out:?}"
    );
    assert_eq!(
        open.ledger().heat_expected(),
        open.heat().total().unwrap() + out.heat_left
    );
    assert_eq!(closed.outflow().heat_left, 0);
    assert_eq!(closed.heat().total(), Some(closed.ledger().heat_expected()));
    // the top really is open: air leaves through some top face
    let g = open.velocity();
    let top = (0..g.nz)
        .flat_map(|k| (0..g.nx).map(move |i| (i, k)))
        .map(|(i, k)| g.v[i + g.nx * (g.ny + (g.ny + 1) * k)])
        .fold(Fix128::ZERO, |m, v| if v > m { v } else { m });
    assert!(top > Fix128::ZERO, "no air leaves through the top");
}

/// Opening the top is refused once the plume has stepped; an open plume's state carries
/// its outflow, rereads to the same bytes, and a closed plume refuses it (and the other way),
/// by its digest even when the lengths agree.
#[test]
fn an_open_plume_state_round_trips_and_is_told_apart() {
    let mut p = plume();
    p.advance(step()).unwrap();
    assert_eq!(p.set_open_top(), Err(SmokeBuildError::BadConfig));
    let mut open = plume();
    open.set_open_top().unwrap();
    for _ in 0..4 {
        open.advance(step() * Fix128::from_int(64)).unwrap();
    }
    assert!(open.outflow().heat_left > 0);
    let bytes = state(&open);
    let mut again = plume();
    again.set_open_top().unwrap();
    again.check_state(&bytes).unwrap();
    again.read_state(&bytes);
    assert_eq!(state(&again), bytes);
    assert_eq!(again.outflow(), open.outflow());
    assert!(plume().check_state(&bytes).is_err());
    assert!(again.check_state(&state(&plume())).is_err());
    // the open state without its outflow bytes has a closed plume's length: only the digest,
    // which names the open top, tells them apart
    let cut = &bytes[..bytes.len() - 16];
    assert_eq!(plume().check_state(cut), Err(StateError::InvalidValue));
}

/// The heat that left through the top of the open plume over each half-second window from
/// `first` (counted from 0) to `windows.len() - 1`, over what the source added in that window:
/// the declared 50 K/s on 4 cells is `4 · 25 · 2²⁶` quanta per step exactly.
fn outflow_ratios(windows: &[SmokePlume], first: usize) -> Vec<f64> {
    let per_window = 64 * 4 * 25 * (1i64 << 26);
    (first..windows.len())
        .map(|w| {
            let last = if w == 0 {
                0
            } else {
                windows[w - 1].outflow().heat_left
            };
            (windows[w].outflow().heat_left - last) as f64 / per_window as f64
        })
        .collect()
}

/// The open plume of [`plume`] comes to a steady state in which the heat leaving through the
/// top balances the source: over each half second from 6.5 to 7.5 s the outflow is the source
/// within 2 %, their mean within 0.5 % (measured, with the run of
/// `the_open_top_carries_the_source_away_from_seven_to_twelve_seconds`: 1.0060, 1.0025,
/// 1.0006, mean 1.0030; the first outflow comes at 2.5 s and overshoots to 1.27 at 3.5 s,
/// within 2 % from 6 s on). The same bounds over 7 to 12 s are that test's, run weekly.
#[test]
fn the_open_top_carries_the_source_away_at_steady_state() {
    let windows = open_plume_windows(15);
    let ratios = outflow_ratios(&windows, 12);
    assert_eq!(ratios.len(), 3);
    assert!(ratios.iter().all(|r| (r - 1.0).abs() < 0.02), "{ratios:?}");
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    assert!((mean - 1.0).abs() < 0.005, "mean {mean}, {ratios:?}");
}

/// The open plume of [`plume`] comes to a steady state in which the heat leaving through the
/// top balances the source: the outflow over each half second from 7 to 12 s is that within
/// 2 %, their mean within 0.5 % (measured: each within 0.0093, the mean 1.000; the first
/// outflow comes at 2.5 s and overshoots to 1.27 at 3.5 s; after about 15 s heat collects in
/// the narrow box again and the ratio falls to 0.96 by 20 s, so the window is where the store
/// is steady).
#[test]
#[ignore = "runtime: 1536 plume steps, about 47 s without optimisation; the lib test checks \
            6.5 to 7.5 s; run by run_ignored.py"]
fn the_open_top_carries_the_source_away_from_seven_to_twelve_seconds() {
    let windows = open_plume_windows(24);
    let ratios = outflow_ratios(&windows, 14);
    assert_eq!(ratios.len(), 10);
    assert!(ratios.iter().all(|r| (r - 1.0).abs() < 0.02), "{ratios:?}");
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    assert!((mean - 1.0).abs() < 0.005, "mean {mean}, {ratios:?}");
}

/// The net outflow of cell `(i, j, k)`, `Σ_faces (u_out − u_in)` (the divergence times the
/// cell's side).
fn net_out(g: &MacGrid, i: usize, j: usize, k: usize) -> Fix128 {
    let (nx, ny) = (g.nx, g.ny);
    (g.u[i + 1 + (nx + 1) * (j + ny * k)] - g.u[i + (nx + 1) * (j + ny * k)])
        + (g.v[i + nx * (j + 1 + (ny + 1) * k)] - g.v[i + nx * (j + (ny + 1) * k)])
        + (g.w[i + nx * (j + ny * (k + 1))] - g.w[i + nx * (j + ny * k)])
}

/// [`plume`] after half a second, prolonged to 16 × 32 × 16 cells of 5 mm: every quantum
/// is kept (each coarse cell's 8 children sum to it and differ by at most one), the ledger
/// carries over, a fine face that coincides with a coarse face has its velocity, every
/// fine cell has its parent's divergence (its net outflow is half the parent's, the cell
/// being half as wide: oracle, the linear-along-the-normal rule written here, to the two
/// units of the last place its halvings round away), and the fine
/// plume steps on with its ledger closed, the doubled source cells adding
/// `32 · ⌊50 · 2³² / 256⌋ = 32 · 50 · 2²⁴` quanta per step of 1/256 s.
#[test]
fn prolongation_keeps_every_quantum_and_each_parents_divergence() {
    let mut coarse = plume();
    coarse.advance(step() * Fix128::from_int(64)).unwrap();
    let mut c = config();
    c.soot_diffusivity_m2_s = Fix128::from_ratio(1, 10_000);
    c.step_max = Fix128::from_ratio(1, 256);
    let mut fine = coarse.prolonged(c).unwrap();
    assert_eq!(fine.ledger(), coarse.ledger());
    assert_eq!(fine.steps(), coarse.steps());
    for (cf, ff) in [(coarse.heat(), fine.heat()), (coarse.soot(), fine.soot())] {
        assert_eq!(ff.total(), cf.total());
        for k in 0..8 {
            for j in 0..16 {
                for i in 0..8 {
                    let kids: Vec<i64> = (0..8)
                        .map(|n| {
                            ff.quanta()[ff.index(
                                2 * i + (n & 1),
                                2 * j + ((n >> 1) & 1),
                                2 * k + (n >> 2),
                            )]
                        })
                        .collect();
                    assert_eq!(kids.iter().sum::<i64>(), cf.quanta()[cf.index(i, j, k)]);
                    let (lo, hi) = (kids.iter().min().unwrap(), kids.iter().max().unwrap());
                    assert!(hi - lo <= 1, "children {kids:?}");
                }
            }
        }
    }
    let (cg, fg) = (coarse.velocity(), fine.velocity());
    assert!(
        cg.v.iter().any(|v| !v.is_zero()),
        "the coarse plume did not move"
    );
    for k in 0..16 {
        for j in 0..32 {
            for i in 0..=16 {
                if i % 2 == 0 {
                    assert_eq!(
                        fg.u[i + 17 * (j + 32 * k)],
                        cg.u[i / 2 + 9 * (j / 2 + 16 * (k / 2))]
                    );
                }
            }
        }
    }
    let half = Fix128::from_ratio(1, 2);
    let mut worst = 0f64;
    for k in 0..16 {
        for j in 0..32 {
            for i in 0..16 {
                let parent = net_out(cg, i / 2, j / 2, k / 2) * half;
                worst = worst.max((net_out(fg, i, j, k) - parent).to_f64().abs());
            }
        }
    }
    // rounding only: each axis's mean face loses at most half a unit of the last place (2⁻⁶⁴)
    // when halved, three axes, and halving the parent's another half: at most 2 · 2⁻⁶⁴
    let ulp = 1.0 / (1u128 << 64) as f64;
    assert!(
        worst > 0.0 && worst <= 2.0 * ulp,
        "a child's divergence differs from its parent's by {worst:e}"
    );
    let added = fine.ledger().heat_added;
    fine.advance(Fix128::from_ratio(1, 256)).unwrap();
    assert_eq!(fine.ledger().heat_added - added, 32 * 50 * (1i64 << 24));
    assert_eq!(fine.heat().total(), Some(fine.heat_held_expected()));
    assert_eq!(fine.soot().total(), Some(fine.soot_held_expected()));
}

/// An open plume prolongs open, with its outflow; a grid past 256 cells or cells below 1 mm
/// is refused.
#[test]
fn prolongation_keeps_the_open_top_and_refuses_what_the_fine_grid_cannot_be() {
    let mut open = plume();
    open.set_open_top().unwrap();
    open.advance(step() * Fix128::from_int(64)).unwrap();
    open.advance(step() * Fix128::from_int(64)).unwrap();
    open.advance(step() * Fix128::from_int(64)).unwrap();
    open.advance(step() * Fix128::from_int(64)).unwrap();
    open.advance(step() * Fix128::from_int(64)).unwrap();
    open.advance(step() * Fix128::from_int(64)).unwrap();
    assert!(open.outflow().heat_left > 0);
    let mut c = config();
    c.step_max = Fix128::from_ratio(1, 256);
    let fine = open.prolonged(c).unwrap();
    assert!(fine.open_top());
    assert_eq!(fine.outflow(), open.outflow());
    assert_eq!(fine.heat().total(), Some(fine.heat_held_expected()));
    let tall = SmokePlume::new(4, 256, 4, Fix128::from_ratio(1, 100), config()).unwrap();
    assert_eq!(
        tall.prolonged(config()).err(),
        Some(SmokeBuildError::BadGrid)
    );
    let small = SmokePlume::new(4, 4, 4, Fix128::from_ratio(1, 1000), config()).unwrap();
    assert_eq!(
        small.prolonged(config()).err(),
        Some(SmokeBuildError::BadGrid)
    );
}

/// The laminar scene is one physical scene at every cell: at 10, 5 and 2.5 mm its box and its
/// source span the same lengths (oracle, in micrometres: the box 320 000 wide and 640 000 tall,
/// the source from 150 000 to 170 000 across and from 10 000 to 30 000 up), and the 10 mm scene
/// prolonged to 5 mm is the 5 mm scene, so the 5 mm scene accepts its state (a source placed a
/// fixed number of cells up, not a fixed height, would sit lower on the finer grid and both
/// fail).
#[test]
fn the_laminar_scene_has_the_same_lengths_at_every_cell() {
    for dx_um in [10_000i64, 5_000, 2_500] {
        let g = laminar_scene::geometry(dx_um);
        let at = |cells: usize| cells as i64 * dx_um;
        assert_eq!(
            (at(g.n_xz), at(g.n_y)),
            (320_000, 640_000),
            "box at {dx_um} µm"
        );
        for axis in [0, 2] {
            assert_eq!(
                (at(g.source.lo[axis]), at(g.source.hi[axis])),
                (150_000, 170_000),
                "source across at {dx_um} µm"
            );
        }
        assert_eq!(
            (at(g.source.lo[1]), at(g.source.hi[1])),
            (10_000, 30_000),
            "source height at {dx_um} µm"
        );
    }
    let fine_scene = laminar_scene::scene(5, 200, 400);
    let fine = laminar_scene::scene(10, 200, 400)
        .prolonged(*fine_scene.config())
        .unwrap();
    let mut bytes = Vec::new();
    fine.write_state(&mut bytes);
    fine_scene.check_state(&bytes).unwrap();
}

/// The averaging window of the laminar plume run opens only on an outflow that has stayed
/// steady for a transit (oracle: the means of the declared ratios, computed outside the rule):
/// (1) the recorded outflow of [`plume`] with its top open (outflow / source per half second,
/// 0 to 6.5 s: 0, 0, 0, 0, 0.0206, 1.0078, 1.2689, 1.1009, 1.0529, 1.0361, 1.0182, 1.0105, 1.0060,
/// 1.0024, 1.0005, 1.0002, the overshoot of the first arrival at the top, then 1.0000): over a
/// span of 4 s the two-sided test first holds at 8 s (1.0159), a one-sided `left >= 0.95 · added`
/// at 7 s inside the overshoot (1.0618), and the sustained rule opens at 11 s, the first second
/// with 8, 9, 10 and 11 s all inside the band; (2) an outflow passing through the band while the
/// box fills (per second 0, 0.2, 0.5, 0.8, 0.95, 1.02, 1.09, 1.16, 1.18, 1.14, 1.08, 1.03, 1.01,
/// then 1.00): the two-sided test alone holds at 7 s (0.965) and leaves the band at 8 s (1.055),
/// the sustained rule waits to 17 s; (3) the span is the box's height at the plume's speed in
/// whole seconds rounded up: 0.64 m at 0.27 m/s is 3 s, at 0.16 m/s 4 s, at 0.64 m/s 1 s, and no
/// span without an upward speed.
#[test]
fn the_plume_window_opens_on_a_sustained_steady_outflow() {
    // one (left, added) pair per whole second
    type History = [(i64, i64)];
    let opens = |hist: &History, rule: &dyn Fn(&History) -> bool| {
        (1..=hist.len()).find(|&s| rule(&hist[..s])).map(|s| s - 1)
    };
    // (1) the ratios per half second in units of 10⁻⁴, against 10⁶ quanta of source each
    let mut halves = vec![
        0i64, 0, 0, 0, 206, 10078, 12689, 11009, 10529, 10361, 10182, 10105, 10060, 10024, 10005,
        10002,
    ];
    halves.extend([10_000; 12]);
    let (mut left, mut added) = (0i64, 0i64);
    let mut hist = vec![(0i64, 0i64)];
    for (h, r) in halves.iter().enumerate() {
        added += 1_000_000;
        left += r * 100;
        if h % 2 == 1 {
            hist.push((left, added));
        }
    }
    assert_eq!(
        opens(&hist, &|h| laminar_scene::outflow_steady(h, 4)),
        Some(8)
    );
    let one_sided = |h: &History| {
        h.len() > 4 && {
            let (now, then) = (h[h.len() - 1], h[h.len() - 5]);
            now.1 > then.1 && (now.0 - then.0) * 100 >= 95 * (now.1 - then.1)
        }
    };
    assert_eq!(opens(&hist, &one_sided), Some(7));
    assert_eq!(
        opens(&hist, &|h| laminar_scene::sustained_steady(h, 4)),
        Some(11)
    );
    // (2) per second, in units of 10⁻⁴
    let seconds = [
        0i64, 2000, 5000, 8000, 9500, 10200, 10900, 11600, 11800, 11400, 10800, 10300, 10100,
        10000, 10000, 10000, 10000, 10000, 10000, 10000,
    ];
    let (mut left, mut added) = (0i64, 0i64);
    let mut hist = vec![(0i64, 0i64)];
    for r in seconds {
        added += 1_000_000;
        left += r * 100;
        hist.push((left, added));
    }
    assert_eq!(
        opens(&hist, &|h| laminar_scene::outflow_steady(h, 4)),
        Some(7)
    );
    assert!(
        !laminar_scene::outflow_steady(&hist[..9], 4),
        "it leaves the band at 8 s"
    );
    assert_eq!(
        opens(&hist, &|h| laminar_scene::sustained_steady(h, 4)),
        Some(17)
    );
    // (3)
    assert_eq!(laminar_scene::transit_span_s(0.27), Some(3));
    assert_eq!(laminar_scene::transit_span_s(0.16), Some(4));
    assert_eq!(laminar_scene::transit_span_s(0.64), Some(1));
    assert_eq!(laminar_scene::transit_span_s(0.0), None);
    assert_eq!(laminar_scene::transit_span_s(-0.1), None);
}
