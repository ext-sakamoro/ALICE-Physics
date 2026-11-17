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
