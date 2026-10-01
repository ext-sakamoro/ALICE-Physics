//! Oracles for the two-way thermoplastic coupling driven to a fixed point.
//!
//! The one-directional legs are pinned elsewhere — `analytic_thermoplastic_
//! softening.rs` for temperature entering the return mapping, and
//! `analytic_plastic_dissipation.rs` for plastic work becoming heat. What is
//! left to pin here is the thing neither of those can see: that the increment
//! the driver returns satisfies **both** legs at once, rather than one leg
//! evaluated at the other's previous guess.
//!
//! # What "satisfies both legs" is checked against
//!
//! Three independent axes, because a fixed point is easy to claim and hard to
//! claim honestly:
//!
//! 1. **The relaxation does not move it.** `ω` changes how the iteration
//!    travels and must not change where it lands. Measured on the scene below,
//!    `ω = 1, 1/2, 1/4` take 10, 34 and 76 sweeps and agree on the converged
//!    rise to eleven significant digits. ⚠️ A driver that returned the last
//!    iterate without converging would be `ω`-dependent, and one that ignored
//!    `ω` would take the same number of sweeps for all three.
//! 2. **The thermal leg holds to the bit.** The returned rise has to be the
//!    deposit of the returned increment's own plastic work — not of some other
//!    sweep's.
//! 3. **The mechanical leg holds too.** Solving the mechanics on its own at
//!    the returned rise has to reproduce the returned increment. Together with
//!    (2) this is what "both legs at once" means, and it is the one claim only
//!    a coupled driver can make.
//!
//! ⚠️ **(3) is not a pointwise closed form, and it cannot be.** The converged
//! rise is non-uniform, so each element gets its own `σ_y(ΔT_e)`; a softer
//! element cannot carry the same stress as a stiffer neighbour, so the strain
//! redistributes and the local axial strain is no longer the prescribed
//! average. Measured on this scene: element 0 carries **3.147794758 MPa**
//! where the pointwise bilinear curve at its own `ΔT = 1.419522305 K` would
//! give **3.132705421 MPa**, a 0.48 % gap that is equilibrium, not error. The
//! pointwise closed form is pinned where it is valid — a **uniform** field —
//! in `analytic_thermoplastic_softening.rs`.
//!
//! And one that proves the exercise is not vacuous: **the coupled answer has
//! to differ from the isothermal one** by a margin far above the tolerances.
//!
//! # ⚠️ What is deliberately not claimed
//!
//! **That any relaxation beats `ω = 1`.** Measured on these scenes nothing
//! does: under-relaxation costs sweeps (10 / 34 / 76 at `ω` = 1 / ½ / ¼,
//! `c_v = 2⁻⁸`) and so does over-relaxation (10 / 14 / 21 / 29 for `ω` = 1 /
//! 1.112 / 1.25 / 1.367), even though the measured spectrum says the second
//! group should win. The spectrum and why it does not decide the sweep count
//! are below; the oracle here checks only that `ω` does not move the *answer*.
//!
//! ⚠️ **Under-relaxation does not repair this splitting, and over-relaxation
//! does not help either — both measured.** The Jacobian at the fixed point was
//! taken by finite differences over the 25 nodes the deposit writes:
//! `λ ∈ {+0.280, −0.078}` at `c_v = 2⁻⁸` and `{+0.938, −0.401}` at
//! `c_v = 2⁻¹⁰`. Which effect owns which sign is the physics one would guess,
//! and switching them off one at a time confirms it: **softening is the
//! positive eigenvalue** (hotter → weaker → more work → hotter) and
//! **expansion is the negative one** (hotter → more eigenstrain → less elastic
//! trial strain → less work) — with `α = 0` the negative eigenvalue vanishes
//! (`−0.000`), with no softening law the dominant one turns negative
//! (`−0.146`). The alternating early iterates are the negative eigenvalue in
//! the transient; the positive one sets the tail.
//!
//! ⚠️ **The sweep count is not predicted by that spectrum**, because the
//! iteration reaches the floor while still in its transient: at `c_v = 2⁻⁸` the
//! early sweeps contract by 0.084 each against a `λ_max` of 0.280. So
//! Richardson's equalising `ω* = 2/(2 − λ_max − λ_min)` — 1.112 and 1.367 here,
//! both over-relaxation — is **worse** in practice: measured sweeps at
//! `c_v = 2⁻⁸` are 10 / 14 / 21 / 29 / 45 / 174 for `ω` = 1 / 1.112 / 1.25 /
//! 1.367 / 1.5 / 1.75, and at `c_v = 2⁻¹⁰` over-relaxation above 1.25 fails
//! outright. `ω = 1` is the best value on every scene tried, which is why the
//! range stops there.
//!
//!
//! **That a non-contracting scene reports `Diverging`.** At `c_v = 2⁻¹⁴` the
//! sweep settles into a **period-two cycle**: the first sweep deposits 177.7 K,
//! which is past the point where `σ_y(T)` clamps at zero, a body with no yield
//! stress carries no stress and so dissipates nothing, the next sweep deposits
//! exactly zero, and the one after is back to 177.7. The residual is flat, so
//! the honest verdict is `Stagnated` with `best_residual == first_residual` —
//! and the oracle below asserts exactly that, rather than a divergence the
//! sequence does not show.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::coupled_iteration::{CoupledIterationError, SubIterationConfig};
use alice_physics::linear_elastic_fem::{
    deposit_increment_heat, step_thermoplastic, Axis, BoundaryConditions, ElasticMaterial,
    ElastoplasticConfig, ElastoplasticIncrementRequest, ElastoplasticProblem, ElastoplasticState,
    FemError, PlasticHeating, SolverConfig, ThermalExpansion, ThermalSoftening,
    ThermoplasticCoupling,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

const E_MPA: f64 = 1024.0;
const NU: f64 = 0.25;
const SIGMA_Y: f64 = 2.0;
const H_PLASTIC: f64 = 1024.0;
const ALPHA_PER_K: f64 = 1.0 / 4096.0;
const YIELD_LOST_PER_K: f64 = 1.0 / 8.0;
const HARDENING_LOST_PER_K: f64 = 1.0 / 16.0;
const EPS_TOTAL: f64 = 0.005;
/// Volumetric heat capacity of the scene the convergence oracles use, MPa/K.
///
/// Chosen from a measured sweep: it puts the converged rise at 2.58 K, well
/// inside the range where `σ_y(T)` has not clamped, and gives `ρ ≈ 0.079` so
/// that `ω = 1/4` still converges inside an 80-sweep budget.
const C_V: f64 = 1.0 / 256.0;
/// Heat capacity at which the sweep settles into a period-two cycle.
const C_V_CYCLE: f64 = 1.0 / 16384.0;
/// `2⁻³⁰` of the first sweep's deposit. See `ThermoplasticCoupling::try_new`.
const FLOOR_FRACTION_RAW: u64 = 1 << 34;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn node(i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * 3 + k * 6).expect("the 3x2x2 lattice fits u32")
}

fn bar_mesh() -> SdfTetMesh {
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let (nx, ny, nz, h) = (2usize, 1usize, 1usize, 2.0f32);
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
            }
        }
    }
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node(i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node(i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

fn bar_bc(eps: f64) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for k in 0..=1 {
        for j in 0..=1 {
            bc.prescribe(node(0, j, k), Axis::X, Fix128::ZERO);
            bc.prescribe(node(2, j, k), Axis::X, fx(eps) * Fix128::from_int(4));
        }
    }
    bc.prescribe(node(0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node(0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node(0, 1, 0), Axis::Z, Fix128::ZERO);
    bc
}

fn config() -> ElastoplasticConfig {
    ElastoplasticConfig::try_new(
        SolverConfig::default(),
        60,
        Fix128::from_raw(0, 1 << 24),
        fx(SIGMA_Y),
        fx(H_PLASTIC),
    )
    .expect("a valid elastoplastic config")
}

/// A grid covering the bar with a one-cell margin, at the reference
/// temperature, so the increment starts from `ΔT = 0`.
fn base_field() -> CoupledField {
    base_field_at(0.0)
}

/// The same grid, filled with an arbitrary absolute temperature.
fn base_field_at(absolute: f64) -> CoupledField {
    CoupledField::try_new_filled(
        7,
        5,
        5,
        (
            Fix128::from_int(-1),
            Fix128::from_int(-1),
            Fix128::from_int(-1),
        ),
        (
            Fix128::from_int(5),
            Fix128::from_int(3),
            Fix128::from_int(3),
        ),
        fx(absolute),
    )
    .expect("a grid with at least two nodes per axis")
}

fn softening() -> ThermalSoftening {
    ThermalSoftening::try_new(fx(YIELD_LOST_PER_K), fx(HARDENING_LOST_PER_K))
        .expect("fractions in range")
}

fn coupling(
    heat_capacity: f64,
    taylor_quinney: f64,
    relaxation: f64,
    max_sweeps: u32,
) -> Result<ThermoplasticCoupling, FemError> {
    coupling_at(heat_capacity, taylor_quinney, relaxation, max_sweeps, 0.0)
}

fn coupling_at(
    heat_capacity: f64,
    taylor_quinney: f64,
    relaxation: f64,
    max_sweeps: u32,
    material_reference: f64,
) -> Result<ThermoplasticCoupling, FemError> {
    ThermoplasticCoupling::try_new(
        PlasticHeating::try_new(fx(taylor_quinney), fx(heat_capacity))?,
        fx(ALPHA_PER_K),
        fx(material_reference),
        Some(softening()),
        fx(relaxation),
        Fix128::from_raw(0, FLOOR_FRACTION_RAW),
        SubIterationConfig::new(
            max_sweeps,
            Fix128::from_raw(0, 1 << 24),
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(5, 4),
            4,
        )
        .expect("a valid sub-iteration config"),
    )
}

/// The bar, its virgin state and a base field, ready for a driver call.
struct Scene {
    mesh: SdfTetMesh,
    problem: ElastoplasticProblem,
    state: ElastoplasticState,
    base: CoupledField,
}

fn scene(eps: f64) -> Scene {
    let mesh = bar_mesh();
    let material = ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and nu in (-1, 0.5)");
    // `ElastoplasticProblem` borrows its inputs, so the scene owns a leaked
    // copy of each: these live for the whole test and nothing else reads them.
    let material: &'static ElasticMaterial = Box::leak(Box::new(material));
    let boundary: &'static BoundaryConditions = Box::leak(Box::new(bar_bc(eps)));
    let config: &'static ElastoplasticConfig = Box::leak(Box::new(config()));
    let owned_mesh: &'static SdfTetMesh = Box::leak(Box::new(bar_mesh()));
    let problem = ElastoplasticProblem::try_new(owned_mesh, material, boundary, config)
        .expect("the bar prepares");
    let state = problem.virgin_state();
    Scene {
        mesh,
        problem,
        state,
        base: base_field(),
    }
}

fn centroid(mesh: &SdfTetMesh, element: usize) -> Vec3Fix {
    let quarter = Fix128::from_raw(0, 1 << 62);
    let mut sum = Vec3Fix::ZERO;
    for &v in &mesh.tets[element].vertices {
        let p = mesh.vertices[v as usize];
        sum = sum
            + Vec3Fix::new(
                Fix128::from_f32(p[0]),
                Fix128::from_f32(p[1]),
                Fix128::from_f32(p[2]),
            );
    }
    sum * quarter
}

/// Largest node-wise difference between two rises on the same grid.
fn rise_gap(a: &CoupledField, b: &CoupledField) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| (x.to_f64() - y.to_f64()).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Oracles
// ---------------------------------------------------------------------------

/// Agreement required between fixed points reached with different `ω`.
///
/// The fixed point itself jitters: across sweeps at a single `ω` the converged
/// rise moves by about `5e-11`, a relative `2e-11` that is the arithmetic's own
/// noise, and the spread across `ω = 1, 1/2, 1/4` was measured at `3e-11` —
/// the same band. `1e-8` is three hundred times that and still eight orders
/// below the 2.58 K answer, so an `ω`-dependent fixed point cannot hide here.
const OMEGA_AGREEMENT: f64 = 1.0e-8;
/// Tolerance on the stress, MPa, for the replay above.
///
/// Derived, not chosen: the floor is `2⁻³⁰` of the first sweep's deposit, so on
/// this scene it is about `2.4e-9` K, and `dσ/dΔT` here is `σ_y₀ w_y` plus the
/// tangent times `α`, about `0.37 MPa/K`. The replay can therefore differ by
/// roughly `9e-10 MPa`. `1e-6` leaves three orders of margin and is still four
/// orders below the `1.5e-2` gap a wrong pairing of the two halves produces.
const STRESS_TOL: f64 = 1.0e-6;
/// Same number, named for the replay it bounds.
const REPLAY_TOL: f64 = STRESS_TOL;

/// Where the iteration lands cannot depend on how it travels.
#[test]
fn the_fixed_point_does_not_depend_on_the_relaxation() {
    let s = scene(EPS_TOTAL);
    let mut answers = Vec::new();
    for omega in [1.0, 1.0 / 2.0, 1.0 / 4.0] {
        let law = coupling(C_V, 1.0, omega, 80).expect("a valid coupling");
        let done = step_thermoplastic(&s.problem, &s.state, &s.mesh, &s.base, &law, Fix128::ONE)
            .unwrap_or_else(|e| panic!("omega {omega} should converge in 80 sweeps, got {e:?}"));
        answers.push((omega, done.report.sweeps, done.temperature_increment));
    }

    for window in answers.windows(2) {
        let (omega_a, _, ref a) = window[0];
        let (omega_b, _, ref b) = window[1];
        let gap = rise_gap(a, b);
        assert!(
            gap <= OMEGA_AGREEMENT,
            "omega {omega_a} and omega {omega_b} reached rises that differ by {gap:.3e} K, \
             above the tolerance {OMEGA_AGREEMENT:.3e}. The relaxation must change the path, \
             not the fixed point"
        );
    }

    // The tooth: if the driver ignored `ω` the three runs would be the same
    // run, and the agreement above would be vacuous.
    let sweeps: Vec<u32> = answers.iter().map(|(_, n, _)| *n).collect();
    assert!(
        sweeps.windows(2).all(|w| w[0] < w[1]),
        "sweep counts {sweeps:?} are not strictly increasing as omega falls, so the relaxation \
         is not reaching the update and the agreement above says nothing"
    );
}

/// The returned rise is the deposit of the returned increment's own work.
#[test]
fn the_returned_rise_is_the_deposit_of_the_returned_increment() {
    let s = scene(EPS_TOTAL);
    let law = coupling(C_V, 1.0, 1.0, 80).expect("a valid coupling");
    let done = step_thermoplastic(&s.problem, &s.state, &s.mesh, &s.base, &law, Fix128::ONE)
        .expect("the coupled increment solves");

    let heating = PlasticHeating::try_new(Fix128::ONE, fx(C_V)).expect("a valid heating law");
    let mut redone = s.base.clone();
    redone.clear();
    deposit_increment_heat(
        &s.mesh,
        &done.increment.plastic_work_increment,
        &heating,
        &mut redone,
    )
    .expect("the deposit succeeds");

    // Exact: both are the same deposit of the same work onto the same grid, so
    // the thermal leg of the coupled pair holds to the bit.
    assert_eq!(
        done.temperature_increment.as_slice(),
        redone.as_slice(),
        "the returned rise is not the deposit of the returned increment's plastic work, so the \
         two halves of the answer came from different sweeps"
    );
    // Not vacuous: an increment that dissipated nothing would make both sides
    // all-zero and the equality would hold for the wrong reason.
    assert!(
        redone.max_value() > Fix128::ZERO,
        "the increment deposited no heat, so the equality above compares two zero grids"
    );
}

/// Solving the mechanics alone at the returned rise reproduces the increment.
///
/// This is the mechanical half of "both legs at once". The thermal half is
/// exact (see above); this one holds to within the residual floor, because the
/// sweep evaluated the mechanics at the relaxed iterate and the returned rise
/// is the deposit of the resulting work — the two differ by at most the floor,
/// which is what convergence means.
#[test]
fn solving_the_mechanics_at_the_returned_rise_reproduces_the_increment() {
    let s = scene(EPS_TOTAL);
    let law = coupling(C_V, 1.0, 1.0, 80).expect("a valid coupling");
    let done = step_thermoplastic(&s.problem, &s.state, &s.mesh, &s.base, &law, Fix128::ONE)
        .expect("the coupled increment solves");

    // The base is at the reference temperature, so the returned rise is the
    // absolute field as well as the rise.
    let rise = TemperatureRise::from_absolute(&done.temperature_increment, Fix128::ZERO);
    let replayed = s
        .problem
        .step(
            &s.state,
            &ElastoplasticIncrementRequest::new(Fix128::ONE).with_thermal(
                ThermalExpansion::from_rise(&rise, fx(ALPHA_PER_K)),
                Some(softening()),
            ),
        )
        .expect("the replay solves");

    for (e, (a, b)) in done
        .increment
        .field
        .element_stress
        .iter()
        .zip(replayed.field.element_stress.iter())
        .enumerate()
    {
        let gap = (a.xx.to_f64() - b.xx.to_f64()).abs();
        assert!(
            gap <= REPLAY_TOL,
            "element {e}: the coupled increment carries {:.12} MPa but replaying the mechanics \
             at the returned rise gives {:.12} MPa, a difference of {gap:.3e} above the floor-\
             derived tolerance {REPLAY_TOL:.3e}. The two halves of the answer do not agree",
            a.xx.to_f64(),
            b.xx.to_f64()
        );
    }

    // Not vacuous: the rise has to be non-zero (otherwise this is the
    // isothermal replay) and non-uniform (otherwise one centroid would do).
    let rises: Vec<f64> = (0..s.mesh.tets.len())
        .map(|e| {
            done.temperature_increment
                .sample(centroid(&s.mesh, e))
                .to_f64()
        })
        .collect();
    let lowest = rises.iter().copied().fold(f64::MAX, f64::min);
    let highest = rises.iter().copied().fold(f64::MIN, f64::max);
    assert!(
        lowest > 0.0,
        "every element sampled a rise of zero, so this is the isothermal replay"
    );
    assert!(
        highest - lowest > 1.0e-3,
        "the converged rise is uniform across the bar ({lowest:.9} to {highest:.9} K), so the \
         per-element sampling is not exercised"
    );
}

/// The coupling has to change the answer, or none of the above means anything.
#[test]
fn the_coupled_answer_differs_from_the_isothermal_one() {
    let s = scene(EPS_TOTAL);
    let law = coupling(C_V, 1.0, 1.0, 80).expect("a valid coupling");
    let coupled = step_thermoplastic(&s.problem, &s.state, &s.mesh, &s.base, &law, Fix128::ONE)
        .expect("the coupled increment solves");
    let isothermal = s
        .problem
        .step(&s.state, &ElastoplasticIncrementRequest::new(Fix128::ONE))
        .expect("the isothermal increment solves");

    let gap = (coupled.increment.field.element_stress[0].xx.to_f64()
        - isothermal.field.element_stress[0].xx.to_f64())
    .abs();
    assert!(
        gap > 1.0e3 * STRESS_TOL,
        "the coupled and isothermal increments differ by {gap:.3e} MPa, within a thousand \
         tolerances of each other, so the coupling is not reaching the answer"
    );
}

/// More of the plastic work turned into heat means a larger converged rise.
///
/// This is a statement about the law the driver applies — `δT ∝ β` through
/// `PlasticHeating` — not about the stress the body settles at, which is an
/// equilibrium and has no required ordering.
#[test]
fn the_converged_rise_grows_with_the_taylor_quinney_fraction() {
    let s = scene(EPS_TOTAL);
    let mut rises = Vec::new();
    for beta in [1.0 / 4.0, 1.0 / 2.0, 1.0] {
        let law = coupling(C_V, beta, 1.0, 80).expect("a valid coupling");
        let done = step_thermoplastic(&s.problem, &s.state, &s.mesh, &s.base, &law, Fix128::ONE)
            .unwrap_or_else(|e| panic!("beta {beta} should converge, got {e:?}"));
        rises.push((beta, done.temperature_increment.max_value().to_f64()));
    }
    for window in rises.windows(2) {
        let (beta_a, a) = window[0];
        let (beta_b, b) = window[1];
        assert!(
            b > a,
            "beta {beta_a} reached a rise of {a:.9} K and the larger beta {beta_b} only \
             {b:.9} K; the fraction of work released as heat scales the deposit directly"
        );
    }
}

/// A flat sequence is stagnation, not divergence.
#[test]
fn a_period_two_cycle_is_reported_as_stagnation_not_divergence() {
    let s = scene(EPS_TOTAL);
    let law = coupling(C_V_CYCLE, 1.0, 1.0, 40).expect("a valid coupling");
    let error = step_thermoplastic(&s.problem, &s.state, &s.mesh, &s.base, &law, Fix128::ONE)
        .expect_err("a period-two cycle is not a fixed point");
    match error {
        FemError::CoupledSubIterationFailed(CoupledIterationError::Stagnated {
            best_residual,
            first_residual,
            ..
        }) => {
            // Flat to the bit: the cycle returns the same two states forever,
            // so the residual never improves on its first value. That is what
            // separates this from a contraction that ran out of budget.
            assert_eq!(
                best_residual, first_residual,
                "the residual improved on its first value, so the sequence is contracting and \
                 the verdict should have been NotConverged rather than Stagnated"
            );
        }
        other => panic!(
            "expected Stagnated on a flat residual sequence, got {other:?}. A period-two cycle \
             neither grows nor shrinks, so Diverging would send a caller to change the splitting \
             and NotConverged would send them to raise the budget, and neither is the problem"
        ),
    }
}

/// An increment that stays elastic is its own fixed point immediately.
#[test]
fn an_elastic_increment_converges_in_one_sweep_with_no_rise() {
    // Half the yield strain: nothing yields, so nothing dissipates.
    let s = scene(1.0 / 1024.0);
    let law = coupling(C_V, 1.0, 1.0, 40).expect("a valid coupling");
    let done = step_thermoplastic(&s.problem, &s.state, &s.mesh, &s.base, &law, Fix128::ONE)
        .expect("an elastic increment solves");
    assert_eq!(
        done.report.sweeps, 1,
        "an increment that dissipates nothing deposits nothing, so the first sweep already \
         reproduces δT = 0 and the residual is exactly zero"
    );
    assert_eq!(
        done.temperature_increment.max_value(),
        Fix128::ZERO,
        "an elastic increment raised the temperature"
    );
    assert_eq!(
        done.increment.plastic_work_increment.iter().copied().max(),
        Some(Fix128::ZERO),
        "the scene was meant to stay elastic but some element dissipated"
    );
}

/// Only the rise above the reference drives the coupling, not the absolute
/// temperature.
///
/// The material parameters were measured at `material_reference`, so a body
/// sitting at that temperature is unsoftened and unexpanded however large the
/// number is. Two scenes with the same *difference* and different absolutes
/// therefore have to give the same answer, to the bit.
///
/// ⚠️ This is what makes the reference observable at all. Every other scene in
/// this file uses a reference of zero, where dropping the reference from the
/// `from_absolute` call is the identity — so without this oracle a driver that
/// ignored `material_reference` entirely would pass the whole file.
#[test]
fn only_the_rise_above_the_reference_drives_the_coupling() {
    let s = scene(EPS_TOTAL);
    const REFERENCE: f64 = 300.0;

    let at_zero = step_thermoplastic(
        &s.problem,
        &s.state,
        &s.mesh,
        &base_field_at(0.0),
        &coupling_at(C_V, 1.0, 1.0, 80, 0.0).expect("a valid coupling"),
        Fix128::ONE,
    )
    .expect("the increment at a zero reference solves");

    let shifted = step_thermoplastic(
        &s.problem,
        &s.state,
        &s.mesh,
        &base_field_at(REFERENCE),
        &coupling_at(C_V, 1.0, 1.0, 80, REFERENCE).expect("a valid coupling"),
        Fix128::ONE,
    )
    .expect("the increment at a shifted reference solves");

    assert_eq!(
        at_zero.temperature_increment.as_slice(),
        shifted.temperature_increment.as_slice(),
        "shifting both the base field and the reference by {REFERENCE} K changed the rise, so \
         the absolute temperature is reaching the constitutive law instead of only the \
         difference"
    );
    assert_eq!(
        at_zero.report.sweeps, shifted.report.sweeps,
        "the shifted scene took a different number of sweeps, so the two iterations are not the \
         same iteration"
    );
    for (e, (a, b)) in at_zero
        .increment
        .field
        .element_stress
        .iter()
        .zip(shifted.increment.field.element_stress.iter())
        .enumerate()
    {
        assert_eq!(
            a.xx, b.xx,
            "element {e} carries a different stress after shifting the reference"
        );
    }

    // The tooth: a scene that is already at its reference everywhere and stays
    // there would make the equality above hold for free. Here the body heats
    // up, so the two runs genuinely sample a shifted field.
    assert!(
        at_zero.temperature_increment.max_value() > Fix128::ZERO,
        "the body did not heat up, so both runs sat at their reference and the equality is free"
    );
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// Every relaxation outside `(0, 1]` is refused, and `1` is accepted.
#[test]
fn an_unusable_relaxation_is_refused() {
    for omega in [0.0, -0.5, 1.0 + 1.0 / 1024.0, 2.0] {
        assert!(
            matches!(
                coupling(C_V, 1.0, omega, 40),
                Err(FemError::InvalidConfig(_))
            ),
            "relaxation {omega} is outside (0, 1] and was accepted"
        );
    }
    assert!(
        coupling(C_V, 1.0, 1.0, 40).is_ok(),
        "relaxation of exactly one is the unrelaxed sweep and must be accepted"
    );
}

/// A floor fraction of zero leaves the stopping rule relative-only.
#[test]
fn an_unusable_residual_floor_fraction_is_refused() {
    let build = |fraction: Fix128| {
        ThermoplasticCoupling::try_new(
            PlasticHeating::try_new(Fix128::ONE, fx(C_V)).expect("a valid heating law"),
            fx(ALPHA_PER_K),
            Fix128::ZERO,
            Some(softening()),
            Fix128::ONE,
            fraction,
            SubIterationConfig::new(
                40,
                Fix128::from_raw(0, 1 << 24),
                Fix128::from_ratio(1, 2),
                Fix128::from_ratio(5, 4),
                4,
            )
            .expect("a valid sub-iteration config"),
        )
    };
    for fraction in [Fix128::ZERO, Fix128::ONE, Fix128::from_int(2), -Fix128::ONE] {
        assert!(
            matches!(build(fraction), Err(FemError::InvalidConfig(_))),
            "a floor fraction of {fraction:?} is outside (0, 1) and was accepted"
        );
    }
    assert!(
        build(Fix128::from_raw(0, FLOOR_FRACTION_RAW)).is_ok(),
        "the measured floor fraction must be accepted"
    );
}

/// A negative expansion coefficient is refused at construction.
#[test]
fn a_negative_expansion_coefficient_is_refused() {
    let built = ThermoplasticCoupling::try_new(
        PlasticHeating::try_new(Fix128::ONE, fx(C_V)).expect("a valid heating law"),
        -fx(ALPHA_PER_K),
        Fix128::ZERO,
        Some(softening()),
        Fix128::ONE,
        Fix128::from_raw(0, FLOOR_FRACTION_RAW),
        SubIterationConfig::new(
            40,
            Fix128::from_raw(0, 1 << 24),
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(5, 4),
            4,
        )
        .expect("a valid sub-iteration config"),
    );
    assert!(
        matches!(built, Err(FemError::InvalidConfig(_))),
        "a negative expansion coefficient was accepted"
    );
}

/// A grid with a single node on an axis is refused rather than deposited into.
#[test]
fn a_degenerate_grid_axis_is_refused() {
    let s = scene(EPS_TOTAL);
    let flat = CoupledField::try_new_filled(
        7,
        5,
        1,
        (
            Fix128::from_int(-1),
            Fix128::from_int(-1),
            Fix128::from_int(-1),
        ),
        (
            Fix128::from_int(5),
            Fix128::from_int(3),
            Fix128::from_int(3),
        ),
        Fix128::ZERO,
    )
    .expect("a grid with one node on z");
    let law = coupling(C_V, 1.0, 1.0, 40).expect("a valid coupling");
    let error = step_thermoplastic(&s.problem, &s.state, &s.mesh, &flat, &law, Fix128::ONE)
        .expect_err("a degenerate axis has no thickness to divide the heat by");
    assert!(
        matches!(
            error,
            FemError::DepositGridHasDegenerateAxis { axis: Axis::Z }
        ),
        "expected a refusal naming the z axis, got {error:?}"
    );
}

/// A field that does not cover the mesh is refused by the mechanical leg.
#[test]
fn a_field_that_does_not_cover_the_mesh_is_refused() {
    let s = scene(EPS_TOTAL);
    // Stops at x = 3, while the bar reaches x = 4.
    let short = CoupledField::try_new_filled(
        5,
        5,
        5,
        (Fix128::ZERO, Fix128::from_int(-1), Fix128::from_int(-1)),
        (
            Fix128::from_int(3),
            Fix128::from_int(3),
            Fix128::from_int(3),
        ),
        Fix128::ZERO,
    )
    .expect("a valid grid");
    let law = coupling(C_V, 1.0, 1.0, 40).expect("a valid coupling");
    let error = step_thermoplastic(&s.problem, &s.state, &s.mesh, &short, &law, Fix128::ONE)
        .expect_err("a field that leaves part of the bar unsampled is refused");
    assert!(
        matches!(error, FemError::TemperatureFieldDoesNotCoverMesh { .. }),
        "expected a coverage refusal, got {error:?}"
    );
}

/// A sweep budget of one sweep cannot reach a fixed point it has not found.
#[test]
fn a_budget_of_one_sweep_is_reported_as_not_converged() {
    let s = scene(EPS_TOTAL);
    let law = coupling(C_V, 1.0, 1.0, 1).expect("a valid coupling");
    let error = step_thermoplastic(&s.problem, &s.state, &s.mesh, &s.base, &law, Fix128::ONE)
        .expect_err("one sweep is not enough for this scene");
    assert!(
        matches!(
            error,
            FemError::CoupledSubIterationFailed(CoupledIterationError::NotConverged { .. })
        ),
        "expected NotConverged on an exhausted budget, got {error:?}"
    );
}
