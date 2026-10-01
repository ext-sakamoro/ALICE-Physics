//! Heat the body with its own plastic work, and let that heat weaken it
//!
//! The two one-directional legs already exist: a temperature field enters the
//! return mapping (`with_thermal`) and plastic work becomes heat on a grid
//! (`deposit_increment_heat`). Running them once each, in order, is a
//! *partitioned* scheme with zero sub-iterations — the mechanics is evaluated
//! at the temperature of the previous guess, and the answer depends on which
//! leg went first.
//!
//! `step_thermoplastic` sub-iterates the pair to a fixed point, so the
//! increment it returns satisfies **both** legs at once. That is what makes the
//! coupling strong rather than formal.
//!
//! # What to watch
//!
//! **The feedback is positive, so it can stop closing.** Softening lowers the
//! yield stress, a weaker body yields more, more plastic work means more heat,
//! and more heat softens it further. Whether the loop closes is a property of
//! the splitting, not of the step size: the first table sweeps the volumetric
//! heat capacity, and below a threshold the sweep stops contracting and the
//! driver says so rather than returning a plausible number.
//!
//! ⚠️ **What it does there is settle into a period-two cycle, not diverge.**
//! At `c_v = 2⁻¹⁴` the first sweep deposits a rise of 177.7 K, which is far
//! past the point where `σ_y(T)` clamps at zero; a body with no yield stress
//! and no hardening carries no stress, so it dissipates nothing, so the next
//! sweep deposits exactly zero — and the one after that is back to 177.7. The
//! residual neither grows nor shrinks, and the monitor reports `Stagnated`
//! with `best_residual == first_residual`, which is the correct reading of a
//! flat sequence. ⚠️ **It is not `Diverging`**, and reporting it as divergence
//! would send a caller to change the splitting when the budget is not the
//! problem either.
//!
//! **Relaxation changes how the iteration travels, not where it lands.** The
//! second table holds `c_v` fixed and sweeps `ω`: all three land on the same
//! fixed point to eleven significant digits while taking 10, 32 and about 60
//! sweeps.
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
//! It does not break the period-two cycle either. Relaxation is the remedy
//! for the added-mass form [`alice_physics::coupled_iteration`] documents,
//! where the dominant eigenvalue is negative; this splitting is the other kind.
//!
//! ```bash
//! cargo run --example thermoplastic_sub_iteration --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::coupled_field::CoupledField;
use alice_physics::coupled_iteration::SubIterationConfig;
use alice_physics::linear_elastic_fem::{
    step_thermoplastic, Axis, BoundaryConditions, ElasticMaterial, ElastoplasticConfig,
    ElastoplasticProblem, FemError, PlasticHeating, SolverConfig, ThermalSoftening,
    ThermoplasticCoupling,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// Young's modulus, MPa. Dyadic, so the yield strain is exact.
const E_MPA: f64 = 1024.0;
/// Poisson's ratio.
const NU: f64 = 0.25;
/// Yield stress at the reference temperature, MPa.
const SIGMA_Y: f64 = 2.0;
/// Plastic modulus at the reference temperature, MPa.
const H_PLASTIC: f64 = 1024.0;
/// Linear expansion coefficient, K⁻¹.
const ALPHA_PER_K: f64 = 1.0 / 4096.0;
/// Fraction of the yield stress lost per kelvin.
const YIELD_LOST_PER_K: f64 = 1.0 / 8.0;
/// Fraction of the hardening modulus lost per kelvin.
const HARDENING_LOST_PER_K: f64 = 1.0 / 16.0;
/// Taylor-Quinney fraction: the share of plastic work that becomes heat.
const BETA: f64 = 1.0;
/// Absolute temperature the material parameters were measured at.
const REFERENCE_K: f64 = 0.0;
/// Total axial strain the bar is pulled to, well past `σ_y/E = 2⁻⁹`.
const EPS_TOTAL: f64 = 0.005;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn node(i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * 3 + k * 6).expect("the 3x2x2 lattice fits u32")
}

/// The bar `[0,4] × [0,2] × [0,2]` as two cells of six Kuhn tetrahedra.
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

fn bar_bc() -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for k in 0..=1 {
        for j in 0..=1 {
            bc.prescribe(node(0, j, k), Axis::X, Fix128::ZERO);
            bc.prescribe(node(2, j, k), Axis::X, fx(EPS_TOTAL) * Fix128::from_int(4));
        }
    }
    bc.prescribe(node(0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node(0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node(0, 1, 0), Axis::Z, Fix128::ZERO);
    bc
}

/// A grid covering the bar with a one-cell margin, filled with the reference
/// temperature so the increment starts from `ΔT = 0`.
fn base_field() -> CoupledField {
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
        fx(REFERENCE_K),
    )
    .expect("a grid with at least two nodes per axis")
}

fn coupling(
    heat_capacity: f64,
    relaxation: f64,
    max_sweeps: u32,
) -> Result<ThermoplasticCoupling, alice_physics::linear_elastic_fem::FemError> {
    ThermoplasticCoupling::try_new(
        PlasticHeating::try_new(fx(BETA), fx(heat_capacity))?,
        fx(ALPHA_PER_K),
        fx(REFERENCE_K),
        Some(
            ThermalSoftening::try_new(fx(YIELD_LOST_PER_K), fx(HARDENING_LOST_PER_K))
                .expect("fractions in range"),
        ),
        fx(relaxation),
        // 2⁻³⁰ of what the first sweep deposits: about 45 times the noise the
        // residual was measured to settle into, which scales with the answer.
        Fix128::from_raw(0, 1 << 34),
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

fn main() {
    let mesh = bar_mesh();
    let material = ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and nu in (-1, 0.5)");
    let config = ElastoplasticConfig::try_new(
        SolverConfig::default(),
        60,
        Fix128::from_raw(0, 1 << 24),
        fx(SIGMA_Y),
        fx(H_PLASTIC),
    )
    .expect("a valid elastoplastic config");
    let problem = ElastoplasticProblem::try_new(&mesh, &material, &bar_bc(), &config)
        .expect("the bar prepares");
    let state = problem.virgin_state();
    let base = base_field();

    println!("unrelaxed sweep (ω = 1), sweeping the volumetric heat capacity\n");
    println!("  c_v (MPa/K)   sweeps   max δT (K)   ratio ρ      verdict");
    for c_v in [1.0, 1.0 / 64.0, 1.0 / 256.0, 1.0 / 1024.0, 1.0 / 16384.0] {
        let law = match coupling(c_v, 1.0, 40) {
            Ok(law) => law,
            Err(error) => {
                println!("  {c_v:>11.5}   -        -            -            {error:?}");
                continue;
            }
        };
        match step_thermoplastic(&problem, &state, &mesh, &base, &law, Fix128::ONE) {
            Ok(done) => {
                let ratio = done
                    .report
                    .observed_ratio
                    .map_or("-".to_string(), |r| format!("{:.6}", r.to_f64()));
                println!(
                    "  {c_v:>11.5}   {:>6}   {:>10.6}   {ratio:>10}   converged",
                    done.report.sweeps,
                    done.temperature_increment.max_value().to_f64()
                );
            }
            Err(FemError::CoupledSubIterationFailed(error)) => {
                println!("  {c_v:>11.5}   -        -            -            {error:?}");
            }
            Err(other) => println!("  {c_v:>11.5}   -        -            -            {other:?}"),
        }
    }

    println!("\nthe same scene under relaxation: where it lands, not how it travels\n");
    println!("  c_v (MPa/K)   ω        sweeps   max δT (K)          (budget 80)");
    for c_v in [1.0 / 256.0] {
        for omega in [1.0, 1.0 / 2.0, 1.0 / 4.0] {
            let law = coupling(c_v, omega, 80).expect("a valid coupling");
            match step_thermoplastic(&problem, &state, &mesh, &base, &law, Fix128::ONE) {
                Ok(done) => println!(
                    "  {c_v:>11.5}   {omega:>5.3}   {:>6}   {:>18.12}",
                    done.report.sweeps,
                    done.temperature_increment.max_value().to_f64()
                ),
                Err(error) => println!("  {c_v:>11.5}   {omega:>5.3}   -        {error:?}"),
            }
        }
    }

    println!("\none-way comparison: the coupling has to change the answer\n");
    let law = coupling(1.0, 1.0, 40).expect("a valid coupling");
    let coupled = step_thermoplastic(&problem, &state, &mesh, &base, &law, Fix128::ONE)
        .expect("the coupled increment solves");
    let isothermal = problem
        .step(
            &state,
            &alice_physics::linear_elastic_fem::ElastoplasticIncrementRequest::new(Fix128::ONE),
        )
        .expect("the isothermal increment solves");
    println!(
        "  coupled   von Mises[0] = {:.9} MPa, ΔW_p[0] = {:.9}",
        coupled.increment.field.element_stress[0]
            .von_mises()
            .to_f64(),
        coupled.increment.plastic_work_increment[0].to_f64()
    );
    println!(
        "  isothermal von Mises[0] = {:.9} MPa, ΔW_p[0] = {:.9}",
        isothermal.field.element_stress[0].von_mises().to_f64(),
        isothermal.plastic_work_increment[0].to_f64()
    );
    println!(
        "  relaxation in use: ω = {:.3}, residual floor fraction = {:.3e}",
        law.relaxation().to_f64(),
        law.residual_floor_fraction().to_f64()
    );
}
