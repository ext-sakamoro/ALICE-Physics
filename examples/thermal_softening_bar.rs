//! A bar that is hotter at one end yields there first
//!
//! The forward leg of the strong thermo-mechanical coupling: a temperature
//! field enters the plastic solve itself, not as a nodal load bolted on
//! afterwards. Two distinct effects arrive together and this example separates
//! them in the output.
//!
//! **Thermal expansion** supplies the eigenstrain `ε_th = α ΔT I`, removed from
//! the elastic trial strain before the return mapping. It is purely
//! volumetric, and J2 yielding reads only the deviator, so expansion never
//! moves the yield criterion directly — it moves the strain that is left over
//! for the elastic-plastic response. Here `u_x` is prescribed at both ends, so
//! the expansion is fought by the constraint and the axial strain available is
//! `ε − α ΔT`.
//!
//! **Thermal softening** shrinks the yield surface itself,
//! `σ_y(T) = σ_y₀ (1 − w_y ΔT)` and `H(T) = H₀ (1 − w_h ΔT)`, each clamped at
//! zero. This is the effect that makes the coupling strong rather than formal:
//! without it a hot element yields at the cold criterion.
//!
//! ⚠️ **The hotter element does not end up carrying less stress.** Softening
//! lowers its yield radius, so it yields earlier and accumulates more
//! equivalent plastic strain, and with hardening its current radius
//! `σ_y(T) + H(T) ε̄_p` can finish *above* a colder element's. The monotone
//! quantity is the radius the law hands out at zero plastic strain, printed in
//! its own column; the stress the body settles at is an equilibrium, not an
//! ordering. The table below shows both so the difference is visible.
//!
//! ```bash
//! cargo run --example thermal_softening_bar --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::linear_elastic_fem::{
    Axis, BoundaryConditions, ElasticMaterial, ElastoplasticConfig, ElastoplasticIncrementRequest,
    ElastoplasticProblem, SolverConfig, ThermalExpansion, ThermalSoftening,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// Young's modulus, MPa. Dyadic, so the yield strain is exact.
const E_MPA: f64 = 1024.0;
/// Poisson's ratio.
const NU: f64 = 0.25;
/// Yield stress at the reference temperature, MPa.
const SIGMA_Y: f64 = 2.0;
/// Plastic modulus `H = dσ_y/dε̄_p` at the reference temperature, MPa.
const H_PLASTIC: f64 = 1024.0;
/// Linear expansion coefficient, K⁻¹. Dyadic, so `α ΔT` is exact.
const ALPHA_PER_K: f64 = 1.0 / 4096.0;
/// Fraction of the yield stress lost per kelvin.
const YIELD_LOST_PER_K: f64 = 1.0 / 8.0;
/// Fraction of the hardening modulus lost per kelvin.
const HARDENING_LOST_PER_K: f64 = 1.0 / 16.0;
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

/// Both ends prescribed in `x`, plus the minimum restraint that removes the
/// rigid rotations without over-constraining the cross section.
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

/// A rise of `ΔT = x` kelvin over a grid that covers the bar with a margin, so
/// the coverage check passes and no centroid falls outside.
///
/// The field is built as absolute temperatures and handed to `from_absolute`
/// with a zero reference, which is the call that records which temperature the
/// material parameters were measured at.
fn graded_rise() -> TemperatureRise {
    let (nx, ny, nz) = (7usize, 5usize, 5usize);
    let lo = (
        Fix128::from_int(-1),
        Fix128::from_int(-1),
        Fix128::from_int(-1),
    );
    let hi = (
        Fix128::from_int(5),
        Fix128::from_int(3),
        Fix128::from_int(3),
    );
    let mut absolute =
        CoupledField::try_new(nx, ny, nz, lo, hi).expect("a grid with two nodes per axis");
    let cell = Fix128::from_int(6) / Fix128::from_int(i64::try_from(nx - 1).expect("small"));
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let x = lo.0 + cell * Fix128::from_int(i64::try_from(i).expect("small"));
                absolute.set(i, j, k, x);
            }
        }
    }
    TemperatureRise::from_absolute(&absolute, Fix128::ZERO)
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

    // The softening law. `try_new` refuses a negative fraction and one so large
    // that a single kelvin would wipe out the parameter, so an unusable law is
    // an error at construction rather than a silently clamped zero later.
    let law = ThermalSoftening::try_new(fx(YIELD_LOST_PER_K), fx(HARDENING_LOST_PER_K))
        .expect("fractions in range");
    println!(
        "softening law: yield {:.6}/K, hardening {:.6}/K",
        law.yield_per_k().to_f64(),
        law.hardening_per_k().to_f64()
    );

    let rise = graded_rise();
    let request = ElastoplasticIncrementRequest::new(Fix128::ONE).with_thermal(
        ThermalExpansion::from_rise(&rise, fx(ALPHA_PER_K)),
        Some(law),
    );

    // The request carries its inputs back out, so a driver that builds requests
    // in a loop can report what it actually asked for rather than what it meant
    // to ask for.
    println!(
        "request: factor {:.3}, temperature field {}, softening {}\n",
        request.factor().to_f64(),
        if request.thermal().is_some() {
            "attached"
        } else {
            "none"
        },
        if request.softening().is_some() {
            "attached"
        } else {
            "none"
        }
    );

    let mut state = problem.virgin_state();
    let hot = problem
        .step(&state, &request)
        .expect("the heated increment solves");
    let newton = hot.newton_iterations();
    let work: Vec<f64> = hot
        .plastic_work_increment
        .iter()
        .map(|w| w.to_f64())
        .collect();
    let mises: Vec<f64> = hot
        .field
        .element_stress
        .iter()
        .map(|t| t.von_mises().to_f64())
        .collect();
    hot.commit(&mut state);
    let eqps: Vec<f64> = state
        .equivalent_plastic_strain()
        .iter()
        .map(|e| e.to_f64())
        .collect();

    println!("element   ΔT     σ_y(T)   H(T)      ε̄_p        ΔW_p       von Mises");
    for (e, tet) in mesh.tets.iter().enumerate() {
        // The solver samples the field at each element's own centroid, so the
        // centroid is also what reproduces the parameters it used.
        let mut x = 0.0f64;
        for v in tet.vertices {
            x += f64::from(mesh.vertices[v as usize][0]) / 4.0;
        }
        let delta_t = x;
        let softened_yield = SIGMA_Y * (1.0 - YIELD_LOST_PER_K * delta_t).max(0.0);
        let softened_hardening = H_PLASTIC * (1.0 - HARDENING_LOST_PER_K * delta_t).max(0.0);
        println!(
            "{e:>5}   {delta_t:>5.3}  {softened_yield:>7.4}  {softened_hardening:>7.1}  \
             {:>9.3e}  {:>9.3e}  {:>9.6}",
            eqps[e], work[e], mises[e]
        );
    }

    let spread = mises.iter().copied().fold(f64::MIN, f64::max)
        - mises.iter().copied().fold(f64::MAX, f64::min);
    println!("\nNewton iterations: {newton}\nstress spread across the bar: {spread:.6} MPa");
    println!(
        "The spread is the observable that distinguishes a per-element sample from\n\
         one centroid reused for the whole mesh: reusing one would make it zero."
    );
}
