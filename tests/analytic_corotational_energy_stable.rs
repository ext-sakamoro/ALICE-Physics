//! Two contracts of the co-rotational transient solve, from a soft letter that inverted when
//! run undamped:
//!
//! 1. Undamped, free and unloaded, a body released from a large deformation keeps its total
//!    energy (kinetic + co-rotated strain energy, [`CorotationalTransientSolver::energy`]):
//!    the continuous system conserves it, and a scheme that is energy-stable for the
//!    nonlinear co-rotational force does not let it grow. The bound: each step solves its
//!    linear system to the conjugate gradient's relative residual `τ`, and the residual's work
//!    over the step is at most `τ` times the step's elastic work, at most `2 τ E₀`; over `n`
//!    steps the energy stays below `E₀ (1 + 2 n τ)`. Checked over 10⁴ steps.
//! 2. An element that inverts (`det F ≤ 0`) is reported by the step that inverts it: every
//!    step that returns `Ok` leaves every element with a positive volume, and the step that
//!    would invert one returns an error with the state unchanged.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: energies and element volumes are read in f64, never by the solver under test"
)]

use alice_physics_core::dynamic_fem::{CorotationalTransientSolver, DynamicsConfig, MassLumping};
use alice_physics_core::linear_elastic_fem::{BoundaryConditions, ElasticMaterial, SolverConfig};
use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

#[path = "support/deformable_column.rs"]
mod column;
use column::column_mesh;

const YOUNG_MPA: f64 = 1.2725;
const POISSON: f64 = 0.45;
/// tonnes per cubic millimetre (1100 kg/m³)
const DENSITY: f64 = 1.1e-9;
const CG_TOL: f64 = 1e-10;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(YOUNG_MPA), fx(POISSON)).expect("valid")
}

/// a 25 × 100 × 25 mm column of 12.5 mm cubes (192 tetrahedra), in mm
fn column_mm() -> SdfTetMesh {
    let m = column_mesh(0.0125, 0.0125, 0.05);
    SdfTetMesh {
        vertices: m
            .vertices
            .iter()
            .map(|v| [v[0] * 1000.0, v[1] * 1000.0, v[2] * 1000.0])
            .collect(),
        tets: m.tets.clone(),
    }
}

fn solver(mesh: &SdfTetMesh, dt: f64) -> CorotationalTransientSolver {
    let dynamics =
        DynamicsConfig::try_new(fx(DENSITY), fx(dt), MassLumping::Lumped).expect("dt > 0");
    let config = SolverConfig::try_new(20_000, fx(CG_TOL)).expect("budget and tolerance");
    CorotationalTransientSolver::new(
        mesh,
        &material(),
        &BoundaryConditions::new(),
        &dynamics,
        &config,
    )
    .expect("the column solves")
}

#[test]
fn an_undamped_large_deformation_keeps_its_energy() {
    let mesh = column_mm();
    // the free-free axial period 2L/c, c = sqrt(E/ρ): 32 steps per period
    let c = (YOUNG_MPA / DENSITY).sqrt(); // mm/s
    let dt = 2.0 * 100.0 / c / 32.0;
    let mut s = solver(&mesh, dt);
    // released from a 30 % stretch along the column and a 10 % shear across it
    let u: Vec<Fix128> = mesh
        .vertices
        .iter()
        .flat_map(|v| {
            let y = f64::from(v[1]);
            [fx(0.1 * y), fx(0.3 * y), Fix128::ZERO]
        })
        .collect();
    s.set_displacements(&u)
        .expect("the stretched state has a polar factor");
    let e0 = s.energy().expect("energy").to_f64();
    assert!(e0 > 0.0, "the premise: a stored energy to keep ({e0})");
    let steps = 10_000u32;
    let mut worst = f64::NEG_INFINITY;
    for n in 1..=steps {
        s.step().expect("the step succeeds");
        let e = s.energy().expect("energy").to_f64();
        let bound = e0 * (1.0 + 2.0 * f64::from(n) * CG_TOL);
        worst = worst.max(e / e0 - 1.0);
        assert!(
            e <= bound,
            "step {n}: energy {e:.6e} above the bound {bound:.6e} (start {e0:.6e}, worst growth {worst:.3e})"
        );
    }
}

/// the signed volume of every tetrahedron at the solver's displacement, over its rest volume
fn volume_ratios(mesh: &SdfTetMesh, u: &[Fix128]) -> Vec<f64> {
    let at = |i: u32, d: bool| {
        let v = mesh.vertices[i as usize];
        let k = 3 * i as usize;
        let off = |j: usize| if d { u[k + j].to_f64() } else { 0.0 };
        [
            f64::from(v[0]) + off(0),
            f64::from(v[1]) + off(1),
            f64::from(v[2]) + off(2),
        ]
    };
    let vol = |t: &Tetrahedron, d: bool| {
        let [a, b, c, e] = t.vertices.map(|i| at(i, d));
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let ae = [e[0] - a[0], e[1] - a[1], e[2] - a[2]];
        ab[0] * (ac[1] * ae[2] - ac[2] * ae[1]) - ab[1] * (ac[0] * ae[2] - ac[2] * ae[0])
            + ab[2] * (ac[0] * ae[1] - ac[1] * ae[0])
    };
    mesh.tets
        .iter()
        .map(|t| vol(t, true) / vol(t, false))
        .collect()
}

#[test]
fn an_inverting_element_is_reported_by_the_step_that_inverts_it() {
    // one tetrahedron, its apex driven through the opposite face
    let mesh = SdfTetMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        tets: vec![Tetrahedron {
            vertices: [0, 1, 2, 3],
        }],
    };
    let mut s = solver(&mesh, 1e-4);
    // the apex's lumped mass ρV/4, V = 1000/6 mm³; 100 m/s toward the base
    let m = DENSITY * 1000.0 / 6.0 / 4.0;
    s.apply_impulses(&[(3, [Fix128::ZERO, Fix128::ZERO, fx(-m * 1.0e5)])])
        .expect("the impulse applies");
    let mut refused = None;
    for n in 1..=40 {
        let before = s.displacements().to_vec();
        match s.step() {
            Ok(_) => {
                let r = volume_ratios(&mesh, s.displacements());
                assert!(
                    r.iter().all(|&x| x > 0.0),
                    "step {n} returned Ok with an inverted element (volume ratio {:?})",
                    r
                );
            }
            Err(e) => {
                assert_eq!(
                    s.displacements(),
                    &before[..],
                    "the refused step changed the state"
                );
                refused = Some((n, e));
                break;
            }
        }
    }
    let (n, e) = refused.expect("the apex crosses the base within 40 steps");
    println!("refused at step {n}: {e:?}");
}

/// The measure itself, on two states whose energy is known: a free body moving rigidly at
/// `v` (translation, no strain) has `½ M v²` and keeps it; a pure stretch at rest has the
/// linear strain energy `½ E* ε² V` of the uniform field along the column, `E*` the
/// laterally confined modulus (no lateral motion at rest: `ε_x = ε_z = 0`).
#[test]
fn the_energy_measure_reads_known_states() {
    let mesh = column_mm();
    let c = (YOUNG_MPA / DENSITY).sqrt();
    let dt = 2.0 * 100.0 / c / 32.0;
    let mut s = solver(&mesh, dt);
    let volume = 25.0 * 100.0 * 25.0;
    let mass = DENSITY * volume;
    // translation: an impulse on every vertex in proportion to its mass gives every vertex
    // the same velocity; total impulse M·v
    let v = 50.0; // mm/s
    let masses = column::lumped_masses(&mesh, DENSITY);
    let impulses: Vec<(u32, [Fix128; 3])> = masses
        .iter()
        .enumerate()
        .map(|(i, m)| (i as u32, [fx(m * v), Fix128::ZERO, Fix128::ZERO]))
        .collect();
    s.apply_impulses(&impulses).expect("impulses");
    let e = s.energy().expect("energy").to_f64();
    let want = 0.5 * mass * v * v;
    assert!(
        (e - want).abs() < 1e-9 * want,
        "translation: {e:.9e} vs ½Mv² {want:.9e}"
    );
    for _ in 0..20 {
        s.step().expect("step");
    }
    let e = s.energy().expect("energy").to_f64();
    assert!(
        (e - want).abs() < 1e-9 * want,
        "translation after 20 steps: {e:.9e} vs {want:.9e}"
    );
    // a uniform stretch along y at rest
    let mut s = solver(&mesh, dt);
    let eps = 1e-3;
    let u: Vec<Fix128> = mesh
        .vertices
        .iter()
        .flat_map(|p| [Fix128::ZERO, fx(eps * f64::from(p[1])), Fix128::ZERO])
        .collect();
    s.set_displacements(&u).expect("polar factor");
    let e = s.energy().expect("energy").to_f64();
    let confined = YOUNG_MPA * (1.0 - POISSON) / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON));
    let want = 0.5 * confined * eps * eps * volume;
    assert!(
        (e - want).abs() < 1e-6 * want,
        "stretch: {e:.9e} vs ½E*ε²V {want:.9e}"
    );
}
