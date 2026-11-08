//! A cantilever under its own weight against Timoshenko beam theory: the closed form of the
//! tip deflection of a prismatic beam of length `L`, square section `w × w`, fixed at one end
//! and loaded by its weight `q = ρ g w²` per unit length, is
//!
//! ```text
//! δ = q L⁴ / (8 E I) + q L² / (2 κ G A),   I = w⁴/12, A = w², G = E / (2(1+ν)),
//! κ = 10 (1+ν) / (12 + 11 ν)   (Cowper's shear coefficient of a rectangle)
//! ```
//!
//! the Euler–Bernoulli bending term plus Timoshenko's shear term. Beam theory itself leaves out
//! terms of relative size `(w/L)²` (the stress state near the clamp and the section's warping),
//! so a three-dimensional solve agrees with it to that order at most: the tolerance used here
//! is `2 (w/L)²`, twice the truncation, with nothing for the discretisation, which a solve of
//! the right kind must bring below that. The same beam is modelled as the `linear_elastic_fem`
//! column (Kuhn tetrahedra, `tests/support/deformable_column.rs`) with its base fixed and
//! gravity sideways, at two mesh densities through the depth and two Poisson's ratios.
//!
//! Known defect (the oracle is red and `#[ignore]`d): linear (P1) tetrahedra lock in bending,
//! their constant strain cannot represent the linear strain of bending, so the mesh is far too
//! stiff: measured tip deflection 0.494 of the closed form with 2 cells through the depth and
//! 0.783 with 4 at ν 0.3 (shear locking), 0.093 and 0.262 at ν 0.49 (volumetric locking on
//! top of it, the near-incompressible P1 element). The fix is the quadratic tier (P2
//! tetrahedra, `quadratic_elastic_fem`) with a volumetric treatment if ν 0.49 still locks,
//! after which this test turns green within the tolerance above.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the beam closed form and the lumped weights are computed in f64 from the mesh, never by the solver under test"
)]

use alice_physics_core::linear_elastic_fem::{
    self, Axis, BoundaryConditions, ElasticMaterial, FemError, SolverConfig,
};
use alice_physics_core::math::Fix128;

#[path = "support/deformable_column.rs"]
mod column;
use column::{base_vertices, column_mesh, lumped_masses, top_vertices};

const E_PA: f64 = 1.27e6;
const DENSITY: f64 = 1100.0;
const G: f64 = 9.81;
/// beam length and width (m): slenderness 8
const LENGTH: f64 = 0.2;
const WIDTH: f64 = 0.025;

/// Timoshenko tip deflection (m) of the cantilever under its own weight
fn timoshenko_tip(nu: f64) -> f64 {
    let area = WIDTH * WIDTH;
    let inertia = WIDTH.powi(4) / 12.0;
    let q = DENSITY * G * area;
    let shear_modulus = E_PA / (2.0 * (1.0 + nu));
    let kappa = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu);
    q * LENGTH.powi(4) / (8.0 * E_PA * inertia)
        + q * LENGTH * LENGTH / (2.0 * kappa * shear_modulus * area)
}

/// the finite element tip deflection (m, along x) of the column fixed at its base under a
/// sideways gravity, `cells_through_depth` cubes across the width
fn fem_tip(nu: f64, cells_through_depth: usize) -> Result<f64, FemError> {
    let cell = (WIDTH / cells_through_depth as f64) as f32;
    let mesh = column_mesh(cell, WIDTH / 2.0, LENGTH / 2.0);
    let mm = alice_physics_core::sdf_fem_mesh::SdfTetMesh {
        vertices: mesh
            .vertices
            .iter()
            .map(|v| [v[0] * 1000.0, v[1] * 1000.0, v[2] * 1000.0])
            .collect(),
        tets: mesh.tets.clone(),
    };
    let material = ElasticMaterial::new(Fix128::from_f64(E_PA * 1e-6), Fix128::from_f64(nu))
        .expect("E > 0, ν in range");
    let mut bc = BoundaryConditions::new();
    for i in base_vertices(&mesh) {
        bc.fix(i as u32);
    }
    // the weight sideways: N = t · mm/s², masses in tonnes
    for (i, m) in lumped_masses(&mesh, DENSITY).iter().enumerate() {
        bc.add_load(i as u32, Axis::X, Fix128::from_f64(-m * 1e-3 * G * 1000.0));
    }
    let config =
        SolverConfig::try_new(20_000, Fix128::from_f64(1e-9)).expect("budget and tolerance");
    let sol = linear_elastic_fem::solve(&mm, &material, &bc, &config)?;
    let top = top_vertices(&mesh);
    Ok(top
        .iter()
        .map(|&i| -sol.displacements[i][0].to_f64())
        .sum::<f64>()
        / top.len() as f64
        * 1e-3)
}

fn check(nu: f64) {
    let closed = timoshenko_tip(nu);
    let tol = 2.0 * (WIDTH / LENGTH).powi(2);
    // measure every density first, so a red run still prints the whole picture
    let ratios: Vec<(usize, f64)> = [2usize, 4]
        .iter()
        .map(|&cells| {
            (
                cells,
                fem_tip(nu, cells).expect("the cantilever solves") / closed,
            )
        })
        .collect();
    for &(cells, ratio) in &ratios {
        eprintln!(
            "ν {nu}: {cells} cells through the depth: tip / Timoshenko {ratio:.3} (closed form {closed:.4e} m, tolerance {tol:.3})"
        );
    }
    for &(cells, ratio) in &ratios {
        assert!(
            (ratio - 1.0).abs() <= tol,
            "ν {nu}, {cells} cells: tip / closed form {ratio:.3} (tolerance {tol:.3})"
        );
    }
}

#[test]
#[ignore = "known defect: P1 shear locking: the tip deflection of a cantilever under its own weight is 0.49 of the Timoshenko closed form with 2 cells through the depth (0.78 with 4) at ν 0.3; the quadratic tier is the fix"]
fn a_cantilever_under_its_own_weight_bends_by_the_timoshenko_tip_deflection_at_nu_0_3() {
    check(0.3);
}

#[test]
#[ignore = "known defect: P1 shear and volumetric locking at ν 0.49: the tip deflection is 0.09 of the Timoshenko closed form with 2 cells through the depth (0.26 with 4); the quadratic tier with a volumetric treatment is the fix"]
fn a_cantilever_under_its_own_weight_bends_by_the_timoshenko_tip_deflection_at_nu_0_49() {
    check(0.49);
}
