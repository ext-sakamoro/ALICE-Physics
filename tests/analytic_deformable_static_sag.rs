//! Static sag of an elastic column under its own weight: the Neo-Hookean
//! [`DeformableBody`] (`Material::NeoHookean`) against the linear elastic finite
//! element solve on the same tetrahedra and against the closed form.
//!
//! A column of height `H`, density `ρ` and Young's modulus `E` standing on a fixed base
//! shortens under gravity: with uniaxial stress `σ(y) = ρ g (H − y)` the strain is `σ / E`
//! and the top moves down by `δ = ρ g H² / (2 E)` (Timoshenko & Goodier, *Theory of
//! Elasticity*, §2 bar under its own weight). A fixed base also holds the lateral (Poisson)
//! expansion in a boundary layer of depth at most the width `w`, which can only stiffen the
//! column there: the sag lies between the closed form and the closed form minus the share
//! of that layer, `(f − f² / 2) / (1 / 2)` with `f = w / H` (the integral of `ρ g (H − y)`
//! over the bottom `f H`), and the deviation shrinks as the column gets slenderer. The
//! finite element solve on the same mesh is the L0 reference the particle body must
//! reproduce.
//!
//! Material: the measured shear modulus of the logo's elastomer `G = 4.27e5 Pa` with
//! `ν = 0.49` gives `E = 2 G (1 + ν) = 1.2725e6 Pa`; density `ρ = 1100 kg/m³`,
//! `g = 9.81 m/s²`. The column is `0.05 × 0.4 × 0.05 m` (`H / w = 8`; `0.025` wide for
//! the slenderness check), meshed by [`sdf_fem_mesh::generate`] at `0.0125 m` cells for the
//! finite element comparisons (the `Fix128` conjugate gradient stagnates on the 4096-vertex
//! mesh of `0.00625 m` cells, so that size is left to the particle body's own convergence
//! sweep: three cell sizes, the error against the closed form must shrink as the cell
//! halves).
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the closed form ρ g H² / (2 E) and the lumped vertex masses are computed in f64 from the mesh, never by the body under test"
)]

use alice_physics_core::deformable::{
    DeformableBody, DeformableConfig, Material, MaterialError, NeoHookeanLaw,
};
use alice_physics_core::linear_elastic_fem::{
    self, Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::sdf_fem_mesh::SdfTetMesh;

#[path = "support/deformable_column.rs"]
mod column;
use column::{base_vertices, column_mesh, height, lumped_masses, max_y, top_vertices};

const SHEAR_MODULUS_PA: f64 = 4.27e5;
const POISSON: f64 = 0.49;
const YOUNG_PA: f64 = 2.0 * SHEAR_MODULUS_PA * (1.0 + POISSON);
const DENSITY: f64 = 1100.0;
const G: f64 = 9.81;
/// half height of the finite element column, metres (height 0.4)
const HALF_H: f64 = 0.2;
/// half width of the finite element column, metres (width 0.05, `H / w = 8`)
const HALF_W: f64 = 0.025;
/// the particle body's column: 0.05 m tall, 0.0125 m wide (one cell wide, four tall at the
/// 0.0125 m cell), so a converged solve stays within a test's budget
const BODY_HALF_H: f64 = 0.025;
const BODY_HALF_W: f64 = 0.00625;
/// substeps per 1/60 s frame at the 0.0125 m cell. The Gauss–Seidel passes converge only once
/// `α̃ = 1/h²` is comparable to `Σ w|∇C|²` (here `≈ 2e7 s⁻²`, so `h ≲ 1/4500 s`): below that the
/// multipliers cannot build up within a substep and the body is far softer than the material
/// (measured: one cube sags 4.6e-5 m at 4 substeps × 8 iterations, 3e-6 m converged).
/// `Σ w|∇C|²` scales like `μ / (ρ L²)`, so the substeps scale with `1 / L` in the sweep.
const SUBSTEPS: usize = 64;
const ITERATIONS: usize = 8;
/// the relative difference to the finite element sag accepted at `4 × SUBSTEPS` (256):
/// the solve converges first order in `h` (measured 0.184 at 64 substeps, 0.092 at 128,
/// 0.028 at 256), so two doublings from the 64-substep error give `0.184 / 4 = 0.046`,
/// rounded up to 0.05
const SETTLE_TOL: f64 = 0.05;
/// frames of 1/60 s the particle body settles for (velocity retention 0.9 per frame; the
/// sag is within the tolerances below after 120 frames, measured)
const SETTLE_FRAMES: usize = 120;

fn closed_form_sag(h: f64) -> f64 {
    DENSITY * G * h * h / (2.0 * YOUNG_PA)
}

/// the share of the closed-form sag that the bottom layer of depth `w` contributes: the
/// most a fixed base can take away by holding that layer's lateral expansion
fn base_layer_share(w: f64, h: f64) -> f64 {
    let f = w / h;
    (f - f * f / 2.0) / 0.5
}

/// the linear elastic finite element sag (metres, positive downward) on `mesh` with the base fixed
fn fem_sag(mesh: &SdfTetMesh) -> f64 {
    // the finite element solve works in mm / N / MPa
    let mm = SdfTetMesh {
        vertices: mesh
            .vertices
            .iter()
            .map(|v| [v[0] * 1000.0, v[1] * 1000.0, v[2] * 1000.0])
            .collect(),
        tets: mesh.tets.clone(),
    };
    let material =
        ElasticMaterial::new(Fix128::from_f64(YOUNG_PA * 1e-6), Fix128::from_f64(POISSON))
            .expect("E > 0, ν in range");
    let mut bc = BoundaryConditions::new();
    for i in base_vertices(mesh) {
        bc.fix(i as u32);
    }
    for (i, m) in lumped_masses(mesh, DENSITY).iter().enumerate() {
        bc.add_load(i as u32, Axis::Y, Fix128::from_f64(-m * G));
    }
    let sol = linear_elastic_fem::solve(&mm, &material, &bc, &SolverConfig::default())
        .expect("the column solves");
    let top = top_vertices(mesh);
    let sum: f64 = top.iter().map(|&i| -sol.displacements[i][1].to_f64()).sum();
    sum / top.len() as f64 * 1e-3
}

/// a body of `material` on `mesh` with the lumped masses and the base pinned
fn body_with(mesh: &SdfTetMesh, material: Material, substeps: usize) -> DeformableBody {
    let positions: Vec<Vec3Fix> = mesh
        .vertices
        .iter()
        .map(|v| {
            Vec3Fix::new(
                Fix128::from_f32(v[0]),
                Fix128::from_f32(v[1]),
                Fix128::from_f32(v[2]),
            )
        })
        .collect();
    let tets: Vec<[usize; 4]> = mesh
        .tets
        .iter()
        .map(|t| {
            [
                t.vertices[0] as usize,
                t.vertices[1] as usize,
                t.vertices[2] as usize,
                t.vertices[3] as usize,
            ]
        })
        .collect();
    let config = DeformableConfig {
        material,
        iterations: ITERATIONS,
        substeps,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(-G), Fix128::ZERO),
        damping: Fix128::from_ratio(90, 100),
        ..DeformableConfig::default()
    };
    let surface: Vec<[usize; 3]> = mesh
        .boundary_faces()
        .expect("a closed tetrahedral mesh has boundary faces")
        .iter()
        .map(|f| [f[0] as usize, f[1] as usize, f[2] as usize])
        .collect();
    let mut body = DeformableBody::new(positions, &tets, surface, Fix128::ONE);
    body.config = config;
    for (i, m) in lumped_masses(mesh, DENSITY).iter().enumerate() {
        body.inv_masses[i] = Fix128::ONE / Fix128::from_f64(*m);
    }
    for i in base_vertices(mesh) {
        body.inv_masses[i] = Fix128::ZERO;
    }
    body
}

fn neo_hookean() -> Material {
    Material::NeoHookean(
        NeoHookeanLaw::new(Fix128::from_f64(YOUNG_PA), Fix128::from_f64(POISSON))
            .expect("E > 0, ν in range"),
    )
}

/// the particle body's sag (metres, positive downward) after it has settled, base particles pinned
fn body_sag(mesh: &SdfTetMesh, substeps: usize) -> f64 {
    let mut body = body_with(mesh, neo_hookean(), substeps);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..SETTLE_FRAMES {
        body.step(dt);
    }
    let top = top_vertices(mesh);
    let rest = f64::from(max_y(mesh));
    let sum: f64 = top
        .iter()
        .map(|&i| rest - body.positions[i].y.to_f64())
        .sum();
    sum / top.len() as f64
}

/// The finite element reference itself sits between the closed form and the closed form
/// minus the base layer's share (the fixed base can only stiffen, and only that layer), and
/// the deviation halves when the column is twice as slender (two slendernesses: `H / w` 8
/// and 16; the deviation at 16 must be below the one at 8).
#[test]
fn the_finite_element_sag_of_a_column_lies_below_rho_g_h_squared_over_two_e_by_at_most_the_base_layer(
) {
    let mut deviations = Vec::new();
    for half_w in [HALF_W, HALF_W / 2.0] {
        let mesh = column_mesh(0.0125, half_w, HALF_H);
        let fem = fem_sag(&mesh);
        let h = height(&mesh);
        let closed = closed_form_sag(h);
        assert!(closed > 0.0);
        let dev = (closed - fem) / closed;
        let bound = base_layer_share(2.0 * half_w, h);
        eprintln!("H/w {:.0}: fem {fem:.4e} m, closed form {closed:.4e} m, deviation {dev:.4}, bound {bound:.4}", h / (2.0 * half_w));
        assert!(
            dev > 0.0 && dev < bound,
            "fem {fem:.3e} m vs closed form {closed:.3e} m: deviation {dev:.3}, bound {bound:.3}"
        );
        deviations.push(dev);
    }
    assert!(
        deviations[1] < deviations[0],
        "deviations {deviations:?} must shrink with slenderness"
    );
}

/// The Neo-Hookean particle body settles to the finite element sag on the same tetrahedra
/// (same nodes, same lumped masses, same fixed base) within `SETTLE_TOL` at `4 × SUBSTEPS`;
/// the small-strain difference between the Neo-Hookean and the linear law at a strain of
/// `ρ g H / E ≈ 4e-4` is far below that, the rest is the solve's first-order convergence.
#[test]
fn the_neo_hookean_body_settles_to_the_finite_element_sag() {
    let mesh = column_mesh(0.0125, BODY_HALF_W, BODY_HALF_H);
    let fem = fem_sag(&mesh);
    let body = body_sag(&mesh, 4 * SUBSTEPS);
    let rel = (body - fem).abs() / fem;
    eprintln!("body {body:.4e} m vs fem {fem:.4e} m ({rel:.4} relative)");
    assert!(
        rel < SETTLE_TOL,
        "body {body:.3e} m vs fem {fem:.3e} m ({rel:.3} relative)"
    );
}

/// Doubling the substeps three times on the body's column (20 tetrahedra), the body's error
/// against the finite element sag on the same tetrahedra must shrink every time (a convergence
/// sweep of four settings), and the last must be within `SETTLE_TOL`: the XPBD solve converges
/// to the implicit solution as `h → 0`, and the comparison on the same mesh isolates the
/// solver's convergence from the discretisation (the closed form differs from the P1 mesh
/// by the base layer and the mesh itself, see the finite element test above).
#[test]
fn the_neo_hookean_sag_converges_to_the_finite_element_sag_as_the_substeps_double() {
    let mesh = column_mesh(0.0125, BODY_HALF_W, BODY_HALF_H);
    let fem = fem_sag(&mesh);
    let mut errors = Vec::new();
    for substeps in [SUBSTEPS / 2, SUBSTEPS, 2 * SUBSTEPS, 4 * SUBSTEPS] {
        let body = body_sag(&mesh, substeps);
        errors.push((body - fem).abs() / fem);
    }
    assert_eq!(errors.len(), 4);
    eprintln!(
        "errors against the finite element sag, substeps doubling three times from {}: {errors:?}",
        SUBSTEPS / 2
    );
    for k in 1..errors.len() {
        assert!(
            errors[k] < errors[k - 1],
            "errors {errors:?} must shrink at every doubling"
        );
    }
    assert!(
        errors[3] < SETTLE_TOL,
        "finest substep error {:.4}",
        errors[3]
    );
}

/// An inverted tetrahedron recovers: the apex of a single rest tetrahedron is pushed through
/// its base, and within 120 frames (no gravity) the tetrahedron is back near its rest volume
/// with the right sign (the stable Neo-Hookean energy stays finite through `J ≤ 0`).
#[test]
fn an_inverted_tetrahedron_recovers_its_rest_shape() {
    let tenth = Fix128::from_ratio(1, 10);
    let rest = [
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(tenth, Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, tenth, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, tenth),
    ];
    let faces = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
    let mut body = DeformableBody::new(
        rest.to_vec(),
        &[[0, 1, 2, 3]],
        faces,
        Fix128::from_ratio(1, 100),
    );
    body.config = DeformableConfig {
        material: neo_hookean(),
        iterations: 16,
        substeps: 4,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        damping: Fix128::from_ratio(90, 100),
        ..DeformableConfig::default()
    };
    let volume = |p: &[Vec3Fix]| (p[1] - p[0]).dot((p[2] - p[0]).cross(p[3] - p[0])).to_f64() / 6.0;
    let rest_volume = volume(&rest);
    // push the apex (z = 0.1) through the base plane z = 0
    body.positions[3] = Vec3Fix::new(
        Fix128::from_ratio(3, 100),
        Fix128::from_ratio(3, 100),
        Fix128::from_ratio(-1, 10),
    );
    assert!(volume(&body.positions) < 0.0, "the start is inverted");
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..120 {
        body.step(dt);
    }
    let v = volume(&body.positions);
    assert!(v > 0.0, "volume after recovery {v:.3e}");
    let rel = (v - rest_volume).abs() / rest_volume;
    assert!(
        rel < 0.05,
        "volume {v:.3e} vs rest {rest_volume:.3e} ({rel:.3} relative)"
    );
}

/// The same body stepped twice from the same start reaches the same bits (determinism).
#[test]
fn a_neo_hookean_run_replays_to_the_same_bits() {
    let mesh = column_mesh(0.0125, BODY_HALF_W, BODY_HALF_H);
    let mut a = body_with(&mesh, neo_hookean(), 8);
    let mut b = body_with(&mesh, neo_hookean(), 8);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        a.step(dt);
        b.step(dt);
    }
    assert_eq!(a.positions, b.positions);
    assert_eq!(a.velocities, b.velocities);
}

/// The law refuses what the constraint cannot carry, on each side of every limit: `E = 0`
/// and the smallest positive `E` (refused / accepted), `E` at and just above `2⁴⁰ Pa`,
/// `ν = −1` and just above, `ν` at `0.5 − 2⁻¹⁶` and just above.
#[test]
fn the_neo_hookean_law_refuses_values_outside_its_range_on_each_side() {
    let e = Fix128::from_f64(YOUNG_PA);
    let nu = Fix128::from_f64(POISSON);
    let eps = Fix128::from_ratio(1, 1 << 30);
    assert_eq!(
        NeoHookeanLaw::new(Fix128::ZERO, nu),
        Err(MaterialError::YoungModulusNotPositive)
    );
    assert!(NeoHookeanLaw::new(eps, nu).is_ok());
    assert!(NeoHookeanLaw::new(NeoHookeanLaw::MAX_YOUNG_MODULUS, nu).is_ok());
    assert_eq!(
        NeoHookeanLaw::new(NeoHookeanLaw::MAX_YOUNG_MODULUS + eps, nu),
        Err(MaterialError::YoungModulusTooLarge)
    );
    assert_eq!(
        NeoHookeanLaw::new(e, Fix128::from_int(-1)),
        Err(MaterialError::PoissonRatioOutOfRange)
    );
    assert!(NeoHookeanLaw::new(e, Fix128::from_int(-1) + eps).is_ok());
    let top = Fix128::from_ratio(1, 2) - NeoHookeanLaw::POISSON_MARGIN;
    assert!(NeoHookeanLaw::new(e, top).is_ok());
    assert_eq!(
        NeoHookeanLaw::new(e, top + eps),
        Err(MaterialError::PoissonRatioOutOfRange)
    );
    let (mu, lambda) = NeoHookeanLaw::new(e, nu).expect("in range").lame();
    let want_mu = YOUNG_PA / (2.0 * (1.0 + POISSON));
    let want_lambda = YOUNG_PA * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON));
    assert!((mu.to_f64() - want_mu).abs() / want_mu < 1e-9);
    assert!((lambda.to_f64() - want_lambda).abs() / want_lambda < 1e-9);
}
