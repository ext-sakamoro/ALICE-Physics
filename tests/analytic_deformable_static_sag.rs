//! Static sag of an elastic column under its own weight: the Neo-Hookean
//! [`DeformableBody`] (`Material::NeoHookean { E, ν }`) against the linear elastic finite
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

use alice_physics_core::deformable::{DeformableBody, DeformableConfig, Material};
use alice_physics_core::linear_elastic_fem::{
    self, Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::sdf_collider::SdfField;
use alice_physics_core::sdf_fem_mesh::{self, SdfTetMesh};

const SHEAR_MODULUS_PA: f64 = 4.27e5;
const POISSON: f64 = 0.49;
const YOUNG_PA: f64 = 2.0 * SHEAR_MODULUS_PA * (1.0 + POISSON);
const DENSITY: f64 = 1100.0;
const G: f64 = 9.81;
/// half height of the column, metres (height 0.4)
const HALF_H: f64 = 0.2;
/// half width of the column, metres (width 0.05, `H / w = 8`)
const HALF_W: f64 = 0.025;

/// an axis-aligned box, exact signed distance outside and inside (metres)
struct BoxSdf([f32; 3]);

impl SdfField for BoxSdf {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        let q = [
            x.abs() - self.0[0],
            y.abs() - self.0[1],
            z.abs() - self.0[2],
        ];
        let outside =
            (q[0].max(0.0).powi(2) + q[1].max(0.0).powi(2) + q[2].max(0.0).powi(2)).sqrt();
        let inside = q[0].max(q[1]).max(q[2]).min(0.0);
        outside + inside
    }
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let q = [
            x.abs() - self.0[0],
            y.abs() - self.0[1],
            z.abs() - self.0[2],
        ];
        let axis = (0..3)
            .max_by(|&a, &b| q[a].partial_cmp(&q[b]).expect("finite"))
            .expect("three axes");
        let p = [x, y, z];
        let mut n = [0.0f32; 3];
        n[axis] = if p[axis] >= 0.0 { 1.0 } else { -1.0 };
        (n[0], n[1], n[2])
    }
}

fn column_mesh(cell: f32, half_w: f64) -> SdfTetMesh {
    let h = [half_w as f32, HALF_H as f32, half_w as f32];
    let pad = cell * 0.5;
    sdf_fem_mesh::generate(
        &BoxSdf(h),
        [-h[0] - pad, -h[1] - pad, -h[2] - pad],
        [h[0] + pad, h[1] + pad, h[2] + pad],
        cell,
    )
}

fn tet_volume(v: &[[f32; 3]], t: [u32; 4]) -> f64 {
    let p = |i: u32| {
        let a = v[i as usize];
        [f64::from(a[0]), f64::from(a[1]), f64::from(a[2])]
    };
    let (a, b, c, d) = (p(t[0]), p(t[1]), p(t[2]), p(t[3]));
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    let cross = [
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    ];
    ((ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]) / 6.0).abs()
}

/// lumped vertex masses, kg: a quarter of each tetrahedron's mass to each of its vertices
fn lumped_masses(mesh: &SdfTetMesh) -> Vec<f64> {
    let mut m = vec![0.0; mesh.vertices.len()];
    for t in &mesh.tets {
        let share = DENSITY * tet_volume(&mesh.vertices, t.vertices) / 4.0;
        for &i in &t.vertices {
            m[i as usize] += share;
        }
    }
    m
}

fn min_y(mesh: &SdfTetMesh) -> f32 {
    mesh.vertices
        .iter()
        .map(|v| v[1])
        .fold(f32::INFINITY, f32::min)
}

fn max_y(mesh: &SdfTetMesh) -> f32 {
    mesh.vertices
        .iter()
        .map(|v| v[1])
        .fold(f32::NEG_INFINITY, f32::max)
}

fn base_vertices(mesh: &SdfTetMesh) -> Vec<usize> {
    let y0 = min_y(mesh);
    (0..mesh.vertices.len())
        .filter(|&i| (mesh.vertices[i][1] - y0).abs() < 1e-6)
        .collect()
}

fn top_vertices(mesh: &SdfTetMesh) -> Vec<usize> {
    let y1 = max_y(mesh);
    (0..mesh.vertices.len())
        .filter(|&i| (mesh.vertices[i][1] - y1).abs() < 1e-6)
        .collect()
}

fn height(mesh: &SdfTetMesh) -> f64 {
    f64::from(max_y(mesh) - min_y(mesh))
}

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
    for (i, m) in lumped_masses(mesh).iter().enumerate() {
        bc.add_load(i as u32, Axis::Y, Fix128::from_f64(-m * G));
    }
    let sol = linear_elastic_fem::solve(&mm, &material, &bc, &SolverConfig::default())
        .expect("the column solves");
    let top = top_vertices(mesh);
    let sum: f64 = top.iter().map(|&i| -sol.displacements[i][1].to_f64()).sum();
    sum / top.len() as f64 * 1e-3
}

/// the particle body's sag (metres, positive downward) after it has settled, base particles pinned
fn body_sag(mesh: &SdfTetMesh, iterations: usize) -> f64 {
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
        material: Material::NeoHookean {
            young_modulus: Fix128::from_f64(YOUNG_PA),
            poisson_ratio: Fix128::from_f64(POISSON),
        },
        iterations,
        substeps: 4,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(-G), Fix128::ZERO),
        damping: Fix128::from_ratio(95, 100),
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
    for (i, m) in lumped_masses(mesh).iter().enumerate() {
        body.inv_masses[i] = Fix128::ONE / Fix128::from_f64(*m);
    }
    for i in base_vertices(mesh) {
        body.inv_masses[i] = Fix128::ZERO;
    }
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..600 {
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
        let mesh = column_mesh(0.0125, half_w);
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
/// (same nodes, same lumped masses, same fixed base): 2 % relative, the small-strain
/// difference between the Neo-Hookean and the linear law at a strain of `ρ g H / E ≈ 3e-3`
/// plus the settling residual.
#[test]
#[ignore = "src gap: Material::NeoHookean tetrahedral constraints are not implemented (DeformableBody::substep panics)"]
fn the_neo_hookean_body_settles_to_the_finite_element_sag() {
    let mesh = column_mesh(0.0125, HALF_W);
    let fem = fem_sag(&mesh);
    let body = body_sag(&mesh, 16);
    let rel = (body - fem).abs() / fem;
    assert!(
        rel < 0.02,
        "body {body:.3e} m vs fem {fem:.3e} m ({rel:.3} relative)"
    );
}

/// Halving the cell three times, the body's error against the closed form must not grow:
/// each refinement's error is at most the previous one (a convergence sweep of ≥ 3 settings).
#[test]
#[ignore = "src gap: Material::NeoHookean tetrahedral constraints are not implemented (DeformableBody::substep panics)"]
fn the_neo_hookean_sag_converges_to_the_closed_form_as_the_cell_halves() {
    let mut errors = Vec::new();
    for cell in [0.025f32, 0.0125, 0.00625] {
        let mesh = column_mesh(cell, HALF_W);
        let closed = closed_form_sag(height(&mesh));
        let body = body_sag(&mesh, 16);
        errors.push((body - closed).abs() / closed);
    }
    assert_eq!(errors.len(), 3);
    assert!(
        errors[1] <= errors[0] && errors[2] <= errors[1],
        "errors {errors:?}"
    );
    assert!(errors[2] < 0.05, "finest cell error {:.3}", errors[2]);
}

/// Today's behaviour, pinned so the coverage of the entry point does not depend on the
/// red oracles above: the default material is the constraint set, and a step with the
/// Neo-Hookean material stops at its stub instead of silently running the old constraints.
#[test]
fn the_default_material_is_constraints_and_neo_hookean_is_a_stub_today() {
    assert_eq!(DeformableConfig::default().material, Material::Constraints);
    let mesh = column_mesh(0.025, HALF_W);
    let fem = fem_sag(&mesh);
    assert!(fem > 0.0);
    let result = std::panic::catch_unwind(|| body_sag(&mesh, 1));
    let err = result.expect_err("the Neo-Hookean step is a stub today");
    let text = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default();
    assert!(
        text.contains("STUB: Material::NeoHookean"),
        "panic text {text:?}"
    );
}
