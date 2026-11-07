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
use alice_physics_core::sdf_collider::SdfField;
use alice_physics_core::sdf_fem_mesh::{self, SdfTetMesh};

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
/// frames of 1/60 s the particle body settles for (velocity retention 0.9 per frame; the
/// sag is within the tolerances below after 120 frames, measured)
const SETTLE_FRAMES: usize = 120;

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

/// the column's tetrahedra: a lattice of `cell`-sized cubes, each split into the six Kuhn
/// tetrahedra (the split is conforming across cubes without any parity rule), so the mesh
/// has the box's exact faces at every cell size and a full layer of vertices on each face
fn column_mesh(cell: f32, half_w: f64, half_h: f64) -> SdfTetMesh {
    let nx = ((2.0 * half_w / f64::from(cell)).round() as usize).max(1);
    let ny = ((2.0 * half_h / f64::from(cell)).round() as usize).max(1);
    let nz = nx;
    let at = |i: usize, j: usize, k: usize| -> u32 { ((i * (ny + 1) + j) * (nz + 1) + k) as u32 };
    let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nz {
                vertices.push([
                    (i as f64 * f64::from(cell) - half_w) as f32,
                    (j as f64 * f64::from(cell) - half_h) as f32,
                    (k as f64 * f64::from(cell) - half_w) as f32,
                ]);
            }
        }
    }
    let mut tets = Vec::with_capacity(6 * nx * ny * nz);
    // the six tetrahedra of a cube share its main diagonal v000–v111; each is (v000, a, b, v111)
    // along one of the six monotone paths of the cube's edges
    let paths: [[(usize, usize, usize); 2]; 6] = [
        [(1, 0, 0), (1, 1, 0)],
        [(1, 0, 0), (1, 0, 1)],
        [(0, 1, 0), (1, 1, 0)],
        [(0, 1, 0), (0, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, 1), (0, 1, 1)],
    ];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let v000 = at(i, j, k);
                let v111 = at(i + 1, j + 1, k + 1);
                for [(a0, a1, a2), (b0, b1, b2)] in paths {
                    let a = at(i + a0, j + a1, k + a2);
                    let b = at(i + b0, j + b1, k + b2);
                    tets.push(sdf_fem_mesh::Tetrahedron {
                        vertices: [v000, a, b, v111],
                    });
                }
            }
        }
    }
    SdfTetMesh { vertices, tets }
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

/// the vertices of the bottom face (the marching tetrahedra put them on the face to the
/// float rounding of the warp, far below `FACE_TOL`; the next layer is a cell away)
const FACE_TOL: f32 = 1e-4;

fn base_vertices(mesh: &SdfTetMesh) -> Vec<usize> {
    let y0 = min_y(mesh);
    (0..mesh.vertices.len())
        .filter(|&i| (mesh.vertices[i][1] - y0).abs() < FACE_TOL)
        .collect()
}

fn top_vertices(mesh: &SdfTetMesh) -> Vec<usize> {
    let y1 = max_y(mesh);
    (0..mesh.vertices.len())
        .filter(|&i| (mesh.vertices[i][1] - y1).abs() < FACE_TOL)
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
    for (i, m) in lumped_masses(mesh).iter().enumerate() {
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
/// (same nodes, same lumped masses, same fixed base): 2 % relative, the small-strain
/// difference between the Neo-Hookean and the linear law at a strain of `ρ g H / E ≈ 3e-3`
/// plus the settling residual.
#[test]
fn the_neo_hookean_body_settles_to_the_finite_element_sag() {
    let mesh = column_mesh(0.0125, BODY_HALF_W, BODY_HALF_H);
    let fem = fem_sag(&mesh);
    let body = body_sag(&mesh, SUBSTEPS);
    let rel = (body - fem).abs() / fem;
    eprintln!("body {body:.4e} m vs fem {fem:.4e} m ({rel:.4} relative)");
    assert!(
        rel < 0.02,
        "body {body:.3e} m vs fem {fem:.3e} m ({rel:.3} relative)"
    );
}

/// Halving the cell twice on the body's column (`H / w = 4`), with the substeps doubling each
/// time (see `SUBSTEPS`), the body's error against the closed form must not grow: each
/// refinement's error is at most the previous one (a convergence sweep of three settings;
/// the finest mesh has 1 280 tetrahedra at 256 substeps).
#[test]
fn the_neo_hookean_sag_converges_to_the_closed_form_as_the_cell_halves() {
    let mut errors = Vec::new();
    for (cell, substeps) in [
        (0.0125f32, SUBSTEPS),
        (0.00625, 2 * SUBSTEPS),
        (0.003125, 4 * SUBSTEPS),
    ] {
        let mesh = column_mesh(cell, BODY_HALF_W, BODY_HALF_H);
        let closed = closed_form_sag(height(&mesh));
        let body = body_sag(&mesh, substeps);
        errors.push((body - closed).abs() / closed);
    }
    assert_eq!(errors.len(), 3);
    eprintln!("errors against the closed form, cell halving twice: {errors:?}");
    assert!(
        errors[1] <= errors[0] && errors[2] <= errors[1],
        "errors {errors:?}"
    );
    assert!(errors[2] < 0.05, "finest cell error {:.3}", errors[2]);
}

/// The default material is the constraint set (the behaviour every earlier caller had), and
/// the Neo-Hookean material is a different law: bodies of each material from the same mesh
/// are at different positions after a frame.
#[test]
fn the_default_material_is_constraints_and_neo_hookean_is_a_different_law() {
    assert_eq!(DeformableConfig::default().material, Material::Constraints);
    let mesh = column_mesh(0.0125, BODY_HALF_W, BODY_HALF_H);
    let mut a = body_with(&mesh, Material::Constraints, 4);
    let mut b = body_with(&mesh, neo_hookean(), 4);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..30 {
        a.step(dt);
        b.step(dt);
    }
    assert_ne!(a.positions, b.positions);
}

/// Doubling the substeps (64 → 128) moves the settled sag by less than 1 % of the finite
/// element answer: the solve is converged, so the stiffness is the material's, not the
/// step count's.
#[test]
fn the_neo_hookean_sag_does_not_depend_on_the_substep_count() {
    let mesh = column_mesh(0.0125, BODY_HALF_W, BODY_HALF_H);
    let fem = fem_sag(&mesh);
    let at_8 = body_sag(&mesh, SUBSTEPS);
    let at_16 = body_sag(&mesh, 2 * SUBSTEPS);
    let rel = (at_8 - at_16).abs() / fem;
    eprintln!(
        "64 substeps {at_8:.4e} m, 128 substeps {at_16:.4e} m, fem {fem:.4e} m ({rel:.4} of fem)"
    );
    assert!(
        rel < 0.01,
        "64 substeps {at_8:.3e} m, 128 substeps {at_16:.3e} m, fem {fem:.3e} m ({rel:.4} of fem)"
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
