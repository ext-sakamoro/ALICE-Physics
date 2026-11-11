//! A measurement, not a check: the cost of a correct frame of a stiff elastic solid by two
//! integrators on the same tetrahedra, against the same reference.
//!
//! (a) [`DeformableBody`] with `Material::NeoHookean`: XPBD, explicit-like substepping, at the
//!     substeps its Gauss–Seidel convergence needs (`h ≲ 1/√(Σ w|∇C|²)`, 64 per frame at
//!     1.25 cm cells for this material).
//! (b) [`dynamic_fem::TransientSolver`]: implicit Newmark (average acceleration,
//!     unconditionally stable for linear elasticity) on P1 tetrahedra, 1 and 2 steps per
//!     frame, each a deterministic `Fix128` conjugate gradient solve to a fixed tolerance.
//!
//! Reference: the static linear elastic solve on the same mesh (the sag of a column under its
//! own weight with its base fixed), and the closed form `ρ g H² / (2E)`. The Newmark run has
//! no damping, so its sag oscillates about the static value: its mean over the second half of
//! the run is what is compared. Run with
//! `cargo test --release --test analytic_deformable_integrators -- --ignored --nocapture`.
#![cfg(feature = "std")]

use alice_physics_core::deformable::{DeformableBody, DeformableConfig, Material, NeoHookeanLaw};
use alice_physics_core::dynamic_fem::{DynamicsConfig, MassLumping, TransientSolver};
use alice_physics_core::linear_elastic_fem::{
    self, Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::sdf_fem_mesh::SdfTetMesh;
use std::time::Instant;

#[path = "support/deformable_column.rs"]
mod column;
use column::{base_vertices, column_mesh, height, lumped_masses, max_y, top_vertices};

const SHEAR_MODULUS_PA: f64 = 4.27e5;
const POISSON: f64 = 0.49;
const YOUNG_PA: f64 = 2.0 * SHEAR_MODULUS_PA * (1.0 + POISSON);
const DENSITY: f64 = 1100.0;
const G: f64 = 9.81;
const FRAMES: usize = 60;

/// the mesh in mm with the finite element boundary conditions (base fixed, gravity loads)
fn fem_setup(mesh: &SdfTetMesh) -> (SdfTetMesh, ElasticMaterial, BoundaryConditions) {
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
    (mm, material, bc)
}

/// static sag, metres, positive downward
fn static_sag(mesh: &SdfTetMesh) -> f64 {
    let (mm, material, bc) = fem_setup(mesh);
    let sol = linear_elastic_fem::solve(&mm, &material, &bc, &SolverConfig::default())
        .expect("the column solves");
    let top = top_vertices(mesh);
    top.iter()
        .map(|&i| -sol.displacements[i][1].to_f64())
        .sum::<f64>()
        / top.len() as f64
        * 1e-3
}

fn top_sag_of_body(mesh: &SdfTetMesh, body: &DeformableBody) -> f64 {
    let top = top_vertices(mesh);
    let rest = f64::from(max_y(mesh));
    top.iter()
        .map(|&i| rest - body.positions[i].y.to_f64())
        .sum::<f64>()
        / top.len() as f64
}

fn xpbd_body(mesh: &SdfTetMesh, substeps: usize, iterations: usize) -> DeformableBody {
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
    let surface: Vec<[usize; 3]> = mesh
        .boundary_faces()
        .expect("a closed mesh has boundary faces")
        .iter()
        .map(|f| [f[0] as usize, f[1] as usize, f[2] as usize])
        .collect();
    let mut body = DeformableBody::new(positions, &tets, surface, Fix128::ONE);
    body.config = DeformableConfig {
        material: Material::NeoHookean(
            NeoHookeanLaw::new(Fix128::from_f64(YOUNG_PA), Fix128::from_f64(POISSON))
                .expect("E > 0, ν in range"),
        ),
        iterations,
        substeps,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(-G), Fix128::ZERO),
        damping: Fix128::from_ratio(90, 100),
        ..DeformableConfig::default()
    };
    for (i, m) in lumped_masses(mesh, DENSITY).iter().enumerate() {
        body.inv_masses[i] = Fix128::ONE / Fix128::from_f64(*m);
    }
    for i in base_vertices(mesh) {
        body.inv_masses[i] = Fix128::ZERO;
    }
    body
}

/// (seconds per frame, sag after the run, metres)
fn run_xpbd(mesh: &SdfTetMesh, substeps: usize, iterations: usize) -> (f64, f64) {
    let mut body = xpbd_body(mesh, substeps, iterations);
    let dt = Fix128::from_ratio(1, 60);
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: the wall-clock start of the measured run, printed, never asserted"
    )]
    let t = Instant::now();
    for _ in 0..FRAMES {
        body.step(dt);
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: the wall-clock cost of the measured run, printed, never asserted"
    )]
    let secs = t.elapsed().as_secs_f64();
    (secs / FRAMES as f64, top_sag_of_body(mesh, &body))
}

/// (seconds per frame, conjugate gradient iterations per step, mean sag over the second half
/// of the run, metres)
fn run_newmark(mesh: &SdfTetMesh, steps_per_frame: usize, tolerance: f64) -> (f64, f64, f64) {
    let (mm, material, bc) = fem_setup(mesh);
    // tonne per mm³: 1100 kg/m³ = 1.1e-9 t/mm³
    let dynamics = DynamicsConfig::try_new(
        Fix128::from_f64(DENSITY * 1e-12),
        Fix128::from_ratio(1, 60 * steps_per_frame as i64),
        MassLumping::Lumped,
    )
    .expect("density and dt positive");
    let config =
        SolverConfig::try_new(20_000, Fix128::from_f64(tolerance)).expect("budget and tolerance");
    let mut solver = match TransientSolver::new(&mm, &material, &bc, &dynamics, &config) {
        Ok(s) => s,
        Err(e) => {
            println!("  newmark  {steps_per_frame} step/frame tol {tolerance:.0e}: the consistent acceleration solve failed: {e:?} (the attainable residual depends on the mesh, see the MODEL's valid range)");
            return (f64::NAN, f64::NAN, f64::NAN);
        }
    };
    let top = top_vertices(mesh);
    let mut iterations = 0u64;
    let mut sag_sum = 0.0;
    let mut sag_count = 0;
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: the wall-clock start of the measured run, printed, never asserted"
    )]
    let t = Instant::now();
    for frame in 0..FRAMES {
        for _ in 0..steps_per_frame {
            match solver.step() {
                Ok(it) => iterations += u64::from(it),
                Err(e) => {
                    println!("  newmark  {steps_per_frame} step/frame tol {tolerance:.0e}: the solve stopped at frame {frame}: {e:?}");
                    return (f64::NAN, f64::NAN, f64::NAN);
                }
            }
        }
        if frame >= FRAMES / 2 {
            let u = solver.displacements();
            sag_sum +=
                top.iter().map(|&i| -u[3 * i + 1].to_f64()).sum::<f64>() / top.len() as f64 * 1e-3;
            sag_count += 1;
        }
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: the wall-clock cost of the measured run, printed, never asserted"
    )]
    let secs = t.elapsed().as_secs_f64();
    (
        secs / FRAMES as f64,
        iterations as f64 / (FRAMES * steps_per_frame) as f64,
        sag_sum / f64::from(sag_count),
    )
}

fn closed_form_sag(h: f64) -> f64 {
    DENSITY * G * h * h / (2.0 * YOUNG_PA)
}

#[test]
#[ignore = "a measurement, not a check"]
fn cost_and_error_of_xpbd_and_newmark_on_the_same_columns() {
    // the oracle's column (20 tetrahedra) and a taller, wider one (768) at the same cell
    for (name, half_w, half_h) in [
        ("column 0.0125 × 0.05 m", 0.00625, 0.025),
        ("column 0.025 × 0.2 m", 0.0125, 0.1),
    ] {
        let mesh = column_mesh(0.0125, half_w, half_h);
        let fem = static_sag(&mesh);
        let closed = closed_form_sag(height(&mesh));
        println!(
            "== {name}: {} vertices, {} tets; static fem sag {fem:.4e} m, closed form {closed:.4e} m",
            mesh.vertices.len(),
            mesh.tets.len()
        );
        for (substeps, iterations) in [(4, 8), (64, 8), (128, 8)] {
            let (per_frame, sag) = run_xpbd(&mesh, substeps, iterations);
            println!(
                "  xpbd nh  substeps {substeps:3} iterations {iterations}: {per_frame:.4} s/frame, sag {sag:.4e} m ({:+.3} of fem)",
                (sag - fem) / fem
            );
        }
        for (steps, tolerance) in [(1, 1e-6), (2, 1e-6), (1, 1e-9)] {
            let (per_frame, cg, sag) = run_newmark(&mesh, steps, tolerance);
            println!(
                "  newmark  {steps} step/frame tol {tolerance:.0e}: {per_frame:.4} s/frame, {cg:.0} cg iterations/step, mean sag {sag:.4e} m ({:+.3} of fem)",
                (sag - fem) / fem
            );
        }
    }
}
