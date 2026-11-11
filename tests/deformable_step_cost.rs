//! A measurement, not a check: what a [`DeformableBody`] step costs on a body the size of
//! the seven-letter logo (about 13 000 tetrahedra, 4 500 particles), with the logo's
//! configuration (8 iterations, 4 substeps, shape stiffness 1, volume compliance 1e-3, no
//! edge compliance). Run with
//! `cargo test --release --test deformable_step_cost -- --ignored --nocapture`, under a
//! sampling profiler to see where the time goes. The body is one block, not the letters:
//! the cost per constraint does not depend on the shape, and the letters' shape data is not
//! in this repository.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "timing harness: std::time::Instant measures the wall-clock cost of the step; nothing it reads enters a result"
)]

use alice_physics_core::deformable::{DeformableBody, DeformableConfig, Material, NeoHookeanLaw};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::sdf_collider::SdfField;
use alice_physics_core::sdf_fem_mesh;

/// an axis-aligned block, exact signed distance (metres)
struct BlockSdf([f32; 3]);

impl SdfField for BlockSdf {
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

fn logo_sized_body(material: Material) -> DeformableBody {
    // 0.02 m cells over a 0.56 × 0.24 × 0.16 m block: 28 × 12 × 8 = 2688 cells, the marching
    // tetrahedra give about 13 000 tetrahedra on 3 500 particles like the seven letters
    let half = [0.28f32, 0.12, 0.08];
    let cell = 0.02f32;
    let mesh = sdf_fem_mesh::generate_marching_tets(
        &BlockSdf(half),
        [-half[0] - cell, -half[1] - cell, -half[2] - cell],
        [half[0] + cell, half[1] + cell, half[2] + cell],
        cell,
    );
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
        .expect("a closed block has boundary faces")
        .iter()
        .map(|f| [f[0] as usize, f[1] as usize, f[2] as usize])
        .collect();
    let mut body = DeformableBody::new(positions, &tets, surface, Fix128::from_ratio(1, 200));
    body.config = DeformableConfig {
        material,
        iterations: 8,
        substeps: 4,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(-9.81), Fix128::ZERO),
        shape_stiffness: Fix128::ONE,
        volume_compliance: Fix128::from_ratio(1, 1000),
        edge_compliance: Fix128::ZERO,
        ..DeformableConfig::default()
    };
    body
}

#[test]
#[ignore = "a measurement, not a check"]
fn what_a_step_costs_on_a_logo_sized_block() {
    let mut body = logo_sized_body(Material::Constraints);
    let (n, tets) = (body.particle_count(), body.tetrahedra.len());
    let dt = Fix128::from_ratio(1, 60);
    body.step(dt); // warm up
    let t = std::time::Instant::now();
    let steps = 20;
    for _ in 0..steps {
        body.step(dt);
    }
    let per_step = t.elapsed() / steps;
    println!(
        "{n} particles, {tets} tets: {per_step:?} a step (iterations {}, substeps {})",
        body.config.iterations, body.config.substeps
    );
}

/// The same block with the Neo-Hookean material at the substeps its convergence needs at this
/// cell (`h ≲ 1/√(Σ w|∇C|²)`, 64 per frame at 1.25 cm cells of the oracle; this block's cells
/// are 2 cm, so 40 would do, 64 is measured for the comparison with the oracle's setting):
/// the cost of a correct L0 frame.
#[test]
#[ignore = "a measurement, not a check"]
fn what_a_converged_neo_hookean_step_costs_on_a_logo_sized_block() {
    let law = NeoHookeanLaw::new(Fix128::from_f64(1.2725e6), Fix128::from_f64(0.49))
        .expect("E > 0, ν in range");
    let mut body = logo_sized_body(Material::NeoHookean(law));
    body.config.substeps = 64;
    body.config.iterations = 8;
    let (n, tets) = (body.particle_count(), body.tetrahedra.len());
    let dt = Fix128::from_ratio(1, 60);
    body.step(dt); // warm up
    let t = std::time::Instant::now();
    let steps = 5;
    for _ in 0..steps {
        body.step(dt);
    }
    let per_step = t.elapsed() / steps;
    println!(
        "{n} particles, {tets} tets, Neo-Hookean: {per_step:?} a step (iterations {}, substeps {})",
        body.config.iterations, body.config.substeps
    );
}
