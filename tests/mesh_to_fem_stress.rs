//! What the two mesh generators are worth as *stress* input.
//!
//! `mesh_conformity.rs` says the elements fit together and `mesh_quality.rs`
//! says they are not slivers. Neither says the mesh can be handed to
//! [`alice_physics::linear_elastic_fem`] and produce a stress field worth
//! reading, and that is the only reason the mesher exists. This file measures
//! it end to end, on both generators, and the two answers are different enough
//! that the difference is the point.
//!
//! # The instrument
//!
//! Prescribe an exact linear displacement field on every boundary node. A P1
//! tetrahedral element reproduces a linear field exactly, so for *any*
//! conforming mesh of *any* domain the interior must come back with the same
//! constant stress, whatever the element shapes are. Deviation from it is
//! therefore not discretisation error — it is the solver failing to converge on
//! an ill-conditioned mesh.
//!
//! This is the one thing a patch test is good for here. It says nothing about
//! whether the faces line up (measured: exact to 3.6e-15 MPa on a mesh with 576
//! of 768 faces dangling), so conformity stays in its own file.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use std::collections::HashMap;

use alice_physics::linear_elastic_fem::{
    solve, Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sdf_fem_mesh::{generate, generate_marching_tets, SdfTetMesh};

/// A bar, deliberately not aligned with the lattice.
///
/// The offsets are there so the surface falls *between* lattice planes. A box
/// whose faces sit exactly on them would put every zero crossing exactly on a
/// corner, no corner would need warping, and the test would exercise none of
/// the machinery it is here for.
fn bar_sdf() -> ClosureSdf {
    const HALF: [f32; 3] = [4.7, 1.3, 1.1];
    const CENTRE: [f32; 3] = [0.17, 0.09, -0.13];
    ClosureSdf::new(
        move |x, y, z| {
            let d = [
                (x - CENTRE[0]).abs() - HALF[0],
                (y - CENTRE[1]).abs() - HALF[1],
                (z - CENTRE[2]).abs() - HALF[2],
            ];
            let outside = [d[0].max(0.0), d[1].max(0.0), d[2].max(0.0)];
            let len = (outside[0] * outside[0] + outside[1] * outside[1] + outside[2] * outside[2])
                .sqrt();
            len + d[0].max(d[1]).max(d[2]).min(0.0)
        },
        |_, _, _| (1.0, 0.0, 0.0),
    )
}

const YOUNGS_MPA: f64 = 200_000.0;
const POISSON: f64 = 0.3;

fn steel() -> ElasticMaterial {
    ElasticMaterial::new(fx(YOUNGS_MPA), fx(POISSON)).expect("steel is a valid material")
}

/// Vertices that lie on a face used by exactly one element.
fn boundary_vertices(mesh: &SdfTetMesh) -> Vec<u32> {
    const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let mut uses: HashMap<[u32; 3], usize> = HashMap::new();
    for tet in &mesh.tets {
        for f in FACES {
            let mut key = [tet.vertices[f[0]], tet.vertices[f[1]], tet.vertices[f[2]]];
            key.sort_unstable();
            *uses.entry(key).or_insert(0) += 1;
        }
    }
    let mut on_boundary = vec![false; mesh.vertices.len()];
    for (face, count) in &uses {
        if *count == 1 {
            for v in face {
                on_boundary[*v as usize] = true;
            }
        }
    }
    (0..mesh.vertices.len())
        .filter(|i| on_boundary[*i])
        .map(|i| u32::try_from(i).expect("vertex count fits u32"))
        .collect()
}

/// Uniaxial strain `eps` along x, with the transverse contraction Poisson's
/// ratio requires, prescribed on the whole boundary.
///
/// The resulting stress is uniform and closed form: `sigma_xx = E * eps` with
/// every other component zero, because the transverse strains are exactly the
/// ones that leave the transverse stresses free.
///
/// `nu` has to match the material's Poisson ratio: it is what makes the
/// transverse stresses vanish, and passing a different one would prescribe a
/// triaxial state whose von Mises stress is not `E * eps`.
fn prescribe_uniaxial(mesh: &SdfTetMesh, nu: f64, eps: f64) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for v in boundary_vertices(mesh) {
        let p = mesh.vertices[v as usize];
        bc.prescribe(v, Axis::X, fx(eps * f64::from(p[0])));
        bc.prescribe(v, Axis::Y, fx(-nu * eps * f64::from(p[1])));
        bc.prescribe(v, Axis::Z, fx(-nu * eps * f64::from(p[2])));
    }
    bc
}

fn fx(value: f64) -> Fix128 {
    Fix128::from_f64(value)
}

/// The mesh a stress analysis is meant to use.
///
/// The claim is not that the answer is approximately right. On a linear field a
/// P1 mesh is *exact*, so anything the solver gives back that is not the
/// constant `E * eps` is the conjugate gradient iteration failing on badly
/// shaped elements — which is exactly what the warp in `sdf_fem_mesh` is there
/// to prevent, and what this measures from the far end.
#[test]
fn marching_tets_mesh_carries_an_exact_uniform_stress() {
    let sdf = bar_sdf();
    let material = steel();
    let eps = 1.0e-3;
    for cell in [0.5_f32, 0.375, 0.25] {
        let mesh = generate_marching_tets(&sdf, [-6.0, -3.0, -3.0], [6.0, 3.0, 3.0], cell);
        let bc = prescribe_uniaxial(&mesh, POISSON, eps);
        let solution = solve(&mesh, &material, &bc, &SolverConfig::default()).unwrap_or_else(|e| {
            panic!("cell {cell}: the FEM rejected a mesh this crate produced: {e:?}")
        });

        let expected = YOUNGS_MPA * eps;
        let mut worst = 0.0_f64;
        for s in &solution.element_stress {
            worst = worst.max((s.von_mises().to_f64() - expected).abs() / expected);
        }
        eprintln!(
            "[stress]  marching  cell {cell:<6} tets {:>5}  iterations {:>5}  \
             worst von Mises deviation {:.3e}  (expected {expected:.1} MPa)",
            mesh.tet_count(),
            solution.iterations,
            worst
        );
        assert!(
            worst < 1.0e-5,
            "cell {cell}: a linear field is exact on P1 elements, so {worst:.3e} relative \
             deviation is the solver losing on element shape, not discretisation"
        );
    }
}

/// The mesh a stress analysis must *not* use, and the measurement that says so.
///
/// [`generate`] keeps only cubes with all eight corners inside, so the domain it
/// hands the FEM is a staircase strictly inside the shape. Measured on a
/// 9.4 x 2.6 x 2.2 bar against its exact volume of 53.768:
///
/// | cell | meshed volume | fraction |
/// |---|---|---|
/// | 0.5 | 27.000 | 50% |
/// | 0.375 | 37.969 | 71% |
/// | 0.25 | 36.422 | 68% |
///
/// The body being solved is a third smaller than the one the caller described,
/// and refining from 0.375 to 0.25 made it *worse*, because a cell size that
/// does not divide the thickness loses a whole layer. A load spread over the
/// real cross-section is therefore applied to a smaller one, and every stress
/// that comes back is wrong by that ratio before element shape is considered at
/// all.
///
/// # What this test does not show
///
/// The uniform-stress instrument at the top of this file is **blind to the
/// defect it is measuring**. `generate` returns the exact constant stress to
/// 6e-9 relative, the same as the good mesh, because a linear displacement
/// field is reproduced exactly by P1 elements on any domain whatever its shape
/// — a staircase included. Said plainly so the green above is not mistaken for
/// evidence that `generate` is fit for stress: the only thing separating the two
/// generators here is the volume, and that is what this test pins.
///
/// The other expected consequence — that the staircase's re-entrant corners are
/// stress singularities, so a peak stress near one grows instead of converging
/// under refinement — is **not measured here**, and is not claimed as a result.
/// Seeing it needs a load case whose exact solution is non-uniform, which is a
/// separate scene from this one.
///
#[test]
fn whole_cube_dicing_is_not_a_stress_mesh() {
    let sdf = bar_sdf();
    let material = steel();
    let eps = 1.0e-3;
    let expected = YOUNGS_MPA * eps;
    let mut volumes = Vec::new();
    for cell in [0.5_f32, 0.375, 0.25] {
        let mesh = generate(&sdf, [-6.0, -3.0, -3.0], [6.0, 3.0, 3.0], cell);
        let bc = prescribe_uniaxial(&mesh, POISSON, eps);
        let solution = solve(&mesh, &material, &bc, &SolverConfig::default())
            .unwrap_or_else(|e| panic!("cell {cell}: {e:?}"));
        let mut worst = 0.0_f64;
        for s in &solution.element_stress {
            worst = worst.max((s.von_mises().to_f64() - expected).abs() / expected);
        }
        let volume = mesh_volume(&mesh);
        eprintln!(
            "[stress]  generate  cell {cell:<6} tets {:>5}  iterations {:>5}  \
             worst von Mises deviation {:.3e}  volume {volume:.3} / {:.3}",
            mesh.tet_count(),
            solution.iterations,
            worst,
            analytic_bar_volume()
        );
        volumes.push(volume);
    }
    let analytic = analytic_bar_volume();
    assert!(
        volumes.iter().all(|v| *v < 0.85 * analytic),
        "this test exists to pin the deficit; if `generate` now meshes most of the shape, \
         the guidance that sends stress users to `generate_marching_tets` needs revisiting. \
         Volumes {volumes:?} against {analytic:.3}"
    );
}

fn analytic_bar_volume() -> f64 {
    (2.0 * 4.7) * (2.0 * 1.3) * (2.0 * 1.1)
}

fn mesh_volume(mesh: &SdfTetMesh) -> f64 {
    let mut total = 0.0;
    for tet in &mesh.tets {
        let p: [[f64; 3]; 4] = tet.vertices.map(|v| {
            let w = mesh.vertices[v as usize];
            [f64::from(w[0]), f64::from(w[1]), f64::from(w[2])]
        });
        let e = |i: usize, k: usize| p[i][k] - p[0][k];
        let a = [e(1, 0), e(1, 1), e(1, 2)];
        let b = [e(2, 0), e(2, 1), e(2, 2)];
        let c = [e(3, 0), e(3, 1), e(3, 2)];
        total += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))
            .abs()
            / 6.0;
    }
    total
}
