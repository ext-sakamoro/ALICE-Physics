//! The lowest axial mode of a column's own discretisation: inverse iteration on the
//! generalised eigenproblem `K φ = ω² M φ` of the P1 mesh (lumped `M`), kept in the subspace
//! that is symmetric under `x → −x` and `z → −z` so it converges to the axial mode and not to
//! the bending modes (the lowest ones of a slender column), with the iterate symmetrised
//! every step so `Fix128` rounding cannot seed an antisymmetric component.
//!
//! `K⁻¹ (M v)` is the public static solve with `M v` as nodal loads (the same solve the sag
//! oracles use), so the mode is exact for the discretisation the transient solvers advance:
//! the comparison that follows isolates the integrator from the mesh.
//!
//! Units are the finite element module's: mm, N, MPa, tonne (`ρ` in t/mm³), so `ω` is in
//! rad/s.
#![allow(
    dead_code,
    reason = "shared by several test files through #[path]; each uses a subset of these helpers"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the Rayleigh quotient and the mirror pairing are computed in f64 from the mesh, never by the integrator under test"
)]

use alice_physics_core::linear_elastic_fem::{
    self, Axis, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_fem_mesh::SdfTetMesh;

/// for every vertex, the vertex at `(−x, y, −z)` (the lattice is symmetric), and the one at
/// `(−x, y, z)` and `(x, y, −z)`
fn mirrors(mesh: &SdfTetMesh) -> Vec<[usize; 3]> {
    let find = |p: [f32; 3]| {
        mesh.vertices
            .iter()
            .position(|v| {
                (v[0] - p[0]).abs() < 1e-5
                    && (v[1] - p[1]).abs() < 1e-5
                    && (v[2] - p[2]).abs() < 1e-5
            })
            .expect("the lattice is mirror symmetric")
    };
    mesh.vertices
        .iter()
        .map(|v| {
            [
                find([-v[0], v[1], v[2]]),
                find([v[0], v[1], -v[2]]),
                find([-v[0], v[1], -v[2]]),
            ]
        })
        .collect()
}

/// project a displacement field onto the x/z-symmetric subspace: `u_y` even in both mirrors,
/// `u_x` odd under `x → −x` and even under `z → −z`, `u_z` the other way round
fn symmetrise(u: &mut [f64], mirrors: &[[usize; 3]]) {
    let n = mirrors.len();
    let mut out = vec![0.0; 3 * n];
    for i in 0..n {
        let [mx, mz, mxz] = mirrors[i];
        out[3 * i] = (u[3 * i] - u[3 * mx] + u[3 * mz] - u[3 * mxz]) / 4.0;
        out[3 * i + 1] = (u[3 * i + 1] + u[3 * mx + 1] + u[3 * mz + 1] + u[3 * mxz + 1]) / 4.0;
        out[3 * i + 2] = (u[3 * i + 2] + u[3 * mx + 2] - u[3 * mz + 2] - u[3 * mxz + 2]) / 4.0;
    }
    u.copy_from_slice(&out);
}

/// `(ω₁, φ₁)` of the axial mode: `ω₁` in rad/s, `φ₁` three components per vertex, scaled to
/// unit `M`-norm. `masses_t` are the lumped vertex masses in tonnes, `fixed` the vertices
/// held (every component), `iterations` the inverse iterations (the Rayleigh quotient
/// converges quadratically; 12 is far more than the 1e-6 relative the oracles ask)
pub fn axial_mode(
    mesh_mm: &SdfTetMesh,
    material: &ElasticMaterial,
    masses_t: &[f64],
    fixed: &[usize],
    iterations: usize,
) -> (f64, Vec<f64>) {
    let n = mesh_mm.vertices.len();
    let mirrors = mirrors(mesh_mm);
    let is_fixed: Vec<bool> = (0..n).map(|i| fixed.contains(&i)).collect();
    // start from a pure axial field growing with height: the shape of the static sag
    let y0 = mesh_mm
        .vertices
        .iter()
        .map(|v| v[1])
        .fold(f32::INFINITY, f32::min);
    let mut v = vec![0.0f64; 3 * n];
    for i in 0..n {
        if !is_fixed[i] {
            v[3 * i + 1] = -f64::from(mesh_mm.vertices[i][1] - y0);
        }
    }
    let mut omega2 = 0.0;
    for _ in 0..iterations {
        // w = K⁻¹ (M v): the static solve with M v as the loads
        let mut bc = BoundaryConditions::new();
        for &i in fixed {
            bc.fix(i as u32);
        }
        for i in 0..n {
            if is_fixed[i] {
                continue;
            }
            for (axis, k) in [(Axis::X, 0), (Axis::Y, 1), (Axis::Z, 2)] {
                let f = masses_t[i] * v[3 * i + k];
                if f != 0.0 {
                    bc.add_load(i as u32, axis, Fix128::from_f64(f));
                }
            }
        }
        let sol = linear_elastic_fem::solve(mesh_mm, material, &bc, &SolverConfig::default())
            .expect("the mode solve");
        let mut w = vec![0.0f64; 3 * n];
        for i in 0..n {
            for k in 0..3 {
                w[3 * i + k] = sol.displacements[i][k].to_f64();
            }
        }
        symmetrise(&mut w, &mirrors);
        for i in fixed {
            for k in 0..3 {
                w[3 * i + k] = 0.0;
            }
        }
        // Rayleigh quotient with the previous iterate: ω² = vᵀ M v / vᵀ M w (K w = M v)
        let mut vmv = 0.0;
        let mut vmw = 0.0;
        for i in 0..n {
            for k in 0..3 {
                vmv += masses_t[i] * v[3 * i + k] * v[3 * i + k];
                vmw += masses_t[i] * v[3 * i + k] * w[3 * i + k];
            }
        }
        omega2 = vmv / vmw;
        // normalise the new iterate to unit M-norm
        let mut wmw = 0.0;
        for i in 0..n {
            for k in 0..3 {
                wmw += masses_t[i] * w[3 * i + k] * w[3 * i + k];
            }
        }
        let scale = 1.0 / wmw.sqrt();
        for x in &mut w {
            *x *= scale;
        }
        v = w;
    }
    (omega2.sqrt(), v)
}

/// the frequency the average-acceleration Newmark scheme oscillates at for a mode of
/// frequency `omega` and step `dt`: `ω̄ = (2/dt)·atan(ω dt/2)` (period elongation, no
/// amplitude change)
pub fn newmark_frequency(omega: f64, dt: f64) -> f64 {
    2.0 / dt * (omega * dt / 2.0).atan()
}
