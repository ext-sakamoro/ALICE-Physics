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
pub fn mirrors(mesh: &SdfTetMesh) -> Vec<[usize; 3]> {
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
pub fn symmetrise(u: &mut [f64], mirrors: &[[usize; 3]]) {
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

/// `(ω₁, φ₁)` of the axial mode with the vertices `fixed` held in every component: see
/// [`axial_mode_held`], whose error bound is dropped here
pub fn axial_mode(
    mesh_mm: &SdfTetMesh,
    material: &ElasticMaterial,
    masses_t: &[f64],
    fixed: &[usize],
    iterations: usize,
) -> (f64, Vec<f64>) {
    let held = held_dofs(fixed, &[], mesh_mm.vertices.len());
    let (omega, phi, _) = axial_mode_held(mesh_mm, material, masses_t, &held, iterations);
    (omega, phi)
}

/// the frequency the average-acceleration Newmark scheme oscillates at for a mode of
/// frequency `omega` and step `dt`: `ω̄ = (2/dt)·atan(ω dt/2)` (period elongation, no
/// amplitude change)
pub fn newmark_frequency(omega: f64, dt: f64) -> f64 {
    2.0 / dt * (omega * dt / 2.0).atan()
}

/// The stiffness matrix of one P1 tetrahedron in `f64` (`12 × 12`, three components per
/// vertex, rows and columns ordered vertex by vertex): `K_e = V Bᵀ D B` with `B` the constant
/// strain-displacement matrix of the linear shape functions and `D` the isotropic Hooke
/// matrix (`λ`, `μ`). Units: the mesh's length unit and the material's stress unit.
pub fn element_stiffness(p: [[f64; 3]; 4], lambda: f64, mu: f64) -> [[f64; 12]; 12] {
    // gradients of the shape functions: ∇Nᵢ = row i of J⁻ᵀ, J the edge matrix from vertex 0
    let e = |a: usize, b: usize| [p[b][0] - p[a][0], p[b][1] - p[a][1], p[b][2] - p[a][2]];
    let (ab, ac, ad) = (e(0, 1), e(0, 2), e(0, 3));
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let det = dot(ab, cross(ac, ad));
    let volume = det.abs() / 6.0;
    // ∇N₁ = (ac × ad)/det, ∇N₂ = (ad × ab)/det, ∇N₃ = (ab × ac)/det, ∇N₀ = −(∇N₁+∇N₂+∇N₃)
    let g1 = cross(ac, ad).map(|x| x / det);
    let g2 = cross(ad, ab).map(|x| x / det);
    let g3 = cross(ab, ac).map(|x| x / det);
    let g0 = [
        -(g1[0] + g2[0] + g3[0]),
        -(g1[1] + g2[1] + g3[1]),
        -(g1[2] + g2[2] + g3[2]),
    ];
    let g = [g0, g1, g2, g3];
    // K_e[3a+i][3b+j] = V (λ gₐᵢ g_bⱼ + μ gₐⱼ g_bᵢ + μ δᵢⱼ (gₐ · g_b))
    let mut k = [[0.0; 12]; 12];
    for a in 0..4 {
        for b in 0..4 {
            let gab = dot(g[a], g[b]);
            for i in 0..3 {
                for j in 0..3 {
                    let delta = if i == j { 1.0 } else { 0.0 };
                    k[3 * a + i][3 * b + j] = volume
                        * (lambda * g[a][i] * g[b][j] + mu * g[a][j] * g[b][i] + mu * delta * gab);
                }
            }
        }
    }
    k
}

/// An upper bound on the largest eigenvalue of the assembled (and, with the same elements
/// rotated, the co-rotationally warped) stiffness: `uᵀ K u = Σ_e u_eᵀ K_e u_e ≤
/// Σ_e λ_max(K_e) ‖u_e‖² ≤ max_i (Σ_{e ∋ i} λ_max(K_e)) ‖u‖²`, with `λ_max(K_e)` bounded by
/// Gershgorin's largest absolute row sum of `K_e` (a rotation `R K_e Rᵀ` keeps its
/// eigenvalues, so the bound holds for every warping). Units: stress × length.
pub fn stiffness_eigenvalue_bound(mesh: &SdfTetMesh, lambda: f64, mu: f64) -> f64 {
    let mut per_vertex = vec![0.0f64; mesh.vertices.len()];
    for t in &mesh.tets {
        let p = t.vertices.map(|i| mesh.vertices[i as usize].map(f64::from));
        let k = element_stiffness(p, lambda, mu);
        let gershgorin = k
            .iter()
            .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0, f64::max);
        for &i in &t.vertices {
            per_vertex[i as usize] += gershgorin;
        }
    }
    per_vertex.iter().copied().fold(0.0, f64::max)
}

/// The polar factor `R` of a `3 × 3` matrix `F` with positive determinant, by Higham's
/// iteration `R ← (R + R⁻ᵀ)/2` (quadratic convergence; 40 steps are far more than `f64`
/// needs for the deformation gradients of a mesh that has not inverted)
pub fn polar_factor(f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = f;
    for _ in 0..40 {
        // inverse transpose via the adjugate
        let c = |i: usize, j: usize| {
            let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            r[i1][j1] * r[i2][j2] - r[i1][j2] * r[i2][j1]
        };
        let det = r[0][0] * c(0, 0) + r[0][1] * c(0, 1) + r[0][2] * c(0, 2);
        let mut next = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                // (R⁻ᵀ)ᵢⱼ = cofactor(i, j) / det
                next[i][j] = 0.5 * (r[i][j] + c(i, j) / det);
            }
        }
        r = next;
    }
    r
}

/// `A u` for the co-rotational Newmark operator `A = (4/dt²) M + K'` in `f64`, with `K'` the
/// warped stiffness `Σₑ Rₑ Kₑ Rₑᵀ` (`Rₑ` the polar factor of each element's deformation
/// gradient at `x₀ + u`) and `M` the lumped (`masses` on the diagonal) or consistent
/// (`(ρ Vₑ / 20)(1 + δᵢⱼ)` per element) mass. Units: the mesh's, `u` in mm, out in N.
pub fn newmark_operator_times(
    mesh_mm: &SdfTetMesh,
    lambda: f64,
    mu: f64,
    masses_t: &[f64],
    consistent: bool,
    inv_beta_dt2: f64,
    u: &[f64],
) -> Vec<f64> {
    warped_operator_times(
        mesh_mm,
        lambda,
        mu,
        masses_t,
        consistent,
        inv_beta_dt2,
        u,
        u,
    )
}

/// [`newmark_operator_times`] with the warping taken at `warp_state` and the operator applied
/// to `u`: linear in `u`, so it assembles `K'(warp_state)` column by column
#[allow(
    clippy::too_many_arguments,
    reason = "the mesh, the two Lamé constants, the masses, the mass kind, the Newmark coefficient, the warping state and the vector are all inputs of one product"
)]
pub fn warped_operator_times(
    mesh_mm: &SdfTetMesh,
    lambda: f64,
    mu: f64,
    masses_t: &[f64],
    consistent: bool,
    inv_beta_dt2: f64,
    warp_state: &[f64],
    u: &[f64],
) -> Vec<f64> {
    let n = mesh_mm.vertices.len();
    let mut out = vec![0.0; 3 * n];
    // the mass term
    if consistent {
        let volume_of = |t: &alice_physics_core::sdf_fem_mesh::Tetrahedron| {
            let q = t
                .vertices
                .map(|i| mesh_mm.vertices[i as usize].map(f64::from));
            let e = |a: usize, b: usize| [q[b][0] - q[a][0], q[b][1] - q[a][1], q[b][2] - q[a][2]];
            let (ab, ac, ad) = (e(0, 1), e(0, 2), e(0, 3));
            (ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
                + ab[1] * (ac[2] * ad[0] - ac[0] * ad[2])
                + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]))
                .abs()
                / 6.0
        };
        // the lumped mass of a vertex is ρ Σ Vₑ / 4 over its elements, so the density the
        // masses imply is their sum over the mesh's volume
        let rho = masses_t.iter().sum::<f64>() / mesh_mm.tets.iter().map(volume_of).sum::<f64>();
        for t in &mesh_mm.tets {
            let volume = volume_of(t);
            let share = rho * volume / 20.0;
            for (a, &ia) in t.vertices.iter().enumerate() {
                for (b, &ib) in t.vertices.iter().enumerate() {
                    let w = if a == b { 2.0 * share } else { share };
                    for k in 0..3 {
                        out[3 * ia as usize + k] += inv_beta_dt2 * w * u[3 * ib as usize + k];
                    }
                }
            }
        }
    } else {
        for i in 0..n {
            for k in 0..3 {
                out[3 * i + k] += inv_beta_dt2 * masses_t[i] * u[3 * i + k];
            }
        }
    }
    // the warped stiffness term
    for t in &mesh_mm.tets {
        let p = t
            .vertices
            .map(|i| mesh_mm.vertices[i as usize].map(f64::from));
        let x: [[f64; 3]; 4] = core::array::from_fn(|a| {
            let i = t.vertices[a] as usize;
            [
                p[a][0] + warp_state[3 * i],
                p[a][1] + warp_state[3 * i + 1],
                p[a][2] + warp_state[3 * i + 2],
            ]
        });
        // F = Ds Dm⁻¹ with the edge matrices from vertex 0
        let dm: [[f64; 3]; 3] =
            core::array::from_fn(|r| core::array::from_fn(|c| p[c + 1][r] - p[0][r]));
        let ds: [[f64; 3]; 3] =
            core::array::from_fn(|r| core::array::from_fn(|c| x[c + 1][r] - x[0][r]));
        let cof = |m: &[[f64; 3]; 3], i: usize, j: usize| {
            let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1]
        };
        let det = dm[0][0] * cof(&dm, 0, 0) + dm[0][1] * cof(&dm, 0, 1) + dm[0][2] * cof(&dm, 0, 2);
        // Dm⁻¹ = adj(Dm)ᵀ / det, adj(Dm)ᵢⱼ = cofactor(j, i)
        let dm_inv: [[f64; 3]; 3] =
            core::array::from_fn(|i| core::array::from_fn(|j| cof(&dm, j, i) / det));
        let f: [[f64; 3]; 3] = core::array::from_fn(|i| {
            core::array::from_fn(|j| (0..3).map(|k| ds[i][k] * dm_inv[k][j]).sum())
        });
        let r = polar_factor(f);
        let k = element_stiffness(p, lambda, mu);
        // Rᵀ u_e, then K_e, then R
        let mut ue = [0.0; 12];
        for (a, &ia) in t.vertices.iter().enumerate() {
            for i in 0..3 {
                ue[3 * a + i] = (0..3).map(|j| r[j][i] * u[3 * ia as usize + j]).sum();
            }
        }
        let mut ke = [0.0; 12];
        for i in 0..12 {
            ke[i] = (0..12).map(|j| k[i][j] * ue[j]).sum();
        }
        for (a, &ia) in t.vertices.iter().enumerate() {
            for i in 0..3 {
                out[3 * ia as usize + i] += (0..3).map(|j| r[i][j] * ke[3 * a + j]).sum::<f64>();
            }
        }
    }
    out
}

/// The lowest axial mode of the co-rotationally warped stiffness at the state `u_state`
/// (mm): `K'(u_state)` assembled densely in `f64` by applying [`newmark_operator_times`]
/// with no mass term to the unit vectors, restricted to the free degrees of freedom,
/// factored by Cholesky, and inverse-iterated against the lumped mass in the x/z-symmetric
/// subspace exactly as [`axial_mode`] does for the unwarped stiffness. Returns `ω` (rad/s).
/// The warped mode is the instantaneous first frequency of the co-rotational system at
/// that state; the difference from the unwarped `ω₁` is the warping's effect on the period.
pub fn warped_axial_mode(
    mesh_mm: &SdfTetMesh,
    lambda: f64,
    mu: f64,
    masses_t: &[f64],
    held: &[bool],
    u_state: &[f64],
    iterations: usize,
) -> f64 {
    let n = mesh_mm.vertices.len();
    let ndof = 3 * n;
    let free: Vec<usize> = (0..ndof).filter(|&d| !held[d]).collect();
    let m = free.len();
    // dense K' on the free degrees of freedom
    let mut k = vec![vec![0.0f64; m]; m];
    for (col, &d) in free.iter().enumerate() {
        let mut e = vec![0.0; ndof];
        e[d] = 1.0;
        let column = warped_operator_times(mesh_mm, lambda, mu, masses_t, false, 0.0, u_state, &e);
        for (row, &r) in free.iter().enumerate() {
            k[row][col] = column[r];
        }
    }
    // Cholesky K' = L Lᵀ
    let mut l = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for j in 0..=i {
            let mut sum = k[i][j];
            for (lip, ljp) in l[i][..j].iter().zip(&l[j][..j]) {
                sum -= lip * ljp;
            }
            if i == j {
                assert!(
                    sum > 0.0,
                    "the warped stiffness is positive definite on the free dofs"
                );
                l[i][i] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    let solve = |b: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0; m];
        for i in 0..m {
            let mut sum = b[i];
            for p in 0..i {
                sum -= l[i][p] * y[p];
            }
            y[i] = sum / l[i][i];
        }
        let mut x = vec![0.0; m];
        for i in (0..m).rev() {
            let mut sum = y[i];
            for p in (i + 1)..m {
                sum -= l[p][i] * x[p];
            }
            x[i] = sum / l[i][i];
        }
        x
    };
    let mirrors = mirrors(mesh_mm);
    let y0 = mesh_mm
        .vertices
        .iter()
        .map(|v| v[1])
        .fold(f32::INFINITY, f32::min);
    let mut v = vec![0.0f64; ndof];
    for i in 0..n {
        if !held[3 * i + 1] {
            v[3 * i + 1] = -f64::from(mesh_mm.vertices[i][1] - y0);
        }
    }
    let mut omega2 = 0.0;
    for _ in 0..iterations {
        let mv: Vec<f64> = free.iter().map(|&d| masses_t[d / 3] * v[d]).collect();
        let wf = solve(&mv);
        let mut w = vec![0.0f64; ndof];
        for (idx, &d) in free.iter().enumerate() {
            w[d] = wf[idx];
        }
        symmetrise(&mut w, &mirrors);
        for d in 0..ndof {
            if held[d] {
                w[d] = 0.0;
            }
        }
        let mut vmv = 0.0;
        let mut vmw = 0.0;
        let mut wmw = 0.0;
        for i in 0..n {
            for k in 0..3 {
                vmv += masses_t[i] * v[3 * i + k] * v[3 * i + k];
                vmw += masses_t[i] * v[3 * i + k] * w[3 * i + k];
                wmw += masses_t[i] * w[3 * i + k] * w[3 * i + k];
            }
        }
        omega2 = vmv / vmw;
        let scale = 1.0 / wmw.sqrt();
        for x in &mut w {
            *x *= scale;
        }
        v = w;
    }
    omega2.sqrt()
}

/// Degrees of freedom held at zero, by `(vertex, axis)`: the general form of the fixed
/// vertices of [`axial_mode`], so a roller base (bottom vertices held in `y` only) can be
/// analysed too
pub fn held_dofs(fixed_vertices: &[usize], fixed_y_vertices: &[usize], n: usize) -> Vec<bool> {
    let mut held = vec![false; 3 * n];
    for &i in fixed_vertices {
        for k in 0..3 {
            held[3 * i + k] = true;
        }
    }
    for &i in fixed_y_vertices {
        held[3 * i + 1] = true;
    }
    held
}

/// [`axial_mode`] with the held degrees of freedom given one by one (`held` from
/// [`held_dofs`]): the static solve prescribes each held component to zero
/// Returns `(ω₁, φ₁, δ)`: `ω₁` in rad/s, `φ₁` of unit `M`-norm, and `δ` a bound on the
/// relative error of `ω₁` from the solves: the estimate is `ω² = vᵀMv / vᵀMw` with
/// `w = K⁻¹Mv + e`, `K e = −r`, so `|δ(vᵀMw)| = |wₑₓₐ꜀ₜᵀ r| ≤ ‖w‖₂ ‖r‖₂` with
/// `‖r‖₂ ≤ max(tol ‖Mv‖₂, floor)` (the solve's stopping rule, the load scaled up by `10⁶` so
/// the floor does not bind), and `δ(ω²)/ω² ≤ ‖w‖₂ ‖r‖₂ ω²`, `δω/ω` half of it; the vector's
/// own convergence (quadratic in the Rayleigh quotient) is far below after 12 iterations.
pub fn axial_mode_held(
    mesh_mm: &SdfTetMesh,
    material: &ElasticMaterial,
    masses_t: &[f64],
    held: &[bool],
    iterations: usize,
) -> (f64, Vec<f64>, f64) {
    const SCALE: f64 = 1e6;
    let config =
        SolverConfig::try_new(20_000, Fix128::from_f64(1e-9)).expect("budget and tolerance");
    let n = mesh_mm.vertices.len();
    let mirrors = mirrors(mesh_mm);
    let y0 = mesh_mm
        .vertices
        .iter()
        .map(|v| v[1])
        .fold(f32::INFINITY, f32::min);
    let mut v = vec![0.0f64; 3 * n];
    for i in 0..n {
        if !held[3 * i + 1] {
            v[3 * i + 1] = -f64::from(mesh_mm.vertices[i][1] - y0);
        }
    }
    let mut omega2 = 0.0;
    let mut omega_err = 0.0;
    for _ in 0..iterations {
        let mut bc = BoundaryConditions::new();
        let mut load_norm2 = 0.0;
        for i in 0..n {
            for (axis, k) in [(Axis::X, 0), (Axis::Y, 1), (Axis::Z, 2)] {
                if held[3 * i + k] {
                    bc.prescribe(i as u32, axis, Fix128::ZERO);
                } else {
                    let f = SCALE * masses_t[i] * v[3 * i + k];
                    load_norm2 += f * f;
                    if f != 0.0 {
                        bc.add_load(i as u32, axis, Fix128::from_f64(f));
                    }
                }
            }
        }
        let sol =
            linear_elastic_fem::solve(mesh_mm, material, &bc, &config).expect("the mode solve");
        let mut w = vec![0.0f64; 3 * n];
        for i in 0..n {
            for k in 0..3 {
                w[3 * i + k] = sol.displacements[i][k].to_f64() / SCALE;
            }
        }
        // the solve's residual bound, in the scaled units, and the error it allows in vᵀMw
        let residual =
            (1e-9 * load_norm2.sqrt()).max(linear_elastic_fem::RESIDUAL_NORM_FLOOR.to_f64());
        let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        symmetrise(&mut w, &mirrors);
        for d in 0..3 * n {
            if held[d] {
                w[d] = 0.0;
            }
        }
        let mut vmv = 0.0;
        let mut vmw = 0.0;
        let mut wmw = 0.0;
        for i in 0..n {
            for k in 0..3 {
                vmv += masses_t[i] * v[3 * i + k] * v[3 * i + k];
                vmw += masses_t[i] * v[3 * i + k] * w[3 * i + k];
                wmw += masses_t[i] * w[3 * i + k] * w[3 * i + k];
            }
        }
        omega2 = vmv / vmw;
        omega_err = 0.5 * w_norm * SCALE * residual / SCALE / vmw;
        let scale = 1.0 / wmw.sqrt();
        for x in &mut w {
            *x *= scale;
        }
        v = w;
    }
    (omega2.sqrt(), v, omega_err)
}
