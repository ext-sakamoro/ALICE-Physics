//! Closed-form oracles for a **hyperelastic** law on the quadratic (P2)
//! tetrahedron.
//!
//! # What is being measured, and what is not
//!
//! The wiring under test is `solve_quadratic_hyperelastic`: a finite-strain
//! total-Lagrangian internal force `Σ_q w_q V₀ P(F(ξ_q)) ∇N(ξ_q)` evaluated at
//! the quadrature points of the element, driven by a modified Newton iteration
//! whose fixed point is `f_ext = f_mat(u)`.
//!
//! ⚠️⚠️ **Every closed-form oracle here uses a deformation with a *uniform* `F`,
//! and that is forced, not a convenience.** The quadrature of the non-linear
//! internal force is not exact and cannot be made so: for Neo-Hookean the first
//! Piola-Kirchhoff stress works out to `P = μF + [K(J−1) − μ]·cof F`, which is
//! degree five in the entries of `F`; with `F` linear in the barycentric
//! coordinates the integrand `P : ∇N` is degree **six** against a rule exact to
//! degree **two**. A manufactured non-uniform solution therefore does **not**
//! satisfy the discrete equations even when it lies in the element's function
//! space, so the "the element reproduces this field exactly" shape of oracle —
//! the one `analytic_quadratic_fem.rs` and `analytic_cubic_fem.rs` use for the
//! linear law — has no non-uniform counterpart here. With `F` uniform, `P` comes
//! out of the integral and what is left is `∫∇N`, degree one, which the rule
//! integrates exactly.
//!
//! ⚠️ **Consequence for what the teeth are.** A uniform deformation is an affine
//! field, which lies in the P1 space as well, so these oracles **do not separate
//! P2 from P1** — a linear tetrahedron would pass them too. What they separate
//! is the **constitutive law**: `uniaxial_tension_is_not_the_linear_answer`
//! runs the identical scene through the small-strain `solve_quadratic` and
//! requires it to miss the closed form by orders of magnitude. Without that
//! pair, an implementation that quietly fell back to linear elasticity would
//! pass everything here.
//!
//! # The closed form
//!
//! Uniaxial tension of an isotropic hyperelastic solid: `F = diag(s, t, t)` with
//! the lateral faces traction free. Writing `q = t²`, `J = s·q`, `B = F Fᵀ`,
//! `I₁ = s² + 2q`, and the module's own volumetric completion
//! `σ = (2/J)[(W₁ + I₁W₂)B − W₂B²] + [κ(J−1) − p_ref/J]·I` with
//! `p_ref = 2(W₁ + 2W₂)` at the undeformed state, multiplying `σ_tt = 0` by `J`
//! gives a quadratic in `q`
//!
//! ```text
//! (2C₂ + κs²) q² + (2C₁ + 2s²C₂ − κs) q − (2C₁ + 4C₂) = 0
//! ```
//!
//! (`q = 1` at `s = 1`, as it must). `κ` is the volumetric modulus the solver
//! derives from the material, `κ = λ − 4C₂`: the value that makes the stress
//! linearise to the enclosing solid's `λ`. This is for a model whose
//! `(W₁, W₂)` are the constants `(C₁, C₂)` — Neo-Hookean is
//! `C₁ = μ/2, C₂ = 0`, Mooney-Rivlin is `(C₁, C₂)` as given. Four operations and
//! two square roots, so the expectation is built without calling the solver's own
//! stress routine.
//!
//! ⚠️ **Why the uniform state is an exact solution of the *discrete* problem.**
//! `P` is constant, so the internal force at node `i` is `P·Σ_e ∫_e ∇N_i`, and
//! `∫_Ω ∇N_i = ∮_∂Ω N_i n dS` vanishes at every interior node and equals the
//! lumped area-normal on the boundary. Traction free means `P·N = 0` on the
//! lateral faces, which is exactly `σ_tt = 0`. The mesh does not enter, which is
//! why the lattice below is perturbed without the expectation changing.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// ⚠️ **Excluded from the `parallel` feature build on purpose.**
// `cargo test --features "parallel"` (ci.yml:77) would otherwise run this file
// a second time, and `parallel` reaches none of the code it exercises:
// `grep -cE 'feature = "parallel"|rayon|par_iter'` is **0** for every module in
// the transitive closure of these tests (`quadratic_elastic_fem`,
// `cubic_elastic_fem`, `linear_elastic_fem`, `hyperelastic`, `math`,
// `sdf_fem_mesh`, `coupled_field`, `sdf_collider`, `sim_field`, `collider`,
// `metric`). The feature bites only the rigid-body solvers (`solver.rs`,
// `solver_tgs*.rs`, `query.rs`, `multi_world.rs`, `laminate.rs`,
// `eulerian_grid.rs`). ⚠️ The duplicate run costs real time and adds zero
// coverage, so the file opts out — which also means it does **not** run under
// `--all-features`.
#![cfg(not(feature = "parallel"))]

use alice_physics::hyperelastic::HyperelasticModel;
use alice_physics::linear_elastic_fem::{
    Axis, BoundaryConditions, CorotationalConfig, ElasticMaterial, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::quadratic_elastic_fem::{
    solve_quadratic, solve_quadratic_hyperelastic, QuadraticMesh,
};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// Cube side (mm).
const SIDE: f64 = 4.0;
/// Young's modulus (MPa) and Poisson's ratio of the enclosing isotropic solid.
///
/// Only the Lamé constant `λ` of this pair reaches the hyperelastic
/// path — it is what fixes the pressure an incompressible strain energy leaves
/// free — but the same pair also drives the linear control, which is the point
/// of using one material for both.
const E_MPA: f64 = 10.0;
const NU: f64 = 0.45;

/// Interior lattice nodes are displaced by this fraction of `h`.
///
/// Carried over from `analytic_cubic_fem.rs` for the same reason: a uniform
/// Kuhn lattice gives the low-order element a superconvergence that is an
/// artefact of the lattice. The oracles here are mesh independent by the
/// argument in the module doc, so the jitter does not change what is expected —
/// it removes the possibility that something in the assembly is resting on the
/// lattice without anyone noticing.
const JITTER: f64 = 0.25;

/// Axial stretch the oracles run at.
///
/// 40 % is far outside small strain: the linear control below misses the lateral
/// contraction by 4.6 % of the stretch, which is five orders of magnitude above
/// [`EXACTNESS_BOUND`].
const STRETCH: f64 = 1.4;

/// What "reproduced exactly" means here, in mm.
///
/// Not the arithmetic floor. The uniform state is an exact solution of the
/// discrete equations, but it still has to be *found*, by a modified Newton
/// iteration whose steps are solved by a conjugate gradient that stops at a
/// relative residual of `2⁻³⁰`. Both residuals are printed on every row.
const EXACTNESS_BOUND: f64 = 2.0e-6;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("valid")
}

/// `(λ, μ)` of the enclosing solid, in f64, for building expectations.
fn lame_f64() -> (f64, f64) {
    let lambda = E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU));
    let mu = E_MPA / (2.0 * (1.0 + NU));
    (lambda, mu)
}

/// `κ = λ − 4C₂`, the volumetric modulus the solver derives from the material
/// for a model with constant `(W₁, W₂) = (C₁, C₂)`: it makes the stress
/// linearise to the enclosing solid's `λ`.
fn bulk_f64(c2: f64) -> f64 {
    let (lambda, _) = lame_f64();
    lambda - 4.0 * c2
}

fn linear_config() -> SolverConfig {
    SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
        .expect("valid")
}

fn hyperelastic_config(model: HyperelasticModel) -> CorotationalConfig {
    CorotationalConfig::try_new(linear_config(), 80, fx(1.0e-7), 4, 64)
        .expect("valid")
        .with_hyperelastic(model)
}

// ---------------------------------------------------------------------------
// the closed form
// ---------------------------------------------------------------------------

/// `(W₁, W₂)` for the models whose derivatives are constants.
///
/// Yeoh is deliberately absent: its `W₁` depends on `I₁`, so `σ_tt = 0` stops
/// being linear in `q` and the closed form below does not apply to it.
fn constant_energy_derivatives(model: &HyperelasticModel) -> (f64, f64) {
    match model {
        HyperelasticModel::NeoHookean { mu_mpa } => (mu_mpa.to_f64() / 2.0, 0.0),
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => (c1_mpa.to_f64(), c2_mpa.to_f64()),
        HyperelasticModel::Yeoh { .. } => {
            panic!("the closed form here is derived for constant (W1, W2); Yeoh has neither")
        }
    }
}

/// Lateral stretch `t` and axial Cauchy stress `σ_xx` of uniaxial tension at
/// axial stretch `s`, derived in the module doc.
fn uniaxial_closed_form(model: &HyperelasticModel, s: f64) -> (f64, f64) {
    let (c1, c2) = constant_energy_derivatives(model);
    let k = bulk_f64(c2);
    // (2C2 + k s²) q² + (2C1 + 2 s² C2 − k s) q − (2C1 + 4C2) = 0, positive root.
    let a = 2.0 * c2 + k * s * s;
    let b = 2.0 * c1 + 2.0 * s * s * c2 - k * s;
    let c = 2.0 * c1 + 4.0 * c2;
    let q = (-b + (b * b + 4.0 * a * c).sqrt()) / (2.0 * a);
    let t = q.sqrt();
    let j = s * q;
    let i1 = s * s + 2.0 * q;
    let p_ref = 2.0 * (c1 + 2.0 * c2);
    let sigma_xx =
        (2.0 / j) * ((c1 + i1 * c2) * s * s - c2 * s * s * s * s) + k * (j - 1.0) - p_ref / j;
    (t, sigma_xx)
}

/// The lateral stretch small-strain elasticity predicts for the same axial one.
///
/// What the control is expected to produce, and the number the hyperelastic
/// answer has to be distinguishable from.
fn linear_lateral_stretch(s: f64) -> f64 {
    1.0 - NU * (s - 1.0)
}

// ---------------------------------------------------------------------------
// mesh
// ---------------------------------------------------------------------------

fn node_index(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// Deterministic displacement in `[-1, 1]` for one lattice node and axis.
fn jitter_unit(i: usize, j: usize, k: usize, axis: usize) -> f64 {
    let mut h = 0x9E37_79B9_7F4A_7C15_u64;
    for v in [i as u64, j as u64, k as u64, axis as u64] {
        h ^= v.wrapping_add(0x9E37_79B9_7F4A_7C15);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 31;
    }
    ((h >> 32) as f64) / ((1u64 << 32) as f64) * 2.0 - 1.0
}

/// Kuhn 6-tet cube on `[0, n·h]³`, with the interior vertices displaced.
fn kuhn_cube(n: usize, h: f64, jitter: f64) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                let boundary = i == 0 || j == 0 || k == 0 || i == n || j == n || k == n;
                let mut p = [i as f64 * h, j as f64 * h, k as f64 * h];
                if !boundary && jitter != 0.0 {
                    for (axis, c) in p.iter_mut().enumerate() {
                        *c += jitter * h * jitter_unit(i, j, k, axis);
                    }
                }
                mesh.vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node_index(n, i, j, k);
                    for (m, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[m + 1] = node_index(n, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

/// Reference position of every node of the quadratic mesh, in f64.
fn node_positions(mesh: &QuadraticMesh) -> Vec<[f64; 3]> {
    (0..mesh.node_count())
        .map(|n| {
            let p = mesh
                .node_position(u32::try_from(n).expect("fits"))
                .expect("node in range");
            [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()]
        })
        .collect()
}

/// Is `value` on the plane `plane` to within half a micron?
fn on_plane(value: f64, plane: f64) -> bool {
    (value - plane).abs() < 5.0e-7
}

/// The uniaxial tension scene: the two `x` faces driven to stretch `s`, the `y`
/// and `z` planes through the origin pinned in their own normal direction, and
/// everything else free.
///
/// ⚠️ The pins on `y = 0` and `z = 0` are **consistent with the exact answer**
/// (`u_y = (t−1)·0 = 0` there), so they remove the three translations and the
/// three rotations without changing the solution. Pinning a node where the exact
/// displacement is non-zero would.
fn uniaxial_scene(positions: &[[f64; 3]], s: f64) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (n, p) in positions.iter().enumerate() {
        let node = u32::try_from(n).expect("fits");
        if on_plane(p[0], 0.0) {
            bc.prescribe(node, Axis::X, Fix128::ZERO);
        } else if on_plane(p[0], SIDE) {
            bc.prescribe(node, Axis::X, fx((s - 1.0) * SIDE));
        }
        if on_plane(p[1], 0.0) {
            bc.prescribe(node, Axis::Y, Fix128::ZERO);
        }
        if on_plane(p[2], 0.0) {
            bc.prescribe(node, Axis::Z, Fix128::ZERO);
        }
    }
    bc
}

/// Worst nodal deviation from the homogeneous map `u = ((s−1)X, (t−1)Y, (t−1)Z)`.
fn worst_deviation(positions: &[[f64; 3]], displacements: &[[Fix128; 3]], s: f64, t: f64) -> f64 {
    let mut worst = 0.0f64;
    for (p, d) in positions.iter().zip(displacements.iter()) {
        let expected = [(s - 1.0) * p[0], (t - 1.0) * p[1], (t - 1.0) * p[2]];
        for axis in 0..3 {
            let e = (d[axis].to_f64() - expected[axis]).abs();
            if e > worst {
                worst = e;
            }
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// the premise the oracles rest on
// ---------------------------------------------------------------------------

/// Every interior lattice node actually moved.
///
/// The jitter is the thing that stops the lattice from flattering the assembly,
/// and every validity check below it (topology, orientation, volume) passes at
/// `JITTER = 0` as well — so without this count, a perturbation that did nothing
/// would be reported as a valid perturbation.
#[test]
fn the_lattice_the_oracles_run_on_is_actually_perturbed() {
    let n = 2usize;
    let h = SIDE / n as f64;
    let plain = kuhn_cube(n, h, 0.0);
    let jittered = kuhn_cube(n, h, JITTER);
    assert_eq!(plain.vertices.len(), jittered.vertices.len());

    let mut moved = 0;
    let mut interior = 0;
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                let idx = node_index(n, i, j, k) as usize;
                if i == 0 || j == 0 || k == 0 || i == n || j == n || k == n {
                    assert_eq!(
                        plain.vertices[idx], jittered.vertices[idx],
                        "boundary vertex {idx} must not move"
                    );
                } else {
                    interior += 1;
                    if plain.vertices[idx] != jittered.vertices[idx] {
                        moved += 1;
                    }
                }
            }
        }
    }
    assert!(interior > 0, "the lattice has no interior vertices to move");
    assert_eq!(
        moved, interior,
        "{moved} of {interior} interior vertices moved; JITTER={JITTER} is not doing its job"
    );
}

/// The deformation gradient the element assembles from an **affine** nodal field
/// is exactly `I + A`, on the perturbed lattice.
///
/// `Σᵢ Xᵢ ⊗ ∇Nᵢ = I` is what makes `F = I + Σᵢ uᵢ ⊗ ∇Nᵢ` the deformation
/// gradient rather than something close to it, and it holds for P2 only because
/// the element reproduces a linear field — a property of the *element*, which is
/// what this measures. The probe is the fully prescribed path, which performs no
/// solve: the stress it reports is `cauchy_stress` read off the assembled `F`,
/// so comparing it against `cauchy_stress` evaluated at `I + A` in this file
/// isolates the assembly from the constitutive law.
///
/// ⚠️ `A` is **not symmetric and not small**: it carries a shear and a 10 %
/// stretch, so an assembly that silently symmetrised `F` or dropped the
/// identity would be caught.
#[test]
fn the_quadratic_gradient_of_an_affine_field_is_exact() {
    use alice_physics::math::{Mat3Fix, Vec3Fix};

    let n = 2usize;
    let h = SIDE / n as f64;
    let tets = kuhn_cube(n, h, JITTER);
    let mesh = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let positions = node_positions(&mesh);

    // `u = A·X` with A asymmetric.
    let a = [
        [0.10, 0.07, -0.03],
        [-0.05, 0.08, 0.02],
        [0.04, -0.06, 0.12],
    ];
    let mut bc = BoundaryConditions::new();
    for (idx, p) in positions.iter().enumerate() {
        let node = u32::try_from(idx).expect("fits");
        let u = [
            a[0][0] * p[0] + a[0][1] * p[1] + a[0][2] * p[2],
            a[1][0] * p[0] + a[1][1] * p[1] + a[1][2] * p[2],
            a[2][0] * p[0] + a[2][1] * p[1] + a[2][2] * p[2],
        ];
        bc.prescribe_all(node, [fx(u[0]), fx(u[1]), fx(u[2])]);
    }

    let model = HyperelasticModel::NeoHookean { mu_mpa: fx(3.0) };
    let solution =
        solve_quadratic_hyperelastic(&mesh, &material(), &bc, &hyperelastic_config(model))
            .expect("fully prescribed solve");

    // `F = I + A`, built here and fed to the public constitutive law.
    let f = Mat3Fix::from_cols(
        Vec3Fix::new(fx(1.0 + a[0][0]), fx(a[1][0]), fx(a[2][0])),
        Vec3Fix::new(fx(a[0][1]), fx(1.0 + a[1][1]), fx(a[2][1])),
        Vec3Fix::new(fx(a[0][2]), fx(a[1][2]), fx(1.0 + a[2][2])),
    );
    let (lambda, _) = lame_f64();
    let kappa =
        alice_physics::hyperelastic::volumetric_modulus(&model, fx(lambda)).expect("kappa >= 0");
    let expected =
        alice_physics::hyperelastic::cauchy_stress(&model, kappa, f).expect("positive det F");

    let mut worst = 0.0f64;
    for s in &solution.field.element_stress {
        let got = [
            s.xx.to_f64(),
            s.yy.to_f64(),
            s.zz.to_f64(),
            s.xy.to_f64(),
            s.yz.to_f64(),
            s.zx.to_f64(),
        ];
        let want = [
            expected.col0.x.to_f64(),
            expected.col1.y.to_f64(),
            expected.col2.z.to_f64(),
            expected.col1.x.to_f64(),
            expected.col2.y.to_f64(),
            expected.col2.x.to_f64(),
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            let e = (g - w).abs();
            if e > worst {
                worst = e;
            }
        }
    }
    println!(
        "affine F: worst component error {worst:.3e} MPa over {} elements",
        solution.field.element_stress.len()
    );
    assert!(
        worst < 1.0e-9,
        "the assembled F does not reproduce the affine field: worst {worst:.3e} MPa"
    );
}

// ---------------------------------------------------------------------------
// the oracles
// ---------------------------------------------------------------------------

/// Uniaxial Neo-Hookean tension matches the closed form derived in the module
/// doc, on a perturbed lattice, at 40 % stretch.
#[test]
fn neo_hookean_uniaxial_tension_matches_the_closed_form() {
    let (_, mu) = lame_f64();
    let model = HyperelasticModel::NeoHookean { mu_mpa: fx(mu) };
    let (t, sigma_xx) = uniaxial_closed_form(&model, STRETCH);

    let n = 2usize;
    let h = SIDE / n as f64;
    let tets = kuhn_cube(n, h, JITTER);
    let mesh = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let positions = node_positions(&mesh);
    let bc = uniaxial_scene(&positions, STRETCH);

    let solution =
        solve_quadratic_hyperelastic(&mesh, &material(), &bc, &hyperelastic_config(model))
            .expect("converged");

    let worst = worst_deviation(&positions, &solution.field.displacements, STRETCH, t);
    println!(
        "neo-hookean s={STRETCH} t={t:.9} worst {worst:.3e} mm  newton={} cg={} rel_resid={:.3e}",
        solution.newton_iterations,
        solution.field.iterations,
        solution.field.relative_residual.to_f64()
    );
    assert!(
        worst < EXACTNESS_BOUND,
        "uniaxial Neo-Hookean missed the closed form by {worst:.3e} mm (bound {EXACTNESS_BOUND:.1e})"
    );

    // The axial stress is the second half of the closed form, and it is not
    // implied by the displacement: a law with the right lateral contraction and
    // the wrong magnitude would pass the test above.
    let mut worst_axial = 0.0f64;
    let mut worst_lateral = 0.0f64;
    for s in &solution.field.element_stress {
        worst_axial = worst_axial.max((s.xx.to_f64() - sigma_xx).abs());
        worst_lateral = worst_lateral
            .max(s.yy.to_f64().abs())
            .max(s.zz.to_f64().abs());
    }
    println!("  sigma_xx expected {sigma_xx:.9} MPa, worst error {worst_axial:.3e}, worst lateral {worst_lateral:.3e}");
    assert!(
        worst_axial < 1.0e-5,
        "axial stress off the closed form by {worst_axial:.3e} MPa (expected {sigma_xx:.6})"
    );
    assert!(
        worst_lateral < 1.0e-5,
        "the lateral faces are not traction free: {worst_lateral:.3e} MPa"
    );
}

/// ⚠️ **The teeth.** The identical scene under the small-strain law misses the
/// closed form by orders of magnitude.
///
/// Without this, an implementation that never reached the hyperelastic branch —
/// a configuration ignored, a correction term dropped, a fall-back to
/// `solve_quadratic` — would pass every other test in this file, because the
/// uniform field they use is in the linear solver's reach too. What is *not* in
/// its reach is the **lateral contraction**: small strain gives `t = 1 − ν(s−1)`
/// and the material law gives something else entirely at 40 % stretch.
#[test]
fn uniaxial_tension_is_not_the_linear_answer() {
    let (_, mu) = lame_f64();
    let model = HyperelasticModel::NeoHookean { mu_mpa: fx(mu) };
    let (t, _) = uniaxial_closed_form(&model, STRETCH);
    let t_linear = linear_lateral_stretch(STRETCH);

    let n = 2usize;
    let h = SIDE / n as f64;
    let tets = kuhn_cube(n, h, JITTER);
    let mesh = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let positions = node_positions(&mesh);
    let bc = uniaxial_scene(&positions, STRETCH);

    let linear = solve_quadratic(&mesh, &material(), &bc, &linear_config()).expect("converged");
    let worst = worst_deviation(&positions, &linear.displacements, STRETCH, t);

    println!(
        "teeth: hyperelastic t={t:.9}, linear t={t_linear:.9}, linear misses the hyperelastic closed form by {worst:.3e} mm"
    );
    assert!(
        (t - t_linear).abs() > 1.0e-3,
        "the two laws predict the same lateral stretch at s={STRETCH}; this scene cannot tell them apart"
    );
    assert!(
        worst > 1000.0 * EXACTNESS_BOUND,
        "the small-strain law reproduced the hyperelastic closed form to {worst:.3e} mm; \
         the hyperelastic oracle above is not measuring the constitutive law"
    );

    // And the linear answer is the linear closed form, which says the scene is
    // set up the way the derivation assumes — the gap above is the law, not a
    // mis-stated boundary condition.
    let worst_linear = worst_deviation(&positions, &linear.displacements, STRETCH, t_linear);
    println!("  linear vs its own closed form: {worst_linear:.3e} mm");
    assert!(
        worst_linear < EXACTNESS_BOUND,
        "the linear control does not match small-strain uniaxial tension either ({worst_linear:.3e} mm); \
         the boundary conditions, not the law, are what differs"
    );
}

/// Mooney-Rivlin, whose `W₂` is non-zero, matches its own closed form.
///
/// Neo-Hookean never reaches the `B²` term of the constitutive law, so without
/// a second model that branch is integrated by nothing here. The closed form
/// changes with `C₂`, so the expectation moves too — a wiring that ignored the
/// model and always used Neo-Hookean would fail this and pass the first oracle.
#[test]
fn mooney_rivlin_uniaxial_tension_matches_the_closed_form() {
    let model = HyperelasticModel::MooneyRivlin {
        c1_mpa: fx(1.2),
        c2_mpa: fx(0.4),
    };
    let (t, sigma_xx) = uniaxial_closed_form(&model, STRETCH);
    let neo = HyperelasticModel::NeoHookean {
        mu_mpa: fx(2.0 * (1.2 + 0.4)),
    };
    let (t_neo, _) = uniaxial_closed_form(&neo, STRETCH);
    assert!(
        (t - t_neo).abs() > 1.0e-4,
        "C2 does not move the closed form here, so this is not a second branch"
    );

    let n = 2usize;
    let h = SIDE / n as f64;
    let tets = kuhn_cube(n, h, JITTER);
    let mesh = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let positions = node_positions(&mesh);
    let bc = uniaxial_scene(&positions, STRETCH);

    let solution =
        solve_quadratic_hyperelastic(&mesh, &material(), &bc, &hyperelastic_config(model))
            .expect("converged");
    let worst = worst_deviation(&positions, &solution.field.displacements, STRETCH, t);
    println!(
        "mooney-rivlin s={STRETCH} t={t:.9} (neo-hookean twin {t_neo:.9}) worst {worst:.3e} mm  newton={}",
        solution.newton_iterations
    );
    assert!(
        worst < EXACTNESS_BOUND,
        "uniaxial Mooney-Rivlin missed the closed form by {worst:.3e} mm"
    );

    let mut worst_axial = 0.0f64;
    for s in &solution.field.element_stress {
        worst_axial = worst_axial.max((s.xx.to_f64() - sigma_xx).abs());
    }
    println!("  sigma_xx expected {sigma_xx:.9} MPa, worst error {worst_axial:.3e}");
    assert!(
        worst_axial < 1.0e-5,
        "axial stress off the closed form by {worst_axial:.3e} MPa"
    );
}

/// Uniaxial **compression** at `s = 0.75`, where `J < 1` and the pressure term
/// changes sign.
///
/// The tension oracles leave `K(J−1)` positive throughout, so a sign error in
/// the volumetric completion would survive them.
#[test]
fn neo_hookean_uniaxial_compression_matches_the_closed_form() {
    const COMPRESSION: f64 = 0.75;
    let (_, mu) = lame_f64();
    let model = HyperelasticModel::NeoHookean { mu_mpa: fx(mu) };
    let (t, sigma_xx) = uniaxial_closed_form(&model, COMPRESSION);
    assert!(t > 1.0, "compression must widen the section, got t={t}");
    assert!(
        sigma_xx < 0.0,
        "compression must give a negative axial stress"
    );

    let n = 2usize;
    let h = SIDE / n as f64;
    let tets = kuhn_cube(n, h, JITTER);
    let mesh = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let positions = node_positions(&mesh);
    let bc = uniaxial_scene(&positions, COMPRESSION);

    let solution =
        solve_quadratic_hyperelastic(&mesh, &material(), &bc, &hyperelastic_config(model))
            .expect("converged");
    let worst = worst_deviation(&positions, &solution.field.displacements, COMPRESSION, t);
    println!(
        "neo-hookean s={COMPRESSION} t={t:.9} sigma_xx={sigma_xx:.9} worst {worst:.3e} mm  newton={}",
        solution.newton_iterations
    );
    assert!(
        worst < EXACTNESS_BOUND,
        "uniaxial compression missed the closed form by {worst:.3e} mm"
    );
}

/// ⚠️ **Objectivity, and the only oracle here whose `F` is not diagonal.**
///
/// Superposing a rigid rotation on the uniaxial solution — `F = R·diag(s,t,t)`
/// — leaves the lateral faces traction free, because the nominal traction
/// carries the rotation out in front: `P = J σ F⁻ᵀ` with `σ = R σ₀ Rᵀ` and
/// `F⁻ᵀ = R F₀⁻ᵀ` gives `P = R·P₀`, and `P₀·ŷ = 0` is exactly the `σ_tt = 0`
/// the closed form was solved for. So the stretch `t` is unchanged and the
/// Cauchy stress is `σ_xx·(Re₁)⊗(Re₁)`, both of which this checks.
///
/// ⚠️⚠️ **Why this test exists separately from the ones above.** Every other
/// oracle in this file drives a deformation whose `F` is **diagonal**, where
/// `F⁻¹ = F⁻ᵀ` — so dropping the transpose in `P = J σ F⁻ᵀ` changes nothing and
/// the whole file stays green. That is measured: the mutation survived the
/// first six oracles and is caught only here. It is the same shape as the
/// `F⁻ᵀ` omission that once passed every test in this crate
/// reproduced here and then closed.
///
/// ⚠️ `cos` and `sin` of the angle are **not binary exact**, so the prescribed
/// data carries a rounding of its own — the test asserts `cos² + sin² = 1` to
/// the size of that rounding rather than claiming it away.
///
/// ⚠️⚠️ **The angle is bounded by the iteration, not by the physics.** The
/// approximate tangent this solver steps with is the **small-strain** operator,
/// which carries no rotation — unlike `solve_corotational`, whose tangent is
/// `Σₑ R Kₑ⁰ Rᵀ`. The stress is objective either way (it is `cauchy_stress(F)`,
/// a function of `F` alone), but the *iteration* stops contracting once the
/// superposed rotation is large. Measured on this scene, sweeping the angle:
///
/// ```text
/// small-strain tangent      co-rotational tangent (what ships)
///   0.00°  OK 1.847e-7 mm      —
///   2.00°  OK 8.432e-8 mm      —
///   5.00°  NotConverged,       OK 1.350e-7 mm, 414 Newton steps
///          residual 11.33x
///  10.00°  —                   OK 1.134e-7 mm, 429 Newton steps
///  20.00°  —                   OK 8.192e-8 mm, 457 Newton steps
///  36.87°  RotationFailed      not measured
///          { cause: Inverted }
/// ```
///
/// ⚠️ **With the small-strain tangent, increments did not help**: at 5° the
/// final residual was bit identical at four increments and at sixteen, so what
/// failed was the contraction of the iteration map and not the size of the
/// step. That is what the co-rotational tangent fixed — and it is a different
/// failure from a budget shortfall, which increments *do* fix.
///
/// ⚠️ **Above 20° is unmeasured, not known to fail.** The sweep serialises with
/// every other build in the tree and was stopped there; 20° is already four
/// times the angle at which the small-strain tangent gave up. If a scene ever
/// needs more, the next lever is a polar decomposition per quadrature point
/// rather than per element.
///
/// ⚠️ The Newton budget here is **200 per increment**, not the 80 the diagonal
/// oracles use: at sixteen increments the measured total is 386 steps and the
/// work is not spread evenly across them, so 80 per increment refuses with
/// `NotConverged` at a residual 29× the target. Measured, not guessed.
///
/// ⚠️ At 36.87° the first increment's opening small-strain solve puts an
/// element through `det F = 0` outright (`RotationFailed { tet: 0, cause:
/// Inverted }`), which is the same limitation seen from the other side.
fn rotated_uniaxial_oracle(config_for: impl Fn(HyperelasticModel) -> CorotationalConfig) {
    // ⚠️ **16.26°, and the angle is load bearing.** It sits between 10° and 20°,
    // both of which were measured to converge (see the sweep above), and it is
    // far enough from diagonal that `F⁻¹` and `F⁻ᵀ` are different matrices —
    // which is the whole reason this scene exists.
    // ⚠️ **A Pythagorean rotation, not `cos`/`sin` of an angle.** `clippy.toml`
    // disallows `f64::sin` / `f64::cos` crate-wide — including in tests — because
    // the platform libm is not bit-exact across targets, and a test that seeds a
    // scene from them is not reproducible either. 7-24-25 gives
    // `cos = 24/25`, `sin = 7/25` as exact rationals with no transcendental
    // call: an angle of 16.26°, which the sweep below brackets on both sides.
    const COS: f64 = 24.0 / 25.0;
    const SIN: f64 = 7.0 / 25.0;
    const DEGREES: f64 = 16.26;
    let (_, mu) = lame_f64();
    let model = HyperelasticModel::NeoHookean { mu_mpa: fx(mu) };
    let (t, sigma_axial) = uniaxial_closed_form(&model, STRETCH);

    // `F = R·diag(s, t, t)`, by rows.
    let f = [
        [COS * STRETCH, -SIN * t, 0.0],
        [SIN * STRETCH, COS * t, 0.0],
        [0.0, 0.0, t],
    ];
    assert!(
        (f[0][1] - f[1][0]).abs() > 1.0e-3,
        "F must be non-symmetric for this oracle to say anything about F transpose"
    );
    assert!(
        (COS * COS + SIN * SIN - 1.0).abs() < 1.0e-15,
        "R is not orthogonal, so the superposed motion is not rigid"
    );

    let n = 2usize;
    let h = SIDE / n as f64;
    let tets = kuhn_cube(n, h, JITTER);
    let mesh = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let positions = node_positions(&mesh);

    let expected = |p: &[f64; 3]| {
        [
            (f[0][0] - 1.0) * p[0] + f[0][1] * p[1] + f[0][2] * p[2],
            f[1][0] * p[0] + (f[1][1] - 1.0) * p[1] + f[1][2] * p[2],
            f[2][0] * p[0] + f[2][1] * p[1] + (f[2][2] - 1.0) * p[2],
        ]
    };

    // Both `x` faces fully prescribed to the rotated map; the `y` and `z` faces
    // stay free, which is where the traction-free condition has to hold.
    let mut bc = BoundaryConditions::new();
    for (idx, p) in positions.iter().enumerate() {
        if !(on_plane(p[0], 0.0) || on_plane(p[0], SIDE)) {
            continue;
        }
        let u = expected(p);
        bc.prescribe_all(
            u32::try_from(idx).expect("fits"),
            [fx(u[0]), fx(u[1]), fx(u[2])],
        );
    }

    let config = config_for(model);
    let solution =
        solve_quadratic_hyperelastic(&mesh, &material(), &bc, &config).expect("converged");

    let mut worst = 0.0f64;
    for (p, d) in positions.iter().zip(solution.field.displacements.iter()) {
        let want = expected(p);
        for axis in 0..3 {
            worst = worst.max((d[axis].to_f64() - want[axis]).abs());
        }
    }
    println!(
        "rotated uniaxial: {DEGREES}deg (cos={COS}, sin={SIN}) t={t:.9} worst {worst:.3e} mm  newton={}",
        solution.newton_iterations
    );
    assert!(
        worst < EXACTNESS_BOUND,
        "the superposed rotation moved the answer by {worst:.3e} mm; \
         either the lateral faces are not traction free under it or F transpose is wrong"
    );

    // `σ = σ_axial·(Re₁)⊗(Re₁)` — six components, three of them non-zero and
    // one of them off-diagonal, so a transposed or symmetrised stress shows up.
    let axis1 = [COS, SIN, 0.0];
    let want = [
        sigma_axial * axis1[0] * axis1[0],
        sigma_axial * axis1[1] * axis1[1],
        0.0,
        sigma_axial * axis1[0] * axis1[1],
        0.0,
        0.0,
    ];
    let mut worst_stress = 0.0f64;
    for s in &solution.field.element_stress {
        let got = [
            s.xx.to_f64(),
            s.yy.to_f64(),
            s.zz.to_f64(),
            s.xy.to_f64(),
            s.yz.to_f64(),
            s.zx.to_f64(),
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            worst_stress = worst_stress.max((g - w).abs());
        }
    }
    println!("  rotated sigma expected {want:?}, worst error {worst_stress:.3e} MPa");
    assert!(
        worst_stress < 1.0e-5,
        "the rotated Cauchy stress is off by {worst_stress:.3e} MPa"
    );
}

/// The rotated oracle on the iteration that ships by default: the
/// co-rotational tangent over sixteen increments (about 450 Newton steps).
///
/// ⚠️ This is the only scene that drives the default iteration through a
/// superposed rotation, so it is what notices when that iteration stops
/// converging under one — a property of the *path*, which the fast twin below
/// does not exercise.
#[test]
#[ignore = "runtime: about 165 s in debug (P2, n = 2, default co-rotational tangent, 16 increments, about 450 Newton steps); the per-push twin is uniaxial_tension_under_a_superposed_rotation_with_the_consistent_tangent; run by run_ignored.py"]
fn uniaxial_tension_under_a_superposed_rotation_is_the_rotated_closed_form() {
    rotated_uniaxial_oracle(|model| {
        CorotationalConfig::try_new(linear_config(), 600, fx(1.0e-7), 16, 64)
            .expect("valid")
            .with_hyperelastic(model)
    });
}

/// The same scene, the same assertions and the same bounds, reached by the
/// consistent tangent in one increment (4 Newton steps).
///
/// ⚠️ **Why it checks the same thing.** What the oracle asserts is the
/// converged state, the fixed point of `f_mat(u) = f_ext`; the residual and the
/// bounds are shared, only the tangent and the load path differ, and the
/// hyperelastic law has no path dependence. Measured: the worst displacement and
/// stress errors match the sixteen-increment run to the printed digits.
///
/// ⚠️ **What it does not check**: that the *default* iteration converges under a
/// rotation. That stays with the ignored test above, in the weekly release run.
/// Dropping `.transpose()` from `P = J σ F⁻ᵀ` turns this twin red (as
/// `NotConverged` after three steps rather than on the value assertion).
#[test]
fn uniaxial_tension_under_a_superposed_rotation_with_the_consistent_tangent() {
    rotated_uniaxial_oracle(|model| {
        CorotationalConfig::try_new(linear_config(), 600, fx(1.0e-7), 1, 64)
            .expect("valid")
            .with_hyperelastic(model)
            .with_consistent_tangent()
    });
}

/// The configuration must carry a law; the entry point does not fall back.
///
/// A silent fall-back to the linear solver would make every oracle above pass
/// against the wrong physics, which is the failure the teeth test exists for —
/// this closes the same hole at the API.
#[test]
fn a_configuration_without_a_law_is_refused() {
    use alice_physics::linear_elastic_fem::FemError;

    let n = 1usize;
    let h = SIDE / n as f64;
    let tets = kuhn_cube(n, h, 0.0);
    let mesh = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let positions = node_positions(&mesh);
    let bc = uniaxial_scene(&positions, STRETCH);
    let config =
        CorotationalConfig::try_new(linear_config(), 80, fx(1.0e-7), 4, 64).expect("valid");

    match solve_quadratic_hyperelastic(&mesh, &material(), &bc, &config) {
        Err(FemError::InvalidConfig(_)) => {}
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

/// A model whose small-strain `λ` offset exceeds the material's `λ` would need a
/// negative volumetric modulus (`κ = λ − 8C₂ < 0` for Yeoh); the solve refuses it
/// up front instead of running a law whose volumetric energy is not convex.
#[test]
fn a_negative_volumetric_modulus_is_refused() {
    let (lambda, _) = lame_f64();
    let model = HyperelasticModel::Yeoh {
        c1_mpa: fx(1.0),
        c2_mpa: fx(lambda / 8.0 + 0.5),
        c3_mpa: fx(0.0),
    };
    let n = 2usize;
    let h = SIDE / n as f64;
    let tets = kuhn_cube(n, h, JITTER);
    let mesh = QuadraticMesh::from_tet_mesh(&tets).expect("well formed");
    let positions = node_positions(&mesh);
    let bc = uniaxial_scene(&positions, STRETCH);
    let refused =
        solve_quadratic_hyperelastic(&mesh, &material(), &bc, &hyperelastic_config(model));
    assert!(
        matches!(
            refused,
            Err(alice_physics::linear_elastic_fem::FemError::InvalidConfig(
                _
            ))
        ),
        "kappa < 0 must be refused: {refused:?}"
    );
}
