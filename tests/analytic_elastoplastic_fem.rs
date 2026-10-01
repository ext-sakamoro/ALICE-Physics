//! Closed-form oracles for `linear_elastic_fem::solve_elastoplastic`.
//!
//! Small-strain J2 (von Mises) plasticity with bilinear isotropic hardening on
//! P1 tetrahedra. Every expected value is written from the equations of the
//! model, never from running the solver.
//!
//! # Notation
//!
//! `E` Young's modulus, `G = E / 2(1+ν)`, `σ_y` initial yield stress, `H` the
//! plastic modulus `dσ_y/dε̄_p` (so the uniaxial tangent past yield is
//! `E_t = E·H / (E + H)`), `ε̄_p` the equivalent plastic strain.
//!
//! # Why these scenes
//!
//! P1 tetrahedra represent a linear displacement field exactly, and J2 with
//! isotropic hardening integrated by radial return is exact whenever the
//! stress direction is fixed during the step. Uniaxial tension and simple
//! shear both keep a fixed stress direction, so the discrete answer equals the
//! closed form up to solver tolerance, for any step size. A failure is a bug in
//! the return mapping, the tangent or the assembly, never "the mesh was too
//! coarse".
//!
//! The numbers are dyadic where the scene allows it: `E = 1024`, `ν = 1/4`,
//! `σ_y = 2`, `H = 1024` give `ε_y = 2⁻⁹` and `E_t = 512` exactly.
//!
//! # Tolerances
//!
//! The Newton and conjugate gradient residuals are `2⁻⁴⁰` and `2⁻³⁰` relative,
//! with an absolute floor of `2⁻³⁰` N (about `1e-9`). Each Newton iteration
//! gains the conjugate gradient's `2⁻³⁰`, so the loop ends with a residual of
//! order the floor. On this stiffness (about `E·h = 2000` N/mm) that bounds the
//! displacement error near `1e-12` mm and the stress error near `1e-9` MPa
//! (measured: a Newton tolerance of `2⁻²⁰` instead left `1.2e-6` MPa on the
//! lateral stress, because the stopping rule is relative to the first
//! residual). Stress comparisons use `1e-6` MPa and strains `1e-8`, three
//! orders of margin over the floor; the one scene with a stress of `4e6` MPa
//! uses a relative `1e-7`.
//!
//! # Failure policy under test
//!
//! Invalid input returns `Err` with the variant stated in each test; nothing
//! panics. A guard whose removal leaves every test green is a guard without a
//! test, so each one has its own case below.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::linear_elastic_fem::{
    solve, solve_elastoplastic, Axis, BoundaryConditions, ElasticMaterial, ElastoplasticConfig,
    ElastoplasticSolution, FemError, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// scene construction
// ---------------------------------------------------------------------------

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

/// Kuhn 6-tet subdivision of `[0,nx·h] × [0,ny·h] × [0,nz·h]` (conforming).
fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f32) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
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
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node_index(nx, ny, i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node_index(nx, ny, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// `n / 2^k`, exact in `Fix128`.
fn dy(n: i64, k: u32) -> Fix128 {
    if k == 0 {
        return Fix128::from_int(n);
    }
    Fix128::from_raw(0, 1u64 << (64 - k)) * Fix128::from_int(n)
}

const E_MPA: f64 = 1024.0;
const NU: f64 = 0.25;
const SIGMA_Y: f64 = 2.0;
const H_PLASTIC: f64 = 1024.0;
/// `E·H / (E+H)`, exact for the dyadic inputs.
const E_TANGENT: f64 = 512.0;
/// Yield strain `σ_y / E = 2⁻⁹`.
const EPS_Y: f64 = 1.0 / 512.0;

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and ν in (-1, 0.5)")
}

fn linear() -> SolverConfig {
    SolverConfig::default()
}

fn newton_tol() -> Fix128 {
    Fix128::from_raw(0, 1 << 24) // 2^-40
}

fn config(sigma_y: f64, h: f64) -> ElastoplasticConfig {
    ElastoplasticConfig::try_new(linear(), 60, newton_tol(), fx(sigma_y), fx(h))
        .expect("valid elastoplastic config")
}

fn close(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() <= tol,
        "{what}: got {g:.12e}, closed form {want:.12e}, difference {:.3e} > tol {tol:.3e}",
        (g - want).abs()
    );
}

/// Uniaxial bar `[0,2h] × [0,h] × [0,h]`, `h = 2`, so `L = 4`, driven in
/// displacement: `u_x = 0` on the `x = 0` face, `u_x = ε_ref · L` on the `x = L`
/// face at load factor 1, lateral faces free. Three more constraints remove the
/// lateral rigid modes (rotation about x, and the y / z translations).
const BAR_L: f64 = 4.0;

fn bar_mesh() -> SdfTetMesh {
    kuhn_box(2, 1, 1, 2.0)
}

fn bar_displacement_bc(eps_ref: Fix128) -> BoundaryConditions {
    let (nx, ny, nz) = (2usize, 1usize, 1usize);
    let mut bc = BoundaryConditions::new();
    let length = Fix128::from_int(4);
    for k in 0..=nz {
        for j in 0..=ny {
            bc.prescribe(node_index(nx, ny, 0, j, k), Axis::X, Fix128::ZERO);
            bc.prescribe(node_index(nx, ny, nx, j, k), Axis::X, eps_ref * length);
        }
    }
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, ny, 0), Axis::Z, Fix128::ZERO);
    bc
}

/// The same bar loaded by a uniform end traction `σ` (load factor 1), as the
/// consistent nodal forces of the two face triangles: the diagonal pair of
/// corners takes `A/3` each, the other pair `A/6` each.
fn bar_traction_bc(sigma: f64) -> BoundaryConditions {
    let (nx, ny, nz) = (2usize, 1usize, 1usize);
    let mut bc = BoundaryConditions::new();
    for k in 0..=nz {
        for j in 0..=ny {
            bc.prescribe(node_index(nx, ny, 0, j, k), Axis::X, Fix128::ZERO);
        }
    }
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, ny, 0), Axis::Z, Fix128::ZERO);
    let area = 4.0_f64; // h * h
    for (node, share) in [
        (node_index(nx, ny, nx, 0, 0), 1.0 / 3.0),
        (node_index(nx, ny, nx, ny, nz), 1.0 / 3.0),
        (node_index(nx, ny, nx, ny, 0), 1.0 / 6.0),
        (node_index(nx, ny, nx, 0, nz), 1.0 / 6.0),
    ] {
        bc.add_load(node, Axis::X, fx(sigma * area * share));
    }
    bc
}

/// Strain path `ε_ref = 2⁻⁷` (so `δ = 2⁻⁵` at factor 1) and its factors in
/// eighths: factor `k/8` is the strain `k · 2⁻¹⁰`.
fn eps_ref() -> Fix128 {
    dy(1, 7)
}

fn eighths(ks: &[i64]) -> Vec<Fix128> {
    ks.iter().map(|&k| dy(k, 3)).collect()
}

/// Closed-form bilinear response for a monotone strain.
fn bilinear_stress(eps: f64) -> f64 {
    if eps <= EPS_Y {
        E_MPA * eps
    } else {
        SIGMA_Y + E_TANGENT * (eps - EPS_Y)
    }
}

fn run(
    mesh: &SdfTetMesh,
    bc: &BoundaryConditions,
    cfg: &ElastoplasticConfig,
    path: &[Fix128],
) -> ElastoplasticSolution {
    solve_elastoplastic(mesh, &material(), bc, cfg, path).expect("elastoplastic solve succeeds")
}

fn assert_uniaxial(sol: &ElastoplasticSolution, sigma: f64, what: &str) {
    for (i, s) in sol.field.element_stress.iter().enumerate() {
        close(s.xx, sigma, 1e-6, &format!("{what}: element {i} σ_xx"));
        for (name, v) in [
            ("σ_yy", s.yy),
            ("σ_zz", s.zz),
            ("σ_xy", s.xy),
            ("σ_yz", s.yz),
            ("σ_zx", s.zx),
        ] {
            close(v, 0.0, 1e-6, &format!("{what}: element {i} {name}"));
        }
    }
}

// ---------------------------------------------------------------------------
// oracle 1: bilinear isotropic hardening, uniaxial tension
// ---------------------------------------------------------------------------

/// oracle: bilinear isotropic hardening closed form.
///
/// `σ = Eε` for `ε ≤ ε_y`, `σ = σ_y + E_t (ε − ε_y)` past it, with the plastic
/// strain `ε_p = ε − σ/E` and `ε̄_p = ε_p` (uniaxial). Strain targets are
/// `k · 2⁻¹⁰` for `k = 1, 2, 3, 4, 8`, covering below, at, and past the yield
/// strain `2⁻⁹ = 2·2⁻¹⁰`. Each is reached by a **single** load step and by
/// eight, because radial return is exact in the step size for a fixed stress
/// direction: a step-size dependence would be an integration error.
#[test]
fn uniaxial_tension_follows_the_bilinear_curve() {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let cfg = config(SIGMA_Y, H_PLASTIC);
    for k in [1i64, 2, 3, 4, 8] {
        let eps = (k as f64) / 1024.0;
        let want = bilinear_stress(eps);
        let eps_p = (eps - want / E_MPA).max(0.0);
        let single = run(&mesh, &bc, &cfg, &eighths(&[k]));
        assert_uniaxial(&single, want, &format!("k={k} one step"));
        let ramp: Vec<Fix128> = (1..=8).map(|n| dy(k * n, 6)).collect();
        let many = run(&mesh, &bc, &cfg, &ramp);
        assert_uniaxial(&many, want, &format!("k={k} eight steps"));
        for (i, a) in many.equivalent_plastic_strain.iter().enumerate() {
            close(*a, eps_p, 1e-9, &format!("k={k} element {i} ε̄_p"));
        }
        for (i, p) in many.plastic_strain.iter().enumerate() {
            close(p.xx, eps_p, 1e-9, &format!("k={k} element {i} ε_p,xx"));
            close(
                p.yy,
                -0.5 * eps_p,
                1e-9,
                &format!("k={k} element {i} ε_p,yy"),
            );
            close(
                p.zz,
                -0.5 * eps_p,
                1e-9,
                &format!("k={k} element {i} ε_p,zz"),
            );
        }
        // the lateral contraction: elastic -ν σ/E plus plastic -ε_p/2
        let lateral = -NU * want / E_MPA - 0.5 * eps_p;
        let y_top = many.field.displacements[node_index(2, 1, 2, 1, 0) as usize][1];
        // node (2h,h,0) sits at y = h = 2 relative to the y = 0 pin
        close(y_top, lateral * 2.0, 1e-9, &format!("k={k} lateral u_y"));
    }
}

/// oracle: the same curve under load control, traction `σ = 5` gives
/// `ε = ε_y + (5 − 2)/E_t = 2⁻⁷`, so the end displacement is `4 · 2⁻⁷ = 2⁻⁵`.
#[test]
fn uniaxial_tension_under_traction_lands_on_the_bilinear_strain() {
    let mesh = bar_mesh();
    let cfg = config(SIGMA_Y, H_PLASTIC);
    let bc = bar_traction_bc(5.0);
    let path: Vec<Fix128> = (1..=5).map(|n| dy(n, 0) / Fix128::from_int(5)).collect();
    let sol = run(&mesh, &bc, &cfg, &path);
    assert_uniaxial(&sol, 5.0, "traction 5");
    let tip = sol.field.displacements[node_index(2, 1, 2, 0, 0) as usize][0];
    close(tip, BAR_L / 128.0, 1e-8, "end displacement");
    // the load factor scales the load: 10 MPa at factor 1/2 is 5 MPa
    let half_way = run(&mesh, &bar_traction_bc(10.0), &cfg, &[dy(1, 1)]);
    assert_uniaxial(&half_way, 5.0, "traction 10 at factor 1/2");
}

// ---------------------------------------------------------------------------
// oracle 2: elastic unloading, residual strain
// ---------------------------------------------------------------------------

/// oracle: elastic unloading with slope `E` and a closed-form residual strain.
///
/// Load to `ε = 8·2⁻¹⁰` (`σ = 5`), unload: `σ = 5 − E (ε_max − ε)`. The stress
/// is zero at `ε = ε_max − 5/E = 3·2⁻¹⁰`, which is the residual strain
/// `ε_p = 3·2⁻¹⁰`. Going on to `ε = 0` gives `σ = −E ε_p = −3`, still inside the
/// hardened yield `σ_y + H ε̄_p = 5` so it is elastic and `ε_p` must not move.
#[test]
fn unloading_is_elastic_and_leaves_the_residual_strain() {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let cfg = config(SIGMA_Y, H_PLASTIC);
    // factors in eighths of 2^-7: k/8 → strain k·2^-10
    let load: Vec<i64> = vec![2, 4, 6, 8];
    let cases: [(i64, f64); 4] = [
        (6, 5.0 - E_MPA * 2.0 / 1024.0), // ε = 6·2^-10: σ = 5 − 2 = 3
        (4, 5.0 - E_MPA * 4.0 / 1024.0), // σ = 1
        (3, 0.0),                        // σ = 0, residual strain reached
        (0, -3.0),                       // σ = −E ε_p
    ];
    for (k, sigma) in cases {
        let mut ks = load.clone();
        ks.push(k);
        let sol = run(&mesh, &bc, &cfg, &eighths(&ks));
        assert_uniaxial(&sol, sigma, &format!("unloaded to k={k}"));
        for (i, a) in sol.equivalent_plastic_strain.iter().enumerate() {
            close(
                *a,
                3.0 / 1024.0,
                1e-9,
                &format!("k={k} element {i} ε̄_p held"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// oracle 3: perfect plasticity
// ---------------------------------------------------------------------------

/// oracle: with `H = 0` the stress cannot exceed `σ_y`.
///
/// Displacement control to `4 ε_y`: `σ = σ_y` exactly and the plastic strain
/// takes up the rest, `ε_p = ε − ε_y`.
#[test]
fn perfect_plasticity_caps_the_stress_at_yield() {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let cfg = config(SIGMA_Y, 0.0);
    // 2^-7 = 4 ε_y at factor 1
    let sol = run(&mesh, &bc, &cfg, &eighths(&[4, 8]));
    assert_uniaxial(&sol, SIGMA_Y, "H = 0 at 4 ε_y");
    for (i, a) in sol.equivalent_plastic_strain.iter().enumerate() {
        close(*a, 3.0 * EPS_Y, 1e-8, &format!("element {i} ε̄_p"));
    }
}

/// oracle: under load control the limit load is `σ_y · A`. Below it the solve
/// succeeds and is elastic; above it there is no equilibrium state and the
/// solve returns `Err` rather than a stress over `σ_y`.
///
/// `1.5` is below the limit and `2.5` above it. Exactly `2.0` is not tried:
/// whether it counts as elastic is a rounding question, not an oracle.
#[test]
fn perfect_plasticity_has_a_limit_load() {
    let mesh = bar_mesh();
    let cfg = config(SIGMA_Y, 0.0);
    let ok = run(&mesh, &bar_traction_bc(1.5), &cfg, &[Fix128::ONE]);
    assert_uniaxial(&ok, 1.5, "below the limit load");
    let over = solve_elastoplastic(
        &mesh,
        &material(),
        &bar_traction_bc(2.5),
        &cfg,
        &[Fix128::ONE],
    );
    assert!(
        over.is_err(),
        "a load of 2.5 MPa on a perfectly plastic σ_y = 2 bar has no equilibrium state, got {:?}",
        over.map(|s| s.field.element_stress[0])
    );
}

// ---------------------------------------------------------------------------
// oracle 4: the plastic strain is deviatoric
// ---------------------------------------------------------------------------

/// oracle: `tr ε_p = 0` (J2 flow is along the deviator), in uniaxial tension and
/// in simple shear, for every element.
#[test]
fn plastic_strain_is_traceless() {
    let mesh = bar_mesh();
    let cfg = config(SIGMA_Y, H_PLASTIC);
    let sol = run(
        &mesh,
        &bar_displacement_bc(eps_ref()),
        &cfg,
        &eighths(&[4, 8]),
    );
    assert!(sol.equivalent_plastic_strain[0].to_f64() > 1e-4);
    for p in &sol.plastic_strain {
        close(p.xx + p.yy + p.zz, 0.0, 1e-12, "tr ε_p (uniaxial)");
    }
    let (mesh, bc) = shear_scene(0.01);
    let sol = run(&mesh, &bc, &cfg, &[Fix128::ONE]);
    assert!(sol.equivalent_plastic_strain[0].to_f64() > 1e-4);
    for p in &sol.plastic_strain {
        close(p.xx + p.yy + p.zz, 0.0, 1e-12, "tr ε_p (shear)");
    }
}

// ---------------------------------------------------------------------------
// simple shear and hydrostatic compression: the parts uniaxial cannot see
// ---------------------------------------------------------------------------

/// 2×2×2 cells, `h = 2`, boundary nodes carrying `u = (γ y, 0, 0)`, the single
/// interior node free. The field is linear, so the interior node must land on
/// it and every element sees the same simple shear.
fn shear_scene(gamma: f64) -> (SdfTetMesh, BoundaryConditions) {
    let n = 2usize;
    let mesh = kuhn_box(n, n, n, 2.0);
    let mut bc = BoundaryConditions::new();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                if i == 1 && j == 1 && k == 1 {
                    continue;
                }
                let v = node_index(n, n, i, j, k);
                bc.prescribe_all(
                    v,
                    [fx(gamma * (2.0 * j as f64)), Fix128::ZERO, Fix128::ZERO],
                );
            }
        }
    }
    (mesh, bc)
}

/// oracle: simple shear `γ`. With `τ = σ_xy`, `q = √3 τ`, `ε̄_p = γ_p/√3` and
/// `γ = τ/G + γ_p`:
///
/// `τ = G γ` while `√3 τ ≤ σ_y`, and `τ = (√3 σ_y + H γ) / (3 + H/G)` after.
///
/// This is the oracle that separates the engineering shear `γ = 2ε_xy` from the
/// tensor shear in the stress, the flow and the tangent: a factor of two there
/// moves `τ` and is invisible to uniaxial tension.
#[test]
fn simple_shear_follows_the_j2_closed_form() {
    let g = E_MPA / (2.0 * (1.0 + NU));
    let cfg = config(SIGMA_Y, H_PLASTIC);
    for gamma in [0.001_f64, 0.01, 0.05] {
        let tau_elastic = g * gamma;
        let tau = if 3.0_f64.sqrt() * tau_elastic <= SIGMA_Y {
            tau_elastic
        } else {
            (3.0_f64.sqrt() * SIGMA_Y + H_PLASTIC * gamma) / (3.0 + H_PLASTIC / g)
        };
        let gamma_p = gamma - tau / g;
        for steps in [1usize, 4] {
            let (mesh, bc) = shear_scene(gamma);
            let path: Vec<Fix128> = (1..=steps)
                .map(|n| dy(n as i64, 0) / Fix128::from_int(steps as i64))
                .collect();
            let sol = run(&mesh, &bc, &cfg, &path);
            for (i, s) in sol.field.element_stress.iter().enumerate() {
                let what = format!("γ={gamma} steps={steps} element {i}");
                close(s.xy, tau, 1e-6, &format!("{what} τ"));
                for (name, v) in [
                    ("xx", s.xx),
                    ("yy", s.yy),
                    ("zz", s.zz),
                    ("yz", s.yz),
                    ("zx", s.zx),
                ] {
                    close(v, 0.0, 1e-6, &format!("{what} σ_{name}"));
                }
            }
            for (i, a) in sol.equivalent_plastic_strain.iter().enumerate() {
                close(
                    *a,
                    gamma_p / 3.0_f64.sqrt(),
                    1e-8,
                    &format!("γ={gamma} steps={steps} element {i} ε̄_p"),
                );
            }
            for p in &sol.plastic_strain {
                close(
                    p.xy,
                    0.5 * gamma_p,
                    1e-8,
                    "ε_p,xy is the tensor shear γ_p/2",
                );
            }
        }
    }
}

/// oracle: hydrostatic compression never yields. `u = −c·x` on the boundary
/// gives `σ = 3K·(−c)·I` with `K = E / 3(1−2ν)`; here `σ_ii ≈ −20`, ten times
/// `σ_y`, and the deviator is zero so `ε_p = 0` and `ε̄_p = 0`.
#[test]
fn hydrostatic_compression_is_purely_elastic() {
    let n = 2usize;
    let mesh = kuhn_box(n, n, n, 2.0);
    let c = 0.01_f64;
    let mut bc = BoundaryConditions::new();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                if i == 1 && j == 1 && k == 1 {
                    continue;
                }
                bc.prescribe_all(
                    node_index(n, n, i, j, k),
                    [
                        fx(-c * 2.0 * i as f64),
                        fx(-c * 2.0 * j as f64),
                        fx(-c * 2.0 * k as f64),
                    ],
                );
            }
        }
    }
    let sol = run(&mesh, &bc, &config(SIGMA_Y, H_PLASTIC), &[Fix128::ONE]);
    let bulk = E_MPA / (3.0 * (1.0 - 2.0 * NU));
    for (i, s) in sol.field.element_stress.iter().enumerate() {
        for (name, v) in [("xx", s.xx), ("yy", s.yy), ("zz", s.zz)] {
            close(v, -3.0 * bulk * c, 1e-6, &format!("element {i} σ_{name}"));
        }
        close(s.xy, 0.0, 1e-6, "σ_xy");
    }
    for a in &sol.equivalent_plastic_strain {
        assert!(
            a.is_zero(),
            "hydrostatic loading cannot produce plastic strain"
        );
    }
    for p in &sol.plastic_strain {
        assert_eq!(p.xx, Fix128::ZERO);
    }
}

// ---------------------------------------------------------------------------
// oracle 5: before yield it is the linear elastic solver, bit for bit
// ---------------------------------------------------------------------------

/// oracle: a load that never reaches yield returns exactly what
/// [`solve`] returns, displacements and stresses, bit for bit (regression
/// against the elastic path).
///
/// The comparison is `assert_eq!` on `Fix128`, not a tolerance: the elastic
/// elements go through the same element routines as `solve`, the first Newton
/// iteration solves the same system, and the second never runs because the
/// first already meets the Newton tolerance.
#[test]
fn below_yield_equals_the_linear_elastic_solver_bit_for_bit() {
    let mesh = bar_mesh();
    let cfg = config(SIGMA_Y, H_PLASTIC);
    for bc in [bar_traction_bc(1.0), bar_displacement_bc(dy(1, 10))] {
        let reference = solve(&mesh, &material(), &bc, &linear()).expect("linear solve");
        let sol = run(&mesh, &bc, &cfg, &[Fix128::ONE]);
        assert_eq!(sol.field.displacements, reference.displacements);
        assert_eq!(sol.field.element_stress, reference.element_stress);
        assert!(sol.equivalent_plastic_strain.iter().all(|a| a.is_zero()));
        assert_eq!(sol.newton_iterations, 1);
    }
}

// ---------------------------------------------------------------------------
// oracle 6: numbering and element order do not matter
// ---------------------------------------------------------------------------

/// A 3×1×1 cantilever clamped at `x = 0`, transverse tip load.
fn cantilever(scale: f64) -> (SdfTetMesh, BoundaryConditions) {
    let (nx, ny, nz) = (3usize, 1usize, 1usize);
    let mesh = kuhn_box(nx, ny, nz, 2.0);
    let mut bc = BoundaryConditions::new();
    for k in 0..=nz {
        for j in 0..=ny {
            bc.prescribe_all(node_index(nx, ny, 0, j, k), [Fix128::ZERO; 3]);
        }
    }
    for k in 0..=nz {
        for j in 0..=ny {
            bc.add_load(node_index(nx, ny, nx, j, k), Axis::Y, fx(scale / 4.0));
        }
    }
    (mesh, bc)
}

/// Reverse the vertex numbering and the tetrahedron order.
fn permuted(mesh: &SdfTetMesh, bc: &BoundaryConditions) -> (SdfTetMesh, BoundaryConditions) {
    let n = mesh.vertices.len();
    let map = |v: u32| (n as u32 - 1) - v;
    let mut out = SdfTetMesh {
        vertices: mesh.vertices.iter().rev().copied().collect(),
        ..SdfTetMesh::default()
    };
    for tet in mesh.tets.iter().rev() {
        let mut v = tet.vertices;
        for x in &mut v {
            *x = map(*x);
        }
        out.tets.push(Tetrahedron { vertices: v });
    }
    let mut b = BoundaryConditions::new();
    for &(v, axis, val) in bc.prescribed() {
        b.prescribe(map(v), axis, val);
    }
    for &(v, axis, val) in bc.loads() {
        b.add_load(map(v), axis, val);
    }
    (out, b)
}

/// oracle: invariance. The answer of a non-uniform plastic scene (a cantilever
/// whose root yields while the tip stays elastic) does not depend on how the
/// vertices are numbered or in which order the tetrahedra are listed. There is
/// no closed form for the field, so the check is the invariance itself, with a
/// tolerance of `1e-8` (the conjugate gradient visits the unknowns in another
/// order, so bit equality is not promised).
#[test]
fn numbering_and_element_order_do_not_change_the_answer() {
    let cfg = config(SIGMA_Y, H_PLASTIC);
    let (mesh, bc) = cantilever(3.0);
    let (pmesh, pbc) = permuted(&mesh, &bc);
    let path = eighths(&[2, 4, 6, 8]);
    let a = run(&mesh, &bc, &cfg, &path);
    let b = run(&pmesh, &pbc, &cfg, &path);
    let plastic = a
        .equivalent_plastic_strain
        .iter()
        .filter(|x| x.to_f64() > 1e-9)
        .count();
    assert!(
        plastic > 0 && plastic < a.equivalent_plastic_strain.len(),
        "the scene must be partly plastic to test anything, {plastic} of {} yielded",
        a.equivalent_plastic_strain.len()
    );
    let n = a.field.displacements.len();
    for v in 0..n {
        for axis in 0..3 {
            close(
                b.field.displacements[n - 1 - v][axis],
                a.field.displacements[v][axis].to_f64(),
                1e-8,
                &format!("u[{v}][{axis}]"),
            );
        }
    }
    let m = a.field.element_stress.len();
    for t in 0..m {
        let sa = a.field.element_stress[t];
        let sb = b.field.element_stress[m - 1 - t];
        close(sb.xx, sa.xx.to_f64(), 1e-6, "σ_xx");
        close(sb.xy, sa.xy.to_f64(), 1e-6, "σ_xy");
        close(
            b.equivalent_plastic_strain[m - 1 - t],
            a.equivalent_plastic_strain[t].to_f64(),
            1e-8,
            "ε̄_p",
        );
    }
}

/// oracle: the consistent tangent. A single load step to `4 ε_y` on the
/// hardening bar converges in 4 Newton iterations measured (one elastic
/// overshoot, then the quadratic tail; the tangent is the derivative of the
/// return map). The bound is 5; a tangent that was not the derivative takes
/// more or does not converge in the budget.
#[test]
fn the_consistent_tangent_converges_in_a_few_iterations() {
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let cfg = config(SIGMA_Y, H_PLASTIC);
    let sol = run(&mesh, &bc, &cfg, &[Fix128::ONE]);
    assert!(
        sol.newton_iterations <= 5,
        "one step to 4 ε_y took {} Newton iterations",
        sol.newton_iterations
    );
}

// ---------------------------------------------------------------------------
// oracle 8: invalid input
// ---------------------------------------------------------------------------

fn invalid_config(res: Result<ElastoplasticConfig, FemError>, what: &str) {
    assert!(
        matches!(res, Err(FemError::InvalidConfig(_))),
        "{what}: expected Err(InvalidConfig), got {res:?}"
    );
}

/// Policy: a setting outside its usable range is `Err(InvalidConfig)` at
/// construction, never a panic and never a silent clamp.
#[test]
fn config_rejects_every_unusable_setting() {
    let ok =
        |sy: f64, h: f64| ElastoplasticConfig::try_new(linear(), 60, newton_tol(), fx(sy), fx(h));
    assert!(ok(2.0, 1024.0).is_ok());
    assert!(ok(2.0, 0.0).is_ok(), "H = 0 is perfect plasticity, valid");
    invalid_config(ok(0.0, 1.0), "σ_y = 0");
    invalid_config(ok(-2.0, 1.0), "σ_y < 0");
    invalid_config(ok(2.0e9, 1.0), "σ_y beyond the arithmetic range");
    invalid_config(ok(2.0, -1.0), "H < 0 (softening)");
    invalid_config(ok(2.0, 2.0e9), "H beyond the arithmetic range");
    invalid_config(
        ElastoplasticConfig::try_new(linear(), 0, newton_tol(), fx(2.0), fx(1.0)),
        "zero Newton budget",
    );
    invalid_config(
        ElastoplasticConfig::try_new(linear(), 60, Fix128::ZERO, fx(2.0), fx(1.0)),
        "zero Newton tolerance",
    );
    invalid_config(
        ElastoplasticConfig::try_new(linear(), 60, fx(-0.5), fx(2.0), fx(1.0)),
        "negative Newton tolerance",
    );
    invalid_config(
        ElastoplasticConfig::try_new(linear(), 60, Fix128::ONE, fx(2.0), fx(1.0)),
        "Newton tolerance of 1",
    );
}

/// Policy: the default configuration is valid and means "plasticity effectively
/// off" (a yield stress far above any physical load), so it cannot yield a bar.
#[test]
fn the_default_config_never_yields_a_physical_load() {
    let cfg = ElastoplasticConfig::default();
    let mesh = bar_mesh();
    let sol = run(&mesh, &bar_displacement_bc(eps_ref()), &cfg, &[Fix128::ONE]);
    assert!(sol.equivalent_plastic_strain.iter().all(|a| a.is_zero()));
    assert_uniaxial(&sol, E_MPA / 128.0, "default config is elastic");
}

/// Policy: structural problems with the model are `Err` with the same variants
/// [`solve`] uses.
#[test]
fn entry_rejects_bad_models_explicitly() {
    let cfg = config(SIGMA_Y, H_PLASTIC);
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let one = [Fix128::ONE];

    let empty = SdfTetMesh::default();
    assert_eq!(
        solve_elastoplastic(&empty, &material(), &bc, &cfg, &one).unwrap_err(),
        FemError::EmptyMesh
    );

    let mut bad = bar_displacement_bc(eps_ref());
    bad.prescribe(9999, Axis::X, Fix128::ZERO);
    assert!(matches!(
        solve_elastoplastic(&mesh, &material(), &bad, &cfg, &one),
        Err(FemError::VertexOutOfRange { vertex: 9999, .. })
    ));
    let mut bad = bar_displacement_bc(eps_ref());
    bad.add_load(9999, Axis::X, Fix128::ONE);
    assert!(matches!(
        solve_elastoplastic(&mesh, &material(), &bad, &cfg, &one),
        Err(FemError::VertexOutOfRange { vertex: 9999, .. })
    ));

    let mut loose = BoundaryConditions::new();
    loose.fix(0);
    loose.prescribe(1, Axis::X, Fix128::ZERO);
    assert_eq!(
        solve_elastoplastic(&mesh, &material(), &loose, &cfg, &one).unwrap_err(),
        FemError::UnderConstrained
    );

    let mut flat = SdfTetMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        ..SdfTetMesh::default()
    };
    flat.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    let mut fb = BoundaryConditions::new();
    fb.fix(0);
    fb.fix(1);
    assert_eq!(
        solve_elastoplastic(&flat, &material(), &fb, &cfg, &one).unwrap_err(),
        FemError::DegenerateElement { tet: 0 }
    );
}

/// Policy: an empty load path and an out-of-range load factor are
/// `Err(InvalidConfig)`.
#[test]
fn entry_rejects_bad_load_paths() {
    let cfg = config(SIGMA_Y, H_PLASTIC);
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    assert!(matches!(
        solve_elastoplastic(&mesh, &material(), &bc, &cfg, &[]),
        Err(FemError::InvalidConfig(_))
    ));
    for bad in [fx(2.0e6), fx(-2.0e6)] {
        assert!(matches!(
            solve_elastoplastic(&mesh, &material(), &bc, &cfg, &[bad]),
            Err(FemError::InvalidConfig(_))
        ));
    }
}

/// Policy: zero load is `Ok` and gives the zero field, not an error.
#[test]
fn zero_load_gives_the_zero_field() {
    let cfg = config(SIGMA_Y, H_PLASTIC);
    let mesh = bar_mesh();
    let sol = run(
        &mesh,
        &bar_displacement_bc(eps_ref()),
        &cfg,
        &[Fix128::ZERO],
    );
    for u in &sol.field.displacements {
        assert_eq!(*u, [Fix128::ZERO; 3]);
    }
    for s in &sol.field.element_stress {
        assert_eq!(s.xx, Fix128::ZERO);
    }
    let sol = run(&mesh, &bar_traction_bc(0.0), &cfg, &[Fix128::ONE]);
    assert!(sol.field.element_stress.iter().all(|s| s.xx.is_zero()));
}

/// Policy: the largest accepted load factor (`2²⁰`) is driven through a heavily
/// plastic state without panic or overflow, and still lands on the closed form
/// (`ε = 2⁻⁷ · 2²⁰ = 2¹³`, `σ = σ_y + E_t (ε − ε_y)`).
#[test]
fn a_huge_load_factor_is_still_the_bilinear_curve() {
    let cfg = config(SIGMA_Y, H_PLASTIC);
    let mesh = bar_mesh();
    let sol = run(
        &mesh,
        &bar_displacement_bc(eps_ref()),
        &cfg,
        &[Fix128::from_int(1 << 20)],
    );
    let eps = 8192.0_f64;
    close(
        sol.field.element_stress[0].xx,
        SIGMA_Y + E_TANGENT * (eps - EPS_Y),
        4.0e6 * 1e-7,
        "σ_xx at 2^13 strain (relative 1e-7)",
    );
}

/// Policy: `E = 0` is refused when the material is built (`Err`), so it never
/// reaches the solver. A nearly incompressible or nearly auxetic solid (`ν`
/// at the edges the material accepts) and a very large hardening modulus
/// (`H → ∞` is elastic) either land on the bilinear closed form (the uniaxial
/// curve does not depend on `ν`) or return an explicit `Err` from the
/// conjugate gradient; neither panics and neither returns a different stress.
#[test]
fn extreme_material_and_hardening_do_not_panic() {
    assert!(ElasticMaterial::new(Fix128::ZERO, fx(NU)).is_err());
    let mesh = bar_mesh();
    let bc = bar_displacement_bc(eps_ref());
    let cases = [
        (0.499_f64, H_PLASTIC),
        (-0.9_f64, H_PLASTIC),
        (NU, 1.0e9),
        (NU, 1.0e-6),
    ];
    for (nu, h) in cases {
        let mat = ElasticMaterial::new(fx(E_MPA), fx(nu)).expect("ν inside (-1, 0.5)");
        let cfg = config(SIGMA_Y, h);
        let r = solve_elastoplastic(&mesh, &mat, &bc, &cfg, &eighths(&[4, 8]));
        if let Ok(sol) = r {
            let et = E_MPA * h / (E_MPA + h);
            let want = SIGMA_Y + et * (1.0 / 128.0 - EPS_Y);
            for s in &sol.field.element_stress {
                close(s.xx, want, 1e-3, &format!("ν={nu} H={h} σ_xx"));
            }
        }
    }
}

/// oracle: the consistent tangent in shear and in a mixed state. Simple shear
/// at `γ = 0.01` in one step and the plastic cantilever each converge within a
/// bound measured on this implementation (shear 3, cantilever 13 over its four
/// steps). A tangent with a wrong shear term or a wrong direction `n` leaves the
/// answer unchanged and only costs iterations, so the count is the only thing
/// that can see it.
#[test]
fn the_consistent_tangent_is_right_in_shear_and_in_mixed_states() {
    let cfg = config(SIGMA_Y, H_PLASTIC);
    let (mesh, bc) = shear_scene(0.01);
    let shear = run(&mesh, &bc, &cfg, &[Fix128::ONE]);
    assert!(
        shear.newton_iterations <= SHEAR_NEWTON_BOUND,
        "simple shear took {} Newton iterations",
        shear.newton_iterations
    );
    let (mesh, bc) = cantilever(3.0);
    let beam = run(&mesh, &bc, &cfg, &eighths(&[2, 4, 6, 8]));
    assert!(
        beam.newton_iterations <= CANTILEVER_NEWTON_BOUND,
        "the cantilever took {} Newton iterations",
        beam.newton_iterations
    );
}

const SHEAR_NEWTON_BOUND: u32 = 3;
const CANTILEVER_NEWTON_BOUND: u32 = 13;
