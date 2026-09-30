//! Acceptance oracles for thermoelastic coupling, written **before** the term
//! exists.
//!
//! Both tests are `#[ignore]`d and both are **red** when the attribute is
//! removed. That is the point: they fix the acceptance criterion for wiring a
//! temperature field into [`alice_physics::linear_elastic_fem`] while the
//! residual still has no eigenstrain term, so the criterion cannot be written
//! to match whatever the implementation happens to produce.
//!
//! Measured on `origin/main` at `f0d8689`:
//!
//! ```text
//! $ grep -rn "thermal\|temperature\|eigenstrain" src/linear_elastic_fem.rs
//! (no output, exit 1)
//! ```
//!
//! `CoupledField` exists (`src/coupled_field.rs`) and is the intended channel,
//! but [`solve`] has no parameter that could receive it, and `ElasticMaterial`
//! carries no expansion coefficient. Both are named in the ignore reasons.
//!
//! # ⚠️ The measured red, so nobody has to guess whether these are broken tests
//!
//! `cargo test --test analytic_thermoelastic -- --ignored --nocapture`, run on
//! `f0d8689` with the native feature set:
//!
//! ```text
//! test free_thermal_expansion_is_affine_and_stress_free ...
//!   n = 2: worst |u − α ΔT X| = 2.000000e-1 mm, worst |σ| = 0.000000e0 MPa,
//!          far-corner u_x = 0.000000e0 mm (want 2.000000e-1)
//! panicked: n = 2: free expansion must be the affine field α ΔT · X; worst node
//!           is 2.000000e-1 mm away
//!
//! test fully_constrained_heating_is_hydrostatic_compression ...
//!   n = 2: worst |u| = 0.000000e0 mm, worst |σ_nn − -583.333333| = 5.833333e2 MPa,
//!          worst |σ_shear| = 0.000000e0 MPa, first element σ_xx = 0.000000e0 MPa
//! panicked: n = 2: every normal stress must be -583.3333333333333 MPa
//!           (compression); worst element is 5.833333e2 MPa away
//!
//! test result: FAILED. 0 passed; 2 failed
//! ```
//!
//! The distinction that matters: **the discrepancy equals the expected value
//! itself** — `2.000000e-1 mm = α ΔT · L` and `5.833333e2 MPa = 1750/3` — so the
//! solve returned the identically zero state, which is the correct answer to the
//! problem it was actually given (no load). The closed-form cross-checks and the
//! vacuity guards all pass *before* those comparisons. That is a missing source
//! term, not a broken test.
//!
//! Note which assert did **not** fire: free expansion's `σ ≡ 0` passes today,
//! because zero stress is right for zero displacement. It only becomes
//! load-bearing once the thermal load lands — see the halfway-wiring table below.
//!
//! # Why a separate file
//!
//! `tests/analytic_coupled_field.rs` pins the *field operations* of
//! `CoupledField` — trilinear interpolation, splat, diffusion, reconciliation —
//! and its module doc scopes itself to exactly those three. It imports no solid
//! mechanics at all. The oracles here are about the **FEM residual**: they need
//! a tet mesh, a material, boundary conditions and a linear solve, and they are
//! red for a reason that lives in `linear_elastic_fem.rs`, not in
//! `coupled_field.rs`. Appending them there would make one file answer two
//! unrelated questions and would make its "every expected value comes from a
//! closed form of the three field operations" doc false. The two tests here also
//! share a single lifecycle — they flip from ignored to required together, on the
//! same commit — which is easier to see in a file of their own.
//!
//! # The two closed forms
//!
//! Thermoelastic constitutive law with a uniform temperature rise `ΔT` and an
//! isotropic linear expansion coefficient `α`:
//!
//! ```text
//! ε_th = α ΔT · I                       (the eigenstrain, uniform)
//! σ    = λ tr(ε − ε_th) I + 2μ (ε − ε_th)
//! ```
//!
//! **(1) Free expansion.** Constrain only the six rigid-body modes, all with
//! value zero. The body is then free to expand and the exact solution is the
//! affine field
//!
//! ```text
//! u(X) = α ΔT · X          ⇒  ε = α ΔT · I = ε_th  ⇒  σ ≡ 0
//! ```
//!
//! Two independent facts, asserted separately below: the displacement is that
//! affine field, and **every** element stress vanishes. The affine field is
//! linear in `X`, so it lies in the P1 space exactly — there is no
//! discretisation error to absorb, for any conforming tet mesh, which is why
//! both mesh refinements are run.
//!
//! **(2) Full constraint.** Prescribe `u = 0` on every boundary node. The
//! eigenstrain is uniform, so `div(C : ε_th) = 0` and the assembled thermal load
//! is zero on every interior degree of freedom — it appears only as a reaction on
//! the constrained boundary. Therefore `u ≡ 0` exactly, and with `ε = 0`:
//!
//! ```text
//! σ = −(3λ + 2μ) α ΔT · I = −(E α ΔT / (1 − 2ν)) · I
//! ```
//!
//! With `E = 3500 MPa`, `ν = 7/20`, `α = 1/1000 K⁻¹`, `ΔT = 50 K`:
//!
//! ```text
//! α ΔT      = 50/1000 = 1/20
//! 1 − 2ν    = 1 − 7/10 = 3/10
//! σ         = −3500 · (1/20) / (3/10) = −175 · 10/3 = −1750/3 ≈ −583.3333 MPa
//! ```
//!
//! Cross-checked through the Lamé route, which shares no factor with the one
//! above: `λ = Eν/((1+ν)(1−2ν)) = 1225/0.405`, `μ = E/(2(1+ν)) = 3500/2.7`, so
//! `3λ + 2μ = 35000/3` and `−(3λ + 2μ) · (1/20) = −1750/3`. The test asserts the
//! two routes agree **and** that both equal the exact rational `−1750/3`, so a
//! typo in either expression is caught by the other.
//!
//! # ⚠️ The tolerances below are targets, not measurements
//!
//! A red test measures nothing about attainable precision. `1e-9 mm` and
//! `1e-2 MPa` are chosen because both closed forms lie in the P1 space exactly,
//! leaving only the conjugate-gradient relative residual (`2⁻³⁰`) to account
//! for. Whoever lands the eigenstrain term **must re-measure the achieved error
//! and record it here**, exactly as the other analytic files do, and tighten or
//! justify the bound at that point.
//!
//! # What each assert catches, if the term is wired only halfway
//!
//! | partial wiring | which assert fires |
//! |---|---|
//! | thermal load added to the residual, `C : ε_th` **not** subtracted from the reported stress | free expansion: displacement passes, `σ ≡ 0` fails; full constraint: stress has the wrong sign |
//! | stress corrected but no thermal load in the residual | free expansion: displacement stays zero and fails; full constraint passes vacuously |
//! | eigenstrain applied with the wrong bulk factor (`E α ΔT` instead of `E α ΔT/(1 − 2ν)`) | full constraint: off by the factor `1/(1 − 2ν) = 10/3` |
//! | field read but sign flipped (heating treated as cooling) | full constraint: sign, caught by the signed comparison rather than a magnitude |
//!
//! Neither test can pass on an all-zero state: each carries a vacuity guard
//! asserting that the quantity it is about to compare against is far larger than
//! the tolerance it uses.
//!
//! # ⚠️ Reversal condition — when to remove `#[ignore]`
//!
//! Remove it from **both** tests on the commit that lands all three of:
//!
//! 1. a way to hand a temperature (or general eigenstrain) field to the solve —
//!    a `CoupledField` parameter, or `α` plus `ΔT` on `ElasticMaterial`;
//! 2. the element load `f_e = ∫ Bᵀ C ε_th dV` in the residual assembled by
//!    [`solve`] (and by `solve_corotational`, if it gains the same term);
//! 3. `C : ε_th` subtracted from `FemSolution::element_stress`.
//!
//! If only some of the three land, the table above says which assert is expected
//! to stay red; do not relax a bound to get past it. If the wiring chooses a
//! different channel than `CoupledField`, replace [`uniform_temperature_rise`]
//! and leave the closed forms untouched — they do not depend on how the data
//! arrives.
//!
//! The field here carries the temperature **rise above the stress-free reference
//! configuration**. A wiring that carries absolute temperature must also carry
//! that reference; only the field construction changes, not the oracle.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_field::CoupledField;
use alice_physics::linear_elastic_fem::{
    solve, Axis, BoundaryConditions, ElasticMaterial, SolverConfig, StressTensor,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// material and scene
// ---------------------------------------------------------------------------

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;
/// Side of the cube, mm.
const SIDE: f64 = 4.0;
/// Linear expansion coefficient, K⁻¹. Chosen so `α ΔT` is the exact rational
/// `1/20` and every number in the closed forms stays rational.
const ALPHA_PER_K: f64 = 1.0 / 1000.0;
/// Uniform temperature rise above the stress-free reference, K.
const DELTA_T_K: f64 = 50.0;

/// Displacement tolerance, mm. See the module doc: a target, not a measurement.
const U_TOL_MM: f64 = 1e-9;
/// Stress tolerance, MPa. See the module doc: a target, not a measurement.
const SIGMA_TOL_MPA: f64 = 1e-2;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn pla() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and ν in (-1, 0.5)")
}

fn lame() -> (f64, f64) {
    (
        E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        E_MPA / (2.0 * (1.0 + NU)),
    )
}

/// `α ΔT`, the uniform eigenstrain magnitude. Exactly `1/20`.
fn alpha_delta_t() -> f64 {
    ALPHA_PER_K * DELTA_T_K
}

/// `σ = −E α ΔT / (1 − 2ν)`, MPa — the fully constrained stress.
fn constrained_stress_mpa() -> f64 {
    -E_MPA * alpha_delta_t() / (1.0 - 2.0 * NU)
}

/// The same number by the Lamé route `−(3λ + 2μ) α ΔT`, sharing no factor with
/// the expression above.
fn constrained_stress_via_lame_mpa() -> f64 {
    let (lambda, mu) = lame();
    -(3.0 * lambda + 2.0 * mu) * alpha_delta_t()
}

fn node_index(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// Kuhn 6-tet cube on `[0, n·h]³`, conforming at every `n`.
fn kuhn_cube(n: usize, h: f64) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                mesh.vertices.push([
                    (i as f64 * h) as f32,
                    (j as f64 * h) as f32,
                    (k as f64 * h) as f32,
                ]);
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

fn vert(mesh: &SdfTetMesh, v: u32) -> [f64; 3] {
    let p = mesh.vertices[v as usize];
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

fn solver_config() -> SolverConfig {
    SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid linear config")
}

/// The channel the residual is meant to read: a uniform temperature rise over
/// the cube.
///
/// The field is uniform, so its resolution is irrelevant to the closed forms —
/// `5³` matches the coarse mesh's node lattice only for readability. The
/// oracles assert the uniformity rather than assuming it, because a field that
/// was accidentally left partly zero would make both closed forms wrong while
/// still looking plausible.
fn uniform_temperature_rise() -> CoupledField {
    CoupledField::try_new_filled(
        5,
        5,
        5,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (fx(SIDE), fx(SIDE), fx(SIDE)),
        fx(DELTA_T_K),
    )
    .expect("5x5x5 over [0, 4]³ is a valid grid")
}

/// Assert the driving data really is the uniform `ΔT` the closed forms assume.
fn assert_field_is_uniform(field: &CoupledField) {
    for p in [
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(fx(1.5), fx(2.5), fx(0.5)),
        Vec3Fix::new(fx(SIDE), fx(SIDE), fx(SIDE)),
    ] {
        let t = field.sample(p).to_f64();
        assert!(
            (t - DELTA_T_K).abs() < 1e-9,
            "the temperature field must be the uniform rise the closed forms assume; \
             sampled {t} K instead of {DELTA_T_K} K"
        );
    }
}

fn worst_stress_component(stresses: &[StressTensor]) -> f64 {
    let mut worst = 0.0_f64;
    for s in stresses {
        for c in [s.xx, s.yy, s.zz, s.xy, s.yz, s.zx] {
            worst = worst.max(c.to_f64().abs());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// oracle 1 — free thermal expansion
// ---------------------------------------------------------------------------

/// **Free expansion: `u = α ΔT · X` and zero stress everywhere.**
///
/// Only the six rigid-body modes are constrained, every one of them with the
/// value the exact solution already has there, so the constraint set adds no
/// information beyond removing the null space: the node at the origin does not
/// move under `u = α ΔT · X`, the node at `(L, 0, 0)` has `u_y = u_z = 0`, and
/// the node at `(0, L, 0)` has `u_z = 0`. Six constraints is also the minimum
/// [`solve`] accepts.
///
/// The two asserts are independent facts, not two views of one. A wiring that
/// produces the right displacement while reporting `σ = C : ε` without the
/// eigenstrain correction passes the first and fails the second.
#[test]
#[ignore = "src gap: linear_elastic_fem has no eigenstrain term; CoupledField is not wired into the residual"]
fn free_thermal_expansion_is_affine_and_stress_free() {
    let temperature = uniform_temperature_rise();
    assert_field_is_uniform(&temperature);

    let strain = alpha_delta_t();
    let far_corner_u = strain * SIDE;
    assert!(
        far_corner_u > 100.0 * U_TOL_MM,
        "vacuity guard: the expected displacement at the far corner is {far_corner_u} mm, \
         which must be far larger than the {U_TOL_MM} mm tolerance or an all-zero \
         solution would pass"
    );

    for n in [2_usize, 4_usize] {
        let mesh = kuhn_cube(n, SIDE / n as f64);
        let far = node_index(n, n, n, n);

        let mut bc = BoundaryConditions::new();
        bc.fix(node_index(n, 0, 0, 0));
        bc.prescribe(node_index(n, n, 0, 0), Axis::Y, Fix128::ZERO);
        bc.prescribe(node_index(n, n, 0, 0), Axis::Z, Fix128::ZERO);
        bc.prescribe(node_index(n, 0, n, 0), Axis::Z, Fix128::ZERO);
        assert_eq!(
            bc.prescribed_count(),
            6,
            "exactly the six rigid-body modes, no more"
        );

        let out = solve(&mesh, &pla(), &bc, &solver_config())
            .unwrap_or_else(|e| panic!("n = {n}: {e:?}"));

        let mut worst_u = 0.0_f64;
        for v in 0..u32::try_from(mesh.vertices.len()).expect("fits") {
            let x = vert(&mesh, v);
            for (coord, got) in x.iter().zip(out.displacements[v as usize].iter()) {
                let want = strain * coord;
                worst_u = worst_u.max((got.to_f64() - want).abs());
            }
        }
        let worst_sigma = worst_stress_component(&out.element_stress);
        eprintln!(
            "  n = {n}: worst |u − α ΔT X| = {worst_u:.6e} mm, worst |σ| = {worst_sigma:.6e} MPa, \
             far-corner u_x = {:.6e} mm (want {far_corner_u:.6e})",
            out.displacements[far as usize][0].to_f64()
        );

        assert!(
            worst_u < U_TOL_MM,
            "n = {n}: free expansion must be the affine field α ΔT · X; worst node is \
             {worst_u:.6e} mm away, and that field is linear in X so P1 carries it exactly"
        );
        assert!(
            worst_sigma < SIGMA_TOL_MPA,
            "n = {n}: an unconstrained uniform expansion stores no energy, so every element \
             stress must vanish; worst component is {worst_sigma:.6e} MPa on a {E_MPA} MPa \
             material"
        );
    }
}

// ---------------------------------------------------------------------------
// oracle 2 — fully constrained heating
// ---------------------------------------------------------------------------

/// **Full constraint: `u = 0` and `σ = −E α ΔT / (1 − 2ν) · I`.**
///
/// Every boundary node is held at zero. The eigenstrain is uniform, so the
/// thermal load vanishes on every interior degree of freedom and `u ≡ 0` is
/// exact rather than approximate; the whole eigenstrain then shows up as stress.
///
/// The stress is compared **signed and component by component**: heating a fully
/// constrained body puts it in compression, and the shear components stay zero
/// because the eigenstrain is isotropic. A magnitude-only comparison would
/// accept a sign error, and a von Mises comparison would accept any hydrostatic
/// state at all — including the correct answer's negation and, since
/// `von_mises` of a pure hydrostatic state is zero, including zero stress.
#[test]
#[ignore = "src gap: linear_elastic_fem has no eigenstrain term; CoupledField is not wired into the residual"]
fn fully_constrained_heating_is_hydrostatic_compression() {
    let temperature = uniform_temperature_rise();
    assert_field_is_uniform(&temperature);

    let want = constrained_stress_mpa();
    let want_lame = constrained_stress_via_lame_mpa();
    assert!(
        (want - want_lame).abs() < 1e-9,
        "the two closed forms must agree: −E α ΔT/(1 − 2ν) = {want} MPa, \
         −(3λ + 2μ) α ΔT = {want_lame} MPa"
    );
    assert!(
        (want - (-1750.0 / 3.0)).abs() < 1e-9,
        "and both must equal the exact rational −1750/3 MPa, not {want}"
    );
    assert!(
        want.abs() > 100.0 * SIGMA_TOL_MPA,
        "vacuity guard: the expected stress is {want} MPa, which must be far larger than \
         the {SIGMA_TOL_MPA} MPa tolerance or a zero-stress solution would pass"
    );

    let eps = SIDE * 1e-9;
    for n in [2_usize, 4_usize] {
        let mesh = kuhn_cube(n, SIDE / n as f64);

        let mut bc = BoundaryConditions::new();
        let mut interior = 0_usize;
        for v in 0..u32::try_from(mesh.vertices.len()).expect("fits") {
            let p = vert(&mesh, v);
            if p.iter().any(|&c| c < eps || c > SIDE - eps) {
                bc.prescribe_all(v, [Fix128::ZERO; 3]);
            } else {
                interior += 1;
            }
        }
        let expected_interior = (n - 1) * (n - 1) * (n - 1);
        assert_eq!(
            interior, expected_interior,
            "n = {n}: an n-cell cube has (n−1)³ interior nodes"
        );

        let out = solve(&mesh, &pla(), &bc, &solver_config())
            .unwrap_or_else(|e| panic!("n = {n}: {e:?}"));

        let mut worst_u = 0.0_f64;
        for u in &out.displacements {
            for c in u {
                worst_u = worst_u.max(c.to_f64().abs());
            }
        }
        let mut worst_normal = 0.0_f64;
        let mut worst_shear = 0.0_f64;
        for s in &out.element_stress {
            for c in [s.xx, s.yy, s.zz] {
                worst_normal = worst_normal.max((c.to_f64() - want).abs());
            }
            for c in [s.xy, s.yz, s.zx] {
                worst_shear = worst_shear.max(c.to_f64().abs());
            }
        }
        eprintln!(
            "  n = {n}: worst |u| = {worst_u:.6e} mm, worst |σ_nn − {want:.6}| = \
             {worst_normal:.6e} MPa, worst |σ_shear| = {worst_shear:.6e} MPa, \
             first element σ_xx = {:.6e} MPa",
            out.element_stress[0].xx.to_f64()
        );

        assert!(
            worst_u < U_TOL_MM,
            "n = {n}: a uniform eigenstrain loads no interior degree of freedom, so the \
             displacement must be exactly zero; worst node is {worst_u:.6e} mm"
        );
        assert!(
            worst_normal < SIGMA_TOL_MPA,
            "n = {n}: every normal stress must be {want} MPa (compression); worst element is \
             {worst_normal:.6e} MPa away"
        );
        assert!(
            worst_shear < SIGMA_TOL_MPA,
            "n = {n}: an isotropic eigenstrain produces no shear; worst component is \
             {worst_shear:.6e} MPa"
        );
    }
}
