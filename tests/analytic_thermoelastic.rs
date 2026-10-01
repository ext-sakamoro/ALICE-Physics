//! Acceptance oracles for thermoelastic coupling, written **before** the term
//! exists.
//!
//! Four of the five tests here are `#[ignore]`d and all four are **red** when
//! the attribute is removed. That is the point: they fix the acceptance
//! criterion for wiring a temperature field into
//! [`alice_physics::linear_elastic_fem`] while the residual still has no
//! eigenstrain term, so the criterion cannot be written to match whatever the
//! implementation happens to produce.
//!
//! The fifth, [`the_discriminating_measurements_are_not_inert`], runs today and
//! pins the measurement helpers the other four compare through — see the
//! controls section.
//!
//! Measured on `origin/main` at `cb2fc36`:
//!
//! ```text
//! $ git grep -cE 'eigenstrain|CoupledField|thermal|temperature' -- src/linear_elastic_fem.rs
//! (no output, exit 1)
//! $ sed -n '1202,1207p' src/linear_elastic_fem.rs
//! pub fn solve(
//!     mesh: &SdfTetMesh,
//!     material: &ElasticMaterial,
//!     boundary: &BoundaryConditions,
//!     config: &SolverConfig,
//! ) -> Result<FemSolution, FemError> {
//! ```
//!
//! `CoupledField` exists (`src/coupled_field.rs`) and is the intended channel,
//! but [`solve`] has no parameter that could receive it, and `ElasticMaterial`
//! carries no expansion coefficient. Both are named in the ignore reasons.
//!
//! # ⚠️ The gap is on **both** sides, and the test side is one function wide
//!
//! The earlier revision of this file built the temperature field, asserted it
//! was uniform, and then called `solve` **without it** — the field was dropped
//! silently at each call site. Removing `#[ignore]` after landing the residual
//! term would therefore still have been red, for a reason that lives in this
//! file rather than in `src/`.
//!
//! Every solve here now goes through [`solve_with_temperature`], which takes the
//! field and documents in one place that `solve` has nowhere to put it. That
//! function is the **only** line that has to change on the test side when the
//! term lands; nothing in the closed forms or the asserts depends on how the
//! data arrives.
//!
//! # ⚠️ The measured red, so nobody has to guess whether these are broken tests
//!
//! `cargo test --features std --test analytic_thermoelastic -- --ignored
//! --nocapture`, run on `cb2fc36`:
//!
//! ```text
//! test fully_constrained_heating_is_hydrostatic_compression ...
//!   n = 2: worst |u| = 0.000000e0 mm, worst |σ_nn − -583.333333| = 5.833333e2 MPa,
//!          worst |σ_shear| = 0.000000e0 MPa, first element σ_xx = 0.000000e0 MPa
//! panicked: n = 2: every normal stress must be -583.3333333333333 MPa
//!           (compression); worst element is 5.833333e2 MPa away
//!
//! test free_thermal_expansion_is_affine_and_stress_free ...
//!   n = 2: worst |u − α ΔT X| = 2.000000e-1 mm, worst |σ| = 0.000000e0 MPa,
//!          far-corner u_x = 0.000000e0 mm (want 2.000000e-1)
//! panicked: n = 2: free expansion must be the affine field α ΔT · X; worst node
//!           is 2.000000e-1 mm away
//!
//! test result: FAILED. 0 passed; 2 failed
//! ```
//!
//! The distinction that matters: **the discrepancy equals the expected value
//! itself** — `2.000000e-1 mm = α ΔT · L` and `5.833333e2 MPa = 1750/3` — so the
//! solve returned the identically zero state, which is the correct answer to the
//! problem it was actually given (no load). The closed-form cross-checks, the
//! scene-data guards and the vacuity guards all pass *before* those comparisons,
//! and nothing panics inside the solver. That is a missing source term, not a
//! broken test and not a solver failure.
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
//! closed form of the three field operations" doc false. The tests here also
//! share a single lifecycle — they flip from ignored to required together, on the
//! same commit — which is easier to see in a file of their own.
//!
//! # ⚠️ Where every expected number comes from
//!
//! Textbook thermoelasticity, evaluated in `f64` from the four scene constants
//! below and nothing else. **No expected value is produced by calling anything
//! under test**: `ElasticMaterial::new` and `CoupledField::try_new_filled`
//! appear only as *inputs* to the solve, and the one place a helper's output is
//! compared against a literal ([`ThermalLoad::assert_drives`], which checks that
//! `CoupledField::sample` returns the `ΔT` the field was filled with) is a
//! contract test on the driving data, not a source of an expected value.
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
//! `3λ + 2μ = 35000/3` and `−(3λ + 2μ) · (1/20) = −1750/3`. The tests assert the
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
//! | stress corrected but no thermal load in the residual | free expansion: displacement stays zero and fails; full constraint passes vacuously — **the two controls below are what catch this**, because a stress built from `ε_th` alone is still a function of `ΔT` only if the field is actually read |
//! | eigenstrain applied with the wrong bulk factor (`E α ΔT` instead of `E α ΔT/(1 − 2ν)`) | full constraint: off by the factor `1/(1 − 2ν) = 10/3` |
//! | field read but sign flipped (heating treated as cooling) | full constraint: sign, caught by the signed comparison rather than a magnitude |
//!
//! # ⚠️ The two controls — what makes the pair above discriminating
//!
//! Oracles 1 and 2 only say "the implementation reproduces one closed form at
//! one `ΔT`". They cannot see a term that produces the right number **without
//! reading the field**, which is the most likely half-landing: a constant load
//! assembled from `α` with `ΔT` hard-coded, or an absolute temperature read
//! against an implicit `0 K` reference, both reproduce the `ΔT = 50 K` answer
//! exactly. The controls close that:
//!
//! | control | varies | goes red on |
//! |---|---|---|
//! | [`an_absent_temperature_rise_must_not_look_like_heating`] | `ΔT = 50 K` vs `ΔT = 0 K`, same scene | a load that ignores the field value (identical states ⇒ no discriminating power); an absolute-temperature read whose reference is not the field's zero (`ΔT = 0` then stresses the body) |
//! | [`cooling_is_the_signed_mirror_of_heating`] | `ΔT = +50 K` vs `ΔT = −50 K` | any even-order use of the rise (`|ΔT|`, `ΔT²`), which passes every single-sign oracle; an offset in the reference temperature, which breaks the exact antisymmetry |
//!
//! The first one is the direct analogue of
//! `eulerian_grid::tests::colour_blind_gauss_seidel_drifts_from_red_black`: it
//! asserts that the quantity the other tests vary actually changes the answer,
//! so their licence to compare against a closed form is measured rather than
//! assumed. **It is red today for exactly that reason** — with no term in the
//! residual the two scenes are bit-identical.
//!
//! Neither oracle can pass on an all-zero state: each carries a vacuity guard
//! asserting that the quantity it is about to compare against is far larger than
//! the tolerance it uses. The `ΔT = 0` leg of the first control is the one place
//! a zero state *is* the right answer, and there the assert that carries the
//! information is the gap against its `ΔT = 50 K` twin, not the zero itself.
//!
//! # ⚠️ Reversal condition — when to remove `#[ignore]`
//!
//! Remove it from **all four ignored** tests on the commit that lands all three
//! of (the fifth test is not ignored and needs no change):
//!
//! 1. a way to hand a temperature (or general eigenstrain) field to the solve —
//!    a `CoupledField` parameter, or `α` plus `ΔT` on `ElasticMaterial` — plus
//!    the one-line change in [`solve_with_temperature`] that forwards it;
//! 2. the element load `f_e = ∫ Bᵀ C ε_th dV` in the residual assembled by
//!    [`solve`] (and by `solve_corotational`, if it gains the same term);
//! 3. `C : ε_th` subtracted from `FemSolution::element_stress`.
//!
//! If only some of the three land, the table above says which assert is expected
//! to stay red; do not relax a bound to get past it. If the wiring chooses a
//! different channel than `CoupledField`, replace [`ThermalLoad`] and leave the
//! closed forms untouched — they do not depend on how the data arrives.
//!
//! The field here carries the temperature **rise above the stress-free reference
//! configuration**. A wiring that carries absolute temperature must also carry
//! that reference; only the field construction changes, not the oracle. Note
//! that the first control is what turns that sentence into a test.
//!
//! ⚠️ Whoever lands the term must also check that the solve **refuses or
//! reports** a field that does not cover the mesh: `CoupledField::sample` clamps
//! a point outside the grid onto the nearest boundary node (documented in
//! `coupled_field.rs`), so a mesh sticking out of the field would be heated by
//! the extruded boundary value instead of failing. Every scene here asserts its
//! own coverage ([`ThermalLoad::assert_drives`]) so that none of them leans on
//! that clamp, but the guard belongs next to the caller, in the same commit.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_field::CoupledField;
use alice_physics::linear_elastic_fem::{
    solve, Axis, BoundaryConditions, ElasticMaterial, FemError, FemSolution, SolverConfig,
    StressTensor,
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

/// Mesh refinements every scene is run at. Both closed forms lie in the P1
/// space exactly, so the answer must not move between them.
const REFINEMENTS: [usize; 2] = [2, 4];

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

/// `α ΔT`, the uniform eigenstrain magnitude. Exactly `1/20` at `ΔT = 50 K`.
///
/// oracle: `ε_th = α ΔT · I`, the isotropic linear-expansion eigenstrain.
fn alpha_delta_t(delta_t_k: f64) -> f64 {
    ALPHA_PER_K * delta_t_k
}

/// `σ = −E α ΔT / (1 − 2ν)`, MPa — the fully constrained stress.
///
/// oracle: `ε = 0` in `σ = λ tr(ε − ε_th) I + 2μ (ε − ε_th)` gives
/// `σ = −(3λ + 2μ) α ΔT · I`, and `3λ + 2μ = 3K = E/(1 − 2ν)`.
fn constrained_stress_mpa(delta_t_k: f64) -> f64 {
    -E_MPA * alpha_delta_t(delta_t_k) / (1.0 - 2.0 * NU)
}

/// The same number by the Lamé route `−(3λ + 2μ) α ΔT`, sharing no factor with
/// the expression above.
///
/// oracle: same closed form, evaluated through `λ = Eν/((1+ν)(1−2ν))` and
/// `μ = E/(2(1+ν))` instead of through the bulk modulus.
fn constrained_stress_via_lame_mpa(delta_t_k: f64) -> f64 {
    let (lambda, mu) = lame();
    -(3.0 * lambda + 2.0 * mu) * alpha_delta_t(delta_t_k)
}

/// Assert the two independent routes to the fully constrained stress agree, and
/// return the value they agree on.
fn constrained_stress_checked(delta_t_k: f64) -> f64 {
    let want = constrained_stress_mpa(delta_t_k);
    let want_lame = constrained_stress_via_lame_mpa(delta_t_k);
    assert!(
        (want - want_lame).abs() < 1e-9,
        "the two closed forms must agree at ΔT = {delta_t_k} K: \
         −E α ΔT/(1 − 2ν) = {want} MPa, −(3λ + 2μ) α ΔT = {want_lame} MPa"
    );
    want
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

fn vertex_count(mesh: &SdfTetMesh) -> u32 {
    u32::try_from(mesh.vertices.len()).expect("lattice fits u32")
}

fn solver_config() -> SolverConfig {
    SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid linear config")
}

// ---------------------------------------------------------------------------
// the driving data, and the one place it has to reach the solve
// ---------------------------------------------------------------------------

/// A uniform temperature rise, together with the `ΔT` the closed forms are
/// evaluated at, so the two cannot drift apart.
struct ThermalLoad {
    field: CoupledField,
    delta_t_k: f64,
}

impl ThermalLoad {
    /// The channel the residual is meant to read: a uniform rise over the cube.
    ///
    /// The field is uniform, so its resolution is irrelevant to the closed forms
    /// — `5³` matches the coarse mesh's node lattice only for readability.
    fn uniform(delta_t_k: f64) -> Self {
        let field = CoupledField::try_new_filled(
            5,
            5,
            5,
            (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
            (fx(SIDE), fx(SIDE), fx(SIDE)),
            fx(delta_t_k),
        )
        .expect("5x5x5 over [0, 4]³ is a valid grid");
        Self { field, delta_t_k }
    }

    /// Assert the driving data really is the uniform `ΔT` the closed forms
    /// assume, and that it covers the mesh.
    ///
    /// Both halves guard against a scene that looks plausible while the closed
    /// form no longer applies to it: a field left partly zero, or a mesh poking
    /// out of the grid, where `CoupledField::sample` silently clamps onto the
    /// nearest boundary node instead of refusing.
    fn assert_drives(&self, mesh: &SdfTetMesh) {
        for p in [
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(fx(1.5), fx(2.5), fx(0.5)),
            Vec3Fix::new(fx(SIDE), fx(SIDE), fx(SIDE)),
        ] {
            let t = self.field.sample(p).to_f64();
            assert!(
                (t - self.delta_t_k).abs() < 1e-9,
                "the temperature field must be the uniform rise the closed forms assume; \
                 sampled {t} K instead of {} K",
                self.delta_t_k
            );
        }
        for v in 0..vertex_count(mesh) {
            let x = vert(mesh, v);
            let p = Vec3Fix::new(fx(x[0]), fx(x[1]), fx(x[2]));
            assert!(
                self.field.contains(p),
                "node {v} at {x:?} mm lies outside the temperature field, where sampling \
                 clamps onto the boundary node; the closed forms assume the field covers \
                 the mesh"
            );
        }
    }
}

/// ⚠️ **The one place the temperature field has to reach the solve.**
///
/// On `cb2fc36` [`solve`] takes `(mesh, material, boundary, config)` and has no
/// parameter that could receive a field, so the load is accepted here, checked
/// against the scene, and then **dropped** — the solve sees a problem with no
/// load at all, which is why every test in this file is `#[ignore]`d and red.
///
/// This is the whole test-side half of the gap. When the eigenstrain term lands,
/// forward `load` to the solve here and nothing else in this file changes: the
/// closed forms, the asserts and the two controls are all written against the
/// physics rather than against the signature.
fn solve_with_temperature(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    config: &SolverConfig,
    load: &ThermalLoad,
) -> Result<FemSolution, FemError> {
    load.assert_drives(mesh);
    // src gap: no eigenstrain term, and no parameter for a CoupledField.
    let _dropped_until_the_eigenstrain_term_lands = load;
    solve(mesh, material, boundary, config)
}

// ---------------------------------------------------------------------------
// boundary conditions
// ---------------------------------------------------------------------------

/// The six rigid-body modes, each prescribed with the value the free-expansion
/// solution already has there, so the constraint set adds no information beyond
/// removing the null space.
fn rigid_modes_only(n: usize) -> BoundaryConditions {
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
    bc
}

/// `u = 0` on every boundary node, with the interior node count checked against
/// the `(n − 1)³` the lattice must have.
fn clamped_boundary(mesh: &SdfTetMesh, n: usize) -> BoundaryConditions {
    let eps = SIDE * 1e-9;
    let mut bc = BoundaryConditions::new();
    let mut interior = 0_usize;
    for v in 0..vertex_count(mesh) {
        let p = vert(mesh, v);
        if p.iter().any(|&c| c < eps || c > SIDE - eps) {
            bc.prescribe_all(v, [Fix128::ZERO; 3]);
        } else {
            interior += 1;
        }
    }
    assert_eq!(
        interior,
        (n - 1) * (n - 1) * (n - 1),
        "n = {n}: an n-cell cube has (n−1)³ interior nodes"
    );
    bc
}

// ---------------------------------------------------------------------------
// measurements
// ---------------------------------------------------------------------------

fn worst_displacement_mm(displacements: &[[Fix128; 3]]) -> f64 {
    let mut worst = 0.0_f64;
    for u in displacements {
        for c in u {
            worst = worst.max(c.to_f64().abs());
        }
    }
    worst
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

/// Worst deviation of any normal component from `want`, MPa.
fn worst_normal_deviation(stresses: &[StressTensor], want: f64) -> f64 {
    let mut worst = 0.0_f64;
    for s in stresses {
        for c in [s.xx, s.yy, s.zz] {
            worst = worst.max((c.to_f64() - want).abs());
        }
    }
    worst
}

fn worst_shear(stresses: &[StressTensor]) -> f64 {
    let mut worst = 0.0_f64;
    for s in stresses {
        for c in [s.xy, s.yz, s.zx] {
            worst = worst.max(c.to_f64().abs());
        }
    }
    worst
}

/// Worst component-wise difference between two stress fields on the same mesh.
fn worst_stress_gap(a: &[StressTensor], b: &[StressTensor]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "the two runs must be on the same mesh to be compared element by element"
    );
    let mut worst = 0.0_f64;
    for (sa, sb) in a.iter().zip(b.iter()) {
        for (ca, cb) in [sa.xx, sa.yy, sa.zz, sa.xy, sa.yz, sa.zx]
            .into_iter()
            .zip([sb.xx, sb.yy, sb.zz, sb.xy, sb.yz, sb.zx])
        {
            worst = worst.max((ca.to_f64() - cb.to_f64()).abs());
        }
    }
    worst
}

/// Worst component-wise violation of `σ(a) = −σ(b)`.
fn worst_antisymmetry_gap(a: &[StressTensor], b: &[StressTensor]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "the two runs must be on the same mesh to be compared element by element"
    );
    let mut worst = 0.0_f64;
    for (sa, sb) in a.iter().zip(b.iter()) {
        for (ca, cb) in [sa.xx, sa.yy, sa.zz, sa.xy, sa.yz, sa.zx]
            .into_iter()
            .zip([sb.xx, sb.yy, sb.zz, sb.xy, sb.yz, sb.zx])
        {
            worst = worst.max((ca.to_f64() + cb.to_f64()).abs());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// contract — the measurements the controls rely on are not inert
// ---------------------------------------------------------------------------

/// The measurement helpers above, against hand-written tensors.
///
/// ⚠️ Every other test in this file is `#[ignore]`d, so until the eigenstrain
/// term lands nothing ever executes [`worst_stress_gap`] or
/// [`worst_antisymmetry_gap`]. A helper that returned `0.0` unconditionally
/// would let both controls pass on the day the term arrives while measuring
/// nothing at all — the controls would be the false green they exist to
/// prevent. A gate that cannot run yet is a gate whose own correctness has to be
/// pinned separately, and this is that pin: the one test here that runs today.
///
/// The two cases that matter are the inert ones: a field compared with itself
/// must give a gap of `0` (which is what the controls read as "the term does not
/// see the field"), and a field compared with itself under the antisymmetry
/// measure must give `2|σ|` rather than `0` (a measure that ignored the sign
/// would report a perfect mirror for two identical compressions).
///
/// Every expected number is arithmetic on the literals in the test, not a closed
/// form of any physics.
///
/// # Measured on `cb2fc36`
///
/// `cargo test --features std --test analytic_thermoelastic`, one mutation at a
/// time with the file restored in between:
///
/// | mutation | result |
/// |---|---|
/// | `worst_stress_gap` returns `0` unconditionally | red — `gap between a loaded and an unloaded state: measured 0, expected 600` |
/// | `worst_antisymmetry_gap` subtracts instead of adding | red — `antisymmetry of ±600 MPa: measured 1200, expected 0` |
/// | `worst_shear` reads the normal components | red — `largest shear component: measured 0, expected 4` |
/// | `worst_normal_deviation` ignores the value it is given | red — `deviation of −600 MPa from its own value: measured 600, expected 0` |
/// | none (control) | green — `1 passed; 0 failed; 4 ignored` |
///
/// Four mutations, four red, control green. The first row is the one that
/// matters most: it is the exact failure mode that would make
/// [`an_absent_temperature_rise_must_not_look_like_heating`] pass while
/// measuring nothing.
#[test]
fn the_discriminating_measurements_are_not_inert() {
    let hydrostatic = |p: f64| StressTensor {
        xx: fx(p),
        yy: fx(p),
        zz: fx(p),
        ..StressTensor::default()
    };
    let close = |got: f64, want: f64, what: &str| {
        assert!(
            (got - want).abs() < 1e-12,
            "{what}: measured {got}, expected {want}"
        );
    };

    let compression = [hydrostatic(-600.0), hydrostatic(-600.0)];
    let unloaded = [hydrostatic(0.0), hydrostatic(0.0)];
    let tension = [hydrostatic(600.0), hydrostatic(600.0)];
    let sheared = [StressTensor {
        xy: fx(3.0),
        yz: fx(-4.0),
        zx: fx(1.0),
        ..StressTensor::default()
    }];

    close(
        worst_stress_component(&compression),
        600.0,
        "worst component of a −600 MPa hydrostatic state",
    );
    close(
        worst_stress_component(&unloaded),
        0.0,
        "worst component of an unloaded state",
    );
    close(
        worst_normal_deviation(&compression, -600.0),
        0.0,
        "deviation of −600 MPa from its own value",
    );
    close(
        worst_normal_deviation(&unloaded, -600.0),
        600.0,
        "deviation of an unloaded state from −600 MPa",
    );
    close(worst_shear(&sheared), 4.0, "largest shear component");
    close(
        worst_shear(&compression),
        0.0,
        "shear of a hydrostatic state",
    );
    close(
        worst_displacement_mm(&[[fx(0.0), fx(-2.5), fx(1.0)]]),
        2.5,
        "largest displacement magnitude",
    );

    close(
        worst_stress_gap(&compression, &unloaded),
        600.0,
        "gap between a loaded and an unloaded state",
    );
    close(
        worst_stress_gap(&compression, &compression),
        0.0,
        "gap of a state with itself — the reading the first control treats as \
         'the term does not see the field'",
    );
    close(
        worst_antisymmetry_gap(&compression, &tension),
        0.0,
        "antisymmetry of ±600 MPa",
    );
    close(
        worst_antisymmetry_gap(&compression, &compression),
        1200.0,
        "antisymmetry of two identical compressions — a sign-blind measure would \
         report 0 here and the second control would have no teeth",
    );
}

// ---------------------------------------------------------------------------
// oracle 1 — free thermal expansion
// ---------------------------------------------------------------------------

/// **Free expansion: `u = α ΔT · X` and zero stress everywhere.**
///
/// Only the six rigid-body modes are constrained, every one of them with the
/// value the exact solution already has there: the node at the origin does not
/// move under `u = α ΔT · X`, the node at `(L, 0, 0)` has `u_y = u_z = 0`, and
/// the node at `(0, L, 0)` has `u_z = 0`. Six constraints is also the minimum
/// [`solve`] accepts.
///
/// The two asserts are independent facts, not two views of one. A wiring that
/// produces the right displacement while reporting `σ = C : ε` without the
/// eigenstrain correction passes the first and fails the second.
#[test]
#[ignore = "src gap on cb2fc36: linear_elastic_fem::solve has no eigenstrain term and no \
            parameter for a temperature field (grep eigenstrain|CoupledField|thermal|temperature \
            over src/linear_elastic_fem.rs = 0 hits), so solve_with_temperature drops the field \
            and the solve answers an unloaded problem"]
fn free_thermal_expansion_is_affine_and_stress_free() {
    let load = ThermalLoad::uniform(DELTA_T_K);
    let strain = alpha_delta_t(load.delta_t_k);
    let far_corner_u = strain * SIDE;
    assert!(
        far_corner_u > 100.0 * U_TOL_MM,
        "vacuity guard: the expected displacement at the far corner is {far_corner_u} mm, \
         which must be far larger than the {U_TOL_MM} mm tolerance or an all-zero \
         solution would pass"
    );

    for n in REFINEMENTS {
        let mesh = kuhn_cube(n, SIDE / n as f64);
        let far = node_index(n, n, n, n);
        let bc = rigid_modes_only(n);

        let out = solve_with_temperature(&mesh, &pla(), &bc, &solver_config(), &load)
            .unwrap_or_else(|e| panic!("n = {n}: {e:?}"));

        let mut worst_u = 0.0_f64;
        for v in 0..vertex_count(&mesh) {
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
#[ignore = "src gap on cb2fc36: linear_elastic_fem::solve has no eigenstrain term and no \
            parameter for a temperature field (grep eigenstrain|CoupledField|thermal|temperature \
            over src/linear_elastic_fem.rs = 0 hits), so solve_with_temperature drops the field \
            and the solve answers an unloaded problem"]
fn fully_constrained_heating_is_hydrostatic_compression() {
    let load = ThermalLoad::uniform(DELTA_T_K);
    let want = constrained_stress_checked(load.delta_t_k);
    assert!(
        (want - (-1750.0 / 3.0)).abs() < 1e-9,
        "and both must equal the exact rational −1750/3 MPa, not {want}"
    );
    assert!(
        want.abs() > 100.0 * SIGMA_TOL_MPA,
        "vacuity guard: the expected stress is {want} MPa, which must be far larger than \
         the {SIGMA_TOL_MPA} MPa tolerance or a zero-stress solution would pass"
    );

    for n in REFINEMENTS {
        let mesh = kuhn_cube(n, SIDE / n as f64);
        let bc = clamped_boundary(&mesh, n);

        let out = solve_with_temperature(&mesh, &pla(), &bc, &solver_config(), &load)
            .unwrap_or_else(|e| panic!("n = {n}: {e:?}"));

        let worst_u = worst_displacement_mm(&out.displacements);
        let worst_normal = worst_normal_deviation(&out.element_stress, want);
        let worst_shear_c = worst_shear(&out.element_stress);
        eprintln!(
            "  n = {n}: worst |u| = {worst_u:.6e} mm, worst |σ_nn − {want:.6}| = \
             {worst_normal:.6e} MPa, worst |σ_shear| = {worst_shear_c:.6e} MPa, \
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
            worst_shear_c < SIGMA_TOL_MPA,
            "n = {n}: an isotropic eigenstrain produces no shear; worst component is \
             {worst_shear_c:.6e} MPa"
        );
    }
}

// ---------------------------------------------------------------------------
// control 1 — the rise has to be what drives the answer
// ---------------------------------------------------------------------------

/// **Teeth for both oracles: `ΔT = 0 K` must give the unloaded state, and must
/// not agree with `ΔT = 50 K`.**
///
/// Same fully constrained scene, run twice, with the temperature rise as the only
/// difference. Two separate facts:
///
/// * `ΔT = 0` ⇒ `ε_th = 0` ⇒ no load and no stress, by the same closed form the
///   oracle above uses, read at zero. A wiring that reads an **absolute**
///   temperature against an implicit `0 K` reference stresses the body here.
/// * the two runs must differ by about the full `1750/3` MPa. If they agree, the
///   term does not read the field, and then the oracles above are comparing a
///   constant against a closed form that happens to match it — they would have
///   no discriminating power at all.
///
/// The second assert is the one that is informative today: with no term in the
/// residual both runs are the identically zero state, so the gap is `0`.
#[test]
#[ignore = "src gap on cb2fc36: with no eigenstrain term the ΔT = 50 K and ΔT = 0 K runs are \
            the same unloaded problem, so the measured gap is 0 MPa instead of 1750/3"]
fn an_absent_temperature_rise_must_not_look_like_heating() {
    let hot = ThermalLoad::uniform(DELTA_T_K);
    let cold = ThermalLoad::uniform(0.0);
    let want_hot = constrained_stress_checked(hot.delta_t_k);
    let want_cold = constrained_stress_checked(cold.delta_t_k);
    assert_eq!(
        want_cold, 0.0,
        "no temperature rise is no eigenstrain, so the closed form must be exactly zero"
    );
    let want_gap = (want_hot - want_cold).abs();
    assert!(
        want_gap > 100.0 * SIGMA_TOL_MPA,
        "vacuity guard: the two scenes are expected to differ by {want_gap} MPa, which must be \
         far larger than the {SIGMA_TOL_MPA} MPa tolerance or two identical zero states \
         would pass"
    );

    for n in REFINEMENTS {
        let mesh = kuhn_cube(n, SIDE / n as f64);
        let bc = clamped_boundary(&mesh, n);
        let cfg = solver_config();

        let out_hot = solve_with_temperature(&mesh, &pla(), &bc, &cfg, &hot)
            .unwrap_or_else(|e| panic!("n = {n}, ΔT = {DELTA_T_K} K: {e:?}"));
        let out_cold = solve_with_temperature(&mesh, &pla(), &bc, &cfg, &cold)
            .unwrap_or_else(|e| panic!("n = {n}, ΔT = 0 K: {e:?}"));

        let cold_u = worst_displacement_mm(&out_cold.displacements);
        let cold_sigma = worst_stress_component(&out_cold.element_stress);
        let gap = worst_stress_gap(&out_hot.element_stress, &out_cold.element_stress);
        eprintln!(
            "  n = {n}: ΔT = 0 gives worst |u| = {cold_u:.6e} mm, worst |σ| = \
             {cold_sigma:.6e} MPa; gap to ΔT = {DELTA_T_K} K is {gap:.6e} MPa \
             (want {want_gap:.6e})"
        );

        assert!(
            cold_u < U_TOL_MM,
            "n = {n}: no temperature rise is no load, so the displacement must be zero; \
             worst node is {cold_u:.6e} mm"
        );
        assert!(
            cold_sigma < SIGMA_TOL_MPA,
            "n = {n}: no temperature rise is no eigenstrain, so every stress component must \
             vanish; worst is {cold_sigma:.6e} MPa — a non-zero value here means the term \
             reads an absolute temperature against a reference the field does not carry"
        );
        assert!(
            (gap - want_gap).abs() < SIGMA_TOL_MPA,
            "n = {n}: heating by {DELTA_T_K} K must change the stress by {want_gap} MPa, but \
             the two runs differ by {gap:.6e} MPa; a gap of zero means the term does not read \
             the temperature field, and then the closed-form oracles have no discriminating \
             power"
        );
    }
}

// ---------------------------------------------------------------------------
// control 2 — the sign of the rise has to carry through
// ---------------------------------------------------------------------------

/// **Teeth against an even-order use of the rise: cooling is heating negated.**
///
/// `ε_th = α ΔT · I` is odd in `ΔT`, and the constrained scene is linear, so
/// `σ(−ΔT) = −σ(ΔT)` exactly — cooling a fully constrained body puts it in
/// tension of the same magnitude, `+1750/3` MPa.
///
/// Oracle 2 pins one sign at one magnitude, which `|ΔT|`, `ΔT²` or `ΔT·|ΔT|`
/// all reproduce. This asserts the cooling closed form **and** the exact
/// antisymmetry between the two runs, so any even-order dependence, and any
/// offset in the reference temperature, is red. The heating closed form is not
/// re-asserted here: oracle 2 owns it, and a common factor wrong in both runs
/// shows up in the cooling comparison.
#[test]
#[ignore = "src gap on cb2fc36: with no eigenstrain term both the heating and the cooling run \
            are the same unloaded problem, so the cooling stress is 0 MPa instead of 1750/3"]
fn cooling_is_the_signed_mirror_of_heating() {
    let hot = ThermalLoad::uniform(DELTA_T_K);
    let cold = ThermalLoad::uniform(-DELTA_T_K);
    let want_cold = constrained_stress_checked(cold.delta_t_k);
    assert!(
        (want_cold - 1750.0 / 3.0).abs() < 1e-9,
        "cooling must be tension of the exact rational +1750/3 MPa, not {want_cold}"
    );
    assert!(
        want_cold > 100.0 * SIGMA_TOL_MPA,
        "vacuity guard: the expected cooling stress is {want_cold} MPa, which must be far \
         larger than the {SIGMA_TOL_MPA} MPa tolerance or a zero-stress solution would pass"
    );

    for n in REFINEMENTS {
        let mesh = kuhn_cube(n, SIDE / n as f64);
        let bc = clamped_boundary(&mesh, n);
        let cfg = solver_config();

        let out_hot = solve_with_temperature(&mesh, &pla(), &bc, &cfg, &hot)
            .unwrap_or_else(|e| panic!("n = {n}, ΔT = +{DELTA_T_K} K: {e:?}"));
        let out_cold = solve_with_temperature(&mesh, &pla(), &bc, &cfg, &cold)
            .unwrap_or_else(|e| panic!("n = {n}, ΔT = −{DELTA_T_K} K: {e:?}"));

        let cold_u = worst_displacement_mm(&out_cold.displacements);
        let cold_normal = worst_normal_deviation(&out_cold.element_stress, want_cold);
        let cold_shear = worst_shear(&out_cold.element_stress);
        let mirror = worst_antisymmetry_gap(&out_hot.element_stress, &out_cold.element_stress);
        eprintln!(
            "  n = {n}: ΔT = −{DELTA_T_K} K gives worst |u| = {cold_u:.6e} mm, \
             worst |σ_nn − {want_cold:.6}| = {cold_normal:.6e} MPa, worst |σ_shear| = \
             {cold_shear:.6e} MPa; worst |σ(+ΔT) + σ(−ΔT)| = {mirror:.6e} MPa"
        );

        assert!(
            cold_u < U_TOL_MM,
            "n = {n}: a uniform eigenstrain loads no interior degree of freedom whatever its \
             sign, so the displacement must be zero; worst node is {cold_u:.6e} mm"
        );
        assert!(
            cold_normal < SIGMA_TOL_MPA,
            "n = {n}: every normal stress must be {want_cold} MPa (tension); worst element is \
             {cold_normal:.6e} MPa away"
        );
        assert!(
            cold_shear < SIGMA_TOL_MPA,
            "n = {n}: an isotropic eigenstrain produces no shear whatever its sign; worst \
             component is {cold_shear:.6e} MPa"
        );
        assert!(
            mirror < SIGMA_TOL_MPA,
            "n = {n}: ε_th is odd in ΔT and the scene is linear, so σ(+ΔT) + σ(−ΔT) must \
             vanish component by component; worst is {mirror:.6e} MPa, which is what an even \
             function of the rise (|ΔT|, ΔT²) or a shifted reference temperature produces"
        );
    }
}
