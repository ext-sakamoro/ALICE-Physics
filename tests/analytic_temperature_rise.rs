//! What `TemperatureRise` fixes, measured against closed forms.
//!
//! # The question this file answers
//!
//! `linear_elastic_fem::ThermalExpansion` drives `ε_th = α ΔT I` from a scalar
//! field, and `ΔT` is a **rise above the stress-free reference**, not a
//! temperature. Both temperature owners in this crate
//! (`thermal::ThermalModifier` and `phase_change::PhaseChangeModifier`) fill
//! their field from `ambient_temperature`, i.e. absolutely, on the same grid
//! and under the same `coupled_field::CoupledScalar::coupled_name` of
//! `"temperature"`. Up to `f842c79` a `&CoupledField` of temperatures and a
//! `&CoupledField` of rises were the same type, so handing an owner's field to
//! the eigenstrain compiled, ran, and loaded the reference temperature itself
//! as if it were a rise.
//!
//! `coupled_field::TemperatureRise` is the subtraction turned into a type:
//! `TemperatureRise::from_absolute(field, reference)` is the only way to build
//! one and `ThermalExpansion::from_rise` is what accepts it, so the reference
//! has to be named at the call site.
//!
//! # ⚠️ A compile error cannot be a test, so this file pins the arithmetic
//!
//! What the type actually buys is that the reference is *stated*; it cannot
//! stop a caller from stating zero for a field that is not measured from zero.
//! Every test below therefore measures the two paths through the same entry
//! point and compares them against closed forms that are far apart:
//!
//! - [`the_stress_follows_the_rise_not_the_absolute_temperature`] — naming the
//!   real reference gives the closed form of `ΔT = T − T_ref`.
//! - [`naming_a_zero_reference_costs_the_stress_of_the_reference_itself`] —
//!   naming zero gives the closed form of `ΔT = T`, and the gap between the
//!   two runs is the closed form of `ΔT = T_ref` to the last digit.
//! - [`only_the_difference_of_the_two_temperatures_reaches_the_solve`] —
//!   different `(T, T_ref)` pairs with the same difference produce
//!   **bit-identical** stress, which needs no tolerance at all.
//! - [`a_one_kelvin_error_in_the_reference_is_outside_every_tolerance_here`] —
//!   the smallest reference mistake worth worrying about is 24 times the
//!   tolerance the other tests use, so none of them can pass with the
//!   subtraction slightly wrong.
//!
//! # Scope
//!
//! Uniform rises and one isotropic material, so that `u ≡ 0` is the exact
//! solution of the clamped scene and the suppressed-stress closed form applies
//! element by element. Non-uniform rises need a different closed form and are
//! filed in the backlog, not covered here. Nothing here exercises the reverse
//! direction (deformation heating), diffusion, or phase change.
//!
//! ⚠️ **Two things this scene is structurally blind to**, measured rather than
//! assumed (destructive run at `8747b62`):
//!
//! - **Where the eigenstrain is sampled.** Replacing
//!   `thermal.field.sample(centroid)` with `sample(Vec3Fix::ZERO)` leaves every
//!   test here green, because a uniform field reads the same everywhere. Only a
//!   non-uniform rise can see the sampling point, and the crate has no such
//!   oracle yet (filed in the backlog).
//! - **Whether the eigenstrain load is added to the right-hand side.** Making
//!   `add_eigenstrain_load` a no-op leaves this file green too: the clamped
//!   answer is `u ≡ 0` with or without the load, and the stress is still
//!   `−σ_th`. The observable for that term is *free* expansion, where the
//!   displacement is what moves — `analytic_thermoelastic::
//!   free_thermal_expansion_is_affine_and_stress_free` is the test that holds
//!   it, and it is the only one of the three files that goes red.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::linear_elastic_fem::{
    solve, solve_with_eigenstrain, BoundaryConditions, ElasticMaterial, FemError, FemSolution,
    SolverConfig, StressTensor, ThermalExpansion,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// material, scene and the temperatures in play
// ---------------------------------------------------------------------------

/// Young's modulus, MPa (PLA).
const E_MPA: f64 = 3500.0;
/// Poisson's ratio.
const NU: f64 = 0.35;
/// Linear expansion coefficient, K⁻¹.
const ALPHA_PER_K: f64 = 1.0 / 1000.0;
/// Side of the cube, mm.
const SIDE: f64 = 4.0;

/// The stress-free reference the mesh was built at, K.
const REFERENCE_K: i64 = 25;
/// The absolute temperature the body is held at, K.
const ABSOLUTE_K: i64 = 65;
/// `ABSOLUTE_K − REFERENCE_K`, the rise the constitutive law is specified on.
const RISE_K: f64 = 40.0;

/// Field resolution. Uniform fields, so this only has to put the outer nodes
/// on the cube's faces.
const RES: usize = 3;

/// Mesh refinements every scene is run at. The exact solution `u ≡ 0` lies in
/// the P1 space, so the answer must not move between them.
const REFINEMENTS: [usize; 2] = [1, 2];

/// Tolerance on a stress in MPa.
///
/// `α = 10⁻³` is not dyadic, so `α ΔT` carries one truncation per multiply and
/// nothing here is exact against an `f64` closed form. The observed worst
/// deviation is far below this; see
/// [`a_one_kelvin_error_in_the_reference_is_outside_every_tolerance_here`] for
/// why this value cannot hide a reference mistake.
const STRESS_TOL_MPA: f64 = 1e-2;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn pla() -> ElasticMaterial {
    ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_ratio(35, 100))
        .expect("E > 0 and ν in (-1, 0.5)")
}

fn lame() -> (f64, f64) {
    (
        E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        E_MPA / (2.0 * (1.0 + NU)),
    )
}

/// `σ = −E α ΔT / (1 − 2ν)`, MPa — every normal component of the fully
/// suppressed stress.
///
/// oracle: the constitutive law is `σ = λ tr(ε − ε_th) I + 2μ (ε − ε_th)` with
/// `ε_th = α ΔT I`. Clamping every boundary node of a body under a *uniform*
/// eigenstrain admits `u ≡ 0` (an interior shape function has
/// `∫∇N_a = ∫_∂Ω N_a n = 0`, so a constant `σ` produces no internal force on a
/// free row), hence `ε = 0` and `σ = −(3λ + 2μ) α ΔT I`. Substituting
/// `3λ + 2μ = 3K = E/(1 − 2ν)` gives the form above, with no shear.
fn suppressed_stress_mpa(delta_t_k: f64) -> f64 {
    -E_MPA * (ALPHA_PER_K * delta_t_k) / (1.0 - 2.0 * NU)
}

/// The same number through `−(3λ + 2μ) α ΔT`, which shares no factor with the
/// bulk-modulus route above.
fn suppressed_stress_via_lame_mpa(delta_t_k: f64) -> f64 {
    let (lambda, mu) = lame();
    -(3.0 * lambda + 2.0 * mu) * (ALPHA_PER_K * delta_t_k)
}

/// Assert the two independent routes agree and return the value.
fn suppressed_stress_checked(delta_t_k: f64) -> f64 {
    let a = suppressed_stress_mpa(delta_t_k);
    let b = suppressed_stress_via_lame_mpa(delta_t_k);
    // Relative, because the two routes differ in their `f64` rounding and the
    // test at a 2²⁰ K reference evaluates them at 1.2e7 MPa, where one unit in
    // the last place is already 2e-9 MPa.
    let slack = 1e-12 * a.abs().max(1.0);
    assert!(
        (a - b).abs() < slack,
        "the two closed forms must agree at ΔT = {delta_t_k} K to {slack} MPa: \
         −E α ΔT/(1 − 2ν) = {a} MPa, −(3λ + 2μ) α ΔT = {b} MPa"
    );
    a
}

// ---------------------------------------------------------------------------
// scene
// ---------------------------------------------------------------------------

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
                    for (slot, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[slot + 1] = node_index(n, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

/// `u = 0` on every boundary node.
fn clamped_boundary(mesh: &SdfTetMesh, n: usize) -> BoundaryConditions {
    let eps = (SIDE * 1e-9) as f32;
    let last = SIDE as f32;
    let mut bc = BoundaryConditions::new();
    let mut interior = 0usize;
    for (v, p) in mesh.vertices.iter().enumerate() {
        let on_face = p.iter().any(|c| *c <= eps || *c >= last - eps);
        if on_face {
            bc.fix(u32::try_from(v).expect("fits"));
        } else {
            interior += 1;
        }
    }
    assert_eq!(
        interior,
        (n - 1) * (n - 1) * (n - 1),
        "the lattice must leave exactly (n − 1)³ interior nodes free at n = {n}"
    );
    bc
}

fn solver_config() -> SolverConfig {
    SolverConfig::default()
}

/// A uniform absolute temperature field covering the cube, as a temperature
/// owner would publish it.
fn absolute_field(value_k: i64) -> CoupledField {
    uniform_field(RES, value_k)
}

/// A uniform field of `value_k` on a `res³` grid over the cube.
fn uniform_field(res: usize, value_k: i64) -> CoupledField {
    CoupledField::try_new_filled(
        res,
        res,
        res,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (
            Fix128::from_int(SIDE as i64),
            Fix128::from_int(SIDE as i64),
            Fix128::from_int(SIDE as i64),
        ),
        Fix128::from_int(value_k),
    )
    .unwrap_or_else(|e| panic!("a {res}³ grid over the cube is valid: {e:?}"))
}

/// ⚠️ **The one place the rise has to reach the residual.**
///
/// A `TemperatureRise` that is built and then not forwarded leaves every
/// stress at zero while the scene still looks right;
/// [`the_rise_actually_reaches_the_residual`] is the control that rules that
/// out by comparing against a run with no eigenstrain at all.
fn solve_with_rise(n: usize, rise: &TemperatureRise) -> FemSolution {
    let mesh = kuhn_cube(n, SIDE / n as f64);
    let bc = clamped_boundary(&mesh, n);
    solve_with_eigenstrain(
        &mesh,
        &pla(),
        &bc,
        &solver_config(),
        Some(ThermalExpansion::from_rise(rise, fx(ALPHA_PER_K))),
    )
    .unwrap_or_else(|e| panic!("the clamped cube at n = {n} is well posed: {e:?}"))
}

/// Worst signed deviation of any normal component from `want`, in MPa.
fn worst_normal_deviation(stresses: &[StressTensor], want: f64) -> f64 {
    assert!(!stresses.is_empty(), "the mesh must have elements");
    let mut worst = 0.0_f64;
    for s in stresses {
        for c in [s.xx, s.yy, s.zz] {
            worst = worst.max((c.to_f64() - want).abs());
        }
    }
    worst
}

/// Worst absolute shear component, in MPa.
fn worst_shear(stresses: &[StressTensor]) -> f64 {
    let mut worst = 0.0_f64;
    for s in stresses {
        for c in [s.xy, s.yz, s.zx] {
            worst = worst.max(c.to_f64().abs());
        }
    }
    worst
}

/// Mean normal component over every element, in MPa.
fn mean_normal_mpa(stresses: &[StressTensor]) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for s in stresses {
        for c in [s.xx, s.yy, s.zz] {
            sum += c.to_f64();
            count += 1;
        }
    }
    assert!(count > 0, "the mesh must have elements");
    sum / count as f64
}

// ---------------------------------------------------------------------------
// 1. the stress follows the rise
// ---------------------------------------------------------------------------

/// Naming the real reference gives the closed form of `ΔT = T − T_ref`.
///
/// oracle: `σ = −E α ΔT / (1 − 2ν)` with `ΔT = 65 − 25 = 40` K, cross-checked
/// against `−(3λ + 2μ) α ΔT`. The absolute temperature 65 K would give
/// −758.333333 MPa and the reference 25 K would give −291.666667 MPa, so the
/// three candidates are hundreds of MPa apart and the assert cannot be
/// satisfied by the wrong one.
#[test]
fn the_stress_follows_the_rise_not_the_absolute_temperature() {
    let want = suppressed_stress_checked(RISE_K);
    assert!(
        (want + 466.666_666_666_666_7).abs() < 1e-9,
        "the closed form at ΔT = 40 K must be −466.666667 MPa, got {want}"
    );

    for n in REFINEMENTS {
        let absolute = absolute_field(ABSOLUTE_K);
        let rise = TemperatureRise::from_absolute(&absolute, Fix128::from_int(REFERENCE_K));
        let out = solve_with_rise(n, &rise);

        let dev = worst_normal_deviation(&out.element_stress, want);
        assert!(
            dev < STRESS_TOL_MPA,
            "at n = {n} every normal stress must be the suppressed {want} MPa of the \
             40 K rise; worst deviation {dev} MPa (the absolute 65 K would read \
             {} MPa)",
            suppressed_stress_mpa(ABSOLUTE_K as f64)
        );
        let shear = worst_shear(&out.element_stress);
        assert!(
            shear < STRESS_TOL_MPA,
            "a hydrostatic eigenstrain on a clamped body stores no shear; \
             worst |σ_shear| = {shear} MPa at n = {n}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. what naming zero costs
// ---------------------------------------------------------------------------

/// Naming a zero reference loads the reference temperature as a rise.
///
/// oracle: the constitutive law is linear in `ΔT`, so the gap between a run at
/// `ΔT = T` and one at `ΔT = T − T_ref` is exactly the suppressed stress of
/// `T_ref` on its own, `E α T_ref / (1 − 2ν) = 291.666667` MPa — 62.5% of the
/// right answer. This is the error the type makes visible rather than
/// impossible, so it is pinned numerically instead of by a compile failure.
#[test]
fn naming_a_zero_reference_costs_the_stress_of_the_reference_itself() {
    const N: usize = 2;
    let absolute = absolute_field(ABSOLUTE_K);

    let right = TemperatureRise::from_absolute(&absolute, Fix128::from_int(REFERENCE_K));
    let wrong = TemperatureRise::from_absolute(&absolute, Fix128::ZERO);

    let right_mpa = mean_normal_mpa(&solve_with_rise(N, &right).element_stress);
    let wrong_mpa = mean_normal_mpa(&solve_with_rise(N, &wrong).element_stress);

    let want_right = suppressed_stress_checked(RISE_K);
    let want_wrong = suppressed_stress_checked(ABSOLUTE_K as f64);
    let want_cost = suppressed_stress_checked(REFERENCE_K as f64).abs();

    assert!(
        (right_mpa - want_right).abs() < STRESS_TOL_MPA,
        "naming the reference must give {want_right} MPa, got {right_mpa} MPa"
    );
    assert!(
        (wrong_mpa - want_wrong).abs() < STRESS_TOL_MPA,
        "naming a zero reference must give the absolute temperature's \
         {want_wrong} MPa, got {wrong_mpa} MPa"
    );

    let cost = (wrong_mpa - right_mpa).abs();
    assert!(
        (cost - want_cost).abs() < STRESS_TOL_MPA,
        "the gap between the two runs must be the suppressed stress of the \
         reference alone, {want_cost} MPa; measured {cost} MPa"
    );
    assert!(
        (cost - 291.666_666_666_666_7).abs() < STRESS_TOL_MPA,
        "the documented cost of skipping the subtraction is 291.666667 MPa; \
         measured {cost} MPa"
    );
}

// ---------------------------------------------------------------------------
// 3. only the difference reaches the solve (no tolerance needed)
// ---------------------------------------------------------------------------

/// Three `(T, T_ref)` pairs with the same difference give bit-identical stress.
///
/// oracle: [`Fix128`] subtraction is exact, so `from_absolute` must produce the
/// *same* field for every pair with the same difference, and the solve is
/// deterministic — hence `assert_eq!` on raw [`Fix128`] rather than a
/// tolerance. A fourth pair with a different difference is included so the test
/// cannot be satisfied by a constructor that ignores its field entirely.
///
/// This is the test that pins the subtraction itself: ignoring the reference
/// leaves the three at 65 / 105 / 1065 K, and adding it instead of subtracting
/// leaves them at 90 / 170 / 2065 K. Either way the three stop agreeing.
#[test]
fn only_the_difference_of_the_two_temperatures_reaches_the_solve() {
    const N: usize = 2;
    const SAME_RISE: [(i64, i64); 3] = [(65, 25), (105, 65), (1065, 1025)];

    let mut reference_stress: Option<Vec<StressTensor>> = None;
    for (absolute_k, reference_k) in SAME_RISE {
        let field = absolute_field(absolute_k);
        let rise = TemperatureRise::from_absolute(&field, Fix128::from_int(reference_k));
        let stress = solve_with_rise(N, &rise).element_stress;
        match &reference_stress {
            None => reference_stress = Some(stress),
            Some(first) => assert_eq!(
                &stress, first,
                "T = {absolute_k} K above T_ref = {reference_k} K is the same 40 K rise \
                 as T = 65 K above 25 K, so the stress must be bit-identical"
            ),
        }
    }

    // A different difference must not land on the same answer, or the three
    // above would agree for the wrong reason.
    let other = absolute_field(105);
    let other_rise = TemperatureRise::from_absolute(&other, Fix128::from_int(REFERENCE_K));
    let other_stress = solve_with_rise(N, &other_rise).element_stress;
    assert_ne!(
        &other_stress,
        reference_stress.as_ref().expect("at least one run"),
        "an 80 K rise must not produce the stress of a 40 K rise"
    );
}

// ---------------------------------------------------------------------------
// 4. the tolerance cannot hide a reference mistake
// ---------------------------------------------------------------------------

/// A one-kelvin error in the reference is 24 times [`STRESS_TOL_MPA`].
///
/// oracle: `E α / (1 − 2ν) = 11.666667` MPa per kelvin of reference, measured
/// here as the gap between two runs whose references differ by exactly 1 K.
/// ⚠️ Without this, a reader cannot tell whether the `1e-2` MPa tolerance the
/// value tests use leaves room for the subtraction to be slightly wrong: it
/// does not, because the smallest error that can occur in the reference is
/// three orders of magnitude larger than the tolerance.
#[test]
fn a_one_kelvin_error_in_the_reference_is_outside_every_tolerance_here() {
    const N: usize = 2;
    let absolute = absolute_field(ABSOLUTE_K);

    let exact = TemperatureRise::from_absolute(&absolute, Fix128::from_int(REFERENCE_K));
    let off_by_one = TemperatureRise::from_absolute(&absolute, Fix128::from_int(REFERENCE_K + 1));

    let a = mean_normal_mpa(&solve_with_rise(N, &exact).element_stress);
    let b = mean_normal_mpa(&solve_with_rise(N, &off_by_one).element_stress);

    let per_kelvin = suppressed_stress_checked(1.0).abs();
    let gap = (a - b).abs();
    assert!(
        (gap - per_kelvin).abs() < STRESS_TOL_MPA,
        "one kelvin of reference must move the stress by {per_kelvin} MPa; \
         measured {gap} MPa"
    );
    assert!(
        gap > 20.0 * STRESS_TOL_MPA,
        "the value tolerance {STRESS_TOL_MPA} MPa must be far below the smallest \
         reference mistake ({gap} MPa), or those asserts could pass with the \
         subtraction wrong"
    );
}

// ---------------------------------------------------------------------------
// 5. control: the rise reaches the residual
// ---------------------------------------------------------------------------

/// A forwarded rise changes the answer; a dropped one would not.
///
/// oracle: with no eigenstrain the clamped cube carries no load, so every
/// stress is **exactly** [`Fix128::ZERO`] (there is nothing to round). With the
/// 40 K rise forwarded it is the suppressed stress. The two must differ by that
/// closed form, which is what rules out a scene that builds a field, asserts on
/// it and then does not pass it to the solve.
#[test]
fn the_rise_actually_reaches_the_residual() {
    const N: usize = 2;
    let mesh = kuhn_cube(N, SIDE / N as f64);
    let bc = clamped_boundary(&mesh, N);

    let bare = solve(&mesh, &pla(), &bc, &solver_config()).expect("the scene is well posed");
    for (e, s) in bare.element_stress.iter().enumerate() {
        for (name, c) in [("xx", s.xx), ("yy", s.yy), ("zz", s.zz)] {
            assert_eq!(
                c,
                Fix128::ZERO,
                "with no eigenstrain element {e} must carry exactly zero σ_{name}"
            );
        }
    }

    let absolute = absolute_field(ABSOLUTE_K);
    let rise = TemperatureRise::from_absolute(&absolute, Fix128::from_int(REFERENCE_K));
    let loaded = solve_with_rise(N, &rise);
    let want = suppressed_stress_checked(RISE_K);
    let dev = worst_normal_deviation(&loaded.element_stress, want);
    assert!(
        dev < STRESS_TOL_MPA,
        "the forwarded rise must move every normal stress to {want} MPa; worst \
         deviation {dev} MPa"
    );
}

// ---------------------------------------------------------------------------
// 6. degenerate inputs — each with the result that is correct for it
// ---------------------------------------------------------------------------

/// `T == T_ref` is no eigenstrain at all, bit-for-bit.
///
/// Expected result: **exactly** the no-eigenstrain solution. This is the
/// contract `ThermalExpansion`'s documentation states ("`ΔT = 0` everywhere
/// must mean no eigenstrain") and the reason [`solve`] can forward [`None`];
/// `Fix128` subtraction is exact, so `T − T = 0` with no residue and the
/// comparison needs no tolerance.
#[test]
fn a_reference_equal_to_the_temperature_is_the_same_as_no_eigenstrain() {
    const N: usize = 2;
    let mesh = kuhn_cube(N, SIDE / N as f64);
    let bc = clamped_boundary(&mesh, N);

    let bare = solve(&mesh, &pla(), &bc, &solver_config()).expect("well posed");

    let absolute = absolute_field(ABSOLUTE_K);
    let zero_rise = TemperatureRise::from_absolute(&absolute, Fix128::from_int(ABSOLUTE_K));
    let loaded = solve_with_rise(N, &zero_rise);

    assert_eq!(
        loaded.element_stress, bare.element_stress,
        "a zero rise must reproduce the no-eigenstrain stress exactly"
    );
    assert_eq!(
        loaded.displacements, bare.displacements,
        "a zero rise must reproduce the no-eigenstrain displacement exactly"
    );
}

/// `α = 0` is no eigenstrain either, whatever the rise is.
///
/// Expected result: **exactly** zero stress. `ε_th = α ΔT I` with `α = 0` is
/// the zero tensor for every finite `ΔT`, and `Fix128` multiplication by zero
/// is exact, so this is an equality rather than a tolerance.
#[test]
fn a_zero_expansion_coefficient_produces_exactly_no_stress() {
    const N: usize = 2;
    let mesh = kuhn_cube(N, SIDE / N as f64);
    let bc = clamped_boundary(&mesh, N);

    let absolute = absolute_field(ABSOLUTE_K);
    let rise = TemperatureRise::from_absolute(&absolute, Fix128::from_int(REFERENCE_K));
    let out = solve_with_eigenstrain(
        &mesh,
        &pla(),
        &bc,
        &solver_config(),
        Some(ThermalExpansion::from_rise(&rise, Fix128::ZERO)),
    )
    .expect("α = 0 is a valid coefficient");

    for (e, s) in out.element_stress.iter().enumerate() {
        for (name, c) in [
            ("xx", s.xx),
            ("yy", s.yy),
            ("zz", s.zz),
            ("xy", s.xy),
            ("yz", s.yz),
            ("zx", s.zx),
        ] {
            assert_eq!(
                c,
                Fix128::ZERO,
                "α = 0 must leave element {e} with exactly zero σ_{name}"
            );
        }
    }
}

/// A reference far above the temperature is a large *negative* rise.
///
/// Expected result: the suppressed-stress closed form, now **tensile**
/// (positive), because suppressing contraction pulls. The reference is `2²⁰`
/// K — physically absurd and deliberately so: it is the sign and the magnitude
/// of the subtraction that are under test, and a 1048511 K drop is still
/// inside the [`Fix128`] range. A relative tolerance is used because the
/// closed form is twelve million MPa.
#[test]
fn a_reference_far_above_the_temperature_gives_a_tensile_stress() {
    const N: usize = 2;
    const HUGE_REFERENCE_K: i64 = 1 << 20;

    let absolute = absolute_field(ABSOLUTE_K);
    let rise = TemperatureRise::from_absolute(&absolute, Fix128::from_int(HUGE_REFERENCE_K));
    let out = solve_with_rise(N, &rise);

    let delta_t = (ABSOLUTE_K - HUGE_REFERENCE_K) as f64;
    let want = suppressed_stress_checked(delta_t);
    assert!(
        want > 0.0,
        "cooling by {} K must put the clamped body in tension; closed form {want} MPa",
        -delta_t
    );

    let dev = worst_normal_deviation(&out.element_stress, want);
    assert!(
        dev < want.abs() * 1e-9,
        "the closed form at ΔT = {delta_t} K is {want} MPa; worst deviation {dev} MPa"
    );
}

/// A single-cell grid carries a uniform rise exactly.
///
/// Expected result: **bit-identical** to the `3³` grid. A degenerate axis is
/// given a cell size of one by `CoupledField::try_new_filled` and
/// `CoupledField::sample` clamps onto the only node, so a uniform field needs
/// no resolution at all. This is the grid/mesh resolution mismatch that is
/// legitimate; the one that is not is the next test.
#[test]
fn a_one_cell_grid_and_a_three_cubed_grid_agree_bit_for_bit() {
    const N: usize = 2;

    let fine = uniform_field(RES, ABSOLUTE_K);
    let coarse = uniform_field(1, ABSOLUTE_K);
    assert_eq!(coarse.cell_count(), 1, "the coarse grid must be one cell");

    let reference = Fix128::from_int(REFERENCE_K);
    let a = solve_with_rise(N, &TemperatureRise::from_absolute(&fine, reference));
    let b = solve_with_rise(N, &TemperatureRise::from_absolute(&coarse, reference));

    assert_eq!(
        a.element_stress, b.element_stress,
        "a uniform rise must not depend on the grid it was stored on"
    );
}

/// A grid that does not cover the mesh is refused, not sampled by clamping.
///
/// Expected result: `Err(FemError::TemperatureFieldDoesNotCoverMesh)` naming a
/// node that really is outside. Clamping would silently extend the boundary
/// value over the uncovered region and produce a plausible wrong stress, so the
/// refusal is the correct behaviour and the returned node index is checked
/// against `CoupledField::contains`.
#[test]
fn a_grid_that_does_not_cover_the_mesh_is_refused() {
    const N: usize = 2;
    let mesh = kuhn_cube(N, SIDE / N as f64);
    let bc = clamped_boundary(&mesh, N);

    // Half the cube in every direction, so the far nodes stick out.
    let half = CoupledField::try_new_filled(
        RES,
        RES,
        RES,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (fx(SIDE / 2.0), fx(SIDE / 2.0), fx(SIDE / 2.0)),
        Fix128::from_int(ABSOLUTE_K),
    )
    .expect("a valid grid over [0, 2]³");
    let rise = TemperatureRise::from_absolute(&half, Fix128::from_int(REFERENCE_K));

    let err = solve_with_eigenstrain(
        &mesh,
        &pla(),
        &bc,
        &solver_config(),
        Some(ThermalExpansion::from_rise(&rise, fx(ALPHA_PER_K))),
    )
    .expect_err("a field covering [0, 2]³ cannot drive a mesh on [0, 4]³");

    match err {
        FemError::TemperatureFieldDoesNotCoverMesh { vertex } => {
            let x = mesh.vertices[vertex as usize];
            let p = Vec3Fix::new(
                fx(f64::from(x[0])),
                fx(f64::from(x[1])),
                fx(f64::from(x[2])),
            );
            assert!(
                !half.contains(p),
                "the refusal must name a node that is actually outside the field, \
                 but node {vertex} at {x:?} mm lies inside [0, 2]³"
            );
        }
        other => panic!(
            "a mesh poking out of its temperature field must be refused as \
             TemperatureFieldDoesNotCoverMesh, not {other:?}"
        ),
    }
}

/// A `TemperatureRise` cannot be built on an empty grid, because the field
/// cannot be.
///
/// Expected result: `Err(CoupledFieldError::EmptyGrid)` from
/// `CoupledField::try_new_filled`, before a rise exists.
/// `TemperatureRise::from_absolute` takes an already validated field, so "the
/// empty field" is not a case it has to handle — it is a case the constructor
/// of the field refuses first, and this test records that the guard is upstream
/// rather than missing.
#[test]
fn an_empty_grid_is_refused_before_a_rise_can_exist() {
    let err = CoupledField::try_new_filled(
        0,
        RES,
        RES,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (
            Fix128::from_int(SIDE as i64),
            Fix128::from_int(SIDE as i64),
            Fix128::from_int(SIDE as i64),
        ),
        Fix128::from_int(ABSOLUTE_K),
    )
    .expect_err("a zero dimension must be refused");
    assert!(
        matches!(
            err,
            alice_physics::coupled_field::CoupledFieldError::EmptyGrid { .. }
        ),
        "a zero dimension must be reported as EmptyGrid, not {err:?}"
    );
}
