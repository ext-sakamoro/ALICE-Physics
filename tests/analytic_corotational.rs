//! Oracles for the co-rotational solve.
//!
//! `tests/analytic_large_rotation.rs` pins the *kinematics*: given a rigid
//! rotation as boundary data on **every** node, the stress must be zero. That
//! test prescribes all degrees of freedom, so no equation is ever solved — it
//! cannot say anything about the solver, only about the strain measure.
//!
//! This file pins the *solve*. The boundary is prescribed and **the interior is
//! free**, so the solver has to find the rigid motion itself. That is what makes
//! the tests here sensitive to the one design decision a co-rotational
//! formulation has to make:
//!
//! # ⚠️ When is the element rotation updated?
//!
//! `Kₑ = R Kₑ⁰ Rᵀ` needs an `R` per element, and there are two policies:
//!
//! - **recompute `R` from the current displacement at every Newton iteration**
//!   (what the implementation does), or
//! - **freeze `R`** at the initial iterate.
//!
//! Because [`alice_physics::linear_elastic_fem::solve`] is a *static* solver —
//! `K u = f`, no time stepping — "freeze at the start of the step" means freeze at
//! `u = 0`, which is `R = I`, which is the small-strain solver. So the frozen
//! policy is not a variant that needs writing to be tested: **the existing linear
//! `solve` is it**, and `the_linear_solver_cannot_reach_zero_stress` measures what
//! it does on the same scene. That is the destruction test for the update policy,
//! and it costs nothing.
//!
//! # Destruction tests, re-measured 2026-10-01
//!
//! One mutation at a time, restored in between. ⚠️ **The whole table was
//! re-measured when the material law was wired in**: the 2026-09-30 numbers were
//! taken when the stress path had no branch, and a branch can move which oracle
//! sees which mutation. The baseline is `13 passed / 0 failed / 1 ignored` here
//! (14 tests), `21 passed` in `cargo test --lib hyperelastic` and `2 passed` in
//! `cargo test --lib linear_elastic_fem`.
//!
//! ⚠️ **Run the whole mutation through one `cargo` invocation**, e.g.
//! `cargo test --no-fail-fast --lib --test analytic_corotational`. Two mutations
//! in the table below red only in the library target, and splitting the run puts
//! the marker and the red in different logs, where nobody outside can pair them.
//!
//! ⚠️ **A mutation run must print `### MUTATION: <what was changed>` on its own
//! line**, and the restored green must land in the same log. A destruction test
//! produces a real red on purpose, and anyone reading the output from outside —
//! another session triaging failures, a later reader of a log — cannot otherwise
//! tell it from a genuine break. The marker costs one line and no coordination;
//! announcing mutations over messages instead does not scale, because destruction
//! tests come in batches.
//!
//! | mutation | oracles that went red |
//! | --- | --- |
//! | `rotate_stress` returns `σ̃` unrotated (drop the `R` in `σ = R σ̃ Rᵀ`) | 2: `the_corotational_linear_law_is_what_the_element_returns_without_a_material` and `rotated_uniform_stretch_matches_the_closed_form` |
//! | `corotational_local_stress` drops the `Rᵀ` (`ε = sym(RᵀF − I)` → `sym(F − I)`) | **10** |
//! | `solve_corotational` ignores the configured model (`law` forced to `None`) | 2: `the_neo_hookean_deviator_is_what_the_element_returns_under_that_law` and `the_material_law_moves_the_answer_and_the_answer_is_equilibrium` |
//! | `energy_derivatives` returns `μ` instead of `μ/2` for Neo-Hookean | the same 2 |
//! | `material_correction` returns without accumulating | 2: `the_material_law_moves_…` and `the_solve_is_equivariant_…`, both `NotConverged` |
//! | `hyperelastic_stress` adds `+1 MPa` to `σ_xx` (a world-fixed axis) | **0 here**, 2 in `cargo test --lib linear_elastic_fem` — see `the_solve_is_equivariant_…` for why a non-objective bias is invisible to a solve |
//! | `cauchy_stress` flips the sign of `K·(J−1)` | 1: `the_material_law_moves_…` |
//! | `hyperelastic_stress` drops the `J` in `P = J σ F⁻ᵀ` | 1: `the_material_law_moves_…` |
//! | `hyperelastic_stress` transposes the product (`F⁻ᵀσ` for `σF⁻ᵀ`) | 1: `the_material_law_moves_…` |
//! | `cauchy_stress` drops the `−p_ref` that makes the reference state stress free | **0 here**, 2 in `cargo test --lib hyperelastic` |
//! | `hyperelastic_stress` drops the `F⁻ᵀ` in `P = J σ F⁻ᵀ` | measured `12 passed / 1 failed / 1 ignored` here (`the_solve_is_equivariant_…`) + 2 in `cargo test --lib linear_elastic_fem`; **0 anywhere before those three existed** |
//!
//! Three things that table says and the prose would not:
//!
//! - The first mutation is confined to the reporting path of the *linear* law, so
//!   it separates cleanly and the two material oracles do not see it. The second
//!   is in the residual and in the frames the material path also solves against,
//!   so it takes almost everything down — including
//!   `characterises_which_stretches_the_corotational_solve_reaches`, which means
//!   that test is sensitive to the strain measure and not only to the stopping
//!   rule it is named for.
//! - ⚠️ **A uniform stress is invisible to a solve.** It puts zero force on an
//!   interior node whatever it is, so a mutation that shifts every element's
//!   stress by the same tensor — dropping `−p_ref` — leaves every oracle in this
//!   file green. That one is checked in `src/hyperelastic.rs` instead, by
//!   `cauchy_stress_vanishes_in_the_reference_state`.
//! - ⚠️ **Dropping the `F⁻ᵀ` survived every oracle in the crate until 2026-10-01**,
//!   and still survives every oracle in this file except the one added that day.
//!   The affine scenes cannot see it for the reason above, and on the
//!   inhomogeneous scene the solver's own equilibrium check uses the same broken
//!   force, so it converges to a different state and calls it equilibrium —
//!   `the_material_law_moves_…` still returns `Ok`. What
//!   catches it is an oracle the internal force cannot satisfy by agreeing with
//!   itself: `P = J σ F⁻ᵀ` rotates to `Q P` under a superposed `Q` where
//!   `P = J σ` rotates to `Q P Qᵀ`. Both now live in `src/linear_elastic_fem.rs`
//!   (`the_first_piola_kirchhoff_stress_is_objective_under_a_superposed_rotation`
//!   and `…_is_what_the_closed_form_says`), because `P` is not on the public
//!   surface — a test in `tests/` can see it only through where a solve lands.
//!   ⚠️ **`the_solve_is_equivariant_…` below is the companion, not the oracle**: it
//!   does red on this mutation, but as a `RotationFailed` on the turned scene
//!   rather than as a measured disagreement, so it cannot say *which* part of the
//!   force was wrong. Its own comment has the four measurements.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::hyperelastic::{
    strain_energy_density, uniaxial_cauchy_stress, HyperelasticModel, Stretch,
};
use alice_physics::linear_elastic_fem::{
    solve, solve_corotational, Axis, BoundaryConditions, CorotationalConfig, ElasticMaterial,
    FemError, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

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

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;
/// Side of the cube, mm.
const SIDE: f64 = 4.0;

fn pla() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and ν in (-1, 0.5)")
}

fn lame() -> (f64, f64) {
    (
        E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        E_MPA / (2.0 * (1.0 + NU)),
    )
}

fn vert(mesh: &SdfTetMesh, v: u32) -> [f64; 3] {
    let p = mesh.vertices[v as usize];
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

/// Rotation about z by an angle given as an exact `(cos, sin)` pair.
#[derive(Clone, Copy)]
struct Turn {
    name: &'static str,
    cos: f64,
    sin: f64,
}

/// A quarter turn: exactly representable, and the strongest case.
const QUARTER: Turn = Turn {
    name: "90° about z",
    cos: 0.0,
    sin: 1.0,
};

/// `cos = 4/5`, `sin = 3/5` — about 36.87°, large enough that the small-strain
/// reading is wrong by 20% of `E` and small enough that Newton reaches it without
/// help.
const THREE_FOUR_FIVE: Turn = Turn {
    name: "36.87° about z (3-4-5)",
    cos: 4.0 / 5.0,
    sin: 3.0 / 5.0,
};

/// `u = (R − I)·x`, the exact rigid motion.
fn rigid_displacement(turn: Turn, p: [f64; 3]) -> [f64; 3] {
    let [x, y, _z] = p;
    [
        turn.cos * x - turn.sin * y - x,
        turn.sin * x + turn.cos * y - y,
        0.0,
    ]
}

/// Prescribe the rigid rotation on the **boundary only**, leaving the interior
/// free. Returns the conditions and the interior node indices.
///
/// The exact solution is the rigid motion everywhere: zero stress satisfies
/// equilibrium and matches the boundary data, and an elliptic problem has only
/// one solution. It is also exactly representable — `u = (R − I)·x` is linear in
/// `x`, so P1 can carry it with no discretisation error at all. **The only thing
/// standing between the solver and the exact answer is the strain measure.**
fn boundary_rotation(mesh: &SdfTetMesh, turn: Turn) -> (BoundaryConditions, Vec<u32>) {
    let eps = SIDE * 1e-9;
    let mut bc = BoundaryConditions::new();
    let mut interior = Vec::new();
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let p = vert(mesh, v);
        let on_boundary = p.iter().any(|&c| c < eps || c > SIDE - eps);
        if on_boundary {
            let u = rigid_displacement(turn, p);
            bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            interior.push(v);
        }
    }
    (bc, interior)
}

fn corotational_config(increments: u32) -> CorotationalConfig {
    corotational_config_with_newton_budget(increments, 32)
}

/// Same, with the per-increment Newton budget spelled out.
///
/// ⚠️ **The Newton budget is the only field that differs from
/// [`corotational_config`]** — same linear `SolverConfig`, same Newton tolerance,
/// same polar budget, same increment count. That restriction is deliberate: the
/// ignored twin below has to red for one reason (no material law), so nothing
/// else about the solve may move underneath it.
///
/// The 32 [`corotational_config`] uses is a *frame-settling* budget, not an
/// equilibrium one: on the last increment the iteration runs until the element
/// frames stop moving, and only then is the residual checked. Large stretches
/// need more of those steps than large rotations do — at the 125 % stretch the
/// twins use, the **measured floor is 47** (46 refuses) and the twins pass 64 for
/// headroom. So a scene can be reported as `NotConverged` with a residual that is
/// already well *inside* tolerance. See
/// `characterises_which_stretches_the_corotational_solve_reaches`.
fn corotational_config_with_newton_budget(
    increments: u32,
    newton_iterations: u32,
) -> CorotationalConfig {
    CorotationalConfig::try_new(
        SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid linear config"),
        newton_iterations,            // Newton iterations per increment
        Fix128::from_raw(0, 1 << 34), // Newton relative tolerance, 2^-30
        increments,
        32, // polar iteration budget, from the measured 6-19
    )
    .expect("valid co-rotational config")
}

/// Largest stress component over the mesh, MPa.
fn worst_stress(stresses: &[alice_physics::linear_elastic_fem::StressTensor]) -> f64 {
    let mut worst = 0.0_f64;
    for s in stresses {
        for c in [s.xx, s.yy, s.zz, s.xy, s.yz, s.zx] {
            worst = worst.max(c.to_f64().abs());
        }
    }
    worst
}

/// Largest interior displacement error against the rigid motion, mm.
fn worst_interior_error(
    mesh: &SdfTetMesh,
    interior: &[u32],
    displacements: &[[Fix128; 3]],
    turn: Turn,
) -> f64 {
    let mut worst = 0.0_f64;
    for &v in interior {
        let want = rigid_displacement(turn, vert(mesh, v));
        for axis in 0..3 {
            worst = worst.max((displacements[v as usize][axis].to_f64() - want[axis]).abs());
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// the discriminating oracle
// ---------------------------------------------------------------------------

/// **The gate for the co-rotational solve.**
///
/// A rigid rotation imposed on the boundary must propagate into the free interior
/// as the same rigid motion, with zero stress. Unlike the all-prescribed test this
/// requires the solver to *find* the motion, so it is sensitive to how the element
/// rotation is updated.
#[test]
fn boundary_rigid_rotation_leaves_the_interior_unstressed() {
    for turn in [THREE_FOUR_FIVE, QUARTER] {
        let mesh = kuhn_cube(4, SIDE / 4.0);
        let (bc, interior) = boundary_rotation(&mesh, turn);
        assert_eq!(interior.len(), 27, "a 4-cell cube has 27 interior nodes");

        let out = solve_corotational(&mesh, &pla(), &bc, &corotational_config(4))
            .unwrap_or_else(|e| panic!("{}: {e:?}", turn.name));

        let stress = worst_stress(&out.field.element_stress);
        let drift = worst_interior_error(&mesh, &interior, &out.field.displacements, turn);
        eprintln!(
            "  {}: worst |σ| = {stress:.6e} MPa, worst interior |u − u_exact| = {drift:.6e} mm, \
             {} Newton iterations over {} increments",
            turn.name, out.newton_iterations, out.increments
        );

        assert!(
            drift < 1e-6,
            "{}: the interior must follow the boundary rigidly; worst node is {drift:.6e} mm \
             from (R − I)·x. The rigid motion is linear in x, so P1 carries it exactly and \
             this is not a discretisation error",
            turn.name
        );
        assert!(
            stress < 1e-3,
            "{}: a rigid motion strains nothing, so the stress must vanish; worst component is \
             {stress:.6e} MPa on a {E_MPA} MPa material",
            turn.name
        );
    }
}

/// **The destruction test for the update policy**, at no cost.
///
/// Freezing the element rotation at the initial iterate is the same thing as not
/// having one, and the crate already ships that solver: `solve`. On the scene
/// above it must leave a stress of the order of `E`, which is the evidence that
/// the test above is measuring the rotation update and not something else.
///
/// The closed form is the small-strain reading of the rotation:
/// `σ_xx = 2λ(cos θ − 1) + 2μ(cos θ − 1)`. The linear solve does not reproduce it
/// exactly here — the interior is free, so it relaxes toward something else — but
/// it cannot get near zero, and the bound below is that it stays within a factor
/// of ten of the prescribed-everywhere value.
#[test]
fn the_linear_solver_cannot_reach_zero_stress() {
    let (lambda, mu) = lame();
    for turn in [THREE_FOUR_FIVE, QUARTER] {
        let c1 = turn.cos - 1.0;
        let prescribed_everywhere = (2.0 * lambda * c1 + 2.0 * mu * c1).abs();

        let mesh = kuhn_cube(4, SIDE / 4.0);
        let (bc, _) = boundary_rotation(&mesh, turn);
        let config = SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid");
        let out = solve(&mesh, &pla(), &bc, &config).expect("well posed");

        let stress = worst_stress(&out.element_stress);
        eprintln!(
            "  {}: linear solve leaves worst |σ| = {stress:.3} MPa \
             (all-prescribed closed form {prescribed_everywhere:.3} MPa)",
            turn.name
        );
        assert!(
            stress > prescribed_everywhere / 10.0,
            "{}: the linear solver left only {stress:.3} MPa where the small-strain reading of \
             this rotation is {prescribed_everywhere:.3} MPa. If the linear solver can reach \
             near-zero stress on this scene then the scene does not discriminate, and \
             `boundary_rigid_rotation_leaves_the_interior_unstressed` proves nothing about the \
             rotation update",
            turn.name
        );
    }
}

// ---------------------------------------------------------------------------
// precision-parameter independence
// ---------------------------------------------------------------------------

/// The converged answer must not depend on the Newton budget.
///
/// Per the analytic-oracle rule, a precision parameter that changes the result is
/// a semantics bug. Budgets above what the problem needs must all give the same
/// answer bit for bit.
#[test]
fn the_answer_does_not_depend_on_the_newton_budget() {
    let mesh = kuhn_cube(4, SIDE / 4.0);
    let (bc, _) = boundary_rotation(&mesh, THREE_FOUR_FIVE);

    let mut reference: Option<Vec<[Fix128; 3]>> = None;
    for budget in [8_u32, 16, 32, 64] {
        let config = CorotationalConfig::try_new(
            SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid"),
            budget,
            Fix128::from_raw(0, 1 << 34),
            4,
            32,
        )
        .expect("valid");
        let out = solve_corotational(&mesh, &pla(), &bc, &config)
            .unwrap_or_else(|e| panic!("budget {budget}: {e:?}"));
        eprintln!(
            "  Newton budget {budget}: {} iterations, worst |σ| = {:.3e} MPa",
            out.newton_iterations,
            worst_stress(&out.field.element_stress)
        );
        match &reference {
            None => reference = Some(out.field.displacements),
            Some(first) => assert_eq!(
                &out.field.displacements, first,
                "a Newton budget of {budget} gave a different answer from the smallest budget \
                 that converged. A budget above what the problem needs must not change the \
                 result"
            ),
        }
    }
}

/// The converged answer must not depend on how the rotation is applied in
/// increments.
///
/// Incremental application exists so Newton has a nearby starting point for a
/// large rotation; it is not part of the answer. If one increment converges its
/// answer must match nine.
///
/// # ⚠️ Why this is ignored, and what is not lost
///
/// It asks for bit-identical displacements and the solver reaches four units in
/// the last place, 2.2e-19 mm. That is not a tolerance waiting to be tightened:
/// the outer iteration is `R ← polar(F(u(R)))`, and it converges to a spread of
/// a few units in the last place and then wanders inside that spread forever
/// rather than landing on a point that maps to itself. A fixed point of the
/// frame map has to exist before any stopping rule can be asked to find it, and
/// measurement says it does not — see `solve_corotational`'s note on the path.
///
/// ⚠️ **Do not delete it, and do not weaken the assertion to 4 units.** The red
/// is the standing record that the co-rotational path is the one place in this
/// crate where the answer is not a pure function of the problem statement, and a
/// weakened version would stop saying so.
///
/// Ignoring it costs no coverage, because
/// [`the_increment_spread_stays_at_the_arithmetic_floor`] is **not** ignored and
/// pins the same quantity from the other side: it asserts the spread is *at
/// least one* unit in the last place — so the day a formulation lands that does
/// reach a fixed point, that companion reds and sends the reader here — and *at
/// most eight*, so any return toward the 3.5e10 units the residual stopping rule
/// used to leave reds it as well.
#[test]
#[ignore = "the red is correct: bit-identical displacements across increment \
            counts need an exact fixed point of the frame map, and measurement \
            says the map has none — it converges to a spread of a few units in \
            the last place and wanders inside it. Remove this attribute in the \
            commit that lands a formulation with a constructive fixed point, and \
            delete `the_increment_spread_stays_at_the_arithmetic_floor` in the \
            same diff. CI coverage is not lost: that companion test is not \
            ignored and pins the same spread from both sides"]
fn the_answer_does_not_depend_on_the_increment_count() {
    let mesh = kuhn_cube(4, SIDE / 4.0);
    let (bc, _) = boundary_rotation(&mesh, THREE_FOUR_FIVE);

    let mut converged: Vec<(u32, Vec<[Fix128; 3]>)> = Vec::new();
    for increments in [1_u32, 2, 4, 9] {
        match solve_corotational(&mesh, &pla(), &bc, &corotational_config(increments)) {
            Ok(out) => {
                eprintln!(
                    "  {increments} increment(s): {} Newton iterations, worst |σ| = {:.3e} MPa",
                    out.newton_iterations,
                    worst_stress(&out.field.element_stress)
                );
                converged.push((increments, out.field.displacements));
            }
            Err(e) => eprintln!("  {increments} increment(s): {e:?} (reported, not asserted)"),
        }
    }

    assert!(
        !converged.is_empty(),
        "no increment count converged, so there is nothing to compare"
    );
    let (first_n, first) = &converged[0];
    for (n, later) in &converged[1..] {
        assert_eq!(
            later, first,
            "{n} increments gave a different answer from {first_n}. Incremental application is \
             a path to the answer, not part of it"
        );
    }
}

/// One unit in the last place of [`Fix128`], in mm.
const ULP_MM: f64 = 5.421_010_862_427_522e-20;

/// Displacement as a raw two's-complement 128-bit integer, so two answers can be
/// differenced in units in the last place rather than in `f64`, which cannot
/// represent the difference.
fn raw(v: Fix128) -> i128 {
    (i128::from(v.hi) << 64) | i128::from(v.lo)
}

/// Worst per-component difference between two displacement fields, in units in
/// the last place.
fn worst_ulp(a: &[[Fix128; 3]], b: &[[Fix128; 3]]) -> u128 {
    let mut worst = 0_u128;
    for (x, y) in a.iter().zip(b.iter()) {
        for axis in 0..3 {
            worst = worst.max((raw(x[axis]) - raw(y[axis])).unsigned_abs());
        }
    }
    worst
}

/// **The guard for [`the_answer_does_not_depend_on_the_increment_count`]**, which
/// is ignored because the spread it asks to be zero cannot be.
///
/// This pins the spread from both sides, so the pair covers what the ignored test
/// would have covered:
///
/// - **at least one unit in the last place.** If a later formulation does reach a
///   constructive fixed point of the frame map, this reds, and its message says
///   to remove the `#[ignore]` from that test and delete this one in the same
///   diff. A guard that only bounded the spread from above would silently stay
///   green through the good news and the ignored test would sit red forever.
/// - **at most eight units in the last place.** The stopping rule is what decides
///   this number. When the last increment stopped on a residual *tolerance*
///   instead of on settled frames, the same six pairs differed by up to 3.5e10
///   units — because a one-increment run met the threshold after a single frame
///   update where a nine-increment run had had several. Anything that puts a
///   tolerance back into where the last increment stops reds this.
///
/// Measured: 4 units at 36.87° and 4 at 90°, as the worst over the six pairs of
/// 1, 2, 4 and 9 increments; 5 is the largest seen over a wider sweep of angles
/// and increment counts, recorded in `solve_corotational`'s table.
#[test]
fn the_increment_spread_stays_at_the_arithmetic_floor() {
    for turn in [THREE_FOUR_FIVE, QUARTER] {
        let mesh = kuhn_cube(4, SIDE / 4.0);
        let (bc, _) = boundary_rotation(&mesh, turn);

        let mut runs: Vec<(u32, Vec<[Fix128; 3]>)> = Vec::new();
        for increments in [1_u32, 2, 4, 9] {
            let out = solve_corotational(&mesh, &pla(), &bc, &corotational_config(increments))
                .unwrap_or_else(|e| panic!("{}: {increments} increment(s): {e:?}", turn.name));
            runs.push((increments, out.field.displacements));
        }

        let mut worst = 0_u128;
        for (i, (n, a)) in runs.iter().enumerate() {
            for (m, b) in &runs[i + 1..] {
                let d = worst_ulp(a, b);
                eprintln!("  {}: {n} vs {m} increments: {d} ulp", turn.name);
                worst = worst.max(d);
            }
        }
        eprintln!(
            "  {}: worst over the six pairs = {worst} ulp = {:.3e} mm",
            turn.name,
            worst as f64 * ULP_MM
        );

        assert!(
            worst >= 1,
            "{}: the four increment counts agreed bit for bit on all six pairs, which \
             `the_answer_does_not_depend_on_the_increment_count` asks for and this crate could \
             not deliver. If that is now true, remove the `#[ignore]` from that test and delete \
             this one in the same diff",
            turn.name
        );
        assert!(
            worst <= 8,
            "{}: the increment counts differ by {worst} units in the last place ({:.3e} mm), \
             above the eight this solver settles at. The stopping rule decides this number: a \
             residual tolerance deciding where the **last** increment stops left 3.5e10 units \
             here, because a one-increment run meets the threshold after a single frame update \
             where a nine-increment run has had several. Check that the last increment still \
             stops only on settled frames",
            turn.name,
            worst as f64 * ULP_MM
        );
    }
}

// ---------------------------------------------------------------------------
// the small-rotation limit
// ---------------------------------------------------------------------------

/// In the small-rotation limit the co-rotational solve must agree with the linear
/// one.
///
/// This is the consistency check that keeps the new path from being a different
/// physics: at a rotation of a thousandth of a radian the geometric correction is
/// of order `θ²/2 ≈ 5e-7`, so the two answers must agree to about that relative
/// size. A co-rotational implementation that disagreed here would be wrong about
/// the *material*, not about the rotation.
#[test]
fn the_small_rotation_limit_agrees_with_the_linear_solver() {
    let theta = 1.0e-3_f64;
    let turn = Turn {
        name: "1 mrad about z",
        cos: 1.0 - theta * theta / 2.0,
        sin: theta,
    };
    let mesh = kuhn_cube(4, SIDE / 4.0);
    let (bc, interior) = boundary_rotation(&mesh, turn);

    let linear_config =
        SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid");
    let linear = solve(&mesh, &pla(), &bc, &linear_config).expect("well posed");
    let coro = solve_corotational(&mesh, &pla(), &bc, &corotational_config(1)).expect("converges");

    let mut worst = 0.0_f64;
    for &v in &interior {
        for axis in 0..3 {
            let a = linear.displacements[v as usize][axis].to_f64();
            let b = coro.field.displacements[v as usize][axis].to_f64();
            worst = worst.max((a - b).abs());
        }
    }
    let scale = SIDE * theta; // the size of the displacements themselves
    eprintln!(
        "  1 mrad: worst |u_linear − u_corotational| = {worst:.3e} mm, displacement scale \
         {scale:.3e} mm, ratio {:.3e}",
        worst / scale
    );
    assert!(
        worst / scale < 1e-4,
        "at 1 mrad the geometric correction is of order θ²/2 ≈ 5e-7, so the two solvers must \
         agree to far better than 1e-4 of the displacement scale; got {:.3e}",
        worst / scale
    );
}

// ---------------------------------------------------------------------------
// non-zero internal force
// ---------------------------------------------------------------------------

/// **Large rotation with real deformation**, which the rigid tests cannot reach.
///
/// Every test above imposes a *rigid* motion, so the exact internal force is zero
/// and the material law is never exercised. A solver could get all of them right
/// and still be wrong about stress under rotation. This one superposes a uniform
/// stretch on the rotation and has a closed form for both.
///
/// Take `F = R·U` with `U` a constant symmetric positive definite stretch and
/// impose `u = (R·U − I)·X` on the boundary. The field is affine, so P1 carries it
/// exactly and the interior must reproduce it. The co-rotational strain is then
///
/// ```text
/// ε = sym(Rᵀ F − I) = sym(U − I) = U − I        (U is symmetric)
/// σ̃ = λ tr(ε) I + 2μ ε                          (in the rotated frame)
/// σ  = R σ̃ Rᵀ                                   (Cauchy, global frame)
/// ```
///
/// which is a closed form for the **co-rotational model** — not for finite-strain
/// elasticity, which this is not claiming to be. That distinction is the point: it
/// pins what the implemented model says, so a later move to a full finite-strain
/// measure has to change this test deliberately rather than silently.
///
/// ⚠️ This is also the test that separates a Newton on the co-rotational residual
/// from re-solving `(R Kₑ⁰ Rᵀ)·u = f` with an updated `R` each iteration. The two
/// differ by the `(Rᵀ − I)·X` term in `Rᵀx − X`, and this scene has both a
/// non-zero `X` contribution and a non-zero strain.
#[test]
fn rotated_uniform_stretch_matches_the_closed_form() {
    let (lambda, mu) = lame();
    let turn = THREE_FOUR_FIVE;
    // U = diag(1.02, 0.99, 1.005): a 2% stretch, well inside small strain in the
    // rotated frame, so the co-rotational model is the right one to compare with.
    let stretch = [1.02_f64, 0.99, 1.005];

    let mesh = kuhn_cube(4, SIDE / 4.0);
    let eps = SIDE * 1e-9;
    let mut bc = BoundaryConditions::new();
    let mut interior = Vec::new();
    // u = (R·U − I)·X
    let field = |p: [f64; 3]| {
        let s = [p[0] * stretch[0], p[1] * stretch[1], p[2] * stretch[2]];
        [
            turn.cos * s[0] - turn.sin * s[1] - p[0],
            turn.sin * s[0] + turn.cos * s[1] - p[1],
            s[2] - p[2],
        ]
    };
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let p = vert(&mesh, v);
        if p.iter().any(|&c| c < eps || c > SIDE - eps) {
            let u = field(p);
            bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            interior.push(v);
        }
    }

    let out = solve_corotational(&mesh, &pla(), &bc, &corotational_config(2))
        .expect("a 2% stretch under rotation converges");

    // interior must reproduce the affine field
    let mut drift = 0.0_f64;
    for &v in &interior {
        let want = field(vert(&mesh, v));
        for (axis, want_axis) in want.iter().enumerate() {
            drift =
                drift.max((out.field.displacements[v as usize][axis].to_f64() - want_axis).abs());
        }
    }

    // closed-form stress: ε = U − I, σ̃ = λ tr(ε) I + 2μ ε, σ = R σ̃ Rᵀ
    let e = [stretch[0] - 1.0, stretch[1] - 1.0, stretch[2] - 1.0];
    let trace = e[0] + e[1] + e[2];
    let s_local = [
        lambda * trace + 2.0 * mu * e[0],
        lambda * trace + 2.0 * mu * e[1],
        lambda * trace + 2.0 * mu * e[2],
    ];
    // R σ̃ Rᵀ for a rotation about z and a diagonal σ̃
    let (c, s) = (turn.cos, turn.sin);
    let want_xx = c * c * s_local[0] + s * s * s_local[1];
    let want_yy = s * s * s_local[0] + c * c * s_local[1];
    let want_zz = s_local[2];
    let want_xy = c * s * (s_local[0] - s_local[1]);

    let mut worst = 0.0_f64;
    for st in &out.field.element_stress {
        for (got, want) in [
            (st.xx.to_f64(), want_xx),
            (st.yy.to_f64(), want_yy),
            (st.zz.to_f64(), want_zz),
            (st.xy.to_f64(), want_xy),
            (st.yz.to_f64(), 0.0),
            (st.zx.to_f64(), 0.0),
        ] {
            worst = worst.max((got - want).abs());
        }
    }

    eprintln!(
        "  rotated 2% stretch: interior drift {drift:.3e} mm, worst |σ − σ_exact| = {worst:.6} MPa"
    );
    eprintln!(
        "    closed form σ = ({want_xx:.4}, {want_yy:.4}, {want_zz:.4}) diag, σ_xy = {want_xy:.4} MPa"
    );
    assert!(
        drift < 1e-6,
        "the affine field is exactly representable by P1, so the interior must reproduce it; \
         worst node is {drift:.3e} mm off"
    );
    assert!(
        worst < 1e-2,
        "the co-rotational stress must match R (D:(U−I)) Rᵀ; worst component is {worst:.6} MPa \
         off a state whose largest component is {:.3} MPa",
        want_xx.abs().max(want_yy.abs()).max(want_zz.abs())
    );
}

// ---------------------------------------------------------------------------
// the hyperelastic twins
// ---------------------------------------------------------------------------
//
// Two oracles on **one geometry**, one per constitutive law:
//
// - `the_neo_hookean_deviator_is_what_the_element_returns_under_that_law` — the
//   deviatoric Cauchy stress an incompressible Neo-Hookean solid carries at this
//   deformation, asked of a solve configured with
//   `CorotationalConfig::with_hyperelastic`.
// - `the_corotational_linear_law_is_what_the_element_returns_without_a_material`
//   — the same geometry with no material set, under the co-rotational *linear*
//   law.
//
// # ⚠️ How the reversal condition was discharged, and where it was not followed
//
// Until `with_hyperelastic` existed, the first of these was `#[ignore]`d as
// `the_neo_hookean_deviator_is_not_what_the_element_returns`, and the comment
// here said, verbatim:
//
// > **On the commit that makes the material law swappable through
// > `HyperelasticModel`, drop the `#[ignore]` from … and flip
// > `the_corotational_linear_law_is_what_the_element_returns_today` to
// > `#[ignore = "superseded"]`.** Both edits belong in that one diff: keeping the
// > linear pin green next to a green Neo-Hookean oracle would assert two
// > different answers for one scene.
//
// The `#[ignore]` was dropped. **The linear pin was kept green**, against the
// letter of that instruction, because its stated reason stopped holding the
// moment the law became swappable: the law is now *part of the scene*, so the
// two tests no longer name one scene with two answers — they name two scenes.
// Ignoring the linear one would have deleted the only test the destruction table
// below is written against, which the instruction itself notes an `#[ignore]`d
// oracle cannot replace. Both were renamed, because "is not what the element
// returns" and "today" had both become false.
//
// # ⚠️ What these two do **not** cover
//
// The solution of this scene is affine, and an affine displacement is an exact
// equilibrium under **every** homogeneous law — uniform stress puts zero force on
// an interior node whatever the stress is. So the two above pin the constitutive
// law and say nothing about whether the *solve* reaches the material's root:
// measured, the material run here takes one Newton step per increment, the same
// as the linear one, and never forms the correction term that carries the law
// into the step. `the_material_law_moves_the_answer_and_the_answer_is_equilibrium`
// is the one that does, on an inhomogeneous scene.
//
// # Why the stretch has to be isochoric
//
// `hyperelastic.rs` carries no volumetric term — its module doc says "All models
// assume incompressibility (`J = λ₁·λ₂·λ₃ = 1`)", and `W = μ/2·(I₁ − 3)` for
// Neo-Hookean. For an incompressible solid the pressure is not a function of the
// deformation; it is whatever the constraint needs it to be. So a stretch with
// `J ≠ 1` has **no** Neo-Hookean stress to compare against, and `U` is chosen on
// the exactly-rational isochoric family `diag(a², 1/a, 1/a)` (`det U = 1` for
// every `a`) with `a = 3/2`.
//
// # Why only the deviator
//
// Every boundary node is prescribed here, so nothing in the scene fixes the
// hydrostatic part: `σ = μ B − p I` holds for any `p`. Both oracles therefore
// subtract `tr σ / 3 · I` from the measured stress and compare deviators only.
// **The trace is never compared.**

/// `U = diag(9/4, 2/3, 2/3)` — `a = 3/2` on `diag(a², 1/a, 1/a)`, so `det U = 1`
/// exactly. A 125 % first principal stretch: far outside small strain, which is
/// the point — it is where the linear and the Neo-Hookean deviators are furthest
/// apart (measured: 1254 MPa on the first component, 46 % of the linear value).
const ISOCHORIC_U: [f64; 3] = [9.0 / 4.0, 2.0 / 3.0, 2.0 / 3.0];

/// Newton budget the twins run with.
///
/// ⚠️ **Not the house default of 32.** At this stretch the frame iteration needs
/// 47 steps on the final increment, and the budget check fires before the
/// residual is looked at, so 32 returns `NotConverged` with
/// `relative_residual = 0.217` — a residual 4.6× *inside* tolerance.
/// `characterises_which_stretches_the_corotational_solve_reaches` pins that threshold.
const STRETCH_NEWTON_BUDGET: u32 = 64;

/// `u = (R·U − I)·X`, the affine field the twins prescribe.
fn rotated_stretch_field(turn: Turn, stretch: [f64; 3], p: [f64; 3]) -> [f64; 3] {
    let s = [p[0] * stretch[0], p[1] * stretch[1], p[2] * stretch[2]];
    [
        turn.cos * s[0] - turn.sin * s[1] - p[0],
        turn.sin * s[0] + turn.cos * s[1] - p[1],
        s[2] - p[2],
    ]
}

/// Prescribe [`rotated_stretch_field`] on the boundary, interior free.
fn boundary_rotated_stretch(
    mesh: &SdfTetMesh,
    turn: Turn,
    stretch: [f64; 3],
) -> (BoundaryConditions, Vec<u32>) {
    let eps = SIDE * 1e-9;
    let mut bc = BoundaryConditions::new();
    let mut interior = Vec::new();
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let p = vert(mesh, v);
        if p.iter().any(|&c| c < eps || c > SIDE - eps) {
            let u = rotated_stretch_field(turn, stretch, p);
            bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            interior.push(v);
        }
    }
    (bc, interior)
}

/// `σ̃ = Rᵀ σ R` for a rotation about z, then `dev σ̃ = σ̃ − tr σ / 3 · I`.
///
/// Returns `(dev σ̃₁₁, dev σ̃₂₂, dev σ̃₃₃, σ̃₁₂)`. The trace is taken from the
/// unrotated stress because it is rotation invariant, which is also a cheap check
/// that the rotation above is the right way round.
fn deviator_in_the_stretch_frame(
    turn: Turn,
    s: &alice_physics::linear_elastic_fem::StressTensor,
) -> [f64; 4] {
    let (c, sn) = (turn.cos, turn.sin);
    let (sxx, syy, szz, sxy) = (s.xx.to_f64(), s.yy.to_f64(), s.zz.to_f64(), s.xy.to_f64());
    let t11 = c * c * sxx + 2.0 * c * sn * sxy + sn * sn * syy;
    let t22 = sn * sn * sxx - 2.0 * c * sn * sxy + c * c * syy;
    let t12 = c * sn * (syy - sxx) + (c * c - sn * sn) * sxy;
    let third = (sxx + syy + szz) / 3.0;
    [t11 - third, t22 - third, szz - third, t12]
}

/// **The target.** The deviatoric stress an incompressible Neo-Hookean solid
/// carries at `F = R·U`, `U = diag(9/4, 2/3, 2/3)`.
///
/// # The closed form, derived here
///
/// `σ = μ B − p I` with `B = F Fᵀ = R U² Rᵀ`, so in the stretch frame
/// `σ̃ = μ U² − p I` and
///
/// ```text
/// U²      = diag(81/16, 4/9, 4/9) = diag(2187, 192, 192) / 432
/// tr U²   = 81/16 + 8/9 = (729 + 128)/144 = 857/144 = 2571/432
/// tr U²/3 = 857/432
/// dev σ̃  = μ (U² − tr(U²)/3 · I)
///         = μ · diag(2187 − 857, 192 − 857, 192 − 857) / 432
///         = μ · diag(1330, −665, −665) / 432
///         = μ · diag(665/216, −665/432, −665/432)
/// ```
///
/// Two independent checks on that line, both asserted below:
///
/// - `Σ dev = 1330 − 665 − 665 = 0`
/// - `dev₁ − dev₂ = μ·1995/432 = μ·665/144`, which must equal the principal
///   stress difference `μ(λ₁² − λ₂²) = μ(81/16 − 4/9) = μ·665/144`
///
/// `μ` is the Lamé `μ` of [`pla`] (`E/(2(1+ν)) = 35000/27 = 1296.296… MPa`), so
/// `dev σ̃ = (3990.912209, −1995.456104, −1995.456104) MPa`.
///
/// ⚠️ The expected values are written from the rationals above. Nothing in
/// `hyperelastic.rs` is called to produce them —
/// `the_neo_hookean_uniaxial_helper_agrees_with_its_closed_form` checks that
/// module against its own closed form separately.
#[test]
fn the_neo_hookean_deviator_is_what_the_element_returns_under_that_law() {
    let mu = lame().1;
    let turn = THREE_FOUR_FIVE;

    // det U = 1 exactly, which is what makes the incompressible model applicable.
    let det_u = ISOCHORIC_U[0] * ISOCHORIC_U[1] * ISOCHORIC_U[2];
    assert!(
        (det_u - 1.0).abs() < 1e-12,
        "the Neo-Hookean models in hyperelastic.rs carry no volumetric term, so \
         the stretch must be isochoric; det U = {det_u}"
    );

    // the closed form, from the rationals in the doc comment
    let dev = [mu * 665.0 / 216.0, -mu * 665.0 / 432.0, -mu * 665.0 / 432.0];
    assert!(
        (dev[0] + dev[1] + dev[2]).abs() < 1e-9,
        "a deviator is traceless; got {:.6}",
        dev[0] + dev[1] + dev[2]
    );
    assert!(
        ((dev[0] - dev[1]) - mu * 665.0 / 144.0).abs() < 1e-9,
        "dev_1 − dev_2 must equal the principal stress difference μ(λ₁² − λ₂²) = μ·665/144"
    );

    let mesh = kuhn_cube(4, SIDE / 4.0);
    let (bc, interior) = boundary_rotated_stretch(&mesh, turn, ISOCHORIC_U);
    // The model's μ is the material's own Lamé μ, read off the same
    // `ElasticMaterial` the solve is given, so the two describe one solid. The
    // closed form above uses the `f64` value of the same expression.
    let out = solve_corotational(
        &mesh,
        &pla(),
        &bc,
        &corotational_config_with_newton_budget(2, STRETCH_NEWTON_BUDGET).with_hyperelastic(
            HyperelasticModel::NeoHookean {
                mu_mpa: pla().lame().1,
            },
        ),
    )
    .expect("the scene converges at this Newton budget; see the_reachable_stretches_…");

    // the affine field is exactly representable by P1, so the kinematics are not
    // what this oracle is about — check them anyway so a red is unambiguous.
    let mut drift = 0.0_f64;
    for &v in &interior {
        let want = rotated_stretch_field(turn, ISOCHORIC_U, vert(&mesh, v));
        for (axis, want_axis) in want.iter().enumerate() {
            drift =
                drift.max((out.field.displacements[v as usize][axis].to_f64() - want_axis).abs());
        }
    }
    assert!(
        drift < 1e-6,
        "interior is {drift:.3e} mm off the affine field"
    );

    let mut worst = 0.0_f64;
    for st in &out.field.element_stress {
        let got = deviator_in_the_stretch_frame(turn, st);
        for (g, w) in got.iter().zip([dev[0], dev[1], dev[2], 0.0]) {
            worst = worst.max((g - w).abs());
        }
    }
    eprintln!(
        "  Neo-Hookean dev σ̃ = ({:.6}, {:.6}, {:.6}) MPa, worst |dev − oracle| = {worst:.3e} MPa, \
         interior drift {drift:.3e} mm, {} Newton steps over {} increments",
        dev[0], dev[1], dev[2], out.newton_iterations, out.increments
    );
    assert!(
        worst < 1e-2,
        "an incompressible Neo-Hookean element must carry dev σ̃ = μ(U² − tr(U²)/3·I); \
         worst component is {worst:.4} MPa off a state whose largest component is {:.3} MPa",
        dev[0]
    );
}

/// **The default law, pinned.** The same geometry with no material model set —
/// the co-rotational *linear* law, which is what `solve_corotational` evaluates
/// when [`CorotationalConfig::with_hyperelastic`] is not called.
///
/// ```text
/// ε      = U − I = diag(5/4, −1/3, −1/3)      tr ε = 5/4 − 2/3 = 7/12
/// σ̃      = λ tr(ε) I + 2μ ε
///        = (5005.144033, 900.205761, 900.205761) MPa   for E = 3500, ν = 0.35
/// σ      = R σ̃ Rᵀ
/// dev σ̃  = (2736.625514, −1368.312757, −1368.312757) MPa
/// ```
///
/// Compared as a deviator for the same reason as its twin: every boundary degree
/// of freedom is prescribed, so the hydrostatic part is not determined by the
/// scene. Measured worst deviation 3.5e-6 MPa against a state whose largest
/// component is 3527 MPa, so the 1e-2 tolerance has three orders of headroom —
/// the affine field is exact in P1 and all that is left is the `Fix128` floor.
///
/// # ⚠️ This is the oracle the linear law's destruction tests break
///
/// Both mutations below are on the co-rotational path, which the material path
/// does not take, so **its twin cannot stand in for this test** — re-measured
/// 2026-10-01 and the twin stays green under the first of them. The full table
/// is in the module comment.
///
/// | mutation in `src/linear_elastic_fem.rs` | measured |
/// | --- | --- |
/// | `σ = R σ̃ Rᵀ` → return `σ̃` unrotated | red here (off-diagonal `σ̃₁₂` term) and in `rotated_uniform_stretch_matches_the_closed_form`, 2 of 13 |
/// | `ε = sym(RᵀF − I)` → drop the `Rᵀ` | red here and in 9 others, including `boundary_rigid_rotation_leaves_the_interior_unstressed` |
///
/// That is also why this test was kept green when its twin's `#[ignore]` came
/// off — see the reversal condition in the module comment above.
#[test]
fn the_corotational_linear_law_is_what_the_element_returns_without_a_material() {
    let (lambda, mu) = lame();
    let turn = THREE_FOUR_FIVE;

    let e = [
        ISOCHORIC_U[0] - 1.0,
        ISOCHORIC_U[1] - 1.0,
        ISOCHORIC_U[2] - 1.0,
    ];
    let trace = e[0] + e[1] + e[2];
    let s_local = [
        lambda * trace + 2.0 * mu * e[0],
        lambda * trace + 2.0 * mu * e[1],
        lambda * trace + 2.0 * mu * e[2],
    ];
    let third = (s_local[0] + s_local[1] + s_local[2]) / 3.0;
    let dev = [
        s_local[0] - third,
        s_local[1] - third,
        s_local[2] - third,
        0.0,
    ];

    let mesh = kuhn_cube(4, SIDE / 4.0);
    let (bc, interior) = boundary_rotated_stretch(&mesh, turn, ISOCHORIC_U);
    let out = solve_corotational(
        &mesh,
        &pla(),
        &bc,
        &corotational_config_with_newton_budget(2, STRETCH_NEWTON_BUDGET),
    )
    .expect("125 % stretch converges with a Newton budget of 64");

    let mut drift = 0.0_f64;
    for &v in &interior {
        let want = rotated_stretch_field(turn, ISOCHORIC_U, vert(&mesh, v));
        for (axis, want_axis) in want.iter().enumerate() {
            drift =
                drift.max((out.field.displacements[v as usize][axis].to_f64() - want_axis).abs());
        }
    }

    let mut worst = 0.0_f64;
    for st in &out.field.element_stress {
        let got = deviator_in_the_stretch_frame(turn, st);
        for (g, w) in got.iter().zip(dev) {
            worst = worst.max((g - w).abs());
        }
    }

    eprintln!(
        "  co-rotational linear dev σ̃ = ({:.6}, {:.6}) MPa, interior drift {drift:.3e} mm, \
         worst |dev − closed form| = {worst:.9} MPa",
        dev[0], dev[1]
    );
    assert!(
        drift < 1e-6,
        "the affine field is exactly representable by P1; worst node is {drift:.3e} mm off"
    );
    assert!(
        worst < 1e-2,
        "the element must return dev(λ tr(U−I) I + 2μ (U−I)) rotated into the global frame; \
         worst component is {worst:.9} MPa off a state whose largest component is {:.3} MPa",
        dev[0].abs()
    );
    // the gap the twin measures, printed so the two numbers sit together
    let nh = mu * 665.0 / 216.0;
    eprintln!(
        "    Neo-Hookean carries {nh:.6} MPa on dev_1, a gap of {:.4} MPa ({:.1}% of this)",
        nh - dev[0],
        100.0 * (nh - dev[0]) / dev[0]
    );
}

/// **The material law decides the answer, not only the reported stress.**
///
/// # Why the twins above cannot say this
///
/// Their solution is affine, and an affine displacement is an exact equilibrium
/// under *every* homogeneous law: a uniform stress puts zero force on an interior
/// node whatever the stress is. So the linear solve already lands on the
/// material's root, the iteration stops after one step per increment, and the
/// term that carries the law into the step is never formed. Measured: **2 Newton
/// steps over 2 increments**, the same as the linear twin's path to the same
/// displacement.
///
/// This test puts a 200 N load on the single node at the centre of the cube, with
/// the same isochoric boundary data. The field is then inhomogeneous, the two laws
/// disagree about it, and the iteration has to work: measured **181 Newton steps**
/// against the linear law's 84.
///
/// # ⚠️ What is asserted, and why `Ok` is the strong part
///
/// There is no closed form for an inhomogeneous field on this mesh, so the
/// oracle is not a value — it is that `solve_corotational` returns `Ok` **under
/// the Neo-Hookean law**. It returns `Ok` only after checking, on the answer, that
/// the largest out-of-balance nodal force is within `newton_tolerance` of the
/// reference load, with the internal force computed from that law. So `Ok` says
/// the field handed back is in equilibrium under Neo-Hookean, and a step that
/// carried the wrong law — or none — would converge somewhere that is not, and
/// report `NotConverged`.
///
/// Two more measurements are pinned because they are what separate "the law is in
/// the step" from "the law is in the report":
///
/// - the answer moves by `9.173e-3 mm` from the linear law's, four orders above
///   the `2.3e-10 mm` the affine scene shows;
/// - it does **not** move with the increment count — `9.356e-11 mm` between one
///   increment and two, five orders below the law gap, which says the iteration
///   is converging to a root rather than stopping where the path left it. ⚠️ It
///   is not *zero*: unlike the linear law's exact step, this one stops on a
///   tolerance, so [`solve_corotational`]'s increment independence is not claimed
///   here.
#[test]
fn the_material_law_moves_the_answer_and_the_answer_is_equilibrium() {
    let turn = THREE_FOUR_FIVE;
    let mesh = kuhn_cube(4, SIDE / 4.0);
    let (bc, _) = boundary_rotated_stretch(&mesh, turn, ISOCHORIC_U);

    // The one node at the centre of the cube, which `boundary_rotated_stretch`
    // left free. `SIDE / 2` is a lattice point at n = 4, checked here rather than
    // assumed, because a mesh change that moved it would otherwise load nothing.
    let centre = (0..u32::try_from(mesh.vertex_count()).expect("fits"))
        .find(|&v| {
            vert(&mesh, v)
                .iter()
                .all(|c| (c - SIDE / 2.0).abs() < SIDE * 1e-9)
        })
        .expect("the 4-cell cube has a node at its centre");
    let mut loaded = bc.clone();
    loaded.add_load(centre, Axis::X, fx(200.0));

    let neo_hookean = HyperelasticModel::NeoHookean {
        mu_mpa: pla().lame().1,
    };
    let solve_with = |increments: u32, model: Option<HyperelasticModel>| {
        let base = corotational_config_with_newton_budget(increments, 512);
        let config = match model {
            Some(m) => base.with_hyperelastic(m),
            None => base,
        };
        solve_corotational(&mesh, &pla(), &loaded, &config)
    };

    let linear = solve_with(1, None).expect("the linear law reaches this load");
    let material = solve_with(1, Some(neo_hookean))
        .expect("the Neo-Hookean field must satisfy Neo-Hookean equilibrium");
    let split = solve_with(2, Some(neo_hookean)).expect("same, applied in two increments");

    let worst = |a: &[[Fix128; 3]], b: &[[Fix128; 3]]| {
        let mut w = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            for (cx, cy) in x.iter().zip(y.iter()) {
                w = w.max((cx.to_f64() - cy.to_f64()).abs());
            }
        }
        w
    };
    let law_gap = worst(&linear.field.displacements, &material.field.displacements);
    let increment_gap = worst(&material.field.displacements, &split.field.displacements);

    eprintln!(
        "  loaded centre node: linear {} Newton steps, Neo-Hookean {} (1 increment) / {} (2)",
        linear.newton_iterations, material.newton_iterations, split.newton_iterations
    );
    eprintln!(
        "    |u_linear − u_neo-hookean| = {law_gap:.3e} mm, \
         |u(1 increment) − u(2)| = {increment_gap:.3e} mm"
    );

    assert!(
        law_gap > 1e-3,
        "the two laws must disagree about an inhomogeneous field; they differ by \
         {law_gap:.3e} mm, which is the order the affine scene reports when the law \
         is not in the step at all"
    );
    assert!(
        increment_gap < 1e-8,
        "the converged field must be the material's root and not where the \
         continuation path stopped; one and two increments differ by \
         {increment_gap:.3e} mm"
    );
}

/// **Characterisation, not correctness.** Records which stretches
/// `solve_corotational` reaches today, and that `NotConverged` covers two
/// unrelated situations.
///
/// ⚠️ **Nothing here is asserted to be the right behaviour.** In particular the
/// `a = 5/4` refusal below is pinned as *the current state*, not as correct: a
/// frame fixed point that fails equilibrium may well be a defect of the stopping
/// rule or of the formulation, and settling that is a design question this file
/// does not answer. If a later change makes `a = 5/4` converge, **this test is
/// supposed to go red** — update it, do not read the red as a regression.
///
/// This replaces a bisection. "The largest stretch that converges" is only a
/// meaningful quantity if the reachable set is an interval, and it is not:
/// measured on the exactly-rational isochoric family `U = diag(a², 1/a, 1/a)`,
///
/// | `a` | first stretch | budget 32 | 64 | 4096 |
/// | --- | --- | --- | --- | --- |
/// | 9/8 | +27 % | ok | ok | ok |
/// | 6/5 | +44 % | ok | ok | ok |
/// | **5/4** | **+56 %** | **NotConverged** | **NotConverged** | **NotConverged** |
/// | 4/3 | +78 % | ok | ok | ok |
/// | 7/5 | +96 % | ok | ok | ok |
/// | **3/2** | **+125 %** | **NotConverged** | **ok** | ok |
///
/// A bisection would cross the `a = 5/4` hole and report a threshold that is not
/// one. The same shape shows up off the isochoric family: sweeping
/// `U(t) = I + t·diag(5/4, −1/3, −1/3)` at 0.01 finds two holes,
/// `t ∈ [0.22, 0.36]` and `t ≥ 0.80`.
///
/// # ⚠️ The two `NotConverged`s
///
/// `relative_residual` is `residual / newton_target`, so **below 1 means the
/// residual already met the tolerance**.
///
/// - `iterations == budget`: the frame iteration on the final increment ran out
///   of steps. The residual is not what refused — at `a = 3/2` it is `0.217`,
///   i.e. 4.6× inside tolerance. Raising the budget fixes it (threshold 47).
/// - `iterations < budget`: the frames settled and the fixed point they settled
///   on does not satisfy equilibrium (`relative_residual` 1.03…1.7). Raising the
///   budget changes nothing — at `a = 5/4` six budgets from 32 to 4096 return the
///   identical `iterations` and `relative_residual`.
///
/// Telling them apart matters because the responses are opposite: a budget for
/// the first, the tolerance or the formulation for the second. The `a = 5/4` band
/// is the second kind and is not addressed here.
///
/// `FemError::Stagnated` has the same shape — three mechanisms behind one variant,
/// measured separately in `tests/locking_p1.rs`. Two variants of one enum now
/// conflate distinguishable causes, so the pattern is the enum's, not one
/// variant's.
#[test]
fn characterises_which_stretches_the_corotational_solve_reaches() {
    let turn = THREE_FOUR_FIVE;
    let mesh = kuhn_cube(4, SIDE / 4.0);

    // U = diag(a², 1/a, 1/a) — isochoric for every a, and rational for rational a.
    let isochoric = |num: f64, den: f64| {
        let a = num / den;
        [a * a, 1.0 / a, 1.0 / a]
    };
    let attempt = |stretch: [f64; 3], budget: u32| {
        let (bc, _) = boundary_rotated_stretch(&mesh, turn, stretch);
        solve_corotational(
            &mesh,
            &pla(),
            &bc,
            &corotational_config_with_newton_budget(2, budget),
        )
        .map(|_| ())
    };

    // the hole: a = 5/4 refuses, while the larger a = 4/3 and a = 7/5 do not
    let hole = isochoric(5.0, 4.0);
    let err = attempt(hole, 32).expect_err("a = 5/4 does not converge");
    let (hole_iterations, hole_residual) = match err {
        FemError::NotConverged {
            iterations,
            relative_residual,
        } => (iterations, relative_residual),
        other => panic!("expected NotConverged at a = 5/4, got {other:?}"),
    };
    assert!(
        hole_iterations < 32,
        "a = 5/4 stops because the frames settled on a non-equilibrium fixed point, \
         so it must report fewer iterations than the budget; got {hole_iterations}"
    );
    assert!(
        hole_residual > Fix128::ONE,
        "…and a residual above the tolerance it was measured against"
    );
    // budget independence: identical iterations and residual at 128x the budget.
    // One extra budget is enough to make the point; 4096 is the largest measured.
    for budget in [4096_u32] {
        match attempt(hole, budget) {
            Err(FemError::NotConverged {
                iterations,
                relative_residual,
            }) => {
                assert_eq!(
                    (iterations, relative_residual),
                    (hole_iterations, hole_residual),
                    "a = 5/4 is budget independent, so budget {budget} must reproduce \
                     the budget-32 result bit for bit"
                );
            }
            other => panic!("a = 5/4 must refuse at budget {budget}; got {other:?}"),
        }
    }
    // larger stretches on the same family do converge — this is the
    // non-monotonicity. Only the two witnesses *above* a = 5/4 are run; a = 9/8
    // and a = 6/5 are in the table but would add nothing an assert can use.
    for (num, den) in [(4.0, 3.0), (7.0, 5.0)] {
        attempt(isochoric(num, den), 32).unwrap_or_else(|e| {
            panic!("a = {num}/{den} is expected to converge at the house budget; got {e:?}")
        });
    }

    // the budget case: a = 3/2 refuses at 32 with a residual *inside* tolerance,
    // and the threshold is 47
    match attempt(ISOCHORIC_U, 32) {
        Err(FemError::NotConverged {
            iterations,
            relative_residual,
        }) => {
            assert_eq!(
                iterations, 32,
                "a = 3/2 runs out of frame-settling steps, so it must report the budget"
            );
            assert!(
                relative_residual < Fix128::ONE,
                "⚠️ the residual is already inside tolerance when the budget fires; \
                 got {relative_residual} × the target"
            );
        }
        other => panic!("a = 3/2 must refuse at the house budget of 32; got {other:?}"),
    }
    attempt(ISOCHORIC_U, 46).expect_err("46 frame-settling steps are not enough at a = 3/2");
    attempt(ISOCHORIC_U, 47).expect("47 is the measured threshold at a = 3/2");
    attempt(ISOCHORIC_U, STRETCH_NEWTON_BUDGET)
        .expect("the budget the twin oracles above use must converge");
    eprintln!(
        "  a = 5/4 refuses at every budget (iterations {hole_iterations}, \
         residual {hole_residual} × target) while a = 4/3 and a = 7/5 converge; \
         a = 3/2 needs 47 frame-settling steps"
    );
}

/// Contract check on the module the twins are *about*.
///
/// The gap they pin is wiring, not constitutive: `hyperelastic.rs` gets the
/// incompressible Neo-Hookean uniaxial stress right. `σ = μ(λ² − 1/λ)` is the
/// standard result, written here from the formula rather than taken from the
/// function under test.
#[test]
fn the_neo_hookean_uniaxial_helper_agrees_with_its_closed_form() {
    let mu = 3.0_f64;
    let model = HyperelasticModel::NeoHookean { mu_mpa: fx(mu) };
    for lambda in [0.5_f64, 0.8, 1.0, 1.25, 2.0, 2.25] {
        let want = mu * (lambda * lambda - 1.0 / lambda);
        let got = uniaxial_cauchy_stress(&model, fx(lambda)).to_f64();
        assert!(
            (got - want).abs() < 1e-9,
            "σ(λ = {lambda}) must be μ(λ² − 1/λ) = {want:.9}; got {got:.9}"
        );
    }
    // the strain energy of the undeformed state is zero, and it grows with stretch
    let w_unity = strain_energy_density(&model, &Stretch::UNITY).to_f64();
    assert!(w_unity.abs() < 1e-9, "W(I) must vanish; got {w_unity:.9}");
}

/// The Newton budget the twins run with does not change the answer they pin.
///
/// [`STRETCH_NEWTON_BUDGET`] is 64 because 32 does not converge at this stretch
/// and the measured floor is 47 — a number chosen to make the solve *finish*. That
/// makes it a load-bearing default of
/// `the_corotational_linear_law_is_what_the_element_returns_without_a_material`, and "the
/// budget was raised to converge, not to move the answer" is exactly what that pin
/// means. So it is checked here rather than asserted in prose.
///
/// ⚠️ `the_answer_does_not_depend_on_the_newton_budget` does **not** cover this.
/// That test is older than the twins, runs a different scene (a rigid rotation, no
/// stretch) through [`corotational_config`], and never reaches a budget above 32 —
/// so nothing in it touches the `64` the twins depend on.
///
/// Doubling the budget past a converged solve is a pure no-op on the iteration:
/// the extra steps are never taken, because the frames have already settled and
/// the loop exits on that, not on the count. The measured difference is therefore
/// `0` ulp on every stress component, and `assert_eq!` is the honest bound.
///
/// ⚠️ **`0 == 0` is the shape a vacuous comparison takes**, so the comparison was
/// checked against a difference it *should* see: replacing the second solve with
/// one at 4 increments instead of a doubled budget moves it to **11 ulp on the
/// displacement and 69819 ulp on the stress**. The zero above is a fact about
/// raising the budget, not an artifact of comparing a value with itself.
#[test]
fn the_budget_the_twins_use_does_not_change_what_they_pin() {
    let turn = THREE_FOUR_FIVE;
    let mesh = kuhn_cube(4, SIDE / 4.0);
    let (bc, _) = boundary_rotated_stretch(&mesh, turn, ISOCHORIC_U);

    let solve_at = |budget: u32| {
        solve_corotational(
            &mesh,
            &pla(),
            &bc,
            &corotational_config_with_newton_budget(2, budget),
        )
        .unwrap_or_else(|e| panic!("budget {budget} must converge at this stretch; got {e:?}"))
    };

    let at_64 = solve_at(STRETCH_NEWTON_BUDGET);
    let at_128 = solve_at(STRETCH_NEWTON_BUDGET * 2);

    // displacements first: same helper the increment-spread oracle uses
    let spread = worst_ulp(&at_64.field.displacements, &at_128.field.displacements);

    // then the stresses, which are what the twins actually assert on
    let mut worst_stress_ulp = 0_u128;
    for (a, b) in at_64
        .field
        .element_stress
        .iter()
        .zip(at_128.field.element_stress.iter())
    {
        for (x, y) in [
            (a.xx, b.xx),
            (a.yy, b.yy),
            (a.zz, b.zz),
            (a.xy, b.xy),
            (a.yz, b.yz),
            (a.zx, b.zx),
        ] {
            worst_stress_ulp = worst_stress_ulp.max((raw(x) - raw(y)).unsigned_abs());
        }
    }

    eprintln!(
        "  budget {} vs {}: displacement spread {spread} ulp, stress spread {worst_stress_ulp} ulp",
        STRETCH_NEWTON_BUDGET,
        STRETCH_NEWTON_BUDGET * 2
    );
    eprintln!(
        "    newton_iterations: {} at budget {}, {} at budget {}",
        at_64.newton_iterations,
        STRETCH_NEWTON_BUDGET,
        at_128.newton_iterations,
        STRETCH_NEWTON_BUDGET * 2
    );

    // ⚠️ The discriminator. A `0` spread would also be produced by two solves
    // that both stopped *on the budget* — but then the pinned value would be "what
    // 64 steps happen to give", not the converged answer, and raising the budget
    // further would keep moving it. Both sides must have exited on the frames
    // settling, which means strictly fewer steps than the budget allowed.
    assert!(
        at_64.newton_iterations < STRETCH_NEWTON_BUDGET,
        "the pinned solve must stop because the frames settled, not because the \
         budget ran out; it used all {} steps",
        at_64.newton_iterations
    );
    assert_eq!(
        at_64.newton_iterations, at_128.newton_iterations,
        "a converged solve does not take more steps when it is allowed more"
    );
    assert_eq!(
        spread, 0,
        "doubling a budget the solve already finished inside must not move the \
         displacement by a single bit"
    );
    assert_eq!(
        worst_stress_ulp, 0,
        "…nor the stress the twin oracles compare against their closed forms"
    );
}

// ---------------------------------------------------------------------------
// objectivity, at the level of the solve
// ---------------------------------------------------------------------------

/// `Q·R` for `Q` the quarter turn about `z` and `R` = [`THREE_FOUR_FIVE`].
///
/// A rotation about `z` composed with another is just another `(cos, sin)` pair,
/// and for a quarter turn it is the same two rationals reordered and negated:
/// `(c, s) → (−s, c)`. So the rotated scene below is not a different *kind* of
/// scene — same mesh, same stretch, same rational boundary data.
const QUARTER_AFTER_THREE_FOUR_FIVE: Turn = Turn {
    name: "126.87° about z (90° after 3-4-5)",
    cos: -3.0 / 5.0,
    sin: 4.0 / 5.0,
};

/// `Q·p` for the same quarter turn about `z`.
fn quarter_turn(p: [f64; 3]) -> [f64; 3] {
    [-p[1], p[0], p[2]]
}

/// The node at the centre of the cube, which `boundary_rotated_stretch` leaves
/// free. Checked rather than assumed, because a mesh change that moved it would
/// otherwise load nothing.
fn centre_node(mesh: &SdfTetMesh) -> u32 {
    (0..u32::try_from(mesh.vertex_count()).expect("fits"))
        .find(|&v| {
            vert(mesh, v)
                .iter()
                .all(|c| (c - SIDE / 2.0).abs() < SIDE * 1e-9)
        })
        .expect("the 4-cell cube has a node at its centre")
}

/// **Objectivity at the level of the solve** — the companion to the two oracles
/// in `src/linear_elastic_fem.rs`, which pin the same property on `P` directly.
///
/// Superposing a rigid rotation `Q` on a scene — the reference mesh untouched,
/// the boundary positions and the load both turned by `Q` — must turn the answer
/// by `Q` and nothing else. The internal force of a frame-indifferent law is
/// equivariant (`f(Qx) = Q f(x)`), so the equilibrium of the turned scene is the
/// turned equilibrium, and **no closed form is needed**: the second solve is the
/// oracle for the first.
///
/// That is what makes this reachable from `tests/`, where `P` is not. It is also
/// what the inhomogeneous-field test above cannot say on its own: `Ok` means the
/// field balances the force the solver assembled, which a wrong `P` also does.
///
/// # ⚠️ Why this is a bound and not an `assert_eq!`
///
/// `solve_corotational` stops on a tolerance, not on an exact fixed point, and
/// the two runs do not even take the same number of steps — **measured 181 and
/// 192**, so the solve is not exactly equivariant and the gap is a convergence
/// floor, not zero. That is the same reason
/// `the_answer_does_not_depend_on_the_increment_count` is ignored. The exact
/// statement of the same property lives in `src/linear_elastic_fem.rs`, on `P`,
/// where no iteration is involved.
///
/// The threshold is **`1e-6` mm, set from the measured clean gap with ~390×
/// headroom** — not back-computed from what any mutation produces. The measured
/// gap is printed on every run so a later reader can re-derive that factor.
///
/// # ⚠️ Where the teeth are, measured — and the bound is **not** one of them
///
/// | mutation | worst `\|u(QS) − Q u(S)\|` |
/// | --- | --- |
/// | clean | **2.563e-9 mm** (node 56), 181 / 192 Newton steps |
/// | `rotate_stress` returns `σ̃` unrotated | **2.563e-9 mm**, unchanged — the linear reporting path is not in this scene |
/// | `hyperelastic_stress` adds `+1 MPa` to `σ_xx`, a **world-fixed** axis (≈0.08 % of `μ`) | ⚠️ **2.563e-9 mm, unchanged** — see below |
/// | `hyperelastic_stress` drops the `F⁻ᵀ` | ⚠️ **no gap to measure**: the turned scene returns `RotationFailed { tet: 25, cause: Inverted }` |
/// | `material_correction` returns without accumulating | the **base** scene returns `NotConverged` at 512 iterations |
///
/// ⚠️⚠️ **The `1e-6` bound below is untested. Every red in that table comes from
/// one of the two `expect`s, never from the bound.** Do not read a green here as
/// "the threshold was checked against something".
///
/// ⚠️⚠️ **Untested is not the same as dead — do not delete it.** A dead assertion
/// is one the earlier assertions already imply, and this one is not: the `expect`s
/// only say that both solves returned `Ok`, and a mutation can converge *and* land
/// on a non-equivariant answer. Row 3 is the measured existence proof of that
/// case — a non-objective stress bias that the direct oracles on `P` both red on
/// (verified: the mutation is live) while **both scenes here still converge**. It
/// leaves this gap unchanged only because its violation falls under the
/// convergence floor, not because the bound could never fire. Raise the bias and
/// the bound is what catches it.
///
/// ⚠️ Row 3 is also the sharpest statement of **what this test does not see**: at
/// `+1 MPa` a world-fixed, non-objective stress is invisible here. The likely
/// reason is the uniform-stress blind spot in its approximate form — this scene's
/// `F` is affine on the boundary and perturbed by one loaded node, so a stress
/// shift that is the same tensor everywhere puts nearly no force on an interior
/// node. **That mechanism is not verified**; what is measured is the invisibility.
/// Finding a bias large enough to converge *and* exceed `1e-6 mm` is in the
/// Backlog, not done here.
///
/// ⚠️ Two of the five rows red for reasons that are not this test's subject (a
/// scene that will not converge at all), so **this test does not separate causes
/// on its own** — it is the companion to the direct oracles on `P`, not a
/// replacement for them.
#[test]
fn the_solve_is_equivariant_under_a_superposed_quarter_turn() {
    let mesh = kuhn_cube(4, SIDE / 4.0);
    let centre = centre_node(&mesh);
    let neo_hookean = HyperelasticModel::NeoHookean {
        mu_mpa: pla().lame().1,
    };
    let config = corotational_config_with_newton_budget(1, 512).with_hyperelastic(neo_hookean);

    let solve_turned_by = |turn: Turn, load_axis: Axis| {
        let (mut bc, _) = boundary_rotated_stretch(&mesh, turn, ISOCHORIC_U);
        bc.add_load(centre, load_axis, fx(200.0));
        solve_corotational(&mesh, &pla(), &bc, &config)
    };

    // `Q·(200, 0, 0) = (0, 200, 0)`, so the turned scene carries its load on y.
    let base = solve_turned_by(THREE_FOUR_FIVE, Axis::X).expect("the base scene converges");
    let turned = solve_turned_by(QUARTER_AFTER_THREE_FOUR_FIVE, Axis::Y)
        .expect("the turned scene converges");

    let mut worst = 0.0_f64;
    let mut worst_node = 0_u32;
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        let x = vert(&mesh, v);
        let u = base.field.displacements[v as usize];
        // `u(QS) = Q·(X + u(S)) − X`, the turned deformed position read back as a
        // displacement from the same reference node.
        let want = quarter_turn([
            x[0] + u[0].to_f64(),
            x[1] + u[1].to_f64(),
            x[2] + u[2].to_f64(),
        ]);
        let got = turned.field.displacements[v as usize];
        for axis in 0..3 {
            let d = (got[axis].to_f64() + x[axis] - want[axis]).abs();
            if d > worst {
                worst = d;
                worst_node = v;
            }
        }
    }

    eprintln!(
        "  base {} Newton steps, turned {}",
        base.newton_iterations, turned.newton_iterations
    );
    eprintln!("    worst |u(QS) − Q u(S)| = {worst:.3e} mm at node {worst_node}");

    // ⚠️ This bound has never fired: every mutation measured so far either leaves
    // the gap unchanged or stops one of the two solves above. It is not implied by
    // those `expect`s — see the table on this test — so it is untested, not dead.
    assert!(
        worst < 1e-6,
        "the turned scene must land on the turned answer; worst component differs \
         by {worst:.3e} mm at node {worst_node}, against a measured clean gap of \
         2.563e-9 mm"
    );
}

// ===========================================================================
// The consistent tangent
// ===========================================================================
//
// `with_consistent_tangent` replaces the co-rotational linear tangent of the
// modified Newton iteration with the tangent of the stress that is implemented
// (`src/linear_elastic_fem/consistent_tangent.rs`; the closed form is checked
// against `hyperelastic_stress` in that file). What these tests pin is what the
// solve does with it.
//
// ⚠️ What was claimed and what was measured. The backlog said the modified
// iteration "does not converge at 2000 N" with a contraction rate of 1.33. It
// does converge: reading the relative residual `NotConverged` reports for
// budgets 1 .. 1024 shows a monotone contraction of about 0.98 per step, and
// 2000 N is `Ok` after **758** steps (181 at 200 N). The problem is speed, and
// that is what the consistent tangent removes: 3 steps at 200 N and 4 at 2000 N.

/// The 4-cell cube of `the_material_law_moves_the_answer_…` with a point load of
/// `load` N on its centre node, the boundary rotated and stretched.
fn centre_loaded_cube(load: f64) -> (SdfTetMesh, BoundaryConditions) {
    let mesh = kuhn_cube(4, SIDE / 4.0);
    let (bc, _) = boundary_rotated_stretch(&mesh, THREE_FOUR_FIVE, ISOCHORIC_U);
    let centre = (0..u32::try_from(mesh.vertex_count()).expect("fits"))
        .find(|&v| {
            vert(&mesh, v)
                .iter()
                .all(|c| (c - SIDE / 2.0).abs() < SIDE * 1e-9)
        })
        .expect("the 4-cell cube has a node at its centre");
    let mut loaded = bc;
    loaded.add_load(centre, Axis::X, fx(load));
    (mesh, loaded)
}

fn neo_hookean_model() -> HyperelasticModel {
    HyperelasticModel::NeoHookean {
        mu_mpa: pla().lame().1,
    }
}

/// Largest component of the difference of two displacement fields, mm.
fn field_gap(a: &[[Fix128; 3]], b: &[[Fix128; 3]]) -> f64 {
    let mut worst = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        for (cx, cy) in x.iter().zip(y.iter()) {
            worst = worst.max((cx.to_f64() - cy.to_f64()).abs());
        }
    }
    worst
}

fn solve_cube(
    load: f64,
    increments: u32,
    budget: u32,
    model: HyperelasticModel,
    consistent: bool,
) -> Result<
    alice_physics::linear_elastic_fem::CorotationalSolution,
    alice_physics::linear_elastic_fem::FemError,
> {
    let (mesh, loaded) = centre_loaded_cube(load);
    let mut config =
        corotational_config_with_newton_budget(increments, budget).with_hyperelastic(model);
    if consistent {
        config = config.with_consistent_tangent();
    }
    solve_corotational(&mesh, &pla(), &loaded, &config)
}

/// **Oracle: the same root, in a handful of steps.** The consistent tangent
/// changes how many steps, not where they land, so its field must be the
/// modified iteration's. Measured: 1.6e-9 mm apart at 200 N (both stop on a
/// `2⁻³⁰` residual), 3 steps against 181.
#[test]
fn the_consistent_tangent_lands_on_the_modified_newton_root_in_a_handful_of_steps() {
    let modified = solve_cube(200.0, 1, 512, neo_hookean_model(), false).expect("modified, 200 N");
    let consistent =
        solve_cube(200.0, 1, 64, neo_hookean_model(), true).expect("consistent, 200 N");
    let gap = field_gap(
        &modified.field.displacements,
        &consistent.field.displacements,
    );
    eprintln!(
        "  200 N: modified {} steps, consistent {} steps, fields {gap:.3e} mm apart",
        modified.newton_iterations, consistent.newton_iterations
    );
    assert!(
        gap < 1e-7,
        "same material, same load, different roots: {gap:.3e} mm apart"
    );
    assert!(
        modified.newton_iterations >= 100,
        "the premise of the comparison is that the modified iteration is slow here; it took {}",
        modified.newton_iterations
    );
    assert!(
        consistent.newton_iterations <= 8,
        "a consistent Newton iteration should need a handful of steps, took {}",
        consistent.newton_iterations
    );
}

/// **Oracle: 2000 N in a handful of steps, and independent of the increment
/// count.** The modified iteration needs 758 steps here (budget 256 is not
/// enough). With the consistent tangent the answer does not move when the load
/// is applied in two increments instead of one: measured 6.5e-13 mm apart, four
/// orders tighter than the modified iteration's `1e-8` bound.
#[test]
fn the_consistent_tangent_reaches_2000_n_in_a_handful_of_steps_whatever_the_increments() {
    let one =
        solve_cube(2000.0, 1, 64, neo_hookean_model(), true).expect("consistent, 1 increment");
    let two =
        solve_cube(2000.0, 2, 64, neo_hookean_model(), true).expect("consistent, 2 increments");
    let gap = field_gap(&one.field.displacements, &two.field.displacements);
    eprintln!(
        "  2000 N: consistent {} steps (1 increment) / {} (2), {gap:.3e} mm apart",
        one.newton_iterations, two.newton_iterations
    );
    assert!(
        one.newton_iterations <= 12,
        "took {} steps",
        one.newton_iterations
    );
    assert!(
        gap < 1e-9,
        "the root must not depend on the increment path: {gap:.3e} mm"
    );
    // Characterisation of what this replaces: the modified iteration runs out of
    // a 256-step budget at this load (it needs 758).
    assert!(
        matches!(
            solve_cube(2000.0, 1, 256, neo_hookean_model(), false),
            Err(alice_physics::linear_elastic_fem::FemError::NotConverged { .. })
        ),
        "the modified iteration is expected to need more than 256 steps at 2000 N"
    );
}

/// A tangent for a law that is not there has nothing to differentiate; the
/// request is refused rather than silently ignored.
#[test]
fn the_consistent_tangent_without_a_hyperelastic_law_is_refused() {
    let (mesh, loaded) = centre_loaded_cube(200.0);
    let config = corotational_config_with_newton_budget(1, 64).with_consistent_tangent();
    assert!(matches!(
        solve_corotational(&mesh, &pla(), &loaded, &config),
        Err(alice_physics::linear_elastic_fem::FemError::InvalidConfig(
            _
        ))
    ));
}

/// **Oracle: the other two laws.** Mooney-Rivlin (`W₂ ≠ 0`) and Yeoh
/// (`W₁₁ ≠ 0`) have terms Neo-Hookean does not, and the unit tests check them
/// against `hyperelastic_stress` on one element. Here the whole solve must agree
/// with itself under a different increment path and with the modified iteration
/// where that converges.
#[test]
fn the_consistent_tangent_solves_mooney_rivlin_and_yeoh() {
    let mu = pla().lame().1.to_f64();
    let models = [
        (
            "Mooney-Rivlin",
            HyperelasticModel::MooneyRivlin {
                c1_mpa: fx(mu * 0.375),
                c2_mpa: fx(mu * 0.125),
            },
        ),
        (
            "Yeoh",
            HyperelasticModel::Yeoh {
                c1_mpa: fx(mu * 0.5),
                c2_mpa: fx(mu * 0.0125),
                c3_mpa: fx(mu * 0.000_125),
            },
        ),
    ];
    for (name, model) in models {
        let one = solve_cube(200.0, 1, 64, model, true).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let two = solve_cube(200.0, 2, 64, model, true).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let gap = field_gap(&one.field.displacements, &two.field.displacements);
        eprintln!(
            "  {name}: {} / {} steps, increments {gap:.3e} mm apart",
            one.newton_iterations, two.newton_iterations
        );
        assert!(
            gap < 1e-8,
            "{name}: the root depends on the increment path, {gap:.3e} mm"
        );
        assert!(
            one.newton_iterations <= 12,
            "{name}: took {} steps",
            one.newton_iterations
        );
        // where the modified iteration also converges, it must land on this root
        if let Ok(modified) = solve_cube(200.0, 1, 1024, model, false) {
            let gap = field_gap(&one.field.displacements, &modified.field.displacements);
            assert!(
                gap < 1e-6,
                "{name}: consistent and modified roots differ by {gap:.3e} mm"
            );
        }
    }
}

/// **Panic oracle: no failure path the modified iteration does not have.** Every
/// degenerate input below is refused by the modified iteration in some way; the
/// consistent tangent must refuse it in the *same* way (same `FemError`
/// variant), not panic, and not return a field. The inputs that are not
/// refused — a boundary that does not move, or crushes the cube without
/// inverting it — must agree with the modified iteration's field.
///
/// ⚠️ `1e12 N` comes back `UnderConstrained` from **both** paths: the load is
/// representable (`Fix128 { hi: 10^12 }`) and the cube is constrained, so the
/// variant is misleading — the first conjugate gradient direction's `pᵀKp`
/// wraps. That is the modified iteration's behaviour, recorded here so the
/// consistent path does not diverge from it; it is not asserted to be right.
#[test]
fn the_consistent_tangent_adds_no_failure_path_on_degenerate_input() {
    let none = Turn {
        name: "none",
        cos: 1.0,
        sin: 0.0,
    };
    let mesh = kuhn_cube(4, SIDE / 4.0);
    let centre = (0..u32::try_from(mesh.vertex_count()).expect("fits"))
        .find(|&v| {
            vert(&mesh, v)
                .iter()
                .all(|c| (c - SIDE / 2.0).abs() < SIDE * 1e-9)
        })
        .expect("the 4-cell cube has a node at its centre");
    let run = |turn: Turn, stretch: [f64; 3], load: f64, increments: u32, consistent: bool| {
        let (mut bc, _) = boundary_rotated_stretch(&mesh, turn, stretch);
        bc.add_load(centre, Axis::X, fx(load));
        let mut config = corotational_config_with_newton_budget(increments, 1024)
            .with_hyperelastic(neo_hookean_model());
        if consistent {
            config = config.with_consistent_tangent();
        }
        solve_corotational(&mesh, &pla(), &bc, &config)
    };
    let cases: [(&str, Turn, [f64; 3], f64, u32); 7] = [
        ("crushed to 1%", none, [0.01, 0.01, 0.01], 0.0, 1),
        (
            "crushed to 1%, 4 increments",
            none,
            [0.01, 0.01, 0.01],
            0.0,
            4,
        ),
        ("reflected", none, [-1.0, 1.0, 1.0], 0.0, 1),
        ("20 kN", THREE_FOUR_FIVE, ISOCHORIC_U, 20_000.0, 1),
        ("1e12 N", THREE_FOUR_FIVE, ISOCHORIC_U, 1.0e12, 1),
        ("-1e12 N", THREE_FOUR_FIVE, ISOCHORIC_U, -1.0e12, 1),
        (
            "identity boundary, zero load",
            none,
            [1.0, 1.0, 1.0],
            0.0,
            1,
        ),
    ];
    for (name, turn, stretch, load, increments) in cases {
        let modified = run(turn, stretch, load, increments, false);
        let consistent = run(turn, stretch, load, increments, true);
        match (&modified, &consistent) {
            (Ok(m), Ok(c)) => {
                let gap = field_gap(&m.field.displacements, &c.field.displacements);
                assert!(
                    gap < 1e-6,
                    "{name}: both solve, but the fields differ by {gap:.3e} mm"
                );
            }
            (Err(m), Err(c)) => assert_eq!(
                std::mem::discriminant(m),
                std::mem::discriminant(c),
                "{name}: modified refuses with {m:?}, consistent with {c:?}"
            ),
            _ => panic!("{name}: modified {modified:?} against consistent {consistent:?}"),
        }
    }
    // The ones with a definite answer are pinned by value, not by comparison.
    let reflected = run(none, [-1.0, 1.0, 1.0], 0.0, 1, true);
    assert!(
        matches!(
            reflected,
            Err(alice_physics::linear_elastic_fem::FemError::RotationFailed { .. })
        ),
        "a reflected boundary has no valid deformation gradient: {reflected:?}"
    );
    let at_rest =
        run(none, [1.0, 1.0, 1.0], 0.0, 1, true).expect("an undeformed cube is a solution");
    assert!(
        field_gap(
            &at_rest.field.displacements,
            &vec![[Fix128::ZERO; 3]; at_rest.field.displacements.len()]
        ) < 1e-9,
        "an undeformed, unloaded cube must not move"
    );
}

/// A budget that is too small is a refusal, not a field: the consistent
/// iteration needs 3 steps at 200 N, so 1 and 2 must report `NotConverged`.
#[test]
fn the_consistent_tangent_reports_a_budget_it_cannot_meet() {
    for budget in [1u32, 2] {
        let result = solve_cube(200.0, 1, budget, neo_hookean_model(), true);
        assert!(
            matches!(result, Err(alice_physics::linear_elastic_fem::FemError::NotConverged { iterations, .. }) if iterations == budget),
            "budget {budget}: {result:?}"
        );
    }
}
