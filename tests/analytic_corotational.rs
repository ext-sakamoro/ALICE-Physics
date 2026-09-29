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
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::linear_elastic_fem::{
    solve, solve_corotational, BoundaryConditions, CorotationalConfig, ElasticMaterial,
    SolverConfig,
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
    CorotationalConfig::try_new(
        SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid linear config"),
        32,                           // Newton iterations per increment
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
#[test]
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
