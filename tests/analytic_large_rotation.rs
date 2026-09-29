//! Oracle: a rigid **rotation** carries no stress.
//!
//! `tests/analytic_linear_elastic_fem.rs` already pins the translation half of
//! the rigid-motion null space (`rigid_translation_produces_zero_stress`). A
//! translation is in the null space of *any* strain measure, so it passes on a
//! small-strain solver and on a finite-strain one alike, and therefore says
//! nothing about which of the two is implemented.
//!
//! A **large rotation** separates them. The true stress is identically zero —
//! the body has not been strained, only moved — but the small-strain tensor
//! `ε = ½(∇u + ∇uᵀ)` reads a rotation as a strain, because for `u = (R − I)·x`
//! the symmetric part of `∇u = R − I` is `diag(cosθ − 1, cosθ − 1, 0)` for a
//! rotation about z, which is not zero unless `θ` is.
//!
//! So this file is the oracle for the geometric-nonlinearity wall:
//!
//! - it **must red on the present solver**, by the amount the closed form below
//!   predicts, and that is the evidence the oracle reaches the property;
//! - it **must green on a co-rotational (or fully finite-strain) solver**, and
//!   that is the evidence the implementation is right.
//!
//! Unlike a convergence study this needs no refinement sequence and no
//! iteration: every degree of freedom is prescribed, so `solve` computes the
//! element stresses directly from the boundary data. The red is therefore a
//! statement about the strain measure alone — not about mesh quality, the
//! conjugate gradient, or the stopping rule.
//!
//! # Closed forms
//!
//! Rotation by `θ` about z, applied to every node as `u = (R − I)·x`:
//!
//! ```text
//! true (finite strain):   σ ≡ 0
//! small strain:           ε = diag(c−1, c−1, 0),  c = cos θ
//!                         σ_xx = σ_yy = 2λ(c−1) + 2μ(c−1)
//!                         σ_zz = 2λ(c−1)
//!                         σ_xy = σ_yz = σ_zx = 0
//! ```
//!
//! with `λ = Eν/((1+ν)(1−2ν))` and `μ = E/(2(1+ν))`. Note `c − 1 ≈ −θ²/2`, so
//! the spurious stress is **second order in the angle** — invisible at 1° and
//! catastrophic at 90°, which is exactly why a small-angle test cannot stand in
//! for this one.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::linear_elastic_fem::{solve, BoundaryConditions, ElasticMaterial, SolverConfig};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// scene construction (Kuhn 6-tet box, conforming for any nx/ny/nz)
// ---------------------------------------------------------------------------

/// Node index within an `(nx+1) × (ny+1) × (nz+1)` lattice.
fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

/// Axis-aligned box `[0,nx·h] × [0,ny·h] × [0,nz·h]` split into Kuhn 6-tet cells.
///
/// Every face diagonal Kuhn's subdivision produces runs between that face's
/// `(0,0)` and `(1,1)` corners, which both cubes sharing the face agree on, so
/// the mesh is conforming for any `nx, ny, nz` (see
/// `feedback_patch_test_blind_to_nonconforming_faces`).
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

/// PLA-like: 3.5 GPa, ν = 0.35 — the same material the other FEM oracles use,
/// so the numbers here can be compared with them directly.
fn pla() -> ElasticMaterial {
    ElasticMaterial::new(fx(3500.0), fx(0.35)).expect("E > 0 and ν in (-1, 0.5)")
}

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;

fn lame() -> (f64, f64) {
    let lambda = E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU));
    let mu = E_MPA / (2.0 * (1.0 + NU));
    (lambda, mu)
}

// ---------------------------------------------------------------------------
// the rotation, prescribed exactly
// ---------------------------------------------------------------------------

/// Cosine and sine of the rotation angle, given **exactly** where the angle is
/// a quarter turn so the scene carries no transcendental error at all.
#[derive(Clone, Copy)]
struct Turn {
    name: &'static str,
    cos: f64,
    sin: f64,
}

const QUARTER: Turn = Turn {
    name: "90° about z",
    cos: 0.0,
    sin: 1.0,
};

const HALF: Turn = Turn {
    name: "180° about z",
    cos: -1.0,
    sin: 0.0,
};

/// Prescribe every node to the rigid rotation `u = (R − I)·x` about the z axis
/// through the origin, which is a corner of the box.
fn prescribe_rigid_rotation(mesh: &SdfTetMesh, turn: Turn) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        let (x, y, z) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
        // R·x − x, with R the rotation about z.
        let ux = turn.cos * x - turn.sin * y - x;
        let uy = turn.sin * x + turn.cos * y - y;
        let uz = 0.0 * z;
        bc.prescribe_all(u32::try_from(v).expect("fits"), [fx(ux), fx(uy), fx(uz)]);
    }
    bc
}

/// Largest absolute stress component over the whole mesh, and where it sits.
fn worst_component(
    stresses: &[alice_physics::linear_elastic_fem::StressTensor],
) -> (f64, &'static str, usize) {
    let mut worst = 0.0_f64;
    let mut name = "xx";
    let mut at = 0usize;
    for (t, s) in stresses.iter().enumerate() {
        for (n, c) in [
            ("xx", s.xx),
            ("yy", s.yy),
            ("zz", s.zz),
            ("xy", s.xy),
            ("yz", s.yz),
            ("zx", s.zx),
        ] {
            let v = c.to_f64().abs();
            if v > worst {
                worst = v;
                name = n;
                at = t;
            }
        }
    }
    (worst, name, at)
}

// ---------------------------------------------------------------------------
// the oracle
// ---------------------------------------------------------------------------

/// **The gate for the geometric-nonlinearity wall.**
///
/// A rigid rotation strains nothing, so every stress component must be
/// identically zero — not small. This is red on the present small-strain
/// solver, by roughly 8.6 GPa on a 3.5 GPa material, and green once the strain
/// measure sees through a rotation (co-rotational formulation, or a full
/// finite-strain one).
///
/// The tolerance is the same `1e-9` MPa the other rigid-motion oracle uses; the
/// answer is exactly representable, so there is nothing for a looser band to
/// buy.
///
/// # Why this is `#[ignore]`d rather than deleted or left red
///
/// **The red is correct.** It is held back only because the implementation has
/// not caught up — not because the assertion, the tolerance or the scene is in
/// doubt. `cargo test -- --ignored` runs it, and the commit that lands a
/// co-rotational strain measure removes this attribute in the same diff.
///
/// ⚠️ **Do not delete it as "a test for an unimplemented feature".** Deleting it
/// loses the measurement that the present solver is wrong by 2.47 × E, and the
/// next reader would have to rediscover it.
///
/// Ignoring it costs no coverage, because
/// [`characterises_the_small_strain_rotation_defect`] is **not** ignored and pins
/// the same numbers from the other side: it asserts the measured stress equals
/// `2λ(c−1) + 2μ(c−1)` to 1e-6 MPa. Any change to the strain measure, the `B`
/// matrix or the material law moves those numbers and reds *that* test in CI. So
/// the pair is: this one is the goal, its companion is the guard, and exactly one
/// of the two is green at any time.
#[test]
#[ignore = "the red is correct: a rigid rotation must carry no stress, and the \
            small-strain solver cannot deliver that yet. Remove this attribute in \
            the commit that lands a co-rotational formulation, and delete \
            `characterises_the_small_strain_rotation_defect` in the same diff. \
            CI coverage is not lost: that companion test is not ignored and pins \
            the same numbers"]
fn rigid_rotation_produces_zero_stress() {
    for turn in [QUARTER, HALF] {
        let mesh = kuhn_box(2, 1, 1, 3.0);
        let bc = prescribe_rigid_rotation(&mesh, turn);
        let out = solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("fully prescribed");

        let (worst, name, at) = worst_component(&out.element_stress);
        eprintln!(
            "  {}: worst |σ| = {:.6e} MPa (σ_{} of tet {})",
            turn.name, worst, name, at
        );

        for (t, s) in out.element_stress.iter().enumerate() {
            for (n, c) in [
                ("xx", s.xx),
                ("yy", s.yy),
                ("zz", s.zz),
                ("xy", s.xy),
                ("yz", s.yz),
                ("zx", s.zx),
            ] {
                let got = c.to_f64();
                assert!(
                    got.abs() <= 1e-9,
                    "{}: tet {t} σ_{n} = {got:.6e} MPa under a rigid rotation, which strains \
                     nothing, so the stress must be zero. A small-strain tensor reads the \
                     rotation as a strain of cos θ − 1 per in-plane axis; see \
                     `characterises_the_small_strain_rotation_defect` for the closed form of \
                     exactly this number",
                    turn.name
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// why it is red — checked, not asserted by inspection
// ---------------------------------------------------------------------------

/// The red above is the small-strain approximation and **nothing else**.
///
/// Per `feedback_prose_oracle_verified_by_red_2026_09_28`, when an oracle reds
/// the first suspect is the oracle. This test removes that suspicion by
/// predicting the failing number in closed form: if the measured stress matches
/// `2λ(c−1) + 2μ(c−1)` component by component, the solver is doing small-strain
/// linear elasticity correctly and the gate above is failing for the one reason
/// it is meant to.
///
/// ⚠️ **Delete this test when a co-rotational formulation lands.** It pins the
/// defect, so it must red exactly when `rigid_rotation_produces_zero_stress`
/// greens. Keeping both would make the suite unsatisfiable.
#[test]
fn characterises_the_small_strain_rotation_defect() {
    let (lambda, mu) = lame();
    for turn in [QUARTER, HALF] {
        let c1 = turn.cos - 1.0;
        let want_xx = 2.0 * lambda * c1 + 2.0 * mu * c1;
        let want_zz = 2.0 * lambda * c1;

        let mesh = kuhn_box(2, 1, 1, 3.0);
        let bc = prescribe_rigid_rotation(&mesh, turn);
        let out = solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("fully prescribed");

        eprintln!(
            "  {}: predicted σ_xx = {want_xx:.6} MPa, σ_zz = {want_zz:.6} MPa",
            turn.name
        );

        for (t, s) in out.element_stress.iter().enumerate() {
            for (n, c, want) in [
                ("xx", s.xx, want_xx),
                ("yy", s.yy, want_xx),
                ("zz", s.zz, want_zz),
                ("xy", s.xy, 0.0),
                ("yz", s.yz, 0.0),
                ("zx", s.zx, 0.0),
            ] {
                let got = c.to_f64();
                assert!(
                    (got - want).abs() <= 1e-6,
                    "{}: tet {t} σ_{n} = {got:.9e} MPa, small-strain closed form \
                     {want:.9e} MPa, difference {:.3e}. The gate's red is then not the \
                     strain measure and must be diagnosed before any nonlinear work",
                    turn.name,
                    (got - want).abs()
                );
            }
        }
    }
}

/// The spurious stress is second order in the angle, so a small-angle test is
/// not a weaker version of the oracle — it is a blind one.
///
/// Reported rather than asserted with a band: the point is the ratio between
/// angles, and the numbers are printed so the scale of the blindness is on the
/// record. The assertion is only the ordering, which no correct solver of either
/// kind can violate for the wrong reason (a finite-strain solver gives zero at
/// every angle and passes trivially).
#[test]
fn small_angles_hide_what_a_quarter_turn_shows() {
    let (lambda, mu) = lame();
    let mut measured: Vec<(f64, f64)> = Vec::new();
    for degrees in [1.0_f64, 5.0, 15.0, 45.0, 90.0] {
        let theta = degrees.to_radians();
        let turn = Turn {
            name: "sweep",
            cos: theta.cos(),
            sin: theta.sin(),
        };
        let mesh = kuhn_box(1, 1, 1, 3.0);
        let bc = prescribe_rigid_rotation(&mesh, turn);
        let out = solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("fully prescribed");
        let (worst, _, _) = worst_component(&out.element_stress);
        let predicted = (2.0 * lambda * (turn.cos - 1.0) + 2.0 * mu * (turn.cos - 1.0)).abs();
        eprintln!(
            "  θ = {degrees:5.1}°  worst |σ| = {worst:12.6} MPa   (−θ²/2 prediction \
             {predicted:12.6} MPa)"
        );
        measured.push((degrees, worst));
    }
    for w in measured.windows(2) {
        assert!(
            w[1].1 >= w[0].1,
            "the spurious stress must not shrink as the rotation grows: {:.1}° gave \
             {:.6} MPa, {:.1}° gave {:.6} MPa",
            w[0].0,
            w[0].1,
            w[1].0,
            w[1].1
        );
    }
}
