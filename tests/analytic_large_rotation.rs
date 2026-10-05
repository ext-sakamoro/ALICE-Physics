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
//! - it **reds on the small-strain `solve`**, by the amount the closed form below
//!   predicts (`characterises_the_small_strain_rotation_defect`), and that is the
//!   evidence the oracle reaches the property;
//! - it **greens on the co-rotational `solve_corotational`**
//!   (`rigid_rotation_produces_zero_stress`), and that is the evidence the
//!   implementation is right.
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

use alice_physics::linear_elastic_fem::{
    solve, solve_corotational, BoundaryConditions, CorotationalConfig, ElasticMaterial,
    SolverConfig,
};
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
/// the mesh is conforming for any `nx, ny, nz`.
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

/// `cos = 4/5`, `sin = 3/5`, about 36.87°. The 3-4-5 triangle keeps `cos² + sin²`
/// equal to one to the last bit of an `f64`, which is all the scene needs.
const THREE_FOUR_FIVE: Turn = Turn {
    name: "36.87° about z (3-4-5)",
    cos: 0.8,
    sin: 0.6,
};

/// `cos = -3/5`, `sin = 4/5`, about 126.87°: past a quarter turn, where a
/// small-strain reading has already changed sign of its error's growth.
const OBTUSE_THREE_FOUR_FIVE: Turn = Turn {
    name: "126.87° about z (3-4-5)",
    cos: -0.6,
    sin: 0.8,
};

/// The co-rotational settings every test below uses. Everything is prescribed,
/// so the conjugate gradient has no free degree of freedom to find; the Newton
/// and polar budgets are the ones `tests/analytic_corotational.rs` uses.
fn corotational_config(increments: u32) -> CorotationalConfig {
    CorotationalConfig::try_new(
        SolverConfig::try_new(200_000, Fix128::from_raw(0, 1 << 34)).expect("valid linear config"),
        32,
        Fix128::from_raw(0, 1 << 34),
        increments,
        32,
    )
    .expect("valid co-rotational config")
}

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
/// # What this solver is, and why the small-strain twin stays
///
/// This runs [`solve_corotational`], whose element strain is
/// `ε = sym(RᵀF − I)` with `R` the polar factor of `F`. For `F = R₀` the polar
/// factor is `R₀` itself, so `ε = 0` and every component is zero to the rounding
/// of the polar iteration, not to a modelling error. The tolerance is the same
/// `1e-9` MPa the other rigid-motion oracle uses.
///
/// It was `#[ignore]`d while the only solver in the crate was the small-strain
/// [`solve`]; the red was correct and is now the green below. ⚠️ **The
/// companion [`characterises_the_small_strain_rotation_defect`] is kept, not
/// deleted**: the original instruction was to delete it in the commit that lands
/// a co-rotational formulation, but [`solve`] still ships and still has that
/// defect, so the companion is the only test that pins what it does. Deleting it
/// would leave the linear solver's rotation behaviour unmeasured.
///
/// The angle list is `90°`, `180°`, `36.87°` and `126.87°`. A 1° row is left out
/// on purpose: at 1° the small-strain error is `θ²/2 ≈ 1.5e-4` of `E`, which a
/// `1e-9` band would still catch, but `cos`/`sin` of 1° are not exact in `f64`,
/// so the scene would no longer carry zero stress and the band would be testing
/// the `f64` rounding of the input.
#[test]
fn rigid_rotation_produces_zero_stress() {
    for turn in [QUARTER, HALF, THREE_FOUR_FIVE, OBTUSE_THREE_FOUR_FIVE] {
        // One increment for every angle. 180° cannot use more: the prescribed
        // displacement is interpolated linearly, so at t = 1/2 the deformation
        // gradient is diag(0, 0, 1) and the polar factor does not exist
        // (`RotationFailed { cause: Inverted }`, measured for 2, 4 and 8
        // increments; recorded as a candidate defect).
        let mesh = kuhn_box(2, 1, 1, 3.0);
        let bc = prescribe_rigid_rotation(&mesh, turn);
        let out = solve_corotational(&mesh, &pla(), &bc, &corotational_config(1))
            .unwrap_or_else(|e| panic!("{}: {e:?}", turn.name));

        let (worst, name, at) = worst_component(&out.field.element_stress);
        eprintln!(
            "  {}: worst |σ| = {:.6e} MPa (σ_{} of tet {})",
            turn.name, worst, name, at
        );

        for (t, s) in out.field.element_stress.iter().enumerate() {
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
                     nothing, so the stress must be zero. The small-strain `solve` reads this \
                     rotation as a strain of cos θ − 1 per in-plane axis; see \
                     `characterises_the_small_strain_rotation_defect`",
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
/// When an oracle reds
/// the first suspect is the oracle. This test removes that suspicion by
/// predicting the failing number in closed form: if the measured stress matches
/// `2λ(c−1) + 2μ(c−1)` component by component, the solver is doing small-strain
/// linear elasticity correctly and the gate above is failing for the one reason
/// it is meant to.
///
/// ⚠️ **This stays after the co-rotational solve landed.** It pins the defect of
/// the small-strain [`solve`], which still ships; `rigid_rotation_produces_zero_stress`
/// now runs [`solve_corotational`] and the two no longer contradict each other.
/// Delete it only with the small-strain solve itself.
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
