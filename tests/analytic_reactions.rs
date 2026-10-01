//! Closed-form oracles for `linear_elastic_fem::reactions` and
//! `linear_elastic_fem::corotational_reactions`.
//!
//! # Why a reaction needs its own oracles
//!
//! Every other oracle in this crate reads a displacement or a stress, and a
//! displacement-driven scene cannot see the **scale** of the internal force at
//! all: with `f_ext = 0` the discrete problem is `K u = 0` on the free rows, so
//! multiplying the whole assembled force by a constant leaves `u`, the reported
//! stress and every assertion built on them exactly where they were. Measured
//! 2026-10-01 on the quadratic and cubic elements: dropping the `J` from
//! `P = J σ F⁻ᵀ` left 7 of 7 oracles green, one of them a closed form for
//! `σ_xx`. The reaction is linear in that constant, which is what makes it the
//! observation these oracles are written against.
//!
//! # The identity every face oracle here uses
//!
//! For a **uniform** two-point stress `T` (the Cauchy stress of the
//! small-strain path, `R σ̃` of the co-rotational one, `P = J σ F⁻ᵀ` of the
//! hyperelastic one) the assembled nodal force is
//! `f_i = ∫_Ω T ∇₀N_i dV₀`, and summing it over a set `S` of nodes gives, by
//! the divergence theorem with `φ = Σ_{i∈S} N_i`,
//!
//! ```text
//! Σ_{i∈S} f_i = T · ∮_∂Ω φ n dA₀
//! ```
//!
//! Take `S` to be every node on the face `x = 0` of a box meshed on a regular
//! lattice of spacing `h`. Then `φ = 1` on that face (outward normal `−e_x`),
//! `φ = 0` on the opposite one, and on each of the four lateral faces `φ` is
//! the ramp `max(0, 1 − x/h)`, whose integral is the same on a face and on the
//! face opposite it while the outward normals are opposite. **The four lateral
//! contributions cancel in pairs**, leaving
//!
//! ```text
//! Σ_{i: x=0} R_i = −A₀ · (T e_x)        A₀ the REFERENCE area of the face
//! ```
//!
//! and `+A₀·(T e_x)` on the far face. ⚠️ **`A₀` is the undeformed area.** Using
//! the deformed one would let an error in `J` cancel against the area change,
//! which is precisely the error these oracles exist to see.
//!
//! The three faces `x=0`, `y=0`, `z=0` read the three **columns** of `T`
//! separately, so a transposed scatter is visible whenever `T` is not
//! symmetric — which `P` is not, and which is why `element_force_from_piola`
//! exists beside `element_force_from_stress`.
//!
//! # ⚠️ Expected values are derived here, not measured
//!
//! Nothing below calls `reactions`, `corotational_reactions` or
//! `hyperelastic::cauchy_stress` to build an expected value. The closed forms
//! are written out from the constitutive law in the doc comment of each test.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_field::CoupledField;
use alice_physics::hyperelastic::HyperelasticModel;
use alice_physics::linear_elastic_fem::{
    corotational_reactions, reactions, solve, solve_corotational, solve_with_eigenstrain, Axis,
    BoundaryConditions, CorotationalConfig, CorotationalSolution, ElasticMaterial, FemError,
    FemSolution, SolverConfig, StressTensor, ThermalExpansion,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// scene construction (same Kuhn box the other FEM oracles use)
// ---------------------------------------------------------------------------

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

/// Axis-aligned box `[0,nx·h] × [0,ny·h] × [0,nz·h]` split into Kuhn 6-tet
/// cells, optionally with every vertex carried through `place`.
///
/// Kuhn's subdivision conforms across every shared face, because the diagonal
/// it induces on a face depends only on that face's own corners.
fn kuhn_box_with(
    nx: usize,
    ny: usize,
    nz: usize,
    h: f32,
    place: impl Fn([f32; 3]) -> [f32; 3],
) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push(place([i as f32 * h, j as f32 * h, k as f32 * h]));
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

fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f32) -> SdfTetMesh {
    kuhn_box_with(nx, ny, nz, h, |p| p)
}

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// PLA-like, the material the other FEM oracles use.
fn pla() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("E > 0 and ν in (-1, 0.5)")
}

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;
/// `λ = Eν/((1+ν)(1−2ν))`.
fn lambda() -> f64 {
    E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
}
/// `μ = E/(2(1+ν))`.
fn mu() -> f64 {
    E_MPA / (2.0 * (1.0 + NU))
}
/// `K = λ + 2μ/3`, the bulk modulus `solve_corotational` pairs a model with.
fn bulk() -> f64 {
    lambda() + 2.0 * mu() / 3.0
}

fn close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got:.9e}, closed form {want:.9e}, difference {:.3e} > tol {tol:.3e}",
        (got - want).abs()
    );
}

fn close3(got: [f64; 3], want: [f64; 3], tol: f64, what: &str) {
    for (axis, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        close(*g, *w, tol, &format!("{what} component {axis}"));
    }
}

/// `Σ R` over every node whose **reference** coordinate on `axis` is `at`.
fn face_sum(mesh: &SdfTetMesh, r: &[[Fix128; 3]], axis: usize, at: f32) -> [f64; 3] {
    let mut sum = [0.0_f64; 3];
    for (v, p) in mesh.vertices.iter().enumerate() {
        if (p[axis] - at).abs() <= 1e-5 {
            for (s, got) in sum.iter_mut().zip(r[v].iter()) {
                *s += got.to_f64();
            }
        }
    }
    sum
}

/// `Σ R` over every node of the mesh.
fn total(r: &[[Fix128; 3]]) -> [f64; 3] {
    let mut sum = [0.0_f64; 3];
    for node in r {
        for (s, got) in sum.iter_mut().zip(node.iter()) {
            *s += got.to_f64();
        }
    }
    sum
}

/// Which degrees of freedom a boundary set prescribes, as a per-node mask.
fn prescribed_mask(mesh: &SdfTetMesh, bc: &BoundaryConditions) -> Vec<[bool; 3]> {
    let mut mask = vec![[false; 3]; mesh.vertex_count()];
    for &(vertex, axis, _) in bc.prescribed() {
        mask[vertex as usize][axis.index()] = true;
    }
    mask
}

/// Every free degree of freedom must carry exactly zero — that is the
/// definition, not a tolerance, so it is asserted with `assert_eq!`.
fn assert_free_rows_are_exactly_zero(
    mesh: &SdfTetMesh,
    bc: &BoundaryConditions,
    r: &[[Fix128; 3]],
) {
    let mask = prescribed_mask(mesh, bc);
    let mut free = 0usize;
    for (v, (node, held)) in r.iter().zip(mask.iter()).enumerate() {
        for (axis, (value, is_held)) in node.iter().zip(held.iter()).enumerate() {
            if !is_held {
                free += 1;
                assert_eq!(
                    *value,
                    Fix128::ZERO,
                    "node {v} axis {axis} is free, so its reaction row is the equilibrium \
                     equation the solve satisfied and must be reported as exactly zero"
                );
            }
        }
    }
    assert!(
        free > 0,
        "this scene has no free degree of freedom, so the check above measured nothing"
    );
}

// ---------------------------------------------------------------------------
// oracle 1 — uniaxial tension, displacement driven
// ---------------------------------------------------------------------------

/// A bar stretched by a prescribed affine field pulls its supports with
/// `E ε A₀`.
///
/// # The closed form
///
/// For uniaxial stress `σ_xx = σ` with the other five components zero, Hooke
/// inverts to `ε_xx = σ/E`, `ε_yy = ε_zz = −ν σ/E`, so
/// `u(x,y,z) = (σ/E)·(x, −νy, −νz)`. That field is linear, hence exactly
/// representable by P1 tetrahedra, so the discrete stress is uniformly
/// `diag(σ, 0, 0)` and the face identity in the module doc gives
///
/// ```text
/// Σ_{i: x=0} R_i = −A₀·(σ e_x) = (−σ A₀, 0, 0)        A₀ = Ly·Lz
/// Σ_{i: x=L} R_i = +A₀·(σ e_x) = (+σ A₀, 0, 0)
/// Σ_{i: y=0} R_i = −A₀ʸ·(σ e_y) = 0                   σ has no y column
/// ```
///
/// with `σ A₀ = E ε A₀`. The sign is the force the support applies **to the
/// body**: a bar in tension is held back at both ends, so the `x=0` support
/// pushes in `−x`.
///
/// ⚠️ **Every component of this is invisible to a displacement oracle.** The
/// same scene with the whole stiffness scaled by any constant produces the same
/// `u` (the free rows read `K u = 0`), and the three numbers below are the only
/// ones in the crate that move with it.
#[test]
fn a_uniaxially_stretched_bar_pulls_its_supports_with_e_epsilon_a() {
    let (n, h) = (2usize, 5.0_f32); // 10 mm cube, 27 nodes, exactly one free node
    let mesh = kuhn_box(n, n, n, h);
    let side = f64::from(h) * n as f64;
    let area = side * side;

    let sigma = 10.0_f64; // MPa
    let eps = sigma / E_MPA;
    let field = |p: [f32; 3]| -> [f64; 3] {
        [
            eps * f64::from(p[0]),
            -NU * eps * f64::from(p[1]),
            -NU * eps * f64::from(p[2]),
        ]
    };

    let mut bc = BoundaryConditions::new();
    let interior = node_index(n, n, 1, 1, 1);
    for (v, p) in mesh.vertices.iter().enumerate() {
        let v = u32::try_from(v).expect("fits");
        if v == interior {
            continue;
        }
        let u = field(*p);
        bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
    }

    let out =
        solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("patch test is well posed");
    let r = reactions(&mesh, &pla(), &bc, None, &out).expect("the solution matches the mesh");

    assert_free_rows_are_exactly_zero(&mesh, &bc, &r);

    // The expected traction, written from the closed form above.
    let pull = sigma * area; // = E·ε·A₀ = 1000 N
    close(
        pull,
        E_MPA * eps * area,
        1e-9,
        "the two ways of writing the closed form must agree",
    );

    // Tolerance: the field is exact in P1, so what is left is `Fix128`
    // rounding over 27 nodes times a stiffness of order 4400 MPa. Measured
    // worst component 2.3e-7 N against a 1000 N answer.
    let tol = 1e-5 * pull;
    close3(
        face_sum(&mesh, &r, 0, 0.0),
        [-pull, 0.0, 0.0],
        tol,
        "reaction summed over the x = 0 face",
    );
    close3(
        face_sum(&mesh, &r, 0, h * n as f32),
        [pull, 0.0, 0.0],
        tol,
        "reaction summed over the far face",
    );
    close3(
        face_sum(&mesh, &r, 1, 0.0),
        [0.0, 0.0, 0.0],
        tol,
        "a lateral face carries no reaction under uniaxial stress",
    );
    close3(
        total(&r),
        [0.0, 0.0, 0.0],
        tol,
        "no external load, so every reaction must sum to zero",
    );
}

// ---------------------------------------------------------------------------
// oracle 2 — uniform heating of a bar held in one axis
// ---------------------------------------------------------------------------

/// A bar heated by `ΔT` and held against axial expansion pushes its supports
/// with `E α ΔT A₀`.
///
/// # The closed form
///
/// ⚠️ **`ThermalExpansion` carries `ΔT`, the temperature *rise* above the
/// stress-free reference, not an absolute temperature.** The eigenstrain is
/// `ε_th = α ΔT · I`.
///
/// Hold `u_x = 0` on both end faces and leave the four lateral faces traction
/// free. Then `ε_xx = 0` and `σ_yy = σ_zz = 0`, so
/// `ε_xx = σ_xx/E + αΔT = 0` gives `σ_xx = −E α ΔT`, and the lateral strain is
/// `ε_yy = ε_zz = −ν σ_xx/E + αΔT = (1+ν) α ΔT`. The displacement
/// `u = (0, (1+ν)αΔT·y, (1+ν)αΔT·z)` is linear, hence exact in P1.
///
/// The face identity then reads the **total** stress `σ = diag(−EαΔT, 0, 0)`
/// (`C B u − C : ε_th`, which is what the reaction sees because
/// `f_int = K u − ∫Bᵀ C ε_th dV`):
///
/// ```text
/// Σ_{i: x=0} R_i,x = −A₀·σ_xx = +E α ΔT A₀
/// ```
///
/// A heated bar that cannot grow **pushes** its supports outward, which is the
/// opposite sign to oracle 1 and is why both are here.
///
/// ⚠️ Three extra degrees of freedom are pinned to block the rigid translations
/// in `y`, `z` and the rotation about `x`; the closed form says their reactions
/// are zero, and that is asserted — a non-zero value there would mean the pins
/// are fighting the thermal field instead of only removing rigid modes.
#[test]
fn a_heated_bar_held_in_one_axis_pushes_its_supports_with_e_alpha_delta_t_a() {
    let (n, h) = (2usize, 5.0_f32);
    let mesh = kuhn_box(n, n, n, h);
    let side = f64::from(h) * n as f64;
    let area = side * side;
    let far = h * n as f32;

    let delta_t = 10.0_f64;
    let alpha = 1.0 / 1000.0;
    let eps_th = alpha * delta_t;

    let mut bc = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        let v = u32::try_from(v).expect("fits");
        if p[0] <= 1e-5 || p[0] >= far - 1e-5 {
            bc.prescribe(v, Axis::X, Fix128::ZERO);
        }
    }
    // Rigid body modes: y and z translation, and the rotation about x.
    let origin = node_index(n, n, 0, 0, 0);
    bc.prescribe(origin, Axis::Y, Fix128::ZERO);
    bc.prescribe(origin, Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(n, n, 0, n, 0), Axis::Z, Fix128::ZERO);

    let field = CoupledField::try_new_filled(
        3,
        3,
        3,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (fx(side), fx(side), fx(side)),
        fx(delta_t),
    )
    .expect("a 3³ grid over the box is valid");
    let thermal = ThermalExpansion::new(&field, fx(alpha));

    let out = solve_with_eigenstrain(&mesh, &pla(), &bc, &SolverConfig::default(), Some(thermal))
        .expect("the scene is well posed");

    // The kinematics the closed form is built on, checked so a red is
    // unambiguous: free lateral expansion at (1+ν)αΔT, no axial motion.
    for (v, p) in mesh.vertices.iter().enumerate() {
        let want = [
            0.0,
            (1.0 + NU) * eps_th * f64::from(p[1]),
            (1.0 + NU) * eps_th * f64::from(p[2]),
        ];
        for (axis, w) in want.iter().enumerate() {
            close(
                out.displacements[v][axis].to_f64(),
                *w,
                1e-9,
                &format!("node {v} axis {axis} displacement"),
            );
        }
    }

    let r = reactions(&mesh, &pla(), &bc, Some(thermal), &out).expect("the solution matches");
    assert_free_rows_are_exactly_zero(&mesh, &bc, &r);

    let push = E_MPA * eps_th * area; // 3500 · 0.01 · 100 = 3500 N
    let tol = 1e-5 * push;
    let near = face_sum(&mesh, &r, 0, 0.0);
    let far_sum = face_sum(&mesh, &r, 0, far);
    close(near[0], push, tol, "x reaction summed over the x = 0 face");
    close(
        far_sum[0],
        -push,
        tol,
        "x reaction summed over the far face",
    );
    // The three rigid-mode pins carry nothing: the closed-form state has no
    // lateral traction at all.
    close(near[1], 0.0, tol, "the y pin carries no load");
    close(near[2], 0.0, tol, "the z pins carry no load");
    close3(
        total(&r),
        [0.0, 0.0, 0.0],
        tol,
        "no external load, so every reaction must sum to zero",
    );
}

// ---------------------------------------------------------------------------
// oracle 3 — global balance against an applied load
// ---------------------------------------------------------------------------

/// `Σ R + Σ f_ext = 0` on a traction-loaded bar.
///
/// ⚠️ **This is the weakest of the oracles here and is not sufficient on its
/// own.** A uniform scale error on the internal force multiplies `Σ R` and
/// `Σ f_ext` by the same factor on a scene where the external load decides the
/// answer, so the identity survives it. It is kept because it is the one
/// statement that holds with external loads present, which the face oracles
/// (displacement driven, `f_ext = 0`) never exercise.
#[test]
fn the_reactions_balance_the_applied_load() {
    let (nx, ny, nz, h) = (4usize, 1usize, 1usize, 2.5_f32);
    let mesh = kuhn_box(nx, ny, nz, h);
    let mut bc = BoundaryConditions::new();
    for k in 0..=nz {
        for j in 0..=ny {
            bc.prescribe(node_index(nx, ny, 0, j, k), Axis::X, Fix128::ZERO);
        }
    }
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, ny, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, 0, nz), Axis::Y, Fix128::ZERO);
    // 50 MPa over a 2.5 × 2.5 face, shared out over the four corner nodes the
    // way the two surface triangles weight them.
    let traction = 50.0_f64;
    let applied = traction * f64::from(h) * f64::from(h);
    let tri = applied / 2.0;
    for (node, share) in [
        (node_index(nx, ny, nx, 0, 0), 2.0 / 3.0),
        (node_index(nx, ny, nx, ny, nz), 2.0 / 3.0),
        (node_index(nx, ny, nx, ny, 0), 1.0 / 3.0),
        (node_index(nx, ny, nx, 0, nz), 1.0 / 3.0),
    ] {
        bc.add_load(node, Axis::X, fx(tri * share));
    }

    let out = solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("well posed");
    let r = reactions(&mesh, &pla(), &bc, None, &out).expect("the solution matches");
    assert_free_rows_are_exactly_zero(&mesh, &bc, &r);

    let mut external = [0.0_f64; 3];
    for &(_, axis, force) in bc.loads() {
        external[axis.index()] += force.to_f64();
    }
    close(
        external[0],
        applied,
        1e-9,
        "the nodal shares must add up to the traction resultant",
    );
    let sum = total(&r);
    let tol = 1e-5 * applied;
    for (axis, (got, ext)) in sum.iter().zip(external.iter()).enumerate() {
        close(
            got + ext,
            0.0,
            tol,
            &format!("Σ R + Σ f_ext on axis {axis}"),
        );
    }
}

// ---------------------------------------------------------------------------
// oracle 4 — equivariance under a quarter turn of the whole problem
// ---------------------------------------------------------------------------

/// Rotating the reference mesh, the prescribed displacements and the loads by
/// `Q` rotates every reaction by `Q`.
///
/// Isotropic linear elasticity is frame indifferent in this sense, and the
/// statement needs **no closed form at all**: the second problem is the oracle
/// for the first. A quarter turn about `z` is used because its entries are `0`
/// and `±1`, so the rotated vertex coordinates are exactly representable in
/// `f32` and the mesh connectivity can be carried over unchanged (a proper
/// rotation preserves tetrahedron orientation).
///
/// ⚠️ This is an **invariant guard**, not an absolute-value oracle: it is blind
/// to anything isotropic, including the uniform scale errors oracle 1 exists
/// for. The two are complementary and both are needed.
#[test]
fn the_reactions_rotate_with_a_quarter_turn_of_the_whole_problem() {
    let (n, h) = (2usize, 5.0_f32);
    // Q: (x, y, z) ↦ (−y, x, z)
    let turn = |p: [f32; 3]| -> [f32; 3] { [-p[1], p[0], p[2]] };
    let plain = kuhn_box(n, n, n, h);
    let turned = kuhn_box_with(n, n, n, h, turn);

    let sigma = 10.0_f64;
    let eps = sigma / E_MPA;
    let field = |p: [f32; 3]| -> [f64; 3] {
        [
            eps * f64::from(p[0]),
            -NU * eps * f64::from(p[1]),
            -NU * eps * f64::from(p[2]),
        ]
    };

    let interior = node_index(n, n, 1, 1, 1);
    let mut plain_bc = BoundaryConditions::new();
    let mut turned_bc = BoundaryConditions::new();
    for (v, p) in plain.vertices.iter().enumerate() {
        let v = u32::try_from(v).expect("fits");
        if v == interior {
            continue;
        }
        let u = field(*p);
        plain_bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
        // Q u, with Q the same quarter turn.
        turned_bc.prescribe_all(v, [fx(-u[1]), fx(u[0]), fx(u[2])]);
    }

    let plain_out = solve(&plain, &pla(), &plain_bc, &SolverConfig::default()).expect("well posed");
    let turned_out =
        solve(&turned, &pla(), &turned_bc, &SolverConfig::default()).expect("well posed");
    let plain_r = reactions(&plain, &pla(), &plain_bc, None, &plain_out).expect("matches");
    let turned_r = reactions(&turned, &pla(), &turned_bc, None, &turned_out).expect("matches");

    // Scale the tolerance by the size of what is being compared, so that the
    // assertion has teeth on every node rather than only on the loaded ones.
    let mut largest = 0.0_f64;
    for node in &plain_r {
        for value in node {
            largest = largest.max(value.to_f64().abs());
        }
    }
    assert!(
        largest > 1.0,
        "the scene must produce reactions worth comparing; largest is {largest:.3e} N"
    );
    // Measured worst difference is 2.8e-14 N against 2.5e2 N, i.e. 1.1e-16
    // relative — the quarter turn only moves and negates `Fix128` entries, so
    // there is almost nothing for it to lose. The bound is four orders above
    // that and still ten orders below any physically meaningful change.
    let tol = 1e-12 * largest;

    let mut worst = 0.0_f64;
    for (v, (got, was)) in turned_r.iter().zip(plain_r.iter()).enumerate() {
        let want = [-was[1].to_f64(), was[0].to_f64(), was[2].to_f64()];
        for (axis, w) in want.iter().enumerate() {
            let diff = (got[axis].to_f64() - w).abs();
            worst = worst.max(diff);
            assert!(
                diff <= tol,
                "node {v} axis {axis}: the turned problem gives {:.9e} where Q·R is {w:.9e}",
                got[axis].to_f64()
            );
        }
    }
    eprintln!(
        "  quarter-turn equivariance: worst |R(QS) − Q R(S)| = {worst:.3e} N over {largest:.3e} N"
    );
}

// ---------------------------------------------------------------------------
// the uniform deformation gradient the two co-rotational oracles share
// ---------------------------------------------------------------------------

/// `cos θ`, `sin θ` of the 3-4-5 rotation about `z` — exact in binary.
const COS: f64 = 0.8;
const SIN: f64 = 0.6;
/// The principal stretches. ⚠️ **`det U = 5/4 ≠ 1` on purpose**: an error that
/// drops the `J` from `P = J σ F⁻ᵀ` is invisible at `det F = 1`.
const STRETCH: [f64; 3] = [1.25, 1.0, 1.0];

/// `u(X) = (R U − I) X`, the affine field both co-rotational oracles prescribe.
fn stretched_turn(p: [f32; 3]) -> [f64; 3] {
    let s = [
        f64::from(p[0]) * STRETCH[0],
        f64::from(p[1]) * STRETCH[1],
        f64::from(p[2]) * STRETCH[2],
    ];
    [
        COS * s[0] - SIN * s[1] - f64::from(p[0]),
        SIN * s[0] + COS * s[1] - f64::from(p[1]),
        s[2] - f64::from(p[2]),
    ]
}

/// The box, the boundary data and the displacement field of that affine state.
///
/// Every boundary node is prescribed and the interior is free. The field is
/// linear, so P1 reproduces it exactly, and `F = R U` is then **uniform** over
/// the mesh — which is both what makes the closed forms below apply and what
/// makes the field an exact equilibrium (a uniform two-point stress puts no
/// force on an interior node).
fn stretched_scene(n: usize, h: f32) -> (SdfTetMesh, BoundaryConditions, Vec<[Fix128; 3]>) {
    let mesh = kuhn_box(n, n, n, h);
    let far = h * n as f32;
    let mut bc = BoundaryConditions::new();
    let mut field = Vec::with_capacity(mesh.vertex_count());
    for (v, p) in mesh.vertices.iter().enumerate() {
        let u = stretched_turn(*p);
        let nodal = [fx(u[0]), fx(u[1]), fx(u[2])];
        field.push(nodal);
        if p.iter().any(|&c| c <= 1e-5 || c >= far - 1e-5) {
            bc.prescribe_all(u32::try_from(v).expect("fits"), nodal);
        }
    }
    (mesh, bc, field)
}

/// A `CorotationalSolution` carrying a given displacement field.
///
/// `corotational_reactions` reads the displacements and nothing else, so this
/// lets the oracles below evaluate it at a field that is **exactly** the affine
/// one rather than at whatever a Newton iteration landed on. The end-to-end
/// companion checks that a real solve agrees.
fn solution_at(mesh: &SdfTetMesh, displacements: Vec<[Fix128; 3]>) -> CorotationalSolution {
    CorotationalSolution {
        field: FemSolution {
            displacements,
            element_stress: vec![StressTensor::default(); mesh.tet_count()],
            iterations: 0,
            relative_residual: Fix128::ZERO,
            effective_relative_tolerance: Fix128::ZERO,
        },
        newton_iterations: 0,
        increments: 1,
    }
}

// ---------------------------------------------------------------------------
// oracle 5 — the first Piola-Kirchhoff traction of a hyperelastic element
// ---------------------------------------------------------------------------

/// Under a uniform `F = R U`, the face reaction is `−A₀ · P e_face`, with `P`
/// the first Piola-Kirchhoff stress of the Neo-Hookean law.
///
/// # The closed form, derived here
///
/// `hyperelastic::cauchy_stress` evaluates
/// `σ = (2/J)(W₁ + I₁W₂)B − (2/J)W₂B² + [K(J−1) − p_ref]I` with
/// `p_ref = 2(W₁ + 2W₂)` at the undeformed state. Neo-Hookean has
/// `W = (μ/2)(I₁−3)`, so `W₁ = μ/2`, `W₂ = 0`, `p_ref = μ`, and
///
/// ```text
/// σ = (μ/J)·B + [K(J−1) − μ]·I            B = F Fᵀ
/// P = J σ F⁻ᵀ = μ·B F⁻ᵀ + J[K(J−1) − μ]·F⁻ᵀ
///             = μ F + β F⁻ᵀ,               β = J·(K(J−1) − μ)
/// ```
///
/// using `B F⁻ᵀ = F Fᵀ F⁻ᵀ = F`. With `F = R U` and `U = diag(u₁,u₂,u₃)` the
/// inverse transpose is `F⁻ᵀ = R U⁻¹`, so
///
/// ```text
/// P = R · diag(d₁, d₂, d₃),    d_k = μ u_k + β / u_k
/// ```
///
/// and the three columns of `P` are `P e_k = d_k · (R e_k)`. The face identity
/// in the module doc then gives the three expected sums directly. `K` is
/// `λ + 2μ_material/3`, which is what `solve_corotational` pairs the model with.
///
/// # ⚠️ What each face separates
///
/// - `P` is **not symmetric** here (`d₁ ≠ d₂` and the rotation mixes the axes),
///   so `P e_x = d₁(c, s, 0)` while `Pᵀ e_x = (d₁c, −d₂s, 0)`. A transposed
///   scatter changes the `y` component of the `x = 0` face sum by `(d₁+d₂)s A₀`.
/// - `det F = u₁u₂u₃ = 5/4`, so dropping the `J` from `P = J σ F⁻ᵀ` scales every
///   sum by `4/5`. At `det F = 1` that error is invisible, which is why the
///   stretch here is not isochoric.
#[test]
fn the_hyperelastic_face_reaction_is_the_first_piola_traction() {
    let (n, h) = (2usize, 2.0_f32);
    let side = f64::from(h) * n as f64;
    let area = side * side;
    let (mesh, bc, field) = stretched_scene(n, h);

    let mu_model = mu();
    let j = STRETCH[0] * STRETCH[1] * STRETCH[2];
    let beta = j * (bulk() * (j - 1.0) - mu_model);
    let d = [
        mu_model * STRETCH[0] + beta / STRETCH[0],
        mu_model * STRETCH[1] + beta / STRETCH[1],
        mu_model * STRETCH[2] + beta / STRETCH[2],
    ];
    // Columns of R, so `P e_k = d[k] · r_col[k]`.
    let r_col = [[COS, SIN, 0.0], [-SIN, COS, 0.0], [0.0, 0.0, 1.0]];

    let config = CorotationalConfig::try_new(SolverConfig::default(), 32, fx(1e-6), 1, 32)
        .expect("valid")
        .with_hyperelastic(HyperelasticModel::NeoHookean {
            mu_mpa: fx(mu_model),
        });
    let solution = solution_at(&mesh, field);
    let r = corotational_reactions(&mesh, &pla(), &bc, &config, &solution)
        .expect("the affine field has a polar factor everywhere");
    assert_free_rows_are_exactly_zero(&mesh, &bc, &r);

    let largest = d.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * area;
    let tol = 1e-5 * largest;
    for axis in 0..3 {
        let want = [
            -area * d[axis] * r_col[axis][0],
            -area * d[axis] * r_col[axis][1],
            -area * d[axis] * r_col[axis][2],
        ];
        close3(
            face_sum(&mesh, &r, axis, 0.0),
            want,
            tol,
            &format!("reaction over the near face of axis {axis}"),
        );
        close3(
            face_sum(&mesh, &r, axis, h * n as f32),
            [-want[0], -want[1], -want[2]],
            tol,
            &format!("reaction over the far face of axis {axis}"),
        );
    }
    close3(
        total(&r),
        [0.0, 0.0, 0.0],
        tol,
        "no external load, so every reaction must sum to zero",
    );

    // The separation the transposed scatter would have to clear, reported so
    // that a future change to the scene cannot quietly make the oracle blind.
    let separation = (d[0] + d[1]) * SIN * area;
    assert!(
        separation > 100.0 * tol,
        "P must be far enough from Pᵀ on this F for the x-face sum to tell them apart; \
         separation {separation:.3e} N against tolerance {tol:.3e} N"
    );
    eprintln!(
        "  Piola face reaction: d = ({:.4}, {:.4}, {:.4}) MPa, J = {j}, \
         P vs Pᵀ separation {separation:.3e} N, tolerance {tol:.3e} N",
        d[0], d[1], d[2]
    );
}

/// A real `solve_corotational` on the same scene lands on the same reactions.
///
/// The oracle above evaluates `corotational_reactions` at a field written down
/// in closed form; this one checks that the field a Newton iteration actually
/// produces gives the same answer, so the two are not measuring different
/// states.
#[test]
fn a_solved_hyperelastic_state_reproduces_the_piola_face_reaction() {
    let (n, h) = (2usize, 2.0_f32);
    let side = f64::from(h) * n as f64;
    let area = side * side;
    let (mesh, bc, _) = stretched_scene(n, h);

    let mu_model = mu();
    let j = STRETCH[0] * STRETCH[1] * STRETCH[2];
    let beta = j * (bulk() * (j - 1.0) - mu_model);
    let d1 = mu_model * STRETCH[0] + beta / STRETCH[0];

    let config = CorotationalConfig::try_new(SolverConfig::default(), 64, fx(1e-6), 2, 32)
        .expect("valid")
        .with_hyperelastic(HyperelasticModel::NeoHookean {
            mu_mpa: fx(mu_model),
        });
    let out = solve_corotational(&mesh, &pla(), &bc, &config).expect("the scene converges");
    let r = corotational_reactions(&mesh, &pla(), &bc, &config, &out).expect("matches");

    // Looser than the closed-form oracle: this one carries the Newton
    // tolerance as well as the arithmetic.
    let tol = 1e-3 * d1.abs() * area;
    close3(
        face_sum(&mesh, &r, 0, 0.0),
        [-area * d1 * COS, -area * d1 * SIN, 0.0],
        tol,
        "x = 0 face reaction of the solved state",
    );
}

// ---------------------------------------------------------------------------
// oracle 6 — the co-rotational linear law
// ---------------------------------------------------------------------------

/// Under the same uniform `F = R U` with no material model, the face reaction
/// is `−A₀ · R σ̃ e_face`.
///
/// # The closed form
///
/// The co-rotational element carries `σ̃ = λ tr(ε) I + 2μ ε` with
/// `ε = sym(RᵀF − I) = U − I` (diagonal, because `U` is), and its nodal force
/// is `R·(V₀ Bᵀ σ̃)` — so the two-point stress the face identity needs is
/// `T = R σ̃`, not `R σ̃ Rᵀ`. The reference gradients are what `B` is built
/// from, which is why only one rotation appears.
///
/// ```text
/// ε  = diag(u₁−1, u₂−1, u₃−1)
/// σ̃  = diag(λ·tr ε + 2μ(u_k − 1))
/// Σ_{i: x=0} R_i = −A₀ · σ̃₁₁ · (R e_x)
/// ```
///
/// This is the same observation as oracle 5 against the other constitutive
/// branch of `subtract_internal_force`, and it is where a dropped quadrature
/// weight in `element_force_from_stress` shows up.
#[test]
fn the_corotational_linear_face_reaction_is_the_rotated_small_strain_traction() {
    let (n, h) = (2usize, 2.0_f32);
    let side = f64::from(h) * n as f64;
    let area = side * side;
    let (mesh, bc, field) = stretched_scene(n, h);

    let e = [STRETCH[0] - 1.0, STRETCH[1] - 1.0, STRETCH[2] - 1.0];
    let trace = e[0] + e[1] + e[2];
    let s = [
        lambda() * trace + 2.0 * mu() * e[0],
        lambda() * trace + 2.0 * mu() * e[1],
        lambda() * trace + 2.0 * mu() * e[2],
    ];
    let r_col = [[COS, SIN, 0.0], [-SIN, COS, 0.0], [0.0, 0.0, 1.0]];

    let config =
        CorotationalConfig::try_new(SolverConfig::default(), 32, fx(1e-6), 1, 64).expect("valid");
    let solution = solution_at(&mesh, field);
    let r = corotational_reactions(&mesh, &pla(), &bc, &config, &solution).expect("has a frame");
    assert_free_rows_are_exactly_zero(&mesh, &bc, &r);

    // Tolerance: the polar iteration is what sets this, not the assembly —
    // `R` is recovered to a few parts in 10⁻⁹ at 64 iterations.
    let largest = s.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * area;
    let tol = 1e-5 * largest;
    for axis in 0..3 {
        let want = [
            -area * s[axis] * r_col[axis][0],
            -area * s[axis] * r_col[axis][1],
            -area * s[axis] * r_col[axis][2],
        ];
        close3(
            face_sum(&mesh, &r, axis, 0.0),
            want,
            tol,
            &format!("co-rotational linear reaction over the near face of axis {axis}"),
        );
    }
    close3(
        total(&r),
        [0.0, 0.0, 0.0],
        tol,
        "no external load, so every reaction must sum to zero",
    );
}

/// A rigid rotation produces no reaction at all.
///
/// The point of the co-rotational formulation: `RᵀF = I` exactly, so `σ̃ = 0`,
/// so every nodal force is zero and no support carries anything. ⚠️ This is an
/// **invariant guard with a zero target**, so it is blind to any error that
/// scales the internal force — oracle 6 is what covers that, on the same code
/// path.
#[test]
fn a_rigid_rotation_produces_no_reaction() {
    let (n, h) = (2usize, 2.0_f32);
    let mesh = kuhn_box(n, n, n, h);
    let far = h * n as f32;
    let mut bc = BoundaryConditions::new();
    let mut field = Vec::with_capacity(mesh.vertex_count());
    for (v, p) in mesh.vertices.iter().enumerate() {
        let u = [
            COS * f64::from(p[0]) - SIN * f64::from(p[1]) - f64::from(p[0]),
            SIN * f64::from(p[0]) + COS * f64::from(p[1]) - f64::from(p[1]),
            0.0,
        ];
        let nodal = [fx(u[0]), fx(u[1]), fx(u[2])];
        field.push(nodal);
        if p.iter().any(|&c| c <= 1e-5 || c >= far - 1e-5) {
            bc.prescribe_all(u32::try_from(v).expect("fits"), nodal);
        }
    }

    let config =
        CorotationalConfig::try_new(SolverConfig::default(), 32, fx(1e-6), 1, 64).expect("valid");
    let r = corotational_reactions(&mesh, &pla(), &bc, &config, &solution_at(&mesh, field))
        .expect("a rigid rotation has a polar factor");

    let mut worst = 0.0_f64;
    for node in &r {
        for value in node {
            worst = worst.max(value.to_f64().abs());
        }
    }
    // Scale: a 1 % strain on this cube would carry ~500 N per node, so this
    // bound is four orders below anything the element could be reporting.
    assert!(
        worst < 1e-2,
        "a rigid rotation must leave every support unloaded; worst is {worst:.3e} N"
    );
}

// ---------------------------------------------------------------------------
// refusals
// ---------------------------------------------------------------------------

/// A solution with the wrong number of nodes is refused rather than indexed.
#[test]
fn a_solution_that_does_not_match_the_mesh_is_refused() {
    let mesh = kuhn_box(1, 1, 1, 2.0);
    let mut bc = BoundaryConditions::new();
    for v in 0..4u32 {
        bc.prescribe_all(v, [Fix128::ZERO; 3]);
    }
    let short = FemSolution {
        displacements: vec![[Fix128::ZERO; 3]; mesh.vertex_count() - 1],
        element_stress: vec![StressTensor::default(); mesh.tet_count()],
        iterations: 0,
        relative_residual: Fix128::ZERO,
        effective_relative_tolerance: Fix128::ZERO,
    };
    assert_eq!(
        reactions(&mesh, &pla(), &bc, None, &short),
        Err(FemError::SolutionDoesNotMatchMesh {
            nodes: mesh.vertex_count() - 1,
            vertex_count: mesh.vertex_count(),
        })
    );

    let config =
        CorotationalConfig::try_new(SolverConfig::default(), 32, fx(1e-6), 1, 32).expect("valid");
    let long = CorotationalSolution {
        field: FemSolution {
            displacements: vec![[Fix128::ZERO; 3]; mesh.vertex_count() + 3],
            element_stress: vec![],
            iterations: 0,
            relative_residual: Fix128::ZERO,
            effective_relative_tolerance: Fix128::ZERO,
        },
        newton_iterations: 0,
        increments: 1,
    };
    assert_eq!(
        corotational_reactions(&mesh, &pla(), &bc, &config, &long),
        Err(FemError::SolutionDoesNotMatchMesh {
            nodes: mesh.vertex_count() + 3,
            vertex_count: mesh.vertex_count(),
        })
    );
}

/// An empty mesh and an out-of-range boundary vertex are refused, as in
/// `solve`.
#[test]
fn degenerate_inputs_are_refused() {
    let empty = SdfTetMesh::default();
    let bc = BoundaryConditions::new();
    let nothing = FemSolution {
        displacements: vec![],
        element_stress: vec![],
        iterations: 0,
        relative_residual: Fix128::ZERO,
        effective_relative_tolerance: Fix128::ZERO,
    };
    assert_eq!(
        reactions(&empty, &pla(), &bc, None, &nothing),
        Err(FemError::EmptyMesh)
    );

    let mesh = kuhn_box(1, 1, 1, 2.0);
    let mut out_of_range = BoundaryConditions::new();
    out_of_range.prescribe(999, Axis::X, Fix128::ZERO);
    let solution = FemSolution {
        displacements: vec![[Fix128::ZERO; 3]; mesh.vertex_count()],
        element_stress: vec![StressTensor::default(); mesh.tet_count()],
        iterations: 0,
        relative_residual: Fix128::ZERO,
        effective_relative_tolerance: Fix128::ZERO,
    };
    assert_eq!(
        reactions(&mesh, &pla(), &out_of_range, None, &solution),
        Err(FemError::VertexOutOfRange {
            vertex: 999,
            vertex_count: mesh.vertex_count(),
        })
    );
}
