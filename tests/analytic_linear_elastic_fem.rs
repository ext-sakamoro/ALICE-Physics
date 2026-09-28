//! Closed-form oracles for `alice_physics::linear_elastic_fem`.
//!
//! Every expected value here comes from the equations of linear elasticity, not
//! from running the solver. Where a tolerance appears, the comment says what it
//! is made of.
//!
//! # Why these scenes
//!
//! P1 tetrahedra represent a *linear* displacement field exactly, so for any
//! field of the form `u(x) = A·x + b` the discrete solution is the analytic one
//! up to solver tolerance and `Fix128` rounding — no discretisation error at
//! all. That makes uniaxial tension, hydrostatic compression and rigid body
//! motion **exact** oracles rather than asymptotic ones, and it means a failure
//! is a bug in `B`, `D`, the assembly or the solver, never "the mesh was too
//! coarse".
//!
//! Bending is deliberately *not* here. Euler-Bernoulli tip deflection is only
//! the limit of the FEM answer, and P1 tets converge to it from below slowly
//! (shear locking), so a tolerance on it would be pinning the mesh resolution
//! rather than the solver. That oracle belongs with a convergence study.
//!
//! The meshes are built directly rather than through `sdf_fem_mesh::generate`,
//! because the generator drops every cube that straddles the surface, so its
//! domain is not the box the closed form is written for (a 20 mm box at 4 mm
//! cells meshes only the inner 16 mm). That is a mesher property and would be
//! measured here as if it were solver error
//! ([[feedback_oracle_scene_hits_verifier_limit]]). Kuhn's 6-tet subdivision,
//! used below, conforms across every shared face because the diagonal it
//! induces on a face depends only on that face's own corners.
//!
//! The generator was also non-conforming when these oracles were written; that
//! is fixed, and `tests/mesh_conformity.rs` now pins it. **Do not add a patch
//! test over there to check conformity**: two triangulations of a square
//! interpolate a linear function identically, and the element contributions
//! telescope to zero over any geometric partition of the domain whatever the
//! faces look like, so a patch test on a non-conforming mesh comes back exact
//! (measured at 3.6e-15 MPa). Only a solution that is not linear across the
//! face sees the difference.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::filament_db::{MaterialCategory, MaterialProperties};
use alice_physics::linear_elastic_fem::{
    solve, stiffness_diagonal_stats, Axis, BoundaryConditions, ElasticMaterial, Preconditioner,
    SolverConfig, StressTensor,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// scene construction
// ---------------------------------------------------------------------------

/// Node index within an `(nx+1) × (ny+1) × (nz+1)` lattice.
fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

/// Axis-aligned box `[0,nx·h] × [0,ny·h] × [0,nz·h]` split into Kuhn 6-tet cells.
///
/// Kuhn's subdivision cuts each cube into the six tetrahedra that share the
/// main diagonal, one per ordering of the three unit steps. Every face diagonal
/// it produces is the one between that face's `(0,0)` and `(1,1)` corners, which
/// both cubes sharing the face agree on, so the mesh is conforming for any
/// `nx, ny, nz`.
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
    // the six orderings of the unit steps x, y, z
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

/// PLA-like: 3.5 GPa, ν = 0.35. Chosen so `1−2ν = 0.30` stays well away from
/// the incompressible limit where λ diverges.
fn pla() -> ElasticMaterial {
    ElasticMaterial::new(fx(3500.0), fx(0.35)).expect("E > 0 and ν in (-1, 0.5)")
}

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;

fn close(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() <= tol,
        "{what}: got {g:.12e}, closed form {want:.12e}, difference {:.3e} > tol {tol:.3e}",
        (g - want).abs()
    );
}

// ---------------------------------------------------------------------------
// oracle 1 — patch test, uniaxial tension
// ---------------------------------------------------------------------------

/// Prescribing the exact linear field on the boundary must reproduce it inside.
///
/// Oracle: for uniaxial stress `σ_xx = σ` with the other five components zero,
/// Hooke's law inverts to `ε_xx = σ/E`, `ε_yy = ε_zz = −ν σ/E`, so
/// `u(x,y,z) = (σ/E)·(x, −ν y, −ν z)`. This field is linear, hence exactly
/// representable by P1 tetrahedra, so **every** element must report
/// `σ_xx = σ` and zero elsewhere, and the interior node must land on the field.
///
/// This is the standard FEM patch test (Irons): it fails if `B`, `D`, the
/// element volume, the assembly or the Dirichlet elimination is wrong.
#[test]
fn patch_test_uniaxial_tension_is_exact() {
    let mesh = kuhn_box(2, 2, 2, 5.0); // 27 nodes, 48 tets, exactly one interior node
    assert_eq!(mesh.vertex_count(), 27);
    assert_eq!(mesh.tet_count(), 48);

    let sigma = 10.0_f64; // MPa
    let exx = sigma / E_MPA;
    let field = |p: [f32; 3]| -> [f64; 3] {
        [
            exx * f64::from(p[0]),
            -NU * exx * f64::from(p[1]),
            -NU * exx * f64::from(p[2]),
        ]
    };

    let mut bc = BoundaryConditions::new();
    let interior = node_index(2, 2, 1, 1, 1);
    for (v, p) in mesh.vertices.iter().enumerate() {
        let v = u32::try_from(v).expect("fits");
        if v == interior {
            continue; // the one degree of freedom the solver has to find
        }
        let u = field(*p);
        bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
    }
    assert_eq!(bc.prescribed_count(), 26 * 3);

    let out =
        solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("patch test is well posed");

    // interior node sits on the analytic field
    let want = field(mesh.vertices[interior as usize]);
    for (axis, w) in Axis::ALL.into_iter().zip(want) {
        close(
            out.displacements[interior as usize][axis.index()],
            w,
            1e-12,
            &format!("interior node u_{axis:?}"),
        );
    }

    // every element carries the same uniaxial stress
    for (t, s) in out.element_stress.iter().enumerate() {
        close(s.xx, sigma, 1e-9, &format!("tet {t} σ_xx"));
        close(s.yy, 0.0, 1e-9, &format!("tet {t} σ_yy"));
        close(s.zz, 0.0, 1e-9, &format!("tet {t} σ_zz"));
        close(s.xy, 0.0, 1e-9, &format!("tet {t} σ_xy"));
        close(s.yz, 0.0, 1e-9, &format!("tet {t} σ_yz"));
        close(s.zx, 0.0, 1e-9, &format!("tet {t} σ_zx"));
        close(s.von_mises(), sigma, 1e-9, &format!("tet {t} von Mises"));
    }
    close(out.max_von_mises_mpa(), sigma, 1e-9, "max von Mises");
}

// ---------------------------------------------------------------------------
// oracle 2 — patch test, hydrostatic compression
// ---------------------------------------------------------------------------

/// Under pressure `p` the stress is `σ = −p·I` and the strain is isotropic:
/// `ε = −p(1−2ν)/E` on each axis, so `u(x) = −p(1−2ν)/E · x`.
///
/// This exercises the part of `D` that uniaxial tension leaves nearly idle —
/// the off-diagonal `λ` block — and pins the von Mises formula from the other
/// side, because a purely hydrostatic state has **zero** equivalent stress.
#[test]
fn patch_test_hydrostatic_is_exact() {
    let mesh = kuhn_box(2, 2, 2, 5.0);
    let pressure = 12.0_f64; // MPa
    let eps = -pressure * (1.0 - 2.0 * NU) / E_MPA;

    let mut bc = BoundaryConditions::new();
    let interior = node_index(2, 2, 1, 1, 1);
    for (v, p) in mesh.vertices.iter().enumerate() {
        let v = u32::try_from(v).expect("fits");
        if v == interior {
            continue;
        }
        bc.prescribe_all(
            v,
            [
                fx(eps * f64::from(p[0])),
                fx(eps * f64::from(p[1])),
                fx(eps * f64::from(p[2])),
            ],
        );
    }

    let out = solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("well posed");

    for (t, s) in out.element_stress.iter().enumerate() {
        close(s.xx, -pressure, 1e-9, &format!("tet {t} σ_xx"));
        close(s.yy, -pressure, 1e-9, &format!("tet {t} σ_yy"));
        close(s.zz, -pressure, 1e-9, &format!("tet {t} σ_zz"));
        close(s.hydrostatic(), -pressure, 1e-9, &format!("tet {t} mean"));
        // deviatoric part is identically zero, so von Mises must vanish
        close(s.von_mises(), 0.0, 1e-9, &format!("tet {t} von Mises"));
    }
}

// ---------------------------------------------------------------------------
// oracle 3 — simple shear, all three shear rows at once
// ---------------------------------------------------------------------------

/// The field `u = (γ₁y + γ₃z, γ₂z, 0)` is linear, so P1 reproduces it exactly,
/// and it has **no** normal strain: differentiating gives `ε_xx = ε_yy = ε_zz =
/// 0` and engineering shears `γ_xy = γ₁`, `γ_yz = γ₂`, `γ_zx = γ₃`. Hooke's law
/// then gives `σ_xy = μγ₁`, `σ_yz = μγ₂`, `σ_zx = μγ₃` and zero normal stress
/// (the trace vanishes, so `λ` contributes nothing), and
/// `von Mises = √(3(σ_xy² + σ_yz² + σ_zx²))`.
///
/// The three shear rates are deliberately **distinct**, so swapping any two
/// rows of `B` — or of `D`'s shear block — moves a component and is caught.
///
/// This test exists because it was missing: the tension, hydrostatic and rigid
/// body scenes all have identically zero shear strain, so swapping two shear
/// rows of `B` left every one of them green. Only the force-driven bar noticed,
/// and it noticed for the wrong reason (a different operator, not a measured
/// shear).
#[test]
fn patch_test_simple_shear_is_exact() {
    let mesh = kuhn_box(2, 2, 2, 5.0);
    let (g1, g2, g3) = (1.0e-3_f64, 2.0e-3, 3.0e-3);
    let field = |p: [f32; 3]| -> [f64; 3] {
        let (_x, y, z) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
        [g1 * y + g3 * z, g2 * z, 0.0]
    };

    let mut bc = BoundaryConditions::new();
    let interior = node_index(2, 2, 1, 1, 1);
    for (v, p) in mesh.vertices.iter().enumerate() {
        let v = u32::try_from(v).expect("fits");
        if v == interior {
            continue;
        }
        let u = field(*p);
        bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
    }

    let out = solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("well posed");

    let mu = E_MPA / (2.0 * (1.0 + NU));
    let (sxy, syz, szx) = (mu * g1, mu * g2, mu * g3);
    let vm = (3.0 * (sxy * sxy + syz * syz + szx * szx)).sqrt();

    let want = field(mesh.vertices[interior as usize]);
    for (axis, w) in Axis::ALL.into_iter().zip(want) {
        close(
            out.displacements[interior as usize][axis.index()],
            w,
            1e-12,
            &format!("interior node u_{axis:?} under simple shear"),
        );
    }

    for (t, s) in out.element_stress.iter().enumerate() {
        close(s.xx, 0.0, 1e-9, &format!("tet {t} σ_xx (shear has none)"));
        close(s.yy, 0.0, 1e-9, &format!("tet {t} σ_yy (shear has none)"));
        close(s.zz, 0.0, 1e-9, &format!("tet {t} σ_zz (shear has none)"));
        close(s.xy, sxy, 1e-9, &format!("tet {t} σ_xy = μγ₁"));
        close(s.yz, syz, 1e-9, &format!("tet {t} σ_yz = μγ₂"));
        close(s.zx, szx, 1e-9, &format!("tet {t} σ_zx = μγ₃"));
        close(s.hydrostatic(), 0.0, 1e-9, &format!("tet {t} mean stress"));
        close(s.von_mises(), vm, 1e-9, &format!("tet {t} von Mises"));
    }
}

// ---------------------------------------------------------------------------
// oracle 4 — rigid body motion carries no stress
// ---------------------------------------------------------------------------

/// A constant displacement is in the null space of the strain operator, so the
/// stress must be identically zero — not "small". Catches a `B` matrix with a
/// spurious constant term and a shape-function gradient that does not sum to
/// zero over the four nodes.
#[test]
fn rigid_translation_produces_zero_stress() {
    let mesh = kuhn_box(2, 1, 1, 3.0);
    let shift = [fx(0.7), fx(-1.3), fx(2.0)];

    let mut bc = BoundaryConditions::new();
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits") {
        bc.prescribe_all(v, shift);
    }

    let out = solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("fully prescribed");

    for (t, s) in out.element_stress.iter().enumerate() {
        for (name, c) in [
            ("xx", s.xx),
            ("yy", s.yy),
            ("zz", s.zz),
            ("xy", s.xy),
            ("yz", s.yz),
            ("zx", s.zx),
        ] {
            close(c, 0.0, 1e-9, &format!("tet {t} σ_{name} under translation"));
        }
    }
}

// ---------------------------------------------------------------------------
// oracle 4 — traction-loaded bar (force driven, free lateral surfaces)
// ---------------------------------------------------------------------------

/// The same uniaxial state, but reached through *loads* instead of prescribed
/// displacements, with the lateral surfaces free.
///
/// Oracle: a bar of length `L` and cross-section `A` pulled by total force `F`
/// carries `σ = F/A` and stretches by `δ = F·L/(A·E)`. The lateral faces are
/// traction free, so the bar also contracts by `−ν·δ·(w/L)` across its width;
/// both come out of the same closed form as oracle 1 with `σ = F/A`.
///
/// The end face is loaded by consistent nodal forces: for a constant traction
/// on a triangulated face, each triangle gives one third of `traction × area`
/// to each of its three nodes. Getting that wrong (lumping equally per node,
/// say) perturbs the answer, which is the point — this test covers the load
/// path that the prescribed-displacement patch tests never touch.
#[test]
fn traction_loaded_bar_matches_closed_form() {
    let (nx, ny, nz, h) = (4usize, 1usize, 1usize, 2.5_f32);
    let mesh = kuhn_box(nx, ny, nz, h);
    let length = f64::from(h) * nx as f64; // 10 mm
    let width = f64::from(h) * ny as f64; // 2.5 mm
    let height = f64::from(h) * nz as f64; // 2.5 mm
    let area = width * height;

    let total_force = 50.0_f64; // N
    let sigma = total_force / area;
    let delta = total_force * length / (area * E_MPA);

    let mut bc = BoundaryConditions::new();

    // x = 0 face: roller in x. Pin exactly three more components to remove the
    // remaining rigid body modes without restraining the lateral contraction:
    // node (0,0,0) also in y and z, node (0,ny,0) in z, node (0,0,nz) in y.
    for k in 0..=nz {
        for j in 0..=ny {
            bc.prescribe(node_index(nx, ny, 0, j, k), Axis::X, Fix128::ZERO);
        }
    }
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, ny, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, 0, nz), Axis::Y, Fix128::ZERO);

    // x = L face: the four corner nodes of a single 2.5 × 2.5 quad, split by
    // Kuhn into two triangles sharing the (0,0)-(1,1) diagonal. Consistent
    // nodal forces for a uniform traction: each triangle hands area/3 of the
    // traction to each of its nodes, so the two diagonal nodes get two thirds
    // of a triangle's share each and the other two get one third.
    let n00 = node_index(nx, ny, nx, 0, 0);
    let n10 = node_index(nx, ny, nx, ny, 0);
    let n01 = node_index(nx, ny, nx, 0, nz);
    let n11 = node_index(nx, ny, nx, ny, nz);
    let tri = total_force / 2.0; // force carried by each of the two triangles
    for (node, share) in [
        (n00, 2.0 / 3.0),
        (n11, 2.0 / 3.0),
        (n10, 1.0 / 3.0),
        (n01, 1.0 / 3.0),
    ] {
        bc.add_load(node, Axis::X, fx(tri * share));
    }

    let out = solve(&mesh, &pla(), &bc, &SolverConfig::default()).expect("well posed");

    // uniform axial stress everywhere
    for (t, s) in out.element_stress.iter().enumerate() {
        close(s.xx, sigma, 1e-6, &format!("tet {t} σ_xx"));
        close(s.yy, 0.0, 1e-6, &format!("tet {t} σ_yy"));
        close(s.zz, 0.0, 1e-6, &format!("tet {t} σ_zz"));
        close(s.xy, 0.0, 1e-6, &format!("tet {t} σ_xy"));
    }

    // tip extension δ = F L / (A E)
    for j in 0..=ny {
        for k in 0..=nz {
            let n = node_index(nx, ny, nx, j, k) as usize;
            close(
                out.displacements[n][0],
                delta,
                1e-9,
                &format!("tip node ({j},{k}) u_x"),
            );
        }
    }

    // lateral contraction at the tip: u_y = −ν·(σ/E)·y
    let n = node_index(nx, ny, nx, ny, 0) as usize;
    close(
        out.displacements[n][1],
        -NU * (sigma / E_MPA) * width,
        1e-9,
        "tip node u_y (Poisson contraction)",
    );
}

// ---------------------------------------------------------------------------
// oracle 5 — material and configuration contracts
// ---------------------------------------------------------------------------

/// `λ = Eν/((1+ν)(1−2ν))`, `μ = E/(2(1+ν))`, and the constructor refuses the
/// values that make the stiffness indefinite.
#[test]
fn lame_parameters_and_material_validation() {
    let m = pla();
    let (lambda, mu) = m.lame();
    let want_lambda = E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU));
    let want_mu = E_MPA / (2.0 * (1.0 + NU));
    close(lambda, want_lambda, 1e-6, "λ");
    close(mu, want_mu, 1e-6, "μ");

    // E = 2μ(1+ν) is an identity, so it holds for any admissible ν
    close(
        mu * fx(2.0) * (Fix128::ONE + fx(NU)),
        E_MPA,
        1e-6,
        "2μ(1+ν) = E",
    );

    assert!(ElasticMaterial::new(fx(0.0), fx(0.3)).is_err(), "E = 0");
    assert!(ElasticMaterial::new(fx(-1.0), fx(0.3)).is_err(), "E < 0");
    assert!(
        ElasticMaterial::new(fx(3500.0), fx(0.5)).is_err(),
        "ν = 0.5 is incompressible, λ diverges"
    );
    assert!(
        ElasticMaterial::new(fx(3500.0), fx(0.6)).is_err(),
        "ν > 0.5 makes the stiffness indefinite"
    );
    assert!(
        ElasticMaterial::new(fx(3500.0), fx(-1.0)).is_err(),
        "ν = -1 makes μ/λ degenerate"
    );

    assert!(SolverConfig::try_new(0, fx(1e-9)).is_err(), "zero budget");
    assert!(
        SolverConfig::try_new(100, Fix128::ZERO).is_err(),
        "zero tolerance is unreachable in fixed point"
    );
    assert!(
        SolverConfig::try_new(100, Fix128::ONE).is_err(),
        "tolerance of 1 accepts the zero vector"
    );
}

/// Von Mises is built from the closed form, not from the solver.
#[test]
fn von_mises_matches_closed_form() {
    // an arbitrary non-symmetric state, so a transposed or dropped shear term
    // shows up
    let s = StressTensor {
        xx: fx(120.0),
        yy: fx(-40.0),
        zz: fx(15.0),
        xy: fx(30.0),
        yz: fx(-12.0),
        zx: fx(7.0),
    };
    let (sxx, syy, szz) = (120.0_f64, -40.0, 15.0);
    let (sxy, syz, szx) = (30.0_f64, -12.0, 7.0);
    let want = (0.5 * ((sxx - syy).powi(2) + (syy - szz).powi(2) + (szz - sxx).powi(2))
        + 3.0 * (sxy * sxy + syz * syz + szx * szx))
        .sqrt();
    close(s.von_mises(), want, 1e-6, "von Mises");
    close(
        s.hydrostatic(),
        (sxx + syy + szz) / 3.0,
        1e-9,
        "mean stress",
    );
}

// ---------------------------------------------------------------------------
// oracle 6 — the unit contract, fixed by numbers rather than by prose
// ---------------------------------------------------------------------------

/// mm in, N in, MPa out.
///
/// The numbers are chosen so that every plausible alternative convention lands
/// somewhere else by a factor of at least 1000:
///
/// - `E = 1000 MPa`, a `2 mm × 2 mm` section, `L = 8 mm`, `F = 200 N`
/// - `σ = F/A = 50 MPa`. If `E` were read as GPa the stress would be unchanged,
///   but the extension would be 1000× smaller.
/// - `δ = σL/E = 50 · 8 / 1000 = 0.4 mm`. If lengths were metres, `δ` would be
///   0.4 m; if forces were kN, `σ` would be 50 kPa.
///
/// A convention change that kept all three consistent is not a bug, which is
/// why the assertion is on the absolute numbers and not on a ratio.
#[test]
fn unit_contract_is_mm_newton_mpa() {
    let (nx, ny, nz, h) = (4usize, 1usize, 1usize, 2.0_f32);
    let mesh = kuhn_box(nx, ny, nz, h);
    let material = ElasticMaterial::new(fx(1000.0), fx(0.30)).expect("valid");

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

    let total_force = 200.0_f64; // N over a 2 mm x 2 mm face
    let tri = total_force / 2.0;
    for (node, share) in [
        (node_index(nx, ny, nx, 0, 0), 2.0 / 3.0),
        (node_index(nx, ny, nx, ny, nz), 2.0 / 3.0),
        (node_index(nx, ny, nx, ny, 0), 1.0 / 3.0),
        (node_index(nx, ny, nx, 0, nz), 1.0 / 3.0),
    ] {
        bc.add_load(node, Axis::X, fx(tri * share));
    }

    let out = solve(&mesh, &material, &bc, &SolverConfig::default()).expect("well posed");

    close(out.max_von_mises_mpa(), 50.0, 1e-6, "σ = F/A in MPa");
    let tip = node_index(nx, ny, nx, 0, 0) as usize;
    close(out.displacements[tip][0], 0.4, 1e-7, "δ = σL/E in mm");
}

/// `from_filament` converts GPa to MPa and takes Poisson's ratio from the
/// category table; `with_poisson` overrides it without touching `E`.
#[test]
fn from_filament_converts_units_and_fills_poisson() {
    let pla_entry = MaterialProperties::pla();
    assert_eq!(pla_entry.category, MaterialCategory::Fdm);

    let m = ElasticMaterial::from_filament(&pla_entry).expect("PLA is a valid elastic material");
    // datasheet entry is 3.5 GPa
    close(
        m.youngs_modulus_mpa(),
        3500.0,
        1e-9,
        "E from filament (GPa -> MPa)",
    );
    close(
        m.poissons_ratio(),
        0.35,
        1e-9,
        "ν from the Fdm row of the category table",
    );
    assert_eq!(
        m.poissons_ratio(),
        ElasticMaterial::default_poissons_ratio(MaterialCategory::Fdm),
        "from_filament must read the same table the accessor exposes"
    );

    let measured = m.with_poisson(fx(0.41)).expect("valid ratio");
    close(measured.poissons_ratio(), 0.41, 1e-9, "overridden ν");
    assert_eq!(
        measured.youngs_modulus_mpa(),
        m.youngs_modulus_mpa(),
        "with_poisson must not disturb E"
    );
    assert!(
        measured.with_poisson(fx(0.5)).is_err(),
        "the override is validated like the constructor"
    );

    // the four categories are distinct enough to be worth a table at all
    let fdm = ElasticMaterial::default_poissons_ratio(MaterialCategory::Fdm);
    let metal = ElasticMaterial::default_poissons_ratio(MaterialCategory::SheetMetal);
    let powder = ElasticMaterial::default_poissons_ratio(MaterialCategory::Powder);
    assert!(
        metal < fdm && fdm < powder,
        "table ordering: steel < FDM < PA12"
    );
}

// ---------------------------------------------------------------------------
// oracle 7 — the errors are reported, not absorbed
// ---------------------------------------------------------------------------

/// A mesh with no constraints has three free translations, so the stiffness is
/// singular; the solver must say so rather than return whatever the iteration
/// happened to reach.
#[test]
fn unconstrained_mesh_is_rejected() {
    let mesh = kuhn_box(1, 1, 1, 4.0);
    let mut bc = BoundaryConditions::new();
    bc.add_load(0, Axis::X, fx(1.0));
    let err = solve(&mesh, &pla(), &bc, &SolverConfig::default()).unwrap_err();
    assert!(
        matches!(
            err,
            alice_physics::linear_elastic_fem::FemError::UnderConstrained
                | alice_physics::linear_elastic_fem::FemError::NotConverged { .. }
        ),
        "expected a rigid-body-mode diagnosis, got {err:?}"
    );
}

/// A boundary condition on a vertex the mesh does not have is a caller bug and
/// must not be silently dropped.
#[test]
fn out_of_range_vertex_is_rejected() {
    let mesh = kuhn_box(1, 1, 1, 4.0);
    let mut bc = BoundaryConditions::new();
    bc.fix(9_999);
    let err = solve(&mesh, &pla(), &bc, &SolverConfig::default()).unwrap_err();
    assert!(
        matches!(
            err,
            alice_physics::linear_elastic_fem::FemError::VertexOutOfRange { vertex: 9_999, .. }
        ),
        "expected VertexOutOfRange, got {err:?}"
    );
}

/// A tolerance below the arithmetic floor must be abandoned quickly and said
/// so, not ground at until the budget runs out.
///
/// `Fix128` carries 64 fractional bits, so an inner product whose terms fall
/// under 2⁻⁶⁴ rounds to zero and the residual cannot improve any further. A
/// tolerance placed under that floor is unreachable by construction. What the
/// solver must not do is spend the whole budget finding that out: measured
/// before this rule existed, an unpreconditioned solve of a 25,600-element
/// cantilever ran 500,000 iterations over 24.8 minutes to stop 1% above its
/// tolerance, and the report could not say whether more iterations would have
/// helped.
///
/// The two outcomes are deliberately different error variants because they call
/// for opposite responses: `NotConverged` means raise the budget, `Stagnated`
/// means the budget is irrelevant.
#[test]
fn unreachable_tolerance_stagnates_instead_of_burning_the_budget() {
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
    for node in [
        node_index(nx, ny, nx, 0, 0),
        node_index(nx, ny, nx, ny, nz),
        node_index(nx, ny, nx, ny, 0),
        node_index(nx, ny, nx, 0, nz),
    ] {
        bc.add_load(node, Axis::X, fx(12.5));
    }

    // 2^-60 is below anything the residual can reach in Q64.64 here.
    let budget = 200_000;
    let cfg = SolverConfig::try_new(budget, Fix128::from_raw(0, 1 << 4)).expect("valid config");
    let err = solve(&mesh, &pla(), &bc, &cfg).unwrap_err();

    match err {
        alice_physics::linear_elastic_fem::FemError::Stagnated {
            iterations,
            relative_residual,
            without_improvement,
        } => {
            eprintln!(
                "[stagnation] gave up after {iterations} of {budget} iterations, \
                 best relative residual {:.3e}, {without_improvement} without improvement",
                relative_residual.to_f64()
            );
            assert!(
                iterations < budget / 10,
                "the point of the rule is to stop early: it used {iterations} of {budget}"
            );
            assert!(
                relative_residual.to_f64() > 0.0,
                "a stagnation report must carry the residual it actually reached"
            );
        }
        other => panic!(
            "an unreachable tolerance must be reported as stagnation, not as {other:?}; \
             NotConverged here would tell the caller to raise a budget that cannot help"
        ),
    }
}

/// A preconditioner changes how the iteration gets there, never where it gets.
///
/// `K x = b` has one solution; `M` only reshapes the path. So both settings must
/// land on the same closed form, to the same tolerance — if they disagree, one
/// of them is solving a different system. What they are allowed to differ in is
/// the iteration count, which is reported.
#[test]
fn preconditioner_does_not_change_the_answer() {
    let (nx, ny, nz, h) = (4usize, 1usize, 1usize, 2.5_f32);
    let mesh = kuhn_box(nx, ny, nz, h);
    let width = f64::from(h) * ny as f64;
    let height = f64::from(h) * nz as f64;
    let area = width * height;
    let total_force = 50.0_f64;
    let sigma = total_force / area;

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
    let tri = total_force / 2.0;
    for (node, share) in [
        (node_index(nx, ny, nx, 0, 0), 2.0 / 3.0),
        (node_index(nx, ny, nx, ny, nz), 2.0 / 3.0),
        (node_index(nx, ny, nx, ny, 0), 1.0 / 3.0),
        (node_index(nx, ny, nx, 0, nz), 1.0 / 3.0),
    ] {
        bc.add_load(node, Axis::X, fx(tri * share));
    }

    let mut results = Vec::new();
    for mode in [Preconditioner::None, Preconditioner::JacobiScaled] {
        let cfg = SolverConfig::default().with_preconditioner(mode);
        let out = solve(&mesh, &pla(), &bc, &cfg).expect("well posed either way");
        eprintln!(
            "[precond] {mode:?}: {} iterations, rel resid {:.3e}, max von Mises {:.6} MPa",
            out.iterations,
            out.relative_residual.to_f64(),
            out.max_von_mises_mpa().to_f64()
        );
        close(
            out.max_von_mises_mpa(),
            sigma,
            1e-6,
            &format!("{mode:?}: sigma = F/A"),
        );
        results.push(out);
    }

    // same problem, same answer
    for (t, (a, b)) in results[0]
        .element_stress
        .iter()
        .zip(results[1].element_stress.iter())
        .enumerate()
    {
        for (name, x, y) in [("xx", a.xx, b.xx), ("yy", a.yy, b.yy), ("zz", a.zz, b.zz)] {
            assert!(
                (x.to_f64() - y.to_f64()).abs() < 1e-6,
                "tet {t} sigma_{name} differs between preconditioners: {:.9} vs {:.9}",
                x.to_f64(),
                y.to_f64()
            );
        }
    }

    let stats = stiffness_diagonal_stats(&mesh, &pla(), &bc).expect("well posed");
    eprintln!(
        "[diag] free dofs {} min {:.4e} mean {:.4e} max {:.4e} (max/min {:.2})",
        stats.free_dofs,
        stats.min.to_f64(),
        stats.mean.to_f64(),
        stats.max.to_f64(),
        stats.max.to_f64() / stats.min.to_f64()
    );
    assert!(
        stats.min > Fix128::ZERO && stats.min <= stats.mean && stats.mean <= stats.max,
        "diagonal statistics must be ordered and positive"
    );
}

/// The stagnation rule is configurable and validated like the rest.
#[test]
fn stagnation_settings_are_validated() {
    let base = SolverConfig::default();
    assert!(base.stagnation_min_window() > 0);
    assert!(base.stagnation_window_fraction() > Fix128::ZERO);
    assert!(base.stagnation_min_improvement() > Fix128::ZERO);
    assert!(base.stagnation_min_improvement() < Fix128::ONE);

    let tuned = base.with_stagnation(50, fx(0.01)).expect("valid");
    assert_eq!(tuned.stagnation_min_window(), 50);
    // the rest of the configuration survives
    assert_eq!(tuned.max_iterations(), base.max_iterations());
    assert_eq!(tuned.relative_tolerance(), base.relative_tolerance());

    assert!(base.with_stagnation(0, fx(0.01)).is_err(), "zero window");
    assert!(
        base.with_stagnation(50, Fix128::ZERO).is_err(),
        "zero improvement never counts as progress, so nothing would ever stagnate"
    );
    assert!(
        base.with_stagnation(50, Fix128::ONE).is_err(),
        "an improvement factor of 1 demands the residual reach zero every window"
    );

    let scaled = base.with_stagnation_fraction(fx(0.25)).expect("valid");
    assert_eq!(scaled.stagnation_window_fraction(), fx(0.25));
    assert_eq!(scaled.stagnation_min_window(), base.stagnation_min_window());
    assert!(
        base.with_stagnation_fraction(Fix128::ZERO).is_err(),
        "a zero fraction degenerates to the fixed window the scaling replaces"
    );
    assert!(
        base.with_stagnation_fraction(fx(-1.0)).is_err(),
        "a negative fraction has no meaning"
    );
    assert!(
        base.with_stagnation_fraction(Fix128::ONE).is_err(),
        "at a fraction of 1 the condition `iterations - last >= iterations` needs \
         `last <= 0`, so the rule can never fire and a hopeless solve burns the budget"
    );
}

/// An empty mesh is an error, not an empty solution.
#[test]
fn empty_mesh_is_rejected() {
    let mesh = SdfTetMesh::default();
    let bc = BoundaryConditions::new();
    let err = solve(&mesh, &pla(), &bc, &SolverConfig::default()).unwrap_err();
    assert_eq!(err, alice_physics::linear_elastic_fem::FemError::EmptyMesh);
}
