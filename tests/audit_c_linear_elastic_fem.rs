//! Audit oracles for `linear_elastic_fem`: the increment schedule and the
//! first-step frame rule of `solve_corotational`, and the zero-curvature
//! branch of the conjugate gradient.
//!
//! Every expected value is a closed form: a rigid rotation carries no stress
//! and moves every node by `(R - I) x`, a laterally confined bar under an end
//! displacement is in uniform uniaxial strain, and a load along a rigid body
//! mode the constraints left free has no equilibrium at all. None of them is
//! read back from the implementation under test.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::cast_precision_loss)]

use alice_physics::linear_elastic_fem::{
    solve, solve_corotational, Axis, BoundaryConditions, CorotationalConfig, ElasticMaterial,
    FemError, SolverConfig,
};
use alice_physics::math::{Fix128, PolarError};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// `2^exponent` for `-63 <= exponent <= 62`, built from raw words.
fn pow2(exponent: i32) -> Fix128 {
    if exponent >= 0 {
        Fix128::from_int(1_i64 << exponent)
    } else {
        Fix128::from_raw(0, 1_u64 << (64 + exponent))
    }
}

/// `nx x ny x nz` unit cubes, each split into the six Kuhn tetrahedra.
/// Every vertex has integer coordinates, so every shape-function gradient is
/// a small integer and is exact in `Fix128`.
fn kuhn_box(nx: usize, ny: usize, nz: usize) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    let idx = |i: usize, j: usize, k: usize| (i + (nx + 1) * (j + (ny + 1) * k)) as u32;
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices.push([i as f32, j as f32, k as f32]);
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
                    let mut corners = [idx(i, j, k); 4];
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = idx(i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

fn material() -> ElasticMaterial {
    ElasticMaterial::new(Fix128::from_int(2000), Fix128::from_ratio(1, 4)).unwrap()
}

fn corotational(increments: u32) -> CorotationalConfig {
    CorotationalConfig::try_new(SolverConfig::default(), 32, pow2(-30), increments, 32).unwrap()
}

/// Every node of a unit cube prescribed to the rigid rotation by 180 degrees
/// about the z axis, `u = (R - I) x = (-2x, -2y, 0)`. All values are exact.
fn half_turn_boundary(mesh: &SdfTetMesh) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        let x = Fix128::from_int(p[0] as i64);
        let y = Fix128::from_int(p[1] as i64);
        bc.prescribe_all(v as u32, [-(x + x), -(y + y), Fix128::ZERO]);
    }
    bc
}

/// Rigid motion: every displacement is the prescribed one bit for bit (every
/// node is prescribed) and every stress component vanishes to the arithmetic
/// floor. The bound `2^-40` MPa is ten decades below a stress of 1 MPa and
/// well above the `1e-15` floor a rigid rotation reaches.
fn assert_rigid_half_turn(increments: u32) {
    let mesh = kuhn_box(1, 1, 1);
    let bc = half_turn_boundary(&mesh);
    let out = solve_corotational(&mesh, &material(), &bc, &corotational(increments))
        .unwrap_or_else(|e| panic!("increments = {increments}: {e:?}"));
    assert_eq!(out.increments, increments);
    for (v, p) in mesh.vertices.iter().enumerate() {
        let x = Fix128::from_int(p[0] as i64);
        let y = Fix128::from_int(p[1] as i64);
        assert_eq!(
            out.field.displacements[v],
            [-(x + x), -(y + y), Fix128::ZERO],
            "increments = {increments}, vertex {v}"
        );
    }
    let floor = pow2(-40);
    for (t, s) in out.field.element_stress.iter().enumerate() {
        assert!(
            s.von_mises() <= floor && s.hydrostatic().abs() <= floor,
            "increments = {increments}, tet {t}: von Mises {} MPa, hydrostatic {} MPa",
            s.von_mises().to_f64(),
            s.hydrostatic().to_f64()
        );
    }
}

/// The documented schedule applies the prescribed field in equal fractions
/// `i / n`. With linear interpolation `F(t) = I + t (R - I)` of a half turn,
/// `F` is singular only at `t = 1/2`. An odd count never visits that state,
/// so a rigid half turn applied in 3 or 5 increments has to come back as the
/// rigid motion. A schedule of `i / (n + 1)` visits `2/4` on `n = 3` and is
/// refused there.
#[test]
fn rigid_half_turn_in_an_odd_number_of_increments_is_stress_free() {
    for increments in [1_u32, 3, 5] {
        assert_rigid_half_turn(increments);
    }
}

/// The same rigid half turn in an even number of increments. The doc says the
/// increments are a path to the answer and not part of it, and a rigid motion
/// has a stress-free answer whatever the path. The current source refuses it.
#[test]
#[ignore = "known defect: AUD-A-S34-030: `solve_corotational` applies a rigid 180 degree rotation in an even number of increments through `F = I + (1/2)(R - I) = diag(0, 0, 1)` and returns `RotationFailed { tet: 0, cause: Inverted }` for increments 2 and 4, while increments 1, 3 and 5 return the rigid motion with zero stress"]
fn rigid_half_turn_in_an_even_number_of_increments_is_stress_free() {
    for increments in [2_u32, 4] {
        assert_rigid_half_turn(increments);
    }
}

/// Probe for the defect above: pins what the current source returns, so a
/// change in the failure mode is visible.
// PIN: AUD-A-S34-030
#[test]
fn rigid_half_turn_in_two_increments_is_currently_refused_as_inverted() {
    let mesh = kuhn_box(1, 1, 1);
    let bc = half_turn_boundary(&mesh);
    let got = solve_corotational(&mesh, &material(), &bc, &corotational(2));
    assert!(
        matches!(
            got,
            Err(FemError::RotationFailed {
                cause: PolarError::Inverted,
                ..
            })
        ),
        "{got:?}"
    );
}

/// A bar of two unit cubes along x, every node confined laterally (`u_y =
/// u_z = 0`), the `x = 0` face held and the `x = 2` face pulled to `u_x = d`.
/// The interior face is free in x. Closed form: uniform uniaxial strain
/// `eps = d / 2`, so the interior face moves by exactly `d / 2`,
/// `sigma_xx = (lambda + 2 mu) eps` and `sigma_yy = sigma_zz = lambda eps`.
///
/// The state the solve opens at moves the end face and leaves the interior
/// at zero. Every element of that state is a pure stretch along x, so its
/// rotation factor is the identity, the same frame the iteration starts from.
/// Accepting frames as settled before any solve has been done therefore
/// returns that unsolved state.
#[test]
fn confined_bar_reaches_the_uniaxial_strain_solution_in_one_increment() {
    let mesh = kuhn_box(2, 1, 1);
    let d = pow2(-2);
    let mut bc = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        let v = v as u32;
        bc.prescribe(v, Axis::Y, Fix128::ZERO);
        bc.prescribe(v, Axis::Z, Fix128::ZERO);
        if p[0] == 0.0 {
            bc.prescribe(v, Axis::X, Fix128::ZERO);
        } else if p[0] == 2.0 {
            bc.prescribe(v, Axis::X, d);
        }
    }
    let mat = material();
    let (lambda, mu) = mat.lame();
    let eps = d.half();
    for increments in [1_u32, 2, 3] {
        let out = solve_corotational(&mesh, &mat, &bc, &corotational(increments))
            .unwrap_or_else(|e| panic!("increments = {increments}: {e:?}"));
        let tol = pow2(-50);
        for (v, p) in mesh.vertices.iter().enumerate() {
            if p[0] == 1.0 {
                let ux = out.field.displacements[v][0];
                assert!(
                    (ux - eps).abs() <= tol,
                    "increments = {increments}, vertex {v}: u_x = {}, want {}",
                    ux.to_f64(),
                    eps.to_f64()
                );
            }
        }
        let sxx = (lambda + mu + mu) * eps;
        let slat = lambda * eps;
        let stol = sxx * pow2(-40);
        for (t, s) in out.field.element_stress.iter().enumerate() {
            assert!(
                (s.xx - sxx).abs() <= stol
                    && (s.yy - slat).abs() <= stol
                    && (s.zz - slat).abs() <= stol
                    && s.xy.abs() <= stol
                    && s.yz.abs() <= stol
                    && s.zx.abs() <= stol,
                "increments = {increments}, tet {t}: {s:?}, want xx {} lateral {}",
                sxx.to_f64(),
                slat.to_f64()
            );
        }
        assert!(
            out.newton_iterations >= increments,
            "increments = {increments}: every increment needs at least one solve, got {}",
            out.newton_iterations
        );
    }
}

/// One corner tetrahedron with `u_y = u_z = 0` on every node (eight
/// constraints) and an equal x load on every node. The x translation is left
/// free, and the load is that translation, so no equilibrium exists: the
/// total force is `4 F` with nothing to react it. The first search direction
/// is the load itself (no preconditioner) and the stiffness maps it to zero
/// exactly, because the gradients of this element are integers and sum to
/// zero. The doc of `FemError::UnderConstrained` names exactly this case,
/// `p^T K p <= 0` on a rigid body mode.
#[test]
fn load_along_a_free_rigid_translation_is_under_constrained() {
    let mesh = SdfTetMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        tets: vec![Tetrahedron {
            vertices: [0, 1, 2, 3],
        }],
    };
    let mut bc = BoundaryConditions::new();
    for v in 0..4_u32 {
        bc.prescribe(v, Axis::Y, Fix128::ZERO);
        bc.prescribe(v, Axis::Z, Fix128::ZERO);
        bc.add_load(v, Axis::X, Fix128::ONE);
    }
    let config = SolverConfig::default();
    let got = solve(&mesh, &material(), &bc, &config);
    assert!(
        matches!(got, Err(FemError::UnderConstrained)),
        "small strain: {got:?}"
    );
    let got = solve_corotational(&mesh, &material(), &bc, &corotational(1));
    assert!(
        matches!(got, Err(FemError::UnderConstrained)),
        "corotational: {got:?}"
    );
}
