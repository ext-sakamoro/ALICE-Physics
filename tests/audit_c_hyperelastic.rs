//! Audit oracles for `hyperelastic`: the consistent tangent that
//! `solve_corotational` builds from the law, measured by the Newton
//! iteration count it produces against an independent Newton iteration whose
//! Jacobian is the central finite difference of `cauchy_stress`.
//!
//! The scene is one corner tetrahedron with three nodes held and a point
//! load on the fourth, so the internal force has the closed form
//! `f = V0 P e_z` with `V0 = 1/6` and `P = J sigma F^-T`, and the whole
//! Newton iteration is a 3 x 3 problem the test solves on its own. A tangent
//! that is the exact derivative of the stress converges quadratically and
//! takes the same number of steps as the finite-difference Newton iteration;
//! a tangent that is off by a constant converges linearly and takes more.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::many_single_char_names)]

use alice_physics::hyperelastic::{cauchy_stress, volumetric_modulus, HyperelasticModel};
use alice_physics::linear_elastic_fem::{
    solve_corotational, Axis, BoundaryConditions, CorotationalConfig, ElasticMaterial, SolverConfig,
};
use alice_physics::math::{Fix128, Mat3Fix, Vec3Fix};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn corner_tet() -> SdfTetMesh {
    SdfTetMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        tets: vec![Tetrahedron {
            vertices: [0, 1, 2, 3],
        }],
    }
}

/// `2^exponent` for `-63 <= exponent < 0`.
fn pow2_neg(exponent: i32) -> Fix128 {
    Fix128::from_raw(0, 1_u64 << (64 + exponent))
}

/// Nodal force on node 3 of the corner tetrahedron for a displacement `u` of
/// that node: `F = I + u e_z^T`, `f = (1/6) J sigma F^-T e_z`.
fn internal_force(model: &HyperelasticModel, kappa: Fix128, u: [f64; 3]) -> [f64; 3] {
    let one = Fix128::ONE;
    let f = Mat3Fix::from_cols(
        Vec3Fix::new(one, Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, one, Fix128::ZERO),
        Vec3Fix::new(
            Fix128::from_f64(u[0]),
            Fix128::from_f64(u[1]),
            one + Fix128::from_f64(u[2]),
        ),
    );
    let sigma = cauchy_stress(model, kappa, f).expect("det F > 0 on every visited state");
    let j = f.determinant();
    let p = sigma
        .mul_mat(f.inverse().expect("invertible").transpose())
        .scale(j);
    let sixth = |x: Fix128| (x / Fix128::from_int(6)).to_f64();
    [sixth(p.col2.x), sixth(p.col2.y), sixth(p.col2.z)]
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    let mut x = [0.0; 3];
    for (c, slot) in x.iter_mut().enumerate() {
        let mut m = a;
        for r in 0..3 {
            m[r][c] = b[r];
        }
        *slot = det(m) / d;
    }
    x
}

/// Newton iteration count of an independent replica of the documented
/// schedule: the first step is the small-strain solve from `u = 0` (the
/// surrogate step `with_consistent_tangent` documents for the first step of
/// an increment), every later state is tested against
/// `max|r| <= tol * max|f_ext|` and, if not met, advanced by a Newton step
/// whose Jacobian is the central difference of `internal_force`.
fn replica_newton_count(
    model: &HyperelasticModel,
    lambda: f64,
    mu: f64,
    kappa: Fix128,
    load: [f64; 3],
    tol: f64,
) -> (u32, [f64; 3]) {
    let mut u = [
        6.0 * load[0] / mu,
        6.0 * load[1] / mu,
        6.0 * load[2] / (lambda + 2.0 * mu),
    ];
    let target = tol * load.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut count = 1_u32;
    loop {
        let fi = internal_force(model, kappa, u);
        let r = [load[0] - fi[0], load[1] - fi[1], load[2] - fi[2]];
        let reach = r.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if reach <= target {
            return (count, u);
        }
        assert!(count < 60, "replica did not converge, residual {reach:e}");
        let h = 1e-6;
        let mut jac = [[0.0; 3]; 3];
        for c in 0..3 {
            let mut up = u;
            let mut um = u;
            up[c] += h;
            um[c] -= h;
            let fp = internal_force(model, kappa, up);
            let fm = internal_force(model, kappa, um);
            for row in 0..3 {
                jac[row][c] = (fp[row] - fm[row]) / (2.0 * h);
            }
        }
        let du = solve3(jac, r);
        for k in 0..3 {
            u[k] += du[k];
        }
        count += 1;
    }
}

struct Outcome {
    crate_count: u32,
    replica_count: u32,
    crate_u: [f64; 3],
    replica_u: [f64; 3],
}

fn run(model: HyperelasticModel, e: f64, nu: f64, load: [f64; 3]) -> Outcome {
    let material = ElasticMaterial::new(Fix128::from_f64(e), Fix128::from_f64(nu)).unwrap();
    let (lambda, mu) = material.lame();
    let kappa = volumetric_modulus(&model, lambda).expect("kappa >= 0");
    let tol = pow2_neg(-30);
    let config = CorotationalConfig::try_new(SolverConfig::default(), 200, tol, 1, 32)
        .unwrap()
        .with_hyperelastic(model)
        .with_consistent_tangent();
    let mut bc = BoundaryConditions::new();
    for v in 0..3_u32 {
        bc.fix(v);
    }
    bc.add_load(3, Axis::X, Fix128::from_f64(load[0]));
    bc.add_load(3, Axis::Y, Fix128::from_f64(load[1]));
    bc.add_load(3, Axis::Z, Fix128::from_f64(load[2]));
    let out = solve_corotational(&corner_tet(), &material, &bc, &config).expect("converges");
    let (replica_count, replica_u) = replica_newton_count(
        &model,
        lambda.to_f64(),
        mu.to_f64(),
        kappa,
        load,
        tol.to_f64(),
    );
    let d = out.field.displacements[3];
    Outcome {
        crate_count: out.newton_iterations,
        replica_count,
        crate_u: [d[0].to_f64(), d[1].to_f64(), d[2].to_f64()],
        replica_u,
    }
}

fn check(what: &str, o: &Outcome) {
    eprintln!(
        "{what}: crate {} steps, replica {} steps",
        o.crate_count, o.replica_count
    );
    for k in 0..3 {
        let scale = o.replica_u[k].abs().max(1e-3);
        assert!(
            (o.crate_u[k] - o.replica_u[k]).abs() <= 1e-6 * scale,
            "{what}: u[{k}] crate {} replica {}",
            o.crate_u[k],
            o.replica_u[k]
        );
    }
    assert!(
        o.crate_count <= o.replica_count,
        "{what}: the consistent tangent took {} Newton steps where an exact-Jacobian Newton \
         iteration takes {}",
        o.crate_count,
        o.replica_count
    );
}

/// Neo-Hookean, tension plus shear on the free node. The volumetric part of
/// the tangent carries `p_ref`, so this scene measures the `p_ref` constant
/// the tangent is built from.
#[test]
fn neo_hookean_consistent_tangent_converges_like_exact_newton() {
    let model = HyperelasticModel::NeoHookean {
        mu_mpa: Fix128::from_int(4),
    };
    let o = run(model, 10.0, 0.25, [0.6, 0.3, 1.5]);
    check("neo-hookean", &o);
}

/// Mooney-Rivlin: `W2 != 0`, so `p_ref = 2 (W1 + 2 W2)` has both terms.
#[test]
fn mooney_rivlin_consistent_tangent_converges_like_exact_newton() {
    let model = HyperelasticModel::MooneyRivlin {
        c1_mpa: Fix128::from_int(1),
        c2_mpa: Fix128::from_ratio(1, 2),
    };
    let o = run(model, 10.0, 0.3, [0.4, -0.2, 1.2]);
    check("mooney-rivlin", &o);
}

/// Yeoh with a large `C3` and a stretch that puts `I1 - 3` near one, so the
/// `6 C3 (I1 - 3)` part of `W11` is a large share of the tangent.
#[test]
fn yeoh_consistent_tangent_converges_like_exact_newton() {
    let model = HyperelasticModel::Yeoh {
        c1_mpa: Fix128::from_int(1),
        c2_mpa: Fix128::from_ratio(1, 4),
        c3_mpa: Fix128::from_int(1),
    };
    let o = run(model, 10.0, 0.3, [0.5, 0.2, 2.5]);
    check("yeoh", &o);
}
