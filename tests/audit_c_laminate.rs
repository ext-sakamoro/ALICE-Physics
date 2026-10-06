//! Audit oracles for laminate: classical lamination theory on symmetric
//! angle-ply and quasi-isotropic stacks of orthotropic plies (`E_L != E_T`),
//! where `B` must vanish and `A` / `D` follow from the through-thickness
//! integration of `Q-bar` (AUD-C-S1W5-007).
//!
//! The reference `Q-bar` is computed here by rotating the plane-stress
//! stiffness as a fourth-order tensor, `C'_ijkl = R_ip R_jq R_kr R_ls C_pqrs`
//! with `R = [[c, -s], [s, c]]` (the fibre axis at `+theta` from x), which is
//! independent of the closed-form `c^4 / s^4` polynomials the crate uses. The
//! integration `A = sum Q-bar t`, `D = sum Q-bar (z_top^3 - z_bot^3) / 3` is
//! done in `f64` from the mid-plane.

use alice_physics::anisotropic::OrthotropicElasticity;
use alice_physics::laminate::{compute_abd, is_symmetric_stack, AbdMatrix, Ply, Sym3};
use alice_physics::math::Fix128;

const E_L: f64 = 140_000.0;
const E_T: f64 = 10_000.0;
const NU_LT: f64 = 0.25;
const G_LT: f64 = 5_000.0;

fn material() -> OrthotropicElasticity {
    OrthotropicElasticity {
        e_l_mpa: Fix128::from_int(140_000),
        e_t_mpa: Fix128::from_int(10_000),
        e_z_mpa: Fix128::from_int(10_000),
        nu_lt: Fix128::from_ratio(1, 4),
        nu_lz: Fix128::from_ratio(1, 4),
        nu_tz: Fix128::from_ratio(2, 5),
        g_lt_mpa: Fix128::from_int(5_000),
        g_lz_mpa: Fix128::from_int(5_000),
        g_tz_mpa: Fix128::from_int(3_500),
    }
}

/// Reduced plane-stress stiffness in material axes (Jones eq. 2.61).
fn q_material() -> [[[[f64; 2]; 2]; 2]; 2] {
    let nu_tl = NU_LT * E_T / E_L;
    let den = 1.0 - NU_LT * nu_tl;
    let (q11, q22, q12, q66) = (E_L / den, E_T / den, NU_LT * E_T / den, G_LT);
    let mut c = [[[[0.0; 2]; 2]; 2]; 2];
    c[0][0][0][0] = q11;
    c[1][1][1][1] = q22;
    c[0][0][1][1] = q12;
    c[1][1][0][0] = q12;
    for (i, j, k, l) in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)] {
        c[i][j][k][l] = q66;
    }
    c
}

/// `(Q11, Q12, Q22, Q16, Q26, Q66)` in laminate axes by tensor rotation.
fn q_bar_tensor(theta: f64) -> [f64; 6] {
    let (s, c) = theta.sin_cos();
    let r = [[c, -s], [s, c]];
    let cm = q_material();
    let mut out = [[[[0.0; 2]; 2]; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    let mut acc = 0.0;
                    for p in 0..2 {
                        for q in 0..2 {
                            for rr in 0..2 {
                                for ss in 0..2 {
                                    acc +=
                                        r[i][p] * r[j][q] * r[k][rr] * r[l][ss] * cm[p][q][rr][ss];
                                }
                            }
                        }
                    }
                    out[i][j][k][l] = acc;
                }
            }
        }
    }
    [
        out[0][0][0][0],
        out[0][0][1][1],
        out[1][1][1][1],
        out[0][0][0][1],
        out[1][1][0][1],
        out[0][1][0][1],
    ]
}

struct Reference {
    a: [f64; 6],
    b: [f64; 6],
    d: [f64; 6],
}

fn reference(stack: &[(f64, f64)]) -> Reference {
    let h: f64 = stack.iter().map(|(t, _)| t).sum();
    let mut z = -h / 2.0;
    let mut r = Reference {
        a: [0.0; 6],
        b: [0.0; 6],
        d: [0.0; 6],
    };
    for (t, theta) in stack {
        let top = z + t;
        let qb = q_bar_tensor(*theta);
        let cube = |x: f64| x * x * x;
        for (n, q) in qb.iter().enumerate() {
            r.a[n] += q * (top - z);
            r.b[n] += q * (top * top - z * z) / 2.0;
            r.d[n] += q * (cube(top) - cube(z)) / 3.0;
        }
        z = top;
    }
    r
}

fn entries(m: &Sym3) -> [f64; 6] {
    [
        m.m11.to_f64(),
        m.m12.to_f64(),
        m.m22.to_f64(),
        m.m13.to_f64(),
        m.m23.to_f64(),
        m.m33.to_f64(),
    ]
}

const NAMES: [&str; 6] = ["11", "12", "22", "16", "26", "66"];

/// Builds the plies with exactly the angle the reference uses: each angle is
/// rounded to `Fix128` first and the reference reads it back.
fn build(stack_deg: &[(i64, f64)]) -> (Vec<Ply>, Vec<(f64, f64)>) {
    let mut plies = Vec::new();
    let mut reference = Vec::new();
    for (den, deg) in stack_deg {
        let t = Fix128::from_ratio(1, *den);
        let magnitude = Fix128::from_f64(deg.abs().to_radians());
        let angle = if *deg < 0.0 { -magnitude } else { magnitude };
        plies.push(Ply {
            thickness_mm: t,
            orientation_rad: angle,
            material: material(),
        });
        reference.push((t.to_f64(), angle.to_f64()));
    }
    (plies, reference)
}

fn check(abd: &AbdMatrix, r: &Reference, what: &str) {
    let scale_a = r.a[0].abs();
    let scale_d = r.d[0].abs();
    let a = entries(&abd.a);
    let d = entries(&abd.d);
    for n in 0..6 {
        assert!(
            (a[n] - r.a[n]).abs() <= 1e-9 * scale_a,
            "{what}: A{} = {}, CLT reference {}",
            NAMES[n],
            a[n],
            r.a[n]
        );
        assert!(
            (d[n] - r.d[n]).abs() <= 1e-9 * scale_d,
            "{what}: D{} = {}, CLT reference {}",
            NAMES[n],
            d[n],
            r.d[n]
        );
    }
}

/// `B` within one unit of the last place (`2^-64`) per ply. The documented
/// claim is `B = 0` exactly, pinned separately below; this bound is what the
/// value checks rely on.
fn assert_b_vanishes(abd: &AbdMatrix, plies: usize, what: &str) {
    let tol = Fix128::from_raw(0, plies as u64);
    assert!(
        abd.is_symmetric(tol),
        "{what}: B = {:?}, a mirrored stack has B = 0",
        abd.b
    );
}

#[test]
fn reference_q_bar_reproduces_the_on_axis_and_cross_ply_limits() {
    // sanity of the independent reference itself: 0 deg is Q, 90 deg swaps
    // Q11 and Q22, and both have no shear coupling.
    let q0 = q_bar_tensor(0.0);
    let q90 = q_bar_tensor(core::f64::consts::FRAC_PI_2);
    let den = 1.0 - NU_LT * NU_LT * E_T / E_L;
    assert!((q0[0] - E_L / den).abs() < 1e-9);
    assert!((q90[2] - E_L / den).abs() < 1e-6);
    assert!((q90[0] - E_T / den).abs() < 1e-6);
    assert!(q0[3].abs() < 1e-9 && q0[4].abs() < 1e-9);
    assert!(q90[3].abs() < 1e-6 && q90[4].abs() < 1e-6);
}

#[test]
fn symmetric_angle_ply_has_zero_b_and_clt_a_and_d() {
    // [+30 / -30]s, four plies of 1/8 mm
    let (plies, stack) = build(&[(8, 30.0), (8, -30.0), (8, -30.0), (8, 30.0)]);
    assert!(is_symmetric_stack(&plies));
    let abd = compute_abd(&plies);
    let r = reference(&stack);
    assert_b_vanishes(&abd, plies.len(), "[+30/-30]s");
    check(&abd, &r, "[+30/-30]s");
    // balanced: A16 = A26 = 0 in closed form; bending-twist D16 does not vanish
    assert!(r.a[3].abs() < 1e-9 * r.a[0] && r.a[4].abs() < 1e-9 * r.a[0]);
    assert!(abd.a.m13.to_f64().abs() < 1e-9 * r.a[0]);
    assert!(abd.a.m23.to_f64().abs() < 1e-9 * r.a[0]);
    assert!(r.d[3].abs() > 1e-3 * r.d[0]);
}

#[test]
fn symmetric_quasi_isotropic_stack_has_zero_b_and_clt_a_and_d() {
    // [0 / +45 / -45 / 90]s, eight plies of 1/16 mm: A is in-plane isotropic
    // (A11 = A22, A16 = A26 = 0, A66 = (A11 - A12) / 2), D is not.
    let half = [(16, 0.0), (16, 45.0), (16, -45.0), (16, 90.0)];
    let mut full: Vec<(i64, f64)> = half.to_vec();
    full.extend(half.iter().rev());
    let (plies, stack) = build(&full);
    assert!(is_symmetric_stack(&plies));
    let abd = compute_abd(&plies);
    let r = reference(&stack);
    assert_b_vanishes(&abd, plies.len(), "[0/45/-45/90]s");
    check(&abd, &r, "[0/45/-45/90]s");
    let a = entries(&abd.a);
    let tol = 1e-9 * a[0];
    assert!((a[0] - a[2]).abs() < tol, "A11 = {}, A22 = {}", a[0], a[2]);
    assert!(
        (a[5] - (a[0] - a[1]) / 2.0).abs() < tol,
        "A66 = (A11 - A12) / 2"
    );
}

#[test]
fn unsymmetric_angle_ply_has_the_clt_coupling_b16() {
    // [+30 / -30] (two plies, not mirrored): B16 = B26 != 0 from the
    // reference; the symmetric-layup checks above would be vacuous if B were
    // zero for every stack of these plies.
    let (plies, stack) = build(&[(8, 30.0), (8, -30.0)]);
    assert!(!is_symmetric_stack(&plies));
    let abd = compute_abd(&plies);
    let r = reference(&stack);
    assert!(r.b[3].abs() > 1.0, "reference B16 = {}", r.b[3]);
    let b = entries(&abd.b);
    for n in 0..6 {
        assert!(
            (b[n] - r.b[n]).abs() <= 1e-9 * r.b[3].abs(),
            "B{} = {}, CLT reference {}",
            NAMES[n],
            b[n],
            r.b[n]
        );
    }
    assert!(!abd.is_symmetric(Fix128::ONE));
}

#[test]
// AUD-A-S34-050
fn symmetric_stack_b_is_exactly_zero_as_documented() {
    let half = [(16, 0.0), (16, 45.0), (16, -45.0), (16, 90.0)];
    let mut quasi: Vec<(i64, f64)> = half.to_vec();
    quasi.extend(half.iter().rev());
    for (what, stack) in [
        (
            "[+30/-30]s",
            vec![(8, 30.0), (8, -30.0), (8, -30.0), (8, 30.0)],
        ),
        ("[0/45/-45/90]s", quasi),
    ] {
        let (plies, _) = build(&stack);
        assert!(is_symmetric_stack(&plies));
        let abd = compute_abd(&plies);
        assert!(
            abd.is_symmetric(Fix128::ZERO),
            "{what}: B = {:?}, documented B = 0 exactly",
            abd.b
        );
    }
}
