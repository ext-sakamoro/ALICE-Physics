//! Audit S1-5 oracles for `alice_physics::laminate` (Jones, Mechanics of Composite
//! Materials ch. 2 and 4: closed-form Q-bar rotation and ABD for simple stacks).
#![allow(clippy::disallowed_methods)]

use alice_physics::anisotropic::OrthotropicElasticity;
use alice_physics::laminate::*;
use alice_physics::math::Fix128;

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn mat(el: f64, et: f64, nu: f64, g: f64) -> OrthotropicElasticity {
    OrthotropicElasticity {
        e_l_mpa: f(el),
        e_t_mpa: f(et),
        e_z_mpa: f(et),
        nu_lt: f(nu),
        nu_lz: f(nu),
        nu_tz: f(nu),
        g_lt_mpa: f(g),
        g_lz_mpa: f(g),
        g_tz_mpa: f(g),
    }
}

/// Graphite/epoxy-like ply (Jones example scale).
fn m() -> OrthotropicElasticity {
    mat(140_000.0, 10_000.0, 0.3, 5_000.0)
}

/// Independent Q from engineering constants (plane stress).
fn q_ref(el: f64, et: f64, nu: f64, g: f64) -> (f64, f64, f64, f64) {
    let nu_tl = nu * et / el;
    let d = 1.0 - nu * nu_tl;
    (el / d, nu * et / d, et / d, g)
}

fn ply(t: f64, th: f64, mat: OrthotropicElasticity) -> Ply {
    Ply {
        thickness_mm: f(t),
        orientation_rad: f(th),
        material: mat,
    }
}

fn near(got: Fix128, want: f64, tol_rel: f64, tol_abs: f64, what: &str) {
    let g = got.to_f64();
    assert!(
        (g - want).abs() <= tol_abs + tol_rel * want.abs(),
        "{what}: got {g} want {want}"
    );
}

#[test]
fn q_matrix_matches_plane_stress_formulas() {
    let (q11, q12, q22, q66) = ply(0.125, 0.0, m()).q_matrix();
    let (r11, r12, r22, r66) = q_ref(140_000.0, 10_000.0, 0.3, 5_000.0);
    near(q11, r11, 1e-12, 0.0, "Q11");
    near(q12, r12, 1e-12, 0.0, "Q12");
    near(q22, r22, 1e-12, 0.0, "Q22");
    near(q66, r66, 1e-12, 0.0, "Q66");
    // Maxwell reciprocity: Q12 = nu_TL * Q11 = nu_LT * Q22
    near(q12, 0.3 * q22.to_f64(), 1e-12, 0.0, "Q12 = nu_LT Q22");
    // degenerate inputs give all-zero (documented guard)
    let z = ply(0.1, 0.0, mat(0.0, 1.0, 0.3, 1.0)).q_matrix();
    assert!(z.0.is_zero() && z.1.is_zero() && z.2.is_zero() && z.3.is_zero());
    let z = ply(0.1, 0.0, mat(1.0, 0.0, 0.3, 1.0)).q_matrix();
    assert!(z.0.is_zero() && z.3.is_zero());
}

#[test]
fn q_bar_on_axis_and_cross_ply() {
    let (r11, r12, r22, r66) = q_ref(140_000.0, 10_000.0, 0.3, 5_000.0);
    let (a11, a12, a22, a16, a26, a66) = ply(0.1, 0.0, m()).q_bar();
    near(a11, r11, 1e-12, 1e-6, "0 deg Q11");
    near(a12, r12, 1e-12, 1e-6, "0 deg Q12");
    near(a22, r22, 1e-12, 1e-6, "0 deg Q22");
    near(a66, r66, 1e-12, 1e-6, "0 deg Q66");
    near(a16, 0.0, 0.0, 1e-6, "0 deg Q16");
    near(a26, 0.0, 0.0, 1e-6, "0 deg Q26");
    let (b11, b12, b22, b16, b26, b66) = ply(0.1, std::f64::consts::FRAC_PI_2, m()).q_bar();
    near(b11, r22, 1e-9, 1e-3, "90 deg Q11 = Q22");
    near(b22, r11, 1e-9, 1e-3, "90 deg Q22 = Q11");
    near(b12, r12, 1e-9, 1e-3, "90 deg Q12");
    near(b66, r66, 1e-9, 1e-3, "90 deg Q66");
    near(b16, 0.0, 0.0, 1e-3, "90 deg Q16");
    near(b26, 0.0, 0.0, 1e-3, "90 deg Q26");
}

#[test]
fn q_bar_45_degree_closed_form() {
    let (q11, q12, q22, q66) = q_ref(140_000.0, 10_000.0, 0.3, 5_000.0);
    for sign in [1.0, -1.0] {
        let (a11, a12, a22, a16, a26, a66) =
            ply(0.1, sign * std::f64::consts::FRAC_PI_4, m()).q_bar();
        let tol = 1e-8;
        near(
            a11,
            0.25 * (q11 + q22 + 2.0 * q12 + 4.0 * q66),
            tol,
            1e-4,
            "Q11 45",
        );
        near(
            a22,
            0.25 * (q11 + q22 + 2.0 * q12 + 4.0 * q66),
            tol,
            1e-4,
            "Q22 45",
        );
        near(
            a12,
            0.25 * (q11 + q22 + 2.0 * q12 - 4.0 * q66),
            tol,
            1e-4,
            "Q12 45",
        );
        near(a66, 0.25 * (q11 + q22 - 2.0 * q12), tol, 1e-4, "Q66 45");
        near(a16, sign * 0.25 * (q11 - q22), tol, 1e-4, "Q16 45");
        near(a26, sign * 0.25 * (q11 - q22), tol, 1e-4, "Q26 45");
    }
}

#[test]
fn q_bar_rotation_invariants_and_isotropic_invariance() {
    let (q11, q12, q22, q66) = q_ref(140_000.0, 10_000.0, 0.3, 5_000.0);
    // Tsai-Pagano invariants: Qbar11 + Qbar22 + 2 Qbar12 = Q11 + Q22 + 2 Q12 and
    // Qbar12 - Qbar66 = Q12 - Q66 hold at every angle.
    let inv1 = q11 + q22 + 2.0 * q12;
    let inv2 = q12 - q66;
    for k in 0..24 {
        let th = k as f64 * 0.13 - 1.4;
        let (a11, a12, a22, _a16, _a26, a66) = ply(0.1, th, m()).q_bar();
        let (g11, g12, g22, g66) = (a11.to_f64(), a12.to_f64(), a22.to_f64(), a66.to_f64());
        assert!((g11 + g22 + 2.0 * g12 - inv1).abs() < 1e-3, "inv1 at {th}");
        assert!((g12 - g66 - inv2).abs() < 1e-3, "inv2 at {th}");
    }
    // isotropic ply (E, nu, G = E/2(1+nu)): Qbar independent of angle, Q16 = Q26 = 0
    let e = 70_000.0;
    let nu = 0.25;
    let iso = mat(e, e, nu, e / (2.0 * (1.0 + nu)));
    for k in 0..12 {
        let th = k as f64 * 0.31;
        let (a11, a12, a22, a16, a26, a66) = ply(0.1, th, iso).q_bar();
        near(a11, e / (1.0 - nu * nu), 1e-9, 1e-3, "iso Q11");
        near(a22, e / (1.0 - nu * nu), 1e-9, 1e-3, "iso Q22");
        near(a12, nu * e / (1.0 - nu * nu), 1e-9, 1e-3, "iso Q12");
        near(a66, e / (2.0 * (1.0 + nu)), 1e-9, 1e-3, "iso Q66");
        near(a16, 0.0, 0.0, 1e-3, "iso Q16");
        near(a26, 0.0, 0.0, 1e-3, "iso Q26");
    }
}

#[test]
fn single_ply_abd_closed_form() {
    let t = 0.5;
    let abd = compute_abd(&[ply(t, 0.0, m())]);
    let (q11, q12, q22, q66) = q_ref(140_000.0, 10_000.0, 0.3, 5_000.0);
    near(abd.a.m11, q11 * t, 1e-9, 1e-6, "A11");
    near(abd.a.m12, q12 * t, 1e-9, 1e-6, "A12");
    near(abd.a.m22, q22 * t, 1e-9, 1e-6, "A22");
    near(abd.a.m33, q66 * t, 1e-9, 1e-6, "A66");
    near(abd.d.m11, q11 * t * t * t / 12.0, 1e-9, 1e-6, "D11");
    near(abd.d.m22, q22 * t * t * t / 12.0, 1e-9, 1e-6, "D22");
    near(abd.d.m12, q12 * t * t * t / 12.0, 1e-9, 1e-6, "D12");
    near(abd.d.m33, q66 * t * t * t / 12.0, 1e-9, 1e-6, "D66");
    near(abd.b.m11, 0.0, 0.0, 1e-9, "B11");
    near(abd.b.m22, 0.0, 0.0, 1e-9, "B22");
    near(abd.b.m12, 0.0, 0.0, 1e-9, "B12");
    near(abd.b.m33, 0.0, 0.0, 1e-9, "B66");
}

#[test]
fn cross_ply_two_layer_abd_closed_form() {
    // [0/90], plies listed bottom to top, each thickness t, mid-plane at z = 0.
    // B = t^2/2 (Q90 - Q0), A = (Q0+Q90) t, D = t^3/3 (Q0+Q90)
    let t = 1.0;
    let abd = compute_abd(&[ply(t, 0.0, m()), ply(t, std::f64::consts::FRAC_PI_2, m())]);
    let (q11, q12, q22, q66) = q_ref(140_000.0, 10_000.0, 0.3, 5_000.0);
    near(abd.a.m11, (q11 + q22) * t, 1e-9, 1e-3, "A11");
    near(abd.a.m22, (q11 + q22) * t, 1e-9, 1e-3, "A22");
    near(abd.a.m12, 2.0 * q12 * t, 1e-9, 1e-3, "A12");
    near(abd.a.m33, 2.0 * q66 * t, 1e-9, 1e-3, "A66");
    near(abd.b.m11, 0.5 * t * t * (q22 - q11), 1e-9, 1e-3, "B11");
    near(abd.b.m22, 0.5 * t * t * (q11 - q22), 1e-9, 1e-3, "B22");
    near(abd.b.m12, 0.0, 0.0, 1e-3, "B12");
    near(abd.b.m33, 0.0, 0.0, 1e-3, "B66");
    near(abd.d.m11, t * t * t / 3.0 * (q11 + q22), 1e-9, 1e-3, "D11");
    near(abd.d.m12, t * t * t / 3.0 * 2.0 * q12, 1e-9, 1e-3, "D12");
    near(abd.d.m33, t * t * t / 3.0 * 2.0 * q66, 1e-9, 1e-3, "D66");
}

#[test]
fn reversing_stack_flips_b_and_keeps_a_d() {
    let stack = [ply(0.2, 0.3, m()), ply(0.4, -0.7, m()), ply(0.1, 1.1, m())];
    let fwd = compute_abd(&stack);
    let mut rev = stack;
    rev.reverse();
    let r = compute_abd(&rev);
    let c = |x: Fix128, y: Fix128, w: &str, neg: bool| {
        let want = if neg { -y.to_f64() } else { y.to_f64() };
        assert!(
            (x.to_f64() - want).abs() <= 1e-6 * (1.0 + want.abs()),
            "{w}: {} vs {}",
            x.to_f64(),
            want
        );
    };
    c(r.a.m11, fwd.a.m11, "A11", false);
    c(r.a.m13, fwd.a.m13, "A16", false);
    c(r.d.m22, fwd.d.m22, "D22", false);
    c(r.d.m33, fwd.d.m33, "D66", false);
    c(r.b.m11, fwd.b.m11, "B11", true);
    c(r.b.m13, fwd.b.m13, "B16", true);
    c(r.b.m23, fwd.b.m23, "B26", true);
    assert!(
        fwd.b.m11.abs().to_f64() > 1.0,
        "test must have non-trivial B"
    );
}

#[test]
fn balanced_angle_ply_has_zero_a16_a26_and_symmetric_has_zero_b() {
    let th = 0.6;
    let abd = compute_abd(&[ply(0.2, th, m()), ply(0.2, -th, m())]);
    near(abd.a.m13, 0.0, 0.0, 1e-3, "A16 balanced");
    near(abd.a.m23, 0.0, 0.0, 1e-3, "A26 balanced");
    // A, D scaling: A ~ total thickness, D ~ total thickness^3 (same ply pattern, 2x thinner plies)
    let thin = compute_abd(&[ply(0.1, th, m()), ply(0.1, -th, m())]);
    near(
        abd.a.m11,
        2.0 * thin.a.m11.to_f64(),
        1e-9,
        1e-6,
        "A scales with t",
    );
    near(
        abd.d.m11,
        8.0 * thin.d.m11.to_f64(),
        1e-9,
        1e-6,
        "D scales with t^3",
    );
}

/// Doc: "Symmetric stacks have B = 0 exactly".
#[test]
#[ignore = "known defect: AUD-A-S1W5-009: compute_abd leaves B != 0 (rounding residue b11 = -2^-64 (raw hi=-1,lo=u64::MAX-1) for symmetric [a,b,c,b,a]; is_symmetric_stack doc says B = 0 exactly"]
fn symmetric_stack_has_exactly_zero_b() {
    let a = ply(0.2, 0.52, m());
    let b = ply(0.15, 1.04, m());
    let c = ply(0.3, 0.2, m());
    let stack = [a, b, c, b, a];
    assert!(is_symmetric_stack(&stack));
    let abd = compute_abd(&stack);
    for (n, v) in [
        ("b11", abd.b.m11),
        ("b12", abd.b.m12),
        ("b22", abd.b.m22),
        ("b13", abd.b.m13),
        ("b23", abd.b.m23),
        ("b33", abd.b.m33),
    ] {
        assert!(v.is_zero(), "{n} = {:?} ({})", v, v.to_f64());
    }
}

#[test]
fn is_symmetric_tolerance_boundary_and_each_entry() {
    let tol = f(0.5);
    let mut abd = AbdMatrix::default();
    assert!(abd.is_symmetric(Fix128::ZERO));
    // each B entry independently trips the check, boundary (== tol) passes, sign ignored
    let set: [fn(&mut AbdMatrix, Fix128); 6] = [
        |a, v| a.b.m11 = v,
        |a, v| a.b.m12 = v,
        |a, v| a.b.m22 = v,
        |a, v| a.b.m13 = v,
        |a, v| a.b.m23 = v,
        |a, v| a.b.m33 = v,
    ];
    for s in set {
        abd = AbdMatrix::default();
        s(&mut abd, tol);
        assert!(abd.is_symmetric(tol), "== tol passes");
        s(&mut abd, -tol);
        assert!(abd.is_symmetric(tol), "== -tol passes (abs)");
        s(&mut abd, f(0.5000001));
        assert!(!abd.is_symmetric(tol));
        s(&mut abd, f(-0.5000001));
        assert!(!abd.is_symmetric(tol));
    }
    // A and D entries are irrelevant
    let mut abd = AbdMatrix::default();
    abd.a.m11 = f(1e6);
    abd.d.m33 = f(1e6);
    assert!(abd.is_symmetric(tol));
}

#[test]
fn is_symmetric_stack_semantics() {
    let a = ply(0.2, 0.0, m());
    let b = ply(0.2, 1.0, m());
    assert!(is_symmetric_stack(&[]));
    assert!(is_symmetric_stack(&[a]));
    assert!(is_symmetric_stack(&[a, a]));
    assert!(!is_symmetric_stack(&[a, b]));
    assert!(is_symmetric_stack(&[a, b, a]));
    assert!(is_symmetric_stack(&[a, b, b, a]));
    assert!(!is_symmetric_stack(&[a, b, a, b]));
    // thickness, orientation, material each break symmetry independently
    let thick = ply(0.3, 0.0, m());
    assert!(!is_symmetric_stack(&[a, b, thick]));
    let mat2 = mat(1.0e5, 1.0e4, 0.3, 5_000.0);
    assert!(!is_symmetric_stack(&[a, b, ply(0.2, 0.0, mat2)]));
    // odd length: middle ply is unconstrained
    assert!(is_symmetric_stack(&[a, b, thick, b, a]));
    // inner mismatch only
    assert!(!is_symmetric_stack(&[a, b, thick, thick, a]));
}

#[test]
fn sym3_add_scale_entrywise() {
    let x = Sym3 {
        m11: f(1.0),
        m12: f(2.0),
        m22: f(3.0),
        m13: f(4.0),
        m23: f(5.0),
        m33: f(6.0),
    };
    let y = Sym3 {
        m11: f(10.0),
        m12: f(20.0),
        m22: f(30.0),
        m13: f(40.0),
        m23: f(50.0),
        m33: f(60.0),
    };
    let s = x.add(&y);
    for (g, w) in [
        (s.m11, 11.0),
        (s.m12, 22.0),
        (s.m22, 33.0),
        (s.m13, 44.0),
        (s.m23, 55.0),
        (s.m33, 66.0),
    ] {
        assert_eq!(g.to_f64(), w);
    }
    let k = x.scale(f(-0.5));
    for (g, w) in [
        (k.m11, -0.5),
        (k.m12, -1.0),
        (k.m22, -1.5),
        (k.m13, -2.0),
        (k.m23, -2.5),
        (k.m33, -3.0),
    ] {
        assert_eq!(g.to_f64(), w);
    }
}
