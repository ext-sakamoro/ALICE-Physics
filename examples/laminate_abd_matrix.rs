//! Classical Laminate Theory (CLT) production entry point for
//! `alice_physics::laminate`: `Ply::q_matrix`, `Ply::q_bar`, `compute_abd`,
//! `is_symmetric_stack`, and `AbdMatrix::is_symmetric`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all five as `unwired`.
//! `tests/engineering_oracles_solid.rs` already drives `q_matrix` / `q_bar`
//! / `compute_abd` / `is_symmetric_stack` / `is_symmetric` against the
//! Jones (*Mechanics of Composite Materials*, 2nd ed.) closed forms for a
//! T300/5208 cross-ply and `+-45` angle-ply stack, but tests do not count
//! as production callers for the wiring guard, and nothing in `src/` /
//! `examples/` / `benches/` called any of the five before this file
//! existed. This example is that caller.
//!
//! `tests/analytic_laminate_wiring.rs` holds additional closed-form
//! oracles that the existing oracle file does not cover: a non-special
//! (30 degree) rotation angle for `q_bar`, the two divide-by-zero guard
//! branches inside `q_matrix`, a single-ply and a zero-thickness-ply
//! laminate for `compute_abd`, and odd-length / material-only-differing
//! stacks for `is_symmetric_stack`.
//!
//! ```bash
//! cargo run --example laminate_abd_matrix --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::anisotropic::OrthotropicElasticity;
use alice_physics::laminate::{compute_abd, is_symmetric_stack, AbdMatrix, Ply, Sym3};
use alice_physics::math::Fix128;

/// Relative-error check against an independently hand-derived f64 closed
/// form. The expected value (`want`) is computed by a formula written out
/// again in plain f64 arithmetic in `main` below -- never by calling the
/// function under test -- per this repo's analytic-oracle-tests
/// discipline (`~/claude-config/rules/analytic-oracle-tests.md`).
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = if want == 0.0 {
        g.abs()
    } else {
        ((g - want) / want).abs()
    };
    assert!(
        err <= tol,
        "[laminate] MISMATCH {what}: got {g}, want {want} (err {err:.3e} > {tol:.1e})"
    );
    println!("[laminate] ok {what}: got {g:.6}, want {want:.6} (err {err:.2e})");
}

/// T300/5208 graphite-epoxy (Jones Table 2.3): E1 = 181 GPa, E2 = 10.3 GPa,
/// nu12 = 0.28, G12 = 7.17 GPa. The through-thickness fields are unused by
/// in-plane CLT and are set to the transverse in-plane values so the
/// struct literal is a well-formed (if unused) material.
fn t300_5208() -> OrthotropicElasticity {
    OrthotropicElasticity {
        e_l_mpa: Fix128::from_int(181_000),
        e_t_mpa: Fix128::from_int(10_300),
        e_z_mpa: Fix128::from_int(10_300),
        nu_lt: Fix128::from_ratio(28, 100),
        nu_lz: Fix128::from_ratio(28, 100),
        nu_tz: Fix128::from_ratio(40, 100),
        g_lt_mpa: Fix128::from_int(7_170),
        g_lz_mpa: Fix128::from_int(7_170),
        g_tz_mpa: Fix128::from_int(3_000),
    }
}

/// Jones eq. 2.61 reduced-stiffness closed form, independent f64
/// reference: `Q11 = E1/(1-nu12*nu21)`, `Q12 = nu12*E2/(1-nu12*nu21)`,
/// `Q22 = E2/(1-nu12*nu21)`, `Q66 = G12`, with `nu21 = nu12*E2/E1`
/// (Maxwell reciprocity).
fn q_ref_f64(e1: f64, e2: f64, nu12: f64, g12: f64) -> (f64, f64, f64, f64) {
    let nu21 = nu12 * e2 / e1;
    let d = 1.0 - nu12 * nu21;
    (e1 / d, nu12 * e2 / d, e2 / d, g12)
}

/// Jones eq. 2.84 axis-transformation closed form, independent f64
/// reference. Takes `cos(theta)` / `sin(theta)` as plain numbers (rather
/// than computing them here) so this function never calls a trigonometric
/// method itself -- `clippy.toml`'s `disallowed-methods` bans `f64::sin` /
/// `f64::cos` crate-wide (determinism gate: every transcendental must go
/// through `alice_physics::det_math`, not the platform libm) and that ban
/// applies to `--all-targets`, including this example.
fn q_bar_ref_f64(
    q11: f64,
    q12: f64,
    q22: f64,
    q66: f64,
    c: f64,
    s: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let c2 = c * c;
    let s2 = s * s;
    let c4 = c2 * c2;
    let s4 = s2 * s2;
    let c2s2 = c2 * s2;
    let c3s = c2 * c * s;
    let cs3 = c * s2 * s;
    let q11_bar = q11 * c4 + 2.0 * (q12 + 2.0 * q66) * c2s2 + q22 * s4;
    let q22_bar = q11 * s4 + 2.0 * (q12 + 2.0 * q66) * c2s2 + q22 * c4;
    let q12_bar = (q11 + q22 - 4.0 * q66) * c2s2 + q12 * (c4 + s4);
    let q66_bar = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * c2s2 + q66 * (c4 + s4);
    let q16_bar = (q11 - q12 - 2.0 * q66) * c3s + (q12 - q22 + 2.0 * q66) * cs3;
    let q26_bar = (q11 - q12 - 2.0 * q66) * cs3 + (q12 - q22 + 2.0 * q66) * c3s;
    (q11_bar, q12_bar, q22_bar, q16_bar, q26_bar, q66_bar)
}

fn main() {
    let tol = 1e-9;
    let (e1, e2, nu12, g12) = (181_000.0_f64, 10_300.0_f64, 0.28_f64, 7_170.0_f64);
    let (q11_ref, q12_ref, q22_ref, q66_ref) = q_ref_f64(e1, e2, nu12, g12);

    // ------------------------------------------------------------------
    // 1. Ply::q_matrix -- Jones eq. 2.61 reduced stiffness, T300/5208.
    //    `q_matrix` does not depend on ply orientation, so the orientation
    //    chosen here is arbitrary.
    // ------------------------------------------------------------------
    let ply0 = Ply {
        thickness_mm: Fix128::from_ratio(1, 4),
        orientation_rad: Fix128::ZERO,
        material: t300_5208(),
    };
    let (q11, q12, q22, q66) = ply0.q_matrix();
    assert_rel(q11, q11_ref, tol, "q_matrix Q11 = E1/(1-nu12*nu21)");
    assert_rel(q12, q12_ref, tol, "q_matrix Q12 = nu12*E2/(1-nu12*nu21)");
    assert_rel(q22, q22_ref, tol, "q_matrix Q22 = E2/(1-nu12*nu21)");
    assert_rel(q66, q66_ref, tol, "q_matrix Q66 = G12");

    // ------------------------------------------------------------------
    // 2. Ply::q_bar -- Jones eq. 2.84 at theta = 30 deg, a rotation that is
    //    neither the 0 deg on-axis case (where q_bar collapses to
    //    q_matrix) nor the 45 deg special case already covered by
    //    tests/engineering_oracles_solid.rs. cos(30 deg) = sqrt(3)/2,
    //    sin(30 deg) = 1/2 exactly, so both are plain algebraic constants,
    //    not calls to a disallowed trigonometric method.
    // ------------------------------------------------------------------
    let cos30 = 3.0_f64.sqrt() / 2.0;
    let sin30 = 0.5_f64;
    let ply30 = Ply {
        thickness_mm: Fix128::from_ratio(1, 4),
        // theta = pi/6: CORDIC sin_cos of this Fix128 constant is what the
        // implementation evaluates against the independent cos30/sin30
        // above.
        orientation_rad: Fix128::PI / Fix128::from_int(6),
        material: t300_5208(),
    };
    let (qb11_ref, qb12_ref, qb22_ref, qb16_ref, qb26_ref, qb66_ref) =
        q_bar_ref_f64(q11_ref, q12_ref, q22_ref, q66_ref, cos30, sin30);
    let (qb11, qb12, qb22, qb16, qb26, qb66) = ply30.q_bar();
    assert_rel(qb11, qb11_ref, tol, "q_bar Qbar11(30deg)");
    assert_rel(qb12, qb12_ref, tol, "q_bar Qbar12(30deg)");
    assert_rel(qb22, qb22_ref, tol, "q_bar Qbar22(30deg)");
    assert_rel(qb16, qb16_ref, tol, "q_bar Qbar16(30deg)");
    assert_rel(qb26, qb26_ref, tol, "q_bar Qbar26(30deg)");
    assert_rel(qb66, qb66_ref, tol, "q_bar Qbar66(30deg)");

    // ------------------------------------------------------------------
    // 3. compute_abd -- two identical on-axis (0 deg) plies stacked. With
    //    Qbar == Q at every point through the thickness this is the
    //    homogeneous-plate closed form (not a CLT-specific citation, just
    //    the definition of A/B/D integrated against a constant Q over
    //    [-h/2, h/2]): A = Q*h, B = 0, D = Q*h^3/12.
    // ------------------------------------------------------------------
    let t = 0.25_f64;
    let t_fix = Fix128::from_ratio(1, 4);
    let homogeneous = [
        Ply {
            thickness_mm: t_fix,
            orientation_rad: Fix128::ZERO,
            material: t300_5208(),
        },
        Ply {
            thickness_mm: t_fix,
            orientation_rad: Fix128::ZERO,
            material: t300_5208(),
        },
    ];
    let abd = compute_abd(&homogeneous);
    let h = 2.0 * t;
    let h3_over_12 = h * h * h / 12.0;
    assert_rel(abd.a.m11, q11_ref * h, tol, "compute_abd A11 = Q11*h");
    assert_rel(abd.a.m22, q22_ref * h, tol, "compute_abd A22 = Q22*h");
    assert_rel(abd.a.m12, q12_ref * h, tol, "compute_abd A12 = Q12*h");
    assert_rel(abd.a.m33, q66_ref * h, tol, "compute_abd A66 = Q66*h");
    assert_rel(
        abd.d.m11,
        q11_ref * h3_over_12,
        tol,
        "compute_abd D11 = Q11*h^3/12",
    );
    assert_rel(
        abd.d.m22,
        q22_ref * h3_over_12,
        tol,
        "compute_abd D22 = Q22*h^3/12",
    );
    assert_rel(
        abd.d.m33,
        q66_ref * h3_over_12,
        tol,
        "compute_abd D66 = Q66*h^3/12",
    );
    assert_rel(abd.b.m11, 0.0, 1e-9, "compute_abd B11 = 0 (homogeneous)");

    // ------------------------------------------------------------------
    // 4. is_symmetric_stack -- the homogeneous 2-ply stack above is a
    //    trivial palindrome (both plies identical), so it must report
    //    symmetric. A stack whose two plies differ only in thickness is
    //    not a palindrome by the module's own equality check and must
    //    report asymmetric.
    // ------------------------------------------------------------------
    let homogeneous_is_symmetric = is_symmetric_stack(&homogeneous);
    assert!(
        homogeneous_is_symmetric,
        "[laminate] MISMATCH: two identical plies must be is_symmetric_stack"
    );
    println!(
        "[laminate] ok is_symmetric_stack(two identical 0deg plies) = {homogeneous_is_symmetric}"
    );

    let differing_thickness = [
        Ply {
            thickness_mm: Fix128::from_ratio(1, 4),
            orientation_rad: Fix128::ZERO,
            material: t300_5208(),
        },
        Ply {
            thickness_mm: Fix128::from_ratio(1, 2),
            orientation_rad: Fix128::ZERO,
            material: t300_5208(),
        },
    ];
    let differing_is_symmetric = is_symmetric_stack(&differing_thickness);
    assert!(
        !differing_is_symmetric,
        "[laminate] MISMATCH: plies with different thickness must not be is_symmetric_stack"
    );
    println!("[laminate] ok is_symmetric_stack(differing thickness) = {differing_is_symmetric}");

    // ------------------------------------------------------------------
    // 5. AbdMatrix::is_symmetric -- a direct, hand-built B matrix at the
    //    tolerance boundary. The check is `|B_ij| <= tol` component-wise,
    //    so `B11 == tol` (every other entry zero) must report symmetric,
    //    and `B11` strictly greater than `tol` must not.
    // ------------------------------------------------------------------
    let computed_is_symmetric = abd.is_symmetric(Fix128::from_ratio(1, 1_000_000_000));
    assert!(
        computed_is_symmetric,
        "[laminate] MISMATCH: homogeneous-stack B must be within tolerance of zero"
    );
    println!(
        "[laminate] ok is_symmetric(homogeneous-stack ABD, tol=1e-9) = {computed_is_symmetric}"
    );

    let tol_fix = Fix128::from_ratio(1, 1_000);
    let at_boundary = AbdMatrix {
        a: Sym3::default(),
        b: Sym3 {
            m11: tol_fix,
            ..Sym3::default()
        },
        d: Sym3::default(),
    };
    let boundary_result = at_boundary.is_symmetric(tol_fix);
    assert!(
        boundary_result,
        "[laminate] MISMATCH: |B11| == tol must report symmetric (<=, not <)"
    );
    println!("[laminate] ok is_symmetric(B11 == tol exactly) = {boundary_result}");

    let past_boundary = AbdMatrix {
        a: Sym3::default(),
        b: Sym3 {
            m11: tol_fix + tol_fix,
            ..Sym3::default()
        },
        d: Sym3::default(),
    };
    let past_result = past_boundary.is_symmetric(tol_fix);
    assert!(
        !past_result,
        "[laminate] MISMATCH: |B11| == 2*tol must report asymmetric"
    );
    println!("[laminate] ok is_symmetric(B11 == 2*tol) = {past_result}");

    println!(
        "[laminate] all 5 production entry points (q_matrix, q_bar, compute_abd, \
         is_symmetric_stack, is_symmetric) verified against hand-derived closed forms"
    );
}
