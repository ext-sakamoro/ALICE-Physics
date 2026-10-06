//! Stress-concentration factor (Peterson) production entry point for
//! `alice_physics::fillet_stress`: `kt_circular_hole_infinite_plate`,
//! `kt_elliptical_hole`, `kt_u_notch_axial`, and
//! `recommended_fillet_radius_mm`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all four as `unwired`.
//! `src/fillet_stress.rs`'s own `#[cfg(test)]` module exercises them, but
//! unit tests do not count as production callers for the wiring guard,
//! and nothing in `src/` / `examples/` / `benches/` called any of the
//! four before this file existed. This example is that caller.
//!
//! `tests/analytic_fillet_stress_wiring.rs` holds additional closed-form
//! oracles that this file does not cover: the elliptical-hole circle
//! limit (`a == b`) and needle limit (`a >> b`), U-notch at the formula's
//! stated validity boundary (`h/r == 10`), and `recommended_fillet_radius_mm`'s
//! degenerate inputs (`kt_target <= 1`, zero-diameter shaft, extreme
//! magnitude geometry).
//!
//! Every expected value below is derived by hand from the textbook
//! closed form quoted in each section's comment -- never by calling the
//! `alice_physics::fillet_stress` function under test.
//!
//! ```bash
//! cargo run --example fillet_stress_concentration --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::fillet_stress::{
    kt_circular_hole_infinite_plate, kt_elliptical_hole, kt_shaft_shoulder_bending,
    kt_u_notch_axial, recommended_fillet_radius_mm,
};
use alice_physics::math::Fix128;

/// Relative-error check against an independently hand-derived f64 closed
/// form. The expected value (`want`) is computed by a formula written
/// out again in plain f64 arithmetic in `main` below -- never by calling
/// the function under test.
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = if want == 0.0 {
        g.abs()
    } else {
        ((g - want) / want).abs()
    };
    assert!(
        err <= tol,
        "[fillet_stress] MISMATCH {what}: got {g}, want {want} (err {err:.3e} > {tol:.1e})"
    );
    println!("[fillet_stress] ok {what}: got {g:.6}, want {want:.6} (err {err:.2e})");
}

/// Absolute-error check, for expectations that are exactly zero or where
/// a relative comparison is otherwise unsuitable (e.g. the two sides are
/// already known to be astronomically large).
fn assert_abs(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs();
    assert!(
        err <= tol,
        "[fillet_stress] MISMATCH {what}: got {g}, want {want} (abs err {err:.3e} > {tol:.1e})"
    );
    println!("[fillet_stress] ok {what}: got {g:.6}, want {want:.6} (abs err {err:.2e})");
}

fn main() {
    let tol = 1e-9;

    // ------------------------------------------------------------------
    // 1. kt_circular_hole_infinite_plate -- Kirsch's 1898 exact elastic
    //    solution for a small circular hole in an infinite plate under
    //    remote uniaxial tension: K_t = 3, independent of hole size.
    // ------------------------------------------------------------------
    let kt_circular = kt_circular_hole_infinite_plate();
    assert_rel(
        kt_circular,
        3.0,
        tol,
        "kt_circular_hole_infinite_plate K_t = 3 (Kirsch 1898)",
    );

    // ------------------------------------------------------------------
    // 2. kt_elliptical_hole -- Inglis (1913): K_t = 1 + 2*a/b, where `a`
    //    is the semi-axis perpendicular to the remote load and `b` the
    //    semi-axis parallel to it.
    //
    //    Case A: a = 8mm, b = 4mm (a > b, ellipse elongated across the
    //    load path) -> K_t = 1 + 2*(8/4) = 5.
    //    Case B: a = 3mm, b = 9mm (a < b, ellipse elongated along the
    //    load path, the weaker concentrator of the two orientations)
    //    -> K_t = 1 + 2*(3/9) = 1 + 2/3 = 1.6666...
    // ------------------------------------------------------------------
    let kt_ellipse_a = kt_elliptical_hole(Fix128::from_int(8), Fix128::from_int(4));
    assert_rel(
        kt_ellipse_a,
        1.0 + 2.0 * (8.0 / 4.0),
        tol,
        "kt_elliptical_hole(a=8,b=4) K_t = 1+2a/b (Inglis 1913)",
    );

    let kt_ellipse_b = kt_elliptical_hole(Fix128::from_int(3), Fix128::from_int(9));
    assert_rel(
        kt_ellipse_b,
        1.0 + 2.0 * (3.0 / 9.0),
        tol,
        "kt_elliptical_hole(a=3,b=9) K_t = 1+2a/b (Inglis 1913)",
    );

    // ------------------------------------------------------------------
    // 3. kt_u_notch_axial -- Peterson curve-fit (Pilkey Table 2-8):
    //    K_t = 0.85 + 2*sqrt(h/r), valid for h/r <= 10.
    //
    //    Case A: h = 6mm, r = 1.5mm -> h/r = 4, K_t = 0.85 + 2*sqrt(4)
    //    = 0.85 + 4 = 4.85 (exact: sqrt(4) = 2 is a perfect square, so
    //    this case also pins the sqrt path at a rational result).
    //    Case B: h = 10mm, r = 1mm -> h/r = 10, the formula's own stated
    //    upper validity bound, K_t = 0.85 + 2*sqrt(10) = 7.174555...
    // ------------------------------------------------------------------
    let kt_unotch_a = kt_u_notch_axial(Fix128::from_int(6), Fix128::from_ratio(3, 2));
    assert_rel(
        kt_unotch_a,
        0.85 + 2.0 * (4.0_f64).sqrt(),
        tol,
        "kt_u_notch_axial(h=6,r=1.5) K_t = 0.85+2*sqrt(h/r) (Pilkey Table 2-8)",
    );

    let kt_unotch_b = kt_u_notch_axial(Fix128::from_int(10), Fix128::from_int(1));
    assert_rel(
        kt_unotch_b,
        0.85 + 2.0 * (10.0_f64).sqrt(),
        tol,
        "kt_u_notch_axial(h=10,r=1) K_t = 0.85+2*sqrt(h/r) at h/r=10 validity bound",
    );

    // ------------------------------------------------------------------
    // 4. recommended_fillet_radius_mm -- binary search over `r/d` on
    //    `kt_shaft_shoulder_bending` (the Peterson curve fit for a stepped
    //    round bar in bending, Pilkey 2nd ed.), inverted to find the
    //    smallest fillet radius that keeps K_t <= target.
    //
    //    Shaft: small_dia = 25mm, large_dia = 45mm: step h = 10 mm,
    //    t = 2h/D = 20/45. With x = h/r the fit is
    //      K_t = C1 + C2 t + C3 t^2 + C4 t^3, Ci = ai + bi sqrt(x) + ci x
    //    (the 2 <= x <= 20 set below); K_t = 2 is solved for x by f64
    //    bisection here, then r = h / x (about 1.76 mm, x about 5.7).
    // ------------------------------------------------------------------
    let small_dia = Fix128::from_int(25);
    let large_dia = Fix128::from_int(45);
    let kt_target = Fix128::from_int(2);
    let r_needed = recommended_fillet_radius_mm(small_dia, large_dia, kt_target);
    let t = 20.0_f64 / 45.0;
    let fit_high = |x: f64| {
        let s = x.sqrt();
        let c1 = 1.232 + 0.832 * s - 0.008 * x;
        let c2 = -3.813 + 0.968 * s - 0.260 * x;
        let c3 = 7.423 - 4.868 * s + 0.869 * x;
        let c4 = -3.839 + 3.070 * s - 0.600 * x;
        c1 + c2 * t + c3 * t * t + c4 * t * t * t
    };
    let (mut lo, mut hi) = (2.0_f64, 20.0_f64); // K_t rises with x
    assert!(fit_high(lo) < 2.0 && fit_high(hi) > 2.0, "root bracketed");
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if fit_high(mid) > 2.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let r_closed = 10.0 / lo;
    assert_rel(
        r_needed,
        r_closed,
        1e-6,
        "recommended_fillet_radius_mm(d=25,D=45,Kt<=2) radius from the fit solved in f64",
    );

    // Cross-check the achieved K_t at that radius, via the already-wired
    // `kt_shaft_shoulder_bending` (not the function under test here) --
    // confirms the returned radius actually satisfies the target, not
    // just that the bisection landed on the expected r/d.
    let kt_achieved = kt_shaft_shoulder_bending(r_needed, small_dia, large_dia);
    // continuous in r, so the minimal radius meets the target with equality
    // (to the bisection's 30 halvings of r/d in [0, 0.5])
    assert_abs(
        kt_achieved,
        2.0,
        1e-6,
        "kt_shaft_shoulder_bending(r_needed,25,45) K_t at recommended radius",
    );
    assert!(
        kt_achieved.to_f64() <= 2.0,
        "[fillet_stress] MISMATCH: recommended_fillet_radius_mm must satisfy K_t <= target"
    );

    println!(
        "[fillet_stress] all 4 production entry points (kt_circular_hole_infinite_plate, \
         kt_elliptical_hole, kt_u_notch_axial, recommended_fillet_radius_mm) verified \
         against hand-derived Peterson/Inglis/Kirsch closed forms"
    );
}
