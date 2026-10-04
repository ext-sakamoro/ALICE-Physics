//! Audit oracles for `alice_physics::non_newtonian` (S2-2 audit).
//!
//! Expected values are the textbook closed forms (Bird-Stewart-Lightfoot ch. 8,
//! Chhabra-Richardson) evaluated in plain f64 with multiplication only (the
//! crate's clippy config forbids platform powf in tests), never the function
//! under test.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::non_newtonian::{Bingham, Carreau, HerschelBulkley, PowerLaw};

fn rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs() / want.abs().max(1e-300);
    assert!(err <= tol, "{what}: got {g}, want {want}, rel err {err:e}");
}

fn fx(num: i64, den: i64) -> Fix128 {
    Fix128::from_ratio(num, den)
}

/// tau = K * rate^(1/n), n in {2,3,4}, over three decades (doc: shear-thinning
/// flow index 1/n). powf_pos keeps 24 fractional exponent bits, so the
/// non-dyadic 1/3 is truncated; the tolerance bounds that truncation.
#[test]
fn shear_thinning_stress_matches_k_rate_to_the_one_over_n() {
    // (n, [(rate, exact n-th root)])
    let cases: [(u32, [(i64, f64); 3]); 3] = [
        (2, [(4, 2.0), (100, 10.0), (10000, 100.0)]),
        (3, [(8, 2.0), (125, 5.0), (1000, 10.0)]),
        (4, [(16, 2.0), (81, 3.0), (10000, 10.0)]),
    ];
    for (n, table) in cases {
        let p = PowerLaw::shear_thinning(fx(5, 2), n);
        for (rate, root) in table {
            rel(
                p.stress(Fix128::from_int(rate)),
                2.5 * root,
                1e-6,
                &format!("n={n} rate={rate}"),
            );
        }
    }
}

/// Doc: the (post 1.2.0) shear-thinning stress keeps growing with shear rate.
#[test]
fn shear_thinning_stress_is_strictly_increasing_in_rate() {
    let p = PowerLaw::shear_thinning(fx(3, 1), 3);
    let mut prev = Fix128::ZERO;
    for r in [1, 2, 3, 5, 8, 13, 100, 1000] {
        let s = p.stress(Fix128::from_int(r));
        assert!(s > prev, "stress not increasing at rate {r}");
        prev = s;
    }
}

/// eta = tau / rate: thickening K rate^(n-1), thinning K rate^(1/n - 1).
#[test]
fn apparent_viscosity_follows_the_power_law_exponent_minus_one() {
    let thick = PowerLaw::shear_thickening(fx(3, 2), 3);
    rel(
        thick.apparent_viscosity(Fix128::from_int(7)),
        1.5 * 49.0,
        1e-12,
        "thick n=3 eta(7)",
    );
    let thin = PowerLaw::shear_thinning(Fix128::ONE, 3);
    // eta(8)/eta(1) = 8^(1/3-1) = 8^(-2/3) = 1/4
    let ratio = thin.apparent_viscosity(Fix128::from_int(8)).to_f64()
        / thin.apparent_viscosity(Fix128::ONE).to_f64();
    assert!((ratio - 0.25).abs() < 1e-6, "ratio {ratio}");
}

/// Herschel-Bulkley with zero yield is the power law; tau - tau_y = K rate^n.
#[test]
fn herschel_bulkley_minus_yield_is_the_power_law_term() {
    for n in [2u32, 3, 4] {
        let hb0 = HerschelBulkley {
            yield_stress: Fix128::ZERO,
            k: fx(7, 4),
            n_int: n,
        };
        let pl = PowerLaw::shear_thickening(fx(7, 4), n);
        let hb = HerschelBulkley {
            yield_stress: Fix128::from_int(12),
            k: fx(7, 4),
            n_int: n,
        };
        for r in [1, 3, 9] {
            let g = Fix128::from_int(r);
            assert_eq!(hb0.stress(g), pl.stress(g));
            assert_eq!(hb.stress(g) - Fix128::from_int(12), pl.stress(g));
        }
    }
}

/// Bingham tau = tau_y + mu_p rate for rate > 0, with exact binary fractions.
#[test]
fn bingham_is_affine_in_rate_above_zero() {
    let b = Bingham {
        yield_stress: fx(21, 2),
        plastic_viscosity: fx(3, 8),
    };
    for r in [1i64, 2, 16, 400] {
        let want = 10.5 + 0.375 * r as f64;
        assert_eq!(b.stress(Fix128::from_int(r)).to_f64(), want);
    }
    // negative rate: no flow, zero
    assert_eq!(b.stress(Fix128::from_int(-3)), Fix128::ZERO);
}

/// Carreau integer path closed forms: eta = eta_inf + (eta0 - eta_inf)(1+(lambda rate)^2)^h.
#[test]
fn carreau_integer_exponent_matches_closed_form() {
    for (h, rate) in [(-2i32, 3i64), (-1, 5), (1, 2), (2, 3)] {
        let c = Carreau {
            eta_zero: Fix128::from_int(120),
            eta_inf: Fix128::from_int(8),
            lambda: fx(1, 2),
            half_exponent: h,
        };
        let x = 0.5 * rate as f64;
        let inside = 1.0 + x * x;
        let mut f = 1.0;
        for _ in 0..h.unsigned_abs() {
            f *= inside;
        }
        let f = if h < 0 { 1.0 / f } else { f };
        let want = 8.0 + 112.0 * f;
        rel(
            c.viscosity(Fix128::from_int(rate)),
            want,
            1e-12,
            &format!("h={h} rate={rate}"),
        );
    }
}

/// Carreau depends on (lambda rate)^2, so it is even in the rate and lambda = 0 is Newtonian eta0.
#[test]
fn carreau_is_even_in_rate_and_lambda_zero_is_newtonian() {
    let c = Carreau {
        eta_zero: Fix128::from_int(100),
        eta_inf: Fix128::from_int(2),
        lambda: fx(3, 4),
        half_exponent: -1,
    };
    assert_eq!(
        c.viscosity(Fix128::from_int(6)),
        c.viscosity(Fix128::from_int(-6))
    );
    let n = Carreau {
        lambda: Fix128::ZERO,
        ..c
    };
    assert_eq!(n.viscosity(Fix128::from_int(1000)), Fix128::from_int(100));
}

/// The fractional-index path agrees with the integer path where both apply
/// (n = 2h + 1) and with the closed form at a genuine fractional index.
#[test]
fn carreau_fractional_and_integer_paths_agree_at_integer_half_exponents() {
    let base = Carreau {
        eta_zero: Fix128::from_int(120),
        eta_inf: Fix128::from_int(8),
        lambda: fx(1, 2),
        half_exponent: 0,
    };
    for (h, n) in [(-1i32, -1i64), (1, 3), (-2, -3), (2, 5)] {
        let c = Carreau {
            half_exponent: h,
            ..base
        };
        for rate in [1i64, 4, 20] {
            let a = c.viscosity(Fix128::from_int(rate)).to_f64();
            let b = base
                .viscosity_with_index(Fix128::from_int(rate), Fix128::from_int(n))
                .to_f64();
            assert!(
                (a - b).abs() / a.abs() < 1e-6,
                "h={h} n={n} rate={rate}: {a} vs {b}"
            );
        }
    }
    // n = 1 is Newtonian at every rate
    assert_eq!(
        base.viscosity_with_index(Fix128::from_int(50), Fix128::ONE),
        Fix128::from_int(120)
    );
    // n = 1/2: (1 + x^2)^(-1/4); at rate=6, x=3, inside=10: 10^(-0.25)
    let want = 8.0 + 112.0 * 0.562_341_325_190_349_1;
    rel(
        base.viscosity_with_index(Fix128::from_int(6), fx(1, 2)),
        want,
        1e-6,
        "n=1/2 rate=6",
    );
}

/// K rate^n overflows Fix128 (+-9.2e18) at rate=5e6, n=3 (1.25e20): the product wraps
/// silently, so a thickening stress that must be positive and increasing comes out garbage.
#[test]
fn thickening_stress_stays_positive_and_increasing_at_high_rate() {
    let p = PowerLaw::shear_thickening(Fix128::ONE, 3);
    let lo = p.stress(Fix128::from_int(1_000_000));
    let hi = p.stress(Fix128::from_int(5_000_000));
    assert!(
        hi > lo,
        "stress(5e6) = {} <= stress(1e6) = {}",
        hi.to_f64(),
        lo.to_f64()
    );
}
