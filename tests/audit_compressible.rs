//! Audit S1-5 oracles for `alice_physics::compressible` (closed forms: Anderson,
//! Modern Compressible Flow; normal-shock tables for gamma = 1.4).
#![allow(clippy::disallowed_methods)]

use alice_physics::compressible::*;
use alice_physics::math::Fix128;

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn close(got: Fix128, want: f64, rel: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs() / want.abs().max(1e-12);
    assert!(err <= rel, "{what}: got {g} want {want} rel {err}");
}

#[test]
fn presets_match_documented_constants() {
    let a = IdealGas::air();
    close(a.gas_constant, 287.0, 1e-15, "air R");
    close(a.gamma, 1.4, 1e-15, "air gamma");
    let h = IdealGas::helium();
    close(h.gas_constant, 2077.0, 1e-15, "He R");
    close(h.gamma, 5.0 / 3.0, 1e-15, "He gamma");
}

#[test]
fn eos_roundtrips_and_closed_forms() {
    let g = IdealGas::air();
    // p = rho R T : 1.2 * 287 * 300 = 103320
    close(g.pressure(f(1.2), f(300.0)), 103_320.0, 1e-15, "p");
    close(g.density(f(103_320.0), f(300.0)), 1.2, 1e-14, "rho");
    close(g.temperature(f(103_320.0), f(1.2)), 300.0, 1e-14, "T");
    // zero guards
    assert!(g.density(f(1e5), Fix128::ZERO).is_zero());
    assert!(g.temperature(f(1e5), Fix128::ZERO).is_zero());
    let zr = IdealGas {
        gas_constant: Fix128::ZERO,
        gamma: f(1.4),
    };
    assert!(zr.density(f(1e5), f(300.0)).is_zero());
    assert!(zr.temperature(f(1e5), f(1.0)).is_zero());
}

#[test]
fn speed_of_sound_closed_forms() {
    let g = IdealGas::air();
    // sqrt(1.4*287*288) = 340.1...
    close(
        g.speed_of_sound(f(288.0)),
        (1.4f64 * 287.0 * 288.0).sqrt(),
        1e-12,
        "a(T)",
    );
    // exact: T=400 -> sqrt(160720)=400.8975...
    close(
        g.speed_of_sound(f(400.0)),
        (1.4f64 * 287.0 * 400.0).sqrt(),
        1e-12,
        "a(400)",
    );
    // consistent with from p,rho when p = rho R T
    let p = g.pressure(f(1.2), f(300.0));
    close(
        g.speed_of_sound_from_pd(p, f(1.2)),
        g.speed_of_sound(f(300.0)).to_f64(),
        1e-12,
        "a(p,rho)",
    );
    assert!(g.speed_of_sound_from_pd(f(1e5), Fix128::ZERO).is_zero());
    // helium
    let h = IdealGas::helium();
    close(
        h.speed_of_sound(f(300.0)),
        (5.0f64 / 3.0 * 2077.0 * 300.0).sqrt(),
        1e-12,
        "He a",
    );
}

#[test]
fn mach_number_closed_forms() {
    let g = IdealGas::air();
    let a = g.speed_of_sound(f(300.0)).to_f64();
    close(g.mach_number(f(2.0 * a), f(300.0)), 2.0, 1e-12, "M=2");
    // sign dropped: |u|
    close(g.mach_number(f(-2.0 * a), f(300.0)), 2.0, 1e-12, "M(-u)");
    // T=0 => a=0 : documented? returns 0
    assert!(g.mach_number(f(10.0), Fix128::ZERO).is_zero());
}

#[test]
fn stagnation_ratios_closed_form() {
    let g = IdealGas::air();
    close(
        stagnation_temp_ratio(&g, Fix128::ZERO),
        1.0,
        1e-15,
        "T0/T M=0",
    );
    close(stagnation_temp_ratio(&g, f(2.0)), 1.8, 1e-14, "T0/T M=2");
    close(stagnation_temp_ratio(&g, f(3.0)), 2.8, 1e-14, "T0/T M=3");
    // p0/p = (T0/T)^3.5
    close(
        stagnation_pressure_ratio(&g, Fix128::ONE),
        1.2f64.powf(3.5),
        1e-9,
        "p0/p M=1",
    );
    close(
        stagnation_pressure_ratio(&g, f(2.0)),
        1.8f64.powf(3.5),
        1e-9,
        "p0/p M=2",
    );
    close(
        stagnation_pressure_ratio(&g, Fix128::ZERO),
        1.0,
        1e-12,
        "p0/p M=0",
    );
    // helium exponent 2.5
    let h = IdealGas::helium();
    let base: f64 = 1.0 + (2.0 / 3.0) / 2.0 * 4.0;
    close(
        stagnation_pressure_ratio(&h, f(2.0)),
        base.powf(2.5),
        1e-8,
        "He p0/p M=2",
    );
    // M and -M symmetric
    close(
        stagnation_pressure_ratio(&g, f(-2.0)),
        1.8f64.powf(3.5),
        1e-9,
        "p0/p M=-2",
    );
}

#[test]
fn normal_shock_air_mach_two_table_values() {
    let g = IdealGas::air();
    let s = normal_shock_jump(&g, f(2.0));
    close(s.density_ratio, 8.0 / 3.0, 1e-12, "rho2/rho1");
    close(s.pressure_ratio, 4.5, 1e-12, "p2/p1");
    close(s.temperature_ratio, 4.5 / (8.0 / 3.0), 1e-12, "T2/T1");
    close(s.mach_downstream, (1.0_f64 / 3.0).sqrt(), 1e-12, "M2");
}

#[test]
fn normal_shock_other_mach_and_gas() {
    let g = IdealGas::air();
    // M=3: rho 3.8571, p 10.333, M2 0.47519
    let s = normal_shock_jump(&g, f(3.0));
    close(
        s.density_ratio,
        2.4 * 9.0 / (0.4 * 9.0 + 2.0),
        1e-12,
        "rho M3",
    );
    close(s.pressure_ratio, 1.0 + 2.8 / 2.4 * 8.0, 1e-12, "p M3");
    close(
        s.mach_downstream,
        ((0.4_f64 * 9.0 + 2.0) / (2.8 * 9.0 - 0.4)).sqrt(),
        1e-12,
        "M2 M3",
    );
    // strong shock limit rho -> (g+1)/(g-1) = 6
    let s = normal_shock_jump(&g, f(1000.0));
    close(s.density_ratio, 6.0, 1e-4, "strong shock rho");
    // helium M=2 : (g+1)=8/3 , (g-1)=2/3
    let h = IdealGas::helium();
    let s = normal_shock_jump(&h, f(2.0));
    close(
        s.density_ratio,
        (8.0 / 3.0 * 4.0) / (2.0 / 3.0 * 4.0 + 2.0),
        1e-12,
        "He rho",
    );
    close(
        s.pressure_ratio,
        1.0 + (10.0 / 3.0) / (8.0 / 3.0) * 3.0,
        1e-12,
        "He p",
    );
    // entropy: shock must raise stagnation-pressure loss (p02/p01 < 1) -- 2nd-law oracle
    // p02/p01 = (p2/p1) * (T0 ratio at M2 / T0 ratio at M1 ...) via isentropic factors
    let g = IdealGas::air();
    let s = normal_shock_jump(&g, f(2.0));
    let p02_over_p1 =
        stagnation_pressure_ratio(&g, s.mach_downstream).to_f64() * s.pressure_ratio.to_f64();
    let p01_over_p1 = stagnation_pressure_ratio(&g, f(2.0)).to_f64();
    close(
        Fix128::from_f64(p02_over_p1 / p01_over_p1),
        0.7208,
        2e-4,
        "p02/p01 table (M=2 air)",
    );
}

#[test]
fn normal_shock_at_or_below_sonic_is_identity() {
    let g = IdealGas::air();
    for m in [0.0, 0.5, 1.0] {
        let s = normal_shock_jump(&g, f(m));
        assert_eq!(s.density_ratio, Fix128::ONE);
        assert_eq!(s.pressure_ratio, Fix128::ONE);
        assert_eq!(s.temperature_ratio, Fix128::ONE);
        close(s.mach_downstream, m.max(1e-300), 1e-15, "M2 passthrough");
    }
    // just above sonic: continuity with the identity branch
    let s = normal_shock_jump(&g, f(1.000001));
    assert!((s.pressure_ratio.to_f64() - 1.0).abs() < 1e-5);
    assert!((s.mach_downstream.to_f64() - 1.0).abs() < 1e-5);
}

#[test]
fn riemann_invariants_closed_form() {
    let g = IdealGas::air();
    let (jp, jm) = riemann_invariants(&g, f(10.0), f(340.0));
    close(jp, 10.0 + 2.0 * 340.0 / 0.4, 1e-12, "J+");
    close(jm, 10.0 - 2.0 * 340.0 / 0.4, 1e-12, "J-");
    // J+ + J- = 2u , J+ - J- = 4a/(g-1)
    close(jp + jm, 20.0, 1e-12, "sum");
    close(jp - jm, 4.0 * 340.0 / 0.4, 1e-12, "diff");
    let h = IdealGas::helium();
    let (jp, jm) = riemann_invariants(&h, f(-5.0), f(1000.0));
    close(jp, -5.0 + 2.0 * 1000.0 / (2.0 / 3.0), 1e-12, "He J+");
    close(jm, -5.0 - 2.0 * 1000.0 / (2.0 / 3.0), 1e-12, "He J-");
}

/// gamma = 1: isothermal limit, 2a/(gamma-1) is infinite; returning (u, u) hides that.
#[test]
// AUD-A-S1W5-007
fn gamma_one_riemann_invariants_are_not_a_silent_identity() {
    let g = IdealGas {
        gas_constant: f(287.0),
        gamma: Fix128::ONE,
    };
    let (jp, jm) = riemann_invariants(&g, f(10.0), f(340.0));
    assert!(jp != jm, "J+ == J- == {}", jp.to_f64());
    let inf = Fix128::from_int(i64::MAX >> 8);
    assert_eq!((jp, jm), (f(10.0) + inf, f(10.0) - inf));
    // no sound speed: no acoustic term
    assert_eq!(
        riemann_invariants(&g, f(10.0), Fix128::ZERO),
        (f(10.0), f(10.0))
    );
}

/// gamma = 1 is the isothermal limit of the isentropic stagnation relation:
/// the exponent `gamma/(gamma-1)` diverges but the limit itself is finite,
/// `lim (1 + (gamma-1)/2 M^2)^(gamma/(gamma-1)) = exp(M^2/2)`.
#[test]
fn gamma_one_stagnation_pressure_ratio_is_the_isothermal_limit() {
    let g = IdealGas {
        gas_constant: f(287.0),
        gamma: Fix128::ONE,
    };
    // isothermal: p0/p = exp(M^2/2) at M=2 = e^2 = 7.389
    close(
        stagnation_pressure_ratio(&g, f(2.0)),
        2.0f64.exp(),
        1e-3,
        "isothermal p0/p",
    );
}

/// p0/p is monotone in |M| and >= 1. Past the Fix128 range (~9.2e18) it must saturate,
/// not wrap: M=1e4 (air) is 3.6e25, which wraps to 1.7e18 (< the M=1e3 value 3.6e18); M=1e5 is negative.
#[test]
// AUD-A-S1W5-008
fn stagnation_pressure_ratio_is_monotone_and_positive_at_large_mach() {
    let g = IdealGas::air();
    let mut prev = 1.0;
    for m in [1e2, 1e3, 1e4, 1e5] {
        let r = stagnation_pressure_ratio(&g, f(m)).to_f64();
        assert!(r >= prev && r >= 1.0, "M={m:e}: p0/p={r:e} after {prev:e}");
        prev = r;
    }
    // below the range the exact power is kept: M = 100, (1 + 0.2e4)^3.5
    let r = stagnation_pressure_ratio(&g, f(100.0)).to_f64();
    let want = 2001.0f64.powf(3.5);
    assert!((r - want).abs() / want < 1e-6, "{r} vs {want}");
}

/// The normal shock past M = 1e9 (where M^2 terms would wrap): the strong-shock
/// limit, rho2/rho1 = (g+1)/(g-1) = 6 and M2 = sqrt((g-1)/(2g)) for air, with the
/// pressure and temperature ratios saturated, never negative
#[test]
fn normal_shock_at_extreme_mach_is_the_strong_shock_limit() {
    let g = IdealGas::air();
    for m in [3e9, 1e12] {
        let j = normal_shock_jump(&g, f(m));
        assert!((j.density_ratio.to_f64() - 6.0).abs() < 1e-12, "M={m:e}");
        assert!((j.mach_downstream.to_f64() - (0.4f64 / 2.8).sqrt()).abs() < 1e-12);
        assert!(j.pressure_ratio.to_f64() > 1e18 && j.temperature_ratio.to_f64() > 1e17);
    }
    // just below the limit the formulas still apply and approach the same values
    let j = normal_shock_jump(&g, f(1e8));
    assert!((j.density_ratio.to_f64() - 6.0).abs() < 1e-9);
    assert!(j.pressure_ratio.to_f64() > 1e16);
}

fn gas_with_gamma(gamma: Fix128) -> IdealGas {
    IdealGas {
        gas_constant: f(287.0),
        gamma,
    }
}

/// `T0/T = 1 + 0.2 M^2` for air leaves Fix128 near M = 6.8e9: past it both T0/T
/// and p0/p saturate at the largest Fix128 instead of wrapping (M^2 wrapped
/// from about 3e9 and p0/p fell back to 1.0 at M = 7e9, 3e10, 3e15)
#[test]
fn stagnation_ratios_saturate_past_the_fix128_range() {
    let g = IdealGas::air();
    let mut prev = 0.0;
    for m in [1e9, 3e9, 5e9, 7e9, 1e10, 3e10, 1e12, 3e15] {
        let p = stagnation_pressure_ratio(&g, f(m)).to_f64();
        assert!(
            p >= prev && p > 9.2e18,
            "M={m:e}: p0/p={p:e} after {prev:e}"
        );
        prev = p;
    }
    // T0/T itself: exact while 0.2 M^2 fits (M = 3e9: 1.8e18), saturated after
    let t = stagnation_temp_ratio(&g, f(3e9)).to_f64();
    assert!((t - (1.0 + 0.2 * 9e18)).abs() / t < 1e-12, "{t:e}");
    for m in [7e9, 3e10, 3e15] {
        assert!(stagnation_temp_ratio(&g, f(m)).to_f64() > 9.2e18, "M={m:e}");
    }
}

/// The saturation threshold is ln(largest Fix128) = 63 ln 2 = 43.668, not 43:
/// air at M = 1118 has ln(p0/p) = 3.5 ln(1 + 0.2 * 1118^2) = 43.50, which fits
/// (p0/p = 7.8e18) and must be the exact power, not the saturated value
#[test]
fn stagnation_pressure_ratio_is_exact_just_below_the_range() {
    let g = IdealGas::air();
    let base = 1.0 + 0.2 * 1118.0_f64 * 1118.0;
    let want = (3.5 * base.ln()).exp();
    assert!(want < 9.2e18 && 3.5 * base.ln() > 43.0);
    let got = stagnation_pressure_ratio(&g, f(1118.0)).to_f64();
    assert!((got - want).abs() / want < 1e-6, "{got:e} vs {want:e}");
    // isothermal: exp(M^2 / 2) at M = 9 is exp(40.5) = 3.9e17 (fits), M = 10
    // (exp 50) saturates
    let iso = gas_with_gamma(Fix128::ONE);
    let got = stagnation_pressure_ratio(&iso, f(9.0)).to_f64();
    assert!((got - 40.5_f64.exp()).abs() / got < 1e-6, "{got:e}");
    assert!(stagnation_pressure_ratio(&iso, f(10.0)).to_f64() > 9.2e18);
}

/// gamma = 1 (isothermal) normal shock: rho2/rho1 = p2/p1 = M^2, T2/T1 = 1,
/// M2 = 1/M. At M = 2e9 (M^2 = 4e18 fits) the formulas give M^2 exactly; past
/// the range of the formulas' M^2 products rho and p are M^2 while it fits and
/// then saturate (they were wrapped negative: -2.2e17 at 3e9)
#[test]
fn isothermal_normal_shock_past_the_range_saturates() {
    let iso = gas_with_gamma(Fix128::ONE);
    let j = normal_shock_jump(&iso, f(2e9));
    assert!((j.density_ratio.to_f64() - 4e18).abs() / 4e18 < 1e-12);
    assert!((j.pressure_ratio.to_f64() - 4e18).abs() / 4e18 < 1e-12);
    for m in [3e9, 3.1e9, 1e10] {
        let j = normal_shock_jump(&iso, f(m));
        // min(M^2, largest Fix128 = 9.22e18): 9e18 still fits at M = 3e9
        let want = (m * m).min(9.223_372_036_854_776e18);
        let rho = j.density_ratio.to_f64();
        let p = j.pressure_ratio.to_f64();
        assert!((rho - want).abs() / want < 1e-12, "M={m:e} rho {rho:e}");
        assert!((p - want).abs() / want < 1e-12, "M={m:e} p {p:e}");
        assert_eq!(j.temperature_ratio, Fix128::ONE, "M={m:e} T");
        let m2 = j.mach_downstream.to_f64();
        assert!((m2 * m - 1.0).abs() < 1e-9, "M={m:e}: M2={m2:e}");
    }
}

/// A gamma above 4.6 overflowed 2 gamma M^2 below the old fixed limit M = 1e9
/// (gamma = 6, M = 9e8 gave M2 = 0): M2 = sqrt((5 + 2/M^2) / (12 - 5/M^2)) =
/// 0.6455 and rho2/rho1 = 7/5
#[test]
fn normal_shock_with_a_large_gamma_does_not_wrap() {
    // air at M = 3e9: (g+1) M^2 does not fit, so the divided-through form runs;
    // T2/T1 = (2.8 M^2 - 0.4)(0.4 M^2 + 2) / (5.76 M^2) = 1.75e18 still fits
    let j = normal_shock_jump(&IdealGas::air(), f(3e9));
    let m_sq = 9e18_f64;
    let t_want = (2.8 * m_sq - 0.4) * (0.4 * m_sq + 2.0) / (5.76 * m_sq);
    let t = j.temperature_ratio.to_f64();
    assert!((t - t_want).abs() / t_want < 1e-9, "T {t:e} vs {t_want:e}");
    let p_want = 1.0 + 2.8 / 2.4 * (m_sq - 1.0);
    assert!(
        p_want > 9.3e18 && j.pressure_ratio.to_f64() > 9.2e18,
        "p saturates"
    );

    let g = gas_with_gamma(Fix128::from_int(6));
    for m in [9e8, 2e9, 1e12] {
        let j = normal_shock_jump(&g, f(m));
        assert!(
            (j.mach_downstream.to_f64() - (5.0_f64 / 12.0).sqrt()).abs() < 1e-9,
            "M={m:e}"
        );
        assert!((j.density_ratio.to_f64() - 1.4).abs() < 1e-9, "M={m:e}");
        assert!(j.pressure_ratio.to_f64() > 1e17 && j.temperature_ratio.to_f64() > 1e17);
    }
}

/// T0/T stays exact while `1 + 0.2 M^2` fits even where `M^2` alone does not
/// (air, M = 4e9: 3.2e18, M = 5e9: 5e18, with M^2 = 1.6e19 / 2.5e19), and
/// saturates from M = 6.8e9. p0/p saturates just past ln(MAX) = 43.67: at
/// M = 1200 its ln is 3.5 ln(1 + 0.2 * 1200^2) = 44.0. A large gamma at the
/// edge (gamma = 5, M = 2^31: 2 * 2^62 = 2^63) saturates instead of wrapping.
#[test]
fn stagnation_ratios_at_the_edges_of_the_range() {
    let g = IdealGas::air();
    for m in [4e9_f64, 5e9, 6.7e9] {
        let t = stagnation_temp_ratio(&g, f(m)).to_f64();
        let want = 1.0 + 0.2 * m * m;
        assert!(
            (t - want).abs() / want < 1e-12,
            "M={m:e}: {t:e} vs {want:e}"
        );
    }
    assert!(stagnation_pressure_ratio(&g, f(1200.0)).to_f64() > 9.2e18);
    let g5 = gas_with_gamma(Fix128::from_int(5));
    let t = stagnation_temp_ratio(&g5, Fix128::from_int(1 << 31)).to_f64();
    assert!(t > 9.2e18, "{t:e}");
}
