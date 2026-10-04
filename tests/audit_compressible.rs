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
#[ignore = "known defect: AUD-A-S1W5-007: riemann_invariants(gamma=1) silently returns (u,u) (J+ == J-) although 2a/(gamma-1) diverges; likewise stagnation_pressure_ratio(gamma<=1) silently returns 1"]
fn gamma_one_degenerate_is_not_silent_identity() {
    let g = IdealGas {
        gas_constant: f(287.0),
        gamma: Fix128::ONE,
    };
    let (jp, jm) = riemann_invariants(&g, f(10.0), f(340.0));
    assert!(jp != jm, "J+ == J- == {}", jp.to_f64());
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
#[ignore = "known defect: AUD-A-S1W5-008: stagnation_pressure_ratio wraps silently when p0/p exceeds the Fix128 range (M=1e4 -> 1.7e18 < value at M=1e3; M=1e5 -> -8.6e18); normal_shock_jump wraps from M~3e9"]
fn stagnation_pressure_ratio_is_monotone_and_positive_at_large_mach() {
    let g = IdealGas::air();
    let mut prev = 1.0;
    for m in [1e2, 1e3, 1e4, 1e5] {
        let r = stagnation_pressure_ratio(&g, f(m)).to_f64();
        assert!(r >= prev && r >= 1.0, "M={m:e}: p0/p={r:e} after {prev:e}");
        prev = r;
    }
}
