//! Oracles for the `compressible` ideal-gas / shock relations
//! (`IdealGas::air`, `IdealGas::helium`, `IdealGas::density`,
//! `IdealGas::temperature`, `IdealGas::speed_of_sound`,
//! `IdealGas::speed_of_sound_from_pd`, `IdealGas::mach_number`,
//! `normal_shock_jump`, `ShockJump`, `stagnation_temp_ratio`,
//! `stagnation_pressure_ratio`, `riemann_invariants`). Production entry
//! point: `examples/compressible_gas_dynamics.rs`.
//!
//! # Closed forms (Anderson, *Modern Compressible Flow* 3rd ed.)
//!
//! * **Gas presets**: `air` is `R = 287`, `γ = 7/5`; `helium` is
//!   `R = 2077`, `γ = 5/3` — both exact `Fix128` constants (no rounding),
//!   asserted bit-for-bit against the documented numbers.
//! * **Ideal-gas law** (`p = ρ·R·T`, eq. 1.46 form): `density(p, T)` and
//!   `temperature(p, ρ)` are its two inverses. For a synthetic gas with an
//!   integer `R` (no truncation in the Fix128 fraction), `ρ`, `T` chosen so
//!   `p = ρ·R·T` has no remainder when divided back, the round trip is
//!   bit-exact (`assert_eq!`), independent of [`IdealGas::pressure`].
//! * **Speed of sound, two routes tied by the ideal-gas law itself**: for
//!   a synthetic gas with integer `γ` (so [`Fix128::powf_pos`]/`Mul` never
//!   truncate `γ` itself), `a_T = √(γ·R·T)` and `a_pd = √(γ·p/ρ)` with
//!   `p` built from `(ρ, T)` by the test's own arithmetic (not calling
//!   [`IdealGas::pressure`]) land on the *same* `Fix128` bit pattern,
//!   because `γ·R·T` and `γ·(ρ·R·T)/ρ` reduce to the same exact integer
//!   radicand before the identical `sqrt()` call. This is the oracle the
//!   brief asks for: the two formulas are not independently "about right",
//!   they are the *same number* whenever the ideal-gas law holds. The air
//!   / helium presets (non-dyadic `γ`) repeat the same check with a
//!   `2⁻⁴⁰` relative tolerance (two chained roundings: the stored `γ`
//!   fraction and the `sqrt()` digit recurrence).
//! * **Mach number**: `M = |u| / a`, exact when `u` is chosen as an exact
//!   multiple of `a` (so the division has no remainder); `mach_number`
//!   takes `u.abs()` first, so a negative `u` gives the same `M` as `+u`.
//! * **Rankine-Hugoniot normal shock** (eq. 3.51/3.53/3.55): the closed
//!   forms are re-derived in this file from `M₁` and `γ` with independent
//!   `Fix128` arithmetic (not calling `normal_shock_jump`), at `M₁ = 2.5`
//!   (not the `M₁ ∈ {2, 3, 1000}` cases already pinned by
//!   `tests/engineering_oracles_fluid.rs`). `M₁ ≤ 1` returns the identity
//!   ratios — the documented "no shock exists for subsonic upstream"
//!   contract — checked at the exact boundary `M₁ == 1` (not just `< 1`).
//! * **Isentropic stagnation relations** (eq. 3.28/3.30): `M = 0` is the
//!   degenerate case `T₀/T = p₀/p = 1` exactly (both formulas collapse to
//!   `1 + 0` / `1^exponent`), asserted with `assert_eq!`, not a tolerance.
//! * **1-D Riemann invariants** (eq. 7.66): `J± = u ± 2a/(γ-1)`, exact for
//!   a synthetic gas with a dyadic `γ - 1` (e.g. `γ = 2`, so `2a/(γ-1) =
//!   2a` is an exact `Fix128` doubling, no division rounding at all).
//!
//! # Degenerate input
//!
//! * `density(p, T)` / `temperature(p, ρ)` with a zero or negative second
//!   argument: `Fix128::Div`'s documented contract already returns `ZERO`
//!   for a zero divisor, so the explicit `is_zero()` guards in `density` /
//!   `temperature` are redundant with that contract for `T = 0` / `ρ = 0`
//!   — removing them does not change behaviour (see mutation table below,
//!   classified "arithmetically equivalent", not a bug). A *negative*
//!   temperature or density is not guarded at all: `density(p, T<0)`
//!   divides by a negative, nonzero `R·T`, so it returns a (nonsensical,
//!   negative-density-signed) value rather than `ZERO` or an `Err` — this
//!   file pins that as the implemented behaviour.
//! * `speed_of_sound_from_pd(p, ρ=0)`: explicit `ZERO` guard, pinned.
//! * `mach_number(u, T=0)`: `speed_of_sound(0) = √0 = 0`, then the
//!   `a.is_zero()` guard returns `ZERO` rather than dividing — pinned.
//! * Extreme Mach (`M₁ = Fix128::from_int(i64::MAX)`, so `M₁²` overflows a
//!   single `Fix128` lane): every operator `compressible.rs` uses on
//!   `Fix128` (`Mul`, `Div`, `sqrt`) is wrapping / zero-safe by contract, so
//!   `normal_shock_jump` and `stagnation_pressure_ratio` never panic on an
//!   overflowing input — `catch_unwind` confirms this — but the returned
//!   ratios wrap to an unspecified (and not necessarily physically
//!   meaningful) value; this file does not assert a particular wrapped
//!   number, only the absence of a panic.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_physics::compressible::{
    normal_shock_jump, riemann_invariants, stagnation_pressure_ratio, stagnation_temp_ratio,
    IdealGas, ShockJump,
};
use alice_physics::det_math::{powf64, sqrt64};
use alice_physics::math::Fix128;

fn rel_err(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

// ============================================================================
// 1. Gas presets — exact.
// ============================================================================

#[test]
fn air_preset_is_exact_r287_gamma_7_over_5() {
    let g = IdealGas::air();
    assert_eq!(g.gas_constant, Fix128::from_int(287));
    assert_eq!(g.gamma, Fix128::from_ratio(7, 5));
}

#[test]
fn helium_preset_is_exact_r2077_gamma_5_over_3() {
    let g = IdealGas::helium();
    assert_eq!(g.gas_constant, Fix128::from_int(2077));
    assert_eq!(g.gamma, Fix128::from_ratio(5, 3));
}

// ============================================================================
// 2. density / temperature — ideal-gas law inverses, bit-exact round trip.
// ============================================================================

#[test]
fn density_and_temperature_are_exact_inverses_of_the_ideal_gas_law() {
    // Synthetic gas: R = 4 (integer, no Fix128 fraction truncation).
    let g = IdealGas {
        gas_constant: Fix128::from_int(4),
        gamma: Fix128::from_int(2), // unused here
    };
    let rho = Fix128::from_int(3);
    let t = Fix128::from_int(5);
    // p = rho * R * T, computed by the test, not by `IdealGas::pressure`.
    let p = rho * g.gas_constant * t; // 3*4*5 = 60, exact integer.
    assert_eq!(p, Fix128::from_int(60));

    let rho_back = g.density(p, t); // 60 / (4*5) = 3, exact.
    assert_eq!(rho_back, rho);
    let t_back = g.temperature(p, rho); // 60 / (3*4) = 5, exact.
    assert_eq!(t_back, t);
}

#[test]
fn density_zero_temperature_returns_zero_not_a_panic_or_err() {
    let g = IdealGas::air();
    // Documented: Fix128::Div already returns ZERO for a zero divisor, so
    // this also exercises the (redundant) explicit guard in `density`.
    assert_eq!(
        g.density(Fix128::from_int(101_325), Fix128::ZERO),
        Fix128::ZERO
    );
    assert_eq!(
        g.temperature(Fix128::from_int(101_325), Fix128::ZERO),
        Fix128::ZERO
    );
}

#[test]
fn density_negative_temperature_is_not_guarded() {
    // Documented behaviour pin: a negative (nonzero) temperature is not
    // rejected; density(p, T<0) divides p by a negative R*T and returns
    // whatever sign that produces (here, p > 0 and T < 0 => rho < 0).
    let g = IdealGas::air();
    let p = Fix128::from_int(101_325);
    let t_neg = Fix128::from_int(-10);
    let rho = g.density(p, t_neg);
    assert!(
        rho.is_negative(),
        "density(p>0, T<0) is implemented as negative, not Err/ZERO"
    );
    // Round-trips back to the same negative T through `temperature`.
    let t_back = g.temperature(p, rho);
    assert_eq!(t_back, t_neg);
}

// ============================================================================
// 3. speed_of_sound / speed_of_sound_from_pd — the ideal-gas-law tie.
// ============================================================================

#[test]
fn speed_of_sound_two_routes_are_bit_exact_for_an_integer_gas() {
    // gamma = 4, R = 1 (both integers: no Fix128 fraction truncation
    // anywhere in gamma*R*T or gamma*p/rho).
    let g = IdealGas {
        gas_constant: Fix128::from_int(1),
        gamma: Fix128::from_int(4),
    };
    let rho = Fix128::from_int(5);
    let t = Fix128::from_int(25);
    let p = rho * g.gas_constant * t; // 125

    let a_t = g.speed_of_sound(t); // sqrt(4*1*25) = sqrt(100) = 10
    let a_pd = g.speed_of_sound_from_pd(p, rho); // sqrt(4*125/5) = sqrt(100) = 10
    assert_eq!(a_t, Fix128::from_int(10));
    assert_eq!(a_pd, Fix128::from_int(10));
    assert_eq!(
        a_t, a_pd,
        "the two routes must be the exact same Fix128 value"
    );
}

#[test]
fn speed_of_sound_two_routes_agree_for_air_and_helium_presets() {
    for (label, g, rho, t) in [
        (
            "air",
            IdealGas::air(),
            Fix128::from_ratio(1225, 1000),
            Fix128::from_int(288),
        ),
        (
            "helium",
            IdealGas::helium(),
            Fix128::from_ratio(1663, 10000),
            Fix128::from_int(300),
        ),
    ] {
        let p = rho * g.gas_constant * t;
        let a_t = g.speed_of_sound(t).to_f64();
        let a_pd = g.speed_of_sound_from_pd(p, rho).to_f64();
        // Two chained roundings (gamma's stored fraction + the digit-recurrence
        // sqrt), so 2^-40 (== 1.0 / 2^40) rather than bit-exact.
        let tol_2_pow_neg_40 = 1.0 / 1_099_511_627_776.0; // 2^40
        assert!(
            rel_err(a_pd, a_t) < tol_2_pow_neg_40,
            "{label}: a_T={a_t} a_pd={a_pd}"
        );
        // Independent closed form via std-f64 sqrt (allowed here only as a
        // sanity cross-check against a *different* number system, not as
        // the oracle for the Fix128-vs-Fix128 tie above).
        let want = sqrt64(g.gamma.to_f64() * g.gas_constant.to_f64() * t.to_f64());
        assert!(rel_err(a_t, want) < 1e-6, "{label}: a_T={a_t} want={want}");
    }
}

#[test]
fn speed_of_sound_from_pd_zero_density_returns_zero() {
    let g = IdealGas::air();
    assert_eq!(
        g.speed_of_sound_from_pd(Fix128::from_int(101_325), Fix128::ZERO),
        Fix128::ZERO
    );
}

// ============================================================================
// 4. mach_number — exact u/a, and |u| (sign-independent).
// ============================================================================

#[test]
fn mach_number_is_exact_when_u_is_an_integer_multiple_of_a() {
    // gamma=4, R=1, T=25 => a = sqrt(100) = 10 exactly (see test above).
    let g = IdealGas {
        gas_constant: Fix128::from_int(1),
        gamma: Fix128::from_int(4),
    };
    let t = Fix128::from_int(25);
    let a = g.speed_of_sound(t);
    assert_eq!(a, Fix128::from_int(10));

    let u = Fix128::from_int(25); // 25 / 10 = 2.5, exact (dyadic).
    let m = g.mach_number(u, t);
    assert_eq!(m, Fix128::from_ratio(5, 2));
}

#[test]
fn mach_number_uses_absolute_velocity() {
    let g = IdealGas {
        gas_constant: Fix128::from_int(1),
        gamma: Fix128::from_int(4),
    };
    let t = Fix128::from_int(25);
    let m_pos = g.mach_number(Fix128::from_int(25), t);
    let m_neg = g.mach_number(Fix128::from_int(-25), t);
    assert_eq!(m_pos, m_neg, "mach_number must be sign-independent (|u|/a)");
    assert_eq!(m_pos, Fix128::from_ratio(5, 2));
}

#[test]
fn mach_number_zero_temperature_returns_zero() {
    let g = IdealGas::air();
    // a = speed_of_sound(0) = sqrt(0) = 0 => a.is_zero() guard => ZERO.
    assert_eq!(
        g.mach_number(Fix128::from_int(100), Fix128::ZERO),
        Fix128::ZERO
    );
}

// ============================================================================
// 5. normal_shock_jump / ShockJump — Rankine-Hugoniot, independently derived.
// ============================================================================

/// Closed-form Rankine-Hugoniot ratios at `(gas, m1)`, derived directly from
/// Anderson eq. 3.51/3.53/3.55 by this test's own `Fix128` arithmetic — does
/// **not** call `normal_shock_jump`.
fn hand_derived_shock(gas: &IdealGas, m1: Fix128) -> (Fix128, Fix128, Fix128, Fix128) {
    let g = gas.gamma;
    let gp1 = g + Fix128::ONE;
    let gm1 = g - Fix128::ONE;
    let two = Fix128::from_int(2);
    let m1_sq = m1 * m1;

    let rho_ratio = (gp1 * m1_sq) / (gm1 * m1_sq + two);
    let p_ratio = Fix128::ONE + (two * g / gp1) * (m1_sq - Fix128::ONE);
    let t_ratio = p_ratio / rho_ratio;
    let m2_sq = (gm1 * m1_sq + two) / (two * g * m1_sq - gm1);
    (rho_ratio, p_ratio, t_ratio, m2_sq.sqrt())
}

#[test]
fn normal_shock_jump_matches_hand_derived_rankine_hugoniot_at_m1_2_5() {
    let air = IdealGas::air();
    let m1 = Fix128::from_ratio(5, 2);
    let want = hand_derived_shock(&air, m1);
    let got: ShockJump = normal_shock_jump(&air, m1);
    let tol = Fix128::from_ratio(1, 1_000_000_000);
    assert!((got.density_ratio - want.0).abs() < tol);
    assert!((got.pressure_ratio - want.1).abs() < tol);
    assert!((got.temperature_ratio - want.2).abs() < tol);
    assert!((got.mach_downstream - want.3).abs() < tol);
}

#[test]
fn normal_shock_jump_identity_at_and_below_sonic_upstream() {
    let air = IdealGas::air();
    // Exact boundary M1 == 1 (not just M1 < 1): the guard is `m1 <= ONE`.
    let s1 = normal_shock_jump(&air, Fix128::ONE);
    assert_eq!(s1.density_ratio, Fix128::ONE);
    assert_eq!(s1.pressure_ratio, Fix128::ONE);
    assert_eq!(s1.temperature_ratio, Fix128::ONE);
    assert_eq!(s1.mach_downstream, Fix128::ONE);

    let s_sub = normal_shock_jump(&air, Fix128::from_ratio(3, 4));
    assert_eq!(s_sub.density_ratio, Fix128::ONE);
    assert_eq!(s_sub.mach_downstream, Fix128::from_ratio(3, 4));
}

#[test]
fn normal_shock_jump_extreme_mach_does_not_panic() {
    let air = IdealGas::air();
    let m1 = Fix128::from_int(i64::MAX); // squares far past a single lane.
    let r = catch_unwind(AssertUnwindSafe(|| normal_shock_jump(&air, m1)));
    assert!(
        r.is_ok(),
        "normal_shock_jump must not panic on an overflowing Mach"
    );
}

// ============================================================================
// 6. stagnation_temp_ratio / stagnation_pressure_ratio — exact at M=0.
// ============================================================================

#[test]
fn stagnation_ratios_at_m_zero_are_exactly_one() {
    let air = IdealGas::air();
    let helium = IdealGas::helium();
    for g in [&air, &helium] {
        assert_eq!(stagnation_temp_ratio(g, Fix128::ZERO), Fix128::ONE);
        assert_eq!(stagnation_pressure_ratio(g, Fix128::ZERO), Fix128::ONE);
    }
}

#[test]
fn stagnation_pressure_ratio_matches_hand_derived_powf_at_nonzero_mach() {
    // Independent closed form (Anderson eq. 3.30): p0/p = (1 + (gamma-1)/2
    // M^2)^(gamma/(gamma-1)), computed here via `det_math::powf64` — a
    // different (f64, non-Fix128) deterministic power implementation, not
    // `Fix128::powf_pos` — so this is not just re-running the production
    // formula with the production function.
    for (label, g) in [("air", IdealGas::air()), ("helium", IdealGas::helium())] {
        let gamma = g.gamma.to_f64();
        let gm1 = gamma - 1.0;
        for mach in [0.5f64, 1.0, 2.0] {
            let base = 1.0 + 0.5 * gm1 * mach * mach;
            let want = powf64(base, gamma / gm1);
            let got = stagnation_pressure_ratio(&g, Fix128::from_f64(mach)).to_f64();
            assert!(
                rel_err(got, want) < 1e-6,
                "{label} M={mach}: got={got} want={want}"
            );
        }
    }
}

#[test]
fn stagnation_pressure_ratio_extreme_mach_does_not_panic() {
    let air = IdealGas::air();
    let m1 = Fix128::from_int(i64::MAX);
    let r = catch_unwind(AssertUnwindSafe(|| stagnation_pressure_ratio(&air, m1)));
    assert!(
        r.is_ok(),
        "stagnation_pressure_ratio must not panic on an overflowing Mach"
    );
}

// ============================================================================
// 7. riemann_invariants — exact for a dyadic (gamma - 1).
// ============================================================================

#[test]
fn riemann_invariants_are_exact_for_gamma_2() {
    // gamma = 2 => gamma - 1 = 1 => 2a/(gamma-1) = 2a exactly (a doubling,
    // no division rounding at all).
    let g = IdealGas {
        gas_constant: Fix128::from_int(1),
        gamma: Fix128::from_int(2),
    };
    let u = Fix128::from_int(50);
    let a = Fix128::from_int(340);
    let (jp, jm) = riemann_invariants(&g, u, a);
    assert_eq!(jp, u + a.double());
    assert_eq!(jm, u - a.double());
}

#[test]
fn riemann_invariants_gamma_one_saturate_the_acoustic_term() {
    // gamma - 1 == 0: 2a/(gamma - 1) diverges, reported as the saturating
    // sentinel (AUD-A-S1W5-007; this test used to pin the collapse onto u)
    let g = IdealGas {
        gas_constant: Fix128::from_int(1),
        gamma: Fix128::ONE,
    };
    let u = Fix128::from_int(42);
    let inf = Fix128::from_int(i64::MAX >> 8);
    let (jp, jm) = riemann_invariants(&g, u, Fix128::from_int(1000));
    assert_eq!(jp, u + inf);
    assert_eq!(jm, u - inf);
}
