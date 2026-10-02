//! Compressible gas dynamics — production entry point for every item of
//! `src/compressible.rs` that had zero production callers
//! (`ShockJump`, `IdealGas::air`, `IdealGas::density`, `IdealGas::helium`,
//! `IdealGas::mach_number`, `normal_shock_jump`, `riemann_invariants`,
//! `IdealGas::speed_of_sound`, `IdealGas::speed_of_sound_from_pd`,
//! `stagnation_pressure_ratio`, `stagnation_temp_ratio`,
//! `IdealGas::temperature`).
//!
//! Every section prints the closed-form value (Anderson, *Modern
//! Compressible Flow* 3rd ed.) next to the crate's `Fix128` output so the
//! two can be read side by side.
//!
//! ```bash
//! cargo run --example compressible_gas_dynamics --features std
//! ```
//!
//! # Closed forms
//!
//! * **Ideal-gas law** (Anderson eq. 1.46 form): `p = ρ·R·T`, so
//!   `density(p, T) = p/(R·T)` and `temperature(p, ρ) = p/(ρ·R)` are the two
//!   inverses of the same relation.
//! * **Speed of sound**, two routes tied by the ideal-gas law itself
//!   (Anderson eq. 8.18 / 8.19): `a = √(γ·R·T)` and `a = √(γ·p/ρ)` must
//!   agree whenever `p = ρ·R·T` — this example constructs `p` from `(ρ, T)`
//!   for both presets and prints both routes next to each other.
//! * **Mach number**: `M = |u| / a`.
//! * **Rankine-Hugoniot normal shock** (Anderson eq. 3.51/3.53/3.55):
//!   `ρ₂/ρ₁ = (γ+1)M₁² / ((γ-1)M₁² + 2)`,
//!   `p₂/p₁ = 1 + (2γ/(γ+1))·(M₁² - 1)`, `T₂/T₁ = (p₂/p₁)/(ρ₂/ρ₁)`,
//!   `M₂² = ((γ-1)M₁² + 2) / (2γM₁² - (γ-1))`. Subsonic upstream (`M₁ ≤ 1`)
//!   returns the identity ratios — no stationary normal shock exists there.
//! * **Isentropic stagnation relations** (Anderson eq. 3.28 / 3.30):
//!   `T₀/T = 1 + (γ-1)/2·M²`, `p₀/p = (T₀/T)^(γ/(γ-1))`.
//! * **1-D Riemann invariants** (Anderson eq. 7.66): `J± = u ± 2a/(γ-1)`.
//!
//! Author: Moroya Sakamoto

use alice_physics::compressible::{
    normal_shock_jump, riemann_invariants, stagnation_pressure_ratio, stagnation_temp_ratio,
    IdealGas, ShockJump,
};
use alice_physics::math::Fix128;

/// Build the pressure that the ideal-gas law predicts for a given `(ρ, T)`
/// pair, independently of [`IdealGas::pressure`] (direct arithmetic), so the
/// two speed-of-sound routes below are fed a *consistent* state.
fn ideal_gas_pressure(gas: &IdealGas, density_kg_per_m3: Fix128, temperature_k: Fix128) -> Fix128 {
    density_kg_per_m3 * gas.gas_constant * temperature_k
}

fn run_gas(label: &str, gas: &IdealGas, rho: Fix128, t: Fix128, u: Fix128) {
    println!("[compressible] == {label} ==");
    println!(
        "[compressible]   preset: R={:.4} gamma={:.4}",
        gas.gas_constant.to_f64(),
        gas.gamma.to_f64()
    );

    // IdealGas::density / IdealGas::temperature — the two inverses of p = rho*R*T.
    let p = ideal_gas_pressure(gas, rho, t);
    let rho_back = gas.density(p, t);
    let t_back = gas.temperature(p, rho);
    println!(
        "[compressible]   density(p={:.3}, T={:.3}) = {:.6} (expect rho={:.6})",
        p.to_f64(),
        t.to_f64(),
        rho_back.to_f64(),
        rho.to_f64()
    );
    println!(
        "[compressible]   temperature(p={:.3}, rho={:.3}) = {:.6} (expect T={:.6})",
        p.to_f64(),
        rho.to_f64(),
        t_back.to_f64(),
        t.to_f64()
    );

    // IdealGas::speed_of_sound vs IdealGas::speed_of_sound_from_pd — two
    // routes to the same `a`, tied together by the ideal-gas law above.
    let a_t = gas.speed_of_sound(t);
    let a_pd = gas.speed_of_sound_from_pd(p, rho);
    println!(
        "[compressible]   speed_of_sound(T) = {:.6} m/s, speed_of_sound_from_pd(p, rho) = {:.6} m/s (must agree)",
        a_t.to_f64(),
        a_pd.to_f64()
    );

    // IdealGas::mach_number
    let m = gas.mach_number(u, t);
    println!(
        "[compressible]   mach_number(u={:.3}, T={:.3}) = {:.6} (expect u/a = {:.6})",
        u.to_f64(),
        t.to_f64(),
        m.to_f64(),
        u.to_f64() / a_t.to_f64()
    );

    // riemann_invariants — J+ and J- at this state.
    let (jp, jm) = riemann_invariants(gas, u, a_t);
    println!(
        "[compressible]   riemann_invariants(u={:.3}, a={:.3}) = (J+={:.6}, J-={:.6})",
        u.to_f64(),
        a_t.to_f64(),
        jp.to_f64(),
        jm.to_f64()
    );

    // normal_shock_jump / ShockJump — subsonic (no shock) vs supersonic.
    for m1 in [
        Fix128::from_ratio(1, 2),
        Fix128::from_int(2),
        Fix128::from_int(4),
    ] {
        let shock: ShockJump = normal_shock_jump(gas, m1);
        println!(
            "[compressible]   normal_shock_jump(M1={:.3}) -> rho2/rho1={:.6} p2/p1={:.6} T2/T1={:.6} M2={:.6}",
            m1.to_f64(),
            shock.density_ratio.to_f64(),
            shock.pressure_ratio.to_f64(),
            shock.temperature_ratio.to_f64(),
            shock.mach_downstream.to_f64()
        );
    }

    // stagnation_temp_ratio / stagnation_pressure_ratio.
    for mach in [Fix128::ZERO, Fix128::ONE, Fix128::from_int(2)] {
        let t0_t = stagnation_temp_ratio(gas, mach);
        let p0_p = stagnation_pressure_ratio(gas, mach);
        println!(
            "[compressible]   stagnation(M={:.3}) -> T0/T={:.6} p0/p={:.6}",
            mach.to_f64(),
            t0_t.to_f64(),
            p0_p.to_f64()
        );
    }
}

fn main() {
    let air = IdealGas::air();
    let helium = IdealGas::helium();

    // ISA sea-level-ish state for air, a cold cryogenic state for helium —
    // both gas presets driven through every item in the baseline list.
    run_gas(
        "air",
        &air,
        Fix128::from_ratio(1225, 1000), // rho ~ 1.225 kg/m^3
        Fix128::from_int(288),          // T = 288 K (15 C)
        Fix128::from_int(100),          // u = 100 m/s
    );
    run_gas(
        "helium",
        &helium,
        Fix128::from_ratio(1663, 10000), // rho ~ 0.1663 kg/m^3 at ~300 K, 1 atm
        Fix128::from_int(300),           // T = 300 K
        Fix128::from_int(500),           // u = 500 m/s (helium's a is ~ 1018 m/s)
    );

    println!("[compressible] done");
}
