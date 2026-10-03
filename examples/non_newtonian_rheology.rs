//! Driving the non-Newtonian fluid constitutive models end to end
//!
//! `non_newtonian` (power-law / Carreau / Bingham / Herschel-Bulkley) is a
//! pure data-and-formula module: every constructor (`PowerLaw::newtonian`,
//! `PowerLaw::shear_thickening`, `PowerLaw::shear_thinning`) and every
//! evaluator (`PowerLaw::stress`, `PowerLaw::apparent_viscosity`,
//! `Carreau::viscosity`, `Carreau::viscosity_with_index`, `Bingham::stress`,
//! `Bingham::flows_under_stress`, `HerschelBulkley::stress`) had no
//! production caller before this example — `fluid.rs` and `eulerian_grid.rs`
//! (named in the module doc as the intended consumers) still take a single
//! constant `FluidConfig::viscosity`, not one of these shear-rate-dependent
//! laws.
//!
//! This walks every constructor through every evaluator at hand-picked shear
//! rates, printing the production output next to the closed-form value it
//! must equal (Bird-Stewart-Lightfoot *Transport Phenomena* Ch. 8 /
//! Chhabra & Richardson *Non-Newtonian Flow and Applied Rheology* Ch. 1 /
//! Bird, Armstrong & Hassager *Dynamics of Polymeric Liquids* Vol. 1 eq.
//! 4.1-9).
//!
//! ```bash
//! cargo run --example non_newtonian_rheology --features std
//! ```

use alice_physics::math::Fix128;
use alice_physics::non_newtonian::{Bingham, Carreau, HerschelBulkley, PowerLaw};

fn main() {
    // --- PowerLaw::newtonian — degenerate case, constant viscosity --------
    // tau = mu * gamma_dot with mu = 2 Pa.s -> at gamma_dot = 5, tau = 10
    let water = PowerLaw::newtonian(Fix128::from_int(2));
    let tau_newton = water.stress(Fix128::from_int(5));
    println!(
        "[non_newtonian] newtonian stress(5) = {} (want 10)",
        tau_newton.to_f64()
    );
    assert_eq!(tau_newton, Fix128::from_int(10));
    let eta_newton_1 = water.apparent_viscosity(Fix128::from_int(1));
    let eta_newton_100 = water.apparent_viscosity(Fix128::from_int(100));
    println!(
        "[non_newtonian] newtonian apparent_viscosity(1)={} apparent_viscosity(100)={} (both want 2, constant)",
        eta_newton_1.to_f64(),
        eta_newton_100.to_f64()
    );
    assert_eq!(eta_newton_1, Fix128::from_int(2));
    assert_eq!(eta_newton_1, eta_newton_100);

    // --- PowerLaw::shear_thickening — tau = K * gamma_dot^n, n = 2 --------
    // K = 1 -> at gamma_dot = 6, tau = 1 * 36 = 36, eta = tau/gamma_dot = 6
    let cornstarch = PowerLaw::shear_thickening(Fix128::from_int(1), 2);
    let tau_thick = cornstarch.stress(Fix128::from_int(6));
    let eta_thick = cornstarch.apparent_viscosity(Fix128::from_int(6));
    println!(
        "[non_newtonian] shear_thickening stress(6)={} (want 36) apparent_viscosity(6)={} (want 6)",
        tau_thick.to_f64(),
        eta_thick.to_f64()
    );
    assert_eq!(tau_thick, Fix128::from_int(36));
    assert_eq!(eta_thick, Fix128::from_int(6));
    // the preset clamps n_int to >= 2: a caller asking for n=1 still gets
    // the n=2 curve, not the (degenerate, linear) n=1 curve.
    let clamped = PowerLaw::shear_thickening(Fix128::from_int(1), 1);
    assert_eq!(clamped.stress(Fix128::from_int(6)), tau_thick);

    // --- PowerLaw::shear_thinning — tau = K * gamma_dot^(1/n), n = 2 ------
    // K = 2 -> at gamma_dot = 4, tau = 2 * 4^0.5 = 4, eta = tau/gamma_dot = 1
    let mud = PowerLaw::shear_thinning(Fix128::from_int(2), 2);
    let tau_thin = mud.stress(Fix128::from_int(4));
    let eta_thin = mud.apparent_viscosity(Fix128::from_int(4));
    println!(
        "[non_newtonian] shear_thinning stress(4)={} (want 4) apparent_viscosity(4)={} (want 1)",
        tau_thin.to_f64(),
        eta_thin.to_f64()
    );
    assert!((tau_thin.to_f64() - 4.0).abs() < 1e-6);
    assert!((eta_thin.to_f64() - 1.0).abs() < 1e-6);
    // same n_int >= 2 clamp as shear_thickening.
    let clamped_thin = PowerLaw::shear_thinning(Fix128::from_int(2), 1);
    assert!((clamped_thin.stress(Fix128::from_int(4)).to_f64() - tau_thin.to_f64()).abs() < 1e-9);

    // --- Carreau::viscosity (integer half_exponent, n = -1) ---------------
    // eta_0 = 1000, eta_inf = 10, lambda = 0.1 -> at lambda*gamma_dot = 1
    // (gamma_dot = 10) the midpoint is eta_inf + (eta_0-eta_inf)/2 = 505.
    let melt = Carreau {
        eta_zero: Fix128::from_int(1000),
        eta_inf: Fix128::from_int(10),
        lambda: Fix128::from_ratio(1, 10),
        half_exponent: -1,
    };
    let eta_rest = melt.viscosity(Fix128::ZERO);
    let eta_mid = melt.viscosity(Fix128::from_int(10));
    println!(
        "[non_newtonian] carreau viscosity(0)={} (want 1000) viscosity(10)={} (want 505)",
        eta_rest.to_f64(),
        eta_mid.to_f64()
    );
    assert_eq!(eta_rest, Fix128::from_int(1000));
    assert!((eta_mid.to_f64() - 505.0).abs() < 1e-6);

    // --- Carreau::viscosity_with_index (exact fractional n) ---------------
    // n = -1 matches the integer `viscosity` path exactly at every shear
    // rate (the f64-closed-form cross-check for the fractional n = 0.4
    // polymer-melt case already lives in
    // `tests/engineering_oracles_fluid.rs::non_newtonian_carreau_fractional_index_matches_closed_form`,
    // not duplicated here).
    let eta_frac_matches_int =
        melt.viscosity_with_index(Fix128::from_int(10), Fix128::from_int(-1));
    println!(
        "[non_newtonian] carreau viscosity_with_index(10, n=-1)={} viscosity(10)={} (must match)",
        eta_frac_matches_int.to_f64(),
        eta_mid.to_f64()
    );
    assert_eq!(eta_frac_matches_int, eta_mid);

    // --- Bingham::stress / flows_under_stress ------------------------------
    // yield_stress = 10, plastic_viscosity = 1 -> at gamma_dot = 5, tau = 15
    let drilling_mud = Bingham {
        yield_stress: Fix128::from_int(10),
        plastic_viscosity: Fix128::from_int(1),
    };
    let tau_bingham = drilling_mud.stress(Fix128::from_int(5));
    println!(
        "[non_newtonian] bingham stress(5)={} (want 15)",
        tau_bingham.to_f64()
    );
    assert_eq!(tau_bingham, Fix128::from_int(15));
    let flows_below = drilling_mud.flows_under_stress(Fix128::from_int(5));
    let flows_above = drilling_mud.flows_under_stress(Fix128::from_int(20));
    println!(
        "[non_newtonian] bingham flows_under_stress(5)={flows_below} (want false) flows_under_stress(20)={flows_above} (want true)"
    );
    assert!(!flows_below);
    assert!(flows_above);

    // --- HerschelBulkley::stress — yield + power-law hybrid ----------------
    // yield_stress = 5, K = 1, n = 2 -> at gamma_dot = 3, tau = 5 + 1*9 = 14
    let hb = HerschelBulkley {
        yield_stress: Fix128::from_int(5),
        k: Fix128::from_int(1),
        n_int: 2,
    };
    let tau_hb = hb.stress(Fix128::from_int(3));
    println!(
        "[non_newtonian] herschel_bulkley stress(3)={} (want 14)",
        tau_hb.to_f64()
    );
    assert_eq!(tau_hb, Fix128::from_int(14));

    println!("[non_newtonian] done: 10 production entry points exercised");
}
