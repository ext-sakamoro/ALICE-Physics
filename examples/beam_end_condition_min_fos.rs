//! `BeamAnalysis::with_end_condition` and `with_min_fos`.
//!
//! A 15x20 mm PLA cantilever (`Z = b h^2 / 6 = 1000 mm^3`) carries 100 N at
//! 250 mm: `sigma = M / Z = 25 MPa`, `FoS = 50 / 25 = 2` exactly.
//!
//! ```text
//! P_cr = pi^2 E I / (K L)^2      E = 3500 MPa, L = 250 mm, I about the weak
//!                                axis = h b^3 / 12 = 5625 mm^4
//!   pin-pin      K = 1    P_cr = 3109 N
//!   fixed-fixed  K = 1/2  P_cr = 4 x pin-pin
//!   cantilever   K = 2    P_cr = 1/4 x pin-pin
//!   fixed-pin    K = 0.7  P_cr = pin-pin / 0.49
//! is_safe = FoS >= min_fos      (2.0 passes at the default 2.0, fails at 2.5)
//! ```
//!
//! ```bash
//! cargo run --example beam_end_condition_min_fos --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{BeamAnalysis, ColumnEndCondition, CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;

fn main() {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(15),
        height_mm: Fix128::from_int(20),
    };
    let load = LoadCase::CantileverEndPoint {
        load_n: Fix128::from_int(100),
        length_mm: Fix128::from_int(250),
    };
    let base = BeamAnalysis::new(section, load, MaterialProperties::pla());
    // buckling about the weak axis: I = h b^3 / 12 = 20 * 15^3 / 12 = 5625 mm^4
    let pin_pin = core::f64::consts::PI.powi(2) * 3500.0 * 5_625.0 / (250.0 * 250.0);

    for (ec, k) in [
        (ColumnEndCondition::PinPin, 1.0),
        (ColumnEndCondition::FixedFixed, 0.5),
        (ColumnEndCondition::Cantilever, 2.0),
        (ColumnEndCondition::FixedPin, 0.7),
    ] {
        let r = base.with_end_condition(ec).analyze();
        let want = pin_pin / (k * k);
        println!(
            "{ec:?}: P_cr {:.1} N (closed form {want:.1} N)",
            r.euler_critical_load_n.to_f64()
        );
        assert!((r.euler_critical_load_n.to_f64() - want).abs() / want < 1e-9);
    }

    for (min, safe) in [(2.0, true), (2.5, false)] {
        let r = base.with_min_fos(Fix128::from_f64(min)).analyze();
        println!(
            "min_fos {min}: FoS {:.3}, is_safe {}",
            r.factor_of_safety.to_f64(),
            r.is_safe
        );
        assert_eq!(r.is_safe, safe);
    }
}
