//! Warp risk of an ABS plate in an open room versus the Bambu-style enclosure:
//! `Footprint::rectangle`, `EnvConditions::enclosed_abs`.
//!
//! Hand check for a 100 x 100 mm ABS part (`alpha = 0.008`, `E = 2.3 GPa`):
//! `raw = 500 * 0.008 * (2.3/5) * (10000/50000) * (100/300) * (dT/200)`
//! * enclosure (print 240, chamber 55, dT 185): raw = 0.1134667 (Low)
//! * open air  (print 240, chamber 20, dT 220): raw = 0.1349333 (Low)
//! and `F/mm = E[MPa] * alpha * dT` = 2300 * 0.008 * dT = 3404 (enclosure), 4048 (open).
//!
//! ```bash
//! cargo run --release --example warp_risk_enclosure --features std
//! ```

use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::warp_risk::{analyze_warp_risk, EnvConditions, Footprint};

fn main() {
    let part = Footprint::rectangle(Fix128::from_int(100), Fix128::from_int(100));
    let abs = MaterialProperties::abs();
    let open = EnvConditions {
        print_temp_c: Fix128::from_int(240),
        chamber_temp_c: Fix128::from_int(20),
        bed_temp_c: Fix128::from_int(90),
    };
    for (label, env) in [
        ("enclosed", EnvConditions::enclosed_abs()),
        ("open air", open),
    ] {
        let r = analyze_warp_risk(&part, &abs, &env);
        println!(
            "{label}: score {:.7} {:?}, F/mm {:.1} N/mm - {}",
            r.score.to_f64(),
            r.category,
            r.warp_force_per_mm.to_f64(),
            r.recommendation
        );
    }
    // Footprint where the enclosure matters: 280 x 250 mm saturates at Critical in the open.
    let big = Footprint::rectangle(Fix128::from_int(280), Fix128::from_int(250));
    let r = analyze_warp_risk(&big, &abs, &EnvConditions::enclosed_abs());
    println!(
        "280x250 enclosed: score {:.3} {:?}",
        r.score.to_f64(),
        r.category
    );
}
