//! Thermal-yield design envelope for a fully constrained part, per material.
//!
//! `yield_temperature_c` gives the temperature at which restrained thermal expansion
//! first reaches the material's yield strength:
//!
//! ```text
//! T_y = T_install + sigma_y / (E alpha c)
//! ```
//!
//! The example prints the envelope for every built-in material at three constraint
//! levels and cross-checks it against `analyze_thermal_stress`: at `T_y` the factor of
//! safety must be exactly 1.
//!
//! ```bash
//! cargo run --release --example thermal_stress_envelope --features std
//! ```

use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::thermal_stress::{analyze_thermal_stress, yield_temperature_c};

fn main() {
    let materials = [
        MaterialProperties::pla(),
        MaterialProperties::petg(),
        MaterialProperties::abs(),
        MaterialProperties::pc(),
        MaterialProperties::nylon(),
        MaterialProperties::peek(),
        MaterialProperties::sus304(),
        MaterialProperties::a5052(),
    ];
    let install = Fix128::from_int(20);
    println!(
        "{:10} {:>12} {:>12} {:>12}",
        "material", "c = 1.0", "c = 0.5", "c = 0.25"
    );
    for m in &materials {
        let mut row = Vec::new();
        for (n, d) in [(1, 1), (1, 2), (1, 4)] {
            let c = Fix128::from_ratio(n, d);
            let ty = yield_temperature_c(m, c, install);
            // at the envelope temperature the factor of safety is exactly 1
            let at = analyze_thermal_stress(m, c, install, ty);
            assert!(
                (at.factor_of_safety.to_f64() - 1.0).abs() < 1e-9,
                "{}",
                m.name
            );
            row.push(ty.to_f64());
        }
        println!(
            "{:10} {:12.1} {:12.1} {:12.1}",
            m.name, row[0], row[1], row[2]
        );
    }
}
