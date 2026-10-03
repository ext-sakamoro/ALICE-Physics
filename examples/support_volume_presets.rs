//! Support volume with the quality preset and spool-length helpers:
//! `SupportConfig::quality`, `SupportVolumeReport::{filament_length_m, is_nontrivial}`.
//!
//! One 10 x 10 mm overhang (area 100 mm^2) standing 10 mm above the plate.
//! Default config: base `100 * 0.6 * 0.9 = 54`, interface `100 * 0.4 * 1 = 40`, bulk
//! `100 * (10 - 1.0) * 0.15 = 135`, total 229 mm^3. Quality preset (layer 0.1, 4 interface
//! layers, bulk 10 %): base `100 * 0.3 * 0.9 = 27`, interface `100 * 0.4 = 40`, bulk
//! `100 * 9.3 * 0.10 = 93`, total 160 mm^3; filament length = V / (pi (1.75/2)^2) = 66.5 mm.
//!
//! ```bash
//! cargo run --release --example support_volume_presets --features std
//! ```

use alice_physics::math::Fix128;
use alice_physics::support_volume::{estimate_support_volume, OverhangRegion, SupportConfig};

fn main() {
    let region = OverhangRegion {
        projected_area_mm2: Fix128::from_int(100),
        support_height_mm: Fix128::from_int(10),
    };
    for (name, cfg) in [
        ("default", SupportConfig::default()),
        ("quality", SupportConfig::quality()),
    ] {
        let r = estimate_support_volume(&[region], &cfg);
        println!(
            "{name}: base {:.1} interface {:.1} bulk {:.1} total {:.1} mm^3, filament {:.2} mm = {:.5} m, {:.4} min, non-trivial {}",
            r.base_volume_mm3.to_f64(),
            r.interface_volume_mm3.to_f64(),
            r.bulk_volume_mm3.to_f64(),
            r.total_volume_mm3.to_f64(),
            r.filament_length_mm.to_f64(),
            r.filament_length_m().to_f64(),
            r.estimated_time_min.to_f64(),
            r.is_nontrivial()
        );
    }
    let none = estimate_support_volume(&[], &SupportConfig::quality());
    println!("no overhangs: non-trivial {}", none.is_nontrivial());
}
