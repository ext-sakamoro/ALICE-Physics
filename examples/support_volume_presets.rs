//! Support volume with the quality and speed presets and spool-length helpers:
//! `SupportConfig::{quality, speed}`, `SupportVolumeReport::{filament_length_m, is_nontrivial}`.
//!
//! One 10 x 10 mm overhang (area 100 mm^2) standing 10 mm above the plate.
//! Default config: base `100 * 0.6 * 0.9 = 54`, interface `100 * 0.4 * 1 = 40`, bulk
//! `100 * (10 - 1.0) * 0.15 = 135`, total 229 mm^3. Quality preset (layer 0.1, 4 interface
//! layers, bulk 10 %): base `100 * 0.3 * 0.9 = 27`, interface `100 * 0.4 = 40`, bulk
//! `100 * 9.3 * 0.10 = 93`, total 160 mm^3; filament length = V / (pi (1.75/2)^2) = 66.5 mm.
//! Speed preset (layer 0.28, 1 interface layer, bulk 20 %, 2400 mm^3/min): base
//! `100 * 0.84 * 0.9 = 75.6`, interface `100 * 0.28 = 28`, bulk `100 * (10 - 1.12) * 0.20 = 177.6`,
//! total 281.2 mm^3, time `281.2 / 2400` min. Every figure is checked against these closed forms.
//!
//! ```bash
//! cargo run --release --example support_volume_presets --features std
//! ```

use alice_physics::math::Fix128;
use alice_physics::support_volume::{estimate_support_volume, OverhangRegion, SupportConfig};

/// `(base, interface, bulk, throughput)` from the documented model
/// `area * slab * density` with the base / interface slabs `layers * layer_h`.
fn closed_form(layer_h: f64, iface_layers: f64, bulk: f64, rate: f64) -> [f64; 4] {
    let (area, height) = (100.0, 10.0);
    let base_t = 3.0 * layer_h;
    let iface_t = iface_layers * layer_h;
    [
        area * base_t * 0.9,
        area * iface_t,
        area * (height - base_t - iface_t) * bulk,
        rate,
    ]
}

fn main() {
    let region = OverhangRegion {
        projected_area_mm2: Fix128::from_int(100),
        support_height_mm: Fix128::from_int(10),
    };
    for (name, cfg) in [
        ("default", SupportConfig::default()),
        ("quality", SupportConfig::quality()),
        ("speed", SupportConfig::speed()),
    ] {
        let r = estimate_support_volume(&[region], &cfg);
        let [base, iface, bulk, rate] = match name {
            "default" => closed_form(0.2, 2.0, 0.15, 1440.0),
            "quality" => closed_form(0.1, 4.0, 0.10, 600.0),
            _ => closed_form(0.28, 1.0, 0.20, 2400.0),
        };
        let total = base + iface + bulk;
        let filament = total / (std::f64::consts::PI * 0.875 * 0.875);
        let close = |got: Fix128, want: f64| (got.to_f64() - want).abs() <= 1e-9 * want.max(1.0);
        assert!(close(r.base_volume_mm3, base), "{name}: base");
        assert!(close(r.interface_volume_mm3, iface), "{name}: interface");
        assert!(close(r.bulk_volume_mm3, bulk), "{name}: bulk");
        assert!(close(r.total_volume_mm3, total), "{name}: total");
        assert!(close(r.filament_length_mm, filament), "{name}: filament");
        assert!(
            close(r.filament_length_m(), filament / 1000.0),
            "{name}: metres"
        );
        assert!(close(r.estimated_time_min, total / rate), "{name}: time");
        assert!(r.is_nontrivial());
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
    assert!(!none.is_nontrivial());
}
