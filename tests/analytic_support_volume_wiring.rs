//! Oracles for `support_volume::{SupportConfig::quality, SupportVolumeReport::filament_length_m,
//! SupportVolumeReport::is_nontrivial}` (`examples/support_volume_presets.rs`) and the estimate they
//! sit on.
//!
//! Hand-derived (module doc), for a region of area `A` and height `H` and a column tall enough for
//! the dense layers (`H >= (base_layers + interface_layers) * layer_h`):
//! `base = A * base_layers * layer_h * base_density`, `interface = A * interface_layers * layer_h * interface_density`,
//! `bulk = A * (H - dense) * bulk_density`; filament length `= V / (pi (d/2)^2)`; time `= V / throughput`.
//! The quality preset is `bulk 0.10, 4 interface layers, layer 0.1 mm, 600 mm^3/min` on top of the defaults.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::support_volume::{
    estimate_region_volume, estimate_support_volume, OverhangRegion, SupportConfig,
    SupportVolumeReport,
};

fn region(area: f64, h: f64) -> OverhangRegion {
    OverhangRegion {
        projected_area_mm2: Fix128::from_f64(area),
        support_height_mm: Fix128::from_f64(h),
    }
}
fn near(a: Fix128, want: f64) {
    assert!(
        (a.to_f64() - want).abs() < 1e-9 * want.abs().max(1.0),
        "{} vs {want}",
        a.to_f64()
    );
}

#[test]
fn quality_preset_values() {
    let q = SupportConfig::quality();
    let d = SupportConfig::default();
    assert_eq!(q.bulk_density, Fix128::from_ratio(10, 100));
    assert_eq!(q.interface_layers, 4);
    assert_eq!(q.layer_height_mm, Fix128::from_ratio(1, 10));
    assert_eq!(q.throughput_mm3_per_min, Fix128::from_int(600));
    // untouched fields come from the default
    assert_eq!(q.interface_density, d.interface_density);
    assert_eq!(q.base_density, d.base_density);
    assert_eq!(q.base_layers, d.base_layers);
    assert_eq!(q.filament_diameter_mm, d.filament_diameter_mm);
    // the other presets stay distinct
    assert_ne!(q.layer_height_mm, SupportConfig::speed().layer_height_mm);
}

#[test]
fn quality_estimate_matches_the_hand_derivation() {
    // A = 100, H = 10: base = 100*0.3*0.9 = 27; interface = 100*0.4*1 = 40; bulk = 100*9.3*0.1 = 93
    let r = estimate_support_volume(&[region(100.0, 10.0)], &SupportConfig::quality());
    near(r.base_volume_mm3, 27.0);
    near(r.interface_volume_mm3, 40.0);
    near(r.bulk_volume_mm3, 93.0);
    near(r.total_volume_mm3, 160.0);
    near(
        r.filament_length_mm,
        160.0 / (std::f64::consts::PI * 0.875 * 0.875),
    );
    near(r.estimated_time_min, 160.0 / 600.0);
    // default config for comparison: 54 + 40 + 135 = 229
    let d = estimate_support_volume(&[region(100.0, 10.0)], &SupportConfig::default());
    near(d.total_volume_mm3, 229.0);
    near(d.estimated_time_min, 229.0 / 1440.0);
}

#[test]
fn per_region_volumes_scale_linearly_and_sum_over_regions() {
    let cfg = SupportConfig::quality();
    for (a, h) in [(1.0, 5.0), (25.0, 12.5), (400.0, 3.0), (0.5, 40.0)] {
        let (b, i, k) = estimate_region_volume(&region(a, h), &cfg);
        // columns tall enough for 0.3 + 0.4 mm of dense layers: closed form
        near(b, a * 0.3 * 0.9);
        near(i, a * 0.4);
        near(k, a * (h - 0.7) * 0.1);
        // doubling the area doubles every part
        let (b2, i2, k2) = estimate_region_volume(&region(2.0 * a, h), &cfg);
        near(b2, 2.0 * b.to_f64());
        near(i2, 2.0 * i.to_f64());
        near(k2, 2.0 * k.to_f64());
    }
    let two = estimate_support_volume(&[region(100.0, 10.0), region(50.0, 20.0)], &cfg);
    near(
        two.total_volume_mm3,
        160.0 + (50.0 * 0.27 + 50.0 * 0.4 + 50.0 * 19.3 * 0.1),
    );
    near(two.base_volume_mm3, 27.0 + 50.0 * 0.27);
}

#[test]
fn filament_length_in_metres_is_millimetres_over_a_thousand() {
    for (a, h) in [(100.0, 10.0), (37.5, 22.0), (1.0, 1.0)] {
        let r = estimate_support_volume(&[region(a, h)], &SupportConfig::quality());
        assert_eq!(
            r.filament_length_m(),
            r.filament_length_mm / Fix128::from_int(1000)
        );
        near(
            r.filament_length_m(),
            r.filament_length_mm.to_f64() / 1000.0,
        );
    }
    let r = SupportVolumeReport {
        filament_length_mm: Fix128::from_int(2500),
        ..Default::default()
    };
    assert_eq!(r.filament_length_m(), Fix128::from_ratio(5, 2));
    assert_eq!(
        SupportVolumeReport::default().filament_length_m(),
        Fix128::ZERO
    );
}

#[test]
fn nontrivial_means_any_total_volume() {
    let cfg = SupportConfig::quality();
    assert!(!estimate_support_volume(&[], &cfg).is_nontrivial());
    assert!(!estimate_support_volume(&[region(0.0, 10.0)], &cfg).is_nontrivial());
    assert!(estimate_support_volume(&[region(100.0, 10.0)], &cfg).is_nontrivial());
    // a height of zero still carries base + interface slabs, so it is non-trivial for a positive area
    assert!(estimate_support_volume(&[region(1.0, 0.0)], &cfg).is_nontrivial());
    // only the total decides (the other fields do not)
    let r = SupportVolumeReport {
        base_volume_mm3: Fix128::ONE,
        total_volume_mm3: Fix128::ZERO,
        ..Default::default()
    };
    assert!(!r.is_nontrivial());
    let r = SupportVolumeReport {
        total_volume_mm3: Fix128::from_ratio(1, 1_000_000),
        ..Default::default()
    };
    assert!(r.is_nontrivial());
}

#[test]
fn degenerate_configs_do_not_divide_by_zero() {
    let r = estimate_support_volume(
        &[region(100.0, 10.0)],
        &SupportConfig {
            filament_diameter_mm: Fix128::ZERO,
            throughput_mm3_per_min: Fix128::ZERO,
            ..SupportConfig::quality()
        },
    );
    assert_eq!(r.filament_length_mm, Fix128::ZERO);
    assert_eq!(r.estimated_time_min, Fix128::ZERO);
    near(r.total_volume_mm3, 160.0);
}
