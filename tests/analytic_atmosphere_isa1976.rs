//! Oracles for `atmosphere::Isa1976` against the published U.S. Standard
//! Atmosphere 1976 table.
//!
//! # Reference values
//!
//! `TABLE` is the 1976 standard's tabulation by **geometric** altitude `Z`
//! (NOAA / NASA / USAF, *U.S. Standard Atmosphere, 1976*, Table I), copied
//! with its printed significant figures. The rows go through
//! `Isa1976::at_geometric_altitude`, so the geometric → geopotential
//! conversion `H = r₀ Z / (r₀ + Z)` is checked by the same rows (at 20 km the
//! two altitudes differ by 63 m, which moves the pressure by 1 %).
//!
//! # Tolerance
//!
//! One unit in the last printed digit of the table entry: the table is
//! rounded from the standard's own intermediate values, so the last digit
//! can differ by one from a fresh evaluation of the same formulas (an f64
//! evaluation gives 12 111.8 Pa at 15 km where the table prints 1.2111E+04).
//! `Fix128::powf_pos` / `exp` add a relative error below `1e-6`, two orders
//! below the smallest bound used here (`1e-4` of the value).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 `exp` computes a closed-form reference outside the crate, not
// simulation state (same convention as the other analytic tests).
#![allow(clippy::disallowed_methods)]

use alice_physics::atmosphere::{AtmosphereState, Isa1976, IsaError};
use alice_physics::buoyancy_zone::ZoneShape;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::wind_zone::WindZone;

/// One table row: geometric altitude (m), then (value, one unit in the last
/// printed digit) for T (K), p (Pa), ρ (kg/m³).
struct Row {
    z_m: f64,
    t: (f64, f64),
    p: (f64, f64),
    rho: (f64, f64),
}

// oracle: U.S. Standard Atmosphere 1976, Table I (geometric altitude).
const TABLE: [Row; 6] = [
    Row {
        z_m: 0.0,
        t: (288.150, 0.001),
        p: (1.013_25e5, 1.0),
        rho: (1.2250, 0.0001),
    },
    Row {
        z_m: 1_000.0,
        t: (281.651, 0.001),
        p: (8.9876e4, 1.0),
        rho: (1.1117, 0.0001),
    },
    Row {
        z_m: 5_000.0,
        t: (255.676, 0.001),
        p: (5.4048e4, 1.0),
        rho: (7.3643e-1, 1e-5),
    },
    Row {
        z_m: 11_000.0,
        t: (216.774, 0.001),
        p: (2.2700e4, 1.0),
        rho: (3.6480e-1, 1e-5),
    },
    Row {
        z_m: 15_000.0,
        t: (216.650, 0.001),
        p: (1.2111e4, 1.0),
        rho: (1.9476e-1, 1e-5),
    },
    Row {
        z_m: 20_000.0,
        t: (216.650, 0.001),
        p: (5.5293e3, 0.1),
        rho: (8.8910e-2, 1e-6),
    },
];

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn assert_within(label: &str, actual: f64, (expected, unit): (f64, f64)) {
    let err = (actual - expected).abs();
    assert!(
        err <= unit,
        "{label}: {actual} vs table {expected} (|diff| {err} > one last-digit unit {unit})"
    );
}

#[test]
fn table_rows_match_us_standard_atmosphere_1976() {
    for row in &TABLE {
        let s = Isa1976::at_geometric_altitude(fx(row.z_m)).expect("in range");
        let z = row.z_m;
        assert_within(&format!("T at {z} m"), s.temperature_k.to_f64(), row.t);
        assert_within(&format!("p at {z} m"), s.pressure_pa.to_f64(), row.p);
        assert_within(&format!("rho at {z} m"), s.density_kg_m3.to_f64(), row.rho);
    }
}

#[test]
fn speed_of_sound_matches_table() {
    // oracle: 1976 Table I, speed of sound column (geometric altitude):
    // 340.294 m/s at 0 m, 295.154 m/s at 11 km, 295.069 m/s at 20 km.
    for (z, a) in [(0.0, 340.294), (11_000.0, 295.154), (20_000.0, 295.069)] {
        let s = Isa1976::at_geometric_altitude(fx(z)).expect("in range");
        assert_within(
            &format!("a at {z} m"),
            s.speed_of_sound_m_s.to_f64(),
            (a, 0.001),
        );
    }
}

#[test]
fn tropopause_base_values() {
    // oracle: 1976 standard, layer-1 base at H = 11 000 m geopotential:
    // T = 216.65 K, P_b = 22 632.06 Pa (the published layer constant).
    let s = Isa1976::at_geopotential_altitude(Fix128::from_int(11_000)).expect("in range");
    assert_within("T_11", s.temperature_k.to_f64(), (216.65, 1e-6));
    assert_within("p_11", s.pressure_pa.to_f64(), (22_632.06, 0.05));
}

#[test]
fn layers_meet_continuously_at_the_tropopause() {
    // T and p are continuous at H = 11 km by construction of the standard;
    // 1 mm on either side may differ by the lapse over 1 mm (6.5e-6 K) and
    // the pressure gradient over 1 mm (about 0.0035 Pa).
    let below = Isa1976::at_geopotential_altitude(fx(10_999.999)).unwrap();
    let above = Isa1976::at_geopotential_altitude(fx(11_000.001)).unwrap();
    assert!((below.temperature_k - above.temperature_k).to_f64().abs() < 1e-5);
    assert!((below.pressure_pa - above.pressure_pa).to_f64().abs() < 0.01);
    assert!((below.density_kg_m3 - above.density_kg_m3).to_f64().abs() < 1e-6);
}

#[test]
fn lapse_rate_is_6_5_kelvin_per_km_in_layer_0_and_zero_in_layer_1() {
    // oracle: L = 0.0065 K/m below 11 km, isothermal above.
    let t = |h: i64| {
        Isa1976::at_geopotential_altitude(Fix128::from_int(h))
            .unwrap()
            .temperature_k
            .to_f64()
    };
    assert!((t(2_000) - t(3_000) - 6.5).abs() < 1e-9);
    assert!((t(12_000) - t(19_000)).abs() < 1e-12);
}

#[test]
fn stratosphere_scale_height_matches_closed_form() {
    // oracle: isothermal layer, p(H₂)/p(H₁) = exp(−g₀ M₀ ΔH / (R* T₁₁)),
    // g₀ M₀ / (R* T₁₁) = 9.80665 · 0.0289644 / (8.31432 · 216.65) per metre.
    let k: f64 = 9.80665 * 0.0289644 / (8.31432 * 216.65);
    let p = |h: i64| {
        Isa1976::at_geopotential_altitude(Fix128::from_int(h))
            .unwrap()
            .pressure_pa
            .to_f64()
    };
    let ratio = p(18_000) / p(12_000);
    let expected = (-k * 6_000.0).exp();
    assert!(
        ((ratio - expected) / expected).abs() < 1e-5,
        "ratio {ratio} vs {expected}"
    );
}

#[test]
fn geopotential_conversion_matches_closed_form() {
    // oracle: H = r₀ Z / (r₀ + Z), r₀ = 6 356 766 m.
    let r0 = 6_356_766.0;
    for z in [0.0, 1_000.0, 11_000.0, 20_000.0] {
        let h = Isa1976::geopotential_altitude_m(fx(z)).to_f64();
        let expected = r0 * z / (r0 + z);
        assert!((h - expected).abs() < 1e-6, "Z {z}: H {h} vs {expected}");
    }
}

#[test]
fn sea_level_density_matches_wind_zone_presets() {
    // oracle: `WindZone`'s presets carry the sea-level ISA density 1.225
    // kg/m³ as a literal (src/wind_zone.rs, field doc "Sea-level ISA =
    // 1.225"); the ISA model at H = 0 must agree within the table's last digit.
    let sea = Isa1976::at_geopotential_altitude(Fix128::ZERO).unwrap();
    let shape = ZoneShape::Aabb {
        min: Vec3Fix::from_int(-1, -1, -1),
        max: Vec3Fix::from_int(1, 1, 1),
    };
    for zone in [WindZone::light_breeze(shape), WindZone::storm(shape)] {
        let diff = (sea.density_kg_m3 - zone.air_density_kg_m3).to_f64().abs();
        assert!(
            diff <= 1e-4,
            "ISA sea level {} vs WindZone {}",
            sea.density_kg_m3.to_f64(),
            zone.air_density_kg_m3.to_f64()
        );
    }
}

#[test]
fn out_of_range_altitudes_are_errors_not_clamped() {
    // Documented: below 0 m or above 20 000 m geopotential is
    // `AltitudeOutOfRange` carrying the rejected altitude.
    for h in [
        fx(-0.001),
        Fix128::from_int(-5_000),
        fx(20_000.001),
        Fix128::from_int(1_000_000),
    ] {
        assert_eq!(
            Isa1976::at_geopotential_altitude(h),
            Err(IsaError::AltitudeOutOfRange {
                geopotential_altitude_m: h
            })
        );
    }
    // geometric 20 063 m is H ≈ 19 999.9 (in range); 20 100 m is H ≈ 20 036.6 (out).
    assert!(Isa1976::at_geometric_altitude(Fix128::from_int(20_063)).is_ok());
    assert!(matches!(
        Isa1976::at_geometric_altitude(Fix128::from_int(20_100)),
        Err(IsaError::AltitudeOutOfRange { .. })
    ));
    // The boundaries themselves are inside.
    let top: AtmosphereState = Isa1976::at_geopotential_altitude(Fix128::from_int(20_000)).unwrap();
    assert!(top.density_kg_m3 > Fix128::ZERO);
    assert!(Isa1976::at_geopotential_altitude(Fix128::ZERO).is_ok());
}

#[test]
fn negative_geometric_altitudes_are_rejected_including_the_pole_of_the_conversion() {
    // Z = −r₀ makes r₀ + Z = 0: the conversion returns ZERO (Fix128 division
    // convention), which would read as sea level. Documented behaviour:
    // `at_geometric_altitude` rejects every Z < 0 itself, so the pole and
    // Z < −r₀ (where the closed form turns positive again) are errors too.
    assert_eq!(
        Isa1976::geopotential_altitude_m(Fix128::from_int(-6_356_766)),
        Fix128::ZERO
    );
    for z in [-1_i64, -6_356_766, -10_000_000] {
        assert!(
            matches!(
                Isa1976::at_geometric_altitude(Fix128::from_int(z)),
                Err(IsaError::AltitudeOutOfRange { .. })
            ),
            "Z = {z} must be rejected"
        );
    }
}
