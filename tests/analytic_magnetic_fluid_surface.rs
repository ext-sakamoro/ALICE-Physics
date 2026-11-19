//! Oracles for the surface of a magnetisable liquid at rest in the field of permanent magnets
//! (`electromagnetic::{magnetic_fluid_pressure, magnetic_fluid_surface_rise}`).
//!
//! oracle (the law, closed form): a liquid of susceptibility χ and saturation magnetisation `M_s`
//! in a field `B` has the magnetic pressure `p* = ∫₀^|B| M dB'`, `M = min(χ·B'/μ0, M_s)`: that is
//! `χ|B|²/(2μ0)` below `B_s = μ0·M_s/χ` and `M_s|B| − M_s·B_s/2` above. Its free surface at rest
//! obeys the ferrohydrostatic Bernoulli equation `ρ g h − p* = const`, so where the field is zero
//! the surface is at its level and elsewhere it stands `h = p*/(ρ g)` higher. On the axis of a
//! point magnet `m` at distance `d`, `|B| = μ0/4π · 2|m|/d³`.
//!
//! With two magnets the fields add before the square: the rise of the sum is the rise of each
//! plus the cross term `χ·B₁·B₂/(μ0 ρ g)`, so summing the rises of each magnet is wrong whenever
//! the fields are not perpendicular.
//!
//! Tolerance: the code's μ0 is `4π·1e-7` to ten significant digits (relative 2.9e-11 short of
//! the exact value, which this file uses), entering `p*` as `1/μ0`; fixed-point rounding is 2⁻⁶⁴
//! per step. Hence the relative bound 1e-10.

use alice_physics_core::electromagnetic::{
    dipole_sample, magnetic_fluid_pressure, magnetic_fluid_surface_rise, MagneticResponse,
};
use alice_physics_core::{Fix128, Vec3Fix};

const REL: f64 = 1e-10;
const MU0: f64 = 4.0 * std::f64::consts::PI * 1e-7;
/// A ferrofluid-like liquid: ρ = 1200 kg/m³, under g = 9.81 m/s²
const RHO: f64 = 1200.0;
const G: f64 = 9.81;

fn fx(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn rho() -> Fix128 {
    Fix128::from_int(1200)
}

fn g() -> Fix128 {
    fx(981, 100)
}

/// χ = 1, M_s = 4e5 A/m (saturates at about 0.50 T, far above the fields used for the linear
/// branch)
fn liquid() -> MagneticResponse {
    MagneticResponse {
        chi_eff: Fix128::ONE,
        saturation_magnetization: Fix128::from_int(400_000),
    }
}

/// A magnet of moment `m` (A·m², along z) at height `d` (m) over the origin of the surface
fn magnet_over(x_mm: i64, m: i64, d_mm: i64) -> Vec3Fix {
    dipole_sample(
        Vec3Fix::new(fx(x_mm, 1000), Fix128::ZERO, fx(d_mm, 1000)),
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(m)),
        Vec3Fix::ZERO,
    )
    .field
}

fn rel_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= REL * a.abs().max(b.abs())
}

#[test]
fn the_rise_under_a_magnet_is_the_closed_form() {
    // m = 2 A·m² at d = 20 mm: |B| = 1e-7·2·2/(0.02)³ = 0.05 T, h ≈ 84 mm
    let field = magnet_over(0, 2, 20);
    let b = 1e-7 * 2.0 * 2.0 / (0.02 * 0.02 * 0.02);
    assert!(
        rel_close(field.z.to_f64(), b),
        "B {} vs {b}",
        field.z.to_f64()
    );
    let h = magnetic_fluid_surface_rise(field, &liquid(), rho(), g()).expect("a liquid at rest");
    let want = 1.0 * b * b / (2.0 * MU0 * RHO * G);
    assert!(rel_close(h.to_f64(), want), "h {} vs {want}", h.to_f64());
    // and the pressure is the rise times the weight per unit area
    let p = magnetic_fluid_pressure(field, &liquid()).to_f64();
    assert!(rel_close(p, want * RHO * G), "p* {p} vs {}", want * RHO * G);
}

#[test]
fn the_saturated_liquid_rises_by_its_saturation_branch() {
    // M_s = 20,000 A/m with χ = 1: B_s = μ0·M_s ≈ 0.0251 T, below the 0.05 T of the magnet
    let sat = MagneticResponse {
        chi_eff: Fix128::ONE,
        saturation_magnetization: Fix128::from_int(20_000),
    };
    let field = magnet_over(0, 2, 20);
    let b = 1e-7 * 2.0 * 2.0 / (0.02 * 0.02 * 0.02);
    let b_s = MU0 * 20_000.0;
    assert!(b > b_s);
    let want = (20_000.0 * b - 20_000.0 * b_s / 2.0) / (RHO * G);
    let h = magnetic_fluid_surface_rise(field, &sat, rho(), g())
        .expect("a liquid at rest")
        .to_f64();
    assert!(rel_close(h, want), "h {h} vs {want}");
    // the linear branch would claim more: saturation caps the rise
    let linear = b * b / (2.0 * MU0 * RHO * G);
    assert!(h < linear * 0.9, "saturated {h} not below linear {linear}");
}

#[test]
fn two_magnets_superpose_their_fields_before_squaring() {
    // two magnets 15 mm either side of the origin, 20 mm up, both along z
    let b1 = magnet_over(-15, 2, 20);
    let b2 = magnet_over(15, 2, 20);
    let rise = |f: Vec3Fix| {
        magnetic_fluid_surface_rise(f, &liquid(), rho(), g())
            .expect("a liquid at rest")
            .to_f64()
    };
    let (h1, h2, h12) = (rise(b1), rise(b2), rise(b1 + b2));
    let (v1, v2) = (
        [b1.x.to_f64(), b1.y.to_f64(), b1.z.to_f64()],
        [b2.x.to_f64(), b2.y.to_f64(), b2.z.to_f64()],
    );
    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    let sum_sq = (0..3)
        .map(|i| (v1[i] + v2[i]) * (v1[i] + v2[i]))
        .sum::<f64>();
    assert!(rel_close(h12, sum_sq / (2.0 * MU0 * RHO * G)), "h12 {h12}");
    // the cross term is what summing the rises would lose, and it is not small here
    let cross = dot / (MU0 * RHO * G);
    assert!(
        rel_close(h12 - h1 - h2, cross),
        "cross {} vs {cross}",
        h12 - h1 - h2
    );
    assert!(
        cross.abs() > 0.1 * (h1 + h2),
        "cross {cross} too small to tell the two apart"
    );
}

#[test]
fn no_field_no_magnetisation_or_no_weight() {
    let field = magnet_over(0, 2, 20);
    let flat = magnetic_fluid_surface_rise(Vec3Fix::ZERO, &liquid(), rho(), g())
        .expect("a liquid at rest");
    assert_eq!(flat, Fix128::ZERO, "no field, the level");
    let water = MagneticResponse {
        chi_eff: Fix128::ZERO,
        saturation_magnetization: Fix128::ZERO,
    };
    assert_eq!(
        magnetic_fluid_surface_rise(field, &water, rho(), g()),
        Some(Fix128::ZERO)
    );
    assert_eq!(
        magnetic_fluid_surface_rise(field, &liquid(), Fix128::ZERO, g()),
        None
    );
    assert_eq!(
        magnetic_fluid_surface_rise(field, &liquid(), rho(), -g()),
        None
    );
}
