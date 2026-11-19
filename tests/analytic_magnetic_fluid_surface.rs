//! Oracles for the surface of a magnetisable liquid at rest in the field of permanent magnets
//! (`electromagnetic::{magnetic_fluid_traction, magnetic_fluid_surface_rise}`).
//!
//! oracle (the law, closed form): the ferrohydrostatic Bernoulli condition at the free surface of a
//! flat layer (Rosensweig, "Ferrohydrodynamics", 1985): `ρ g h = μ0·M̄·H + μ0·M_n²/2`, where `H` is
//! the field inside the liquid, `μ0·M̄·H = μ0∫₀^H M dH'` and `M = min(χH, M_s)` along `H`. Across a
//! flat surface the tangential `H` and the normal `B` are continuous. Below saturation, with `B` the
//! applied field just above the surface:
//! `h = χ/(2μ0 ρ g)·(B_t² + B_n²/(1+χ))`; a normal field gives `χB²/(2μ0 ρ g (1+χ))`, a tangential
//! one `χB²/(2μ0 ρ g)`, and `χ → 0` gives the old form `χB²/(2μ0 ρ g)` for any direction. Saturated
//! in a normal field: `H = B/μ0 − M_s` and `ρ g h = μ0(M_s·H − M_s·H_s/2) + μ0·M_s²/2`, `H_s = M_s/χ`.
//!
//! oracle (independent): an oblique field above saturation, against a reference solver written here
//! (Newton on the normal `B` condition `x + M_s x/√(H_t² + x²) = B_n/μ0`, in f64).
//!
//! With two magnets the fields add before any square: the rise of the sum is not the sum of rises.
//!
//! Tolerance: the code's μ0 is `4π·1e-7` to ten significant digits (relative 2.9e-11 short of the
//! exact value used here), and fixed point rounds at 2⁻⁶⁴ a step; the bisection of the saturated
//! branch runs 96 halvings of a range under 1e6 A/m. Hence the relative bound 1e-9.

use alice_physics_core::electromagnetic::{
    dipole_sample, magnetic_fluid_surface_rise, magnetic_fluid_traction, MagneticResponse,
};
use alice_physics_core::{Fix128, Vec3Fix};

const REL: f64 = 1e-9;
/// 2⁻³⁶: the bound a normalised normal's traction keeps against a unit normal's (see the test)
const REL_NORMAL: f64 = 1.0 / 68_719_476_736.0;
const MU0: f64 = 4.0 * std::f64::consts::PI * 1e-7;
const RHO: f64 = 1200.0;
const G: f64 = 9.81;

fn fx(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn up() -> Vec3Fix {
    Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO)
}

fn liquid(chi: Fix128, m_s: i64) -> MagneticResponse {
    MagneticResponse {
        chi_eff: chi,
        saturation_magnetization: Fix128::from_int(m_s),
    }
}

fn rise(field: Vec3Fix, r: &MagneticResponse) -> f64 {
    magnetic_fluid_surface_rise(field, up(), r, Fix128::from_int(1200), fx(981, 100))
        .expect("a liquid at rest")
        .to_f64()
}

fn rel_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= REL * a.abs().max(b.abs())
}

/// A magnet of moment `m` (A·m², pointing down at the liquid) at height `d_mm` over (`x_mm`, 0) of the surface
fn magnet_field(x_mm: i64, m: i64, d_mm: i64) -> Vec3Fix {
    dipole_sample(
        Vec3Fix::new(fx(x_mm, 1000), fx(d_mm, 1000), Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-m), Fix128::ZERO),
        Vec3Fix::ZERO,
    )
    .field
}

fn split(b: Vec3Fix) -> (f64, f64) {
    let (x, y, z) = (b.x.to_f64(), b.y.to_f64(), b.z.to_f64());
    (x * x + z * z, y * y)
}

#[test]
fn a_normal_field_lifts_by_the_flat_layer_closed_form() {
    // on the axis of a magnet the field is normal to the surface: m = 2 A·m² at 20 mm, |B| = 0.05 T
    let b = magnet_field(0, 2, 20);
    let (bt2, bn2) = split(b);
    assert!(bt2 < 1e-20 && rel_close(bn2.sqrt(), 0.05), "{b:?}");
    let h = rise(b, &liquid(Fix128::ONE, 400_000));
    let want = 1.0 * bn2 / (2.0 * MU0 * RHO * G * (1.0 + 1.0));
    assert!(rel_close(h, want), "h {h} vs {want}");
}

#[test]
fn a_tangential_field_lifts_by_the_tangential_closed_form() {
    let b = Vec3Fix::new(fx(5, 100), Fix128::ZERO, Fix128::ZERO);
    let h = rise(b, &liquid(Fix128::ONE, 400_000));
    let want = 0.05 * 0.05 / (2.0 * MU0 * RHO * G);
    assert!(rel_close(h, want), "h {h} vs {want}");
}

#[test]
fn an_oblique_field_lifts_by_its_two_parts() {
    // B = (0.03, 0.04, 0) T with χ = 2: tangential 0.03, normal 0.04
    let b = Vec3Fix::new(fx(3, 100), fx(4, 100), Fix128::ZERO);
    let h = rise(b, &liquid(Fix128::from_int(2), 400_000));
    let want = 2.0 / (2.0 * MU0 * RHO * G) * (0.03 * 0.03 + 0.04 * 0.04 / 3.0);
    assert!(rel_close(h, want), "h {h} vs {want}");
}

#[test]
fn a_weak_liquid_reduces_to_the_unscreened_form() {
    // χ = 1e-3: the normal part is χB²/(2μ0ρg(1+χ)), within χ of the unscreened χB²/(2μ0ρg)
    let chi = 1e-3;
    let b = Vec3Fix::new(fx(3, 100), fx(4, 100), Fix128::ZERO);
    let h = rise(b, &liquid(fx(1, 1000), 400_000));
    let unscreened = chi * 0.05 * 0.05 / (2.0 * MU0 * RHO * G);
    assert!(
        (h - unscreened).abs() <= 1.1 * chi * unscreened,
        "h {h} vs {unscreened}"
    );
    assert!(h < unscreened, "screening lowers the normal part");
}

#[test]
fn a_saturated_liquid_in_a_normal_field_follows_its_closed_form() {
    // χ = 1, M_s = H_s = 10000 A/m, B = 0.05 T: H₀ = B/μ0 ≈ 39789 A/m, the linear H₀/2 is above H_s
    let b = Vec3Fix::new(Fix128::ZERO, fx(5, 100), Fix128::ZERO);
    let h = rise(b, &liquid(Fix128::ONE, 10_000));
    let (m_s, h_s) = (10_000.0, 10_000.0);
    let hin = 0.05 / MU0 - m_s;
    let want = (MU0 * (m_s * hin - m_s * h_s / 2.0) + MU0 * m_s * m_s / 2.0) / (RHO * G);
    assert!(rel_close(h, want), "h {h} vs {want}");
    // the linear form would claim more: saturation caps the rise
    assert!(h < 0.05 * 0.05 / (4.0 * MU0 * RHO * G));
}

#[test]
fn the_rise_is_continuous_at_the_saturation_field() {
    // normal field with χ = 1, M_s = 10000: the liquid saturates where H₀/2 = H_s, B = 2μ0·H_s
    let b_at = 2.0 * MU0 * 10_000.0;
    let at = |s: f64| {
        let b = Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(b_at * s), Fix128::ZERO);
        rise(b, &liquid(Fix128::ONE, 10_000))
    };
    let (below, above) = (at(1.0 - 1e-7), at(1.0 + 1e-7));
    assert!((above - below).abs() <= 1e-5 * below, "{below} {above}");
}

#[test]
fn an_oblique_saturated_field_matches_a_reference_solver() {
    // B = (0.03, 0.04, 0) T, χ = 2, M_s = 5000 A/m (H_s = 2500 A/m, far below the field)
    let b = Vec3Fix::new(fx(3, 100), fx(4, 100), Fix128::ZERO);
    let h = rise(b, &liquid(Fix128::from_int(2), 5000));
    let (m_s, chi) = (5000.0, 2.0);
    let (ht, h0n) = (0.03 / MU0, 0.04 / MU0);
    // Newton on f(x) = x + M_s x / √(H_t² + x²) − H₀n, from the unsaturated side
    let mut x = h0n;
    for _ in 0..60 {
        let len = (ht * ht + x * x).sqrt();
        let f = x + m_s * x / len - h0n;
        let df = 1.0 + m_s * ht * ht / (len * len * len);
        x -= f / df;
    }
    let hin = (ht * ht + x * x).sqrt();
    let m_n = m_s * x / hin;
    let want = (MU0 * (m_s * hin - m_s * (m_s / chi) / 2.0) + MU0 * m_n * m_n / 2.0) / (RHO * G);
    assert!(rel_close(h, want), "h {h} vs {want}");
}

#[test]
fn two_magnets_superpose_their_fields_before_squaring() {
    // two magnets 15 mm either side, 20 mm up: midway their tangential fields cancel, the normal ones add
    let b1 = magnet_field(-15, 2, 20);
    let b2 = magnet_field(15, 2, 20);
    let r = liquid(Fix128::ONE, 400_000);
    let (h1, h2, h12) = (rise(b1, &r), rise(b2, &r), rise(b1 + b2, &r));
    let (bt2, bn2) = split(b1 + b2);
    let want = 1.0 / (2.0 * MU0 * RHO * G) * (bt2 + bn2 / 2.0);
    assert!(rel_close(h12, want), "h12 {h12} vs {want}");
    assert!(bt2 < 1e-12 * bn2, "the tangential parts cancel midway");
    assert!(
        (h12 - (h1 + h2)).abs() > 0.1 * (h1 + h2),
        "the rises do not add: {h12} vs {}",
        h1 + h2
    );
}

#[test]
fn no_field_no_magnetisation_or_no_weight() {
    let b = magnet_field(0, 2, 20);
    let r = liquid(Fix128::ONE, 400_000);
    assert_eq!(
        magnetic_fluid_traction(Vec3Fix::ZERO, up(), &r),
        Some(Fix128::ZERO),
        "no field, the level"
    );
    let water = liquid(Fix128::ZERO, 0);
    assert_eq!(
        magnetic_fluid_surface_rise(b, up(), &water, Fix128::ONE, Fix128::ONE),
        Some(Fix128::ZERO)
    );
    assert_eq!(
        magnetic_fluid_surface_rise(b, up(), &r, Fix128::ZERO, Fix128::ONE),
        None
    );
    assert_eq!(
        magnetic_fluid_surface_rise(b, up(), &r, Fix128::ONE, -Fix128::ONE),
        None
    );
}

/// The surface normal may have any length: the traction and the rise are those of its unit normal. An exact
/// scaling (2·up, up/8) gives the same bits; a general one ((3, 4, 0)/5 against (3, 4, 0), (1, 2, 2)/3 against
/// (1, 2, 2), in an oblique field and both regimes) agrees to `2⁻³⁶` relative: the normalisation is good to `2⁻⁴⁰`
/// per component, the inexact unit normal from `from_ratio` is within `2⁻⁶³`, and the traction is quadratic in
/// the normal's components with factors of order one (`B₀_n²` and `B₀_t² = |B₀|² − B₀_n²`), so a few `2⁻⁴⁰`
/// of the traction, under `2⁻³⁶`. A normal shorter than `2⁻¹⁶` (rounding noise) gives `None` for both
#[test]
fn the_normal_is_normalised_and_a_short_one_has_no_direction() {
    let b = magnet_field(7, 2, 20);
    let lin = liquid(Fix128::ONE, 400_000);
    let sat = liquid(Fix128::ONE, 2_000);
    let t = |n: Vec3Fix, r: &MagneticResponse| magnetic_fluid_traction(b, n, r);
    let scale = |n: Vec3Fix, k: Fix128| Vec3Fix::new(n.x * k, n.y * k, n.z * k);
    for r in [&lin, &sat] {
        let at_up = t(up(), r).expect("a unit normal");
        assert!(at_up > Fix128::ZERO);
        assert_eq!(t(scale(up(), Fix128::from_int(2)), r), Some(at_up));
        assert_eq!(t(scale(up(), fx(1, 8)), r), Some(at_up));
        let pairs = [
            (
                Vec3Fix::new(fx(3, 5), fx(4, 5), Fix128::ZERO),
                Vec3Fix::new(Fix128::from_int(3), Fix128::from_int(4), Fix128::ZERO),
            ),
            (
                Vec3Fix::new(fx(1, 3), fx(2, 3), fx(2, 3)),
                Vec3Fix::new(Fix128::ONE, Fix128::from_int(2), Fix128::from_int(2)),
            ),
        ];
        for (unit, long) in pairs {
            let (a, c) = (
                t(unit, r).expect("unit").to_f64(),
                t(long, r).expect("long").to_f64(),
            );
            assert!(a > 0.0);
            assert!((a - c).abs() <= a * REL_NORMAL, "{a} vs {c}");
            let ru = magnetic_fluid_surface_rise(b, unit, r, Fix128::from_int(1200), fx(981, 100))
                .expect("rise")
                .to_f64();
            let rl = magnetic_fluid_surface_rise(b, long, r, Fix128::from_int(1200), fx(981, 100))
                .expect("rise")
                .to_f64();
            assert!((ru - rl).abs() <= ru * REL_NORMAL, "{ru} vs {rl}");
        }
        let noise = Vec3Fix::new(Fix128::ZERO, Fix128::from_raw(0, 1 << 40), Fix128::ZERO);
        assert_eq!(t(noise, r), None);
        assert_eq!(t(Vec3Fix::ZERO, r), None);
        assert_eq!(
            magnetic_fluid_surface_rise(b, Vec3Fix::ZERO, r, Fix128::ONE, Fix128::ONE),
            None
        );
    }
}
