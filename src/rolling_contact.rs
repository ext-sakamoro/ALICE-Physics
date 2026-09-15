//! Hertzian rolling-contact stress + Basquin-style fatigue life.
//!
//! Complements [`crate::fatigue`] (bending-stress fatigue) with a
//! rolling-contact model targeted at ball / roller bearings and gear
//! tooth flanks:
//!
//! - **Hertzian point contact** between two spheres of radii `R1`,
//!   `R2` under a compressive load `P`, producing a contact patch of
//!   radius `a` and peak subsurface pressure `p_max`.
//! - **Basquin fatigue life** — number of load cycles to failure at a
//!   given rolling-contact stress, using a power-law S–N curve
//!   `N_f = C · σ⁻ᵐ`.
//!
//! Only elastic contact is modelled; plastic deformation is future
//! work (callers wanting shakedown analysis can extend the API).

/// Result of a Hertzian sphere-on-sphere contact calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HertzianContact {
    /// Radius of the circular contact patch (m).
    pub contact_radius_m: f32,
    /// Peak contact pressure at the centre of the patch (Pa).
    pub peak_pressure_pa: f32,
    /// Depth at which the maximum shear stress occurs (m); this is
    /// where classic rolling-contact fatigue cracks initiate.
    pub max_shear_depth_m: f32,
}

/// Compute the Hertzian contact patch parameters for two elastic
/// spheres pressed together by `load_n`.
///
/// - `radius_1_m`, `radius_2_m`: sphere radii. Use `f32::INFINITY` for
///   a flat surface (roller against a rail).
/// - `elastic_modulus_1_pa`, `elastic_modulus_2_pa`: Young's moduli.
/// - `poisson_1`, `poisson_2`: Poisson's ratios.
///
/// Returns the contact patch radius `a`, peak pressure `p_max`, and
/// depth of maximum shear (≈ `0.48 · a`).
///
/// # Panics
///
/// Panics if any radius / modulus / load is negative or zero.
#[must_use]
pub fn hertzian_sphere_sphere(
    load_n: f32,
    radius_1_m: f32,
    radius_2_m: f32,
    elastic_modulus_1_pa: f32,
    elastic_modulus_2_pa: f32,
    poisson_1: f32,
    poisson_2: f32,
) -> HertzianContact {
    assert!(load_n > 0.0, "load must be positive");
    assert!(radius_1_m > 0.0, "radius_1 must be positive");
    assert!(radius_2_m > 0.0, "radius_2 must be positive");
    assert!(
        elastic_modulus_1_pa > 0.0,
        "elastic_modulus_1 must be positive"
    );
    assert!(
        elastic_modulus_2_pa > 0.0,
        "elastic_modulus_2 must be positive"
    );

    // Effective radius: 1/R* = 1/R1 + 1/R2.
    let inv_r_star = 1.0 / radius_1_m + 1.0 / radius_2_m;
    let r_star = 1.0 / inv_r_star;

    // Effective modulus:
    //   1/E* = (1 - ν1²)/E1 + (1 - ν2²)/E2.
    let inv_e_star = (1.0 - poisson_1 * poisson_1) / elastic_modulus_1_pa
        + (1.0 - poisson_2 * poisson_2) / elastic_modulus_2_pa;
    let e_star = 1.0 / inv_e_star;

    // Contact patch radius: a = ((3 P R*) / (4 E*))^(1/3).
    let contact_radius_m = crate::det_math::cbrt((3.0 * load_n * r_star) / (4.0 * e_star));

    // Peak pressure: p_max = 3 P / (2 π a²).
    let peak_pressure_pa =
        3.0 * load_n / (2.0 * core::f32::consts::PI * contact_radius_m * contact_radius_m);

    // Maximum subsurface shear occurs at z ≈ 0.48 · a for a sphere-sphere
    // Hertzian contact (Johnson, "Contact Mechanics", ch. 3).
    let max_shear_depth_m = 0.48 * contact_radius_m;

    HertzianContact {
        contact_radius_m,
        peak_pressure_pa,
        max_shear_depth_m,
    }
}

/// Basquin-style S–N (stress-life) model.
///
/// ```text
/// N_f = C · σ⁻ᵐ
/// ```
///
/// where:
///
/// - `σ` — cyclic stress amplitude (Pa),
/// - `C` — fatigue coefficient (fitted from material tests),
/// - `m` — Basquin exponent (typically 3 for through-hardened
///   bearing steel, 9-10 for high-strength alloys).
///
/// Returns the predicted number of cycles to failure. A stress amplitude
/// below the endurance limit yields `f32::INFINITY`.
///
/// # Panics
///
/// Panics if `stress_pa < 0`, `basquin_coefficient <= 0`, or
/// `basquin_exponent <= 0`.
#[must_use]
pub fn basquin_cycles_to_failure(
    stress_pa: f32,
    basquin_coefficient: f32,
    basquin_exponent: f32,
    endurance_limit_pa: f32,
) -> f32 {
    assert!(stress_pa >= 0.0, "stress must be non-negative");
    assert!(basquin_coefficient > 0.0, "coefficient must be positive");
    assert!(basquin_exponent > 0.0, "exponent must be positive");
    if stress_pa <= endurance_limit_pa {
        return f32::INFINITY;
    }
    basquin_coefficient * crate::det_math::powf(stress_pa, -basquin_exponent)
}

/// Convenience wrapper combining [`hertzian_sphere_sphere`] with
/// [`basquin_cycles_to_failure`].
///
/// Returns the predicted number of cycles the contact will survive
/// under repeated application of `load_n`.
///
/// The effective cyclic stress fed into the Basquin equation is the
/// peak Hertzian pressure `p_max`. Callers who need a different stress
/// metric (e.g. subsurface orthogonal shear) can call the two helpers
/// separately.
///
/// # Panics
///
/// Same conditions as [`hertzian_sphere_sphere`] and
/// [`basquin_cycles_to_failure`].
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn rolling_contact_life_cycles(
    load_n: f32,
    radius_1_m: f32,
    radius_2_m: f32,
    elastic_modulus_1_pa: f32,
    elastic_modulus_2_pa: f32,
    poisson_1: f32,
    poisson_2: f32,
    basquin_coefficient: f32,
    basquin_exponent: f32,
    endurance_limit_pa: f32,
) -> f32 {
    let contact = hertzian_sphere_sphere(
        load_n,
        radius_1_m,
        radius_2_m,
        elastic_modulus_1_pa,
        elastic_modulus_2_pa,
        poisson_1,
        poisson_2,
    );
    basquin_cycles_to_failure(
        contact.peak_pressure_pa,
        basquin_coefficient,
        basquin_exponent,
        endurance_limit_pa,
    )
}

/// Material presets: `(E [Pa], ν, Basquin C, Basquin m,
/// endurance limit [Pa])`.
///
/// Values are representative; production callers should supply fits
/// from their own material tests.
pub mod materials {
    /// Through-hardened bearing steel (52100 / SUJ2).
    #[must_use]
    pub const fn bearing_steel_52100() -> (f32, f32, f32, f32, f32) {
        (210.0e9, 0.30, 5.0e34, 3.0, 1.5e9)
    }

    /// Case-hardened gear steel (Ni-Cr-Mo 8620 / SNCM220).
    #[must_use]
    pub const fn gear_steel_8620() -> (f32, f32, f32, f32, f32) {
        (205.0e9, 0.29, 2.5e34, 3.2, 1.3e9)
    }

    /// Silicon nitride ceramic (hybrid bearings).
    #[must_use]
    pub const fn silicon_nitride() -> (f32, f32, f32, f32, f32) {
        (310.0e9, 0.27, 8.0e34, 3.5, 2.0e9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    fn steel_modulus() -> f32 {
        210.0e9
    }

    #[test]
    fn hertzian_contact_matches_textbook_case() {
        // Two identical steel spheres, R = 5 mm, load = 100 N.
        // Textbook result: p_max ≈ 3 P / (2 π a²) with a ≈ 0.16 mm.
        let c = hertzian_sphere_sphere(
            100.0,
            5.0e-3,
            5.0e-3,
            steel_modulus(),
            steel_modulus(),
            0.30,
            0.30,
        );
        assert!(c.contact_radius_m > 0.0);
        assert!(c.contact_radius_m < 1.0e-3, "a = {}", c.contact_radius_m);
        // Peak pressure sanity: 3 P / (2 π a²) exact.
        let expected = 3.0 * 100.0 / (2.0 * PI * c.contact_radius_m * c.contact_radius_m);
        let rel = (c.peak_pressure_pa - expected).abs() / expected;
        assert!(rel < 1.0e-4);
    }

    #[test]
    fn contact_radius_grows_with_load() {
        let low = hertzian_sphere_sphere(
            100.0,
            5.0e-3,
            5.0e-3,
            steel_modulus(),
            steel_modulus(),
            0.30,
            0.30,
        );
        let high = hertzian_sphere_sphere(
            1000.0,
            5.0e-3,
            5.0e-3,
            steel_modulus(),
            steel_modulus(),
            0.30,
            0.30,
        );
        assert!(high.contact_radius_m > low.contact_radius_m);
        assert!(high.peak_pressure_pa > low.peak_pressure_pa);
    }

    #[test]
    fn contact_radius_shrinks_with_stiffer_material() {
        let steel = hertzian_sphere_sphere(
            100.0,
            5.0e-3,
            5.0e-3,
            steel_modulus(),
            steel_modulus(),
            0.30,
            0.30,
        );
        let ceramic = hertzian_sphere_sphere(100.0, 5.0e-3, 5.0e-3, 310.0e9, 310.0e9, 0.27, 0.27);
        assert!(ceramic.contact_radius_m < steel.contact_radius_m);
        assert!(ceramic.peak_pressure_pa > steel.peak_pressure_pa);
    }

    #[test]
    fn max_shear_depth_is_positive_fraction_of_contact_radius() {
        let c = hertzian_sphere_sphere(
            100.0,
            5.0e-3,
            5.0e-3,
            steel_modulus(),
            steel_modulus(),
            0.30,
            0.30,
        );
        let ratio = c.max_shear_depth_m / c.contact_radius_m;
        assert!((ratio - 0.48).abs() < 1.0e-3);
    }

    #[test]
    fn basquin_infinite_life_below_endurance() {
        let cycles = basquin_cycles_to_failure(500.0e6, 5.0e34, 3.0, 1.0e9);
        assert!(cycles.is_infinite());
    }

    #[test]
    fn basquin_finite_life_above_endurance() {
        let cycles = basquin_cycles_to_failure(2.0e9, 5.0e34, 3.0, 1.5e9);
        assert!(cycles.is_finite());
        assert!(cycles > 0.0);
    }

    #[test]
    fn basquin_lower_stress_gives_longer_life() {
        let low = basquin_cycles_to_failure(2.5e9, 5.0e34, 3.0, 1.5e9);
        let high = basquin_cycles_to_failure(3.5e9, 5.0e34, 3.0, 1.5e9);
        assert!(low > high);
    }

    #[test]
    fn combined_rolling_contact_life_returns_finite_positive() {
        let (e, nu, c, m, sigma_e) = materials::bearing_steel_52100();
        let cycles =
            rolling_contact_life_cycles(5000.0, 6.0e-3, 6.0e-3, e, e, nu, nu, c, m, sigma_e);
        assert!(cycles.is_finite());
        assert!(cycles > 0.0);
    }

    #[test]
    #[should_panic(expected = "load must be positive")]
    fn hertzian_panics_on_zero_load() {
        let _ = hertzian_sphere_sphere(0.0, 1.0e-3, 1.0e-3, 210.0e9, 210.0e9, 0.3, 0.3);
    }

    #[test]
    #[should_panic(expected = "exponent must be positive")]
    fn basquin_panics_on_nonpositive_exponent() {
        let _ = basquin_cycles_to_failure(1.0e9, 5.0e34, 0.0, 1.0e8);
    }

    /// Physical sanity of a `(E, ν, C, m, σ_e)` preset tuple.
    fn assert_preset_sane(name: &str, p: (f32, f32, f32, f32, f32), e_lo: f32, e_hi: f32) {
        let (e, nu, c, m, sigma_e) = p;
        assert!(e >= e_lo && e <= e_hi, "{name}: E = {e} Pa out of range");
        assert!(nu > 0.0 && nu < 0.5, "{name}: ν = {nu} not in (0, 0.5)");
        assert!(c > 0.0 && c.is_finite(), "{name}: Basquin C = {c}");
        assert!(m > 0.0 && m < 20.0, "{name}: Basquin m = {m}");
        assert!(
            sigma_e > 0.0 && sigma_e < e,
            "{name}: endurance limit {sigma_e} must be positive and below E"
        );
        // Basquin C must be usable: with σ = σ_e the life is finite and > 1 cycle.
        let n = basquin_cycles_to_failure(sigma_e * 1.01, c, m, sigma_e);
        assert!(n.is_finite() && n > 1.0, "{name}: N({sigma_e}·1.01) = {n}");
    }

    /// Two presets are distinct if any field differs by more than a
    /// relative 1e-6 (tolerance-based, never an exact float compare).
    fn presets_differ(a: (f32, f32, f32, f32, f32), b: (f32, f32, f32, f32, f32)) -> bool {
        let rel = |x: f32, y: f32| (x - y).abs() > 1.0e-6 * x.abs().max(y.abs());
        rel(a.0, b.0) || rel(a.1, b.1) || rel(a.2, b.2) || rel(a.3, b.3) || rel(a.4, b.4)
    }

    #[test]
    fn gear_steel_8620_preset_is_sane_and_distinct() {
        let gear = materials::gear_steel_8620();
        // Steel: E in [190, 215] GPa.
        assert_preset_sane("gear_steel_8620", gear, 190.0e9, 215.0e9);
        assert!((gear.0 - 205.0e9).abs() < 1.0, "E = 205 GPa");
        assert!((gear.1 - 0.29).abs() < 1e-6, "ν = 0.29");
        // Differs from the other presets in at least the modulus.
        let bearing = materials::bearing_steel_52100();
        let ceramic = materials::silicon_nitride();
        assert!(presets_differ(gear, bearing));
        assert!(presets_differ(gear, ceramic));
        assert!(
            gear.0 < bearing.0,
            "case-hardened 8620 is slightly less stiff than 52100"
        );
        assert!(gear.4 < bearing.4, "8620 endurance limit below 52100");
    }

    #[test]
    fn silicon_nitride_preset_is_sane_and_distinct() {
        let sn = materials::silicon_nitride();
        // Si3N4 ceramic: E in [290, 330] GPa.
        assert_preset_sane("silicon_nitride", sn, 290.0e9, 330.0e9);
        assert!((sn.0 - 310.0e9).abs() < 1.0, "E = 310 GPa");
        assert!((sn.1 - 0.27).abs() < 1e-6, "ν = 0.27");
        let bearing = materials::bearing_steel_52100();
        let gear = materials::gear_steel_8620();
        assert!(presets_differ(sn, bearing));
        assert!(presets_differ(sn, gear));
        // Ceramic is the stiffest, has the lowest Poisson ratio and the
        // highest endurance limit of the three presets.
        assert!(sn.0 > bearing.0 && sn.0 > gear.0);
        assert!(sn.1 < bearing.1 && sn.1 < gear.1);
        assert!(sn.4 > bearing.4 && sn.4 > gear.4);
    }

    #[test]
    fn presets_feed_hertz_and_match_closed_form_relations() {
        // Hybrid bearing: silicon-nitride ball on 8620 gear-steel flank.
        let (e1, nu1, _, _, _) = materials::silicon_nitride();
        let (e2, nu2, _, _, _) = materials::gear_steel_8620();
        let load = 250.0;
        let r1 = 4.0e-3;
        let r2 = 12.0e-3;
        let c = hertzian_sphere_sphere(load, r1, r2, e1, e2, nu1, nu2);

        // Closed-form Hertz relations documented by the module:
        //   1/R* = 1/R1 + 1/R2
        //   1/E* = (1-ν1²)/E1 + (1-ν2²)/E2
        //   a    = (3 P R* / (4 E*))^(1/3)  ⇔  a³ = 3 P R* / (4 E*)
        //   p_max = 3 P / (2 π a²)
        //   z_max_shear = 0.48 a
        let r_star = 1.0 / (1.0 / r1 + 1.0 / r2);
        let e_star = 1.0 / ((1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2);
        let a = c.contact_radius_m;
        let a_cubed = a * a * a;
        let expected_a_cubed = 3.0 * load * r_star / (4.0 * e_star);
        let rel = (a_cubed - expected_a_cubed).abs() / expected_a_cubed;
        assert!(rel < 1.0e-4, "a³ relation: rel err = {rel}");

        let expected_p = 3.0 * load / (2.0 * PI * a * a);
        let rel_p = (c.peak_pressure_pa - expected_p).abs() / expected_p;
        assert!(rel_p < 1.0e-5, "p_max relation: rel err = {rel_p}");

        let ratio = c.max_shear_depth_m / a;
        assert!((ratio - 0.48).abs() < 1.0e-5, "shear depth ratio = {ratio}");

        // Sanity of magnitude: a bearing-scale patch is sub-millimetre and
        // p_max is in the GPa regime.
        assert!(a > 1.0e-5 && a < 1.0e-3, "a = {a}");
        assert!(c.peak_pressure_pa > 1.0e8 && c.peak_pressure_pa < 1.0e10);

        // Symmetry: swapping the two bodies leaves the result unchanged.
        let swapped = hertzian_sphere_sphere(load, r2, r1, e2, e1, nu2, nu1);
        assert!((swapped.contact_radius_m - a).abs() / a < 1.0e-6);
        assert!(
            (swapped.peak_pressure_pa - c.peak_pressure_pa).abs() / c.peak_pressure_pa < 1.0e-6
        );

        // Stiffer pairing (ceramic on ceramic) → smaller patch, higher p_max
        // than the gear-steel-on-gear-steel pairing at the same load.
        let cc = hertzian_sphere_sphere(load, r1, r2, e1, e1, nu1, nu1);
        let gg = hertzian_sphere_sphere(load, r1, r2, e2, e2, nu2, nu2);
        assert!(cc.contact_radius_m < gg.contact_radius_m);
        assert!(cc.peak_pressure_pa > gg.peak_pressure_pa);
    }
}
