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
}
