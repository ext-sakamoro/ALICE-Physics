//! Oracles for the `filament_db` unit-conversion constants (`GPA_TO_PA` /
//! `MPA_TO_PA` / `G_CM3_TO_KG_M3`), the SI-unit accessors (`density_si` /
//! `yield_pa` / `youngs_pa`), the Z-axis anisotropic accessors (`yield_z` /
//! `youngs_z` / `tensile_z`), the angle-mixing law (`yield_at_angle` /
//! `youngs_at_angle`), the category predicates (`is_fdm` / `is_sheet_metal`)
//! and the registry filter (`FilamentDb::by_category`). Production entry
//! point: `examples/filament_database_properties.rs`.
//!
//! # Closed forms
//!
//! * **Unit conversion**: `GPA_TO_PA = {hi: 1_000_000_000, lo: 0}` and
//!   `MPA_TO_PA = {hi: 1_000_000, lo: 0}` and
//!   `G_CM3_TO_KG_M3 = {hi: 1_000, lo: 0}` are pure integers (verified from
//!   the constants' own raw fields, not assumed), so `youngs_pa =
//!   youngs_modulus_gpa * GPA_TO_PA`, `yield_pa = yield_strength_mpa *
//!   MPA_TO_PA` and `density_si = density_g_cm3 * G_CM3_TO_KG_M3` are each a
//!   single `Fix128` multiplication by an exact integer, reproduced here as
//!   `field * CONST` directly against the raw public fields (never by
//!   calling `youngs_pa` / `yield_pa` / `density_si` themselves).
//! * **Z-axis accessors**: `yield_z = yield_strength_mpa * anisotropy_z_ratio`,
//!   `youngs_z = youngs_modulus_gpa * anisotropy_z_ratio`, `tensile_z =
//!   tensile_strength_mpa * anisotropy_z_ratio` — again reproduced against
//!   the raw fields, not by calling the accessors.
//! * **Angle-mixing law**: `yield_at_angle(theta) = yield_strength_mpa *
//!   cos(theta)^2 + yield_z * sin(theta)^2` (and the `youngs_*` analogue with
//!   `youngs_modulus_gpa` / `youngs_z`). At `theta = 0` this is exactly
//!   `yield_strength_mpa` (`cos=1, sin=0`); at `theta = pi/2` it is exactly
//!   `yield_z` (`cos=0, sin=1`) — both boundary values are closed forms that
//!   do not require evaluating `sin_cos` at all. For a midpoint (`pi/4`) this
//!   file computes an **independent** reference using
//!   `alice_physics::det_math::{sin, cos}` (`f32`, a different code path
//!   from `Fix128::sin_cos`'s CORDIC — see `src/lib.rs`'s `det_math` module
//!   doc, which delegates to the `alice-det-math` crate rather than the
//!   engine's own CORDIC) and cross-checks the production `Fix128` result
//!   against it in `f64`.
//! * **by_category**: a direct re-filter (`db.iter().filter(|m| m.category ==
//!   cat).count()`) independent of calling `by_category` itself, checked
//!   against the production call's `.len()`.
//!
//! # Degenerate input
//!
//! * **Angle outside `[0, pi/2]`**: neither `yield_at_angle` nor
//!   `youngs_at_angle` validates or clamps `theta` — the formula is
//!   `sigma_xy*cos^2 + sigma_z*sin^2`, and `cos^2`/`sin^2` are even and
//!   `pi`-periodic for *any* real `theta`, so a negative angle must equal its
//!   positive mirror (`theta` vs `-theta`) and an angle one full half-turn
//!   away (`theta` vs `theta + pi`) must land on the same value. `Fix128::
//!   sin_cos` additionally performs a full modular range reduction (see
//!   `cordic_sin_cos` in `src/math.rs`: `k = floor((theta+pi)/(2*pi))`
//!   subtracts `k * 2*pi` before the CORDIC loop runs), so a large multiple
//!   of `pi` (`1000*pi`) still reduces correctly, modulo the rounding
//!   `Fix128::PI` itself carries (`PI` is a `2^-64`-rounded approximation, so
//!   `1000*PI` accumulates ~`1000 * 2^-64` of drift before reduction, far
//!   below this test's tolerance).
//! * **Empty / unregistered category**: `FilamentDb::with_defaults` never
//!   registers `MaterialCategory::Sla` or `::Powder`, so `by_category` on
//!   either returns an empty `Vec`, not an error and not the unfiltered
//!   registry.
//! * **`is_fdm` / `is_sheet_metal` both false**: possible and expected for
//!   `Sla` / `Powder` materials (neither predicate matches). **Both true is
//!   impossible by construction** — `category` is a single `MaterialCategory`
//!   enum value, so the two `matches!` arms are mutually exclusive; this file
//!   asserts the impossibility holds for every preset, rather than merely
//!   assuming it.
//! * **Extreme values (overflow)**: `Fix128`'s `Mul` is documented as
//!   wrapping, never panicking (`src/math.rs`, the `Mul for Fix128` doc, WM-01
//!   / doctrine B-12). `yield_pa` / `youngs_pa` / `density_si` call plain
//!   `Mul`, not `checked_mul`, so a material property large enough that its
//!   product with the unit constant does not fit `Fix128`'s `i64` integer
//!   half wraps modulo `2^64` instead of panicking. This file re-derives the
//!   wrapped value independently (pure `i128` arithmetic mirroring the
//!   documented 128x128->128 middle-bits formula for two pure-integer
//!   operands, **without calling `Fix128::mul`**) and cross-checks it against
//!   the production accessor, via `catch_unwind` to also confirm no panic.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::det_math;
use alice_physics::filament_db::{
    FilamentDb, MaterialCategory, MaterialProperties, GPA_TO_PA, G_CM3_TO_KG_M3, MPA_TO_PA,
};
use alice_physics::math::Fix128;

fn near(a: Fix128, b: Fix128, tol: Fix128) -> bool {
    (a - b).abs() < tol
}

// ===========================================================================
// A. Unit conversion constants: verified from their own raw fields, not
//    assumed from the module doc comment.
// ===========================================================================

#[test]
fn unit_conversion_constants_are_exact_pure_integers() {
    assert_eq!(GPA_TO_PA, Fix128::from_raw(1_000_000_000, 0));
    assert_eq!(MPA_TO_PA, Fix128::from_raw(1_000_000, 0));
    assert_eq!(G_CM3_TO_KG_M3, Fix128::from_raw(1_000, 0));
    // Each is also exactly its integer constructor (double-checks from_raw
    // and from_int agree for these values).
    assert_eq!(GPA_TO_PA, Fix128::from_int(1_000_000_000));
    assert_eq!(MPA_TO_PA, Fix128::from_int(1_000_000));
    assert_eq!(G_CM3_TO_KG_M3, Fix128::from_int(1_000));
}

// ===========================================================================
// B. SI-unit accessors: closed form against the raw fields, every default
//    preset, not just PLA.
// ===========================================================================

#[test]
fn density_si_yield_pa_youngs_pa_equal_the_raw_field_times_the_constant() {
    for m in FilamentDb::with_defaults().iter() {
        assert_eq!(
            m.density_si(),
            m.density_g_cm3 * G_CM3_TO_KG_M3,
            "{}: density_si",
            m.name
        );
        assert_eq!(
            m.yield_pa(),
            m.yield_strength_mpa * MPA_TO_PA,
            "{}: yield_pa",
            m.name
        );
        assert_eq!(
            m.youngs_pa(),
            m.youngs_modulus_gpa * GPA_TO_PA,
            "{}: youngs_pa",
            m.name
        );
    }
}

// ===========================================================================
// C. Z-axis anisotropic accessors: closed form against the raw fields.
// ===========================================================================

#[test]
fn z_axis_accessors_equal_the_raw_field_times_anisotropy_ratio() {
    for m in FilamentDb::with_defaults().iter() {
        assert_eq!(
            m.yield_z(),
            m.yield_strength_mpa * m.anisotropy_z_ratio,
            "{}: yield_z",
            m.name
        );
        assert_eq!(
            m.youngs_z(),
            m.youngs_modulus_gpa * m.anisotropy_z_ratio,
            "{}: youngs_z",
            m.name
        );
        assert_eq!(
            m.tensile_z(),
            m.tensile_strength_mpa * m.anisotropy_z_ratio,
            "{}: tensile_z",
            m.name
        );
    }
}

#[test]
fn isotropic_sheet_metal_z_axis_equals_the_in_plane_value() {
    // anisotropy_z_ratio == ONE for sheet metal -> Z accessors are identity.
    for m in [MaterialProperties::sus304(), MaterialProperties::a5052()] {
        assert_eq!(m.anisotropy_z_ratio, Fix128::ONE);
        assert_eq!(m.yield_z(), m.yield_strength_mpa);
        assert_eq!(m.youngs_z(), m.youngs_modulus_gpa);
        assert_eq!(m.tensile_z(), m.tensile_strength_mpa);
    }
}

// ===========================================================================
// D. Angle-mixing law: boundary closed forms + independent midpoint oracle
//    + both-directions (negative / periodic) invariants.
// ===========================================================================

#[test]
fn yield_and_youngs_at_angle_zero_equal_the_in_plane_value_exactly_in_form() {
    // theta = 0: cos=1, sin=0 exactly (no sin_cos evaluation needed for the
    // *closed form*; CORDIC's own cos(0)/sin(0) still carries ~2^-64 of
    // accumulated iteration error, hence the tolerance, not equality).
    let pla = MaterialProperties::pla();
    let tol_yield = pla.yield_strength_mpa * Fix128::from_ratio(1, 10_000);
    let tol_youngs = pla.youngs_modulus_gpa * Fix128::from_ratio(1, 10_000);
    assert!(near(
        pla.yield_at_angle(Fix128::ZERO),
        pla.yield_strength_mpa,
        tol_yield
    ));
    assert!(near(
        pla.youngs_at_angle(Fix128::ZERO),
        pla.youngs_modulus_gpa,
        tol_youngs
    ));
}

#[test]
fn yield_and_youngs_at_angle_ninety_degrees_equal_the_z_value() {
    let pla = MaterialProperties::pla();
    let tol_yield = pla.yield_strength_mpa * Fix128::from_ratio(1, 1_000);
    let tol_youngs = pla.youngs_modulus_gpa * Fix128::from_ratio(1, 1_000);
    assert!(near(
        pla.yield_at_angle(Fix128::HALF_PI),
        pla.yield_z(),
        tol_yield
    ));
    assert!(near(
        pla.youngs_at_angle(Fix128::HALF_PI),
        pla.youngs_z(),
        tol_youngs
    ));
}

/// Independent midpoint oracle: `det_math::{sin, cos}` (f32, delegates to the
/// `alice-det-math` crate) is a different code path from `Fix128::sin_cos`'s
/// CORDIC, so cross-checking against it in f64 is a genuine second
/// implementation, not a restatement of the function under test.
#[test]
fn yield_and_youngs_at_angle_quarter_pi_matches_an_independent_f32_trig_oracle() {
    let pla = MaterialProperties::pla();
    let theta = Fix128::HALF_PI.shr_bits(1); // exact pi/4 (bit shift, no rounding)
    let theta_f32 = theta.to_f32();
    let c = f64::from(det_math::cos(theta_f32));
    let s = f64::from(det_math::sin(theta_f32));
    let sigma_xy_mpa = pla.yield_strength_mpa.to_f64();
    let sigma_z_mpa = pla.yield_z().to_f64();
    let e_xy_gpa = pla.youngs_modulus_gpa.to_f64();
    let e_z_gpa = pla.youngs_z().to_f64();
    let expected_yield = sigma_xy_mpa * c * c + sigma_z_mpa * s * s;
    let expected_youngs = e_xy_gpa * c * c + e_z_gpa * s * s;

    let got_yield = pla.yield_at_angle(theta).to_f64();
    let got_youngs = pla.youngs_at_angle(theta).to_f64();
    assert!(
        (got_yield - expected_yield).abs() < 1e-3,
        "yield_at_angle(pi/4): got {got_yield}, f32-trig oracle {expected_yield}"
    );
    assert!(
        (got_youngs - expected_youngs).abs() < 1e-4,
        "youngs_at_angle(pi/4): got {got_youngs}, f32-trig oracle {expected_youngs}"
    );
}

#[test]
fn angle_mixing_law_is_even_negative_angle_matches_its_positive_mirror() {
    // cos(-t)^2 == cos(t)^2 and sin(-t)^2 == sin(t)^2 for any real t -> the
    // mixing law must agree at +theta and -theta. Neither accessor documents
    // or enforces a [0, pi/2] domain, so this is the "argument moves, answer
    // doesn't" invariant for the full real-line domain they actually accept.
    let pla = MaterialProperties::pla();
    let theta = Fix128::HALF_PI.shr_bits(1) + Fix128::HALF_PI.shr_bits(2); // 3*pi/8-ish mix, inside domain
    let tol = pla.yield_strength_mpa * Fix128::from_ratio(1, 1_000);
    let tol_e = pla.youngs_modulus_gpa * Fix128::from_ratio(1, 1_000);
    assert!(near(
        pla.yield_at_angle(theta),
        pla.yield_at_angle(Fix128::ZERO - theta),
        tol
    ));
    assert!(near(
        pla.youngs_at_angle(theta),
        pla.youngs_at_angle(Fix128::ZERO - theta),
        tol_e
    ));
}

#[test]
fn angle_mixing_law_out_of_the_documented_range_still_reduces_correctly() {
    // No clamp, no Err: an angle a full half-turn (pi) away from 0 lands back
    // on the in-plane value (cos(pi)^2 = 1, sin(pi)^2 = 0, same as theta=0),
    // and a large multiple of pi (1000*pi, far outside the "[-pi, pi] for
    // best precision" doc note on `sin_cos`) still reduces close to theta=0
    // via the function's own full modular range reduction, not by panicking
    // or returning a nonsensical value.
    let pla = MaterialProperties::pla();
    let tol = pla.yield_strength_mpa * Fix128::from_ratio(1, 100);

    let at_pi = pla.yield_at_angle(Fix128::PI);
    let at_zero = pla.yield_at_angle(Fix128::ZERO);
    assert!(
        near(at_pi, at_zero, tol),
        "yield_at_angle(pi) should match yield_at_angle(0): {at_pi:?} vs {at_zero:?}"
    );

    let far = Fix128::PI * Fix128::from_int(1000);
    let at_far = pla.yield_at_angle(far);
    assert!(
        near(at_far, at_zero, tol),
        "yield_at_angle(1000*pi) should still reduce near yield_at_angle(0): \
         {at_far:?} vs {at_zero:?}"
    );
}

// ===========================================================================
// E. by_category: independent re-filter + empty-category degenerate input.
// ===========================================================================

#[test]
fn by_category_matches_an_independent_refilter_for_every_category() {
    let db = FilamentDb::with_defaults();
    for cat in [
        MaterialCategory::Fdm,
        MaterialCategory::SheetMetal,
        MaterialCategory::Sla,
        MaterialCategory::Powder,
    ] {
        let independent_count = db.iter().filter(|m| m.category == cat).count();
        let production = db.by_category(cat);
        assert_eq!(
            production.len(),
            independent_count,
            "{cat:?}: by_category len vs independent refilter"
        );
        // Distinct input materials -> distinct output materials: every id
        // returned actually has the requested category (not a different one
        // that happens to share the count).
        for m in &production {
            assert_eq!(m.category, cat);
        }
    }
}

#[test]
fn by_category_on_an_unregistered_category_is_empty_not_an_error_not_everything() {
    let db = FilamentDb::with_defaults();
    let sla = db.by_category(MaterialCategory::Sla);
    let powder = db.by_category(MaterialCategory::Powder);
    assert!(
        sla.is_empty(),
        "no SLA preset is registered by with_defaults"
    );
    assert!(
        powder.is_empty(),
        "no Powder preset is registered by with_defaults"
    );
    assert_ne!(sla.len(), db.len());
}

// ===========================================================================
// F. is_fdm / is_sheet_metal: both-false is possible, both-true is not.
// ===========================================================================

#[test]
fn is_fdm_and_is_sheet_metal_are_never_both_true() {
    for m in FilamentDb::with_defaults().iter() {
        assert!(
            !(m.is_fdm() && m.is_sheet_metal()),
            "{}: category cannot satisfy both predicates",
            m.name
        );
    }
}

#[test]
fn is_fdm_and_is_sheet_metal_are_both_false_for_sla_and_powder() {
    // No default preset exists for Sla/Powder, so build one by hand to
    // reach the category that neither predicate recognises.
    let mut sla = MaterialProperties::pla();
    sla.category = MaterialCategory::Sla;
    assert!(!sla.is_fdm());
    assert!(!sla.is_sheet_metal());

    let mut powder = MaterialProperties::pla();
    powder.category = MaterialCategory::Powder;
    assert!(!powder.is_fdm());
    assert!(!powder.is_sheet_metal());
}

// ===========================================================================
// G. Extreme values: yield_pa / youngs_pa / density_si wrap, not panic.
// ===========================================================================

/// Independent of `Fix128::mul`: for two *pure-integer* operands (`lo == 0`
/// on both sides), the documented 128x128->128 middle-bits product collapses
/// to `hi = low 64 bits of the exact i128 product a*b, reinterpreted as
/// signed`, `lo = 0` (every cross term that involves a zero `lo` vanishes).
fn pure_int_mul_oracle(a: i64, b: i64) -> Fix128 {
    let product: i128 = i128::from(a) * i128::from(b);
    let low64 = (product as u128 & u128::from(u64::MAX)) as u64;
    Fix128::from_raw(low64 as i64, 0)
}

#[test]
fn pure_int_mul_oracle_matches_fix128_mul_for_non_overflowing_inputs() {
    // Sanity: the independent oracle must agree with Fix128::mul when the
    // product plainly fits (no wrap in play yet).
    assert_eq!(pure_int_mul_oracle(7, 6), Fix128::from_int(42));
    assert_eq!(
        pure_int_mul_oracle(7, 6),
        Fix128::from_int(7) * Fix128::from_int(6)
    );
    assert_eq!(
        pure_int_mul_oracle(50, 1_000_000),
        Fix128::from_int(50) * MPA_TO_PA
    );
}

#[test]
fn yield_pa_wraps_per_the_documented_mul_contract_not_panic() {
    // yield_strength_mpa = 10^13: 10^13 * 1_000_000 (MPA_TO_PA) = 10^19,
    // which exceeds i64::MAX (~9.223e18) and wraps.
    let huge_mpa: i64 = 10_000_000_000_000;
    let mut m = MaterialProperties::pla();
    m.yield_strength_mpa = Fix128::from_int(huge_mpa);
    let expected = pure_int_mul_oracle(huge_mpa, 1_000_000);
    // Confirm this input genuinely overflows (the oracle disagrees with the
    // true mathematical product), otherwise the "extreme" case would be
    // vacuous.
    let true_product_fits = i128::from(huge_mpa) * 1_000_000 <= i128::from(i64::MAX);
    assert!(!true_product_fits, "the test input must actually overflow");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.yield_pa()));
    let got = result.unwrap_or_else(|e| panic!("yield_pa must not panic on overflow: {e:?}"));
    assert_eq!(
        got, expected,
        "yield_pa must wrap per the documented Mul contract"
    );
    // Determinism: calling again reproduces the identical wrapped bits.
    assert_eq!(m.yield_pa(), got);
}

#[test]
fn youngs_pa_wraps_per_the_documented_mul_contract_not_panic() {
    // youngs_modulus_gpa = 10^13: 10^13 * 1_000_000_000 (GPA_TO_PA) = 10^22.
    let huge_gpa: i64 = 10_000_000_000_000;
    let mut m = MaterialProperties::pla();
    m.youngs_modulus_gpa = Fix128::from_int(huge_gpa);
    let expected = pure_int_mul_oracle(huge_gpa, 1_000_000_000);
    let true_product_fits = i128::from(huge_gpa) * 1_000_000_000 <= i128::from(i64::MAX);
    assert!(!true_product_fits, "the test input must actually overflow");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.youngs_pa()));
    let got = result.unwrap_or_else(|e| panic!("youngs_pa must not panic on overflow: {e:?}"));
    assert_eq!(
        got, expected,
        "youngs_pa must wrap per the documented Mul contract"
    );
    assert_eq!(m.youngs_pa(), got);
}

#[test]
fn density_si_wraps_per_the_documented_mul_contract_not_panic() {
    // density_g_cm3 = 10^16: 10^16 * 1_000 (G_CM3_TO_KG_M3) = 10^19.
    let huge_density: i64 = 10_000_000_000_000_000;
    let mut m = MaterialProperties::pla();
    m.density_g_cm3 = Fix128::from_int(huge_density);
    let expected = pure_int_mul_oracle(huge_density, 1_000);
    let true_product_fits = i128::from(huge_density) * 1_000 <= i128::from(i64::MAX);
    assert!(!true_product_fits, "the test input must actually overflow");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.density_si()));
    let got = result.unwrap_or_else(|e| panic!("density_si must not panic on overflow: {e:?}"));
    assert_eq!(
        got, expected,
        "density_si must wrap per the documented Mul contract"
    );
    assert_eq!(m.density_si(), got);
}

// ===========================================================================
// H. Determinism across calls (new argument coverage: category is not a
//    default-only parameter -- Sla/Powder exercise non-default scenes above).
// ===========================================================================

#[test]
fn filament_db_register_and_get_round_trip_every_preset() {
    let db = FilamentDb::with_defaults();
    for m in db.iter() {
        let fetched = db.get(m.id).expect("registered id must resolve");
        assert_eq!(fetched.name, m.name);
        assert_eq!(fetched.category, m.category);
    }
}
