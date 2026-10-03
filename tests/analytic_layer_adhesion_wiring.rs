//! Independent oracle coverage for `layer_adhesion`'s factor-of-safety API
//! (`EffectiveStrength::{fos_normal_x, fos_normal_z, fos_shear_xy,
//! fos_shear_xz, min_fos}`), one test per item/scenario, plus degenerate and
//! extreme-magnitude inputs.
//!
//! Every expected value here is computed from the module's documented
//! closed form (`src/layer_adhesion.rs:94-105`) using the published material
//! constants in `src/filament_db.rs`, independently of calling the function
//! under test for the expected side:
//!
//! ```text
//! allowables (for_material):
//!     normal_x = normal_y = sigma_y
//!     normal_z = sigma_y * anisotropy_z_ratio
//!     shear_xy = sigma_y * 0.6
//!     shear_yz = shear_xz = shear_xy * (1 + anisotropy_z_ratio) / 2
//!
//! component_fos(applied, allowable):
//!     if applied == 0          -> sentinel = i64::MAX >> 8
//!     else if allowable == 0   -> 0
//!     else                      -> allowable / |applied|
//!
//! min_fos(pairs) -> min over pairs of component_fos(applied, allowable)
//! ```
//!
//! Nothing here touches `src/`.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::filament_db::MaterialProperties;
use alice_physics::layer_adhesion::{EffectiveStrength, PrintOrientation};
use alice_physics::math::Fix128;

/// `|a - b| <= 2^-48`: agreement within the truncation noise of the I64F64
/// format (products/quotients of non-dyadic ratios carry a few thousand ulp
/// depending on evaluation order), same tolerance as
/// `tests/engineering_oracles_misc.rs::fix_close`.
fn fix_close(a: Fix128, b: Fix128) -> bool {
    (a - b).abs() <= Fix128::from_raw(0, 1 << 16)
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// `component_fos`'s sentinel for zero applied stress
/// (`src/layer_adhesion.rs:99`, `i64::MAX >> 8`), written down independently.
fn sentinel() -> Fix128 {
    Fix128::from_int(i64::MAX >> 8)
}

/// (name, yield_strength_mpa, anisotropy_z_ratio) for the four presets this
/// file exercises, copied from `src/filament_db.rs` doc comments/literals
/// rather than read off a constructed `MaterialProperties`.
fn presets() -> [(&'static str, MaterialProperties, i64, i64, i64); 4] {
    [
        ("PLA", MaterialProperties::pla(), 50, 65, 100),
        ("TPU", MaterialProperties::tpu(), 30, 90, 100),
        ("CF-Nylon", MaterialProperties::cf_nylon(), 85, 50, 100),
        ("SUS304", MaterialProperties::sus304(), 215, 1, 1), // r = 1/1 = isotropic
    ]
}

// ============================================================================
// fos_normal_x
// ============================================================================

#[test]
fn fos_normal_x_equals_allowable_over_applied_magnitude() {
    for (name, m, sigma, _r_num, _r_den) in presets() {
        let s = EffectiveStrength::for_material(&m, PrintOrientation::XYFlat);
        let allow_x = Fix128::from_int(sigma); // normal_x = normal_y = sigma_y, exact
        for applied in [sigma / 2, sigma, 2 * sigma] {
            let got = s.fos_normal_x(Fix128::from_int(applied));
            let want = allow_x / Fix128::from_int(applied);
            assert!(
                fix_close(got, want),
                "{name}: fos_normal_x({applied}) = {got:?}, want {want:?}"
            );
        }
    }
}

// ============================================================================
// fos_normal_z — the critical layer-bond direction
// ============================================================================

#[test]
fn fos_normal_z_equals_allowable_over_applied_magnitude() {
    for (name, m, sigma, r_num, r_den) in presets() {
        let s = EffectiveStrength::for_material(&m, PrintOrientation::XYFlat);
        let allow_z = Fix128::from_int(sigma) * r(r_num, r_den);
        for applied in [sigma / 2, sigma, 2 * sigma] {
            if applied == 0 {
                continue;
            }
            let got = s.fos_normal_z(Fix128::from_int(applied));
            let want = allow_z / Fix128::from_int(applied);
            assert!(
                fix_close(got, want),
                "{name}: fos_normal_z({applied}) = {got:?}, want {want:?}"
            );
        }
    }
    // Isotropic limit (r = 1): normal_z's FoS equals normal_x's FoS for the
    // same applied stress -- sheet metal has no weak direction.
    let sus =
        EffectiveStrength::for_material(&MaterialProperties::sus304(), PrintOrientation::XYFlat);
    assert_eq!(
        sus.fos_normal_z(Fix128::from_int(40)),
        sus.fos_normal_x(Fix128::from_int(40)),
        "isotropic material: Z and X FoS coincide"
    );
}

// ============================================================================
// fos_shear_xy — within-layer shear
// ============================================================================

#[test]
fn fos_shear_xy_equals_allowable_over_applied_magnitude() {
    for (name, m, sigma, _r_num, _r_den) in presets() {
        let s = EffectiveStrength::for_material(&m, PrintOrientation::XYFlat);
        let tau_xy = Fix128::from_int(sigma) * r(6, 10);
        for applied in [sigma / 3, sigma] {
            if applied == 0 {
                continue;
            }
            let got = s.fos_shear_xy(Fix128::from_int(applied));
            let want = tau_xy / Fix128::from_int(applied);
            assert!(
                fix_close(got, want),
                "{name}: fos_shear_xy({applied}) = {got:?}, want {want:?}"
            );
        }
    }
}

// ============================================================================
// fos_shear_xz — across-layer shear
// ============================================================================

#[test]
fn fos_shear_xz_equals_allowable_over_applied_magnitude() {
    for (name, m, sigma, r_num, r_den) in presets() {
        let s = EffectiveStrength::for_material(&m, PrintOrientation::XYFlat);
        let tau_xy = Fix128::from_int(sigma) * r(6, 10);
        let inter = (Fix128::ONE + r(r_num, r_den)) * r(1, 2);
        let tau_xz = tau_xy * inter;
        for applied in [sigma / 5, sigma] {
            if applied == 0 {
                continue;
            }
            let got = s.fos_shear_xz(Fix128::from_int(applied));
            let want = tau_xz / Fix128::from_int(applied);
            assert!(
                fix_close(got, want),
                "{name}: fos_shear_xz({applied}) = {got:?}, want {want:?}"
            );
        }
    }
    // Isotropic limit: inter = 1 -> XZ and XY allowables (and therefore
    // their FoS for equal load) coincide.
    let sus =
        EffectiveStrength::for_material(&MaterialProperties::sus304(), PrintOrientation::XYFlat);
    assert_eq!(sus.shear_xz_mpa, sus.shear_xy_mpa);
    assert_eq!(
        sus.fos_shear_xz(Fix128::from_int(37)),
        sus.fos_shear_xy(Fix128::from_int(37))
    );
}

// ============================================================================
// min_fos — componentwise minimum
// ============================================================================

#[test]
fn min_fos_equals_componentwise_minimum_of_independent_calc() {
    let pla = MaterialProperties::pla();
    let s = EffectiveStrength::for_material(&pla, PrintOrientation::XYFlat);

    // PLA: allow_x = 50, allow_z = 32.5, tau_xy = 30, tau_xz = 24.75
    // (independently from sigma=50, r=0.65, formulas above).
    let allow_x = Fix128::from_int(50);
    let allow_z = Fix128::from_int(50) * r(65, 100);
    let tau_xy = Fix128::from_int(50) * r(6, 10);
    let tau_xz = tau_xy * ((Fix128::ONE + r(65, 100)) * r(1, 2));

    let stress = [
        (Fix128::from_int(10), s.normal_x_mpa),
        (Fix128::from_int(10), s.normal_y_mpa),
        (Fix128::from_int(28), s.normal_z_mpa), // worst: 32.5/28 ~= 1.1607
        (Fix128::from_int(-5), s.shear_xy_mpa), // 30/5 = 6
        (Fix128::ZERO, s.shear_yz_mpa),         // sentinel, not the min
        (Fix128::ZERO, s.shear_xz_mpa),         // sentinel, not the min
    ];

    let candidates = [
        allow_x / Fix128::from_int(10),
        allow_x / Fix128::from_int(10),
        allow_z / Fix128::from_int(28),
        tau_xy / Fix128::from_int(5),
        sentinel(),
        sentinel(),
    ];
    let want = candidates
        .into_iter()
        .fold(sentinel(), |acc, c| if c < acc { c } else { acc });
    // Sanity: the independently-built "want" says the Z component wins.
    assert!(fix_close(want, allow_z / Fix128::from_int(28)));

    let got = s.min_fos(&stress);
    assert!(
        fix_close(got, want),
        "min_fos = {got:?}, independent componentwise min = {want:?}"
    );
    let _ = tau_xz; // not the winner here, computed for completeness/documentation
}

#[test]
fn min_fos_all_zero_applied_returns_sentinel_not_smallest_allowable() {
    // Every component unloaded: min_fos must report the sentinel (safe),
    // not e.g. the smallest *allowable* -- a plausible but wrong
    // alternative implementation would return min(allowables) here.
    let pla = MaterialProperties::pla();
    let s = EffectiveStrength::for_material(&pla, PrintOrientation::XYFlat);
    let none = [
        (Fix128::ZERO, s.normal_x_mpa),
        (Fix128::ZERO, s.normal_y_mpa),
        (Fix128::ZERO, s.normal_z_mpa),
        (Fix128::ZERO, s.shear_xy_mpa),
        (Fix128::ZERO, s.shear_yz_mpa),
        (Fix128::ZERO, s.shear_xz_mpa),
    ];
    assert_eq!(s.min_fos(&none), sentinel());
}

// ============================================================================
// Degenerate inputs
// ============================================================================

#[test]
fn fos_zero_applied_stress_returns_sentinel_for_all_four_components() {
    let pla = MaterialProperties::pla();
    let s = EffectiveStrength::for_material(&pla, PrintOrientation::XYFlat);
    assert_eq!(s.fos_normal_x(Fix128::ZERO), sentinel());
    assert_eq!(s.fos_normal_z(Fix128::ZERO), sentinel());
    assert_eq!(s.fos_shear_xy(Fix128::ZERO), sentinel());
    assert_eq!(s.fos_shear_xz(Fix128::ZERO), sentinel());
}

#[test]
fn fos_zero_allowable_with_nonzero_applied_returns_zero_exactly() {
    // Zero allowable is not reachable through `for_material` (every
    // preset's yield strength and anisotropy ratio are positive), so build
    // it by hand via the all-public-field struct -- this only constructs
    // the *input*, the expected ZERO is read off the documented contract
    // at src/layer_adhesion.rs:101-103, not produced by calling the method.
    let zero_allow = EffectiveStrength {
        normal_x_mpa: Fix128::ZERO,
        normal_y_mpa: Fix128::ZERO,
        normal_z_mpa: Fix128::ZERO,
        shear_xy_mpa: Fix128::ZERO,
        shear_yz_mpa: Fix128::ZERO,
        shear_xz_mpa: Fix128::ZERO,
    };
    assert_eq!(zero_allow.fos_normal_x(Fix128::from_int(5)), Fix128::ZERO);
    assert_eq!(zero_allow.fos_normal_z(Fix128::from_int(5)), Fix128::ZERO);
    assert_eq!(zero_allow.fos_shear_xy(Fix128::from_int(5)), Fix128::ZERO);
    assert_eq!(zero_allow.fos_shear_xz(Fix128::from_int(5)), Fix128::ZERO);
}

#[test]
fn fos_zero_applied_and_zero_allowable_returns_sentinel_not_zero() {
    // `component_fos` checks `applied == 0` *before* `allowable == 0`
    // (src/layer_adhesion.rs:97-100). A 0/0 ambiguous case therefore
    // resolves to the "infinite" sentinel, not ZERO -- an order-dependent
    // detail worth pinning explicitly rather than assuming either answer.
    let zero_allow = EffectiveStrength {
        normal_x_mpa: Fix128::ZERO,
        normal_y_mpa: Fix128::ZERO,
        normal_z_mpa: Fix128::ZERO,
        shear_xy_mpa: Fix128::ZERO,
        shear_yz_mpa: Fix128::ZERO,
        shear_xz_mpa: Fix128::ZERO,
    };
    assert_eq!(zero_allow.fos_normal_x(Fix128::ZERO), sentinel());
    assert_eq!(zero_allow.fos_normal_z(Fix128::ZERO), sentinel());
    assert_eq!(zero_allow.fos_shear_xy(Fix128::ZERO), sentinel());
    assert_eq!(zero_allow.fos_shear_xz(Fix128::ZERO), sentinel());
}

#[test]
fn fos_sign_insensitive_to_applied_stress_sign() {
    // Every FoS reads |applied|: compressive (-) and tensile (+) stress of
    // the same magnitude must give identical factors of safety.
    let pla = MaterialProperties::pla();
    let s = EffectiveStrength::for_material(&pla, PrintOrientation::XYFlat);
    for applied in [7_i64, 19, 50] {
        let pos = Fix128::from_int(applied);
        let neg = Fix128::from_int(-applied);
        assert_eq!(s.fos_normal_x(pos), s.fos_normal_x(neg));
        assert_eq!(s.fos_normal_z(pos), s.fos_normal_z(neg));
        assert_eq!(s.fos_shear_xy(pos), s.fos_shear_xy(neg));
        assert_eq!(s.fos_shear_xz(pos), s.fos_shear_xz(neg));
    }
}

// ============================================================================
// Extreme Fix128 magnitudes
// ============================================================================

#[test]
fn fos_normal_x_extreme_large_applied_stress_gives_tiny_but_exact_fos() {
    // A load far beyond anything physically meaningful, but still well
    // within Fix128's representable range (|value| << 2^63): FoS should be
    // tiny (near failure) and exactly `allowable / applied` in rationals.
    let pla = MaterialProperties::pla();
    let s = EffectiveStrength::for_material(&pla, PrintOrientation::XYFlat);
    let applied = Fix128::from_int(1_000_000_000); // 1e9 MPa
    let want = Fix128::from_int(50) / applied; // 5e-8, representable (>> 2^-64)
    let got = s.fos_normal_x(applied);
    assert!(
        fix_close(got, want),
        "fos_normal_x(1e9) = {got:?}, want {want:?}"
    );
    assert!(got.to_f64() > 0.0, "still a positive, nonzero FoS");
}

#[test]
fn fos_normal_x_extreme_tiny_applied_stress_overflows_and_wraps_to_zero() {
    // The smallest representable positive Fix128 (2^-64) as the applied
    // stress, against an ordinary integer-MPa allowable (PLA: 50 MPa). The
    // true FoS is 50 * 2^64, far beyond Fix128's ~9.2e18 integer range.
    //
    // `Fix128::div` computes the integer quotient in u128 and casts it to
    // `i64` (truncating to the low 64 bits). For *any* integer-valued
    // allowable k (fractional part `lo == 0`), the true quotient is
    // exactly `k * 2^64`, an exact multiple of 2^64 -- so its low 64 bits,
    // and therefore the cast result, are always zero, independent of k.
    // That is derived from the arithmetic alone (k * 2^64 mod 2^64 == 0
    // for every integer k), not by calling the function under test.
    //
    // The division silently reports FoS == 0 for a load that is in truth
    // negligible: the opposite of the correct answer. This test does not
    // assert that is correct -- it pins the current, surprising behaviour
    // (and that it does not panic) so a future overflow-handling fix shows
    // up as a visible, intentional test change rather than a silent one.
    let pla = MaterialProperties::pla();
    let s = EffectiveStrength::for_material(&pla, PrintOrientation::XYFlat);
    let tiny_applied = Fix128::from_raw(0, 1); // 2^-64
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        s.fos_normal_x(tiny_applied)
    }));
    assert!(result.is_ok(), "must not panic on extreme magnitude input");
    assert_eq!(
        result.unwrap(),
        Fix128::ZERO,
        "50 MPa / 2^-64 overflows the i64 integer part and wraps to exactly zero"
    );
}
