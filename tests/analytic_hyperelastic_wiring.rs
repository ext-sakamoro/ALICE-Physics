//! Oracles for the production entry points of `alice_physics::hyperelastic`
//! driven by `examples/hyperelastic_material_presets.rs`: `Stretch::{UNITY,
//! uniaxial, equibiaxial, i1, i2, volume_ratio}`, `HyperelasticModel::{
//! tpu_soft, silicone_soft, natural_rubber}`, `strain_energy_density`,
//! `uniaxial_cauchy_stress` and `small_strain_shear_modulus`.
//!
//! # What this file is and is not
//!
//! This is a **separate, standalone closed-form library** from the
//! tensor-valued FEM integration in `linear_elastic_fem.rs` /
//! `cubic_elastic_fem.rs` / `quadratic_elastic_fem.rs` (see
//! `tests/analytic_quadratic_hyperelastic.rs` / `analytic_cubic_hyperelastic.rs`
//! / `analytic_hyperelastic_degenerate.rs` for that oracle work, and the
//! module's own `#[cfg(test)]` block for `cauchy_stress` / `tangent_constants`
//! / `volumetric_modulus`, none of which are touched here). Those three FEM
//! modules call `HyperelasticModel`, `cauchy_stress`, `volumetric_modulus` and
//! `tangent_constants` directly on a deformation gradient `F`; they never
//! construct a `Stretch` or call a preset constructor, so the twelve items
//! above had zero production callers before
//! `examples/hyperelastic_material_presets.rs` existed.
//!
//! # Closed forms and where they come from
//!
//! * **Uniaxial** (`Stretch::uniaxial(λ)`): incompressibility forces
//!   `λ₂ = λ₃ = 1/√λ` (the module's own doc comment), so by hand
//!   `I₁ = λ² + 2/λ` and `I₂ = 1/λ² + 2λ` (the standard incompressible-uniaxial
//!   invariants; `I₂ = 1/λ₁² + 1/λ₂² + 1/λ₃² = 1/λ² + λ + λ`). `J = λ·λ₂·λ₃ = 1`
//!   exactly in real arithmetic; in `Fix128` it carries the rounding of one
//!   `sqrt` and is checked to a tolerance, same as the module's own
//!   `stretch_uniaxial_preserves_volume` test.
//! * **Equibiaxial** (`Stretch::equibiaxial(λ)`): `λ₁ = λ₂ = λ`,
//!   `λ₃ = 1/λ²` (module doc), so `I₁ = 2λ² + 1/λ⁴` and `I₂ = 2/λ² + λ⁴`.
//! * **Strain energy** (module doc on [`HyperelasticModel`]):
//!   `W_NH = (μ/2)(I₁−3)`, `W_MR = C₁(I₁−3) + C₂(I₂−3)`,
//!   `W_Yeoh = C₁d + C₂d² + C₃d³` with `d = I₁ − 3`.
//! * **Uniaxial Cauchy stress** (module doc on [`uniaxial_cauchy_stress`]):
//!   `σ_NH = μ·(λ²−1/λ)`, `σ_MR = 2(C₁+C₂/λ)(λ²−1/λ)`,
//!   `σ_Yeoh = 2(λ²−1/λ)(C₁+2C₂d+3C₃d²)` — each is the derivative of its `W`
//!   under the incompressibility constraint, reproduced here from the strain
//!   energy formula above rather than from the function under test.
//! * **Small-strain shear modulus**: the `λ → 1` limit of
//!   `uniaxial_cauchy_stress`. By hand: `dσ/dλ|_{λ=1} = 3·μ` (Neo-Hookean),
//!   `= 6(C₁+C₂)` (Mooney-Rivlin, product rule on
//!   `2(C₁+C₂/λ)(λ²−1/λ)` at `λ=1`) and `= 6·C₁` (Yeoh — the `d`-dependent
//!   factor multiplies `(λ²−1/λ)`, which is itself zero at `λ=1`, so only the
//!   `d=0` term of the other factor survives). Each is `3× ` the module's
//!   documented closed form for `small_strain_shear_modulus` (`μ`,
//!   `2(C₁+C₂)`, `2C₁`), which is the independent check
//!   `small_strain_limit_matches_three_times_the_uniaxial_stress_slope` uses:
//!   it calls `uniaxial_cauchy_stress` at `λ = 1 ± ε` (the function under
//!   test) only to confirm the hand-derived `3×` identity above, not to
//!   manufacture the oracle itself — the primary oracles
//!   (`small_strain_shear_modulus_matches_the_documented_closed_form_per_preset`)
//!   never call `small_strain_shear_modulus`.
//!
//! # Degenerate input
//!
//! `uniaxial` / `equibiaxial` collapse to `Stretch::UNITY` for `λ ≤ 0` (module
//! doc); a material with `μ = 0` carries zero energy and zero stress at every
//! stretch; a stretch so large that `Fix128`'s wrapping multiply overflows the
//! ±2⁶³ integer range (`λ = 2³²`) is **not refused** — `Stretch::uniaxial`
//! and `uniaxial_cauchy_stress` silently wrap to a small, wrong-sign value
//! instead of the astronomically large one real arithmetic would give, and
//! `Stretch::equibiaxial` hits its own `l2.is_zero()` guard (meant for `λ = 0`)
//! by accident and returns `UNITY`. All three are measured and pinned exactly,
//! not just checked for "did not panic" (`Fix128` arithmetic never panics —
//! `Mul` wraps, `Div` by zero returns `ZERO`, `sqrt` of a non-positive value
//! returns `ZERO` — so "ran to completion" is a vacuous assertion here; see
//! `rules/analytic-oracle-tests.md` on overflow paths).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::hyperelastic::{
    small_strain_shear_modulus, strain_energy_density, uniaxial_cauchy_stress, HyperelasticModel,
    Stretch,
};
use alice_physics::math::Fix128;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// Loose tolerance for comparisons that cross a `sqrt` (the module's own
/// `approx_eq` in its `#[cfg(test)]` block uses the same 1/1000).
fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
    let d = if a > b { a - b } else { b - a };
    d <= tol
}

fn loose_tol() -> Fix128 {
    Fix128::from_ratio(1, 1000)
}

// ============================================================================
// Section 1: Stretch — UNITY, uniaxial, equibiaxial, i1, i2, volume_ratio
// ============================================================================

#[test]
fn unity_is_the_undeformed_state() {
    assert_eq!(Stretch::UNITY.l1, Fix128::ONE, "UNITY.l1");
    assert_eq!(Stretch::UNITY.l2, Fix128::ONE, "UNITY.l2");
    assert_eq!(Stretch::UNITY.l3, Fix128::ONE, "UNITY.l3");
    assert_eq!(Stretch::UNITY.i1(), Fix128::from_int(3), "I1(UNITY) = 3");
    assert_eq!(Stretch::UNITY.i2(), Fix128::from_int(3), "I2(UNITY) = 3");
    assert_eq!(
        Stretch::UNITY.volume_ratio(),
        Fix128::ONE,
        "J(UNITY) = 1 exactly (pure integer product, no sqrt)"
    );
}

/// `I1 = λ² + 2/λ`, `I2 = 1/λ² + 2λ` by hand from `λ₂ = λ₃ = 1/√λ`.
#[test]
fn uniaxial_matches_the_closed_form_invariants() {
    for lambda in [fx(1.0), fx(1.2), fx(2.0), fx(3.0), fx(0.5)] {
        let s = Stretch::uniaxial(lambda);
        let lam2 = lambda * lambda;
        let inv_lam = Fix128::ONE / lambda;
        let want_i1 = lam2 + inv_lam + inv_lam;
        let want_i2 = Fix128::ONE / lam2 + lambda + lambda;
        assert!(
            approx_eq(s.i1(), want_i1, loose_tol()),
            "lambda={}: I1 got {}, want {}",
            lambda.to_f64(),
            s.i1().to_f64(),
            want_i1.to_f64()
        );
        assert!(
            approx_eq(s.i2(), want_i2, loose_tol()),
            "lambda={}: I2 got {}, want {}",
            lambda.to_f64(),
            s.i2().to_f64(),
            want_i2.to_f64()
        );
    }
}

/// `I1 = 2λ² + 1/λ⁴`, `I2 = 2/λ² + λ⁴` by hand from `λ₁=λ₂=λ`, `λ₃ = 1/λ²`.
#[test]
fn equibiaxial_matches_the_closed_form_invariants() {
    for lambda in [fx(1.0), fx(1.2), fx(2.0), fx(1.5)] {
        let s = Stretch::equibiaxial(lambda);
        let lam2 = lambda * lambda;
        let lam4 = lam2 * lam2;
        let want_i1 = lam2 + lam2 + Fix128::ONE / lam4;
        let want_i2 = Fix128::ONE / lam2 + Fix128::ONE / lam2 + lam4;
        assert!(
            approx_eq(s.i1(), want_i1, loose_tol()),
            "lambda={}: I1 got {}, want {}",
            lambda.to_f64(),
            s.i1().to_f64(),
            want_i1.to_f64()
        );
        assert!(
            approx_eq(s.i2(), want_i2, loose_tol()),
            "lambda={}: I2 got {}, want {}",
            lambda.to_f64(),
            s.i2().to_f64(),
            want_i2.to_f64()
        );
    }
}

#[test]
fn uniaxial_and_equibiaxial_preserve_volume_near_unity() {
    for lambda in [fx(1.3), fx(2.0), fx(0.6)] {
        let u = Stretch::uniaxial(lambda).volume_ratio();
        let b = Stretch::equibiaxial(lambda).volume_ratio();
        assert!(
            approx_eq(u, Fix128::ONE, loose_tol()),
            "uniaxial J at lambda={} = {}",
            lambda.to_f64(),
            u.to_f64()
        );
        assert!(
            approx_eq(b, Fix128::ONE, loose_tol()),
            "equibiaxial J at lambda={} = {}",
            lambda.to_f64(),
            b.to_f64()
        );
    }
}

/// A compressible `Stretch` the constructors never produce (incompressibility
/// is baked into `uniaxial`/`equibiaxial`, not into the struct itself): the
/// three fields are public, so this is a plain, bit-exact integer product
/// with no `sqrt` anywhere — `volume_ratio` must return it exactly.
#[test]
fn volume_ratio_of_a_hand_built_compressible_stretch_is_the_exact_product() {
    let s = Stretch {
        l1: Fix128::from_int(2),
        l2: Fix128::from_int(3),
        l3: Fix128::from_int(5),
    };
    assert_eq!(s.volume_ratio(), Fix128::from_int(30));
    assert_eq!(s.i1(), Fix128::from_int(4 + 9 + 25));
}

/// Two distinct inputs must give two distinct outputs — catches a wiring
/// mutation that drops the argument and always returns `UNITY` or a constant.
#[test]
fn uniaxial_and_equibiaxial_are_injective_on_distinct_positive_lambdas() {
    let lambdas = [fx(1.0), fx(1.5), fx(2.0), fx(3.0)];
    let uni: Vec<Fix128> = lambdas.iter().map(|&l| Stretch::uniaxial(l).l1).collect();
    let bia: Vec<Fix128> = lambdas
        .iter()
        .map(|&l| Stretch::equibiaxial(l).l3)
        .collect();
    for i in 0..uni.len() {
        for j in (i + 1)..uni.len() {
            assert_ne!(uni[i], uni[j], "uniaxial collapsed distinct lambdas");
            assert_ne!(bia[i], bia[j], "equibiaxial collapsed distinct lambdas");
        }
    }
}

// ============================================================================
// Section 2: presets + small_strain_shear_modulus (closed form, not calling
// small_strain_shear_modulus to derive the expectation)
// ============================================================================

#[test]
fn tpu_soft_is_neo_hookean_mu_3() {
    match HyperelasticModel::tpu_soft() {
        HyperelasticModel::NeoHookean { mu_mpa } => {
            assert_eq!(mu_mpa, Fix128::from_int(3));
            assert_eq!(
                small_strain_shear_modulus(&HyperelasticModel::tpu_soft()),
                mu_mpa
            );
        }
        other => panic!("tpu_soft is not Neo-Hookean: {other:?}"),
    }
}

/// ⚠️ **Mutation survivor, classified as unobservable, not an oracle gap.**
/// A 1-ULP change to `c2_mpa`'s raw mantissa (±5.4e-20, `Fix128`'s own
/// precision floor) survives the `approx_eq(..., loose_tol())` checks below
/// *and* survives `small_strain_shear_modulus`'s bit-exact check (the
/// doubled sum still rounds the same way) — because `Fix128::from_f64(0.1)`
/// and `Fix128::from_f64(0.05)` do **not** reproduce the module's hand-tuned
/// hex literals bit for bit either (checked directly: both differ from the
/// stored constant by a handful of ULPs), there is no independently-derived
/// bit-exact reference to pin against; the preset's hex *is* the
/// specification, not a computation this test can re-derive. The loose
/// tolerance is intentional for an engineering material constant at MPa
/// scale.
#[test]
fn silicone_soft_is_mooney_rivlin_and_shear_modulus_is_2x_c1_plus_c2() {
    let model = HyperelasticModel::silicone_soft();
    match model {
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => {
            let want = (c1_mpa + c2_mpa).double();
            assert_eq!(small_strain_shear_modulus(&model), want);
            // sanity on the documented magnitudes (~0.1, ~0.05 MPa).
            assert!(approx_eq(c1_mpa, fx(0.1), loose_tol()));
            assert!(approx_eq(c2_mpa, fx(0.05), loose_tol()));
        }
        other => panic!("silicone_soft is not Mooney-Rivlin: {other:?}"),
    }
}

#[test]
fn natural_rubber_is_yeoh_and_shear_modulus_is_2x_c1() {
    let model = HyperelasticModel::natural_rubber();
    match model {
        HyperelasticModel::Yeoh { c1_mpa, .. } => {
            let want = c1_mpa.double();
            assert_eq!(small_strain_shear_modulus(&model), want);
            assert!(approx_eq(c1_mpa, fx(0.5), loose_tol()));
        }
        other => panic!("natural_rubber is not Yeoh: {other:?}"),
    }
}

#[test]
fn the_three_presets_have_distinct_shear_moduli() {
    let mu = [
        small_strain_shear_modulus(&HyperelasticModel::tpu_soft()),
        small_strain_shear_modulus(&HyperelasticModel::silicone_soft()),
        small_strain_shear_modulus(&HyperelasticModel::natural_rubber()),
    ];
    assert_ne!(mu[0], mu[1]);
    assert_ne!(mu[1], mu[2]);
    assert_ne!(mu[0], mu[2]);
}

// ============================================================================
// Section 3: strain_energy_density + uniaxial_cauchy_stress closed forms
// across all three presets and several stretches
// ============================================================================

fn energy_closed_form(model: &HyperelasticModel, i1: Fix128, i2: Fix128) -> Fix128 {
    let three = Fix128::from_int(3);
    match model {
        HyperelasticModel::NeoHookean { mu_mpa } => {
            *mu_mpa * (i1 - three) * Fix128::from_ratio(1, 2)
        }
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => {
            *c1_mpa * (i1 - three) + *c2_mpa * (i2 - three)
        }
        HyperelasticModel::Yeoh {
            c1_mpa,
            c2_mpa,
            c3_mpa,
        } => {
            let d = i1 - three;
            *c1_mpa * d + *c2_mpa * d * d + *c3_mpa * d * d * d
        }
    }
}

fn stress_closed_form(model: &HyperelasticModel, lambda: Fix128) -> Fix128 {
    let lam2 = lambda * lambda;
    let inv_lam = Fix128::ONE / lambda;
    let base = lam2 - inv_lam;
    match model {
        HyperelasticModel::NeoHookean { mu_mpa } => *mu_mpa * base,
        HyperelasticModel::MooneyRivlin { c1_mpa, c2_mpa } => {
            let mix = *c1_mpa + *c2_mpa * inv_lam;
            Fix128::from_int(2) * mix * base
        }
        HyperelasticModel::Yeoh {
            c1_mpa,
            c2_mpa,
            c3_mpa,
        } => {
            let i1 = lam2 + inv_lam + inv_lam;
            let d = i1 - Fix128::from_int(3);
            let dw =
                *c1_mpa + Fix128::from_int(2) * *c2_mpa * d + Fix128::from_int(3) * *c3_mpa * d * d;
            Fix128::from_int(2) * dw * base
        }
    }
}

#[test]
fn strain_energy_density_matches_the_closed_form_for_every_preset_and_stretch() {
    for model in [
        HyperelasticModel::tpu_soft(),
        HyperelasticModel::silicone_soft(),
        HyperelasticModel::natural_rubber(),
    ] {
        for lambda in [fx(1.0), fx(1.2), fx(2.0), fx(3.0)] {
            let s = Stretch::uniaxial(lambda);
            let want = energy_closed_form(&model, s.i1(), s.i2());
            let got = strain_energy_density(&model, &s);
            assert!(
                approx_eq(got, want, loose_tol()),
                "{model:?} at lambda={}: W got {}, want {}",
                lambda.to_f64(),
                got.to_f64(),
                want.to_f64()
            );
        }
    }
}

#[test]
fn uniaxial_cauchy_stress_matches_the_closed_form_for_every_preset_and_stretch() {
    for model in [
        HyperelasticModel::tpu_soft(),
        HyperelasticModel::silicone_soft(),
        HyperelasticModel::natural_rubber(),
    ] {
        for lambda in [fx(1.0), fx(1.2), fx(2.0), fx(3.0), fx(0.6)] {
            let want = stress_closed_form(&model, lambda);
            let got = uniaxial_cauchy_stress(&model, lambda);
            assert!(
                approx_eq(got, want, loose_tol()),
                "{model:?} at lambda={}: sigma got {}, want {}",
                lambda.to_f64(),
                got.to_f64(),
                want.to_f64()
            );
        }
    }
}

/// The documented small-strain limit, read off at `λ = 1 ± ε`: confirms (does
/// not define) the primary oracle above via the hand-derived identity
/// `dσ/dλ|_{λ=1} = 3·μ_0` — see the module doc on this file's own header.
#[test]
fn small_strain_limit_matches_three_times_the_uniaxial_stress_slope() {
    for model in [
        HyperelasticModel::tpu_soft(),
        HyperelasticModel::silicone_soft(),
        HyperelasticModel::natural_rubber(),
    ] {
        let eps = fx(1.0 / 4096.0);
        let s_plus = uniaxial_cauchy_stress(&model, Fix128::ONE + eps).to_f64();
        let s_minus = uniaxial_cauchy_stress(&model, Fix128::ONE - eps).to_f64();
        let slope = (s_plus - s_minus) / (2.0 * eps.to_f64());
        let mu0 = small_strain_shear_modulus(&model).to_f64();
        let rel_err = (slope - 3.0 * mu0).abs() / (3.0 * mu0).max(1e-6);
        assert!(
            rel_err < 0.02,
            "{model:?}: slope={slope}, 3*mu0={}, rel_err={rel_err}",
            3.0 * mu0
        );
    }
}

// ============================================================================
// Section 4: degenerate input
// ============================================================================

/// The guard is on `uniaxial_cauchy_stress` itself, not inherited from
/// `Stretch::uniaxial` (the stress function takes `lambda` directly, never
/// builds a `Stretch`) — a dedicated test here, not reuse of the `Stretch`
/// degenerate test above.
#[test]
fn uniaxial_cauchy_stress_returns_zero_for_nonpositive_lambda() {
    for model in [
        HyperelasticModel::tpu_soft(),
        HyperelasticModel::silicone_soft(),
        HyperelasticModel::natural_rubber(),
    ] {
        for lambda in [Fix128::ZERO, Fix128::from_int(-1), Fix128::from_int(-50)] {
            let got = catch_unwind(AssertUnwindSafe(|| uniaxial_cauchy_stress(&model, lambda)))
                .expect("uniaxial_cauchy_stress must not panic on non-positive lambda");
            assert_eq!(got, Fix128::ZERO, "{model:?} at lambda={}", lambda.to_f64());
        }
    }
}

#[test]
fn uniaxial_and_equibiaxial_collapse_to_unity_for_nonpositive_lambda() {
    for lambda in [Fix128::ZERO, Fix128::from_int(-1), Fix128::from_int(-100)] {
        let u = catch_unwind(AssertUnwindSafe(|| Stretch::uniaxial(lambda)))
            .expect("uniaxial must not panic on non-positive lambda");
        let b = catch_unwind(AssertUnwindSafe(|| Stretch::equibiaxial(lambda)))
            .expect("equibiaxial must not panic on non-positive lambda");
        assert_eq!(u, Stretch::UNITY, "uniaxial({})", lambda.to_f64());
        assert_eq!(b, Stretch::UNITY, "equibiaxial({})", lambda.to_f64());
    }
}

/// A material with zero shear/constants carries no energy and no stress at
/// any stretch, including the overflow case below (`0 * anything == 0` even
/// when the other factor wrapped).
#[test]
fn zero_shear_modulus_material_has_zero_energy_and_zero_stress_everywhere() {
    let model = HyperelasticModel::NeoHookean {
        mu_mpa: Fix128::ZERO,
    };
    for lambda in [fx(1.0), fx(2.0), fx(5.0), Fix128::from_int(1i64 << 32)] {
        let s = Stretch::uniaxial(lambda);
        assert_eq!(strain_energy_density(&model, &s), Fix128::ZERO);
        assert_eq!(uniaxial_cauchy_stress(&model, lambda), Fix128::ZERO);
    }
    assert_eq!(small_strain_shear_modulus(&model), Fix128::ZERO);
}

/// `I2`'s explicit `is_zero()` guard: a stretch with one axis at zero returns
/// `I2 = 0` (refusing the `1/0` it would otherwise need), not a wrapped
/// division artifact. `I1` has no such guard and is just the sum of squares.
#[test]
fn i2_guards_a_zero_stretch_axis_instead_of_dividing_by_zero() {
    let degenerate = Stretch {
        l1: Fix128::ZERO,
        l2: Fix128::ONE,
        l3: Fix128::ONE,
    };
    let got = catch_unwind(AssertUnwindSafe(|| degenerate.i2()))
        .expect("i2 must not panic on a zero stretch axis");
    assert_eq!(got, Fix128::ZERO, "I2 with a zero axis");
    assert_eq!(
        degenerate.i1(),
        Fix128::from_int(2),
        "I1 is unguarded arithmetic: 0+1+1"
    );
}

/// `λ = 2³²`: `λ·λ` overflows `Fix128`'s ±2⁶³ integer range and wraps to
/// exactly zero (`(2³²)² = 2⁶⁴ ≡ 0 mod 2⁶⁴`, reinterpreted as a signed `i64`
/// — the module's `Mul` keeps only the middle 128 bits of the 256-bit
/// product, see `rules/analytic-oracle-tests.md`). `uniaxial` does not guard
/// against this (its guard is `lambda <= ZERO`, which `2³²` is not), so `I1`
/// comes back a few times `2⁻³²` instead of astronomically large, and
/// `equibiaxial`'s own `l2.is_zero()` check — written for the `λ = 0` case —
/// fires on the wrapped value and returns `UNITY` by accident.
#[test]
fn extreme_stretch_wraps_instead_of_growing_without_bound() {
    let lambda = Fix128::from_int(1i64 << 32);

    // The wrap itself, independent of `Stretch`: (2^32)^2 mod 2^64 == 0.
    let lam2 = catch_unwind(AssertUnwindSafe(|| lambda * lambda))
        .expect("Fix128 multiply never panics, wrapping or not");
    assert_eq!(
        lam2,
        Fix128::ZERO,
        "(2^32)^2 wraps to exactly zero in Fix128"
    );

    let uni = catch_unwind(AssertUnwindSafe(|| Stretch::uniaxial(lambda)))
        .expect("uniaxial must not panic on an overflowing lambda");
    // l1 = lambda unchanged (no squaring in the constructor itself).
    assert_eq!(uni.l1, lambda);
    // l2 = l3 = 1/sqrt(2^32) = 1/2^16 = 2^-16 exactly (power of two, no
    // rounding from the integer sqrt).
    let expected_lateral = Fix128::ONE / Fix128::from_int(65536);
    assert_eq!(uni.l2, expected_lateral);
    assert_eq!(uni.l3, expected_lateral);
    // I1 = l1^2 + l2^2 + l3^2 = 0 (wrapped) + 2 * 2^-32, not ~1.8e19.
    let expected_i1 = expected_lateral * expected_lateral + expected_lateral * expected_lateral;
    assert_eq!(
        uni.i1(),
        expected_i1,
        "I1 wrapped to a tiny value, not huge"
    );
    assert!(
        uni.i1().to_f64() < 1.0,
        "I1 at the overflow point must read as small: {}",
        uni.i1().to_f64()
    );

    // uniaxial_cauchy_stress(tpu_soft, 2^32) = mu * (lam2 - 1/lambda)
    //   = 3 * (0 - 2^-32) = -3 * 2^-32: small and the *wrong sign* for a
    // material in tension, not the enormous positive value a real material
    // under this stretch would report.
    let model = HyperelasticModel::tpu_soft();
    let sigma = catch_unwind(AssertUnwindSafe(|| uniaxial_cauchy_stress(&model, lambda)))
        .expect("uniaxial_cauchy_stress must not panic on an overflowing lambda");
    let inv_lambda = Fix128::ONE / lambda;
    let expected_sigma = Fix128::from_int(3) * (lam2 - inv_lambda);
    assert_eq!(sigma, expected_sigma);
    assert!(
        sigma < Fix128::ZERO,
        "wrapped uniaxial stress came back negative (wrong sign): {}",
        sigma.to_f64()
    );

    // equibiaxial's l2-is-zero guard (meant for lambda <= 0) fires here too,
    // because the local `l2 = lambda * lambda` wraps to zero.
    let bia = catch_unwind(AssertUnwindSafe(|| Stretch::equibiaxial(lambda)))
        .expect("equibiaxial must not panic on an overflowing lambda");
    assert_eq!(
        bia,
        Stretch::UNITY,
        "equibiaxial's zero-stretch guard fires on the wrapped square, not a real zero input"
    );
}
