//! Oracles for the production entry points of
//! `alice_physics::laminate_failure` driven by
//! `examples/laminate_failure_criteria.rs`: `LaminateStrengths::{cfrp_ud,
//! gfrp_ud}`, `StressState::zero`, `FailureCriterion`, `FailureMode` and the
//! `failure_index` dispatcher.
//!
//! # What this file is and is not
//!
//! `tsai_wu_failure_index` / `tsai_hill_failure_index` / `hashin_failure_mode`
//! / `puck_failure_mode` already have closed-form oracles against the Jones
//! (1999, §2.9) / Hashin (1980) uniaxial-strength envelope in
//! `tests/engineering_oracles_solid.rs::laminate_failure_uniaxial_strengths_lie_on_the_envelope`
//! and `::laminate_failure_hashin_modes_at_uniaxial_strengths` — those are
//! not repeated here. What that file does **not** cover, and this file
//! does:
//!
//! * the `failure_index` dispatcher (no caller anywhere before
//!   `examples/laminate_failure_criteria.rs` existed, so nothing checked
//!   that it actually forwards to the criterion it names rather than a
//!   swapped or hard-coded one),
//! * `LaminateStrengths::{cfrp_ud, gfrp_ud}` against the published
//!   strengths named in the module's own doc comment (T300/5208 CFRP,
//!   E-glass/epoxy GFRP),
//! * `StressState::zero`,
//! * the Tsai–Wu `F₁₂` cross-coupling term (`2·F₁₂·σ1·σ2`), which every
//!   existing test exercises with `σ1 = 0` or `σ2 = 0`, so the term is
//!   always multiplied by zero and never actually checked,
//! * a pure-uniaxial cross-check against the simple 1-D strength ratio for
//!   each axis independently (Tsai–Hill collapses to `(σ/X)²` exactly when
//!   the other stress components are zero — an algebraic identity derived
//!   from the module's own formula, not from calling it), and
//! * two degenerate inputs the existing file does not touch: all-zero
//!   strengths (every `F_ij` coefficient divides by zero) and a stress
//!   magnitude that overflows `Fix128`'s ±2⁶³ integer range.
//!
//! # Degenerate input
//!
//! Zero strengths: every `1/X` and `1/(X·Y)` term in Tsai–Wu / Tsai–Hill /
//! Hashin / Puck divides by `Fix128::ZERO`, which `Fix128::Div`'s documented
//! contract (`src/math.rs`) returns as `ZERO` — not a panic, not `NaN` —
//! regardless of the numerator. Every `F_ij` therefore collapses to zero and
//! every criterion silently reports "safe" no matter how large the stress
//! is; this is measured and pinned exactly below, not just checked for "did
//! not panic".
//!
//! Extreme stress: `σ1 = Fix128::from_int(i64::MAX)` squared overflows the
//! ±2⁶³ integer range. `Fix128::Mul` is unconditionally wrapping (never
//! panics, see `rules/analytic-oracle-tests.md`), and by hand: with
//! `a = b = Fix128{hi: i64::MAX, lo: 0}`, the mixed cross terms (`hi*lo`,
//! `lo*hi`, `lo*lo`) are all zero because `lo = 0` on both operands, and the
//! `hi*hi` term is `(2⁶³−1)² = 2¹²⁶ − 2⁶⁴ + 1`, which fits in `i128` without
//! itself overflowing; truncating that to the low 64 bits (`hi` of the
//! result) gives `2¹²⁶ − 2⁶⁴ + 1 ≡ 1 (mod 2⁶⁴)` (both `2¹²⁶` and `2⁶⁴` are
//! multiples of `2⁶⁴`), so `σ1·σ1` wraps to exactly `Fix128::ONE` — not an
//! astronomically large value. The criteria check their squared terms
//! (AUD-A-S4W2-004), so with strengths all `Fix128::ONE` this stress is
//! reported as failed: Tsai–Wu / Tsai–Hill return the largest Fix128 and
//! Hashin / Puck `FibreTension` (before, the wrapped square made it "exactly
//! borderline", `FI = 1`).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::laminate_failure::{
    failure_index, hashin_failure_mode, puck_failure_mode, tsai_hill_failure_index,
    tsai_wu_failure_index, FailureCriterion, FailureMode, LaminateStrengths, StressState,
};
use alice_physics::math::Fix128;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn stress(s1: f64, s2: f64, t12: f64) -> StressState {
    StressState {
        sigma_1: Fix128::from_f64(s1),
        sigma_2: Fix128::from_f64(s2),
        tau_12: Fix128::from_f64(t12),
    }
}

/// Relative-error assertion against an f64 oracle (Fix128 keeps ~1e-19
/// relative precision for these magnitudes; the slack is for the f64
/// reference computation itself, same convention as
/// `tests/engineering_oracles_solid.rs`).
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = ((g - want) / want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (rel err {err:.3e} > {tol:.1e})"
    );
}

const FIX_TOL: f64 = 1e-9;

// ============================================================================
// Section 1: presets — published strengths from the module's own doc
// comment (T300/5208 CFRP, E-glass/epoxy GFRP).
// ============================================================================

#[test]
fn cfrp_ud_matches_the_published_t300_5208_strengths() {
    let s = LaminateStrengths::cfrp_ud();
    assert_eq!(s.xt, Fix128::from_int(1500), "Xt");
    assert_eq!(s.xc, Fix128::from_int(1500), "Xc");
    assert_eq!(s.yt, Fix128::from_int(40), "Yt");
    assert_eq!(s.yc, Fix128::from_int(246), "Yc");
    assert_eq!(s.s, Fix128::from_int(68), "S");
}

#[test]
fn gfrp_ud_matches_the_published_e_glass_epoxy_strengths() {
    let s = LaminateStrengths::gfrp_ud();
    assert_eq!(s.xt, Fix128::from_int(1080), "Xt");
    assert_eq!(s.xc, Fix128::from_int(620), "Xc");
    assert_eq!(s.yt, Fix128::from_int(39), "Yt");
    assert_eq!(s.yc, Fix128::from_int(128), "Yc");
    assert_eq!(s.s, Fix128::from_int(89), "S");
}

// ============================================================================
// Section 2: StressState::zero + failure_index dispatcher at zero stress.
// At sigma1=sigma2=tau12=0, Tsai-Wu/Tsai-Hill collapse to 0 by inspection of
// the formula (every term has a stress factor) and Hashin/Puck's branch
// conditions are strict inequalities against positive strengths, so they
// are always Safe => dispatcher 0.0.
// ============================================================================

#[test]
fn zero_stress_state_has_all_three_components_zero() {
    let z = StressState::zero();
    assert_eq!(z.sigma_1, Fix128::ZERO);
    assert_eq!(z.sigma_2, Fix128::ZERO);
    assert_eq!(z.tau_12, Fix128::ZERO);
}

#[test]
fn dispatcher_reports_zero_at_zero_stress_for_every_criterion_and_preset() {
    for s in [LaminateStrengths::cfrp_ud(), LaminateStrengths::gfrp_ud()] {
        let z = StressState::zero();
        for criterion in [
            FailureCriterion::TsaiWu,
            FailureCriterion::TsaiHill,
            FailureCriterion::Hashin,
            FailureCriterion::Puck,
        ] {
            assert_eq!(
                failure_index(criterion, s, z),
                Fix128::ZERO,
                "{criterion:?} at zero stress"
            );
        }
    }
}

// ============================================================================
// Section 3: failure_index dispatcher must forward to the criterion it
// names, not a swapped or hard-coded one — the four criteria disagree at
// this loaded, non-degenerate stress state, so a wiring mutation (dropped
// call, swapped match arm) changes the comparison's outcome.
// ============================================================================

#[test]
fn dispatcher_matches_the_direct_criterion_call_it_names() {
    let s = LaminateStrengths::cfrp_ud();
    // Biaxial + shear so every criterion's formula sees a nonzero value on
    // every term (no term is silently multiplied by zero here).
    let loaded = stress(600.0, 15.0, 20.0);

    let direct_tsai_wu = tsai_wu_failure_index(s, loaded);
    let direct_tsai_hill = tsai_hill_failure_index(s, loaded);
    let direct_hashin_mode = hashin_failure_mode(s, loaded);
    let direct_puck_mode = puck_failure_mode(s, loaded);

    assert_eq!(
        failure_index(FailureCriterion::TsaiWu, s, loaded),
        direct_tsai_wu,
        "TsaiWu dispatch"
    );
    assert_eq!(
        failure_index(FailureCriterion::TsaiHill, s, loaded),
        direct_tsai_hill,
        "TsaiHill dispatch"
    );
    let want_hashin = if direct_hashin_mode == FailureMode::Safe {
        Fix128::ZERO
    } else {
        Fix128::ONE
    };
    assert_eq!(
        failure_index(FailureCriterion::Hashin, s, loaded),
        want_hashin,
        "Hashin dispatch (direct mode: {direct_hashin_mode:?})"
    );
    let want_puck = if direct_puck_mode == FailureMode::Safe {
        Fix128::ZERO
    } else {
        Fix128::ONE
    };
    assert_eq!(
        failure_index(FailureCriterion::Puck, s, loaded),
        want_puck,
        "Puck dispatch (direct mode: {direct_puck_mode:?})"
    );

    // The four criteria must actually disagree somewhere at this stress
    // state (otherwise a mutation that collapses the dispatcher to always
    // call the same criterion would not be observable through this test).
    assert_ne!(
        direct_tsai_wu, direct_tsai_hill,
        "TsaiWu and TsaiHill must diverge at this stress state for the dispatcher comparison to be meaningful"
    );
}

// ============================================================================
// Section 4: pure uniaxial cross-check against the simple 1-D strength
// ratio, independent material (GFRP) and fraction (30%) from the existing
// 100%-strength envelope tests in engineering_oracles_solid.rs. By hand:
// with sigma2 = tau12 = 0, Tsai-Hill's cross/b/c terms are all zero
// (cross = sigma1*sigma2/x^2, b = sigma2^2/y^2, c = tau12^2/s^2), leaving
// FI = (sigma1/X)^2 exactly — same for a pure sigma2 or pure tau12 load.
// ============================================================================

#[test]
fn tsai_hill_pure_uniaxial_matches_the_1d_ratio_at_30_percent_gfrp() {
    let s = LaminateStrengths::gfrp_ud();
    let frac = 0.3_f64;

    let xt = s.xt.to_f64();
    let sigma1_tension = stress(frac * xt, 0.0, 0.0);
    assert_rel(
        failure_index(FailureCriterion::TsaiHill, s, sigma1_tension),
        frac * frac,
        FIX_TOL,
        "sigma1 = 0.3*Xt",
    );

    let xc = s.xc.to_f64();
    let sigma1_compression = stress(-frac * xc, 0.0, 0.0);
    assert_rel(
        failure_index(FailureCriterion::TsaiHill, s, sigma1_compression),
        frac * frac,
        FIX_TOL,
        "sigma1 = -0.3*Xc",
    );

    let yt = s.yt.to_f64();
    let sigma2_tension = stress(0.0, frac * yt, 0.0);
    assert_rel(
        failure_index(FailureCriterion::TsaiHill, s, sigma2_tension),
        frac * frac,
        FIX_TOL,
        "sigma2 = 0.3*Yt",
    );

    let yc = s.yc.to_f64();
    let sigma2_compression = stress(0.0, -frac * yc, 0.0);
    assert_rel(
        failure_index(FailureCriterion::TsaiHill, s, sigma2_compression),
        frac * frac,
        FIX_TOL,
        "sigma2 = -0.3*Yc",
    );

    let ss = s.s.to_f64();
    let shear = stress(0.0, 0.0, frac * ss);
    assert_rel(
        failure_index(FailureCriterion::TsaiHill, s, shear),
        frac * frac,
        FIX_TOL,
        "tau12 = 0.3*S",
    );
}

// ============================================================================
// Section 5: Tsai-Wu's F12 cross-coupling term (2*F12*sigma1*sigma2), which
// every test in engineering_oracles_solid.rs multiplies by zero (each of
// their cases has sigma1 = 0 or sigma2 = 0). This exercises it with both
// stresses nonzero, on GFRP (independent of the CFRP numbers used
// elsewhere), closed form derived by hand from the module's documented
// formula, not from calling tsai_wu_failure_index.
// ============================================================================

#[test]
fn tsai_wu_biaxial_cross_term_matches_the_documented_closed_form_gfrp() {
    let s = LaminateStrengths::gfrp_ud();
    let (xt, xc, yt, yc, ss) = (
        s.xt.to_f64(),
        s.xc.to_f64(),
        s.yt.to_f64(),
        s.yc.to_f64(),
        s.s.to_f64(),
    );
    let f1 = 1.0 / xt - 1.0 / xc;
    let f2 = 1.0 / yt - 1.0 / yc;
    let f11 = 1.0 / (xt * xc);
    let f22 = 1.0 / (yt * yc);
    let f66 = 1.0 / (ss * ss);
    let f12 = -0.5 * (f11 * f22).sqrt();

    let (sigma1, sigma2, tau12) = (100.0_f64, 20.0_f64, 10.0_f64);
    let want = f1 * sigma1
        + f2 * sigma2
        + f11 * sigma1 * sigma1
        + f22 * sigma2 * sigma2
        + f66 * tau12 * tau12
        + 2.0 * f12 * sigma1 * sigma2;

    let got = tsai_wu_failure_index(s, stress(sigma1, sigma2, tau12));
    assert_rel(got, want, FIX_TOL, "Tsai-Wu biaxial+shear, GFRP");

    // Independent sanity: the cross term alone must be nonzero and
    // negative here (F12 < 0, sigma1*sigma2 > 0), so removing it (a sign
    // mutation to `-Fix128::from_ratio(1, 2)` or dropping the `2*` factor)
    // changes the result by a measurable amount, not an unobservable one.
    let without_cross = f1 * sigma1
        + f2 * sigma2
        + f11 * sigma1 * sigma1
        + f22 * sigma2 * sigma2
        + f66 * tau12 * tau12;
    assert!(
        (want - without_cross).abs() > 1e-6,
        "cross term must move the result by more than f64 noise: with={want}, without={without_cross}"
    );
}

// ============================================================================
// Section 6: degenerate input — all-zero strengths. A ply with zero strength
// under non-zero load has FI -> infinity in closed form, so it is reported
// as failed: the largest representable Fix128 for the Tsai indices and a
// non-Safe mode for Hashin / Puck (FI marker 1 from the dispatcher). No panic.
// ============================================================================

#[test]
fn zero_strengths_report_failed_at_any_nonzero_stress() {
    let zero_strengths = LaminateStrengths {
        xt: Fix128::ZERO,
        xc: Fix128::ZERO,
        yt: Fix128::ZERO,
        yc: Fix128::ZERO,
        s: Fix128::ZERO,
    };
    // Large, not just nonzero: the verdict does not depend on stress magnitude.
    let failed = Fix128::from_raw(i64::MAX, u64::MAX);
    let loaded = stress(1.0e6, -5.0e5, 3.0e5);

    let tsai_wu = catch_unwind(AssertUnwindSafe(|| {
        tsai_wu_failure_index(zero_strengths, loaded)
    }))
    .expect("tsai_wu_failure_index must not panic on zero strengths");
    assert_eq!(tsai_wu, failed, "Tsai-Wu with zero strengths");

    let tsai_hill = catch_unwind(AssertUnwindSafe(|| {
        tsai_hill_failure_index(zero_strengths, loaded)
    }))
    .expect("tsai_hill_failure_index must not panic on zero strengths");
    assert_eq!(tsai_hill, failed, "Tsai-Hill with zero strengths");

    let hashin = catch_unwind(AssertUnwindSafe(|| {
        hashin_failure_mode(zero_strengths, loaded)
    }))
    .expect("hashin_failure_mode must not panic on zero strengths");
    assert_eq!(
        hashin,
        FailureMode::FibreTension,
        "Hashin with zero strengths"
    );

    let puck = catch_unwind(AssertUnwindSafe(|| {
        puck_failure_mode(zero_strengths, loaded)
    }))
    .expect("puck_failure_mode must not panic on zero strengths");
    assert_eq!(puck, FailureMode::FibreTension, "Puck with zero strengths");

    for criterion in [
        FailureCriterion::TsaiWu,
        FailureCriterion::TsaiHill,
        FailureCriterion::Hashin,
        FailureCriterion::Puck,
    ] {
        let got = catch_unwind(AssertUnwindSafe(|| {
            failure_index(criterion, zero_strengths, loaded)
        }))
        .expect("failure_index must not panic on zero strengths");
        let want = match criterion {
            FailureCriterion::TsaiWu | FailureCriterion::TsaiHill => failed,
            FailureCriterion::Hashin | FailureCriterion::Puck => Fix128::ONE,
        };
        assert_eq!(got, want, "{criterion:?} dispatcher with zero strengths");
    }
}

// ============================================================================
// Section 7: degenerate input — stress that overflows Fix128's ±2^63
// integer range. sigma1 = Fix128::from_int(i64::MAX) squared wraps to
// exactly Fix128::ONE in raw Fix128 arithmetic (derived by hand in the module
// doc above). The criteria check their squared terms (AUD-A-S4W2-004): the
// index of a stress this far beyond unit strengths is the largest Fix128 for
// Tsai-Wu / Tsai-Hill (failed), and Hashin / Puck report fibre tension.
// ============================================================================

#[test]
fn extreme_overflowing_stress_saturates_instead_of_wrapping() {
    let unity_strengths = LaminateStrengths {
        xt: Fix128::ONE,
        xc: Fix128::ONE,
        yt: Fix128::ONE,
        yc: Fix128::ONE,
        s: Fix128::ONE,
    };
    let extreme = StressState {
        sigma_1: Fix128::from_int(i64::MAX),
        sigma_2: Fix128::ZERO,
        tau_12: Fix128::ZERO,
    };

    // The wrap itself, independent of laminate_failure: (i64::MAX)^2 wraps
    // to exactly Fix128::ONE, not ~8.5e37.
    let squared = catch_unwind(AssertUnwindSafe(|| extreme.sigma_1 * extreme.sigma_1))
        .expect("Fix128 multiply never panics, wrapping or not");
    assert_eq!(
        squared,
        Fix128::ONE,
        "(i64::MAX)^2 must wrap to exactly Fix128::ONE in Fix128"
    );

    let largest = Fix128::from_raw(i64::MAX, u64::MAX);
    for (criterion, want) in [
        (FailureCriterion::TsaiWu, largest),
        (FailureCriterion::TsaiHill, largest),
        (FailureCriterion::Hashin, Fix128::ONE),
        (FailureCriterion::Puck, Fix128::ONE),
    ] {
        let got = catch_unwind(AssertUnwindSafe(|| {
            failure_index(criterion, unity_strengths, extreme)
        }))
        .unwrap_or_else(|_| {
            panic!("failure_index({criterion:?}) must not panic on overflowing stress")
        });
        assert_eq!(
            got, want,
            "{criterion:?} at overflowing stress saturates (failed), it does not wrap"
        );
    }

    // Hashin/Puck's mode classifier directly: must be FibreTension, not
    // Safe and not a panic from the wrapped arithmetic.
    let hashin_mode = catch_unwind(AssertUnwindSafe(|| {
        hashin_failure_mode(unity_strengths, extreme)
    }))
    .expect("hashin_failure_mode must not panic on overflowing stress");
    assert_eq!(hashin_mode, FailureMode::FibreTension);

    let puck_mode = catch_unwind(AssertUnwindSafe(|| {
        puck_failure_mode(unity_strengths, extreme)
    }))
    .expect("puck_failure_mode must not panic on overflowing stress");
    assert_eq!(puck_mode, FailureMode::FibreTension);
}
