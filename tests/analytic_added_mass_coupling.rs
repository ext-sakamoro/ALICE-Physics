//! Analytic oracle for the added-mass instability of partitioned coupling.
//!
//! The crate's existing coupling (`cloth_fluid`, `fsi_advanced`) applies
//! one-directional forces in sequence with **no sub-iteration**. That is a
//! fixed-point iteration, and whether it converges is a property of the
//! splitting rather than of the time step. This file pins that property
//! against a closed form.
//!
//! # Why not a manufactured solution
//!
//! An accuracy oracle — a manufactured solution, a convergence study — passes
//! on a partitioned scheme and on a monolithic one alike whenever both
//! converge, so it says nothing about *which* scheme was needed. What
//! discriminates is the regime where the partitioned scheme stops converging,
//! and the useful fact about that regime is that **it is not a time-step
//! problem**.
//!
//! # The scene
//!
//! A piston of mass `m` on a spring `k` drives an incompressible fluid column
//! of mass `m_f`. Incompressibility makes the pressure a Lagrange multiplier —
//! instantaneous, carrying no time scale of its own — so the fluid contributes
//! pure added mass:
//!
//! ```text
//! monolithic : (m + m_f)·a = −k·x
//! ```
//!
//! Advancing one step by implicit Euler and sweeping fluid-then-structure
//! inside it gives
//!
//! ```text
//! a^(j+1) = (−k·xⁿ − k·Δt·vⁿ − m_f·a^(j)) / (m + k·Δt²)
//! ```
//!
//! whose error map is a contraction by
//!
//! ```text
//! ρ(Δt) = m_f / (m + k·Δt²)
//! ```
//!
//! ⚠️ **Refining `Δt` moves `ρ` *up* toward `m_f/m`, never down.** A splitting
//! with `m_f > m` therefore diverges at every time step, and "take smaller
//! steps" is not a remedy. Measured, with `m = k = 1`:
//!
//! | `Δt` | `ρ` at `m_f/m = 1/2` | `ρ` at `m_f/m = 2` |
//! |---|---|---|
//! | `1/8` | `0.4923076923076923` | `1.9692307692307693` |
//! | `1/80` | `0.4999218872051242` | `1.9996875488204968` |
//! | `1/800` | `0.4999992187512207` | `1.9999968750048827` |
//! | `1/8000` | `0.4999999921875001` | `1.9999999687500005` |
//!
//! This is the added-mass instability of partitioned fluid-structure
//! interaction (Causin, Gerbeau & Nobile 2005; Förster, Wall & Ramm 2007).
//!
//! # What each test pins
//!
//! - `refining_the_time_step_does_not_rescue_a_divergent_splitting` — the
//!   discriminating claim.
//! - `the_contraction_ratio_matches_the_closed_form` — the instrument reads the
//!   right number.
//! - `the_weak_scheme_approaches_the_monolithic_one_as_the_coupling_weakens` —
//!   agreement in the weak limit, asserted as a **rate**, not a threshold.
//! - `the_coupled_period_carries_the_added_mass` — the monolithic solve really
//!   is the coupled system, not the uncoupled one.
//! - `a_diverging_iteration_reports_a_zero_l2_residual` — the silent-failure
//!   path this work exists to close.
//! - `a_contracting_splitting_trips_no_guard` — control: the guards are not
//!   simply always-on.

#![cfg(feature = "std")]
// The f64 values here are closed-form oracle references, not simulation state,
// so the determinism gate on f64 arithmetic does not apply. Note that f64
// cannot resolve differences below 2⁻⁵³, so no claim finer than that is made
// through this path; the ulp-level claims use the raw Q64.64 representation.
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_iteration::{
    residual_norm_inf, residual_norm_l2_checked, ConfigFault, ContractionMonitor,
    CoupledIterationError, MonitorVerdict, SubIterationConfig, SubIterationReport,
};
use alice_physics::math::Fix128;

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// Unit spring and unit structural mass; the coupling strength is carried
/// entirely by `m_f/m`, which is what the oracle sweeps.
const K: Fix128 = Fix128::ONE;
const M: Fix128 = Fix128::ONE;

fn raw(v: Fix128) -> i128 {
    (i128::from(v.hi) << 64) | i128::from(v.lo)
}

fn ulp_gap(a: Fix128, b: Fix128) -> u128 {
    raw(a).abs_diff(raw(b))
}

/// Largest deviation allowed between a measured ratio and its closed form.
///
/// Bit equality is unavailable: the ratio is a truncating division of two
/// residuals. Measured worst over `m_f/m ∈ [1/8, 8]` at four sweeps is
/// 144 ulp, at `m_f/m = 1/8` where the residual sits closest to the rounding
/// floor; this bound leaves a factor of about 1.8. In relative terms the
/// measured worst is `6.2e-17`, far below any difference a wrong ratio would
/// produce.
const RATIO_ULP_BOUND: u128 = 256;

/// State carried across one implicit-Euler step.
#[derive(Clone, Copy)]
struct Step {
    x: Fix128,
    v: Fix128,
    dt: Fix128,
}

impl Step {
    /// `m + k·Δt²`, the structural operator of one implicit-Euler step.
    fn denominator(self) -> Fix128 {
        M + K * self.dt * self.dt
    }

    /// `−k·xⁿ − k·Δt·vⁿ`, the part of the right-hand side the fluid does not
    /// supply.
    fn rhs(self) -> Fix128 {
        -K * self.x - K * self.dt * self.v
    }

    /// Acceleration the monolithic system produces: the fluid's added mass
    /// joins the structural operator instead of being lagged.
    fn monolithic_acceleration(self, mass_ratio: Fix128) -> Fix128 {
        self.rhs() / (self.denominator() + mass_ratio)
    }

    /// One fluid-then-structure sweep from a given structural iterate.
    fn staggered_sweep(self, mass_ratio: Fix128, previous: Fix128) -> Fix128 {
        (self.rhs() - mass_ratio * previous) / self.denominator()
    }
}

/// Closed-form contraction ratio of the splitting: `m_f / (m + k·Δt²)`.
fn closed_form_ratio(mass_ratio: Fix128, dt: Fix128) -> Fix128 {
    mass_ratio / (M + K * dt * dt)
}

/// Drive the sub-iteration under the monitor, returning its report or error.
fn run_scene(
    step: Step,
    mass_ratio: Fix128,
    config: SubIterationConfig,
) -> Result<SubIterationReport, CoupledIterationError> {
    let target = step.monolithic_acceleration(mass_ratio);
    let mut a = Fix128::ZERO;
    alice_physics::coupled_iteration::run_sub_iteration(config, |_| {
        a = step.staggered_sweep(mass_ratio, a);
        residual_norm_inf(&[a - target])
    })
}

/// Read the contraction ratio the splitting actually produces, by running the
/// sweep directly rather than through the monitor's stopping rules.
fn measured_ratio(step: Step, mass_ratio: Fix128, sweeps: u32) -> Fix128 {
    let target = step.monolithic_acceleration(mass_ratio);
    let mut a = Fix128::ZERO;
    let mut previous = Fix128::ZERO;
    let mut ratio = Fix128::ZERO;
    for sweep in 0..sweeps {
        a = step.staggered_sweep(mass_ratio, a);
        let error = (a - target).abs();
        if sweep >= 1 && !previous.is_zero() {
            ratio = error / previous;
        }
        previous = error;
    }
    ratio
}

// ---------------------------------------------------------------------------
// The discriminating oracle
// ---------------------------------------------------------------------------

#[test]
fn refining_the_time_step_does_not_rescue_a_divergent_splitting() {
    // Four time steps spanning three decades. A divergent splitting must stay
    // divergent at all of them, and the ratio must move *toward* m_f/m rather
    // than away — the opposite of what a time-step problem would do.
    let steps = [(1i64, 8i64), (1, 80), (1, 800), (1, 8000)];
    let divergent = Fix128::from_int(2);

    let mut previous_ratio = Fix128::ZERO;
    for (index, &(num, den)) in steps.iter().enumerate() {
        let dt = Fix128::from_ratio(num, den);
        let step = Step {
            x: Fix128::ONE,
            v: Fix128::ZERO,
            dt,
        };
        let ratio = measured_ratio(step, divergent, 4);

        assert!(
            ratio > Fix128::ONE,
            "m_f/m = 2 must not contract at dt = {num}/{den}; ratio was {ratio:?}"
        );
        assert!(
            ratio < divergent,
            "the ratio must stay below m_f/m at dt = {num}/{den}; ratio was {ratio:?}"
        );
        if index > 0 {
            assert!(
                ratio > previous_ratio,
                "refining dt to {num}/{den} must move the ratio up toward m_f/m, \
                 but it went from {previous_ratio:?} to {ratio:?}"
            );
        }
        previous_ratio = ratio;
    }

    // And the monitor reaches the same verdict at the finest step, where a
    // "just use a smaller dt" reading would have expected a rescue.
    let finest = Step {
        x: Fix128::ONE,
        v: Fix128::ZERO,
        dt: Fix128::from_ratio(1, 8000),
    };
    let error = run_scene(finest, divergent, SubIterationConfig::default())
        .expect_err("m_f/m = 2 diverges at every time step");
    assert!(
        matches!(error, CoupledIterationError::Diverging { .. }),
        "expected Diverging at the finest step, got {error:?}"
    );
}

#[test]
fn a_contracting_splitting_stays_contracting_at_every_time_step() {
    // Control for the test above: the same sweep of time steps on a splitting
    // below the threshold must converge at all of them. Without this, a guard
    // that simply rejected everything would pass the discriminating test.
    let contracting = Fix128::from_ratio(1, 2);
    for &(num, den) in &[(1i64, 8i64), (1, 80), (1, 800), (1, 8000)] {
        let step = Step {
            x: Fix128::ONE,
            v: Fix128::ZERO,
            dt: Fix128::from_ratio(num, den),
        };
        let ratio = measured_ratio(step, contracting, 4);
        assert!(
            ratio < Fix128::ONE,
            "m_f/m = 1/2 must contract at dt = {num}/{den}; ratio was {ratio:?}"
        );
        run_scene(step, contracting, SubIterationConfig::default())
            .unwrap_or_else(|e| panic!("m_f/m = 1/2 must converge at dt = {num}/{den}: {e:?}"));
    }
}

#[test]
fn the_contraction_ratio_matches_the_closed_form() {
    // Across the threshold in both directions. `m_f/m = 1` is deliberately
    // absent: it is the marginal case, and truncation leaves a dead band about
    // one ulp wide just below it, so the oracle brackets the threshold rather
    // than probing at it.
    let dt = Fix128::from_ratio(1, 8);
    let step = Step {
        x: Fix128::ONE,
        v: Fix128::ZERO,
        dt,
    };
    for &(num, den) in &[
        (1i64, 4i64),
        (1, 2),
        (3, 4),
        (9, 10),
        (11, 10),
        (3, 2),
        (2, 1),
        (4, 1),
    ] {
        let mass_ratio = Fix128::from_ratio(num, den);
        let measured = measured_ratio(step, mass_ratio, 4);
        let closed = closed_form_ratio(mass_ratio, dt);
        let gap = ulp_gap(measured, closed);
        assert!(
            gap <= RATIO_ULP_BOUND,
            "m_f/m = {num}/{den}: measured ratio {measured:?} is {gap} ulp from the \
             closed form {closed:?}, above the measured bound {RATIO_ULP_BOUND}"
        );

        // The closed form must also land the verdict on the right side of one.
        let contracts = closed < Fix128::ONE;
        assert_eq!(
            contracts,
            mass_ratio < step.denominator(),
            "m_f/m = {num}/{den}: the closed form and the threshold disagree"
        );
    }
}

#[test]
fn the_weak_scheme_approaches_the_monolithic_one_as_the_coupling_weakens() {
    // Agreement in the weak limit, asserted as a rate rather than a threshold:
    // one un-sub-iterated sweep (what the crate does today) differs from the
    // monolithic answer by an amount proportional to m_f, so halving m_f must
    // halve the difference. The ratio of successive differences approaches 2
    // from below, exactly as 2·(D + m_f/2)/(D + m_f) predicts.
    let step = Step {
        x: Fix128::ONE,
        v: Fix128::ZERO,
        dt: Fix128::from_ratio(1, 8),
    };

    let mut differences = Vec::new();
    for shift in 3..8u32 {
        let mass_ratio = Fix128::from_ratio(1, 1i64 << shift);
        let monolithic = step.monolithic_acceleration(mass_ratio);
        let one_sweep = step.staggered_sweep(mass_ratio, Fix128::ZERO);
        differences.push((mass_ratio, (one_sweep - monolithic).abs()));
    }

    let denominator = step.denominator();
    let two = Fix128::from_int(2);
    let mut previous_shrinkage = Fix128::ZERO;
    for window in differences.windows(2) {
        let (coarse_ratio, coarse) = window[0];
        let (fine_ratio, fine) = window[1];
        assert!(
            fine < coarse,
            "halving m_f from {coarse_ratio:?} to {fine_ratio:?} must shrink the gap, \
             but it went from {coarse:?} to {fine:?}"
        );

        // Compare against the closed form rather than a hand-picked window:
        //   gap(m_f) = |R|·m_f / (D·(D + m_f))
        //   gap(m_f) / gap(m_f/2) = 2·(D + m_f/2) / (D + m_f)
        let shrinkage = coarse / fine;
        let closed = two * (denominator + fine_ratio) / (denominator + coarse_ratio);
        let gap = ulp_gap(shrinkage, closed);
        assert!(
            gap <= RATIO_ULP_BOUND,
            "m_f/m = {coarse_ratio:?} → {fine_ratio:?}: shrinkage {shrinkage:?} is {gap} ulp \
             from the closed form {closed:?}, above the measured bound {RATIO_ULP_BOUND}"
        );
        assert!(
            shrinkage < two,
            "the shrinkage approaches 2 from below and must never reach it: {shrinkage:?}"
        );
        assert!(
            shrinkage > previous_shrinkage,
            "the shrinkage must approach 2 from below as the coupling weakens, \
             but it went from {previous_shrinkage:?} to {shrinkage:?}"
        );
        previous_shrinkage = shrinkage;
    }
}

#[test]
fn the_coupled_period_carries_the_added_mass() {
    // The monolithic solve must be the *coupled* system. Integrating it
    // symplectically, the period is 2π√((m + m_f)/k); if the fluid's added mass
    // were dropped the period would be 2π√(m/k), which is a factor √3 shorter
    // at m_f/m = 2. Measuring the period therefore pins that the coupling is
    // present in the answer, not merely in the wiring.
    let mass_ratio = Fix128::from_int(2);
    let dt = Fix128::from_ratio(1, 256);
    let effective_mass = M + mass_ratio;

    // Semi-implicit (symplectic) Euler on (m + m_f)·a = −k·x.
    let mut x = Fix128::ONE;
    let mut v = Fix128::ZERO;
    // Starting at maximum displacement, the first zero crossing is a quarter
    // period in, so crossings one and three are exactly one period apart.
    let mut crossings = 0u32;
    let mut first_crossing = 0u32;
    let mut third_crossing = 0u32;
    for step in 1..40_000u32 {
        let previous_x = x;
        v = v + dt * (-K * x / effective_mass);
        x = x + dt * v;
        if (previous_x > Fix128::ZERO) != (x > Fix128::ZERO) {
            crossings += 1;
            if crossings == 1 {
                first_crossing = step;
            } else if crossings == 3 {
                third_crossing = step;
                break;
            }
        }
    }
    assert!(
        third_crossing > 0,
        "the oscillator never completed a period"
    );

    let measured_period = f64::from(third_crossing - first_crossing) * dt.to_f64();
    let coupled_period = 2.0 * core::f64::consts::PI * (effective_mass.to_f64() / 1.0).sqrt();
    let uncoupled_period = 2.0 * core::f64::consts::PI * (M.to_f64() / 1.0).sqrt();

    let relative_error = (measured_period - coupled_period).abs() / coupled_period;
    assert!(
        relative_error < 0.01,
        "measured period {measured_period} should match the coupled closed form \
         {coupled_period} within 1%, relative error was {relative_error}"
    );
    // The two closed forms must be far apart, or matching one of them would say
    // nothing. √3 ≈ 1.73, so this margin is structural, not tuned.
    assert!(
        (measured_period - uncoupled_period).abs() / uncoupled_period > 0.5,
        "the measured period is indistinguishable from the uncoupled one, so this \
         test cannot tell whether the added mass is present"
    );
}

// ---------------------------------------------------------------------------
// The silent-failure path
// ---------------------------------------------------------------------------

#[test]
fn a_diverging_iteration_reports_a_zero_l2_residual() {
    // This is the defect the work exists to close. `Fix128` multiplication
    // wraps and `sqrt` returns zero for a negative argument, so the squared L2
    // residual of a diverging iteration eventually reads exactly zero — which a
    // convergence test reads as success. The checked norm must refuse instead.
    let dt = Fix128::from_ratio(1, 8);
    let step = Step {
        x: Fix128::ONE,
        v: Fix128::ZERO,
        dt,
    };
    let mass_ratio = Fix128::from_int(4);
    let target = step.monolithic_acceleration(mass_ratio);

    let mut a = Fix128::ZERO;
    let mut naive_reported_zero_at = None;
    let mut checked_refused_at = None;
    for sweep in 1..=64u32 {
        a = step.staggered_sweep(mass_ratio, a);
        let error = a - target;

        // What a caller writing the textbook norm would compute.
        let naive = (error * error).sqrt();
        if naive.is_zero() && naive_reported_zero_at.is_none() && !error.is_zero() {
            naive_reported_zero_at = Some(sweep);
        }
        if residual_norm_l2_checked(&[error]).is_err() && checked_refused_at.is_none() {
            checked_refused_at = Some(sweep);
        }
        if naive_reported_zero_at.is_some() && checked_refused_at.is_some() {
            break;
        }
    }

    let naive_at = naive_reported_zero_at.expect(
        "the naive squared norm must eventually read zero on a diverging iteration; \
         if it no longer does, this oracle is no longer testing the defect",
    );
    let checked_at =
        checked_refused_at.expect("the checked norm must refuse a residual it cannot represent");
    assert!(
        checked_at <= naive_at,
        "the checked norm must refuse no later than the naive one starts lying: \
         checked refused at sweep {checked_at}, naive read zero at sweep {naive_at}"
    );

    // And the monitor must not be fooled either: it reaches a verdict long
    // before the wrap, because it reads the contraction ratio rather than the
    // magnitude.
    let error =
        run_scene(step, mass_ratio, SubIterationConfig::default()).expect_err("m_f/m = 4 diverges");
    match error {
        CoupledIterationError::Diverging { sweeps, .. } => {
            assert!(
                sweeps < naive_at,
                "the monitor must decide before the arithmetic wraps: decided at \
                 sweep {sweeps}, wrap showed at sweep {naive_at}"
            );
        }
        other => panic!("expected Diverging, got {other:?}"),
    }
}

#[test]
fn a_contracting_splitting_trips_no_guard() {
    // Control for every guard at once. A monitor that stopped a healthy solve
    // would be worse than no monitor, and a guard suite is only evidence if
    // something passes it.
    let step = Step {
        x: Fix128::ONE,
        v: Fix128::ZERO,
        dt: Fix128::from_ratio(1, 8),
    };
    for &(num, den) in &[(1i64, 8i64), (1, 4), (1, 2), (3, 4), (9, 10)] {
        let mass_ratio = Fix128::from_ratio(num, den);
        let report = run_scene(step, mass_ratio, SubIterationConfig::default())
            .unwrap_or_else(|e| panic!("m_f/m = {num}/{den} must converge cleanly: {e:?}"));
        assert!(
            report.sweeps >= 1,
            "a converged report must have run at least one sweep"
        );
        let target = step.monolithic_acceleration(mass_ratio);
        let mut a = Fix128::ZERO;
        for _ in 0..report.sweeps {
            a = step.staggered_sweep(mass_ratio, a);
        }
        // The converged sub-iteration must agree with the monolithic answer:
        // this is the "converged partitioned equals monolithic" statement that
        // makes the cost argument, rather than a correctness argument, the
        // reason to prefer one scheme over the other.
        let gap = (a - target).abs();
        assert!(
            gap <= step.rhs().abs() * SubIterationConfig::default().relative_tolerance,
            "m_f/m = {num}/{den}: the converged sweep differs from the monolithic \
             answer by {gap:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Downstream construction (integration tests are a separate crate)
// ---------------------------------------------------------------------------

#[test]
fn the_public_types_can_be_built_and_matched_from_another_crate() {
    // `SubIterationConfig` carries public fields and a `Default`; the error
    // enums are `#[non_exhaustive]`. This crate has previously shipped a type
    // that became impossible to construct from outside, and the breakage was
    // invisible to in-crate tests because a crate can always name its own
    // fields. This test lives in `tests/`, which is a separate crate, so it
    // fails to compile the moment the public construction path closes.
    let config = SubIterationConfig::new(
        16,
        Fix128::from_ratio(1, 256),
        Fix128::from_ratio(1, 4),
        Fix128::from_int(2),
        3,
    )
    .expect("these values satisfy every bound");
    assert_eq!(config.max_sweeps, 16);

    let overridden = SubIterationConfig {
        max_sweeps: 8,
        ..SubIterationConfig::default()
    };
    assert_eq!(overridden.max_sweeps, 8);
    assert_eq!(overridden.ratio_samples, 4);

    assert_eq!(
        SubIterationConfig::new(
            0,
            Fix128::from_ratio(1, 256),
            Fix128::from_ratio(1, 4),
            Fix128::from_int(2),
            3,
        ),
        Err(ConfigFault::ZeroSweepBudget)
    );

    let mut monitor = ContractionMonitor::new(overridden).expect("valid config");
    assert_eq!(monitor.observe(Fix128::ONE), MonitorVerdict::Continue);
    assert_eq!(monitor.sweeps(), 1);
    assert_eq!(monitor.best_residual(), Fix128::ONE);
    assert_eq!(monitor.observed_ratio(), None);

    // Constructing an error value from outside the crate, and matching it.
    let error = CoupledIterationError::ArithmeticWrapped { sweeps: 7 };
    match error {
        CoupledIterationError::ArithmeticWrapped { sweeps } => assert_eq!(sweeps, 7),
        other => panic!("unexpected variant {other:?}"),
    }
}
