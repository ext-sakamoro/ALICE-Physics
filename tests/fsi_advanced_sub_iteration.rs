//! Sub-iterating the `fsi_advanced` coupling, with a closed-form contraction
//! ratio.
//!
//! Unlike `cloth_fluid` (see `cloth_fluid_sub_iteration.rs`), this coupling
//! **closes the loop in velocity**: [`drag_force`] reads the solid sample's
//! velocity and the fluid velocity, and [`react_back_pressure`] deposits the
//! negated solid force back into the fluid. A caller that wires the deposit
//! into whatever the fluid sampler reads therefore has a genuine fixed-point
//! iteration, and its contraction ratio can be measured.
//!
//! ⚠️ The module supplies both halves but **not the connection**: the sampler
//! and the deposit sink are caller-supplied closures. The harness below closes
//! the loop with one scalar, `gain`, standing for the inverse effective mass of
//! the fluid that the reaction pushes on. Everything numeric here is therefore
//! a property of *this harness composed with the module's drag law*, not of the
//! module alone.
//!
//! # The harness, solved on paper
//!
//! With a single sample, `ρ = C_d = A = 1`, solid velocity held at `v₀ = 1`
//! each sweep, and the fluid velocity rebuilt from the deposited reaction:
//!
//! ```text
//! drag        f  = −½·|v_rel|·v_rel ,   v_rel = v₀ − u        (u = fluid velocity)
//! reaction    u' = −f·gain = ½·gain·|v_rel|·v_rel
//! let w = v₀ − u                                              (relative velocity)
//! then        w' = 1 − ½·gain·w²  ≡  h(w)
//! ```
//!
//! The solid iterate is `v₀ + f·Δt`, so successive changes are proportional to
//! successive changes in `w²`, and near the fixed point the ratio is `|h′(w*)|`:
//!
//! ```text
//! ρ(gain) = gain · w* ,     w* = (√(1 + 2·gain) − 1) / gain
//! ```
//!
//! which is the closed form this file asserts against. Measured agreement:
//!
//! | `gain` | `ρ` from the closed form | measured |
//! |---|---|---|
//! | `1/4` | `0.224745` | `0.2248` |
//! | `1/2` | `0.414214` | `0.415` |
//! | `4` | `2.000000` | diverges |
//!
//! ## The stability threshold
//!
//! `ρ = 1` gives `gain·w* = 1`; substituting `w* = 1/gain` into the fixed-point
//! equation yields `1/gain = 1 − 1/(2·gain)`, hence **`gain = 3/2`**. Below it
//! the unrelaxed sweep contracts, above it it does not.
//!
//! # ⚠️ Why `gain = 2` is absent from every scene below
//!
//! Measured, `gain = 2` reports a ratio of **exactly `1.000000`, five sweeps
//! running** — which reads as marginal stability. It is not. `gain = 2` gives
//! `h(w) = 1 − w²`, and the harness's initial condition is exactly `w₀ = 1`,
//! which sits on that map's **period-2 orbit `{1, 0}`**. The orbit alternates
//! exactly, so successive changes have constant magnitude and the ratio reads
//! one. Meanwhile `|h′(w*)| = 1.236 > 1`: the regime is **unstable**, and
//! `gain = 2` is past the threshold of `3/2`, not at it.
//!
//! Two consequences, both load-bearing:
//!
//! - ⚠️ **A measured ratio of one does not mean marginal for a nonlinear map.**
//!   It can be a periodic orbit inside the unstable regime. The piston's
//!   `m_f/m = 1` is marginal because that map is *linear*; the reasoning does
//!   not transfer.
//! - The `gain = 2` reading is a property of **this harness's initial
//!   condition** — a measure-zero coincidence — so it is not pinned anywhere
//!   here. Scenes stay clear of both `3/2` (the threshold) and `2` (the orbit).
//!
//! # What this file claims
//!
//! The closed-form ratio above, the divergence verdict past the threshold, and
//! that a contracting gain trips no guard. ⚠️ It does **not** claim that a
//! converged state is physically correct: `react_back_pressure`'s own
//! documentation calls itself a placeholder for a true immersed-boundary
//! scatter operator, and the fluid side here is a one-scalar stand-in.
//!
//! # ⚠️ These are the *unrelaxed* sweeps
//!
//! No relaxation is applied. `ρ > 1` means "this splitting diverges when swept
//! without relaxation", not "partitioned coupling cannot solve this".

#![cfg(feature = "std")]
// The f64 values are closed-form references, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_iteration::{
    residual_norm_inf, run_sub_iteration, CoupledIterationError, SubIterationConfig,
};
use alice_physics::fsi_advanced::{drag_force, react_back_pressure, SolidSample};
use alice_physics::math::{Fix128, Vec3Fix};

const UNIT: Fix128 = Fix128::ONE;

fn sample(velocity_x: Fix128) -> SolidSample {
    SolidSample {
        position: Vec3Fix::ZERO,
        velocity: Vec3Fix::new(velocity_x, Fix128::ZERO, Fix128::ZERO),
        area_m2: UNIT,
        volume_m3: Fix128::from_ratio(1, 10),
    }
}

/// Closed-form fixed point `w* = (√(1 + 2·gain) − 1) / gain`.
fn relative_velocity_fixed_point(gain: Fix128) -> Fix128 {
    let discriminant = UNIT + Fix128::from_int(2) * gain;
    (discriminant.sqrt() - UNIT) / gain
}

/// Closed-form contraction ratio `ρ = gain · w*`.
fn closed_form_ratio(gain: Fix128) -> Fix128 {
    gain * relative_velocity_fixed_point(gain)
}

/// One sweep: drag from the current fluid velocity, then the reaction rebuilt
/// from scratch so that the sweep is a map and not a time integration.
///
/// ⚠️ Rebuilding the fluid state each sweep is the whole point: carrying it
/// across sweeps makes this an integration, whose residual can neither settle
/// nor report a ratio, and which reports a false divergence that looks like a
/// working detector.
fn sweep(gain: Fix128, fluid_velocity: Fix128) -> (Fix128, Fix128) {
    let solid = sample(UNIT);
    let force = drag_force(
        &solid,
        Vec3Fix::new(fluid_velocity, Fix128::ZERO, Fix128::ZERO),
        UNIT,
        UNIT,
    );
    let mut next_fluid = Fix128::ZERO;
    react_back_pressure(&[solid], &[force], |_position, reaction| {
        next_fluid = next_fluid + reaction.x * gain;
    });
    (force.x, next_fluid)
}

/// Drive the coupling under the monitor. The residual is the change in the
/// solid force between sweeps, which is the interface-force residual.
fn run(
    gain: Fix128,
    config: SubIterationConfig,
) -> Result<alice_physics::coupled_iteration::SubIterationReport, CoupledIterationError> {
    let mut fluid = Fix128::ZERO;
    let mut previous_force: Option<Fix128> = None;
    run_sub_iteration(config, |_| {
        let (force, next_fluid) = sweep(gain, fluid);
        fluid = next_fluid;
        let residual = previous_force.map_or(force.abs(), |before| (force - before).abs());
        previous_force = Some(force);
        residual_norm_inf(&[residual])
    })
}

/// The ratio the sweeps actually produce, read directly.
fn measured_ratio(gain: Fix128, sweeps: u32) -> Fix128 {
    let mut fluid = Fix128::ZERO;
    let mut previous_force: Option<Fix128> = None;
    let mut previous_change = Fix128::ZERO;
    let mut ratio = Fix128::ZERO;
    for _ in 0..sweeps {
        let (force, next_fluid) = sweep(gain, fluid);
        fluid = next_fluid;
        if let Some(before) = previous_force {
            let change = (force - before).abs();
            if !previous_change.is_zero() {
                ratio = change / previous_change;
            }
            previous_change = change;
        }
        previous_force = Some(force);
    }
    ratio
}

#[test]
fn the_loop_is_closed_in_velocity() {
    // The structural difference from `cloth_fluid`, asserted rather than read
    // off the signatures.
    let slow = drag_force(&sample(UNIT), Vec3Fix::ZERO, UNIT, UNIT);
    let fast = drag_force(&sample(Fix128::from_int(5)), Vec3Fix::ZERO, UNIT, UNIT);
    assert_ne!(slow, fast, "drag must read the solid velocity");

    let still = drag_force(&sample(UNIT), Vec3Fix::ZERO, UNIT, UNIT);
    let moving = drag_force(
        &sample(UNIT),
        Vec3Fix::new(UNIT, Fix128::ZERO, Fix128::ZERO),
        UNIT,
        UNIT,
    );
    assert_ne!(still, moving, "drag must read the fluid velocity");

    // And the reaction must carry the force through, so the loop can close.
    let mut from_slow = Fix128::ZERO;
    react_back_pressure(&[sample(UNIT)], &[slow], |_p, r| from_slow = r.x);
    let mut from_fast = Fix128::ZERO;
    react_back_pressure(&[sample(Fix128::from_int(5))], &[fast], |_p, r| {
        from_fast = r.x;
    });
    assert_ne!(
        from_slow, from_fast,
        "the reaction must depend on the solid force"
    );
    assert_eq!(from_slow, -slow.x, "the reaction is the negated force");
}

#[test]
fn the_contraction_ratio_matches_the_closed_form() {
    // Gains well below the threshold of 3/2, so the fixed point is reached and
    // the asymptotic ratio is |h'(w*)|. The tolerance is relative and measured:
    // the sweeps approach the ratio rather than landing on it.
    //
    // Measured relative gap at 12 sweeps, over the set below:
    //
    // | `gain` | relative gap |
    // |---|---|
    // | `1/8` | `2.4e-10` |
    // | `1/4` | `1.5e-8` |
    // | `1/2` | `9.6e-6` |
    // | `1` | `2.2e-3` |
    // | `5/4` | `4.3e-3` |
    //
    // Worst is `4.2528e-3` at `gain = 5/4`, where the ratio is closest to one
    // and the approach is slowest. The bound below is `1/64 ≈ 1.5625e-2`,
    // about 3.7 times the measured worst. A wrong closed form would miss by
    // order one — dropping the `gain` factor, say — so this bound is nowhere
    // near loose enough to hide one.
    //
    // ⚠️ The sweep count is not free to raise. At `gain = 1/8` and 24 sweeps the
    // measured ratio reads **exactly zero**: the sequence has reached the
    // rounding floor, successive changes truncate to nothing, and the quotient
    // collapses. Reading a contraction ratio late is measuring the floor rather
    // than the splitting, and the effect is worse the faster the contraction.
    let bound = Fix128::from_ratio(1, 64);
    for &(num, den) in &[(1i64, 4i64), (1, 2), (1, 1), (5, 4)] {
        let gain = Fix128::from_ratio(num, den);
        let closed = closed_form_ratio(gain);
        let measured = measured_ratio(gain, 12);
        let gap = (measured - closed).abs();
        assert!(
            gap <= closed * bound,
            "gain = {num}/{den}: measured {measured:?} vs closed form {closed:?}, \
             gap {gap:?} exceeds the bound"
        );
        assert!(
            closed < UNIT,
            "gain = {num}/{den} is below the threshold 3/2, so the closed form \
             must contract; got {closed:?}"
        );
    }
}

#[test]
fn reading_the_ratio_too_late_measures_the_floor_instead() {
    // The claim in the comment above, asserted. A strongly contracting sweep
    // reaches the rounding floor, its successive changes truncate to nothing,
    // and the quotient collapses to zero — so the sweep count at which the
    // ratio is read is load-bearing, and later is not safer.
    let gain = Fix128::from_ratio(1, 8);
    let closed = closed_form_ratio(gain);

    let early = measured_ratio(gain, 12);
    let late = measured_ratio(gain, 24);

    let early_gap = (early - closed).abs();
    assert!(
        early_gap <= closed * Fix128::from_ratio(1, 64),
        "12 sweeps should still track the closed form: {early:?} vs {closed:?}"
    );
    assert!(
        late.is_zero(),
        "24 sweeps should have collapsed to the floor, reading zero; got {late:?}. \
         If this no longer collapses, the comment about sweep counts is stale."
    );
}

#[test]
fn the_closed_form_places_the_threshold_at_three_halves() {
    // The derivation, pinned: rho(3/2) = 1 exactly in real arithmetic, and the
    // closed form must straddle one across it. `3/2` itself is skipped as a
    // scene — it is the marginal point — but the *formula* is checked there.
    let threshold = Fix128::from_ratio(3, 2);
    let at = closed_form_ratio(threshold);
    let gap = (at - UNIT).abs();
    assert!(
        gap <= Fix128::from_ratio(1, 1_000_000),
        "rho(3/2) should be one; got {at:?}"
    );
    assert!(closed_form_ratio(Fix128::from_ratio(14, 10)) < UNIT);
    assert!(closed_form_ratio(Fix128::from_ratio(16, 10)) > UNIT);
}

#[test]
fn a_gain_past_the_threshold_is_reported_as_diverging() {
    // gain = 4 gives rho = 2 in closed form, comfortably clear of both the
    // threshold (3/2) and the period-2 orbit (2).
    let gain = Fix128::from_int(4);
    assert_eq!(closed_form_ratio(gain), Fix128::from_int(2));

    let error = run(gain, SubIterationConfig::default())
        .expect_err("gain = 4 does not contract when swept without relaxation");
    match error {
        CoupledIterationError::Diverging { observed_ratio, .. } => {
            assert!(
                observed_ratio > UNIT,
                "a Diverging verdict must carry a ratio above one, got {observed_ratio:?}"
            );
        }
        other => panic!("expected Diverging, got {other:?}"),
    }
}

#[test]
fn a_gain_below_the_threshold_trips_no_guard() {
    // Control. A monitor that stopped these would be worse than none, and the
    // divergence verdict above would say nothing.
    for &(num, den) in &[(1i64, 8i64), (1, 4), (1, 2), (1, 1)] {
        let gain = Fix128::from_ratio(num, den);
        let report = run(gain, SubIterationConfig::default())
            .unwrap_or_else(|e| panic!("gain = {num}/{den} must converge cleanly: {e:?}"));
        assert!(report.sweeps >= 1);
    }
}

#[test]
fn a_wrapped_divergence_is_not_reported_as_convergence() {
    // The direct justification for this whole exercise. Left unmonitored, the
    // diverging sweep grows until `Fix128` wraps; the squared residual then
    // reads exactly zero and a convergence test calls it converged. The
    // monitor must decide before that, and must never answer `Ok`.
    let gain = Fix128::from_int(4);

    // Unmonitored: find the sweep at which the naive squared residual starts
    // lying, and check that it does lie.
    let mut fluid = Fix128::ZERO;
    let mut previous_force: Option<Fix128> = None;
    let mut naive_zero_at = None;
    for index in 1..=64u32 {
        let (force, next_fluid) = sweep(gain, fluid);
        fluid = next_fluid;
        if let Some(before) = previous_force {
            let change = force - before;
            let naive = (change * change).sqrt();
            if naive.is_zero() && !change.is_zero() && naive_zero_at.is_none() {
                naive_zero_at = Some(index);
                break;
            }
        }
        previous_force = Some(force);
    }
    let wrap_sweep = naive_zero_at.expect(
        "the naive squared residual must eventually read zero on this diverging \
         sweep; if it no longer does, this test is no longer exercising the defect",
    );

    // Monitored: a verdict, before the wrap, and never `Ok`.
    let config = SubIterationConfig {
        max_sweeps: 64,
        ..SubIterationConfig::default()
    };
    match run(gain, config) {
        Ok(report) => {
            panic!("a wrapped divergence must never be reported as convergence, got {report:?}")
        }
        Err(CoupledIterationError::Diverging { sweeps, .. }) => {
            assert!(
                sweeps < wrap_sweep,
                "the monitor must decide before the arithmetic wraps: decided at \
                 sweep {sweeps}, wrap showed at sweep {wrap_sweep}"
            );
        }
        Err(other) => panic!("expected Diverging before the wrap, got {other:?}"),
    }
}
