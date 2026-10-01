//! Re-measures, under `Fix128`, the four hypotheses that were rejected in `f64`
//! when looking for a scene where partitioned coupling fails and monolithic
//! coupling does not.
//!
//! # Why re-measure
//!
//! The rejections were produced by an `f64` probe. `f64` rounds to nearest;
//! `Fix128` truncates, and its two operators truncate in **different**
//! directions (measured, see below). A rejection that rested on a rounding
//! accident would therefore not survive the move to fixed point, and the
//! argument for or against building a monolithic solver rests on those
//! rejections. Each test below runs the same scene twice — once in `f64`, once
//! in `Fix128` — because "it behaves like this in `Fix128`" cannot say whether
//! a verdict *changed*.
//!
//! # Outcome: all four verdicts hold
//!
//! | Hypothesis | `f64` | `Fix128` | contraction ratio | changed |
//! |---|---|---|---|---|
//! | added-mass ratio defeats relaxed staggered | rejected | rejected | error factor `1 − ω(1+r)`: `0` in `f64`, `+(1+r)·2⁻⁶⁴` in `Fix128` | no |
//! | eigenvalue spread defeats a constant `ω` | partly holds | partly holds | `0.994029850746269` at spread 1000, both | no |
//! | skew coupling defeats staggered | rejected (reversed) | rejected (reversed) | `1/√(1+a²)`: `0.049937616943892` at `a = 20`, both | no |
//! | a nonlinear fold defeats Aitken | rejected | rejected | upper branch reached at `Λ = 0.20 / 0.30 / 0.35`, both | no |
//!
//! So the move to fixed point does not reopen the case for a monolithic
//! solver: a sub-iterated partitioned scheme still solves the same equations,
//! and relaxation still repairs the splitting.
//!
//! # What *is* different in fixed point
//!
//! Three fixed-point-only effects came out of the re-measurement. None of them
//! changes a verdict; all three are properties of the **instrument**.
//!
//! 1. **One lattice step above the critical mass ratio, a single-step
//!    contraction ratio reads `1` at every even sweep count and above `1` at
//!    every odd one.** The cause is not resolution. At `m_f/m = 1 + 2⁻⁶⁴` the
//!    error magnitude grows in a strict **period-2** pattern — `+2, 0, +2, 0, …`
//!    raw ulp — so a ratio of *successive* errors samples one phase of it and
//!    which phase you land on is decided by where the sweep count stops:
//!
//!    ```text
//!    |e|  …809 →  …808 →  …810 →  …810 →  …812 →  …812 →  …814 → …
//!    ratio      −3 ulp   +4 ulp    = 1    +3 ulp    = 1    +3 ulp
//!    n = 39 → ONE + 3 ulp (> 1)        n = 40 → ONE exactly (= 1)
//!    n = 999 → ONE + 3 ulp             n = 1000 → ONE exactly
//!    n = 99999 → ONE + 3 ulp           n = 100000 → ONE exactly
//!    ```
//!
//!    ⚠️ So it never washes out with more sweeps — it is a parity artefact, not
//!    a transient. The **two-sweep** ratio `|eʲ⁺²|/|eʲ|` reads above one at
//!    *both* phases, and is therefore the reading that does not depend on where
//!    the loop happened to stop.
//!    ⚠️ One lattice step **below** the critical ratio, the reading of `1` is
//!    **correct rather than blind**: truncation gives the iteration a genuine
//!    fixed point there — `|e|` drops one ulp and is then exactly constant
//!    forever — so the decay really has stopped, and "marginal" is the truth
//!    about the implemented map. [`ContractionMonitor`] reports `Stagnated` for
//!    all three lattice points, which is safe but does not name the divergence
//!    above the critical ratio.
//!    `f64` sees none of this: `1 ± 2⁻⁶⁴` both round to `1.0` before the
//!    iteration starts.
//! 2. **A strong contraction reaches the *squaring* floor before the ratio can
//!    be read twice.** The skew residual's L2 norm reads exactly zero by sweep
//!    8 at `a = 20` and by sweep 5 at `a = 100` — which a magnitude-based
//!    convergence test reads as success.
//!    ⚠️ The **iterate itself is still non-zero** there: what it has dropped
//!    below is not `2⁻⁶⁴` but `√(2⁻⁶⁴) = 2⁻³²`, the magnitude whose *square*
//!    survives truncation, so `residual_norm_l2_checked` returns `Ok(0)` by its
//!    documented contract rather than refusing. The two floors are nine orders
//!    of magnitude apart and the squaring one binds first.
//! 3. **The nonlinear fold is limited by [`Fix128::exp`], not by `2⁻⁶⁴`.**
//!    `exp` routes through `powf_pos`, which consumes only the top 24 fraction
//!    bits of its exponent, so its relative resolution is `ln 2 · 2⁻²⁴ =
//!    4.13e-8` — about **twelve** orders of magnitude above the
//!    `2⁻⁶⁴ = 5.42e-20` word floor (measured worst `4.005e-8`, a ratio of
//!    `7.39e11`). The Aitken limit lands `1.8e-8 … 5.0e-8` away from the true
//!    root *in relative terms* in consequence, while its residual under the
//!    *implemented* map is exactly zero.
//!
//! # The truncation directions these rest on (measured)
//!
//! ```text
//! (+2⁻⁶⁴) * 0.5  ->  0        (-2⁻⁶⁴) * 0.5  ->  -2⁻⁶⁴     Mul: floor toward −∞
//! (+1) / 3       ->  6148914691236517205
//! (−1) / 3       -> −6148914691236517205                    Div: truncate toward zero
//! ```
//!
//! ⚠️ The two differ, so the bias of the staggered sweep `(rhs − m_f·a)/D`
//! depends on the sign of the iterate — which is what makes the growth above
//! the critical ratio **period-2** rather than smooth. With `|e₀| = 0.5` the
//! exact change is `0.5` ulp per sweep in either direction, and measured
//! against that:
//!
//! | `m_f/m` | exact | measured | effect of the bias |
//! |---|---|---|---|
//! | `1 − 2⁻⁶⁴` | `−0.5` ulp/sweep | `−1` ulp once, then **exactly `0`** | the decay is arrested outright: the truncated map acquires a fixed point |
//! | `1` | `0` | `0` (bit-identical forever) | none; `×ONE` is exact |
//! | `1 + 2⁻⁶⁴` | `+0.5` ulp/sweep | `+1` ulp/sweep, as `+2, 0, +2, 0, …` | the growth roughly doubles, and arrives in alternate sweeps |
//!
//! ⚠️ The bias therefore acts on `|e|` in the **conservative** direction on both
//! sides (it never makes a growing error look like a shrinking one). What is
//! *not* conservative is reading a single-step ratio of a period-2 sequence.
//!
//! # What is not measured here
//!
//! - Nothing in `src/` is exercised beyond [`Fix128`] and
//!   [`alice_physics::coupled_iteration`]. These are scene-level fixed-point
//!   maps, not the crate's coupling call sites.
//! - The scenes are one- and few-degree-of-freedom model problems with
//!   closed-form error maps. They say nothing about a discretised field.
//! - No claim is made below `2⁻⁶⁴` in `Fix128` or below `2⁻⁵³` in `f64`.

#![cfg(feature = "std")]
// The `f64` arithmetic here is the control arm of a fixed-point-versus-float
// comparison, plus closed-form oracle references — never simulation state — so
// the determinism gate on float transcendentals does not apply. `f64::exp` is
// used for the reference root of the nonlinear scene.
#![allow(clippy::disallowed_methods)]

use alice_physics::coupled_iteration::{residual_norm_inf, SubIterationConfig};
use alice_physics::math::Fix128;

// ---------------------------------------------------------------------------
// Instruments
// ---------------------------------------------------------------------------

fn raw(v: Fix128) -> i128 {
    (i128::from(v.hi) << 64) | i128::from(v.lo)
}

fn ulp_gap(a: Fix128, b: Fix128) -> u128 {
    raw(a).abs_diff(raw(b))
}

/// `|e|` of the one-ulp lattice: `2⁻⁶⁴` as the raw integer `1`.
const ONE_ULP: Fix128 = Fix128::from_raw(0, 1);

/// `m_f/m` as an exact lattice offset from the critical ratio.
fn critical_plus(ulps: i64) -> Fix128 {
    let mut v = Fix128::ONE;
    for _ in 0..ulps.abs() {
        if ulps > 0 {
            v = v + ONE_ULP;
        } else {
            v = v - ONE_ULP;
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Scene: piston on a spring driving an incompressible fluid column
//
//   structure : m·a = −k·x − p·A        fluid: p = (m_f/A)·a
//   monolithic: (m + m_f)·a = −k·x      staggered: a⁺ = (−k·x − m_f·a)/m
//
// Written in the Δt-free form with `m = k = x = 1`, so the exact fixed point is
// `a* = −1/(1 + r)` with `r = m_f/m`, and the exact error map is `e⁺ = −r·e`.
// Dropping Δt isolates the splitting from the time stepping: the contraction
// ratio is then `r` itself, which puts the critical point exactly on `r = 1`
// and makes the lattice probe below meaningful.
// ---------------------------------------------------------------------------

fn fixed_point_fix(r: Fix128) -> Fix128 {
    -(Fix128::ONE / (Fix128::ONE + r))
}

fn fixed_point_f64(r: f64) -> f64 {
    -1.0 / (1.0 + r)
}

/// One fluid-then-structure sweep, no relaxation.
fn sweep_fix(r: Fix128, a: Fix128) -> Fix128 {
    -Fix128::ONE - r * a
}

fn sweep_f64(r: f64, a: f64) -> f64 {
    -1.0 - r * a
}

/// The raw error magnitudes of the un-relaxed iteration from `a₀ = 0`.
fn error_trace_fix(r: Fix128, sweeps: u32) -> Vec<u128> {
    let star = fixed_point_fix(r);
    let mut a = Fix128::ZERO;
    (0..sweeps)
        .map(|_| {
            a = sweep_fix(r, a);
            raw((a - star).abs()).unsigned_abs()
        })
        .collect()
}

/// What a caller reads off the iteration: the ratio of successive errors.
fn measured_ratio_fix(r: Fix128, sweeps: u32) -> Fix128 {
    let star = fixed_point_fix(r);
    let mut a = Fix128::ZERO;
    let mut previous = Fix128::ZERO;
    let mut ratio = Fix128::ZERO;
    for sweep in 0..sweeps {
        a = sweep_fix(r, a);
        let error = (a - star).abs();
        if sweep >= 1 && !previous.is_zero() {
            ratio = error / previous;
        }
        previous = error;
    }
    ratio
}

fn measured_ratio_f64(r: f64, sweeps: u32) -> f64 {
    let star = fixed_point_f64(r);
    let mut a = 0.0f64;
    let mut previous = 0.0f64;
    let mut ratio = 0.0f64;
    for sweep in 0..sweeps {
        a = sweep_f64(r, a);
        let error = (a - star).abs();
        if sweep >= 1 && previous != 0.0 {
            ratio = error / previous;
        }
        previous = error;
    }
    ratio
}

/// Sweeps used by every lattice probe. Large enough that one ulp of exact decay
/// (`0.5` ulp of magnitude per sweep from `|e₀| = 0.5`) accumulates past the
/// per-sweep truncation drift, small enough that `r = 2` has not yet wrapped.
const LATTICE_SWEEPS: u32 = 40;

// ---------------------------------------------------------------------------
// Hypothesis 1 — "the added-mass ratio defeats relaxed staggered too"
// ---------------------------------------------------------------------------

/// Verdict in `f64`: rejected. Optimal constant relaxation `ω* = m/(m + m_f)`
/// annihilates the error in a single sweep at every mass ratio, so a
/// one-degree-of-freedom scene has no discriminating power against a *relaxed*
/// splitting. This re-measures the same claim in `Fix128`.
///
/// The error map of the relaxed sweep is `e⁺ = (1 − ω(1 + r))·e`, so the whole
/// claim is the single number `1 − ω(1 + r)`. In exact arithmetic it is zero.
/// In `Fix128`, `ω = 1/(1 + r)` truncates **toward zero**, hence `ω ≤ ω*`,
/// hence the factor is **non-negative** and bounded by `(1 + r)·2⁻⁶⁴` — a
/// one-directional residue that grows linearly in the mass ratio. Measured:
///
/// | `r` | `Fix128` factor (ulp) | `(1 + r)` bound | `f64` factor |
/// |---|---|---|---|
/// | `2` | `1` | `3` | `0` |
/// | `10` | `5` | `11` | `0` |
/// | `100` | `79` | `101` | `0` |
/// | `10⁶` | `924633` | `1000001` | `0` |
///
/// ⚠️ The verdict is unchanged, but the mechanism is not "exactly zero": the
/// relaxed splitting stays a one-sweep scheme only while `(1 + r)·2⁻⁶⁴` sits
/// below the caller's tolerance, i.e. while `r ≪ tol·2⁶⁴`.
#[test]
fn optimal_constant_relaxation_still_collapses_the_error_in_one_sweep() {
    let one = Fix128::ONE;
    let cases: [(i64, i64); 10] = [
        (2, 1),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1_000_000, 1),
        (9, 10),
        (99, 100),
        (1, 1),
        (101, 100),
        (11, 10),
    ];

    for (num, den) in cases {
        let r = Fix128::from_ratio(num, den);
        let omega = one / (one + r);
        let factor = one - omega * (one + r);

        // Truncation direction: `ω` is truncated toward zero from `ω*`, so the
        // residual factor can only be non-negative.
        assert!(
            raw(factor) >= 0,
            "r = {num}/{den}: the relaxation residue must inherit the \
             toward-zero truncation of omega and stay non-negative, got raw {}",
            raw(factor)
        );
        // And bounded by (1 + r)·2⁻⁶⁴, the closed-form worst case of one
        // truncation of `ω` amplified by `(1 + r)`.
        let bound = (raw(one + r) >> 64) + 1;
        assert!(
            raw(factor) <= bound,
            "r = {num}/{den}: residue {} ulp exceeds the closed-form bound \
             (1 + r) = {bound} ulp",
            raw(factor)
        );

        // The f64 control on the identical scene: rounding to nearest leaves
        // the factor at zero or half an epsilon, never a one-directional bias.
        let rf = num as f64 / den as f64;
        let omega_f = 1.0 / (1.0 + rf);
        let factor_f = 1.0 - omega_f * (1.0 + rf);
        assert!(
            factor_f.abs() <= 4.0 * f64::EPSILON,
            "r = {num}/{den}: the f64 control should annihilate the error, got {factor_f:e}"
        );

        // Driving one actual relaxed sweep from a non-zero seed. A seed of zero
        // would make `a₁ = −ω` and `a*` the *same division*, so the agreement
        // would be a tautology rather than a measurement.
        let seed = Fix128::from_ratio(1, 4);
        let star = fixed_point_fix(r);
        let g = sweep_fix(r, seed);
        let a1 = (one - omega) * seed + omega * g;
        let e1 = (a1 - star).abs();
        assert!(
            raw(e1) <= bound + 1,
            "r = {num}/{den}: one relaxed sweep from a non-zero seed left {} ulp \
             of error, above the bound {} ulp",
            raw(e1),
            bound + 1
        );
    }
}

/// The fixed-point-only finding: one lattice step above the critical mass
/// ratio, a **single-step** contraction ratio reads `1` or `> 1` according to
/// the parity of the sweep count, because the error magnitude grows with
/// period 2. A **two-sweep** ratio reads `> 1` at either parity.
///
/// ⚠️ This pins a defect of the single-step ratio, not of the arithmetic's
/// resolution: the divergence *is* representable, and the odd-sweep reading
/// finds it. If the parity dependence ever disappears, this test goes red and
/// the period-2 trace in the module doc must be re-measured.
#[test]
fn the_single_step_ratio_above_the_critical_mass_ratio_depends_on_sweep_parity() {
    let one = Fix128::ONE;
    let above = critical_plus(1);

    // The growth is period 2: `+2, 0, +2, 0, …` after the first sweep.
    let trace = error_trace_fix(above, 13);
    let deltas: Vec<i128> = trace
        .windows(2)
        .map(|w| w[1] as i128 - w[0] as i128)
        .collect();
    // `deltas[0]` is the first-sweep transient (`-1`); the period-2 pattern
    // starts after it. `chunks_exact` so a trailing half-period is not compared
    // against a full one.
    assert_eq!(deltas[0], -1, "measured deltas {deltas:?}");
    let periodic = &deltas[1..];
    assert!(
        periodic.chunks_exact(2).all(|c| c == [2, 0]),
        "at m_f/m = 1 + 1 ulp the growth must arrive as +2, 0, +2, 0, …; \
         measured deltas {deltas:?}"
    );
    assert!(
        periodic.chunks_exact(2).count() >= 5,
        "the period-2 claim needs several periods to stand on; only {} were \
         compared from {deltas:?}",
        periodic.chunks_exact(2).count()
    );
    assert!(
        trace[12] > trace[0],
        "the error magnitude must grow overall; it went {} -> {}",
        trace[0],
        trace[12]
    );

    // Hence the single-step ratio alternates, and no sweep count washes it out.
    for even in [40u32, 1000, 100_000] {
        assert_eq!(
            measured_ratio_fix(above, even),
            one,
            "at an even sweep count ({even}) the single-step ratio lands on the \
             zero-growth phase and reads exactly ONE"
        );
    }
    for odd in [39u32, 999, 99_999] {
        assert!(
            measured_ratio_fix(above, odd) > one,
            "at an odd sweep count ({odd}) the single-step ratio lands on the \
             growth phase and must read above one; raw {}",
            raw(measured_ratio_fix(above, odd))
        );
    }

    // ⚠️ The reading that does not depend on the parity: two sweeps per ratio.
    for start in [8usize, 9] {
        let numerator = trace[start + 2];
        let denominator = trace[start];
        assert!(
            numerator > denominator,
            "the two-sweep ratio must exceed one at phase {start}; |e| went \
             {denominator} -> {numerator}"
        );
    }

    // ⚠️ One step *below* the critical ratio the reading of one is correct, not
    // blind: truncation arrests the decay outright, so the sequence really is
    // constant and the monitor's "no improvement" is the truth about the
    // implemented map.
    let below = critical_plus(-1);
    let trace_below = error_trace_fix(below, 13);
    assert!(
        trace_below[1] < trace_below[0],
        "the first sweep below the critical ratio must still shed an ulp; \
         {} -> {}",
        trace_below[0],
        trace_below[1]
    );
    assert!(
        trace_below[1..].windows(2).all(|w| w[0] == w[1]),
        "after that single ulp the truncated map must sit on an exact fixed \
         point, so the trace is constant; measured {trace_below:?}"
    );

    // Exactly critical: `×ONE` is exact in `Fix128`, so the marginal case is
    // reproduced bit-for-bit from the very first sweep.
    let trace_critical = error_trace_fix(one, 13);
    assert!(
        trace_critical.windows(2).all(|w| w[0] == w[1]),
        "m_f/m = 1 is marginal and multiplication by ONE is exact, so the error \
         magnitude must be bit-identical at every sweep; measured \
         {trace_critical:?}"
    );
    assert_eq!(
        measured_ratio_fix(one, LATTICE_SWEEPS),
        one,
        "and the measured ratio must be exactly ONE"
    );

    // Two lattice steps out, the single-step ratio is correct at this parity on
    // both sides, so the parity effect is confined to the first step above.
    for ulps in [-4i64, -2] {
        let ratio = measured_ratio_fix(critical_plus(ulps), LATTICE_SWEEPS);
        assert!(
            ratio <= one,
            "m_f/m = 1 {ulps} ulp does not grow, so the ratio must not read \
             above one; raw {}",
            raw(ratio)
        );
    }
    for ulps in [2i64, 4] {
        let ratio = measured_ratio_fix(critical_plus(ulps), LATTICE_SWEEPS);
        assert!(
            ratio > one,
            "m_f/m = 1 +{ulps} ulp diverges fast enough to show at either \
             parity; raw {}",
            raw(ratio)
        );
    }

    // And none of this extends to the ordinary ratios the oracle sweeps, so it
    // is a lattice-scale effect and not a broken instrument.
    for &(num, den) in &[(9i64, 10i64), (99, 100)] {
        assert!(
            measured_ratio_fix(Fix128::from_ratio(num, den), LATTICE_SWEEPS) < one,
            "m_f/m = {num}/{den} must still read as contracting"
        );
    }
    for &(num, den) in &[(101i64, 100i64), (11, 10), (2, 1)] {
        assert!(
            measured_ratio_fix(Fix128::from_ratio(num, den), LATTICE_SWEEPS) > one,
            "m_f/m = {num}/{den} must still read as diverging"
        );
    }
}

/// Why the `f64` probe could not have found any of this: its own lattice is
/// `2⁻⁵²` wide at one, so the three distinct `Fix128` mass ratios collapse to a
/// single `f64` value before the iteration begins.
#[test]
fn the_f64_probe_is_structurally_blind_to_the_lattice_scale_effects() {
    let two_pow_minus_64 = 2.0f64.powi(-64);
    assert!(
        two_pow_minus_64 > 0.0,
        "the offset itself must be representable"
    );
    assert_eq!(
        1.0f64 + two_pow_minus_64,
        1.0,
        "f64 cannot separate 1 + 2^-64 from 1, so an f64 probe of the critical \
         point measures only the critical point itself"
    );
    assert_eq!(1.0f64 - two_pow_minus_64, 1.0, "nor 1 - 2^-64 from 1");

    // The three Fix128 ratios are genuinely distinct, so the blindness is the
    // float's and not the scene's.
    assert_ne!(critical_plus(-1), Fix128::ONE);
    assert_ne!(critical_plus(1), Fix128::ONE);
    assert_eq!(ulp_gap(critical_plus(1), Fix128::ONE), 1);

    // The f64 reading one lattice step above the critical ratio is exactly the
    // marginal reading, and it is the same at both parities — the parity effect
    // cannot exist in f64 because the perturbation does not survive the input.
    for sweeps in [LATTICE_SWEEPS, LATTICE_SWEEPS - 1] {
        let ratio = measured_ratio_f64(1.0 + two_pow_minus_64, sweeps);
        assert!(
            (ratio - 1.0).abs() < 1e-15,
            "the f64 ratio at 1 + 2^-64 reads marginal at {sweeps} sweeps, got {ratio}"
        );
    }
}

/// The parity effect does not open a silent-success path:
/// [`ContractionMonitor`] tracks the best residual rather than a ratio, so at
/// all three lattice points it refuses.
///
/// ⚠️ It refuses with `Stagnated` rather than `Diverging` one step above the
/// critical ratio, so it is safe without naming the divergence. This is the
/// control that makes the previous test a measurement of the instrument rather
/// than a report of a hazard.
#[test]
fn the_monitor_refuses_at_every_lattice_point_rather_than_reporting_success() {
    let one = Fix128::ONE;
    for (label, r) in [
        ("1 - 1 ulp", critical_plus(-1)),
        ("1", one),
        ("1 + 1 ulp", critical_plus(1)),
    ] {
        let star = fixed_point_fix(r);
        let mut a = Fix128::ZERO;
        let outcome = alice_physics::coupled_iteration::run_sub_iteration(
            SubIterationConfig::default(),
            |_| {
                a = sweep_fix(r, a);
                residual_norm_inf(&[a - star])
            },
        );
        assert!(
            outcome.is_err(),
            "m_f/m = {label} is at best marginal, so the monitor must not report \
             convergence; it returned {outcome:?}"
        );
    }

    // Control: a splitting that genuinely diverges is still named as diverging,
    // so the refusal above is not simply a monitor that refuses everything.
    let r = Fix128::from_int(2);
    let star = fixed_point_fix(r);
    let mut a = Fix128::ZERO;
    let outcome =
        alice_physics::coupled_iteration::run_sub_iteration(SubIterationConfig::default(), |_| {
            a = sweep_fix(r, a);
            residual_norm_inf(&[a - star])
        });
    assert!(
        matches!(
            outcome,
            Err(alice_physics::coupled_iteration::CoupledIterationError::Diverging { .. })
        ),
        "m_f/m = 2 must be reported as Diverging, got {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// Hypothesis 2 — "a spread of eigenvalues defeats a constant ω"
// ---------------------------------------------------------------------------

/// Verdict in `f64`: **partly holds** — this is the one hypothesis that was not
/// rejected. With coupling ratios spread over a range, a single constant `ω`
/// cannot annihilate every mode, and its worst-case contraction degrades toward
/// one as the spread grows.
///
/// The closed form, for `r` spread over `[r_min, r_max]` and the optimal
/// constant `ω = 2/(2 + r_min + r_max)`, is
///
/// ```text
/// max_i |1 − ω(1 + r_i)| = (r_max − r_min) / (2 + r_min + r_max)
/// ```
///
/// Measured with `r_min = 1/2`, worst case at the endpoints:
///
/// | spread | closed form | `f64` | `Fix128` | gap |
/// |---|---|---|---|---|
/// | `1` | `0` | `0` | `0` | 1 ulp |
/// | `10` | `0.6` | `0.600000000000000` | `0.600000000000000` | 1 ulp |
/// | `100` | `0.942857142857143` | `0.942857142857143` | `0.942857142857143` | 1 ulp |
/// | `1000` | `0.994029850746269` | `0.994029850746269` | `0.994029850746269` | 1 ulp |
///
/// ⚠️ The verdict is unchanged, and so is its weight: it says a constant `ω`
/// degrades, not that a *sub-iterated* partitioned scheme fails. The quantity
/// is `O(1)`, far from the rounding floor, which is why fixed point reproduces
/// it to a single ulp.
#[test]
fn an_eigenvalue_spread_degrades_constant_relaxation_identically_in_f64_and_fix128() {
    let one = Fix128::ONE;
    let two = Fix128::from_int(2);
    // `r_min = 1/2` and `r_max = spread/2` are both exact in Fix128, so the
    // closed form can be evaluated as an exact Fix128 rational and the
    // comparison is not polluted by a conversion.
    let mut previous = Fix128::ZERO;
    for spread in [1i64, 10, 100, 1000] {
        let r_min = Fix128::from_ratio(1, 2);
        let r_max = Fix128::from_ratio(spread, 2);
        let omega = two / (two + r_min + r_max);
        let worst = {
            let lo = (one - omega * (one + r_min)).abs();
            let hi = (one - omega * (one + r_max)).abs();
            if lo > hi {
                lo
            } else {
                hi
            }
        };
        let closed = (r_max - r_min) / (two + r_min + r_max);
        assert!(
            ulp_gap(worst, closed) <= 4,
            "spread {spread}: Fix128 worst factor {worst:?} is {} ulp from the \
             closed form {closed:?}",
            ulp_gap(worst, closed)
        );

        // The f64 control on the identical scene.
        let r_min_f = 0.5f64;
        let r_max_f = spread as f64 / 2.0;
        let omega_f = 2.0 / (2.0 + r_min_f + r_max_f);
        let worst_f = (1.0 - omega_f * (1.0 + r_min_f))
            .abs()
            .max((1.0 - omega_f * (1.0 + r_max_f)).abs());
        let closed_f = (r_max_f - r_min_f) / (2.0 + r_min_f + r_max_f);
        assert!(
            (worst_f - closed_f).abs() <= 8.0 * f64::EPSILON,
            "spread {spread}: f64 worst factor {worst_f} differs from the closed \
             form {closed_f}"
        );
        // The two arms must agree, which is the statement that the verdict did
        // not change.
        assert!(
            (worst.to_f64() - worst_f).abs() < 1e-15,
            "spread {spread}: Fix128 read {} and f64 read {worst_f}; a verdict \
             that depended on the number system would show up here",
            worst.to_f64()
        );

        // Degradation is monotone in the spread, and the no-spread case is the
        // control: a single `r` is annihilated exactly, so the instrument is not
        // simply reporting a number close to one for every input.
        if spread == 1 {
            assert!(
                raw(worst) <= 4,
                "a single coupling ratio must be annihilated by the optimal \
                 constant omega, got raw {}",
                raw(worst)
            );
        } else {
            assert!(
                worst > previous,
                "widening the spread to {spread} must degrade the worst-case \
                 contraction, but it went {previous:?} -> {worst:?}"
            );
        }
        previous = worst;
    }

    // And the degradation is still a contraction: the hypothesis is that a
    // constant omega gets *slow*, not that it diverges.
    assert!(
        previous < one,
        "even at spread 1000 the worst-case factor must stay below one, got {previous:?}"
    );
}

// ---------------------------------------------------------------------------
// Hypothesis 3 — "skew coupling defeats staggered"
// ---------------------------------------------------------------------------

/// Verdict in `f64`: **rejected, and the hypothesis was backwards** — a skew
/// coupling term (the `v×B` of a Lorentz force) *helps* the splitting. The
/// error map becomes multiplication by `1/(1 + i·a)`, whose magnitude
/// `1/√(1 + a²)` is below one for every non-zero `a`.
///
/// Written on real pairs, `z ↦ z/(1 + i·a)` is
/// `(x, y) ↦ ((x + a·y)/(1 + a²), (y − a·x)/(1 + a²))`, and `1 + a²` is an
/// exact integer in `Fix128` for integer `a`, so the only truncation is the
/// division. Comparing squares avoids comparing against a `sqrt`:
///
/// | `a` | closed form `1/√(1+a²)` | `f64` sweep 1 | `Fix128` sweep 1 | `ratio²` gap |
/// |---|---|---|---|---|
/// | `1` | `0.707106781186548` | `0.707106781186548` | `0.707106781186548` | 1 ulp |
/// | `5` | `0.196116135138184` | `0.196116135138184` | `0.196116135138184` | 2 ulp |
/// | `20` | `0.049937616943892` | `0.049937616943892` | `0.049937616943892` | 2 ulp |
/// | `100` | `0.009999500037497` | `0.009999500037497` | `0.009999500037497` | 1 ulp |
///
/// ⚠️ The verdict is unchanged. The record of the `f64` probe quotes `0.04` at
/// `a = 20`; the closed form is `0.0499376…`, so that figure was rounded in
/// transcription — it does not indicate a different measurement.
#[test]
fn skew_coupling_contracts_under_fix128_exactly_as_the_closed_form_says() {
    let one = Fix128::ONE;

    // Control first: no skew means no contraction, so a test that passed for
    // every `a` would be measuring nothing.
    {
        let denom = one; // 1 + 0²
        let (mut x, mut y) = (one, Fix128::ZERO);
        let nx = (x + Fix128::ZERO * y) / denom;
        let ny = (y - Fix128::ZERO * x) / denom;
        (x, y) = (nx, ny);
        let magnitude = (x * x + y * y).sqrt();
        assert_eq!(
            magnitude, one,
            "a = 0 is the uncoupled case and must neither contract nor grow"
        );
    }

    for a_int in [1i64, 2, 5, 20, 100] {
        let a = Fix128::from_int(a_int);
        let denom = one + a * a;
        let exact_square = one / denom; // (1/√(1+a²))²

        let (mut x, mut y) = (one, Fix128::ZERO);
        let nx = (x + a * y) / denom;
        let ny = (y - a * x) / denom;
        (x, y) = (nx, ny);
        let ratio = (x * x + y * y).sqrt(); // |e₁|/|e₀| with |e₀| = 1
        let square = ratio * ratio;

        assert!(
            ulp_gap(square, exact_square) <= 8,
            "a = {a_int}: the squared contraction {square:?} is {} ulp from the \
             exact rational 1/(1 + a²) = {exact_square:?}",
            ulp_gap(square, exact_square)
        );
        assert!(
            ratio < one,
            "a = {a_int}: skew coupling must contract, not grow; ratio {ratio:?}"
        );

        // f64 control on the identical scene.
        let af = a_int as f64;
        let denom_f = 1.0 + af * af;
        let (xf, yf) = ((1.0 + af * 0.0) / denom_f, (0.0 - af * 1.0) / denom_f);
        let ratio_f = (xf * xf + yf * yf).sqrt();
        let closed_f = 1.0 / denom_f.sqrt();
        assert!(
            (ratio_f - closed_f).abs() <= 8.0 * f64::EPSILON,
            "a = {a_int}: f64 ratio {ratio_f} differs from the closed form {closed_f}"
        );
        assert!(
            (ratio.to_f64() - ratio_f).abs() < 1e-15,
            "a = {a_int}: Fix128 read {} and f64 read {ratio_f}; the verdict does \
             not depend on the number system",
            ratio.to_f64()
        );
    }
}

/// The fixed-point-only consequence of hypothesis 3: the contraction is so
/// strong that the iterate drops below the **squaring** floor within a handful
/// of sweeps, after which the L2 residual reads exactly `0` while the iterate
/// is still non-zero.
///
/// ⚠️ The two floors are different and the squaring one binds first. A
/// component survives down to `2⁻⁶⁴`, but its *square* vanishes below
/// `√(2⁻⁶⁴) = 2⁻³² = 2.33e-10`, which is exactly
/// [`L2_TERM_FLOOR`](alice_physics::coupled_iteration::L2_TERM_FLOOR). So
/// `residual_norm_l2_checked` returns `Ok(0)` here rather than refusing — by
/// its documented contract, since nothing wrapped — and a magnitude-based
/// convergence test reads that `0` as success.
///
/// Measured first sweep at which the norm reads zero: `a = 5` -> 14,
/// `a = 20` -> 8, `a = 100` -> 5. The `f64` arm does not underflow over the
/// same span.
#[test]
fn the_skew_iterate_reaches_the_rounding_floor_before_the_ratio_can_be_read_twice() {
    use alice_physics::coupled_iteration::{residual_norm_l2_checked, L2_TERM_FLOOR};

    let one = Fix128::ONE;
    for (a_int, expected_by) in [(5i64, 14u32), (20, 8), (100, 5)] {
        let a = Fix128::from_int(a_int);
        let denom = one + a * a;
        let (mut x, mut y) = (one, Fix128::ZERO);
        let mut zero_at = None;
        for sweep in 1..=20u32 {
            let nx = (x + a * y) / denom;
            let ny = (y - a * x) / denom;
            (x, y) = (nx, ny);
            let norm = (x * x + y * y).sqrt();
            if norm.is_zero() && zero_at.is_none() {
                zero_at = Some(sweep);
                // ⚠️ The iterate itself has not vanished — only its square has.
                assert!(
                    !x.is_zero() || !y.is_zero(),
                    "a = {a_int}: the point of this test is that the norm reads \
                     zero while the iterate does not; if the iterate has also \
                     reached zero the squaring floor is no longer what binds"
                );
                let largest = if x.abs() > y.abs() { x.abs() } else { y.abs() };
                assert!(
                    largest < L2_TERM_FLOOR,
                    "a = {a_int}: the components must be below L2_TERM_FLOOR for \
                     the zero norm to be truncation working as designed; largest \
                     was {largest:?}"
                );
                // Hence the crate's guard accepts it: nothing wrapped.
                assert_eq!(
                    residual_norm_l2_checked(&[x, y]),
                    Ok(Fix128::ZERO),
                    "a = {a_int}: below the term floor the checked norm reports a \
                     faithful zero, which is what makes this a reachability \
                     problem rather than a wrap"
                );
            }
        }
        let at = zero_at.unwrap_or_else(|| {
            panic!(
                "a = {a_int}: the Fix128 L2 residual must read zero within 20 \
                 sweeps; if it no longer does, this measurement is stale"
            )
        });
        assert!(
            at <= expected_by,
            "a = {a_int}: expected a zero norm by sweep {expected_by}, got it at {at}"
        );

        // The f64 arm over the same span: still non-zero, so the effect is the
        // fixed-point word's and not the contraction's.
        let af = a_int as f64;
        let denom_f = 1.0 + af * af;
        let (mut xf, mut yf) = (1.0f64, 0.0f64);
        for _ in 1..=20u32 {
            let nxf = (xf + af * yf) / denom_f;
            let nyf = (yf - af * xf) / denom_f;
            (xf, yf) = (nxf, nyf);
        }
        assert!(
            xf != 0.0 || yf != 0.0,
            "a = {a_int}: the f64 arm must not underflow over 20 sweeps, which is \
             what makes the Fix128 floor a fixed-point-only effect"
        );
    }
}

// ---------------------------------------------------------------------------
// Hypothesis 4 — "a nonlinear fold defeats Aitken"
// ---------------------------------------------------------------------------

/// `u ↦ Λ·e^u`, the lumped Joule-heating map. Below `Λ = 1/e` it has two roots:
/// a stable lower branch (`g' = u < 1`) and an **unstable** upper branch
/// (`g' = u > 1`). Plain iteration cannot reach the upper branch from any seed;
/// the hypothesis was that neither could Aitken.
fn joule_map_fix(lambda: Fix128, u: Fix128) -> Fix128 {
    lambda * u.exp()
}

/// Aitken's Δ² acceleration of `g`, which reaches a repelling fixed point by
/// extrapolating with an effective relaxation factor outside `(0, 1)`.
fn aitken_fix(lambda: Fix128, seed: Fix128, steps: u32) -> Fix128 {
    let two = Fix128::from_int(2);
    let mut u = seed;
    for _ in 0..steps {
        let u1 = joule_map_fix(lambda, u);
        let u2 = joule_map_fix(lambda, u1);
        let second_difference = u2 - two * u1 + u;
        if second_difference.is_zero() {
            break;
        }
        let next = u2 - (u2 - u1) * (u2 - u1) / second_difference;
        if next == u {
            break;
        }
        u = next;
    }
    u
}

/// Newton on `f(u) = u − Λ·e^u` in `f64`, the reference root.
fn joule_root_f64(lambda: f64, seed: f64) -> f64 {
    let mut u = seed;
    for _ in 0..200 {
        let e = u.exp();
        u -= (u - lambda * e) / (1.0 - lambda * e);
    }
    u
}

/// Verdict in `f64`: rejected. Aitken reaches the unstable upper branch at
/// `Λ = 0.20 / 0.30 / 0.35`, so the fold does not separate partitioned from
/// monolithic — and at the fold itself the Jacobian is singular, where a
/// monolithic Newton is equally badly conditioned.
///
/// Measured in `Fix128` (upper branch from a seed 10% below it):
///
/// | `Λ` | upper root | lower root | `Fix128` Aitken | sweeps | residual | relative displacement |
/// |---|---|---|---|---|---|---|
/// | `0.20` | `2.542641357773526` | `0.259171101819074` | `2.542641404254685` | 9 | `0` | `1.83e-8` |
/// | `0.30` | `1.781337023421627` | `0.489402227180215` | `1.781337088677384` | 8 | `0` | `3.66e-8` |
/// | `0.35` | `1.349717252192249` | `0.716638816456074` | `1.349717319476777` | 15 | `0` | `4.99e-8` |
///
/// ⚠️ The residual under the *implemented* map is exactly zero while the root
/// is displaced by `~1e-8`: Aitken converges to the fixed point of
/// `Λ·Fix128::exp(u)`, which is not the fixed point of `Λ·e^u`. The
/// displacement tracks `exp`'s resolution, not the word's — see the next test.
#[test]
fn aitken_reaches_the_unstable_upper_branch_under_fix128_too() {
    for (num, den) in [(20i64, 100i64), (30, 100), (35, 100)] {
        let lambda_f = num as f64 / den as f64;
        assert!(
            lambda_f < 1.0 / core::f64::consts::E,
            "Lambda = {num}/{den} must sit below the fold at 1/e for two roots to exist"
        );
        let upper = joule_root_f64(lambda_f, 3.0);
        let lower = joule_root_f64(lambda_f, 0.1);
        // The upper root is the repelling one: g'(u) = u > 1 there.
        assert!(
            upper > 1.0 && lower < 1.0,
            "Lambda = {num}/{den}: expected an unstable upper root above one and a \
             stable lower root below it, got {upper} and {lower}"
        );

        let lambda = Fix128::from_ratio(num, den);
        let seed = Fix128::from_f64(upper * 0.9);
        let limit = aitken_fix(lambda, seed, 80);

        // It landed on the upper branch, not back on the lower one.
        assert!(
            limit.to_f64() > 1.0,
            "Lambda = {num}/{den}: Aitken must reach the unstable branch above one, \
             got {}",
            limit.to_f64()
        );
        let to_upper = (limit.to_f64() - upper).abs();
        let to_lower = (limit.to_f64() - lower).abs();
        assert!(
            to_upper * 1e6 < to_lower,
            "Lambda = {num}/{den}: the limit {} must be unambiguously the upper root \
             ({to_upper} away) rather than the lower one ({to_lower} away)",
            limit.to_f64()
        );
        assert!(
            to_upper / upper < 1e-6,
            "Lambda = {num}/{den}: relative displacement {} exceeds the resolution \
             of Fix128::exp",
            to_upper / upper
        );

        // It is a genuine fixed point of the implemented map.
        let residual = (limit - joule_map_fix(lambda, limit)).abs();
        assert!(
            raw(residual) <= 2,
            "Lambda = {num}/{den}: the Aitken limit must be a fixed point of the \
             implemented map, residual raw {}",
            raw(residual)
        );

        // ⚠️ Control: the same seed under *plain* iteration falls to the lower
        // branch. Without this the test above could be passing because the seed
        // was already the answer.
        let mut plain = seed;
        for _ in 0..80 {
            plain = joule_map_fix(lambda, plain);
        }
        assert!(
            (plain.to_f64() - lower).abs() / lower < 1e-6,
            "Lambda = {num}/{den}: plain iteration from the same seed must fall to \
             the stable lower root {lower}, got {}; if it reaches the upper branch \
             then Aitken is not what carried the previous assertion",
            plain.to_f64()
        );
    }
}

/// Where the nonlinear scene's accuracy actually comes from.
///
/// [`Fix128::exp`] evaluates `2^(x·log₂e)` through `powf_pos`, which walks only
/// **24** successive square roots and therefore consumes only the top 24
/// fraction bits of its exponent. Its relative resolution is consequently
/// `ln 2 · 2⁻²⁴ ≈ 4.13e-8`, independent of the `2⁻⁶⁴ = 5.42e-20` word.
///
/// ⚠️ So the limit on hypothesis 4 under fixed point sits about **twelve**
/// orders of magnitude above the rounding floor, and reasoning about the fold
/// from the floor would be wrong by that factor. Measured worst over the branch
/// range is `4.005e-8` at `u = 3`, just inside the derived bound, which is
/// `7.39e11` times the word.
#[test]
fn the_nonlinear_branch_is_limited_by_fix128_exp_not_by_the_rounding_floor() {
    // Derived from the algorithm, not fitted: 24 fractional bits of exponent.
    let derived_bound = core::f64::consts::LN_2 * 2.0f64.powi(-24);
    let word_floor = 2.0f64.powi(-64);

    let mut worst = 0.0f64;
    for u in [0.25f64, 0.5, 0.72, 1.0, 1.35, 1.78, 2.0, 2.54, 3.0] {
        let reference = u.exp();
        let measured = Fix128::from_f64(u).exp().to_f64();
        let relative = (measured - reference).abs() / reference;
        assert!(
            relative <= derived_bound * 1.05,
            "exp({u}) is off by {relative:e}, above the derived resolution \
             ln2 * 2^-24 = {derived_bound:e}"
        );
        if relative > worst {
            worst = relative;
        }
    }

    // The load-bearing half: the limit is `exp`, and it is far above the word.
    assert!(
        worst > 1e-9,
        "if exp has become accurate to better than 1e-9 then the module doc's \
         claim that it, and not the 2^-64 floor, limits the nonlinear scene is \
         stale; worst was {worst:e}"
    );
    assert!(
        worst / word_floor > 1e11,
        "exp's resolution must sit about twelve orders above the word floor for \
         the distinction to matter, which is what the module doc claims; the \
         ratio was {}",
        worst / word_floor
    );
}

// ---------------------------------------------------------------------------
// The instrument itself
// ---------------------------------------------------------------------------

/// Every test above reads a contraction through `measured_ratio_fix` or
/// `error_trace_fix`. If either collapsed to a constant, the sweeps would pass
/// while measuring nothing, so both are pinned against inputs whose answers are
/// known without running the scene.
#[test]
fn the_measurements_are_not_inert() {
    // A ratio of exactly one half, from a scene chosen so the answer is forced:
    // `r = 1/2` contracts by `1/2` per sweep in exact arithmetic, and the scene
    // is far enough from the floor that truncation cannot move the leading bits.
    let half = Fix128::from_ratio(1, 2);
    let ratio = measured_ratio_fix(half, 8);
    assert!(
        ulp_gap(ratio, half) <= 1 << 20,
        "the ratio instrument must read 1/2 on an r = 1/2 scene, got {ratio:?}"
    );

    // The trace must be a decreasing sequence there, and an increasing one at
    // `r = 2`, so neither direction is hard-coded.
    let decreasing = error_trace_fix(half, 8);
    assert!(
        decreasing.windows(2).all(|w| w[1] < w[0]),
        "the trace at r = 1/2 must decrease monotonically: {decreasing:?}"
    );
    let increasing = error_trace_fix(Fix128::from_int(2), 8);
    assert!(
        increasing.windows(2).all(|w| w[1] > w[0]),
        "the trace at r = 2 must increase monotonically: {increasing:?}"
    );

    // And the lattice helper really walks the lattice.
    assert_eq!(ulp_gap(critical_plus(4), Fix128::ONE), 4);
    assert_eq!(ulp_gap(critical_plus(-4), Fix128::ONE), 4);
    assert!(critical_plus(-4) < Fix128::ONE);
    assert!(critical_plus(4) > Fix128::ONE);
}
