//! Analytic-solution oracle tests for `ccd::adaptive_toi_substeps`'s wiring
//! into `solver_tgs::adaptive_substeps_for_ccd`.
//!
//! `adaptive_toi_substeps` used to be a skeleton: it ignored both bodies'
//! velocities and the collider radii entirely, returning either a
//! constant `max_substeps.max(1)` (pair on a collision course) or `1`
//! (pair not closing). The wiring replaces the constant-on-collision
//! branch with a real call into `solver_tgs::adaptive_substeps_for_ccd`,
//! so the sub-step count now actually scales with how fast the pair is
//! closing and with how small the colliders are.
//!
//! The documented formula (see `ccd::adaptive_toi_substeps`'s own doc,
//! and `solver_tgs::adaptive_substeps_for`/`adaptive_substeps_for_ccd`'s
//! doc for the upstream algorithm it now delegates to):
//!
//! 1. If the pair is not predicted to breach the gap within `dt`
//!    (`speculative_contact` returns `None`), return `1` — unchanged
//!    from the old skeleton, this branch was never a skeleton.
//! 2. Otherwise: `v_max` = the larger of the two bodies' own L∞
//!    velocity norms (not the *relative*/closing speed — each body's
//!    own speed). `effective_step` = the smaller of (a) `0.1` world
//!    units (the crate-wide `AdaptiveSubStepConfig` default) and (b)
//!    half the smaller collider's radius. `n` is then the smallest
//!    integer with `n * effective_step >= v_max * dt`, clamped into
//!    `[1, max_substeps.max(1)]`.
//!
//! Rule (analytic-oracle-tests): every expected value below is computed
//! by hand from that documented formula/clamp (see the per-case comment
//! for the arithmetic), never by calling `adaptive_toi_substeps`,
//! `speculative_contact`, or any `solver_tgs` internal (which is
//! `pub(crate)` and unreachable from this external test crate anyway) to
//! produce its own expected value.

use alice_physics::ccd::adaptive_toi_substeps;
use alice_physics::math::{Fix128, Vec3Fix};

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}
fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

/// oracle: a separating pair (closing speed <= 0, so `speculative_contact`
/// returns `None`) always gets a single sub-step, independent of
/// velocity magnitude, radii, or `max_substeps` — this branch of
/// `adaptive_toi_substeps` was never part of the skeleton that this
/// wiring replaces, so it must be unchanged.
#[test]
fn separating_pair_always_gets_one_substep() {
    // a moves left, b moves right: both leaving, closing_speed < 0.
    let n = adaptive_toi_substeps(
        Vec3Fix::ZERO,
        v(-5, 0, 0),
        Fix128::ONE,
        v(10, 0, 0),
        v(5, 0, 0),
        Fix128::ONE,
        Fix128::ONE,
        20,
    );
    assert_eq!(n, 1);
}

/// oracle: colliding pair, tiny colliders (radius 0.02 each) so the
/// CCD-derived cap (`0.02 * 0.5 = 0.01`) is tighter than the crate-wide
/// base cap (`0.1`) — `effective_step = 0.01`.
///
/// `v_max = 5` (body A's own speed; B is at rest), `dt = 1` →
/// `travel = 5`. The smallest `n` with `n * 0.01 >= 5` is `n = 500`,
/// but `max_substeps = 12` clamps it down to `12`.
///
/// `gap = dist(0,1) - (0.02+0.02) = 0.96`; `closing_speed = 5` (A moving
/// straight at stationary B); `predicted_gap = 0.96 - 5*1 = -4.04 < 0` →
/// the pair is on a collision course, so the real (non-`1`) branch runs.
#[test]
fn tiny_colliders_make_the_ccd_cap_dominate_and_clamp_at_max_substeps() {
    let n = adaptive_toi_substeps(
        Vec3Fix::ZERO,
        v(5, 0, 0),
        r(2, 100),
        v(1, 0, 0),
        Vec3Fix::ZERO,
        r(2, 100),
        Fix128::ONE,
        12,
    );
    assert_eq!(n, 12);
}

/// oracle: colliding pair, large colliders (radius 1 each) so the
/// CCD-derived cap (`1 * 0.5 = 0.5`) is *looser* than the base cap
/// (`0.1`) — `effective_step = 0.1` (the base cap dominates, same shape
/// as the crate's pre-existing `fi(2)`/`6` ccd.rs unit test, but with an
/// independently chosen velocity/radius/`max_substeps` combination).
///
/// `v_max = 3`, `dt = 1` → `travel = 3`. Smallest `n` with
/// `n * 0.1 >= 3` is `n = 30`, clamped by `max_substeps = 10` down to
/// `10`.
///
/// `gap = dist(0, 2.5) - (1+1) = 0.5`; `closing_speed = 3`;
/// `predicted_gap = 0.5 - 3*1 = -2.5 < 0` → collision course.
#[test]
fn large_colliders_make_the_base_cap_dominate_and_clamp_at_max_substeps() {
    let n = adaptive_toi_substeps(
        Vec3Fix::ZERO,
        v(3, 0, 0),
        Fix128::ONE,
        Vec3Fix::new(r(5, 2), Fix128::ZERO, Fix128::ZERO), // pos_b.x = 2.5
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ONE,
        10,
    );
    assert_eq!(n, 10);
}

/// oracle: same geometry/velocity shape as the previous case (base cap
/// `0.1` dominates), but with `v_max = 0.25` instead of `3` — small
/// enough that the sub-step count does **not** hit the `max_substeps`
/// clamp. `travel = 0.25`. Smallest `n` with `n * 0.1 >= 0.25`:
/// `n=1 -> 0.1 < 0.25`, `n=2 -> 0.2 < 0.25`, `n=3 -> 0.3 >= 0.25` — so
/// `n = 3`.
///
/// This is the case that actually distinguishes the real formula from
/// the old skeleton: the skeleton returned the constant `max_substeps`
/// (here `20`) for *every* colliding pair, never `3`.
///
/// `gap = dist(0, 2.1) - (1+1) = 0.1`; `closing_speed = 0.25`;
/// `predicted_gap = 0.1 - 0.25*1 = -0.15 < 0` → collision course.
#[test]
fn slow_closing_speed_yields_a_genuinely_scaled_substep_count_not_the_max() {
    let n = adaptive_toi_substeps(
        Vec3Fix::ZERO,
        Vec3Fix::new(r(1, 4), Fix128::ZERO, Fix128::ZERO), // v_max = 0.25
        Fix128::ONE,
        Vec3Fix::new(r(21, 10), Fix128::ZERO, Fix128::ZERO), // pos_b.x = 2.1
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ONE,
        20,
    );
    assert_eq!(n, 3);
}

/// oracle: a pair that is **already overlapping** (`gap <= 0`) is
/// reported as a collision by `speculative_contact` regardless of
/// velocity — including the degenerate case of both bodies at rest.
/// With `v_max = 0`, `travel = 0 <= effective_step` for any
/// non-negative `effective_step`, so `adaptive_substeps_for_ccd`'s early
/// return gives `min_substeps = 1` — never a panic, divide-by-zero, or
/// unspecified value, and independent of `max_substeps`.
#[test]
fn already_overlapping_and_stationary_returns_min_substeps_not_a_degenerate_value() {
    let n = adaptive_toi_substeps(
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::ONE,
        v(1, 0, 0),
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ONE,
        5,
    );
    assert_eq!(n, 1);
}

/// oracle: the `max_substeps == 0` clamp (already covered by the crate's
/// pre-existing ccd.rs unit test for the *base-cap-dominated* case) must
/// also clamp to `1` on the *CCD-cap-dominated* path (tiny colliders),
/// since `AdaptiveSubStepConfig`'s `max_substeps: max_substeps.max(1)`
/// override is applied before the CCD-vs-base cap comparison, not after.
/// Same geometry/velocity as the tiny-collider case above, with
/// `max_substeps = 0`: `cfg.max_substeps = max(0,1) = 1`,
/// `cfg.min_substeps = 1`, so the search loop's `n < max` is `1 < 1`
/// (false) on the very first check and returns the initial `n = 1`.
#[test]
fn max_substeps_zero_clamps_to_one_on_the_ccd_cap_dominated_path_too() {
    let n = adaptive_toi_substeps(
        Vec3Fix::ZERO,
        v(5, 0, 0),
        r(2, 100),
        v(1, 0, 0),
        Vec3Fix::ZERO,
        r(2, 100),
        Fix128::ONE,
        0,
    );
    assert_eq!(n, 1);
}

/// Degenerate input: zero `dt` means `travel = v_max * 0 = 0` for any
/// velocity, which is always `<= effective_step` — so even a fast,
/// genuinely colliding pair must get `min_substeps = 1`, not panic or
/// divide by zero. The pair is already overlapping (`gap = -1 <= 0`),
/// so `speculative_contact` reports a collision regardless of `dt` (the
/// `gap <= 0` branch never reads `dt`); it is the downstream
/// `adaptive_substeps_for_ccd` call, fed `dt = 0`, that this test
/// actually exercises for a div-by-zero/panic surface.
#[test]
fn zero_dt_does_not_panic_and_returns_min_substeps() {
    let result = std::panic::catch_unwind(|| {
        adaptive_toi_substeps(
            Vec3Fix::ZERO,
            v(100, 0, 0),
            Fix128::ONE,
            v(1, 0, 0),
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ZERO,
            16,
        )
    });
    assert_eq!(result.expect("must not panic on dt=0"), 1);
}
