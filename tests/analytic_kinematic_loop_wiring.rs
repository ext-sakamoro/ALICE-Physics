//! Oracles for the production entry points of `alice_physics::kinematic_loop`
//! driven by `examples/kinematic_loop_four_bar.rs`: `FourBarLinkage` (via
//! `try_four_bar_linkage`), `four_bar_linkage`, and
//! `LoopClosureConstraint::centre_to_centre`.
//!
//! # What this file is and is not
//!
//! `src/kinematic_loop.rs`'s own `#[cfg(test)]` module already checks a
//! single valid crank-rocker (`four_bar_linkage_registers_expected_bodies`,
//! `four_bar_closure_targets_ground_and_rocker`,
//! `four_bar_link_lengths_hold_at_rest_and_in_motion`) and two rejection
//! cases (`four_bar_rejects_lengths_that_cannot_close`: `d > sum` and
//! `crank_length == 0`). What those do **not** cover, and this file does:
//!
//! * the `dir = -1` branch (`ground_length < crank_length`),
//! * both Grashof-condition *boundary* configurations, where the triangle
//!   inequality on `d`/`r3`/`r4` holds with equality (fully extended:
//!   `d == r3 + r4`; folded: `d == |r3 - r4|`) -- the module's own `>`/`<`
//!   comparisons are strict, so these must succeed, not error,
//! * the `d.is_zero()` guard in isolation from the sum/diff bounds (a
//!   configuration where `ground_length == crank_length` but `r3 == r4`
//!   too, so neither `d > sum` nor `d < diff` would catch it -- without
//!   this explicit guard, `try_four_bar_linkage` would divide by zero
//!   computing `a = .. / (2*d)`),
//! * each of the four length parameters' `<= 0` guard individually (the
//!   existing test only exercises `crank_length == 0`),
//! * `d < diff` in isolation (the existing test only exercises `d > sum`),
//! * extreme-magnitude cases on both the success and error paths,
//! * `four_bar_linkage`'s documented panic behaviour on an invalid
//!   configuration (its own doc comment's `# Panics` section), and
//! * `LoopClosureConstraint::centre_to_centre`'s exact field values in
//!   isolation (the existing tests only exercise it indirectly through
//!   `LoopClosureConstraint`'s `residual`/`apply` behaviour).
//!
//! # Degenerate / extreme input summary
//!
//! * Non-positive `crank_length`, `coupler_length`, `rocker_length`, or
//!   `ground_length` (zero or negative): `Err(InvalidConfiguration)`,
//!   nothing added to `world`.
//! * `d > r3 + r4` (circles too far apart) and `d < |r3 - r4|` (one circle
//!   nested inside the other, too close): `Err(InvalidConfiguration)`.
//! * `d == 0` with `r3 == r4` (the explicit zero-distance guard, distinct
//!   from the sum/diff bounds): `Err(InvalidConfiguration)`, not a
//!   division-by-zero panic.
//! * `d == r3 + r4` and `d == |r3 - r4|` exactly (Grashof boundary):
//!   `Ok`, with the coupler pin's perpendicular offset `h` exactly zero
//!   (the three pins become collinear).
//! * Extreme magnitude (link lengths around `1e5`-`4e5`): the closed form
//!   holds at scale without overflow, and a triangle-inequality violation
//!   at a much larger extreme magnitude (`1e6` vs `1.0`) still returns
//!   `Err`, not a panic.
//! * `four_bar_linkage` on an invalid configuration: panics (per its own
//!   doc comment), verified via `std::panic::catch_unwind`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::error::PhysicsError;
use alice_physics::kinematic_loop::{
    four_bar_linkage, try_four_bar_linkage, LoopClosureConstraint,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, SolverConfig};

const FIX_TOL: f64 = 1e-9;

fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = if want == 0.0 {
        g.abs()
    } else {
        ((g - want) / want).abs()
    };
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (rel err {err:.3e} > {tol:.1e})"
    );
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig::default())
}

/// Independent f64 closed form for the coupler pin, via the law of cosines
/// on triangle `A-O4-B` (same derivation as
/// `examples/kinematic_loop_four_bar.rs`'s `four_bar_oracle_f64`,
/// duplicated here because test binaries cannot import helpers from an
/// example binary).
fn coupler_pin_f64(r2: f64, r3: f64, r4: f64, r1: f64) -> (f64, f64) {
    let d = (r1 - r2).abs();
    let a = (d * d + r3 * r3 - r4 * r4) / (2.0 * d);
    let h = (r3 * r3 - a * a).max(0.0).sqrt();
    let dir = if r1 >= r2 { 1.0 } else { -1.0 };
    (r2 + dir * a, h)
}

// ============================================================================
// Section 1: try_four_bar_linkage -- valid configurations, both dir
// branches.
// ============================================================================

#[test]
fn try_four_bar_linkage_crank_rocker_matches_circle_intersection_closed_form() {
    let (r2, r3, r4, r1) = (1.0_f64, 3.0_f64, 2.5_f64, 4.0_f64);
    let (bx, by) = coupler_pin_f64(r2, r3, r4, r1);
    let mut w = world();
    let l = try_four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2),
        Fix128::from_f64(r3),
        Fix128::from_f64(r4),
        Fix128::from_f64(r1),
        Fix128::ONE,
    )
    .expect("must close");
    assert_rel(w.bodies[l.coupler].position.x, bx, FIX_TOL, "coupler.x");
    assert_rel(w.bodies[l.coupler].position.y, by, FIX_TOL, "coupler.y");
    assert_eq!(
        w.bodies[l.crank].position.y,
        Fix128::ZERO,
        "crank pin stays on the x-axis"
    );
}

/// `dir = -1`: `ground_length < crank_length`, so O4 is to the *left* of
/// A. `src/kinematic_loop.rs`'s own tests only ever use
/// `ground_length > crank_length` (`dir = +1`).
#[test]
fn try_four_bar_linkage_dir_negative_branch_matches_closed_form() {
    let (r2, r3, r4, r1) = (4.0_f64, 3.0_f64, 2.5_f64, 1.0_f64);
    assert!(
        r1 < r2,
        "this test must exercise ground_length < crank_length"
    );
    let (bx, by) = coupler_pin_f64(r2, r3, r4, r1);
    let mut w = world();
    let l = try_four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2),
        Fix128::from_f64(r3),
        Fix128::from_f64(r4),
        Fix128::from_f64(r1),
        Fix128::ONE,
    )
    .expect("must close");
    assert_rel(
        w.bodies[l.coupler].position.x,
        bx,
        FIX_TOL,
        "coupler.x (dir=-1)",
    );
    assert_rel(
        w.bodies[l.coupler].position.y,
        by,
        FIX_TOL,
        "coupler.y (dir=-1)",
    );
    // With O4 to the left of A, the coupler pin's x must be less than A's.
    assert!(
        w.bodies[l.coupler].position.x < w.bodies[l.crank].position.x,
        "dir=-1 must place B to the left of A"
    );
}

// ============================================================================
// Section 2: Grashof-condition boundary -- d == r3+r4 (fully extended) and
// d == |r3-r4| (folded), both exactly at the inclusive/exclusive edge of
// the module's `d > sum || d < diff` rejection.
// ============================================================================

/// `d == r3 + r4` exactly: the coupler and rocker are fully extended into
/// a straight line, `h == 0`. `try_four_bar_linkage` rejects only `d >
/// sum` (strict), so this boundary must succeed.
#[test]
fn try_four_bar_linkage_fully_extended_boundary_succeeds_with_zero_height() {
    let (r2, r3, r4) = (1.0_f64, 3.0_f64, 2.5_f64);
    let d = r3 + r4; // == 5.5
    let r1 = r2 + d; // ground_length, so |r1-r2| == d exactly
    let mut w = world();
    let l = try_four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2),
        Fix128::from_f64(r3),
        Fix128::from_f64(r4),
        Fix128::from_f64(r1),
        Fix128::ONE,
    )
    .expect("d == r3+r4 must be accepted (boundary is inclusive)");
    assert_rel(
        w.bodies[l.coupler].position.y,
        0.0,
        FIX_TOL,
        "fully-extended coupler pin has zero height",
    );
    // B sits beyond A at exactly r3 along +x (collinear with O4).
    assert_rel(
        w.bodies[l.coupler].position.x,
        r2 + r3,
        FIX_TOL,
        "fully-extended coupler pin x = r2 + r3",
    );
}

/// `d == |r3 - r4|` exactly: the coupler/rocker circles are internally
/// tangent (folded configuration), `h == 0` again but on the other side
/// of the valid range. `try_four_bar_linkage` rejects only `d < diff`
/// (strict), so this boundary must also succeed.
#[test]
fn try_four_bar_linkage_folded_boundary_succeeds_with_zero_height() {
    let (r2, r3, r4) = (1.0_f64, 3.0_f64, 2.5_f64);
    let d = (r3 - r4).abs(); // == 0.5
    let r1 = r2 + d;
    let mut w = world();
    let l = try_four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2),
        Fix128::from_f64(r3),
        Fix128::from_f64(r4),
        Fix128::from_f64(r1),
        Fix128::ONE,
    )
    .expect("d == |r3-r4| must be accepted (boundary is inclusive)");
    assert_rel(
        w.bodies[l.coupler].position.y,
        0.0,
        FIX_TOL,
        "folded coupler pin has zero height",
    );
    // B sits beyond A at exactly r3 along +x, with O4 between A and B
    // (internal tangency).
    assert_rel(
        w.bodies[l.coupler].position.x,
        r2 + r3,
        FIX_TOL,
        "folded coupler pin x = r2 + r3",
    );
    let o4_x = w.bodies[l.rocker].position.x.to_f64();
    let a_x = w.bodies[l.crank].position.x.to_f64();
    let b_x = w.bodies[l.coupler].position.x.to_f64();
    assert!(
        a_x < o4_x && o4_x < b_x,
        "O4 lies between A and B at this boundary"
    );
}

// ============================================================================
// Section 3: degenerate / unreachable configurations -- d > sum, d < diff,
// the explicit d.is_zero() guard, and non-positive lengths.
// ============================================================================

#[test]
fn try_four_bar_linkage_d_less_than_diff_is_rejected() {
    // r3=5, r4=1 -> diff=4; pick r1,r2 so that d=1 < 4 (one circle nested
    // inside the other, cannot intersect).
    let mut w = world();
    let r = try_four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::from_int(1),
        Fix128::from_int(5),
        Fix128::ONE,
        Fix128::from_int(2), // ground_length = 2, crank_length = 1 -> d = 1
        Fix128::ONE,
    );
    assert!(matches!(r, Err(PhysicsError::InvalidConfiguration { .. })));
    assert_eq!(w.bodies.len(), 0, "nothing added on error");
    assert_eq!(w.distance_constraints.len(), 0, "no joints added on error");
}

/// `d.is_zero()` is checked explicitly, separately from `d > sum` / `d <
/// diff`. With `ground_length == crank_length` (`d == 0`) and `r3 ==
/// r4` (`diff == 0` too), neither the sum nor the diff comparison would
/// reject this input (`0 > sum` is false, `0 < 0` is false) -- only the
/// explicit guard does, and it does so *before* `a = (d*d + ..) / (2*d)`
/// would otherwise divide by zero.
#[test]
fn try_four_bar_linkage_zero_distance_with_equal_coupler_rocker_is_rejected_not_a_division_panic() {
    let mut w = world();
    let r = try_four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::ONE,         // crank_length = 1
        Fix128::from_int(2), // coupler_length = 2
        Fix128::from_int(2), // rocker_length = 2 (== coupler_length, diff == 0)
        Fix128::ONE,         // ground_length = 1 (== crank_length, d == 0)
        Fix128::ONE,
    );
    assert!(
        matches!(r, Err(PhysicsError::InvalidConfiguration { .. })),
        "d == 0 must be rejected explicitly, not divide by zero"
    );
    assert_eq!(w.bodies.len(), 0, "nothing added on error");
}

#[test]
fn try_four_bar_linkage_rejects_each_non_positive_length_individually() {
    let base = (
        Fix128::ONE,
        Fix128::from_int(2),
        Fix128::from_int(2),
        Fix128::from_int(3),
    );
    // Each tuple replaces exactly one of (crank, coupler, rocker, ground)
    // with a non-positive value, leaving the other three at the `base`
    // values (which alone would close validly).
    let cases: [(Fix128, Fix128, Fix128, Fix128, &str); 8] = [
        (Fix128::ZERO, base.1, base.2, base.3, "crank_length == 0"),
        (-Fix128::ONE, base.1, base.2, base.3, "crank_length < 0"),
        (base.0, Fix128::ZERO, base.2, base.3, "coupler_length == 0"),
        (base.0, -Fix128::ONE, base.2, base.3, "coupler_length < 0"),
        (base.0, base.1, Fix128::ZERO, base.3, "rocker_length == 0"),
        (base.0, base.1, -Fix128::ONE, base.3, "rocker_length < 0"),
        (base.0, base.1, base.2, Fix128::ZERO, "ground_length == 0"),
        (base.0, base.1, base.2, -Fix128::ONE, "ground_length < 0"),
    ];
    for (crank, coupler, rocker, ground, label) in cases {
        let mut w = world();
        let r = try_four_bar_linkage(
            &mut w,
            Vec3Fix::ZERO,
            crank,
            coupler,
            rocker,
            ground,
            Fix128::ONE,
        );
        assert!(
            matches!(r, Err(PhysicsError::InvalidConfiguration { .. })),
            "case {label} must be rejected"
        );
        assert_eq!(w.bodies.len(), 0, "case {label}: nothing added on error");
    }
}

// ============================================================================
// Section 4: extreme magnitude -- large-but-safe valid configuration, and
// an extreme-magnitude triangle-inequality violation (error path, not a
// panic).
// ============================================================================

#[test]
fn try_four_bar_linkage_extreme_magnitude_valid_configuration_matches_closed_form() {
    let (r2, r3, r4, r1) = (100_000.0_f64, 300_000.0_f64, 250_000.0_f64, 400_000.0_f64);
    let (bx, by) = coupler_pin_f64(r2, r3, r4, r1);
    let mut w = world();
    let l = try_four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2),
        Fix128::from_f64(r3),
        Fix128::from_f64(r4),
        Fix128::from_f64(r1),
        Fix128::ONE,
    )
    .expect("extreme-but-safe magnitude must still close");
    assert_rel(
        w.bodies[l.coupler].position.x,
        bx,
        FIX_TOL,
        "extreme coupler.x",
    );
    assert_rel(
        w.bodies[l.coupler].position.y,
        by,
        FIX_TOL,
        "extreme coupler.y",
    );
}

/// `d` is six orders of magnitude larger than `r3 + r4`: an extreme
/// triangle-inequality violation. Must return `Err`, not panic or
/// silently produce a nonsensical result from an overflowed intermediate.
#[test]
fn try_four_bar_linkage_extreme_magnitude_violation_is_rejected_not_a_panic() {
    let mut w = world();
    let r = try_four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::ONE,                 // crank_length = 1
        Fix128::ONE,                 // coupler_length = 1
        Fix128::ONE,                 // rocker_length = 1 (sum = 2)
        Fix128::from_int(1_000_000), // ground_length = 1e6 -> d ~= 1e6 - 1 >> 2
        Fix128::ONE,
    );
    assert!(matches!(r, Err(PhysicsError::InvalidConfiguration { .. })));
    assert_eq!(w.bodies.len(), 0, "nothing added on error");
}

// ============================================================================
// Section 5: four_bar_linkage -- the panicking wrapper.
// ============================================================================

#[test]
fn four_bar_linkage_matches_try_four_bar_linkage_on_valid_input() {
    let (r2, r3, r4, r1) = (1.0_f64, 3.0_f64, 2.5_f64, 4.0_f64);
    let mut w_try = world();
    let l_try = try_four_bar_linkage(
        &mut w_try,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2),
        Fix128::from_f64(r3),
        Fix128::from_f64(r4),
        Fix128::from_f64(r1),
        Fix128::ONE,
    )
    .expect("must close");

    let mut w_panic = world();
    let l_panic = four_bar_linkage(
        &mut w_panic,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2),
        Fix128::from_f64(r3),
        Fix128::from_f64(r4),
        Fix128::from_f64(r1),
        Fix128::ONE,
    );
    assert_eq!(l_panic.ground, l_try.ground);
    assert_eq!(l_panic.crank, l_try.crank);
    assert_eq!(l_panic.coupler, l_try.coupler);
    assert_eq!(l_panic.rocker, l_try.rocker);
    assert_eq!(
        w_panic.bodies[l_panic.coupler].position, w_try.bodies[l_try.coupler].position,
        "four_bar_linkage and try_four_bar_linkage must agree bit-exactly"
    );
}

/// Per `four_bar_linkage`'s own `# Panics` doc section: "If the four
/// lengths cannot close at crank angle 0 ... [it] panics." Verified via
/// `catch_unwind` rather than `#[should_panic]` so the test can also
/// assert that nothing was left half-added to `world` (a `#[should_panic]`
/// test cannot inspect state after the panic unwinds past it).
#[test]
fn four_bar_linkage_panics_on_a_configuration_that_cannot_close() {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // silence the panic backtrace for this expected panic
    let result = std::panic::catch_unwind(|| {
        let mut w = world();
        // |A O4| = 5 > r3 + r4 = 2: cannot close.
        let _ = four_bar_linkage(
            &mut w,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ONE,
            Fix128::ONE,
            Fix128::from_int(6),
            Fix128::ONE,
        );
    });
    std::panic::set_hook(prev_hook);
    assert!(
        result.is_err(),
        "four_bar_linkage must panic on an unclosable configuration"
    );
}

// ============================================================================
// Section 6: LoopClosureConstraint::centre_to_centre -- exact field
// values, in isolation from `residual`/`apply`.
// ============================================================================

#[test]
fn centre_to_centre_produces_coincident_zero_compliance_closure_exactly() {
    let closure = LoopClosureConstraint::centre_to_centre(7, 11);
    assert_eq!(closure.body_a, 7);
    assert_eq!(closure.body_b, 11);
    assert_eq!(closure.local_anchor_a, Vec3Fix::ZERO);
    assert_eq!(closure.local_anchor_b, Vec3Fix::ZERO);
    assert_eq!(closure.compliance, Fix128::ZERO);
}

/// Hand-derived inverse-mass-weighted convergence point (same derivation
/// as `examples/kinematic_loop_four_bar.rs`'s scenario 4, duplicated here
/// for the reason given in `coupler_pin_f64`'s doc comment): `inv_a =
/// 1/2`, `inv_b = 1/3`, so `w_a = inv_a/(inv_a+inv_b) = 3/5` and `w_b =
/// 2/5`; starting at `x=0` and `x=10`, both bodies land on `x = 6.0`.
#[test]
fn centre_to_centre_apply_converges_to_inverse_mass_weighted_point() {
    use alice_physics::solver::RigidBody;
    let mut w = world();
    let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
    let b = w.add_body(RigidBody::new(
        Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO),
        Fix128::from_int(3),
    ));
    let closure = LoopClosureConstraint::centre_to_centre(a, b);
    closure.apply(&mut w);
    assert_rel(
        w.bodies[a].position.x,
        6.0,
        FIX_TOL,
        "heavy_a converges to x=6.0",
    );
    assert_rel(
        w.bodies[b].position.x,
        6.0,
        FIX_TOL,
        "heavy_b converges to x=6.0",
    );
}
