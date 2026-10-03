//! Classical four-bar linkage production entry point for
//! `alice_physics::kinematic_loop`: `FourBarLinkage`,
//! `LoopClosureConstraint::centre_to_centre`, `four_bar_linkage`, and
//! `try_four_bar_linkage`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all four as `unwired`.
//! `tests/engineering_oracles_misc.rs` already drives `four_bar_linkage` and
//! `LoopClosureConstraint::centre_to_centre` through a motion-tracking test
//! (`four_bar_link_lengths_hold_at_rest_and_in_motion`-style checks), but
//! tests do not count as production callers for the wiring guard (the
//! `tests/` directory is excluded from its corpus), and nothing in `src/` /
//! `examples/` / `benches/` called any of the four before this file
//! existed. This example is that caller.
//!
//! `tests/analytic_kinematic_loop_wiring.rs` holds the closed-form oracles:
//! Grashof-condition boundary cases (fully-extended / folded configurations
//! at the exact `|r3-r4| <= d <= r3+r4` bounds), the `d.is_zero()`
//! division-by-zero guard distinct from the sum/diff bounds, degenerate
//! non-positive lengths for all four link parameters individually, a
//! `four_bar_linkage` panic test matching its documented `# Panics`
//! contract, and extreme-magnitude cases on both the success and error
//! paths.
//!
//! Norton, *Design of Machinery*, 4th ed., ch. 3: a Grashof linkage
//! satisfies `s + l <= p + q` where `s`/`l` are the shortest/longest of the
//! four link lengths and `p`/`q` are the other two; when the shortest link
//! is a side link adjacent to the ground link (as here: the crank), the
//! mechanism is a *crank-rocker* -- the crank makes a full revolution while
//! the rocker oscillates. `try_four_bar_linkage` only ever assembles the
//! mechanism at crank angle zero (it does not simulate the full rotation),
//! so what this example and its oracle test verify is the circle-circle
//! intersection that places the coupler pin at that one configuration --
//! the Grashof classification is reported as a diagnostic, not something
//! the function itself computes or depends on.
//!
//! ```bash
//! cargo run --example kinematic_loop_four_bar --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::det_math::acos64;
use alice_physics::kinematic_loop::{
    four_bar_linkage, try_four_bar_linkage, LoopClosureConstraint,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

/// Relative-error check against an independently hand-derived f64 closed
/// form (never computed by calling the function under test). Mirrors
/// `examples/laminate_abd_matrix.rs`'s `assert_rel`.
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = if want == 0.0 {
        g.abs()
    } else {
        ((g - want) / want).abs()
    };
    assert!(
        err <= tol,
        "[kinematic_loop] MISMATCH {what}: got {g}, want {want} (err {err:.3e} > {tol:.1e})"
    );
    println!("[kinematic_loop] ok {what}: got {g:.6}, want {want:.6} (err {err:.2e})");
}

/// Circle-circle intersection closed form for the coupler pin `B`, written
/// out independently in plain f64 (the same family of formula
/// `try_four_bar_linkage` evaluates in `Fix128`, but re-derived here via
/// the law of cosines on triangle `A-O4-B` rather than by calling the
/// function): with `d = |r1 - r2|` the ground-to-crank-pin distance,
/// `cos(angle O4-A-B) = (d^2 + r3^2 - r4^2) / (2*d*r3)` by the law of
/// cosines, so the projection of `A->B` onto the `A->O4` axis is
/// `a = r3*cos(angle) = (d^2 + r3^2 - r4^2) / (2*d)` and the perpendicular
/// offset is `h = r3*sin(angle) = sqrt(r3^2 - a^2)`.
struct FourBarOracle {
    d: f64,
    a: f64,
    h: f64,
    dir: f64,
    pin_a: (f64, f64),
    pin_o4: (f64, f64),
    pin_b: (f64, f64),
    /// Transmission angle `mu` at vertex B (angle A-B-O4), law of cosines on
    /// triangle A-B-O4 with sides `r3` (AB), `r4` (BO4), `d` (AO4) --
    /// an independent second derivation (different triangle, same three
    /// hand-chosen link lengths) used only as a diagnostic cross-check,
    /// not as the primary position oracle above.
    cos_transmission_angle: f64,
}

fn four_bar_oracle_f64(r2: f64, r3: f64, r4: f64, r1: f64) -> FourBarOracle {
    let d = (r1 - r2).abs();
    let a = (d * d + r3 * r3 - r4 * r4) / (2.0 * d);
    let h = (r3 * r3 - a * a).max(0.0).sqrt();
    let dir = if r1 >= r2 { 1.0 } else { -1.0 };
    let pin_a = (r2, 0.0);
    let pin_o4 = (r1, 0.0);
    let pin_b = (r2 + dir * a, h);
    let cos_transmission_angle = (r3 * r3 + r4 * r4 - d * d) / (2.0 * r3 * r4);
    FourBarOracle {
        d,
        a,
        h,
        dir,
        pin_a,
        pin_o4,
        pin_b,
        cos_transmission_angle,
    }
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig::default())
}

fn main() {
    let tol = 1e-9;

    // ------------------------------------------------------------------
    // 1. try_four_bar_linkage -- a classic Grashof crank-rocker. The
    //    shortest link (r2 = 1, the crank) is a side link adjacent to the
    //    ground link (r1 = 4, the longest), and s + l = 5 <= p + q = 5.5,
    //    so by Norton's criterion this is a crank-rocker: the crank can
    //    fully rotate while the rocker oscillates (the function only
    //    assembles the mechanism at crank angle zero; it does not drive
    //    the rotation).
    // ------------------------------------------------------------------
    let (r2, r3, r4, r1) = (1.0_f64, 3.0_f64, 2.5_f64, 4.0_f64);
    let (s, l, p, q) = (r2, r1, r3, r4); // s=shortest, l=longest, p/q=others
    assert!(
        s + l <= p + q,
        "[kinematic_loop] scenario 1 must satisfy the Grashof condition"
    );
    println!(
        "[kinematic_loop] scenario 1: Grashof crank-rocker (r2={r2}, r3={r3}, r4={r4}, r1={r1}), \
         s+l={} <= p+q={}",
        s + l,
        p + q
    );
    let oracle1 = four_bar_oracle_f64(r2, r3, r4, r1);
    println!(
        "[kinematic_loop] scenario 1 law-of-cosines intermediates: d={:.6}, a={:.6}, h={:.6}",
        oracle1.d, oracle1.a, oracle1.h
    );

    let mut w1 = world();
    let linkage1 = try_four_bar_linkage(
        &mut w1,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2),
        Fix128::from_f64(r3),
        Fix128::from_f64(r4),
        Fix128::from_f64(r1),
        Fix128::ONE,
    )
    .expect("[kinematic_loop] scenario 1 must close at crank angle 0");

    assert_rel(
        w1.bodies[linkage1.crank].position.x,
        oracle1.pin_a.0,
        tol,
        "scenario1 crank pin A.x",
    );
    assert_rel(
        w1.bodies[linkage1.crank].position.y,
        oracle1.pin_a.1,
        tol,
        "scenario1 crank pin A.y",
    );
    assert_rel(
        w1.bodies[linkage1.rocker].position.x,
        oracle1.pin_o4.0,
        tol,
        "scenario1 ground pin O4.x",
    );
    assert_rel(
        w1.bodies[linkage1.coupler].position.x,
        oracle1.pin_b.0,
        tol,
        "scenario1 coupler pin B.x",
    );
    assert_rel(
        w1.bodies[linkage1.coupler].position.y,
        oracle1.pin_b.1,
        tol,
        "scenario1 coupler pin B.y",
    );

    // Diagnostic angles, independently hand-derived via the law of cosines
    // (not read back from the function's own output): the coupler angle
    // (angle O4-A-B, from cos(angle) = a/r3) and the transmission angle at
    // B (angle A-B-O4, from the A-B-O4 triangle's own three sides).
    let coupler_angle_rad = acos64(oracle1.a / r3);
    let transmission_angle_rad = acos64(oracle1.cos_transmission_angle);
    println!(
        "[kinematic_loop] scenario 1 coupler angle (O4-A-B) = {:.6} rad ({:.3} deg)",
        coupler_angle_rad,
        coupler_angle_rad.to_degrees()
    );
    println!(
        "[kinematic_loop] scenario 1 transmission angle (A-B-O4) = {:.6} rad ({:.3} deg)",
        transmission_angle_rad,
        transmission_angle_rad.to_degrees()
    );
    // Cross-check: the transmission angle derived from the A-B-O4 triangle's
    // side lengths alone must match the angle recovered from the actual
    // solved pin positions via the dot product of BA and BO4 (a second,
    // independent path to the same angle).
    let ba = (
        oracle1.pin_a.0 - oracle1.pin_b.0,
        oracle1.pin_a.1 - oracle1.pin_b.1,
    );
    let bo4 = (
        oracle1.pin_o4.0 - oracle1.pin_b.0,
        oracle1.pin_o4.1 - oracle1.pin_b.1,
    );
    let dot = ba.0 * bo4.0 + ba.1 * bo4.1;
    let mag = (ba.0 * ba.0 + ba.1 * ba.1).sqrt() * (bo4.0 * bo4.0 + bo4.1 * bo4.1).sqrt();
    let cos_check = dot / mag;
    let err = (cos_check - oracle1.cos_transmission_angle).abs();
    assert!(
        err <= tol,
        "[kinematic_loop] MISMATCH transmission angle cross-check: {cos_check} vs {} (err {err:.3e})",
        oracle1.cos_transmission_angle
    );
    println!("[kinematic_loop] ok transmission angle cross-check (coordinate dot product vs law of cosines), err {err:.2e}");

    // The three link joints carry exactly the three input lengths.
    assert_rel(
        w1.distance_constraints[linkage1.joints[0]].target_distance,
        r2,
        tol,
        "scenario1 joint[0] (ground-crank) target_distance",
    );
    assert_rel(
        w1.distance_constraints[linkage1.joints[1]].target_distance,
        r3,
        tol,
        "scenario1 joint[1] (crank-coupler) target_distance",
    );
    assert_rel(
        w1.distance_constraints[linkage1.joints[2]].target_distance,
        r4,
        tol,
        "scenario1 joint[2] (coupler-rocker) target_distance",
    );
    // The loop closure re-closes the rocker pin against the ground body's
    // (r1, 0, 0) anchor; at construction the residual is exactly zero.
    assert_eq!(
        linkage1.closure.residual(&w1),
        Vec3Fix::ZERO,
        "scenario1 closure residual must be exactly zero at construction"
    );
    println!("[kinematic_loop] ok scenario 1: all 3 joints + closure verified");

    // ------------------------------------------------------------------
    // 2. try_four_bar_linkage again, with ground_length < crank_length so
    //    the ground pin O4 sits to the *left* of the crank pin A -- the
    //    `dir = -1` branch in `try_four_bar_linkage` (unexercised by
    //    scenario 1, where O4 is to the right). Same coupler/rocker
    //    lengths as scenario 1, so `d` happens to come out identical (3.0),
    //    but the sign of the projection flips.
    // ------------------------------------------------------------------
    let (r2b, r3b, r4b, r1b) = (4.0_f64, 3.0_f64, 2.5_f64, 1.0_f64);
    let oracle2 = four_bar_oracle_f64(r2b, r3b, r4b, r1b);
    assert_eq!(
        oracle2.dir, -1.0,
        "scenario 2 must exercise the dir=-1 branch"
    );

    let mut w2 = world();
    let linkage2 = try_four_bar_linkage(
        &mut w2,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2b),
        Fix128::from_f64(r3b),
        Fix128::from_f64(r4b),
        Fix128::from_f64(r1b),
        Fix128::ONE,
    )
    .expect("[kinematic_loop] scenario 2 must close at crank angle 0");
    assert_rel(
        w2.bodies[linkage2.coupler].position.x,
        oracle2.pin_b.0,
        tol,
        "scenario2 (dir=-1) coupler pin B.x",
    );
    assert_rel(
        w2.bodies[linkage2.coupler].position.y,
        oracle2.pin_b.1,
        tol,
        "scenario2 (dir=-1) coupler pin B.y",
    );
    println!("[kinematic_loop] ok scenario 2 (dir=-1 branch, ground_length < crank_length)");

    // ------------------------------------------------------------------
    // 3. four_bar_linkage -- the panicking convenience wrapper. On a valid
    //    configuration it must produce the exact same body indices and
    //    link lengths as calling try_four_bar_linkage directly on an
    //    equally fresh world (both start empty, so both assign indices
    //    0, 1, 2, 3 in ground/crank/coupler/rocker order).
    // ------------------------------------------------------------------
    let mut w3 = world();
    let linkage3 = four_bar_linkage(
        &mut w3,
        Vec3Fix::ZERO,
        Fix128::from_f64(r2),
        Fix128::from_f64(r3),
        Fix128::from_f64(r4),
        Fix128::from_f64(r1),
        Fix128::ONE,
    );
    assert_eq!(
        linkage3.ground, linkage1.ground,
        "four_bar_linkage ground index"
    );
    assert_eq!(
        linkage3.crank, linkage1.crank,
        "four_bar_linkage crank index"
    );
    assert_eq!(
        linkage3.coupler, linkage1.coupler,
        "four_bar_linkage coupler index"
    );
    assert_eq!(
        linkage3.rocker, linkage1.rocker,
        "four_bar_linkage rocker index"
    );
    assert_eq!(
        w3.bodies[linkage3.coupler].position, w1.bodies[linkage1.coupler].position,
        "four_bar_linkage and try_four_bar_linkage must agree bit-exactly on the same input"
    );
    println!(
        "[kinematic_loop] ok four_bar_linkage matches try_four_bar_linkage on the same valid input"
    );

    // ------------------------------------------------------------------
    // 4. LoopClosureConstraint::centre_to_centre -- a standalone ball-
    //    joint-style closure between two bodies that are *not* part of
    //    the four-bar linkage, demonstrating the production call site the
    //    wiring guard requires. Hand-derived closed form: with
    //    compliance = 0, inv_mass_a = 1/2 (mass 2), inv_mass_b = 1/3
    //    (mass 3), the Baumgarte correction moves both bodies to the
    //    inverse-mass-weighted point `w_a = inv_a/(inv_a+inv_b) = 3/5`,
    //    `w_b = inv_b/(inv_a+inv_b) = 2/5` of the way between them:
    //    starting at x=0 and x=10, both land on x = 0 + 10*w_a = 6.0 and
    //    x = 10 - 10*w_b = 6.0 respectively.
    // ------------------------------------------------------------------
    let mut w4 = world();
    let heavy_a = w4.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
    let heavy_b = w4.add_body(RigidBody::new(
        Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO),
        Fix128::from_int(3),
    ));
    let closure = LoopClosureConstraint::centre_to_centre(heavy_a, heavy_b);
    assert_eq!(closure.body_a, heavy_a);
    assert_eq!(closure.body_b, heavy_b);
    assert_eq!(closure.local_anchor_a, Vec3Fix::ZERO);
    assert_eq!(closure.local_anchor_b, Vec3Fix::ZERO);
    assert_eq!(closure.compliance, Fix128::ZERO);
    let residual_before = closure.residual(&w4);
    assert_eq!(
        residual_before,
        Vec3Fix::new(-Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO),
        "centre_to_centre residual before apply() = pos_a - pos_b exactly"
    );
    closure.apply(&mut w4);
    assert_rel(
        w4.bodies[heavy_a].position.x,
        6.0,
        tol,
        "centre_to_centre heavy_a.x after apply",
    );
    assert_rel(
        w4.bodies[heavy_b].position.x,
        6.0,
        tol,
        "centre_to_centre heavy_b.x after apply",
    );
    println!("[kinematic_loop] ok centre_to_centre: inverse-mass-weighted convergence at x=6.0");

    println!(
        "[kinematic_loop] all 4 production entry points (FourBarLinkage via try_four_bar_linkage, \
         four_bar_linkage, centre_to_centre) verified against hand-derived closed forms"
    );
}
