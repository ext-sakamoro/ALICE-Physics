//! Production entry point for `alice_physics::contact_viz`:
//! `ContactArrow`, `FrictionCone`, `generate_contact_arrows`,
//! `generate_friction_arrows`, and `generate_friction_cones`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all five as `unwired` --
//! `src/contact_viz.rs`'s own `#[cfg(all(test, feature = "std"))]` module
//! calls every one of them, but tests do not count as production callers
//! for the wiring guard, and nothing in `src/` / `examples/` / `benches/`
//! called any of the five before this file existed. This example is that
//! caller.
//!
//! All five items are pure, deterministic geometry transforms over
//! `(Vec3Fix, Vec3Fix, Fix128)` contact tuples -- no RNG, no solver state
//! -- so every expected value below is an independently hand-derived
//! closed form (vector algebra worked out by hand against the module's
//! own documented formulas: `generate_contact_arrows` normalizes the
//! input normal and passes the force magnitude through unchanged;
//! `generate_friction_arrows` scales `normal_force` by `mu` and builds an
//! orthonormal tangent pair via the "least-aligned axis" Gram-Schmidt
//! trick; `generate_friction_cones` sets `half_angle = atan(mu)` and
//! `height = normal_force` verbatim), never a value re-derived by calling
//! the function under test on itself.
//!
//! `tests/analytic_contact_viz_wiring.rs` holds additional closed-form
//! oracles and degenerate-input cases that this file does not cover: a
//! second normalize vector (5-12-13 instead of 3-4-5), the empty-slice
//! input for `generate_friction_arrows` / `generate_friction_cones`,
//! negative `normal_force`, and the `mu == 0` boundary.
//!
//! ```bash
//! cargo run --example contact_visualization --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::contact_viz::{
    generate_contact_arrows, generate_friction_arrows, generate_friction_cones, ContactArrow,
    FrictionCone,
};
use alice_physics::math::{Fix128, Vec3Fix};

/// Absolute-error assertion against an independently hand-derived f64
/// closed form.
fn assert_fix_abs(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs();
    assert!(
        err <= tol,
        "[contact_viz] MISMATCH {what}: got {g}, want {want} (abs err {err:.3e} > {tol:.1e})"
    );
    println!("[contact_viz] ok {what}: got {g:.9}, want {want:.9} (err {err:.2e})");
}

/// Component-wise absolute-error assertion for a `Vec3Fix` against an
/// independently hand-derived `(x, y, z)` f64 closed form.
fn assert_vec3_abs(got: Vec3Fix, want: (f64, f64, f64), tol: f64, what: &str) {
    let g = (got.x.to_f64(), got.y.to_f64(), got.z.to_f64());
    let err = (
        (g.0 - want.0).abs(),
        (g.1 - want.1).abs(),
        (g.2 - want.2).abs(),
    );
    assert!(
        err.0 <= tol && err.1 <= tol && err.2 <= tol,
        "[contact_viz] MISMATCH {what}: got {g:?}, want {want:?} (err {err:?} > {tol:.1e})"
    );
    println!("[contact_viz] ok {what}: got {g:?}, want {want:?}");
}

fn main() {
    let tol = 1e-9;

    // ------------------------------------------------------------------
    // 1. generate_contact_arrows -- a non-trivial normalize (3-4-5
    //    triangle: (0, 3, 4) has length 5, so the normalized direction is
    //    the exact rational (0, 0.6, 0.8)) and an already-unit normal
    //    (UNIT_X, where normalize is the identity). Position and force
    //    magnitude pass through unchanged; `is_friction` is always false.
    // ------------------------------------------------------------------
    let contacts_a = [
        (
            Vec3Fix::from_int(1, 2, 3),
            Vec3Fix::from_int(0, 3, 4),
            Fix128::from_int(10),
        ),
        (
            Vec3Fix::from_int(5, 0, 0),
            Vec3Fix::UNIT_X,
            Fix128::from_int(7),
        ),
    ];
    let arrows: Vec<ContactArrow> = generate_contact_arrows(&contacts_a);
    assert_eq!(arrows.len(), 2, "[contact_viz] MISMATCH arrow count");
    assert_vec3_abs(
        arrows[0].position,
        (1.0, 2.0, 3.0),
        tol,
        "contact_arrows[0].position passes through",
    );
    assert_vec3_abs(
        arrows[0].normal,
        (0.0, 0.6, 0.8),
        tol,
        "contact_arrows[0].normal = normalize(0,3,4) = (0, 3/5, 4/5)",
    );
    assert_fix_abs(
        arrows[0].force_magnitude,
        10.0,
        tol,
        "contact_arrows[0].force_magnitude passes through",
    );
    assert!(
        !arrows[0].is_friction && !arrows[1].is_friction,
        "[contact_viz] MISMATCH: generate_contact_arrows must never set is_friction"
    );
    assert_vec3_abs(
        arrows[1].normal,
        (1.0, 0.0, 0.0),
        tol,
        "contact_arrows[1].normal = normalize(UNIT_X) = UNIT_X (already unit)",
    );
    assert_fix_abs(
        arrows[1].force_magnitude,
        7.0,
        tol,
        "contact_arrows[1].force_magnitude passes through",
    );

    // ------------------------------------------------------------------
    // 2. generate_contact_arrows -- zero-magnitude edge cases. A zero
    //    input normal cannot be normalized (no direction is more correct
    //    than any other), and `Vec3Fix::normalize`'s own documented
    //    contract (`src/math.rs`) is to return `Vec3Fix::ZERO` rather
    //    than NaN/panic; the force magnitude still passes through
    //    unchanged. A zero force magnitude passes through as exactly
    //    zero regardless of a well-formed unit normal.
    // ------------------------------------------------------------------
    let zero_normal_contact = [(
        Vec3Fix::from_int(9, 9, 9),
        Vec3Fix::ZERO,
        Fix128::from_int(5),
    )];
    let zero_normal_arrows: Vec<ContactArrow> = generate_contact_arrows(&zero_normal_contact);
    assert_vec3_abs(
        zero_normal_arrows[0].normal,
        (0.0, 0.0, 0.0),
        tol,
        "contact_arrows: normalize(ZERO) = ZERO (documented zero-length contract)",
    );
    assert_fix_abs(
        zero_normal_arrows[0].force_magnitude,
        5.0,
        tol,
        "contact_arrows: force_magnitude passes through even when normal is zero",
    );

    let zero_force_contact = [(Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::ZERO)];
    let zero_force_arrows: Vec<ContactArrow> = generate_contact_arrows(&zero_force_contact);
    assert_fix_abs(
        zero_force_arrows[0].force_magnitude,
        0.0,
        tol,
        "contact_arrows: zero force_magnitude passes through as exactly zero",
    );

    // ------------------------------------------------------------------
    // 3. generate_friction_arrows -- exercises all three branches of the
    //    private `tangent_basis` helper's "least-aligned axis" selection
    //    (`abs_x < abs_y && abs_x < abs_z` / `abs_y < abs_z` / else), each
    //    hand-derived from the cross-product definition
    //    `cross(a,b) = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)`,
    //    never by calling `Vec3Fix::cross` itself here:
    //
    //    * n = normalize(0,3,4) = (0, 0.6, 0.8): abs_x=0 < abs_y=0.6 and
    //      abs_x=0 < abs_z=0.8 -> helper = UNIT_X.
    //      t1 = n x UNIT_X = (0.6*0-0.8*0, 0.8*1-0*0, 0*0-0.6*1) = (0, 0.8, -0.6)
    //      (already unit length: 0.8^2+0.6^2=1, so normalize is the identity).
    //      t2 = n x t1 = (0.6*-0.6-0.8*0.8, 0.8*0-0*-0.6, 0*0.8-0.6*0) = (-1, 0, 0).
    //    * n = UNIT_X = (1,0,0): abs_x=1, abs_y=0, abs_z=0 ->
    //      abs_x < abs_y is false, and abs_y < abs_z (0<0) is false -> helper = UNIT_Z.
    //      t1 = n x UNIT_Z = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0, -1, 0).
    //      t2 = n x t1 = (0*0-0*-1, 0*0-1*0, 1*-1-0*0) = (0, 0, -1).
    //    * n = UNIT_Z = (0,0,1): abs_x=0, abs_y=0, abs_z=1 ->
    //      abs_x < abs_y (0<0) is false, and abs_y < abs_z (0<1) is true -> helper = UNIT_Y.
    //      t1 = n x UNIT_Y = (0*0-1*1, 1*0-0*0, 0*1-0*0) = (-1, 0, 0).
    //      t2 = n x t1 = (0*0-1*0, 1*-1-0*0, 0*0-0*-1) = (0, -1, 0).
    //
    //    `friction_force = normal_force * mu` is independent linear
    //    scaling, checked against plain multiplication.
    // ------------------------------------------------------------------
    let mu = Fix128::from_ratio(3, 10); // 0.3
    let contacts_b = [
        (
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(0, 3, 4),
            Fix128::from_int(10),
        ),
        (
            Vec3Fix::from_int(2, 0, 0),
            Vec3Fix::UNIT_X,
            Fix128::from_int(4),
        ),
        (
            Vec3Fix::from_int(3, 0, 0),
            Vec3Fix::UNIT_Z,
            Fix128::from_int(2),
        ),
    ];
    let friction_arrows: Vec<ContactArrow> = generate_friction_arrows(&contacts_b, mu);
    assert_eq!(
        friction_arrows.len(),
        6,
        "[contact_viz] MISMATCH: 2 tangent arrows per contact x 3 contacts"
    );
    for arrow in &friction_arrows {
        assert!(
            arrow.is_friction,
            "[contact_viz] MISMATCH: generate_friction_arrows must always set is_friction"
        );
    }

    assert_vec3_abs(
        friction_arrows[0].normal,
        (0.0, 0.8, -0.6),
        tol,
        "friction_arrows[0] t1 for n=normalize(0,3,4), helper=UNIT_X branch",
    );
    assert_vec3_abs(
        friction_arrows[1].normal,
        (-1.0, 0.0, 0.0),
        tol,
        "friction_arrows[1] t2 for n=normalize(0,3,4), helper=UNIT_X branch",
    );
    assert_fix_abs(
        friction_arrows[0].force_magnitude,
        3.0,
        tol,
        "friction_arrows[0,1].force_magnitude = 10 * 0.3",
    );

    assert_vec3_abs(
        friction_arrows[2].normal,
        (0.0, -1.0, 0.0),
        tol,
        "friction_arrows[2] t1 for n=UNIT_X, helper=UNIT_Z branch",
    );
    assert_vec3_abs(
        friction_arrows[3].normal,
        (0.0, 0.0, -1.0),
        tol,
        "friction_arrows[3] t2 for n=UNIT_X, helper=UNIT_Z branch",
    );
    assert_fix_abs(
        friction_arrows[2].force_magnitude,
        1.2,
        tol,
        "friction_arrows[2,3].force_magnitude = 4 * 0.3",
    );

    assert_vec3_abs(
        friction_arrows[4].normal,
        (-1.0, 0.0, 0.0),
        tol,
        "friction_arrows[4] t1 for n=UNIT_Z, helper=UNIT_Y branch",
    );
    assert_vec3_abs(
        friction_arrows[5].normal,
        (0.0, -1.0, 0.0),
        tol,
        "friction_arrows[5] t2 for n=UNIT_Z, helper=UNIT_Y branch",
    );
    assert_fix_abs(
        friction_arrows[4].force_magnitude,
        0.6,
        tol,
        "friction_arrows[4,5].force_magnitude = 2 * 0.3",
    );

    // ------------------------------------------------------------------
    // 4. generate_friction_cones -- half_angle = atan(mu) (independent
    //    oracle: platform f64::atan, a different algorithm from
    //    `Fix128::atan`'s CORDIC implementation -- see
    //    `src/math.rs::atan_matches_f64_within_1e12` for the same
    //    cross-check convention), normal = normalize(input normal),
    //    height = normal_force passed through verbatim (not scaled by
    //    mu, unlike the friction arrows above).
    // ------------------------------------------------------------------
    #[allow(clippy::disallowed_methods)]
    // independent f64 libm oracle, never fed back into Fix128 state
    let half_angle_ref = 0.3_f64.atan();
    let contacts_c = [
        (
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(0, 3, 4),
            Fix128::from_int(10),
        ),
        (
            Vec3Fix::from_int(2, 0, 0),
            Vec3Fix::UNIT_Y,
            Fix128::from_int(5),
        ),
    ];
    let cones: Vec<FrictionCone> = generate_friction_cones(&contacts_c, mu);
    assert_eq!(cones.len(), 2, "[contact_viz] MISMATCH cone count");
    assert_vec3_abs(
        cones[0].normal,
        (0.0, 0.6, 0.8),
        tol,
        "cones[0].normal = normalize(0,3,4)",
    );
    assert_fix_abs(
        cones[0].half_angle,
        half_angle_ref,
        1e-9,
        "cones[0].half_angle = atan(0.3)",
    );
    assert_fix_abs(
        cones[0].height,
        10.0,
        tol,
        "cones[0].height = normal_force passed through verbatim",
    );
    assert_vec3_abs(
        cones[1].normal,
        (0.0, 1.0, 0.0),
        tol,
        "cones[1].normal = normalize(UNIT_Y) = UNIT_Y",
    );
    assert_fix_abs(
        cones[1].half_angle,
        half_angle_ref,
        1e-9,
        "cones[1].half_angle = atan(0.3) (independent of normal_force)",
    );
    assert_fix_abs(
        cones[1].height,
        5.0,
        tol,
        "cones[1].height = normal_force passed through verbatim",
    );

    println!("[contact_viz] all checks passed");
}
