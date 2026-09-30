//! Analytic oracles for the SDF character controller's up axis and its
//! push clamp.
//!
//! Two properties are pinned here, both with closed-form expectations
//! that do not call the implementation to build the expected value.
//!
//! # 1. Ground detection on a sphere world
//!
//! `is_grounded` asks whether the character stands on a surface whose
//! normal points along the character's up axis. On a planet of radius
//! `R` the up axis is **radial** (`p / |p|`), so the answer has a closed
//! form: a character whose centre sits at `|p| = R + radius` has its
//! probe point at `|p| = R - ground_probe_epsilon`, where the planet SDF
//! `|p| - R` evaluates to exactly `-ground_probe_epsilon` and the normal
//! is the radial direction itself. That is grounded for every point of
//! the sphere.
//!
//! With a fixed `+Y` up axis the same closed form only holds at the
//! north pole: at the equator the probe point lands at
//! `sqrt((R + radius)^2 + (radius + eps)^2) - R`, which is larger than
//! `eps` for any `radius > 0`, so a `+Y`-only controller reports "not
//! grounded" while standing on the ground. `equator_*` pins both sides
//! of that.
//!
//! # 2. Push clamp on a field that is not an exact distance field
//!
//! The penetration-resolution step pushes the character out by
//! `radius - d`, which assumes `d` is the true distance. For a field
//! with `|grad f| = L > 1` (gyroid walls and other implicit surfaces are
//! in this class) `|d|` **over**states the penetration by up to `L`, so
//! an unclamped push can jump clean over the free gap and land inside
//! the next sheet.
//!
//! The field used below is periodic solid sheets, and every quantity is
//! a closed form:
//!
//! - sheets of half-thickness `t` centred on `y = k * s`
//! - `u(y) = y - s * round(y / s)`   (signed offset to the nearest sheet)
//! - `f(y) = L * (|u| - t)`, unit normal `sign(u) * y_hat`
//! - the controller's exit condition `f >= radius` is therefore
//!   **`|u| >= t + radius / L`** — an analytic free band that exists iff
//!   `s / 2 > t + radius / L`
//!
//! The oracle is that band **plus which pocket the controller ends in**.
//! The clamped controller must stop in the free pocket adjacent to its
//! start; the unclamped one overshoots it on the first iteration and
//! lands inside the next sheet.
//!
//! ⚠️ The unclamped run does eventually report `converged == true` (the
//! hop lengths vary with the penetration depth, so one of them lands in
//! a gap by chance — measured: iteration 5, at `y = 5.156713`). It is
//! 2.5 periods from where it started, having crossed three solid sheets.
//! **`converged` does not mean the character stayed on its side of the
//! geometry**, which is why the oracle below tests the pocket rather
//! than the flag.

use alice_physics::sdf_character::SdfCharacter;
use alice_physics::sdf_collider::ClosureSdf;

// ─────────────────────────── sphere world ───────────────────────────

/// Planet radius (m). Matches the scale the sphere-world consumers use,
/// so the `radius`-vs-`R` ratio in the equator case is realistic.
const PLANET_R: f32 = 300.0;
/// Capsule radius (m).
const CAP_R: f32 = 0.35;
/// Ground probe distance (m) — `SdfCharacter::default`'s value.
const PROBE_EPS: f32 = 5.0e-2;

/// `f(p) = |p| - R`, the exact distance field of a solid planet.
fn planet() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - PLANET_R,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(1.0e-6);
            (x / len, y / len, z / len)
        },
    )
}

/// Character centre standing on the planet along `dir` (unit).
fn standing_on(dir: [f32; 3]) -> [f32; 3] {
    let r = PLANET_R + CAP_R;
    [dir[0] * r, dir[1] * r, dir[2] * r]
}

#[test]
fn north_pole_is_grounded_with_the_default_y_up() {
    // Closed form: probe sits at |p| = R + CAP_R - (CAP_R + PROBE_EPS)
    // = R - PROBE_EPS, so the planet SDF is exactly -PROBE_EPS (<= eps)
    // and the normal there is +Y, i.e. the up axis. Grounded.
    let ch = SdfCharacter::new(standing_on([0.0, 1.0, 0.0]), CAP_R, 1.8);
    assert_eq!(ch.up, [0.0, 1.0, 0.0], "default up axis must stay +Y");
    assert!(ch.is_grounded(&planet()));
}

#[test]
fn equator_is_not_grounded_with_a_fixed_y_up_axis() {
    // Why the `up` field has to exist. Probe point is
    // (R + CAP_R, -(CAP_R + PROBE_EPS), 0), whose distance to the
    // surface is sqrt((R + CAP_R)^2 + (CAP_R + PROBE_EPS)^2) - R.
    let ch = SdfCharacter::new(standing_on([1.0, 0.0, 0.0]), CAP_R, 1.8);
    let expected = ((PLANET_R + CAP_R) * (PLANET_R + CAP_R)
        + (CAP_R + PROBE_EPS) * (CAP_R + PROBE_EPS))
        .sqrt()
        - PLANET_R;
    assert!(
        expected > PROBE_EPS,
        "closed form must exceed the probe epsilon: {expected} vs {PROBE_EPS}"
    );
    assert!(
        !ch.is_grounded(&planet()),
        "a +Y-only probe cannot find the ground at the equator"
    );
}

#[test]
fn equator_is_grounded_once_the_up_axis_is_radial() {
    let mut ch = SdfCharacter::new(standing_on([1.0, 0.0, 0.0]), CAP_R, 1.8);
    ch.up = [1.0, 0.0, 0.0];
    assert!(ch.is_grounded(&planet()));
}

#[test]
fn every_direction_on_the_sphere_is_grounded_with_a_radial_up_axis() {
    // The closed form is direction-independent, so sweep the sphere: any
    // failure means the probe is still using a privileged axis.
    //
    // Directions come from normalized integer lattice points rather than
    // `sin`/`cos` — this crate's `clippy.toml` disallows the platform
    // trig functions because they are not cross-platform bit-exact, and
    // the lattice needs no transcendentals at all.
    let field = planet();
    let mut checked = 0_usize;
    for ix in -2_i32..=2 {
        for iy in -2_i32..=2 {
            for iz in -2_i32..=2 {
                if ix == 0 && iy == 0 && iz == 0 {
                    continue;
                }
                let raw = [ix as f32, iy as f32, iz as f32];
                let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
                let dir = [raw[0] / len, raw[1] / len, raw[2] / len];
                let mut ch = SdfCharacter::new(standing_on(dir), CAP_R, 1.8);
                ch.up = dir;
                assert!(
                    ch.is_grounded(&field),
                    "not grounded at lattice {ix},{iy},{iz} -> dir {dir:?}"
                );
                checked += 1;
            }
        }
    }
    // 5^3 lattice points minus the origin.
    assert_eq!(checked, 124);
}

#[test]
fn a_short_non_unit_up_axis_still_reaches_the_ground() {
    // The probe distance is `radius + ground_probe_epsilon` along `up`.
    // An axis shorter than unit length must not shorten it: with
    // |up| = 0.1 the raw probe would only reach 0.04 m down, landing
    // 0.31 m ABOVE the surface (> ground_probe_epsilon), so a controller
    // that skipped normalization would report "not grounded" here.
    let mut ch = SdfCharacter::new(standing_on([1.0, 0.0, 0.0]), CAP_R, 1.8);
    ch.up = [0.1, 0.0, 0.0];
    let raw_reach = 0.1 * (CAP_R + PROBE_EPS);
    let raw_probe_distance = (PLANET_R + CAP_R - raw_reach) - PLANET_R;
    assert!(
        raw_probe_distance > PROBE_EPS,
        "oracle needs the un-normalized probe to miss: {raw_probe_distance} vs {PROBE_EPS}"
    );
    assert!(ch.is_grounded(&planet()));
}

#[test]
fn a_long_non_unit_up_axis_does_not_scale_the_reported_alignment() {
    // `up_alignment` is `normal · up_unit`, so it stays a cosine in
    // [-1, 1] however long the caller's axis is. Without normalization
    // this would read 7.5 and any `ground_up_threshold` below that would
    // be satisfied by geometry at almost any angle.
    let mut ch = SdfCharacter::new(standing_on([1.0, 0.0, 0.0]), CAP_R, 1.8);
    ch.up = [7.5, 0.0, 0.0];
    let contact = ch.ground_contact(&planet()).expect("probe hits the planet");
    assert!(
        (contact.up_alignment - 1.0).abs() < 1.0e-5,
        "alignment must be a cosine, got {}",
        contact.up_alignment
    );
}

#[test]
fn a_degenerate_up_axis_falls_back_to_y_instead_of_producing_nan() {
    // Without the fallback the probe coordinates are `0/0` and `NaN/NaN`,
    // and every comparison against them is false — the controller would
    // silently report "not grounded" forever rather than fail loudly.
    // Checked at the north pole, where the +Y fallback is the right answer.
    let field = planet();
    for axis in [
        [0.0, 0.0, 0.0],
        [f32::NAN, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
    ] {
        let mut ch = SdfCharacter::new(standing_on([0.0, 1.0, 0.0]), CAP_R, 1.8);
        ch.up = axis;
        let contact = ch
            .ground_contact(&field)
            .unwrap_or_else(|| panic!("degenerate axis {axis:?} lost the ground"));
        assert!(
            contact.distance.is_finite(),
            "axis {axis:?} produced a non-finite distance {}",
            contact.distance
        );
        assert!(
            (contact.up_alignment - 1.0).abs() < 1.0e-5,
            "axis {axis:?} should behave as +Y, alignment {}",
            contact.up_alignment
        );
        assert!(ch.is_grounded(&field), "axis {axis:?} lost the ground");
    }
}

#[test]
fn ground_contact_reports_the_closed_form_distance_and_normal() {
    let mut ch = SdfCharacter::new(standing_on([1.0, 0.0, 0.0]), CAP_R, 1.8);
    ch.up = [1.0, 0.0, 0.0];
    let contact = ch
        .ground_contact(&planet())
        .expect("standing on the planet must report a contact");
    // Probe sits at |p| = R - PROBE_EPS, so the SDF there is -PROBE_EPS.
    assert!(
        (contact.distance - (-PROBE_EPS)).abs() < 1.0e-4,
        "distance {} != {}",
        contact.distance,
        -PROBE_EPS
    );
    // Radial normal at the probe point is the up axis itself.
    assert!((contact.normal[0] - 1.0).abs() < 1.0e-5);
    assert!(contact.normal[1].abs() < 1.0e-5);
    assert!(contact.normal[2].abs() < 1.0e-5);
    // And the up-alignment the grounded test thresholds on.
    assert!((contact.up_alignment - 1.0).abs() < 1.0e-5);
}

#[test]
fn ground_contact_is_none_when_the_probe_finds_nothing() {
    let mut ch = SdfCharacter::new(standing_on([1.0, 0.0, 0.0]), CAP_R, 1.8);
    ch.up = [1.0, 0.0, 0.0];
    // Lift the character far above the surface: the probe is in free air.
    ch.position = [PLANET_R + 50.0, 0.0, 0.0];
    assert!(ch.ground_contact(&planet()).is_none());
    assert!(!ch.is_grounded(&planet()));
}

// ───────────────────── non-exact field / push clamp ─────────────────────

/// Sheet spacing (m).
const SHEET_S: f32 = 2.0;
/// Sheet half-thickness (m).
const SHEET_T: f32 = 0.5;
/// Lipschitz constant of the field: `|grad f| = L`, i.e. NOT a distance
/// field. Gyroid walls are in this class.
const SHEET_L: f32 = 4.0;

/// Signed offset from `y` to the nearest sheet centre, in `[-s/2, s/2]`.
fn sheet_offset(y: f32) -> f32 {
    y - SHEET_S * (y / SHEET_S).round()
}

/// `f(y) = L * (|u| - t)` with the unit normal `sign(u) * y_hat`.
fn periodic_sheets() -> ClosureSdf {
    ClosureSdf::new(
        |_x, y, _z| SHEET_L * (sheet_offset(y).abs() - SHEET_T),
        |_x, y, _z| {
            let u = sheet_offset(y);
            (0.0, if u < 0.0 { -1.0 } else { 1.0 }, 0.0)
        },
    )
}

/// Analytic exit band: the controller stops when `f >= radius`, i.e.
/// `|u| >= t + radius / L`. The band is non-empty iff `s/2` exceeds it.
fn required_clearance(radius: f32) -> f32 {
    SHEET_T + radius / SHEET_L
}

#[test]
fn the_free_band_the_oracle_uses_is_non_empty() {
    // Guards the oracle itself: if the geometry left no room for the
    // capsule, "did not converge" would be correct rather than a defect.
    let need = required_clearance(CAP_R);
    assert!(
        SHEET_S / 2.0 > need,
        "no free band: s/2 = {} is not above {need}",
        SHEET_S / 2.0
    );
}

/// The free pocket immediately above the sheet centred on `y = 0`:
/// `y ∈ [t + radius/L, s − (t + radius/L)]`.
fn pocket_above_origin(radius: f32) -> (f32, f32) {
    let need = required_clearance(radius);
    (need, SHEET_S - need)
}

#[test]
fn one_unclamped_push_overshoots_the_free_pocket_into_the_next_sheet() {
    // Closed form: f(0.1) = L * (0.1 - t) = -1.6, so the push is
    // radius - f + skin = 0.35 + 1.6 + 1e-4 = 1.9501, while reaching the
    // pocket only needs 0.5875 - 0.1 = 0.4875. The character lands at
    // y = 2.0501, whose offset to the sheet at y = 2 is 0.0501 — inside
    // solid geometry, one whole sheet past where it should have stopped.
    let mut ch = SdfCharacter::new([0.0, 0.1, 0.0], CAP_R, 1.8);
    assert!(
        ch.max_push.is_infinite(),
        "default must stay unclamped so existing arithmetic is untouched"
    );
    ch.max_iterations = 1;
    let out = ch.move_and_slide(&periodic_sheets(), [0.0, 0.0, 0.0]);
    let (lo, hi) = pocket_above_origin(CAP_R);
    assert!(
        (out.position[1] - 2.0501).abs() < 1.0e-4,
        "expected the closed-form overshoot y = 2.0501, got {}",
        out.position[1]
    );
    assert!(
        out.position[1] < lo || out.position[1] > hi,
        "overshoot must miss the pocket [{lo}, {hi}], got {}",
        out.position[1]
    );
    assert!(
        sheet_offset(out.position[1]).abs() < required_clearance(CAP_R),
        "overshoot must land inside solid, offset {}",
        sheet_offset(out.position[1]).abs()
    );
}

#[test]
fn an_unclamped_run_converges_in_a_different_pocket_after_tunnelling() {
    // ⚠️ The flag is not the oracle. Measured: the run reports
    // `converged` on iteration 5 at y = 5.156713, which is outside the
    // pocket adjacent to the start — it crossed three solid sheets to get
    // there. A caller that trusts `converged` alone cannot tell this from
    // a clean resolution.
    let ch = SdfCharacter::new([0.0, 0.1, 0.0], CAP_R, 1.8);
    let out = ch.move_and_slide(&periodic_sheets(), [0.0, 0.0, 0.0]);
    let (lo, hi) = pocket_above_origin(CAP_R);
    assert!(out.converged, "measured behaviour is that it does converge");
    assert!(
        out.position[1] < lo || out.position[1] > hi,
        "expected a pocket other than [{lo}, {hi}], got {}",
        out.position[1]
    );
    assert!(
        out.position[1] > SHEET_S,
        "expected at least one whole period of tunnelling, got {}",
        out.position[1]
    );
}

#[test]
fn a_clamped_push_lands_in_the_analytic_free_band() {
    let mut ch = SdfCharacter::new([0.0, 0.1, 0.0], CAP_R, 1.8);
    ch.max_push = 0.2;
    let out = ch.move_and_slide(&periodic_sheets(), [0.0, 0.0, 0.0]);
    assert!(out.converged, "clamped push must escape the sheet");
    // Closed form: steps of 0.2 from y = 0.1 give 0.3, 0.5, 0.7; the
    // exit condition |u| >= 0.5875 first holds at y = 0.7.
    assert!(
        (out.position[1] - 0.7).abs() < 1.0e-5,
        "expected y = 0.7, got {}",
        out.position[1]
    );
    assert_eq!(out.iterations, 3);
    assert!(sheet_offset(out.position[1]).abs() >= required_clearance(CAP_R));
    // And — the part the flag cannot tell you — it is the pocket the
    // character started next to, so nothing was tunnelled through.
    let (lo, hi) = pocket_above_origin(CAP_R);
    assert!(
        out.position[1] >= lo && out.position[1] <= hi,
        "clamped run must stay in the adjacent pocket [{lo}, {hi}], got {}",
        out.position[1]
    );
}

#[test]
fn the_clamp_only_bounds_the_step_and_never_reverses_it() {
    // A clamp that flipped sign or overshot would still "converge" on the
    // sheet field, so pin the monotonicity directly: every intermediate
    // position must move along +Y by at most `max_push`.
    let field = periodic_sheets();
    let mut ch = SdfCharacter::new([0.0, 0.1, 0.0], CAP_R, 1.8);
    ch.max_push = 0.2;
    ch.max_iterations = 1;
    let mut y = ch.position[1];
    for _ in 0..8 {
        let out = ch.move_and_slide(&field, [0.0, 0.0, 0.0]);
        let step = out.position[1] - y;
        assert!(
            step >= 0.0 && step <= ch.max_push + 1.0e-6,
            "step {step} outside (0, max_push]"
        );
        y = out.position[1];
        ch.position = out.position;
        if out.converged {
            break;
        }
    }
    assert!(sheet_offset(y).abs() >= required_clearance(CAP_R));
}

#[test]
fn an_exact_distance_field_is_untouched_by_the_clamp_being_reachable() {
    // Back-compat: on an exact distance field the unclamped push already
    // lands outside in one iteration, and the closed-form landing point
    // is radius + skin above the plane. Pinning it keeps the default
    // path's arithmetic visible if the formula is ever reordered.
    let plane = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    let ch = SdfCharacter::new([0.0, -0.1, 0.0], CAP_R, 1.8);
    let out = ch.move_and_slide(&plane, [0.0, 0.0, 0.0]);
    assert!(out.converged);
    assert_eq!(out.iterations, 1);
    // push = radius - d + skin = 0.35 + 0.1 + 1e-4, from y = -0.1.
    let expected = -0.1 + (CAP_R - (-0.1) + ch.skin_width);
    assert!(
        (out.position[1] - expected).abs() < 1.0e-6,
        "expected {expected}, got {}",
        out.position[1]
    );
}

// ───────────── inward-normal guard / least-penetrating sample ─────────────

/// A field whose gradient points **inward** (toward the planet centre) at
/// the sample point: `f(p) = x0 - x`, so `∇f = (-1, 0, 0)` everywhere and
/// the field is negative for `x > x0`.
///
/// This is the seam case: geometry buried in the ground reports a normal
/// that faces the centre, and pushing along it drives the character
/// underground instead of out.
fn inward_normal_field(x0: f32) -> ClosureSdf {
    ClosureSdf::new(move |x, _y, _z| x0 - x, |_x, _y, _z| (-1.0, 0.0, 0.0))
}

#[test]
fn an_inward_normal_drives_the_character_toward_the_centre_by_default() {
    // Closed form: at x = R the field is x0 - R = -1.0 (x0 = R - 1), so
    // the push is radius - d + skin = 0.35 + 1.0 + 1e-4 = 1.3501 along
    // (-1, 0, 0) — i.e. 1.35 m DEEPER. Pinned as the default behaviour so
    // the guard below is visibly the thing that changes it.
    let x0 = PLANET_R - 1.0;
    let mut ch = SdfCharacter::new([PLANET_R, 0.0, 0.0], CAP_R, 1.8);
    ch.up = [1.0, 0.0, 0.0];
    ch.max_iterations = 1;
    assert!(
        ch.min_up_alignment.is_infinite() && ch.min_up_alignment.is_sign_negative(),
        "default must not substitute the push direction"
    );
    let out = ch.move_and_slide(&inward_normal_field(x0), [0.0, 0.0, 0.0]);
    let expected = PLANET_R - (CAP_R - (x0 - PLANET_R) + ch.skin_width);
    assert!(
        (out.position[0] - expected).abs() < 1.0e-4,
        "expected {expected}, got {}",
        out.position[0]
    );
    assert!(
        out.position[0] < PLANET_R,
        "default push must move inward here, got {}",
        out.position[0]
    );
}

#[test]
fn the_guard_substitutes_the_up_axis_when_the_normal_faces_the_centre() {
    // `n · up = -1`, below the -0.2 threshold, so the push direction
    // becomes `up` itself. Magnitude is unchanged, so the closed-form
    // landing point is the mirror of the test above.
    let x0 = PLANET_R - 1.0;
    let mut ch = SdfCharacter::new([PLANET_R, 0.0, 0.0], CAP_R, 1.8);
    ch.up = [1.0, 0.0, 0.0];
    ch.min_up_alignment = -0.2;
    ch.max_iterations = 1;
    let out = ch.move_and_slide(&inward_normal_field(x0), [0.0, 0.0, 0.0]);
    let expected = PLANET_R + (CAP_R - (x0 - PLANET_R) + ch.skin_width);
    assert!(
        (out.position[0] - expected).abs() < 1.0e-4,
        "expected {expected}, got {}",
        out.position[0]
    );
    assert!(
        out.position[0] > PLANET_R,
        "guarded push must move outward, got {}",
        out.position[0]
    );
}

#[test]
fn the_guard_leaves_a_sideways_normal_alone() {
    // A wall pushing horizontally has `n · up = 0`, which is above the
    // -0.2 threshold: the guard must not hijack it, otherwise the
    // character cannot be pushed out of a wall at all.
    let wall = ClosureSdf::new(|_x, _y, z| z, |_x, _y, _z| (0.0, 0.0, 1.0));
    let mut ch = SdfCharacter::new([0.0, PLANET_R, -0.1], CAP_R, 1.8);
    ch.up = [0.0, 1.0, 0.0];
    ch.min_up_alignment = -0.2;
    ch.max_iterations = 1;
    let out = ch.move_and_slide(&wall, [0.0, 0.0, 0.0]);
    let expected = -0.1 + (CAP_R - (-0.1) + ch.skin_width);
    assert!(
        (out.position[2] - expected).abs() < 1.0e-6,
        "sideways push must survive the guard: expected {expected}, got {}",
        out.position[2]
    );
    assert!(
        (out.position[1] - PLANET_R).abs() < 1.0e-6,
        "guard must not add an up component here, got {}",
        out.position[1]
    );
}

#[test]
fn the_outcome_reports_the_least_penetrating_sample_it_saw() {
    // On the periodic-sheet field the first push makes things WORSE:
    // f(0.1) = -1.6, and the landing point y = 2.0501 samples
    // f = L * (0.0501 - 0.5) = -1.7996. A caller that takes `position`
    // blindly is deeper than when it started; `best_position` is the
    // input, which is the least-penetrating sample of the run.
    let field = periodic_sheets();
    let mut ch = SdfCharacter::new([0.0, 0.1, 0.0], CAP_R, 1.8);
    ch.max_iterations = 1;
    let out = ch.move_and_slide(&field, [0.0, 0.0, 0.0]);
    assert!(!out.converged);
    let final_d = SHEET_L * (sheet_offset(out.position[1]).abs() - SHEET_T);
    assert!(
        (final_d - (-1.7996)).abs() < 1.0e-3,
        "closed-form final sample is -1.7996, got {final_d}"
    );
    assert!(
        (out.best_distance - (-1.6)).abs() < 1.0e-4,
        "closed-form best sample is -1.6, got {}",
        out.best_distance
    );
    assert!(
        (out.best_position[1] - 0.1).abs() < 1.0e-6,
        "best position must be the input here, got {}",
        out.best_position[1]
    );
    assert!(
        out.best_distance > final_d,
        "best must beat the final sample: {} vs {final_d}",
        out.best_distance
    );
}

#[test]
fn the_best_sample_equals_the_final_one_when_the_run_converges() {
    // Back-compat framing: on an exact field the run improves
    // monotonically, so `best_*` adds nothing and callers can keep using
    // `position`.
    let plane = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    let ch = SdfCharacter::new([0.0, -0.1, 0.0], CAP_R, 1.8);
    let out = ch.move_and_slide(&plane, [0.0, 0.0, 0.0]);
    assert!(out.converged);
    assert_eq!(out.best_position, out.position);
    assert!(
        (out.best_distance - (CAP_R + ch.skin_width)).abs() < 1.0e-6,
        "converged sample is radius + skin above the plane, got {}",
        out.best_distance
    );
}

#[test]
fn a_collision_free_move_reports_the_sampled_distance_as_best() {
    // Zero iterations: the first sample already cleared `radius`, so
    // `best_*` must describe that sample rather than stay at a sentinel.
    let plane = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    let ch = SdfCharacter::new([0.0, 2.0, 0.0], CAP_R, 1.8);
    let out = ch.move_and_slide(&plane, [1.0, 0.0, 0.0]);
    assert_eq!(out.iterations, 0);
    assert!(out.converged);
    assert_eq!(out.best_position, out.position);
    assert!((out.best_distance - 2.0).abs() < 1.0e-6);
}

/// A field where every push makes things worse: the distance decreases
/// with `y` while the normal keeps pointing `+Y`.
///
/// `ClosureSdf` takes the distance and the normal as independent
/// closures, and a field that is not an exact distance field is free to
/// have a normal that disagrees with the distance's own gradient — which
/// is precisely the situation the least-penetrating sample exists for.
///
/// `f(y) = -0.5 - 0.1 y`, `n = +Y`.
fn worsening_funnel() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| -0.5 - 0.1 * y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

#[test]
fn the_best_sample_is_the_start_when_every_push_makes_it_worse() {
    // Closed form: f(0) = -0.5 is the maximum of the whole run, since f
    // is strictly decreasing in y and every push moves +Y. The three
    // loop samples are -0.5, -0.585, -0.6785 (pushes 0.8501 and 0.935),
    // so a controller that kept only the LAST sample would report
    // -0.6785 — worse than the position it was handed.
    let mut ch = SdfCharacter::new([0.0, 0.0, 0.0], CAP_R, 1.8);
    ch.max_iterations = 3;
    let out = ch.move_and_slide(&worsening_funnel(), [0.0, 0.0, 0.0]);
    assert!(!out.converged);
    assert!(
        (out.best_distance - (-0.5)).abs() < 1.0e-5,
        "best must be the starting sample -0.5, got {}",
        out.best_distance
    );
    assert!(
        out.best_position[1].abs() < 1.0e-6,
        "best position must be the start, got {}",
        out.best_position[1]
    );
    // And the raw final position is strictly worse, which is the reason
    // the field exists.
    let final_d = -0.5 - 0.1 * out.position[1];
    assert!(
        final_d < out.best_distance,
        "final {final_d} should be worse than best {}",
        out.best_distance
    );
}

#[test]
fn the_best_sample_includes_the_position_the_last_push_produced() {
    // The loop only samples at its top, so the position created by the
    // final push is measured after the loop. Closed form on the sheet
    // field with a budget of 5: the five loop samples are
    // -1.6, -1.7996, -1.2008, -1.0028, -0.40882, and the sixth position
    // (y = 5.156713) samples L * (0.843287 - 0.5) = +1.3731 — the best of
    // the run by a wide margin, and above `radius`.
    //
    // ⚠️ `converged` is still false here: the budget ran out on the very
    // push that escaped. A caller reading only `converged` would discard
    // a position that is actually clear of the geometry.
    let mut ch = SdfCharacter::new([0.0, 0.1, 0.0], CAP_R, 1.8);
    ch.max_iterations = 5;
    let out = ch.move_and_slide(&periodic_sheets(), [0.0, 0.0, 0.0]);
    assert!(!out.converged, "budget must run out on the escaping push");
    assert_eq!(out.iterations, 5);
    assert!(
        (out.best_distance - 1.3731).abs() < 1.0e-3,
        "expected the post-loop sample +1.3731, got {}",
        out.best_distance
    );
    assert!(
        (out.best_position[1] - 5.156_713).abs() < 1.0e-4,
        "best must be the final position, got {}",
        out.best_position[1]
    );
    assert_eq!(out.best_position, out.position);
    assert!(
        out.best_distance >= ch.radius,
        "the escaping sample clears the capsule radius"
    );
}

// ─────────────── velocity state / contact response (`step`) ───────────────

/// `f(p) = y`, the exact distance field of the half-space below `y = 0`.
fn ground_plane() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

#[test]
fn gravity_accumulates_into_the_velocity_state() {
    // `v += g dt`, twice, on all three axes with distinct magnitudes.
    //
    // ⚠️ The first draft only checked `velocity[1]`, and a mutation that
    // dropped `* dt` on the X axis passed it. A per-axis integrator needs
    // a per-axis oracle.
    let mut ch = SdfCharacter::new([0.0, 10.0, 0.0], CAP_R, 1.8);
    assert_eq!(
        ch.velocity,
        [0.0, 0.0, 0.0],
        "default velocity must be zero"
    );
    let dt = 1.0 / 60.0;
    let g = [1.5_f32, -14.0, 3.25];
    ch.apply_gravity(g, dt);
    for (axis, (got, accel)) in ch.velocity.iter().zip(g.iter()).enumerate() {
        let want = accel * dt;
        assert!(
            (got - want).abs() < 1.0e-6,
            "axis {axis} after one step: expected {want}, got {got}"
        );
    }
    ch.apply_gravity(g, dt);
    for (axis, (got, accel)) in ch.velocity.iter().zip(g.iter()).enumerate() {
        let want = 2.0 * accel * dt;
        assert!(
            (got - want).abs() < 1.0e-6,
            "axis {axis} after two steps: expected {want}, got {got}"
        );
    }
}

#[test]
fn a_free_fall_step_advances_by_velocity_times_dt_and_keeps_the_velocity() {
    // No contact, so `step` is exactly the integrator: 10 - 5/60.
    let mut ch = SdfCharacter::new([0.0, 10.0, 0.0], CAP_R, 1.8);
    ch.velocity = [0.0, -5.0, 0.0];
    let dt = 1.0 / 60.0;
    let out = ch.step(&ground_plane(), dt, [0.0, 0.0, 0.0]);
    assert_eq!(out.iterations, 0, "free fall must not resolve anything");
    assert!((ch.position[1] - (10.0 - 5.0 * dt)).abs() < 1.0e-6);
    assert!(
        (ch.velocity[1] - (-5.0)).abs() < 1.0e-7,
        "velocity must survive a contact-free step, got {}",
        ch.velocity[1]
    );
}

#[test]
fn landing_removes_the_velocity_into_the_surface_and_keeps_the_tangent() {
    // Falling fast enough to penetrate in one step. Closed form: the
    // resolution puts the centre at `radius + skin` above the plane, the
    // normal is +Y, and the inelastic kinematic law removes only the
    // normal component — so `vy` becomes 0 while `vx` is untouched.
    let mut ch = SdfCharacter::new([0.0, CAP_R, 0.0], CAP_R, 1.8);
    ch.velocity = [3.0, -20.0, 0.0];
    let out = ch.step(&ground_plane(), 1.0 / 60.0, [0.0, 0.0, 0.0]);
    assert!(out.iterations >= 1, "the fall must have penetrated");
    assert!(
        (ch.position[1] - (CAP_R + ch.skin_width)).abs() < 1.0e-5,
        "expected to rest at radius + skin, got {}",
        ch.position[1]
    );
    assert!(
        ch.velocity[1].abs() < 1.0e-6,
        "normal velocity must be removed, got {}",
        ch.velocity[1]
    );
    assert!(
        (ch.velocity[0] - 3.0).abs() < 1.0e-6,
        "tangential velocity must survive (slide, not stop), got {}",
        ch.velocity[0]
    );
}

#[test]
fn a_contact_does_not_add_velocity_away_from_the_surface() {
    // Already moving up and out: the contact must not touch the velocity,
    // otherwise a character brushing the floor while jumping loses its
    // jump. Only the component pointing INTO the surface is removed.
    let mut ch = SdfCharacter::new([0.0, CAP_R * 0.5, 0.0], CAP_R, 1.8);
    ch.velocity = [0.0, 8.0, 0.0];
    let out = ch.step(&ground_plane(), 1.0 / 60.0, [0.0, 0.0, 0.0]);
    assert!(
        out.iterations >= 1,
        "must have been penetrating at the start"
    );
    assert!(
        (ch.velocity[1] - 8.0).abs() < 1.0e-6,
        "outward velocity must be preserved, got {}",
        ch.velocity[1]
    );
}

#[test]
fn a_wall_removes_only_the_component_into_the_wall() {
    // Wall at z = 0 with normal +Z. Moving (1, 0, -6): the z part is
    // removed, the x part slides along the wall.
    let wall = ClosureSdf::new(|_x, _y, z| z, |_x, _y, _z| (0.0, 0.0, 1.0));
    let mut ch = SdfCharacter::new([0.0, 0.0, CAP_R * 0.5], CAP_R, 1.8);
    ch.velocity = [1.0, 0.0, -6.0];
    let out = ch.step(&wall, 1.0 / 60.0, [0.0, 0.0, 0.0]);
    assert!(out.iterations >= 1);
    assert!(
        ch.velocity[2].abs() < 1.0e-6,
        "into-wall component must go, got {}",
        ch.velocity[2]
    );
    assert!(
        (ch.velocity[0] - 1.0).abs() < 1.0e-6,
        "along-wall component must stay, got {}",
        ch.velocity[0]
    );
}

#[test]
fn the_control_displacement_is_added_on_top_of_the_velocity() {
    // Tangent locomotion (the control term) and ballistic motion (the
    // velocity term) must both land in the same displacement, or a walking
    // character stops falling.
    let mut ch = SdfCharacter::new([0.0, 10.0, 0.0], CAP_R, 1.8);
    ch.velocity = [0.0, -5.0, 0.0];
    let dt = 1.0 / 60.0;
    let out = ch.step(&ground_plane(), dt, [2.0 * dt, 0.0, 0.0]);
    assert_eq!(out.iterations, 0);
    assert!((ch.position[0] - 2.0 * dt).abs() < 1.0e-6);
    assert!((ch.position[1] - (10.0 - 5.0 * dt)).abs() < 1.0e-6);
}

#[test]
fn step_on_a_sphere_world_rests_at_the_surface_with_zero_radial_velocity() {
    // The whole point of the up axis: the same `step` works with radial
    // gravity. Closed form: the character ends at |p| = R + radius + skin
    // and the radial velocity is gone, on a direction that is not +Y.
    let field = planet();
    let dir = [1.0, 0.0, 0.0];
    let mut ch = SdfCharacter::new([PLANET_R + CAP_R, 0.0, 0.0], CAP_R, 1.8);
    ch.up = dir;
    ch.min_up_alignment = -0.2;
    ch.velocity = [-20.0, 0.0, 2.0];
    let out = ch.step(&field, 1.0 / 60.0, [0.0, 0.0, 0.0]);
    assert!(out.iterations >= 1, "the fall must have penetrated");
    let radius_now = (ch.position[0] * ch.position[0]
        + ch.position[1] * ch.position[1]
        + ch.position[2] * ch.position[2])
        .sqrt();
    assert!(
        (radius_now - (PLANET_R + CAP_R + ch.skin_width)).abs() < 1.0e-2,
        "expected |p| = R + radius + skin, got {radius_now}"
    );
    assert!(
        ch.velocity[0].abs() < 1.0e-3,
        "radial velocity must be removed, got {}",
        ch.velocity[0]
    );

    // Closed form for the response. The contact normal is the radial
    // direction at the penetrating point, and that point has already moved
    // tangentially by `v_z · dt`, so the normal is tilted off `+X` by
    // `atan(0.0333 / 300)` ≈ 1.11e-4 rad. Removing `v · n` along that
    // tilted normal therefore feeds a little into `+Z`:
    // `v_z' = v_z − n_z (v · n)` = 2 + 2.22e-3 = 2.002222.
    //
    // ⚠️ The first draft of this test asserted `v_z == 2.0 ± 1e-3` and
    // failed at 2.002222. The tolerance was the error, not the response —
    // so the expectation is derived here instead of widened.
    let dt = 1.0 / 60.0;
    let unresolved = [PLANET_R + CAP_R - 20.0 * dt, 0.0, 2.0 * dt];
    let len = (unresolved[0] * unresolved[0] + unresolved[2] * unresolved[2]).sqrt();
    let n = [unresolved[0] / len, 0.0, unresolved[2] / len];
    let v0 = [-20.0_f32, 0.0, 2.0];
    let into = v0[0] * n[0] + v0[2] * n[2];
    let expected = [v0[0] - n[0] * into, 0.0, v0[2] - n[2] * into];
    for (axis, (got, want)) in ch.velocity.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1.0e-4,
            "axis {axis}: expected {want}, got {got}"
        );
    }
    // The law itself: nothing is left pointing into the surface, and the
    // response never adds speed.
    let residual = ch.velocity[0] * n[0] + ch.velocity[1] * n[1] + ch.velocity[2] * n[2];
    assert!(
        residual.abs() < 1.0e-4,
        "velocity into the surface must be gone, residual {residual}"
    );
    let speed_before = (v0[0] * v0[0] + v0[2] * v0[2]).sqrt();
    let speed_after = (ch.velocity[0] * ch.velocity[0]
        + ch.velocity[1] * ch.velocity[1]
        + ch.velocity[2] * ch.velocity[2])
        .sqrt();
    assert!(
        speed_after <= speed_before + 1.0e-4,
        "the contact must not add speed: {speed_after} > {speed_before}"
    );
    assert!(
        (ch.velocity[2] - 2.0).abs() < 5.0e-3,
        "tangential velocity must survive, got {}",
        ch.velocity[2]
    );
}

#[test]
fn step_reports_the_same_outcome_move_and_slide_would() {
    // `step` must not become a second, divergent implementation of the
    // resolution: its outcome has to agree with `move_and_slide` given
    // the same total displacement.
    let field = periodic_sheets();
    let dt = 1.0 / 60.0;
    let mut stepped = SdfCharacter::new([0.0, 0.1, 0.0], CAP_R, 1.8);
    stepped.velocity = [0.0, -1.0, 0.0];
    let direct = {
        let probe = stepped;
        probe.move_and_slide(&field, [0.0, -dt, 0.0])
    };
    let out = stepped.step(&field, dt, [0.0, 0.0, 0.0]);
    assert_eq!(out.position, direct.position);
    assert_eq!(out.converged, direct.converged);
    assert_eq!(out.iterations, direct.iterations);
    assert_eq!(out.best_position, direct.best_position);
}

#[test]
fn step_takes_the_least_penetrating_sample_when_the_run_did_not_converge() {
    // The consumer-facing consequence of `best_position`: `step` must not
    // hand back a position that is deeper than where it started, which is
    // what taking `position` blindly would do on this field.
    let field = periodic_sheets();
    let mut ch = SdfCharacter::new([0.0, 0.1, 0.0], CAP_R, 1.8);
    ch.max_iterations = 1;
    let out = ch.step(&field, 1.0 / 60.0, [0.0, 0.0, 0.0]);
    assert!(!out.converged);
    assert_eq!(
        ch.position, out.best_position,
        "step must adopt the least-penetrating sample when it did not converge"
    );
    assert!(
        (ch.position[1] - out.best_position[1]).abs() < 1.0e-9
            && (ch.position[1] - out.position[1]).abs() > 1.0e-3,
        "and that sample must differ from the raw final position here"
    );
}

// ───────────── なぜ 2 つの安全弁に「安全な既定」が無いか ─────────────
//
// `max_push` と `min_up_alignment` は、実測で必要と分かった不変条件を opt-in
// にしたものなので「既定を安全側にすべきでは」という問いが自然に出る
// ⚠️ **両方とも、既定にすると別の正しい場を壊す** ここはその反例を閉形式で
// 固定して、既定を動かす変更が red になるようにする

#[test]
fn capping_the_push_by_default_would_strand_a_deeply_penetrating_character() {
    // 厳密距離場に深く入った場合、正しい 1 回の押し出しは `radius - d` で、
    // `d` は任意に深くなりうる `max_push` を capsule の大きさ程度で既定 cap
    // すると、`max_iterations` 予算内で脱出できなくなる
    //
    // 閉形式: 平面 `y` の内側 `y = -5` から出るには `radius + 5 = 5.35` 必要
    // cap を `radius`(0.35) にすると 1 反復 0.35 なので 8 反復で 2.8 しか進めず、
    // `-5 → -2.2` で予算切れ (脱出に要る反復数は ceil(5.35/0.35) = 16)
    let plane = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    let deep = [0.0, -5.0, 0.0];

    let unclamped = SdfCharacter::new(deep, CAP_R, 1.8);
    let out = unclamped.move_and_slide(&plane, [0.0, 0.0, 0.0]);
    assert!(
        out.converged && out.iterations == 1,
        "既定 (INFINITY) は 1 反復で脱出する: converged={} iters={}",
        out.converged,
        out.iterations
    );
    // 閉形式: `-5 + (radius + 5 + skin)` = `radius + skin`
    // ⚠️ 初稿は `CAP_R + 5.0 + skin` と書いて外した (開始点が -5 であることを
    // 足し忘れた) 押し出し量と着地点を混同しない
    assert!(
        (out.position[1] - (CAP_R + unclamped.skin_width)).abs() < 1.0e-4,
        "expected {}, got {}",
        CAP_R + unclamped.skin_width,
        out.position[1]
    );

    let mut capped = SdfCharacter::new(deep, CAP_R, 1.8);
    capped.max_push = CAP_R;
    let out = capped.move_and_slide(&plane, [0.0, 0.0, 0.0]);
    assert!(
        !out.converged,
        "cap を既定にすると脱出できない (この test が既定変更の red 化を担う)"
    );
    assert_eq!(out.iterations, capped.max_iterations);
    let reached = -5.0 + CAP_R * (capped.max_iterations as f32);
    assert!(
        (out.position[1] - reached).abs() < 1.0e-4,
        "閉形式 {reached} に届くだけ: {}",
        out.position[1]
    );
}

#[test]
fn substituting_up_by_default_would_push_a_character_into_a_ceiling() {
    // `min_up_alignment` は「地面に埋まった geometry の seam で法線が世界の
    // 中心を向く」場のためのもの ⚠️ **天井の下にいる状態も `n·up ≈ -1`** で、
    // 法線だけからは区別できない
    //
    // 天井 = `y = 0` より上が solid、外向き法線は `-Y` 天井にめり込んだ
    // character は `-Y` に押し出されるのが正しい `up = +Y` を代入すると
    // **さらに天井の中へ** 進む
    //
    // 閉形式: `f(y) = -y` なので `y = 0.1` で `d = -0.1`、押し出し量は
    // `radius + 0.1 + skin = 0.4501` 正しい向き (-Y) なら `y = -0.3501`
    // (天井の外)、`up` 代入なら `y = +0.5501` (天井の中、しかも深い)
    let ceiling = ClosureSdf::new(|_x, y, _z| -y, |_x, _y, _z| (0.0, -1.0, 0.0));
    let inside = [0.0, 0.1, 0.0];
    let push = CAP_R + 0.1 + 1.0e-4;

    let plain = SdfCharacter::new(inside, CAP_R, 1.8);
    let out = plain.move_and_slide(&ceiling, [0.0, 0.0, 0.0]);
    assert!(
        (out.position[1] - (0.1 - push)).abs() < 1.0e-5,
        "既定は法線方向 (-Y) に逃がす: expected {}, got {}",
        0.1 - push,
        out.position[1]
    );
    assert!(out.converged, "天井の外に出ている");

    // 1 反復だけ見ると閉形式どおり `+Y` に `push` だけ動く (= 天井の中へ)
    let mut one_step = SdfCharacter::new(inside, CAP_R, 1.8);
    one_step.min_up_alignment = -0.2;
    one_step.max_iterations = 1;
    let out = one_step.move_and_slide(&ceiling, [0.0, 0.0, 0.0]);
    assert!(
        (out.position[1] - (0.1 + push)).abs() < 1.0e-5,
        "guard を既定にすると +Y に押される: expected {}, got {}",
        0.1 + push,
        out.position[1]
    );

    // ⚠️ **予算いっぱい回すと発散する** 深く入るほど `radius - d` が増えるので
    // 押し出しが毎反復 大きくなる ⚠️ 初稿は 1 反復分 (0.5501) を期待して
    // 外した (実測 114.88) 反復の累積を忘れない
    let mut guarded = SdfCharacter::new(inside, CAP_R, 1.8);
    guarded.min_up_alignment = -0.2;
    let out = guarded.move_and_slide(&ceiling, [0.0, 0.0, 0.0]);
    assert!(!out.converged, "天井の中を昇り続けるので収束しない");
    assert_eq!(out.iterations, guarded.max_iterations);
    assert!(
        out.position[1] > 100.0,
        "予算 8 反復で発散する (実測 114.88): {}",
        out.position[1]
    );
    assert!(
        out.position[1] > 0.1,
        "元より深く入っている: {} > 0.1",
        out.position[1]
    );
}

#[test]
fn the_resolved_position_helper_is_the_safe_read_of_an_outcome() {
    // `converged` だけを読む消費側の誤りを 1 呼び出しで潰す
    // (収束時は `position`、非収束時は `best_position`)
    let field = periodic_sheets();
    let mut ch = SdfCharacter::new([0.0, 0.1, 0.0], CAP_R, 1.8);
    ch.max_iterations = 1;
    let out = ch.move_and_slide(&field, [0.0, 0.0, 0.0]);
    assert!(!out.converged);
    assert_eq!(out.resolved_position(), out.best_position);
    assert_ne!(out.resolved_position(), out.position);

    let plane = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    let ch = SdfCharacter::new([0.0, -0.1, 0.0], CAP_R, 1.8);
    let out = ch.move_and_slide(&plane, [0.0, 0.0, 0.0]);
    assert!(out.converged);
    assert_eq!(out.resolved_position(), out.position);
}
