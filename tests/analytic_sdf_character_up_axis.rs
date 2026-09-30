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
