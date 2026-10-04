//! Analytic oracles for the sphere-world building blocks:
//! [`SphericalHeightField`], [`SdfUnion`], [`central_gravity`] and the
//! arc locomotion of [`SdfCharacter::step_on_sphere`].
//!
//! Every expected value is a closed form written here; none is obtained by
//! calling the implementation under test.
//!
//! - height field: `f(p) = (|p - c| - R - h(dir)) / sqrt(1 + s^2)`,
//!   `dir = (p - c) / |p - c|`; for `h(dir) = a * (dir . k)` the spatial
//!   gradient of `|p - c| - R - h` is `r_hat - a * (k - (k . r_hat) r_hat) / |p - c|`
//! - Lipschitz: with `s >= max |dh/dtheta| / R`, `|grad f| <= 1` for `|p - c| >= R`
//! - union: `min(f_a, f_b)`, normal of the smaller operand
//! - central gravity: `-g * r_hat`, zero at the centre
//! - arc step: angle travelled `= |v_t| * dt / r`, radius unchanged
//! - rest: on a sphere of radius `R` a character of radius `rho` settles at
//!   `R + rho + skin` with no tangential drift, at every latitude
//! - walkable slope: on a plane tilted below `ground_up_threshold` gravity
//!   along `-up` does not make the character creep downhill; above it, it slides

use alice_physics::sdf_character::SdfCharacter;
use alice_physics::sdf_collider::{SdfField, SdfUnion};
use alice_physics::spherical_terrain::{central_gravity, SphericalHeightField};

const R: f32 = 50.0;

fn len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn unit(a: [f32; 3]) -> [f32; 3] {
    scale(a, 1.0 / len(a))
}

/// Deterministic LCG in [-1, 1).
struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
    fn dir(&mut self) -> [f32; 3] {
        loop {
            let v = [self.next(), self.next(), self.next()];
            let l = len(v);
            if l > 0.1 && l <= 1.0 {
                return scale(v, 1.0 / l);
            }
        }
    }
}

/// Directions spanning the pole, the equator, the south and two oblique ones.
const DIRS: [[f32; 3]; 6] = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.577_350_3, 0.577_350_3, 0.577_350_3],
    [-0.6, 0.0, 0.8],
];

// ── height field ──

#[test]
fn constant_height_is_the_radial_offset_with_a_radial_normal() {
    let c = [1.0, -2.0, 3.0];
    let field = SphericalHeightField::new(c, R, |_d: [f32; 3]| 1.5);
    for d in DIRS {
        for alt in [-3.0_f32, 0.0, 0.25, 7.0] {
            let p = [
                c[0] + d[0] * (R + alt),
                c[1] + d[1] * (R + alt),
                c[2] + d[2] * (R + alt),
            ];
            let want = alt - 1.5;
            let got = field.distance(p[0], p[1], p[2]);
            assert!(
                (got - want).abs() < 1e-4,
                "dir {d:?} alt {alt}: {got} vs {want}"
            );
            assert!((field.radial_height(p) - want).abs() < 1e-4);
            let (nx, ny, nz) = field.normal(p[0], p[1], p[2]);
            assert!(
                (nx - d[0]).abs() < 1e-4 && (ny - d[1]).abs() < 1e-4 && (nz - d[2]).abs() < 1e-4,
                "dir {d:?}: normal ({nx},{ny},{nz})"
            );
        }
        assert!((field.surface_radius(d) - (R + 1.5)).abs() < 1e-4);
    }
}

#[test]
fn linear_height_normal_matches_the_closed_form_gradient() {
    let a = 4.0_f32;
    let k = unit([0.3, 0.5, -0.8]);
    let field = SphericalHeightField::new([0.0; 3], R, move |d: [f32; 3]| a * dot(d, k));
    let mut rng = Lcg(7);
    for _ in 0..64 {
        let d = rng.dir();
        let r = R + 2.0;
        let p = scale(d, r);
        // closed form: r_hat - a (k - (k . r_hat) r_hat) / r
        let kt = sub(k, scale(d, dot(k, d)));
        let g = sub(d, scale(kt, a / r));
        let want = unit(g);
        let (nx, ny, nz) = field.normal(p[0], p[1], p[2]);
        let err = len(sub([nx, ny, nz], want));
        assert!(err < 2e-3, "dir {d:?}: normal error {err}");
        // the radial value is exact regardless of the slope
        assert!((field.radial_height(p) - (2.0 - a * dot(d, k))).abs() < 1e-4);
    }
}

#[test]
fn declared_slope_bounds_the_lipschitz_constant_above_the_reference_sphere() {
    // h = a * (dir . x_hat): |dh/dtheta| <= a, so the slope bound is a / R
    let a = 0.8 * R;
    let h = move |d: [f32; 3]| a * d[0];
    let raw = SphericalHeightField::new([0.0; 3], R, h);
    let bounded = SphericalHeightField::new([0.0; 3], R, h).with_max_slope(a / R);
    let mut rng = Lcg(11);
    let mut worst_raw = 0.0_f32;
    let mut worst_bounded = 0.0_f32;
    for _ in 0..4000 {
        let d = rng.dir();
        let r = R * (1.0 + 0.5 * (rng.next() + 1.0));
        let p = scale(d, r);
        let step = scale(rng.dir(), 0.05);
        let q = [p[0] + step[0], p[1] + step[1], p[2] + step[2]];
        if len(q) < R {
            continue;
        }
        let dp = len(step);
        worst_raw = worst_raw
            .max((raw.distance(q[0], q[1], q[2]) - raw.distance(p[0], p[1], p[2])).abs() / dp);
        worst_bounded = worst_bounded.max(
            (bounded.distance(q[0], q[1], q[2]) - bounded.distance(p[0], p[1], p[2])).abs() / dp,
        );
    }
    // the sample really is steep: the unscaled radial offset is not 1-Lipschitz
    assert!(
        worst_raw > 1.1,
        "sample not steep enough: raw L = {worst_raw}"
    );
    assert!(
        worst_bounded <= 1.0 + 2e-3,
        "bounded field L = {worst_bounded}"
    );
}

#[test]
fn degenerate_queries_stay_finite() {
    // the centre itself has no direction: documented fallback is +Y, so the
    // value is -R - h(+Y) = -R - 2 for h(dir) = 2 * dir.y
    let field = SphericalHeightField::new([0.0; 3], R, |d: [f32; 3]| 2.0 * d[1]);
    let d0 = field.distance(0.0, 0.0, 0.0);
    assert!((d0 + R + 2.0).abs() < 1e-4, "{d0}");
    let (nx, ny, nz) = field.normal(0.0, 0.0, 0.0);
    assert!(nx.is_finite() && ny.is_finite() && nz.is_finite());
    assert!((nx * nx + ny * ny + nz * nz - 1.0).abs() < 1e-4);
}

#[test]
#[should_panic(expected = "radius")]
fn non_positive_radius_is_rejected() {
    let _ = SphericalHeightField::new([0.0; 3], 0.0, |_d: [f32; 3]| 0.0);
}

#[test]
#[should_panic(expected = "max_slope")]
fn negative_slope_bound_is_rejected() {
    let _ = SphericalHeightField::new([0.0; 3], R, |_d: [f32; 3]| 0.0).with_max_slope(-1.0);
}

// ── union ──

#[test]
fn union_takes_the_nearer_surface_and_its_normal() {
    // land: R + 1 on the north side, R - 6 on the south side; water at R - 2
    let land = SphericalHeightField::new(
        [0.0; 3],
        R,
        |d: [f32; 3]| if d[1] > 0.0 { 1.0 } else { -6.0 },
    );
    let water = SphericalHeightField::new([0.0; 3], R, |_d: [f32; 3]| -2.0);
    let walk = SdfUnion::new(land, water);
    for d in DIRS {
        let standing = if d[1] > 0.0 { 1.0_f32 } else { -2.0 }; // max(land, water)
        for alt in [-1.0_f32, 0.0, 3.0] {
            let p = scale(d, R + standing + alt);
            let got = walk.distance(p[0], p[1], p[2]);
            assert!((got - alt).abs() < 1e-4, "dir {d:?} alt {alt}: {got}");
        }
    }
    // normal of the winning operand
    let a = SphericalHeightField::new([0.0; 3], R, |_d: [f32; 3]| 0.0);
    // half-space x < 60 is solid: distance 60 - x, outward normal -x
    let plane = alice_physics::ClosureSdf::new(|x, _y, _z| 60.0 - x, |_x, _y, _z| (-1.0, 0.0, 0.0));
    let u = SdfUnion::new(a, plane);
    let (nx, ny, nz) = u.normal(0.0, R + 3.0, 0.0); // sphere 3 < plane 60
    assert!(nx.abs() < 1e-4 && (ny - 1.0).abs() < 1e-4 && nz.abs() < 1e-4);
    let (nx, ny, nz) = u.normal(70.0, 0.0, 0.0); // plane -10 < sphere 20
    assert!((nx + 1.0).abs() < 1e-6 && ny == 0.0 && nz == 0.0);
}

// ── central gravity ──

#[test]
fn central_gravity_points_at_the_centre_with_constant_magnitude() {
    let c = [2.0, 3.0, -1.0];
    for d in DIRS {
        for r in [0.5_f32, R, 3.0 * R] {
            let p = [c[0] + d[0] * r, c[1] + d[1] * r, c[2] + d[2] * r];
            let g = central_gravity(c, 9.8, p);
            let want = scale(d, -9.8);
            assert!(len(sub(g, want)) < 1e-4, "dir {d:?} r {r}: {g:?}");
        }
    }
    assert_eq!(central_gravity(c, 9.8, c), [0.0; 3]);
}

#[test]
fn apply_central_gravity_integrates_toward_the_centre() {
    let mut ch = SdfCharacter::new([R + 2.0, 0.0, 0.0], 0.3, 1.7);
    ch.apply_central_gravity([0.0; 3], 10.0, 0.5);
    assert!(
        len(sub(ch.velocity, [-5.0, 0.0, 0.0])) < 1e-6,
        "{:?}",
        ch.velocity
    );
}

// ── arc locomotion ──

/// A field with no surface anywhere near.
fn empty() -> alice_physics::ClosureSdf {
    alice_physics::ClosureSdf::new(|_x, _y, _z| 1.0e6, |_x, _y, _z| (0.0, 1.0, 0.0))
}

#[test]
fn arc_step_travels_speed_times_time_along_the_great_circle() {
    let field = empty();
    let dt = 1.0 / 120.0;
    for (d, t) in [
        ([0.0_f32, 1.0, 0.0], [1.0_f32, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        (unit([0.3, -0.4, 0.5]), unit([0.4, 0.3, 0.0])),
    ] {
        // tangent part of t only (closed form of the travel direction)
        let tt = unit(sub(t, scale(d, dot(t, d))));
        let r = R + 1.7;
        let speed = 6.0;
        let mut ch = SdfCharacter::new(scale(d, r), 0.3, 1.7);
        let steps = 600;
        for _ in 0..steps {
            // the caller holds a constant heading along the great circle
            // through d and tt: heading = (d x tt) x up
            let up = unit(ch.position);
            let axis = [
                d[1] * tt[2] - d[2] * tt[1],
                d[2] * tt[0] - d[0] * tt[2],
                d[0] * tt[1] - d[1] * tt[0],
            ];
            let heading = unit([
                axis[1] * up[2] - axis[2] * up[1],
                axis[2] * up[0] - axis[0] * up[2],
                axis[0] * up[1] - axis[1] * up[0],
            ]);
            ch.step_on_sphere(&field, [0.0; 3], dt, scale(heading, speed));
        }
        let angle_want = speed * dt * steps as f32 / r;
        let angle_got = alice_physics::det_math::acos(dot(unit(ch.position), d).clamp(-1.0, 1.0));
        assert!(
            (angle_got - angle_want).abs() < 1e-4,
            "dir {d:?}: angle {angle_got} vs arc law {angle_want}"
        );
        assert!(
            (len(ch.position) - r).abs() < 2e-3,
            "radius drift {}",
            len(ch.position) - r
        );
        // stays on the great circle spanned by d and tt
        let normal = [
            d[1] * tt[2] - d[2] * tt[1],
            d[2] * tt[0] - d[0] * tt[2],
            d[0] * tt[1] - d[1] * tt[0],
        ];
        assert!(dot(unit(ch.position), normal).abs() < 1e-4);
    }
}

#[test]
fn radial_part_of_the_tangent_velocity_is_ignored() {
    let field = empty();
    let mut ch = SdfCharacter::new([0.0, R, 0.0], 0.3, 1.7);
    ch.step_on_sphere(&field, [0.0; 3], 0.1, [0.0, 50.0, 0.0]);
    assert!(
        len(sub(ch.position, [0.0, R, 0.0])) < 1e-4,
        "{:?}",
        ch.position
    );
}

#[test]
fn radial_velocity_stays_radial_across_an_arc_step() {
    // a jump while walking: the radial speed is carried to the new up axis
    let field = empty();
    let mut ch = SdfCharacter::new([0.0, R, 0.0], 0.3, 1.7);
    ch.velocity = [0.0, 4.0, 0.0];
    let dt = 0.05;
    ch.step_on_sphere(&field, [0.0; 3], dt, [10.0, 0.0, 0.0]);
    let up = unit(ch.position);
    let radial = dot(ch.velocity, up);
    let tangential = len(sub(ch.velocity, scale(up, radial)));
    assert!((radial - 4.0).abs() < 1e-4, "radial {radial}");
    assert!(tangential < 1e-4, "tangential {tangential}");
    // up follows the position
    assert!(len(sub(unit(ch.up), up)) < 1e-3);
}

#[test]
fn zero_dt_and_centre_position_are_harmless() {
    let field = empty();
    let mut ch = SdfCharacter::new([0.0, R, 0.0], 0.3, 1.7);
    ch.step_on_sphere(&field, [0.0; 3], 0.0, [5.0, 0.0, 0.0]);
    assert_eq!(ch.position, [0.0, R, 0.0]);
    let mut at_centre = SdfCharacter::new([0.0; 3], 0.3, 1.7);
    at_centre.apply_central_gravity([0.0; 3], 9.8, 0.1);
    at_centre.step_on_sphere(&field, [0.0; 3], 0.1, [5.0, 0.0, 0.0]);
    assert!(at_centre.position.iter().all(|v| v.is_finite()));
    assert!(at_centre.velocity.iter().all(|v| v.is_finite()));
    // no up axis at the centre: up is left as it was (+Y from the constructor)
    assert_eq!(at_centre.up, [0.0, 1.0, 0.0]);
}

// ── rest under central gravity ──

#[test]
fn central_gravity_settles_on_the_sphere_at_every_latitude() {
    let field = SphericalHeightField::new([0.0; 3], R, |_d: [f32; 3]| 0.0);
    let dt = 1.0 / 120.0;
    for d in DIRS {
        let mut ch = SdfCharacter::new(scale(d, R + 3.0), 0.3, 1.7);
        for _ in 0..600 {
            ch.apply_central_gravity([0.0; 3], 9.8, dt);
            ch.step_on_sphere(&field, [0.0; 3], dt, [0.0; 3]);
        }
        let want = R + 0.3 + ch.skin_width;
        assert!(
            (len(ch.position) - want).abs() < 2e-3,
            "dir {d:?}: radius {}",
            len(ch.position)
        );
        assert!(
            len(sub(unit(ch.position), d)) < 1e-4,
            "dir {d:?}: drifted to {:?}",
            unit(ch.position)
        );
        assert!(
            len(ch.velocity) < 0.2,
            "dir {d:?}: velocity {:?}",
            ch.velocity
        );
        assert!(ch.is_grounded(&field), "dir {d:?}: not grounded");
    }
}

// ── walkable slope ──

fn tilted_plane(slope_deg: f32) -> alice_physics::ClosureSdf {
    let (s, c) = alice_physics::det_math::sin_cos(slope_deg.to_radians());
    // plane through the origin with normal (s, c, 0): rises toward -x
    alice_physics::ClosureSdf::new(move |x, y, _z| s * x + c * y, move |_x, _y, _z| (s, c, 0.0))
}

#[test]
fn walkable_slope_holds_and_steep_slope_slides() {
    let dt = 1.0 / 120.0;
    // a planet so large that up is +Y everywhere near the origin
    let centre = [0.0, -1.0e5, 0.0];
    let g = 9.8;
    for (deg, should_hold) in [(20.0_f32, true), (40.0, true), (60.0, false)] {
        let field = tilted_plane(deg);
        let mut ch = SdfCharacter::new([0.0, 0.31, 0.0], 0.3, 1.7);
        for _ in 0..240 {
            ch.apply_central_gravity(centre, g, dt);
            ch.step_on_sphere(&field, centre, dt, [0.0; 3]);
        }
        let drift = ch.position[0].abs();
        if should_hold {
            assert!(drift < 1e-3, "{deg} deg (walkable): crept {drift} m");
        } else {
            assert!(drift > 0.1, "{deg} deg (steep): did not slide ({drift} m)");
        }
    }
}

#[test]
fn standing_still_is_stationary_with_a_thick_skin() {
    // A skin thicker than one frame's fall (g·dt² = 9.8/120² ≈ 6.8e-4 m):
    // without a ground snap the character free-falls through the skin every
    // few frames and bobs by up to `skin_width`. Closed form of rest: the
    // radius stays at R + radius + skin, frame after frame.
    let field = SphericalHeightField::new([0.0; 3], R, |_d: [f32; 3]| 0.0);
    let dt = 1.0 / 120.0;
    for d in DIRS {
        let mut ch = SdfCharacter::new(scale(d, R + 1.0), 0.3, 1.7);
        ch.skin_width = 0.01;
        for _ in 0..600 {
            ch.apply_central_gravity([0.0; 3], 9.8, dt);
            ch.step_on_sphere(&field, [0.0; 3], dt, [0.0; 3]);
        }
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for _ in 0..120 {
            ch.apply_central_gravity([0.0; 3], 9.8, dt);
            ch.step_on_sphere(&field, [0.0; 3], dt, [0.0; 3]);
            let r = len(ch.position);
            lo = lo.min(r);
            hi = hi.max(r);
        }
        let want = R + 0.3 + 0.01;
        assert!(hi - lo < 2e-5, "dir {d:?}: bobs between {lo} and {hi}");
        assert!(
            (hi - want).abs() < 1e-4,
            "dir {d:?}: rests at {hi}, closed form {want}"
        );
    }
}

#[test]
fn a_jump_from_the_ground_is_not_snapped_back() {
    // starting inside the skin band and moving away slowly: still inside the
    // band after the frame, but moving up, so it must be left in free flight
    let field = SphericalHeightField::new([0.0; 3], R, |_d: [f32; 3]| 0.0);
    let dt = 1.0 / 120.0;
    let mut ch = SdfCharacter::new([0.0, R + 0.305, 0.0], 0.3, 1.7);
    ch.skin_width = 0.01;
    ch.velocity = [0.0, 0.1, 0.0];
    ch.step_on_sphere(&field, [0.0; 3], dt, [0.0; 3]);
    let want = R + 0.305 + 0.1 * dt;
    assert!(
        (ch.position[1] - want).abs() < 1e-5,
        "{:?} vs {want}",
        ch.position
    );
    assert!((ch.velocity[1] - 0.1).abs() < 1e-6, "{:?}", ch.velocity);
}

#[test]
fn a_steep_contact_inside_the_skin_band_is_not_snapped() {
    // 60 degrees is steeper than the default 0.7 threshold (cos 60 = 0.5):
    // the snap would lift by (band - d) / 0.5 along up; it must not run
    let field = tilted_plane(60.0);
    let centre = [0.0, -1.0e5, 0.0];
    let mut ch = SdfCharacter::new([0.0, 0.305 / 0.5, 0.0], 0.3, 1.7); // d = 0.305
    ch.skin_width = 0.01;
    let before = ch.position;
    ch.step_on_sphere(&field, centre, 0.0, [0.0; 3]);
    assert!(
        len(sub(ch.position, before)) < 1e-6,
        "{:?} -> {:?}",
        before,
        ch.position
    );
}
