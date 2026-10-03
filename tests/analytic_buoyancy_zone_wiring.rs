//! Closed-form oracles for `buoyancy_zone`'s two previously-unwired items,
//! `ZoneShape::depth_below_surface` and `BuoyancyZone::water_pool`, and for the
//! module's own stated properties (`force_on`, `submerged_fraction`).
//!
//! Closed forms (textbook, not copied from the implementation):
//!
//! ```text
//! cap volume fraction  f(h) = h^2 (3r - h) / (4 r^3),  h = clamp(d + r, 0, 2r)
//! buoyancy             F_b  = rho * (4/3 pi r^3) * f * g          (Archimedes)
//! drag                 F_d  = -c_lin v - c_quad |v| v
//! ```
//!
//! The second, independent oracle for the cap fraction is a midpoint-rule
//! integral of the disc areas `pi (r^2 - z^2)` over the submerged slab, which
//! shares no algebra with the closed form.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 `powi` / `sqrt` are oracle references computed outside the crate.
#![allow(clippy::disallowed_methods)]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_physics::buoyancy_zone::{BuoyancyZone, ZoneShape};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{BodyType, RigidBody};

fn body_at(x: i64, y: i64, z: i64, v: Vec3Fix) -> RigidBody {
    RigidBody {
        position: Vec3Fix::from_int(x, y, z),
        velocity: v,
        inv_mass: Fix128::ONE,
        inv_inertia: Vec3Fix::ZERO,
        prev_position: Vec3Fix::from_int(x, y, z),
        rotation: QuatFix::IDENTITY,
        angular_velocity: Vec3Fix::ZERO,
        prev_rotation: QuatFix::IDENTITY,
        restitution: Fix128::ZERO,
        friction: Fix128::ZERO,
        gravity_scale: Fix128::ONE,
        linear_damping: Fix128::ONE,
        angular_damping: Fix128::ONE,
        is_sensor: false,
        body_type: BodyType::Dynamic,
        kinematic_target: None,
    }
}

fn pool() -> ZoneShape {
    // x,z in [-1, 1], y in [-2, 2]: surface at y = 2.
    ZoneShape::Aabb {
        min: Vec3Fix::from_int(-1, -2, -1),
        max: Vec3Fix::from_int(1, 2, 1),
    }
}

fn ball() -> ZoneShape {
    // centre (0, 1, 0), radius 2: top at y = 3.
    ZoneShape::Sphere {
        centre: Vec3Fix::from_int(0, 1, 0),
        radius: Fix128::from_int(2),
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

// ---------------------------------------------------------------- depth

#[test]
fn aabb_depth_is_top_minus_y_inside() {
    let s = pool();
    // integers only, so exact
    assert_eq!(
        s.depth_below_surface(Vec3Fix::from_int(0, 0, 0)),
        Fix128::from_int(2)
    );
    assert_eq!(
        s.depth_below_surface(Vec3Fix::from_int(1, -2, -1)),
        Fix128::from_int(4),
        "corner of the box is inside (inclusive bounds), floor depth = full height"
    );
    assert_eq!(
        s.depth_below_surface(Vec3Fix::from_int(0, 2, 0)),
        Fix128::ZERO,
        "exactly at the surface"
    );
    // fractional depth: 2 - 0.25 = 1.75 (dyadic, exact)
    let p = Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 4), Fix128::ZERO);
    assert_eq!(
        s.depth_below_surface(p),
        Fix128::from_ratio(7, 4),
        "top - y with y = 1/4"
    );
}

#[test]
fn aabb_depth_is_zero_outside_every_face() {
    let s = pool();
    for p in [
        Vec3Fix::from_int(0, 3, 0),  // above the surface
        Vec3Fix::from_int(0, -3, 0), // below the floor (pre-1.2.0 clamped to full depth)
        Vec3Fix::from_int(2, 0, 0),  // +x outside
        Vec3Fix::from_int(-2, 0, 0), // -x outside
        Vec3Fix::from_int(0, 0, 2),  // +z outside
        Vec3Fix::from_int(0, 0, -2), // -z outside
    ] {
        assert_eq!(s.depth_below_surface(p), Fix128::ZERO, "{p:?}");
    }
}

#[test]
fn sphere_depth_is_top_minus_y_strictly_inside_the_ball() {
    let s = ball();
    // (0,0,0): |d| = 1 < 2, top = 3 => 3
    assert_eq!(
        s.depth_below_surface(Vec3Fix::from_int(0, 0, 0)),
        Fix128::from_int(3)
    );
    // (1,1,0): in the ball, depth 2
    assert_eq!(
        s.depth_below_surface(Vec3Fix::from_int(1, 1, 0)),
        Fix128::from_int(2)
    );
    // on the sphere surface (dist == r) is *not* inside: ZERO
    assert_eq!(
        s.depth_below_surface(Vec3Fix::from_int(2, 1, 0)),
        Fix128::ZERO
    );
    assert_eq!(
        s.depth_below_surface(Vec3Fix::from_int(0, 3, 0)),
        Fix128::ZERO,
        "the north pole is on the sphere"
    );
    // outside the ball
    assert_eq!(
        s.depth_below_surface(Vec3Fix::from_int(3, 1, 0)),
        Fix128::ZERO
    );
    // (1, 3, 1): dist = sqrt(1 + 4 + 1) > 2 although inside the horizontal extent
    assert_eq!(
        s.depth_below_surface(Vec3Fix::from_int(1, 3, 1)),
        Fix128::ZERO
    );
}

#[test]
fn sphere_depth_agrees_with_contains_off_the_boundary() {
    // depth > 0 <=> contains (strictly inside), over a lattice that avoids ties
    let s = ball();
    let mut n_inside = 0;
    for ix in -6..=6 {
        for iy in -6..=8 {
            for iz in -6..=6 {
                let p = Vec3Fix::new(
                    Fix128::from_ratio(ix * 7, 10),
                    Fix128::from_ratio(iy * 7, 10),
                    Fix128::from_ratio(iz * 7, 10),
                );
                let d = s.depth_below_surface(p);
                let inside = s.contains(p);
                // `contains` is inclusive of the boundary, `depth` is strict; a
                // point at dist == r (e.g. (2,1,0)-like) is the only difference
                if d > Fix128::ZERO {
                    assert!(inside, "{p:?}");
                    n_inside += 1;
                }
                // reference ball in f64
                let (x, y, z) = (
                    f64::from(ix as i32) * 0.7,
                    f64::from(iy as i32) * 0.7 - 1.0,
                    f64::from(iz as i32) * 0.7,
                );
                let in_ref = x * x + y * y + z * z < 4.0 - 1e-9;
                let on_shell = (x * x + y * y + z * z - 4.0).abs() <= 1e-9;
                if !on_shell {
                    assert_eq!(d > Fix128::ZERO, in_ref, "lattice point {ix},{iy},{iz}");
                }
            }
        }
    }
    assert!(n_inside > 50, "lattice must exercise the interior");
}

#[test]
fn depth_equals_signed_depth_where_both_are_defined_for_aabb() {
    let s = pool();
    for iy in -2..=2 {
        let p = Vec3Fix::from_int(0, iy, 0);
        assert_eq!(
            Some(s.depth_below_surface(p)),
            s.signed_depth_below_surface(p),
            "y = {iy}"
        );
    }
}

// ------------------------------------------------------------ water_pool

#[test]
fn water_pool_defaults_match_the_documented_water_constants() {
    let z = BuoyancyZone::water_pool(pool());
    assert!(rel(z.density_kg_m3.to_f64(), 1000.0) < 1e-15);
    assert!(rel(z.gravity.to_f64(), 9.81) < 1e-15);
    assert!(rel(z.drag_linear.to_f64(), 0.5) < 1e-15);
    assert!(rel(z.drag_quadratic.to_f64(), 1.0) < 1e-15);
    // the supplied shape is stored, not replaced
    assert_eq!(
        z.shape.depth_below_surface(Vec3Fix::from_int(0, 0, 0)),
        Fix128::from_int(2)
    );
}

// ------------------------------------------------------ cap fraction

#[test]
fn submerged_fraction_matches_cap_closed_form_and_slab_integral() {
    let z = BuoyancyZone::water_pool(ZoneShape::Aabb {
        min: Vec3Fix::from_int(-5, -5, -5),
        max: Vec3Fix::from_int(5, 5, 5),
    });
    let r = 1.0_f64;
    // centre y relative to surface y = 5: depth d = 5 - y
    for k in -20..=20 {
        let d = f64::from(k) * 0.1; // -2.0 .. 2.0
        let y = 5.0 - d;
        let pos = Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(y), Fix128::ZERO);
        let got = z.submerged_fraction(pos, Fix128::ONE).to_f64();
        let h = (d + r).clamp(0.0, 2.0 * r);
        let closed = h * h * (3.0 * r - h) / (4.0 * r * r * r);
        assert!(
            (got - closed).abs() < 1e-12,
            "d = {d}: got {got}, closed form {closed}"
        );
        // independent: integrate disc areas of the part of the ball below z = h - r
        let n = 20_000;
        let top = h - r; // submerged slab is z in [-r, top]
        let dz = (top + r) / f64::from(n);
        let mut vol = 0.0;
        for i in 0..n {
            let zc = -r + (f64::from(i) + 0.5) * dz;
            vol += std::f64::consts::PI * (r * r - zc * zc) * dz;
        }
        let frac = vol / (4.0 / 3.0 * std::f64::consts::PI * r * r * r);
        assert!(
            (got - frac).abs() < 1e-6,
            "d = {d}: got {got}, slab integral {frac}"
        );
    }
}

#[test]
fn submerged_fraction_is_exactly_half_at_the_surface_and_continuous() {
    let z = BuoyancyZone::water_pool(pool());
    // centre exactly at the surface y = 2: half the sphere is wet
    let half = z.submerged_fraction(Vec3Fix::from_int(0, 2, 0), Fix128::ONE);
    assert_eq!(half, Fix128::from_ratio(1, 2));
    // r above the surface: 0, r below: 1
    assert_eq!(
        z.submerged_fraction(Vec3Fix::from_int(0, 3, 0), Fix128::ONE),
        Fix128::ZERO
    );
    assert_eq!(
        z.submerged_fraction(Vec3Fix::from_int(0, 1, 0), Fix128::ONE),
        Fix128::ONE
    );
    // monotone non-increasing in height
    let mut last = Fix128::ONE;
    for k in 0..=40 {
        let y = Fix128::from_ratio(k, 10); // 0 .. 4
        let f = z.submerged_fraction(Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO), Fix128::ONE);
        assert!(f <= last, "fraction rose with height at y = {k}/10");
        last = f;
    }
}

#[test]
fn centre_outside_footprint_or_below_floor_is_dry() {
    let z = BuoyancyZone::water_pool(pool());
    for p in [
        Vec3Fix::from_int(5, 0, 0),
        Vec3Fix::from_int(0, 0, -5),
        Vec3Fix::from_int(0, -3, 0),
    ] {
        assert_eq!(z.submerged_fraction(p, Fix128::ONE), Fix128::ZERO, "{p:?}");
    }
}

// ----------------------------------------------------------- force_on

#[test]
fn buoyancy_is_archimedes_with_full_precision_four_thirds_pi() {
    // fully submerged r = 1/2 at rest: F_y = 1000 * (4/3 pi (1/2)^3) * 9.81
    let z = BuoyancyZone::water_pool(ZoneShape::Aabb {
        min: Vec3Fix::from_int(-5, -5, -5),
        max: Vec3Fix::from_int(5, 5, 5),
    });
    let body = body_at(0, -4, 0, Vec3Fix::ZERO);
    let f = z.force_on(&body, Fix128::from_ratio(1, 2));
    let expect = 1000.0 * (4.0 / 3.0 * std::f64::consts::PI * 0.125) * 9.81;
    assert_eq!(f.x, Fix128::ZERO);
    assert_eq!(f.z, Fix128::ZERO);
    // 4.1888 vs 4.18879020...: 2.3e-6 relative error before the constant was fixed
    assert!(
        rel(f.y.to_f64(), expect) < 1e-12,
        "F_y = {} expected {expect}",
        f.y.to_f64()
    );
}

#[test]
fn drag_is_linear_plus_quadratic_and_opposes_velocity() {
    // v = (3, 4, 0), |v| = 5 exactly. F_drag = -(c_lin + c_quad |v|) v = -(0.5 + 5) v.
    let z = BuoyancyZone::water_pool(ZoneShape::Aabb {
        min: Vec3Fix::from_int(-5, -5, -5),
        max: Vec3Fix::from_int(5, 5, 5),
    });
    // centre 10 above the floor... place the body at the surface so buoyancy is
    // known: centre at y=5 (surface) => fraction 1/2
    let body = body_at(0, 5, 0, Vec3Fix::from_int(3, 4, 0));
    let f = z.force_on(&body, Fix128::ONE);
    let buoy = 1000.0 * (4.0 / 3.0 * std::f64::consts::PI) * 0.5 * 9.81;
    assert!(rel(f.x.to_f64(), -5.5 * 3.0) < 1e-12, "{}", f.x.to_f64());
    assert!(
        rel(f.y.to_f64(), buoy - 5.5 * 4.0) < 1e-12,
        "{} vs {}",
        f.y.to_f64(),
        buoy - 22.0
    );
    assert_eq!(f.z, Fix128::ZERO);
}

#[test]
fn quadratic_drag_scales_with_speed_squared() {
    // c_lin = 0 isolates the quadratic term
    let mut z = BuoyancyZone::water_pool(ZoneShape::Aabb {
        min: Vec3Fix::from_int(-50, -50, -50),
        max: Vec3Fix::from_int(50, 50, 50),
    });
    z.drag_linear = Fix128::ZERO;
    let one = z.force_on(&body_at(0, -40, 0, Vec3Fix::from_int(2, 0, 0)), Fix128::ONE);
    let two = z.force_on(&body_at(0, -40, 0, Vec3Fix::from_int(4, 0, 0)), Fix128::ONE);
    // -c_quad * v * |v|: 4 vs 16
    assert!(rel(one.x.to_f64(), -4.0) < 1e-12);
    assert!(rel(two.x.to_f64(), -16.0) < 1e-12);
    // and a zero quadratic coefficient with a non-zero linear one: -c_lin v
    z.drag_quadratic = Fix128::ZERO;
    z.drag_linear = Fix128::from_int(3);
    let lin = z.force_on(
        &body_at(0, -40, 0, Vec3Fix::from_int(0, 0, -2)),
        Fix128::ONE,
    );
    assert!(rel(lin.z.to_f64(), 6.0) < 1e-12, "{}", lin.z.to_f64());
}

#[test]
fn dry_body_gets_zero_force_even_when_moving() {
    let z = BuoyancyZone::water_pool(pool());
    let f = z.force_on(
        &body_at(0, 10, 0, Vec3Fix::from_int(9, 9, 9)),
        Fix128::from_ratio(1, 2),
    );
    assert_eq!(f, Vec3Fix::ZERO);
}

#[test]
fn light_body_settles_at_the_depth_archimedes_predicts() {
    // sphere r = 1/2 with half the density of water floats with its centre exactly
    // at the surface (cap fraction 1/2 by symmetry). Net force there is 0 at rest.
    let z = BuoyancyZone::water_pool(ZoneShape::Aabb {
        min: Vec3Fix::from_int(-5, -5, -5),
        max: Vec3Fix::from_int(5, 5, 5),
    });
    let r = 0.5_f64;
    let vol = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
    let weight = 500.0 * vol * 9.81;
    let f = z.force_on(&body_at(0, 5, 0, Vec3Fix::ZERO), Fix128::from_ratio(1, 2));
    assert!(rel(f.y.to_f64(), weight) < 1e-12);
}

// -------------------------------------------------------- degenerate

#[test]
fn non_positive_radius_gives_no_buoyancy() {
    let z = BuoyancyZone::water_pool(pool());
    let body = body_at(0, 0, 0, Vec3Fix::ZERO);
    assert_eq!(
        z.submerged_fraction(body.position, Fix128::ZERO),
        Fix128::ZERO
    );
    assert_eq!(
        z.submerged_fraction(body.position, Fix128::from_int(-1)),
        Fix128::ZERO
    );
    assert_eq!(z.force_on(&body, Fix128::ZERO), Vec3Fix::ZERO);
    assert_eq!(z.force_on(&body, Fix128::from_int(-1)), Vec3Fix::ZERO);
}

#[test]
fn inverted_box_is_empty_and_nothing_panics() {
    // min > max on every axis: no point is inside
    let s = ZoneShape::Aabb {
        min: Vec3Fix::from_int(1, 1, 1),
        max: Vec3Fix::from_int(-1, -1, -1),
    };
    let r = catch_unwind(AssertUnwindSafe(|| {
        (
            s.depth_below_surface(Vec3Fix::ZERO),
            BuoyancyZone::water_pool(s).force_on(&body_at(0, 0, 0, Vec3Fix::ZERO), Fix128::ONE),
        )
    }));
    let (d, f) = r.expect("inverted box must not panic");
    assert_eq!(d, Fix128::ZERO);
    assert_eq!(f, Vec3Fix::ZERO);
}

#[test]
fn zero_radius_sphere_zone_is_empty() {
    let s = ZoneShape::Sphere {
        centre: Vec3Fix::ZERO,
        radius: Fix128::ZERO,
    };
    assert_eq!(s.depth_below_surface(Vec3Fix::ZERO), Fix128::ZERO);
}

// ------------------------------------------------- boundary / sphere zone

#[test]
fn aabb_side_faces_are_inclusive_for_depth() {
    // x = min.x, x = max.x, z = min.z, z = max.z are all inside (documented
    // inclusive bounds), so the depth there is the plain top - y.
    let s = pool();
    for p in [
        Vec3Fix::from_int(-1, 0, 0),
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::from_int(0, 0, -1),
        Vec3Fix::from_int(0, 0, 1),
    ] {
        assert_eq!(s.depth_below_surface(p), Fix128::from_int(2), "{p:?}");
    }
}

#[test]
fn sphere_zone_buoyancy_follows_the_flat_top_surface_model() {
    // Sphere zone: centre (0,1,0), r = 2, flat surface at y = 3 (module model:
    // "upper surface" = centre.y + radius). Body radius 1/2.
    let z = BuoyancyZone::water_pool(ball());
    let r = Fix128::from_ratio(1, 2);
    // deep inside the footprint, 2 below the top: h = 2.5 >= 2r => fully wet
    assert_eq!(
        z.submerged_fraction(Vec3Fix::from_int(0, 1, 0), r),
        Fix128::ONE
    );
    // centre exactly at the surface: half
    assert_eq!(
        z.submerged_fraction(Vec3Fix::from_int(0, 3, 0), r),
        Fix128::from_ratio(1, 2)
    );
    // on the horizontal footprint edge (dx^2 + dz^2 == r^2): not in the fluid
    assert_eq!(
        z.submerged_fraction(Vec3Fix::from_int(2, 1, 0), r),
        Fix128::ZERO
    );
    // below the ball's floor y = centre - r = -1
    assert_eq!(
        z.submerged_fraction(Vec3Fix::from_int(0, -2, 0), r),
        Fix128::ZERO
    );
    // outside the footprint
    assert_eq!(
        z.submerged_fraction(Vec3Fix::from_int(3, 1, 0), r),
        Fix128::ZERO
    );
    // and the force is Archimedes on the wet part
    let f = z.force_on(&body_at(0, 1, 0, Vec3Fix::ZERO), r);
    let expect = 1000.0 * (4.0 / 3.0 * std::f64::consts::PI * 0.125) * 9.81;
    assert!(rel(f.y.to_f64(), expect) < 1e-12);
}

/// Known inconsistency, pinned so a future change is a deliberate decision:
/// for a sphere zone `depth_below_surface` tests membership in the *ball*,
/// `signed_depth_below_surface` (used by `submerged_fraction` / `force_on`) tests
/// the *horizontal footprint*. A point above the ball's curved top but inside
/// its horizontal extent is "dry" for the first and "wet" for the second.
#[test]
fn sphere_zone_depth_functions_disagree_above_the_curved_top() {
    let s = ball();
    let p = Vec3Fix::from_int(1, 3, 1); // outside the ball, inside the footprint
    assert!(!s.contains(p));
    assert_eq!(s.depth_below_surface(p), Fix128::ZERO);
    assert!(s.signed_depth_below_surface(p).is_some());
}
