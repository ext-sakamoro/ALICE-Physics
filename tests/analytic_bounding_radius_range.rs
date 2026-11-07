//! Oracle: a shape's bounding radius is right for huge shapes and is flagged when
//! it cannot be represented.
//!
//! # What is measured
//!
//! `Shape::bounding_radius` (the collision radius `set_body_shape` stores, which
//! the broad-phase, the sleep tree and the CCD read) was `sqrt(a² + b² + c²)` in
//! `Fix128`, whose squares wrap from `2³¹·⁵`: a box of half extents `2⁴⁰` got a
//! wrapped, small radius and the world then treated the body as tiny. The
//! lengths are now formed without squaring out of the range, so every radius
//! that fits is exact, and a radius that does not fit (dimensions near `2⁶³`)
//! saturates and raises the world's overflow flag.
//!
//! Every expected value is an exact integer from a Pythagorean triple scaled by
//! `2⁴⁰` (`3, 4, 5` / `24, 7, 25` / `12, 4, 6, 14`), so the reference needs no
//! rounding. Inside the range the bits are unchanged: `tests/analytic_shape.rs`
//! and the determinism goldens pin them.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::collider::Sphere;
use alice_physics_core::compound::CompoundShape;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::shape::{Shape, ShapeError};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

const M: i64 = 1 << 40;

fn fi(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    })
}

/// (shape, exact bounding radius): each from a triple, each beyond `2³¹·⁵` in its
/// squares.
fn huge_shapes() -> Vec<(&'static str, Shape, i64)> {
    vec![
        (
            "box 3-4-0 → 5",
            Shape::Box {
                half_extents: Vec3Fix::from_int(3 * M, 4 * M, 0),
            },
            5 * M,
        ),
        (
            "cylinder r 3, h 4 → 5",
            Shape::Cylinder {
                radius: fi(3 * M),
                half_height: fi(4 * M),
            },
            5 * M,
        ),
        (
            // quarter = 7m, rim = 25m, apex = 21m < rim
            "cone r 24, half height 14 → rim 25",
            Shape::Cone {
                radius: fi(24 * M),
                half_height: fi(14 * M),
            },
            25 * M,
        ),
        (
            // hw 12, third 4, hd 6: corner 14, apex sqrt(8² + 6²) = 10 < corner
            "wedge w 24, h 12, d 12 → corner 14",
            Shape::Wedge {
                width: fi(24 * M),
                height: fi(12 * M),
                depth: fi(12 * M),
            },
            14 * M,
        ),
        (
            "ellipsoid radii 2, 5, 3 → 5",
            Shape::Ellipsoid {
                radii: Vec3Fix::from_int(2 * M, 5 * M, 3 * M),
            },
            5 * M,
        ),
        (
            "torus 3 + 1 → 4",
            Shape::Torus {
                major_radius: fi(3 * M),
                minor_radius: fi(M),
            },
            4 * M,
        ),
    ]
}

#[test]
fn a_huge_shape_has_its_exact_bounding_radius() {
    for (name, shape, want) in huge_shapes() {
        assert_eq!(shape.bounding_radius(), fi(want), "{name}: bounding_radius");
        assert_eq!(
            shape.checked_bounding_radius(),
            Some(fi(want)),
            "{name}: checked_bounding_radius"
        );
    }
}

/// The world stores that radius: a ray from far along +x meets the body's
/// collision sphere at exactly `x − radius` (`raycast` tests bounding spheres).
#[test]
fn the_world_stores_the_exact_radius_of_a_huge_shape() {
    for (name, shape, want) in huge_shapes() {
        let mut w = world();
        let i = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        assert!(w.set_body_shape(i, &shape), "{name}");
        assert!(
            !w.overflow_detected(),
            "{name}: a radius that fits raises no flag"
        );
        let hit = w.raycast(
            Vec3Fix::from_int(64 * M, 0, 0),
            Vec3Fix::from_int(-1, 0, 0),
            fi(128 * M),
        );
        assert_eq!(
            hit,
            Some((i, fi(64 * M - want))),
            "{name}: raycast distance"
        );
    }
}

/// A compound beyond the range is refused before its radius matters: the mass
/// check (`mass · r² < 2⁶¹`, the largest quantity the solve forms) rejects two
/// spheres at `±(3, 4, 0)·2⁴⁰` however light they are, and accepts the same
/// pair at `2²⁰` (where the radius `5·2²⁰` is in range and exact).
#[test]
fn a_compound_beyond_the_range_is_refused_and_one_inside_has_its_exact_radius() {
    let pair = |m: i64| {
        let mut c = CompoundShape::new();
        for s in [1i64, -1] {
            c.add_sphere(
                Sphere {
                    center: Vec3Fix::from_int(s * 3 * m, s * 4 * m, 0),
                    radius: Fix128::ONE,
                },
                Vec3Fix::ZERO,
                QuatFix::IDENTITY,
            );
        }
        c
    };
    let mut w = world();
    assert!(matches!(
        w.add_compound_body(&pair(M), Fix128::from_raw(0, 1 << 44), Vec3Fix::ZERO),
        Err(ShapeError::MassNotRepresentable)
    ));
    let m = 1i64 << 20;
    let i = w
        .add_compound_body(&pair(m), Fix128::ONE, Vec3Fix::ZERO)
        .expect("a compound inside the range");
    // the farthest box corner: (3m + 1, 4m + 1, 1), within 2 of 5m
    let hit = w
        .raycast(
            Vec3Fix::from_int(64 * m, 0, 0),
            Vec3Fix::from_int(-1, 0, 0),
            fi(128 * m),
        )
        .expect("the compound's sphere is hit");
    assert_eq!(hit.0, i);
    let err = (hit.1 - fi(64 * m - 5 * m)).abs();
    assert!(err <= fi(2), "radius off by {err:?}");
}

/// A radius past the range (`1.5 · 2⁶² · √3 > 2⁶³`): `set_body_shape` keeps the body, stores the end of
/// the range and raises the flag; `add_shaped_body` and `add_compound_body`
/// refuse the shape as not representable.
#[test]
fn a_radius_that_does_not_fit_is_flagged_or_refused() {
    let huge = Shape::Box {
        half_extents: Vec3Fix::from_int(3 << 61, 3 << 61, 3 << 61),
    };
    assert_eq!(huge.checked_bounding_radius(), None);
    assert_eq!(
        huge.bounding_radius(),
        Fix128 {
            hi: i64::MAX,
            lo: u64::MAX
        },
        "saturates at the end of the range"
    );
    let mut w = world();
    let i = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    assert!(!w.overflow_detected());
    assert!(w.set_body_shape(i, &huge));
    assert!(w.overflow_detected(), "set_body_shape raises the flag");
    let mut w = world();
    assert!(matches!(
        w.add_shaped_body(&huge, Fix128::ONE, Vec3Fix::ZERO),
        Err(ShapeError::MassNotRepresentable)
    ));
}

/// Inside the range the checked radius is the radius, bit for bit.
#[test]
fn inside_the_range_the_checked_radius_is_the_radius() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        Fix128::from_raw(((seed >> 40) as i64) + 1, seed << 20)
    };
    for _ in 0..2000 {
        let (a, b, c) = (next(), next(), next());
        let shapes = [
            Shape::Box {
                half_extents: Vec3Fix::new(a, b, c),
            },
            Shape::Cylinder {
                radius: a,
                half_height: b,
            },
            Shape::Cone {
                radius: a,
                half_height: b,
            },
            Shape::Ellipsoid {
                radii: Vec3Fix::new(a, b, c),
            },
            Shape::Wedge {
                width: a,
                height: b,
                depth: c,
            },
            Shape::Torus {
                major_radius: a,
                minor_radius: b,
            },
        ];
        for s in shapes {
            assert_eq!(
                s.checked_bounding_radius(),
                Some(s.bounding_radius()),
                "{s:?}"
            );
        }
    }
}
