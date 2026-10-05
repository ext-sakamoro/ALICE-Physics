//! `PosedShape::support` builds a turned cylinder, ellipsoid and torus with
//! their `with_rotation` constructors. These oracles pin that the posed solid
//! is the shape built by `new` with `rotation` assigned afterwards, to the bit,
//! and that the rotation is not dropped: the turned support differs from the
//! unturned one.
//!
//! The rotation is `(1/2, 1/2, 1/2, 1/2)`, the 120 degree turn about `(1, 1, 1)`
//! that cycles the axes. Every component is exact in `Fix128`, and none of the
//! three solids is symmetric under it, so a dropped rotation shows.

use alice_physics::collider::Support;
use alice_physics::cylinder::Cylinder;
use alice_physics::ellipsoid::Ellipsoid;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::{PosedShape, Shape};
use alice_physics::torus::Torus;

fn half() -> Fix128 {
    Fix128::from_ratio(1, 2)
}

fn cycle_axes() -> QuatFix {
    QuatFix::new(half(), half(), half(), half())
}

fn position() -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_int(3),
        Fix128::from_int(-1),
        Fix128::from_ratio(5, 4),
    )
}

/// Axis directions, diagonals and a few off-axis directions with small
/// integer components.
fn directions() -> Vec<Vec3Fix> {
    let mut out = Vec::new();
    for x in -2..=2i64 {
        for y in -2..=2i64 {
            for z in -2..=2i64 {
                if x != 0 || y != 0 || z != 0 {
                    out.push(Vec3Fix::new(
                        Fix128::from_int(x),
                        Fix128::from_int(y),
                        Fix128::from_int(z),
                    ));
                }
            }
        }
    }
    out
}

/// The posed support equals `reference` for every direction and differs from
/// the unturned posed support for at least one.
fn check(shape: Shape, reference: &dyn Fn(Vec3Fix) -> Vec3Fix, name: &str) {
    let turned = PosedShape {
        shape,
        position: position(),
        rotation: cycle_axes(),
    };
    let unturned = PosedShape {
        rotation: QuatFix::IDENTITY,
        ..turned
    };
    let mut differs = 0usize;
    for d in directions() {
        let got = turned.support(d);
        assert_eq!(got, reference(d), "{name}: direction {d:?}");
        differs += usize::from(got != unturned.support(d));
    }
    assert!(differs > 0, "{name}: the rotation changed no support point");
}

#[test]
fn posed_cylinder_is_new_with_rotation_assigned() {
    let (radius, half_height) = (Fix128::from_ratio(1, 2), Fix128::from_int(2));
    check(
        Shape::Cylinder {
            radius,
            half_height,
        },
        &|d| {
            let mut c = Cylinder::new(position(), half_height, radius);
            c.rotation = cycle_axes();
            c.support(d)
        },
        "cylinder",
    );
}

#[test]
fn posed_ellipsoid_is_new_with_rotation_assigned() {
    let radii = Vec3Fix::new(Fix128::ONE, Fix128::from_int(2), Fix128::from_int(3));
    check(
        Shape::Ellipsoid { radii },
        &|d| {
            let mut e = Ellipsoid::new(position(), radii);
            e.rotation = cycle_axes();
            e.support(d)
        },
        "ellipsoid",
    );
}

#[test]
fn posed_torus_is_new_with_rotation_assigned() {
    let (major, minor) = (Fix128::from_int(2), Fix128::from_ratio(1, 2));
    check(
        Shape::Torus {
            major_radius: major,
            minor_radius: minor,
        },
        &|d| {
            let mut t = Torus::new(position(), major, minor);
            t.rotation = cycle_axes();
            t.support(d)
        },
        "torus",
    );
}
