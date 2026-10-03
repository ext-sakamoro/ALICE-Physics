//! The convex colliders a body can carry and the narrow-phase between them.
//!
//! A body's collider is either one convex [`Shape`] or a [`CompoundShape`] — a
//! union of convex children, which is **not** convex and so cannot go through GJK as
//! one piece (its support function is the support of its convex hull, which fills
//! the gaps between children). A collider is therefore broken into *pieces*, each
//! convex, and two colliders are in contact when any pair of their pieces is; the
//! contact reported is the deepest pair's.
//!
//! Author: Moroya Sakamoto

use crate::collider::{contact, gjk, Contact, Support, AABB};
use crate::compound::{CompoundChild, CompoundShape};
use crate::math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};
use crate::shape::{PosedShape, Shape};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// What a body collides as, beyond its collision sphere.
#[derive(Clone, Debug)]
pub(crate) enum BodyCollider {
    /// One convex solid.
    Shape(Shape),
    /// A union of convex children, in the body's frame with its centre of mass at the
    /// origin.
    Compound(CompoundShape),
}

/// One convex piece of a collider, placed in the world.
enum Piece<'a> {
    Shape(PosedShape),
    Child {
        child: &'a CompoundChild,
        position: Vec3Fix,
        rotation: QuatFix,
    },
}

impl Support for Piece<'_> {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        match self {
            Self::Shape(posed) => posed.support(direction),
            Self::Child {
                child,
                position,
                rotation,
            } => child.support_world(direction, *position, *rotation),
        }
    }
}

impl BodyCollider {
    /// The convex pieces of this collider for a body at `position` turned by
    /// `rotation`, each with a box that contains it.
    fn pieces(&self, position: Vec3Fix, rotation: QuatFix) -> Vec<(Piece<'_>, AABB)> {
        match self {
            Self::Shape(shape) => {
                let r = shape.bounding_radius();
                let half = Vec3Fix::new(r, r, r);
                vec![(
                    Piece::Shape(PosedShape {
                        shape: *shape,
                        position,
                        rotation,
                    }),
                    AABB::from_center_half(position, half),
                )]
            }
            Self::Compound(compound) => compound
                .children
                .iter()
                .enumerate()
                .map(|(i, child)| {
                    (
                        Piece::Child {
                            child,
                            position,
                            rotation,
                        },
                        compound.child_world_aabb(i, position, rotation),
                    )
                })
                .collect(),
        }
    }

    /// Radius of a sphere about the body's origin (its centre of mass) that contains
    /// the whole collider.
    pub(crate) fn bounding_radius(&self) -> Fix128 {
        match self {
            Self::Shape(shape) => shape.bounding_radius(),
            Self::Compound(compound) => {
                // The corners of each child's box in the body's own frame bound it;
                // the farthest corner from the origin bounds the union.
                let mut radius = Fix128::ZERO;
                for i in 0..compound.children.len() {
                    let b = compound.child_world_aabb(i, Vec3Fix::ZERO, QuatFix::IDENTITY);
                    for &x in &[b.min.x, b.max.x] {
                        for &y in &[b.min.y, b.max.y] {
                            for &z in &[b.min.z, b.max.z] {
                                let d = Vec3Fix::new(x, y, z).length();
                                if d > radius {
                                    radius = d;
                                }
                            }
                        }
                    }
                }
                radius
            }
        }
    }
}

/// The deepest contact between two colliders, or `None`: GJK/EPA on every pair of
/// pieces whose boxes meet. The normal points from `b` to `a`.
pub(crate) fn contact_between(
    a: &BodyCollider,
    (position_a, rotation_a): (Vec3Fix, QuatFix),
    b: &BodyCollider,
    (position_b, rotation_b): (Vec3Fix, QuatFix),
) -> Option<Contact> {
    let pieces_a = a.pieces(position_a, rotation_a);
    let pieces_b = b.pieces(position_b, rotation_b);
    let mut deepest: Option<Contact> = None;
    for (pa, box_a) in &pieces_a {
        for (pb, box_b) in &pieces_b {
            if !box_a.intersects(box_b) {
                continue;
            }
            if let Some(hit) = contact(pa, pb) {
                if deepest.is_none_or(|d| hit.depth > d.depth) {
                    deepest = Some(hit);
                }
            }
        }
    }
    deepest
}

/// Whether any piece of `a` meets any piece of `b` (GJK; touching counts).
pub(crate) fn colliders_meet(
    a: &BodyCollider,
    (position_a, rotation_a): (Vec3Fix, QuatFix),
    b: &BodyCollider,
    (position_b, rotation_b): (Vec3Fix, QuatFix),
) -> bool {
    let pieces_a = a.pieces(position_a, rotation_a);
    let pieces_b = b.pieces(position_b, rotation_b);
    pieces_a.iter().any(|(pa, box_a)| {
        pieces_b
            .iter()
            .any(|(pb, box_b)| box_a.intersects(box_b) && gjk(pa, pb).colliding)
    })
}

/// The unit quaternion of a rotation matrix (Shepperd's method: the largest of the
/// four candidates is divided by, so no division by a small number).
pub(crate) fn quat_from_rotation(m: Mat3Fix) -> QuatFix {
    let (r00, r10, r20) = (m.col0.x, m.col0.y, m.col0.z);
    let (r01, r11, r21) = (m.col1.x, m.col1.y, m.col1.z);
    let (r02, r12, r22) = (m.col2.x, m.col2.y, m.col2.z);
    let one = Fix128::ONE;
    let trace = r00 + r11 + r22;
    let q = if trace > Fix128::ZERO {
        let s = (trace + one).sqrt().double();
        QuatFix::new(
            (r21 - r12) / s,
            (r02 - r20) / s,
            (r10 - r01) / s,
            s / Fix128::from_int(4),
        )
    } else if r00 >= r11 && r00 >= r22 {
        let s = (one + r00 - r11 - r22).sqrt().double();
        QuatFix::new(
            s / Fix128::from_int(4),
            (r01 + r10) / s,
            (r02 + r20) / s,
            (r21 - r12) / s,
        )
    } else if r11 >= r22 {
        let s = (one + r11 - r00 - r22).sqrt().double();
        QuatFix::new(
            (r01 + r10) / s,
            s / Fix128::from_int(4),
            (r12 + r21) / s,
            (r02 - r20) / s,
        )
    } else {
        let s = (one + r22 - r00 - r11).sqrt().double();
        QuatFix::new(
            (r02 + r20) / s,
            (r12 + r21) / s,
            s / Fix128::from_int(4),
            (r10 - r01) / s,
        )
    };
    q.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(q: QuatFix) -> Mat3Fix {
        Mat3Fix::from_cols(
            q.rotate_vec(Vec3Fix::UNIT_X),
            q.rotate_vec(Vec3Fix::UNIT_Y),
            q.rotate_vec(Vec3Fix::UNIT_Z),
        )
    }

    fn same_rotation(a: QuatFix, b: QuatFix) -> bool {
        // Two unit quaternions are the same rotation up to overall sign.
        let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        (dot.abs() - Fix128::ONE).abs() < Fix128::from_ratio(1, 1_000_000_000)
    }

    /// A matrix built from a quaternion returns the same rotation, on each of
    /// Shepperd's four branches: a small turn (trace branch) and half turns about
    /// each axis (the three largest-diagonal branches).
    #[test]
    fn a_rotation_matrix_round_trips_through_the_quaternion_on_every_branch() {
        let axes = [
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_Y,
            Vec3Fix::UNIT_Z,
            Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ZERO),
            Vec3Fix::new(Fix128::ONE, -Fix128::ONE, Fix128::from_int(2)),
        ];
        let angles = [
            Fix128::from_ratio(3, 10),
            Fix128::PI,
            Fix128::HALF_PI,
            Fix128::from_ratio(5, 2),
        ];
        for axis in axes {
            for angle in angles {
                let q = QuatFix::from_axis_angle(axis, angle);
                let back = quat_from_rotation(matrix(q));
                assert!(
                    same_rotation(q, back),
                    "axis {axis:?} angle {angle:?}: {q:?} vs {back:?}"
                );
            }
        }
        assert!(same_rotation(
            quat_from_rotation(Mat3Fix::IDENTITY),
            QuatFix::IDENTITY
        ));
    }
}
