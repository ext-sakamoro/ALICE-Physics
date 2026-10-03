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

use crate::box_collider::OrientedBox;
use crate::collider::{contact, gjk, Contact, Support, AABB};
use crate::compound::{CompoundChild, CompoundShape, ShapeRef};
use crate::math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};
#[cfg(feature = "std")]
use crate::sdf_collider::{
    box_sample_points, collide_aabb_sdf, collide_capsule_sdf, collide_point_sdf,
    collide_points_sdf, collide_sphere_sdf, SdfCollider,
};
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
    /// A box that contains the whole collider, for a body at `position` turned by
    /// `rotation`.
    fn world_aabb(&self, position: Vec3Fix, rotation: QuatFix) -> AABB {
        match self {
            Self::Shape(shape) => {
                let r = shape.bounding_radius();
                AABB::from_center_half(position, Vec3Fix::new(r, r, r))
            }
            Self::Compound(compound) => compound.world_aabb(position, rotation),
        }
    }

    /// The convex pieces of this collider for a body at `position` turned by
    /// `rotation`, each with a box that contains it. Only the pieces whose box meets
    /// `against` are returned: a piece outside it cannot touch whatever `against`
    /// bounds.
    fn pieces(
        &self,
        position: Vec3Fix,
        rotation: QuatFix,
        against: &AABB,
    ) -> Vec<(Piece<'_>, AABB)> {
        match self {
            Self::Shape(shape) => vec![(
                Piece::Shape(PosedShape {
                    shape: *shape,
                    position,
                    rotation,
                }),
                self.world_aabb(position, rotation),
            )],
            Self::Compound(compound) => compound
                .overlapping_children(against, position, rotation)
                .into_iter()
                .map(|i| {
                    (
                        Piece::Child {
                            child: &compound.children[i],
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

/// How many times the support point is chased along the SDF's gradient.
#[cfg(feature = "std")]
const SDF_SUPPORT_ITERATIONS: usize = 6;

/// Keeps the deeper of two optional contacts.
#[cfg(feature = "std")]
fn deeper(best: &mut Option<Contact>, candidate: Option<Contact>) {
    if let Some(c) = candidate {
        if best.is_none_or(|b| c.depth > b.depth) {
            *best = Some(c);
        }
    }
}

/// A convex solid against an SDF by its support function: the point of the solid
/// furthest *into* the field is found by repeatedly taking the support point
/// against the field's outward normal (at the centre first, then at each support
/// point found). That is exact for a flat field (the normal does not change, so
/// the first support point is the deepest) and follows the surface for a curved
/// one, where it finds a point on the solid that is locally deepest.
#[cfg(feature = "std")]
fn convex_sdf_contact(solid: &impl Support, centre: Vec3Fix, sdf: &SdfCollider) -> Option<Contact> {
    let outward = |at: Vec3Fix| {
        let (lx, ly, lz) = sdf.world_to_local(at);
        let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
        sdf.local_normal_to_world(nx, ny, nz)
    };
    let mut at = centre;
    let mut best = None;
    for _ in 0..SDF_SUPPORT_ITERATIONS {
        let point = solid.support(-outward(at));
        deeper(&mut best, collide_point_sdf(point, sdf));
        if point == at {
            break;
        }
        at = point;
    }
    best
}

/// A box against an SDF: the deepest of the 27 points that
/// [`collide_aabb_sdf`] samples (an axis-aligned box goes through it), turned with
/// the box. Exact for a flat field, whose deepest point of a box is a corner.
#[cfg(feature = "std")]
fn box_sdf_contact(b: &OrientedBox, sdf: &SdfCollider) -> Option<Contact> {
    if b.rotation == QuatFix::IDENTITY {
        let aabb = b.aabb();
        return collide_aabb_sdf(aabb.min, aabb.max, sdf);
    }
    collide_points_sdf(
        &box_sample_points(b.center, b.half_extents, b.rotation),
        sdf,
    )
}

/// One shape, placed in the world, against an SDF.
#[cfg(feature = "std")]
fn posed_sdf_contact(posed: &PosedShape, sdf: &SdfCollider) -> Option<Contact> {
    match posed.shape {
        Shape::Box { half_extents } => box_sdf_contact(
            &OrientedBox::new(posed.position, half_extents, posed.rotation),
            sdf,
        ),
        Shape::Ellipsoid { radii } if radii.x == radii.y && radii.y == radii.z => {
            collide_sphere_sdf(posed.position, radii.x, sdf)
        }
        _ => convex_sdf_contact(posed, posed.position, sdf),
    }
}

/// One child of a compound, placed in the world by its body's pose, against an SDF.
#[cfg(feature = "std")]
fn child_sdf_contact(
    child: &CompoundChild,
    body_position: Vec3Fix,
    body_rotation: QuatFix,
    sdf: &SdfCollider,
) -> Option<Contact> {
    let position = body_position + body_rotation.rotate_vec(child.local_position);
    let rotation = body_rotation.mul(child.local_rotation);
    match &child.shape {
        ShapeRef::Sphere(s) => {
            collide_sphere_sdf(position + rotation.rotate_vec(s.center), s.radius, sdf)
        }
        ShapeRef::Capsule(c) => collide_capsule_sdf(
            position + rotation.rotate_vec(c.a),
            position + rotation.rotate_vec(c.b),
            c.radius,
            sdf,
        ),
        ShapeRef::Box(b) => box_sdf_contact(
            &OrientedBox::new(
                position + rotation.rotate_vec(b.center),
                b.half_extents,
                rotation.mul(b.rotation),
            ),
            sdf,
        ),
        ShapeRef::ConvexHull(hull) => {
            let world: Vec<Vec3Fix> = hull
                .vertices
                .iter()
                .map(|&v| position + rotation.rotate_vec(v))
                .collect();
            collide_points_sdf(&world, sdf)
        }
    }
}

#[cfg(feature = "std")]
impl BodyCollider {
    /// The deepest contact of this collider, for a body at `position` turned by
    /// `rotation`, with an SDF, or `None` when no piece reaches into it. The normal
    /// pushes the collider out of the field; `depth` is how far.
    ///
    /// A sphere, an ellipsoid with equal radii and a capsule child use their own
    /// sphere / capsule tests, a box its corners and centre, a hull its vertices,
    /// and any other convex shape its support point against the field's normal
    /// ([`convex_sdf_contact`]). Against a flat field every one of these is exact;
    /// against a curved one a face or an edge can reach deeper than the points
    /// sampled.
    pub(crate) fn sdf_contact(
        &self,
        position: Vec3Fix,
        rotation: QuatFix,
        sdf: &SdfCollider,
    ) -> Option<Contact> {
        match self {
            Self::Shape(shape) => posed_sdf_contact(
                &PosedShape {
                    shape: *shape,
                    position,
                    rotation,
                },
                sdf,
            ),
            Self::Compound(compound) => {
                let mut best = None;
                for child in &compound.children {
                    deeper(&mut best, child_sdf_contact(child, position, rotation, sdf));
                }
                best
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
    let (box_a, box_b) = (
        a.world_aabb(position_a, rotation_a),
        b.world_aabb(position_b, rotation_b),
    );
    if !box_a.intersects(&box_b) {
        return None;
    }
    let pieces_a = a.pieces(position_a, rotation_a, &box_b);
    let pieces_b = b.pieces(position_b, rotation_b, &box_a);
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
    let (box_a, box_b) = (
        a.world_aabb(position_a, rotation_a),
        b.world_aabb(position_b, rotation_b),
    );
    if !box_a.intersects(&box_b) {
        return false;
    }
    let pieces_a = a.pieces(position_a, rotation_a, &box_b);
    let pieces_b = b.pieces(position_b, rotation_b, &box_a);
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
