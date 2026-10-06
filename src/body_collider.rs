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
use crate::collider::{contact, gjk, Contact, Sphere, Support, AABB};
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
    /// `rotation`: the shape's own closed-form world box, or the union of a
    /// compound's children's world boxes. Tighter than the cube of the bounding
    /// sphere for any collider that is not round.
    pub(crate) fn world_aabb(&self, position: Vec3Fix, rotation: QuatFix) -> AABB {
        match self {
            Self::Shape(shape) => PosedShape {
                shape: *shape,
                position,
                rotation,
            }
            .world_aabb(),
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
/// `collide_aabb_sdf` samples (an axis-aligned box goes through it), turned with
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

/// The deepest contact between a collider and a plain sphere (a body that has a
/// collision radius but no collider): GJK/EPA on every piece of the collider whose
/// box meets the sphere's box, with the sphere as its own support. When
/// `collider_is_a` the collider is body A and the normal points from the sphere to
/// the collider; otherwise the sphere is body A and the normal points from the
/// collider to the sphere (B to A either way, the [`Contact`] contract).
pub(crate) fn contact_with_sphere(
    collider: &BodyCollider,
    (position, rotation): (Vec3Fix, QuatFix),
    sphere: Sphere,
    collider_is_a: bool,
) -> Option<Contact> {
    let r = sphere.radius;
    let sphere_box = AABB::from_center_half(sphere.center, Vec3Fix::new(r, r, r));
    if !collider
        .world_aabb(position, rotation)
        .intersects(&sphere_box)
    {
        return None;
    }
    let mut deepest: Option<Contact> = None;
    for (piece, piece_box) in collider.pieces(position, rotation, &sphere_box) {
        if !piece_box.intersects(&sphere_box) {
            continue;
        }
        let hit = if collider_is_a {
            contact(&piece, &sphere)
        } else {
            contact(&sphere, &piece)
        };
        if let Some(hit) = hit {
            if deepest.is_none_or(|d| hit.depth > d.depth) {
                deepest = Some(hit);
            }
        }
    }
    deepest
}

/// Whether any piece of a collider meets a plain sphere (GJK; touching counts).
pub(crate) fn collider_meets_sphere(
    collider: &BodyCollider,
    (position, rotation): (Vec3Fix, QuatFix),
    sphere: Sphere,
) -> bool {
    let r = sphere.radius;
    let sphere_box = AABB::from_center_half(sphere.center, Vec3Fix::new(r, r, r));
    if !collider
        .world_aabb(position, rotation)
        .intersects(&sphere_box)
    {
        return false;
    }
    collider
        .pieces(position, rotation, &sphere_box)
        .iter()
        .any(|(piece, piece_box)| {
            piece_box.intersects(&sphere_box) && gjk(piece, &sphere).colliding
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

    use crate::collider::{Capsule, ConvexHull};

    fn fx(num: i64, den: i64) -> Fix128 {
        Fix128::from_ratio(num, den)
    }

    fn v(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
        Vec3Fix::new(x, y, z)
    }

    fn assert_near(actual: Fix128, expected: f64, tol: f64, what: &str) {
        let a = actual.to_f64();
        assert!(
            (a - expected).abs() <= tol,
            "{what}: {a} vs expected {expected} (tol {tol})"
        );
    }

    fn assert_vec_near(actual: Vec3Fix, expected: [f64; 3], tol: f64, what: &str) {
        assert_near(actual.x, expected[0], tol, what);
        assert_near(actual.y, expected[1], tol, what);
        assert_near(actual.z, expected[2], tol, what);
    }

    fn cube(half: Fix128) -> BodyCollider {
        BodyCollider::Shape(Shape::Box {
            half_extents: v(half, half, half),
        })
    }

    /// A quarter turn about `Z` halved: the box turned into a diamond in `XY`.
    fn eighth_turn_z() -> QuatFix {
        QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI.half())
    }

    /// Two unit compound spheres (radius 1/2) at body-local `x = ±2`: a dumbbell
    /// with an empty gap at the origin that its convex hull would fill.
    fn dumbbell() -> BodyCollider {
        let mut c = CompoundShape::new();
        for x in [-2, 2] {
            c.add_sphere(
                Sphere::new(Vec3Fix::ZERO, fx(1, 2)),
                Vec3Fix::from_int(x, 0, 0),
                QuatFix::IDENTITY,
            );
        }
        BodyCollider::Compound(c)
    }

    /// The bounding radius of a box collider is the length of its half-extent
    /// vector (the corner is the farthest point): `|(1, 2, 2)| = 3`.
    #[test]
    fn a_box_colliders_bounding_radius_is_its_corner_distance() {
        let c = BodyCollider::Shape(Shape::Box {
            half_extents: Vec3Fix::from_int(1, 2, 2),
        });
        assert_near(c.bounding_radius(), 3.0, 1e-12, "box bounding radius");
    }

    /// Two axis-aligned unit cubes whose centres are 9/10 apart along `X` overlap
    /// by `1 - 9/10 = 1/10` (sum of half-extents minus the distance), and the
    /// normal points from `b` (at the origin) to `a` (at `+x`): `(1, 0, 0)`.
    /// Moved to 12/10 apart the gap is 1/5 and there is no contact.
    #[test]
    fn two_overlapping_cubes_report_the_face_overlap_and_the_b_to_a_normal() {
        let (a, b) = (cube(fx(1, 2)), cube(fx(1, 2)));
        let at = |x: Fix128| (v(x, Fix128::ZERO, Fix128::ZERO), QuatFix::IDENTITY);
        let origin = (Vec3Fix::ZERO, QuatFix::IDENTITY);

        let hit = contact_between(&a, at(fx(9, 10)), &b, origin).expect("cubes overlap");
        assert_near(hit.depth, 0.1, 1e-9, "depth");
        assert_vec_near(hit.normal, [1.0, 0.0, 0.0], 1e-9, "normal");
        assert!(colliders_meet(&a, at(fx(9, 10)), &b, origin));

        assert!(contact_between(&a, at(fx(12, 10)), &b, origin).is_none());
        assert!(!colliders_meet(&a, at(fx(12, 10)), &b, origin));
    }

    /// Two unit cubes turned 1/8 turn about `Z` are diamonds `|x| + |y| <= √2/2`
    /// in `XY`; their world boxes have half-width `√2/2 ≈ 0.707`. With centres at
    /// the origin and at `(1, 1, 0)` the boxes overlap (`1 < 2·0.707`) but the
    /// diamonds do not: along the diagonal the L1 distance 2 exceeds the sum of the
    /// L1 radii `√2`. The box test alone would report a contact; the shapes must not.
    #[test]
    fn turned_cubes_whose_world_boxes_overlap_but_whose_solids_do_not_are_apart() {
        let (a, b) = (cube(fx(1, 2)), cube(fx(1, 2)));
        let pose_a = (Vec3Fix::from_int(1, 1, 0), eighth_turn_z());
        let pose_b = (Vec3Fix::ZERO, eighth_turn_z());
        assert!(a
            .world_aabb(pose_a.0, pose_a.1)
            .intersects(&b.world_aabb(pose_b.0, pose_b.1)));
        assert!(contact_between(&a, pose_a, &b, pose_b).is_none());
        assert!(!colliders_meet(&a, pose_a, &b, pose_b));

        // Moved to (1/2, 1/2, 0) the L1 distance is 1 < √2: the diamonds overlap
        // along the diagonal by (√2 - 1) in L1, i.e. (√2 - 1)/√2 = 1 - √2/2 along
        // the unit diagonal normal (the faces are perpendicular to (1, 1)/√2).
        let near = (v(fx(1, 2), fx(1, 2), Fix128::ZERO), eighth_turn_z());
        let hit = contact_between(&a, near, &b, pose_b).expect("diamonds overlap");
        let s = 0.5 * core::f64::consts::SQRT_2;
        assert_near(hit.depth, 1.0 - s, 1e-9, "diamond depth");
        assert_vec_near(hit.normal, [s, s, 0.0], 1e-9, "diamond normal");
        assert!(colliders_meet(&a, near, &b, pose_b));
    }

    /// A compound is the union of its children, not their hull: a sphere sitting
    /// in the dumbbell's gap touches nothing, although the hull would contain it.
    /// The same sphere at `x = 2.8` overlaps the `x = 2` child by
    /// `1/2 + 1/2 - 0.8 = 0.2`; the normal points from body B to body A, so it is
    /// `-x` when the collider is A and `+x` when the sphere is A.
    #[test]
    fn a_compound_meets_a_sphere_through_its_children_not_its_hull() {
        let c = dumbbell();
        let pose = (Vec3Fix::ZERO, QuatFix::IDENTITY);
        let in_gap = Sphere::new(Vec3Fix::ZERO, fx(1, 2));
        assert!(!collider_meets_sphere(&c, pose, in_gap));
        assert!(contact_with_sphere(&c, pose, in_gap, true).is_none());
        assert!(contact_with_sphere(&c, pose, in_gap, false).is_none());

        let far = Sphere::new(Vec3Fix::from_int(5, 0, 0), fx(1, 2));
        assert!(!collider_meets_sphere(&c, pose, far));
        assert!(contact_with_sphere(&c, pose, far, true).is_none());

        let touching = Sphere::new(v(fx(28, 10), Fix128::ZERO, Fix128::ZERO), fx(1, 2));
        assert!(collider_meets_sphere(&c, pose, touching));
        // EPA approximates two round surfaces by a polytope: the depth agrees to
        // 1e-3 and the normal to 1e-2 (measured 4.6e-3 off axis).
        let as_a = contact_with_sphere(&c, pose, touching, true).expect("overlap");
        assert_near(as_a.depth, 0.2, 1e-3, "depth (collider is A)");
        assert_vec_near(
            as_a.normal,
            [-1.0, 0.0, 0.0],
            1e-2,
            "normal (collider is A)",
        );
        let as_b = contact_with_sphere(&c, pose, touching, false).expect("overlap");
        assert_near(as_b.depth, 0.2, 1e-3, "depth (sphere is A)");
        assert_vec_near(as_b.normal, [1.0, 0.0, 0.0], 1e-2, "normal (sphere is A)");
    }

    /// A compound against a convex collider: a unit cube centred at `x = 2.9`
    /// reaches down to `x = 2.4` and overlaps the dumbbell's `x = 2` child (which
    /// reaches `x = 2.5`) by `0.1`, pushing the compound (A) towards `-x`. Turning
    /// the dumbbell a quarter turn about `Z` swings that child to `y = 2`, out of
    /// the cube's reach, so the turned compound is clear of it.
    #[test]
    fn a_compound_against_a_box_uses_its_posed_children() {
        let c = dumbbell();
        let block = cube(fx(1, 2));
        let pose_c = (Vec3Fix::ZERO, QuatFix::IDENTITY);
        let pose_block = (v(fx(29, 10), Fix128::ZERO, Fix128::ZERO), QuatFix::IDENTITY);
        let hit = contact_between(&c, pose_c, &block, pose_block).expect("child overlaps");
        assert_near(hit.depth, 0.1, 1e-3, "depth");
        assert_vec_near(hit.normal, [-1.0, 0.0, 0.0], 1e-3, "normal");
        assert!(colliders_meet(&c, pose_c, &block, pose_block));

        let turned = (
            Vec3Fix::ZERO,
            QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI),
        );
        assert!(contact_between(&c, turned, &block, pose_block).is_none());
        assert!(!colliders_meet(&c, turned, &block, pose_block));
        // Both sides compound: the other dumbbell shifted by 4.8 along x puts its
        // x = -2 child at 2.8, 0.2 into this one's x = 2 child.
        let other = (v(fx(48, 10), Fix128::ZERO, Fix128::ZERO), QuatFix::IDENTITY);
        let hit = contact_between(&c, pose_c, &dumbbell(), other).expect("children overlap");
        assert_near(hit.depth, 0.2, 1e-3, "compound-compound depth");
        assert!(colliders_meet(&c, pose_c, &dumbbell(), other));
    }

    /// The half-space `y < 0` (distance `y`, outward normal `+y`), fixed at the
    /// origin.
    #[cfg(feature = "std")]
    fn floor() -> SdfCollider {
        SdfCollider::new_static(
            Box::new(crate::sdf_collider::ClosureSdf::new(
                |_, y, _| y,
                |_, _, _| (0.0, 1.0, 0.0),
            )),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        )
    }

    /// Each convex shape against a flat floor reports its lowest point's depth
    /// below `y = 0` with the floor's normal `+y`:
    /// - an upright unit cube centred at `y = 0.4` reaches `0.4 - 1/2 = -0.1`;
    /// - the same cube turned 1/8 turn about `Z` at `y = 0.6` stands on an edge at
    ///   `0.6 - √2/2`;
    /// - a ball (ellipsoid of equal radii 1/2) at `y = 0.3` reaches `-0.2`;
    /// - an ellipsoid of radii `(1, 1/2, 1)` at `y = 0.4` reaches `-0.1` (its
    ///   support point along `-y`);
    /// - a cylinder of radius 1/2 and half-height 1/2 at `y = 0.45` reaches `-0.05`;
    /// - the upright cube lifted to `y = 2` does not reach the floor.
    ///
    /// The field is evaluated in `f32`, hence the tolerance.
    #[cfg(feature = "std")]
    #[test]
    fn convex_shapes_on_a_flat_floor_report_their_lowest_points_depth() {
        let floor = floor();
        let up = |y: Fix128| v(Fix128::ZERO, y, Fix128::ZERO);
        let cases: [(BodyCollider, Fix128, QuatFix, f64); 5] = [
            (cube(fx(1, 2)), fx(4, 10), QuatFix::IDENTITY, 0.1),
            (
                cube(fx(1, 2)),
                fx(6, 10),
                eighth_turn_z(),
                0.5 * core::f64::consts::SQRT_2 - 0.6,
            ),
            (
                BodyCollider::Shape(Shape::Ellipsoid {
                    radii: v(fx(1, 2), fx(1, 2), fx(1, 2)),
                }),
                fx(3, 10),
                QuatFix::IDENTITY,
                0.2,
            ),
            (
                BodyCollider::Shape(Shape::Ellipsoid {
                    radii: v(Fix128::ONE, fx(1, 2), Fix128::ONE),
                }),
                fx(4, 10),
                QuatFix::IDENTITY,
                0.1,
            ),
            (
                BodyCollider::Shape(Shape::Cylinder {
                    radius: fx(1, 2),
                    half_height: fx(1, 2),
                }),
                fx(45, 100),
                QuatFix::IDENTITY,
                0.05,
            ),
        ];
        for (i, (collider, y, rotation, depth)) in cases.iter().enumerate() {
            let hit = collider
                .sdf_contact(up(*y), *rotation, &floor)
                .unwrap_or_else(|| panic!("case {i}: no contact"));
            assert_near(hit.depth, *depth, 1e-5, "floor depth");
            assert_vec_near(hit.normal, [0.0, 1.0, 0.0], 1e-6, "floor normal");
        }
        assert!(cube(fx(1, 2))
            .sdf_contact(up(Fix128::from_int(2)), QuatFix::IDENTITY, &floor)
            .is_none());
    }

    /// Each kind of compound child against the floor, placed by the body's pose:
    /// - a sphere child (radius 1/4) at body-local `(-3/2, 0, 0)` on a body at
    ///   `y = 1` turned a quarter turn about `Z` lands at world `(0, -1/2, 0)`
    ///   (the turn maps local `-x` to world `-y`): depth `1/2 + 1/4 = 3/4`
    ///   (unturned it would sit at `y = 1`, clear of the floor);
    /// - a capsule child along `y` from `-1/2` to `1/2`, radius 1/4, body at
    ///   `y = 0.6`: lowest point `0.6 - 1/2 - 1/4 = -0.15`;
    /// - a unit-cube child turned 1/8 turn about `Z`, body at `y = 0.6`: lowest
    ///   edge at `0.6 - √2/2`;
    /// - a hull child with vertices `(0, -1, 0), (±1, 0, 0), (0, 0, 1)`, body at
    ///   `y = 0.75`: lowest vertex at `-0.25`.
    ///
    /// A compound holding the capsule and the hull reports the deeper (the hull).
    #[cfg(feature = "std")]
    #[test]
    fn compound_children_on_a_flat_floor_report_the_deepest_child() {
        let floor = floor();
        let up = |y: Fix128| v(Fix128::ZERO, y, Fix128::ZERO);
        let half = fx(1, 2);
        let quarter = fx(1, 4);

        let mut sphere = CompoundShape::new();
        sphere.add_sphere(
            Sphere::new(Vec3Fix::ZERO, quarter),
            v(fx(-3, 2), Fix128::ZERO, Fix128::ZERO),
            QuatFix::IDENTITY,
        );
        let sphere = BodyCollider::Compound(sphere);
        let quarter_turn = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI);
        let hit = sphere
            .sdf_contact(up(Fix128::ONE), quarter_turn, &floor)
            .expect("turned sphere child reaches the floor");
        assert_near(hit.depth, 0.75, 1e-5, "sphere child depth");
        assert!(sphere
            .sdf_contact(up(Fix128::ONE), QuatFix::IDENTITY, &floor)
            .is_none());

        let capsule = Capsule::new(v(Fix128::ZERO, -half, Fix128::ZERO), up(half), quarter);
        let mut c = CompoundShape::new();
        c.add_capsule(capsule, Vec3Fix::ZERO, QuatFix::IDENTITY);
        let hit = BodyCollider::Compound(c)
            .sdf_contact(up(fx(6, 10)), QuatFix::IDENTITY, &floor)
            .expect("capsule child reaches the floor");
        assert_near(hit.depth, 0.15, 1e-5, "capsule child depth");

        let mut c = CompoundShape::new();
        c.add_box(
            OrientedBox::new(Vec3Fix::ZERO, v(half, half, half), eighth_turn_z()),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        let hit = BodyCollider::Compound(c)
            .sdf_contact(up(fx(6, 10)), QuatFix::IDENTITY, &floor)
            .expect("box child reaches the floor");
        assert_near(
            hit.depth,
            0.5 * core::f64::consts::SQRT_2 - 0.6,
            1e-5,
            "box child depth",
        );

        let hull = || {
            ConvexHull::new(vec![
                Vec3Fix::from_int(0, -1, 0),
                Vec3Fix::from_int(1, 0, 0),
                Vec3Fix::from_int(-1, 0, 0),
                Vec3Fix::from_int(0, 0, 1),
            ])
        };
        let mut both = CompoundShape::new();
        both.add_capsule(capsule, Vec3Fix::ZERO, QuatFix::IDENTITY);
        both.add_convex_hull(hull(), Vec3Fix::ZERO, QuatFix::IDENTITY);
        // Body at y = 0.75: capsule bottom at 0.75 - 3/4 = 0 (touching, no
        // contact), hull bottom at -0.25.
        let hit = BodyCollider::Compound(both)
            .sdf_contact(up(fx(3, 4)), QuatFix::IDENTITY, &floor)
            .expect("hull child reaches the floor");
        assert_near(hit.depth, 0.25, 1e-5, "deepest child depth");
        assert_vec_near(hit.normal, [0.0, 1.0, 0.0], 1e-6, "floor normal");
    }
}
