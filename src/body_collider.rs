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

#[cfg(feature = "std")]
use crate::box_collider::OrientedBox;
use crate::collider::{contact, gjk, Contact, Sphere, Support, AABB};
use crate::compound::{CompoundChild, CompoundShape, ShapeRef};
use crate::heightfield::HeightField;
use crate::math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};
use crate::plane_collider::PlaneCollider;
#[cfg(feature = "std")]
use crate::sdf_collider::{
    box_sample_points, collide_aabb_sdf, collide_capsule_sdf, collide_point_sdf,
    collide_points_sdf, collide_sphere_sdf, SdfCollider,
};
use crate::shape::{PosedShape, Shape};
use crate::static_collider::StaticCollider;
use crate::trimesh::TriMesh;
use crate::world_shape_query::TriangleSupport;

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

impl Piece<'_> {
    /// This piece as a ball, when it is one: a compound's sphere child, or an
    /// ellipsoid with equal semi-axes.
    fn as_ball(&self) -> Option<Sphere> {
        match self {
            Self::Shape(PosedShape {
                shape: Shape::Ellipsoid { radii },
                position,
                ..
            }) if radii.x == radii.y && radii.y == radii.z => Some(Sphere::new(*position, radii.x)),
            Self::Child {
                child,
                position,
                rotation,
            } => match &child.shape {
                ShapeRef::Sphere(sphere) => {
                    let body_rot = rotation.unit_rotation();
                    let child_rot = body_rot.mul(child.local_rotation);
                    let child_pos = *position + body_rot.rotate_vec(child.local_position);
                    Some(Sphere::new(
                        child_pos + child_rot.rotate_vec(sphere.center),
                        sphere.radius,
                    ))
                }
                _ => None,
            },
            Self::Shape(_) => None,
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
    let body_rotation = body_rotation.unit_rotation();
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
    // LIMITATION(COV-RIGID-068): against a curved one a face or an edge can reach deeper than the points sampled.
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

/// How many times the tangent plane under a piece is taken again at the piece's
/// lowest point, against a height field.
const FIELD_SUPPORT_ITERATIONS: usize = 2;

impl BodyCollider {
    /// The deepest contact of this collider, for a body at `position` turned by
    /// `rotation`, with a static collider, or `None` when no piece reaches into it.
    /// The normal points from the surface toward the body; translating the body by
    /// `depth · normal` moves the deepest piece out.
    ///
    /// - **Plane**: closed form. The plane is two-sided, as for a sphere: the side
    ///   is the one the body's centre of mass is on, and the piece's support point
    ///   against that side's normal is its deepest point.
    /// - **Triangle mesh**: GJK/EPA of each piece with every triangle whose box
    ///   meets the piece's box (the mesh's BVH), the deepest kept.
    /// - **Height field**: the tangent plane of the field under the piece's lowest
    ///   point, taken again at the new lowest point ([`FIELD_SUPPORT_ITERATIONS`]),
    ///   as for a sphere (the distance along the surface normal). Exact where the
    ///   field is planar under the piece (flat, or one constant slope).
    // LIMITATION(COV-RIGID-065): a height field whose bilinear cells curve under a piece is met at the tangent plane under its lowest point
    pub(crate) fn static_contact(
        &self,
        position: Vec3Fix,
        rotation: QuatFix,
        collider: &StaticCollider,
    ) -> Option<Contact> {
        let own = self.world_aabb(position, rotation);
        let mut best: Option<Contact> = None;
        for (piece, piece_box) in self.pieces(position, rotation, &own) {
            let hit = match collider {
                StaticCollider::Plane(plane) => plane_piece_contact(&piece, position, plane),
                StaticCollider::TriMesh(mesh) => {
                    mesh_piece_contact(&piece, &piece_box, position, mesh)
                }
                StaticCollider::HeightField(field) => {
                    field_piece_contact(&piece, &piece_box, field)
                }
            };
            if let Some(c) = hit {
                if best.is_none_or(|b| c.depth > b.depth) {
                    best = Some(c);
                }
            }
        }
        best
    }
}

/// A convex piece against a two-sided plane: the side is the one `reference` (the
/// body's centre of mass) is on.
fn plane_piece_contact(
    piece: &Piece<'_>,
    reference: Vec3Fix,
    plane: &PlaneCollider,
) -> Option<Contact> {
    let side = if plane.distance_to_point(reference) < Fix128::ZERO {
        -plane.normal
    } else {
        plane.normal
    };
    let lowest = piece.support(-side);
    // the signed height of the lowest point over the plane, on the body's side
    let height = if side == plane.normal {
        plane.distance_to_point(lowest)
    } else {
        -plane.distance_to_point(lowest)
    };
    (height < Fix128::ZERO).then(|| Contact {
        depth: -height,
        normal: side,
        point_a: lowest,
        point_b: lowest - side * height,
    })
}

/// A convex piece against a triangle mesh: GJK/EPA with each candidate triangle.
fn mesh_piece_contact(
    piece: &Piece<'_>,
    piece_box: &AABB,
    reference: Vec3Fix,
    mesh: &TriMesh,
) -> Option<Contact> {
    let mut best: Option<Contact> = None;
    for index in mesh.bvh.query(piece_box) {
        let Some(triangle) = mesh.triangles.get(index as usize) else {
            continue;
        };
        if !triangle.aabb().intersects(piece_box) {
            continue;
        }
        let hit = match triangle_face_contact(piece, reference, triangle) {
            Face::Clear => None,
            Face::Hit(c) => Some(c),
            Face::Outside => match piece.as_ball() {
                Some(ball) => ball_triangle_boundary_contact(ball, reference, triangle),
                None => contact(piece, &TriangleSupport(triangle)),
            },
        };
        if let Some(hit) = hit {
            if hit.depth > Fix128::ZERO && best.is_none_or(|d| hit.depth > d.depth) {
                best = Some(hit);
            }
        }
    }
    best
}

/// How a convex piece meets one triangle's face
enum Face {
    /// The piece lies wholly on the reference's side of the triangle's plane: it
    /// cannot meet the triangle
    Clear,
    /// The piece's deepest point below the plane projects inside the triangle:
    /// the contact is the face's, exactly (the face normal, the depth below it)
    Hit(Contact),
    /// The deepest point projects outside the triangle: an edge or a corner is
    /// met, left to GJK/EPA
    Outside,
}

/// A convex piece against the face of one triangle, two-sided like a plane (the
/// side is the one `reference`, the body's centre of mass, is on). On a face the
/// answer is exact: GJK/EPA on a curved piece only bounds the depth and its
/// normal, and a normal a little off the face's pushes a body resting on a mesh
/// sideways every substep.
fn triangle_face_contact(
    piece: &Piece<'_>,
    reference: Vec3Fix,
    triangle: &crate::trimesh::Triangle,
) -> Face {
    let n = triangle.unit_normal();
    if n.length_squared().is_zero() {
        return Face::Outside;
    }
    let above = |p: Vec3Fix| (p - triangle.v0).dot(n);
    let side = if above(reference) < Fix128::ZERO {
        -n
    } else {
        n
    };
    let lowest = piece.support(-side);
    let height = (lowest - triangle.v0).dot(side);
    if height >= Fix128::ZERO {
        return Face::Clear;
    }
    let on_plane = lowest - side * height;
    let (a, b, c) = (triangle.v0, triangle.v1, triangle.v2);
    let inside = [(a, b), (b, c), (c, a)]
        .iter()
        .all(|&(p, q)| (q - p).cross(on_plane - p).dot(n) >= Fix128::ZERO);
    if inside {
        Face::Hit(Contact {
            depth: -height,
            normal: side,
            point_a: lowest,
            point_b: on_plane,
        })
    } else {
        Face::Outside
    }
}

/// The point of the segment `a`-`b` closest to `p`; `a` when the segment has no length.
fn closest_on_segment(a: Vec3Fix, b: Vec3Fix, p: Vec3Fix) -> Vec3Fix {
    let e = b - a;
    let len2 = e.length_squared();
    if len2.is_zero() {
        return a;
    }
    let t = (p - a).dot(e) / len2;
    if t <= Fix128::ZERO {
        a
    } else if t >= Fix128::ONE {
        b
    } else {
        a + e * t
    }
}

/// The normal of a ball whose centre lies on the triangle's boundary, where the offset
/// to the closest point has no direction: the face's on the reference's side; for a
/// triangle with no area, perpendicular to its longest edge, built from the unit axis
/// least along that edge (the first such, in the order X, Y, Z); for a triangle that
/// is a single point, +Y.
fn centre_on_boundary_normal(reference: Vec3Fix, tri: &crate::trimesh::Triangle) -> Vec3Fix {
    let n = tri.unit_normal();
    if !n.length_squared().is_zero() {
        return if (reference - tri.v0).dot(n) < Fix128::ZERO {
            -n
        } else {
            n
        };
    }
    let mut edge = tri.v1 - tri.v0;
    for e in [tri.v2 - tri.v1, tri.v0 - tri.v2] {
        if e.length_squared() > edge.length_squared() {
            edge = e;
        }
    }
    if edge.length_squared().is_zero() {
        return Vec3Fix::UNIT_Y;
    }
    let (x, y, z) = (edge.x.abs(), edge.y.abs(), edge.z.abs());
    let axis = if x <= y && x <= z {
        Vec3Fix::UNIT_X
    } else if y <= z {
        Vec3Fix::UNIT_Y
    } else {
        Vec3Fix::UNIT_Z
    };
    edge.cross(axis).normalize()
}

/// A ball against the boundary of a triangle (its edges and corners, and the whole of
/// a triangle with no area), in closed form: the closest point of the three edges, the
/// depth `r - d` to it and the normal along the offset, where GJK/EPA stops once its
/// polytope is within 2^-16 and leaves the normal up to 0.45 degrees off. Edges are
/// taken in order and a later one must be strictly closer, so a tie has one answer.
fn ball_triangle_boundary_contact(
    ball: Sphere,
    reference: Vec3Fix,
    tri: &crate::trimesh::Triangle,
) -> Option<Contact> {
    let c = ball.center;
    let mut closest = closest_on_segment(tri.v0, tri.v1, c);
    let mut best = (c - closest).length_squared();
    for (p, q) in [(tri.v1, tri.v2), (tri.v2, tri.v0)] {
        let on_edge = closest_on_segment(p, q, c);
        let d2 = (c - on_edge).length_squared();
        if d2 < best {
            best = d2;
            closest = on_edge;
        }
    }
    let d = best.sqrt();
    if d >= ball.radius {
        return None;
    }
    let normal = if d.is_zero() {
        centre_on_boundary_normal(reference, tri)
    } else {
        (c - closest).rescaled_direction().normalize()
    };
    Some(Contact {
        depth: ball.radius - d,
        normal,
        point_a: c - normal * ball.radius,
        point_b: closest,
    })
}

/// A convex piece against a height field, at the tangent plane under its lowest
/// point. Outside the grid (by more than the 2 cells a sphere is also tested in)
/// there is no surface.
fn field_piece_contact(
    piece: &Piece<'_>,
    piece_box: &AABB,
    field: &HeightField,
) -> Option<Contact> {
    if field.width == 0 || field.depth == 0 || field.spacing <= Fix128::ZERO {
        return None;
    }
    let mid = (piece_box.min + piece_box.max) * Fix128::from_ratio(1, 2);
    let mut normal = field.sample_normal(mid.x, mid.z);
    let mut lowest = piece.support(-normal);
    for _ in 0..FIELD_SUPPORT_ITERATIONS {
        normal = field.sample_normal(lowest.x, lowest.z);
        lowest = piece.support(-normal);
    }
    let (gx, gz) = field.world_to_grid(lowest);
    let margin = Fix128::from_int(2);
    let last = |n: u32| Fix128::from_int(i64::from(n) - 1);
    if gx < -margin
        || gx > last(field.width) + margin
        || gz < -margin
        || gz > last(field.depth) + margin
    {
        return None;
    }
    let ground = field.sample_height(lowest.x, lowest.z);
    // distance from the lowest point to the tangent plane at the point below it
    let height = (lowest.y - ground) * normal.y;
    (height < Fix128::ZERO).then(|| Contact {
        depth: -height,
        normal,
        point_a: lowest,
        point_b: lowest - normal * height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box with a non-unit orientation (norm 2) meets a plane SDF exactly as
    /// the unit one does: its sample points are not scaled by |q|^2.
    #[cfg(feature = "std")]
    #[test]
    fn box_sdf_contact_ignores_the_orientation_norm() {
        let plane = || {
            crate::sdf_collider::SdfCollider::new_static(
                Box::new(crate::sdf_collider::ClosureSdf::new(
                    |_x, y, _z| y,
                    |_x, _y, _z| (0.0, 1.0, 0.0),
                )),
                Vec3Fix::ZERO,
                QuatFix::IDENTITY,
            )
        };
        let half = Vec3Fix::new(Fix128::ONE, Fix128::from_ratio(1, 2), Fix128::ONE);
        let center = Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 4), Fix128::ZERO);
        let turn = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::from_ratio(1, 5));
        let doubled = QuatFix::new(
            turn.x.double(),
            turn.y.double(),
            turn.z.double(),
            turn.w.double(),
        );
        let unit = box_sdf_contact(&OrientedBox::new(center, half, turn), &plane());
        let scaled = box_sdf_contact(&OrientedBox::new(center, half, doubled), &plane());
        let (u, s) = (unit.expect("touches"), scaled.expect("touches"));
        assert!(
            (u.depth - s.depth).abs() < Fix128::from_ratio(1, 1_000_000),
            "{u:?} {s:?}"
        );
    }

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

    /// Compound children against the floor for a body whose orientation is not a
    /// unit quaternion (norm 2 and 1/2): each child is placed by the rotation the
    /// quaternion stands for, so the contact is the normalized orientation's, bit
    /// for bit, and the unit orientation's up to the normalization's rounding.
    /// Unnormalized, a child at body-local `x = 1` would sit `|q|^2` times as far
    /// from the body origin.
    #[cfg(feature = "std")]
    #[test]
    fn compound_sdf_contact_ignores_the_orientation_norm() {
        let floor = floor();
        let position = v(Fix128::ZERO, fx(1, 2), Fix128::ZERO);
        let turn = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(-6, 10));
        let local = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, fx(3, 10));
        let at = v(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let quarter = fx(1, 4);
        let mut kinds = Vec::new();
        let mut c = CompoundShape::new();
        c.add_sphere(
            Sphere::new(v(quarter, Fix128::ZERO, Fix128::ZERO), quarter),
            at,
            local,
        );
        kinds.push(("sphere", c));
        let mut c = CompoundShape::new();
        c.add_capsule(
            Capsule::new(
                v(-quarter, Fix128::ZERO, Fix128::ZERO),
                v(quarter, Fix128::ZERO, Fix128::ZERO),
                quarter,
            ),
            at,
            local,
        );
        kinds.push(("capsule", c));
        let mut c = CompoundShape::new();
        c.add_box(
            OrientedBox::new(
                v(Fix128::ZERO, Fix128::ZERO, quarter),
                v(quarter, quarter, quarter),
                eighth_turn_z(),
            ),
            at,
            local,
        );
        kinds.push(("box", c));
        let mut c = CompoundShape::new();
        c.add_convex_hull(
            ConvexHull::new(vec![
                Vec3Fix::from_int(0, -1, 0),
                Vec3Fix::from_int(1, 0, 0),
                Vec3Fix::from_int(-1, 0, 0),
                Vec3Fix::from_int(0, 0, 1),
            ]),
            at,
            local,
        );
        kinds.push(("hull", c));
        for (name, c) in kinds {
            let collider = BodyCollider::Compound(c);
            let unit = collider
                .sdf_contact(position, turn, &floor)
                .unwrap_or_else(|| panic!("{name}: the unit orientation reaches the floor"));
            for s in [Fix128::from_int(2), fx(1, 2)] {
                let scaled = QuatFix::new(turn.x * s, turn.y * s, turn.z * s, turn.w * s);
                let got = collider.sdf_contact(position, scaled, &floor);
                assert_eq!(
                    got,
                    collider.sdf_contact(position, scaled.normalize(), &floor),
                    "{name}, s = {}",
                    s.to_f64()
                );
                let got = got.unwrap_or_else(|| panic!("{name}: no contact"));
                assert_near(got.depth, unit.depth.to_f64(), 1e-6, name);
            }
        }
    }

    /// A compound of one sphere of radius `r` at the body origin: its lowest point
    /// along any direction is unique (a box's face has many).
    fn ball(r: Fix128) -> BodyCollider {
        let mut c = CompoundShape::new();
        c.add_sphere(
            Sphere::new(Vec3Fix::ZERO, r),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        BodyCollider::Compound(c)
    }

    fn ground_plane() -> StaticCollider {
        StaticCollider::Plane(PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO))
    }

    fn tri(a: [i64; 3], b: [i64; 3], c: [i64; 3]) -> crate::trimesh::Triangle {
        crate::trimesh::Triangle {
            v0: Vec3Fix::from_int(a[0], a[1], a[2]),
            v1: Vec3Fix::from_int(b[0], b[1], b[2]),
            v2: Vec3Fix::from_int(c[0], c[1], c[2]),
        }
    }

    /// The square `[-10, 10]^2` at `y = 0` as two triangles, normals up.
    fn floor_mesh() -> StaticCollider {
        StaticCollider::TriMesh(TriMesh::from_triangles(vec![
            tri([-10, 0, -10], [-10, 0, 10], [10, 0, 10]),
            tri([-10, 0, -10], [10, 0, 10], [10, 0, -10]),
        ]))
    }

    /// A unit cube (half 1/2) sunk 1/4 into the plane `y = 0` from either side
    /// meets it 1/4 deep with the normal on its own side; clear of it, no contact.
    #[test]
    fn a_box_meets_a_two_sided_plane_on_its_own_side() {
        let c = cube(fx(1, 2));
        let up = c
            .static_contact(
                v(Fix128::ZERO, fx(1, 4), Fix128::ZERO),
                QuatFix::IDENTITY,
                &ground_plane(),
            )
            .expect("sunk from above");
        assert_eq!(up.normal, Vec3Fix::UNIT_Y);
        assert_eq!(up.depth, fx(1, 4));
        let down = c
            .static_contact(
                v(Fix128::ZERO, -fx(1, 4), Fix128::ZERO),
                QuatFix::IDENTITY,
                &ground_plane(),
            )
            .expect("sunk from below");
        assert_eq!(down.normal, -Vec3Fix::UNIT_Y);
        assert_eq!(down.depth, fx(1, 4));
        assert!(c
            .static_contact(
                Vec3Fix::from_int(0, 2, 0),
                QuatFix::IDENTITY,
                &ground_plane()
            )
            .is_none());
    }

    /// A box sunk into a mesh's face meets it exactly as a plane: the face normal
    /// on the box's side and the depth below the face. Above the face, or away
    /// from the mesh (no BVH candidate), there is no contact.
    #[test]
    fn a_box_meets_a_mesh_face_exactly() {
        let c = cube(fx(1, 2));
        let mesh = floor_mesh();
        let up = c
            .static_contact(v(fx(1, 3), fx(1, 4), fx(2, 7)), QuatFix::IDENTITY, &mesh)
            .expect("on the face");
        assert_eq!(up.normal, Vec3Fix::UNIT_Y);
        assert_eq!(up.depth, fx(1, 4));
        let down = c
            .static_contact(v(fx(1, 3), -fx(1, 4), fx(2, 7)), QuatFix::IDENTITY, &mesh)
            .expect("under the face");
        assert_eq!(down.normal, -Vec3Fix::UNIT_Y);
        assert_eq!(down.depth, fx(1, 4));
        // the box straddles the face's plane only when sunk: 1/2 above is clear
        // although its box touches the triangle's
        assert!(c
            .static_contact(
                v(Fix128::ZERO, fx(1, 2), Fix128::ZERO),
                QuatFix::IDENTITY,
                &mesh
            )
            .is_none());
        assert!(c
            .static_contact(Vec3Fix::from_int(50, 0, 0), QuatFix::IDENTITY, &mesh)
            .is_none());
    }

    /// A ball beside a triangle's edge, its lowest point projecting outside the
    /// triangle, is met at the edge: depth `r − d` where `d` is the
    /// distance from the centre to the edge line, along the centre's offset.
    #[test]
    fn a_ball_beside_a_triangle_edge_meets_the_edge() {
        let mesh = StaticCollider::TriMesh(TriMesh::from_triangles(vec![tri(
            [0, 0, 0],
            [0, 0, 4],
            [4, 0, 0],
        )]));
        let r = fx(1, 2);
        // centre (−1/5, 3/10, 1): the edge x = y = 0 is at d = √(1/25 + 9/100)
        let hit = ball(r)
            .static_contact(
                v(-fx(1, 5), fx(3, 10), Fix128::ONE),
                QuatFix::IDENTITY,
                &mesh,
            )
            .expect("the edge is within r");
        let d = (0.04_f64 + 0.09).sqrt();
        assert_near(hit.depth, 0.5 - d, 1e-15, "edge depth");
        let n = [-0.2 / d, 0.3 / d, 0.0];
        assert_vec_near(hit.normal, n, 1e-15, "edge normal");
    }

    /// A degenerate (zero-area) triangle has no face: the ball is met on the
    /// segment it is.
    #[test]
    fn a_degenerate_triangle_is_met_as_its_segment() {
        let mesh = StaticCollider::TriMesh(TriMesh::from_triangles(vec![tri(
            [0, 0, 0],
            [2, 0, 0],
            [4, 0, 0],
        )]));
        let hit = ball(fx(1, 2))
            .static_contact(
                v(Fix128::ONE, fx(3, 10), Fix128::ZERO),
                QuatFix::IDENTITY,
                &mesh,
            )
            .expect("the segment is within r");
        assert_near(hit.depth, 0.2, 1e-15, "segment depth");
        assert_vec_near(hit.normal, [0.0, 1.0, 0.0], 1e-15, "segment normal");
    }

    /// A ball off a triangle in a general orientation, each case built so the closest
    /// point is known without computing it: the centre is `q + o`, with `q` on an edge or
    /// a corner and `o` in the normal cone there (`o · (p - q) <= 0` for every corner
    /// `p`, checked below). Depth `r - |o|`, normal `o / |o|`. The f64 reference is good
    /// to about 1e-15 at these sizes and Fix128 rounds at 2^-64, so 1e-14 is the bound.
    #[test]
    fn a_ball_off_a_general_triangle_is_met_at_the_known_closest_point() {
        let corners = [[1_i64, 2, -1], [4, 3, 0], [2, 5, 3]];
        let mesh = StaticCollider::TriMesh(TriMesh::from_triangles(vec![tri(
            corners[0], corners[1], corners[2],
        )]));
        // twice q (to stay in integers), then o before its scale of 1/10
        let cases: [([i64; 3], [i64; 3]); 6] = [
            ([5, 5, -1], [1, -3, 0]),  // edge v0-v1
            ([6, 8, 3], [1, 1, 0]),    // edge v1-v2
            ([3, 7, 2], [-3, 1, 0]),   // edge v2-v0
            ([2, 4, -2], [-1, -1, 0]), // corner v0
            ([8, 6, 0], [1, 0, -1]),   // corner v1
            ([4, 10, 6], [0, 1, 1]),   // corner v2
        ];
        let r = 0.5_f64;
        for (twice_q, o) in cases {
            for p in corners {
                let dot: i64 = (0..3).map(|i| o[i] * (2 * p[i] - twice_q[i])).sum();
                assert!(dot <= 0, "{o:?} is not in the normal cone at {twice_q:?}");
            }
            let centre = v(
                fx(5 * twice_q[0] + o[0], 10),
                fx(5 * twice_q[1] + o[1], 10),
                fx(5 * twice_q[2] + o[2], 10),
            );
            let hit = ball(fx(1, 2))
                .static_contact(centre, QuatFix::IDENTITY, &mesh)
                .expect("the closest point is within r");
            let o = o.map(|c| c as f64 / 10.0);
            let d = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
            let what = format!("q/2 {twice_q:?}");
            assert_near(hit.depth, r - d, 1e-14, &what);
            assert_vec_near(hit.normal, [o[0] / d, o[1] / d, o[2] / d], 1e-14, &what);
            let q = [
                twice_q[0] as f64 / 2.0,
                twice_q[1] as f64 / 2.0,
                twice_q[2] as f64 / 2.0,
            ];
            assert_vec_near(hit.point_b, q, 1e-14, &what);
            let n = hit.normal;
            let surface = centre - n * fx(1, 2);
            assert_eq!(hit.point_a, surface, "{what}");
        }
    }

    /// An ellipsoid with equal semi-axes is a ball, and goes the same way as a sphere
    /// child; one with unequal semi-axes is not one.
    #[test]
    fn only_a_round_piece_is_a_ball() {
        let radii = |a: i64, b: i64| Shape::Ellipsoid {
            radii: Vec3Fix::from_int(a, b, a),
        };
        let at = Vec3Fix::from_int(1, 2, 3);
        let piece = |shape| {
            Piece::Shape(PosedShape {
                shape,
                position: at,
                rotation: QuatFix::IDENTITY,
            })
        };
        let round = piece(radii(2, 2)).as_ball().expect("round");
        assert_eq!((round.center, round.radius), (at, Fix128::from_int(2)));
        assert!(piece(radii(2, 3)).as_ball().is_none());
        // one pair equal is not enough, whichever pair
        for radii in [Vec3Fix::from_int(2, 2, 3), Vec3Fix::from_int(3, 2, 2)] {
            assert!(piece(Shape::Ellipsoid { radii }).as_ball().is_none());
        }
        assert!(piece(radii(3, 2)).as_ball().is_none());
        assert!(piece(Shape::Box {
            half_extents: Vec3Fix::from_int(1, 1, 1)
        })
        .as_ball()
        .is_none());
    }

    /// A sphere child is placed by the body's pose, its local offset and turn, and the
    /// sphere's own centre: the same point its support mapping is about.
    #[test]
    fn a_sphere_child_as_a_ball_sits_where_its_support_says() {
        let mut c = CompoundShape::new();
        c.add_sphere(
            Sphere::new(Vec3Fix::from_int(1, 0, 0), fx(1, 2)),
            Vec3Fix::from_int(0, 2, 0),
            QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI),
        );
        let position = Vec3Fix::from_int(3, 0, -1);
        let rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::HALF_PI);
        let piece = Piece::Child {
            child: &c.children[0],
            position,
            rotation,
        };
        let b = piece.as_ball().expect("a sphere child");
        // +X in the child turns to +Y, offset (0, 3, 0) in the body; the body's quarter
        // turn about X takes that to (0, 0, 3). The quarter turns are built from a
        // rounded sine, measured off by 1.1e-14 here, so the bound is 1e-12
        assert_vec_near(b.center, [3.0, 0.0, 2.0], 1e-12, "centre");
        let top = piece.support(Vec3Fix::UNIT_X);
        assert_vec_near(top, [3.5, 0.0, 2.0], 1e-12, "support about the centre");
        assert_eq!(b.radius, fx(1, 2));
    }

    /// A centre exactly on the segment of an area-free triangle has no offset direction:
    /// the normal is fixed by the edge alone, unit and perpendicular to it, the same
    /// bits each time, and the depth is the whole radius.
    #[test]
    fn a_centre_on_a_degenerate_triangle_has_one_normal() {
        let r = fx(1, 2);
        let reference = Vec3Fix::ZERO;
        let along_x = tri([0, 0, 0], [2, 0, 0], [4, 0, 0]);
        let ball_at = |c: Vec3Fix| Sphere::new(c, r);
        let hit = ball_triangle_boundary_contact(
            ball_at(Vec3Fix::from_int(1, 0, 0)),
            reference,
            &along_x,
        )
        .expect("on the segment");
        assert_eq!(hit.depth, r);
        // the longest edge is v2-v0, so the edge runs along -X
        assert_eq!(hit.normal, (-Vec3Fix::UNIT_X).cross(Vec3Fix::UNIT_Y));
        // the axis is the one least along the edge, whichever that is
        let along_y = tri([0, 0, 0], [0, 2, 0], [0, 4, 0]);
        let hit = ball_triangle_boundary_contact(
            ball_at(Vec3Fix::from_int(0, 1, 0)),
            reference,
            &along_y,
        )
        .expect("on the segment");
        assert_eq!(hit.normal, (-Vec3Fix::UNIT_Y).cross(Vec3Fix::UNIT_X));
        let along_z = tri([0, 0, 0], [0, 0, 2], [0, 0, 4]);
        let hit = ball_triangle_boundary_contact(
            ball_at(Vec3Fix::from_int(0, 0, 1)),
            reference,
            &along_z,
        )
        .expect("on the segment");
        assert_eq!(hit.normal, (-Vec3Fix::UNIT_Z).cross(Vec3Fix::UNIT_X));
        let along_xz = tri([0, 0, 0], [3, 1, 4], [6, 2, 8]);
        let hit = ball_triangle_boundary_contact(
            ball_at(Vec3Fix::from_int(3, 1, 4)),
            reference,
            &along_xz,
        )
        .expect("on the segment");
        let e = Vec3Fix::from_int(3, 1, 4).normalize();
        assert!(hit.normal.dot(e).abs() < fx(1, 1_000_000_000));
        assert_near(hit.normal.length(), 1.0, 1e-15, "unit");
        // the longest edge decides, not the first: v0 = v1 leaves v1-v2 to be used
        let repeated = tri([1, 0, 0], [1, 0, 0], [1, 4, 0]);
        let hit = ball_triangle_boundary_contact(
            ball_at(Vec3Fix::from_int(1, 2, 0)),
            reference,
            &repeated,
        )
        .expect("on the segment");
        assert_eq!(hit.point_b, Vec3Fix::from_int(1, 2, 0));
        assert_eq!(hit.normal, Vec3Fix::UNIT_Y.cross(Vec3Fix::UNIT_X));
        // one point: +Y
        let point = tri([1, 1, 1], [1, 1, 1], [1, 1, 1]);
        let hit =
            ball_triangle_boundary_contact(ball_at(Vec3Fix::from_int(1, 1, 1)), reference, &point)
                .expect("on the point");
        assert_eq!(hit.normal, Vec3Fix::UNIT_Y);
    }

    /// A centre on the boundary of a triangle with area takes the face's normal on the
    /// reference's side.
    #[test]
    fn a_centre_on_a_real_edge_takes_the_face_on_the_reference_side() {
        let t = tri([0, 0, 0], [0, 0, 4], [4, 0, 0]);
        let ball = Sphere::new(Vec3Fix::from_int(2, 0, 0), fx(1, 2));
        let above = Vec3Fix::from_int(0, 3, 0);
        let hit = ball_triangle_boundary_contact(ball, above, &t).expect("on the edge");
        assert_eq!(hit.normal, Vec3Fix::UNIT_Y);
        assert_eq!(hit.depth, fx(1, 2));
        let below = Vec3Fix::from_int(0, -3, 0);
        let hit = ball_triangle_boundary_contact(ball, below, &t).expect("on the edge");
        assert_eq!(hit.normal, -Vec3Fix::UNIT_Y);
    }

    /// A ball just out of reach of the boundary, or exactly touching it, has no contact.
    #[test]
    fn a_ball_at_or_past_its_radius_from_the_boundary_has_no_contact() {
        let t = tri([0, 0, 0], [2, 0, 0], [4, 0, 0]);
        let r = fx(1, 2);
        let reference = Vec3Fix::ZERO;
        let touching = Sphere::new(v(Fix128::ONE, r, Fix128::ZERO), r);
        assert!(ball_triangle_boundary_contact(touching, reference, &t).is_none());
        let past = Sphere::new(v(Fix128::ONE, fx(6, 10), Fix128::ZERO), r);
        assert!(ball_triangle_boundary_contact(past, reference, &t).is_none());
        let within = Sphere::new(v(Fix128::ONE, fx(4, 10), Fix128::ZERO), r);
        assert!(ball_triangle_boundary_contact(within, reference, &t).is_some());
    }

    /// Past the end of an edge the closest point clamps to the corner, and a segment
    /// with no length is its one point.
    #[test]
    fn the_closest_point_on_a_segment_clamps_to_its_ends() {
        let a = Vec3Fix::from_int(1, 0, 0);
        let b = Vec3Fix::from_int(3, 0, 0);
        assert_eq!(closest_on_segment(a, b, Vec3Fix::from_int(0, 1, 0)), a);
        assert_eq!(closest_on_segment(a, b, Vec3Fix::from_int(5, 1, 0)), b);
        assert_eq!(closest_on_segment(a, b, a), a);
        assert_eq!(closest_on_segment(a, b, b), b);
        assert_eq!(
            closest_on_segment(a, b, Vec3Fix::from_int(2, 7, 0)),
            Vec3Fix::from_int(2, 0, 0)
        );
        assert_eq!(closest_on_segment(a, a, b), a);
    }

    /// GJK/EPA run out of their 64 iterations on a ball very near a segment (measured at
    /// all four offsets here), and then return their best upper bound, not a converged
    /// depth; the normal is not bounded on this exit (measured up to 11 degrees off at
    /// 1/10000, and arbitrary at 1/100000). A piece that is not a ball can take this exit,
    /// so its depth is pinned: never short of the true depth by more than the threshold,
    /// and no more than the radius and the threshold.
    #[test]
    fn epa_out_of_iterations_returns_an_upper_bound() {
        const EPA_THRESHOLD: f64 = 1.0 / 65536.0;
        let t = tri([0, 0, 0], [2, 0, 0], [4, 0, 0]);
        for den in [100_i64, 1_000, 10_000, 100_000] {
            let ball = Sphere::new(v(Fix128::ONE, fx(1, den), Fix128::ZERO), fx(1, 2));
            let hit = contact(&ball, &TriangleSupport(&t)).expect("overlap");
            let want = 0.5 - 1.0 / den as f64;
            let got = hit.depth.to_f64();
            assert!(got >= want - EPA_THRESHOLD, "1/{den}: {got} vs {want}");
            assert!(got <= 0.5 + EPA_THRESHOLD, "1/{den}: {got} vs {want}");
        }
    }

    /// A piece that is not a ball stays on GJK/EPA at an edge, which stops once its
    /// polytope is within 2^-16: a capsule along Z, its side over the edge x = y = 0.
    /// Depth `r - d`, the capsule's flat side keeps the normal exact to the same bound.
    #[test]
    fn a_capsule_beside_a_triangle_edge_stays_within_the_epa_threshold() {
        const EPA_THRESHOLD: f64 = 1.0 / 65536.0;
        let mesh = StaticCollider::TriMesh(TriMesh::from_triangles(vec![tri(
            [0, 0, 0],
            [0, 0, 4],
            [4, 0, 0],
        )]));
        let mut c = CompoundShape::new();
        c.add_capsule(
            Capsule::new(
                Vec3Fix::from_int(0, 0, -1),
                Vec3Fix::from_int(0, 0, 1),
                fx(1, 2),
            ),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        let collider = BodyCollider::Compound(c);
        let hit = collider
            .static_contact(v(-fx(1, 5), fx(3, 10), fx(2, 1)), QuatFix::IDENTITY, &mesh)
            .expect("the edge is within r");
        let d = (0.04_f64 + 0.09).sqrt();
        assert_near(hit.depth, 0.5 - d, EPA_THRESHOLD, "capsule depth");
    }

    /// The deepest of several pieces is the contact: a dumbbell tilted so its
    /// right ball sits lower meets the plane at that ball's depth.
    #[test]
    fn the_deepest_piece_is_the_static_contact() {
        // body origin at y = 1/4; balls (radius 1/2) at local x = ±2 turned by a
        // about z: heights 1/4 ∓ 2 sin a for the right and left ball
        let a = fx(1, 10);
        let turn = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, -a);
        let hit = dumbbell()
            .static_contact(
                v(Fix128::ZERO, fx(1, 4), Fix128::ZERO),
                turn,
                &ground_plane(),
            )
            .expect("both balls sunk");
        let deeper = 0.5 - (0.25 - 2.0 * 0.099_833_416_646_828_15); // sin(1/10)
        assert_near(hit.depth, deeper, 1e-9, "the lower ball's depth");
        assert!(hit.point_a.x > Fix128::ZERO, "{hit:?}");
    }

    /// A box sunk into a flat height field meets it at the tangent plane; above
    /// it, beyond the grid's two-cell margin on each side, or on an empty or
    /// zero-spacing field there is no contact.
    #[test]
    fn a_box_meets_a_height_field_inside_its_grid_only() {
        let field = |w: u32, d: u32, spacing: Fix128| {
            StaticCollider::HeightField(HeightField::flat(
                w,
                d,
                spacing,
                Vec3Fix::ZERO,
                Fix128::ZERO,
            ))
        };
        let c = cube(fx(1, 2));
        let grid = field(5, 5, Fix128::ONE);
        let hit = c
            .static_contact(
                v(Fix128::from_int(2), fx(1, 4), Fix128::from_int(2)),
                QuatFix::IDENTITY,
                &grid,
            )
            .expect("inside the grid");
        assert_vec_near(hit.normal, [0.0, 1.0, 0.0], 1e-12, "field normal");
        assert_near(hit.depth, 0.25, 1e-12, "field depth");
        assert!(c
            .static_contact(Vec3Fix::from_int(2, 3, 2), QuatFix::IDENTITY, &grid)
            .is_none());
        // the grid spans 0..4 in x and z; the margin is 2 cells
        for (x, z) in [(-10, 2), (20, 2), (2, -10), (2, 20)] {
            assert!(
                c.static_contact(
                    v(Fix128::from_int(x), fx(1, 4), Fix128::from_int(z)),
                    QuatFix::IDENTITY,
                    &grid
                )
                .is_none(),
                "({x}, {z}) is outside the grid"
            );
        }
        let at = v(Fix128::ZERO, fx(1, 4), Fix128::ZERO);
        assert!(c
            .static_contact(at, QuatFix::IDENTITY, &field(0, 5, Fix128::ONE))
            .is_none());
        assert!(c
            .static_contact(at, QuatFix::IDENTITY, &field(5, 0, Fix128::ONE))
            .is_none());
        assert!(c
            .static_contact(at, QuatFix::IDENTITY, &field(5, 5, Fix128::ZERO))
            .is_none());
    }
}
