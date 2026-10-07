//! Ray queries against the geometry a [`PhysicsWorld`] actually collides with.
//!
//! [`PhysicsWorld::raycast`] and [`crate::query::batch_raycast`] test a ray against
//! each body's **bounding sphere** (its collision radius): a ray that passes the
//! corner gap of a box is reported as a hit, the distance to a long box is the
//! distance to its sphere, and planes, height fields, triangle meshes and SDF
//! colliders are not seen at all. The queries here test the real geometry:
//!
//! | geometry | where it comes from | method |
//! |---|---|---|
//! | sphere | a body with a collision radius and no shape | quadratic |
//! | box | [`Shape::Box`], compound box child | slab clipping of 6 planes |
//! | cylinder | [`Shape::Cylinder`] | side quadratic + two caps |
//! | cone | [`Shape::Cone`] | lateral quadratic + base cap |
//! | ellipsoid | [`Shape::Ellipsoid`] | quadratic in scaled coordinates |
//! | wedge | [`Shape::Wedge`] | clipping of its 5 planes |
//! | torus | [`Shape::Torus`] | sphere tracing of the exact distance, then Newton on the quartic |
//! | sphere, capsule | compound children | quadratic (capsule: side + two spheres) |
//! | convex hull | compound children | clipping of the hull's face planes |
//! | plane | [`StaticCollider::Plane`] | [`crate::raycast::ray_plane`] |
//! | height field | [`StaticCollider::HeightField`] | cell walk, exact quadratic on each bilinear cell |
//! | triangle mesh | [`StaticCollider::TriMesh`] | [`TriMesh::raycast`](crate::trimesh::TriMesh::raycast) (BVH + Möller–Trumbore) |
//! | SDF | [`PhysicsWorld::sdf_colliders`] (`std` only) | [`crate::sdf_ccd::ray_march_sdf`] |
//!
//! Bodies are culled with a BVH of their collider boxes, walked by the ray itself
//! (a node is entered only when the ray segment passes through its box), so a
//! query touches the bodies along the ray and not every body in the world.
//!
//! # Conventions
//!
//! - The direction need not be unit length; it is normalized, and `t` is a
//!   **distance** along it. A direction whose squared length is zero in `Fix128`
//!   (every component below about `2⁻³²`) gives no hit.
//! - Hits have `0 ≤ t ≤ max_t`. `max_t ≤ 0` is an empty segment: no hit.
//! - **Solids** (every body shape, compound children, SDFs): a ray whose origin is
//!   strictly inside reports a hit at `t = 0`, at the origin, with normal
//!   `−direction`. A ray that starts outside reports its entry point with the
//!   outward normal there.
//! - **Surfaces** (plane, height field, triangle mesh) have no inside: the first
//!   crossing is reported from either side, and the normal is turned to face the
//!   incoming ray.
//! - A ray lying in a flat face (parallel to it within `2⁻³²`) does not hit that
//!   face; it can still enter through an adjacent face.
//! - Each collider reports at most one hit, its first crossing; a compound body
//!   reports its nearest child.
//! - Ties in `t` are broken by target: bodies (by index), then static colliders
//!   (by index), then SDF colliders (by index). The result does not depend on the
//!   order the BVH visits bodies in.
//!
//! # Precision
//!
//! The analytic shapes are exact up to `Fix128` truncation (`2⁻⁶⁴` per operation)
//! and the CORDIC rotations: distances agree with closed forms to about `1e-12`
//! for scenes of size 1–100 (`tests/analytic_shape_raycast.rs`). The torus stops
//! sphere tracing at `2⁻³²` and refines with Newton on its quartic. An SDF hit is
//! where the field drops below [`SdfCcdConfig::tolerance`] (an `f32` field), so it
//! is up to that tolerance (divided by the cosine of incidence) short of the
//! surface.
//!
//! # Height fields
//!
//! The surface is the one [`HeightField::sample_height`] describes: bilinear over
//! each cell, with the stored heights as absolute world `Y` (`origin.y` is not
//! read, as by every other height-field query). The ray is clipped to the grid's
//! `XZ` footprint `[origin, origin + (n − 1)·spacing]`; a grid with fewer than two
//! points along either axis has no surface. The normal is
//! [`HeightField::sample_normal`] at the hit point.
//!
//! Author: Moroya Sakamoto

use crate::body_collider::BodyCollider;
use crate::bvh::{BvhPrimitive, LinearBvh};
use crate::collider::{Capsule, ConvexHull, AABB};
use crate::compound::{CompoundChild, ShapeRef};
use crate::filter::CollisionFilter;
use crate::heightfield::HeightField;
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::raycast::{ray_plane, Ray};
use crate::sdf_ccd::SdfCcdConfig;
use crate::shape::Shape;
use crate::solver::PhysicsWorld;
use crate::static_collider::StaticCollider;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Denominators below this are treated as zero (a ray parallel to a face): `2⁻³²`.
pub(crate) const PARALLEL_EPSILON: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000_0001_0000_0000,
};

/// Sphere tracing of a torus stops when the distance is below this: `2⁻³²`.
const TORUS_TOLERANCE: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000_0001_0000_0000,
};

/// Sphere tracing steps for a torus before giving up (a grazing ray).
const TORUS_MAX_STEPS: usize = 512;

/// Newton steps on the torus quartic after sphere tracing.
const TORUS_NEWTON_STEPS: usize = 4;

/// The first crossing of a ray with one piece of geometry: the distance along the
/// unit direction and the normal there (unit, see the module conventions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocalHit {
    pub(crate) t: Fix128,
    pub(crate) normal: Vec3Fix,
}

/// The hit at `t = 0` of a ray that starts inside a solid.
fn inside_hit(direction: Vec3Fix) -> LocalHit {
    LocalHit {
        t: Fix128::ZERO,
        normal: -direction,
    }
}

/// Keep the nearer of two hits; on a tie the first one.
pub(crate) fn nearer(a: Option<LocalHit>, b: Option<LocalHit>) -> Option<LocalHit> {
    match (a, b) {
        (Some(x), Some(y)) => Some(if y.t < x.t { y } else { x }),
        (x, None) => x,
        (None, y) => y,
    }
}

/// `t` within the segment `[0, max_t]`.
fn in_segment(t: Fix128, max_t: Fix128) -> bool {
    t >= Fix128::ZERO && t <= max_t
}

/// The real roots of `a·t² + 2·half_b·t + c = 0`, ascending; a linear equation when
/// `|a|` is below [`PARALLEL_EPSILON`], none when that is degenerate too.
fn quadratic_roots(a: Fix128, half_b: Fix128, c: Fix128) -> [Option<Fix128>; 2] {
    if a.abs() < PARALLEL_EPSILON {
        if half_b.abs() < PARALLEL_EPSILON {
            return [None, None];
        }
        return [Some(-c / half_b.double()), None];
    }
    let disc = half_b * half_b - a * c;
    if disc.is_negative() {
        return [None, None];
    }
    let s = disc.sqrt();
    let r0 = (-half_b - s) / a;
    let r1 = (-half_b + s) / a;
    if r0 <= r1 {
        [Some(r0), Some(r1)]
    } else {
        [Some(r1), Some(r0)]
    }
}

/// A ray moved into a solid's frame: `(origin, direction)` in local coordinates.
pub(crate) fn to_local(
    origin: Vec3Fix,
    direction: Vec3Fix,
    center: Vec3Fix,
    rotation: QuatFix,
) -> (Vec3Fix, Vec3Fix) {
    let inv = rotation.conjugate();
    (inv.rotate_vec(origin - center), inv.rotate_vec(direction))
}

/// A local hit turned back into the world.
pub(crate) fn to_world(hit: Option<LocalHit>, rotation: QuatFix) -> Option<LocalHit> {
    hit.map(|h| LocalHit {
        t: h.t,
        normal: rotation.rotate_vec(h.normal),
    })
}

/// A ray against the convex region `n·x ≤ e` of every `(n, e)` (unit `n`), by
/// clipping the segment `[0, max_t]` plane by plane (Cyrus–Beck). The entry
/// normal is the plane the segment enters last.
pub(crate) fn ray_planes(
    origin: Vec3Fix,
    direction: Vec3Fix,
    planes: &[(Vec3Fix, Fix128)],
    max_t: Fix128,
) -> Option<LocalHit> {
    let mut enter: Option<(Fix128, Vec3Fix)> = None;
    let mut exit = max_t;
    for &(n, e) in planes {
        let denom = n.dot(direction);
        let num = e - n.dot(origin);
        if denom.abs() < PARALLEL_EPSILON {
            // Parallel: inside this slab for every t, or outside for every t.
            if num.is_negative() {
                return None;
            }
            continue;
        }
        let t = num / denom;
        if denom.is_negative() {
            if enter.is_none_or(|(te, _)| t > te) {
                enter = Some((t, n));
            }
        } else if t < exit {
            exit = t;
        }
    }
    match enter {
        // No plane is entered: the origin is on the inside of every one.
        None => (exit >= Fix128::ZERO).then(|| inside_hit(direction)),
        Some((t, n)) => {
            if t > exit || exit.is_negative() {
                None
            } else if t.is_negative() {
                Some(inside_hit(direction))
            } else if t > max_t {
                None
            } else {
                Some(LocalHit { t, normal: n })
            }
        }
    }
}

/// A solid sphere.
pub(crate) fn ray_solid_sphere(
    origin: Vec3Fix,
    direction: Vec3Fix,
    center: Vec3Fix,
    radius: Fix128,
    max_t: Fix128,
) -> Option<LocalHit> {
    let oc = origin - center;
    let c = oc.dot(oc) - radius * radius;
    if c.is_negative() {
        return Some(inside_hit(direction));
    }
    let a = direction.dot(direction);
    let half_b = oc.dot(direction);
    let [first, _] = quadratic_roots(a, half_b, c);
    let t = first?;
    if !in_segment(t, max_t) {
        return None;
    }
    let p = oc + direction * t;
    Some(LocalHit {
        t,
        normal: p.normalize(),
    })
}

/// A solid capsule (segment `a`–`b`, radius `r`): the side of the cylinder between
/// the end planes, and the two end spheres.
pub(crate) fn ray_solid_capsule(
    origin: Vec3Fix,
    direction: Vec3Fix,
    capsule: &Capsule,
    max_t: Fix128,
) -> Option<LocalHit> {
    let axis = capsule.b - capsule.a;
    let (u, len) = axis.normalize_with_length();
    if len.is_zero() {
        return ray_solid_sphere(origin, direction, capsule.a, capsule.radius, max_t);
    }
    let ao = origin - capsule.a;
    let s0 = ao.dot(u);
    let closest = capsule.a + u * clamp(s0, Fix128::ZERO, len);
    let r2 = capsule.radius * capsule.radius;
    if (origin - closest).length_squared() < r2 {
        return Some(inside_hit(direction));
    }
    let d_perp = direction - u * direction.dot(u);
    let ao_perp = ao - u * s0;
    let mut side = None;
    for t in quadratic_roots(
        d_perp.dot(d_perp),
        d_perp.dot(ao_perp),
        ao_perp.dot(ao_perp) - r2,
    )
    .into_iter()
    .flatten()
    {
        if !in_segment(t, max_t) {
            continue;
        }
        let s = s0 + direction.dot(u) * t;
        if s >= Fix128::ZERO && s <= len {
            side = Some(LocalHit {
                t,
                normal: (ao_perp + d_perp * t).normalize(),
            });
            break;
        }
    }
    let cap_a = ray_solid_sphere(origin, direction, capsule.a, capsule.radius, max_t);
    let cap_b = ray_solid_sphere(origin, direction, capsule.b, capsule.radius, max_t);
    nearer(nearer(side, cap_a), cap_b)
}

fn clamp(x: Fix128, lo: Fix128, hi: Fix128) -> Fix128 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Unit axis vectors and their negatives as the 6 planes of a box of half-extents
/// `h` centred at the origin.
pub(crate) fn box_planes(h: Vec3Fix) -> [(Vec3Fix, Fix128); 6] {
    [
        (Vec3Fix::UNIT_X, h.x),
        (-Vec3Fix::UNIT_X, h.x),
        (Vec3Fix::UNIT_Y, h.y),
        (-Vec3Fix::UNIT_Y, h.y),
        (Vec3Fix::UNIT_Z, h.z),
        (-Vec3Fix::UNIT_Z, h.z),
    ]
}

/// A solid oriented box.
fn ray_box(
    origin: Vec3Fix,
    direction: Vec3Fix,
    center: Vec3Fix,
    half_extents: Vec3Fix,
    rotation: QuatFix,
    max_t: Fix128,
) -> Option<LocalHit> {
    let (o, d) = to_local(origin, direction, center, rotation);
    to_world(ray_planes(o, d, &box_planes(half_extents), max_t), rotation)
}

/// A solid cylinder along local `Y`, in local coordinates.
pub(crate) fn ray_local_cylinder(
    o: Vec3Fix,
    d: Vec3Fix,
    radius: Fix128,
    half_height: Fix128,
    max_t: Fix128,
) -> Option<LocalHit> {
    let r2 = radius * radius;
    if o.x * o.x + o.z * o.z < r2 && o.y.abs() < half_height {
        return Some(inside_hit(d));
    }
    let mut best = None;
    for t in quadratic_roots(
        d.x * d.x + d.z * d.z,
        o.x * d.x + o.z * d.z,
        o.x * o.x + o.z * o.z - r2,
    )
    .into_iter()
    .flatten()
    {
        if in_segment(t, max_t) && (o.y + d.y * t).abs() <= half_height {
            let p = o + d * t;
            best = Some(LocalHit {
                t,
                normal: Vec3Fix::new(p.x, Fix128::ZERO, p.z).normalize(),
            });
            break;
        }
    }
    if d.y.abs() >= PARALLEL_EPSILON {
        for (y, ny) in [(half_height, Fix128::ONE), (-half_height, Fix128::NEG_ONE)] {
            let t = (y - o.y) / d.y;
            if !in_segment(t, max_t) {
                continue;
            }
            let p = o + d * t;
            if p.x * p.x + p.z * p.z <= r2 {
                best = nearer(
                    best,
                    Some(LocalHit {
                        t,
                        normal: Vec3Fix::new(Fix128::ZERO, ny, Fix128::ZERO),
                    }),
                );
            }
        }
    }
    best
}

/// A solid cone along local `Y`, apex at `+half_height`, base of `radius` at
/// `−half_height`, in local coordinates (centred on its geometric centre).
fn ray_local_cone(
    o: Vec3Fix,
    d: Vec3Fix,
    radius: Fix128,
    half_height: Fix128,
    max_t: Fix128,
) -> Option<LocalHit> {
    // Lateral surface: x² + z² = k²·(h − y)², k = r / 2h, for −h ≤ y ≤ h.
    let height = half_height.double();
    let k = radius / height;
    let k2 = k * k;
    let apex_gap = half_height - o.y; // Y = h − y at the origin
    let rr = o.x * o.x + o.z * o.z;
    if apex_gap > Fix128::ZERO && apex_gap < height && rr < k2 * apex_gap * apex_gap {
        return Some(inside_hit(d));
    }
    let mut best = None;
    for t in quadratic_roots(
        d.x * d.x + d.z * d.z - k2 * d.y * d.y,
        o.x * d.x + o.z * d.z + k2 * apex_gap * d.y,
        rr - k2 * apex_gap * apex_gap,
    )
    .into_iter()
    .flatten()
    {
        if !in_segment(t, max_t) {
            continue;
        }
        let gap = apex_gap - d.y * t;
        if gap >= Fix128::ZERO && gap <= height {
            let p = o + d * t;
            let normal = Vec3Fix::new(p.x, k2 * gap, p.z).normalize();
            best = Some(LocalHit {
                t,
                normal: if normal == Vec3Fix::ZERO {
                    Vec3Fix::UNIT_Y
                } else {
                    normal
                },
            });
            break;
        }
    }
    if d.y.abs() >= PARALLEL_EPSILON {
        let t = (-half_height - o.y) / d.y;
        if in_segment(t, max_t) {
            let p = o + d * t;
            if p.x * p.x + p.z * p.z <= radius * radius {
                best = nearer(
                    best,
                    Some(LocalHit {
                        t,
                        normal: -Vec3Fix::UNIT_Y,
                    }),
                );
            }
        }
    }
    best
}

/// A solid ellipsoid of semi-axes `radii`, in local coordinates.
fn ray_local_ellipsoid(o: Vec3Fix, d: Vec3Fix, radii: Vec3Fix, max_t: Fix128) -> Option<LocalHit> {
    let os = Vec3Fix::new(o.x / radii.x, o.y / radii.y, o.z / radii.z);
    let ds = Vec3Fix::new(d.x / radii.x, d.y / radii.y, d.z / radii.z);
    let c = os.dot(os) - Fix128::ONE;
    if c.is_negative() {
        return Some(inside_hit(d));
    }
    let [first, _] = quadratic_roots(ds.dot(ds), os.dot(ds), c);
    let t = first?;
    if !in_segment(t, max_t) {
        return None;
    }
    let p = o + d * t;
    Some(LocalHit {
        t,
        normal: Vec3Fix::new(
            p.x / (radii.x * radii.x),
            p.y / (radii.y * radii.y),
            p.z / (radii.z * radii.z),
        )
        .normalize(),
    })
}

/// The 5 planes of a wedge (see [`crate::wedge`]) about its geometric centre.
fn wedge_planes(width: Fix128, height: Fix128, depth: Fix128) -> [(Vec3Fix, Fix128); 5] {
    let hw = width.half();
    let hh = height.half();
    let hd = depth.half();
    // A slanted face runs from the base corner (±w/2, −h/2) to the apex (0, h/2):
    // its outward normal is (±h, w/2, 0), through the apex.
    let right = Vec3Fix::new(height, hw, Fix128::ZERO).normalize();
    let left = Vec3Fix::new(-height, hw, Fix128::ZERO).normalize();
    [
        (-Vec3Fix::UNIT_Y, hh),
        (Vec3Fix::UNIT_Z, hd),
        (-Vec3Fix::UNIT_Z, hd),
        (right, right.y * hh),
        (left, left.y * hh),
    ]
}

/// The exact signed distance of a torus (ring of `major` in the local `XZ` plane,
/// tube of `minor`).
pub(crate) fn torus_distance(p: Vec3Fix, major: Fix128, minor: Fix128) -> Fix128 {
    let ring = (p.x * p.x + p.z * p.z).sqrt() - major;
    (ring * ring + p.y * p.y).sqrt() - minor
}

/// A solid torus, in local coordinates.
pub(crate) fn ray_local_torus(
    o: Vec3Fix,
    d: Vec3Fix,
    major: Fix128,
    minor: Fix128,
    max_t: Fix128,
) -> Option<LocalHit> {
    if torus_distance(o, major, minor).is_negative() {
        return Some(inside_hit(d));
    }
    // Start where the ray enters the bounding sphere (no surface before it).
    let bound = ray_solid_sphere(o, d, Vec3Fix::ZERO, major + minor, max_t)?;
    let mut t = bound.t;
    let mut converged = false;
    for _ in 0..TORUS_MAX_STEPS {
        let dist = torus_distance(o + d * t, major, minor);
        if dist < TORUS_TOLERANCE {
            converged = true;
            break;
        }
        t = t + dist;
        if t > max_t {
            return None;
        }
    }
    if !converged {
        return None;
    }
    // Newton on F(t) = (|p|² + R² − r²)² − 4R²(x² + z²).
    let r2 = major * major;
    let k = r2 - minor * minor;
    let four_r2 = r2.double().double();
    for _ in 0..TORUS_NEWTON_STEPS {
        let p = o + d * t;
        let s = p.dot(p) + k;
        let f = s * s - four_r2 * (p.x * p.x + p.z * p.z);
        let df = (s * p.dot(d)).double().double() - (four_r2 * (p.x * d.x + p.z * d.z)).double();
        if df.abs() < PARALLEL_EPSILON {
            break;
        }
        t = t - f / df;
    }
    if !in_segment(t, max_t) {
        return None;
    }
    let p = o + d * t;
    let (ring_dir, _) = Vec3Fix::new(p.x, Fix128::ZERO, p.z).normalize_with_length();
    Some(LocalHit {
        t,
        normal: (p - ring_dir * major).normalize(),
    })
}

/// A solid [`Shape`] whose centre of mass is at `position`, turned by `rotation`
/// (see [`crate::shape`] for the frames).
fn ray_posed_shape(
    origin: Vec3Fix,
    direction: Vec3Fix,
    shape: &Shape,
    position: Vec3Fix,
    rotation: QuatFix,
    max_t: Fix128,
) -> Option<LocalHit> {
    let rotation = rotation.unit_rotation();
    let center = position - rotation.rotate_vec(shape.center_of_mass_offset());
    let (o, d) = to_local(origin, direction, center, rotation);
    let hit = match *shape {
        Shape::Box { half_extents } => ray_planes(o, d, &box_planes(half_extents), max_t),
        Shape::Cylinder {
            radius,
            half_height,
        } => ray_local_cylinder(o, d, radius, half_height, max_t),
        Shape::Cone {
            radius,
            half_height,
        } => ray_local_cone(o, d, radius, half_height, max_t),
        Shape::Ellipsoid { radii } => ray_local_ellipsoid(o, d, radii, max_t),
        Shape::Wedge {
            width,
            height,
            depth,
        } => ray_planes(o, d, &wedge_planes(width, height, depth), max_t),
        Shape::Torus {
            major_radius,
            minor_radius,
        } => ray_local_torus(o, d, major_radius, minor_radius, max_t),
    };
    to_world(hit, rotation)
}

/// The face planes of a convex hull (unit outward normals), or `None` when its
/// points span no volume.
pub(crate) fn hull_planes(hull: &ConvexHull) -> Option<Vec<(Vec3Fix, Fix128)>> {
    let mesh = crate::convex_mesh_builder::build_hull_mesh(&hull.vertices)?;
    Some(
        mesh.faces
            .iter()
            .filter_map(|&[a, b, c]| {
                let (va, vb, vc) = (mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]);
                let n = (vb - va).cross(vc - va).try_normalize()?;
                Some((n, n.dot(va)))
            })
            .collect(),
    )
}

/// One compound child of a body at `body_pos` turned by `body_rot`, placed the way
/// [`CompoundChild::support_world`] places it.
fn ray_compound_child(
    origin: Vec3Fix,
    direction: Vec3Fix,
    child: &CompoundChild,
    body_pos: Vec3Fix,
    body_rot: QuatFix,
    max_t: Fix128,
) -> Option<LocalHit> {
    let body_rot = body_rot.unit_rotation();
    let child_rot = body_rot.mul(child.local_rotation);
    let child_pos = body_pos + body_rot.rotate_vec(child.local_position);
    match &child.shape {
        ShapeRef::Sphere(s) => ray_solid_sphere(
            origin,
            direction,
            child_pos + child_rot.rotate_vec(s.center),
            s.radius,
            max_t,
        ),
        ShapeRef::Capsule(c) => ray_solid_capsule(
            origin,
            direction,
            &Capsule::new(
                child_pos + child_rot.rotate_vec(c.a),
                child_pos + child_rot.rotate_vec(c.b),
                c.radius,
            ),
            max_t,
        ),
        ShapeRef::Box(b) => ray_box(
            origin,
            direction,
            child_pos + child_rot.rotate_vec(b.center),
            b.half_extents,
            child_rot.mul(b.rotation),
            max_t,
        ),
        ShapeRef::ConvexHull(hull) => {
            let planes = hull_planes(hull)?;
            let (o, d) = to_local(origin, direction, child_pos, child_rot);
            to_world(ray_planes(o, d, &planes, max_t), child_rot)
        }
    }
}

/// The surface of a height field (see the module doc).
fn ray_heightfield(
    origin: Vec3Fix,
    direction: Vec3Fix,
    field: &HeightField,
    max_t: Fix128,
) -> Option<LocalHit> {
    if field.width < 2 || field.depth < 2 || field.spacing <= Fix128::ZERO {
        return None;
    }
    let s = field.spacing;
    let x0 = field.origin.x;
    let z0 = field.origin.z;
    let x1 = x0 + s * Fix128::from_int(i64::from(field.width - 1));
    let z1 = z0 + s * Fix128::from_int(i64::from(field.depth - 1));
    // Clip the segment to the footprint in XZ.
    let mut t_lo = Fix128::ZERO;
    let mut t_hi = max_t;
    for (o, d, lo, hi) in [
        (origin.x, direction.x, x0, x1),
        (origin.z, direction.z, z0, z1),
    ] {
        if d.abs() < PARALLEL_EPSILON {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let (mut a, mut b) = ((lo - o) / d, (hi - o) / d);
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        if a > t_lo {
            t_lo = a;
        }
        if b < t_hi {
            t_hi = b;
        }
    }
    if t_lo > t_hi {
        return None;
    }
    let last_x = field.width - 2;
    let last_z = field.depth - 2;
    let cell_of = |w: Fix128, w0: Fix128, last: u32| -> u32 {
        let g = (w - w0) / s;
        if g.is_negative() {
            0
        } else {
            (g.hi.min(i64::from(last))) as u32
        }
    };
    let p_start = origin + direction * t_lo;
    let mut cx = cell_of(p_start.x, x0, last_x);
    let mut cz = cell_of(p_start.z, z0, last_z);
    let mut t_cell = t_lo;
    // Each step leaves through at least one cell wall, so the walk is bounded by
    // the number of cells the footprint has along X plus along Z.
    let max_steps = (field.width as usize) + (field.depth as usize) + 2;
    for _ in 0..max_steps {
        let cell_x0 = x0 + s * Fix128::from_int(i64::from(cx));
        let cell_z0 = z0 + s * Fix128::from_int(i64::from(cz));
        // Where the ray leaves this cell (or the clipped segment).
        let mut t_exit = t_hi;
        let mut step_x = 0i32;
        let mut step_z = 0i32;
        if direction.x.abs() >= PARALLEL_EPSILON {
            let wall = if direction.x.is_negative() {
                cell_x0
            } else {
                cell_x0 + s
            };
            let t = (wall - origin.x) / direction.x;
            if t < t_exit {
                t_exit = t;
                step_x = if direction.x.is_negative() { -1 } else { 1 };
                step_z = 0;
            } else if t == t_exit {
                step_x = if direction.x.is_negative() { -1 } else { 1 };
            }
        }
        if direction.z.abs() >= PARALLEL_EPSILON {
            let wall = if direction.z.is_negative() {
                cell_z0
            } else {
                cell_z0 + s
            };
            let t = (wall - origin.z) / direction.z;
            if t < t_exit {
                t_exit = t;
                step_z = if direction.z.is_negative() { -1 } else { 1 };
                step_x = 0;
            } else if t == t_exit {
                step_z = if direction.z.is_negative() { -1 } else { 1 };
            }
        }
        if t_exit < t_cell {
            t_exit = t_cell;
        }
        if let Some(hit) = ray_bilinear_cell(
            origin,
            direction,
            field,
            (cx, cz),
            (cell_x0, cell_z0),
            t_cell,
            t_exit,
        ) {
            return Some(hit);
        }
        if t_exit >= t_hi {
            return None;
        }
        let nx = i64::from(cx) + i64::from(step_x);
        let nz = i64::from(cz) + i64::from(step_z);
        if nx < 0 || nz < 0 || nx > i64::from(last_x) || nz > i64::from(last_z) {
            return None;
        }
        cx = nx as u32;
        cz = nz as u32;
        t_cell = t_exit;
    }
    None
}

/// The first crossing in `[t0, t1]` of the ray with the bilinear patch of one cell.
#[allow(clippy::too_many_arguments)]
fn ray_bilinear_cell(
    origin: Vec3Fix,
    direction: Vec3Fix,
    field: &HeightField,
    (cx, cz): (u32, u32),
    (cell_x0, cell_z0): (Fix128, Fix128),
    t0: Fix128,
    t1: Fix128,
) -> Option<LocalHit> {
    let h00 = field.get_height(cx, cz);
    let h10 = field.get_height(cx + 1, cz);
    let h01 = field.get_height(cx, cz + 1);
    let h11 = field.get_height(cx + 1, cz + 1);
    let b = h10 - h00;
    let c = h01 - h00;
    let dd = h00 - h10 - h01 + h11;
    // Parametrize by u = t − t0 so the numbers stay cell-sized.
    let p0 = origin + direction * t0;
    let inv_s = Fix128::ONE / field.spacing;
    let ax = (p0.x - cell_x0) * inv_s;
    let az = (p0.z - cell_z0) * inv_s;
    let bx = direction.x * inv_s;
    let bz = direction.z * inv_s;
    // f(u) = y(u) − h(u) = q0 + q1·u + q2·u²
    let q0 = p0.y - h00 - b * ax - c * az - dd * ax * az;
    let q1 = direction.y - b * bx - c * bz - dd * (ax * bz + bx * az);
    let q2 = -(dd * bx * bz);
    let span = t1 - t0;
    if q0.is_zero() && q1.is_zero() && q2.is_zero() {
        // The ray lies in the cell's surface (a flat cell, ray in its plane).
        return None;
    }
    let root = if q0.is_zero() {
        Some(Fix128::ZERO)
    } else {
        quadratic_roots(q2, q1.half(), q0)
            .into_iter()
            .flatten()
            .find(|&u| u >= Fix128::ZERO && u <= span)
    }?;
    let t = t0 + root;
    let p = origin + direction * t;
    let n = field.sample_normal(p.x, p.z);
    Some(LocalHit {
        t,
        normal: if n.dot(direction) > Fix128::ZERO {
            -n
        } else {
            n
        },
    })
}

/// A static surface.
fn ray_static(
    origin: Vec3Fix,
    direction: Vec3Fix,
    collider: &StaticCollider,
    max_t: Fix128,
) -> Option<LocalHit> {
    let ray = Ray { origin, direction };
    match collider {
        StaticCollider::Plane(plane) => {
            ray_plane(&ray, plane.normal, plane.offset, max_t).map(|h| LocalHit {
                t: h.t,
                normal: h.normal,
            })
        }
        StaticCollider::HeightField(field) => ray_heightfield(origin, direction, field, max_t),
        StaticCollider::TriMesh(mesh) => mesh.raycast(&ray, max_t).map(|h| LocalHit {
            t: h.t,
            normal: h.normal,
        }),
    }
}

/// An SDF collider: inside (negative distance at the origin) is a hit at `t = 0`;
/// otherwise [`crate::sdf_ccd::ray_march_sdf`].
#[cfg(feature = "std")]
fn ray_sdf(
    origin: Vec3Fix,
    direction: Vec3Fix,
    sdf: &crate::sdf_collider::SdfCollider,
    config: &SdfCcdConfig,
    max_t: Fix128,
) -> Option<LocalHit> {
    let (lx, ly, lz) = sdf.world_to_local(origin);
    if sdf.field.distance(lx, ly, lz) * sdf.scale_f32 < 0.0 {
        return Some(inside_hit(direction));
    }
    let toi = crate::sdf_ccd::ray_march_sdf(origin, direction, max_t, sdf, config)?;
    in_segment(toi.t, max_t).then_some(LocalHit {
        t: toi.t,
        normal: toi.normal,
    })
}

// ============================================================================
// Public types
// ============================================================================

/// What a ray hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum RayTarget {
    /// A body, by index: its shape, its compound children, or its collision sphere.
    Body(usize),
    /// A static collider ([`PhysicsWorld::add_static_collider`]), by index.
    StaticCollider(usize),
    /// An SDF collider ([`PhysicsWorld::sdf_colliders`]), by index.
    Sdf(usize),
}

/// A hit of [`PhysicsWorld::cast_ray`] and its variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct WorldRayHit {
    /// Distance along the (normalized) direction.
    pub t: Fix128,
    /// World-space hit point, `origin + direction·t`.
    pub point: Vec3Fix,
    /// Unit normal at the hit (see the module conventions for its side).
    pub normal: Vec3Fix,
    /// The collider hit.
    pub target: RayTarget,
    /// The body the hit belongs to: the body for [`RayTarget::Body`], the body an
    /// SDF collider is attached to (if any) for [`RayTarget::Sdf`], `None` for a
    /// static collider.
    pub body: Option<usize>,
}

/// Which colliders a ray query sees.
///
/// The default sees every body on any layer except sensors, every static collider
/// and every SDF collider. A body is seen when `layer_mask & body_filter.layer != 0`
/// ([`PhysicsWorld::body_filter`]); an SDF collider attached to a body is seen only
/// when that body is.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct RayFilter {
    /// Layers the ray sees (bitmask against [`CollisionFilter::layer`]).
    pub layer_mask: u32,
    /// A body the ray ignores (typically the one carrying the sensor).
    pub exclude_body: Option<usize>,
    /// Whether sensor bodies are seen.
    pub include_sensors: bool,
    /// Whether static colliders are seen.
    pub include_static: bool,
    /// Whether SDF colliders are seen.
    pub include_sdf: bool,
    /// Sphere-tracing settings for SDF colliders.
    pub sdf: SdfCcdConfig,
}

impl Default for RayFilter {
    fn default() -> Self {
        Self {
            layer_mask: u32::MAX,
            exclude_body: None,
            include_sensors: false,
            include_static: true,
            include_sdf: true,
            sdf: SdfCcdConfig::default(),
        }
    }
}

impl RayFilter {
    /// The default filter (see the type doc).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Only bodies on these layers.
    #[must_use]
    pub const fn with_layer_mask(mut self, mask: u32) -> Self {
        self.layer_mask = mask;
        self
    }

    /// Ignore this body (and any SDF collider attached to it).
    #[must_use]
    pub const fn excluding_body(mut self, body: usize) -> Self {
        self.exclude_body = Some(body);
        self
    }

    /// Whether sensor bodies are seen.
    #[must_use]
    pub const fn with_sensors(mut self, include: bool) -> Self {
        self.include_sensors = include;
        self
    }

    /// Whether static colliders are seen.
    #[must_use]
    pub const fn with_static(mut self, include: bool) -> Self {
        self.include_static = include;
        self
    }

    /// Whether SDF colliders are seen.
    #[must_use]
    pub const fn with_sdf(mut self, include: bool) -> Self {
        self.include_sdf = include;
        self
    }

    /// Sphere-tracing settings for SDF colliders.
    #[must_use]
    pub const fn with_sdf_config(mut self, config: SdfCcdConfig) -> Self {
        self.sdf = config;
        self
    }

    pub(crate) fn sees_body(&self, world: &PhysicsWorld, i: usize) -> bool {
        let Some(body) = world.bodies.get(i) else {
            return false;
        };
        if self.exclude_body == Some(i) || (body.is_sensor && !self.include_sensors) {
            return false;
        }
        let filter: CollisionFilter = world.body_filter(i);
        filter.layer & self.layer_mask != 0
    }
}

/// A ray query against one world, with the body BVH built once: cast many rays
/// (a lidar scan) without rebuilding it. Made by [`PhysicsWorld::ray_caster`].
pub struct WorldRayCaster<'w> {
    world: &'w PhysicsWorld,
    filter: RayFilter,
    bvh: LinearBvh,
    /// Each body's collider box, by body index (unused for bodies not in the tree).
    boxes: Vec<AABB>,
}

impl PhysicsWorld {
    /// A ray caster over this world's current state with `filter`; the body BVH is
    /// built here, once.
    #[must_use]
    pub fn ray_caster(&self, filter: RayFilter) -> WorldRayCaster<'_> {
        let (bvh, boxes) = self.query_body_bvh(&filter);
        WorldRayCaster {
            world: self,
            filter,
            bvh,
            boxes,
        }
    }

    /// The BVH of the collider boxes of the bodies `filter` sees, and each body's
    /// box by body index (unused for bodies not in the tree).
    pub(crate) fn query_body_bvh(&self, filter: &RayFilter) -> (LinearBvh, Vec<AABB>) {
        let mut primitives = Vec::new();
        let mut boxes = vec![AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO); self.bodies.len()];
        for (i, slot) in boxes.iter_mut().enumerate() {
            if !filter.sees_body(self, i) {
                continue;
            }
            let Some((radius, collider)) = self.ray_geometry(i) else {
                continue;
            };
            let body = &self.bodies[i];
            let aabb = match collider {
                Some(c) => c.world_aabb(body.position, body.rotation),
                None => AABB::from_center_half(body.position, Vec3Fix::new(radius, radius, radius)),
            };
            *slot = aabb;
            primitives.push(BvhPrimitive {
                aabb,
                index: i as u32,
                morton: 0,
            });
        }
        (LinearBvh::build(primitives), boxes)
    }

    /// The nearest hit of a ray with the world's geometry (see [`crate::shape_raycast`]).
    #[must_use]
    pub fn cast_ray(
        &self,
        origin: Vec3Fix,
        direction: Vec3Fix,
        max_t: Fix128,
        filter: &RayFilter,
    ) -> Option<WorldRayHit> {
        self.ray_caster(*filter).closest(origin, direction, max_t)
    }

    /// Every collider the ray hits, one hit each, nearest first.
    #[must_use]
    pub fn cast_ray_all(
        &self,
        origin: Vec3Fix,
        direction: Vec3Fix,
        max_t: Fix128,
        filter: &RayFilter,
    ) -> Vec<WorldRayHit> {
        self.ray_caster(*filter).all(origin, direction, max_t)
    }

    /// Whether the ray hits anything (stops at the first hit found).
    #[must_use]
    pub fn cast_ray_any(
        &self,
        origin: Vec3Fix,
        direction: Vec3Fix,
        max_t: Fix128,
        filter: &RayFilter,
    ) -> bool {
        self.ray_caster(*filter).any(origin, direction, max_t)
    }
}

/// The unit direction of a query, or `None` for an empty one.
fn query_direction(direction: Vec3Fix, max_t: Fix128) -> Option<Vec3Fix> {
    if max_t <= Fix128::ZERO {
        return None;
    }
    direction.try_normalize()
}

/// Whether the ray segment `[0, max_t]` passes through `aabb`.
fn segment_meets_box(origin: Vec3Fix, direction: Vec3Fix, aabb: &AABB, max_t: Fix128) -> bool {
    let mut t0 = Fix128::ZERO;
    let mut t1 = max_t;
    for (o, d, lo, hi) in [
        (origin.x, direction.x, aabb.min.x, aabb.max.x),
        (origin.y, direction.y, aabb.min.y, aabb.max.y),
        (origin.z, direction.z, aabb.min.z, aabb.max.z),
    ] {
        if d.is_zero() {
            if o < lo || o > hi {
                return false;
            }
            continue;
        }
        let (mut a, mut b) = ((lo - o) / d, (hi - o) / d);
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        if a > t0 {
            t0 = a;
        }
        if b < t1 {
            t1 = b;
        }
        if t0 > t1 {
            return false;
        }
    }
    true
}

impl WorldRayCaster<'_> {
    /// Bodies whose collider box the ray segment passes through, found by walking
    /// the BVH (a subtree is skipped when the segment misses its box), in the order
    /// the walk visits them: the only bodies the narrow phase tests. Empty for an
    /// empty query (zero direction, `max_t ≤ 0`).
    #[must_use]
    pub fn candidates(&self, origin: Vec3Fix, direction: Vec3Fix, max_t: Fix128) -> Vec<usize> {
        let mut out = Vec::new();
        let Some(dir) = query_direction(direction, max_t) else {
            return out;
        };
        let nodes = &self.bvh.nodes;
        let mut idx = 0u32;
        while (idx as usize) < nodes.len() {
            let node = &nodes[idx as usize];
            if segment_meets_box(origin, dir, &node.get_aabb(), max_t) {
                if node.is_leaf() {
                    let start = node.first_child_or_prim as usize;
                    let end = start + node.prim_count() as usize;
                    for &p in self.bvh.primitives.get(start..end).unwrap_or(&[]) {
                        let i = p as usize;
                        if self
                            .boxes
                            .get(i)
                            .is_some_and(|b| segment_meets_box(origin, dir, b, max_t))
                        {
                            out.push(i);
                        }
                    }
                    idx = node.escape_idx();
                } else {
                    idx = node.first_child_or_prim;
                }
            } else {
                idx = node.escape_idx();
            }
        }
        out
    }

    /// One body's geometry.
    fn body_hit(&self, i: usize, origin: Vec3Fix, dir: Vec3Fix, max_t: Fix128) -> Option<LocalHit> {
        let (radius, collider) = self.world.ray_geometry(i)?;
        let body = &self.world.bodies[i];
        match collider {
            None => ray_solid_sphere(origin, dir, body.position, radius, max_t),
            Some(BodyCollider::Shape(shape)) => {
                ray_posed_shape(origin, dir, shape, body.position, body.rotation, max_t)
            }
            Some(BodyCollider::Compound(compound)) => {
                let mut best = None;
                for child in &compound.children {
                    best = nearer(
                        best,
                        ray_compound_child(origin, dir, child, body.position, body.rotation, max_t),
                    );
                }
                best
            }
        }
    }

    /// Visit every hit (any order); `visit` returns `false` to stop.
    fn for_each_hit(
        &self,
        origin: Vec3Fix,
        direction: Vec3Fix,
        max_t: Fix128,
        mut visit: impl FnMut(WorldRayHit) -> bool,
    ) {
        let Some(dir) = query_direction(direction, max_t) else {
            return;
        };
        let make = |h: LocalHit, target: RayTarget, body: Option<usize>| WorldRayHit {
            t: h.t,
            point: origin + dir * h.t,
            normal: h.normal,
            target,
            body,
        };
        for i in self.candidates(origin, dir, max_t) {
            if let Some(h) = self.body_hit(i, origin, dir, max_t) {
                if !visit(make(h, RayTarget::Body(i), Some(i))) {
                    return;
                }
            }
        }
        if self.filter.include_static {
            for (j, collider) in self.world.static_colliders_slice().iter().enumerate() {
                if let Some(h) = ray_static(origin, dir, collider, max_t) {
                    if !visit(make(h, RayTarget::StaticCollider(j), None)) {
                        return;
                    }
                }
            }
        }
        #[cfg(feature = "std")]
        if self.filter.include_sdf {
            for (k, sdf) in self.world.sdf_colliders.iter().enumerate() {
                let body = (sdf.body_index < self.world.bodies.len()).then_some(sdf.body_index);
                if body.is_some_and(|b| !self.filter.sees_body(self.world, b)) {
                    continue;
                }
                if let Some(h) = ray_sdf(origin, dir, sdf, &self.filter.sdf, max_t) {
                    if !visit(make(h, RayTarget::Sdf(k), body)) {
                        return;
                    }
                }
            }
        }
    }

    /// The nearest hit (ties: see the module conventions).
    #[must_use]
    pub fn closest(
        &self,
        origin: Vec3Fix,
        direction: Vec3Fix,
        max_t: Fix128,
    ) -> Option<WorldRayHit> {
        let mut best: Option<WorldRayHit> = None;
        self.for_each_hit(origin, direction, max_t, |h| {
            if best.is_none_or(|b| (h.t, h.target) < (b.t, b.target)) {
                best = Some(h);
            }
            true
        });
        best
    }

    /// Every hit, one per collider, sorted by `t` (ties: see the module conventions).
    #[must_use]
    pub fn all(&self, origin: Vec3Fix, direction: Vec3Fix, max_t: Fix128) -> Vec<WorldRayHit> {
        let mut hits = Vec::new();
        self.for_each_hit(origin, direction, max_t, |h| {
            hits.push(h);
            true
        });
        hits.sort_by_key(|h| (h.t, h.target));
        hits
    }

    /// Whether anything is hit.
    #[must_use]
    pub fn any(&self, origin: Vec3Fix, direction: Vec3Fix, max_t: Fix128) -> bool {
        let mut found = false;
        self.for_each_hit(origin, direction, max_t, |_| {
            found = true;
            false
        });
        found
    }
}
