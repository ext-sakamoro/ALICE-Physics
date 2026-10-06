//! Swept and overlap queries against the geometry a [`PhysicsWorld`] actually
//! collides with.
//!
//! [`crate::query::sphere_cast`], [`crate::query::capsule_cast`],
//! [`crate::query::overlap_sphere`] and [`crate::query::overlap_aabb`] test each
//! body's bounding sphere (or its position) and see no static or SDF collider.
//! The world methods here test the same geometry as
//! [`PhysicsWorld::cast_ray`] (see [`crate::shape_raycast`]): body shapes,
//! compound children, planes, height fields, triangle meshes and (`std` only) SDF
//! colliders.
//!
//! | method | query |
//! |---|---|
//! | [`PhysicsWorld::cast_sphere`] | a sphere swept along a direction, nearest hit |
//! | [`PhysicsWorld::cast_capsule`] | a capsule (segment `a`–`b` grown by a radius) swept along a direction, nearest hit |
//! | [`PhysicsWorld::overlap_sphere`] | every collider a sphere overlaps |
//! | [`PhysicsWorld::overlap_aabb`] | every collider an axis-aligned box overlaps |
//!
//! # How each geometry is tested
//!
//! A swept sphere of radius `r` first touches a solid `S` where its centre
//! reaches the boundary of the Minkowski sum `S ⊕ ball(r)`; a ray against that sum
//! is used wherever it has a closed form:
//!
//! | geometry | sphere cast | capsule cast |
//! |---|---|---|
//! | sphere (body radius, compound child) | ray vs sphere of `R + r` | ray from the sphere's centre along `−direction` vs the capsule grown by `R` |
//! | box (shape, compound child) | ray vs rounded box (3 slabs + 12 edge capsules) | conservative advancement, GJK distance |
//! | cylinder | ray vs rounded cylinder (2 cylinders + 2 rim tori) | conservative advancement, GJK distance |
//! | torus | ray vs torus of tube `minor + r` | conservative advancement, segment–torus distance |
//! | compound capsule | ray vs capsule of `radius + r` | conservative advancement, GJK distance |
//! | cone, ellipsoid, wedge, convex hull child | conservative advancement, GJK distance | conservative advancement, GJK distance |
//! | plane | closed form (two-sided) | closed form at the nearer end |
//! | triangle mesh | per triangle: the two offset triangles + 3 edge capsules | conservative advancement, GJK distance per triangle |
//! | height field | conservative advancement, distance to the bilinear cells | conservative advancement, segment–cell distance |
//! | SDF (`std`) | sphere tracing of `field − r` | sphere tracing of the field minimum along the segment |
//!
//! Overlaps use the distance from the sphere's centre to the geometry (closed form
//! for spheres, capsules, boxes, cylinders, tori, planes and triangles, GJK for the
//! other convex pieces). A box overlaps a convex piece when GJK finds them
//! intersecting, a plane when the plane passes between its corners, a height field
//! when the surface heights over the box's footprint (the bilinear surface takes
//! its extremes at the corners of each cell's part of the footprint) reach into
//! its `Y` range, and an SDF when an octree subdivision of the box finds a point
//! inside.
//!
//! # Conventions
//!
//! Those of [`crate::shape_raycast`], with these additions:
//!
//! - The direction need not be unit length; it is normalized and `t` is a
//!   **distance** along it. A zero direction or `max_t ≤ 0` gives no hit.
//! - A negative radius is not a sphere: no hit, no overlap. Radius `0` is allowed:
//!   [`PhysicsWorld::cast_sphere`] with radius `0` **is** [`PhysicsWorld::cast_ray`]
//!   (same target, `t`, point and normal), and an overlap of radius `0` is a point
//!   strictly inside a solid.
//! - A capsule whose two ends coincide is a sphere: [`PhysicsWorld::cast_capsule`]
//!   then returns [`PhysicsWorld::cast_sphere`].
//! - A cast that starts overlapping a collider reports it at `t = 0`, normal
//!   `−direction`, point the cast shape's centre (the sphere's centre, the
//!   capsule segment's midpoint). Otherwise the hit's point is the contact point on
//!   the collider and the normal is the collider's surface normal there, pointing
//!   toward the cast shape (for surfaces: toward the side the shape comes from).
//! - Touching is not overlapping: an overlap needs a distance below the radius
//!   (or, for boxes against convex pieces, a GJK intersection).
//! - Ties in `t` are broken by target (bodies, then static colliders, then SDF
//!   colliders, each by index); overlaps are returned sorted by target. Neither
//!   depends on the order the BVH visits bodies in.
//! - An [`AABB`] with `min > max` on any axis contains nothing.
//!
//! # Precision
//!
//! Closed-form paths are exact up to `Fix128` truncation and the CORDIC rotations,
//! like the ray queries (`tests/analytic_world_shape_query.rs` checks them to
//! `1e-12`). Conservative advancement stops when the gap is below `2⁻³²`
//! GJK stops when its bound is within `2⁻⁴⁰` of the
//! distance; together they are checked to `1e-9` for approaches up to about 60°
//! from the surface normal. A grazing approach converges slowly and gives up
//! (no hit) after 1024 steps. Height-field cells are refined by
//! Newton steps from the projection of the point, which is exact for a planar cell
//! and a local minimum on a twisted one. Segment distances to tori, height fields
//! and SDFs minimise a 1-Lipschitz point distance along the segment by
//! branch-and-bound to `2⁻¹⁶` and golden-section refinement after it. SDF fields
//! are `f32`: a hit is where `field − r` drops below [`crate::sdf_ccd::SdfCcdConfig::tolerance`].
//!
//! Author: Moroya Sakamoto

use crate::body_collider::BodyCollider;
use crate::collider::{Capsule, Support, AABB};
use crate::compound::{CompoundChild, ShapeRef};
use crate::heightfield::HeightField;
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::plane_collider::PlaneCollider;
#[cfg(feature = "std")]
use crate::sdf_ccd::SdfCcdConfig;
use crate::shape::{PosedShape, Shape};
use crate::shape_raycast::{
    box_planes, nearer, ray_local_cylinder, ray_local_torus, ray_planes, ray_solid_capsule,
    ray_solid_sphere, to_local, to_world, torus_distance, LocalHit, RayFilter, RayTarget,
    PARALLEL_EPSILON,
};
use crate::solver::PhysicsWorld;
use crate::static_collider::StaticCollider;
use crate::trimesh::{ray_triangle, TriMesh, Triangle};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Conservative advancement stops when the gap is below this: `2⁻³²`.
const TRACE_TOLERANCE: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000_0001_0000_0000,
};

/// Conservative advancement steps before giving up (a grazing approach).
const TRACE_MAX_STEPS: usize = 1024;

/// GJK stops when `|v|² − v·w ≤ |v|² · 2⁻⁴⁰`.
const GJK_RELATIVE: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000_0000_0100_0000,
};

/// GJK treats `|v|² ≤ 2⁻⁸⁰` as intersecting: `|v| ≤ 2⁻⁴⁰`.
const GJK_INTERSECT: Fix128 = Fix128 { hi: 0, lo: 1 << 48 };

const GJK_MAX_ITERATIONS: usize = 64;

/// Branch-and-bound along a segment stops subdividing below this bound gap:
/// `2⁻¹⁶`.
const CURVE_COARSE: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0001_0000_0000_0000,
};

/// Golden-section steps after the branch-and-bound (shrinks by `0.618⁶⁴`).
const GOLDEN_STEPS: usize = 64;

/// Branch-and-bound node budget per minimisation.
const CURVE_MAX_NODES: usize = 256;

/// SDF box overlap: octree depth.
#[cfg(feature = "std")]
const SDF_BOX_DEPTH: u32 = 10;

// ============================================================================
// Public types
// ============================================================================

/// A hit of [`PhysicsWorld::cast_sphere`] or [`PhysicsWorld::cast_capsule`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct WorldShapeHit {
    /// Distance the shape travels along the (normalized) direction before it
    /// touches.
    pub t: Fix128,
    /// The contact point on the collider (the cast shape's centre for a cast that
    /// starts overlapping; see the module conventions).
    pub point: Vec3Fix,
    /// Unit surface normal of the collider at the contact, toward the cast shape
    /// (`−direction` for a cast that starts overlapping).
    pub normal: Vec3Fix,
    /// The collider hit.
    pub target: RayTarget,
    /// The body the hit belongs to, as in [`crate::shape_raycast::WorldRayHit::body`].
    pub body: Option<usize>,
}

/// One collider a capsule overlaps (see [`PhysicsWorld::capsule_penetrations`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Penetration {
    /// The collider.
    pub(crate) target: RayTarget,
    /// How far the capsule reaches into it, `r −` the segment's distance (`> 0`).
    pub(crate) depth: Fix128,
    /// Unit normal of the collider at the nearest point, toward the segment:
    /// moving the capsule by `depth · normal` makes it touch.
    pub(crate) normal: Vec3Fix,
}

// ============================================================================
// Distances
// ============================================================================

/// The distance from a query core (a point or a segment) to one piece of geometry.
#[derive(Clone, Copy, Debug)]
enum Dist {
    /// The core meets the inside of a solid (or lies on a surface).
    Inside,
    /// The nearest point of the geometry, its distance and the unit normal there
    /// pointing toward the core.
    Outside {
        dist: Fix128,
        point: Vec3Fix,
        normal: Vec3Fix,
    },
    /// Nothing nearer than this.
    AtLeast(Fix128),
}

impl Dist {
    /// A lower bound on the distance (`−1` inside), for comparisons.
    fn value(&self) -> Fix128 {
        match *self {
            Self::Inside => Fix128::NEG_ONE,
            Self::Outside { dist, .. } => dist,
            Self::AtLeast(d) => d,
        }
    }

    /// Whether a sphere of `r` about the core overlaps the geometry.
    fn within(&self, r: Fix128) -> bool {
        match *self {
            Self::Inside => true,
            Self::Outside { dist, .. } => dist < r,
            Self::AtLeast(_) => false,
        }
    }

    /// The outside distance from `core` to `point`, the normal toward the core.
    fn toward(core: Vec3Fix, point: Vec3Fix) -> Self {
        let (n, dist) = (core - point).normalize_with_length();
        if dist.is_zero() {
            return Self::Inside;
        }
        Self::Outside {
            dist,
            point,
            normal: n,
        }
    }

    /// This distance less `amount` (the core grown, a solid's radius).
    fn shrunk(self, amount: Fix128) -> Self {
        match self {
            Self::Outside {
                dist,
                point,
                normal,
            } => {
                let d = dist - amount;
                if d.is_negative() {
                    Self::Inside
                } else {
                    Self::Outside {
                        dist: d,
                        point: point + normal * amount,
                        normal,
                    }
                }
            }
            Self::AtLeast(d) => Self::AtLeast(d - amount),
            Self::Inside => Self::Inside,
        }
    }
}

fn half_vec(v: Vec3Fix) -> Vec3Fix {
    Vec3Fix::new(v.x.half(), v.y.half(), v.z.half())
}

fn min_fix(a: Fix128, b: Fix128) -> Fix128 {
    if b < a {
        b
    } else {
        a
    }
}

fn max_fix(a: Fix128, b: Fix128) -> Fix128 {
    if b > a {
        b
    } else {
        a
    }
}

fn clamp(x: Fix128, lo: Fix128, hi: Fix128) -> Fix128 {
    max_fix(lo, min_fix(x, hi))
}

/// The point of segment `a`–`b` nearest `p`.
fn closest_on_segment(a: Vec3Fix, b: Vec3Fix, p: Vec3Fix) -> Vec3Fix {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2.is_zero() {
        return a;
    }
    a + ab * clamp((p - a).dot(ab) / len2, Fix128::ZERO, Fix128::ONE)
}

/// The point of `aabb` nearest `p` (the point itself inside).
fn closest_on_aabb(aabb: &AABB, p: Vec3Fix) -> Vec3Fix {
    Vec3Fix::new(
        clamp(p.x, aabb.min.x, aabb.max.x),
        clamp(p.y, aabb.min.y, aabb.max.y),
        clamp(p.z, aabb.min.z, aabb.max.z),
    )
}

/// A point, for GJK.
struct PointSupport(Vec3Fix);

impl Support for PointSupport {
    fn support(&self, _direction: Vec3Fix) -> Vec3Fix {
        self.0
    }
}

/// A segment, for GJK.
struct SegmentSupport(Vec3Fix, Vec3Fix);

impl Support for SegmentSupport {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        if self.1.dot(direction) > self.0.dot(direction) {
            self.1
        } else {
            self.0
        }
    }
}

/// A triangle, for GJK.
struct TriangleSupport<'a>(&'a Triangle);

impl Support for TriangleSupport<'_> {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        let t = self.0;
        let (d0, d1, d2) = (
            t.v0.dot(direction),
            t.v1.dot(direction),
            t.v2.dot(direction),
        );
        if d1 > d0 && d1 >= d2 {
            t.v1
        } else if d2 > d0 && d2 > d1 {
            t.v2
        } else {
            t.v0
        }
    }
}

/// A compound child placed in the world, for GJK.
struct ChildSupport<'a> {
    child: &'a CompoundChild,
    position: Vec3Fix,
    rotation: QuatFix,
}

impl Support for ChildSupport<'_> {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        self.child
            .support_world(direction, self.position, self.rotation)
    }
}

/// One vertex of a GJK simplex: the Minkowski-difference point and the two
/// support points it came from.
#[derive(Clone, Copy)]
struct Vertex {
    w: Vec3Fix,
    a: Vec3Fix,
    b: Vec3Fix,
}

fn vertex<A: Support, B: Support>(a: &A, b: &B, direction: Vec3Fix) -> Vertex {
    let pa = a.support(direction);
    let pb = b.support(-direction);
    Vertex {
        w: pa - pb,
        a: pa,
        b: pb,
    }
}

/// The point of the simplex nearest the origin, the sub-simplex it lies on and
/// its barycentric weights; `None` when the origin is inside a tetrahedron.
fn closest_on_simplex(s: &[Vertex]) -> Option<(Vec<Vertex>, Vec<Fix128>)> {
    match s.len() {
        1 => Some((s.to_vec(), vec![Fix128::ONE])),
        2 => Some(closest_on_edge(s[0], s[1])),
        3 => Some(closest_on_triangle(s[0], s[1], s[2])),
        _ => closest_on_tetrahedron(s[0], s[1], s[2], s[3]),
    }
}

fn closest_on_edge(a: Vertex, b: Vertex) -> (Vec<Vertex>, Vec<Fix128>) {
    let ab = b.w - a.w;
    let len2 = ab.length_squared();
    let t = if len2.is_zero() {
        Fix128::ZERO
    } else {
        -a.w.dot(ab) / len2
    };
    if t <= Fix128::ZERO {
        (vec![a], vec![Fix128::ONE])
    } else if t >= Fix128::ONE {
        (vec![b], vec![Fix128::ONE])
    } else {
        (vec![a, b], vec![Fix128::ONE - t, t])
    }
}

/// Ericson, *Real-Time Collision Detection* §5.1.5, for the origin.
fn closest_on_triangle(a: Vertex, b: Vertex, c: Vertex) -> (Vec<Vertex>, Vec<Fix128>) {
    let ab = b.w - a.w;
    let ac = c.w - a.w;
    let ap = -a.w;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= Fix128::ZERO && d2 <= Fix128::ZERO {
        return (vec![a], vec![Fix128::ONE]);
    }
    let bp = -b.w;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= Fix128::ZERO && d4 <= d3 {
        return (vec![b], vec![Fix128::ONE]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= Fix128::ZERO && d1 >= Fix128::ZERO && d3 <= Fix128::ZERO {
        let v = d1 / (d1 - d3);
        return (vec![a, b], vec![Fix128::ONE - v, v]);
    }
    let cp = -c.w;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= Fix128::ZERO && d5 <= d6 {
        return (vec![c], vec![Fix128::ONE]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= Fix128::ZERO && d2 >= Fix128::ZERO && d6 <= Fix128::ZERO {
        let w = d2 / (d2 - d6);
        return (vec![a, c], vec![Fix128::ONE - w, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= Fix128::ZERO && d4 - d3 >= Fix128::ZERO && d5 - d6 >= Fix128::ZERO {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (vec![b, c], vec![Fix128::ONE - w, w]);
    }
    let denom = va + vb + vc;
    if denom <= Fix128::ZERO {
        // A degenerate (flat) triangle: the nearest of its edges.
        let mut best = closest_on_edge(a, b);
        for cand in [closest_on_edge(b, c), closest_on_edge(a, c)] {
            if weighted(&cand).length_squared() < weighted(&best).length_squared() {
                best = cand;
            }
        }
        return best;
    }
    let v = vb / denom;
    let w = vc / denom;
    (vec![a, b, c], vec![Fix128::ONE - v - w, v, w])
}

fn closest_on_tetrahedron(
    a: Vertex,
    b: Vertex,
    c: Vertex,
    d: Vertex,
) -> Option<(Vec<Vertex>, Vec<Fix128>)> {
    let faces = [(a, b, c, d), (a, c, d, b), (a, d, b, c), (b, d, c, a)];
    let mut best: Option<(Vec<Vertex>, Vec<Fix128>)> = None;
    let mut inside = true;
    for (p, q, r, opposite) in faces {
        let n = (q.w - p.w).cross(r.w - p.w);
        let origin_side = n.dot(-p.w);
        let other_side = n.dot(opposite.w - p.w);
        // The origin is outside this face when it is strictly on the other side
        // from the fourth vertex; every face of a flat tetrahedron counts.
        let outside = other_side.is_zero()
            || (!origin_side.is_zero() && origin_side.is_negative() != other_side.is_negative());
        if outside {
            inside = false;
            let cand = closest_on_triangle(p, q, r);
            if best
                .as_ref()
                .is_none_or(|b| weighted(&cand).length_squared() < weighted(b).length_squared())
            {
                best = Some(cand);
            }
        }
    }
    if inside {
        None
    } else {
        best
    }
}

fn weighted(s: &(Vec<Vertex>, Vec<Fix128>)) -> Vec3Fix {
    s.0.iter()
        .zip(&s.1)
        .fold(Vec3Fix::ZERO, |acc, (v, &l)| acc + v.w * l)
}

/// The distance between two convex sets by GJK: `(distance, point on a, point on
/// b)`, or `None` when they intersect (closer than `2⁻⁴⁰`).
fn gjk_distance<A: Support, B: Support>(a: &A, b: &B) -> Option<(Fix128, Vec3Fix, Vec3Fix)> {
    let mut simplex = vec![vertex(a, b, Vec3Fix::UNIT_X)];
    let mut weights = vec![Fix128::ONE];
    let mut v = simplex[0].w;
    for _ in 0..GJK_MAX_ITERATIONS {
        let vv = v.length_squared();
        if vv <= GJK_INTERSECT {
            return None;
        }
        let w = vertex(a, b, -v);
        if vv - v.dot(w.w) <= vv * GJK_RELATIVE {
            break;
        }
        if simplex.iter().any(|s| s.w == w.w) {
            break;
        }
        let mut grown = simplex.clone();
        grown.push(w);
        let (sub, lambda) = closest_on_simplex(&grown)?;
        let next = weighted(&(sub.clone(), lambda.clone()));
        if next.length_squared() >= vv {
            // No progress (rounding): keep the previous simplex.
            break;
        }
        simplex = sub;
        weights = lambda;
        v = next;
    }
    let pa = simplex
        .iter()
        .zip(&weights)
        .fold(Vec3Fix::ZERO, |acc, (s, &l)| acc + s.a * l);
    let pb = simplex
        .iter()
        .zip(&weights)
        .fold(Vec3Fix::ZERO, |acc, (s, &l)| acc + s.b * l);
    let dist = v.length();
    if dist.is_zero() {
        return None;
    }
    Some((dist, pa, pb))
}

/// The distance from the core `a`–`b` to a convex set, by GJK.
fn convex_dist<S: Support>(a: Vec3Fix, b: Vec3Fix, solid: &S) -> Dist {
    let found = if a == b {
        gjk_distance(&PointSupport(a), solid)
    } else {
        gjk_distance(&SegmentSupport(a, b), solid)
    };
    match found {
        None => Dist::Inside,
        Some((dist, pa, pb)) => Dist::Outside {
            dist,
            point: pb,
            normal: (pa - pb) / dist,
        },
    }
}

/// The distance between two segments `p0`–`p1` and `q0`–`q1`, by GJK on the two
/// segments: `(distance, point on p, point on q)`, `None` when they cross.
fn segment_segment(
    p0: Vec3Fix,
    p1: Vec3Fix,
    q0: Vec3Fix,
    q1: Vec3Fix,
) -> Option<(Fix128, Vec3Fix, Vec3Fix)> {
    if p0 == p1 && q0 == q1 {
        let (_, d) = (p0 - q0).normalize_with_length();
        return (!d.is_zero()).then_some((d, p0, q0));
    }
    if p0 == p1 {
        let q = closest_on_segment(q0, q1, p0);
        let d = (p0 - q).length();
        return (!d.is_zero()).then_some((d, p0, q));
    }
    if q0 == q1 {
        let p = closest_on_segment(p0, p1, q0);
        let d = (p - q0).length();
        return (!d.is_zero()).then_some((d, p, q0));
    }
    gjk_distance(&SegmentSupport(p0, p1), &SegmentSupport(q0, q1))
}

/// The distance from the core `a`–`b` to a solid sphere.
fn sphere_dist(a: Vec3Fix, b: Vec3Fix, center: Vec3Fix, radius: Fix128) -> Dist {
    let s = closest_on_segment(a, b, center);
    let (n, d) = (s - center).normalize_with_length();
    if d < radius || d.is_zero() {
        return Dist::Inside;
    }
    Dist::Outside {
        dist: d - radius,
        point: center + n * radius,
        normal: n,
    }
}

/// The distance from the core `a`–`b` to a solid capsule.
fn capsule_dist(a: Vec3Fix, b: Vec3Fix, capsule: &Capsule) -> Dist {
    match segment_segment(a, b, capsule.a, capsule.b) {
        None => Dist::Inside,
        Some((d, pa, pb)) => Dist::Outside {
            dist: d,
            point: pb,
            normal: (pa - pb) / d,
        }
        .shrunk(capsule.radius),
    }
}

/// The distance from `p` to a solid box of half-extents `h` about the origin, in
/// its frame.
fn point_box_local(p: Vec3Fix, h: Vec3Fix) -> Dist {
    let q = Vec3Fix::new(
        clamp(p.x, -h.x, h.x),
        clamp(p.y, -h.y, h.y),
        clamp(p.z, -h.z, h.z),
    );
    if q == p {
        return Dist::Inside;
    }
    Dist::toward(p, q)
}

/// The distance from `p` to a solid cylinder along `Y` (radius `radius`, half
/// height `hh`), in its frame.
fn point_cylinder_local(p: Vec3Fix, radius: Fix128, hh: Fix128) -> Dist {
    let (dir, rho) = Vec3Fix::new(p.x, Fix128::ZERO, p.z).normalize_with_length();
    let qy = clamp(p.y, -hh, hh);
    if rho < radius && p.y.abs() < hh {
        return Dist::Inside;
    }
    let qr = min_fix(rho, radius);
    let q = if rho.is_zero() {
        Vec3Fix::new(Fix128::ZERO, qy, Fix128::ZERO)
    } else {
        Vec3Fix::new(dir.x * qr, qy, dir.z * qr)
    };
    Dist::toward(p, q)
}

/// The distance from `p` to a solid torus (ring `major` in `XZ`, tube `minor`), in
/// its frame.
fn point_torus_local(p: Vec3Fix, major: Fix128, minor: Fix128) -> Dist {
    if torus_distance(p, major, minor).is_negative() {
        return Dist::Inside;
    }
    let (ring_dir, rho) = Vec3Fix::new(p.x, Fix128::ZERO, p.z).normalize_with_length();
    let ring_dir = if rho.is_zero() {
        Vec3Fix::UNIT_X
    } else {
        ring_dir
    };
    let ring = ring_dir * major;
    match Dist::toward(p, ring) {
        Dist::Outside { dist, normal, .. } => Dist::Outside {
            dist,
            point: ring,
            normal,
        }
        .shrunk(minor),
        other => other,
    }
}

/// A distance in a solid's frame turned back into the world.
fn dist_to_world(d: Dist, center: Vec3Fix, rotation: QuatFix) -> Dist {
    match d {
        Dist::Outside {
            dist,
            point,
            normal,
        } => Dist::Outside {
            dist,
            point: center + rotation.rotate_vec(point),
            normal: rotation.rotate_vec(normal),
        },
        other => other,
    }
}

/// The minimum over `s ∈ [0, 1]` of an `L`-Lipschitz distance `f(s)`:
/// branch-and-bound to [`CURVE_COARSE`], then golden-section refinement about
/// the best node. `f` may report [`Dist::AtLeast`] (a lower bound).
fn curve_min(lipschitz: Fix128, f: impl Fn(Fix128) -> Dist) -> Dist {
    const SEEDS: i64 = 16;
    // (s, half width of its node, distance)
    let mut best: Option<(Fix128, Fix128, Dist)> = None;
    // Open nodes: (midpoint, half width, lower bound).
    let mut stack: Vec<(Fix128, Fix128, Fix128)> = Vec::new();
    let seed_half = Fix128::from_ratio(1, 2 * SEEDS);
    let mut nodes = 0usize;
    let mut todo: Vec<(Fix128, Fix128)> = (0..SEEDS)
        .map(|k| (Fix128::from_ratio(2 * k + 1, 2 * SEEDS), seed_half))
        .collect();
    loop {
        for (m, hw) in todo.drain(..) {
            let d = f(m);
            if matches!(d, Dist::Inside) {
                return Dist::Inside;
            }
            if best.as_ref().is_none_or(|(_, _, b)| d.value() < b.value()) {
                best = Some((m, hw, d));
            }
            // A node's lower bound: the Lipschitz cone below its midpoint value.
            stack.push((m, hw, d.value() - lipschitz * hw));
        }
        let best_value = best.as_ref().map_or(Fix128::ZERO, |(_, _, b)| b.value());
        // Drop the nodes that cannot hold a value below the best by more than the
        // coarse tolerance, and the ones already narrow enough.
        stack.retain(|&(_, hw, bound)| {
            bound < best_value - CURVE_COARSE && lipschitz * hw > CURVE_COARSE
        });
        // Best first: the lowest bound (the first such node on a tie).
        let Some(k) = (0..stack.len()).min_by_key(|&k| stack[k].2) else {
            break;
        };
        nodes += 1;
        if nodes > CURVE_MAX_NODES {
            break;
        }
        let (mid, hw, _) = stack.remove(k);
        let quarter = hw.half();
        todo.push((mid - quarter, quarter));
        todo.push((mid + quarter, quarter));
    }
    let Some((s_best, hw_best, d_best)) = best else {
        return Dist::AtLeast(Fix128::ZERO);
    };
    // Golden-section over the best node and its neighbours.
    let mut lo = max_fix(Fix128::ZERO, s_best - hw_best.double());
    let mut hi = min_fix(Fix128::ONE, s_best + hw_best.double());
    let ratio = Fix128::from_ratio(618_033_988_749_895, 1_000_000_000_000_000);
    let mut x1 = hi - (hi - lo) * ratio;
    let mut x2 = lo + (hi - lo) * ratio;
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    let mut refined = d_best;
    for _ in 0..GOLDEN_STEPS {
        for d in [f1, f2] {
            if matches!(d, Dist::Inside) {
                return Dist::Inside;
            }
            if d.value() < refined.value() {
                refined = d;
            }
        }
        if f1.value() <= f2.value() {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - (hi - lo) * ratio;
            f1 = f(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + (hi - lo) * ratio;
            f2 = f(x2);
        }
    }
    refined
}

/// The distance from the segment `a`–`b` to a geometry given by its point
/// distance `f`.
fn segment_min(a: Vec3Fix, b: Vec3Fix, f: impl Fn(Vec3Fix) -> Dist) -> Dist {
    if a == b {
        return f(a);
    }
    let ab = b - a;
    curve_min(ab.length(), |s| f(a + ab * s))
}

// ============================================================================
// Height fields
// ============================================================================

/// Whether the field has a surface (see [`crate::shape_raycast`]).
fn field_has_surface(field: &HeightField) -> bool {
    field.width >= 2 && field.depth >= 2 && field.spacing > Fix128::ZERO
}

/// The cell range `[lo, hi]` (clamped to the grid) covering `[x0, x1]` along an
/// axis with `n` points.
fn cell_range(
    x0: Fix128,
    x1: Fix128,
    origin: Fix128,
    spacing: Fix128,
    n: u32,
) -> Option<(u32, u32)> {
    let last = i64::from(n) - 2;
    let lo = ((x0 - origin) / spacing).floor().hi;
    let hi = ((x1 - origin) / spacing).floor().hi;
    if hi < 0 || lo > last {
        return None;
    }
    Some((lo.clamp(0, last) as u32, hi.clamp(0, last) as u32))
}

/// The world `(x, z)` of cell `(gx, gz)`'s first corner.
fn cell_origin(field: &HeightField, gx: u32, gz: u32) -> (Fix128, Fix128) {
    (
        field.origin.x + field.spacing * Fix128::from_int(i64::from(gx)),
        field.origin.z + field.spacing * Fix128::from_int(i64::from(gz)),
    )
}

/// The four corner heights of cell `(gx, gz)`: `(h00, h10, h01, h11)`.
fn cell_heights(field: &HeightField, gx: u32, gz: u32) -> [Fix128; 4] {
    [
        field.get_height(gx, gz),
        field.get_height(gx + 1, gz),
        field.get_height(gx, gz + 1),
        field.get_height(gx + 1, gz + 1),
    ]
}

/// The point of the bilinear cell nearest `p`: the nearest of its four edges
/// (straight lines) and, when Newton's method from the projection of `p`
/// converges inside the cell, that interior point.
fn cell_closest(x0: Fix128, z0: Fix128, s: Fix128, h: [Fix128; 4], p: Vec3Fix) -> Vec3Fix {
    let [h00, h10, h01, h11] = h;
    let corner = |u: bool, v: bool, hh: Fix128| {
        Vec3Fix::new(if u { x0 + s } else { x0 }, hh, if v { z0 + s } else { z0 })
    };
    let c00 = corner(false, false, h00);
    let c10 = corner(true, false, h10);
    let c01 = corner(false, true, h01);
    let c11 = corner(true, true, h11);
    let mut best = closest_on_segment(c00, c10, p);
    for (e0, e1) in [(c10, c11), (c11, c01), (c01, c00)] {
        let q = closest_on_segment(e0, e1, p);
        if (q - p).length_squared() < (best - p).length_squared() {
            best = q;
        }
    }
    let k = h00 - h10 - h01 + h11;
    let mut u = clamp((p.x - x0) / s, Fix128::ZERO, Fix128::ONE);
    let mut v = clamp((p.z - z0) / s, Fix128::ZERO, Fix128::ONE);
    let mut ok = true;
    for _ in 0..8 {
        let one = Fix128::ONE;
        let hh =
            h00 * (one - u) * (one - v) + h10 * u * (one - v) + h01 * (one - u) * v + h11 * u * v;
        let hu = (h10 - h00) * (one - v) + (h11 - h01) * v;
        let hv = (h01 - h00) * (one - u) + (h11 - h10) * u;
        let e = Vec3Fix::new(x0 + s * u, hh, z0 + s * v) - p;
        let pu = Vec3Fix::new(s, hu, Fix128::ZERO);
        let pv = Vec3Fix::new(Fix128::ZERO, hv, s);
        let gu = e.dot(pu);
        let gv = e.dot(pv);
        let huu = pu.dot(pu);
        let hvv = pv.dot(pv);
        let huv = pu.dot(pv) + e.y * k;
        let det = huu * hvv - huv * huv;
        if det <= Fix128::ZERO {
            ok = false;
            break;
        }
        let du = (hvv * gu - huv * gv) / det;
        let dv = (huu * gv - huv * gu) / det;
        u = u - du;
        v = v - dv;
        if u.is_negative() || v.is_negative() || u > one || v > one {
            ok = false;
            break;
        }
    }
    if ok {
        let one = Fix128::ONE;
        let hh =
            h00 * (one - u) * (one - v) + h10 * u * (one - v) + h01 * (one - u) * v + h11 * u * v;
        let q = Vec3Fix::new(x0 + s * u, hh, z0 + s * v);
        if (q - p).length_squared() < (best - p).length_squared() {
            best = q;
        }
    }
    best
}

/// The distance from `p` to the height-field surface, searching the cells within
/// `cap` of `p` ([`Dist::AtLeast`] beyond).
fn point_heightfield(field: &HeightField, p: Vec3Fix, cap: Fix128) -> Dist {
    if !field_has_surface(field) {
        return Dist::AtLeast(cap);
    }
    let bounds = field.aabb();
    let far = (closest_on_aabb(&bounds, p) - p).length();
    if far >= cap {
        return Dist::AtLeast(far);
    }
    let s = field.spacing;
    let (Some((gx0, gx1)), Some((gz0, gz1))) = (
        cell_range(p.x - cap, p.x + cap, field.origin.x, s, field.width),
        cell_range(p.z - cap, p.z + cap, field.origin.z, s, field.depth),
    ) else {
        return Dist::AtLeast(cap);
    };
    // The cells by their box's distance from `p`, nearest first: once a cell's
    // box is farther than the best point found, no later cell can be nearer.
    let mut cells: Vec<(Fix128, u32, u32)> = Vec::new();
    for gz in gz0..=gz1 {
        for gx in gx0..=gx1 {
            let h = cell_heights(field, gx, gz);
            let (x0, z0) = cell_origin(field, gx, gz);
            let lo = min_fix(min_fix(h[0], h[1]), min_fix(h[2], h[3]));
            let hi = max_fix(max_fix(h[0], h[1]), max_fix(h[2], h[3]));
            let cell_box = AABB::new(Vec3Fix::new(x0, lo, z0), Vec3Fix::new(x0 + s, hi, z0 + s));
            let lb2 = (closest_on_aabb(&cell_box, p) - p).length_squared();
            if lb2 < cap * cap {
                cells.push((lb2, gz, gx));
            }
        }
    }
    cells.sort_unstable();
    let mut best: Option<(Fix128, Vec3Fix)> = None;
    for (lb2, gz, gx) in cells {
        if best.is_some_and(|(d2, _)| lb2 >= d2) {
            break;
        }
        let (x0, z0) = cell_origin(field, gx, gz);
        let q = cell_closest(x0, z0, s, cell_heights(field, gx, gz), p);
        let d2 = (q - p).length_squared();
        if best.is_none_or(|(b, _)| d2 < b) {
            best = Some((d2, q));
        }
    }
    match best {
        Some((d2, q)) if d2 < cap * cap => Dist::toward(p, q),
        _ => Dist::AtLeast(cap),
    }
}

/// Whether the height-field surface passes through `aabb`: over each cell's part
/// of the box's `XZ` footprint the bilinear surface takes its extremes at the
/// corners of that part.
fn heightfield_meets_aabb(field: &HeightField, aabb: &AABB) -> bool {
    if !field_has_surface(field) {
        return false;
    }
    let s = field.spacing;
    let x_end = field.origin.x + s * Fix128::from_int(i64::from(field.width) - 1);
    let z_end = field.origin.z + s * Fix128::from_int(i64::from(field.depth) - 1);
    let fx0 = max_fix(aabb.min.x, field.origin.x);
    let fx1 = min_fix(aabb.max.x, x_end);
    let fz0 = max_fix(aabb.min.z, field.origin.z);
    let fz1 = min_fix(aabb.max.z, z_end);
    if fx0 > fx1 || fz0 > fz1 {
        return false;
    }
    let (Some((gx0, gx1)), Some((gz0, gz1))) = (
        cell_range(fx0, fx1, field.origin.x, s, field.width),
        cell_range(fz0, fz1, field.origin.z, s, field.depth),
    ) else {
        return false;
    };
    for gz in gz0..=gz1 {
        for gx in gx0..=gx1 {
            let x0 = field.origin.x + s * Fix128::from_int(i64::from(gx));
            let z0 = field.origin.z + s * Fix128::from_int(i64::from(gz));
            let ux0 = max_fix(fx0, x0);
            let ux1 = min_fix(fx1, x0 + s);
            let uz0 = max_fix(fz0, z0);
            let uz1 = min_fix(fz1, z0 + s);
            if ux0 > ux1 || uz0 > uz1 {
                continue;
            }
            let [h00, h10, h01, h11] = cell_heights(field, gx, gz);
            let mut lo: Option<Fix128> = None;
            let mut hi: Option<Fix128> = None;
            for x in [ux0, ux1] {
                for z in [uz0, uz1] {
                    let u = (x - x0) / s;
                    let v = (z - z0) / s;
                    let one = Fix128::ONE;
                    let h = h00 * (one - u) * (one - v)
                        + h10 * u * (one - v)
                        + h01 * (one - u) * v
                        + h11 * u * v;
                    lo = Some(lo.map_or(h, |l| min_fix(l, h)));
                    hi = Some(hi.map_or(h, |m| max_fix(m, h)));
                }
            }
            if let (Some(lo), Some(hi)) = (lo, hi) {
                if lo < aabb.max.y && hi > aabb.min.y {
                    return true;
                }
            }
        }
    }
    false
}

// ============================================================================
// Triangle meshes
// ============================================================================

/// The triangles of `mesh` whose box meets `query`, in BVH order.
fn mesh_candidates(mesh: &TriMesh, query: &AABB) -> Vec<u32> {
    mesh.bvh.query(query)
}

/// The distance from the core `a`–`b` to the mesh, searching the triangles within
/// `cap`.
fn mesh_dist(mesh: &TriMesh, a: Vec3Fix, b: Vec3Fix, cap: Fix128) -> Dist {
    if mesh.triangles.is_empty() {
        return Dist::AtLeast(cap);
    }
    let core = core_box(a, b, Fix128::ZERO);
    let far = aabb_gap(&core, &mesh.bounds);
    if far >= cap {
        return Dist::AtLeast(far);
    }
    let mut best: Option<Dist> = None;
    for i in mesh_candidates(mesh, &core_box(a, b, cap)) {
        let tri = &mesh.triangles[i as usize];
        let d = if a == b {
            Dist::toward(a, tri.closest_point(a))
        } else {
            convex_dist(a, b, &TriangleSupport(tri))
        };
        if matches!(d, Dist::Inside) {
            return Dist::Inside;
        }
        if best.as_ref().is_none_or(|x| d.value() < x.value()) {
            best = Some(d);
        }
    }
    match best {
        Some(d) if d.value() < cap => d,
        _ => Dist::AtLeast(cap),
    }
}

/// A sphere swept against one triangle: the two copies of the triangle moved
/// `±r` along its normal and the three edges grown into capsules of `r`.
fn sweep_triangle(
    o: Vec3Fix,
    d: Vec3Fix,
    tri: &Triangle,
    r: Fix128,
    max_t: Fix128,
) -> Option<LocalHit> {
    let ray = crate::raycast::Ray {
        origin: o,
        direction: d,
    };
    let mut best = None;
    if let Some(n) = tri.normal().try_normalize() {
        for off in [n * r, -(n * r)] {
            let moved = Triangle::new(tri.v0 + off, tri.v1 + off, tri.v2 + off);
            best = nearer(
                best,
                ray_triangle(&ray, &moved, max_t).map(|h| LocalHit {
                    t: h.t,
                    normal: h.normal,
                }),
            );
        }
    }
    for (e0, e1) in [(tri.v0, tri.v1), (tri.v1, tri.v2), (tri.v2, tri.v0)] {
        best = nearer(
            best,
            ray_solid_capsule(o, d, &Capsule::new(e0, e1, r), max_t),
        );
    }
    best
}

// ============================================================================
// Geometry pieces
// ============================================================================

/// One piece of geometry a query tests, placed in the world.
enum Piece<'a> {
    Sphere {
        center: Vec3Fix,
        radius: Fix128,
    },
    Capsule(Capsule),
    Box {
        center: Vec3Fix,
        half: Vec3Fix,
        rotation: QuatFix,
    },
    Posed(PosedShape),
    Hull(ChildSupport<'a>),
    Plane(&'a PlaneCollider),
    Height(&'a HeightField),
    Mesh(&'a TriMesh),
    #[cfg(feature = "std")]
    Sdf(&'a crate::sdf_collider::SdfCollider),
}

/// The pieces of a body (see [`crate::shape_raycast`] for which geometry).
fn body_pieces(world: &PhysicsWorld, i: usize) -> Vec<Piece<'_>> {
    let Some((radius, collider)) = world.ray_geometry(i) else {
        return Vec::new();
    };
    let body = &world.bodies[i];
    match collider {
        None => vec![Piece::Sphere {
            center: body.position,
            radius,
        }],
        Some(BodyCollider::Shape(shape)) => {
            let posed = PosedShape {
                shape: *shape,
                position: body.position,
                rotation: body.rotation,
            };
            vec![match *shape {
                Shape::Box { half_extents } => Piece::Box {
                    center: body.position - body.rotation.rotate_vec(shape.center_of_mass_offset()),
                    half: half_extents,
                    rotation: body.rotation,
                },
                _ => Piece::Posed(posed),
            }]
        }
        Some(BodyCollider::Compound(compound)) => compound
            .children
            .iter()
            .map(|child| child_piece(child, body.position, body.rotation))
            .collect(),
    }
}

/// One compound child, placed the way [`CompoundChild::support_world`] places it.
fn child_piece(child: &CompoundChild, body_pos: Vec3Fix, body_rot: QuatFix) -> Piece<'_> {
    let child_rot = body_rot.mul(child.local_rotation);
    let child_pos = body_pos + body_rot.rotate_vec(child.local_position);
    match &child.shape {
        ShapeRef::Sphere(s) => Piece::Sphere {
            center: child_pos + child_rot.rotate_vec(s.center),
            radius: s.radius,
        },
        ShapeRef::Capsule(c) => Piece::Capsule(Capsule::new(
            child_pos + child_rot.rotate_vec(c.a),
            child_pos + child_rot.rotate_vec(c.b),
            c.radius,
        )),
        ShapeRef::Box(b) => Piece::Box {
            center: child_pos + child_rot.rotate_vec(b.center),
            half: b.half_extents,
            rotation: child_rot.mul(b.rotation),
        },
        ShapeRef::ConvexHull(_) => Piece::Hull(ChildSupport {
            child,
            position: body_pos,
            rotation: body_rot,
        }),
    }
}

/// The geometric centre and frame of a posed shape.
fn posed_frame(posed: &PosedShape) -> (Vec3Fix, QuatFix) {
    (
        posed.position
            - posed
                .rotation
                .rotate_vec(posed.shape.center_of_mass_offset()),
        posed.rotation,
    )
}

/// The query settings that pieces need.
struct Settings {
    #[cfg(feature = "std")]
    sdf: SdfCcdConfig,
}

impl Piece<'_> {
    /// The distance from the core `a`–`b`; `cap` bounds the search where a piece
    /// searches cells or triangles (it may report [`Dist::AtLeast`] beyond).
    fn dist(&self, a: Vec3Fix, b: Vec3Fix, cap: Fix128) -> Dist {
        match self {
            Self::Sphere { center, radius } => sphere_dist(a, b, *center, *radius),
            Self::Capsule(c) => capsule_dist(a, b, c),
            Self::Box {
                center,
                half,
                rotation,
            } => {
                if a == b {
                    let inv = rotation.conjugate();
                    let p = inv.rotate_vec(a - *center);
                    dist_to_world(point_box_local(p, *half), *center, *rotation)
                } else {
                    let obb = crate::box_collider::OrientedBox::new(*center, *half, *rotation);
                    convex_dist(a, b, &obb)
                }
            }
            Self::Posed(posed) => {
                let (center, rotation) = posed_frame(posed);
                let inv = rotation.conjugate();
                match posed.shape {
                    Shape::Cylinder {
                        radius,
                        half_height,
                    } if a == b => dist_to_world(
                        point_cylinder_local(inv.rotate_vec(a - center), radius, half_height),
                        center,
                        rotation,
                    ),
                    Shape::Torus {
                        major_radius,
                        minor_radius,
                    } => {
                        let f = |p: Vec3Fix| {
                            dist_to_world(
                                point_torus_local(
                                    inv.rotate_vec(p - center),
                                    major_radius,
                                    minor_radius,
                                ),
                                center,
                                rotation,
                            )
                        };
                        segment_min(a, b, f)
                    }
                    _ => convex_dist(a, b, posed),
                }
            }
            Self::Hull(child) => convex_dist(a, b, child),
            Self::Plane(plane) => plane_dist(plane, a, b),
            Self::Height(field) => segment_min(a, b, |p| point_heightfield(field, p, cap)),
            Self::Mesh(mesh) => mesh_dist(mesh, a, b, cap),
            #[cfg(feature = "std")]
            Self::Sdf(sdf) => segment_min(a, b, |p| point_sdf(sdf, p)),
        }
    }

    /// The first time the core `o`, `o + (b − a)` grown by `r` touches this
    /// piece moving along the unit `d`, when it does not overlap it at the start.
    fn sweep(
        &self,
        a: Vec3Fix,
        b: Vec3Fix,
        r: Fix128,
        d: Vec3Fix,
        max_t: Fix128,
        settings: &Settings,
    ) -> Option<Contact> {
        #[cfg(not(feature = "std"))]
        let _ = settings;
        let point_core = a == b;
        match self {
            Self::Sphere { center, radius } => {
                if point_core {
                    let h = ray_solid_sphere(a, d, *center, *radius + r, max_t)?;
                    Some(Contact::from_ray(a, d, r, h))
                } else {
                    // The capsule moving +d meets the sphere when the sphere's
                    // centre, moving −d, meets the capsule grown by its radius.
                    let grown = Capsule::new(a, b, r + *radius);
                    let h = ray_solid_capsule(*center, -d, &grown, max_t)?;
                    let normal = -h.normal;
                    Some(Contact {
                        t: h.t,
                        point: *center + normal * *radius,
                        normal,
                    })
                }
            }
            Self::Capsule(c) if point_core => {
                let grown = Capsule::new(c.a, c.b, c.radius + r);
                let h = ray_solid_capsule(a, d, &grown, max_t)?;
                Some(Contact::from_ray(a, d, r, h))
            }
            Self::Box {
                center,
                half,
                rotation,
            } if point_core => {
                let (o, dl) = to_local(a, d, *center, *rotation);
                let h = to_world(sweep_local_box(o, dl, *half, r, max_t), *rotation)?;
                Some(Contact::from_ray(a, d, r, h))
            }
            Self::Posed(posed) if point_core => {
                let (center, rotation) = posed_frame(posed);
                let (o, dl) = to_local(a, d, center, rotation);
                let local = match posed.shape {
                    Shape::Cylinder {
                        radius,
                        half_height,
                    } => Some(sweep_local_cylinder(o, dl, radius, half_height, r, max_t)),
                    Shape::Torus {
                        major_radius,
                        minor_radius,
                    } => Some(ray_local_torus(
                        o,
                        dl,
                        major_radius,
                        minor_radius + r,
                        max_t,
                    )),
                    _ => None,
                };
                match local {
                    Some(h) => {
                        let h = to_world(h, rotation)?;
                        Some(Contact::from_ray(a, d, r, h))
                    }
                    None => self.trace(a, b, r, d, max_t, TRACE_TOLERANCE),
                }
            }
            Self::Plane(plane) => sweep_plane(plane, a, b, r, d, max_t),
            Self::Mesh(mesh) if point_core => {
                let swept = core_box(a, a + d * max_t, r);
                let mut best: Option<LocalHit> = None;
                for i in mesh_candidates(mesh, &swept) {
                    best = nearer(
                        best,
                        sweep_triangle(a, d, &mesh.triangles[i as usize], r, max_t),
                    );
                }
                best.map(|h| Contact::from_ray(a, d, r, h))
            }
            #[cfg(feature = "std")]
            Self::Sdf(_) => {
                let tol = Fix128::from_f32(settings.sdf.tolerance);
                self.trace(a, b, r, d, max_t, tol)
            }
            _ => self.trace(a, b, r, d, max_t, TRACE_TOLERANCE),
        }
        .filter(|c| c.t >= Fix128::ZERO && c.t <= max_t)
    }

    /// Conservative advancement: step by the gap (the distance less `r`) until it
    /// is below `tol`.
    fn trace(
        &self,
        a: Vec3Fix,
        b: Vec3Fix,
        r: Fix128,
        d: Vec3Fix,
        max_t: Fix128,
        tol: Fix128,
    ) -> Option<Contact> {
        let mut t = Fix128::ZERO;
        let cap = r + r + Fix128::ONE;
        for _ in 0..TRACE_MAX_STEPS {
            let off = d * t;
            let gap = match self.dist(a + off, b + off, cap) {
                Dist::Inside => {
                    return Some(Contact {
                        t,
                        point: half_vec(a + b) + off,
                        normal: -d,
                    })
                }
                Dist::AtLeast(bound) => bound - r,
                Dist::Outside {
                    dist,
                    point,
                    normal,
                } => {
                    let gap = dist - r;
                    if gap <= tol {
                        return Some(Contact { t, point, normal });
                    }
                    gap
                }
            };
            if gap <= Fix128::ZERO {
                return None;
            }
            t = t + gap;
            if t > max_t {
                return None;
            }
        }
        None
    }

    /// Whether this piece overlaps `aabb` (see the module doc).
    fn meets_aabb(&self, aabb: &AABB) -> bool {
        match self {
            Self::Sphere { center, radius } => {
                (closest_on_aabb(aabb, *center) - *center).length_squared() < *radius * *radius
            }
            Self::Capsule(c) => match gjk_distance(&SegmentSupport(c.a, c.b), aabb) {
                None => true,
                Some((dist, _, _)) => dist < c.radius,
            },
            Self::Box {
                center,
                half,
                rotation,
            } => {
                let obb = crate::box_collider::OrientedBox::new(*center, *half, *rotation);
                gjk_distance(&obb, aabb).is_none()
            }
            Self::Posed(posed) => match posed.shape {
                Shape::Torus {
                    major_radius,
                    minor_radius,
                } => torus_meets_aabb(posed, major_radius, minor_radius, aabb),
                _ => gjk_distance(posed, aabb).is_none(),
            },
            Self::Hull(child) => gjk_distance(child, aabb).is_none(),
            Self::Plane(plane) => {
                let c = half_vec(aabb.min + aabb.max);
                let h = half_vec(aabb.max - aabb.min);
                let n = plane.normal;
                let reach = n.x.abs() * h.x + n.y.abs() * h.y + n.z.abs() * h.z;
                (n.dot(c) - plane.offset).abs() < reach
            }
            Self::Height(field) => heightfield_meets_aabb(field, aabb),
            Self::Mesh(mesh) => mesh_candidates(mesh, aabb).into_iter().any(|i| {
                gjk_distance(&TriangleSupport(&mesh.triangles[i as usize]), aabb).is_none()
            }),
            #[cfg(feature = "std")]
            Self::Sdf(sdf) => sdf_meets_aabb(sdf, aabb),
        }
    }
}

/// A contact of a cast: distance along the direction, contact point, normal.
#[derive(Clone, Copy, Debug)]
struct Contact {
    t: Fix128,
    point: Vec3Fix,
    normal: Vec3Fix,
}

impl Contact {
    /// A ray hit on the Minkowski sum turned into the contact of the sphere of `r`
    /// whose centre moved from `o` along `d`.
    fn from_ray(o: Vec3Fix, d: Vec3Fix, r: Fix128, h: LocalHit) -> Self {
        Self {
            t: h.t,
            point: o + d * h.t - h.normal * r,
            normal: h.normal,
        }
    }
}

/// A box of half-extents `h` grown by `r` (rounded), in its frame: the three
/// slabs (the box pushed out by `r` along one axis) and the 12 edges as capsules.
fn sweep_local_box(
    o: Vec3Fix,
    d: Vec3Fix,
    h: Vec3Fix,
    r: Fix128,
    max_t: Fix128,
) -> Option<LocalHit> {
    let mut best = None;
    for grow in [
        Vec3Fix::new(r, Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, r, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, r),
    ] {
        best = nearer(best, ray_planes(o, d, &box_planes(h + grow), max_t));
    }
    let signs = [Fix128::ONE, Fix128::NEG_ONE];
    for &s1 in &signs {
        for &s2 in &signs {
            for (a, b) in [
                (
                    Vec3Fix::new(-h.x, s1 * h.y, s2 * h.z),
                    Vec3Fix::new(h.x, s1 * h.y, s2 * h.z),
                ),
                (
                    Vec3Fix::new(s1 * h.x, -h.y, s2 * h.z),
                    Vec3Fix::new(s1 * h.x, h.y, s2 * h.z),
                ),
                (
                    Vec3Fix::new(s1 * h.x, s2 * h.y, -h.z),
                    Vec3Fix::new(s1 * h.x, s2 * h.y, h.z),
                ),
            ] {
                best = nearer(best, ray_solid_capsule(o, d, &Capsule::new(a, b, r), max_t));
            }
        }
    }
    best
}

/// A cylinder along `Y` grown by `r` (rounded), in its frame: the cylinder of
/// radius `radius + r`, the one of half height `hh + r`, and the two rim tori.
fn sweep_local_cylinder(
    o: Vec3Fix,
    d: Vec3Fix,
    radius: Fix128,
    hh: Fix128,
    r: Fix128,
    max_t: Fix128,
) -> Option<LocalHit> {
    let mut best = ray_local_cylinder(o, d, radius + r, hh, max_t);
    best = nearer(best, ray_local_cylinder(o, d, radius, hh + r, max_t));
    for y in [hh, -hh] {
        let shift = Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO);
        best = nearer(best, ray_local_torus(o - shift, d, radius, r, max_t));
    }
    best
}

/// The distance from the core `a`–`b` to a two-sided plane.
fn plane_dist(plane: &PlaneCollider, a: Vec3Fix, b: Vec3Fix) -> Dist {
    let n = plane.normal;
    let sa = n.dot(a) - plane.offset;
    let sb = n.dot(b) - plane.offset;
    if sa.is_zero() || sb.is_zero() || sa.is_negative() != sb.is_negative() {
        return Dist::Inside;
    }
    let (p, s) = if sb.abs() < sa.abs() {
        (b, sb)
    } else {
        (a, sa)
    };
    let side = if s.is_negative() { -n } else { n };
    Dist::Outside {
        dist: s.abs(),
        point: p - n * s,
        normal: side,
    }
}

/// A capsule (or sphere) swept against a two-sided plane: the nearer end reaches
/// distance `r`.
fn sweep_plane(
    plane: &PlaneCollider,
    a: Vec3Fix,
    b: Vec3Fix,
    r: Fix128,
    d: Vec3Fix,
    max_t: Fix128,
) -> Option<Contact> {
    let n = plane.normal;
    let sa = n.dot(a) - plane.offset;
    let sb = n.dot(b) - plane.offset;
    if sa.is_zero() || sb.is_zero() || sa.is_negative() != sb.is_negative() {
        return None;
    }
    // The nearer end (the first on a tie).
    let (p, s) = if sb.abs() < sa.abs() {
        (b, sb)
    } else {
        (a, sa)
    };
    let normal = if s.is_negative() { -n } else { n };
    let speed = -normal.dot(d);
    if speed < PARALLEL_EPSILON {
        return None;
    }
    let t = (s.abs() - r) / speed;
    if t.is_negative() || t > max_t {
        return None;
    }
    Some(Contact {
        t,
        point: p + d * t - normal * r,
        normal,
    })
}

/// Whether a torus meets `aabb`: its ring comes within `minor` of the box.
fn torus_meets_aabb(posed: &PosedShape, major: Fix128, minor: Fix128, aabb: &AABB) -> bool {
    let (center, rotation) = posed_frame(posed);
    // Quick rejection by the bounding sphere.
    if (closest_on_aabb(aabb, center) - center).length() >= major + minor {
        return false;
    }
    let ring = |s: Fix128| {
        let (sin, cos) = (Fix128::TWO_PI * s).sin_cos();
        center + rotation.rotate_vec(Vec3Fix::new(major * cos, Fix128::ZERO, major * sin))
    };
    let d = curve_min(Fix128::TWO_PI * major, |s| {
        let p = ring(s);
        let q = closest_on_aabb(aabb, p);
        if q == p {
            Dist::Inside
        } else {
            Dist::toward(p, q)
        }
    });
    d.within(minor)
}

#[cfg(feature = "std")]
fn point_sdf(sdf: &crate::sdf_collider::SdfCollider, p: Vec3Fix) -> Dist {
    let (lx, ly, lz) = sdf.world_to_local(p);
    let (d, (nx, ny, nz)) = sdf.field.distance_and_normal(lx, ly, lz);
    let d = d * sdf.scale_f32;
    if !d.is_finite() || d < 0.0 {
        return Dist::Inside;
    }
    let dist = Fix128::from_f32(d);
    let normal = sdf.local_normal_to_world(nx, ny, nz);
    Dist::Outside {
        dist,
        point: p - normal * dist,
        normal,
    }
}

/// Whether an SDF solid meets `aabb`: an octree subdivision of the box, pruning
/// cells whose centre is farther than their half-diagonal from the surface and
/// accepting a centre inside; a leaf (depth `SDF_BOX_DEPTH`) within its
/// half-diagonal is counted as meeting.
#[cfg(feature = "std")]
fn sdf_meets_aabb(sdf: &crate::sdf_collider::SdfCollider, aabb: &AABB) -> bool {
    let mut stack = vec![(aabb.min, aabb.max, 0u32)];
    while let Some((lo, hi, depth)) = stack.pop() {
        let c = half_vec(lo + hi);
        let half_diag = half_vec(hi - lo).length();
        let d = match point_sdf(sdf, c) {
            Dist::Inside => return true,
            Dist::Outside { dist, .. } => dist,
            Dist::AtLeast(d) => d,
        };
        if d >= half_diag {
            continue;
        }
        if depth >= SDF_BOX_DEPTH {
            return true;
        }
        for k in 0..8u8 {
            let pick = |bit: u8, l: Fix128, m: Fix128, h: Fix128| {
                if k & bit == 0 {
                    (l, m)
                } else {
                    (m, h)
                }
            };
            let (x0, x1) = pick(1, lo.x, c.x, hi.x);
            let (y0, y1) = pick(2, lo.y, c.y, hi.y);
            let (z0, z1) = pick(4, lo.z, c.z, hi.z);
            stack.push((
                Vec3Fix::new(x0, y0, z0),
                Vec3Fix::new(x1, y1, z1),
                depth + 1,
            ));
        }
    }
    false
}

/// The box of the segment `a`–`b` grown by `r`.
fn core_box(a: Vec3Fix, b: Vec3Fix, r: Fix128) -> AABB {
    let lo = Vec3Fix::new(min_fix(a.x, b.x), min_fix(a.y, b.y), min_fix(a.z, b.z));
    let hi = Vec3Fix::new(max_fix(a.x, b.x), max_fix(a.y, b.y), max_fix(a.z, b.z));
    let g = Vec3Fix::new(r, r, r);
    AABB::new(lo - g, hi + g)
}

/// The distance between two boxes (zero when they meet).
fn aabb_gap(a: &AABB, b: &AABB) -> Fix128 {
    let gap = |lo1: Fix128, hi1: Fix128, lo2: Fix128, hi2: Fix128| {
        max_fix(Fix128::ZERO, max_fix(lo2 - hi1, lo1 - hi2))
    };
    Vec3Fix::new(
        gap(a.min.x, a.max.x, b.min.x, b.max.x),
        gap(a.min.y, a.max.y, b.min.y, b.max.y),
        gap(a.min.z, a.max.z, b.min.z, b.max.z),
    )
    .length()
}

fn box_is_valid(aabb: &AABB) -> bool {
    aabb.min.x <= aabb.max.x && aabb.min.y <= aabb.max.y && aabb.min.z <= aabb.max.z
}

// ============================================================================
// World queries
// ============================================================================

/// The colliders a query sees whose box meets `query`: each target with its body
/// and its pieces, in target order.
fn for_each_target<'w>(
    world: &'w PhysicsWorld,
    filter: &RayFilter,
    query: &AABB,
    mut visit: impl FnMut(RayTarget, Option<usize>, &[Piece<'w>]),
) {
    let (bvh, _) = world.query_body_bvh(filter);
    let mut bodies: Vec<u32> = bvh.query(query);
    bodies.sort_unstable();
    bodies.dedup();
    for i in bodies {
        let i = i as usize;
        let pieces = body_pieces(world, i);
        visit(RayTarget::Body(i), Some(i), &pieces);
    }
    if filter.include_static {
        for (j, collider) in world.static_colliders_slice().iter().enumerate() {
            let piece = match collider {
                StaticCollider::Plane(p) => Piece::Plane(p),
                StaticCollider::HeightField(f) => Piece::Height(f),
                StaticCollider::TriMesh(m) => Piece::Mesh(m),
            };
            visit(
                RayTarget::StaticCollider(j),
                None,
                core::slice::from_ref(&piece),
            );
        }
    }
    #[cfg(feature = "std")]
    if filter.include_sdf {
        for (k, sdf) in world.sdf_colliders.iter().enumerate() {
            let body = (sdf.body_index < world.bodies.len()).then_some(sdf.body_index);
            if body.is_some_and(|b| !filter.sees_body(world, b)) {
                continue;
            }
            let piece = Piece::Sdf(sdf);
            visit(RayTarget::Sdf(k), body, core::slice::from_ref(&piece));
        }
    }
}

impl PhysicsWorld {
    /// The nearest collider a sphere of `radius` touches when its centre moves
    /// from `center` along `direction` for at most `max_t` (see
    /// [`crate::world_shape_query`] for the conventions). Radius `0` is
    /// [`PhysicsWorld::cast_ray`]; a negative radius, a zero direction or
    /// `max_t ≤ 0` gives `None`.
    #[must_use]
    pub fn cast_sphere(
        &self,
        center: Vec3Fix,
        radius: Fix128,
        direction: Vec3Fix,
        max_t: Fix128,
        filter: &RayFilter,
    ) -> Option<WorldShapeHit> {
        if radius.is_negative() {
            return None;
        }
        if radius.is_zero() {
            return self
                .cast_ray(center, direction, max_t, filter)
                .map(|h| WorldShapeHit {
                    t: h.t,
                    point: h.point,
                    normal: h.normal,
                    target: h.target,
                    body: h.body,
                });
        }
        self.cast_core(center, center, radius, direction, max_t, filter)
    }

    /// The nearest collider a capsule (segment `a`–`b` grown by `radius`) touches
    /// when it moves along `direction` for at most `max_t`. Coinciding ends are
    /// [`PhysicsWorld::cast_sphere`]; a negative radius, a zero direction or
    /// `max_t ≤ 0` gives `None`.
    #[must_use]
    pub fn cast_capsule(
        &self,
        a: Vec3Fix,
        b: Vec3Fix,
        radius: Fix128,
        direction: Vec3Fix,
        max_t: Fix128,
        filter: &RayFilter,
    ) -> Option<WorldShapeHit> {
        if a == b {
            return self.cast_sphere(a, radius, direction, max_t, filter);
        }
        if radius.is_negative() {
            return None;
        }
        self.cast_core(a, b, radius, direction, max_t, filter)
    }

    /// Every collider a sphere of `radius` about `center` overlaps, sorted by
    /// target. A negative radius gives an empty list; radius `0` finds the solids
    /// `center` is strictly inside.
    #[must_use]
    pub fn overlap_sphere(
        &self,
        center: Vec3Fix,
        radius: Fix128,
        filter: &RayFilter,
    ) -> Vec<RayTarget> {
        let mut out = Vec::new();
        if radius.is_negative() {
            return out;
        }
        let query = core_box(center, center, radius);
        let cap = radius + radius + Fix128::ONE;
        for_each_target(self, filter, &query, |target, _, pieces| {
            if pieces
                .iter()
                .any(|p| p.dist(center, center, cap).within(radius))
            {
                out.push(target);
            }
        });
        out.sort_unstable();
        out
    }

    /// Every collider `aabb` overlaps, sorted by target (see
    /// [`crate::world_shape_query`]). A box with `min > max` on any axis gives an
    /// empty list.
    #[must_use]
    pub fn overlap_aabb(&self, aabb: &AABB, filter: &RayFilter) -> Vec<RayTarget> {
        let mut out = Vec::new();
        if !box_is_valid(aabb) {
            return out;
        }
        for_each_target(self, filter, aabb, |target, _, pieces| {
            if pieces.iter().any(|p| p.meets_aabb(aabb)) {
                out.push(target);
            }
        });
        out.sort_unstable();
        out
    }

    /// The colliders the capsule (segment `a`–`b` grown by `r`) overlaps, each
    /// with its penetration along the normal at the segment point nearest it,
    /// sorted by target (the pieces of one target, such as compound children, in
    /// their order); `None` when the segment itself meets a solid (no distance, so
    /// no push-out direction). Used by [`PhysicsWorld::move_character`] to push a capsule out.
    pub(crate) fn capsule_penetrations(
        &self,
        a: Vec3Fix,
        b: Vec3Fix,
        r: Fix128,
        filter: &RayFilter,
    ) -> Option<Vec<Penetration>> {
        let query = core_box(a, b, r);
        let cap = r + r + Fix128::ONE;
        let mut out = Vec::new();
        let mut core_inside = false;
        for_each_target(self, filter, &query, |target, _, pieces| {
            for piece in pieces {
                match piece.dist(a, b, cap) {
                    Dist::Inside => core_inside = true,
                    Dist::Outside { dist, normal, .. } if dist < r => out.push(Penetration {
                        target,
                        depth: r - dist,
                        normal,
                    }),
                    _ => {}
                }
            }
        });
        if core_inside {
            return None;
        }
        out.sort_by_key(|x| x.target);
        Some(out)
    }

    /// The nearest hit of the core `a`–`b` grown by `r > 0` (or `r = 0` with
    /// `a ≠ b`) moving along `direction`.
    fn cast_core(
        &self,
        a: Vec3Fix,
        b: Vec3Fix,
        r: Fix128,
        direction: Vec3Fix,
        max_t: Fix128,
        filter: &RayFilter,
    ) -> Option<WorldShapeHit> {
        if max_t <= Fix128::ZERO {
            return None;
        }
        let d = direction.try_normalize()?;
        let settings = Settings {
            #[cfg(feature = "std")]
            sdf: filter.sdf,
        };
        let swept = {
            let start = core_box(a, b, r);
            let end = core_box(a + d * max_t, b + d * max_t, r);
            AABB::new(
                Vec3Fix::new(
                    min_fix(start.min.x, end.min.x),
                    min_fix(start.min.y, end.min.y),
                    min_fix(start.min.z, end.min.z),
                ),
                Vec3Fix::new(
                    max_fix(start.max.x, end.max.x),
                    max_fix(start.max.y, end.max.y),
                    max_fix(start.max.z, end.max.z),
                ),
            )
        };
        let start_cap = r + r + Fix128::ONE;
        let centre = half_vec(a + b);
        let mut best: Option<WorldShapeHit> = None;
        for_each_target(self, filter, &swept, |target, body, pieces| {
            let mut nearest: Option<Contact> = None;
            for piece in pieces {
                let c = if piece.dist(a, b, start_cap).within(r) {
                    Some(Contact {
                        t: Fix128::ZERO,
                        point: centre,
                        normal: -d,
                    })
                } else {
                    piece.sweep(a, b, r, d, max_t, &settings)
                };
                if let Some(c) = c {
                    if nearest.is_none_or(|n| c.t < n.t) {
                        nearest = Some(c);
                    }
                }
            }
            if let Some(c) = nearest {
                let hit = WorldShapeHit {
                    t: c.t,
                    point: c.point,
                    normal: c.normal,
                    target,
                    body,
                };
                if best.is_none_or(|b| (hit.t, hit.target) < (b.t, b.target)) {
                    best = Some(hit);
                }
            }
        });
        best
    }
}
