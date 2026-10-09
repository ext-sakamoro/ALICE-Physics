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
//! | box (shape, compound child) | ray vs rounded box (3 slabs + 12 edge capsules) | convex time of impact |
//! | cylinder | ray vs rounded cylinder (2 cylinders + 2 rim tori) | convex time of impact |
//! | torus | ray vs torus of tube `minor + r` | arcs of the ring: time of impact against each arc's hull grown by `minor`, arcs split until the hull is within `2⁻³²` of the arc |
//! | compound capsule | ray vs capsule of `radius + r` | convex time of impact |
//! | cone, ellipsoid, wedge, convex hull child | convex time of impact | convex time of impact |
//! | plane | closed form (two-sided) | closed form at the nearer end |
//! | triangle mesh | per triangle: the two offset triangles + 3 edge capsules | convex time of impact per triangle |
//! | height field | per cell: time of impact against the hull of the cell's corners, the cell split into parts until each hull is within `2⁻³²` of the surface (a planar cell is its own hull) | the same, with the segment |
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
//! - A cast that starts overlapping a collider by more than `2⁻³²` reports it at
//!   `t = 0`, normal `−direction`, point the cast shape's centre (the sphere's
//!   centre, the capsule segment's midpoint). Otherwise the hit's point is the
//!   contact point on the collider and the normal is the collider's surface
//!   normal there, pointing toward the cast shape (for surfaces: toward the side
//!   the shape comes from).
//! - A cast that starts touching a collider (within `2⁻³²`) hits it at `t = 0`
//!   only when the direction goes into the surface (`direction·normal < 0`);
//!   moving away from it or along it is not a hit, and the cast goes on to what
//!   lies further along the path. The same holds wherever a cast touches a
//!   surface on its way: a contact whose direction is within `2⁻⁶` of the
//!   tangent plane is a hit only if the path then goes more than `2⁻³²` into
//!   that piece, so a shape resting on a mesh floor or a flat height field
//!   crosses the edges between triangles and cells.
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
//! `1e-12`).
//!
//! The convex time of impact steps by Newton's method on the gap `g(t)` (GJK
//! distance less the radii; GJK stops when its bound is within `2⁻⁴⁰` of the
//! distance). For two convex sets moving apart linearly `g` is convex in `t`,
//! so the tangent's root never passes the first contact, and a gap that is not
//! decreasing proves there is none. It stops when the gap is below `2⁻³²`
//! (checked to `1e-9`); a grazing approach converges linearly instead of
//! quadratically but still in a few dozen steps (an approach `0.1°` from
//! tangent is checked). Meshes, height-field cells and torus arcs are split into
//! convex parts, so moving along one part (a mesh floor) costs nothing against
//! another (a wall of the same mesh).
//!
//! **No missed hits:** a cast reports no hit only when there is none within
//! `max_t` (to the `2⁻³²` gap tolerance). If a step budget runs out (a part
//! hierarchy deeper than its node budget, an SDF traced for more than 4096
//! steps), the cast reports the earliest position it could not rule out, which
//! is never after the true contact.
//!
//! A bilinear height-field cell restricted to a rectangle of its `(u, v)` is
//! again bilinear with its four corners as control points, so it lies inside
//! their convex hull, within a quarter of the part's twist `|h00 − h10 − h01 +
//! h11|·Δu·Δv` of it. Distances to a cell are found by branch-and-bound over
//! such parts (the hull distance bounds a part from below), so they never
//! exceed the true distance by more than `2⁻³²`; casts take the earliest hull
//! contact over parts, splitting a part until its hull is that close to it, so
//! a cast never moves past the surface. The contact is then moved onto the
//! surface by Newton's method. Cells are visited in the order the swept shape
//! reaches their boxes. Segment distances to tori and
//! SDFs minimise a 1-Lipschitz point distance along the segment by
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

/// A touching contact whose motion is closer than this to the surface's tangent
/// plane (`|d·n| < 2⁻⁶`) is a hit only if the path then goes more than
/// [`TRACE_TOLERANCE`] into the piece (see [`dips_below`]); a steeper one is.
const NEAR_TANGENT: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0400_0000_0000_0000,
};

/// Newton steps of a convex time of impact before the last clear time is
/// reported (it needs a few dozen at most, even at a grazing approach).
const TRACE_MAX_STEPS: usize = 1024;

/// Sphere-tracing steps for an SDF before the last position is reported.
#[cfg(feature = "std")]
const SDF_MAX_STEPS: usize = 4096;

/// GJK stops when `|v|² − v·w ≤ |v|² · 2⁻⁴⁰`.
const GJK_RELATIVE: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000_0000_0100_0000,
};

/// GJK treats `|v|² ≤ 2⁻⁶⁴` as intersecting: `|v| ≤ 2⁻³²`, the conservative
/// advancement tolerance. `lo` is a 64-bit fraction (`lo = 1` is `2⁻⁶⁴`, the
/// smallest positive `Fix128`), so this is the smallest threshold a squared
/// length can be compared with; a larger one (`lo = 1 << 48` is `2⁻¹⁶`, `|v| ≤
/// 2⁻⁸`) would count any pair nearer than about `0.0039` as intersecting.
const GJK_INTERSECT: Fix128 = Fix128 { hi: 0, lo: 1 };

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

/// Branch-and-bound parts per height-field cell distance.
const CELL_MAX_NODES: usize = 512;

/// Parts expanded per hierarchical time of impact (one height-field cell).
const TREE_MAX_NODES: usize = 1024;

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

/// Whether a cast of radius `r` starts overlapping a piece at distance `d`: more
/// than [`TRACE_TOLERANCE`] inside it. Within the tolerance it is touching, and
/// the sweep decides by the direction of motion.
fn starts_overlapping(d: &Dist, r: Fix128) -> bool {
    match *d {
        Dist::Inside => true,
        Dist::Outside { dist, .. } => dist + TRACE_TOLERANCE < r,
        Dist::AtLeast(_) => false,
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

/// Below this squared length (`|v| < 2⁻¹⁶`) a length or direction is taken from
/// `v` scaled up by a power of two: `length_squared` keeps only multiples of
/// `2⁻⁶⁴`, so `|v|` near `2⁻³²` would have one or two significant bits.
const FINE_LENGTH_SQUARED: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000_0001_0000_0000,
};

/// `2⁻¹⁶`, the square root of [`FINE_LENGTH_SQUARED`].
const FINE_LENGTH: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0001_0000_0000_0000,
};

/// `v · 2ᵏ` with the largest component in `[1, 2)` and that `k`, for
/// `0 < |v| < 1` (`k > 0`, an exact scaling); zero gives `(ZERO, 0)`.
fn scaled_up(v: Vec3Fix) -> (Vec3Fix, u32) {
    let raw = |f: Fix128| ((f.hi as i128) << 64) | (f.lo as i128);
    let (x, y, z) = (raw(v.x), raw(v.y), raw(v.z));
    let m = x.unsigned_abs().max(y.unsigned_abs()).max(z.unsigned_abs());
    if m == 0 {
        return (Vec3Fix::ZERO, 0);
    }
    // bit 64 is 1.0
    let msb = 127 - m.leading_zeros();
    if msb >= 64 {
        return (v, 0);
    }
    let k = 64 - msb;
    let up = |r: i128| {
        let s = r << k;
        Fix128::from_raw((s >> 64) as i64, s as u64)
    };
    (Vec3Fix::new(up(x), up(y), up(z)), k)
}

/// `|v|`, accurate for a short `v` too (measured on `v` scaled up, see
/// [`FINE_LENGTH_SQUARED`]); `None` for zero. A `v` not shorter than `2⁻¹⁶`
/// gives `v.length()`.
fn fine_length(v: Vec3Fix) -> Option<Fix128> {
    if v.length_squared() >= FINE_LENGTH_SQUARED {
        let len = v.length();
        return (!len.is_zero()).then_some(len);
    }
    let (s, k) = scaled_up(v);
    let len = s.length();
    if len.is_zero() {
        return None;
    }
    let back = (((len.hi as i128) << 64) | (len.lo as i128)) >> k;
    Some(Fix128::from_raw((back >> 64) as i64, back as u64))
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

/// An ellipsoid, whose support mapping normalizes its direction, given that
/// direction scaled up to a largest component
/// in `[1, 2)`: GJK's direction near a contact is about as long as the gap, and
/// such a mapping (which leaves directions down to `2⁻²⁴` unscaled) loses that
/// many bits of the support point, enough to put it about `2⁻³²` off the
/// surface. The scaling is exact and the direction unchanged. Polytopes are not
/// wrapped: their support compares dot products, and a different rounding would
/// only pick another of two tied vertices. Cylinders and cones are not wrapped
/// either: their support jumps between the rims (and the apex), and with the
/// scaled direction casts onto their sides went further in, and some stopped
/// early, so they keep the previous support direction.
struct RoundSupport<'a>(&'a PosedShape);

impl Support for RoundSupport<'_> {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        self.0.support(scaled_up(direction).0)
    }
}

/// Whether a posed shape is wrapped in [`RoundSupport`] (an ellipsoid).
fn is_round(posed: &PosedShape) -> bool {
    matches!(posed.shape, Shape::Ellipsoid { .. })
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

/// A sub-simplex, its weights, and whether it is a face whose point GJK takes
/// by [`within_face`] (see [`closest_on_triangle`]).
type Nearest = (Vec<Vertex>, Vec<Fix128>, bool);

/// The point of the simplex nearest the origin, the sub-simplex it lies on and
/// its barycentric weights; `None` when the origin is inside a tetrahedron.
///
/// The weights are the same for the simplex scaled about the origin, and are
/// computed on it scaled up by `2ᵏ` (exactly) when its edges are shorter than
/// [`SMALL_SIMPLEX`]: their dot products, of the size of an edge to the fourth
/// power, would otherwise keep few significant bits (on a curved surface the
/// simplex shrinks around the nearest point as GJK converges).
fn closest_on_simplex(s: &[Vertex]) -> Option<Nearest> {
    let k = simplex_scale(s);
    if k == 0 {
        return closest_on_scaled_simplex(s, 0);
    }
    let up: Vec<Vertex> = s
        .iter()
        .map(|v| Vertex {
            w: shift_vec(v.w, k as i32),
            ..*v
        })
        .collect();
    let (sub, lambda, face) = closest_on_scaled_simplex(&up, k)?;
    let sub = sub
        .into_iter()
        .map(|v| Vertex {
            w: shift_vec(v.w, -(k as i32)),
            ..v
        })
        .collect();
    Some((sub, lambda, face))
}

/// [`closest_on_simplex`] for `s` scaled up by `2ᵏ`.
fn closest_on_scaled_simplex(s: &[Vertex], k: u32) -> Option<Nearest> {
    match s.len() {
        1 => Some((s.to_vec(), vec![Fix128::ONE], false)),
        2 => {
            let (sub, lambda) = closest_on_edge(s[0], s[1]);
            Some((sub, lambda, false))
        }
        3 => Some(closest_on_triangle(s[0], s[1], s[2])),
        _ => closest_on_tetrahedron(s[0], s[1], s[2], s[3], k),
    }
}

/// A simplex whose edges have every component below this (`2⁻⁸`) is scaled up
/// for its barycentric weights (see [`closest_on_simplex`]).
const SMALL_SIMPLEX_BITS: u32 = 56;

/// The `k` that brings the largest edge component of `s` to `[1, 2)`, `0` when
/// it is at least `2⁻⁸` (or the simplex is a point), and limited so that no
/// vertex gets a component of `2²⁴` or more.
fn simplex_scale(s: &[Vertex]) -> u32 {
    let raw = |f: Fix128| ((f.hi as i128) << 64) | (f.lo as i128);
    let mag = |v: Vec3Fix| {
        raw(v.x)
            .unsigned_abs()
            .max(raw(v.y).unsigned_abs())
            .max(raw(v.z).unsigned_abs())
    };
    let Some(first) = s.first() else {
        return 0;
    };
    let edge = s.iter().map(|v| mag(v.w - first.w)).max().unwrap_or(0);
    if edge == 0 {
        return 0;
    }
    // bit 64 is 1.0
    let msb = 127 - edge.leading_zeros();
    if msb >= SMALL_SIMPLEX_BITS {
        return 0;
    }
    let far = s.iter().map(|v| mag(v.w)).max().unwrap_or(0);
    let far_msb = 127 - far.leading_zeros();
    // keep components below 2²⁴ (bit 88)
    (64 - msb).min(88u32.saturating_sub(far_msb))
}

/// `v · 2ᵏ` (`k` may be negative), exact when no bit leaves the range.
fn shift_vec(v: Vec3Fix, k: i32) -> Vec3Fix {
    let raw = |f: Fix128| ((f.hi as i128) << 64) | (f.lo as i128);
    let sh = |f: Fix128| {
        let r = raw(f);
        let s = if k >= 0 { r << k } else { r >> (-k) };
        Fix128::from_raw((s >> 64) as i64, s as u64)
    };
    Vec3Fix::new(sh(v.x), sh(v.y), sh(v.z))
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

/// [`log2_size`] of zero.
const NO_SIZE: i32 = i32::MIN / 4;

/// `⌊log₂⌋` of the largest component of `v` ([`NO_SIZE`] for zero).
fn log2_size(v: Vec3Fix) -> i32 {
    let raw = |f: Fix128| ((f.hi as i128) << 64) | (f.lo as i128);
    let m = raw(v.x)
        .unsigned_abs()
        .max(raw(v.y).unsigned_abs())
        .max(raw(v.z).unsigned_abs());
    if m == 0 {
        return NO_SIZE;
    }
    // bit 64 is 1.0
    63 - m.leading_zeros() as i32
}

/// Whether the barycentric form gives `point`, the nearest point of triangle
/// `a b c`, to within `2⁻³²` of its length, so that its direction (GJK's next
/// search direction) is within `2⁻³²` rad. The ratio does not depend on a
/// scaling of the simplex.
///
/// Its region tests and weights are differences of products of two dot
/// products: each dot product of vectors of sizes `L` (an edge) and `P` (a
/// vertex) is rounded to about `2⁻⁶²·(L + P)`, so a weight, such a difference
/// over `|n|²` (`n = (b − a) × (c − a)`), is off by about `2⁻⁶²·L·P / |n|²`,
/// and the point by `L` times that. On a curved solid the simplex near the
/// surface is a long thin triangle of support points far from the origin
/// (`|n|` a small fraction of `L²`): the point was off by up to `2⁻²⁶` there,
/// its direction turned the next support back to a vertex already in the
/// simplex, and GJK stopped up to `2⁻²⁶` away from the surface. The faces of
/// the polytopes cast at (boxes, wedges, meshes) stay well below the bound and
/// keep the barycentric form. The sizes are taken from the leading bits of
/// the components, so the estimate is within a factor of 8.
fn barycentric_is_precise(a: Vertex, b: Vertex, c: Vertex, point: Vec3Fix) -> bool {
    let ab = b.w - a.w;
    let ac = c.w - a.w;
    let n = log2_size(ab.cross(ac));
    let near = log2_size(point);
    if n == NO_SIZE || near == NO_SIZE {
        return false;
    }
    let l = log2_size(ab).max(log2_size(ac)).max(log2_size(c.w - b.w));
    let p = log2_size(a.w).max(log2_size(b.w)).max(log2_size(c.w));
    // log₂ of the point's error over its length
    -62 + 2 * l + p - 2 * n - near <= -32
}

/// The point of triangle `a b c` nearest the origin: by
/// [`closest_on_triangle_barycentric`] when that is precise (see
/// [`barycentric_is_precise`]), otherwise by [`closest_on_triangle_by_area`],
/// whose face point GJK then takes by [`within_face`] (the third element).
fn closest_on_triangle(a: Vertex, b: Vertex, c: Vertex) -> Nearest {
    let (sub, lambda) = closest_on_triangle_barycentric(a, b, c);
    if barycentric_is_precise(a, b, c, weighted(&(sub.clone(), lambda.clone()))) {
        return (sub, lambda, false);
    }
    let (sub, lambda) = closest_on_triangle_by_area(a, b, c);
    let face = sub.len() == 3;
    (sub, lambda, face)
}

/// The point of triangle `a b c` nearest the origin, its sub-simplex and
/// weights, for a triangle whose barycentric form is not precise (see
/// [`barycentric_is_precise`]). The regions are decided by the signed areas
/// `n·((q − p) × (o − p))` of the origin `o` projected on the plane, one per
/// edge `p q` (`n = (b − a) × (c − a)`, scaled up to a largest component in
/// `[1, 2)`): each is one cross product of edge vectors and a dot product, so
/// its rounding is about `2⁻⁶²` of the size of the triangle, and a weight,
/// such an area over their sum `n·n`, is off by that over `|n|`. The
/// barycentric form ([`closest_on_triangle_barycentric`]) takes the same
/// quantities as differences of products of dot products, whose weights are
/// off by that size squared over `|n|²`.
///
/// When the origin is outside an edge (a negative area), the nearest point is
/// on such an edge (the nearest point of a convex polygon to a point outside it
/// is on an edge whose line separates them), and the nearest of those edges is
/// taken.
fn closest_on_triangle_by_area(a: Vertex, b: Vertex, c: Vertex) -> (Vec<Vertex>, Vec<Fix128>) {
    let ab = b.w - a.w;
    let ac = c.w - a.w;
    let bc = c.w - b.w;
    let n = scaled_up(ab.cross(ac)).0;
    // the area for the edge opposite each vertex (positive inside)
    let area_a = n.dot(bc.cross(-b.w));
    let area_b = n.dot((-ac).cross(-c.w));
    let area_c = n.dot(ab.cross(-a.w));
    let sum = area_a + area_b + area_c;
    if n.length_squared().is_zero() || sum <= Fix128::ZERO {
        // A degenerate (flat) triangle: the nearest of its edges.
        let mut best = closest_on_edge(a, b);
        for cand in [closest_on_edge(b, c), closest_on_edge(a, c)] {
            if weighted(&cand).length_squared() < weighted(&best).length_squared() {
                best = cand;
            }
        }
        return best;
    }
    if !area_a.is_negative() && !area_b.is_negative() && !area_c.is_negative() {
        let (wb, wc) = (area_b / sum, area_c / sum);
        return (vec![a, b, c], vec![Fix128::ONE - wb - wc, wb, wc]);
    }
    let mut best: Option<(Vec<Vertex>, Vec<Fix128>)> = None;
    for (area, p, q) in [(area_c, a, b), (area_a, b, c), (area_b, a, c)] {
        if !area.is_negative() {
            continue;
        }
        let cand = closest_on_edge(p, q);
        if best
            .as_ref()
            .is_none_or(|b| weighted(&cand).length_squared() < weighted(b).length_squared())
        {
            best = Some(cand);
        }
    }
    // not reached: one of the areas is negative here
    best.unwrap_or_else(|| (vec![a], vec![Fix128::ONE]))
}

/// Ericson, *Real-Time Collision Detection* §5.1.5, for the origin.
fn closest_on_triangle_barycentric(a: Vertex, b: Vertex, c: Vertex) -> (Vec<Vertex>, Vec<Fix128>) {
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

/// The nearest point of tetrahedron `a b c d` (scaled up by `2ᵏ`), `None` when
/// the origin is inside it.
///
/// A tetrahedron whose least height (the volume `n·(d − a)` over its largest
/// face's `|n|`, both from differences of its vertices) is at most `2⁻³²` (in
/// unscaled units) is taken as flat, and the nearest of its four faces
/// returned: four support points on one circle or one generator of a curved
/// solid are coplanar but for rounding, and the sign of such a volume (and of
/// the origin's side of a face) is that rounding, which counted the origin
/// inside and reported an intersection far from the surface. Taking it as flat
/// moves the result by at most that height: an origin within it is within
/// `2⁻³²` of a face, the distance GJK counts as intersecting.
fn closest_on_tetrahedron(a: Vertex, b: Vertex, c: Vertex, d: Vertex, k: u32) -> Option<Nearest> {
    let faces = [(a, b, c, d), (a, c, d, b), (a, d, b, c), (b, d, c, a)];
    let flat_height = if k >= 32 {
        Fix128::from_raw(1i64 << (k - 32), 0)
    } else {
        Fix128::from_raw(0, 1u64 << (32 + k))
    };
    let flat = faces.iter().any(|&(p, q, r, opposite)| {
        let n = (q.w - p.w).cross(r.w - p.w);
        n.dot(opposite.w - p.w).abs() <= n.length() * flat_height
    });
    let mut best: Option<Nearest> = None;
    let mut inside = true;
    for (p, q, r, opposite) in faces {
        let n = (q.w - p.w).cross(r.w - p.w);
        let origin_side = n.dot(-p.w);
        let other_side = n.dot(opposite.w - p.w);
        // The origin is outside this face when it is strictly on the other side
        // from the fourth vertex; every face of a flat tetrahedron counts.
        let outside = flat
            || other_side.is_zero()
            || (!origin_side.is_zero() && origin_side.is_negative() != other_side.is_negative());
        if outside {
            inside = false;
            let cand = closest_on_triangle(p, q, r);
            if best.as_ref().is_none_or(|b| {
                nearest_point(&cand).length_squared() < nearest_point(b).length_squared()
            }) {
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

/// The point of face `s` with weights `lambda`, from its first vertex along its
/// edges: `s₀ + λ₁(s₁ − s₀) + λ₂(s₂ − s₀)`.
///
/// For a face whose barycentric form is not precise (see
/// [`closest_on_triangle`]): such a face is long and thin, its plane (the cross
/// product of a long edge and a short one) is not known to more than the
/// rounding of the short edge, and a projection on it misses the distance by
/// that tilt, in either direction (a cast stopped short of the surface). A
/// point of the face is in the Minkowski difference, so its length is never
/// below the distance, and the error of its weights moves it within the face,
/// across the thin side (along the surface): its length changes only by the
/// square of that over the distance. Taken from `s₀`, the edges are
/// differences of the support points, exact, and the point is not rounded to
/// the size of the vertices.
fn within_face(s: &[Vertex], lambda: &[Fix128]) -> Vec3Fix {
    s[0].w + (s[1].w - s[0].w) * lambda[1] + (s[2].w - s[0].w) * lambda[2]
}

/// The point of a [`Nearest`] by its weights.
fn nearest_point(n: &Nearest) -> Vec3Fix {
    n.0.iter()
        .zip(&n.1)
        .fold(Vec3Fix::ZERO, |acc, (v, &l)| acc + v.w * l)
}

/// The point of the line or plane through the sub-simplex `s` (1 to 3 vertices,
/// the one [`closest_on_simplex`] chose) nearest the origin, by projection: when
/// the simplex is near the origin its barycentric combination cancels terms of
/// the size of its vertices and keeps few significant bits, while `a − (a·ê)ê`
/// and `(a·n̂)n̂` lose only rounding of that size.
fn nearest_on_affine_hull(s: &[Vertex]) -> Vec3Fix {
    match *s {
        [a] => a.w,
        [a, b] => {
            let e = scaled_up(b.w - a.w).0;
            let ee = e.length_squared();
            if ee.is_zero() {
                return a.w;
            }
            a.w - e * (a.w.dot(e) / ee)
        }
        [a, b, c, ..] => {
            let n = scaled_up((b.w - a.w).cross(c.w - a.w)).0;
            let nn = n.length_squared();
            if nn.is_zero() {
                return a.w;
            }
            n * (a.w.dot(n) / nn)
        }
        [] => Vec3Fix::ZERO,
    }
}

fn weighted(s: &(Vec<Vertex>, Vec<Fix128>)) -> Vec3Fix {
    s.0.iter()
        .zip(&s.1)
        .fold(Vec3Fix::ZERO, |acc, (v, &l)| acc + v.w * l)
}

/// The distance between two convex sets by GJK: `(distance, point on a, point on
/// b)`, or `None` when they intersect (closer than `2⁻³²`).
fn gjk_distance<A: Support, B: Support>(a: &A, b: &B) -> Option<(Fix128, Vec3Fix, Vec3Fix)> {
    let mut simplex = vec![vertex(a, b, Vec3Fix::UNIT_X)];
    let mut weights = vec![Fix128::ONE];
    let mut v = simplex[0].w;
    // `v` was taken within a face (see `within_face`).
    let mut in_face = false;
    for _ in 0..GJK_MAX_ITERATIONS {
        let vv = v.length_squared();
        if vv <= GJK_INTERSECT {
            return None;
        }
        // A short `v` is compared through `u`, `v` scaled up: `|v|² − v·w ≤
        // |v|²·ε` is `u·(v − w) ≤ (u·v)·ε`, whose terms keep their precision
        // where `|v|²` is a few multiples of `2⁻⁶⁴`.
        let fine = vv < FINE_LENGTH_SQUARED;
        let u = if fine { scaled_up(v).0 } else { v };
        let w = vertex(a, b, -u);
        let converged = if fine {
            u.dot(v - w.w) <= u.dot(v) * GJK_RELATIVE
        } else {
            vv - v.dot(w.w) <= vv * GJK_RELATIVE
        };
        if converged {
            break;
        }
        if simplex.iter().any(|s| s.w == w.w) {
            break;
        }
        let mut grown = simplex.clone();
        grown.push(w);
        let (sub, lambda, face) = closest_on_simplex(&grown)?;
        let next = if face {
            within_face(&sub, &lambda)
        } else if fine {
            nearest_on_affine_hull(&sub)
        } else {
            weighted(&(sub.clone(), lambda.clone()))
        };
        let no_progress = if fine {
            // Both short: compare the lengths measured on the scaled vectors.
            let len = |x: Vec3Fix| fine_length(x).unwrap_or(Fix128::ZERO);
            len(next) >= len(v)
        } else {
            next.length_squared() >= vv
        };
        if no_progress {
            // No progress (rounding): keep the previous simplex.
            break;
        }
        simplex = sub;
        weights = lambda;
        in_face = face;
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
    let dist = fine_length(v)?;
    if v.length_squared() < FINE_LENGTH_SQUARED || in_face {
        // `v` is projected (see `nearest_on_affine_hull`) or taken within a
        // face from its first vertex (see `within_face`), the weights are not
        // as precise: the witness on `a` is taken as `pb + v`, so that `pa − pb`
        // is `v` exactly.
        return Some((dist, pb + v, pb));
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
            normal: gjk_normal(pa - pb, dist),
        },
    }
}

/// The unit normal along `diff` (`pa − pb` of [`gjk_distance`]) of length
/// `dist`: `diff / dist`, or for a short `diff` the scaled-up `diff` normalized,
/// since a quotient of two lengths near `2⁻³²` is unit only to about `2⁻³³`.
fn gjk_normal(diff: Vec3Fix, dist: Fix128) -> Vec3Fix {
    if diff.length_squared() >= FINE_LENGTH_SQUARED {
        return diff / dist;
    }
    let s = scaled_up(diff).0;
    s / s.length()
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
            normal: gjk_normal(pa - pb, d),
        }
        .shrunk(capsule.radius),
    }
}

/// The distance from `p` to a solid box of half-extents `h` about the origin, in
/// its frame.
fn point_box_local(p: Vec3Fix, h: Vec3Fix) -> Dist {
    if p.x.abs() < h.x && p.y.abs() < h.y && p.z.abs() < h.z {
        return Dist::Inside;
    }
    let q = Vec3Fix::new(
        clamp(p.x, -h.x, h.x),
        clamp(p.y, -h.y, h.y),
        clamp(p.z, -h.z, h.z),
    );
    if q == p {
        // On the boundary: touching, with the normal of a face it lies on.
        let axis = |v: Fix128, e: Fix128, unit: Vec3Fix| {
            (v.abs() == e).then(|| if v.is_negative() { -unit } else { unit })
        };
        let z = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
        let normal = axis(p.x, h.x, Vec3Fix::UNIT_X)
            .or_else(|| axis(p.y, h.y, Vec3Fix::UNIT_Y))
            .or_else(|| axis(p.z, h.z, z))
            .unwrap_or(Vec3Fix::UNIT_Y);
        return Dist::Outside {
            dist: Fix128::ZERO,
            point: p,
            normal,
        };
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
    if (q - p).length_squared().is_zero() {
        // On the boundary: touching, with the cap or side normal there.
        let normal = if p.y.abs() == hh {
            if p.y.is_negative() {
                -Vec3Fix::UNIT_Y
            } else {
                Vec3Fix::UNIT_Y
            }
        } else {
            dir
        };
        return Dist::Outside {
            dist: Fix128::ZERO,
            point: p,
            normal,
        };
    }
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

/// Four points, for GJK: their convex hull. The control points of a bilinear
/// patch, whose hull contains the patch.
struct QuadSupport([Vec3Fix; 4]);

impl Support for QuadSupport {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        let mut best = self.0[0];
        let mut best_dot = best.dot(direction);
        for &p in &self.0[1..] {
            let d = p.dot(direction);
            if d > best_dot {
                best = p;
                best_dot = d;
            }
        }
        best
    }
}

/// One bilinear height-field cell: first corner `(x0, z0)`, spacing `s`, corner
/// heights `(h00, h10, h01, h11)`.
#[derive(Clone, Copy)]
struct Cell {
    x0: Fix128,
    z0: Fix128,
    s: Fix128,
    h: [Fix128; 4],
}

impl Cell {
    fn new(field: &HeightField, gx: u32, gz: u32) -> Self {
        let (x0, z0) = cell_origin(field, gx, gz);
        Self {
            x0,
            z0,
            s: field.spacing,
            h: cell_heights(field, gx, gz),
        }
    }

    /// The bilinear height at `(u, v) ∈ [0, 1]²`.
    fn height(&self, u: Fix128, v: Fix128) -> Fix128 {
        let [h00, h10, h01, h11] = self.h;
        let one = Fix128::ONE;
        h00 * (one - u) * (one - v) + h10 * u * (one - v) + h01 * (one - u) * v + h11 * u * v
    }

    fn point(&self, u: Fix128, v: Fix128) -> Vec3Fix {
        Vec3Fix::new(
            self.x0 + self.s * u,
            self.height(u, v),
            self.z0 + self.s * v,
        )
    }

    /// `h00 − h10 − h01 + h11`: zero for a planar cell.
    fn twist(&self) -> Fix128 {
        let [h00, h10, h01, h11] = self.h;
        h00 - h10 - h01 + h11
    }

    /// The unit upward normal of the surface at `(u, v)`.
    fn normal(&self, u: Fix128, v: Fix128) -> Vec3Fix {
        let [h00, h10, h01, h11] = self.h;
        let one = Fix128::ONE;
        let hu = (h10 - h00) * (one - v) + (h11 - h01) * v;
        let hv = (h01 - h00) * (one - u) + (h11 - h10) * u;
        Vec3Fix::new(-hu, self.s, -hv).normalize()
    }

    /// The `(u, v)` of the world `(x, z)` of `p`, clamped to `[u0, u1] × [v0, v1]`.
    fn uv(&self, p: Vec3Fix, part: &Patch) -> (Fix128, Fix128) {
        (
            clamp((p.x - self.x0) / self.s, part.u0, part.u1),
            clamp((p.z - self.z0) / self.s, part.v0, part.v1),
        )
    }
}

/// The part `u ∈ [u0, u1]`, `v ∈ [v0, v1]` of a [`Cell`]. A bilinear patch
/// restricted to a rectangle of `(u, v)` is again bilinear, with its four corners
/// as control points, so it lies inside the convex hull of those corners.
#[derive(Clone, Copy)]
struct Patch {
    u0: Fix128,
    u1: Fix128,
    v0: Fix128,
    v1: Fix128,
}

impl Patch {
    const WHOLE: Self = Self {
        u0: Fix128::ZERO,
        u1: Fix128::ONE,
        v0: Fix128::ZERO,
        v1: Fix128::ONE,
    };

    fn hull(&self, cell: &Cell) -> QuadSupport {
        QuadSupport([
            cell.point(self.u0, self.v0),
            cell.point(self.u1, self.v0),
            cell.point(self.u0, self.v1),
            cell.point(self.u1, self.v1),
        ])
    }

    /// A bound on how far a point of the hull is from the patch: the hull lies
    /// between the two triangulations of the corners, which differ from the
    /// bilinear surface by at most a quarter of the part's twist
    /// `|k|·(u1 − u0)·(v1 − v0)` along `Y`.
    fn thickness(&self, cell: &Cell) -> Fix128 {
        (cell.twist().abs() * (self.u1 - self.u0) * (self.v1 - self.v0))
            .half()
            .half()
    }

    fn split(&self) -> [Self; 4] {
        let um = (self.u0 + self.u1).half();
        let vm = (self.v0 + self.v1).half();
        [
            Self {
                u1: um,
                v1: vm,
                ..*self
            },
            Self {
                u0: um,
                v1: vm,
                ..*self
            },
            Self {
                u1: um,
                v0: vm,
                ..*self
            },
            Self {
                u0: um,
                v0: vm,
                ..*self
            },
        ]
    }

    /// The surface point of this part over the `(x, z)` of `p`.
    fn surface_under(&self, cell: &Cell, p: Vec3Fix) -> Vec3Fix {
        let (u, v) = cell.uv(p, self);
        cell.point(u, v)
    }
}

/// The distance from the core `a`–`b` to one cell and the nearest surface point
/// found, by branch-and-bound over parts of the cell: the distance to a part's
/// hull is a lower bound for the part, the distance to the surface point under
/// the hull's nearest point an upper bound, and parts are split until the two
/// meet within [`TRACE_TOLERANCE`]. The result is never above the true distance
/// by more than that tolerance; if the node budget runs out it is the lowest
/// open lower bound (below the true distance).
fn cell_dist(cell: &Cell, a: Vec3Fix, b: Vec3Fix) -> (Fix128, Vec3Fix) {
    let to_core = |q: Vec3Fix| (closest_on_segment(a, b, q) - q).length();
    let eval = |part: &Patch| {
        let hull = part.hull(cell);
        let found = if a == b {
            gjk_distance(&PointSupport(a), &hull)
        } else {
            gjk_distance(&SegmentSupport(a, b), &hull)
        };
        let (lower, near) = match found {
            Some((d, _, pb)) => (d, pb),
            None => (
                Fix128::ZERO,
                cell.point((part.u0 + part.u1).half(), (part.v0 + part.v1).half()),
            ),
        };
        let q = part.surface_under(cell, near);
        (lower, q, to_core(q))
    };
    let (lower, q, upper) = eval(&Patch::WHOLE);
    let mut best = (upper, q);
    let mut open = vec![(lower, Patch::WHOLE)];
    let mut nodes = 0usize;
    while let Some(k) = (0..open.len()).min_by_key(|&k| open[k].0) {
        let (lower, part) = open.remove(k);
        // Every open part is at least `lower` away: nothing can beat the best by
        // more than the tolerance.
        if lower + TRACE_TOLERANCE >= best.0 {
            break;
        }
        nodes += 1;
        if nodes > CELL_MAX_NODES {
            return (lower, best.1);
        }
        for child in part.split() {
            let (lower, q, upper) = eval(&child);
            if upper < best.0 {
                best = (upper, q);
            }
            // A part thinner than the tolerance has its upper bound within the
            // tolerance of its lower bound, so it is settled by `best`.
            if lower + TRACE_TOLERANCE < best.0 && child.thickness(cell) > TRACE_TOLERANCE {
                open.push((lower, child));
            }
        }
    }
    best
}

/// The distance from the core `a`–`b` to the height-field surface, searching the
/// cells within `cap` of it ([`Dist::AtLeast`] beyond). A core within
/// [`TRACE_TOLERANCE`] of the surface is touching: a point gets distance `0` and
/// the surface normal on its side, a segment is [`Dist::Inside`] (it meets the
/// surface).
fn heightfield_dist(field: &HeightField, a: Vec3Fix, b: Vec3Fix, cap: Fix128) -> Dist {
    if !field_has_surface(field) {
        return Dist::AtLeast(cap);
    }
    let core = core_box(a, b, Fix128::ZERO);
    let far = aabb_gap(&core, &field.aabb());
    if far >= cap {
        return Dist::AtLeast(far);
    }
    let reach = core_box(a, b, cap);
    let s = field.spacing;
    let (Some((gx0, gx1)), Some((gz0, gz1))) = (
        cell_range(reach.min.x, reach.max.x, field.origin.x, s, field.width),
        cell_range(reach.min.z, reach.max.z, field.origin.z, s, field.depth),
    ) else {
        return Dist::AtLeast(cap);
    };
    // The cells by their box's distance from the core, nearest first: once a
    // cell's box is farther than the best distance found, no later cell is nearer.
    let mut cells: Vec<(Fix128, u32, u32)> = Vec::new();
    for gz in gz0..=gz1 {
        for gx in gx0..=gx1 {
            let lower = aabb_gap(&core, &cell_box(field, gx, gz));
            if lower < cap {
                cells.push((lower, gz, gx));
            }
        }
    }
    cells.sort_unstable();
    let mut best: Option<(Fix128, Vec3Fix, Cell)> = None;
    for (lower, gz, gx) in cells {
        if best.is_some_and(|(d, _, _)| lower >= d) {
            break;
        }
        let cell = Cell::new(field, gx, gz);
        let (d, q) = cell_dist(&cell, a, b);
        if best.is_none_or(|(bd, _, _)| d < bd) {
            best = Some((d, q, cell));
        }
    }
    let Some((d, q, cell)) = best.filter(|&(d, _, _)| d < cap) else {
        return Dist::AtLeast(cap);
    };
    let core_point = closest_on_segment(a, b, q);
    if d <= TRACE_TOLERANCE {
        if a != b {
            return Dist::Inside;
        }
        let (u, v) = cell.uv(q, &Patch::WHOLE);
        let n = cell.normal(u, v);
        let side = if (a - q).dot(n).is_negative() { -n } else { n };
        return Dist::Outside {
            dist: d,
            point: q,
            normal: side,
        };
    }
    let normal = match (core_point - q).try_normalize() {
        Some(n) => n,
        None => return Dist::Inside,
    };
    Dist::Outside {
        dist: d,
        point: q,
        normal,
    }
}

/// The box of cell `(gx, gz)`: its `XZ` square and the range of its corner
/// heights (the bilinear surface stays within them).
fn cell_box(field: &HeightField, gx: u32, gz: u32) -> AABB {
    let h = cell_heights(field, gx, gz);
    let (x0, z0) = cell_origin(field, gx, gz);
    let s = field.spacing;
    let lo = min_fix(min_fix(h[0], h[1]), min_fix(h[2], h[3]));
    let hi = max_fix(max_fix(h[0], h[1]), max_fix(h[2], h[3]));
    AABB::new(Vec3Fix::new(x0, lo, z0), Vec3Fix::new(x0 + s, hi, z0 + s))
}

/// The first time the core `a`–`b` grown by `r`, moving along the unit `d`,
/// touches one cell: against the hull of the whole cell when it is planar (the
/// hull is the cell), otherwise by [`toi_tree`] over its parts, which accepts a
/// part once its hull is within [`TRACE_TOLERANCE`] of its surface.
fn cell_toi(
    cell: &Cell,
    a: Vec3Fix,
    b: Vec3Fix,
    r: Fix128,
    d: Vec3Fix,
    max_t: Fix128,
) -> Option<Contact> {
    let toi = |part: &Patch| toi_convex(a, b, r, Fix128::ZERO, d, max_t, &part.hull(cell));
    if cell.twist().is_zero() {
        return toi(&Patch::WHOLE);
    }
    let found = toi_tree(
        &[Patch::WHOLE],
        toi,
        |part| part.thickness(cell) <= TRACE_TOLERANCE,
        |part| part.split().to_vec(),
    )?;
    let off = d * found.t;
    Some(refine_on_cell(cell, a + off, b + off, found))
}

/// A contact found on a part's hull moved onto the surface: Newton's method on
/// the squared distance from the core to the bilinear surface, started at the
/// contact's `(u, v)`. The hull contact is within the part (a few `2⁻¹⁶` of the
/// cell), where the distance has a single minimum; the refined point is taken
/// only if it is not farther from the core than the hull contact.
fn refine_on_cell(cell: &Cell, a: Vec3Fix, b: Vec3Fix, found: Contact) -> Contact {
    let [h00, h10, h01, h11] = cell.h;
    let k = cell.twist();
    let s = cell.s;
    let one = Fix128::ONE;
    let (mut u, mut v) = cell.uv(found.point, &Patch::WHOLE);
    for _ in 0..8 {
        let q = cell.point(u, v);
        let e = q - closest_on_segment(a, b, q);
        let hu = (h10 - h00) * (one - v) + (h11 - h01) * v;
        let hv = (h01 - h00) * (one - u) + (h11 - h10) * u;
        let pu = Vec3Fix::new(s, hu, Fix128::ZERO);
        let pv = Vec3Fix::new(Fix128::ZERO, hv, s);
        let (gu, gv) = (e.dot(pu), e.dot(pv));
        let (huu, hvv) = (pu.dot(pu), pv.dot(pv));
        let huv = pu.dot(pv) + e.y * k;
        let det = huu * hvv - huv * huv;
        if det <= Fix128::ZERO {
            return found;
        }
        u = clamp(u - (hvv * gu - huv * gv) / det, Fix128::ZERO, one);
        v = clamp(v - (huu * gv - huv * gu) / det, Fix128::ZERO, one);
    }
    let q = cell.point(u, v);
    let core = closest_on_segment(a, b, q);
    let before = (closest_on_segment(a, b, found.point) - found.point).length();
    match (core - q).normalize_with_length() {
        (n, dist)
            if !dist.is_zero()
                && dist <= before + TRACE_TOLERANCE + TRACE_TOLERANCE
                && n.dot(found.normal) > Fix128::ZERO =>
        {
            Contact {
                t: found.t,
                point: q,
                normal: n,
            }
        }
        _ => found,
    }
}

/// The first contact of the core `a`–`b` grown by `r` moving along the unit `d`
/// with a height field: every cell its swept box crosses, in order of the time
/// the moving box first reaches the cell's box, stopping once that time is past
/// the best contact.
fn sweep_heightfield(
    field: &HeightField,
    a: Vec3Fix,
    b: Vec3Fix,
    r: Fix128,
    d: Vec3Fix,
    max_t: Fix128,
) -> Option<Contact> {
    if !field_has_surface(field) {
        return None;
    }
    let start = core_box(a, b, r);
    let swept = swept_box(a, b, r, d, max_t);
    let s = field.spacing;
    let (Some((gx0, gx1)), Some((gz0, gz1))) = (
        cell_range(swept.min.x, swept.max.x, field.origin.x, s, field.width),
        cell_range(swept.min.z, swept.max.z, field.origin.z, s, field.depth),
    ) else {
        return None;
    };
    let mut cells: Vec<(Fix128, u32, u32)> = Vec::new();
    for gz in gz0..=gz1 {
        for gx in gx0..=gx1 {
            if let Some(entry) = box_entry(&start, d, max_t, &cell_box(field, gx, gz)) {
                cells.push((entry, gz, gx));
            }
        }
    }
    cells.sort_unstable();
    let mut best: Option<Contact> = None;
    for (entry, gz, gx) in cells {
        if best.is_some_and(|c| c.t < entry) {
            break;
        }
        if let Some(c) = cell_toi(&Cell::new(field, gx, gz), a, b, r, d, max_t) {
            if best.is_none_or(|bc| c.t < bc.t) {
                best = Some(c);
            }
        }
    }
    best
}

/// `num / den` limited to `[−big, big]` (no overflow for a tiny `den`).
fn ratio_within(num: Fix128, den: Fix128, big: Fix128) -> Fix128 {
    if num.abs() >= den.abs() * big {
        if num.is_negative() == den.is_negative() {
            big
        } else {
            -big
        }
    } else {
        num / den
    }
}

/// The first `t ∈ [0, max_t]` at which `moving` shifted by `t·d` meets `target`
/// (boxes touching count), or `None`.
fn box_entry(moving: &AABB, d: Vec3Fix, max_t: Fix128, target: &AABB) -> Option<Fix128> {
    let big = max_t + Fix128::ONE;
    let mut lo = Fix128::ZERO;
    let mut hi = max_t;
    for (m0, m1, t0, t1, di) in [
        (moving.min.x, moving.max.x, target.min.x, target.max.x, d.x),
        (moving.min.y, moving.max.y, target.min.y, target.max.y, d.y),
        (moving.min.z, moving.max.z, target.min.z, target.max.z, d.z),
    ] {
        if di.abs() < PARALLEL_EPSILON {
            // Nearly no motion on this axis: at most max_t·|di| either way.
            let slack = max_t * di.abs();
            if m1 + slack < t0 || m0 - slack > t1 {
                return None;
            }
            continue;
        }
        let (enter, leave) = if di.is_negative() {
            (
                ratio_within(t1 - m0, di, big),
                ratio_within(t0 - m1, di, big),
            )
        } else {
            (
                ratio_within(t0 - m1, di, big),
                ratio_within(t1 - m0, di, big),
            )
        };
        lo = max_fix(lo, enter);
        hi = min_fix(hi, leave);
    }
    (lo <= hi).then_some(lo)
}

/// The box swept by the core `a`–`b` grown by `r` moving along `d` for `max_t`.
fn swept_box(a: Vec3Fix, b: Vec3Fix, r: Fix128, d: Vec3Fix, max_t: Fix128) -> AABB {
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
}

/// The first contact of the core `a`–`b` grown by `r` moving along the unit `d`
/// with a triangle mesh: a convex time of impact per triangle the swept box
/// meets, in order of the time the moving box reaches the triangle's box. Each
/// triangle is convex, so running along one triangle (a floor) is not a step
/// budget spent against another (a wall).
fn sweep_mesh(
    mesh: &TriMesh,
    a: Vec3Fix,
    b: Vec3Fix,
    r: Fix128,
    d: Vec3Fix,
    max_t: Fix128,
) -> Option<Contact> {
    let start = core_box(a, b, r);
    let mut order: Vec<(Fix128, u32)> = mesh_candidates(mesh, &swept_box(a, b, r, d, max_t))
        .into_iter()
        .filter_map(|i| {
            let tri = &mesh.triangles[i as usize];
            let tri_box = AABB::new(
                Vec3Fix::new(
                    min_fix(min_fix(tri.v0.x, tri.v1.x), tri.v2.x),
                    min_fix(min_fix(tri.v0.y, tri.v1.y), tri.v2.y),
                    min_fix(min_fix(tri.v0.z, tri.v1.z), tri.v2.z),
                ),
                Vec3Fix::new(
                    max_fix(max_fix(tri.v0.x, tri.v1.x), tri.v2.x),
                    max_fix(max_fix(tri.v0.y, tri.v1.y), tri.v2.y),
                    max_fix(max_fix(tri.v0.z, tri.v1.z), tri.v2.z),
                ),
            );
            box_entry(&start, d, max_t, &tri_box).map(|entry| (entry, i))
        })
        .collect();
    order.sort_unstable();
    let mut best: Option<Contact> = None;
    for (entry, i) in order {
        if best.is_some_and(|c| c.t < entry) {
            break;
        }
        let tri = &mesh.triangles[i as usize];
        if let Some(c) = toi_convex(a, b, r, Fix128::ZERO, d, max_t, &TriangleSupport(tri)) {
            if best.is_none_or(|bc| c.t < bc.t) {
                best = Some(c);
            }
        }
    }
    best
}

/// The ring of a torus (radius `major` in its local `XZ` plane) and its tube
/// radius: the solid torus is the ring grown by `minor`.
#[derive(Clone, Copy)]
struct Ring {
    center: Vec3Fix,
    rotation: QuatFix,
    major: Fix128,
    minor: Fix128,
}

/// The arc of a [`Ring`] from the unit local direction `e0` counter-clockwise
/// (from `+X` toward `+Z`) to `e1`, less than half a turn.
#[derive(Clone, Copy)]
struct RingArc {
    e0: Vec3Fix,
    e1: Vec3Fix,
}

/// `(a × b)·Y` for vectors in the `XZ` plane: positive when `b` is
/// counter-clockwise from `a` (from `+X` toward `+Z`).
fn turn(a: Vec3Fix, b: Vec3Fix) -> Fix128 {
    a.x * b.z - a.z * b.x
}

impl RingArc {
    /// How far the chord is inside the arc (the sagitta): every point of the
    /// arc's hull is within it of the arc.
    fn thickness(&self, major: Fix128) -> Fix128 {
        let half_chord = ((self.e1 - self.e0).length() * major).half();
        major - (major * major - half_chord * half_chord).sqrt()
    }

    fn split(&self) -> [Self; 2] {
        let mid = (self.e0 + self.e1).normalize();
        [
            Self {
                e0: self.e0,
                e1: mid,
            },
            Self {
                e0: mid,
                e1: self.e1,
            },
        ]
    }
}

/// An arc of a ring placed in the world, for GJK: its convex hull (the circular
/// segment).
struct ArcSupport {
    ring: Ring,
    arc: RingArc,
}

impl Support for ArcSupport {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        let local = self.ring.rotation.conjugate().rotate_vec(direction);
        let flat = Vec3Fix::new(local.x, Fix128::ZERO, local.z);
        let (e0, e1) = (self.arc.e0, self.arc.e1);
        let e = match flat.try_normalize() {
            Some(u) if !turn(e0, u).is_negative() && !turn(u, e1).is_negative() => u,
            _ if e1.dot(flat) > e0.dot(flat) => e1,
            _ => e0,
        };
        self.ring.center + self.ring.rotation.rotate_vec(e * self.ring.major)
    }
}

impl Ring {
    /// The first contact of the core `a`–`b` grown by `r` with the solid torus:
    /// [`toi_tree`] over arcs of the ring, each arc's hull grown by the tube
    /// radius (a convex set containing that part of the torus), split until the
    /// hull is within [`TRACE_TOLERANCE`] of the arc.
    fn sweep(
        &self,
        a: Vec3Fix,
        b: Vec3Fix,
        r: Fix128,
        d: Vec3Fix,
        max_t: Fix128,
    ) -> Option<Contact> {
        let x = Vec3Fix::UNIT_X;
        let z = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
        let quarters = [
            RingArc { e0: x, e1: z },
            RingArc { e0: z, e1: -x },
            RingArc { e0: -x, e1: -z },
            RingArc { e0: -z, e1: x },
        ];
        toi_tree(
            &quarters,
            |arc| {
                let hull = ArcSupport {
                    ring: *self,
                    arc: *arc,
                };
                toi_convex(a, b, r, self.minor, d, max_t, &hull)
            },
            |arc| arc.thickness(self.major) <= TRACE_TOLERANCE,
            |arc| arc.split().to_vec(),
        )
        .map(|c| self.refine(a + d * c.t, b + d * c.t, c))
    }

    /// A contact found on an arc's hull moved onto the torus: the ring point
    /// nearest the core point nearest the contact, grown by the tube radius
    /// (the hull contact is within the arc's sagitta of it).
    fn refine(&self, a: Vec3Fix, b: Vec3Fix, found: Contact) -> Contact {
        let core = closest_on_segment(a, b, found.point);
        let inv = self.rotation.conjugate();
        let local = inv.rotate_vec(core - self.center);
        let Some(dir) = Vec3Fix::new(local.x, Fix128::ZERO, local.z).try_normalize() else {
            return found;
        };
        let ring = self.center + self.rotation.rotate_vec(dir * self.major);
        match (core - ring).try_normalize() {
            Some(n) if n.dot(found.normal) > Fix128::ZERO => Contact {
                t: found.t,
                point: ring + n * self.minor,
                normal: n,
            },
            _ => found,
        }
    }
}

// ============================================================================
// Time of impact
// ============================================================================

/// The first time the core `a`–`b` grown by `r`, moving along the unit `d` for at
/// most `max_t`, touches a convex `solid` grown by `inflate`.
///
/// The gap `g(t) = dist(core + t·d, solid) − r − inflate` of two convex sets
/// moving apart linearly is convex in `t`, and `g′(t) = d·n` with `n` the unit
/// normal from the solid toward the core. Newton's step from below,
/// `t + g / (−g′)`, then never passes the first root (the tangent of a convex
/// function lies below it) and converges quadratically (linearly at a grazing,
/// double root); a gap that is not decreasing (`g′ ≥ 0`) never decreases again,
/// which proves there is no contact. A step that lands past the root (the
/// distance and normal are computed to within GJK's tolerance) is undone by
/// bisection between the last clear time and that time; once the two are within
/// the tolerance, the tangent's root from the clear time is the contact (a core
/// of reach `0` always ends this way, since it is touching only within GJK's
/// intersection tolerance).
///
/// Returns the contact once the gap is within [`TRACE_TOLERANCE`] and the core
/// moves into the solid (see [`touch_is_hit`]): a core touching the solid and
/// moving away from it or along it does not hit. A core that overlaps the solid at `t = 0` by
/// more than the tolerance hits at `t = 0` with normal `−d`. If the step budget
/// runs out (it does not in practice: Newton needs a few dozen steps even at a
/// grazing approach), the last clear time is reported, which is never after the
/// true contact.
fn toi_convex<S: Support>(
    a: Vec3Fix,
    b: Vec3Fix,
    r: Fix128,
    inflate: Fix128,
    d: Vec3Fix,
    max_t: Fix128,
    solid: &S,
) -> Option<Contact> {
    let reach = r + inflate;
    let mut t = Fix128::ZERO;
    // The last time known clear (gap above the tolerance), its near contact, and
    // the gap and closing speed there.
    let mut clear: Option<Contact> = None;
    let mut clear_gap = (Fix128::ZERO, Fix128::ZERO, false);
    // A time known to overlap, once a step has gone past the root.
    let mut deep: Option<Fix128> = None;
    for _ in 0..TRACE_MAX_STEPS {
        let off = d * t;
        let state = match convex_dist(a + off, b + off, solid) {
            Dist::Outside {
                dist,
                point,
                normal,
            } if dist - reach >= -TRACE_TOLERANCE => {
                let gap = dist - reach;
                let slope = d.dot(normal);
                let contact = Contact {
                    t,
                    point: point + normal * inflate,
                    normal,
                };
                if gap <= TRACE_TOLERANCE {
                    let gap_at = |t: Fix128| {
                        let off = d * t;
                        match convex_dist(a + off, b + off, solid) {
                            Dist::Outside { dist, .. } => Some(dist - reach),
                            _ => None,
                        }
                    };
                    return touch_is_hit(contact, d, max_t, gap_at).then_some(contact);
                }
                if !slope.is_negative() {
                    if slope < NEAR_TANGENT && gap <= GRAZE_GAP {
                        // Not a proof of separation: see `GRAZE_GAP`.
                        let probe = |t: Fix128| {
                            let off = d * t;
                            convex_dist(a + off, b + off, solid)
                        };
                        return graze_contact(probe, reach, inflate, t, max_t);
                    }
                    return None;
                }
                Some((gap, -slope, contact, dist < FINE_LENGTH))
            }
            _ => None,
        };
        match state {
            Some((gap, speed, contact, fine)) => {
                clear = Some(contact);
                clear_gap = (gap, speed, fine);
                let limit = deep.unwrap_or(max_t);
                // The tangent's root, unless it is at or past the limit.
                if gap >= (limit - t) * speed {
                    match deep {
                        None if gap > (limit - t) * speed => return None,
                        None => t = limit,
                        Some(hi) => t = (t + hi).half(),
                    }
                } else {
                    t = t + gap / speed;
                }
            }
            None => {
                if t.is_zero() {
                    return Some(Contact {
                        t,
                        point: half_vec(a + b),
                        normal: -d,
                    });
                }
                deep = Some(t);
                let lo = clear.map_or(Fix128::ZERO, |c| c.t);
                if t - lo <= TRACE_TOLERANCE {
                    // Bracketed to within the tolerance. When the clear end is
                    // nearer than 2⁻¹⁶ (a core of reach below that), report the
                    // tangent's root from it, which is not past the contact
                    // (convexity): a core of reach 0 never sees a touching gap
                    // (GJK counts `|v| ≤ 2⁻³²` as intersecting), so every such
                    // cast ends here; at `t` its gap is at most 2⁻³², not
                    // necessarily below 0, so the contact is before
                    // `t + 2⁻³² / speed`. Otherwise the clear end itself.
                    let (gap, speed, fine) = clear_gap;
                    if !fine {
                        return clear;
                    }
                    return clear.map(|c| {
                        let slack = ratio_within(TRACE_TOLERANCE, speed, max_t);
                        let step = ratio_within(gap, speed, t - lo + slack);
                        Contact {
                            t: min_fix(lo + max_fix(step, Fix128::ZERO), max_t),
                            ..c
                        }
                    });
                }
                t = (lo + t).half();
            }
        }
    }
    clear
}

/// A clear gap up to this (`2⁻²⁴`) with the motion within [`NEAR_TANGENT`] of
/// the tangent plane but not into it: the slope `d·n` then is not a proof that
/// the gap only grows. The normal is known to about the error of the distance
/// over the distance, and a path at `9·10⁻¹⁰` rad to a cone's base showed a
/// slope of `+3·10⁻⁹` where it was `−1.5·10⁻⁹`, at a gap of
/// `3·2⁻³²`, and went `1.7·2⁻³²` in further on: `2⁻²⁸` of slope error over a
/// path of 16 is `2⁻²⁴` of gap.
const GRAZE_GAP: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000_0100_0000_0000,
};

/// The contact of a path that is near tangent at `t0` (see [`GRAZE_GAP`]): the
/// gap (`dist` less `reach`, convex) is minimised on `[t0, max_t]` by
/// golden-section search, and if it goes below `−2⁻³²` the time it first comes
/// within `2⁻³²` is found by bisection between `t0` and that point (a clear
/// end, never after the contact, if the bisection does not land on it); `None`
/// if it does not go in.
fn graze_contact(
    dist: impl Fn(Fix128) -> Dist,
    reach: Fix128,
    inflate: Fix128,
    t0: Fix128,
    max_t: Fix128,
) -> Option<Contact> {
    let gap_at = |t: Fix128| match dist(t) {
        Dist::Outside { dist, .. } => Some(dist - reach),
        _ => None,
    };
    let deep = |g: Option<Fix128>| g.is_none_or(|g| g < -TRACE_TOLERANCE);
    let mut inside = deep(gap_at(max_t)).then_some(max_t);
    if inside.is_none() {
        let ratio = Fix128::from_ratio(618_033_988_749_895, 1_000_000_000_000_000);
        let (mut lo, mut hi) = (t0, max_t);
        for _ in 0..GOLDEN_STEPS {
            let x1 = hi - (hi - lo) * ratio;
            let x2 = lo + (hi - lo) * ratio;
            let (g1, g2) = (gap_at(x1), gap_at(x2));
            if deep(g1) {
                inside = Some(x1);
                break;
            }
            if deep(g2) {
                inside = Some(x2);
                break;
            }
            if g1 <= g2 {
                hi = x2;
            } else {
                lo = x1;
            }
        }
    }
    let (mut lo, mut hi) = (t0, inside?);
    for _ in 0..GOLDEN_STEPS {
        let mid = (lo + hi).half();
        match gap_at(mid) {
            Some(g) if g > TRACE_TOLERANCE => lo = mid,
            Some(g) if g >= -TRACE_TOLERANCE => {
                lo = mid;
                break;
            }
            _ => hi = mid,
        }
    }
    match dist(lo) {
        Dist::Outside { point, normal, .. } => Some(Contact {
            t: lo,
            point: point + normal * inflate,
            normal,
        }),
        _ => None,
    }
}

/// Whether a contact found touching a convex piece (gap at most `2⁻³²`, normal
/// `n` toward the core) is a hit for the motion along the unit `d`: moving away
/// or along the surface (`d·n ≥ 0`) is not; moving into it is, except that when
/// the motion is within [`NEAR_TANGENT`] of the tangent plane the contact is a
/// hit only if `gap_at` (the gap along the path, `None` inside) then drops
/// below `−2⁻³²` before `max_t`. A shape resting on a mesh floor or a flat
/// height field touches the next triangle or cell only tangentially at their
/// shared edge, and must not stop there; a grazing approach that really goes
/// in is still a hit.
fn touch_is_hit(
    contact: Contact,
    d: Vec3Fix,
    max_t: Fix128,
    gap_at: impl Fn(Fix128) -> Option<Fix128>,
) -> bool {
    let slope = d.dot(contact.normal);
    if slope < -NEAR_TANGENT {
        return true;
    }
    if !slope.is_negative() {
        return false;
    }
    match gap_at(contact.t) {
        None => true,
        Some(gap) => dips_below(gap_at, contact.t, gap, -slope, max_t),
    }
}

/// The least drop of the gap along the tangent over the first step of
/// [`dips_below`]: `2⁻²⁸`, 25 times the largest error measured for the distance
/// from a point to a cone or a cylinder near its surface (`0.62·2⁻³²` over
/// 40000 points `2⁻¹⁶` to `2⁻⁷` from the side or a cap, against the distance
/// to its meridian section in `f64`). With a first step of `2⁻³²`, a path
/// sinking at a slope of `0.009` dropped by `0.01·2⁻³²` a step, the rounding of
/// the distance (`0.2·2⁻³²`) made the gap grow, and a cast that went
/// `38152·2⁻³²` in was not a hit.
const DIPS_FIRST_DROP: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000_0010_0000_0000,
};

/// Whether a convex gap function (`None` inside) that is `f0` at `t0` and
/// decreasing at `speed` drops below `−2⁻³²` in `[t0, max_t]`: steps doubling
/// from the tangent's root until the gap goes below that (yes), grows again
/// (its minimum is then bracketed and found by golden-section search) or
/// `max_t` is reached.
fn dips_below(
    gap_at: impl Fn(Fix128) -> Option<Fix128>,
    t0: Fix128,
    f0: Fix128,
    speed: Fix128,
    max_t: Fix128,
) -> bool {
    let deep = |g: Option<Fix128>| g.is_none_or(|g| g < -TRACE_TOLERANCE);
    // The far end first: a path that sinks so slowly that a doubling step
    // changes the gap by less than its rounding would see no decrease and stop
    // the search before it reaches the depth it has by `max_t`.
    if deep(gap_at(max_t)) {
        return true;
    }
    let big = max_t + Fix128::ONE;
    // The first step goes at least as far as the gap drops by `DIPS_FIRST_DROP`
    // along the tangent: a step whose drop is below the error of the distance
    // can see the gap grow from rounding alone, and then brackets a minimum
    // that is not there.
    let mut step = max_fix(
        max_fix(
            ratio_within(max_fix(f0, Fix128::ZERO), speed, big),
            ratio_within(DIPS_FIRST_DROP, speed, big),
        ),
        TRACE_TOLERANCE,
    );
    let (mut before, mut last, mut last_gap) = (t0, t0, f0);
    for _ in 0..GOLDEN_STEPS {
        let t1 = min_fix(t0 + step, max_t);
        let g1 = gap_at(t1);
        if deep(g1) {
            return true;
        }
        let g1 = g1.unwrap_or(Fix128::ZERO);
        if g1 > last_gap || t1 >= max_t {
            // Convex: the minimum is in [before, t1].
            let (mut lo, mut hi) = (before, t1);
            let ratio = Fix128::from_ratio(618_033_988_749_895, 1_000_000_000_000_000);
            for _ in 0..GOLDEN_STEPS {
                let x1 = hi - (hi - lo) * ratio;
                let x2 = lo + (hi - lo) * ratio;
                let (g1, g2) = (gap_at(x1), gap_at(x2));
                if deep(g1) || deep(g2) {
                    return true;
                }
                if g1 <= g2 {
                    hi = x2;
                } else {
                    lo = x1;
                }
            }
            return false;
        }
        before = last;
        last = t1;
        last_gap = g1;
        step = step.double();
    }
    false
}

/// The first contact with a non-convex surface described as a hierarchy of parts
/// whose convex hulls contain them: `toi` is the contact with a part's hull,
/// `leaf` says a part's hull is within tolerance of the part, `split` divides a
/// part. Parts are taken best first (earliest hull contact); a hull contact is
/// never after the contact with the part it contains, so the first leaf taken is
/// the first contact of the whole surface to within the leaf tolerance. If the
/// node budget runs out, the earliest open hull contact is reported (never after
/// the true contact).
fn toi_tree<N: Copy>(
    roots: &[N],
    toi: impl Fn(&N) -> Option<Contact>,
    leaf: impl Fn(&N) -> bool,
    split: impl Fn(&N) -> Vec<N>,
) -> Option<Contact> {
    let mut open: Vec<(Contact, N)> = roots
        .iter()
        .filter_map(|n| toi(n).map(|c| (c, *n)))
        .collect();
    let mut nodes = 0usize;
    loop {
        let k = (0..open.len()).min_by_key(|&k| open[k].0.t)?;
        let (contact, node) = open.remove(k);
        if leaf(&node) {
            return Some(contact);
        }
        nodes += 1;
        if nodes > TREE_MAX_NODES {
            return Some(contact);
        }
        for child in split(&node) {
            if let Some(c) = toi(&child) {
                open.push((c, child));
            }
        }
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
            match Dist::toward(a, tri.closest_point(a)) {
                // On the triangle: touching, with its normal.
                Dist::Inside => Dist::Outside {
                    dist: Fix128::ZERO,
                    point: a,
                    normal: tri.normal().try_normalize().unwrap_or(Vec3Fix::UNIT_Y),
                },
                other => other,
            }
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
                Shape::Box { half_extents } => {
                    let rotation = body.rotation.unit_rotation();
                    Piece::Box {
                        center: body.position - rotation.rotate_vec(shape.center_of_mass_offset()),
                        half: half_extents,
                        rotation,
                    }
                }
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
    let body_rot = body_rot.unit_rotation();
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
    let rotation = posed.rotation.unit_rotation();
    (
        posed.position - rotation.rotate_vec(posed.shape.center_of_mass_offset()),
        rotation,
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
                    _ if is_round(posed) => convex_dist(a, b, &RoundSupport(posed)),
                    _ => convex_dist(a, b, posed),
                }
            }
            Self::Hull(child) => convex_dist(a, b, child),
            Self::Plane(plane) => plane_dist(plane, a, b),
            Self::Height(field) => heightfield_dist(field, a, b, cap),
            Self::Mesh(mesh) => mesh_dist(mesh, a, b, cap),
            #[cfg(feature = "std")]
            Self::Sdf(sdf) => segment_min(a, b, |p| point_sdf(sdf, p)),
        }
    }

    /// The first time the core `a`–`b` grown by `r` touches this piece moving
    /// along the unit `d`, when it does not overlap it at the start: closed forms
    /// for a point core against spheres, capsules, boxes, cylinders, tori,
    /// planes and triangles, the convex time of impact ([`toi_convex`]) for a
    /// segment core and the other convex pieces, a hierarchy of hulls for
    /// height-field cells and torus arcs ([`toi_tree`]), one time of impact per
    /// triangle for meshes, and sphere tracing for SDFs.
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
                let c = if point_core {
                    let h = ray_solid_sphere(a, d, *center, *radius + r, max_t)?;
                    Contact::from_ray(a, d, r, h)
                } else {
                    // The capsule moving +d meets the sphere when the sphere's
                    // centre, moving −d, meets the capsule grown by its radius.
                    let grown = Capsule::new(a, b, r + *radius);
                    let h = ray_solid_capsule(*center, -d, &grown, max_t)?;
                    let normal = -h.normal;
                    Contact {
                        t: h.t,
                        point: *center + normal * *radius,
                        normal,
                    }
                };
                self.entering(c, a, b, r, d, max_t)
            }
            Self::Capsule(c) => {
                if point_core {
                    let grown = Capsule::new(c.a, c.b, c.radius + r);
                    let h = ray_solid_capsule(a, d, &grown, max_t)?;
                    self.entering(Contact::from_ray(a, d, r, h), a, b, r, d, max_t)
                } else {
                    toi_convex(a, b, r, c.radius, d, max_t, &SegmentSupport(c.a, c.b))
                }
            }
            Self::Box {
                center,
                half,
                rotation,
            } => {
                if point_core {
                    let (o, dl) = to_local(a, d, *center, *rotation);
                    let h = to_world(sweep_local_box(o, dl, *half, r, max_t), *rotation)?;
                    self.entering(Contact::from_ray(a, d, r, h), a, b, r, d, max_t)
                } else {
                    let obb = crate::box_collider::OrientedBox::new(*center, *half, *rotation);
                    toi_convex(a, b, r, Fix128::ZERO, d, max_t, &obb)
                }
            }
            Self::Posed(posed) => {
                let (center, rotation) = posed_frame(posed);
                match posed.shape {
                    Shape::Cylinder {
                        radius,
                        half_height,
                    } if point_core => {
                        let (o, dl) = to_local(a, d, center, rotation);
                        let h = to_world(
                            Some(sweep_local_cylinder(o, dl, radius, half_height, r, max_t)?),
                            rotation,
                        )?;
                        self.entering(Contact::from_ray(a, d, r, h), a, b, r, d, max_t)
                    }
                    Shape::Torus {
                        major_radius,
                        minor_radius,
                    } => {
                        let ring = Ring {
                            center,
                            rotation,
                            major: major_radius,
                            minor: minor_radius,
                        };
                        if point_core {
                            let (o, dl) = to_local(a, d, center, rotation);
                            let h = ray_local_torus(o, dl, major_radius, minor_radius + r, max_t);
                            let c = Contact::from_ray(a, d, r, to_world(h, rotation)?);
                            // A torus is not convex: leaving it where the sphere
                            // touches it, the path can still enter it further on.
                            self.entering(c, a, b, r, d, max_t)
                                .or_else(|| ring.sweep(a, b, r, d, max_t))
                        } else {
                            ring.sweep(a, b, r, d, max_t)
                        }
                    }
                    _ if is_round(posed) => {
                        toi_convex(a, b, r, Fix128::ZERO, d, max_t, &RoundSupport(posed))
                    }
                    _ => toi_convex(a, b, r, Fix128::ZERO, d, max_t, posed),
                }
            }
            Self::Hull(child) => toi_convex(a, b, r, Fix128::ZERO, d, max_t, child),
            Self::Plane(plane) => sweep_plane(plane, a, b, r, d, max_t),
            Self::Height(field) => sweep_heightfield(field, a, b, r, d, max_t),
            Self::Mesh(mesh) => {
                if point_core {
                    let swept = core_box(a, a + d * max_t, r);
                    let mut best: Option<Contact> = None;
                    for i in mesh_candidates(mesh, &swept) {
                        let tri = &mesh.triangles[i as usize];
                        let Some(h) = sweep_triangle(a, d, tri, r, max_t) else {
                            continue;
                        };
                        let gap_at = |t: Fix128| {
                            let p = a + d * t;
                            match Dist::toward(p, tri.closest_point(p)) {
                                Dist::Outside { dist, .. } => Some(dist - r),
                                _ => None,
                            }
                        };
                        let mut c = Contact::from_ray(a, d, r, h);
                        if c.t <= TRACE_TOLERANCE {
                            // Touching this triangle at the start: its own
                            // nearest point and normal (see `entering`).
                            if let Dist::Outside { point, normal, .. } =
                                Dist::toward(a, tri.closest_point(a))
                            {
                                c = Contact {
                                    t: Fix128::ZERO,
                                    point,
                                    normal,
                                };
                            }
                        }
                        if !touch_is_hit(c, d, max_t, gap_at) {
                            continue;
                        }
                        if best.is_none_or(|bc| c.t < bc.t) {
                            best = Some(c);
                        }
                    }
                    best
                } else {
                    sweep_mesh(mesh, a, b, r, d, max_t)
                }
            }
            #[cfg(feature = "std")]
            Self::Sdf(_) => {
                let tol = Fix128::from_f32(settings.sdf.tolerance);
                self.trace(a, b, r, d, max_t, tol)
            }
        }
        .filter(|c| c.t >= Fix128::ZERO && c.t <= max_t)
    }

    /// A closed-form contact is a hit by [`touch_is_hit`]. One at `t ≤ 2⁻³²`
    /// comes from a cast that starts touching the piece (an overlap is reported
    /// before the sweep): its normal and point are the piece's nearest point and
    /// normal toward the core (a closed form may report a touching origin as
    /// inside). Moving away or along the surface is not a hit, nor is a tangent
    /// arrival.
    fn entering(
        &self,
        c: Contact,
        a: Vec3Fix,
        b: Vec3Fix,
        r: Fix128,
        d: Vec3Fix,
        max_t: Fix128,
    ) -> Option<Contact> {
        let cap = r + r + Fix128::ONE;
        let gap_at = |t: Fix128| {
            let off = d * t;
            match self.dist(a + off, b + off, cap) {
                Dist::Outside { dist, .. } => Some(dist - r),
                Dist::AtLeast(bound) => Some(bound - r),
                Dist::Inside => None,
            }
        };
        let c = if c.t > TRACE_TOLERANCE {
            c
        } else {
            match self.dist(a, b, cap) {
                Dist::Outside { point, normal, .. } => Contact {
                    t: Fix128::ZERO,
                    point,
                    normal,
                },
                _ => return Some(c),
            }
        };
        touch_is_hit(c, d, max_t, gap_at).then_some(c)
    }

    /// Sphere tracing (for SDFs): step by the gap (the distance less `r`) until
    /// it is below `tol`. A field that is a true distance never lets a step pass
    /// the surface; if the step budget runs out the last position is reported
    /// (never after the contact), not "no hit".
    #[cfg(feature = "std")]
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
        let mut last: Option<Contact> = None;
        for _ in 0..SDF_MAX_STEPS {
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
                    let contact = Contact { t, point, normal };
                    last = Some(contact);
                    if gap <= tol {
                        // Touching: a hit only when moving into the surface;
                        // moving away or along it, step past by the tolerance.
                        if d.dot(normal).is_negative() {
                            return Some(contact);
                        }
                        tol
                    } else {
                        gap
                    }
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
        last
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
            } => obb_meets_aabb(*center, *half, *rotation, aabb),
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

/// Whether an oriented box overlaps an axis-aligned one, by the separating axis
/// theorem (the 3 + 3 face axes and the 9 edge cross products): they overlap
/// when no axis separates them, and boxes that only touch (projections meeting
/// at a point) do not overlap. Cross products of parallel edges are zero and
/// separate nothing; they are skipped.
fn obb_meets_aabb(center: Vec3Fix, half: Vec3Fix, rotation: QuatFix, aabb: &AABB) -> bool {
    let z = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
    let a_axes = [
        rotation.rotate_vec(Vec3Fix::UNIT_X),
        rotation.rotate_vec(Vec3Fix::UNIT_Y),
        rotation.rotate_vec(z),
    ];
    let a_half = [half.x, half.y, half.z];
    let b_axes = [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, z];
    let b_center = half_vec(aabb.min + aabb.max);
    let b_ext = half_vec(aabb.max - aabb.min);
    let b_half = [b_ext.x, b_ext.y, b_ext.z];
    let between = b_center - center;
    let separates = |l: Vec3Fix| {
        let reach_a: Fix128 = (0..3).fold(Fix128::ZERO, |acc, i| {
            acc + a_half[i] * a_axes[i].dot(l).abs()
        });
        let reach_b: Fix128 = (0..3).fold(Fix128::ZERO, |acc, i| {
            acc + b_half[i] * b_axes[i].dot(l).abs()
        });
        between.dot(l).abs() >= reach_a + reach_b
    };
    let mut axes: Vec<Vec3Fix> = Vec::with_capacity(15);
    axes.extend_from_slice(&a_axes);
    axes.extend_from_slice(&b_axes);
    for ea in a_axes {
        for eb in b_axes {
            let l = ea.cross(eb);
            if !l.length_squared().is_zero() {
                axes.push(l);
            }
        }
    }
    !axes.into_iter().any(separates)
}

/// The distance from the core `a`–`b` to a two-sided plane.
fn plane_dist(plane: &PlaneCollider, a: Vec3Fix, b: Vec3Fix) -> Dist {
    let n = plane.normal;
    let sa = n.dot(a) - plane.offset;
    let sb = n.dot(b) - plane.offset;
    if !sa.is_zero() && !sb.is_zero() && sa.is_negative() != sb.is_negative() {
        // The segment crosses the plane.
        return Dist::Inside;
    }
    let (p, s) = if sb.abs() < sa.abs() {
        (b, sb)
    } else {
        (a, sa)
    };
    // An end on the plane is touching it, on the side of the other end.
    let other = if sb.abs() < sa.abs() { sa } else { sb };
    let side = if s.is_negative() || (s.is_zero() && other.is_negative()) {
        -n
    } else {
        n
    };
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
    if speed <= Fix128::ZERO {
        return None;
    }
    let gap = s.abs() - r;
    // A sink slower than 2⁻³² per unit of travel is a hit when the core is more
    // than 2⁻³² inside the plane by `max_t` (the rule of `touch_is_hit` for a
    // motion near the tangent plane); its time is bounded by `ratio_within`.
    if speed < PARALLEL_EPSILON && gap - speed * max_t >= -TRACE_TOLERANCE {
        return None;
    }
    // A start within the tolerance inside the plane's reach (deeper is a start
    // overlap, reported before the sweep) is touching: contact at once.
    let t = max_fix(ratio_within(gap, speed, max_t + Fix128::ONE), Fix128::ZERO);
    if t > max_t {
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

/// `direction` scaled down by its largest component when that is above `2²⁰`,
/// so that normalizing it does not overflow the squared length (`i64::MAX / 2`
/// squared wraps); smaller directions are returned as they are.
fn tame_direction(direction: Vec3Fix) -> Vec3Fix {
    let m = max_fix(
        max_fix(direction.x.abs(), direction.y.abs()),
        direction.z.abs(),
    );
    if m > Fix128::from_int(1 << 20) {
        direction / m
    } else {
        direction
    }
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
        let direction = tame_direction(direction);
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
        self.cast_core(a, b, radius, tame_direction(direction), max_t, filter)
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
                    Dist::Outside { dist, normal, .. } if dist + TRACE_TOLERANCE < r => {
                        out.push(Penetration {
                            target,
                            depth: r - dist,
                            normal,
                        })
                    }
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
    ///
    /// `direction` must have gone through `tame_direction` (both callers do):
    /// the `try_normalize` below squares it and wraps for components of `2³¹·⁵`
    /// or more.
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
                let c = if starts_overlapping(&piece.dist(a, b, start_cap), r) {
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

#[cfg(test)]
mod tests {
    //! Closed-form unit tests of the swept and overlap queries: every expected
    //! distance, normal and point is written from the geometry by hand (the
    //! closed form is in a comment next to each assertion); none calls the code
    //! under test. A sphere of radius `r` swept along a unit direction first
    //! touches a solid `S` where its centre reaches the boundary of
    //! `S ⊕ ball(r)`.
    //!
    //! `EXACT = 1e-12` for closed-form paths (spheres, boxes against a point
    //! core, planes, triangles, cylinders); `ITER = 1e-8` for iterative ones (GJK,
    //! convex time of impact, height-field parts, torus arcs).

    use super::*;
    use crate::box_collider::OrientedBox;
    use crate::collider::{ConvexHull, Sphere};
    use crate::compound::CompoundShape;
    use crate::solver::{PhysicsConfig, RigidBody};

    const EXACT: f64 = 1e-12;
    const ITER: f64 = 1e-8;

    fn fx(v: f64) -> Fix128 {
        Fix128::from_f64(v)
    }

    fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
        Vec3Fix::new(fx(x), fx(y), fx(z))
    }

    fn p3(p: [f64; 3]) -> Vec3Fix {
        v3(p[0], p[1], p[2])
    }

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsConfig::default())
    }

    fn all() -> RayFilter {
        RayFilter::default()
    }

    fn sphere_cast(
        w: &PhysicsWorld,
        c: [f64; 3],
        r: f64,
        d: [f64; 3],
        max: f64,
    ) -> Option<WorldShapeHit> {
        w.cast_sphere(p3(c), fx(r), p3(d), fx(max), &all())
    }

    fn capsule_cast(
        w: &PhysicsWorld,
        a: [f64; 3],
        b: [f64; 3],
        r: f64,
        d: [f64; 3],
        max: f64,
    ) -> Option<WorldShapeHit> {
        w.cast_capsule(p3(a), p3(b), fx(r), p3(d), fx(max), &all())
    }

    fn overlap_s(w: &PhysicsWorld, c: [f64; 3], r: f64) -> Vec<RayTarget> {
        w.overlap_sphere(p3(c), fx(r), &all())
    }

    fn overlap_b(w: &PhysicsWorld, lo: [f64; 3], hi: [f64; 3]) -> Vec<RayTarget> {
        w.overlap_aabb(&AABB::new(p3(lo), p3(hi)), &all())
    }

    fn unit(v: [f64; 3]) -> [f64; 3] {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    }

    #[track_caller]
    fn assert_vec(got: Vec3Fix, want: [f64; 3], tol: f64, what: &str) {
        let g = [got.x.to_f64(), got.y.to_f64(), got.z.to_f64()];
        for k in 0..3 {
            assert!(
                (g[k] - want[k]).abs() < tol,
                "{what} {g:?} but the closed form is {want:?}"
            );
        }
    }

    /// A hit on `target` at `t` with `normal` (normalized here) and `point`.
    #[track_caller]
    fn assert_hit(
        hit: Option<WorldShapeHit>,
        target: RayTarget,
        t: f64,
        normal: [f64; 3],
        point: [f64; 3],
        tol: f64,
    ) {
        let h = hit.unwrap_or_else(|| panic!("expected a hit at t = {t}, got none"));
        assert_eq!(h.target, target, "target");
        assert!(
            (h.t.to_f64() - t).abs() < tol,
            "t = {} but the closed form is {t}",
            h.t.to_f64()
        );
        assert_vec(h.normal, unit(normal), tol, "normal");
        assert_vec(h.point, point, tol, "point");
    }

    /// A capsule cast onto a flat face parallel to its segment: every point under
    /// the segment is a contact, so only `t`, the normal and the contact's height
    /// `y` are unique; its `x` lies between the ends' `x0..x1`.
    #[track_caller]
    fn assert_face_hit(
        hit: Option<WorldShapeHit>,
        target: RayTarget,
        t: f64,
        y: f64,
        x0: f64,
        x1: f64,
    ) {
        let h = hit.unwrap_or_else(|| panic!("expected a hit at t = {t}, got none"));
        assert_eq!(h.target, target, "target");
        assert!((h.t.to_f64() - t).abs() < ITER, "t = {}", h.t.to_f64());
        assert_vec(h.normal, [0.0, 1.0, 0.0], ITER, "normal");
        assert!(
            (h.point.y.to_f64() - y).abs() < ITER,
            "y = {}",
            h.point.y.to_f64()
        );
        let x = h.point.x.to_f64();
        assert!(x > x0 - ITER && x < x1 + ITER, "x = {x}");
    }

    fn shaped(w: &mut PhysicsWorld, shape: Shape, pos: Vec3Fix) -> usize {
        w.add_shaped_body(&shape, Fix128::ONE, pos)
            .expect("valid shape")
    }

    fn unit_box(w: &mut PhysicsWorld) -> usize {
        shaped(
            w,
            Shape::Box {
                half_extents: v3(1.0, 1.0, 1.0),
            },
            Vec3Fix::ZERO,
        )
    }

    fn sphere_body(w: &mut PhysicsWorld, pos: Vec3Fix, r: f64) -> usize {
        w.add_body_with_radius(RigidBody::new_static(pos), fx(r))
    }

    fn plane(w: &mut PhysicsWorld, normal: Vec3Fix, offset: f64) -> usize {
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            normal,
            fx(offset),
        )))
    }

    /// The two triangles of the floor square `[0, 4]²` at `y = 0`.
    fn floor_mesh(w: &mut PhysicsWorld) -> usize {
        let verts = [
            v3(0.0, 0.0, 0.0),
            v3(4.0, 0.0, 0.0),
            v3(4.0, 0.0, 4.0),
            v3(0.0, 0.0, 4.0),
        ];
        w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
            &verts,
            &[0, 1, 2, 0, 2, 3],
        )))
    }

    /// One bilinear cell over `[0, 1]²` with corner heights `0, 0, 0, 1`: the
    /// twisted surface `y = x·z`.
    fn saddle_field(w: &mut PhysicsWorld) -> usize {
        let heights = vec![Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ONE];
        w.add_static_collider(StaticCollider::HeightField(HeightField::new(
            heights,
            2,
            2,
            Fix128::ONE,
            Vec3Fix::ZERO,
        )))
    }

    // ------------------------------------------------------------ sphere cast

    #[test]
    fn sphere_cast_against_a_sphere_body() {
        let mut w = world();
        let b = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        // oracle: head-on, the centre reaches |c| = R + r = 1.5: t = 10 − 1.5;
        // the direction (2, 0, 0) is normalized first.
        assert_hit(
            sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [2.0, 0.0, 0.0], 100.0),
            RayTarget::Body(b),
            8.5,
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            EXACT,
        );
        // oracle: on the line y = 0.6, x = −√(1.5² − 0.36) = −√1.89 at contact,
        // t = 10 − √1.89, normal c/1.5, contact R·normal.
        let s = 1.89f64.sqrt();
        let n = [-s / 1.5, 0.6 / 1.5, 0.0];
        assert_hit(
            sphere_cast(&w, [-10.0, 0.6, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
            RayTarget::Body(b),
            10.0 - s,
            n,
            n,
            EXACT,
        );
        // oracle: the line y = 1.6 passes 1.6 > 1.5 from the centre: no contact.
        assert_eq!(
            sphere_cast(&w, [-10.0, 1.6, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
            None
        );
        // oracle: the contact at t = 8.5 is beyond max_t = 5.
        assert_eq!(
            sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 5.0),
            None
        );
        let h = sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0).expect("hit");
        assert_eq!(h.body, Some(b));
    }

    #[test]
    fn degenerate_casts_give_no_hit() {
        let mut w = world();
        sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        let c = [-10.0, 0.0, 0.0];
        // A negative radius, a zero direction and max_t ≤ 0 give no hit.
        assert_eq!(sphere_cast(&w, c, -0.5, [1.0, 0.0, 0.0], 100.0), None);
        assert_eq!(sphere_cast(&w, c, 0.5, [0.0, 0.0, 0.0], 100.0), None);
        assert_eq!(sphere_cast(&w, c, 0.5, [1.0, 0.0, 0.0], 0.0), None);
        assert_eq!(sphere_cast(&w, c, 0.5, [1.0, 0.0, 0.0], -1.0), None);
        assert_eq!(
            capsule_cast(&w, c, [-10.0, 1.0, 0.0], -0.5, [1.0, 0.0, 0.0], 100.0),
            None
        );
        assert_eq!(
            capsule_cast(&w, c, [-10.0, 1.0, 0.0], 0.5, [1.0, 0.0, 0.0], 0.0),
            None
        );
    }

    #[test]
    fn radius_zero_is_a_ray_and_a_huge_direction_is_normalized() {
        let mut w = world();
        let b = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        // oracle: a ray from x = −10 meets the unit sphere at x = −1: t = 9.
        assert_hit(
            sphere_cast(&w, [-10.0, 0.0, 0.0], 0.0, [1.0, 0.0, 0.0], 100.0),
            RayTarget::Body(b),
            9.0,
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            EXACT,
        );
        // oracle: the direction (1e7, 0, 0) is the unit +X: t = 8.5.
        assert_hit(
            sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1e7, 0.0, 0.0], 100.0),
            RayTarget::Body(b),
            8.5,
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            EXACT,
        );
    }

    #[test]
    fn a_cast_that_starts_overlapping_hits_at_zero() {
        let mut w = world();
        let b = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        // oracle: the centre 0.5 from a body of radius 1: overlapping, t = 0,
        // normal −direction, point the cast centre.
        assert_hit(
            sphere_cast(&w, [0.5, 0.0, 0.0], 0.5, [0.0, 1.0, 0.0], 100.0),
            RayTarget::Body(b),
            0.0,
            [0.0, -1.0, 0.0],
            [0.5, 0.0, 0.0],
            EXACT,
        );
        // oracle: a capsule with an end inside: t = 0, point the segment midpoint.
        assert_hit(
            capsule_cast(
                &w,
                [0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                0.5,
                [1.0, 0.0, 0.0],
                100.0,
            ),
            RayTarget::Body(b),
            0.0,
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            EXACT,
        );
    }

    #[test]
    fn a_cast_that_starts_touching_hits_only_when_it_moves_in() {
        let mut w = world();
        let b = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        // The cast sphere of radius 0.5 at (−1.5, 0, 0) touches the body.
        let c = [-1.5, 0.0, 0.0];
        // oracle: moving in (+X): t = 0, normal −X, contact (−1, 0, 0).
        assert_hit(
            sphere_cast(&w, c, 0.5, [1.0, 0.0, 0.0], 100.0),
            RayTarget::Body(b),
            0.0,
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            EXACT,
        );
        // oracle: moving away (−X) or along the tangent plane (+Y): no hit.
        assert_eq!(sphere_cast(&w, c, 0.5, [-1.0, 0.0, 0.0], 100.0), None);
        assert_eq!(sphere_cast(&w, c, 0.5, [0.0, 1.0, 0.0], 100.0), None);

        // A sphere resting on a plane: down is a hit at 0, sideways and up not.
        let mut w = world();
        let p = plane(&mut w, Vec3Fix::UNIT_Y, 0.0);
        let c = [1.0, 0.5, 2.0];
        assert_hit(
            sphere_cast(&w, c, 0.5, [0.0, -1.0, 0.0], 100.0),
            RayTarget::StaticCollider(p),
            0.0,
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 2.0],
            EXACT,
        );
        assert_eq!(sphere_cast(&w, c, 0.5, [1.0, 0.0, 0.0], 100.0), None);
        assert_eq!(sphere_cast(&w, c, 0.5, [0.0, 1.0, 0.0], 100.0), None);

        // A sphere resting on the unit box top: down hits at 0, sideways not.
        let mut w = world();
        let bx = unit_box(&mut w);
        let c = [0.2, 1.5, 0.3];
        assert_hit(
            sphere_cast(&w, c, 0.5, [0.0, -1.0, 0.0], 100.0),
            RayTarget::Body(bx),
            0.0,
            [0.0, 1.0, 0.0],
            [0.2, 1.0, 0.3],
            EXACT,
        );
        assert_eq!(sphere_cast(&w, c, 0.5, [1.0, 0.0, 0.0], 100.0), None);
    }

    #[test]
    fn sphere_cast_against_box_face_edge_and_turned_box() {
        let mut w = world();
        let b = unit_box(&mut w);
        // oracle: face x = −1 pushed out to x = −1.5: t = 8.5, contact on the face.
        assert_hit(
            sphere_cast(&w, [-10.0, 0.2, 0.3], 0.5, [1.0, 0.0, 0.0], 100.0),
            RayTarget::Body(b),
            8.5,
            [-1.0, 0.0, 0.0],
            [-1.0, 0.2, 0.3],
            EXACT,
        );
        // oracle: edge (x, y) = (−1, 1): the centre (x, 1.3) is 0.5 from it at
        // x = −1 − √(0.25 − 0.09) = −1.4, t = 8.6, normal (−0.4, 0.3)/0.5.
        assert_hit(
            sphere_cast(&w, [-10.0, 1.3, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
            RayTarget::Body(b),
            8.6,
            [-0.8, 0.6, 0.0],
            [-1.0, 1.0, 0.0],
            EXACT,
        );
        // The same box turned 45° about Y: its vertical edge points at −X, at
        // x = −√2. oracle: t = 10 − √2 − 0.5, normal −X, contact (−√2, 0, 0).
        let mut w = world();
        let b = unit_box(&mut w);
        w.bodies[b].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::HALF_PI.half());
        let s2 = 2f64.sqrt();
        assert_hit(
            sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
            RayTarget::Body(b),
            9.5 - s2,
            [-1.0, 0.0, 0.0],
            [-s2, 0.0, 0.0],
            ITER,
        );
    }

    #[test]
    fn sphere_cast_against_cylinder_side_cap_and_rim() {
        let mut w = world();
        let b = shaped(
            &mut w,
            Shape::Cylinder {
                radius: Fix128::ONE,
                half_height: Fix128::ONE,
            },
            Vec3Fix::ZERO,
        );
        // oracle: side x = −1 pushed out to −1.5: t = 8.5.
        assert_hit(
            sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
            RayTarget::Body(b),
            8.5,
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            EXACT,
        );
        // oracle: top cap y = 1 pushed up to 1.5: t = 8.5.
        assert_hit(
            sphere_cast(&w, [0.2, 10.0, 0.0], 0.5, [0.0, -1.0, 0.0], 100.0),
            RayTarget::Body(b),
            8.5,
            [0.0, 1.0, 0.0],
            [0.2, 1.0, 0.0],
            EXACT,
        );
        // oracle: rim (1, 1, 0): the centre (1.3, y, 0) is 0.5 from it at
        // y = 1 + √(0.25 − 0.09) = 1.4, t = 8.6, normal (0.3, 0.4)/0.5.
        assert_hit(
            sphere_cast(&w, [1.3, 10.0, 0.0], 0.5, [0.0, -1.0, 0.0], 100.0),
            RayTarget::Body(b),
            8.6,
            [0.6, 0.8, 0.0],
            [1.0, 1.0, 0.0],
            EXACT,
        );
        // oracle: a horizontal capsule over the cap: t = 5 − 1 − 0.5.
        assert_face_hit(
            capsule_cast(
                &w,
                [-0.5, 5.0, 0.0],
                [0.5, 5.0, 0.0],
                0.5,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            RayTarget::Body(b),
            3.5,
            1.0,
            -0.5,
            0.5,
        );
        // Overlaps: 0.2 above the cap is within 0.3, not within 0.1.
        assert_eq!(
            overlap_s(&w, [0.0, 1.2, 0.0], 0.3),
            vec![RayTarget::Body(b)]
        );
        assert!(overlap_s(&w, [0.0, 1.2, 0.0], 0.1).is_empty());
        // Radius 0: a point strictly inside overlaps, one on the surface does not.
        assert_eq!(
            overlap_s(&w, [0.0, 0.0, 0.0], 0.0),
            vec![RayTarget::Body(b)]
        );
        assert!(overlap_s(&w, [0.0, 1.0, 0.0], 0.0).is_empty());
    }

    #[test]
    fn sphere_and_capsule_casts_against_an_ellipsoid() {
        let mut w = world();
        let b = shaped(
            &mut w,
            Shape::Ellipsoid {
                radii: v3(1.0, 2.0, 1.0),
            },
            Vec3Fix::ZERO,
        );
        // oracle: the top (0, 2, 0) reached at y = 2.5: t = 10 − 2.5.
        assert_hit(
            sphere_cast(&w, [0.0, 10.0, 0.0], 0.5, [0.0, -1.0, 0.0], 100.0),
            RayTarget::Body(b),
            7.5,
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            ITER,
        );
        // oracle: the side (1, 0, 0) reached by a vertical capsule at x = 1.5.
        assert_hit(
            capsule_cast(
                &w,
                [-10.0, -0.5, 0.0],
                [-10.0, 0.5, 0.0],
                0.5,
                [1.0, 0.0, 0.0],
                100.0,
            ),
            RayTarget::Body(b),
            8.5,
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            ITER,
        );
        // oracle: the line x = 1.6 is 0.6 > 0.5 from the widest point: no contact.
        assert_eq!(
            sphere_cast(&w, [1.6, 10.0, 0.0], 0.5, [0.0, -1.0, 0.0], 100.0),
            None
        );
        // Overlaps: the side point (1, 0, 0) is 0.2 from (1.2, 0, 0).
        assert_eq!(
            overlap_s(&w, [1.2, 0.0, 0.0], 0.3),
            vec![RayTarget::Body(b)]
        );
        assert!(overlap_s(&w, [1.2, 0.0, 0.0], 0.1).is_empty());
        assert_eq!(
            overlap_b(&w, [0.9, -0.1, -0.1], [1.5, 0.1, 0.1]),
            vec![RayTarget::Body(b)]
        );
        assert!(overlap_b(&w, [1.1, -0.1, -0.1], [1.5, 0.1, 0.1]).is_empty());
    }

    #[test]
    fn casts_and_overlaps_against_a_torus() {
        let mut w = world();
        // Ring of radius 2 in XZ, tube 0.5.
        let b = shaped(
            &mut w,
            Shape::Torus {
                major_radius: fx(2.0),
                minor_radius: fx(0.5),
            },
            Vec3Fix::ZERO,
        );
        // oracle: the top of the tube (2, 0.5, 0) reached at y = 0.75: t = 9.25.
        assert_hit(
            sphere_cast(&w, [2.0, 10.0, 0.0], 0.25, [0.0, -1.0, 0.0], 100.0),
            RayTarget::Body(b),
            9.25,
            [0.0, 1.0, 0.0],
            [2.0, 0.5, 0.0],
            EXACT,
        );
        // oracle: down the hole, 1.5 > 0.25 from the tube: no contact.
        assert_eq!(
            sphere_cast(&w, [0.0, 10.0, 0.0], 0.25, [0.0, -1.0, 0.0], 100.0),
            None
        );
        // oracle: a capsule across the tube, its point (2, y, 0) nearest the ring:
        // y = 0.5 + 0.25, t = 4.25.
        assert_hit(
            capsule_cast(
                &w,
                [1.5, 5.0, 0.0],
                [2.5, 5.0, 0.0],
                0.25,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            RayTarget::Body(b),
            4.25,
            [0.0, 1.0, 0.0],
            [2.0, 0.5, 0.0],
            ITER,
        );
        // Overlaps: the axis is 1.5 from the tube.
        assert!(overlap_s(&w, [0.0, 0.0, 0.0], 1.4).is_empty());
        assert_eq!(
            overlap_s(&w, [0.0, 0.0, 0.0], 1.6),
            vec![RayTarget::Body(b)]
        );
        assert_eq!(
            overlap_s(&w, [2.0, 0.0, 0.0], 0.1),
            vec![RayTarget::Body(b)]
        );
        // oracle: the ring at 45° (√2, 0, √2) is (√2 − c)·√2 from the box corner
        // (c, ·, c): c = 1 gives 2 − √2 = 0.59 > 0.5 (no overlap), c = 1.2 gives
        // 0.30 < 0.5 (overlap).
        assert!(overlap_b(&w, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]).is_empty());
        assert_eq!(
            overlap_b(&w, [-1.2, -1.0, -1.2], [1.2, 1.0, 1.2]),
            vec![RayTarget::Body(b)]
        );
        // oracle: a box far outside the bounding sphere (radius 2.5).
        assert!(overlap_b(&w, [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]).is_empty());
    }

    #[test]
    fn casts_and_overlaps_against_compound_children() {
        let mut w = world();
        // Box children at x = ±3, sphere children at z = ±3, a capsule child
        // along X and cube hull children at y = ±4: symmetric, centre of mass at
        // the origin.
        let mut c = CompoundShape::new();
        for s in [3.0, -3.0] {
            c.add_box(
                OrientedBox::new(Vec3Fix::ZERO, v3(0.5, 0.5, 0.5), QuatFix::IDENTITY),
                v3(s, 0.0, 0.0),
                QuatFix::IDENTITY,
            );
            c.add_sphere(
                Sphere::new(Vec3Fix::ZERO, fx(0.5)),
                v3(0.0, 0.0, s),
                QuatFix::IDENTITY,
            );
            let mut cube = Vec::new();
            for x in [-0.5, 0.5] {
                for y in [-0.5, 0.5] {
                    for z in [-0.5, 0.5] {
                        cube.push(v3(x, y, z));
                    }
                }
            }
            c.add_convex_hull(
                ConvexHull::new(cube),
                v3(0.0, s + s / 3.0, 0.0),
                QuatFix::IDENTITY,
            );
        }
        c.add_capsule(
            Capsule::new(v3(-1.5, 0.0, 0.0), v3(1.5, 0.0, 0.0), fx(0.5)),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        let b = w
            .add_compound_body(&c, Fix128::ONE, Vec3Fix::ZERO)
            .expect("valid compound");
        let body = RayTarget::Body(b);
        // oracle: box child top y = 0.5 reached at 0.75: t = 9.25.
        assert_hit(
            sphere_cast(&w, [3.0, 10.0, 0.0], 0.25, [0.0, -1.0, 0.0], 100.0),
            body,
            9.25,
            [0.0, 1.0, 0.0],
            [3.0, 0.5, 0.0],
            EXACT,
        );
        // oracle: sphere child at (0, 0, 3): radius 0.75 above it, t = 9.25.
        assert_hit(
            sphere_cast(&w, [0.0, 10.0, 3.0], 0.25, [0.0, -1.0, 0.0], 100.0),
            body,
            9.25,
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 3.0],
            EXACT,
        );
        // oracle: capsule child side at x = 1, from +Z: 0.75 from the segment,
        // t = 10 − 0.75.
        assert_hit(
            sphere_cast(&w, [1.0, 0.0, 10.0], 0.25, [0.0, 0.0, -1.0], 100.0),
            body,
            9.25,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.5],
            EXACT,
        );
        // oracle: hull cube at y = 4 (top 4.5) from the side: its face x = 0.5
        // reached at x = 0.75, t = 10 − 0.75.
        assert_hit(
            sphere_cast(&w, [10.0, 4.0, 0.0], 0.25, [-1.0, 0.0, 0.0], 100.0),
            body,
            9.25,
            [1.0, 0.0, 0.0],
            [0.5, 4.0, 0.0],
            ITER,
        );
        // oracle: a capsule over the box child at x = 3: t = 5 − 0.5 − 0.25.
        assert_face_hit(
            capsule_cast(
                &w,
                [2.8, 5.0, 0.0],
                [3.2, 5.0, 0.0],
                0.25,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            body,
            4.25,
            0.5,
            2.8,
            3.2,
        );
        // oracle: a capsule crossing over the capsule child (below the hull cube
        // at y = 4): t = 2 − 0.75.
        assert_hit(
            capsule_cast(
                &w,
                [0.0, 2.0, -1.0],
                [0.0, 2.0, 1.0],
                0.25,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            body,
            1.25,
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            ITER,
        );
        // oracle: a capsule over the sphere child at (0, 0, −3): t = 5 − 0.75.
        assert_hit(
            capsule_cast(
                &w,
                [-1.0, 5.0, -3.0],
                [1.0, 5.0, -3.0],
                0.25,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            body,
            4.25,
            [0.0, 1.0, 0.0],
            [0.0, 0.5, -3.0],
            EXACT,
        );
        // oracle: (2.1, 0, 0) is 0.1 from the capsule child (its end cap reaches
        // x = 2) and 0.4 from the box child (face x = 2.5): not within 0.05,
        // within 0.45.
        assert!(overlap_s(&w, [2.1, 0.0, 0.0], 0.05).is_empty());
        assert_eq!(overlap_s(&w, [2.1, 0.0, 0.0], 0.45), vec![body]);
        // Box overlaps: each child alone, and the empty space between them.
        assert_eq!(overlap_b(&w, [3.4, 0.4, -0.1], [3.6, 0.6, 0.1]), vec![body]);
        assert_eq!(overlap_b(&w, [-0.1, 0.4, 2.9], [0.1, 0.6, 3.1]), vec![body]);
        assert_eq!(overlap_b(&w, [1.0, 0.4, -0.1], [1.2, 0.6, 0.1]), vec![body]);
        assert_eq!(overlap_b(&w, [0.4, 4.4, -0.1], [0.6, 4.6, 0.1]), vec![body]);
        assert!(overlap_b(&w, [1.0, 2.0, 1.0], [2.0, 3.0, 2.0]).is_empty());
    }

    #[test]
    fn casts_and_overlaps_against_a_plane() {
        let mut w = world();
        let p = plane(&mut w, Vec3Fix::UNIT_Y, 0.0);
        let sp = RayTarget::StaticCollider(p);
        // oracle: oblique (1, −1, 0)/√2: the height drops 4.5 after 4.5·√2.
        assert_hit(
            sphere_cast(&w, [0.0, 5.0, 0.0], 0.5, [1.0, -1.0, 0.0], 100.0),
            sp,
            4.5 * 2f64.sqrt(),
            [0.0, 1.0, 0.0],
            [4.5, 0.0, 0.0],
            EXACT,
        );
        // oracle: two-sided: from below the centre reaches y = −0.5.
        assert_hit(
            sphere_cast(&w, [0.0, -5.0, 0.0], 0.5, [0.0, 1.0, 0.0], 100.0),
            sp,
            4.5,
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            EXACT,
        );
        // oracle: parallel 1 above: never closer than 1 > 0.5.
        assert_eq!(
            sphere_cast(&w, [0.0, 1.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
            None
        );
        // oracle: capsule, the lower end (−1, 5, 0) reaches y = 0.5: t = 4.5.
        assert_hit(
            capsule_cast(
                &w,
                [-1.0, 5.0, 0.0],
                [1.0, 6.0, 0.0],
                0.5,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            sp,
            4.5,
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            EXACT,
        );
        // oracle: a capsule whose segment crosses the plane starts overlapping.
        assert_hit(
            capsule_cast(
                &w,
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                0.5,
                [1.0, 0.0, 0.0],
                100.0,
            ),
            sp,
            0.0,
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            EXACT,
        );
        // Overlaps: 0.4 above is within 0.5; 0.6 is not.
        assert_eq!(overlap_s(&w, [0.0, 0.4, 0.0], 0.5), vec![sp]);
        assert!(overlap_s(&w, [0.0, 0.6, 0.0], 0.5).is_empty());
        // oracle: a box straddling y = 0 overlaps, one resting on it does not.
        assert_eq!(overlap_b(&w, [0.0, -0.1, 0.0], [1.0, 0.1, 1.0]), vec![sp]);
        assert!(overlap_b(&w, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).is_empty());
    }

    #[test]
    fn casts_along_a_mesh_floor_cross_its_edges() {
        let mut w = world();
        let m = floor_mesh(&mut w);
        let sm = RayTarget::StaticCollider(m);
        // oracle: face region: the plane y = 0 reached at y = 0.5: t = 2.5.
        assert_hit(
            sphere_cast(&w, [1.0, 3.0, 3.0], 0.5, [0.0, -1.0, 0.0], 100.0),
            sm,
            2.5,
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 3.0],
            EXACT,
        );
        // oracle: edge x = 0: the centre (x, 0.3, 1) is 0.5 from (0, 0, 1) at
        // x = −0.4, t = 9.6, normal (−0.8, 0.6, 0).
        assert_hit(
            sphere_cast(&w, [-10.0, 0.3, 1.0], 0.5, [1.0, 0.0, 0.0], 100.0),
            sm,
            9.6,
            [-0.8, 0.6, 0.0],
            [0.0, 0.0, 1.0],
            EXACT,
        );
        // oracle: a sphere resting on the floor sliding across the diagonal
        // edge, and a capsule doing the same: tangent, no hit.
        assert_eq!(
            sphere_cast(&w, [1.0, 0.5, 3.0], 0.5, [1.0, 0.0, -1.0], 2.0),
            None
        );
        assert_eq!(
            capsule_cast(
                &w,
                [1.0, 0.5, 2.0],
                [1.0, 1.5, 2.0],
                0.5,
                [1.0, 0.0, 0.0],
                2.0
            ),
            None
        );
        // oracle: resting and moving down: t = 0 on the floor.
        assert_hit(
            sphere_cast(&w, [1.0, 0.5, 3.0], 0.5, [0.0, -1.0, 0.0], 100.0),
            sm,
            0.0,
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 3.0],
            EXACT,
        );
        // oracle: tilted capsule, lower end (2, 3, 1) reaches y = 0.5: t = 2.5.
        assert_hit(
            capsule_cast(
                &w,
                [2.0, 3.0, 1.0],
                [3.0, 4.0, 1.0],
                0.5,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            sm,
            2.5,
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            ITER,
        );
        // Overlaps: 0.3 above the floor is within 0.5, outside it 0.6 is not.
        assert_eq!(overlap_s(&w, [2.0, 0.3, 2.0], 0.5), vec![sm]);
        assert!(overlap_s(&w, [-0.6, 0.0, 2.0], 0.5).is_empty());
        assert_eq!(overlap_b(&w, [1.0, -0.1, 1.0], [2.0, 0.1, 2.0]), vec![sm]);
        assert!(overlap_b(&w, [1.0, 0.1, 1.0], [2.0, 1.0, 2.0]).is_empty());
    }

    #[test]
    fn casts_and_overlaps_against_height_fields() {
        // oracle: flat field at 0.25 over [0, 4]²: t = 5 − 0.25 − 0.5.
        let mut w = world();
        let h = w.add_static_collider(StaticCollider::HeightField(HeightField::flat(
            5,
            5,
            Fix128::ONE,
            Vec3Fix::ZERO,
            fx(0.25),
        )));
        let sh = RayTarget::StaticCollider(h);
        assert_hit(
            sphere_cast(&w, [2.0, 5.0, 2.0], 0.5, [0.0, -1.0, 0.0], 100.0),
            sh,
            4.25,
            [0.0, 1.0, 0.0],
            [2.0, 0.25, 2.0],
            ITER,
        );
        // oracle: a horizontal capsule over it: the same t.
        assert_face_hit(
            capsule_cast(
                &w,
                [1.5, 5.0, 2.0],
                [2.5, 5.0, 2.0],
                0.5,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            sh,
            4.25,
            0.25,
            1.5,
            2.5,
        );
        // oracle: resting on it and sliding across cells: no hit.
        assert_eq!(
            sphere_cast(&w, [0.5, 0.75, 0.5], 0.5, [1.0, 0.0, 0.0], 3.0),
            None
        );
        // oracle: outside the field's footprint: no hit.
        assert_eq!(
            sphere_cast(&w, [10.0, 5.0, 10.0], 0.5, [0.0, -1.0, 0.0], 100.0),
            None
        );
        assert_eq!(overlap_s(&w, [2.0, 0.5, 2.0], 0.3), vec![sh]);
        assert!(overlap_s(&w, [2.0, 0.6, 2.0], 0.3).is_empty());
        assert_eq!(overlap_b(&w, [1.0, 0.2, 1.0], [3.0, 0.3, 3.0]), vec![sh]);
        assert!(overlap_b(&w, [1.0, 0.3, 1.0], [3.0, 1.0, 3.0]).is_empty());
        assert!(overlap_b(&w, [5.0, 0.0, 5.0], [6.0, 1.0, 6.0]).is_empty());
    }

    #[test]
    fn casts_and_overlaps_against_a_twisted_height_field_cell() {
        let mut w = world();
        let h = saddle_field(&mut w);
        let sh = RayTarget::StaticCollider(h);
        // The surface y = x·z at (0.5, 0.25, 0.5) has the unit normal
        // n = (−0.5, 1, −0.5)/√1.5. A sphere of radius 0.1 (its curvature 10
        // beats the surface's ±1, so the contact is unique) whose centre moves
        // down the line through (0.5, 0.25, 0.5) + 0.1·n touches there.
        let k = 1.5f64.sqrt();
        let n = [-0.5 / k, 1.0 / k, -0.5 / k];
        let r = 0.1;
        let cx = 0.5 + r * n[0];
        // oracle: t = 5 − (0.25 + r·n.y).
        assert_hit(
            sphere_cast(&w, [cx, 5.0, cx], r, [0.0, -1.0, 0.0], 100.0),
            sh,
            5.0 - (0.25 + r * n[1]),
            n,
            [0.5, 0.25, 0.5],
            ITER,
        );
        // oracle: a point 0.09 along n from the surface point is within 0.1 of
        // the surface, one 0.11 along n is not.
        let at = |s: f64| [0.5 + s * n[0], 0.25 + s * n[1], 0.5 + s * n[2]];
        assert_eq!(overlap_s(&w, at(0.09), 0.1), vec![sh]);
        assert!(overlap_s(&w, at(0.11), 0.1).is_empty());
        // oracle: the surface spans y ∈ [0, 1]: a box at y ∈ [0.9, 2] over the
        // corner (1, 1) meets it (height 1 there), one over (0, 0) does not.
        assert_eq!(overlap_b(&w, [0.9, 0.9, 0.9], [1.0, 2.0, 1.0]), vec![sh]);
        assert!(overlap_b(&w, [0.0, 0.9, 0.0], [0.1, 2.0, 0.1]).is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn casts_and_overlaps_against_an_sdf() {
        use crate::sdf_collider::{ClosureSdf, SdfCollider};
        let mut w = world();
        // The unit-sphere field about the origin.
        w.sdf_colliders.push(SdfCollider::new_static(
            Box::new(ClosureSdf::new(
                |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
                |x, y, z| {
                    let l = (x * x + y * y + z * z).sqrt();
                    (x / l, y / l, z / l)
                },
            )),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        ));
        let tol = f64::from(all().sdf.tolerance);
        // oracle: the centre reaches |c| = 1.5: t ∈ [3.5 − tol, 3.5].
        let h = sphere_cast(&w, [-5.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0).expect("hit");
        assert_eq!(h.target, RayTarget::Sdf(0));
        assert!(h.t.to_f64() > 3.5 - tol - 1e-5 && h.t.to_f64() < 3.5 + 1e-5);
        assert_vec(h.normal, [-1.0, 0.0, 0.0], 1e-3, "normal");
        // oracle: a vertical capsule at the same height: the same t.
        let h = capsule_cast(
            &w,
            [-5.0, -0.5, 0.0],
            [-5.0, 0.5, 0.0],
            0.5,
            [1.0, 0.0, 0.0],
            100.0,
        )
        .expect("hit");
        assert!(h.t.to_f64() > 3.5 - tol - 1e-4 && h.t.to_f64() < 3.5 + 1e-4);
        // oracle: the line y = 2 stays 2 − 1 = 1 > 0.5 from the surface.
        assert_eq!(
            sphere_cast(&w, [-5.0, 2.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
            None
        );
        // Overlaps: (1.2, 0, 0) is 0.2 from the surface.
        assert_eq!(overlap_s(&w, [1.2, 0.0, 0.0], 0.3), vec![RayTarget::Sdf(0)]);
        assert!(overlap_s(&w, [1.2, 0.0, 0.0], 0.1).is_empty());
        assert_eq!(
            overlap_b(&w, [0.8, -0.1, -0.1], [1.2, 0.1, 0.1]),
            vec![RayTarget::Sdf(0)]
        );
        assert!(overlap_b(&w, [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]).is_empty());
        // The filter can hide it.
        assert!(w
            .overlap_sphere(v3(1.2, 0.0, 0.0), fx(0.3), &all().with_sdf(false))
            .is_empty());
    }

    #[test]
    fn capsule_cast_against_a_sphere_body_and_a_box() {
        let mut w = world();
        let b = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        // oracle: the segment (1..3, y, 0): its end (1, y, 0) reaches 1.5 from
        // the centre at y = √1.25: t = 5 − √1.25, normal (1, √1.25, 0)/1.5.
        let s = 1.25f64.sqrt();
        let n = [1.0 / 1.5, s / 1.5, 0.0];
        assert_hit(
            capsule_cast(
                &w,
                [1.0, 5.0, 0.0],
                [3.0, 5.0, 0.0],
                0.5,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            RayTarget::Body(b),
            5.0 - s,
            n,
            n,
            EXACT,
        );
        // oracle: coinciding ends are the sphere cast: t = 8.5.
        assert_hit(
            capsule_cast(
                &w,
                [-10.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0],
                0.5,
                [1.0, 0.0, 0.0],
                100.0,
            ),
            RayTarget::Body(b),
            8.5,
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            EXACT,
        );
        let mut w = world();
        let b = unit_box(&mut w);
        // oracle: horizontal capsule over the top face y = 1: t = 5 − 1 − 0.5.
        assert_face_hit(
            capsule_cast(
                &w,
                [-0.5, 5.0, 0.0],
                [0.5, 5.0, 0.0],
                0.5,
                [0.0, -1.0, 0.0],
                100.0,
            ),
            RayTarget::Body(b),
            3.5,
            1.0,
            -0.5,
            0.5,
        );
        // oracle: moving away from the box: no hit.
        assert_eq!(
            capsule_cast(
                &w,
                [-0.5, 5.0, 0.0],
                [0.5, 5.0, 0.0],
                0.5,
                [0.0, 1.0, 0.0],
                100.0
            ),
            None
        );
    }

    #[test]
    fn overlaps_of_bodies_and_the_box_edge_gap() {
        let mut w = world();
        let s = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        // oracle: (1.4, 0, 0) is 0.4 from the sphere: within 0.5, not 0.3.
        assert_eq!(
            overlap_s(&w, [1.4, 0.0, 0.0], 0.5),
            vec![RayTarget::Body(s)]
        );
        assert!(overlap_s(&w, [1.4, 0.0, 0.0], 0.3).is_empty());
        // A negative radius overlaps nothing.
        assert!(overlap_s(&w, [0.0, 0.0, 0.0], -1.0).is_empty());
        // oracle: the box [0.5, 2]³'s nearest point (0.5, 0.5, 0.5) is √0.75 =
        // 0.866 from the centre: inside radius 1.
        assert_eq!(
            overlap_b(&w, [0.5, 0.5, 0.5], [2.0, 2.0, 2.0]),
            vec![RayTarget::Body(s)]
        );
        // oracle: the box [0.6, 2]³: √1.08 = 1.04 > 1, no overlap.
        assert!(overlap_b(&w, [0.6, 0.6, 0.6], [2.0, 2.0, 2.0]).is_empty());
        // A box with min > max contains nothing.
        assert!(overlap_b(&w, [1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]).is_empty());

        let mut w = world();
        let b = unit_box(&mut w);
        // oracle: (1.3, 1.3, 0) is 0.3·√2 = 0.424 from the edge (1, 1, z): within
        // 0.5, not 0.4.
        assert_eq!(
            overlap_s(&w, [1.3, 1.3, 0.0], 0.5),
            vec![RayTarget::Body(b)]
        );
        assert!(overlap_s(&w, [1.3, 1.3, 0.0], 0.4).is_empty());
        // The box turned 45° about Y reaches x = √2 at z = 0: an AABB from
        // x = 1.3 meets it, one from x = 1.5 does not.
        w.bodies[b].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::HALF_PI.half());
        assert_eq!(
            overlap_b(&w, [1.3, -0.5, -0.1], [2.0, 0.5, 0.1]),
            vec![RayTarget::Body(b)]
        );
        assert!(overlap_b(&w, [1.5, -0.5, -0.1], [2.0, 0.5, 0.1]).is_empty());
    }

    #[test]
    fn filters_and_ties() {
        let mut w = world();
        let b = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        let p = plane(&mut w, Vec3Fix::UNIT_Y, -5.0);
        // oracle: down from (0, 10, 0): the sphere first (t = 10 − 1.5), the
        // plane y = −5 at t = 14.5 when the body is excluded, nothing when the
        // static colliders are hidden too.
        let c = v3(0.0, 10.0, 0.0);
        let down = -Vec3Fix::UNIT_Y;
        let h = w
            .cast_sphere(c, fx(0.5), down, fx(100.0), &all())
            .expect("hit");
        assert_eq!(h.target, RayTarget::Body(b));
        let h = w
            .cast_sphere(c, fx(0.5), down, fx(100.0), &all().excluding_body(b))
            .expect("hit");
        assert_eq!(h.target, RayTarget::StaticCollider(p));
        assert!((h.t.to_f64() - 14.5).abs() < EXACT);
        assert_eq!(
            w.cast_sphere(
                c,
                fx(0.5),
                down,
                fx(100.0),
                &all().excluding_body(b).with_static(false)
            ),
            None
        );
        // Two identical bodies: the tie goes to the lower index; overlaps are
        // sorted by target.
        let mut w = world();
        let b0 = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        let b1 = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
        let h = sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0).expect("hit");
        assert_eq!(h.target, RayTarget::Body(b0));
        assert_eq!(
            overlap_s(&w, [0.0, 0.0, 0.0], 0.1),
            vec![RayTarget::Body(b0), RayTarget::Body(b1)]
        );
    }

    #[test]
    fn capsule_penetrations_closed_forms() {
        let mut w = world();
        let p = plane(&mut w, Vec3Fix::UNIT_Y, 0.0);
        // oracle: the segment's lower end is 0.2 above the plane: depth
        // 0.3 − 0.2 = 0.1 along +Y.
        let pens = w
            .capsule_penetrations(v3(0.0, 0.2, 0.0), v3(0.0, 1.4, 0.0), fx(0.3), &all())
            .expect("the segment is outside");
        assert_eq!(pens.len(), 1);
        assert_eq!(pens[0].target, RayTarget::StaticCollider(p));
        assert!((pens[0].depth.to_f64() - 0.1).abs() < EXACT);
        assert_vec(pens[0].normal, [0.0, 1.0, 0.0], EXACT, "normal");
        // oracle: 0.4 above: no overlap.
        assert!(w
            .capsule_penetrations(v3(0.0, 0.4, 0.0), v3(0.0, 1.4, 0.0), fx(0.3), &all())
            .expect("outside")
            .is_empty());
        // oracle: a segment crossing the plane has no push-out direction.
        assert!(w
            .capsule_penetrations(v3(0.0, -0.2, 0.0), v3(0.0, 1.4, 0.0), fx(0.3), &all())
            .is_none());
    }

    #[test]
    fn helper_closed_forms() {
        // aabb_gap: x gap 1, z gap 2, y overlapping: √5.
        let a = AABB::new(v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0));
        let b = AABB::new(v3(2.0, 0.5, 3.0), v3(3.0, 2.0, 4.0));
        assert!((aabb_gap(&a, &b).to_f64() - 5f64.sqrt()).abs() < EXACT);
        assert_eq!(aabb_gap(&a, &a), Fix128::ZERO);
        // box_entry: [0, 1]³ moving +X meets [3, 4] × [0, 1]² at t = 2; not
        // within max_t = 1; never when the Y ranges are apart.
        let target = AABB::new(v3(3.0, 0.0, 0.0), v3(4.0, 1.0, 1.0));
        let entry = box_entry(&a, Vec3Fix::UNIT_X, fx(10.0), &target).expect("meets");
        assert!((entry.to_f64() - 2.0).abs() < EXACT);
        assert_eq!(box_entry(&a, Vec3Fix::UNIT_X, Fix128::ONE, &target), None);
        let above = AABB::new(v3(3.0, 5.0, 0.0), v3(4.0, 6.0, 1.0));
        assert_eq!(box_entry(&a, Vec3Fix::UNIT_X, fx(10.0), &above), None);
        // oracle: moving −X toward [−4, −3]: t = 3.
        let behind = AABB::new(v3(-4.0, 0.0, 0.0), v3(-3.0, 1.0, 1.0));
        let entry = box_entry(&a, -Vec3Fix::UNIT_X, fx(10.0), &behind).expect("meets");
        assert!((entry.to_f64() - 3.0).abs() < EXACT);
        // ratio_within: 1/4 = 0.25; a tiny denominator is clamped to ±big.
        let tiny = Fix128 { hi: 0, lo: 1 };
        assert!((ratio_within(Fix128::ONE, fx(4.0), fx(5.0)).to_f64() - 0.25).abs() < EXACT);
        assert_eq!(ratio_within(Fix128::ONE, tiny, fx(5.0)), fx(5.0));
        assert_eq!(ratio_within(Fix128::NEG_ONE, tiny, fx(5.0)), fx(-5.0));
        // tame_direction: (2²¹, 0, 0) is scaled to (1, 0, 0); (3, 4, 0) is kept.
        assert_eq!(
            tame_direction(Vec3Fix::new(
                Fix128::from_int(1 << 21),
                Fix128::ZERO,
                Fix128::ZERO
            )),
            Vec3Fix::UNIT_X
        );
        assert_eq!(tame_direction(v3(3.0, 4.0, 0.0)), v3(3.0, 4.0, 0.0));
        assert!(box_is_valid(&a));
        assert!(!box_is_valid(&AABB::new(
            v3(0.0, 1.0, 0.0),
            v3(1.0, 0.0, 1.0)
        )));
        // closest_on_segment: clamped to the ends, degenerate segment.
        let (s0, s1) = (v3(0.0, 0.0, 0.0), v3(2.0, 0.0, 0.0));
        assert_eq!(
            closest_on_segment(s0, s1, v3(1.0, 5.0, 0.0)),
            v3(1.0, 0.0, 0.0)
        );
        assert_eq!(closest_on_segment(s0, s1, v3(-3.0, 1.0, 0.0)), s0);
        assert_eq!(closest_on_segment(s0, s0, v3(5.0, 5.0, 5.0)), s0);
        // segment_segment: crossing → None; skew at distance 1; point cases.
        assert!(segment_segment(
            v3(-1.0, 0.0, 0.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, -1.0, 0.0),
            v3(0.0, 1.0, 0.0)
        )
        .is_none());
        let (d, _, _) = segment_segment(
            v3(-1.0, 0.0, 0.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, -1.0, 1.0),
            v3(0.0, 1.0, 1.0),
        )
        .expect("apart");
        assert!((d.to_f64() - 1.0).abs() < ITER);
        let (d, _, _) =
            segment_segment(s0, s0, v3(3.0, 4.0, 0.0), v3(3.0, 4.0, 0.0)).expect("apart");
        assert!((d.to_f64() - 5.0).abs() < EXACT);
        let (d, _, _) =
            segment_segment(v3(1.0, 3.0, 0.0), v3(1.0, 3.0, 0.0), s0, s1).expect("apart");
        assert!((d.to_f64() - 3.0).abs() < EXACT);
        let (d, _, _) =
            segment_segment(s0, s1, v3(1.0, 0.0, 2.0), v3(1.0, 0.0, 2.0)).expect("apart");
        assert!((d.to_f64() - 2.0).abs() < EXACT);
        // point_box_local: on the face x = 1 it touches with normal +X; outside
        // it is the distance to the clamped point; inside it is Inside.
        let h = v3(1.0, 1.0, 1.0);
        match point_box_local(v3(1.0, 0.5, 0.5), h) {
            Dist::Outside { dist, normal, .. } => {
                assert_eq!(dist, Fix128::ZERO);
                assert_eq!(normal, Vec3Fix::UNIT_X);
            }
            other => panic!("expected touching, got {other:?}"),
        }
        assert!((point_box_local(v3(4.0, 5.0, 0.0), h).value().to_f64() - 5.0).abs() < EXACT);
        assert!(matches!(point_box_local(Vec3Fix::ZERO, h), Dist::Inside));
        // point_cylinder_local: on the cap; 2 beside the side.
        match point_cylinder_local(v3(0.0, -1.0, 0.0), Fix128::ONE, Fix128::ONE) {
            Dist::Outside { dist, normal, .. } => {
                assert_eq!(dist, Fix128::ZERO);
                assert_eq!(normal, -Vec3Fix::UNIT_Y);
            }
            other => panic!("expected touching, got {other:?}"),
        }
        assert!(
            (point_cylinder_local(v3(3.0, 0.0, 0.0), Fix128::ONE, Fix128::ONE)
                .value()
                .to_f64()
                - 2.0)
                .abs()
                < EXACT
        );
        // Dist: a distance shrunk past zero is Inside; a bound shrinks by it.
        assert!(matches!(
            Dist::toward(v3(2.0, 0.0, 0.0), Vec3Fix::ZERO).shrunk(fx(3.0)),
            Dist::Inside
        ));
        assert_eq!(
            Dist::AtLeast(fx(2.0)).shrunk(Fix128::ONE).value(),
            Fix128::ONE
        );
        assert!(!Dist::AtLeast(Fix128::ZERO).within(fx(10.0)));
        assert!(Dist::Inside.within(Fix128::ZERO));
        assert_eq!(Dist::Inside.value(), Fix128::NEG_ONE);
        assert!(matches!(
            Dist::toward(Vec3Fix::ZERO, Vec3Fix::ZERO),
            Dist::Inside
        ));
    }
}

#[cfg(test)]
mod zz_s34_gjk_error {
    use super::*;
    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn r(&mut self, a: f64, b: f64) -> f64 {
            a + (b - a) * self.u()
        }
    }
    fn seg(px: f64, py: f64, a: (f64, f64), b: (f64, f64)) -> f64 {
        let (ex, ey) = (b.0 - a.0, b.1 - a.1);
        let u = (((px - a.0) * ex + (py - a.1) * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
        ((px - a.0 - u * ex).powi(2) + (py - a.1 - u * ey).powi(2)).sqrt()
    }
    fn sd(cone: bool, l: [f64; 3]) -> f64 {
        let rho = (l[0] * l[0] + l[2] * l[2]).sqrt();
        if cone {
            let y = l[1] - 0.4;
            let (a, b, c) = ((0.0, -0.8), (0.0, 0.8), (1.0, -0.8));
            seg(rho, y, a, b)
                .min(seg(rho, y, b, c))
                .min(seg(rho, y, c, a))
        } else {
            let qx = rho - 0.9;
            let qy = l[1].abs() - 0.7;
            (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0)
        }
    }
    #[test]
    fn measure_gjk_point_error() {
        if std::env::var("S34E").is_err() {
            return;
        }
        let mut rng = Rng(77);
        for cone in [true, false] {
            let posed = PosedShape {
                shape: if cone {
                    Shape::Cone {
                        radius: Fix128::ONE,
                        half_height: Fix128::from_f64(0.8),
                    }
                } else {
                    Shape::Cylinder {
                        radius: Fix128::from_f64(0.9),
                        half_height: Fix128::from_f64(0.7),
                    }
                },
                position: Vec3Fix::ZERO,
                rotation: QuatFix::IDENTITY,
            };
            let mut worst = 0.0f64;
            let mut hist = [0usize; 40];
            for _ in 0..20000 {
                let ph = rng.r(0.0, std::f64::consts::TAU);
                let (rho, y, nr, ny) = if cone {
                    let s = rng.r(0.02, 0.98);
                    let (nr, ny) = (
                        1.6 / (1.6f64 * 1.6 + 1.0).sqrt(),
                        1.0 / (1.6f64 * 1.6 + 1.0).sqrt(),
                    );
                    (s, 1.2 - 1.6 * s, nr, ny)
                } else if rng.u() < 0.5 {
                    (0.9, rng.r(-0.68, 0.68), 1.0, 0.0)
                } else {
                    (rng.r(0.0, 0.88), 0.7, 0.0, 1.0)
                };
                let h = 2f64.powi(-(8 + (rng.u() * 9.0) as i32)) * rng.r(1.0, 2.0);
                let p = [
                    (rho + h * nr) * ph.cos(),
                    y + h * ny,
                    (rho + h * nr) * ph.sin(),
                ];
                let pf = Vec3Fix::new(
                    Fix128::from_f64(p[0]),
                    Fix128::from_f64(p[1]),
                    Fix128::from_f64(p[2]),
                );
                let pp = [pf.x.to_f64(), pf.y.to_f64(), pf.z.to_f64()];
                let tru = sd(cone, pp);
                let seg = std::env::var("S34SEG").is_ok();
                let (d, tru) = if seg {
                    // a short segment through p along the surface's tangent (phi direction)
                    let tdir = [-ph.sin(), 0.0, ph.cos()];
                    let hl = rng.r(0.05, 0.3);
                    let a = [p[0] - hl * tdir[0], p[1], p[2] - hl * tdir[2]];
                    let b = [p[0] + hl * tdir[0], p[1], p[2] + hl * tdir[2]];
                    let af = Vec3Fix::new(
                        Fix128::from_f64(a[0]),
                        Fix128::from_f64(a[1]),
                        Fix128::from_f64(a[2]),
                    );
                    let bf = Vec3Fix::new(
                        Fix128::from_f64(b[0]),
                        Fix128::from_f64(b[1]),
                        Fix128::from_f64(b[2]),
                    );
                    let (a, b) = (
                        [af.x.to_f64(), af.y.to_f64(), af.z.to_f64()],
                        [bf.x.to_f64(), bf.y.to_f64(), bf.z.to_f64()],
                    );
                    let at = |s: f64| {
                        [
                            a[0] + s * (b[0] - a[0]),
                            a[1] + s * (b[1] - a[1]),
                            a[2] + s * (b[2] - a[2]),
                        ]
                    };
                    let (mut lo, mut hi) = (0.0f64, 1.0f64);
                    for _ in 0..200 {
                        let m1 = lo + (hi - lo) / 3.0;
                        let m2 = hi - (hi - lo) / 3.0;
                        if sd(cone, at(m1)) <= sd(cone, at(m2)) {
                            hi = m2
                        } else {
                            lo = m1
                        }
                    }
                    let tr = sd(cone, at(0.5 * (lo + hi)))
                        .min(sd(cone, at(0.0)))
                        .min(sd(cone, at(1.0)));
                    if tr <= 0.0 {
                        continue;
                    }
                    let Some((d, _, _)) = gjk_distance(&SegmentSupport(af, bf), &posed) else {
                        continue;
                    };
                    (d, tr)
                } else {
                    let Some((d, _, _)) = gjk_distance(&PointSupport(pf), &posed) else {
                        continue;
                    };
                    (d, tru)
                };
                let e = (d.to_f64() - tru).abs() / 2f64.powi(-64);
                if e > 2f64.powi(31) && std::env::var("S34V").is_ok() {
                    eprintln!("BIG cone={cone} rho={rho:.4} y={y:.4} h={h:.3e} gjk={:.17e} tru={tru:.17e}", d.to_f64());
                }
                worst = worst.max(e);
                let b = if e < 1.0 {
                    0
                } else {
                    (e.log2() as usize + 1).min(39)
                };
                hist[b] += 1;
            }
            eprintln!(
                "ERR cone={cone} worst={:.3e} x2^-64 (log2 {:.1}) hist={:?}",
                worst,
                worst.log2(),
                hist
            );
        }
    }
}
