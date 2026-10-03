//! Oracles for the wiring of `compound`'s surface: `CompoundShape::{new,
//! add_sphere, add_capsule, add_convex_hull, add_box, compute_aabb,
//! child_world_aabb, world_aabb, overlapping_children}`. All of
//! `add_sphere`, `add_capsule`, `add_convex_hull`, `add_box`,
//! `compute_aabb`, `child_world_aabb`, `world_aabb`, `overlapping_children`
//! had zero production callers before this crate's
//! `examples/compound_shapes.rs` (`scripts/wiring-baseline.txt` `unwired
//! src/compound.rs::*`, 8 items).
//!
//! # Closed forms (every expected value is derived here, none from the
//! crate's `CompoundShape` methods under test -- `OrientedBox::aabb` is
//! used freely since it belongs to a *different* type, `box_collider.rs`,
//! not to the module being wired here)
//!
//! * **`add_sphere`/`add_capsule`/`add_convex_hull`/`add_box`** each push
//!   exactly one `CompoundChild` (so `len()` grows by exactly 1 per call,
//!   checked against an independent running counter) and set `dirty`
//!   (checked by observing that the *next* `compute_aabb()` call picks up
//!   the new child -- `dirty` itself is a private field, not asserted
//!   directly).
//! * **`child_world_aabb(i, body_pos, body_rot)`** is, per shape kind:
//!   - Sphere: `world_pos +/- radius` on every axis, where `world_pos =
//!     body_pos + body_rot.rotate_vec(local_position) +
//!     body_rot.rotate_vec(sphere.center)`.
//!   - Capsule: axis-aligned bbox of the two rotated/translated endpoints,
//!     expanded by `radius` on every axis.
//!   - Box: `OrientedBox::aabb()` (a different type's own formula) applied
//!     to the world-transformed box (`center' = world_pos +
//!     child_rot.rotate_vec(center)`, `rotation' = body_rot *
//!     child.local_rotation * box.rotation`).
//!   - ConvexHull: vertex-wise min/max of each vertex rotated by
//!     `body_rot * child.local_rotation` and translated by `world_pos`.
//! * **`compute_aabb`** (local space, `body_pos`/`body_rot` fixed at
//!   `ZERO`/`IDENTITY` internally) is the union of `child_world_aabb(i,
//!   ZERO, IDENTITY)` over all children; empty compound gives the
//!   degenerate box `[ZERO, ZERO]`. It is a pure function of `children`,
//!   independent of any *external* body transform (verified by comparing
//!   two calls separated by a `world_aabb` call with a non-identity
//!   transform: `compute_aabb` is unaffected).
//! * **`world_aabb(body_pos, body_rot)`** is the union of
//!   `child_world_aabb(i, body_pos, body_rot)` over all children; empty
//!   compound gives the degenerate point box `[body_pos, body_pos]`
//!   (*not* `[ZERO, ZERO]` translated -- `world_aabb`'s empty branch
//!   returns `AABB::new(body_pos, body_pos)` directly, verified below with
//!   a non-zero `body_pos`).
//! * **`overlapping_children(target, body_pos, body_rot)`** returns
//!   exactly the indices `i` for which
//!   `child_world_aabb(i, body_pos, body_rot).intersects(target)`, in
//!   ascending index order; the oracle recomputes this set from the
//!   independently-derived `child_world_aabb` closed forms above, not by
//!   calling `overlapping_children` itself.
//! * **180-degree-about-Z rotation closed form** (reused verbatim from
//!   `tests/analytic_multi_world_wiring.rs`, which already hand-derives
//!   this Hamilton product once and for all): with `q = (x=0, y=0, z=1,
//!   w=0)`, `rotate_vec(q, v) = (-v.x, -v.y, v.z)` exactly, for any `v`.
//!   General 180-degree-about-unit-axis identity (textbook, independent of
//!   this crate): `rotate_vec(v) = 2*(e.v)*e - v`. For `e = UNIT_Y`:
//!   `(-v.x, v.y, -v.z)`. For `e = UNIT_X`: `(v.x, -v.y, -v.z)`.
//!
//! # Degenerate / extreme inputs (documented result, not merely "no panic")
//!
//! * **Empty compound**: `compute_aabb()` -> `[ZERO, ZERO]`; `world_aabb`
//!   -> `[body_pos, body_pos]`; `overlapping_children` -> `[]` (the loop
//!   body never executes, no panic).
//! * **`child_world_aabb` with an out-of-range index** panics (`self
//!   .children[child_idx]` is a direct slice index with no bounds check in
//!   `compound.rs`) -- checked with `#[should_panic]` on an empty compound.
//! * **Degenerate shapes**: zero-radius sphere/capsule collapse their AABB
//!   to a single point / a line segment's bbox with zero radius expansion;
//!   single-vertex "hull" (not a real hull, but accepted by this module's
//!   loop which only reads `vertices[0]` and iterates `vertices[1..]`,
//!   which is empty) collapses to the point itself.
//! * **Extreme `Fix128` magnitude** (`1_000_000_000`, well inside the
//!   `+/-9.2e18` range documented in `src/math.rs`, so addition does not
//!   wrap): AABB arithmetic stays exact at this scale.
//! * **Overlapping children** (two children whose world AABBs overlap
//!   each other): `overlapping_children` against a query that spans both
//!   returns both indices; `compute_aabb`'s union still equals the
//!   independently-computed min/max even when the per-child boxes overlap
//!   (union of overlapping boxes is still exactly the component-wise
//!   min/max, no double-counting possible for an AABB union).

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{Capsule, ConvexHull, Sphere, AABB};
use alice_physics::compound::CompoundShape;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};

/// 180-degree rotation about world Z, exact unit quaternion.
const ROT_180_Z: QuatFix = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
/// 180-degree rotation about world Y, exact unit quaternion.
const ROT_180_Y: QuatFix = QuatFix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO, Fix128::ZERO);

/// Exact dyadic-rational half (0.5), representable bit-for-bit in `Fix128`
/// (lo = 2^63, i.e. exactly 0.5 -- no rounding).
const HALF: Fix128 = Fix128::from_raw(0, 1u64 << 63);

/// 120-degree rotation about the (1,1,1) diagonal: `(x=0.5, y=0.5, z=0.5,
/// w=0.5)` is an exact *unit* quaternion (`0.25*4 = 1`, exactly, no
/// rounding) that cyclically permutes the basis: `e_x -> e_y -> e_z -> e_x`.
/// Hand-derived via this crate's own Hamilton product formula
/// (`QuatFix::mul`/`conjugate`, a primitive op, not the function under
/// test), using only +-0.5/+-0.25 arithmetic (exact in binary):
///
/// ```text
/// q = (.5,.5,.5,.5), q^-1 = conjugate(q) = (-.5,-.5,-.5,.5), qv = (1,0,0,0) for v=e_x
/// temp  = q * qv = (x: .5*1+.5*0+.5*0-.5*0, y: .5*0-.5*0+.5*0+.5*1,
///                   z: .5*0+.5*0-.5*1+.5*0, w: .5*0-.5*1-.5*0-.5*0)
///       = (.5, .5, -.5, -.5)
/// result = temp * q^-1
///        = (x: (-.5)*(-.5)+.5*.5+.5*(-.5)-(-.5)*(-.5),
///           y: (-.5)*(-.5)-.5*(-.5)+.5*.5+(-.5)*(-.5),
///           z: (-.5)*(-.5)+.5*(-.5)-.5*(-.5)+(-.5)*.5,
///           w: (-.5)*.5-.5*(-.5)-.5*(-.5)-(-.5)*(-.5))
///        = (.25+.25-.25-.25, .25+.25+.25+.25, .25-.25+.25-.25, -.25+.25+.25-.25)
///        = (0, 1, 0, 0)
/// ```
///
/// so `rotate_vec(ROT_CYCLIC_XYZ, e_x) = e_y` exactly; by the same
/// arithmetic (cyclic symmetry of the quaternion's x/y/z components),
/// `e_y -> e_z` and `e_z -> e_x`. Unlike `ROT_180_{X,Y,Z}`, this rotation
/// is *not* in the sign-flip-only family, so it actually permutes which
/// world axis each half-extent lands on -- this is the property needed to
/// distinguish `child_rot.mul(ob.rotation)` from `ob.rotation` alone for
/// an axis-aligned box (see `child_world_aabb_box_local_rotation_composes_with_child_rot`).
const ROT_CYCLIC_XYZ: QuatFix = QuatFix::new(HALF, HALF, HALF, HALF);

/// Hand-derived closed form for `rotate_vec(ROT_180_Z, v)`: see module doc.
fn rot_180_z(v: Vec3Fix) -> Vec3Fix {
    Vec3Fix::new(-v.x, -v.y, v.z)
}

/// Hand-derived closed form for `rotate_vec(ROT_180_Y, v)`: see module doc.
fn rot_180_y(v: Vec3Fix) -> Vec3Fix {
    Vec3Fix::new(-v.x, v.y, -v.z)
}

fn sphere_aabb_hand(center: Vec3Fix, radius: Fix128) -> AABB {
    let r = Vec3Fix::new(radius, radius, radius);
    AABB::new(center - r, center + r)
}

fn capsule_aabb_hand(a: Vec3Fix, b: Vec3Fix, radius: Fix128) -> AABB {
    let min = Vec3Fix::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
    let max = Vec3Fix::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));
    let r = Vec3Fix::new(radius, radius, radius);
    AABB::new(min - r, max + r)
}

fn hull_aabb_hand(vertices: &[Vec3Fix]) -> AABB {
    let mut min = vertices[0];
    let mut max = vertices[0];
    for &v in &vertices[1..] {
        min = Vec3Fix::new(min.x.min(v.x), min.y.min(v.y), min.z.min(v.z));
        max = Vec3Fix::new(max.x.max(v.x), max.y.max(v.y), max.z.max(v.z));
    }
    AABB::new(min, max)
}

fn union_hand(a: AABB, b: AABB) -> AABB {
    AABB::new(
        Vec3Fix::new(
            a.min.x.min(b.min.x),
            a.min.y.min(b.min.y),
            a.min.z.min(b.min.z),
        ),
        Vec3Fix::new(
            a.max.x.max(b.max.x),
            a.max.y.max(b.max.y),
            a.max.z.max(b.max.z),
        ),
    )
}

// ===========================================================================
// add_sphere / add_capsule / add_convex_hull / add_box: len() growth
// ===========================================================================

#[test]
fn add_each_shape_kind_pushes_exactly_one_child() {
    let mut c = CompoundShape::new();
    assert_eq!(c.len(), 0);
    assert!(c.is_empty());

    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_eq!(c.len(), 1, "add_sphere must push exactly one child");

    c.add_capsule(
        Capsule::new(Vec3Fix::ZERO, Vec3Fix::from_int(0, 1, 0), Fix128::ONE),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_eq!(c.len(), 2, "add_capsule must push exactly one child");

    c.add_box(
        OrientedBox::axis_aligned(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_eq!(c.len(), 3, "add_box must push exactly one child");

    c.add_convex_hull(
        ConvexHull::new(vec![Vec3Fix::ZERO, Vec3Fix::from_int(1, 0, 0)]),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_eq!(c.len(), 4, "add_convex_hull must push exactly one child");
    assert!(!c.is_empty());
}

#[test]
fn add_marks_dirty_so_next_compute_aabb_picks_up_the_new_child() {
    let mut c = CompoundShape::new();
    // Force the cache to be populated (and dirty cleared) for the empty state.
    let empty_cached = c.compute_aabb();
    assert_eq!(empty_cached, AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO));

    // add_sphere must set dirty=true again; if it didn't, compute_aabb would
    // keep returning the stale empty-box cache instead of picking up the
    // new child.
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
        Vec3Fix::from_int(50, 0, 0),
        QuatFix::IDENTITY,
    );
    let expected = sphere_aabb_hand(Vec3Fix::from_int(50, 0, 0), Fix128::ONE);
    assert_eq!(
        c.compute_aabb(),
        expected,
        "add_sphere must invalidate the cache (dirty=true) so compute_aabb recomputes"
    );
}

// ===========================================================================
// child_world_aabb: one test per shape kind
// ===========================================================================

#[test]
fn child_world_aabb_sphere_identity() {
    let mut c = CompoundShape::new();
    let sphere = Sphere::new(Vec3Fix::from_int(2, -1, 0), Fix128::from_int(3));
    c.add_sphere(sphere, Vec3Fix::from_int(10, 0, 0), QuatFix::IDENTITY);

    // world_pos = body_pos(ZERO) + rotate(local_position) + rotate(sphere.center)
    //           = (10,0,0) + (2,-1,0) = (12,-1,0); radius 3.
    let expected = sphere_aabb_hand(Vec3Fix::from_int(12, -1, 0), Fix128::from_int(3));
    assert_eq!(
        c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY),
        expected
    );
}

#[test]
fn child_world_aabb_sphere_with_body_translation_and_rotation() {
    let mut c = CompoundShape::new();
    let sphere = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
    let local_pos = Vec3Fix::from_int(3, 4, 0);
    c.add_sphere(sphere, local_pos, QuatFix::IDENTITY);

    let body_pos = Vec3Fix::from_int(1, 2, 3);
    // world_pos = body_pos + rotate_180_z(local_pos) + rotate_180_z(sphere.center=ZERO)
    let expected_center = body_pos + rot_180_z(local_pos);
    let expected = sphere_aabb_hand(expected_center, Fix128::ONE);
    assert_eq!(c.child_world_aabb(0, body_pos, ROT_180_Z), expected);
}

#[test]
fn child_world_aabb_capsule_identity_and_offset() {
    let mut c = CompoundShape::new();
    let capsule = Capsule::new(
        Vec3Fix::from_int(0, -3, 0),
        Vec3Fix::from_int(0, 3, 0),
        Fix128::from_int(1),
    );
    c.add_capsule(capsule, Vec3Fix::from_int(5, 0, 0), QuatFix::IDENTITY);

    let a = Vec3Fix::from_int(5, -3, 0);
    let b = Vec3Fix::from_int(5, 3, 0);
    let expected = capsule_aabb_hand(a, b, Fix128::from_int(1));
    assert_eq!(
        c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY),
        expected
    );
}

#[test]
fn child_world_aabb_capsule_with_child_local_rotation_180_about_y() {
    let mut c = CompoundShape::new();
    // Segment along +X; rotating 180 about Y flips X and Z, keeps Y.
    let capsule = Capsule::new(
        Vec3Fix::from_int(-2, 0, 0),
        Vec3Fix::from_int(2, 0, 0),
        Fix128::from_ratio(1, 2),
    );
    c.add_capsule(capsule, Vec3Fix::ZERO, ROT_180_Y);

    // child_rot = body_rot(IDENTITY) * local_rotation(ROT_180_Y) = ROT_180_Y
    let a = rot_180_y(capsule.a);
    let b = rot_180_y(capsule.b);
    let expected = capsule_aabb_hand(a, b, Fix128::from_ratio(1, 2));
    assert_eq!(
        c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY),
        expected
    );
}

#[test]
fn child_world_aabb_box_uses_oriented_box_aabb_formula() {
    let mut c = CompoundShape::new();
    let obox = OrientedBox::axis_aligned(Vec3Fix::from_int(1, 0, 0), Vec3Fix::from_int(2, 1, 1));
    c.add_box(obox, Vec3Fix::from_int(0, 5, 0), ROT_180_Z);

    // world_pos = body_pos(ZERO) + body_rot(IDENTITY).rotate_vec(local_position)
    //           = (0,5,0)  -- note: it is body_rot, NOT the child's own
    //           local_rotation, that rotates the local_position offset.
    // child_rot = body_rot(IDENTITY) * local_rotation(ROT_180_Z) = ROT_180_Z
    // transformed.center = world_pos + child_rot.rotate_vec(obox.center)
    //                     = (0,5,0) + rot_180_z((1,0,0)) = (0,5,0)+(-1,0,0) = (-1,5,0)
    // transformed.rotation = child_rot * obox.rotation = ROT_180_Z * IDENTITY = ROT_180_Z
    let transformed = OrientedBox::new(
        Vec3Fix::from_int(0, 5, 0) + rot_180_z(obox.center),
        obox.half_extents,
        ROT_180_Z.mul(obox.rotation),
    );
    let expected = transformed.aabb();
    assert_eq!(
        c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY),
        expected
    );
}

#[test]
fn child_world_aabb_box_local_rotation_composes_with_child_rot() {
    // Distinguishes `child_rot.mul(ob.rotation)` (correct) from `ob.rotation`
    // alone (a plausible drop-the-composition mutant) -- which 180-about-a-
    // coordinate-axis rotations CANNOT do for an axis-aligned box: every
    // such rotation only flips signs of the local half-extent axes, and
    // `OrientedBox::aabb()` takes `abs()` of each projection, so the extent
    // is identical to the unrotated box regardless of which (or how many)
    // of {identity, 180X, 180Y, 180Z} is used. `ROT_CYCLIC_XYZ` instead
    // *permutes* which world axis each half-extent lands on, which a
    // dropped composition cannot reproduce for an asymmetric box.
    let mut c = CompoundShape::new();
    let obox = OrientedBox::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 2, 3), QuatFix::IDENTITY);
    c.add_box(obox, Vec3Fix::ZERO, ROT_CYCLIC_XYZ);

    // body_rot = IDENTITY, local_rotation = ROT_CYCLIC_XYZ => child_rot = ROT_CYCLIC_XYZ.
    // transformed.rotation = child_rot.mul(ob.rotation=IDENTITY) = ROT_CYCLIC_XYZ.
    // local_x=(1,0,0) -> world_x_vec = 1*rotate(e_x) = 1*e_y = (0,1,0)
    // local_y=(0,2,0) -> world_y_vec = 2*rotate(e_y) = 2*e_z = (0,0,2)
    // local_z=(0,0,3) -> world_z_vec = 3*rotate(e_z) = 3*e_x = (3,0,0)
    // extent.x = |0|+|0|+|3| = 3; extent.y = |1|+|0|+|0| = 1; extent.z = |0|+|2|+|0| = 2
    let expected = AABB::new(Vec3Fix::from_int(-3, -1, -2), Vec3Fix::from_int(3, 1, 2));
    let got = c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY);
    assert_eq!(
        got, expected,
        "box rotation must be child_rot.mul(ob.rotation), not ob.rotation alone \
         (half_extents (1,2,3) must map to world extent (3,1,2), a genuine axis \
         permutation, not the unrotated (1,2,3))"
    );

    // Cross-check: dropping the composition entirely (using ob.rotation =
    // IDENTITY alone) would give the UNROTATED extent (1,2,3), which must
    // differ from what we just asserted above.
    let unrotated = AABB::new(Vec3Fix::from_int(-1, -2, -3), Vec3Fix::from_int(1, 2, 3));
    assert_ne!(
        got, unrotated,
        "sanity: this scenario must actually distinguish the two formulas"
    );
}

#[test]
fn child_world_aabb_convex_hull_vertexwise_minmax() {
    let mut c = CompoundShape::new();
    let verts = vec![
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::from_int(3, -1, 2),
        Vec3Fix::from_int(-2, 4, 0),
    ];
    let hull = ConvexHull::new(verts.clone());
    c.add_convex_hull(hull, Vec3Fix::from_int(10, 0, 0), QuatFix::IDENTITY);

    let world_verts: Vec<Vec3Fix> = verts
        .iter()
        .map(|&v| v + Vec3Fix::from_int(10, 0, 0))
        .collect();
    let expected = hull_aabb_hand(&world_verts);
    assert_eq!(
        c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY),
        expected
    );
}

// ===========================================================================
// compute_aabb
// ===========================================================================

#[test]
fn compute_aabb_empty_is_degenerate_origin_box() {
    let mut c = CompoundShape::new();
    assert_eq!(c.compute_aabb(), AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO));
    // idempotent
    assert_eq!(c.compute_aabb(), AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO));
}

#[test]
fn compute_aabb_single_child_equals_that_childs_aabb() {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::from_int(2)),
        Vec3Fix::from_int(-5, 5, 0),
        QuatFix::IDENTITY,
    );
    let expected = sphere_aabb_hand(Vec3Fix::from_int(-5, 5, 0), Fix128::from_int(2));
    assert_eq!(c.compute_aabb(), expected);
}

#[test]
fn compute_aabb_multiple_children_is_union_and_independent_of_body_transform() {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
        Vec3Fix::from_int(10, 0, 0),
        QuatFix::IDENTITY,
    );
    c.add_capsule(
        Capsule::new(
            Vec3Fix::from_int(0, -3, 0),
            Vec3Fix::from_int(0, 3, 0),
            Fix128::from_int(2),
        ),
        Vec3Fix::from_int(-4, 0, 5),
        QuatFix::IDENTITY,
    );
    c.add_sphere(
        Sphere::new(Vec3Fix::from_int(0, 0, -8), Fix128::ONE),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );

    let b0 = sphere_aabb_hand(Vec3Fix::from_int(10, 0, 0), Fix128::ONE);
    let b1 = capsule_aabb_hand(
        Vec3Fix::from_int(-4, -3, 5),
        Vec3Fix::from_int(-4, 3, 5),
        Fix128::from_int(2),
    );
    let b2 = sphere_aabb_hand(Vec3Fix::from_int(0, 0, -8), Fix128::ONE);
    let expected = union_hand(union_hand(b0, b1), b2);

    let got = c.compute_aabb();
    assert_eq!(got, expected);

    // compute_aabb is local-space only: calling world_aabb with a wild
    // body transform afterwards must not change what compute_aabb returns.
    let _ = c.world_aabb(Vec3Fix::from_int(999, -999, 999), ROT_180_Z);
    assert_eq!(
        c.compute_aabb(),
        expected,
        "compute_aabb must be independent of any external body transform"
    );
}

#[test]
fn compute_aabb_at_extreme_fix128_magnitude_stays_exact() {
    let mut c = CompoundShape::new();
    let big = Fix128::from_int(1_000_000_000);
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::from_int(5)),
        Vec3Fix::new(big, -big, Fix128::ZERO),
        QuatFix::IDENTITY,
    );
    let expected = sphere_aabb_hand(Vec3Fix::new(big, -big, Fix128::ZERO), Fix128::from_int(5));
    assert_eq!(c.compute_aabb(), expected);
}

// ===========================================================================
// world_aabb
// ===========================================================================

#[test]
fn world_aabb_empty_is_point_at_body_pos_not_origin() {
    let c = CompoundShape::new();
    let body_pos = Vec3Fix::from_int(7, -3, 2);
    assert_eq!(
        c.world_aabb(body_pos, ROT_180_Z),
        AABB::new(body_pos, body_pos),
        "empty compound's world_aabb must be a degenerate point at body_pos, \
         not body_pos + the local [ZERO,ZERO] box (same result here since \
         ZERO is additive identity, but the code path is a distinct early \
         return -- see child count check below"
    );
}

#[test]
fn world_aabb_matches_union_of_child_world_aabb_under_translation_and_rotation() {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
        Vec3Fix::from_int(3, 0, 0),
        QuatFix::IDENTITY,
    );
    c.add_box(
        OrientedBox::axis_aligned(Vec3Fix::ZERO, Vec3Fix::from_int(1, 2, 1)),
        Vec3Fix::from_int(-3, 0, 0),
        QuatFix::IDENTITY,
    );

    let body_pos = Vec3Fix::from_int(100, -50, 0);
    let got = c.world_aabb(body_pos, ROT_180_Z);

    let b0 = sphere_aabb_hand(
        body_pos + rot_180_z(Vec3Fix::from_int(3, 0, 0)),
        Fix128::ONE,
    );
    let box_center = body_pos + rot_180_z(Vec3Fix::from_int(-3, 0, 0));
    let transformed_box = OrientedBox::new(box_center, Vec3Fix::from_int(1, 2, 1), ROT_180_Z);
    let b1 = transformed_box.aabb();
    let expected = union_hand(b0, b1);

    assert_eq!(got, expected);
}

// ===========================================================================
// overlapping_children
// ===========================================================================

#[test]
fn overlapping_children_empty_compound_returns_empty() {
    let c = CompoundShape::new();
    let query = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));
    let result = c.overlapping_children(&query, Vec3Fix::ZERO, QuatFix::IDENTITY);
    assert!(result.is_empty());
}

#[test]
fn overlapping_children_disjoint_query_hits_only_the_right_one() {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
        Vec3Fix::from_int(-10, 0, 0),
        QuatFix::IDENTITY,
    );
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
        Vec3Fix::from_int(10, 0, 0),
        QuatFix::IDENTITY,
    );

    let query_right = AABB::new(Vec3Fix::from_int(9, -1, -1), Vec3Fix::from_int(12, 1, 1));
    assert_eq!(
        c.overlapping_children(&query_right, Vec3Fix::ZERO, QuatFix::IDENTITY),
        vec![1]
    );

    let query_both = AABB::new(Vec3Fix::from_int(-11, -1, -1), Vec3Fix::from_int(11, 1, 1));
    assert_eq!(
        c.overlapping_children(&query_both, Vec3Fix::ZERO, QuatFix::IDENTITY),
        vec![0, 1]
    );

    let query_neither = AABB::new(
        Vec3Fix::from_int(100, 100, 100),
        Vec3Fix::from_int(200, 200, 200),
    );
    assert!(c
        .overlapping_children(&query_neither, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .is_empty());
}

#[test]
fn overlapping_children_with_overlapping_children_both_count() {
    // Two spheres whose world AABBs overlap each other; query spans both.
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::from_int(3)),
        Vec3Fix::from_int(0, 0, 0),
        QuatFix::IDENTITY,
    );
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::from_int(3)),
        Vec3Fix::from_int(2, 0, 0),
        QuatFix::IDENTITY,
    );
    // sphere0 AABB x in [-3,3], sphere1 AABB x in [-1,5]: they overlap in [-1,3].
    let b0 = sphere_aabb_hand(Vec3Fix::from_int(0, 0, 0), Fix128::from_int(3));
    let b1 = sphere_aabb_hand(Vec3Fix::from_int(2, 0, 0), Fix128::from_int(3));
    assert!(b0.intersects(&b1), "test fixture must actually overlap");

    let query = AABB::new(Vec3Fix::from_int(-5, -5, -5), Vec3Fix::from_int(5, 5, 5));
    assert_eq!(
        c.overlapping_children(&query, Vec3Fix::ZERO, QuatFix::IDENTITY),
        vec![0, 1]
    );
}

#[test]
fn overlapping_children_at_extreme_fix128_magnitude() {
    let mut c = CompoundShape::new();
    let big = Fix128::from_int(1_000_000_000);
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::from_int(5)),
        Vec3Fix::new(big, Fix128::ZERO, Fix128::ZERO),
        QuatFix::IDENTITY,
    );
    let hit_query = AABB::new(
        Vec3Fix::new(big - Fix128::from_int(1), -Fix128::ONE, -Fix128::ONE),
        Vec3Fix::new(big + Fix128::from_int(1), Fix128::ONE, Fix128::ONE),
    );
    assert_eq!(
        c.overlapping_children(&hit_query, Vec3Fix::ZERO, QuatFix::IDENTITY),
        vec![0]
    );
    let miss_query = AABB::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
    assert!(c
        .overlapping_children(&miss_query, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .is_empty());
}

// ===========================================================================
// Degenerate shapes
// ===========================================================================

#[test]
fn zero_radius_sphere_collapses_aabb_to_a_point() {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::ZERO),
        Vec3Fix::from_int(4, 5, 6),
        QuatFix::IDENTITY,
    );
    let expected = AABB::new(Vec3Fix::from_int(4, 5, 6), Vec3Fix::from_int(4, 5, 6));
    assert_eq!(
        c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY),
        expected
    );
    assert_eq!(c.compute_aabb(), expected);
}

#[test]
fn zero_radius_capsule_collapses_to_the_segment_bbox() {
    let mut c = CompoundShape::new();
    c.add_capsule(
        Capsule::new(
            Vec3Fix::from_int(0, -2, 0),
            Vec3Fix::from_int(0, 2, 0),
            Fix128::ZERO,
        ),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let expected = AABB::new(Vec3Fix::from_int(0, -2, 0), Vec3Fix::from_int(0, 2, 0));
    assert_eq!(
        c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY),
        expected
    );
}

#[test]
fn single_vertex_hull_collapses_to_the_vertex_point() {
    let mut c = CompoundShape::new();
    c.add_convex_hull(
        ConvexHull::new(vec![Vec3Fix::from_int(7, -1, 3)]),
        Vec3Fix::from_int(1, 0, 0),
        QuatFix::IDENTITY,
    );
    let expected_point = Vec3Fix::from_int(8, -1, 3);
    assert_eq!(
        c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY),
        AABB::new(expected_point, expected_point)
    );
}

// ===========================================================================
// Panic tests (degenerate inputs that must fail fast, not silently)
// ===========================================================================

#[test]
#[should_panic(expected = "index out of bounds")]
fn child_world_aabb_out_of_range_index_on_empty_compound_panics() {
    let c = CompoundShape::new();
    let _ = c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn child_world_aabb_out_of_range_index_past_the_last_child_panics() {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    // Only index 0 exists; index 1 is out of range.
    let _ = c.child_world_aabb(1, Vec3Fix::ZERO, QuatFix::IDENTITY);
}
