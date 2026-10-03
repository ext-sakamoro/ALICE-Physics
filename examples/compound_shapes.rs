//! Compound rigid-body shapes: combine boxes / spheres / capsules / convex
//! hulls into one shape, with per-child local transform, local-space
//! bounding AABB, world-space AABB under an arbitrary body transform, and
//! AABB-based broad-phase pruning of children against a query box.
//!
//! Wiring: `CompoundShape::{add_sphere, add_capsule, add_convex_hull,
//! add_box, compute_aabb, child_world_aabb, world_aabb,
//! overlapping_children}` had zero production callers
//! (`scripts/wiring-baseline.txt` `unwired src/compound.rs::*`, 8 items).
//! This is their production entry point.
//!
//! `CompoundShape` has no path through `PhysicsWorld::add_shaped_body` /
//! `set_body_shape`: both take `crate::shape::Shape`, and that enum (see
//! `src/shape.rs`) has no `Compound` arm -- `grep -n "pub enum Shape" -A40
//! src/shape.rs` shows only the leaf primitive variants. So, same as the
//! other standalone-but-unwired modules already wired in this program,
//! `CompoundShape` stands on its own: it is the production type any caller
//! builds up by hand to get GJK support (`TransformedCompound` implements
//! `Support`) and broad-phase AABB queries for a body made of several
//! primitive children.
//!
//! Every expected value below is derived independently of the function
//! under test, from the primitive types' own known formulas (sphere/capsule
//! AABB by center±radius, `OrientedBox::aabb`'s own sum-of-abs-projections
//! formula, convex hull by vertex-wise min/max) -- never by calling the
//! `CompoundShape` method whose output is being checked.
//!
//! ```bash
//! cargo run --example compound_shapes --features std
//! ```

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{Capsule, ConvexHull, Sphere, AABB};
use alice_physics::compound::CompoundShape;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};

/// 180-degree rotation about the world Z axis, as an exact unit quaternion
/// (`sin(90deg) = 1`, `cos(90deg) = 0`, both exactly representable in
/// `Fix128`). Hand-derived Hamilton product `q*v*q^-1` reduces to exactly
/// `(-v.x, -v.y, v.z)` for any `v` -- see `tests/analytic_multi_world_wiring.rs`
/// for the full derivation of this same identity (reused here, not
/// rederived, since it is independent of `compound.rs`).
const ROT_180_Z: QuatFix = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);

fn rot_180_z(v: Vec3Fix) -> Vec3Fix {
    Vec3Fix::new(-v.x, -v.y, v.z)
}

fn main() {
    // ------------------------------------------------------------------
    // add_sphere / child_world_aabb (identity transform): closed form is
    // the sphere's own `center +/- radius` box, computed here by hand.
    // ------------------------------------------------------------------
    let mut compound = CompoundShape::new();
    assert!(compound.is_empty(), "fresh CompoundShape must be empty");

    let sphere = Sphere::new(Vec3Fix::from_int(1, 0, 0), Fix128::from_int(2));
    compound.add_sphere(sphere, Vec3Fix::ZERO, QuatFix::IDENTITY);
    assert_eq!(compound.len(), 1, "add_sphere must push exactly one child");

    let expected_sphere_aabb = AABB::new(
        Vec3Fix::from_int(1 - 2, -2, -2),
        Vec3Fix::from_int(1 + 2, 2, 2),
    );
    let got = compound.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY);
    assert_eq!(
        got, expected_sphere_aabb,
        "sphere child_world_aabb must equal center +/- radius (hand-derived)"
    );
    println!(
        "[compound] add_sphere -> child_world_aabb(identity) = [{:?}, {:?}] (expected [{:?}, {:?}])",
        got.min, got.max, expected_sphere_aabb.min, expected_sphere_aabb.max
    );

    // ------------------------------------------------------------------
    // add_capsule / child_world_aabb: closed form is the segment's
    // axis-aligned bbox (min/max of the two endpoints) expanded by radius
    // on every axis, computed here by hand.
    // ------------------------------------------------------------------
    let capsule = Capsule::new(
        Vec3Fix::from_int(0, -3, 0),
        Vec3Fix::from_int(0, 3, 0),
        Fix128::from_int(1),
    );
    compound.add_capsule(capsule, Vec3Fix::from_int(5, 0, 0), QuatFix::IDENTITY);
    assert_eq!(compound.len(), 2, "add_capsule must push exactly one child");

    let expected_capsule_aabb = AABB::new(
        Vec3Fix::from_int(5 - 1, -3 - 1, -1),
        Vec3Fix::from_int(5 + 1, 3 + 1, 1),
    );
    let got = compound.child_world_aabb(1, Vec3Fix::ZERO, QuatFix::IDENTITY);
    assert_eq!(
        got, expected_capsule_aabb,
        "capsule child_world_aabb must equal segment bbox +/- radius (hand-derived)"
    );
    println!(
        "[compound] add_capsule -> child_world_aabb(identity) = [{:?}, {:?}] (expected [{:?}, {:?}])",
        got.min, got.max, expected_capsule_aabb.min, expected_capsule_aabb.max
    );

    // ------------------------------------------------------------------
    // add_box / child_world_aabb with a non-identity child local rotation:
    // closed form reuses `OrientedBox::aabb`'s own formula (a primitive
    // type's method, not the `CompoundShape` method under test) applied by
    // hand to the box's world-space center/rotation.
    // ------------------------------------------------------------------
    let obox = OrientedBox::axis_aligned(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
    compound.add_box(obox, Vec3Fix::from_int(0, 10, 0), ROT_180_Z);
    assert_eq!(compound.len(), 3, "add_box must push exactly one child");

    let expected_box_world = OrientedBox::new(
        Vec3Fix::from_int(0, 10, 0) + ROT_180_Z.rotate_vec(obox.center),
        obox.half_extents,
        ROT_180_Z.mul(obox.rotation),
    );
    let expected_box_aabb = expected_box_world.aabb();
    let got = compound.child_world_aabb(2, Vec3Fix::ZERO, QuatFix::IDENTITY);
    assert_eq!(
        got, expected_box_aabb,
        "box child_world_aabb must equal OrientedBox::aabb() of the world-transformed box"
    );
    println!(
        "[compound] add_box (child-local 180deg about Z) -> child_world_aabb(identity) = [{:?}, {:?}]",
        got.min, got.max
    );

    // ------------------------------------------------------------------
    // add_convex_hull / child_world_aabb: closed form is the vertex-wise
    // min/max over the hull's vertices, computed here by hand.
    // ------------------------------------------------------------------
    let hull = ConvexHull::new(vec![
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::from_int(4, 0, 0),
        Vec3Fix::from_int(0, 4, 0),
        Vec3Fix::from_int(0, 0, 4),
    ]);
    compound.add_convex_hull(hull, Vec3Fix::from_int(-10, 0, 0), QuatFix::IDENTITY);
    assert_eq!(
        compound.len(),
        4,
        "add_convex_hull must push exactly one child"
    );

    let expected_hull_aabb = AABB::new(
        Vec3Fix::from_int(-10, 0, 0),
        Vec3Fix::from_int(-10 + 4, 4, 4),
    );
    let got = compound.child_world_aabb(3, Vec3Fix::ZERO, QuatFix::IDENTITY);
    assert_eq!(
        got, expected_hull_aabb,
        "convex hull child_world_aabb must equal vertex-wise min/max (hand-derived)"
    );
    println!(
        "[compound] add_convex_hull -> child_world_aabb(identity) = [{:?}, {:?}]",
        got.min, got.max
    );

    // ------------------------------------------------------------------
    // compute_aabb: the local-space union of all 4 children above,
    // computed independently here as the running min/max of the 4
    // expected AABBs already derived above (not by calling world_aabb or
    // compute_aabb for the expected side).
    // ------------------------------------------------------------------
    let mut expected_min = expected_sphere_aabb.min;
    let mut expected_max = expected_sphere_aabb.max;
    for b in [expected_capsule_aabb, expected_box_aabb, expected_hull_aabb] {
        expected_min = Vec3Fix::new(
            expected_min.x.min(b.min.x),
            expected_min.y.min(b.min.y),
            expected_min.z.min(b.min.z),
        );
        expected_max = Vec3Fix::new(
            expected_max.x.max(b.max.x),
            expected_max.y.max(b.max.y),
            expected_max.z.max(b.max.z),
        );
    }
    let expected_union = AABB::new(expected_min, expected_max);
    let got = compound.compute_aabb();
    assert_eq!(
        got, expected_union,
        "compute_aabb must equal the independently-computed union of the 4 children"
    );
    println!(
        "[compound] compute_aabb (4 children) = [{:?}, {:?}]",
        got.min, got.max
    );

    // ------------------------------------------------------------------
    // world_aabb: same union, translated by the body position (the body
    // rotation is identity here, exercised with a non-identity rotation
    // below). Closed form: expected_union.min/max + body translation.
    // ------------------------------------------------------------------
    let body_pos = Vec3Fix::from_int(100, 0, 0);
    let expected_world = AABB::new(expected_union.min + body_pos, expected_union.max + body_pos);
    let got = compound.world_aabb(body_pos, QuatFix::IDENTITY);
    assert_eq!(
        got, expected_world,
        "world_aabb under pure translation must equal the local union shifted by body_pos"
    );
    println!(
        "[compound] world_aabb(translate +100x, identity rot) = [{:?}, {:?}]",
        got.min, got.max
    );

    // ------------------------------------------------------------------
    // world_aabb under a *body* rotation (180deg about Z), single sphere
    // child at a local offset so the rotation actually moves it. Closed
    // form: rotate the child's local offset by hand (`rot_180_z`, the same
    // exact identity used above), add body_pos, expand by the sphere's
    // radius on every axis.
    // ------------------------------------------------------------------
    let mut rotated_compound = CompoundShape::new();
    let r_sphere = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
    let local_offset = Vec3Fix::from_int(3, 4, 0);
    rotated_compound.add_sphere(r_sphere, local_offset, QuatFix::IDENTITY);

    let body_pos2 = Vec3Fix::from_int(1, 2, 3);
    let expected_center = body_pos2 + rot_180_z(local_offset);
    let expected_rotated_aabb = AABB::new(
        expected_center - Vec3Fix::from_int(1, 1, 1),
        expected_center + Vec3Fix::from_int(1, 1, 1),
    );
    let got = rotated_compound.world_aabb(body_pos2, ROT_180_Z);
    assert_eq!(
        got, expected_rotated_aabb,
        "world_aabb under a body rotation must rotate the child's local offset first"
    );
    println!(
        "[compound] world_aabb(body rot 180deg about Z, offset child) = [{:?}, {:?}]",
        got.min, got.max
    );

    // ------------------------------------------------------------------
    // overlapping_children: AABB-overlap broad-phase pruning against two
    // disjoint children; closed form is "which of the 4 known child AABBs
    // (from compute_aabb's inputs, at identity body transform) intersects
    // the query box", checked by hand with the same `AABB` the type
    // itself exposes (`intersects`), on the *expected* boxes, not by
    // calling `overlapping_children` for the expected side.
    // ------------------------------------------------------------------
    let query = AABB::new(Vec3Fix::from_int(-20, -1, -1), Vec3Fix::from_int(-6, 5, 5));
    let mut expected_overlap: Vec<usize> = Vec::new();
    for (i, b) in [
        expected_sphere_aabb,
        expected_capsule_aabb,
        expected_box_aabb,
        expected_hull_aabb,
    ]
    .iter()
    .enumerate()
    {
        if b.intersects(&query) {
            expected_overlap.push(i);
        }
    }
    let got = compound.overlapping_children(&query, Vec3Fix::ZERO, QuatFix::IDENTITY);
    assert_eq!(
        got, expected_overlap,
        "overlapping_children must match the independently-checked AABB::intersects set"
    );
    println!(
        "[compound] overlapping_children(query=[-20,-6]x) = {got:?} (expected {expected_overlap:?})"
    );

    println!("[compound] all 8 wiring items exercised and matched hand-derived closed forms");
}
