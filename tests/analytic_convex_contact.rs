//! Oracles for convex narrow-phase contacts: `collider::contact` (GJK followed by
//! EPA) and the bodies of a `PhysicsWorld` that carry a `Shape` as their collider.
//!
//! # What is measured
//!
//! The penetration depth of two overlapping convex solids is the length of the
//! shortest translation that separates them; the contact normal (this crate's
//! contract: from B to A) is its direction. For two **boxes** that translation is
//! the minimum over the 15 separating axes of the separating-axis test, which is
//! written out below from the boxes' actual rotated axes and compared with the
//! GJK/EPA answer. For shapes with flat faces facing each other (stacked boxes,
//! coaxial cylinders) the answer is simply the overlap along the axis.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{contact, Contact, Sphere};
use alice_physics::cylinder::Cylinder;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn rot(angle: f64, axis: [f64; 3]) -> QuatFix {
    QuatFix::from_axis_angle(v3(axis[0], axis[1], axis[2]), fx(angle))
}

fn axes(q: QuatFix) -> [[f64; 3]; 3] {
    [
        arr(q.rotate_vec(v3(1.0, 0.0, 0.0))),
        arr(q.rotate_vec(v3(0.0, 1.0, 0.0))),
        arr(q.rotate_vec(v3(0.0, 0.0, 1.0))),
    ]
}

/// The separating-axis answer for two oriented boxes: `(depth, normal B→A)`, or
/// `None` when an axis separates them. Written from the boxes' rotated axes.
fn sat(a: &OrientedBox, b: &OrientedBox) -> Option<(f64, [f64; 3])> {
    let (ax, bx) = (axes(a.rotation), axes(b.rotation));
    let (ha, hb) = (arr(a.half_extents), arr(b.half_extents));
    let t = [
        (a.center.x - b.center.x).to_f64(),
        (a.center.y - b.center.y).to_f64(),
        (a.center.z - b.center.z).to_f64(),
    ];
    let mut candidates: Vec<[f64; 3]> = Vec::new();
    candidates.extend(ax);
    candidates.extend(bx);
    for u in &ax {
        for w in &bx {
            let c = cross(*u, *w);
            if norm(c) > 1e-9 {
                candidates.push(c);
            }
        }
    }
    let mut best: Option<(f64, [f64; 3])> = None;
    for axis in candidates {
        let l = norm(axis);
        let n = [axis[0] / l, axis[1] / l, axis[2] / l];
        let ra: f64 = (0..3).map(|k| ha[k] * dot(ax[k], n).abs()).sum();
        let rb: f64 = (0..3).map(|k| hb[k] * dot(bx[k], n).abs()).sum();
        let sep = dot(t, n);
        let overlap = ra + rb - sep.abs();
        if overlap <= 0.0 {
            return None;
        }
        let dir = if sep >= 0.0 { n } else { [-n[0], -n[1], -n[2]] };
        if best.is_none_or(|(d, _)| overlap < d) {
            best = Some((overlap, dir));
        }
    }
    best
}

fn assert_contact(hit: &Contact, depth: f64, normal: [f64; 3], tol: f64, what: &str) {
    assert!(
        (hit.depth.to_f64() - depth).abs() <= tol,
        "{what}: depth {} but the closed form says {depth}",
        hit.depth.to_f64()
    );
    let n = arr(hit.normal);
    assert!(
        (0..3).all(|k| (n[k] - normal[k]).abs() <= tol.max(1e-6)),
        "{what}: normal {n:?} but the closed form says {normal:?}"
    );
}

fn boxx(c: [f64; 3], h: [f64; 3], q: QuatFix) -> OrientedBox {
    OrientedBox::new(v3(c[0], c[1], c[2]), v3(h[0], h[1], h[2]), q)
}

// ---------------------------------------------------------------------------
// collider::contact
// ---------------------------------------------------------------------------

/// Two unit boxes stacked with 0.2 of overlap: depth 0.2, normal +y (from B to A).
#[test]
fn stacked_boxes_overlap_by_the_gap_along_the_stacking_axis() {
    let a = boxx([0.0, 1.8, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    let b = boxx([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    let hit = contact(&a, &b).expect("they overlap");
    assert_contact(&hit, 0.2, [0.0, 1.0, 0.0], 1e-9, "stacked");
    // Swapping the arguments flips the normal and keeps the depth.
    let flipped = contact(&b, &a).expect("they overlap");
    assert_contact(&flipped, 0.2, [0.0, -1.0, 0.0], 1e-9, "swapped");
}

/// Offset in `x` as well: the shortest way out is still `y` (0.3 < 1.5).
#[test]
fn the_shortest_translation_wins() {
    let a = boxx([0.5, 1.7, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    let b = boxx([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    let hit = contact(&a, &b).expect("they overlap");
    assert_contact(&hit, 0.3, [0.0, 1.0, 0.0], 1e-9, "offset boxes");
}

/// Rotated boxes against the separating-axis test, over a spread of rotations and
/// offsets: the GJK/EPA depth is the SAT minimum, and the normal is its axis.
#[test]
fn rotated_boxes_agree_with_the_separating_axis_test() {
    let cases = [
        (rot(0.6, [0.0, 0.0, 1.0]), [1.1, 0.4, 0.2]),
        (rot(0.6, [0.0, 0.0, 1.0]), [0.3, 1.5, -0.1]),
        (rot(1.1, [1.0, 0.0, 0.0]), [0.2, 0.3, 1.4]),
        (rot(0.8, [1.0, 1.0, 0.0]), [1.0, 0.0, 0.6]),
        (rot(0.4, [0.0, 1.0, 1.0]), [-0.9, 0.7, 0.5]),
    ];
    for (i, (q, offset)) in cases.iter().enumerate() {
        let a = boxx(*offset, [1.0, 0.7, 0.5], *q);
        let b = boxx([0.0, 0.0, 0.0], [0.8, 0.6, 1.0], rot(0.3, [0.0, 1.0, 0.0]));
        let want = sat(&a, &b).unwrap_or_else(|| panic!("case {i}: SAT separates them"));
        let hit = contact(&a, &b).unwrap_or_else(|| panic!("case {i}: no contact"));
        assert_contact(&hit, want.0, want.1, 2e-4, &format!("rotated case {i}"));
    }
}

/// Boxes that an axis separates have no contact.
#[test]
fn separated_shapes_have_no_contact() {
    let a = boxx([0.0, 2.5, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    let b = boxx([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    assert!(contact(&a, &b).is_none());
    // Far apart along a diagonal, rotated: SAT separates, so does GJK.
    let c = boxx([5.0, 5.0, 0.0], [1.0, 1.0, 1.0], rot(0.7, [0.0, 0.0, 1.0]));
    assert!(sat(&c, &b).is_none());
    assert!(contact(&c, &b).is_none());
}

/// Two coaxial cylinders (radius 1, half-height 1) 1.7 apart overlap by 0.3 along
/// the axis; the flat caps face each other, so the answer is exact.
#[test]
fn coaxial_cylinders_overlap_by_the_gap() {
    let a = Cylinder::new(v3(0.0, 1.7, 0.0), fx(1.0), fx(1.0));
    let b = Cylinder::new(v3(0.0, 0.0, 0.0), fx(1.0), fx(1.0));
    let hit = contact(&a, &b).expect("they overlap");
    assert_contact(&hit, 0.3, [0.0, 1.0, 0.0], 1e-6, "cylinders");
}

/// Two spheres: depth `r₁ + r₂ − d` along the line of centres. EPA approximates a
/// smooth Minkowski sum by a polytope, so the tolerance is the polytope's.
#[test]
fn spheres_overlap_by_the_sum_of_radii_minus_the_distance() {
    let a = Sphere::new(v3(1.5, 0.0, 0.0), fx(1.0));
    let b = Sphere::new(v3(0.0, 0.0, 0.0), fx(1.0));
    let hit = contact(&a, &b).expect("they overlap");
    assert_contact(&hit, 0.5, [1.0, 0.0, 0.0], 5e-3, "spheres");
}

/// Identical boxes at the same place: the shortest way out is a full side (2).
#[test]
fn coincident_boxes_separate_by_the_shortest_side() {
    let a = boxx([0.0, 0.0, 0.0], [1.0, 1.5, 2.0], QuatFix::IDENTITY);
    let hit = contact(&a, &a).expect("a body overlaps itself");
    assert!(
        (hit.depth.to_f64() - 2.0).abs() < 1e-9,
        "depth {}",
        hit.depth.to_f64()
    );
    let n = arr(hit.normal);
    assert!(
        (n[0].abs() - 1.0).abs() < 1e-9 && n[1].abs() < 1e-9 && n[2].abs() < 1e-9,
        "the shortest side is x: normal {n:?}"
    );
}

/// A box with a zero extent is a flat sheet; asking for its contact must not
/// panic, and anything reported is a sane (non-negative) depth.
#[test]
fn a_flat_box_does_not_panic() {
    let flat = boxx([0.0, 0.5, 0.0], [1.0, 0.0, 1.0], QuatFix::IDENTITY);
    let b = boxx([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    if let Some(hit) = contact(&flat, &b) {
        assert!(hit.depth.to_f64() >= 0.0);
    }
    let point = boxx([0.0, 0.5, 0.0], [0.0, 0.0, 0.0], QuatFix::IDENTITY);
    if let Some(hit) = contact(&point, &b) {
        assert!(hit.depth.to_f64() >= 0.0);
    }
}

// ---------------------------------------------------------------------------
// Bodies that carry a shape
// ---------------------------------------------------------------------------

/// One substep per step, zero gravity: the step's movement is the contact
/// correction alone (see `analytic_static_collider.rs`).
fn config() -> SolverConfig {
    SolverConfig {
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    }
}

fn unit_box() -> Shape {
    Shape::Box {
        half_extents: v3(1.0, 1.0, 1.0),
    }
}

/// Two equal boxes overlapping by 0.2: each is moved half the depth along the
/// normal (equal masses).
#[test]
fn two_shaped_boxes_are_separated_half_the_depth_each() {
    let mut w = PhysicsWorld::new(config());
    let a = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 1.8, 0.0))
        .expect("valid");
    let b = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    w.step(fx(1.0 / 60.0));
    let pa = arr(w.get_body(a).expect("a").position);
    let pb = arr(w.get_body(b).expect("b").position);
    assert!((pa[1] - 1.9).abs() < 1e-9, "a ends at y = {}", pa[1]);
    assert!((pb[1] + 0.1).abs() < 1e-9, "b ends at y = {}", pb[1]);
    assert!(pa[0].abs() < 1e-9 && pa[2].abs() < 1e-9 && pb[0].abs() < 1e-9 && pb[2].abs() < 1e-9);
}

/// Boxes whose bounding spheres overlap but whose faces do not: the sphere test
/// would collide them; the shape does not. A at (2.2, 2.2, 0): the gap along x is
/// 0.2, the centres are 3.11 apart against bounding radii summing to 3.46.
#[test]
fn boxes_that_only_their_bounding_spheres_overlap_do_not_collide() {
    let mut w = PhysicsWorld::new(config());
    let a = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(2.2, 2.2, 0.0))
        .expect("valid");
    let b = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    w.step(fx(1.0 / 60.0));
    assert_eq!(arr(w.get_body(a).expect("a").position), [2.2, 2.2, 0.0]);
    assert_eq!(arr(w.get_body(b).expect("b").position), [0.0, 0.0, 0.0]);
    assert!(w.contact_events().is_empty());
}

/// A shaped box over a static box: the static one does not move, the dynamic one is
/// lifted by the full depth.
#[test]
fn a_box_on_a_static_box_is_lifted_by_the_whole_depth() {
    let mut w = PhysicsWorld::new(config());
    let floor = w.add_body(RigidBody::new_static(v3(0.0, 0.0, 0.0)));
    assert!(w.set_body_shape(floor, &unit_box()));
    let top = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 1.7, 0.0))
        .expect("valid");
    w.step(fx(1.0 / 60.0));
    assert_eq!(
        arr(w.get_body(floor).expect("floor").position),
        [0.0, 0.0, 0.0]
    );
    let p = arr(w.get_body(top).expect("top").position);
    assert!((p[1] - 2.0).abs() < 1e-9, "the box ends at y = {}", p[1]);
}

/// `set_body_shape` refuses an index that is not a body.
#[test]
fn a_shape_cannot_be_attached_to_a_body_that_does_not_exist() {
    let mut w = PhysicsWorld::new(config());
    assert!(!w.set_body_shape(0, &unit_box()));
    w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    assert!(!w.set_body_shape(1, &unit_box()));
    assert!(w.set_body_shape(0, &unit_box()));
}

/// Dropped on a floor box under gravity, a box rests with its bottom on the floor's
/// top: centre at floor top + half-extent.
#[test]
fn a_box_dropped_on_a_floor_box_rests_on_its_top_face() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let floor = w.add_body(RigidBody::new_static(v3(0.0, 0.0, 0.0)));
    assert!(w.set_body_shape(floor, &unit_box()));
    let top = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 4.0, 0.0))
        .expect("valid");
    for _ in 0..400 {
        w.step(fx(1.0 / 60.0));
    }
    let p = arr(w.get_body(top).expect("top").position);
    assert!((p[1] - 2.0).abs() < 1e-3, "rests at y = {}", p[1]);
    assert!(
        p[0].abs() < 1e-6 && p[2].abs() < 1e-6,
        "drifted sideways: {p:?}"
    );
}

/// A shaped body against a plain sphere body still collides, as spheres (the
/// shape's bounding sphere), exactly as before shapes had a narrow-phase.
#[test]
fn a_shaped_body_against_a_plain_sphere_body_collides_as_spheres() {
    let mut w = PhysicsWorld::new(config());
    let shaped = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(3.0, 0.0, 0.0), Fix128::ONE),
        fx(2.0),
    );
    w.step(fx(1.0 / 60.0));
    // Bounding radius of the unit box is √3; the sphere sum is √3 + 2 = 3.732, the
    // distance 3, so the depth is 0.732. The box has mass 8 (density 1, volume 8)
    // and the ball mass 1: each moves in inverse proportion to its mass, the box
    // `depth/9` and the ball `8·depth/9`.
    let depth = 3f64.sqrt() + 2.0 - 3.0;
    let ps = arr(w.get_body(shaped).expect("shaped").position);
    let pb = arr(w.get_body(ball).expect("ball").position);
    assert!(
        (ps[0] + depth / 9.0).abs() < 1e-9,
        "shaped ends at x = {}",
        ps[0]
    );
    assert!(
        (pb[0] - (3.0 + 8.0 * depth / 9.0)).abs() < 1e-9,
        "ball ends at x = {}",
        pb[0]
    );
}

/// Two shaped boxes with the same centre overlap everywhere; the sphere path
/// cannot give such a pair a normal, the shapes can: the shortest way out of two
/// boxes of half-extents (1, 1.5, 2) is a full side along x, shared equally.
#[test]
fn shaped_bodies_with_coincident_centres_are_separated() {
    let shape = Shape::Box {
        half_extents: v3(1.0, 1.5, 2.0),
    };
    let mut w = PhysicsWorld::new(config());
    let a = w
        .add_shaped_body(&shape, fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    let b = w
        .add_shaped_body(&shape, fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    w.step(fx(1.0 / 60.0));
    let pa = arr(w.get_body(a).expect("a").position);
    let pb = arr(w.get_body(b).expect("b").position);
    assert!((pa[0].abs() - 1.0).abs() < 1e-9, "a moved {pa:?}");
    assert!(
        (pa[0] + pb[0]).abs() < 1e-9,
        "equal masses move oppositely: {pa:?} {pb:?}"
    );
    assert!(pa[1].abs() < 1e-9 && pa[2].abs() < 1e-9);
}

/// A cone and a wedge rest their flat base on a box's top at the height their
/// **centre of mass** puts it: the body's position is the centre of mass, not the
/// geometric centre. Cone (r = 1, half-height 2): the base is `half_height / 2 =
/// 1` below the centre of mass. Wedge (h = 3): the base is `h / 3 = 1` below it.
#[test]
fn a_cone_and_a_wedge_sit_on_the_floor_by_their_centre_of_mass() {
    let shapes = [
        Shape::Cone {
            radius: fx(1.0),
            half_height: fx(2.0),
        },
        Shape::Wedge {
            width: fx(2.0),
            height: fx(3.0),
            depth: fx(2.0),
        },
    ];
    for shape in shapes {
        let mut w = PhysicsWorld::new(config());
        let floor = w.add_body(RigidBody::new_static(v3(0.0, 0.0, 0.0)));
        assert!(w.set_body_shape(floor, &unit_box()));
        // The base is 1 below the centre of mass, so a centre of mass at 1.7 puts the
        // base at 0.7, 0.3 into the floor top at 1.
        let body = w
            .add_shaped_body(&shape, fx(1.0), v3(0.0, 1.7, 0.0))
            .expect("valid");
        w.step(fx(1.0 / 60.0));
        let p = arr(w.get_body(body).expect("body").position);
        assert!(
            (p[1] - 2.0).abs() < 1e-6,
            "{shape:?}: the centre of mass ends at y = {}",
            p[1]
        );
    }
}

/// Removing a body keeps the other bodies' shapes with them: after the first of
/// three bodies is removed (the last takes its slot), the moved body still collides
/// as its own — small — box.
#[test]
fn removing_a_body_keeps_the_shapes_with_their_bodies() {
    let big = Shape::Box {
        half_extents: v3(3.0, 3.0, 3.0),
    };
    let small = Shape::Box {
        half_extents: v3(0.5, 0.5, 0.5),
    };
    let mut w = PhysicsWorld::new(config());
    let first = w
        .add_shaped_body(&big, fx(1.0), v3(100.0, 0.0, 0.0))
        .expect("valid");
    let floor = w.add_body(RigidBody::new_static(v3(0.0, 0.0, 0.0)));
    assert!(w.set_body_shape(floor, &unit_box()));
    let moved = w
        .add_shaped_body(&small, fx(1.0), v3(0.0, 1.3, 0.0))
        .expect("valid");
    assert!(w.remove_body(first).is_some());
    // `moved` now sits in slot `first`. The small box's bottom is at 0.8, 0.2 into the
    // floor top at 1: it is lifted by 0.2. Under the big box's shape it would be 1.7
    // into the floor, and lifted by that much.
    w.step(fx(1.0 / 60.0));
    let p = arr(w.get_body(first).expect("the moved body").position);
    assert!(
        (p[1] - 1.5).abs() < 1e-6,
        "ends at y = {}, expected 1.5",
        p[1]
    );
    let _ = (moved, floor);
}

/// Faces exactly touching: GJK says they meet, and the contact reports depth zero
/// along the shared normal (a polytope of zero thickness has nothing to expand,
/// but the tetrahedron completed from the touching triangle does). A millionth of
/// a gap either way is a miss or a depth of the same size.
#[test]
fn boxes_that_exactly_touch_report_zero_depth() {
    let b = boxx([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    let touching = boxx([0.0, 2.0, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    let hit = contact(&touching, &b).expect("touching faces still meet");
    assert_contact(&hit, 0.0, [0.0, 1.0, 0.0], 1e-9, "touching");
    let sunk = boxx([0.0, 2.0 - 1e-6, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    let hit = contact(&sunk, &b).expect("overlapping by a millionth");
    assert_contact(&hit, 1e-6, [0.0, 1.0, 0.0], 1e-9, "a millionth in");
    let apart = boxx([0.0, 2.0 + 1e-6, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    assert!(contact(&apart, &b).is_none());
}

/// In the world, touching shapes are not a contact (no event, no push): depth zero
/// is filtered, as the sphere path filters `dist == combined`.
#[test]
fn touching_shaped_bodies_raise_no_contact() {
    let mut w = PhysicsWorld::new(config());
    w.add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 2.0, 0.0))
        .expect("valid");
    w.add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    w.step(fx(1.0 / 60.0));
    assert!(w.contact_events().is_empty());
    assert_eq!(arr(w.get_body(0).expect("a").position), [0.0, 2.0, 0.0]);
}

/// The contact event of a shaped pair carries the contact's normal (this crate's
/// contract: from B to A) and the closing speed along it. A moving down at 2 onto B
/// closes at −2; the substep first moves A by `−2·dt` (positions integrate before
/// collisions are detected), so the overlap the event reports is `0.2 + 2·dt`.
#[test]
fn the_contact_event_of_a_shaped_pair_has_the_normal_and_the_closing_speed() {
    let dt = 1.0 / 60.0;
    let mut w = PhysicsWorld::new(config());
    let a = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 1.8, 0.0))
        .expect("valid");
    let b = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    w.get_body_mut(a).expect("a").velocity = v3(0.0, -2.0, 0.0);
    w.step(fx(dt));
    let events = w.contact_events();
    assert_eq!(events.len(), 1);
    let e = &events[0];
    assert_eq!((e.body_a, e.body_b), (a, b));
    assert_eq!(arr(e.normal), [0.0, 1.0, 0.0]);
    assert!(
        (e.depth.to_f64() - (0.2 + 2.0 * dt)).abs() < 1e-6,
        "depth {} expected {}",
        e.depth.to_f64(),
        0.2 + 2.0 * dt
    );
    assert!(
        (e.relative_velocity.to_f64() + 2.0).abs() < 1e-9,
        "closing {}",
        e.relative_velocity.to_f64()
    );
}

/// A body's rotation turns its collider: a box turned 45° about z has a vertex
/// pointing down, `√2` below its centre, so over a floor box whose top is at 1 and
/// with its centre of mass at `1 + √2 − 0.3` it is lifted to `1 + √2`.
#[test]
fn a_rotated_shaped_body_collides_as_the_rotated_shape() {
    let mut w = PhysicsWorld::new(config());
    let floor = w.add_body(RigidBody::new_static(v3(0.0, 0.0, 0.0)));
    assert!(w.set_body_shape(floor, &unit_box()));
    let rest = 1.0 + 2f64.sqrt();
    let body = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, rest - 0.3, 0.0))
        .expect("valid");
    w.get_body_mut(body).expect("body").rotation =
        QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), Fix128::HALF_PI.half());
    w.step(fx(1.0 / 60.0));
    let p = arr(w.get_body(body).expect("body").position);
    assert!(
        (p[1] - rest).abs() < 1e-6,
        "ends at y = {}, expected {rest}",
        p[1]
    );
}

/// Boxes touching only along an edge, or at a corner, meet at a single point of the
/// Minkowski difference — the origin is its first support point and GJK stops on a
/// one-point simplex. EPA needs a tetrahedron, so the contact completes it; the
/// answer is depth zero along an axis-aligned unit normal (which of the touching
/// faces' normals is not determined by the geometry).
#[test]
fn boxes_touching_at_an_edge_or_a_corner_report_zero_depth() {
    let b = boxx([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], QuatFix::IDENTITY);
    for (name, at) in [("edge", [2.0, 2.0, 0.0]), ("corner", [2.0, 2.0, 2.0])] {
        let a = boxx(at, [1.0, 1.0, 1.0], QuatFix::IDENTITY);
        let hit = contact(&a, &b).unwrap_or_else(|| panic!("{name}: touching shapes still meet"));
        assert!(
            hit.depth.to_f64().abs() < 1e-9,
            "{name}: depth {}",
            hit.depth.to_f64()
        );
        let n = arr(hit.normal);
        let axis_aligned = n.iter().filter(|v| v.abs() > 1e-9).count() == 1
            && n.iter().any(|v| (v.abs() - 1.0).abs() < 1e-9);
        assert!(
            axis_aligned,
            "{name}: normal {n:?} is not an axis-aligned unit vector"
        );
        // Pointing away from B: the touching point of A is on the +x/+y/+z side.
        assert!(
            n.iter().all(|v| *v >= -1e-9),
            "{name}: normal {n:?} points into B"
        );
    }
}

/// `colliders_overlap` decides by the shapes when both bodies have one, by the
/// collision spheres otherwise, and says `false` for what is not a collider.
#[test]
fn colliders_overlap_asks_the_shapes_when_it_can_and_the_spheres_when_it_cannot() {
    let mut w = PhysicsWorld::new(config());
    let a = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(2.2, 2.2, 0.0))
        .expect("valid");
    let b = w
        .add_shaped_body(&unit_box(), fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    // The bounding spheres (radius √3 each, 3.11 apart) overlap; the boxes do not.
    assert!(!w.colliders_overlap(a, b) && !w.colliders_overlap(b, a));
    w.get_body_mut(a).expect("a").position = v3(1.8, 1.8, 0.0);
    assert!(w.colliders_overlap(a, b) && w.colliders_overlap(b, a));
    // A sphere-only body: the sphere test, from its radius and the shaped body's
    // bounding radius (1 + √3 = 2.732 reach).
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(2.5, 0.0, 0.0), Fix128::ONE),
        fx(1.0),
    );
    assert!(w.colliders_overlap(ball, b), "2.5 < 1 + √3");
    w.get_body_mut(ball).expect("ball").position = v3(2.8, 0.0, 0.0);
    assert!(!w.colliders_overlap(ball, b), "2.8 > 1 + √3");
    // No collider, or no body: false.
    let bare = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    assert!(!w.colliders_overlap(bare, b));
    assert!(!w.colliders_overlap(b, 99) && !w.colliders_overlap(99, 98));
}
