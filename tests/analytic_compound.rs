//! Oracles for compound shapes: the geometry of their children, their mass
//! properties, the principal axes of an inertia tensor, and the bodies a
//! `PhysicsWorld` builds from them.
//!
//! # What is measured
//!
//! - **Geometry**: where each child's bounding box is, for a body at a given pose,
//!   from the child's own offset and the two rotations — written from the geometry
//!   (a sphere at `(1,0,0)` in a child turned a quarter turn is at `(0,1,0)`).
//! - **Mass**: a compound's mass, centre of mass and inertia about it are the sums
//!   over its children with the parallel-axis theorem; checked against closed forms
//!   for boxes and against a quadrature of the union.
//! - **Principal axes**: the eigen-decomposition of a symmetric tensor `R diag(λ) Rᵀ`
//!   is recovered, up to the order and sign of the axes.
//! - **Collision**: a compound is *not* its convex hull. A probe between two
//!   separated children touches nothing, though it lies inside the hull.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]
// Matrix oracles index rows and columns on purpose: `t[a][b]` reads as the maths does.
#![allow(clippy::needless_range_loop)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{Capsule, ConvexHull, Sphere, Support, AABB};
use alice_physics::compound::{CompoundShape, TransformedCompound};
use alice_physics::mass_properties::{principal_axes, MassProperties};
use alice_physics::math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};
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

fn rot(angle: f64, axis: [f64; 3]) -> QuatFix {
    QuatFix::from_axis_angle(v3(axis[0], axis[1], axis[2]), fx(angle))
}

fn quarter_z() -> QuatFix {
    QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), Fix128::HALF_PI)
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{what}: got {got:.12e}, expected {want:.12e}"
    );
}

fn assert_vec(got: [f64; 3], want: [f64; 3], tol: f64, what: &str) {
    for k in 0..3 {
        assert!(
            (got[k] - want[k]).abs() <= tol,
            "{what}: axis {k} is {} but the closed form says {}",
            got[k],
            want[k]
        );
    }
}

fn aabb_arr(a: &AABB) -> ([f64; 3], [f64; 3]) {
    (arr(a.min), arr(a.max))
}

fn unit_box_at(c: [f64; 3]) -> OrientedBox {
    OrientedBox::new(v3(c[0], c[1], c[2]), v3(1.0, 1.0, 1.0), QuatFix::IDENTITY)
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// A sphere child: the sphere's own centre offset is turned by the **child's**
/// rotation (as every other child type does), then placed at the child's position.
/// A sphere at `(1, 0, 0)` in a child at `(10, 0, 0)` turned a quarter turn about z
/// is at `(10, 1, 0)`.
#[test]
fn a_sphere_child_aabb_follows_the_child_rotation() {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(v3(1.0, 0.0, 0.0), fx(0.5)),
        v3(10.0, 0.0, 0.0),
        quarter_z(),
    );
    let (lo, hi) = aabb_arr(&c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY));
    assert_vec(lo, [9.5, 0.5, -0.5], 1e-9, "sphere child min");
    assert_vec(hi, [10.5, 1.5, 0.5], 1e-9, "sphere child max");
}

/// Under a body pose: the body's rotation turns the child's offset too. Body at
/// `(5, 0, 0)` turned a quarter turn about z, child offset `(2, 0, 0)`: world
/// `(5, 2, 0)`.
#[test]
fn a_child_aabb_follows_the_body_pose() {
    let mut c = CompoundShape::new();
    c.add_box(unit_box_at([0.0; 3]), v3(2.0, 0.0, 0.0), QuatFix::IDENTITY);
    let (lo, hi) = aabb_arr(&c.child_world_aabb(0, v3(5.0, 0.0, 0.0), quarter_z()));
    assert_vec(lo, [4.0, 1.0, -1.0], 1e-9, "box child min");
    assert_vec(hi, [6.0, 3.0, 1.0], 1e-9, "box child max");
}

/// A box child turned 45° about z reaches `√2` along x and y.
#[test]
fn a_rotated_box_child_has_the_diagonal_extent() {
    let mut c = CompoundShape::new();
    c.add_box(
        unit_box_at([0.0; 3]),
        Vec3Fix::ZERO,
        rot(std::f64::consts::FRAC_PI_4, [0.0, 0.0, 1.0]),
    );
    let (lo, hi) = aabb_arr(&c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY));
    let e = 2f64.sqrt();
    assert_vec(lo, [-e, -e, -1.0], 1e-6, "rotated box min");
    assert_vec(hi, [e, e, 1.0], 1e-6, "rotated box max");
}

/// Capsule and hull children: the capsule's endpoints and radius, the hull's vertices.
#[test]
fn capsule_and_hull_children_have_their_extents() {
    let mut c = CompoundShape::new();
    c.add_capsule(
        Capsule::new(v3(0.0, -1.0, 0.0), v3(0.0, 1.0, 0.0), fx(0.5)),
        v3(3.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    let hull = ConvexHull::new(vec![
        v3(0.0, 0.0, 0.0),
        v3(1.0, 0.0, 0.0),
        v3(0.0, 2.0, 0.0),
        v3(0.0, 0.0, 3.0),
    ]);
    c.add_convex_hull(hull, v3(0.0, 0.0, -10.0), QuatFix::IDENTITY);
    let (lo, hi) = aabb_arr(&c.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY));
    assert_vec(lo, [2.5, -1.5, -0.5], 1e-9, "capsule min");
    assert_vec(hi, [3.5, 1.5, 0.5], 1e-9, "capsule max");
    let (lo, hi) = aabb_arr(&c.child_world_aabb(1, Vec3Fix::ZERO, QuatFix::IDENTITY));
    assert_vec(lo, [0.0, 0.0, -10.0], 1e-9, "hull min");
    assert_vec(hi, [1.0, 2.0, -7.0], 1e-9, "hull max");
}

/// The union over the children; an empty compound is a point at the body.
#[test]
fn the_world_and_local_boxes_are_the_union_of_the_children() {
    let mut c = CompoundShape::new();
    assert!(c.is_empty());
    let (lo, hi) = aabb_arr(&c.world_aabb(v3(4.0, 5.0, 6.0), QuatFix::IDENTITY));
    assert_eq!((lo, hi), ([4.0, 5.0, 6.0], [4.0, 5.0, 6.0]));
    let (lo, hi) = aabb_arr(&c.compute_aabb());
    assert_eq!((lo, hi), ([0.0; 3], [0.0; 3]));
    c.add_box(unit_box_at([0.0; 3]), v3(-3.0, 0.0, 0.0), QuatFix::IDENTITY);
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, fx(2.0)),
        v3(5.0, 1.0, 0.0),
        QuatFix::IDENTITY,
    );
    assert_eq!(c.len(), 2);
    let (lo, hi) = aabb_arr(&c.world_aabb(Vec3Fix::ZERO, QuatFix::IDENTITY));
    assert_vec(lo, [-4.0, -1.0, -2.0], 1e-9, "union min");
    assert_vec(hi, [7.0, 3.0, 2.0], 1e-9, "union max");
    let (lo, hi) = aabb_arr(&c.compute_aabb());
    assert_vec(lo, [-4.0, -1.0, -2.0], 1e-9, "local min");
    assert_vec(hi, [7.0, 3.0, 2.0], 1e-9, "local max");
}

/// Only the children whose box meets the target are returned, as indices.
#[test]
fn overlapping_children_returns_the_children_the_target_touches() {
    let mut c = CompoundShape::new();
    c.add_box(unit_box_at([0.0; 3]), v3(-5.0, 0.0, 0.0), QuatFix::IDENTITY);
    c.add_box(unit_box_at([0.0; 3]), v3(0.0, 0.0, 0.0), QuatFix::IDENTITY);
    c.add_box(unit_box_at([0.0; 3]), v3(5.0, 0.0, 0.0), QuatFix::IDENTITY);
    let target = AABB::new(v3(-6.5, -0.5, -0.5), v3(-3.5, 0.5, 0.5));
    assert_eq!(
        c.overlapping_children(&target, Vec3Fix::ZERO, QuatFix::IDENTITY),
        vec![0]
    );
    let wide = AABB::new(v3(-6.5, -0.5, -0.5), v3(0.5, 0.5, 0.5));
    assert_eq!(
        c.overlapping_children(&wide, Vec3Fix::ZERO, QuatFix::IDENTITY),
        vec![0, 1]
    );
    let none = AABB::new(v3(20.0, 20.0, 20.0), v3(21.0, 21.0, 21.0));
    assert!(c
        .overlapping_children(&none, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .is_empty());
    // Following the body: moved by +5 in x, the third child (now at x = 10)... and the
    // second (now at 5) are where the target is.
    let at_five = AABB::new(v3(4.5, -0.5, -0.5), v3(5.5, 0.5, 0.5));
    assert_eq!(
        c.overlapping_children(&at_five, v3(5.0, 0.0, 0.0), QuatFix::IDENTITY),
        vec![1]
    );
}

/// The support point is the farthest point of the union along the direction, wherever
/// the compound is — including far from the origin, where the farthest point's dot
/// product with the direction is a large negative number.
#[test]
fn the_support_point_is_the_farthest_point_of_the_union_anywhere() {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, fx(1.0)),
        v3(0.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, fx(1.0)),
        v3(10.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    for body in [[0.0, 0.0, 0.0], [-2.0e6, 3.0, 1.0], [5.0e6, -2.0e6, 0.0]] {
        let p = c.support_world(
            Vec3Fix::UNIT_X,
            v3(body[0], body[1], body[2]),
            QuatFix::IDENTITY,
        );
        // +x: the far sphere, whose rightmost point is `body.x + 11`.
        assert_close(
            p.x.to_f64(),
            body[0] + 11.0,
            1e-9,
            &format!("support at {body:?}"),
        );
        let q = c.support_world(
            -Vec3Fix::UNIT_X,
            v3(body[0], body[1], body[2]),
            QuatFix::IDENTITY,
        );
        assert_close(
            q.x.to_f64(),
            body[0] - 1.0,
            1e-9,
            &format!("support (-x) at {body:?}"),
        );
    }
    // Through the `Support` wrapper as well.
    let wrapped = TransformedCompound {
        compound: &c,
        position: v3(-2.0e6, 0.0, 0.0),
        rotation: QuatFix::IDENTITY,
    };
    assert_close(
        wrapped.support(Vec3Fix::UNIT_X).x.to_f64(),
        -2.0e6 + 11.0,
        1e-9,
        "wrapper",
    );
    // An empty compound supports at the body.
    let empty = CompoundShape::new();
    assert_eq!(
        arr(empty.support_world(Vec3Fix::UNIT_Y, v3(1.0, 2.0, 3.0), QuatFix::IDENTITY)),
        [1.0, 2.0, 3.0]
    );
}

// ---------------------------------------------------------------------------
// Mass properties
// ---------------------------------------------------------------------------

fn tensor(p: &MassProperties) -> [[f64; 3]; 3] {
    let m = p.inertia_tensor;
    let c = [arr(m.col0), arr(m.col1), arr(m.col2)];
    [
        [c[0][0], c[1][0], c[2][0]],
        [c[0][1], c[1][1], c[2][1]],
        [c[0][2], c[1][2], c[2][2]],
    ]
}

/// Two unit-half-extent boxes at `x = ∓2`, density 1.5: mass `2·12`, COM at the
/// origin, and by the parallel-axis theorem `Ixx = 2·(2m/3)`, `Iyy = Izz = 2·(2m/3 + 4m)`.
#[test]
fn two_boxes_have_the_parallel_axis_inertia() {
    let rho = 1.5;
    let mut c = CompoundShape::new();
    c.add_box(unit_box_at([0.0; 3]), v3(-2.0, 0.0, 0.0), QuatFix::IDENTITY);
    c.add_box(unit_box_at([0.0; 3]), v3(2.0, 0.0, 0.0), QuatFix::IDENTITY);
    let p = c.mass_properties(fx(rho));
    let m = rho * 8.0;
    assert_close(p.mass.to_f64(), 2.0 * m, 1e-11, "mass");
    assert_vec(arr(p.center_of_mass), [0.0; 3], 1e-9, "centre of mass");
    let t = tensor(&p);
    assert_close(t[0][0], 2.0 * (2.0 * m / 3.0), 1e-10, "Ixx");
    assert_close(t[1][1], 2.0 * (2.0 * m / 3.0 + 4.0 * m), 1e-10, "Iyy");
    assert_close(t[2][2], 2.0 * (2.0 * m / 3.0 + 4.0 * m), 1e-10, "Izz");
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        assert!(
            t[a][b].abs() < 1e-9 && t[b][a].abs() < 1e-9,
            "product of inertia ({a},{b})"
        );
    }
}

/// The centre of mass is mass-weighted: a box (mass 8) at the origin and a second
/// of twice the mass... here, a sphere child changes the weighting.
#[test]
fn the_centre_of_mass_is_mass_weighted() {
    let mut c = CompoundShape::new();
    c.add_box(unit_box_at([0.0; 3]), v3(0.0, 0.0, 0.0), QuatFix::IDENTITY); // mass 8
    c.add_box(
        OrientedBox::new(Vec3Fix::ZERO, v3(2.0, 1.0, 1.0), QuatFix::IDENTITY),
        v3(6.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    ); // mass 16
    let p = c.mass_properties(fx(1.0));
    assert_close(p.mass.to_f64(), 24.0, 1e-11, "mass");
    assert_vec(
        arr(p.center_of_mass),
        [4.0, 0.0, 0.0],
        1e-9,
        "centre of mass",
    );
}

/// A compound of a turned box, a sphere and a capsule, with no symmetry: mass,
/// centre of mass and the full tensor against a quadrature of the union.
#[test]
fn a_mixed_compound_matches_a_quadrature_of_its_union() {
    let mut c = CompoundShape::new();
    let q = rot(0.5, [0.0, 0.0, 1.0]);
    c.add_box(
        OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 0.5, 0.75), QuatFix::IDENTITY),
        v3(-2.0, 0.0, 0.0),
        q,
    );
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, fx(0.8)),
        v3(0.5, 2.0, 0.5),
        QuatFix::IDENTITY,
    );
    c.add_capsule(
        Capsule::new(v3(0.0, -1.0, 0.0), v3(0.0, 1.0, 0.0), fx(0.5)),
        v3(2.5, -1.5, -0.5),
        QuatFix::IDENTITY,
    );
    let p = c.mass_properties(fx(2.0));

    // Quadrature of the same union at unit density, from the geometry.
    let (cq, sq) = (0.5f64.cos(), 0.5f64.sin());
    let inside = move |x: f64, y: f64, z: f64| {
        // child 0: a box turned by `q` about z, at (-2, 0, 0)
        let (dx, dy, dz) = (x + 2.0, y, z);
        let (lx, ly) = (cq * dx + sq * dy, -sq * dx + cq * dy);
        let in_box = lx.abs() <= 1.0 && ly.abs() <= 0.5 && dz.abs() <= 0.75;
        // child 1: a sphere at (0.5, 2, 0.5)
        let in_sphere = (x - 0.5).powi(2) + (y - 2.0).powi(2) + (z - 0.5).powi(2) <= 0.64;
        // child 2: a capsule along y from -1 to 1 at (2.5, -1.5, -0.5), radius 0.5
        let (cx, cy, cz) = (x - 2.5, y + 1.5, z + 0.5);
        let dyc = (cy.abs() - 1.0).max(0.0);
        let in_capsule = cx * cx + cz * cz + dyc * dyc <= 0.25;
        in_box || in_sphere || in_capsule
    };
    let (lo, hi) = ([-3.5, -3.0, -1.5], [3.5, 3.0, 1.5]);
    let n = 120usize;
    let step = [
        (hi[0] - lo[0]) / n as f64,
        (hi[1] - lo[1]) / n as f64,
        (hi[2] - lo[2]) / n as f64,
    ];
    let cell = step[0] * step[1] * step[2];
    let (mut m, mut mean, mut s) = (0.0, [0.0; 3], [[0.0; 3]; 3]);
    for i in 0..n {
        let x = lo[0] + (i as f64 + 0.5) * step[0];
        for j in 0..n {
            let y = lo[1] + (j as f64 + 0.5) * step[1];
            for k in 0..(n / 2) {
                let z = lo[2] + (k as f64 + 0.5) * step[2] * 2.0;
                if inside(x, y, z) {
                    let pt = [x, y, z];
                    let w = cell * 2.0;
                    m += w;
                    for a in 0..3 {
                        mean[a] += pt[a] * w;
                        for b in 0..3 {
                            s[a][b] += pt[a] * pt[b] * w;
                        }
                    }
                }
            }
        }
    }
    let com = [mean[0] / m, mean[1] / m, mean[2] / m];
    assert_close(p.mass.to_f64(), 2.0 * m, 0.02, "mixed mass");
    assert_vec(arr(p.center_of_mass), com, 0.03, "mixed centre of mass");
    let t = tensor(&p);
    let trace = (s[0][0] - m * com[0] * com[0])
        + (s[1][1] - m * com[1] * com[1])
        + (s[2][2] - m * com[2] * com[2]);
    for a in 0..3 {
        for b in 0..3 {
            let second = s[a][b] - m * com[a] * com[b];
            let want = 2.0 * if a == b { trace - second } else { -second };
            assert!(
                (t[a][b] - want).abs() < 0.03 * want.abs().max(2.0 * trace),
                "I[{a}][{b}] = {} against the union's {want}",
                t[a][b]
            );
        }
    }
}

/// A hull child contributes its exact mass properties, moved to where it is.
#[test]
fn a_hull_child_has_its_exact_mass_and_position() {
    let mut c = CompoundShape::new();
    let cube = ConvexHull::new(vec![
        v3(-1.0, -1.0, -1.0),
        v3(1.0, -1.0, -1.0),
        v3(-1.0, 1.0, -1.0),
        v3(1.0, 1.0, -1.0),
        v3(-1.0, -1.0, 1.0),
        v3(1.0, -1.0, 1.0),
        v3(-1.0, 1.0, 1.0),
        v3(1.0, 1.0, 1.0),
    ]);
    c.add_convex_hull(cube, v3(7.0, 0.0, 0.0), QuatFix::IDENTITY);
    let p = c.mass_properties(fx(3.0));
    assert_close(p.mass.to_f64(), 24.0, 1e-10, "hull mass");
    assert_vec(arr(p.center_of_mass), [7.0, 0.0, 0.0], 1e-9, "hull COM");
    let t = tensor(&p);
    assert_close(t[0][0], 24.0 * 8.0 / 12.0, 1e-9, "hull Ixx about its COM");
}

/// Nothing to weigh: zero properties, not a panic or a division by zero.
#[test]
fn an_empty_compound_has_zero_mass_properties() {
    let c = CompoundShape::new();
    let p = c.mass_properties(fx(1.0));
    assert_eq!(p.mass.to_f64(), 0.0);
    let mut one = CompoundShape::new();
    one.add_box(unit_box_at([0.0; 3]), Vec3Fix::ZERO, QuatFix::IDENTITY);
    assert_eq!(
        one.mass_properties(fx(0.0)).mass.to_f64(),
        0.0,
        "no density"
    );
}

// ---------------------------------------------------------------------------
// Principal axes
// ---------------------------------------------------------------------------

fn matmul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    r
}

fn mat_of(m: Mat3Fix) -> [[f64; 3]; 3] {
    let c = [arr(m.col0), arr(m.col1), arr(m.col2)];
    [
        [c[0][0], c[1][0], c[2][0]],
        [c[0][1], c[1][1], c[2][1]],
        [c[0][2], c[1][2], c[2][2]],
    ]
}

fn mat_to_fix(m: [[f64; 3]; 3]) -> Mat3Fix {
    Mat3Fix::from_cols(
        v3(m[0][0], m[1][0], m[2][0]),
        v3(m[0][1], m[1][1], m[2][1]),
        v3(m[0][2], m[1][2], m[2][2]),
    )
}

/// `R diag(1, 2, 3.5) Rᵀ` for a rotation `R` about `(1, 1, 0)`, and a tensor already
/// diagonal in a permuted order: the moments come back as a set, and the axes
/// reproduce the tensor and are orthonormal and right-handed.
#[test]
fn the_principal_axes_diagonalise_a_symmetric_tensor() {
    let q = rot(0.7, [1.0, 1.0, 0.0]);
    let r = [
        arr(q.rotate_vec(v3(1.0, 0.0, 0.0))),
        arr(q.rotate_vec(v3(0.0, 1.0, 0.0))),
        arr(q.rotate_vec(v3(0.0, 0.0, 1.0))),
    ];
    // Columns of R are the rotated basis vectors.
    let rm = [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][2], r[1][2], r[2][2]],
    ];
    let lambda = [1.0, 2.0, 3.5];
    let d = [
        [lambda[0], 0.0, 0.0],
        [0.0, lambda[1], 0.0],
        [0.0, 0.0, lambda[2]],
    ];
    let rt = [
        [rm[0][0], rm[1][0], rm[2][0]],
        [rm[0][1], rm[1][1], rm[2][1]],
        [rm[0][2], rm[1][2], rm[2][2]],
    ];
    let tensor = matmul(matmul(rm, d), rt);
    let (moments, axes) = principal_axes(mat_to_fix(tensor));
    let mut got = [moments.x.to_f64(), moments.y.to_f64(), moments.z.to_f64()];
    got.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    for k in 0..3 {
        assert_close(got[k], lambda[k], 1e-8, &format!("moment {k}"));
    }
    // R' diag(moments) R'ᵀ = tensor, R' orthonormal with det +1.
    let a = mat_of(axes);
    let at = [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ];
    let m = [
        [moments.x.to_f64(), 0.0, 0.0],
        [0.0, moments.y.to_f64(), 0.0],
        [0.0, 0.0, moments.z.to_f64()],
    ];
    let rebuilt = matmul(matmul(a, m), at);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (rebuilt[i][j] - tensor[i][j]).abs() < 1e-8,
                "rebuilt[{i}][{j}]"
            );
        }
    }
    let id = matmul(a, at);
    for (i, row) in id.iter().enumerate() {
        for (j, &got) in row.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (got - want).abs() < 1e-8,
                "axes are not orthonormal at [{i}][{j}]"
            );
        }
    }
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    assert!(
        (det - 1.0).abs() < 1e-8,
        "the axes are not right-handed: det = {det}"
    );
}

/// Already diagonal, repeated moments (a sphere's tensor): the identity frame, no
/// division by a vanishing off-diagonal.
#[test]
fn a_diagonal_tensor_is_its_own_principal_frame() {
    let t = Mat3Fix::diagonal(fx(4.0), fx(4.0), fx(9.0));
    let (moments, axes) = principal_axes(t);
    let mut got = [moments.x.to_f64(), moments.y.to_f64(), moments.z.to_f64()];
    got.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    assert_close(got[0], 4.0, 1e-12, "m0");
    assert_close(got[1], 4.0, 1e-12, "m1");
    assert_close(got[2], 9.0, 1e-12, "m2");
    let a = mat_of(axes);
    let id = matmul(
        a,
        [
            [a[0][0], a[1][0], a[2][0]],
            [a[0][1], a[1][1], a[2][1]],
            [a[0][2], a[1][2], a[2][2]],
        ],
    );
    for i in 0..3 {
        for j in 0..3 {
            assert!((id[i][j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-10);
        }
    }
    // The zero tensor: zero moments, a valid frame.
    let (z, za) = principal_axes(Mat3Fix::ZERO);
    assert_eq!(arr(z), [0.0; 3]);
    assert!((mat_of(za)[0][0] - 1.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// A compound in the world
// ---------------------------------------------------------------------------

fn config() -> SolverConfig {
    SolverConfig {
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    }
}

/// Boxes at `(−1, −1, 0)` and `(1, 1, 0)` (half-extent 1): the tensor about the COM
/// is `2m·[[1,−1,0],[−1,1,0],[0,0,2]] + (4m/3)·E` with `m = 8ρ`, whose principal
/// moments are `4m/3` (along `(1,1,0)`), `4m/3 + 4m` (along `(1,−1,0)` and `z`).
fn diagonal_pair() -> CompoundShape {
    let mut c = CompoundShape::new();
    c.add_box(
        unit_box_at([0.0; 3]),
        v3(-1.0, -1.0, 0.0),
        QuatFix::IDENTITY,
    );
    c.add_box(unit_box_at([0.0; 3]), v3(1.0, 1.0, 0.0), QuatFix::IDENTITY);
    c
}

/// The body's mass is the compound's, its principal inertia is that of the tensor
/// (checked through the world tensor `R diag(I) Rᵀ`, which does not depend on which
/// axis is called which), and it sits where it was put.
#[test]
fn a_compound_body_has_the_mass_and_inertia_of_its_children() {
    let rho = 1.0;
    let m = 8.0 * rho;
    let mut w = PhysicsWorld::new(config());
    let idx = w
        .add_compound_body(&diagonal_pair(), fx(rho), v3(10.0, 20.0, 30.0))
        .expect("a valid compound");
    let b = w.get_body(idx).expect("body");
    assert_close(b.mass().to_f64(), 2.0 * m, 1e-9, "mass");
    assert_eq!(arr(b.position), [10.0, 20.0, 30.0]);
    // World tensor = R diag(1/inv_inertia) Rᵀ with the body's rotation R.
    let q = b.rotation;
    let cols = [
        arr(q.rotate_vec(v3(1.0, 0.0, 0.0))),
        arr(q.rotate_vec(v3(0.0, 1.0, 0.0))),
        arr(q.rotate_vec(v3(0.0, 0.0, 1.0))),
    ];
    let inv = arr(b.inv_inertia);
    let mut world_tensor = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            world_tensor[i][j] = (0..3)
                .map(|k| cols[k][i] * (1.0 / inv[k]) * cols[k][j])
                .sum();
        }
    }
    let base = 4.0 * m / 3.0;
    let want = [
        [2.0 * m + base, -2.0 * m, 0.0],
        [-2.0 * m, 2.0 * m + base, 0.0],
        [0.0, 0.0, 4.0 * m + base],
    ];
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (world_tensor[i][j] - want[i][j]).abs() < 1e-6 * want[0][0],
                "world inertia [{i}][{j}] = {} expected {}",
                world_tensor[i][j],
                want[i][j]
            );
        }
    }
}

/// The compound is *not* its convex hull. Boxes at `x = −3` and `x = +5` (half-extent
/// one) have their centre of mass at `x = 1`; the body is put with that at the origin,
/// so the boxes sit at `−4` and `+4` and the gap runs from `−3` to `3`. A probe in
/// the gap, inside the hull of the pair, touches neither child. A probe on a child
/// collides and is moved out by the closed-form depth.
#[test]
fn a_compound_collides_as_its_children_not_as_its_hull() {
    let mut c = CompoundShape::new();
    c.add_box(unit_box_at([0.0; 3]), v3(-3.0, 0.0, 0.0), QuatFix::IDENTITY);
    c.add_box(unit_box_at([0.0; 3]), v3(5.0, 0.0, 0.0), QuatFix::IDENTITY);
    let probe = Shape::Box {
        half_extents: v3(1.0, 1.0, 1.0),
    };

    // In the gap, at the centre: no contact, no push.
    let mut w = PhysicsWorld::new(config());
    let body = w
        .add_compound_body(&c, fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    let gap = w
        .add_shaped_body(&probe, fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    assert!(
        !w.colliders_overlap(body, gap),
        "the gap is not part of the compound"
    );
    w.step(fx(1.0 / 60.0));
    assert_eq!(
        arr(w.get_body(gap).expect("probe").position),
        [0.0, 0.0, 0.0]
    );
    assert!(w.contact_events().is_empty());

    // On the right child (now spanning [3, 5]), a probe at 3.5 spans [2.5, 4.5]: the
    // shortest way out is to the left, by 1.5, to x = 2.
    let mut w = PhysicsWorld::new(config());
    let body = w
        .add_compound_body(&c, fx(1.0), v3(0.0, 0.0, 0.0))
        .expect("valid");
    w.get_body_mut(body).expect("body").inv_mass = Fix128::ZERO;
    let probe_body = w
        .add_shaped_body(&probe, fx(1.0), v3(3.5, 0.0, 0.0))
        .expect("valid");
    assert!(w.colliders_overlap(body, probe_body));
    w.step(fx(1.0 / 60.0));
    let p = arr(w.get_body(probe_body).expect("probe").position);
    assert!(
        (p[0] - 2.0).abs() < 1e-6,
        "pushed to x = {} instead of 2",
        p[0]
    );
}

/// A probe that meets two children at different depths is moved by the deeper one.
/// Children span `[2, 4]` and `[3.5, 5.5]` along x after centring (boxes at −2 and 0
/// relative to a COM at... here: placed so the numbers are plain).
#[test]
fn the_deepest_child_contact_decides() {
    // Child boxes at x = 3 and x = 5.5 (half-extent 1, and 1): spans [2, 4] and
    // [4.5, 6.5]. Their centre of mass is at 4.25, so the body is put at 4.25 and the
    // children keep their authored x.
    let mut c = CompoundShape::new();
    c.add_box(unit_box_at([0.0; 3]), v3(3.0, 0.0, 0.0), QuatFix::IDENTITY);
    c.add_box(unit_box_at([0.0; 3]), v3(5.5, 0.0, 0.0), QuatFix::IDENTITY);
    let mut w = PhysicsWorld::new(config());
    let body = w
        .add_compound_body(&c, fx(1.0), v3(4.25, 0.0, 0.0))
        .expect("valid");
    w.get_body_mut(body).expect("body").inv_mass = Fix128::ZERO;
    // A probe at x = 4.25 spans [3.25, 5.25]: 0.75 into the first child's right
    // side (to push right, 4 − 3.25 = 0.75; to push left: 5.25 − 2 = 3.25) and 0.75
    // into the second's left side (4.5 → 5.25 − 4.5 = 0.75; the other way 6.5 − 3.25).
    // A probe at 4.0 spans [3, 5]: 1.0 into the first from the right, 0.5 into the
    // second from the left; the deeper one (1.0, pushing right) decides.
    let probe = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(1.0, 1.0, 1.0),
            },
            fx(1.0),
            v3(4.0, 0.0, 0.0),
        )
        .expect("valid");
    w.step(fx(1.0 / 60.0));
    let p = arr(w.get_body(probe).expect("probe").position);
    assert!(
        (p[0] - 5.0).abs() < 1e-6,
        "pushed to x = {} instead of 5",
        p[0]
    );
}

/// A rotated child with an offset centre: the child's own centre is turned by the
/// child's rotation (a sphere at `(1, 0, 0)` in a child turned a quarter turn about z
/// is at `(0, 1, 0)` from the child's position), and the centre of mass follows.
#[test]
fn the_centre_of_mass_follows_a_rotated_child_with_an_offset_centre() {
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(v3(1.0, 0.0, 0.0), fx(1.0)),
        v3(10.0, 0.0, 0.0),
        quarter_z(),
    );
    let p = c.mass_properties(fx(1.0));
    assert_vec(
        arr(p.center_of_mass),
        [10.0, 1.0, 0.0],
        1e-9,
        "COM of a rotated, offset sphere",
    );
}

/// Refusals: an empty compound, a non-positive density.
#[test]
fn a_compound_without_volume_or_density_is_refused() {
    use alice_physics::shape::ShapeError;
    let mut w = PhysicsWorld::new(config());
    let empty = CompoundShape::new();
    assert_eq!(
        w.add_compound_body(&empty, fx(1.0), Vec3Fix::ZERO),
        Err(ShapeError::DegenerateShape)
    );
    assert_eq!(
        w.add_compound_body(&diagonal_pair(), fx(0.0), Vec3Fix::ZERO),
        Err(ShapeError::NonPositiveDensity)
    );
    assert_eq!(
        w.add_compound_body(&diagonal_pair(), fx(-2.0), Vec3Fix::ZERO),
        Err(ShapeError::NonPositiveDensity)
    );
    assert_eq!(w.body_count(), 0, "a refused compound adds no body");
    let _ = RigidBody::new_static(Vec3Fix::ZERO);
}

/// One box child turned a quarter about z: the closed-form tensor of a box with
/// its x and y extents exchanged (a child rotation the tensor must follow).
#[test]
fn a_turned_box_child_exchanges_its_x_and_y_moments() {
    let mut c = CompoundShape::new();
    c.add_box(
        OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 0.5, 0.75), QuatFix::IDENTITY),
        Vec3Fix::ZERO,
        quarter_z(),
    );
    let p = c.mass_properties(fx(2.0));
    let m = 2.0 * 2.0 * 1.0 * 1.5;
    assert_close(p.mass.to_f64(), m, 1e-9, "turned box mass");
    let (ixx, iyy, izz) = (
        m * (1.0 + 2.25) / 12.0,
        m * (4.0 + 2.25) / 12.0,
        m * (4.0 + 1.0) / 12.0,
    );
    let t = tensor(&p);
    assert_vec(
        [t[0][0], t[1][1], t[2][2]],
        [iyy, ixx, izz],
        1e-8,
        "turned box moments",
    );
}

/// A capsule along x (a child tilted away from the solid's own y axis): the
/// closed-form cylinder plus two hemispheres, `diag(I∥, I⊥, I⊥)`.
#[test]
fn an_x_capsule_child_has_the_axial_moment_about_x() {
    let mut c = CompoundShape::new();
    c.add_capsule(
        Capsule::new(v3(-1.0, 0.0, 0.0), v3(1.0, 0.0, 0.0), fx(0.5)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let rho = 2.0;
    let p = c.mass_properties(fx(rho));
    let (r, h) = (0.5f64, 2.0f64);
    let pi = std::f64::consts::PI;
    let mc = rho * pi * r * r * h;
    let ms = rho * 4.0 / 3.0 * pi * r * r * r;
    let axial = mc * r * r / 2.0 + ms * 0.4 * r * r;
    let perp = mc * (r * r / 4.0 + h * h / 12.0)
        + ms * (0.4 * r * r - 9.0 / 64.0 * r * r + (h / 2.0 + 3.0 * r / 8.0).powi(2));
    assert_close(p.mass.to_f64(), mc + ms, 1e-9, "capsule mass");
    let t = tensor(&p);
    assert_vec(
        [t[0][0], t[1][1], t[2][2]],
        [axial, perp, perp],
        1e-8,
        "capsule moments",
    );
}

/// `overlapping_children` returns exactly the children whose boxes meet the target
/// (touching counts), for a body at a position and with a turn.
#[test]
fn the_overlapping_children_are_those_whose_boxes_meet_the_target() {
    let mut c = CompoundShape::new();
    for x in [-4.0, 0.0, 4.0] {
        c.add_box(unit_box_at([0.0; 3]), v3(x, 0.0, 0.0), QuatFix::IDENTITY);
    }
    let target =
        |lo: [f64; 3], hi: [f64; 3]| AABB::new(v3(lo[0], lo[1], lo[2]), v3(hi[0], hi[1], hi[2]));
    let at = |t: &AABB, pos: Vec3Fix, rot: QuatFix| c.overlapping_children(t, pos, rot);
    let id = QuatFix::IDENTITY;

    // The boxes are [-5,-3], [-1,1], [3,5] in x.
    assert_eq!(
        at(
            &target([2.5, -1.0, -1.0], [6.0, 1.0, 1.0]),
            Vec3Fix::ZERO,
            id
        ),
        vec![2]
    );
    assert_eq!(
        at(
            &target([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]),
            Vec3Fix::ZERO,
            id
        ),
        vec![1]
    );
    assert_eq!(
        at(
            &target([-9.0, -9.0, -9.0], [9.0, 9.0, 9.0]),
            Vec3Fix::ZERO,
            id
        ),
        vec![0, 1, 2]
    );
    assert!(at(
        &target([1.5, -1.0, -1.0], [2.5, 1.0, 1.0]),
        Vec3Fix::ZERO,
        id
    )
    .is_empty());
    assert!(at(
        &target([-1.0, 3.0, -1.0], [1.0, 4.0, 1.0]),
        Vec3Fix::ZERO,
        id
    )
    .is_empty());
    // Touching counts: the target starts exactly where child 2 ends.
    assert_eq!(
        at(
            &target([5.0, -1.0, -1.0], [6.0, 1.0, 1.0]),
            Vec3Fix::ZERO,
            id
        ),
        vec![2]
    );
    // Touching from the other side: the target ends exactly where child 0 begins.
    assert_eq!(
        at(
            &target([-6.0, -1.0, -1.0], [-5.0, 1.0, 1.0]),
            Vec3Fix::ZERO,
            id
        ),
        vec![0]
    );
    // The body at x = 4: the boxes are [-1,1], [3,5], [7,9].
    assert_eq!(
        at(
            &target([2.5, -1.0, -1.0], [5.5, 1.0, 1.0]),
            v3(4.0, 0.0, 0.0),
            id
        ),
        vec![1]
    );
    // The body turned a quarter about z: x -> y, so the boxes sit at y = -4, 0, 4.
    assert_eq!(
        at(
            &target([-1.0, 3.0, -1.0], [1.0, 5.0, 1.0]),
            Vec3Fix::ZERO,
            quarter_z()
        ),
        vec![2]
    );
}

/// A probe above a child, separated from the compound only along z, still meets
/// it: the boxes the narrow-phase prunes with have the extent on every axis.
#[test]
fn a_probe_that_overlaps_along_z_only_is_not_pruned() {
    let mut c = CompoundShape::new();
    for x in [-4.0, 4.0] {
        c.add_box(unit_box_at([0.0; 3]), v3(x, 0.0, 0.0), QuatFix::IDENTITY);
    }
    let mut world = PhysicsWorld::new(SolverConfig::default());
    let body = world
        .add_compound_body(&c, Fix128::from_int(1000), Vec3Fix::ZERO)
        .expect("two boxes have volume");
    let probe = Shape::Ellipsoid {
        radii: v3(0.5, 0.5, 0.5),
    };
    // The child at x = -4 spans z in [-1, 1]; a ball of radius 0.5 at z = 1.3
    // reaches down to 0.8, inside it. At z = 1.7 it only reaches 1.2: apart.
    let near = world
        .add_shaped_body(&probe, Fix128::from_int(1000), v3(-4.0, 0.0, 1.3))
        .expect("a valid solid");
    let far = world
        .add_shaped_body(&probe, Fix128::from_int(1000), v3(-4.0, 0.0, 1.7))
        .expect("a valid solid");
    assert!(world.colliders_overlap(body, near));
    assert!(!world.colliders_overlap(body, far));
}
