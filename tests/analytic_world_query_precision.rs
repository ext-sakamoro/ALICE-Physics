//! Oracles for the precision of world casts at the conservative-advancement
//! tolerance `2⁻³²`: a segment cast (radius `0`, `a ≠ b`) reports the closed-form
//! time of impact and a unit normal; a step or a sinking path that goes more
//! than `2⁻³²` (and less than `2⁻²⁸`) into a collider is a hit; a contact
//! exactly at `max_t` is a hit; and a path that sinks into a surface more slowly
//! than `2⁻³²` per unit of travel still hits it when it is more than `2⁻³²`
//! inside by `max_t`.
//!
//! # Expected values
//!
//! Every expected value is written from the geometry by hand; the closed form is
//! in a comment next to each assertion. Nothing here calls the code under test to
//! make an expected value.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;
use alice_physics::world_shape_query::WorldShapeHit;

/// The conservative-advancement tolerance, `2⁻³²`.
const TOL: f64 = 1.0 / 4_294_967_296.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn p3(p: [f64; 3]) -> Vec3Fix {
    v3(p[0], p[1], p[2])
}

fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig::default())
}

fn sphere_cast(
    w: &PhysicsWorld,
    c: [f64; 3],
    r: f64,
    d: [f64; 3],
    max: f64,
) -> Option<WorldShapeHit> {
    w.cast_sphere(p3(c), fx(r), p3(d), fx(max), &RayFilter::default())
}

fn capsule_cast(
    w: &PhysicsWorld,
    a: [f64; 3],
    b: [f64; 3],
    r: f64,
    d: [f64; 3],
    max: f64,
) -> Option<WorldShapeHit> {
    w.cast_capsule(p3(a), p3(b), fx(r), p3(d), fx(max), &RayFilter::default())
}

/// A segment cast: radius `0`, `a ≠ b`. A direction with a component below
/// `2⁻⁶⁴` is passed raw.
fn segment_cast(
    w: &PhysicsWorld,
    a: [f64; 3],
    b: [f64; 3],
    d: Vec3Fix,
    max: f64,
) -> Option<WorldShapeHit> {
    w.cast_capsule(
        p3(a),
        p3(b),
        Fix128::ZERO,
        d,
        fx(max),
        &RayFilter::default(),
    )
}

fn static_box(w: &mut PhysicsWorld, c: [f64; 3], h: [f64; 3]) {
    let i = w.add_body(RigidBody::new_static(p3(c)));
    assert!(w.set_body_shape(
        i,
        &Shape::Box {
            half_extents: p3(h)
        }
    ));
}

fn static_shape(w: &mut PhysicsWorld, c: [f64; 3], shape: Shape) {
    let i = w.add_body(RigidBody::new_static(p3(c)));
    assert!(w.set_body_shape(i, &shape));
}

/// A mesh of quads `[p0, p1, p2, p3]`, each as the triangles `p0 p1 p2` and
/// `p0 p2 p3`.
fn quad_mesh(quads: &[[[f64; 3]; 4]]) -> TriMesh {
    let mut v = Vec::new();
    let mut idx = Vec::new();
    for q in quads {
        let b = v.len() as u32;
        for p in q {
            v.push(p3(*p));
        }
        idx.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    TriMesh::from_indexed(&v, &idx)
}

/// The flat floors `y = 0` (top face or surface) a test runs on, `l` the half
/// width: a plane, a box, a two-triangle mesh and a flat height field.
fn floors(l: f64) -> Vec<(&'static str, PhysicsWorld)> {
    let mut out = Vec::new();
    let mut w = world();
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    out.push(("plane", w));
    let mut w = world();
    static_box(&mut w, [0.0, -1.0, 0.0], [l, 1.0, l]);
    out.push(("box", w));
    let mut w = world();
    w.add_static_collider(StaticCollider::TriMesh(quad_mesh(&[[
        [-l, 0.0, -l],
        [-l, 0.0, l],
        [l, 0.0, l],
        [l, 0.0, -l],
    ]])));
    out.push(("mesh", w));
    let mut w = world();
    let n = 5u32;
    w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        vec![Fix128::ZERO; (n * n) as usize],
        n,
        n,
        fx(2.0 * l / f64::from(n - 1)),
        v3(-l, 0.0, -l),
    )));
    out.push(("field", w));
    out
}

#[track_caller]
fn assert_unit_normal(h: &WorldShapeHit, want: [f64; 3], tol: f64, what: &str) {
    let n = f3(h.normal);
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    // unit to within 2⁻³² is the contract; it is met to rounding (2⁻⁴⁰ here)
    assert!(
        (len - 1.0).abs() <= TOL / 256.0,
        "{what}: |n| = {len} (normal {n:?}), not unit to within 2⁻⁴⁰"
    );
    for k in 0..3 {
        assert!(
            (n[k] - want[k]).abs() <= tol,
            "{what}: normal {n:?} but the closed form is {want:?}"
        );
    }
}

// ============================================================================
// Segment casts (radius 0)
// ============================================================================

/// The three colliders under the segment `x ∈ [0.5, 1.5]`, `z = 1`: a triangle in
/// `y = 0` (vertices `(0,0,0)`, `(4,0,0)`, `(0,0,4)`), a box of half-extents
/// `(3, 1, 3)` about `(1, −1, 1)` (top face `y = 0`) and an ellipsoid of radii
/// `(3, 1, 3)` about `(1, −1, 1)` (top `(1, 0, 1)`).
fn segment_targets() -> Vec<(&'static str, PhysicsWorld)> {
    let mut wm = world();
    wm.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
        &[v3(0.0, 0.0, 0.0), v3(4.0, 0.0, 0.0), v3(0.0, 0.0, 4.0)],
        &[0, 1, 2],
    )));
    let mut wb = world();
    static_box(&mut wb, [1.0, -1.0, 1.0], [3.0, 1.0, 3.0]);
    let mut we = world();
    static_shape(
        &mut we,
        [1.0, -1.0, 1.0],
        Shape::Ellipsoid {
            radii: v3(3.0, 1.0, 3.0),
        },
    );
    vec![("mesh", wm), ("box", wb), ("ellipsoid", we)]
}

#[test]
fn segment_cast_straight_down_is_exact_to_the_tolerance() {
    for (name, w) in segment_targets() {
        for y0 in [3.0, 1.0, 0.25] {
            // oracle: the segment at height y0 moving −Y reaches y = 0 (the
            // triangle, the box's top face, the ellipsoid's top at x = 1) after
            // t = y0, normal +Y.
            let h = segment_cast(
                &w,
                [0.5, y0, 1.0],
                [1.5, y0, 1.0],
                v3(0.0, -1.0, 0.0),
                100.0,
            )
            .unwrap_or_else(|| panic!("{name} y0 = {y0}: no hit"));
            let t = h.t.to_f64();
            assert!(
                (t - y0).abs() <= TOL,
                "{name} y0 = {y0}: t = {t:.15}, closed form {y0} (error {:e} > 2⁻³²)",
                t - y0
            );
            // not after the contact beyond rounding
            assert!(
                t <= y0 + 1e-12,
                "{name} y0 = {y0}: t = {t} after the contact"
            );
            assert_unit_normal(&h, [0.0, 1.0, 0.0], 1e-9, name);
        }
    }
}

#[test]
fn segment_cast_on_a_slant_is_exact_to_the_tolerance() {
    // The direction (0, −1, dz) / √(1 + dz²).
    let cases: [(f64, f64); 2] = [(0.37, 0.3), (2.0, -0.7)];
    for (name, w) in segment_targets() {
        for (y0, dz) in cases {
            let k = (1.0 + dz * dz).sqrt();
            let d = v3(0.0, -1.0 / k, dz / k);
            let h = segment_cast(&w, [0.5, y0, 1.0], [1.5, y0, 1.0], d, 100.0);
            // The segment moves by u = t / k down and u·dz along z.
            let (t_want, n_want) = match name {
                // oracle: the plane y = 0 is reached at u = y0, t = y0·k, while
                // z = 1 + dz·y0 is 1.111 (inside the triangle x + z ≤ 4 for x ≤
                // 1.5) for dz = 0.3; for dz = −0.7, z = −0.4 leaves the triangle
                // (z < 0) before y = 0 and the segment passes it.
                "mesh" if dz < 0.0 => {
                    assert!(h.is_none(), "{name} y0 = {y0} dz = {dz}: {h:?}");
                    continue;
                }
                "mesh" | "box" => (y0 * k, [0.0, 1.0, 0.0]),
                _ => {
                    // oracle: the point x = 1 of the segment meets the ellipse
                    // (y + 1)² + ((z − 1)/3)² = 1 at the smaller root of
                    // (y0 + 1 − u)² + (dz·u/3)² = 1:
                    // (1 + dz²/9)u² − 2(y0 + 1)u + (y0 + 1)² − 1 = 0.
                    let qa = 1.0 + dz * dz / 9.0;
                    let qb = -2.0 * (y0 + 1.0);
                    let qc = (y0 + 1.0) * (y0 + 1.0) - 1.0;
                    let u = (-qb - (qb * qb - 4.0 * qa * qc).sqrt()) / (2.0 * qa);
                    // normal ∝ (0, y + 1, (z − 1)/9) there
                    let (ny, nz) = (y0 + 1.0 - u, dz * u / 9.0);
                    let l = (ny * ny + nz * nz).sqrt();
                    (u * k, [0.0, ny / l, nz / l])
                }
            };
            let h = h.unwrap_or_else(|| panic!("{name} y0 = {y0} dz = {dz}: no hit"));
            let t = h.t.to_f64();
            assert!(
                (t - t_want).abs() <= TOL,
                "{name} y0 = {y0} dz = {dz}: t = {t:.15}, closed form {t_want:.15} (error {:e})",
                t - t_want
            );
            assert_unit_normal(&h, n_want, 1e-6, name);
        }
    }
}

// ============================================================================
// Steps and sinking paths between 2⁻³² and 2⁻²⁸ deep
// ============================================================================

#[test]
fn thin_step_above_the_tolerance_is_a_hit() {
    let r = 0.3;
    for k in [31, 30, 29] {
        let h = (2f64).powi(-k);
        // A box step on a plane floor: top y = h from x = 12.
        let mut wb = world();
        wb.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            Vec3Fix::UNIT_Y,
            Fix128::ZERO,
        )));
        static_box(&mut wb, [17.0, h - 0.5, 0.0], [5.0, 0.5, 5.0]);
        // The same step as a mesh: floor to x = 12, riser, top at y = h.
        let mut wm = world();
        wm.add_static_collider(StaticCollider::TriMesh(quad_mesh(&[
            [
                [-20.0, 0.0, -20.0],
                [-20.0, 0.0, 20.0],
                [12.0, 0.0, 20.0],
                [12.0, 0.0, -20.0],
            ],
            [
                [12.0, 0.0, -20.0],
                [12.0, 0.0, 20.0],
                [12.0, h, 20.0],
                [12.0, h, -20.0],
            ],
            [
                [12.0, h, -20.0],
                [12.0, h, 20.0],
                [30.0, h, 20.0],
                [30.0, h, -20.0],
            ],
        ])));
        // oracle: the sphere resting on y = 0 (centre (1 + t, r)) moving +X meets
        // the step's edge (12, h) when √((11 − t)² + (r − h)²) = r, at
        // t = 11 − √(2rh − h²); the capsule's lower sphere is the same.
        let gap = |t: f64| ((11.0 - t).powi(2) + (r - h).powi(2)).sqrt() - r;
        for (name, w) in [("box", &wb), ("mesh", &wm)] {
            for (kind, hit) in [
                (
                    "sphere",
                    sphere_cast(w, [1.0, r, 0.0], r, [1.0, 0.0, 0.0], 20.0),
                ),
                (
                    "capsule",
                    capsule_cast(
                        w,
                        [1.0, r, 0.0],
                        [1.0, r + 1.2, 0.0],
                        r,
                        [1.0, 0.0, 0.0],
                        20.0,
                    ),
                ),
            ] {
                let hit =
                    hit.unwrap_or_else(|| panic!("{name} {kind} step 2^-{k}: missed the step"));
                let t = hit.t.to_f64();
                assert!(
                    gap(t).abs() <= 2.0 * TOL && t < 11.0,
                    "{name} {kind} step 2^-{k}: t = {t}, gap there {:e} (closed form t = {})",
                    gap(t),
                    11.0 - (2.0 * r * h - h * h).sqrt()
                );
            }
        }
    }
}

#[test]
fn sphere_sinking_slowly_into_a_floor_is_a_hit() {
    let r = 0.3;
    // A slope of 2⁻³⁶ (|d·n| far below 2⁻⁶) over max_t = 100: the sphere resting
    // on y = 0 is 100·2⁻³⁶ = 6.25·2⁻³² inside the floor at max_t.
    let s = (2f64).powi(-36);
    let d = [1.0, -s, 0.0];
    for (name, w) in floors(250.0) {
        if name == "plane" {
            continue;
        }
        // oracle: it touches at the start and goes more than 2⁻³² in: t = 0.
        let h = sphere_cast(&w, [0.0, r, 0.0], r, d, 100.0)
            .unwrap_or_else(|| panic!("{name}: missed a path 6.25·2⁻³² into the floor"));
        assert_eq!(h.t.to_f64(), 0.0, "{name}");
    }
}

// ============================================================================
// A contact exactly at max_t
// ============================================================================

#[test]
fn contact_exactly_at_max_t_is_a_hit() {
    let mut w = world();
    w.add_shaped_body(
        &Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Fix128::ONE,
        Vec3Fix::ZERO,
    )
    .expect("box");
    // oracle: a sphere of 0.5 from x = −10 meets the face x = −1 at t = 8.5.
    let h = sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 8.5).expect("box at max_t");
    assert!((h.t.to_f64() - 8.5).abs() <= 1e-12, "t = {}", h.t.to_f64());

    let mut w = world();
    w.add_shaped_body(
        &Shape::Ellipsoid {
            radii: v3(2.0, 1.0, 1.0),
        },
        Fix128::ONE,
        Vec3Fix::ZERO,
    )
    .expect("ellipsoid");
    // oracle: the ellipsoid's vertex is x = −2, so the sphere meets it at
    // t = 10 − 2 − 0.5 = 7.5.
    let h =
        sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 7.5).expect("ellipsoid at max_t");
    assert!((h.t.to_f64() - 7.5).abs() <= 1e-8, "t = {}", h.t.to_f64());
}

// ============================================================================
// Paths that sink more slowly than 2⁻³² per unit
// ============================================================================

#[test]
fn path_sinking_below_parallel_threshold_is_a_hit() {
    let r = 0.3;
    for k in [33, 36] {
        let s = (2f64).powi(-k);
        for max_t in [100.0, 1000.0, 10000.0] {
            for (name, w) in floors(2.5 * max_t) {
                // oracle: resting on y = 0 and moving along (1, −s, 0), the core
                // is s·max_t ≥ 1.4e−9 > 2⁻³² inside the floor at max_t, so it
                // hits at once: t = 0.
                let hs = sphere_cast(&w, [0.0, r, 0.0], r, [1.0, -s, 0.0], max_t);
                let hc = capsule_cast(
                    &w,
                    [0.0, r, 0.0],
                    [0.0, r + 1.0, 0.0],
                    r,
                    [1.0, -s, 0.0],
                    max_t,
                );
                for (kind, h) in [("sphere", hs), ("capsule", hc)] {
                    let h = h.unwrap_or_else(|| {
                        panic!(
                            "{name} {kind} slope 2^-{k} max_t {max_t}: missed a path {:e} into it",
                            s * max_t
                        )
                    });
                    assert_eq!(
                        h.t.to_f64(),
                        0.0,
                        "{name} {kind} slope 2^-{k} max_t {max_t}"
                    );
                    // oracle: the floor's normal +Y
                    assert!(
                        (h.normal.y.to_f64() - 1.0).abs() < 1e-9,
                        "{name} {kind}: normal {:?}",
                        f3(h.normal)
                    );
                }
            }
        }
    }
}

#[test]
fn path_sinking_less_than_the_tolerance_is_not_a_hit() {
    let r = 0.3;
    // oracle: slope 2⁻³⁶ over max_t = 4 is 2⁻³⁴ < 2⁻³² inside at the end: it stays
    // touching, not a hit.
    let s = (2f64).powi(-36);
    for (name, w) in floors(10.0) {
        let hs = sphere_cast(&w, [0.0, r, 0.0], r, [1.0, -s, 0.0], 4.0);
        let hc = capsule_cast(
            &w,
            [0.0, r, 0.0],
            [0.0, r + 1.0, 0.0],
            r,
            [1.0, -s, 0.0],
            4.0,
        );
        assert!(hs.is_none(), "{name} sphere: {hs:?}");
        assert!(hc.is_none(), "{name} capsule: {hc:?}");
    }
}

#[test]
fn plane_approach_below_parallel_threshold_is_a_hit() {
    let r = 0.3;
    // A sphere 2⁻³⁰ above the plane y = 0 (clear of it: more than 2⁻³²) moving
    // along (1, −2⁻³⁶, 0) over max_t = 2⁸: it reaches the plane at
    // t = 2⁻³⁰ / 2⁻³⁶ = 64 and is 2⁸·2⁻³⁶ − 2⁻³⁰ = 3·2⁻³⁰ inside at max_t.
    let s = (2f64).powi(-36);
    let g = (2f64).powi(-30);
    let mut w = world();
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    // oracle: t = g / s = 64 (the direction (1, −s, 0) has length 1 + s²/2,
    // below 2⁻⁶⁴ from 1), normal +Y.
    let h = sphere_cast(&w, [0.0, r + g, 0.0], r, [1.0, -s, 0.0], 256.0).expect("plane hit");
    assert!((h.t.to_f64() - 64.0).abs() < 1e-6, "t = {}", h.t.to_f64());
    assert!((h.normal.y.to_f64() - 1.0).abs() < 1e-12);
}
