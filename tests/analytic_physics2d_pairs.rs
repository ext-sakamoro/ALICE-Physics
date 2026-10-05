//! Oracles for the 2D narrow-phase pairs between capsules, edges and convex polygons
//! (capsule-capsule, capsule-polygon, edge-polygon, edge-capsule, edge-edge) through the
//! public narrow phase `PhysicsWorld2D::check_collision_2d` and the production entry
//! `PhysicsWorld2D::step`.
//!
//! Every shape is a core (a segment, or a convex polygon) inflated by a radius: a capsule
//! is its segment inflated by its radius, an edge its segment with radius 0, a polygon its
//! vertices with radius 0. Closed forms, with `Contact2D::normal` from `body_a` to `body_b`:
//!
//! * cores apart: `depth = r_a + r_b - |q - p|` along `(q - p) / |q - p|`, `p` / `q` the
//!   closest points of the cores (segment-segment, segment-edge of the polygon)
//! * cores overlapping: `depth = r_a + r_b + PD`, `PD` the smallest overlap of the cores
//!   along the edge normals of both (the exact penetration of the Minkowski difference)
//! * contact point: halfway between the two surfaces along the normal, and at the centre
//!   of the shared extent of the two supporting features along the tangent
//! * both argument orders: normal negated, depth and point unchanged
//! * edge-edge: no contact (two zero-thickness segments enclose no area)
//!
//! Rest under gravity (both insertion orders, default config and `substeps` / `iterations`
//! swept to 1/1 and 8/16): a capsule of radius `r` on a static box of half extent `h` rests
//! at `y = h + r`, on a static edge at `y = r`; a box of half extent `h` on a static edge at
//! `y = h`; a horizontal capsule of radius `r` across the top of a static upright capsule
//! (radius `R`, half length `L`) at `y = L + R + r`; velocity and angle 0.
//!
//! Rest tolerance as in `analytic_physics2d_contact_normals.rs`: each substep sinks the body
//! by `g h_s^2` (at least `4.3e-5` m over the swept configs) and the projection removes the
//! whole overlap re-evaluated from the current positions, so the rest state is exact up to
//! fixed-point rounding (`~1e-18`); `1e-9` is four orders below one substep's sink. The
//! narrow-phase values are closed forms of exactly representable inputs (rotations by
//! `pi/2` / `pi/4` are the only inexact step), checked to `1e-12`.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::physics2d::{
    Contact2D, PhysicsConfig2D, PhysicsWorld2D, RigidBody2D, Shape2D, Vec2Fix,
};

const TOL: f64 = 1e-12;
const TOL_REST: f64 = 1e-9;
const S: f64 = std::f64::consts::FRAC_1_SQRT_2;

fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn v(x: f64, y: f64) -> Vec2Fix {
    Vec2Fix::new(f(x), f(y))
}
fn square(h: f64) -> Shape2D {
    Shape2D::Polygon {
        vertices: vec![v(-h, -h), v(h, -h), v(h, h), v(-h, h)],
    }
}
fn capsule(r: f64, half_length: f64) -> Shape2D {
    Shape2D::Capsule {
        radius: f(r),
        half_length: f(half_length),
    }
}
fn edge(x0: f64, y0: f64, x1: f64, y1: f64) -> Shape2D {
    Shape2D::Edge {
        start: v(x0, y0),
        end: v(x1, y1),
    }
}
fn ground_edge() -> Shape2D {
    edge(-50.0, 0.0, 50.0, 0.0)
}

/// A body placement: shape, position, angle (rad, as a ratio of `Fix128::HALF_PI`).
#[derive(Clone)]
struct Place {
    shape: Shape2D,
    pos: (f64, f64),
    quarter_turns: Fix128,
}

fn at(shape: Shape2D, x: f64, y: f64) -> Place {
    Place {
        shape,
        pos: (x, y),
        quarter_turns: Fix128::ZERO,
    }
}
fn upright(shape: Shape2D, x: f64, y: f64) -> Place {
    Place {
        quarter_turns: Fix128::ONE,
        ..at(shape, x, y)
    }
}
fn diamond(shape: Shape2D, x: f64, y: f64) -> Place {
    Place {
        quarter_turns: Fix128::from_ratio(1, 2),
        ..at(shape, x, y)
    }
}

fn world(lower: &Place, upper: &Place, lower_first: bool) -> PhysicsWorld2D {
    let mut w = PhysicsWorld2D::new(PhysicsConfig2D::default());
    let mut l = RigidBody2D::new_static(v(lower.pos.0, lower.pos.1), lower.shape.clone());
    l.angle = Fix128::HALF_PI * lower.quarter_turns;
    let mut u = RigidBody2D::new_dynamic(
        v(upper.pos.0, upper.pos.1),
        Fix128::ONE,
        upper.shape.clone(),
    );
    u.angle = Fix128::HALF_PI * upper.quarter_turns;
    if lower_first {
        w.add_body(l);
        w.add_body(u);
    } else {
        w.add_body(u);
        w.add_body(l);
    }
    w
}

fn query(lower: &Place, upper: &Place, lower_first: bool) -> Option<Contact2D> {
    let c = world(lower, upper, lower_first).check_collision_2d(0, 1);
    if let Some(c) = c {
        assert_eq!((c.body_a, c.body_b), (0, 1));
    }
    c
}

fn close(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < TOL && (a.1 - b.1).abs() < TOL
}

/// `n_up`: analytic unit direction from `lower` to `upper`; `point`: analytic contact point.
fn assert_pair(
    name: &str,
    lower: &Place,
    upper: &Place,
    n_up: (f64, f64),
    depth: f64,
    point: (f64, f64),
) {
    for lower_first in [true, false] {
        let order = if lower_first {
            "(lower, upper)"
        } else {
            "(upper, lower)"
        };
        let c = query(lower, upper, lower_first)
            .unwrap_or_else(|| panic!("{name} {order}: overlapping shapes reported no contact"));
        let want = if lower_first {
            n_up
        } else {
            (-n_up.0, -n_up.1)
        };
        let got = (c.normal.x.to_f64(), c.normal.y.to_f64());
        assert!(
            close(got, want),
            "{name} {order}: normal {got:?}, want {want:?} (body_a -> body_b)"
        );
        assert!(
            (c.depth.to_f64() - depth).abs() < TOL,
            "{name} {order}: depth {}, want {depth}",
            c.depth.to_f64()
        );
        let p = (c.point.x.to_f64(), c.point.y.to_f64());
        assert!(
            close(p, point),
            "{name} {order}: point {p:?}, want {point:?}"
        );
    }
}

fn assert_apart(name: &str, lower: &Place, upper: &Place) {
    for lower_first in [true, false] {
        let c = query(lower, upper, lower_first);
        assert!(
            c.is_none(),
            "{name} (lower first: {lower_first}): want no contact, got {c:?}"
        );
    }
}

// ---------------------------------------------------------------- capsule-capsule

#[test]
fn capsule_capsule_parallel_overlap() {
    // segments 0.75 apart, r_a + r_b = 1: depth 0.25; shared extent x in [-0.5, 1.5]
    // -> x = 0.5; surfaces at y = 0.5 and 0.25 -> y = 0.375
    assert_pair(
        "capsule-capsule parallel",
        &at(capsule(0.5, 2.0), 0.0, 0.0),
        &at(capsule(0.5, 1.0), 0.5, 0.75),
        (0.0, 1.0),
        0.25,
        (0.5, 0.375),
    );
}

#[test]
fn capsule_capsule_end_to_end() {
    // closest ends (1, 0) and (1.5, 0.5): distance sqrt(0.5) along (1, 1)/sqrt(2);
    // point halfway between the ends
    assert_pair(
        "capsule-capsule ends",
        &at(capsule(0.5, 1.0), 0.0, 0.0),
        &at(capsule(0.5, 1.0), 2.5, 0.5),
        (S, S),
        1.0 - 0.5f64.sqrt(),
        (1.25, 0.25),
    );
}

#[test]
fn capsule_capsule_crossing_cores() {
    // upright core x = 0, y in [-0.75, 1.25] crosses the horizontal core y = 0, x in [-2, 2];
    // smallest core overlap 0.75 (lift B's lower end to y = 0) -> depth 0.75 + 1;
    // surfaces at y = 0.5 and -1.25 -> y = -0.375 at the lower end's x = 0
    assert_pair(
        "capsule-capsule crossing",
        &at(capsule(0.5, 2.0), 0.0, 0.0),
        &upright(capsule(0.5, 1.0), 0.0, 0.25),
        (0.0, 1.0),
        1.75,
        (0.0, -0.375),
    );
}

#[test]
fn capsule_capsule_apart() {
    assert_apart(
        "capsule-capsule",
        &at(capsule(0.5, 2.0), 0.0, 0.0),
        &at(capsule(0.5, 1.0), 0.0, 1.01),
    );
}

// ---------------------------------------------------------------- capsule-polygon

#[test]
fn capsule_box_face() {
    // segment 0.25 above the top face (y = 1): depth 0.5 - 0.25; x in [-0.25, 0.75] on
    // the face -> x = 0.25; surfaces y = 1 and 0.75 -> y = 0.875
    assert_pair(
        "capsule-box face",
        &at(square(1.0), 0.0, 0.0),
        &at(capsule(0.5, 0.5), 0.25, 1.25),
        (0.0, 1.0),
        0.25,
        (0.25, 0.875),
    );
}

#[test]
fn capsule_box_corner() {
    // capsule end (1.25, 1.25) against corner (1, 1): distance sqrt(0.125) along
    // (1, 1)/sqrt(2); point corner - n depth / 2
    let d = 0.125f64.sqrt();
    let depth = 0.5 - d;
    assert_pair(
        "capsule-box corner",
        &at(square(1.0), 0.0, 0.0),
        &at(capsule(0.5, 1.0), 2.25, 1.25),
        (S, S),
        depth,
        (1.0 - S * depth / 2.0, 1.0 - S * depth / 2.0),
    );
}

#[test]
fn capsule_core_inside_box() {
    // core y = 0.75 inside the box: core overlap 1 - 0.75, depth 0.25 + 0.5; surfaces
    // y = 1 and 0.25 -> y = 0.625
    assert_pair(
        "capsule-box core inside",
        &at(square(1.0), 0.0, 0.0),
        &at(capsule(0.5, 0.5), 0.0, 0.75),
        (0.0, 1.0),
        0.75,
        (0.0, 0.625),
    );
}

#[test]
fn capsule_box_apart() {
    let b = at(square(1.0), 0.0, 0.0);
    assert_apart("capsule-box face", &b, &at(capsule(0.5, 0.5), 0.0, 1.6));
    // end (1.25, 1.5) is sqrt(0.3125) > 0.5 from the corner
    assert_apart("capsule-box corner", &b, &at(capsule(0.5, 1.0), 2.25, 1.5));
}

// ---------------------------------------------------------------- edge-polygon

#[test]
fn edge_box_face() {
    // bottom face y = -0.25 below the edge: depth 0.25; x in [-0.5, 1.5] -> x = 0.5;
    // surfaces y = 0 and -0.25 -> y = -0.125
    assert_pair(
        "edge-box face",
        &at(ground_edge(), 0.0, 0.0),
        &at(square(1.0), 0.5, 0.75),
        (0.0, 1.0),
        0.25,
        (0.5, -0.125),
    );
}

#[test]
fn edge_box_vertex() {
    // a box turned by pi/4 puts its lowest vertex at (0, 0.9 - sqrt(2)): depth sqrt(2) - 0.9
    let depth = 2f64.sqrt() - 0.9;
    assert_pair(
        "edge-box vertex",
        &at(edge(-5.0, 0.0, 5.0, 0.0), 0.0, 0.0),
        &diamond(square(1.0), 0.0, 0.9),
        (0.0, 1.0),
        depth,
        (0.0, -depth / 2.0),
    );
}

#[test]
fn edge_box_apart() {
    assert_apart(
        "edge-box",
        &at(ground_edge(), 0.0, 0.0),
        &at(square(1.0), 0.0, 1.01),
    );
}

// ---------------------------------------------------------------- edge-capsule

#[test]
fn edge_capsule_flat() {
    // segment 0.25 above the edge: depth 0.5 - 0.25; x in [-0.5, 1.5] -> x = 0.5;
    // surfaces y = 0 and -0.25 -> y = -0.125
    assert_pair(
        "edge-capsule flat",
        &at(ground_edge(), 0.0, 0.0),
        &at(capsule(0.5, 1.0), 0.5, 0.25),
        (0.0, 1.0),
        0.25,
        (0.5, -0.125),
    );
}

#[test]
fn edge_capsule_crossing_core() {
    // upright core y in [-0.75, 1.25] crosses the edge: core overlap 0.75, depth 0.75 + 0.5;
    // surfaces y = 0 and -1.25 -> y = -0.625
    assert_pair(
        "edge-capsule crossing",
        &at(ground_edge(), 0.0, 0.0),
        &upright(capsule(0.5, 1.0), 0.0, 0.25),
        (0.0, 1.0),
        1.25,
        (0.0, -0.625),
    );
}

#[test]
fn edge_capsule_apart() {
    assert_apart(
        "edge-capsule",
        &at(ground_edge(), 0.0, 0.0),
        &at(capsule(0.5, 1.0), 0.0, 0.51),
    );
}

// ---------------------------------------------------------------- edge-edge

#[test]
fn edge_edge_never_collides() {
    // even crossing segments: zero-thickness edges do not collide with each other
    assert_apart(
        "edge-edge crossing",
        &at(ground_edge(), 0.0, 0.0),
        &at(edge(-1.0, -1.0, 1.0, 1.0), 0.0, 0.0),
    );
}

// ---------------------------------------------------------------- rest through step

/// Drop `upper` onto the static `lower` (both insertion orders) and return the upper
/// body's `(y, vy, angle)` after 5 s.
fn drop_onto(lower: &Place, upper: &Place, lower_first: bool, config: PhysicsConfig2D) -> [f64; 3] {
    let mut w = world(lower, upper, lower_first);
    w.config = config;
    let id = usize::from(lower_first);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..300 {
        w.step(dt);
    }
    let b = &w.bodies[id];
    [
        b.position.y.to_f64(),
        b.velocity.y.to_f64(),
        b.angle.to_f64(),
    ]
}

fn assert_rest(name: &str, lower: &Place, upper: &Place, rest_y: f64) {
    let configs = [
        PhysicsConfig2D::default(),
        PhysicsConfig2D {
            substeps: 1,
            iterations: 1,
            ..PhysicsConfig2D::default()
        },
        PhysicsConfig2D {
            substeps: 8,
            iterations: 16,
            ..PhysicsConfig2D::default()
        },
    ];
    let angle0 = (Fix128::HALF_PI * upper.quarter_turns).to_f64();
    for config in configs {
        let (substeps, iterations) = (config.substeps, config.iterations);
        for lower_first in [true, false] {
            let [y, vy, angle] = drop_onto(lower, upper, lower_first, config.clone());
            assert!(
                (y - rest_y).abs() < TOL_REST
                    && vy.abs() < TOL_REST
                    && (angle - angle0).abs() < TOL_REST,
                "{name} (lower inserted first: {lower_first}, substeps {substeps}, iterations \
                 {iterations}): y = {y}, vy = {vy}, angle = {angle}; want y = {rest_y}, vy = 0, \
                 angle = {angle0}"
            );
        }
    }
}

#[test]
fn capsule_dropped_on_static_box_rests_on_its_top() {
    // h + r = 2 + 0.5
    assert_rest(
        "capsule on box",
        &at(square(2.0), 0.0, 0.0),
        &at(capsule(0.5, 1.0), 0.0, 4.0),
        2.5,
    );
}

#[test]
fn capsule_dropped_on_static_edge_rests_on_it() {
    assert_rest(
        "capsule on edge",
        &at(ground_edge(), 0.0, 0.0),
        &at(capsule(0.5, 1.0), 0.0, 2.0),
        0.5,
    );
}

#[test]
fn box_dropped_on_static_edge_rests_on_it() {
    assert_rest(
        "box on edge",
        &at(ground_edge(), 0.0, 0.0),
        &at(square(1.0), 0.0, 3.0),
        1.0,
    );
}

#[test]
fn capsule_dropped_across_upright_capsule_rests_on_its_top() {
    // L + R + r = 1 + 1 + 0.25 (R > r: the bar rolls back to the top when tilted)
    assert_rest(
        "capsule across capsule",
        &upright(capsule(1.0, 1.0), 0.0, 0.0),
        &at(capsule(0.25, 1.5), 0.0, 4.0),
        2.25,
    );
}

// ---------------------------------------------------------------- degenerate inputs

#[test]
fn zero_length_capsules_reduce_to_circles() {
    // half_length 0: the core is a point, so the pair is circle-circle:
    // depth r_a + r_b - d = 1 - 0.75, point halfway between the surfaces
    assert_pair(
        "point capsules",
        &at(capsule(0.5, 0.0), 0.0, 0.0),
        &at(capsule(0.5, 0.0), 0.0, 0.75),
        (0.0, 1.0),
        0.25,
        (0.0, 0.375),
    );
}

#[test]
fn zero_length_edge_against_box_is_a_point() {
    // a point edge at (0, 0.5) inside a box whose top is at y = 1: no edge normal, the box
    // face normals alone give the penetration 0.5 toward the top face (B = box below,
    // so the edge is pushed up: normal from box to edge +y)
    assert_pair(
        "point edge in box",
        &at(square(1.0), 0.0, 0.0),
        &at(edge(0.0, 0.0, 0.0, 0.0), 0.0, 0.5),
        (0.0, 1.0),
        0.5,
        (0.0, 0.75),
    );
}

#[test]
fn coincident_capsules_separate_along_the_first_core_normal() {
    // identical cores: zero core overlap on the segment normal, so depth = r_a + r_b and
    // the direction is the normal of A's segment (deterministic tie: +perp of A's axis)
    for lower_first in [true, false] {
        let c = query(
            &at(capsule(0.5, 1.0), 0.0, 0.0),
            &at(capsule(0.5, 1.0), 0.0, 0.0),
            lower_first,
        )
        .expect("coincident capsules overlap");
        assert!(close(
            (c.normal.x.to_f64(), c.normal.y.to_f64()),
            (0.0, 1.0)
        ));
        assert!((c.depth.to_f64() - 1.0).abs() < TOL);
    }
}

#[test]
fn coincident_point_capsules_use_the_unit_y_fallback() {
    // both cores the same point: no axis exists, the normal falls back to +y with depth
    // r_a + r_b
    let c = query(
        &at(capsule(0.5, 0.0), 1.0, 1.0),
        &at(capsule(0.25, 0.0), 1.0, 1.0),
        true,
    )
    .expect("coincident point capsules overlap");
    assert!(close(
        (c.normal.x.to_f64(), c.normal.y.to_f64()),
        (0.0, 1.0)
    ));
    assert!((c.depth.to_f64() - 0.75).abs() < TOL);
}

#[test]
fn point_capsule_on_the_line_of_a_segment_uses_the_end_distance() {
    // a point core on the extension of a segment's line overlaps it on the segment normal,
    // so separation is decided by the distance to the end (1, 0): 0.6 < 0.75 overlaps with
    // depth 0.15 along +x, 0.9 > 0.75 is apart
    assert_pair(
        "point capsule past the end",
        &at(capsule(0.5, 1.0), 0.0, 0.0),
        &at(capsule(0.25, 0.0), 1.6, 0.0),
        (1.0, 0.0),
        0.15,
        (1.425, 0.0),
    );
    assert_apart(
        "point capsule past the end",
        &at(capsule(0.5, 1.0), 0.0, 0.0),
        &at(capsule(0.25, 0.0), 1.9, 0.0),
    );
}
