//! Oracles for the contact normal convention of `physics2d`
//! (`Contact2D::normal` points from `body_a` toward `body_b`) through the public
//! narrow phase `PhysicsWorld2D::check_collision_2d` and the production entry
//! `PhysicsWorld2D::step`.
//!
//! Closed forms (lower body `L` fixed at the origin, upper body `U` above it):
//!
//! * narrow phase, both argument orders: `check_collision_2d(L, U).normal = (0, 1)`,
//!   `check_collision_2d(U, L).normal = (0, -1)`, `depth = sum of extents - centre distance`
//!   (circle-circle `r_a + r_b - d`, circle on a box face `h + r - y`, circle on a box corner
//!   `r - |c - corner|` along `(c - corner) / |c - corner|`, box on box `2h - y`,
//!   circle on an edge `r - y`, circle on a capsule `r_cap + r - y`)
//! * rest under gravity: a circle of radius `r` on a static box of half extent `h` rests at
//!   `y = h + r`, on a static edge at `y = r`, on a static capsule of radius `r_cap` at
//!   `y = r_cap + r`; a box of half extent `h` on a static box of half extent `h` rests at
//!   `y = 2h`; the vertical velocity is 0, for the default config and with `substeps` /
//!   `iterations` swept (1/1 and 8/16)
//!
//! Rest tolerance: at rest each substep first sinks the body by `g h_s^2`
//! (`10 * (1/240)^2 = 1.7e-4` m with the default 4 substeps at 60 Hz, at least
//! `10 * (1/480)^2 = 4.3e-5` m in the sweep) and the position
//! projection removes the whole overlap, re-evaluated from the current positions, so the
//! rest height is exact up to fixed-point rounding of the normalisations (`~1e-18`). The
//! oracles allow `1e-9` m and `1e-9` m/s, four orders of magnitude below one substep's sink:
//! a solver that leaves any measurable fraction of the sink, or pushes the wrong way, fails.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::physics2d::{
    Contact2D, PhysicsConfig2D, PhysicsWorld2D, RigidBody2D, Shape2D, Vec2Fix,
};

const TOL_NORMAL: f64 = 1e-12;
const TOL_DEPTH: f64 = 1e-12;
const TOL_REST: f64 = 1e-9;

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
fn circle(r: f64) -> Shape2D {
    Shape2D::Circle { radius: f(r) }
}
fn capsule(r: f64, half_length: f64) -> Shape2D {
    Shape2D::Capsule {
        radius: f(r),
        half_length: f(half_length),
    }
}
fn ground_edge() -> Shape2D {
    Shape2D::Edge {
        start: v(-50.0, 0.0),
        end: v(50.0, 0.0),
    }
}

/// Contact between a static `lower` body at `lower_pos` and a dynamic `upper` body at
/// `upper_pos`, queried as `(A, B) = (lower, upper)` when `lower_first`, else reversed.
fn contact(
    lower: &Shape2D,
    lower_pos: (f64, f64),
    upper: &Shape2D,
    upper_pos: (f64, f64),
    lower_first: bool,
) -> Contact2D {
    let mut w = PhysicsWorld2D::new(PhysicsConfig2D::default());
    let l = RigidBody2D::new_static(v(lower_pos.0, lower_pos.1), lower.clone());
    let u = RigidBody2D::new_dynamic(v(upper_pos.0, upper_pos.1), Fix128::ONE, upper.clone());
    if lower_first {
        w.add_body(l);
        w.add_body(u);
    } else {
        w.add_body(u);
        w.add_body(l);
    }
    let c = w
        .check_collision_2d(0, 1)
        .expect("overlapping bodies must report a contact");
    assert_eq!((c.body_a, c.body_b), (0, 1));
    c
}

/// `n_up` is the analytic unit direction from the lower body to the upper body.
fn assert_pair(
    name: &str,
    lower: &Shape2D,
    lower_pos: (f64, f64),
    upper: &Shape2D,
    upper_pos: (f64, f64),
    n_up: (f64, f64),
    depth: f64,
) {
    for lower_first in [true, false] {
        let c = contact(lower, lower_pos, upper, upper_pos, lower_first);
        let want = if lower_first {
            n_up
        } else {
            (-n_up.0, -n_up.1)
        };
        let got = (c.normal.x.to_f64(), c.normal.y.to_f64());
        let order = if lower_first {
            "(lower, upper)"
        } else {
            "(upper, lower)"
        };
        assert!(
            (got.0 - want.0).abs() < TOL_NORMAL && (got.1 - want.1).abs() < TOL_NORMAL,
            "{name} {order}: normal {got:?}, want {want:?} (body_a -> body_b)"
        );
        assert!(
            (c.depth.to_f64() - depth).abs() < TOL_DEPTH,
            "{name} {order}: depth {}, want {depth}",
            c.depth.to_f64()
        );
    }
}

#[test]
fn circle_circle_normal_points_from_a_to_b_in_both_orders() {
    // r_a + r_b - d = 1 + 1 - 1.5
    assert_pair(
        "circle-circle",
        &circle(1.0),
        (0.0, 0.0),
        &circle(1.0),
        (0.0, 1.5),
        (0.0, 1.0),
        0.5,
    );
}

#[test]
fn circle_box_face_normal_points_from_a_to_b_in_both_orders() {
    // box top at y = 1, circle bottom at 1.75 - 1 = 0.75: overlap 0.25
    assert_pair(
        "circle-box face",
        &square(1.0),
        (0.0, 0.0),
        &circle(1.0),
        (0.0, 1.75),
        (0.0, 1.0),
        0.25,
    );
}

#[test]
fn circle_box_corner_normal_points_from_a_to_b_in_both_orders() {
    // corner (1, 1), centre (1.5, 1.5): distance sqrt(0.5) along (1, 1)/sqrt(2)
    let s = 0.5f64.sqrt();
    assert_pair(
        "circle-box corner",
        &square(1.0),
        (0.0, 0.0),
        &circle(1.0),
        (1.5, 1.5),
        (s, s),
        1.0 - s,
    );
}

#[test]
fn box_box_normal_points_from_a_to_b_in_both_orders() {
    // 2h - y = 2 - 1.75
    assert_pair(
        "box-box",
        &square(1.0),
        (0.0, 0.0),
        &square(1.0),
        (0.0, 1.75),
        (0.0, 1.0),
        0.25,
    );
}

#[test]
fn circle_edge_normal_points_from_a_to_b_in_both_orders() {
    // r - y = 1 - 0.75
    assert_pair(
        "circle-edge",
        &ground_edge(),
        (0.0, 0.0),
        &circle(1.0),
        (0.0, 0.75),
        (0.0, 1.0),
        0.25,
    );
}

#[test]
fn circle_capsule_normal_points_from_a_to_b_in_both_orders() {
    // r_cap + r - y = 0.5 + 1 - 1.25
    assert_pair(
        "circle-capsule",
        &capsule(0.5, 2.0),
        (0.0, 0.0),
        &circle(1.0),
        (0.0, 1.25),
        (0.0, 1.0),
        0.25,
    );
}

/// Drop `upper` from `start_y` onto the static `lower` at the origin, inserting the two
/// bodies in the given order, and return the upper body's final `(y, vy)` after 5 s.
fn drop_onto(
    lower: &Shape2D,
    upper: &Shape2D,
    start_y: f64,
    lower_first: bool,
    config: PhysicsConfig2D,
) -> (f64, f64) {
    let mut w = PhysicsWorld2D::new(config);
    let l = RigidBody2D::new_static(Vec2Fix::ZERO, lower.clone());
    let u = RigidBody2D::new_dynamic(v(0.0, start_y), Fix128::ONE, upper.clone());
    let id = if lower_first {
        w.add_body(l);
        w.add_body(u)
    } else {
        let id = w.add_body(u);
        w.add_body(l);
        id
    };
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..300 {
        w.step(dt);
    }
    let b = &w.bodies[id];
    (b.position.y.to_f64(), b.velocity.y.to_f64())
}

fn assert_rest(name: &str, lower: &Shape2D, upper: &Shape2D, start_y: f64, rest_y: f64) {
    // default config first, then the precision parameters swept: the rest height of a
    // single contact does not depend on them
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
    for config in configs {
        let (substeps, iterations) = (config.substeps, config.iterations);
        for lower_first in [true, false] {
            let (y, vy) = drop_onto(lower, upper, start_y, lower_first, config.clone());
            assert!(
                (y - rest_y).abs() < TOL_REST && vy.abs() < TOL_REST,
                "{name} (lower inserted first: {lower_first}, substeps {substeps}, iterations \
                 {iterations}): y = {y}, vy = {vy}; want y = {rest_y}, vy = 0"
            );
        }
    }
}

#[test]
fn circle_dropped_on_static_box_rests_on_its_top() {
    // h + r = 1 + 0.5
    assert_rest("circle on box", &square(1.0), &circle(0.5), 3.0, 1.5);
}

#[test]
fn circle_dropped_on_static_edge_rests_on_it() {
    assert_rest("circle on edge", &ground_edge(), &circle(0.5), 2.0, 0.5);
}

#[test]
fn circle_dropped_on_static_capsule_rests_on_its_top() {
    // r_cap + r = 0.5 + 0.5
    assert_rest(
        "circle on capsule",
        &capsule(0.5, 3.0),
        &circle(0.5),
        2.5,
        1.0,
    );
}

#[test]
fn box_dropped_on_static_box_rests_on_its_top() {
    // 2h = 2
    assert_rest("box on box", &square(1.0), &square(1.0), 3.0, 2.0);
}
