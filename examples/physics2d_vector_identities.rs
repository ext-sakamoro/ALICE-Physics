//! 2D vector identities, body constructors and a single joint projection,
//! each checked against a closed form the example computes itself.
//!
//! - `Vec2Fix`: `ONE = UNIT_X + UNIT_Y`, `from_int` matches `new`,
//!   `perpendicular(v) · v = 0` and `|perpendicular(v)| = |v|`,
//!   `rotate(π/2) = perpendicular`, `|normalize(v)| = 1` (and `normalize(0) = 0`),
//!   `lerp(a, b, 0) = a`, `lerp(a, b, 1) = b`, `lerp(a, b, 1/2) = (a + b) / 2`.
//! - `RigidBody2D::apply_impulse`: `Δv = J / m` on a dynamic body, no change on a
//!   static or kinematic one even when its inverse mass is set nonzero by hand
//!   (`is_static_or_kinematic` reports which is which).
//! - `RigidBody2D::world_point`: `x + R(θ) p`; at `θ = π/2` the local `(1, 0)` maps to
//!   `x + (0, 1)`.
//! - `solve_joints_2d` (one rigid distance projection, compliance 0): two point masses
//!   `m_a = 1`, `m_b = 3` at distance 4 with target 2. The error `C = 2` is split by
//!   inverse mass, `Δx_a = C w_a / (w_a + w_b) = 1.5`, `Δx_b = −C w_b / (w_a + w_b) = −0.5`,
//!   so the distance becomes exactly the target and `m_a Δx_a + m_b Δx_b = 0`.
//!   Against a static partner the dynamic body takes the whole correction.
//!
//! ```bash
//! cargo run --release --example physics2d_vector_identities --features std
//! ```

use alice_physics::math::Fix128;
use alice_physics::physics2d::{solve_joints_2d, Joint2D, RigidBody2D, Shape2D, Vec2Fix};

/// Tolerance for values that go through `sqrt` / `sin_cos` (Fix128 has 64 fractional bits).
const EPS: f64 = 1e-12;

fn close(a: Fix128, b: f64, what: &str) {
    let d = (a.to_f64() - b).abs();
    assert!(
        d < EPS,
        "{what}: got {} expected {b} (|diff| = {d:e})",
        a.to_f64()
    );
}

fn vec_identities() {
    // constants
    assert_eq!(
        Vec2Fix::ONE,
        Vec2Fix::UNIT_X + Vec2Fix::UNIT_Y,
        "ONE = UNIT_X + UNIT_Y"
    );
    assert_eq!(
        Vec2Fix::UNIT_X.dot(Vec2Fix::UNIT_Y),
        Fix128::ZERO,
        "UNIT_X ⟂ UNIT_Y"
    );
    assert_eq!(
        Vec2Fix::from_int(3, -4),
        Vec2Fix::new(Fix128::from_int(3), Fix128::from_int(-4)),
        "from_int = new(from_int, from_int)"
    );

    // perpendicular: (-y, x), orthogonal and same length
    let v = Vec2Fix::from_int(3, -4);
    let p = v.perpendicular();
    assert_eq!(p, Vec2Fix::from_int(4, 3), "perpendicular(3, -4) = (4, 3)");
    assert_eq!(p.dot(v), Fix128::ZERO, "perpendicular(v) · v = 0");
    assert_eq!(
        p.length_squared(),
        v.length_squared(),
        "|perpendicular(v)| = |v|"
    );
    assert_eq!(
        Vec2Fix::UNIT_X.perpendicular(),
        Vec2Fix::UNIT_Y,
        "perpendicular(UNIT_X) = UNIT_Y"
    );

    // rotate by π/2 equals perpendicular (up to the sin/cos rounding)
    let r = v.rotate(Fix128::HALF_PI);
    close(r.x, p.x.to_f64(), "rotate(π/2).x");
    close(r.y, p.y.to_f64(), "rotate(π/2).y");
    // rotate by 0 is the identity (cos 0 from `sin_cos` carries ~1e-14 rounding)
    let r0 = v.rotate(Fix128::ZERO);
    close(r0.x, 3.0, "rotate(0).x");
    close(r0.y, -4.0, "rotate(0).y");

    // normalize: |v / |v|| = 1, (3, -4)/5 exactly, zero stays zero
    let n = v.normalize();
    close(n.length(), 1.0, "|normalize(v)|");
    close(n.x, 3.0 / 5.0, "normalize(3, -4).x");
    close(n.y, -4.0 / 5.0, "normalize(3, -4).y");
    assert_eq!(Vec2Fix::ZERO.normalize(), Vec2Fix::ZERO, "normalize(0) = 0");

    // lerp endpoints and midpoint
    let a = Vec2Fix::from_int(-2, 6);
    let b = Vec2Fix::from_int(10, -2);
    assert_eq!(a.lerp(b, Fix128::ZERO), a, "lerp(a, b, 0) = a");
    assert_eq!(a.lerp(b, Fix128::ONE), b, "lerp(a, b, 1) = b");
    assert_eq!(
        a.lerp(b, Fix128::from_ratio(1, 2)),
        Vec2Fix::from_int(4, 2),
        "lerp(a, b, 1/2) = (a + b) / 2"
    );
    println!("Vec2Fix identities: ok (perp(3,-4) = (4,3), |normalize| = 1, lerp endpoints)");
}

fn circle(r_num: i64, r_den: i64) -> Shape2D {
    Shape2D::Circle {
        radius: Fix128::from_ratio(r_num, r_den),
    }
}

fn body_impulses_and_frames() {
    let mass = Fix128::from_int(4);
    let mut dynamic = RigidBody2D::new_dynamic(Vec2Fix::from_int(1, 2), mass, circle(1, 2));
    let mut fixed = RigidBody2D::new_static(Vec2Fix::from_int(0, 0), circle(1, 1));
    let mut moving = RigidBody2D::new_kinematic(Vec2Fix::from_int(0, 5), circle(1, 1));
    assert!(
        !dynamic.is_static_or_kinematic(),
        "dynamic body is not static/kinematic"
    );
    assert!(
        fixed.is_static_or_kinematic(),
        "static body is static/kinematic"
    );
    assert!(
        moving.is_static_or_kinematic(),
        "kinematic body is static/kinematic"
    );

    // Δv = J / m: J = (8, -2), m = 4 ⇒ Δv = (2, -1/2)
    let j = Vec2Fix::from_int(8, -2);
    dynamic.apply_impulse(j);
    let expected = Vec2Fix::new(j.x / mass, j.y / mass);
    assert_eq!(dynamic.velocity, expected, "Δv = J / m");
    assert_eq!(
        dynamic.velocity,
        Vec2Fix::new(Fix128::from_int(2), Fix128::from_ratio(-1, 2)),
        "Δv = (2, -1/2)"
    );
    // a second impulse adds linearly
    dynamic.apply_impulse(j);
    assert_eq!(dynamic.velocity, expected + expected, "impulses add");

    // static and kinematic bodies ignore impulses
    fixed.apply_impulse(j);
    moving.apply_impulse(j);
    assert_eq!(fixed.velocity, Vec2Fix::ZERO, "static body keeps v = 0");
    assert_eq!(moving.velocity, Vec2Fix::ZERO, "kinematic body keeps v = 0");
    // the body type decides, not the inverse mass: give both a nonzero inverse
    // mass and they must still ignore the impulse
    fixed.inv_mass = Fix128::ONE;
    moving.inv_mass = Fix128::ONE;
    fixed.apply_impulse(j);
    moving.apply_impulse(j);
    assert_eq!(
        fixed.velocity,
        Vec2Fix::ZERO,
        "static ignores J even with inv_mass 1"
    );
    assert_eq!(
        moving.velocity,
        Vec2Fix::ZERO,
        "kinematic ignores J even with inv_mass 1"
    );

    // world_point = x + R(θ) p
    let w0 = dynamic.world_point(Vec2Fix::from_int(3, 0));
    close(w0.x, 4.0, "θ = 0: world_point(3, 0).x = x.x + 3");
    close(w0.y, 2.0, "θ = 0: world_point(3, 0).y = x.y");
    dynamic.angle = Fix128::HALF_PI;
    let w = dynamic.world_point(Vec2Fix::UNIT_X);
    close(w.x, 1.0, "θ = π/2: world_point(1, 0).x = x.x");
    close(w.y, 3.0, "θ = π/2: world_point(1, 0).y = x.y + 1");
    println!("RigidBody2D: Δv = J/m = (2, -0.5), static/kinematic unchanged, world_point ok");
}

fn single_distance_projection() {
    let (m_a, m_b) = (1_i64, 3_i64);
    let mut bodies = vec![
        RigidBody2D::new_dynamic(Vec2Fix::from_int(0, 0), Fix128::from_int(m_a), circle(1, 4)),
        RigidBody2D::new_dynamic(Vec2Fix::from_int(4, 0), Fix128::from_int(m_b), circle(1, 4)),
    ];
    let target = 2.0;
    let joints = [Joint2D::Distance {
        body_a: 0,
        body_b: 1,
        local_anchor_a: Vec2Fix::ZERO,
        local_anchor_b: Vec2Fix::ZERO,
        target_distance: Fix128::from_int(2),
        compliance: Fix128::ZERO,
    }];
    solve_joints_2d(&mut bodies, &joints, Fix128::from_ratio(1, 60));

    let c = 4.0 - target;
    let (w_a, w_b) = (1.0 / m_a as f64, 1.0 / m_b as f64);
    let dx_a = c * w_a / (w_a + w_b);
    let dx_b = -c * w_b / (w_a + w_b);
    close(bodies[0].position.x, 0.0 + dx_a, "x_a after projection");
    close(bodies[1].position.x, 4.0 + dx_b, "x_b after projection");
    close(bodies[0].position.y, 0.0, "y_a unchanged");
    close(bodies[1].position.y, 0.0, "y_b unchanged");
    close(
        bodies[0].position.distance_to(bodies[1].position),
        target,
        "distance = target after one rigid projection",
    );
    let momentum = m_a as f64 * (bodies[0].position.x.to_f64() - 0.0)
        + m_b as f64 * (bodies[1].position.x.to_f64() - 4.0);
    assert!(
        momentum.abs() < EPS,
        "m_a Δx_a + m_b Δx_b = 0 (got {momentum:e})"
    );

    // static partner: the dynamic body takes the whole correction
    let mut pinned = vec![
        RigidBody2D::new_static(Vec2Fix::from_int(0, 0), circle(1, 4)),
        RigidBody2D::new_dynamic(Vec2Fix::from_int(0, -5), Fix128::from_int(2), circle(1, 4)),
    ];
    let rod = [Joint2D::Distance {
        body_a: 0,
        body_b: 1,
        local_anchor_a: Vec2Fix::ZERO,
        local_anchor_b: Vec2Fix::ZERO,
        target_distance: Fix128::from_int(3),
        compliance: Fix128::ZERO,
    }];
    solve_joints_2d(&mut pinned, &rod, Fix128::from_ratio(1, 60));
    assert_eq!(
        pinned[0].position,
        Vec2Fix::ZERO,
        "static anchor does not move"
    );
    close(
        pinned[1].position.y,
        -3.0,
        "dynamic end pulled to the target length",
    );
    println!(
        "solve_joints_2d: x_a = {:.3}, x_b = {:.3} (expect {dx_a}, {}), static partner ok",
        bodies[0].position.x.to_f64(),
        bodies[1].position.x.to_f64(),
        4.0 + dx_b
    );
}

fn main() {
    vec_identities();
    body_impulses_and_frames();
    single_distance_projection();
    println!("physics2d_vector_identities: all closed forms hold");
}
