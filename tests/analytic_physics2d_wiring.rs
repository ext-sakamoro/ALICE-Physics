//! Oracles for `physics2d::{RigidBody2D::apply_impulse_at_point, RigidBody2D::apply_force,
//! Vec2Fix::distance_to}` (`examples/physics2d_impulse_spin.rs`).
//!
//! * impulse `J` at world point `p`, body at `c`, mass `m`, inertia `I`, `r = p - c`:
//!   `dv = J / m`, `dw = (r.x J.y - r.y J.x) / I`; angular momentum about `p` is unchanged
//!   (`I dw - r x J = 0`), a line of action through the centre gives `dw = 0`
//! * force over `dt`: `dv = F dt / m`, no spin
//! * non-dynamic bodies ignore both
//! * `distance_to = |b - a|`, symmetric, zero on itself
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::physics2d::{BodyType2D, RigidBody2D, Shape2D, Vec2Fix};

fn fv(x: f64, y: f64) -> Vec2Fix {
    Vec2Fix::new(Fix128::from_f64(x), Fix128::from_f64(y))
}
fn square(h: i64) -> Shape2D {
    let h = Fix128::from_int(h);
    Shape2D::Polygon {
        vertices: vec![
            Vec2Fix::new(-h, -h),
            Vec2Fix::new(h, -h),
            Vec2Fix::new(h, h),
            Vec2Fix::new(-h, h),
        ],
    }
}
fn plate(mass: i64, pos: (f64, f64)) -> RigidBody2D {
    RigidBody2D::new_dynamic(fv(pos.0, pos.1), Fix128::from_int(mass), square(1))
}
fn close(a: Fix128, want: f64) {
    assert!(
        (a.to_f64() - want).abs() < 1e-12,
        "{} vs {want}",
        a.to_f64()
    );
}

#[test]
fn impulse_at_point_splits_into_linear_and_angular_parts() {
    // square h=1: I = m * 8 / 12 = 2 for m = 3
    let mut b = plate(3, (5.0, 5.0));
    b.apply_impulse_at_point(fv(0.0, 4.0), fv(6.0, 5.0)); // r = (1, 0): r x J = 4
    close(b.velocity.x, 0.0);
    close(b.velocity.y, 4.0 / 3.0);
    close(b.angular_velocity, 2.0);
    // second kick adds linearly (superposition): J = (3, -3) at r = (-1, 2): r x J = (-1)(-3) - 2*3 = -3
    b.apply_impulse_at_point(fv(3.0, -3.0), fv(4.0, 7.0));
    close(b.velocity.x, 1.0);
    close(b.velocity.y, 4.0 / 3.0 - 1.0);
    close(b.angular_velocity, 2.0 - 1.5);
    // position and angle are untouched by an impulse
    assert_eq!(b.position, fv(5.0, 5.0));
    assert_eq!(b.angle, Fix128::ZERO);
}

#[test]
fn line_of_action_through_the_centre_gives_no_spin_and_matches_apply_impulse() {
    let mut a = plate(2, (1.0, -1.0));
    let mut c = plate(2, (1.0, -1.0));
    // J along r: r = (2, 3) x J = (4, 6) -> r x J = 2*6 - 3*4 = 0
    a.apply_impulse_at_point(fv(4.0, 6.0), fv(3.0, 2.0));
    c.apply_impulse(fv(4.0, 6.0));
    assert_eq!(a.velocity, c.velocity);
    close(a.angular_velocity, 0.0);
    // at the centre itself: r = 0
    let mut d = plate(2, (1.0, -1.0));
    d.apply_impulse_at_point(fv(4.0, 6.0), fv(1.0, -1.0));
    assert_eq!(d.velocity, c.velocity);
    assert_eq!(d.angular_velocity, Fix128::ZERO);
}

#[test]
fn angular_momentum_about_the_application_point_is_conserved() {
    for (m, pos, j, p) in [
        (3i64, (0.0, 0.0), (2.0, -5.0), (1.5, 0.5)),
        (1, (4.0, 2.0), (-7.0, 1.0), (3.0, 6.0)),
        (5, (-2.0, 3.0), (0.5, 0.25), (-2.5, 1.0)),
    ] {
        let mut b = plate(m, pos);
        let inertia = 1.0 / b.inv_inertia.to_f64();
        b.apply_impulse_at_point(fv(j.0, j.1), fv(p.0, p.1));
        let dv = (b.velocity.x.to_f64(), b.velocity.y.to_f64());
        let dw = b.angular_velocity.to_f64();
        // dL_p = I dw + (c - p) x (m dv)
        let cx = pos.0 - p.0;
        let cy = pos.1 - p.1;
        let dl =
            inertia * dw + cx * (f64::from(m as i32) * dv.1) - cy * (f64::from(m as i32) * dv.0);
        assert!(dl.abs() < 1e-9, "m={m}: dL_p = {dl}");
        // and dv = J / m
        assert!((dv.0 - j.0 / m as f64).abs() < 1e-12 && (dv.1 - j.1 / m as f64).abs() < 1e-12);
    }
}

#[test]
fn force_integrates_over_dt_without_spin() {
    let mut b = plate(3, (0.0, 0.0));
    b.apply_force(fv(6.0, -3.0), Fix128::from_ratio(1, 2));
    close(b.velocity.x, 1.0);
    close(b.velocity.y, -0.5);
    close(b.angular_velocity, 0.0);
    b.apply_force(fv(6.0, -3.0), Fix128::from_int(2)); // accumulates
    close(b.velocity.x, 5.0);
    close(b.velocity.y, -2.5);
    // zero dt / zero force change nothing
    let before = b.velocity;
    b.apply_force(fv(100.0, 100.0), Fix128::ZERO);
    b.apply_force(Vec2Fix::ZERO, Fix128::ONE);
    assert_eq!(b.velocity, before);
    assert_eq!(b.position, Vec2Fix::ZERO);
}

#[test]
fn static_and_kinematic_bodies_ignore_impulses_and_forces() {
    for mut b in [
        RigidBody2D::new_static(fv(1.0, 1.0), square(1)),
        RigidBody2D::new_kinematic(fv(1.0, 1.0), square(1)),
    ] {
        assert_ne!(b.body_type, BodyType2D::Dynamic);
        b.apply_impulse_at_point(fv(5.0, 5.0), fv(2.0, 0.0));
        b.apply_force(fv(5.0, 5.0), Fix128::ONE);
        assert_eq!(b.velocity, Vec2Fix::ZERO);
        assert_eq!(b.angular_velocity, Fix128::ZERO);
    }
    // a dynamic body with zero mass has zero inverse mass: impulses are absorbed
    let mut z = RigidBody2D::new_dynamic(Vec2Fix::ZERO, Fix128::ZERO, square(1));
    z.apply_impulse_at_point(fv(5.0, 5.0), fv(2.0, 0.0));
    z.apply_force(fv(5.0, 5.0), Fix128::ONE);
    assert_eq!(
        (z.velocity, z.angular_velocity),
        (Vec2Fix::ZERO, Fix128::ZERO)
    );
}

#[test]
fn circle_inertia_gives_the_textbook_spin() {
    // disc m = 2, r = 3: I = m r^2 / 2 = 9; tangential kick J = 6 at the rim: dw = r J / I = 2
    let mut b = RigidBody2D::new_dynamic(
        Vec2Fix::ZERO,
        Fix128::from_int(2),
        Shape2D::Circle {
            radius: Fix128::from_int(3),
        },
    );
    b.apply_impulse_at_point(fv(0.0, 6.0), fv(3.0, 0.0));
    close(b.angular_velocity, 2.0);
    close(b.velocity.y, 3.0);
    // the rim velocity w x r + v at the kick point: 2 * 3 + 3 = 9 = J (1/m + r^2/I) = 6 (1/2 + 1) = 9
    close(b.velocity.y + b.angular_velocity * Fix128::from_int(3), 9.0);
}

#[test]
fn distance_is_the_euclidean_norm_of_the_difference() {
    let a = fv(1.0, 2.0);
    let b = fv(4.0, 6.0);
    close(a.distance_to(b), 5.0);
    assert_eq!(a.distance_to(b), b.distance_to(a));
    assert_eq!(a.distance_to(a), Fix128::ZERO);
    close(fv(-2.0, -3.0).distance_to(fv(10.0, 2.0)), 13.0); // (12, 5)
    close(Vec2Fix::ZERO.distance_to(fv(0.0, -7.0)), 7.0);
    close(fv(0.0, 0.0).distance_to(fv(1.0, 1.0)), 2f64.sqrt());
}

#[test]
fn body_type_not_inverse_mass_decides_whether_forces_apply() {
    // pub fields allow a static / kinematic body with a stray inverse mass: it must still ignore forces
    for mut b in [
        RigidBody2D::new_static(fv(1.0, 1.0), square(1)),
        RigidBody2D::new_kinematic(fv(1.0, 1.0), square(1)),
    ] {
        b.inv_mass = Fix128::ONE;
        b.inv_inertia = Fix128::ONE;
        b.apply_impulse_at_point(fv(5.0, 5.0), fv(2.0, 0.0));
        b.apply_force(fv(5.0, 5.0), Fix128::ONE);
        assert_eq!(
            (b.velocity, b.angular_velocity),
            (Vec2Fix::ZERO, Fix128::ZERO)
        );
    }
}

#[test]
fn rectangle_inertia_through_the_polygon_formula() {
    // 4 x 2 rectangle centred at the origin: I = m (w^2 + h^2) / 12 = 3 * 20 / 12 = 5
    let (a, b) = (Fix128::from_int(2), Fix128::ONE);
    let rect = Shape2D::Polygon {
        vertices: vec![
            Vec2Fix::new(-a, -b),
            Vec2Fix::new(a, -b),
            Vec2Fix::new(a, b),
            Vec2Fix::new(-a, b),
        ],
    };
    let mut body = RigidBody2D::new_dynamic(Vec2Fix::ZERO, Fix128::from_int(3), rect);
    close(body.inv_inertia, 0.2);
    body.apply_impulse_at_point(fv(0.0, 1.0), fv(2.0, 0.0)); // r x J = 2
    close(body.angular_velocity, 0.4);
}

#[test]
fn triangle_inertia_through_the_polygon_formula() {
    // right isosceles triangle, legs 3, centroid at the origin: I = m a^2 / 18 * 2 = m (Jz / A = 1)
    let tri = Shape2D::Polygon {
        vertices: vec![fv(-1.0, -1.0), fv(2.0, -1.0), fv(-1.0, 2.0)],
    };
    let mut body = RigidBody2D::new_dynamic(Vec2Fix::ZERO, Fix128::from_int(3), tri);
    close(body.inv_inertia, 1.0 / 3.0);
    body.apply_impulse_at_point(fv(0.0, 3.0), fv(1.0, 0.0)); // r x J = 3
    close(body.angular_velocity, 1.0);
}
