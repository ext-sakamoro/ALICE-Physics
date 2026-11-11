//! A ball on the top of a curved static solid: the contact normal is exact, so a
//! ball exactly above the apex stays there, and one set off it rolls off on its
//! own side.
//!
//! oracle (statics and the sign of the tipping torque): at the apex of an
//! ellipsoid or on the top of a torus's tube the surface normal is vertical, so
//! gravity and the contact force are collinear and the ball is in equilibrium
//! (unstable: any offset grows). Offset by `δ` along `x`, the normal leans
//! towards `+x · sign(δ)`, the reaction has a horizontal part of that sign, and
//! the ball rolls off that way (its centre's `x` grows with the sign of `δ`).
//!
//! What it catches: a normal from an iterative search (GJK/EPA) that wobbles
//! about `1e-3` off vertical at the apex kicks the ball off it.

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// The solids and the height of a ball of radius 0.5 resting on top, with the
/// ball's start above the top (`x` of the top for the torus).
fn solids() -> Vec<(&'static str, Shape, f64, f64)> {
    vec![
        (
            "ellipsoid",
            Shape::Ellipsoid {
                radii: Vec3Fix::new(fx(1.5), fx(0.7), fx(1.1)),
            },
            0.0,
            0.7 + 0.5,
        ),
        (
            "torus",
            Shape::Torus {
                major_radius: fx(1.0),
                minor_radius: fx(0.3),
            },
            1.0,
            0.3 + 0.5,
        ),
    ]
}

/// The ball's centre after `frames` frames, dropped from 0.2 above its rest
/// height at `x = x_top + offset`.
fn after(shape: &Shape, x_top: f64, rest: f64, offset: f64, frames: usize) -> (f64, f64) {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let k = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    assert!(w.set_body_shape(k, shape));
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(fx(x_top + offset), fx(rest + 0.2), Fix128::ZERO),
            Fix128::ONE,
        ),
        fx(0.5),
    );
    for _ in 0..frames {
        w.step(fx(1.0 / 60.0));
    }
    let p = w.get_body(b).expect("ball").position;
    (p.x.to_f64() - x_top, p.y.to_f64())
}

/// Exactly above the top it comes to rest there (5 s), within `5e-3` m.
#[test]
fn a_ball_dropped_on_the_apex_stays_on_it() {
    let mut wrong = Vec::new();
    for (name, shape, x_top, rest) in solids() {
        let (x, y) = after(&shape, x_top, rest, 0.0, 300);
        if x.abs() > 5e-3 || (y - rest).abs() > 5e-3 {
            wrong.push(format!(
                "{name}: x off the top {x:.4e}, y {y:.4} (want {rest})"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Set off the top by `±0.02` it rolls off on that side: after 5 s its centre is
/// more than 0.1 m off the top with the sign of the offset.
#[test]
fn a_ball_dropped_off_the_apex_rolls_off_on_its_side() {
    let mut wrong = Vec::new();
    for (name, shape, x_top, rest) in solids() {
        for offset in [0.02, -0.02] {
            let (x, _) = after(&shape, x_top, rest, offset, 300);
            if x * offset.signum() <= 0.1 || x.is_nan() {
                wrong.push(format!("{name} offset {offset}: ends {x:.4} off the top"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
