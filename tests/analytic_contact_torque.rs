//! Oracles for the turning a contact gives a shaped body resting on or falling
//! onto a static floor.
//!
//! # What is measured
//!
//! - **A face at rest stays at rest.** A box set on a face of its own on a level
//!   floor neither turns nor drifts: the floor holds every corner of that face, so
//!   the contact's torques about the centre of mass cancel.
//! - **A domino tips iff its centre of mass is past the edge.** A box of width `w`
//!   (along `x`) and height `h` (along `y`) stood on its bottom edge and tilted by
//!   `θ` about it has its centre of mass over the edge exactly at `tan θ = w / h`.
//!   Below that angle gravity's torque about the edge turns it back onto its face
//!   (it ends upright, centre of mass `h / 2` up); above it, the torque turns it
//!   over (it ends on its long side, centre of mass `w / 2` up). The angles tried
//!   are the threshold `± 0.05 rad`.
//! - **Replay.** The same scene stepped twice gives the same bits.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the tipping threshold atan(w / h) and the tilted pose's sin / cos are the closed form"
)]

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn floor_world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -10.0, 0.0),
        ..SolverConfig::default()
    });
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )));
    w
}

/// A box set on its face stays where it is, turning not at all.
#[test]
fn a_box_on_its_face_neither_turns_nor_drifts() {
    let mut w = floor_world();
    let shape = Shape::Box {
        half_extents: v3(0.3, 0.1, 0.2),
    };
    let b = w
        .add_shaped_body(&shape, fx(500.0), v3(1.0, 0.1, -2.0))
        .expect("box");
    for _ in 0..600 {
        w.step(fx(1.0 / 60.0));
    }
    let body = w.get_body(b).expect("body");
    let p = [
        body.position.x.to_f64(),
        body.position.y.to_f64(),
        body.position.z.to_f64(),
    ];
    let spin = body.angular_velocity.length().to_f64();
    let r = body.rotation;
    let turned = (1.0 - r.w.to_f64().abs()).abs();
    assert!(
        (p[0] - 1.0).abs() < 1e-6 && (p[1] - 0.1).abs() < 1e-6 && (p[2] + 2.0).abs() < 1e-6 && spin < 1e-6 && turned < 1e-9,
        "after 10 s at rest on a face: position {p:?} (want [1, 0.1, -2]), spin {spin:.3e}, rotation {r:?}"
    );
}

/// The height of a domino's centre of mass after it settles, when it is stood on
/// its bottom `+x` edge (at the origin) tilted by `theta` toward `+x`.
fn domino_settles_at(theta: f64) -> f64 {
    let (w, h, d) = (0.1, 0.5, 0.3);
    let mut world = floor_world();
    let shape = Shape::Box {
        half_extents: v3(w / 2.0, h / 2.0, d / 2.0),
    };
    // from the edge to the centre of mass: (−w/2, h/2) in the domino's frame,
    // turned by −θ about z
    let (s, c) = theta.sin_cos();
    let com = v3(-w / 2.0 * c + h / 2.0 * s, w / 2.0 * s + h / 2.0 * c, 0.0);
    let b = world
        .add_shaped_body(&shape, fx(500.0), com)
        .expect("domino");
    world
        .get_body_mut(b)
        .expect("body")
        .set_rotation(QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), fx(-theta)));
    for _ in 0..600 {
        world.step(fx(1.0 / 60.0));
    }
    world.get_body(b).expect("body").position.y.to_f64()
}

#[test]
fn a_domino_tips_over_iff_its_centre_of_mass_is_past_the_edge() {
    let (w, h) = (0.1_f64, 0.5_f64);
    let threshold = (w / h).atan();
    let below = domino_settles_at(threshold - 0.05);
    let above = domino_settles_at(threshold + 0.05);
    assert!(
        (below - h / 2.0).abs() < 1e-3,
        "tilted {:.4} rad (below the threshold {threshold:.4}): settles with its centre of mass at {below:.4}, want upright at {:.4}",
        threshold - 0.05,
        h / 2.0
    );
    assert!(
        (above - w / 2.0).abs() < 1e-3,
        "tilted {:.4} rad (past the threshold {threshold:.4}): settles with its centre of mass at {above:.4}, want on its side at {:.4}",
        threshold + 0.05,
        w / 2.0
    );
}

#[test]
fn a_tipping_domino_replays_to_the_same_bits() {
    let run = || {
        let mut world = floor_world();
        let shape = Shape::Box {
            half_extents: v3(0.05, 0.25, 0.15),
        };
        let b = world
            .add_shaped_body(&shape, fx(500.0), v3(0.1, 0.24, 0.0))
            .expect("domino");
        world
            .get_body_mut(b)
            .expect("body")
            .set_rotation(QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), fx(-0.4)));
        for _ in 0..180 {
            world.step(fx(1.0 / 60.0));
        }
        let body = world.get_body(b).expect("body");
        format!(
            "{:?} {:?} {:?} {:?}",
            body.position, body.rotation, body.velocity, body.angular_velocity
        )
    };
    assert_eq!(run(), run());
}
