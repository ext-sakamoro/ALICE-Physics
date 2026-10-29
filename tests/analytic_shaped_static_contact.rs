//! Oracles for shaped bodies against static colliders, and for overlap that a
//! body starts with.
//!
//! # What is measured
//!
//! - **Rest height.** A convex solid dropped on a level floor comes to rest with
//!   its lowest point on the floor: its centre of mass ends at the solid's support
//!   distance below it, `h = max_{p ∈ solid} (c − p)·up`. For each shape and
//!   orientation below that is a closed form of its dimensions (a box on a face:
//!   the half-extent along up; a cylinder on its end: the half-height, on its
//!   side: the radius; a cone on its base: `half_height / 2`, its centre of mass
//!   being `half_height / 2` below the geometric centre; a wedge on its base:
//!   `height / 3`; an ellipsoid: the semi-axis along up; a torus lying flat: the
//!   tube radius). The floor is a plane, a triangle mesh and a flat height field,
//!   the three kinds of static collider, and the answer is the same on each.
//! - **No launch from overlap.** A body that starts overlapping a static surface
//!   is moved out of it, and that correction injects no velocity: the overlap was
//!   there before the substep, so moving out of it is not motion the body made.
//!   Only the part of the correction that cancels the body's approach this
//!   substep changes the velocity. A body set down overlapping a floor by δ with
//!   no gravity and no velocity ends on the floor at rest.
//! - **Replay.** The same scene stepped twice gives the same bits.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::heightfield::HeightField;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::trimesh::{TriMesh, Triangle};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn config(gravity: f64) -> SolverConfig {
    SolverConfig {
        gravity: v3(0.0, gravity, 0.0),
        ..SolverConfig::default()
    }
}

/// The three kinds of static floor at `y = 0`, facing up
fn floors() -> Vec<(&'static str, StaticCollider)> {
    let (a, b, c, d) = (
        v3(-10.0, 0.0, -10.0),
        v3(10.0, 0.0, -10.0),
        v3(10.0, 0.0, 10.0),
        v3(-10.0, 0.0, 10.0),
    );
    vec![
        (
            "plane",
            StaticCollider::Plane(PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(0.0))),
        ),
        (
            "mesh",
            // counter-clockwise seen from above, so the faces point up
            StaticCollider::TriMesh(TriMesh::from_triangles(vec![
                Triangle::new(a, c, b),
                Triangle::new(a, d, c),
            ])),
        ),
        (
            "height field",
            StaticCollider::HeightField(HeightField::flat(
                21,
                21,
                fx(1.0),
                v3(-10.0, 0.0, -10.0),
                fx(0.0),
            )),
        ),
    ]
}

fn quarter_turn_about_z() -> QuatFix {
    QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), fx(core::f64::consts::FRAC_PI_2))
}

/// (what, shape, orientation, the rest height of its centre of mass, closed form)
fn resting_solids() -> Vec<(&'static str, Shape, QuatFix, f64)> {
    let (r, hh) = (0.25, 0.3);
    vec![
        (
            "cube on a face",
            Shape::Box {
                half_extents: v3(0.25, 0.25, 0.25),
            },
            QuatFix::IDENTITY,
            0.25,
        ),
        (
            "flat pad on its face",
            Shape::Box {
                half_extents: v3(0.45, 0.06, 0.45),
            },
            QuatFix::IDENTITY,
            0.06,
        ),
        (
            "box on its long side (turned about z: its x half-extent is up)",
            Shape::Box {
                half_extents: v3(0.4, 0.1, 0.2),
            },
            quarter_turn_about_z(),
            0.4,
        ),
        (
            "cylinder on its end",
            Shape::Cylinder {
                radius: fx(r),
                half_height: fx(hh),
            },
            QuatFix::IDENTITY,
            hh,
        ),
        (
            "cylinder on its side",
            Shape::Cylinder {
                radius: fx(r),
                half_height: fx(hh),
            },
            quarter_turn_about_z(),
            r,
        ),
        (
            "cone on its base",
            Shape::Cone {
                radius: fx(r),
                half_height: fx(hh),
            },
            QuatFix::IDENTITY,
            hh / 2.0,
        ),
        (
            "ellipsoid",
            Shape::Ellipsoid {
                radii: v3(0.3, 0.2, 0.25),
            },
            QuatFix::IDENTITY,
            0.2,
        ),
        (
            "wedge on its base",
            Shape::Wedge {
                width: fx(0.6),
                height: fx(0.45),
                depth: fx(0.3),
            },
            QuatFix::IDENTITY,
            0.45 / 3.0,
        ),
        (
            "torus lying flat",
            Shape::Torus {
                major_radius: fx(0.3),
                minor_radius: fx(0.08),
            },
            QuatFix::IDENTITY,
            0.08,
        ),
    ]
}

fn rest_height(floor: StaticCollider, shape: &Shape, rotation: QuatFix, h: f64) -> (f64, f64) {
    let mut w = PhysicsWorld::new(config(-10.0));
    w.add_static_collider(floor);
    let b = w
        .add_shaped_body(shape, fx(500.0), v3(0.0, h + 0.2, 0.0))
        .expect("a valid shape");
    w.get_body_mut(b).expect("body").set_rotation(rotation);
    for _ in 0..600 {
        w.step(fx(1.0 / 60.0));
    }
    let body = w.get_body(b).expect("body");
    (body.position.y.to_f64(), body.velocity.length().to_f64())
}

#[test]
fn shaped_bodies_rest_at_their_support_distance_on_every_kind_of_floor() {
    let mut wrong = Vec::new();
    let mut compared = 0;
    for (what, shape, rotation, h) in resting_solids() {
        for (floor_name, floor) in floors() {
            let (y, speed) = rest_height(floor, &shape, rotation, h);
            compared += 1;
            if (y - h).abs() > 1e-6 || speed > 1e-6 {
                wrong.push(format!(
                    "{what} on the {floor_name}: rests at {y:.6} (closed form {h:.6}, the bounding sphere would give more), speed {speed:.3e}"
                ));
            }
        }
    }
    assert_eq!(compared, 27, "9 solids x 3 floors");
    assert!(
        wrong.is_empty(),
        "{} of {compared}:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// One step of a zero-gravity world: (the body's height after one step, its speed
/// after one step, its height after two seconds, its speed after two seconds)
fn after_overlap(floor: StaticCollider, add: impl FnOnce(&mut PhysicsWorld) -> usize) -> [f64; 4] {
    let mut w = PhysicsWorld::new(config(0.0));
    w.add_static_collider(floor);
    let b = add(&mut w);
    w.step(fx(1.0 / 60.0));
    let one = w.get_body(b).expect("body");
    let (y1, v1) = (one.position.y.to_f64(), one.velocity.length().to_f64());
    for _ in 0..119 {
        w.step(fx(1.0 / 60.0));
    }
    let two = w.get_body(b).expect("body");
    [
        y1,
        v1,
        two.position.y.to_f64(),
        two.velocity.length().to_f64(),
    ]
}

/// The separation of a body that starts overlapping is not motion: a sphere set
/// 0.1 into a floor (no shape involved) is moved out and left at rest. This is the
/// velocity half of the launch, measured apart from the shape half.
#[test]
fn a_sphere_set_into_a_floor_is_moved_out_and_left_at_rest() {
    for (floor_name, floor) in floors() {
        let [y1, v1, y2, v2] = after_overlap(floor, |w| {
            w.add_body_with_radius(
                RigidBody::new_dynamic(v3(0.0, 0.4, 0.0), Fix128::ONE),
                fx(0.5),
            )
        });
        assert!(
            (y1 - 0.5).abs() <= 1e-9 && v1 <= 1e-9 && (y2 - 0.5).abs() <= 1e-9 && v2 <= 1e-9,
            "{floor_name}: overlap 0.1 at rest: after one step y {y1} speed {v1}, after 2 s y {y2} speed {v2} (want y 0.5, speed 0: the overlap is not motion)"
        );
    }
}

/// A box set 0.1 into a floor ends on its face at rest, on every kind of floor.
#[test]
fn a_box_set_into_a_floor_is_moved_out_and_left_at_rest() {
    let shape = Shape::Box {
        half_extents: v3(0.25, 0.25, 0.25),
    };
    for (floor_name, floor) in floors() {
        let [y1, v1, y2, v2] = after_overlap(floor, |w| {
            w.add_shaped_body(&shape, fx(500.0), v3(0.0, 0.15, 0.0))
                .expect("box")
        });
        assert!(
            (y1 - 0.25).abs() <= 1e-9 && v1 <= 1e-9 && (y2 - 0.25).abs() <= 1e-9 && v2 <= 1e-9,
            "{floor_name}: cube (half 0.25) set at 0.15: after one step y {y1} speed {v1}, after 2 s y {y2} speed {v2} (want y 0.25, speed 0)"
        );
    }
}

/// The same scene stepped twice ends in the same bits.
#[test]
fn a_shaped_scene_replays_to_the_same_bits() {
    let run = || {
        let mut w = PhysicsWorld::new(config(-10.0));
        for (_, floor) in floors() {
            w.add_static_collider(floor);
        }
        let mut ids = Vec::new();
        for (k, (_, shape, rotation, h)) in resting_solids().into_iter().enumerate() {
            let x = -8.0 + 2.0 * k as f64;
            let b = w
                .add_shaped_body(&shape, fx(500.0), v3(x, h + 0.05 * k as f64, 0.0))
                .expect("shape");
            w.get_body_mut(b).expect("body").set_rotation(rotation);
            ids.push(b);
        }
        for _ in 0..240 {
            w.step(fx(1.0 / 60.0));
        }
        ids.iter()
            .map(|&b| {
                let body = w.get_body(b).expect("body");
                format!(
                    "{:?} {:?} {:?}",
                    body.position, body.velocity, body.rotation
                )
            })
            .collect::<Vec<_>>()
    };
    let (first, second) = (run(), run());
    assert_eq!(first.len(), 9);
    assert_eq!(first, second);
}
