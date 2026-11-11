//! A box set down flat on a floor comes to rest flat, on both backends and at
//! every step rate.
//!
//! oracle (statics): a box on its face is in equilibrium with the floor's
//! reaction spread over the face; nothing makes it move or tip, so it ends at
//! rest (`v = 0`, `ω = 0`) with its face level (`tilt = 0`), whatever the step.
//! The tolerances are one substep of the forces involved: the speed below
//! `g h` (`h` the substep, what gravity adds between two contact solves), every
//! point of the box below `g h` by turning (`|ω| r_max`), the tilt below 1 mrad.
//! The same box at 240, 480 and 960 Hz ends in the same state within those
//! bounds (a limit cycle depends on the step, rest does not).
//!
//! What it catches: a floor that touches the box at one point per tick (its
//! deepest corner) leaves the box turning about that corner for the whole tick;
//! the next tick another corner is deepest, so the box rocks from corner to
//! corner and never settles (measured on the temporal Gauss–Seidel backend:
//! 0.19 m/s and 0.4 rad/s at 240 Hz, the tilt flipping each tick).

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 acos of the tilt of the box's up axis"
)]

use alice_physics_core::heightfield::HeightField;
use alice_physics_core::material::{PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverBackend, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::trimesh::{TriMesh, Triangle};

const G: f64 = 10.0;
const HALF: (f64, f64, f64) = (0.45, 0.06, 0.45);

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

#[derive(Clone, Copy, Debug)]
enum Floor {
    Plane,
    Mesh,
    HeightField,
    StaticBox,
    Sdf,
}

const FLOORS: [Floor; 5] = [
    Floor::Plane,
    Floor::Mesh,
    Floor::HeightField,
    Floor::StaticBox,
    Floor::Sdf,
];

fn install(w: &mut PhysicsWorld, floor: Floor) {
    let (a, b, c, d) = (
        v3(-20.0, 0.0, -20.0),
        v3(20.0, 0.0, -20.0),
        v3(20.0, 0.0, 20.0),
        v3(-20.0, 0.0, 20.0),
    );
    match floor {
        Floor::Plane => {
            w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
                v3(0.0, 1.0, 0.0),
                Fix128::ZERO,
            )));
        }
        Floor::Mesh => {
            w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_triangles(vec![
                Triangle::new(a, c, b),
                Triangle::new(a, d, c),
            ])));
        }
        Floor::HeightField => {
            w.add_static_collider(StaticCollider::HeightField(HeightField::flat(
                41,
                41,
                fx(1.0),
                v3(-20.0, 0.0, -20.0),
                Fix128::ZERO,
            )));
        }
        Floor::StaticBox => {
            let f = w.add_body(RigidBody::new_static(v3(0.0, -1.0, 0.0)));
            w.set_body_shape(
                f,
                &Shape::Box {
                    half_extents: v3(40.0, 1.0, 40.0),
                },
            );
        }
        Floor::Sdf => {
            let field = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
            w.add_sdf_collider(SdfCollider::new_static(
                Box::new(field),
                Vec3Fix::ZERO,
                QuatFix::IDENTITY,
            ));
        }
    }
}

/// The state of a box set down flat (resting on its face, no speed) after
/// `seconds` at `hz`: speed, fastest point speed by turning, tilt (rad).
fn settle(backend: SolverBackend, floor: Floor, hz: f64, seconds: f64) -> (f64, f64, f64, f64) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -G, 0.0),
        solver_backend: backend,
        ..SolverConfig::default()
    });
    install(&mut w, floor);
    let m = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(0.4), Fix128::ZERO));
    w.material_table
        .set_pair_override(m, DEFAULT_MATERIAL, fx(0.4), Fix128::ZERO);
    let shape = Shape::Box {
        half_extents: v3(HALF.0, HALF.1, HALF.2),
    };
    let b = w
        .add_shaped_body(&shape, fx(500.0), v3(0.0, HALF.1, 0.0))
        .expect("box");
    w.set_body_material(b, m);
    let dt = 1.0 / hz;
    for _ in 0..(seconds * hz).round() as usize {
        w.step(fx(dt));
    }
    let h = dt / w.config.substeps as f64;
    let body = w.get_body(b).expect("box");
    let r_max = (HALF.0 * HALF.0 + HALF.1 * HALF.1 + HALF.2 * HALF.2).sqrt();
    let up = body.rotation.rotate_vec(v3(0.0, 1.0, 0.0));
    (
        body.velocity.length().to_f64(),
        body.angular_velocity.length().to_f64() * r_max,
        up.y.to_f64().clamp(-1.0, 1.0).acos(),
        G * h,
    )
}

fn check(backend: SolverBackend, floors: &[Floor]) {
    let mut wrong = Vec::new();
    let mut compared = 0;
    for &floor in floors {
        for hz in [240.0, 480.0, 960.0] {
            let (speed, turn, tilt, bound) = settle(backend, floor, hz, 1.0);
            compared += 1;
            if speed > bound || turn > bound || tilt > 1e-3 {
                wrong.push(format!(
                    "{backend:?} {floor:?} {hz} Hz: speed {speed:.3e}, turning {turn:.3e} \
                     (bound g h = {bound:.3e}), tilt {tilt:.3e} rad (bound 1e-3)"
                ));
            }
        }
    }
    assert_eq!(compared, floors.len() * 3);
    assert!(
        wrong.is_empty(),
        "{} of {compared}:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The plane-like floors (plane, triangle mesh, height field, a static box).
#[test]
fn a_box_set_down_flat_rests_flat_on_plane_like_floors_under_tgs() {
    check(
        SolverBackend::Tgs,
        &[
            Floor::Plane,
            Floor::Mesh,
            Floor::HeightField,
            Floor::StaticBox,
        ],
    );
}

/// The SDF floor (the half-space `y ≤ 0`).
#[test]
fn a_box_set_down_flat_rests_flat_on_an_sdf_floor_under_tgs() {
    check(SolverBackend::Tgs, &[Floor::Sdf]);
}

/// The position-based backend, which detects every substep, on every floor
/// (it must stay at rest when the floor gives several points).
#[test]
fn a_box_set_down_flat_rests_flat_on_every_floor_under_xpbd() {
    check(SolverBackend::Xpbd, &FLOORS);
}
