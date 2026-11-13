//! An overlap that the dynamics did not make is pushed out without velocity.
//!
//! A cube (half size 1/2, density 1000 kg/m³, so 1000 kg) is let fall asleep on
//! the plane `y = 0`, then turned 30° about `z` from outside the step: through
//! the pose setter, and through a raw write of `bodies[..].rotation`. The turn
//! sinks its lowest edge `1/2 (cos 30° + sin 30°) − 1/2 ≈ 0.183 m` into the
//! floor. The previous substep's dynamics left no such overlap (the cube was at
//! rest), so the compliant solve must not take it: it is pushed out by position,
//! and the velocity the step ends with is what gravity gives in one step at most.
//!
//! Oracles (closed form, the scene's own numbers):
//!
//! * after one step `|v| ≤ g·dt` (an overlap solved by the compliant solve
//!   leaves `|v| ≈ depth / h`, 88 m/s here);
//! * the cube then topples back onto its face (its centre of mass is not over
//!   the edge it rests on after the push, since 30° < 45°), and comes to rest at
//!   height `1/2` with the face flat: the centre sinks into the floor by the
//!   compliant overlap of four corners carrying `m g` together,
//!   `(m g / 4) · 2 α` with `α = 1e-10 m/N` per side (each α a `Fix128`, so to
//!   its resolution `2⁻⁶⁴`), and the face tilts by at most that overlap across
//!   the side of 1 m.

use alice_physics_core::material::DEFAULT_CONTACT_COMPLIANCE;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;

const G: f64 = 10.0;
const DT: f64 = 1.0 / 60.0;
const MASS: f64 = 1000.0;

fn cube_asleep_on_a_floor() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::from_int(0, 1, 0),
        Fix128::ZERO,
    )));
    let half = Fix128::from_ratio(1, 2);
    let b = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: Vec3Fix::new(half, half, half),
            },
            Fix128::from_int(1000),
            Vec3Fix::new(Fix128::ZERO, half, Fix128::ZERO),
        )
        .expect("a 1 m cube of density 1000");
    assert_eq!(b, 0);
    for _ in 0..240 {
        w.step(Fix128::from_f64(DT));
    }
    assert!(w.is_sleeping(0), "the cube does not fall asleep");
    w
}

/// 30° about `z`: `(0, 0, sin 15°, cos 15°)`.
fn turn_30_about_z() -> QuatFix {
    QuatFix::new(
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::from_f64(0.258_819_045_102_520_74),
        Fix128::from_f64(0.965_925_826_289_068_3),
    )
    .unit_rotation()
}

fn speed(w: &PhysicsWorld) -> f64 {
    let v = w.bodies[0].velocity;
    let (x, y, z) = (v.x.to_f64(), v.y.to_f64(), v.z.to_f64());
    (x * x + y * y + z * z).sqrt()
}

#[test]
fn a_pose_set_into_the_floor_is_pushed_out_without_velocity() {
    let mut w = cube_asleep_on_a_floor();
    w.bodies[0].set_rotation(turn_30_about_z());
    w.wake_body(0);
    w.step(Fix128::from_f64(DT));
    let v = speed(&w);
    assert!(
        v <= G * DT,
        "|v| = {v} m/s after one step, more than g dt = {}",
        G * DT
    );
}

#[test]
fn a_raw_rotation_write_into_the_floor_is_pushed_out_without_velocity() {
    let mut w = cube_asleep_on_a_floor();
    // only `rotation`: `prev_rotation` keeps the pose the dynamics left
    w.bodies[0].rotation = turn_30_about_z();
    w.wake_body(0);
    w.step(Fix128::from_f64(DT));
    let v = speed(&w);
    assert!(
        v <= G * DT,
        "|v| = {v} m/s after one step, more than g dt = {}",
        G * DT
    );
}

#[test]
fn the_tilted_cube_topples_onto_its_face_and_rests_at_half_height() {
    let mut w = cube_asleep_on_a_floor();
    w.bodies[0].set_rotation(turn_30_about_z());
    w.wake_body(0);
    for _ in 0..600 {
        w.step(Fix128::from_f64(DT));
    }
    let alpha = DEFAULT_CONTACT_COMPLIANCE.to_f64();
    // each side's compliance is a Fix128 of resolution 2^-64, so the two summed
    // carry a relative error of at most 2 · 2^-64 / α (5e-10 here)
    let resolution = 1.0 / 18_446_744_073_709_551_616.0;
    let sink = MASS * G / 4.0 * 2.0 * alpha * (1.0 + 2.0 * resolution / alpha);
    let y = w.bodies[0].position.y.to_f64();
    assert!(
        (0.5 - y) >= -1e-12 && (0.5 - y) <= sink,
        "centre at {y}, rest height 0.5 sunk by at most {sink}"
    );
    // the tilt of the face: the angle of the cube's up axis from +y
    let up = w.bodies[0].rotation.rotate_vec(Vec3Fix::from_int(0, 1, 0));
    let (ux, uz) = (up.x.to_f64(), up.z.to_f64());
    let tilt = (ux * ux + uz * uz).sqrt();
    assert!(tilt <= sink, "face tilted by {tilt} rad, more than {sink}");
}
