//! World sweeps example
//!
//! `PhysicsWorld::time_of_impact`: a ball thrown at a sphere body and dropped
//! onto a ground plane, reported as the fraction of the step at first contact.
//! `PhysicsWorld::move_sdf_character`: an `SdfCharacter` falling onto an SDF
//! floor and walking into an SDF wall, both colliders in the world.
//! Each printed value is compared with its closed form (written from the scene,
//! not computed by the function under test).
//!
//! ```bash
//! cargo run --example world_sweeps
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn check(label: &str, got: f64, want: f64, tol: f64) {
    let ok = (got - want).abs() <= tol;
    println!(
        "{label:<40} {got:.6} closed form {want:.6} {}",
        if ok { "ok" } else { "MISMATCH" }
    );
    assert!(ok, "{label}: {got} vs {want}");
}

fn time_of_impact() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    world.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    let ball = world.add_body_with_radius(
        RigidBody::new(v3(0.0, 2.0, 0.0), Fix128::ONE),
        Fix128::from_f64(0.5),
    );
    world.add_body_with_radius(
        RigidBody::new_static(v3(3.0, 2.0, 0.0)),
        Fix128::from_f64(0.25),
    );
    let filter = RayFilter::default();

    // Gap 3 − 0.5 − 0.25 = 2.25 over a travel of 3 · 1.5 = 4.5: t = 0.5.
    let hit = world
        .time_of_impact(ball, v3(3.0, 0.0, 0.0), Fix128::from_f64(1.5), &filter)
        .expect("the ball reaches the sphere");
    check("ball vs sphere: t", hit.t.to_f64(), 0.5, 1e-12);
    check(
        "ball vs sphere: contact x",
        hit.point.x.to_f64(),
        2.75,
        1e-12,
    );

    // Height 2 − radius 0.5 = 1.5 over a fall of 3 · 1: t = 0.5.
    let hit = world
        .time_of_impact(ball, v3(0.0, -3.0, 0.0), Fix128::ONE, &filter)
        .expect("the ball reaches the ground");
    check("ball vs ground: t", hit.t.to_f64(), 0.5, 1e-12);
    check(
        "ball vs ground: normal y",
        hit.normal.y.to_f64(),
        1.0,
        1e-12,
    );

    // Moving away: no contact in the step.
    let miss = world.time_of_impact(ball, v3(-3.0, 0.0, 0.0), Fix128::ONE, &filter);
    println!("{:<40} {miss:?} closed form None", "ball moving away");
    assert!(miss.is_none());
}

#[cfg(feature = "std")]
fn sdf_character() {
    use alice_physics::math::QuatFix;
    use alice_physics::sdf_character::SdfCharacter;
    use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    // Solid below y = 0 and beyond x = 5.
    world.add_sdf_collider(SdfCollider::new_static(
        Box::new(ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    world.add_sdf_collider(SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, _y, _z| 5.0 - x,
            |_x, _y, _z| (-1.0, 0.0, 0.0),
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));

    let mut ch = SdfCharacter::new([0.0, 3.0, 0.0], 0.35, 1.8);
    let dt = 1.0 / 60.0;
    for _ in 0..240 {
        ch.apply_gravity([0.0, -9.81, 0.0], dt);
        world.move_sdf_character(&mut ch, dt, [0.0, 0.0, 0.0]);
    }
    // Resting at radius + skin above the floor.
    let rest = f64::from(ch.radius + ch.skin_width);
    check(
        "character on the floor: y",
        f64::from(ch.position[1]),
        rest,
        1e-5,
    );

    // Walk 6 m toward the wall: stops radius + skin short of x = 5.
    world.move_sdf_character(&mut ch, dt, [6.0, 0.0, 0.0]);
    check(
        "character at the wall: x",
        f64::from(ch.position[0]),
        5.0 - rest,
        1e-5,
    );
}

#[cfg(not(feature = "std"))]
fn sdf_character() {
    println!("SDF colliders need the `std` feature");
}

fn main() {
    time_of_impact();
    sdf_character();
}
