//! World character controller example
//!
//! A `CharacterController` moved by `PhysicsWorld::move_character` over a
//! plane floor, into a box wall, and up a low ledge. Each printed value is
//! compared with its closed form (written from the scene, not computed by the
//! function under test).
//!
//! ```bash
//! cargo run --example world_character
//! ```

use alice_physics::character::CharacterController;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
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

fn show(label: &str, p: Vec3Fix, grounded: bool, want: [f64; 3]) {
    let got = [p.x.to_f64(), p.y.to_f64(), p.z.to_f64()];
    let ok = got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-8);
    println!(
        "{label:<28} ({:.6}, {:.6}, {:.6}) grounded={grounded} closed form {want:?} {}",
        got[0],
        got[1],
        got[2],
        if ok { "ok" } else { "MISMATCH" }
    );
    assert!(ok, "{label}: {got:?} vs {want:?}");
}

fn main() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    world.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    // a wall box with its face at x = 4
    let wall = world.add_body(RigidBody::new_static(v3(5.0, 2.0, 0.0)));
    world.set_body_shape(
        wall,
        &Shape::Box {
            half_extents: v3(1.0, 2.0, 10.0),
        },
    );
    // a ledge 0.2 high (below the default step height 0.3) with its face at x = 1, z ∈ [3, 7]
    let ledge = world.add_body(RigidBody::new_static(v3(2.0, 0.1, 5.0)));
    world.set_body_shape(
        ledge,
        &Shape::Box {
            half_extents: v3(1.0, 0.1, 2.0),
        },
    );

    // default config: height 1.8, radius 0.3, skin 0.01 -> standing centre y = 0.91
    let mut ctrl = CharacterController::new_default(v3(0.0, 1.5, 0.0));
    let filter = RayFilter::default();

    let r = world.move_character_with_filter(&mut ctrl, v3(1.0, -1.0, 0.0), &filter);
    show(
        "land on the plane",
        r.position,
        r.grounded,
        [1.0, 0.91, 0.0],
    );

    let r = world.move_character(&mut ctrl, v3(4.0, 0.0, 1.0));
    show(
        "slide along the wall",
        r.position,
        r.grounded,
        [4.0 - 0.31, 0.91, 1.0],
    );

    let mut climber = CharacterController::new_default(v3(0.0, 0.91, 5.0));
    world.move_character(&mut climber, Vec3Fix::ZERO);
    let r = world.move_character(&mut climber, v3(1.5, 0.0, 0.0));
    show(
        "step onto the ledge",
        r.position,
        r.grounded,
        [1.5, 1.11, 5.0],
    );
    println!(
        "ground body = {:?} (ledge {ledge}), platform velocity = {:?}",
        climber.ground_body_index,
        climber.get_platform_velocity()
    );
}
