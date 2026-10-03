//! Deformable cube dropped on a rigid sphere:
//! `DeformableBody::{new_cube, center_of_mass, resolve_rigid_body_collisions}`.
//!
//! A 4 kg cube (half-extent 0.5) falls from `y = 3` onto a 2 kg free sphere of radius 1 at the origin.
//! Each frame the cube steps and its particles are resolved against the sphere, which receives the
//! reaction (shared by inverse mass; perfectly inelastic along the contact normal). The sphere is only
//! moved by these reactions here (no rigid-body step), so it is pushed down by the landing cube.
//! `tests/analytic_deformable_wiring.rs` checks the split and the momentum balance in closed form.
//!
//! ```bash
//! cargo run --release --example deformable_cube_impact --features std
//! ```

use alice_physics::deformable::DeformableBody;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

fn main() {
    let mut cube = DeformableBody::new_cube(
        Vec3Fix::from_int(0, 3, 0),
        Fix128::from_ratio(1, 2),
        Fix128::from_int(4),
    );
    cube.config.damping = Fix128::ONE;
    let mut sphere = [RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2))];
    sphere[0].gravity_scale = Fix128::ZERO; // hold the sphere in place until the cube lands
    let radius = [Fix128::ONE];
    let dt = Fix128::from_ratio(1, 60);

    println!(
        "start: cube centre y = {:.3}, fresh cube centre = (0, 3, 0)",
        cube.center_of_mass().y.to_f64()
    );
    for frame in 0..60 {
        cube.step(dt);
        cube.resolve_rigid_body_collisions(&mut sphere, &radius, dt);
        if frame % 10 == 9 {
            let c = cube.center_of_mass();
            println!(
                "frame {:2}: cube y = {:.3}, sphere y = {:.4}, sphere vy = {:.3}",
                frame + 1,
                c.y.to_f64(),
                sphere[0].position.y.to_f64(),
                sphere[0].velocity.y.to_f64()
            );
        }
    }
}
