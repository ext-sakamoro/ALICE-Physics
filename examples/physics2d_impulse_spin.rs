//! 2D rigid-body impulses and forces:
//! `RigidBody2D::{apply_impulse_at_point, apply_force}` and `Vec2Fix::distance_to`.
//!
//! A 3 kg square plate (half-extent 1, `I = m (w^2 + h^2) / 12 = 3 * 8 / 12 = 2`) is kicked with
//! `J = (0, 4)` at the world point `(6, 5)`; the centre is at `(5, 5)`, so `r = (1, 0)`,
//! `dv = J / m = (0, 4/3)`, `dw = (r x J) / I = 4 / 2 = 2`. A `F = (6, 0)` force over `dt = 0.5`
//! then adds `dv = F dt / m = (1, 0)`. The distance from the plate to the kick point is `1`.
//!
//! ```bash
//! cargo run --release --example physics2d_impulse_spin --features std
//! ```

use alice_physics::math::Fix128;
use alice_physics::physics2d::{RigidBody2D, Shape2D, Vec2Fix};

fn v(x: i64, y: i64) -> Vec2Fix {
    Vec2Fix::new(Fix128::from_int(x), Fix128::from_int(y))
}

fn main() {
    let one = Fix128::ONE;
    let square = Shape2D::Polygon {
        vertices: vec![
            Vec2Fix::new(-one, -one),
            Vec2Fix::new(one, -one),
            Vec2Fix::new(one, one),
            Vec2Fix::new(-one, one),
        ],
    };
    let mut plate = RigidBody2D::new_dynamic(v(5, 5), Fix128::from_int(3), square);
    let kick_point = v(6, 5);
    println!(
        "inverse inertia {:.4} (expect 0.5), lever arm {}",
        plate.inv_inertia.to_f64(),
        plate.position.distance_to(kick_point).to_f64()
    );
    plate.apply_impulse_at_point(v(0, 4), kick_point);
    println!(
        "after impulse: v = ({:.4}, {:.4}), w = {:.4}",
        plate.velocity.x.to_f64(),
        plate.velocity.y.to_f64(),
        plate.angular_velocity.to_f64()
    );
    plate.apply_force(v(6, 0), Fix128::from_ratio(1, 2));
    println!(
        "after force: v = ({:.4}, {:.4})",
        plate.velocity.x.to_f64(),
        plate.velocity.y.to_f64()
    );
}
