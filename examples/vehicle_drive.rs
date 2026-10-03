//! Four-wheel vehicle: settle on its suspension, then accelerate and shift.
//!
//! Static deflection per wheel is `c = m g / (4 k)` and the chassis rides at
//! `y = 0.2 + radius + rest (1 - c)` (wheels hang 0.2 m below the chassis
//! origin). With full throttle in gear `n` the drive force on the two driven
//! wheels is `max_torque * ratio_n / radius` and the frame's velocity change
//! `F dt / m`.
//!
//! ```bash
//! cargo run --release --example vehicle_drive --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::vehicle::{Vehicle, VehicleConfig};

fn main() {
    let mass = 1000.0_f64;
    let mut cfg = VehicleConfig::default();
    cfg.anti_roll_stiffness = Fix128::ZERO;
    let (k, rest, radius) = (50000.0_f64, 0.3_f64, 0.3_f64);
    let mut vehicle = Vehicle::new(cfg);
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let g = -world.config.gravity.y.to_f64();
    let car = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_f32(0.0, 0.79, 0.0),
        Fix128::from_int(mass as i64),
    ));
    let dt = Fix128::from_ratio(1, 60);

    for _ in 0..300 {
        vehicle.update(&mut world.bodies[car], dt);
        world.step(dt);
    }
    let c_ref = mass * g / (4.0 * k);
    let y_ref = 0.2 + radius + rest * (1.0 - c_ref);
    let y = world.bodies[car].position.y.to_f64();
    println!(
        "[vehicle] settled: {} wheels grounded, ride height {y:.4} m (closed form {y_ref:.4})",
        vehicle.grounded_wheels()
    );
    assert_eq!(vehicle.grounded_wheels(), 4);
    assert!((y - y_ref).abs() < 0.01);

    // One full-throttle frame per gear: dv = 300 * ratio / 0.3 / 1000 / 60.
    vehicle.throttle = Fix128::ONE;
    let ratios = [3.5_f64, 2.2, 1.5, 1.1, 0.8];
    for (gear, ratio) in ratios.iter().enumerate() {
        assert_eq!(vehicle.current_gear, gear);
        let mut probe = world.bodies[car];
        let before = probe.velocity.z.to_f64();
        vehicle.update(&mut probe, dt);
        let dv = probe.velocity.z.to_f64() - before;
        let want = 300.0 * ratio / radius / mass / 60.0;
        println!(
            "[vehicle] gear {} ratio {ratio}: dv_z = {dv:.5} (closed form {want:.5})",
            gear + 1
        );
        assert!((dv - want).abs() < 1e-4);
        vehicle.shift_up();
    }
    vehicle.shift_up(); // already in 5th: stays
    assert_eq!(vehicle.current_gear, 4);
    for _ in 0..10 {
        vehicle.shift_down();
    }
    assert_eq!(vehicle.current_gear, 0);
    println!("[vehicle] shifting clamps at 1st and 5th gear");
}
