//! Four-wheel vehicle: settle on its suspension, then accelerate and shift.
//!
//! Static deflection per wheel is `c = m g / (4 k)` and the chassis rides at
//! `y = 0.2 + radius + rest (1 - c)` (wheels hang 0.2 m below the chassis
//! origin). With full throttle in gear `n` the drive force on the two driven
//! wheels is `max_torque * ratio_n / radius` and the frame's velocity change
//! `F dt / m`.
//!
//! `Vehicle::new_default` is documented as the default 4-wheel vehicle, i.e.
//! `Vehicle::new(VehicleConfig::default())`: it must carry that configuration,
//! start at rest in 1st gear with the engine at idle, and, since its anti-roll
//! bar only acts on a left / right compression difference, settle on a flat
//! ground at the same closed-form ride height as the bar-less car above.
//!
//! ```bash
//! cargo run --release --example vehicle_drive --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::vehicle::{Vehicle, VehicleConfig, WheelState};

fn main() {
    let mass = 1000.0_f64;
    let cfg = VehicleConfig {
        anti_roll_stiffness: Fix128::ZERO,
        ..VehicleConfig::default()
    };
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

    // The stock car: Vehicle::new(VehicleConfig::default()).
    let mut stock = Vehicle::new_default();
    let want = VehicleConfig::default();
    assert_eq!(stock.config.wheels, want.wheels);
    assert_eq!(stock.config.wheels.len(), 4);
    assert_eq!(
        stock.config.wheels.iter().filter(|w| w.driven).count(),
        2,
        "rear-wheel drive"
    );
    assert_eq!(stock.config.engine, want.engine);
    assert_eq!(stock.config.gear_ratios, want.gear_ratios);
    assert_eq!(stock.config.brake_force, want.brake_force);
    assert_eq!(stock.config.aero_drag, want.aero_drag);
    assert_eq!(stock.config.downforce, want.downforce);
    assert_eq!(stock.config.ground_height, want.ground_height);
    assert_eq!(stock.config.anti_roll_stiffness, want.anti_roll_stiffness);
    assert!(!stock.config.anti_roll_stiffness.is_zero(), "bar fitted");
    assert_eq!(stock.wheel_states, vec![WheelState::default(); 4]);
    assert_eq!(stock.current_gear, 0);
    assert_eq!(stock.engine_rpm, want.engine.idle_rpm);
    for input in [stock.throttle, stock.brake, stock.steering, stock.speed_kmh] {
        assert!(input.is_zero(), "starts at rest with no input");
    }
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let car = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_f32(0.0, 0.79, 0.0),
        Fix128::from_int(mass as i64),
    ));
    for _ in 0..300 {
        stock.update(&mut world.bodies[car], dt);
        world.step(dt);
    }
    let y = world.bodies[car].position.y.to_f64();
    println!(
        "[vehicle] new_default: {} wheels grounded, ride height {y:.4} m (closed form {y_ref:.4})",
        stock.grounded_wheels()
    );
    assert_eq!(stock.grounded_wheels(), 4);
    assert!((y - y_ref).abs() < 0.01);
}
