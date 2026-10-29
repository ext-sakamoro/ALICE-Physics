//! The bits of `Vehicle::update` driving a chassis through `PhysicsWorld::step`,
//! pinned so that a refactor of the vehicle step (the force computed by a
//! separate function, applied by `update` as before) is shown to change
//! nothing. The digests were taken before that refactor.
//!
//! Each scene drives the default four-wheel vehicle for 300 frames at 60 Hz
//! with inputs that change over time (throttle, brake, steering, a gear
//! shift) and hashes, every frame, the chassis state and every field of the
//! vehicle state.

use alice_physics_core::vehicle::{Vehicle, VehicleConfig};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
use sha2::{Digest, Sha256};

fn put(h: &mut Sha256, f: Fix128) {
    h.update(f.hi.to_le_bytes());
    h.update(f.lo.to_le_bytes());
}

fn put3(h: &mut Sha256, v: Vec3Fix) {
    put(h, v.x);
    put(h, v.y);
    put(h, v.z);
}

fn run(start_y: Fix128, steering: Fix128) -> String {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let chassis = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, start_y, Fix128::ZERO),
        Fix128::from_int(1200),
    ));
    let mut v = Vehicle::new(VehicleConfig::default());
    let dt = Fix128::from_ratio(1, 60);
    let mut h = Sha256::new();
    for frame in 0..300 {
        v.throttle = if frame < 150 {
            Fix128::from_ratio(3, 4)
        } else {
            Fix128::ZERO
        };
        v.brake = if frame >= 200 {
            Fix128::from_ratio(1, 2)
        } else {
            Fix128::ZERO
        };
        v.steering = if frame >= 60 { steering } else { Fix128::ZERO };
        if frame == 100 {
            v.shift_up();
        }
        v.update(&mut world.bodies[chassis], dt);
        world.step(dt);
        let b = &world.bodies[chassis];
        put3(&mut h, b.position);
        put3(&mut h, b.velocity);
        put(&mut h, b.rotation.w);
        put3(
            &mut h,
            Vec3Fix::new(b.rotation.x, b.rotation.y, b.rotation.z),
        );
        put3(&mut h, b.angular_velocity);
        h.update((v.current_gear as u64).to_le_bytes());
        for f in [v.engine_rpm, v.speed_kmh] {
            put(&mut h, f);
        }
        for w in &v.wheel_states {
            for f in [
                w.compression,
                w.suspension_force,
                w.spin_angle,
                w.spin_speed,
                w.lateral_slip,
                w.longitudinal_slip,
            ] {
                put(&mut h, f);
            }
            h.update([u8::from(w.grounded)]);
            put3(&mut h, w.contact_point);
            put3(&mut h, w.ground_normal);
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

// re-recorded (from 6e8b1dae… / 54630c89…) when damping became a rate per second with default 0:
// `run` uses the default config, which had damped the chassis by a retention 0.99 per step
const DROP_AND_TURN: &str = "6261ba824ba94b5ed3e0a81a9c76bdd559dc7a8af3faabb98d8925e9ec490d97";
const AIRBORNE_START: &str = "fd270b269e2b6652d529dc850eedf3558589f38c2c18a4598e482890b2b10623";

/// golden (refactor guard): a chassis dropped onto its wheels, driven,
/// turned right, shifted up, coasting and braking.
#[test]
fn drop_and_turn_bits_are_pinned() {
    assert_eq!(
        run(Fix128::from_ratio(3, 4), Fix128::from_ratio(1, 2)),
        DROP_AND_TURN
    );
}

/// golden (refactor guard): a chassis starting 3 m up (every wheel off the
/// ground for the first frames, then landing) and turning left.
#[test]
fn airborne_start_bits_are_pinned() {
    assert_eq!(
        run(Fix128::from_int(3), -Fix128::from_ratio(1, 3)),
        AIRBORNE_START
    );
}
