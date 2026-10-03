//! Oracles for `vehicle`: gear shifting, wheel grounding and the force
//! terms of one `Vehicle::update`.
//!
//! One update applies `F * dt` to the chassis as an impulse, so with chassis
//! mass `M` the velocity change is `F * dt / M`. Every expected force below is
//! a hand computation from the documented terms (spring `k * compression`,
//! compression = `1 - (height - radius) / rest`, drive `torque * gear / n_driven
//! / radius`, brake `brake_force * brake` ...). Forces not under test are
//! zeroed in the config (`aero_drag`, `downforce`, `anti_roll_stiffness`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;
use alice_physics::vehicle::{EngineConfig, Vehicle, VehicleConfig, WheelConfig};

const M: f64 = 1000.0;
const DT: f64 = 1.0 / 60.0;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn wheel(x: f64, y: f64, z: f64) -> WheelConfig {
    WheelConfig {
        local_position: Vec3Fix::from_f32(x as f32, y as f32, z as f32),
        // radius 0.3, rest 0.3, k 50000, damping 4500 (defaults)
        ..WheelConfig::default()
    }
}

/// Four wheels at local y = -0.2 (default layout), every extra force off.
fn quiet_config() -> VehicleConfig {
    let mut c = VehicleConfig::default();
    c.aero_drag = Fix128::ZERO;
    c.downforce = Fix128::ZERO;
    c.anti_roll_stiffness = Fix128::ZERO;
    for w in &mut c.wheels {
        w.max_steer_angle = Fix128::ZERO;
    }
    c
}

fn chassis(y: f64) -> RigidBody {
    RigidBody::new_dynamic(Vec3Fix::from_f32(0.0, y as f32, 0.0), fx(M))
}

/// Run one update and return the chassis velocity change.
fn dv(v: &mut Vehicle, c: &mut RigidBody) -> (f64, f64, f64) {
    let before = c.velocity;
    v.update(c, fx(DT));
    let d = c.velocity - before;
    (d.x.to_f64(), d.y.to_f64(), d.z.to_f64())
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn shifting_is_clamped_to_the_gear_table() {
    let mut v = Vehicle::new_default(); // 5 ratios
    assert_eq!(v.current_gear, 0);
    v.shift_down();
    assert_eq!(v.current_gear, 0, "no gear below first");
    for want in [1, 2, 3, 4, 4, 4] {
        v.shift_up();
        assert_eq!(v.current_gear, want);
    }
    for want in [3, 2, 1, 0, 0] {
        v.shift_down();
        assert_eq!(v.current_gear, want);
    }
    // Table size decides the top gear.
    let mut two = VehicleConfig::default();
    two.gear_ratios.truncate(2);
    let mut v = Vehicle::new(two);
    v.shift_up();
    v.shift_up();
    assert_eq!(v.current_gear, 1);
    let mut none = VehicleConfig::default();
    none.gear_ratios.clear();
    let mut v = Vehicle::new(none);
    v.shift_up();
    assert_eq!(v.current_gear, 0);
}

#[test]
fn grounded_wheels_follow_the_raycast_window() {
    // A wheel is grounded iff 0 < (chassis y - 0.2 - ground) < rest + radius = 0.6.
    let mut cfg = quiet_config();
    cfg.ground_height = fx(1.0);
    for (y, want) in [
        (5.0, 0),
        (1.801, 0),
        (1.799, 4),
        (1.65, 4),
        (1.201, 4),
        (1.199, 0),
        (1.1, 0),
    ] {
        let mut v = Vehicle::new(cfg.clone());
        let mut c = chassis(y);
        v.update(&mut c, fx(DT));
        assert_eq!(v.grounded_wheels(), want, "chassis y = {y}");
        assert_eq!(v.wheel_states.iter().filter(|w| w.grounded).count(), want);
    }
    // Mixed: left wheels low enough, right wheels hanging 1 m below the chassis' reach.
    let mut cfg = quiet_config();
    cfg.wheels[1].local_position = Vec3Fix::from_f32(0.8, 0.5, 1.2);
    cfg.wheels[3].local_position = Vec3Fix::from_f32(0.8, 0.5, -1.2);
    let mut v = Vehicle::new(cfg);
    assert_eq!(
        v.grounded_wheels(),
        0,
        "nothing grounded before the first update"
    );
    let mut c = chassis(0.5);
    v.update(&mut c, fx(DT));
    assert_eq!(v.grounded_wheels(), 2);
    assert!(v.wheel_states[0].grounded && !v.wheel_states[1].grounded);
    assert!(v.wheel_states[2].grounded && !v.wheel_states[3].grounded);
    assert!(v.wheel_states[1].compression.is_zero());
    // Contact point is the foot of the wheel on the ground plane.
    let p = v.wheel_states[0].contact_point.to_f32();
    assert!((p.0 + 0.8).abs() < 1e-4 && p.1.abs() < 1e-6 && (p.2 - 1.2).abs() < 1e-4);
}

#[test]
fn spring_force_is_stiffness_times_compression() {
    // y = 0.5 -> height 0.3 = radius -> compression 1 -> 50000 N per wheel.
    let mut v = Vehicle::new(quiet_config());
    let mut c = chassis(0.5);
    let (_, dy, _) = dv(&mut v, &mut c);
    assert!(near(dy, 4.0 * 50000.0 * DT / M), "dy {dy}");
    assert!(near(v.wheel_states[0].compression.to_f64(), 1.0));
    assert!(near(v.wheel_states[0].suspension_force.to_f64(), 50000.0));
    // y = 0.65 -> height 0.45 -> compression 0.5 -> 25000 N per wheel.
    let mut v = Vehicle::new(quiet_config());
    let mut c = chassis(0.65);
    let (_, dy, _) = dv(&mut v, &mut c);
    assert!(near(dy, 4.0 * 25000.0 * DT / M), "dy {dy}");
    assert!(near(v.wheel_states[3].compression.to_f64(), 0.5));
    // Below the radius the compression saturates at 1.
    let mut v = Vehicle::new(quiet_config());
    let mut c = chassis(0.3);
    let (_, dy, _) = dv(&mut v, &mut c);
    assert!(near(v.wheel_states[0].compression.to_f64(), 1.0));
    assert!(near(dy, 4.0 * 50000.0 * DT / M), "dy {dy}");
}

#[test]
fn progressive_rate_and_bump_stop_add_to_the_spring() {
    // progressive 2 at compression 0.5: k (1 + 2 * 0.5) * 0.5 = 50000 per wheel.
    let mut cfg = quiet_config();
    for w in &mut cfg.wheels {
        w.progressive_rate = fx(2.0);
    }
    let mut v = Vehicle::new(cfg);
    let mut c = chassis(0.65);
    let (_, dy, _) = dv(&mut v, &mut c);
    assert!(near(dy, 4.0 * 50000.0 * DT / M), "dy {dy}");
    // bump stop: stiffness 100000, threshold 0.9, compression 1 -> + 100000 * 0.1.
    let mut cfg = quiet_config();
    for w in &mut cfg.wheels {
        w.bump_stop_stiffness = fx(100000.0);
    }
    let mut v = Vehicle::new(cfg.clone());
    let mut c = chassis(0.5);
    let (_, dy, _) = dv(&mut v, &mut c);
    assert!(near(dy, 4.0 * 60000.0 * DT / M), "dy {dy}");
    // Compression exactly at the threshold (0.9): no bump stop yet.
    // height = radius + (1 - 0.9) * rest = 0.33 -> y = 0.53.
    let mut v = Vehicle::new(cfg);
    let mut c = chassis(0.53);
    let (_, dy, _) = dv(&mut v, &mut c);
    assert!(near(v.wheel_states[0].compression.to_f64(), 0.9));
    assert!(near(dy, 4.0 * 0.9 * 50000.0 * DT / M), "dy {dy}");
}

#[test]
fn damping_resists_velocity_with_separate_bump_and_rebound_rates() {
    // compression 1 -> spring 50000 per wheel; chassis vy = -2 (bump) or +2 (rebound).
    let run = |vy: f64, bump: f64, rebound: f64| {
        let mut cfg = quiet_config();
        for w in &mut cfg.wheels {
            w.bump_damping = fx(bump);
            w.rebound_damping = fx(rebound);
        }
        let mut v = Vehicle::new(cfg);
        let mut c = chassis(0.5).with_velocity(Vec3Fix::from_f32(0.0, vy as f32, 0.0));
        dv(&mut v, &mut c).1
    };
    // Symmetric (0 = use `damping` 4500): susp = 50000 - 4500 * vy per wheel.
    assert!(near(run(-2.0, 0.0, 0.0), 4.0 * (50000.0 + 9000.0) * DT / M));
    assert!(near(run(2.0, 0.0, 0.0), 4.0 * (50000.0 - 9000.0) * DT / M));
    // Dedicated rates.
    assert!(near(
        run(-2.0, 1000.0, 1500.0),
        4.0 * (50000.0 + 2000.0) * DT / M
    ));
    assert!(near(
        run(2.0, 1000.0, 1500.0),
        4.0 * (50000.0 - 3000.0) * DT / M
    ));
}

#[test]
fn anti_roll_bar_acts_on_the_compression_difference() {
    // Left wheel (compression 0.5) grounded, right wheel above the ground window:
    // diff = 0.5 - 0 -> anti-roll force 5000 * 0.5 = 2500 pushing the grounded left side down.
    let mk = |stiffness: f64| {
        let mut cfg = quiet_config();
        cfg.wheels = vec![wheel(-0.8, -0.2, 0.0), wheel(0.8, 5.0, 0.0)];
        cfg.anti_roll_stiffness = fx(stiffness);
        Vehicle::new(cfg)
    };
    let mut v = mk(5000.0);
    let mut c = chassis(0.65);
    let (_, dy, _) = dv(&mut v, &mut c);
    assert!(near(dy, (25000.0 - 2500.0) * DT / M), "dy {dy}");
    let mut v = mk(0.0);
    let mut c = chassis(0.65);
    let (_, dy, _) = dv(&mut v, &mut c);
    assert!(near(dy, 25000.0 * DT / M), "dy {dy}");
}

#[test]
fn drive_force_is_torque_over_radius_split_across_driven_grounded_wheels() {
    // default layout: wheels 2 and 3 driven. throttle 1, gear 0 (3.5): 300 * 3.5 = 1050 Nm,
    // 525 per wheel, / 0.3 m = 1750 N per wheel, 3500 N total, forward = +Z.
    let run = |throttle: f64, gear: usize| {
        let mut v = Vehicle::new(quiet_config());
        v.throttle = fx(throttle);
        v.current_gear = gear;
        let mut c = chassis(0.5);
        let (dx, _, dz) = dv(&mut v, &mut c);
        assert!(dx.abs() < 1e-9);
        dz
    };
    assert!(near(run(1.0, 0), 3500.0 * DT / M));
    assert!(near(run(0.5, 0), 1750.0 * DT / M));
    assert!(near(run(1.0, 1), 2200.0 * DT / M)); // ratio 2.2
    assert!(near(run(1.0, 4), 800.0 * DT / M)); // ratio 0.8
    assert!(near(run(1.0, 99), 1000.0 * DT / M)); // out-of-range gear -> ratio 1
    assert_eq!(run(0.0, 0), 0.0);
    // The torque is split over *driven* wheels (2) even if only one touches the ground.
    let mut cfg = quiet_config();
    cfg.wheels[3].local_position = Vec3Fix::from_f32(0.8, 5.0, -1.2);
    let mut v = Vehicle::new(cfg);
    v.throttle = Fix128::ONE;
    let mut c = chassis(0.5);
    let (_, _, dz) = dv(&mut v, &mut c);
    assert!(near(dz, 1750.0 * DT / M), "dz {dz}");
}

#[test]
fn brake_force_opposes_motion_only_above_a_small_speed() {
    let run = |vz: f64, brake: f64| {
        let mut v = Vehicle::new(quiet_config());
        v.brake = fx(brake);
        let mut c = chassis(0.5).with_velocity(Vec3Fix::from_f32(0.0, 0.0, vz as f32));
        dv(&mut v, &mut c).2
    };
    // 4 braked wheels * 10000 N * brake.
    assert!(near(run(10.0, 0.5), -20000.0 * DT / M));
    assert!(near(run(10.0, 1.0), -40000.0 * DT / M));
    assert!(near(run(-10.0, 0.5), 20000.0 * DT / M));
    assert_eq!(run(10.0, 0.0), 0.0);
    assert_eq!(run(0.05, 1.0), 0.0, "below 0.1 m/s the brake does not push");
    // Wheels without brakes do not brake.
    let mut cfg = quiet_config();
    cfg.wheels[0].has_brake = false;
    cfg.wheels[1].has_brake = false;
    let mut v = Vehicle::new(cfg);
    v.brake = Fix128::ONE;
    let mut c = chassis(0.5).with_velocity(Vec3Fix::from_f32(0.0, 0.0, 10.0));
    assert!(near(dv(&mut v, &mut c).2, -20000.0 * DT / M));
}

#[test]
fn lateral_friction_and_steering_follow_the_coded_model() {
    // Lateral friction: -5000 * vx per grounded wheel.
    let mut v = Vehicle::new(quiet_config());
    let mut c = chassis(0.5).with_velocity(Vec3Fix::from_f32(3.0, 0.0, 0.0));
    let (dx, _, _) = dv(&mut v, &mut c);
    assert!(near(dx, -4.0 * 5000.0 * 3.0 * DT / M), "dx {dx}");
    // Steering (as modelled): each grounded steerable wheel adds
    // |vz| * 1000 * (right sin d + forward (cos d - 1)), d = input * max angle (0.5 rad).
    let mut cfg = quiet_config();
    cfg.wheels[0].max_steer_angle = fx(0.5);
    cfg.wheels[1].max_steer_angle = fx(0.5);
    let mut v = Vehicle::new(cfg);
    v.steering = Fix128::ONE;
    let mut c = chassis(0.5).with_velocity(Vec3Fix::from_f32(0.0, 0.0, 10.0));
    let (dx, _, dz) = dv(&mut v, &mut c);
    let d: f64 = 0.5;
    assert!(near(dx, 2.0 * 10.0 * 1000.0 * d.sin() * DT / M), "dx {dx}");
    // z: steering part plus nothing else (brake 0, throttle 0)
    assert!(
        near(dz, 2.0 * 10.0 * 1000.0 * (d.cos() - 1.0) * DT / M),
        "dz {dz}"
    );
    // Mirror image for negative steering.
    let mut cfg = quiet_config();
    cfg.wheels[0].max_steer_angle = fx(0.5);
    let mut v = Vehicle::new(cfg);
    v.steering = fx(-1.0);
    let mut c = chassis(0.5).with_velocity(Vec3Fix::from_f32(0.0, 0.0, 10.0));
    let (dx, _, _) = dv(&mut v, &mut c);
    assert!(near(dx, -10.0 * 1000.0 * d.sin() * DT / M));
}

#[test]
fn aero_drag_and_downforce_scale_with_speed_squared() {
    let mut cfg = quiet_config();
    cfg.aero_drag = fx(0.4);
    cfg.downforce = fx(0.1);
    // Airborne (y = 5): only the aero terms. v = 10 along Z: drag 0.4 * 100 = 40 N backwards,
    // downforce 0.1 * 100 = 10 N down.
    let mut v = Vehicle::new(cfg.clone());
    let mut c = chassis(5.0).with_velocity(Vec3Fix::from_f32(0.0, 0.0, 10.0));
    let (dx, dy, dz) = dv(&mut v, &mut c);
    assert!(near(dx, 0.0));
    assert!(near(dy, -10.0 * DT / M), "dy {dy}");
    assert!(near(dz, -40.0 * DT / M), "dz {dz}");
    // At rest nothing happens (and no division by zero).
    let mut v = Vehicle::new(cfg);
    let mut c = chassis(5.0);
    assert_eq!(dv(&mut v, &mut c), (0.0, 0.0, 0.0));
}

#[test]
fn wheel_spin_and_speed_readouts() {
    let mut v = Vehicle::new(quiet_config());
    let mut c = chassis(0.5).with_velocity(Vec3Fix::from_f32(0.0, 0.0, 6.0));
    v.update(&mut c, fx(DT));
    // speed readout is taken before the impulse: 6 m/s = 21.6 km/h.
    assert!(near(v.speed_kmh.to_f64(), 21.6));
    // grounded wheel: spin_speed = v / radius = 20 rad/s, angle advances by spin * dt.
    assert!(near(v.wheel_states[0].spin_speed.to_f64(), 20.0));
    assert!(near(v.wheel_states[0].spin_angle.to_f64(), 20.0 * DT));
    // Airborne wheels keep their last spin speed and keep turning.
    let mut c = chassis(5.0);
    v.update(&mut c, fx(DT));
    assert!(near(v.wheel_states[0].spin_speed.to_f64(), 20.0));
    assert!(near(v.wheel_states[0].spin_angle.to_f64(), 40.0 * DT));
    // Idle RPM floor.
    assert!(v.engine_rpm >= v.config.engine.idle_rpm);
    // Heading: a chassis turned 90 degrees about Y moving along +X reads forward speed.
    let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, fx(core::f64::consts::FRAC_PI_2));
    let mut c = chassis(5.0).with_velocity(Vec3Fix::from_f32(10.0, 0.0, 0.0));
    c.rotation = q;
    v.update(&mut c, fx(DT));
    assert!(
        (v.speed_kmh.to_f64() - 36.0).abs() < 1e-3,
        "{}",
        v.speed_kmh.to_f64()
    );
}

#[test]
fn static_chassis_is_left_alone() {
    let mut v = Vehicle::new(quiet_config());
    let mut c = RigidBody::new_static(Vec3Fix::from_f32(0.0, 0.5, 0.0));
    v.throttle = Fix128::ONE;
    v.update(&mut c, fx(DT));
    assert_eq!(c.velocity, Vec3Fix::ZERO);
    assert_eq!(
        v.grounded_wheels(),
        0,
        "a static chassis is not even raycast"
    );
}

#[test]
fn engine_config_default_matches_documented_values() {
    let e = EngineConfig::default();
    assert_eq!(
        (
            e.max_torque.to_f64(),
            e.max_rpm.to_f64(),
            e.idle_rpm.to_f64()
        ),
        (300.0, 7000.0, 800.0)
    );
    assert_eq!(e.num_gears, 5);
    assert_eq!(VehicleConfig::default().gear_ratios.len(), e.num_gears);
}

#[test]
fn ground_window_edges_are_exclusive_with_exact_numbers() {
    // Dyadic geometry: radius 0.25, rest 0.25, wheel at local y = -0.25 -> window 0 < h < 0.5,
    // h = chassis y - 0.25.
    let mk = || {
        let mut cfg = quiet_config();
        for w in &mut cfg.wheels {
            w.radius = fx(0.25);
            w.suspension_rest = fx(0.25);
            w.local_position = Vec3Fix::from_f32(
                w.local_position.x.to_f32(),
                -0.25,
                w.local_position.z.to_f32(),
            );
        }
        Vehicle::new(cfg)
    };
    for (y, want) in [(0.75, 0), (0.7499, 4), (0.25, 0), (0.2501, 4)] {
        let mut v = mk();
        let mut c = chassis(y);
        v.update(&mut c, fx(DT));
        assert_eq!(v.grounded_wheels(), want, "chassis y = {y}");
    }
}

#[test]
fn damping_uses_the_chassis_up_axis_not_world_y() {
    // Chassis upside down (180 degrees about Z): up = -Y. Wheels hang at world y = 0.3 + 0.2 = 0.5
    // -> height 0.5, compression 1 - 0.2 / 0.3 = 1/3 -> spring 16666.67 along -Y.
    // vy = -2 is *rebound* along the chassis' up axis (v . up = +2): susp = spring - 4500 * 2.
    let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(core::f64::consts::PI));
    let mut cfg = quiet_config();
    for w in &mut cfg.wheels {
        w.bump_damping = fx(9000.0);
        w.rebound_damping = fx(4500.0);
    }
    let mut v = Vehicle::new(cfg);
    let mut c = chassis(0.3).with_velocity(Vec3Fix::from_f32(0.0, -2.0, 0.0));
    c.rotation = q;
    let (_, dy, _) = dv(&mut v, &mut c);
    assert!(v.grounded_wheels() == 4);
    let want = -4.0 * (50000.0 / 3.0 - 4500.0 * 2.0) * DT / M;
    assert!((dy - want).abs() < 1e-3, "dy {dy} vs {want}");
}
