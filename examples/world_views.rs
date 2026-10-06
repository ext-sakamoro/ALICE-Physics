//! Read-only views of a running world: a lidar scan, a contact sensor and an IMU,
//! debug-draw data, a stress heatmap of the contacts and audio events for them,
//! each through a `PhysicsWorld` method.
//!
//! A ball dropped onto static ground: the frame it lands produces an impact
//! sound; once it rests, the contact sensor reads its weight and the heatmap
//! peaks under it.
//!
//! Run: `cargo run --example world_views`
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::audio_physics::{AudioConfig, AudioGenerator, AudioMaterial};
use alice_physics::debug_render::{DebugDrawData, DebugDrawFlags};
use alice_physics::heatmap::{HeatmapConfig, SliceAxis};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sensors::{ContactSensor, Imu, Lidar};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn main() {
    let dt = Fix128::from_ratio(1, 60);
    let mut world = PhysicsWorld::new(SolverConfig {
        substeps: 8,
        iterations: 4,
        ..SolverConfig::default()
    });
    // The ground is a large static ball, top at y = -0.5.
    let floor = world.add_body_with_radius(
        RigidBody::new_static(v3(0.0, -10.5, 0.0)),
        Fix128::from_int(10),
    );
    let ball = world.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 1.0, 0.0), Fix128::ONE),
        Fix128::from_ratio(1, 2),
    );

    let mut imu = Imu::new(&world, ball).expect("ball is a body");
    let mut audio = AudioGenerator::new(world.bodies.len(), AudioConfig::default());
    audio.set_material(floor, AudioMaterial::STONE);
    audio.set_material(ball, AudioMaterial::METAL);

    for frame in 0..60 {
        world.step(dt);
        let reading = world.sample_imu(&mut imu, dt).expect("reading");
        audio.begin_frame();
        world.emit_contact_audio(&mut audio);
        for e in audio.get_events() {
            println!(
                "frame {frame:3}: {:?} sound, volume {:.3}, pitch {:.3}, accelerometer y {:.2}",
                e.event_type,
                e.volume.to_f64(),
                e.pitch.to_f64(),
                reading.specific_force.y.to_f64()
            );
        }
    }

    let contact = world.read_contact_sensor(&mut ContactSensor::new(ball), dt);
    println!(
        "contact sensor: {} contact(s), normal force {:.3} N (weight {:.3} N)",
        contact.contact_count,
        contact.normal_force.to_f64(),
        9.81
    );

    let mut lidar = Lidar::new(
        (Fix128::ZERO, Fix128::ZERO, 1),
        (Fix128::ZERO, Fix128::ZERO, 1),
        Fix128::from_int(20),
    )
    .with_pose(v3(-5.0, 0.0, 0.0), alice_physics::math::QuatFix::IDENTITY);
    let scan = world.lidar_scan(&mut lidar);
    println!(
        "lidar from x = -5 along +X: range {:?}",
        scan.range(0, 0).map(Fix128::to_f64)
    );
    let mut mounted = Lidar::new(
        (Fix128::ZERO, Fix128::ZERO, 1),
        (-Fix128::HALF_PI, -Fix128::HALF_PI, 1),
        Fix128::from_int(20),
    );
    let down = world
        .lidar_scan_from_body(&mut mounted, ball)
        .expect("ball is a body");
    println!(
        "lidar on the ball, straight down: range {:?}",
        down.range(0, 0).map(Fix128::to_f64)
    );

    let mut draw = DebugDrawData::new();
    world.debug_draw(&DebugDrawFlags::default(), &mut draw);
    println!(
        "debug draw: {} lines, {} points",
        draw.lines.len(),
        draw.points.len()
    );

    let config = HeatmapConfig {
        resolution_x: 16,
        resolution_y: 16,
        slice_axis: SliceAxis::Y,
        slice_offset: world.bodies[ball].position.y - Fix128::from_ratio(1, 2),
        bounds_min: v3(-2.0, 0.0, -2.0),
        bounds_max: v3(2.0, 0.0, 2.0),
    };
    let map = world.stress_heatmap(dt, &config);
    println!(
        "stress heatmap {}x{}: peak {:.3}",
        map.width,
        map.height,
        map.max_value.to_f64()
    );
}
