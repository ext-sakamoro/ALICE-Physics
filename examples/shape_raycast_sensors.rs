//! Ray queries against real collider shapes, and the simulated sensors built on them.
//!
//! A box, a torus, a compound, a sloped height field and an SDF sphere; rays cast
//! with every query mode and filter; a lidar scan, a contact sensor and an IMU with
//! deterministic Gaussian noise.
//!
//! Run: `cargo run --example shape_raycast_sensors`
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Capsule;
use alice_physics::compound::CompoundShape;
use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::rng::DeterministicRng;
use alice_physics::sdf_ccd::SdfCcdConfig;
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sensors::{ContactSensor, GaussianNoise, Imu, Lidar};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget, WorldRayHit};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::static_collider::StaticCollider;

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn describe(label: &str, hit: Option<WorldRayHit>) {
    match hit {
        Some(h) => {
            let what = match h.target {
                RayTarget::Body(i) => format!("body {i}"),
                RayTarget::StaticCollider(j) => format!("static collider {j}"),
                RayTarget::Sdf(k) => format!("SDF {k}"),
                _ => "other".to_string(),
            };
            println!(
                "{label}: {what} at t = {:.6}, normal ({:.3}, {:.3}, {:.3}), body {:?}",
                h.t.to_f64(),
                h.normal.x.to_f64(),
                h.normal.y.to_f64(),
                h.normal.z.to_f64(),
                h.body
            );
        }
        None => println!("{label}: no hit"),
    }
}

fn main() {
    let dt = Fix128::from_ratio(1, 60);
    let mut world = PhysicsWorld::new(SolverConfig::default());

    // A long thin box: its bounding sphere is much larger than the box.
    let slab = world.add_body_with_radius(RigidBody::new_static(v3(0.0, 2.0, 0.0)), Fix128::ONE);
    world.set_body_shape(
        slab,
        &Shape::Box {
            half_extents: v3(2.0, 0.2, 0.2),
        },
    );
    let torus = world
        .add_shaped_body(
            &Shape::Torus {
                major_radius: Fix128::from_int(2),
                minor_radius: Fix128::from_ratio(1, 2),
            },
            Fix128::ONE,
            v3(10.0, 2.0, 0.0),
        )
        .expect("torus");
    let mut compound = CompoundShape::new();
    compound.add_capsule(
        Capsule::new(
            v3(-1.0, 0.0, 0.0),
            v3(1.0, 0.0, 0.0),
            Fix128::from_ratio(1, 4),
        ),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let capsule = world
        .add_compound_body(&compound, Fix128::ONE, v3(-6.0, 2.0, 0.0))
        .expect("compound");

    // Sloped terrain y = 0.1·x over [−20, 20]², and an SDF sphere.
    let mut heights = Vec::new();
    for _z in 0..41 {
        for x in 0..41 {
            heights.push(Fix128::from_f64(0.1 * (x as f64 - 20.0)));
        }
    }
    world.add_static_collider(StaticCollider::HeightField(HeightField::new(
        heights,
        41,
        41,
        Fix128::ONE,
        v3(-20.0, 0.0, -20.0),
    )));
    world.add_sdf_collider(SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )),
        v3(0.0, 2.0, 8.0),
        QuatFix::IDENTITY,
    ));

    let filter = RayFilter::new();
    let far = Fix128::from_int(100);
    // Above the box: the bounding sphere says hit, the box says no.
    println!(
        "old bounding-sphere raycast: {:?}",
        world
            .raycast(v3(-10.0, 2.5, 0.0), Vec3Fix::UNIT_X, far)
            .map(|(i, t)| (i, t.to_f64()))
    );
    describe(
        "shape raycast above the box",
        world.cast_ray(v3(-10.0, 2.5, 0.0), Vec3Fix::UNIT_X, far, &filter),
    );
    describe(
        "down the torus hole",
        world.cast_ray(v3(10.0, 10.0, 0.0), -Vec3Fix::UNIT_Y, far, &filter),
    );
    describe(
        "onto the capsule",
        world.cast_ray(v3(-6.0, 10.0, 0.0), -Vec3Fix::UNIT_Y, far, &filter),
    );
    describe(
        "onto the terrain",
        world.cast_ray(v3(5.0, 10.0, 5.0), -Vec3Fix::UNIT_Y, far, &filter),
    );
    describe(
        "into the SDF",
        world.cast_ray(
            v3(0.0, 2.0, 4.0),
            Vec3Fix::UNIT_Z,
            far,
            &filter.with_sdf_config(SdfCcdConfig::default()),
        ),
    );

    let along = v3(-20.0, 2.0, 0.0);
    let all = world.cast_ray_all(along, Vec3Fix::UNIT_X, far, &filter);
    println!("all hits along y = 2: {}", all.len());
    for h in &all {
        describe("  ", Some(*h));
    }
    println!(
        "any: {}",
        world.cast_ray_any(along, Vec3Fix::UNIT_X, far, &filter)
    );
    // One excluded body; static colliders and SDFs switched off.
    let only_torus = filter
        .excluding_body(capsule)
        .with_static(false)
        .with_sdf(false)
        .with_sensors(false)
        .with_layer_mask(u32::MAX);
    let caster = world.ray_caster(only_torus);
    println!(
        "candidates (BVH): {:?}",
        caster.candidates(along, Vec3Fix::UNIT_X, far)
    );
    describe(
        "caster closest",
        caster.closest(along, Vec3Fix::UNIT_X, far),
    );
    println!(
        "caster all: {}, any: {}",
        caster.all(along, Vec3Fix::UNIT_X, far).len(),
        caster.any(along, Vec3Fix::UNIT_X, far)
    );
    let _ = torus;

    // Lidar: 9 azimuths × 3 elevations, with 1 cm range noise.
    let mut lidar = Lidar::new(
        (Fix128::from_f64(-0.4), Fix128::from_f64(0.4), 9),
        (Fix128::from_f64(-0.1), Fix128::from_f64(0.1), 3),
        Fix128::from_int(50),
    )
    .with_pose(v3(-20.0, 2.0, 0.0), QuatFix::IDENTITY)
    .with_filter(RayFilter::default())
    .with_noise(GaussianNoise::new(Fix128::from_ratio(1, 100), 42));
    println!("lidar directions: {}", lidar.directions().len());
    let scan = lidar.scan(&world);
    println!(
        "lidar centre range: {:?}",
        scan.range(4, 1).map(Fix128::to_f64)
    );

    // A box resting on a static floor, with a lidar, an IMU and a contact sensor.
    let mut sim = PhysicsWorld::new(SolverConfig {
        iterations: 4,
        ..SolverConfig::default()
    });
    let floor = sim.add_body_with_radius(RigidBody::new_static(v3(0.0, -1.0, 0.0)), Fix128::ONE);
    sim.set_body_shape(
        floor,
        &Shape::Box {
            half_extents: v3(5.0, 0.5, 5.0),
        },
    );
    let boxed = sim
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(0.5, 0.5, 0.5),
            },
            Fix128::ONE,
            Vec3Fix::ZERO,
        )
        .expect("box");
    let mut contact =
        ContactSensor::new(boxed).with_noise(GaussianNoise::new(Fix128::from_ratio(1, 10), 7));
    let mut imu = Imu::new(&sim, boxed)
        .expect("body")
        .with_accel_noise(GaussianNoise::new(Fix128::from_ratio(1, 100), 8))
        .with_gyro_noise(GaussianNoise::new(Fix128::from_ratio(1, 1000), 9));
    let mut down = Lidar::new(
        (Fix128::ZERO, Fix128::ZERO, 1),
        (-Fix128::HALF_PI, -Fix128::HALF_PI, 1),
        Fix128::from_int(10),
    );
    for _ in 0..30 {
        sim.step(dt);
    }
    let reading = contact.read(&sim, dt);
    println!(
        "contact: {} contacts, normal force {:.3} N",
        reading.contact_count,
        reading.normal_force.to_f64()
    );
    let imu_reading = imu.sample(&sim, dt).expect("reading");
    println!(
        "imu specific force y: {:.3}",
        imu_reading.specific_force.y.to_f64()
    );
    if let Some(scan) = down.scan_from_body(&sim, boxed) {
        println!(
            "height above floor (from the box centre): {:?}",
            scan.range(0, 0).map(Fix128::to_f64)
        );
    }

    // The Gaussian itself.
    let mut rng = DeterministicRng::new(1);
    let (a, b) = rng.next_gaussian_pair();
    let c = rng.next_gaussian();
    let d = rng.next_gaussian_with(Fix128::from_int(10), Fix128::from_int(2));
    let mut noise = GaussianNoise::new(Fix128::ONE, 3);
    println!(
        "gaussian draws: {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
        a.to_f64(),
        b.to_f64(),
        c.to_f64(),
        d.to_f64(),
        noise.sample().to_f64(),
        noise.apply(Fix128::ONE).to_f64(),
        noise.apply_vec(Vec3Fix::ZERO).x.to_f64()
    );
}
