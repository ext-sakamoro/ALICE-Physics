//! Oracles for the read-only `PhysicsWorld` views over the sensor, debug-draw,
//! heatmap and audio modules: `lidar_scan`, `lidar_scan_from_body`,
//! `read_contact_sensor`, `sample_imu`, `debug_draw`, `stress_heatmap` and
//! `emit_contact_audio`.
//!
//! Each method is checked twice: for parity (it returns what the module's own
//! function returns on the same world, bit for bit) and against a closed form
//! derived by hand, independent of the implementation.
//!
//! # Expected values
//!
//! - Lidar: a ray along `+X` from the origin meets a sphere of radius `r` centred
//!   at `(d, 0, 0)` at `d − r` (`d = 10`, `r = 2`: 8). Tolerance `1e-9`.
//! - IMU: free fall with no damping has acceleration `g = (0, −9.81, 0)` and
//!   specific force (acceleration minus gravity) zero, and angular rate zero
//!   (the body does not turn). Tolerance `1e-9`, as in `tests/analytic_sensors.rs`.
//! - Contact sensor: a box at rest on a static box carries its weight `m·g`;
//!   tolerance 2 % as in `tests/analytic_contact_forces.rs` (30 frames of settling).
//! - Heatmap: `generate_stress_heatmap` adds `F·(1 − dist/2)` for every contact
//!   within distance 2 of a sample. With one contact and a grid laid out so one
//!   sample sits exactly on the contact point, that sample reads `F` (the contact's
//!   normal force) and is the maximum; `F` is the resting weight `m·g` within 2 %.
//! - Audio: with no gravity, a ball moving at `v = 4` straight down into a static
//!   ball meets it with normal relative speed exactly 4 (no force acts before the
//!   contact). `process_contact` then gives volume `√(min(4/20, 1)) = √0.2`, pitch
//!   `(1 + 4·(1/20))·(1000/600) = 2` (two wood bodies, density 600), and an
//!   `Impact` event because the contact is new. Tolerance `1e-9`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::audio_physics::{AudioConfig, AudioEventType, AudioGenerator};
use alice_physics::collider::Contact;
use alice_physics::debug_render::{debug_draw_world, DebugDrawData, DebugDrawFlags};
use alice_physics::event::ContactEventType;
use alice_physics::heatmap::{generate_stress_heatmap, Heatmap, HeatmapConfig, SliceAxis};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sensors::{ContactSensor, GaussianNoise, Imu, Lidar};
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

const G: f64 = 9.81;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn undamped(gravity: Vec3Fix) -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity,
        damping: Fix128::ONE,
        ..SolverConfig::default()
    })
}

/// One ray along `+X` (azimuth 0, elevation 0).
fn single_ray(max_range: f64) -> Lidar {
    Lidar::new(
        (Fix128::ZERO, Fix128::ZERO, 1),
        (Fix128::ZERO, Fix128::ZERO, 1),
        fx(max_range),
    )
}

/// A box (half-extent 0.5, density 2) resting on a static box for 30 frames.
fn resting_box() -> (PhysicsWorld, usize, usize) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -G, 0.0),
        substeps: 8,
        iterations: 4,
        ..SolverConfig::default()
    });
    let floor = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -1.0, 0.0)), Fix128::ONE);
    w.set_body_shape(
        floor,
        &Shape::Box {
            half_extents: v3(5.0, 0.5, 5.0),
        },
    );
    let boxed = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(0.5, 0.5, 0.5),
            },
            fx(2.0),
            Vec3Fix::ZERO,
        )
        .expect("box");
    for _ in 0..30 {
        w.step(dt());
    }
    (w, floor, boxed)
}

/// A unit-mass ball of radius 0.5 resting on a static ball of radius 10, so the
/// world has exactly one contact.
fn resting_ball() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -G, 0.0),
        substeps: 8,
        iterations: 4,
        ..SolverConfig::default()
    });
    w.add_body_with_radius(RigidBody::new_static(v3(0.0, -10.0, 0.0)), fx(10.0));
    w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE),
        fx(0.5),
    );
    for _ in 0..60 {
        w.step(dt());
    }
    w
}

fn assert_heatmaps_equal(a: &Heatmap, b: &Heatmap) {
    assert_eq!((a.width, a.height), (b.width, b.height));
    assert_eq!(a.data, b.data);
    assert_eq!((a.min_value, a.max_value), (b.min_value, b.max_value));
}

// ---------------------------------------------------------------- lidar

/// A ray to a sphere of radius 2 at distance 10 returns 8; the world method gives
/// the same scan as `Lidar::scan`.
#[test]
fn lidar_scan_to_a_sphere_is_distance_minus_radius() {
    let mut w = undamped(Vec3Fix::ZERO);
    let target = w.add_body_with_radius(RigidBody::new_static(v3(10.0, 0.0, 0.0)), fx(2.0));
    let mut via_world = single_ray(100.0);
    let mut direct = via_world.clone();
    let scan = w.lidar_scan(&mut via_world);
    assert_eq!(scan, direct.scan(&w));
    let range = scan.range(0, 0).expect("the ray meets the sphere").to_f64();
    assert!((range - 8.0).abs() < 1e-9, "range {range}");
    let hit = scan.hits[0].expect("hit");
    assert_eq!(
        hit.target,
        alice_physics::shape_raycast::RayTarget::Body(target)
    );
}

/// Mounted on a body the sensor does not see that body, and the scan equals
/// `Lidar::scan_from_body`; a missing body gives `None`.
#[test]
fn lidar_scan_from_body_matches_the_module_and_skips_the_carrier() {
    let mut w = undamped(Vec3Fix::ZERO);
    let carrier = w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), fx(1.0));
    w.add_body_with_radius(RigidBody::new_static(v3(10.0, 0.0, 0.0)), fx(2.0));
    let mut via_world = single_ray(100.0);
    let mut direct = via_world.clone();
    let scan = w
        .lidar_scan_from_body(&mut via_world, carrier)
        .expect("body");
    assert_eq!(Some(scan.clone()), direct.scan_from_body(&w, carrier));
    let range = scan.range(0, 0).expect("hit").to_f64();
    assert!((range - 8.0).abs() < 1e-9, "range {range}");
    assert_eq!(w.lidar_scan_from_body(&mut single_ray(100.0), 99), None);
}

/// Noise draws advance the sensor's own generator through the world method just
/// as through `scan`: two scans in a row match two direct scans in a row.
#[test]
fn lidar_scan_advances_the_noise_stream_like_the_module() {
    let mut w = undamped(Vec3Fix::ZERO);
    w.add_body_with_radius(RigidBody::new_static(v3(10.0, 0.0, 0.0)), fx(2.0));
    let mut via_world = single_ray(100.0).with_noise(GaussianNoise::new(fx(0.1), 7));
    let mut direct = via_world.clone();
    let first = w.lidar_scan(&mut via_world);
    let second = w.lidar_scan(&mut via_world);
    assert_eq!(first, direct.scan(&w));
    assert_eq!(second, direct.scan(&w));
    assert_ne!(first.ranges, second.ranges, "each scan draws new noise");
}

/// An empty world returns no hit for every ray.
#[test]
fn lidar_scan_in_an_empty_world_sees_nothing() {
    let w = undamped(Vec3Fix::ZERO);
    let mut l = Lidar::new(
        (fx(-0.5), fx(0.5), 5),
        (Fix128::ZERO, Fix128::ZERO, 1),
        fx(50.0),
    );
    let scan = w.lidar_scan(&mut l);
    assert_eq!(scan.ranges, vec![None; 5]);
    assert!(scan.hits.iter().all(Option::is_none));
}

// ---------------------------------------------------------------- IMU

/// Free fall: acceleration `g`, specific force zero, angular rate zero; equal to
/// `Imu::sample` on the same world.
#[test]
fn sample_imu_in_free_fall_reads_g_and_matches_the_module() {
    let mut w = undamped(v3(0.0, -G, 0.0));
    let b = w.add_body(RigidBody::new_dynamic(v3(0.0, 100.0, 0.0), Fix128::ONE));
    let mut via_world = Imu::new(&w, b).expect("body");
    let mut direct = via_world.clone();
    for _ in 0..5 {
        w.step(dt());
        let r = w.sample_imu(&mut via_world, dt()).expect("reading");
        assert_eq!(Some(r), direct.sample(&w, dt()));
        let a = r.acceleration;
        assert!(
            a.x.to_f64().abs() < 1e-9
                && (a.y.to_f64() + G).abs() < 1e-9
                && a.z.to_f64().abs() < 1e-9,
            "a = {a:?}"
        );
        assert!(r.specific_force.length().to_f64() < 1e-9, "{r:?}");
        assert!(r.angular_velocity.length().to_f64() < 1e-9, "{r:?}");
    }
}

/// `dt ≤ 0` and a body that is gone give no reading, as `Imu::sample` does.
#[test]
fn sample_imu_degenerate_input() {
    let mut w = undamped(v3(0.0, -G, 0.0));
    let b = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let mut imu = Imu::new(&w, b).expect("body");
    w.step(dt());
    assert_eq!(w.sample_imu(&mut imu, Fix128::ZERO), None);
    assert_eq!(w.sample_imu(&mut imu, fx(-1.0)), None);
    // The skipped samples kept the reference: the next reading is still g.
    let a = w.sample_imu(&mut imu, dt()).expect("reading").acceleration;
    assert!((a.y.to_f64() + G).abs() < 1e-9, "{}", a.y.to_f64());
    imu.body = 42;
    assert_eq!(w.sample_imu(&mut imu, dt()), None);
}

// ---------------------------------------------------------------- contact sensor

/// A resting box reads its weight `m·g` within 2 %, as `ContactSensor::read`
/// does; a body out of range reads no contact.
#[test]
fn read_contact_sensor_reads_the_weight_and_matches_the_module() {
    let (w, floor, boxed) = resting_box();
    let mass = 1.0 / w.bodies[boxed].inv_mass.to_f64();
    let mut via_world = ContactSensor::new(boxed);
    let r = w.read_contact_sensor(&mut via_world, dt());
    assert_eq!(r, ContactSensor::new(boxed).read(&w, dt()));
    assert!(r.in_contact && r.contact_count >= 1);
    let want = mass * G;
    let got = r.normal_force.to_f64();
    assert!(
        (got - want).abs() < 0.02 * want,
        "normal force {got}, weight {want}"
    );
    let floor_r = w.read_contact_sensor(&mut ContactSensor::new(floor), dt());
    assert!((floor_r.net_force.y + r.net_force.y).to_f64().abs() < 1e-9);
    let missing = w.read_contact_sensor(&mut ContactSensor::new(999), dt());
    assert!(!missing.in_contact);
    assert_eq!(missing.contact_count, 0);
    // Noise goes through the sensor's own stream.
    let mut noisy_world = ContactSensor::new(boxed).with_noise(GaussianNoise::new(fx(0.5), 3));
    let mut noisy_direct = noisy_world.clone();
    assert_eq!(
        w.read_contact_sensor(&mut noisy_world, dt()),
        noisy_direct.read(&w, dt())
    );
}

/// `dt = 0` gives the contacts but no force (a zero step carries no force), as in
/// the module; an empty world has no contact.
#[test]
fn read_contact_sensor_degenerate_input() {
    let (w, _, boxed) = resting_box();
    let r = w.read_contact_sensor(&mut ContactSensor::new(boxed), Fix128::ZERO);
    assert_eq!(r, ContactSensor::new(boxed).read(&w, Fix128::ZERO));
    assert!(r.in_contact);
    assert_eq!(r.normal_force, Fix128::ZERO);
    let empty = undamped(Vec3Fix::ZERO);
    let r = empty.read_contact_sensor(&mut ContactSensor::new(0), dt());
    assert!(!r.in_contact);
    assert_eq!((r.normal_force, r.net_force), (Fix128::ZERO, Vec3Fix::ZERO));
}

// ---------------------------------------------------------------- debug draw

/// One static ball of radius 2 at `(1, 2, 3)` with only boxes and centres on: a
/// 12-edge cube of half-extent 2 around it and one centre point; the same data as
/// `debug_draw_world`.
#[test]
fn debug_draw_draws_the_broad_phase_box_and_matches_the_module() {
    let mut w = undamped(Vec3Fix::ZERO);
    w.add_body_with_radius(RigidBody::new_static(v3(1.0, 2.0, 3.0)), fx(2.0));
    let flags = DebugDrawFlags {
        draw_aabbs: true,
        draw_centers: true,
        draw_velocities: false,
        draw_contacts: false,
        draw_contact_normals: false,
        draw_joints: false,
        draw_bvh: false,
        draw_axes: false,
    };
    let mut via_world = DebugDrawData::new();
    let mut direct = DebugDrawData::new();
    w.debug_draw(&flags, &mut via_world);
    debug_draw_world(&w, &flags, &mut direct);
    assert_eq!(via_world.lines, direct.lines);
    assert_eq!(via_world.points, direct.points);
    assert_eq!((via_world.lines.len(), via_world.points.len()), (12, 1));
    assert_eq!(via_world.points[0].position, v3(1.0, 2.0, 3.0));
    let (lo, hi) = (v3(-1.0, 0.0, 1.0), v3(3.0, 4.0, 5.0));
    for l in &via_world.lines {
        for p in [l.start, l.end] {
            for (c, (a, b)) in [
                (p.x, (lo.x, hi.x)),
                (p.y, (lo.y, hi.y)),
                (p.z, (lo.z, hi.z)),
            ] {
                assert!(c == a || c == b, "corner {p:?} off the box");
            }
        }
        // Every edge has length 4 along exactly one axis.
        assert_eq!((l.end - l.start).length().to_f64(), 4.0);
    }
}

/// An empty world clears the data and draws nothing.
#[test]
fn debug_draw_in_an_empty_world_clears_the_data() {
    let w = undamped(Vec3Fix::ZERO);
    let mut data = DebugDrawData::new();
    data.line(
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_X,
        alice_physics::DebugColor::RED,
    );
    w.debug_draw(&DebugDrawFlags::default(), &mut data);
    assert_eq!(data.primitive_count(), 0);
}

// ---------------------------------------------------------------- heatmap

/// A grid on the plane `y = p.y` whose cell `(2, 2)` sits exactly on `p`.
fn grid_on(p: Vec3Fix) -> HeatmapConfig {
    HeatmapConfig {
        resolution_x: 4,
        resolution_y: 4,
        slice_axis: SliceAxis::Y,
        slice_offset: p.y,
        bounds_min: Vec3Fix::new(p.x - Fix128::ONE, p.y, p.z - Fix128::ONE),
        bounds_max: Vec3Fix::new(p.x + Fix128::ONE, p.y, p.z + Fix128::ONE),
    }
}

/// One resting contact of force `F ≈ m·g`: the cell on the contact point reads
/// exactly `F` and is the peak; the map equals `generate_stress_heatmap` over
/// `contact_forces`.
#[test]
fn stress_heatmap_peaks_at_the_contact_with_its_force() {
    let w = resting_ball();
    let forces = w.contact_forces(dt());
    assert_eq!(forces.len(), 1, "one contact: {forces:?}");
    let (point, _, force) = forces[0];
    let want = 1.0 * G;
    assert!(
        (force.to_f64() - want).abs() < 0.02 * want,
        "force {} vs weight {want}",
        force.to_f64()
    );
    let cfg = grid_on(point);
    let map = w.stress_heatmap(dt(), &cfg);
    let positions: Vec<Vec3Fix> = w.bodies.iter().map(|b| b.position).collect();
    assert_heatmaps_equal(&map, &generate_stress_heatmap(&positions, &forces, &cfg));
    assert_eq!(map.data[2 * 4 + 2], force, "cell on the contact reads F");
    assert_eq!(map.max_value, force);
    // A neighbour half a unit away reads F·(1 − 0.5/2) = 0.75·F.
    let neighbour = map.data[2 * 4 + 1].to_f64();
    assert!(
        (neighbour - 0.75 * force.to_f64()).abs() < 1e-9,
        "{neighbour}"
    );
}

/// An empty world, and `dt = 0` (no force), give an all-zero map.
#[test]
fn stress_heatmap_degenerate_input() {
    let cfg = grid_on(Vec3Fix::ZERO);
    let empty = undamped(Vec3Fix::ZERO).stress_heatmap(dt(), &cfg);
    assert!(empty.data.iter().all(|v| v.is_zero()));
    assert_eq!(empty.data.len(), 16);
    let w = resting_ball();
    let zero_dt = w.stress_heatmap(Fix128::ZERO, &cfg);
    assert!(zero_dt.data.iter().all(|v| v.is_zero()));
}

// ---------------------------------------------------------------- audio

/// A ball of radius 0.5 just touching a static ball of radius 1 below it, moving
/// down at `speed`, no gravity, no damping.
fn impact(world: &mut PhysicsWorld, x: f64, speed: f64) -> usize {
    world.add_body_with_radius(RigidBody::new_static(v3(x, -1.0, 0.0)), fx(1.0));
    let mut ball = RigidBody::new_dynamic(v3(x, 0.49, 0.0), Fix128::ONE);
    ball.velocity = v3(0.0, -speed, 0.0);
    world.add_body_with_radius(ball, fx(0.5))
}

/// One new contact at normal speed 4: one impact of volume `√0.2` and pitch 2
/// (see the module doc); the same events as feeding the contact event to
/// `process_contact` by hand.
#[test]
fn emit_contact_audio_one_impact_follows_the_formula() {
    let mut w = undamped(Vec3Fix::ZERO);
    impact(&mut w, 0.0, 4.0);
    w.step(dt());
    let events: Vec<_> = w.contact_events().to_vec();
    assert_eq!(events.len(), 1, "{events:?}");
    let e = events[0];
    assert_eq!(e.event_type, ContactEventType::Begin);
    assert_eq!(e.relative_velocity.abs(), Fix128::from_int(4));

    let mut via_world = AudioGenerator::new(2, AudioConfig::default());
    w.emit_contact_audio(&mut via_world);
    let got = via_world.get_events();
    assert_eq!(got.len(), 1);
    let a = got[0];
    assert_eq!(a.event_type, AudioEventType::Impact);
    assert!((a.volume.to_f64() - 0.2f64.sqrt()).abs() < 1e-9, "{a:?}");
    assert!((a.pitch.to_f64() - 2.0).abs() < 1e-9, "{a:?}");
    assert_eq!(a.position, e.point);

    // By hand: the module fed the contact the event describes.
    let mut direct = AudioGenerator::new(2, AudioConfig::default());
    let (va, vb) = (w.bodies[e.body_a].velocity, w.bodies[e.body_b].velocity);
    let rel = va - vb;
    let tangential = rel - e.normal * rel.dot(e.normal);
    let contact = Contact {
        depth: e.depth,
        normal: e.normal,
        point_a: e.point,
        point_b: e.point + e.normal * e.depth,
    };
    direct.process_contact(
        e.body_a,
        e.body_b,
        &contact,
        e.normal * e.relative_velocity + tangential,
        true,
    );
    assert_eq!(via_world.get_events(), direct.get_events());
}

/// Two impacts in one step follow the contact-event order, so the result does
/// not depend on anything but the world.
#[test]
fn emit_contact_audio_follows_the_contact_event_order() {
    let mut w = undamped(Vec3Fix::ZERO);
    impact(&mut w, 0.0, 4.0);
    impact(&mut w, 10.0, 9.0);
    w.step(dt());
    let begins: Vec<_> = w
        .contact_events()
        .iter()
        .filter(|e| e.event_type != ContactEventType::End)
        .copied()
        .collect();
    assert_eq!(begins.len(), 2);
    let mut g = AudioGenerator::new(4, AudioConfig::default());
    w.emit_contact_audio(&mut g);
    let got = g.get_events();
    assert_eq!(got.len(), 2);
    for (a, e) in got.iter().zip(&begins) {
        assert_eq!(a.position, e.point);
        let speed = e.relative_velocity.abs().to_f64();
        assert!(
            (a.volume.to_f64() - (speed / 20.0).sqrt()).abs() < 1e-9,
            "{a:?} vs speed {speed}"
        );
    }
    assert_ne!(got[0].volume, got[1].volume, "the two impacts differ");
    // A second call appends the same events again (no frame reset).
    w.emit_contact_audio(&mut g);
    assert_eq!(g.get_events().len(), 4);
    assert_eq!(g.get_events()[..2], g.get_events()[2..]);
}

/// The step after the impact the pair persists at rest: no new impact, and the
/// slow contact is below the module's speed gate. An empty world emits nothing.
#[test]
fn emit_contact_audio_degenerate_input() {
    let empty = undamped(Vec3Fix::ZERO);
    let mut g = AudioGenerator::new(0, AudioConfig::default());
    empty.emit_contact_audio(&mut g);
    assert!(g.get_events().is_empty());

    let mut w = undamped(Vec3Fix::ZERO);
    impact(&mut w, 0.0, 4.0);
    w.step(dt());
    w.step(dt());
    let mut g = AudioGenerator::new(2, AudioConfig::default());
    w.emit_contact_audio(&mut g);
    assert!(
        g.get_events()
            .iter()
            .all(|a| a.event_type != AudioEventType::Impact),
        "{:?} / {:?}",
        g.get_events(),
        w.contact_events()
    );
}

/// A ball sliding at 3 along a floor it rests on: a continuing contact whose
/// tangential speed is above the slide threshold 0.5, so one `Slide` event with
/// roughness `(|v_t| / 20)·0.6` (two wood bodies, hardness 0.6), `v_t` being the
/// tangential part of the bodies' current velocity difference.
#[test]
fn emit_contact_audio_sliding_contact_carries_the_tangential_speed() {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -G, 0.0),
        substeps: 8,
        iterations: 4,
        ..SolverConfig::default()
    });
    let floor = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -1.0, 0.0)), fx(1.0));
    w.set_body_shape(
        floor,
        &Shape::Box {
            half_extents: v3(50.0, 0.5, 50.0),
        },
    );
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), Fix128::ONE),
        fx(0.5),
    );
    for _ in 0..30 {
        w.step(dt());
    }
    w.bodies[ball].velocity = v3(3.0, 0.0, 0.0);
    w.step(dt());
    let persist: Vec<_> = w
        .contact_events()
        .iter()
        .filter(|e| e.event_type == ContactEventType::Persist)
        .copied()
        .collect();
    assert_eq!(persist.len(), 1, "{:?}", w.contact_events());
    let e = persist[0];
    let rel = w.bodies[e.body_a].velocity - w.bodies[e.body_b].velocity;
    let vt = (rel - e.normal * rel.dot(e.normal)).length().to_f64();
    assert!(vt > 2.0, "still sliding: {vt}");
    let mut g = AudioGenerator::new(2, AudioConfig::default());
    w.emit_contact_audio(&mut g);
    let got = g.get_events();
    assert_eq!(got.len(), 1, "{got:?}");
    assert_eq!(got[0].event_type, AudioEventType::Slide);
    let want = vt / 20.0 * 0.6;
    assert!(
        (got[0].roughness.to_f64() - want).abs() < 1e-9,
        "{} vs {want}",
        got[0].roughness.to_f64()
    );
}
