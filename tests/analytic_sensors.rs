//! Oracles for the simulated sensors (`alice_physics::sensors`) and the
//! deterministic Gaussian of `DeterministicRng`.
//!
//! # Expected values
//!
//! - Lidar against a wall `x = 5`: a ray of azimuth `a` and elevation `e` has
//!   direction `(cos e cos a, sin e, cos e sin a)` and meets the wall at
//!   `5 / (cos e · cos a)`. The angles are taken from the `Fix128` grid the sensor
//!   uses (`min + span·i/(n − 1)`), converted to `f64`, so the comparison is of
//!   geometry, not of angle rounding. Tolerance `1e-9`: CORDIC sine and cosine are
//!   good to about `1e-14` (measured worst `7e-15` over `[−π, π]`), multiplied by
//!   ranges below 25 and `1/cos` below 1.3.
//! - IMU: free fall with no damping has `a = g` exactly in closed form and specific
//!   force 0. Tolerance `1e-9` (the velocity is a sum of `substeps` products `g·h`,
//!   each truncated by `2⁻⁶⁴`, then divided by `dt = 1/60`). A body spinning at
//!   constant `ω` about world `Z`, first turned 90° about `X`, reads
//!   `Rx(−90°)·ω = (0, 2, 0)` in its frame at every step; the XPBD backend
//!   recovers `ω` from each substep's rotation increment, which is accurate to
//!   `O((ω·h)²)` relative (`ω·h = 2/480`, measured deviation `1.2e-5` absolute), so
//!   the closed form is checked to `1e-4` after the first frame (the solver's `|ω|`
//!   then keeps shrinking by about that much per frame); at every frame the
//!   reading stays on the body's `Y` axis to `1e-9` and equals the frame change of
//!   the world-frame `ω` the solver holds to `1e-9`.
//! - Contact: a box at rest on a static box carries its weight `m·g`; tolerance 2 %
//!   as in `tests/analytic_contact_forces.rs` (the solver's iterations converge to
//!   the static load to that accuracy in 30 frames).
//! - Gaussian: with `n = 20 000` standard normal draws the sample mean has standard
//!   error `1/√n ≈ 0.0071`, the sample variance `√(2/(n − 1)) ≈ 0.010`, the fraction
//!   within ±1 `√(p(1 − p)/n) ≈ 0.0033` (`p = 0.682 69`), the fourth moment
//!   `√(96/n) ≈ 0.069`. Each is asserted within 4 standard errors (two-sided
//!   probability `6e-5` for a correct generator; with a fixed seed the outcome is
//!   fixed, so this bounds the chance the chosen seed was unlucky).
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::rng::DeterministicRng;
use alice_physics::sensors::{ContactSensor, GaussianNoise, Imu, Lidar};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::static_collider::StaticCollider;

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

fn wall_world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    // Plane n·x = 5 with n = +X: the wall x = 5.
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_X,
        fx(5.0),
    )));
    w
}

/// The `Fix128` grid the sensor builds, in `f64`.
fn grid(min: f64, max: f64, n: usize) -> Vec<f64> {
    let (lo, hi) = (fx(min), fx(max));
    (0..n)
        .map(|i| {
            if n == 1 {
                lo.to_f64()
            } else {
                (lo + (hi - lo) * Fix128::from_int(i as i64) / Fix128::from_int((n - 1) as i64))
                    .to_f64()
            }
        })
        .collect()
}

fn lidar_7x3() -> Lidar {
    Lidar::new((fx(-0.5), fx(0.5), 7), (fx(-0.2), fx(0.2), 3), fx(100.0))
}

// ---------------------------------------------------------------- lidar

/// Every ray of a 7 × 3 scan meets the wall `x = 5` at `5 / (cos e cos a)`.
#[test]
fn lidar_ranges_to_a_wall_match_closed_form() {
    let w = wall_world();
    let scan = lidar_7x3().scan(&w);
    assert_eq!((scan.azimuth_count, scan.elevation_count), (7, 3));
    assert_eq!(scan.ranges.len(), 21);
    let az = grid(-0.5, 0.5, 7);
    let el = grid(-0.2, 0.2, 3);
    for (ei, e) in el.iter().enumerate() {
        for (ai, a) in az.iter().enumerate() {
            let want = 5.0 / (e.cos() * a.cos());
            let got = scan
                .range(ai, ei)
                .expect("every ray meets the wall")
                .to_f64();
            assert!(
                (got - want).abs() < 1e-9,
                "a = {a}, e = {e}: {got} vs {want}"
            );
            let hit = scan.hits[ei * 7 + ai].expect("hit");
            assert_eq!(hit.target, RayTarget::StaticCollider(0));
        }
    }
    // Out-of-range indices read as no return.
    assert_eq!(scan.range(7, 0), None);
    assert_eq!(scan.range(0, 3), None);
}

/// A unit box at `(10, 0, 0)` in front of the wall: the centre ray returns 9, the
/// box face; rays that miss it return the wall. Beyond `max_range` nothing.
#[test]
fn lidar_sees_the_nearest_surface_and_respects_max_range() {
    let mut w = wall_world();
    w.remove_static_collider(0);
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_X,
        fx(20.0),
    )));
    w.add_shaped_body(
        &Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Fix128::ONE,
        v3(10.0, 0.0, 0.0),
    )
    .expect("box");
    let mut lidar = Lidar::new(
        (fx(-0.5), fx(0.5), 3),
        (Fix128::ZERO, Fix128::ZERO, 1),
        fx(100.0),
    );
    let scan = lidar.scan(&w);
    assert_eq!(scan.range(1, 0), Some(fx(9.0)));
    let side = scan.range(0, 0).expect("wall").to_f64();
    assert!((side - 20.0 / 0.5f64.cos()).abs() < 1e-9, "{side}");
    let mut short = Lidar::new(
        (fx(-0.5), fx(0.5), 3),
        (Fix128::ZERO, Fix128::ZERO, 1),
        fx(8.0),
    );
    assert!(short.scan(&w).ranges.iter().all(Option::is_none));
}

/// A lidar mounted on a body at `(3, 0, 0)` (offset (0, 1, 0), turned 90° about
/// `Y` so its `+X` looks along world `−Z`) does not see its own body, and sees a
/// wall `z = −6` at distance 6, at the point `(3, 1, −6)`.
#[test]
fn lidar_on_a_body_uses_the_body_pose_and_ignores_the_body() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let b = w.add_body_with_radius(RigidBody::new_static(v3(3.0, 0.0, 0.0)), fx(2.0));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        -Vec3Fix::UNIT_Z,
        fx(6.0),
    )));
    let mut lidar = Lidar::new(
        (Fix128::ZERO, Fix128::ZERO, 1),
        (Fix128::ZERO, Fix128::ZERO, 1),
        fx(100.0),
    )
    .with_pose(
        v3(0.0, 1.0, 0.0),
        QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::HALF_PI),
    );
    let scan = lidar.scan_from_body(&w, b).expect("body exists");
    let r = scan.range(0, 0).expect("wall").to_f64();
    assert!((r - 6.0).abs() < 1e-9, "range {r}");
    let p = scan.hits[0].expect("hit").point;
    assert!(
        (p.x.to_f64() - 3.0).abs() < 1e-9
            && (p.y.to_f64() - 1.0).abs() < 1e-9
            && (p.z.to_f64() + 6.0).abs() < 1e-9,
        "{p:?}"
    );
    assert!(lidar.scan_from_body(&w, 99).is_none());
    // Unmounted, the same lidar starts inside the body's sphere: a hit at 0.
    let mut free = lidar.clone().with_filter(RayFilter::default());
    free.position = v3(3.0, 0.0, 0.0);
    assert_eq!(free.scan(&w).range(0, 0), Some(Fix128::ZERO));
}

/// The same scan twice gives the same bits; noise with the same seed too, a
/// different seed different bits. Noise residuals over a 40 × 25 scan have mean
/// and standard deviation within 4 standard errors of `(0, σ)`.
#[test]
fn lidar_is_bit_reproducible_and_noise_has_the_stated_statistics() {
    let w = wall_world();
    assert_eq!(lidar_7x3().scan(&w), lidar_7x3().scan(&w));
    let noisy = |seed| {
        lidar_7x3()
            .with_noise(GaussianNoise::new(fx(0.01), seed))
            .scan(&w)
    };
    assert_eq!(noisy(3), noisy(3));
    assert_ne!(noisy(3).ranges, noisy(4).ranges);

    let sigma = 0.01;
    let mut lidar = Lidar::new((fx(-0.6), fx(0.6), 40), (fx(-0.3), fx(0.3), 25), fx(100.0))
        .with_noise(GaussianNoise::new(fx(sigma), 11));
    let scan = lidar.scan(&w);
    let res: Vec<f64> = scan
        .ranges
        .iter()
        .zip(&scan.hits)
        .map(|(r, h)| r.expect("noisy").to_f64() - h.expect("hit").t.to_f64())
        .collect();
    let n = res.len() as f64;
    let mean = res.iter().sum::<f64>() / n;
    let var = res.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    assert!(mean.abs() < 4.0 * sigma / n.sqrt(), "mean {mean}");
    assert!(
        (var / (sigma * sigma) - 1.0).abs() < 4.0 * (2.0 / (n - 1.0)).sqrt(),
        "var {var}"
    );
}

/// No azimuths or no elevations: an empty scan.
#[test]
fn lidar_with_an_empty_grid_scans_nothing() {
    let w = wall_world();
    let mut l = Lidar::new(
        (fx(-0.5), fx(0.5), 0),
        (Fix128::ZERO, Fix128::ZERO, 1),
        fx(10.0),
    );
    let s = l.scan(&w);
    assert!(s.ranges.is_empty() && s.hits.is_empty());
    assert!(l.directions().is_empty());
    let mut l = Lidar::new(
        (fx(-0.5), fx(0.5), 4),
        (Fix128::ZERO, Fix128::ZERO, 0),
        fx(10.0),
    );
    assert!(l.scan(&w).ranges.is_empty());
}

// ---------------------------------------------------------------- IMU

fn undamped(gravity: Vec3Fix) -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity,
        damping: Fix128::ONE,
        ..SolverConfig::default()
    })
}

/// Free fall: acceleration `g`, accelerometer zero; with `gravity_scale = 2`,
/// acceleration `2g` and the accelerometer reads the extra weight, `(0, −g, 0)`.
#[test]
fn imu_in_free_fall_reads_g_and_zero_specific_force() {
    for scale in [1.0, 2.0] {
        let mut w = undamped(v3(0.0, -G, 0.0));
        let mut body = RigidBody::new_dynamic(v3(0.0, 100.0, 0.0), Fix128::ONE);
        body.gravity_scale = fx(scale);
        let b = w.add_body(body);
        let mut imu = Imu::new(&w, b).expect("body");
        for _ in 0..5 {
            w.step(dt());
            let r = imu.sample(&w, dt()).expect("reading");
            let a = r.acceleration;
            assert!(
                (a.y.to_f64() + scale * G).abs() < 1e-9,
                "a = {}",
                a.y.to_f64()
            );
            assert!(a.x.to_f64().abs() < 1e-9 && a.z.to_f64().abs() < 1e-9);
            let f = r.specific_force;
            let want_y = -(scale - 1.0) * G;
            assert!(
                f.x.to_f64().abs() < 1e-9
                    && (f.y.to_f64() - want_y).abs() < 1e-9
                    && f.z.to_f64().abs() < 1e-9,
                "f = {:?}",
                f
            );
        }
    }
}

/// At rest (a static body under gravity) the accelerometer reads `−g` turned into
/// the body frame: unturned `(0, +g, 0)`; turned 90° about `Z`,
/// `Rz(−90°)(0, g, 0) = (g, 0, 0)`.
#[test]
fn imu_at_rest_reads_minus_gravity_in_the_body_frame() {
    let mut w = undamped(v3(0.0, -G, 0.0));
    let flat = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let mut turned = RigidBody::new_static(v3(5.0, 0.0, 0.0));
    turned.rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI);
    let turned = w.add_body(turned);
    let mut imu_flat = Imu::new(&w, flat).expect("body");
    let mut imu_turned = Imu::new(&w, turned).expect("body");
    w.step(dt());
    let f = imu_flat.sample(&w, dt()).expect("reading").specific_force;
    assert!(
        (f.y.to_f64() - G).abs() < 1e-9 && f.x.to_f64().abs() < 1e-9 && f.z.to_f64().abs() < 1e-9,
        "{f:?}"
    );
    let f = imu_turned.sample(&w, dt()).expect("reading").specific_force;
    assert!(
        (f.x.to_f64() - G).abs() < 1e-9 && f.y.to_f64().abs() < 1e-9 && f.z.to_f64().abs() < 1e-9,
        "{f:?}"
    );
}

/// Constant `ω = (0, 0, 2)` in the world on a body first turned 90° about `X`
/// (isotropic inertia, no gravity, no damping): the gyroscope reads
/// `R⁻¹ω = (0, 2, 0)` at every step, while the body keeps turning.
#[test]
fn imu_gyro_reads_angular_velocity_in_the_body_frame() {
    let mut w = undamped(Vec3Fix::ZERO);
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    body.rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, Fix128::HALF_PI);
    body.angular_velocity = v3(0.0, 0.0, 2.0);
    let b = w.add_body(body);
    let mut imu = Imu::new(&w, b).expect("body");
    let start = w.bodies[b].rotation;
    for step in 0..10 {
        w.step(dt());
        let g = imu.sample(&w, dt()).expect("reading").angular_velocity;
        if step == 0 {
            assert!(
                g.x.to_f64().abs() < 1e-4
                    && (g.y.to_f64() - 2.0).abs() < 1e-4
                    && g.z.to_f64().abs() < 1e-4,
                "{g:?}"
            );
        }
        assert!(
            g.x.to_f64().abs() < 1e-9 && g.z.to_f64().abs() < 1e-9,
            "off-axis {g:?}"
        );
        let body = &w.bodies[b];
        let want = body.rotation.conjugate().rotate_vec(body.angular_velocity);
        assert!((g - want).length().to_f64() < 1e-9, "{g:?} vs {want:?}");
    }
    assert_ne!(w.bodies[b].rotation, start, "the body turned");
}

/// `dt ≤ 0` or a missing body: no reading, and the reference velocity is kept.
#[test]
fn imu_degenerate_input() {
    let mut w = undamped(v3(0.0, -G, 0.0));
    let b = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    assert!(Imu::new(&w, 7).is_none());
    let mut imu = Imu::new(&w, b).expect("body");
    w.step(dt());
    assert!(imu.sample(&w, Fix128::ZERO).is_none());
    assert!(imu.sample(&w, fx(-1.0)).is_none());
    // The skipped samples did not move the reference: the next reading is still g.
    let a = imu.sample(&w, dt()).expect("reading").acceleration;
    assert!((a.y.to_f64() + G).abs() < 1e-9, "{}", a.y.to_f64());
    let mut gone = Imu::new(&w, b).expect("body");
    gone.body = 42;
    assert!(gone.sample(&w, dt()).is_none());
}

/// Noise on the IMU is reproducible from its seed and changes the reading.
#[test]
fn imu_noise_is_seeded() {
    let run = |seed| {
        let mut w = undamped(v3(0.0, -G, 0.0));
        let b = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let mut imu = Imu::new(&w, b)
            .expect("body")
            .with_accel_noise(GaussianNoise::new(fx(0.1), seed))
            .with_gyro_noise(GaussianNoise::new(fx(0.1), seed + 1));
        w.step(dt());
        imu.sample(&w, dt()).expect("reading")
    };
    assert_eq!(run(1), run(1));
    assert_ne!(run(1).specific_force, run(2).specific_force);
    assert_ne!(run(1).angular_velocity, Vec3Fix::ZERO);
    assert_eq!(
        run(1).acceleration,
        Vec3Fix::ZERO,
        "acceleration carries no noise"
    );
}

// ---------------------------------------------------------------- contact sensor

/// A 0.5-half-extent box of mass `8·ρ` (ρ = 2) at rest on a static box: one
/// contact or more, normal forces summing to `m·g` within 2 %, the net force
/// pointing up. A body in the air has no contact.
#[test]
fn contact_sensor_reads_the_weight_of_a_resting_box() {
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
    let density = 2.0;
    let boxed = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(0.5, 0.5, 0.5),
            },
            fx(density),
            v3(0.0, 0.0, 0.0),
        )
        .expect("box");
    let air = w.add_body_with_radius(RigidBody::new_static(v3(50.0, 50.0, 0.0)), fx(0.5));
    let mass = 1.0 / w.bodies[boxed].inv_mass.to_f64();
    assert!((mass - density).abs() < 1e-9, "unit-volume box");
    for _ in 0..30 {
        w.step(dt());
    }
    let mut sensor = ContactSensor::new(boxed);
    let r = sensor.read(&w, dt());
    assert!(r.in_contact && r.contact_count >= 1);
    let want = mass * G;
    let got = r.normal_force.to_f64();
    assert!(
        (got - want).abs() < 0.02 * want,
        "normal force {got}, weight {want}"
    );
    let net = r.net_force;
    assert!((net.y.to_f64() - want).abs() < 0.02 * want, "net {net:?}");
    // The floor feels the same force, downward.
    let floor_r = ContactSensor::new(floor).read(&w, dt());
    assert!((floor_r.net_force.y.to_f64() + net.y.to_f64()).abs() < 1e-9);
    let none = ContactSensor::new(air).read(&w, dt());
    assert!(!none.in_contact);
    assert_eq!(
        (none.contact_count, none.normal_force, none.net_force),
        (0, Fix128::ZERO, Vec3Fix::ZERO)
    );
    let missing = ContactSensor::new(999).read(&w, dt());
    assert!(!missing.in_contact);
    // Noise moves the sum, reproducibly, and never below zero.
    let noisy = |seed| {
        ContactSensor::new(boxed)
            .with_noise(GaussianNoise::new(fx(0.5), seed))
            .read(&w, dt())
    };
    assert_eq!(noisy(5), noisy(5));
    assert_ne!(noisy(5).normal_force, r.normal_force);
    let huge = ContactSensor::new(air)
        .with_noise(GaussianNoise::new(fx(10.0), 1))
        .read(&w, dt());
    assert!(huge.normal_force >= Fix128::ZERO);
}

// ---------------------------------------------------------------- Gaussian

/// Sample mean, variance, ±1 fraction and fourth moment of 20 000 draws within 4
/// standard errors of the standard normal's (see the module doc for the errors).
#[test]
fn gaussian_moments_match_the_standard_normal() {
    let mut rng = DeterministicRng::new(2026);
    let n = 20_000usize;
    let z: Vec<f64> = (0..n).map(|_| rng.next_gaussian().to_f64()).collect();
    let nf = n as f64;
    let mean = z.iter().sum::<f64>() / nf;
    let var = z.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let within = z.iter().filter(|x| x.abs() < 1.0).count() as f64 / nf;
    let m4 = z.iter().map(|x| x.powi(4)).sum::<f64>() / nf;
    assert!(mean.abs() < 4.0 / nf.sqrt(), "mean {mean}");
    assert!(
        (var - 1.0).abs() < 4.0 * (2.0 / (nf - 1.0)).sqrt(),
        "variance {var}"
    );
    let p = 0.682_689_492;
    assert!(
        (within - p).abs() < 4.0 * (p * (1.0 - p) / nf).sqrt(),
        "within ±1: {within}"
    );
    assert!(
        (m4 - 3.0).abs() < 4.0 * (96.0 / nf).sqrt(),
        "fourth moment {m4}"
    );
    // The pair's second value has the same statistics and is uncorrelated with the first.
    let mut rng = DeterministicRng::new(7);
    let pairs: Vec<(f64, f64)> = (0..n)
        .map(|_| {
            let (a, b) = rng.next_gaussian_pair();
            (a.to_f64(), b.to_f64())
        })
        .collect();
    let corr = pairs.iter().map(|(a, b)| a * b).sum::<f64>() / nf;
    assert!(corr.abs() < 4.0 / nf.sqrt(), "correlation {corr}");
    let var_b = pairs.iter().map(|(_, b)| b * b).sum::<f64>() / nf;
    assert!(
        (var_b - 1.0).abs() < 4.0 * (2.0 / nf).sqrt(),
        "second variance {var_b}"
    );
}

/// Box–Muller in `Fix128` agrees with the same transform in `f64` from the same
/// two uniforms to `1e-9` (CORDIC and the `ln` series, measured worst case
/// reported by the assertion message).
#[test]
fn gaussian_matches_f64_box_muller() {
    let mut rng = DeterministicRng::new(99);
    let mut worst: f64 = 0.0;
    for _ in 0..5_000 {
        let mut copy = rng.clone();
        let u1 = 1.0 - copy.next_fix128().to_f64();
        let u2 = copy.next_fix128().to_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2 - std::f64::consts::PI;
        let (z0, z1) = rng.next_gaussian_pair();
        worst = worst
            .max((z0.to_f64() - r * th.cos()).abs())
            .max((z1.to_f64() - r * th.sin()).abs());
    }
    assert!(worst < 1e-9, "worst difference {worst:e}");
}

/// The same seed gives the same bits; `next_gaussian` is the pair's first value;
/// `next_gaussian_with` is `mean + σ·z`; `u₁ = 1` (the uniform draw 0) gives 0.
#[test]
fn gaussian_is_bit_reproducible() {
    let draw = |seed| {
        let mut r = DeterministicRng::new(seed);
        (0..100).map(|_| r.next_gaussian()).collect::<Vec<_>>()
    };
    assert_eq!(draw(1), draw(1));
    assert_ne!(draw(1), draw(2));
    let mut a = DeterministicRng::new(5);
    let mut b = DeterministicRng::new(5);
    assert_eq!(a.next_gaussian(), b.next_gaussian_pair().0);
    let mut a = DeterministicRng::new(5);
    let mut b = DeterministicRng::new(5);
    let z = a.next_gaussian();
    assert_eq!(
        b.next_gaussian_with(fx(3.0), fx(2.0)),
        fx(3.0) + fx(2.0) * z
    );
    // σ = 0: exactly the mean.
    let mut c = DeterministicRng::new(5);
    assert_eq!(c.next_gaussian_with(fx(3.0), Fix128::ZERO), fx(3.0));
}
