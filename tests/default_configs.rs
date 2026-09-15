//! Default-configuration oracles (1.2.0, 解析解突合テスト規律).
//!
//! Golden hashes only detect *change*; a simulation can be wrong from the
//! start. Every `Config` type with a `Default` impl is therefore run here on
//! its default values in the most ordinary scenario of its consumer, and the
//! result is compared with a physically meaningful expectation: a closed form
//! where one exists (static suspension deflection `m g / k`, `x = v t`,
//! support-column volume, `sqrt(v / v_max)` loudness), otherwise a
//! qualitative invariant (monotone settling, bounded positions, symmetric
//! result, parameter independence, bit-exact round trip).
//!
//! Every tolerance is stated in the assertion message. No float `assert_eq!`.
//!
//! Not covered here: `TgsConfig`, `AdaptiveSubStepConfig`, `PgsConfig` live in
//! `pub(crate) mod solver_tgs*` (hidden since v0.14.0-preview.8, v1.0 Item B)
//! and are unreachable from an integration test; they need a unit test next
//! to their `impl Default` or a re-export.

#![cfg(feature = "std")]
// The f64 values in this file are the *oracle* (closed-form references and
// tolerance arithmetic), not simulation state, so the det_math determinism
// gate (disallowed f32/f64 libm calls) does not apply.
#![allow(clippy::disallowed_methods)]

use alice_physics::audio_physics::{AudioConfig, AudioEventType, AudioGenerator, AudioMaterial};
use alice_physics::character::{CharacterConfig, CharacterController};
use alice_physics::cloth::{Cloth, ClothConfig};
use alice_physics::collider::Contact;
use alice_physics::deformable::{DeformableBody, DeformableConfig};
use alice_physics::fluid::{Fluid, FluidConfig};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::netcode::{DeterministicSimulation, FrameInput, NetcodeConfig};
use alice_physics::rope::{Rope, RopeConfig};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::sleeping::SleepConfig;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::support_volume::{estimate_support_volume, OverhangRegion, SupportConfig};
use alice_physics::vehicle::{EngineConfig, Vehicle, VehicleConfig, WheelConfig};

const DT60: f64 = 1.0 / 60.0;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn dt60() -> Fix128 {
    r(1, 60)
}

fn v3(v: Vec3Fix) -> (f64, f64, f64) {
    (v.x.to_f64(), v.y.to_f64(), v.z.to_f64())
}

/// Ground plane `y = 0` as an SDF collider (distance = height above plane).
fn floor_sdf() -> SdfCollider {
    let plane = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    SdfCollider::new_static(Box::new(plane), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

/// Discrete closed form of a particle integrator that, per substep of length
/// `dt_s`, does `v = (v + g dt_s) * d; y += v dt_s` (cloth / rope / fluid /
/// deformable all use this scheme). Returns the total drop after `frames`.
/// 1.2.0 particle-module scheme: symplectic Euler over `substeps` inside the
/// frame, `damping` applied once at the end of the frame:
///   y_{n+1} = y_n + v_n·dt + g·dt²·(s+1)/(2s),  v_{n+1} = (v_n + g·dt)·d
fn damped_fall_per_frame(g: f64, d: f64, substeps: usize, frames: usize) -> f64 {
    let s = substeps as f64;
    let mut y = 0.0;
    let mut v = 0.0;
    for _ in 0..frames {
        y += v * DT60 + g * DT60 * DT60 * (s + 1.0) / (2.0 * s);
        v = (v + g * DT60) * d;
    }
    y
}

// ---------------------------------------------------------------------------
// SolverConfig::default() — projectile under the default world; the frame
// damping law is applied once per frame so x is independent of `substeps`
// ---------------------------------------------------------------------------

#[test]
fn solver_config_default_projectile_matches_discrete_closed_form_and_is_substep_independent() {
    // 1.2.0 scheme: within a frame `x += vx * dt` (vx constant over the
    // substeps), then `vx *= damping` once per frame.
    let default = SolverConfig::default();
    assert_eq!(
        default,
        PhysicsConfig::default(),
        "PhysicsConfig is an alias of SolverConfig"
    );
    let (d, vx0) = (default.damping.to_f64(), 3.0);
    let mut x_ref = 0.0;
    let mut vx = vx0;
    for _ in 0..60 {
        x_ref += vx * DT60;
        vx *= d;
    }
    // x_ref = 3 * dt * (1 - d^60) / (1 - d) = 2.257 (undamped would be 3.0)
    let mut xs = Vec::new();
    for substeps in [default.substeps, 2 * default.substeps] {
        let mut world = PhysicsWorld::new(SolverConfig {
            substeps,
            ..default
        });
        let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        body.velocity = Vec3Fix::from_int(3, 0, 0);
        let b = world.add_body(body);
        for _ in 0..60 {
            world.step(dt60());
        }
        let p = world.bodies[b].position;
        xs.push(p.x.to_f64());
        assert!(
            (p.x.to_f64() - x_ref).abs() < 1e-9,
            "substeps {substeps}: x = {} vs closed form {x_ref} (tol 1e-9)",
            p.x.to_f64()
        );
        assert!(
            p.y.to_f64() < -3.5 && p.y.to_f64() > -5.2,
            "substeps {substeps}: y = {} should be between the damped (-4.17) and undamped (-5.08) fall",
            p.y.to_f64()
        );
        assert!(p.z.is_zero(), "no lateral drift: z = {:?}", p.z);
    }
    assert!(
        (xs[0] - xs[1]).abs() < 1e-9,
        "substeps changed x: {xs:?} (tol 1e-9)"
    );
}

// ---------------------------------------------------------------------------
// SleepConfig::default() — a body at rest falls asleep after exactly
// `frames_to_sleep` idle frames, a moving body never does
// ---------------------------------------------------------------------------

#[test]
fn sleep_config_default_body_at_rest_sleeps_after_frames_to_sleep_not_before() {
    let sleep = SleepConfig::default();
    let frames_to_sleep = sleep.frames_to_sleep as usize;
    assert!(
        frames_to_sleep > 1,
        "default frames_to_sleep = {frames_to_sleep}"
    );

    // No gravity so "at rest" is exact and the idle counter is the only clock.
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    world.set_sleep_config(sleep);
    let still = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let mut mover = RigidBody::new_dynamic(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
    mover.velocity = Vec3Fix::from_int(1, 0, 0); // 1 m/s ≫ 0.01 m/s threshold
    let mover = world.add_body(mover);

    for frame in 1..frames_to_sleep {
        world.step(dt60());
        assert!(
            !world.is_sleeping(still),
            "resting body slept at frame {frame} < frames_to_sleep {frames_to_sleep}"
        );
    }
    world.step(dt60());
    assert!(
        world.is_sleeping(still),
        "resting body still awake after {frames_to_sleep} idle frames"
    );
    assert!(
        !world.is_sleeping(mover),
        "1 m/s body must not sleep (threshold {} m/s)",
        sleep.linear_threshold.to_f64()
    );
    // Sleeping body is frozen; the mover is not (damping 0.99^60 = 0.55 of 1 m/s)
    for _ in 0..10 {
        world.step(dt60());
    }
    assert_eq!(
        world.bodies[still].position,
        Vec3Fix::ZERO,
        "sleeping body moved"
    );
    let vx = world.bodies[mover].velocity.x.to_f64();
    assert!(
        vx > 0.4 && vx < 0.6,
        "mover vx = {vx}, expected 0.99^70 = 0.495 ± 0.1"
    );
    assert!(world.active_body_count() == 1, "one active body expected");

    // Ordinary scenario: a ball resting on a static support (default gravity)
    // sleeps too, no earlier than frames_to_sleep and within a settling bound.
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    world.set_sleep_config(sleep);
    world.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
    let ball = world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 2, 0), Fix128::ONE),
        Fix128::ONE,
    );
    let mut first_sleep = None;
    for frame in 1..=600 {
        world.step(dt60());
        if world.is_sleeping(ball) {
            first_sleep = Some(frame);
            break;
        }
    }
    let first_sleep = first_sleep.expect("resting ball never slept within 10 s");
    assert!(
        first_sleep >= frames_to_sleep && first_sleep <= frames_to_sleep + 120,
        "ball slept at frame {first_sleep}, expected in [{frames_to_sleep}, {}]",
        frames_to_sleep + 120
    );
}

// ---------------------------------------------------------------------------
// ClothConfig::default() — curtain: pinned top row, sags under gravity,
// never rises above the pins, edges keep their rest length, stays bounded
// ---------------------------------------------------------------------------

#[test]
fn cloth_config_default_pinned_curtain_sags_and_stays_bounded() {
    let (res_x, res_y) = (5usize, 5usize);
    let mut cloth = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(2),
        Fix128::from_int(2),
        res_x,
        res_y,
        r(1, 100),
    );
    cloth.config = ClothConfig::default();
    cloth.pin_top_row(res_x);
    let start: Vec<Vec3Fix> = cloth.positions.clone();
    let bottom_row: Vec<usize> = (0..res_x).map(|i| (res_y - 1) * res_x + i).collect();

    // 1.2.0: damping is 0.99 per *frame* (not per substep), so the curtain
    // swings for a few seconds before settling — 15 s of simulation.
    let mut lowest_bottom = start[bottom_row[0]].y.to_f64();
    for frame in 0..900 {
        cloth.step(dt60());
        // top row is pinned: bit-identical
        for (i, (p, s)) in cloth.positions.iter().zip(&start).enumerate().take(res_x) {
            assert_eq!(p, s, "pinned particle {i} moved");
        }
        for (i, p) in cloth.positions.iter().enumerate() {
            let (x, y, z) = v3(*p);
            assert!(
                y <= 1e-3,
                "frame {frame}: particle {i} rose above the pins: y = {y} (tol 1e-3)"
            );
            assert!(
                x.abs() < 3.5 && z.abs() < 3.5 && y > -3.5,
                "frame {frame}: particle {i} left the 3.5 m bound: {p:?}"
            );
        }
        // the bottom row never overshoots the inextensible length: it can swing
        // but its lowest point is bounded by 2 m of cloth below the pins
        let y = cloth.positions[bottom_row[0]].y.to_f64();
        lowest_bottom = lowest_bottom.min(y);
        assert!(
            lowest_bottom > -2.05,
            "frame {frame}: bottom row went below the cloth length: {lowest_bottom}"
        );
    }
    assert!(
        lowest_bottom < -1.8,
        "bottom row never reached the hanging length: lowest {lowest_bottom}"
    );
    // hanging curtain: bottom row at ≈ -2 (2 m of inextensible cloth below the pins)
    for &i in &bottom_row {
        let (_, y, z) = v3(cloth.positions[i]);
        assert!(
            y < -1.8 && y > -2.05,
            "bottom particle {i}: y = {y}, expected -2.0 (-2.05 .. -1.8)"
        );
        assert!(
            z.abs() < 0.3,
            "bottom particle {i}: z = {z}, curtain should hang under the pins (|z| < 0.3)"
        );
    }
    // symmetric input → symmetric settled state, mirrored about x = 1. The
    // sequential Gauss-Seidel passes are order dependent, so the *transient*
    // is asymmetric (up to 0.15 m while the curtain swings down); only the
    // rest state is physically constrained: 1 cm on a 2 m cloth.
    for j in 0..res_y {
        for i in 0..res_x / 2 {
            let a = cloth.positions[j * res_x + i];
            let b = cloth.positions[j * res_x + (res_x - 1 - i)];
            let (ax, ay, az) = v3(a);
            let (bx, by, bz) = v3(b);
            assert!(
                (ax + bx - 2.0).abs() < 1e-2 && (ay - by).abs() < 1e-2 && (az - bz).abs() < 1e-2,
                "row {j}: settled mirror pair {i} / {} asymmetric: {a:?} vs {b:?} (tol 1e-2)",
                res_x - 1 - i
            );
        }
    }
    // edges are inextensible (stretch compliance 0): every edge within 1 % of 0.5 m
    for j in 0..res_y {
        for i in 0..res_x {
            let p = cloth.positions[j * res_x + i];
            if i + 1 < res_x {
                let d = (cloth.positions[j * res_x + i + 1] - p).length().to_f64();
                assert!(
                    (d - 0.5).abs() < 5e-3,
                    "row {j} edge {i}: {d} (0.5 ± 0.005)"
                );
            }
            if j + 1 < res_y {
                let d = (cloth.positions[(j + 1) * res_x + i] - p).length().to_f64();
                assert!(
                    (d - 0.5).abs() < 5e-3,
                    "column {i} edge {j}: {d} (0.5 ± 0.005)"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RopeConfig::default() — a rope released horizontally from one anchor swings
// down and hangs vertically, keeping its length within 2 %
// ---------------------------------------------------------------------------

#[test]
fn rope_config_default_hanging_rope_keeps_length_and_ends_below_anchor() {
    let segments = 10;
    let mut rope = Rope::new(
        Vec3Fix::ZERO,
        Vec3Fix::from_int(1, 0, 0),
        segments,
        Fix128::ONE,
    );
    rope.config = RopeConfig::default();
    rope.pin_start();
    let rest = rope.total_length.to_f64();
    assert!((rest - 1.0).abs() < 1e-9, "rest length {rest}");

    // 1.2.0: damping 0.99 per frame — the pendulum swing takes ~10 s to die out
    for frame in 0..900 {
        rope.step(dt60());
        let len = rope.current_length().to_f64();
        assert!(
            (len - rest).abs() / rest < 0.02,
            "frame {frame}: rope length {len} drifted > 2 % from {rest}"
        );
        assert_eq!(rope.positions[0], Vec3Fix::ZERO, "anchor moved");
        for (i, p) in rope.positions.iter().enumerate() {
            let (x, y, z) = v3(*p);
            assert!(
                y <= 1e-3 && x.abs() <= 1.01 && z.abs() < 1e-9,
                "frame {frame}: particle {i} outside the reachable half disc: {p:?}"
            );
        }
    }
    // settled: hangs straight down (pendulum damped by 0.99 per frame)
    let tip = rope.positions[segments];
    let (x, y, _) = v3(tip);
    assert!(
        (y + 1.0).abs() < 0.02 && x.abs() < 0.02,
        "free end at ({x}, {y}), expected (0, -1) ± 0.02"
    );
    for i in 0..=segments {
        let want = -(i as f64) / segments as f64;
        let got = rope.positions[i].y.to_f64();
        assert!(
            (got - want).abs() < 0.02,
            "particle {i}: y = {got}, expected {want} ± 0.02 (uniform spacing)"
        );
    }
    let speed: f64 = rope
        .velocities
        .iter()
        .map(|v| v.length().to_f64())
        .fold(0.0, f64::max);
    assert!(speed < 0.05, "rope still swinging: max speed {speed} m/s");
}

// ---------------------------------------------------------------------------
// FluidConfig::default() — a free block of particles falls as a whole:
// count preserved, centre of mass follows the damped fall, no explosion
// ---------------------------------------------------------------------------

#[test]
fn fluid_config_default_block_falls_as_a_whole_without_explosion() {
    let config = FluidConfig::default();
    let mut fluid = Fluid::new_block(
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::new(r(4, 10), r(4, 10), r(4, 10)),
        r(1, 10),
        config,
    );
    let n = fluid.particle_count();
    assert_eq!(n, 125, "5x5x5 block");
    let com = |f: &Fluid| {
        let mut s = (0.0, 0.0, 0.0);
        for p in &f.positions {
            let (x, y, z) = v3(*p);
            s = (s.0 + x, s.1 + y, s.2 + z);
        }
        let k = f.positions.len() as f64;
        (s.0 / k, s.1 / k, s.2 / k)
    };
    let com0 = com(&fluid);

    let frames = 60;
    let mut prev_y = com0.1;
    for frame in 0..frames {
        fluid.step(dt60());
        assert_eq!(fluid.particle_count(), n, "particle count changed");
        let c = com(&fluid);
        assert!(
            c.1 < prev_y,
            "frame {frame}: centre of mass did not move down ({prev_y} -> {})",
            c.1
        );
        prev_y = c.1;
        // A 0.4 m block in free fall stays a block: after `frame` frames no
        // particle can be further than the free-fall drop plus its own size.
        let bound = 2.0 + 10.0 * ((frame + 1) as f64 * DT60).powi(2) / 2.0;
        for (i, p) in fluid.positions.iter().enumerate() {
            let (x, y, z) = v3(*p);
            assert!(
                x.abs() < bound && y.abs() < bound && z.abs() < bound,
                "frame {frame}: particle {i} of a 0.4 m block exploded to {p:?} (bound {bound:.2} m). \
                 With surface_tension = 0 and vorticity_strength = 0 the block falls exactly on the \
                 damped closed form, so the default cohesion / confinement terms are the unstable part"
            );
        }
    }
    let c = com(&fluid);
    // Internal PBF / XSPH corrections are pairwise antisymmetric, so the centre
    // of mass falls like a single particle of the same integrator.
    let want = damped_fall_per_frame(
        config.gravity.y.to_f64(),
        config.damping.to_f64(),
        config.substeps,
        frames,
    );
    assert!(
        (c.1 - com0.1 - want).abs() < 0.05 * want.abs(),
        "centre of mass drop {} vs damped closed form {want} (tol 5 %)",
        c.1 - com0.1
    );
    assert!(
        (c.0 - com0.0).abs() < 1e-3 && (c.2 - com0.2).abs() < 1e-3,
        "gravity is vertical but the centre of mass drifted laterally: {:?} -> {:?} (tol 1e-3)",
        (com0.0, com0.2),
        (c.0, c.2)
    );
}

// ---------------------------------------------------------------------------
// DeformableConfig::default() — a free cube translates rigidly (all
// constraints satisfied), then a dropped cube rests on the floor with its
// volume preserved
// ---------------------------------------------------------------------------

fn tet_volume(p: [Vec3Fix; 4]) -> f64 {
    let (a, b, c, d) = (v3(p[0]), v3(p[1]), v3(p[2]), v3(p[3]));
    let e1 = (a.0 - d.0, a.1 - d.1, a.2 - d.2);
    let e2 = (b.0 - d.0, b.1 - d.1, b.2 - d.2);
    let e3 = (c.0 - d.0, c.1 - d.1, c.2 - d.2);
    let cross = (
        e2.1 * e3.2 - e2.2 * e3.1,
        e2.2 * e3.0 - e2.0 * e3.2,
        e2.0 * e3.1 - e2.1 * e3.0,
    );
    (e1.0 * cross.0 + e1.1 * cross.1 + e1.2 * cross.2).abs() / 6.0
}

fn body_volume(body: &DeformableBody) -> f64 {
    body.tetrahedra
        .iter()
        .map(|t| tet_volume([t[0], t[1], t[2], t[3]].map(|i| body.positions[i])))
        .sum()
}

#[test]
fn deformable_config_default_free_cube_translates_rigidly_and_rests_on_floor() {
    let config = DeformableConfig::default();
    let half = r(1, 2);
    let mut cube = DeformableBody::new_cube(Vec3Fix::from_int(0, 5, 0), half, Fix128::from_int(8));
    cube.config = config;
    let start = cube.positions.clone();
    let vol0 = body_volume(&cube);
    assert!((vol0 - 1.0).abs() < 1e-9, "unit cube volume {vol0}");

    let frames = 30;
    for _ in 0..frames {
        cube.step(dt60());
    }
    let want = damped_fall_per_frame(
        config.gravity.y.to_f64(),
        config.damping.to_f64(),
        config.substeps,
        frames,
    );
    let drop0 = cube.positions[0].y.to_f64() - start[0].y.to_f64();
    assert!(
        (drop0 - want).abs() < 1e-6,
        "free cube fell {drop0}, damped closed form {want} (tol 1e-6)"
    );
    for (i, (p, s)) in cube.positions.iter().zip(&start).enumerate() {
        let (px, py, pz) = v3(*p);
        let (sx, sy, sz) = v3(*s);
        assert!(
            (px - sx).abs() < 1e-6 && (pz - sz).abs() < 1e-6 && ((py - sy) - drop0).abs() < 1e-6,
            "particle {i} did not translate rigidly: {p:?} vs start {s:?} (tol 1e-6)"
        );
    }
    let vol = body_volume(&cube);
    assert!(
        (vol - vol0).abs() < 1e-6,
        "free-fall volume {vol} vs {vol0} (tol 1e-6)"
    );

    // Drop onto the floor: bottom face lands on y = 0, volume within 5 %.
    let floor = [floor_sdf()];
    let mut cube = DeformableBody::new_cube(Vec3Fix::from_int(0, 1, 0), half, Fix128::from_int(8));
    cube.config = config;
    for frame in 0..180 {
        cube.step_with_sdf(dt60(), &floor);
        for (i, p) in cube.positions.iter().enumerate() {
            let (x, y, z) = v3(*p);
            assert!(
                y > -0.05 && y < 1.6 && x.abs() < 1.5 && z.abs() < 1.5,
                "frame {frame}: particle {i} at {p:?} left the expected box"
            );
        }
    }
    let com = cube.center_of_mass();
    let (cx, cy, cz) = v3(com);
    assert!(
        (cy - 0.5).abs() < 0.1,
        "resting cube centre y = {cy}, expected 0.5 ± 0.1"
    );
    assert!(
        cx.abs() < 1e-3 && cz.abs() < 1e-3,
        "resting cube drifted laterally: ({cx}, {cz}) (tol 1e-3)"
    );
    let vol = body_volume(&cube);
    assert!(
        (vol - vol0).abs() < 0.05 * vol0,
        "resting volume {vol} vs rest {vol0} (tol 5 %)"
    );
    let bottom_min = [0usize, 1, 4, 5]
        .iter()
        .map(|&i| cube.positions[i].y.to_f64())
        .fold(f64::MAX, f64::min);
    assert!(
        bottom_min > -0.02 && bottom_min < 0.05,
        "bottom face y = {bottom_min}, expected on the floor (-0.02 .. 0.05)"
    );
    let speed = cube
        .velocities
        .iter()
        .map(|v| v.length().to_f64())
        .fold(0.0, f64::max);
    assert!(speed < 0.1, "resting cube still moving: {speed} m/s");
}

// ---------------------------------------------------------------------------
// WheelConfig::default() — static deflection of a single suspension:
// k c = m g  →  c = m g / k, ride height = radius + rest (1 − c)
// ---------------------------------------------------------------------------

/// Settle for 5 s and return `(vehicle, world, chassis, max |Δy| over the last second)`.
fn settle_single_wheel(mass_kg: i64) -> (Vehicle, PhysicsWorld, usize, f64) {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let wheel = WheelConfig::default();
    // start with the wheel just touching the ground (max_dist = rest + radius)
    let start_y = wheel.suspension_rest + wheel.radius - r(1, 1000);
    let chassis = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, start_y, Fix128::ZERO),
        Fix128::from_int(mass_kg),
    ));
    let mut vehicle = Vehicle::new(VehicleConfig {
        wheels: vec![wheel],
        engine: EngineConfig::default(),
        ..VehicleConfig::default()
    });
    let mut max_dy = 0.0f64;
    for frame in 0..300 {
        let before = world.bodies[chassis].position.y.to_f64();
        vehicle.update(&mut world.bodies[chassis], dt60());
        world.step(dt60());
        if frame >= 240 {
            max_dy = max_dy.max((world.bodies[chassis].position.y.to_f64() - before).abs());
        }
    }
    (vehicle, world, chassis, max_dy)
}

#[test]
fn wheel_config_default_static_deflection_matches_mg_over_k() {
    let wheel = WheelConfig::default();
    let (k, rest, radius) = (
        wheel.spring_stiffness.to_f64(),
        wheel.suspension_rest.to_f64(),
        wheel.radius.to_f64(),
    );
    let g = -PhysicsConfig::default().gravity.y.to_f64();
    for mass in [1000i64, 500] {
        let (vehicle, world, chassis, max_dy) = settle_single_wheel(mass);
        let c_ref = mass as f64 * g / k; // 0.2 / 0.1
        let y_ref = radius + rest * (1.0 - c_ref); // 0.54 / 0.57
        let state = vehicle.wheel_states[0];
        assert!(state.grounded, "mass {mass}: wheel not grounded");
        let c = state.compression.to_f64();
        assert!(
            (c - c_ref).abs() < 0.01,
            "mass {mass}: compression {c}, closed form m g / k = {c_ref} (tol 0.01)"
        );
        let y = world.bodies[chassis].position.y.to_f64();
        assert!(
            (y - y_ref).abs() < 0.01,
            "mass {mass}: ride height {y}, closed form {y_ref} (tol 0.01)"
        );
        let f = state.suspension_force.to_f64();
        assert!(
            (f - mass as f64 * g).abs() < 0.02 * mass as f64 * g,
            "mass {mass}: suspension force {f} N should carry the weight {} N (tol 2 %)",
            mass as f64 * g
        );
        // Stationary: the ride height no longer changes frame to frame. The
        // frame-end velocity is *not* zero: the suspension impulse is applied
        // once per frame while gravity is integrated per substep, so `vy`
        // saw-tooths within the frame and is sampled near −g·dt/2 = −0.083 m/s.
        assert!(
            max_dy < 1e-4,
            "mass {mass}: ride height still changing by {max_dy} m/frame after 4 s (tol 1e-4)"
        );
        let vy = world.bodies[chassis].velocity.y.to_f64();
        assert!(
            vy.abs() < g * DT60,
            "mass {mass}: frame-end vy = {vy} m/s exceeds one frame of gravity {} (saw-tooth bound)",
            g * DT60
        );
    }
}

// ---------------------------------------------------------------------------
// EngineConfig::default() — full throttle in first gear on the default 4-wheel
// car: Δv per frame = max_torque · gear / radius / m · dt, RPM floored at idle
// ---------------------------------------------------------------------------

#[test]
fn engine_config_default_full_throttle_accelerates_by_closed_form_first_frame() {
    let engine = EngineConfig::default();
    let vehicle_cfg = VehicleConfig::default();
    assert_eq!(
        engine.num_gears,
        vehicle_cfg.gear_ratios.len(),
        "num_gears must match the gear ratio table"
    );
    let mass = 1500i64;
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let chassis = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 1, 0),
        Fix128::from_int(mass),
    ));
    let mut vehicle = Vehicle::new(VehicleConfig {
        engine,
        ..vehicle_cfg.clone()
    });
    assert_eq!(vehicle.engine_rpm, engine.idle_rpm, "engine starts at idle");

    // settle on the suspension first (ride height stationary, see the
    // saw-tooth note in the WheelConfig test: frame-end vy ≈ −g·dt/2 ≠ 0)
    let mut max_dy = 0.0f64;
    for frame in 0..300 {
        let before = world.bodies[chassis].position.y.to_f64();
        vehicle.update(&mut world.bodies[chassis], dt60());
        world.step(dt60());
        if frame >= 240 {
            max_dy = max_dy.max((world.bodies[chassis].position.y.to_f64() - before).abs());
        }
    }
    assert_eq!(vehicle.grounded_wheels(), 4, "all four wheels grounded");
    assert!(
        max_dy < 1e-4,
        "car still settling before throttle: Δy {max_dy} m/frame (tol 1e-4)"
    );
    let v = world.bodies[chassis].velocity;
    assert!(
        v.x.is_zero() && v.z.is_zero(),
        "car drifted horizontally before throttle: {v:?}"
    );

    vehicle.throttle = Fix128::ONE;
    vehicle.update(&mut world.bodies[chassis], dt60());
    let radius = vehicle_cfg.wheels[0].radius.to_f64();
    let gear = vehicle_cfg.gear_ratios[0].to_f64();
    let force = engine.max_torque.to_f64() * gear / radius; // 3500 N over the driven wheels
    let dv_ref = force / mass as f64 * DT60;
    let dv = world.bodies[chassis].velocity.z.to_f64();
    assert!(
        (dv - dv_ref).abs() < 1e-6,
        "first-frame Δv = {dv} m/s vs T·gear/(r m)·dt = {dv_ref} (tol 1e-6)"
    );

    // keeps accelerating forward for 2 s, RPM stays within [idle, max]
    world.step(dt60());
    let mut prev = world.bodies[chassis].velocity.z.to_f64();
    for frame in 0..120 {
        vehicle.update(&mut world.bodies[chassis], dt60());
        world.step(dt60());
        let vz = world.bodies[chassis].velocity.z.to_f64();
        assert!(vz > prev, "frame {frame}: speed fell {prev} -> {vz}");
        prev = vz;
        assert!(
            vehicle.engine_rpm >= engine.idle_rpm && vehicle.engine_rpm <= engine.max_rpm,
            "frame {frame}: rpm {:?} outside [idle, max]",
            vehicle.engine_rpm
        );
    }
    // Discrete closed form of the whole drive line over the 121 update+step
    // cycles since the throttle went down: the drive / drag impulse is applied
    // before the step, the 0.99 frame damping after it, so
    //   v_{n+1} = (v_n + (F − c_d v_n²) / m · dt) · d
    // whose terminal speed is a·dt·d/(1−d) = 3.85 m/s (not the drag terminal
    // 93 m/s): the world damping, not aero drag, limits the default car.
    let (c_d, d) = (
        vehicle_cfg.aero_drag.to_f64(),
        PhysicsConfig::default().damping.to_f64(),
    );
    let mut v_ref = 0.0;
    for _ in 0..121 {
        v_ref = (v_ref + (force - c_d * v_ref * v_ref) / mass as f64 * DT60) * d;
    }
    assert!(
        (prev - v_ref).abs() < 1e-3,
        "v after 2 s = {prev} m/s vs discrete closed form {v_ref} (tol 1e-3; drag on the \
         0.07 m/s suspension saw-tooth is < 1e-4 N)"
    );
    let speed_kmh = vehicle.speed_kmh.to_f64();
    assert!(
        (speed_kmh - prev * 3.6).abs() < 0.2,
        "speed_kmh {speed_kmh} vs 3.6 · {prev} (tol 0.2)"
    );
    // gear index is bounded by num_gears
    for _ in 0..10 {
        vehicle.shift_up();
    }
    assert_eq!(vehicle.current_gear, engine.num_gears - 1, "top gear index");
}

// ---------------------------------------------------------------------------
// SupportConfig::default() — support column volume is the closed-form sum of
// base + interface + bulk slabs, and never exceeds the solid column
// ---------------------------------------------------------------------------

#[test]
fn support_config_default_column_volume_matches_closed_form() {
    let cfg = SupportConfig::default();
    let (area, height) = (100.0, 10.0);
    let region = OverhangRegion {
        projected_area_mm2: Fix128::from_int(100),
        support_height_mm: Fix128::from_int(10),
    };
    let report = estimate_support_volume(&[region], &cfg);

    let lh = cfg.layer_height_mm.to_f64();
    let base_t = lh * f64::from(cfg.base_layers); // 0.6 mm
    let iface_t = lh * f64::from(cfg.interface_layers); // 0.4 mm
    let base = area * base_t * cfg.base_density.to_f64(); // 54
    let iface = area * iface_t * cfg.interface_density.to_f64(); // 40
    let bulk = area * (height - base_t - iface_t) * cfg.bulk_density.to_f64(); // 135
    let total = base + iface + bulk; // 229 mm³
    let d = cfg.filament_diameter_mm.to_f64();
    let length = total / (core::f64::consts::PI * (d / 2.0) * (d / 2.0)); // 95.21 mm
    let minutes = total / cfg.throughput_mm3_per_min.to_f64(); // 0.159 min

    let tol = 1e-6;
    for (name, got, want) in [
        ("base", report.base_volume_mm3.to_f64(), base),
        ("interface", report.interface_volume_mm3.to_f64(), iface),
        ("bulk", report.bulk_volume_mm3.to_f64(), bulk),
        ("total", report.total_volume_mm3.to_f64(), total),
        (
            "filament_length",
            report.filament_length_mm.to_f64(),
            length,
        ),
        ("time", report.estimated_time_min.to_f64(), minutes),
    ] {
        assert!(
            (got - want).abs() < tol,
            "{name}: {got} vs closed form {want} (tol {tol})"
        );
    }
    assert!(
        total < area * height,
        "support ({total} mm³) must be lighter than the solid column ({} mm³)",
        area * height
    );
    assert!(report.is_nontrivial());
    // two identical regions = twice the volume (linearity)
    let twice = estimate_support_volume(&[region, region], &cfg);
    assert!(
        (twice.total_volume_mm3.to_f64() - 2.0 * total).abs() < tol,
        "two regions: {} vs {} (tol {tol})",
        twice.total_volume_mm3.to_f64(),
        2.0 * total
    );
}

// ---------------------------------------------------------------------------
// CharacterConfig::default() — a character dropped above an SDF floor lands
// after the free-fall time, stands with its capsule bottom on the floor and
// walks 1 m/s without sinking or lifting
// ---------------------------------------------------------------------------

#[test]
fn character_config_default_lands_on_floor_and_walks_level() {
    let cfg = CharacterConfig::default();
    let floor = [floor_sdf()];
    let g = PhysicsConfig::default().gravity;
    let stand_y = cfg.height.to_f64() / 2.0; // capsule centre when the bottom touches y = 0
    let start_y = 2.0;
    let mut cc = CharacterController::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2), Fix128::ZERO),
        cfg,
    );
    let mut vel = Vec3Fix::ZERO;
    let mut landed_at = None;
    for frame in 1..=120 {
        if !cc.grounded {
            vel = vel + g * dt60();
        }
        let res = cc.move_and_slide(vel * dt60(), &[], &floor);
        if res.grounded {
            vel.y = Fix128::ZERO;
            if landed_at.is_none() {
                landed_at = Some(frame);
            }
        }
        assert!(
            cc.position.y.to_f64() >= stand_y - 0.01,
            "frame {frame}: character sank into the floor: y = {} (stand {stand_y}, tol 0.01)",
            cc.position.y.to_f64()
        );
    }
    let landed_at = landed_at.expect("never landed");
    // t = sqrt(2 h / g) = sqrt(2 · 1.1 / 10) = 0.469 s = 28 frames (symplectic Euler ± 2)
    let t_ref = (2.0 * (start_y - stand_y) / -g.y.to_f64()).sqrt();
    let frames_ref = t_ref / DT60;
    assert!(
        (landed_at as f64 - frames_ref).abs() <= 3.0,
        "landed at frame {landed_at}, free-fall closed form {frames_ref:.1} (tol 3 frames)"
    );
    assert!(cc.grounded, "not grounded after landing");
    // The controller reports `grounded` once the feet are within
    // `ground_probe_distance` (+ skin) of the floor and does not snap down, so
    // the standing height lies in [stand_y, stand_y + probe + skin] = [0.9, 1.01].
    let hover = cfg.ground_probe_distance.to_f64() + cfg.skin_width.to_f64();
    let y_landed = cc.position.y.to_f64();
    assert!(
        y_landed >= stand_y - 0.01 && y_landed <= stand_y + hover + 1e-6,
        "standing height {y_landed}, expected within the ground probe band [{stand_y}, {}]",
        stand_y + hover
    );

    // walk +x at 1 m/s for 1 s: x advances by exactly v t, height unchanged
    let x0 = cc.position.x.to_f64();
    let walk = Vec3Fix::from_int(1, 0, 0);
    for frame in 0..60 {
        let res = cc.move_and_slide(walk * dt60(), &[], &floor);
        assert!(res.grounded, "frame {frame}: lost the ground while walking");
        assert!(
            (cc.position.y.to_f64() - y_landed).abs() < 1e-9,
            "frame {frame}: height changed {y_landed} -> {} while walking on a flat floor (tol 1e-9)",
            cc.position.y.to_f64()
        );
    }
    let dx = cc.position.x.to_f64() - x0;
    assert!(
        (dx - 1.0).abs() < 1e-9,
        "walked {dx} m, expected v t = 1.0 (tol 1e-9)"
    );
    assert!(cc.position.z.is_zero(), "no lateral drift while walking");
}

// ---------------------------------------------------------------------------
// NetcodeConfig::default() — two clients with the same inputs stay checksum
// identical, a player moves x = v t, and a rollback + replay is bit-exact
// ---------------------------------------------------------------------------

fn make_sim() -> DeterministicSimulation {
    let mut sim = DeterministicSimulation::new(NetcodeConfig::default());
    let p0 = sim.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 10, 0),
        Fix128::ONE,
    ));
    let p1 = sim.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(5, 10, 0),
        Fix128::ONE,
    ));
    sim.assign_player_body(0, p0);
    sim.assign_player_body(1, p1);
    sim
}

fn inputs(frame: u64) -> [FrameInput; 2] {
    // player 0 walks +x, player 1 walks -z and jumps on frame 3
    let jump = u32::from(frame == 3);
    [
        FrameInput::new(0).with_movement(Vec3Fix::from_int(1, 0, 0)),
        FrameInput::new(1)
            .with_movement(Vec3Fix::from_int(0, 0, -1))
            .with_actions(jump),
    ]
}

#[test]
fn netcode_config_default_lockstep_clients_agree_and_rollback_is_bit_exact() {
    let cfg = NetcodeConfig::default();
    let mut a = make_sim();
    let mut b = make_sim();
    let mut snapshot_frame = None;
    let mut checksums = Vec::new();
    for frame in 1..=60u64 {
        let ca = a.advance_frame(&inputs(frame));
        let cb = b.advance_frame(&inputs(frame));
        assert_eq!(ca, cb, "frame {frame}: clients desynced");
        assert_eq!(a.verify_checksum(frame, cb), Some(true));
        checksums.push(ca);
        if frame == 30 {
            let snap = a.save_snapshot();
            snapshot_frame = Some(snap.frame);
        }
    }
    assert_eq!(snapshot_frame, Some(30));
    // x = v t with the default applicator (5 m/s · 1 s), bit-exact in fixed point
    let px = a.world.bodies[0].position.x.to_f64();
    assert!(
        (px - 5.0).abs() < 1e-12,
        "player 0 x = {px}, expected v t = 5.0 (tol 1e-12)"
    );
    let pz = a.world.bodies[1].position.z.to_f64();
    assert!(
        (pz + 5.0).abs() < 1e-12,
        "player 1 z = {pz}, expected -5.0 (tol 1e-12)"
    );
    assert_eq!(a.world.bodies[0].position, b.world.bodies[0].position);

    // rollback to frame 30 and replay the same inputs → identical frame-60 state
    let final_bodies: Vec<RigidBody> = a.world.bodies.clone();
    assert!(a.load_snapshot(30), "snapshot 30 missing");
    assert_eq!(a.frame(), 30);
    assert_eq!(a.checksum(), checksums[29], "restored state checksum");
    assert_eq!(a.checksum_at(31), None, "history trimmed past the rollback");
    for frame in 31..=60u64 {
        let c = a.advance_frame(&inputs(frame));
        assert_eq!(c, checksums[frame as usize - 1], "replay frame {frame}");
    }
    assert_eq!(
        a.world.bodies, final_bodies,
        "replayed bodies differ bit-wise"
    );

    // ring buffers honour the defaults
    for _ in 0..(cfg.max_snapshots + 5) {
        a.save_snapshot();
    }
    assert_eq!(a.snapshot_count(), cfg.max_snapshots);
    assert!(a.get_snapshot(30).is_none(), "oldest snapshot evicted");
    for frame in 61..=(cfg.checksum_history_len as u64 + 1) {
        a.advance_frame(&inputs(frame));
    }
    assert_eq!(a.checksum_at(1), None, "checksum history bounded");
    assert!(a.checksum_at(2).is_some());
    assert_eq!(a.dt(), cfg.fixed_dt);
}

// ---------------------------------------------------------------------------
// AudioConfig::default() — a 5 m/s wood-on-wood impact: volume sqrt(v/v_max)
// = 0.5, pitch (1 + 0.25) · 1000/600, decay 0.42 s; thresholds and per-frame
// cap behave as documented
// ---------------------------------------------------------------------------

fn contact() -> Contact {
    Contact {
        depth: r(1, 100),
        normal: Vec3Fix::UNIT_Y,
        point_a: Vec3Fix::from_int(1, 2, 3),
        point_b: Vec3Fix::from_int(1, 2, 3),
    }
}

#[test]
fn audio_config_default_impact_parameters_match_closed_form() {
    let cfg = AudioConfig::default();
    let mut gen = AudioGenerator::new(2, cfg);
    gen.begin_frame();
    gen.process_contact(0, 1, &contact(), Vec3Fix::from_int(0, -5, 0), true);
    assert_eq!(gen.get_events().len(), 1);
    let e = gen.get_events()[0];
    let wood = AudioMaterial::WOOD;
    let v_max = cfg.max_velocity.to_f64();
    let volume_ref = (5.0 / v_max).sqrt(); // 0.5
    let pitch_ref =
        (1.0 + 5.0 * cfg.velocity_pitch_factor.to_f64()) * (1000.0 / wood.density.to_f64());
    let decay_ref = 0.1 + 2.0 * wood.resonance.to_f64() * (1.0 - wood.damping.to_f64()); // 0.42
    let bright_ref = wood.hardness.to_f64() * (0.5 + 0.5 * 5.0 / v_max); // 0.375
    let tol = 1e-6;
    for (name, got, want) in [
        ("volume", e.volume.to_f64(), volume_ref),
        ("pitch", e.pitch.to_f64(), pitch_ref),
        ("decay", e.decay.to_f64(), decay_ref),
        ("brightness", e.brightness.to_f64(), bright_ref),
        ("roughness", e.roughness.to_f64(), 0.0),
    ] {
        assert!(
            (got - want).abs() < tol,
            "{name}: {got} vs closed form {want} (tol {tol})"
        );
    }
    assert_eq!(e.event_type, AudioEventType::Impact);
    assert_eq!(e.position, Vec3Fix::from_int(1, 2, 3));

    // loudness is monotone in speed and saturates at max_velocity
    let mut prev = 0.0;
    for speed in [1i64, 5, 10, 20, 40] {
        gen.begin_frame();
        gen.process_contact(0, 1, &contact(), Vec3Fix::from_int(0, -speed, 0), true);
        let v = gen.get_events()[0].volume.to_f64();
        assert!(v >= prev, "volume fell at {speed} m/s: {prev} -> {v}");
        prev = v;
    }
    assert!(
        (prev - 1.0).abs() < tol,
        "40 m/s volume {prev} should saturate at 1.0 (tol {tol})"
    );

    // below min_velocity: silent
    gen.begin_frame();
    gen.process_contact(
        0,
        1,
        &contact(),
        Vec3Fix::new(Fix128::ZERO, -r(5, 100), Fix128::ZERO),
        true,
    );
    assert!(
        gen.get_events().is_empty(),
        "0.05 m/s < min_velocity must be silent"
    );

    // persistent contact: tangential 1 m/s > slide threshold → Slide, 0.3 m/s → Roll
    gen.begin_frame();
    gen.process_contact(0, 1, &contact(), Vec3Fix::from_int(1, 0, 0), false);
    gen.process_contact(
        0,
        1,
        &contact(),
        Vec3Fix::new(r(3, 10), Fix128::ZERO, Fix128::ZERO),
        false,
    );
    let kinds: Vec<AudioEventType> = gen.get_events().iter().map(|e| e.event_type).collect();
    assert_eq!(kinds, [AudioEventType::Slide, AudioEventType::Roll]);

    // per-frame cap
    gen.begin_frame();
    for _ in 0..(cfg.max_events_per_frame + 8) {
        gen.process_contact(0, 1, &contact(), Vec3Fix::from_int(0, -5, 0), true);
    }
    assert_eq!(gen.get_events().len(), cfg.max_events_per_frame);
}

// ---------------------------------------------------------------------------
// ControllerConfig::default() — a 117 → 24 identity-like ternary network that
// copies each observed body's velocity into its joint torque: the controller
// output equals the velocity bit-exactly and is clamped to ± max_torque
// ---------------------------------------------------------------------------

#[cfg(feature = "neural")]
#[test]
fn controller_config_default_velocity_passthrough_network_is_exact_and_clamped() {
    use alice_ml::TernaryWeight;
    use alice_physics::neural::{
        Activation, ControllerConfig, DeterministicNetwork, FixedTernaryWeight, RagdollController,
        FEATURES_PER_BODY,
    };

    let cfg = ControllerConfig::default();
    assert_eq!(cfg.features_per_body, FEATURES_PER_BODY);
    let (n_in, n_out) = (cfg.num_bodies * cfg.features_per_body, cfg.num_joints * 3);
    // row (joint j, axis a) ← +1 at body j's velocity component a (feature 3 + a)
    let mut values = vec![0i8; n_out * n_in];
    for joint in 0..cfg.num_joints {
        for axis in 0..3 {
            let row = joint * 3 + axis;
            let col = joint * cfg.features_per_body + 3 + axis;
            values[row * n_in + col] = 1;
        }
    }
    let weight = FixedTernaryWeight::from_ternary_weight_with_scale(
        TernaryWeight::from_ternary(&values, n_out, n_in),
        Fix128::ONE,
    );
    let network = DeterministicNetwork::new(vec![weight], vec![Activation::None]);
    let mut controller = RagdollController::new(network, cfg.clone());

    let mut bodies = Vec::new();
    for i in 0..cfg.num_bodies as i64 {
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(i, 1, -i), Fix128::ONE);
        b.velocity = Vec3Fix::new(r(i, 3), r(-i, 7), Fix128::from_int(i * 30));
        bodies.push(b);
    }
    let out = controller.compute(&bodies);
    assert_eq!(out.torques.len(), cfg.num_joints);
    let max = cfg.max_torque;
    for (j, t) in out.torques.iter().enumerate() {
        let v = bodies[j].velocity;
        assert_eq!(t.x, v.x, "joint {j}: x torque must equal vx bit-exactly");
        assert_eq!(t.y, v.y, "joint {j}: y torque must equal vy bit-exactly");
        let want_z = if v.z > max { max } else { v.z };
        assert_eq!(t.z, want_z, "joint {j}: z torque clamped to max_torque");
        assert!(t.z <= max && t.z >= -max);
    }
    // 30 · j exceeds max_torque = 100 from joint 4 on
    assert_eq!(out.torques[4].z, max);
    assert_eq!(out.torques[3].z, Fix128::from_int(90));
    // determinism: a second evaluation is bit-identical
    let again = controller.compute(&bodies);
    for (a, b) in out.torques.iter().zip(&again.torques) {
        assert_eq!(a, b);
    }
    // fewer bodies than num_bodies: missing observations are zero → zero torque
    let out = controller.compute(&bodies[..2]);
    assert_eq!(out.torques[2], Vec3Fix::ZERO);
    assert_eq!(out.torques[1], bodies[1].velocity);
}
