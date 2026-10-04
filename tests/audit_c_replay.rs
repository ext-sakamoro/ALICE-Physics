//! Audit oracles for `replay`: a recorded simulation reads back the engine state
//! of every frame bit for bit (as the `to_f32` it stores), the state itself equals
//! the closed-form trajectory, and a second run of the same deterministic world
//! reproduces the played-back data.
//!
//! Expected values: free fall from rest with 8 substeps of `h = 1/512`,
//! `y_n = y0 - 10 h^2 n (n + 1) / 2`, `v_n = -10 h n`; a weightless body with
//! velocity `u` at `x0 + u t`; a static body fixed. The recorder keys are not used
//! by the oracle (only the public player API).

#![cfg(all(feature = "std", feature = "replay"))]

use alice_physics::replay::{ReplayPlayer, ReplayRecorder};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

const FRAMES: u64 = 24;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn bits(v: (f32, f32, f32)) -> [u32; 3] {
    [v.0.to_bits(), v.1.to_bits(), v.2.to_bits()]
}

fn fbits(v: Vec3Fix) -> [u32; 3] {
    bits(v.to_f32())
}

/// Falling body, weightless coasting body, static body; no frame damping.
fn scene() -> PhysicsWorld {
    let config = PhysicsConfig {
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    assert_eq!(config.substeps, 8, "closed forms assume 8 substeps");
    let mut w = PhysicsWorld::new(config);
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(1, 10, -2),
        Fix128::ONE,
    ));
    let mut coast = RigidBody::new_dynamic(Vec3Fix::from_int(-4, 2, 0), Fix128::ONE)
        .with_gravity_scale(Fix128::ZERO);
    coast.velocity = Vec3Fix::from_int(3, 0, -1);
    w.add_body(coast);
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    w
}

/// Closed-form state after `frame + 1` steps: (position, velocity) per body.
fn closed_form(frame: u64) -> [(Vec3Fix, Vec3Fix); 3] {
    let n = 8 * (i64::try_from(frame).expect("small") + 1);
    let y = Fix128::from_int(10) - Fix128::from_ratio(10 * n * (n + 1) / 2, 512 * 512);
    let vy = Fix128::from_ratio(-10 * n, 512);
    let t = Fix128::from_ratio(n, 512);
    [
        (
            Vec3Fix::new(Fix128::ONE, y, Fix128::from_int(-2)),
            Vec3Fix::new(Fix128::ZERO, vy, Fix128::ZERO),
        ),
        (
            Vec3Fix::new(
                Fix128::from_int(-4) + Fix128::from_int(3) * t,
                Fix128::from_int(2),
                -t,
            ),
            Vec3Fix::from_int(3, 0, -1),
        ),
        (Vec3Fix::from_int(0, -1, 0), Vec3Fix::ZERO),
    ]
}

/// Record every frame of the scene; the engine state equals the closed form and the
/// player returns its `to_f32` bit for bit, for positions and velocities.
#[test]
fn recorded_frames_equal_the_engine_state_and_the_closed_form() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fall");
    let mut world = scene();
    let mut rec = ReplayRecorder::new(&path, 3).unwrap();
    for f in 0..FRAMES {
        world.step(dt());
        let want = closed_form(f);
        for (b, (p, v)) in want.iter().enumerate() {
            assert_eq!(world.bodies[b].position, *p, "engine position f{f} b{b}");
            assert_eq!(world.bodies[b].velocity, *v, "engine velocity f{f} b{b}");
        }
        rec.record_frame(&world).unwrap();
    }
    assert_eq!(rec.frame_count(), FRAMES);
    rec.close().unwrap();

    let player = ReplayPlayer::open(&path, 3).unwrap();
    for f in 0..FRAMES {
        let want = closed_form(f);
        for (b, (p, v)) in want.iter().enumerate() {
            let gp = player.get_position(f, b).unwrap().expect("position");
            let gv = player.get_velocity(f, b).unwrap().expect("velocity");
            assert_eq!(bits(gp), fbits(*p), "position f{f} b{b}");
            assert_eq!(bits(gv), fbits(*v), "velocity f{f} b{b}");
        }
    }
    // nothing beyond the recorded frames or bodies
    assert!(player.get_position(FRAMES, 0).unwrap().is_none());
    assert!(player.get_position(0, 3).unwrap().is_none());
    player.close().unwrap();
}

/// Playback reproduces a second, independent run of the same world: every frame of
/// the second run equals the played-back positions bit for bit, and a position-only
/// recording of that run carries no velocity.
#[test]
fn playback_reproduces_an_independent_rerun_of_the_world() {
    let dir = tempfile::tempdir().unwrap();
    let full = dir.path().join("full");
    let pos_only = dir.path().join("pos");
    {
        let mut world = scene();
        let mut rec = ReplayRecorder::new(&full, 3).unwrap();
        for _ in 0..FRAMES {
            world.step(dt());
            rec.record_frame(&world).unwrap();
        }
        rec.close().unwrap();
    }
    let mut rerun = scene();
    let mut rec = ReplayRecorder::new(&pos_only, 3).unwrap();
    let player = ReplayPlayer::open(&full, 3).unwrap();
    for f in 0..FRAMES {
        rerun.step(dt());
        rec.record_positions(&rerun).unwrap();
        for b in 0..3 {
            let gp = player.get_position(f, b).unwrap().expect("position");
            assert_eq!(bits(gp), fbits(rerun.bodies[b].position), "f{f} b{b}");
        }
    }
    rec.close().unwrap();
    let scan = player.scan_positions(0, 0, FRAMES - 1).unwrap();
    assert_eq!(scan.len(), usize::try_from(FRAMES).unwrap());
    for (i, (frame, x, y, z)) in scan.iter().enumerate() {
        assert_eq!(*frame, i as u64);
        assert_eq!(bits((*x, *y, *z)), fbits(closed_form(*frame)[0].0));
    }
    player.close().unwrap();

    let pos_player = ReplayPlayer::open(&pos_only, 3).unwrap();
    for f in 0..FRAMES {
        for b in 0..3 {
            let gp = pos_player.get_position(f, b).unwrap().expect("position");
            assert_eq!(bits(gp), fbits(closed_form(f)[b].0), "pos-only f{f} b{b}");
            assert!(pos_player.get_velocity(f, b).unwrap().is_none());
        }
    }
    pos_player.close().unwrap();
}
