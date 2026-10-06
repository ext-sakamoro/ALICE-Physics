//! Record a replay and simulation metrics in process memory, take the
//! recording out as bytes and play it back from those bytes.
//!
//! This path needs no filesystem, so it is the one to use in a browser
//! (`wasm32-unknown-unknown`): keep the bytes wherever the host stores data.
//!
//! Run: `cargo run --example replay_in_memory --features replay`

use alice_physics::db_bridge::PhysicsMetricsSink;
use alice_physics::replay::{ReplayPlayer, ReplayRecorder};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn main() -> std::io::Result<()> {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 20, 0),
        Fix128::ONE,
    ));
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(2, 30, 0),
        Fix128::ONE,
    ));

    let mut recorder = ReplayRecorder::in_memory(world.bodies.len())?;
    let metrics = PhysicsMetricsSink::in_memory()?;
    let dt = Fix128::from_ratio(1, 60);
    let mut last = (0.0_f32, 0.0_f32, 0.0_f32);
    for step in 0..120 {
        world.step(dt);
        recorder.record_frame(&world)?;
        let energy: f32 = world
            .bodies
            .iter()
            .map(|b| {
                let (vx, vy, vz) = b.velocity.to_f32();
                0.5 * (vx * vx + vy * vy + vz * vz)
            })
            .sum();
        metrics.record_energy(step, energy)?;
        last = world.bodies[0].position.to_f32();
    }

    let bytes = recorder.to_bytes()?;
    println!(
        "recording: {} frames, {} bytes",
        recorder.frame_count(),
        bytes.len()
    );

    // the layout travels with the bytes, so the body count passed here is
    // only a fallback for recordings without one
    let player = ReplayPlayer::from_bytes(&bytes, 0)?;
    let replayed = player
        .get_position(119, 0)?
        .expect("frame 119 was recorded");
    println!("body 0 at frame 119: world {last:?}, replay {replayed:?}");
    assert_eq!(
        (
            replayed.0.to_bits(),
            replayed.1.to_bits(),
            replayed.2.to_bits()
        ),
        (last.0.to_bits(), last.1.to_bits(), last.2.to_bits()),
        "the replay reads back the bits the world held"
    );

    // series are queried after a flush, as with the file-backed sink
    metrics.flush()?;
    let energy = metrics.query_energy(0, 119)?;
    println!("kinetic energy samples: {}", energy.len());
    assert_eq!(energy.len(), 120, "every recorded step reads back");
    Ok(())
}
