//! Reading a world snapshot written before damping became a rate per second.
//!
//! A snapshot of format 1 to 5 stores damping as a retention per step (0.99 here). Without
//! the step that retention has no per-second meaning, so `restore_world_checked` refuses it
//! and names the retention; `restore_world_legacy_damping` with the writer's step converts it
//! to the rate `−ln(retention) / step` and reads the rest of the world unchanged.
//!
//! oracle: −ln(0.99) · 60 = 0.603020151210086…, computed apart from the library.
//!
//! cargo run --release --example restore_legacy_snapshot

use alice_physics_core::{
    math::Fix128, solver::PhysicsConfig, LawCheck, LegacyPolicy, PhysicsWorld, WorldSnapshotError,
};

/// A world written by the build where damping was a retention per step (0.99, at 60 steps
/// per second)
const LEGACY: &[u8] = include_bytes!("../tests/fixtures/world_snapshot_v2_stacked.bin");

fn main() {
    let step = Fix128::from_ratio(1, 60);
    let mut world = PhysicsWorld::new(PhysicsConfig::default());

    // without the writer's step the retention cannot become a rate: refused, nothing read
    let refused = world.restore_world_checked(LEGACY, None, LegacyPolicy::AcceptUnpinned);
    assert_eq!(
        refused,
        Err(WorldSnapshotError::LegacyDampingNeedsStep {
            retention: Fix128::from_ratio(99, 100)
        })
    );
    assert!(
        world.bodies.is_empty(),
        "a refused snapshot must leave the world as it was"
    );
    println!("without the step: {}", refused.unwrap_err());

    // with it: read as legacy data (no identifiers), the retention converted
    assert_eq!(
        world.restore_world_legacy_damping(LEGACY, step),
        Ok(LawCheck::Unpinned)
    );
    let rate = world.config.damping_rate.to_f64();
    let want = 0.603_020_151_210_086_5;
    assert!((rate - want).abs() < 1e-12, "rate {rate}, expected {want}");
    println!(
        "with a 1/60 s step: {} bodies, damping rate {rate:.12} per second",
        world.bodies.len()
    );

    // every body's own damping was retention 1 (none): rate 0
    assert!(world
        .bodies
        .iter()
        .all(|b| b.linear_damping_rate == Fix128::ZERO && b.angular_damping_rate == Fix128::ZERO));
}
