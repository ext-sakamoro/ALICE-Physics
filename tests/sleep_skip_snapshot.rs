//! Oracles for the interplay of the sleep skip and the whole-world snapshot.
//!
//! The sleep skip keeps a derived parked-body state (which bodies are skipped, the
//! tree over them) between steps. A snapshot restore replaces the body population,
//! so that derived state must not survive it: stepping a restored world has to give
//! exactly what stepping a world built from the same blob gives.

use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SleepState, Vec3Fix};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// `n` bodies on a 2 m grid at `y = 0`; the first `awake` are lifted to `y = 100`
/// (falling freely), the rest are put to sleep at rest.
fn grid_world(n: usize, awake: usize) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let side = (n as f64).sqrt().ceil() as i64;
    for i in 0..n {
        let (x, y, z) = if i < awake {
            ((i as i64 % 32) * 2, 100, (i as i64 / 32) * 2)
        } else {
            ((i as i64 % side) * 2, 0, (i as i64 / side) * 2)
        };
        w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(x, y, z), Fix128::ONE),
            Fix128::from_ratio(1, 2),
        );
    }
    for i in awake..n {
        w.islands.sleep_data[i].state = SleepState::Sleeping;
        w.islands.sleep_data[i].idle_frames = 100;
    }
    w
}

fn run(w: &mut PhysicsWorld, frames: usize) {
    for _ in 0..frames {
        w.step(dt());
    }
}

/// Restoring a snapshot into a world whose sleep skip already parked other bodies
/// gives the same future as a world built from the blob.
#[test]
fn restoring_into_a_world_with_parked_bodies_matches_a_fresh_world() {
    let mut donor = grid_world(16, 4);
    donor.set_sleep_skip(true);
    run(&mut donor, 3);
    let blob = donor.snapshot_world();

    // a different, larger population that has parked many bodies
    let mut target = grid_world(64, 8);
    target.set_sleep_skip(true);
    run(&mut target, 3);
    // the scenario has teeth only if the skip really parked bodies before the restore
    assert!(
        target.stage_work().parked_sleep_updates > 0,
        "nothing was parked"
    );
    target.restore_world(&blob).expect("restore");

    let mut fresh = PhysicsWorld::from_world_snapshot(&blob).expect("fresh");
    fresh.set_sleep_skip(true);

    run(&mut target, 20);
    run(&mut fresh, 20);
    assert_eq!(target.snapshot_world(), fresh.snapshot_world());
}

/// The same restore with the skip on both sides and the donor's bodies asleep, so
/// the restored world has something to park as well.
#[test]
fn a_restored_all_sleeping_world_steps_like_a_fresh_one() {
    let mut donor = grid_world(32, 0);
    donor.set_sleep_skip(true);
    run(&mut donor, 3);
    let blob = donor.snapshot_world();

    let mut target = grid_world(32, 8);
    target.set_sleep_skip(true);
    run(&mut target, 3);
    target.restore_world(&blob).expect("restore");

    let mut fresh = PhysicsWorld::from_world_snapshot(&blob).expect("fresh");
    fresh.set_sleep_skip(true);

    run(&mut target, 10);
    run(&mut fresh, 10);
    assert_eq!(target.snapshot_world(), fresh.snapshot_world());
}

/// Skip on and skip off give the same snapshot after a restore (the skip is an
/// optimisation only).
#[test]
fn skip_on_and_off_agree_after_a_restore() {
    let mut donor = grid_world(16, 4);
    run(&mut donor, 2);
    let blob = donor.snapshot_world();

    let mut on = grid_world(48, 6);
    on.set_sleep_skip(true);
    run(&mut on, 3);
    on.restore_world(&blob).expect("restore");

    let mut off = PhysicsWorld::from_world_snapshot(&blob).expect("off");
    off.set_sleep_skip(false);

    run(&mut on, 15);
    run(&mut off, 15);
    assert_eq!(on.snapshot_world(), off.snapshot_world());
}
