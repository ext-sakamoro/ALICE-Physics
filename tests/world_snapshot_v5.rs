//! Version 5 world snapshot (the broad-phase tree written as its leaf set)
//! and the frozen blobs of earlier releases, through the public API only.
//!
//! oracle: blobs written by released builds, committed under
//! `tests/fixtures/` (`world_snapshot_v4_*.bin` by the 2.1.0 tag,
//! `world_snapshot_v3_*.bin` by the 2.0.0 tag), and, for each, the same
//! world stepped `FIXTURE_LATER_STEPS` more times by that release
//! (`*_later.bin`). A frozen blob restored here and stepped must reach the
//! state the release itself reached. The version 5 blobs are pinned against
//! the current writer. How every fixture was generated is recorded in
//! `tests/fixtures/README.md`; the scenes are `tests/fixtures/world_snapshot_scenes.rs`.

use alice_physics::{LawCheck, PhysicsWorld, WorldSnapshotError, PHYSICS_SEMANTICS_ID};

include!("fixtures/world_snapshot_scenes.rs");

const VERSION_AT: usize = 4;
const SEMANTICS_AT: usize = 16;
const LAW_AT: usize = 48;
const CHECKSUM: usize = 8;

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!("fixtures/world_snapshot_", $name, ".bin")).as_slice()
    };
}

/// (scene, version 5, version 4 by 2.1.0, its later state by 2.1.0)
type V4Fixture = (&'static str, &'static [u8], &'static [u8], &'static [u8]);

fn v4_fixtures() -> [V4Fixture; 5] {
    [
        (
            "joint_pair",
            fixture!("v5_joint_pair"),
            fixture!("v4_joint_pair"),
            fixture!("v4_joint_pair_later"),
        ),
        (
            "dynamic_tree_freed",
            fixture!("v5_dynamic_tree_freed"),
            fixture!("v4_dynamic_tree_freed"),
            fixture!("v4_dynamic_tree_freed_later"),
        ),
        (
            "dynamic_tree",
            fixture!("v5_dynamic_tree"),
            fixture!("v4_dynamic_tree"),
            fixture!("v4_dynamic_tree_later"),
        ),
        (
            "bvh",
            fixture!("v5_bvh"),
            fixture!("v4_bvh"),
            fixture!("v4_bvh_later"),
        ),
        (
            "hybrid",
            fixture!("v5_hybrid"),
            fixture!("v4_hybrid"),
            fixture!("v4_hybrid_later"),
        ),
    ]
}

/// (scene, version 3 by 2.0.0, its later state by 2.0.0)
fn v3_fixtures() -> [(&'static str, &'static [u8], &'static [u8]); 5] {
    [
        (
            "joint_pair",
            fixture!("v3_joint_pair"),
            fixture!("v3_joint_pair_later"),
        ),
        (
            "dynamic_tree_freed",
            fixture!("v3_dynamic_tree_freed"),
            fixture!("v3_dynamic_tree_freed_later"),
        ),
        (
            "dynamic_tree",
            fixture!("v3_dynamic_tree"),
            fixture!("v3_dynamic_tree_later"),
        ),
        ("bvh", fixture!("v3_bvh"), fixture!("v3_bvh_later")),
        ("hybrid", fixture!("v3_hybrid"), fixture!("v3_hybrid_later")),
    ]
}

fn version(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[VERSION_AT], b[VERSION_AT + 1]])
}

fn id(b: &[u8], at: usize) -> [u8; 32] {
    b[at..at + 32].try_into().expect("32 bytes")
}

/// Recompute the trailing FNV-1a 64 so only the targeted check can fire.
fn reseal(mut b: Vec<u8>) -> Vec<u8> {
    let n = b.len() - CHECKSUM;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in &b[..n] {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x0000_0001_0000_01b3);
    }
    b[n..].copy_from_slice(&h.to_le_bytes());
    b
}

/// Position, velocity, rotation and angular velocity of every body, compared
/// apart from the encoder.
fn bodies(w: &PhysicsWorld) -> Vec<String> {
    w.bodies
        .iter()
        .map(|b| {
            format!(
                "{:?} {:?} {:?} {:?}",
                b.position, b.velocity, b.rotation, b.angular_velocity
            )
        })
        .collect()
}

fn scene(name: &str) -> PhysicsWorld {
    fixture_scenes()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, w)| w)
        .expect("scene")
}

/// A restored world re-serializes to a blob that restores to itself
/// (bytes and bodies).
fn assert_fixed_point(w: &PhysicsWorld, ctx: &str) {
    let b = w.snapshot_world();
    assert_eq!(version(&b), 5, "{ctx}");
    let r = PhysicsWorld::from_world_snapshot(&b).expect(ctx);
    assert_eq!(r.snapshot_world(), b, "{ctx}: fixed point");
    assert_eq!(bodies(&r), bodies(w), "{ctx}: bodies");
}

/// `blob` restored here and stepped `FIXTURE_LATER_STEPS` times reaches the
/// state `later` holds, which the writing release reached by stepping the
/// same world; both are compared through this build's writer and body by
/// body.
fn assert_steps_to(blob: &[u8], later: &[u8], ctx: &str) {
    let mut r = PhysicsWorld::from_world_snapshot(blob).expect(ctx);
    let want = PhysicsWorld::from_world_snapshot(later).expect(ctx);
    r.step_n(FIXTURE_LATER_STEPS, fixture_dt());
    assert_eq!(bodies(&r), bodies(&want), "{ctx}: bodies after stepping");
    assert_eq!(
        r.snapshot_world(),
        want.snapshot_world(),
        "{ctx}: state after stepping"
    );
}

#[test]
fn the_writer_emits_version_5() {
    assert_eq!(PhysicsWorld::WORLD_SNAPSHOT_VERSION, 5);
    for (name, w) in fixture_scenes() {
        assert_eq!(version(&w.snapshot_world()), 5, "{name}");
    }
}

/// The current writer's output is the committed version 5 fixture, byte for
/// byte, for every scene.
#[test]
fn the_writer_matches_the_version_5_fixtures() {
    for (name, v5, _, _) in v4_fixtures() {
        assert_eq!(scene(name).snapshot_world(), v5, "{name}");
    }
}

/// Every version 5 fixture restores to a world that writes the same bytes,
/// carries this build's identifiers, and steps on like the scene built
/// here.
#[test]
fn version_5_fixtures_round_trip_bit_exact() {
    for (name, v5, _, _) in v4_fixtures() {
        let mut live = scene(name);
        let law = live.law_id(&PHYSICS_SEMANTICS_ID);
        assert_eq!(id(v5, SEMANTICS_AT), PHYSICS_SEMANTICS_ID, "{name}");
        assert_eq!(id(v5, LAW_AT), law, "{name}");
        let mut r = PhysicsWorld::new(alice_physics::solver::SolverConfig::default());
        assert_eq!(
            r.restore_world_checked(v5, Some(&law)),
            Ok(LawCheck::Verified),
            "{name}"
        );
        assert_eq!(r.snapshot_world(), v5, "{name}");
        for i in 0..FIXTURE_LATER_STEPS {
            live.step(fixture_dt());
            r.step(fixture_dt());
            assert_eq!(r.snapshot_world(), live.snapshot_world(), "{name} step {i}");
        }
    }
}

/// Every version 4 blob written by 2.1.0 (the broad-phase tree node by
/// node, freed nodes included for `dynamic_tree_freed`) is read: its
/// identifiers are this build's, the checked restore verifies them, the
/// restored world re-serializes as version 5 to a fixed point, and stepping
/// it reaches the state 2.1.0 reached.
#[test]
fn version_4_fixtures_of_2_1_0_restore_and_step_like_2_1_0() {
    for (name, _, v4, later) in v4_fixtures() {
        assert_eq!(version(v4), 4, "{name}");
        assert_eq!(version(later), 4, "{name}");
        assert_eq!(id(v4, SEMANTICS_AT), PHYSICS_SEMANTICS_ID, "{name}");
        let law = id(v4, LAW_AT);
        let mut r = PhysicsWorld::new(alice_physics::solver::SolverConfig::default());
        assert_eq!(
            r.restore_world_checked(v4, Some(&law)),
            Ok(LawCheck::Verified),
            "{name}"
        );
        assert_fixed_point(&r, name);
        assert_steps_to(v4, later, name);
        assert_fixed_point(&PhysicsWorld::from_world_snapshot(later).expect(name), name);
    }
}

/// Every 2.1.0 blob restores to the world this build constructs from the
/// same scene: the same version 5 bytes and the same rule identifier. (The
/// wake rule of `remove_body` changed after 2.1.0; in `dynamic_tree_freed`
/// no body is asleep when the extra body is removed, so both builds reach
/// the same world.)
#[test]
fn version_4_fixtures_of_2_1_0_hold_the_scene_built_here() {
    for (name, v5, v4, _) in v4_fixtures() {
        let w = scene(name);
        assert_eq!(id(v4, LAW_AT), w.law_id(&PHYSICS_SEMANTICS_ID), "{name}");
        let r = PhysicsWorld::from_world_snapshot(v4).expect(name);
        assert_eq!(bodies(&r), bodies(&w), "{name}");
        assert_eq!(r.snapshot_world(), v5, "{name}");
    }
}

/// Every version 3 blob written by 2.0.0 is read as `Unpinned`, re-serializes
/// as version 5 to a fixed point, and stepping it reaches the state 2.0.0
/// reached.
#[test]
fn version_3_fixtures_of_2_0_0_restore_and_step_like_2_0_0() {
    for (name, v3, later) in v3_fixtures() {
        assert_eq!(version(v3), 3, "{name}");
        assert_eq!(version(later), 3, "{name}");
        for expected in [None, Some(&[0xAB; 32])] {
            let mut r = PhysicsWorld::new(alice_physics::solver::SolverConfig::default());
            assert_eq!(
                r.restore_world_checked(v3, expected),
                Ok(LawCheck::Unpinned),
                "{name}"
            );
            assert_fixed_point(&r, name);
        }
        assert_steps_to(v3, later, name);
    }
}

/// A version 6 header (checksum recomputed) is refused by version, whatever
/// the payload, and the target is untouched.
#[test]
fn version_6_is_refused() {
    for (name, v5, _, _) in v4_fixtures() {
        let mut x = v5.to_vec();
        x[VERSION_AT..VERSION_AT + 2].copy_from_slice(&6u16.to_le_bytes());
        let x = reseal(x);
        let mut t = scene("joint_pair");
        let before = t.snapshot_world();
        assert_eq!(
            t.restore_world(&x),
            Err(WorldSnapshotError::UnsupportedVersion {
                found: 6,
                supported: 5
            }),
            "{name}"
        );
        assert_eq!(t.snapshot_world(), before, "{name}");
    }
}

/// The version alone picks the tree reader: a 2.1.0 blob relabelled as
/// version 5, and a version 5 blob relabelled as version 4 (checksums
/// recomputed), are both refused, and neither restores silently.
#[test]
fn the_version_picks_the_tree_reader() {
    for (name, v5, v4, _) in v4_fixtures() {
        let mut x = v4.to_vec();
        x[VERSION_AT..VERSION_AT + 2].copy_from_slice(&5u16.to_le_bytes());
        assert!(
            PhysicsWorld::from_world_snapshot(&reseal(x)).is_err(),
            "{name}: version 4 tree read as version 5"
        );
        let mut x = v5.to_vec();
        x[VERSION_AT..VERSION_AT + 2].copy_from_slice(&4u16.to_le_bytes());
        assert!(
            PhysicsWorld::from_world_snapshot(&reseal(x)).is_err(),
            "{name}: version 5 tree read as version 4"
        );
    }
}

/// Every 2.0.0 blob restores to the world this build constructs from the
/// same scene (the same version 5 bytes).
#[test]
fn version_3_fixtures_of_2_0_0_hold_the_scene_built_here() {
    for (name, v3, _) in v3_fixtures() {
        let w = scene(name);
        let r = PhysicsWorld::from_world_snapshot(v3).expect(name);
        assert_eq!(bodies(&r), bodies(&w), "{name}");
        assert_eq!(r.snapshot_world(), w.snapshot_world(), "{name}");
    }
}
