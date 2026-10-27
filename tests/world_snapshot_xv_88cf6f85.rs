//! Cross-version oracle of `88cf6f85…`, the stepping semantics identifier of the build before the
//! secondary chain was pinned (main `b672ff2e`, `alice-det-math` 0.5.2 at 2d01c63f).
//!
//! Pinning the secondary chain added one entry to `PHYSICS_SEMANTICS_PINS` and changed no other,
//! so every path the earlier identifier pinned steps to the same bits here. Two checks show it:
//!
//! * the fold of this build's pins without the `"secondary chain"` entry is `88cf6f85…` itself;
//! * `tests/fixtures/world_snapshot_xv_88cf6f85_a.bin` is `scene()` stepped 30 times by `dt()` and
//!   written by `snapshot_world` with the build at `b672ff2e`, and `…_b.bin` the same world stepped
//!   30 more times by that build. Restored here and stepped 30 times, `a` gives the world `b`
//!   restores to, byte for byte. The writer printed semantics `88cf6f85…` and
//!   `scene().law_id(..)` `bce0de45…`; `a` is 3289 bytes (sha256 `c4192689…`), `b` 3128 bytes
//!   (sha256 `93226904…`).
//!
//! The scene has a contact (two spheres meet) and a distance constraint, so the contact and
//! constraint solves run, not only free flight.
//!
//! oracle: the header identifiers are compared with what the writing build printed, and `b` is
//! that build's own steps, not anything this build computes.

use alice_physics_core::solver::DistanceConstraint;
use alice_physics_core::{
    Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, PHYSICS_SEMANTICS_ID,
    PHYSICS_SEMANTICS_PINS, PREVIOUS_PHYSICS_SEMANTICS_IDS,
};
use sha2::{Digest, Sha256};

const A: &[u8] = include_bytes!("fixtures/world_snapshot_xv_88cf6f85_a.bin");
const B: &[u8] = include_bytes!("fixtures/world_snapshot_xv_88cf6f85_b.bin");
const SEMANTICS_AT: usize = 16;
const LAW_AT: usize = 48;
/// What the build at `b672ff2e` printed
const WRITER_SEMANTICS: &str = "88cf6f85f1943db5d348d22275e4962602082837a0213c4ca3219d105e1fe7a6";
const WRITER_LAW_ID: &str = "bce0de453c17e5f66d330c11c503bf7e6a552822fd2e7c4488fb0c23bc56753c";
const A_SHA256: &str = "c41926894f7b04d565243ee5a0857fe509ef23aac741fe04272252de77a3558d";
const B_SHA256: &str = "9322690468d96cb9ce7b849f8c901fe3efd6cc37e4923c4e92b6fd9eeb54d4eb";

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -7, 0),
        ..PhysicsConfig::default()
    });
    // two spheres that meet, so the contact solve runs
    w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 4, 0), Fix128::ONE),
        Fix128::ONE,
    );
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(1, 1, 0), Fix128::from_int(2));
    b.velocity = Vec3Fix::from_int(0, 3, 0);
    w.add_body_with_radius(b, Fix128::ONE);
    // a pendulum pair held by a distance constraint
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(-4, 6, 1),
        Fix128::ONE,
    ));
    let mut c = RigidBody::new_dynamic(Vec3Fix::from_int(-2, 6, 1), Fix128::ONE);
    c.velocity = Vec3Fix::from_int(0, 0, 2);
    w.add_body(c);
    w.add_distance_constraint(DistanceConstraint {
        body_a: 2,
        body_b: 3,
        local_anchor_a: Vec3Fix::ZERO,
        local_anchor_b: Vec3Fix::ZERO,
        target_distance: Fix128::from_int(2),
        compliance: Fix128::ZERO,
        cached_lambda: Fix128::ZERO,
    });
    w
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The fold of `PHYSICS_SEMANTICS_PINS` as `alice_physics_core::semantics` documents it: sorted by
/// name, then for each entry the name length (u32, big-endian), the name and the 32 digest bytes
fn fold(pins: &[(&str, [u8; 32])]) -> [u8; 32] {
    let mut v: Vec<_> = pins.to_vec();
    v.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut h = Sha256::new();
    for (n, d) in v {
        h.update(u32::try_from(n.len()).expect("short name").to_be_bytes());
        h.update(n.as_bytes());
        h.update(d);
    }
    h.finalize().into()
}

#[test]
fn the_pins_without_the_secondary_chain_fold_to_the_earlier_identifier() {
    // the fold reproduces this build's identifier, so it is the documented procedure
    assert_eq!(fold(PHYSICS_SEMANTICS_PINS), PHYSICS_SEMANTICS_ID);
    let without: Vec<_> = PHYSICS_SEMANTICS_PINS
        .iter()
        .copied()
        .filter(|(n, _)| *n != "secondary chain")
        .collect();
    assert_eq!(without.len() + 1, PHYSICS_SEMANTICS_PINS.len());
    assert_eq!(hex(&fold(&without)), WRITER_SEMANTICS);
}

#[test]
fn the_fixtures_are_what_the_writer_wrote() {
    assert_eq!(hex(&Sha256::digest(A)), A_SHA256);
    assert_eq!(hex(&Sha256::digest(B)), B_SHA256);
    for blob in [A, B] {
        assert_eq!(
            hex(&blob[SEMANTICS_AT..SEMANTICS_AT + 32]),
            WRITER_SEMANTICS
        );
    }
    assert_eq!(hex(&A[LAW_AT..LAW_AT + 32]), WRITER_LAW_ID);
    let writer: [u8; 32] = A[SEMANTICS_AT..SEMANTICS_AT + 32].try_into().unwrap();
    assert_ne!(writer, PHYSICS_SEMANTICS_ID);
    assert_eq!(scene().law_id(&writer), A[LAW_AT..LAW_AT + 32]);
    assert!(PREVIOUS_PHYSICS_SEMANTICS_IDS.contains(&writer));
}

#[test]
fn a_snapshot_of_the_earlier_build_steps_here_to_that_builds_own_steps() {
    let mut here = scene();
    here.restore_world(A)
        .expect("an earlier identifier this build lists");
    for _ in 0..30 {
        here.step(dt());
    }
    let mut theirs = scene();
    theirs
        .restore_world(B)
        .expect("an earlier identifier this build lists");
    let (h, t) = (here.snapshot_world(), theirs.snapshot_world());
    assert!(h.len() > 1000, "a snapshot of the scene, {} bytes", h.len());
    assert_eq!(hex(&Sha256::digest(&h)), hex(&Sha256::digest(&t)));
    assert_eq!(h, t);
    // and the 30 steps moved the world (the comparison is not of two unmoved worlds)
    let mut a_only = scene();
    a_only.restore_world(A).unwrap();
    assert_ne!(a_only.snapshot_world(), t);
}
