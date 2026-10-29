//! `PhysicsWorld::state_digest`: what it covers, its encoding, and that the
//! whole-world snapshot it hashes round-trips byte for byte.
//!
//! oracle: the digest's definition in its doc (SHA-256 over the BE u64 length
//! of `PhysicsWorld::STATE_DIGEST_DOMAIN`, the domain, then `snapshot_world`), recomputed here
//! with `sha2` directly; the JavaScript test recomputes it with WebCrypto and the
//! Python test with hashlib, each from the snapshot bytes.

use alice_physics_core::joint::{BallJoint, Joint};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::{
    Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, PHYSICS_SEMANTICS_ID,
};
use sha2::{Digest, Sha256};

fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

/// Two bodies on a ball joint above a plane: every class of world content the
/// digest must cover (bodies, a joint, a static collider, config).
fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let a = w.add_body(RigidBody::new_dynamic(v(0, 5, 0), Fix128::ONE));
    let b = w.add_body(RigidBody::new_dynamic(v(1, 5, 0), Fix128::from_int(2)));
    w.add_joint(Joint::Ball(BallJoint::new(a, b, v(0, 0, 0), v(-1, 0, 0))));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v(0, 1, 0),
        Fix128::ZERO,
    )));
    w
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

#[test]
fn the_digest_is_sha256_of_the_domain_and_the_snapshot() {
    let w = scene();
    let mut h = Sha256::new();
    h.update((PhysicsWorld::STATE_DIGEST_DOMAIN.len() as u64).to_be_bytes());
    h.update(PhysicsWorld::STATE_DIGEST_DOMAIN);
    h.update(w.snapshot_world());
    let want: [u8; 32] = h.finalize().into();
    assert_eq!(w.state_digest(), want);
    assert_eq!(
        PhysicsWorld::STATE_DIGEST_DOMAIN,
        b"alice-physics/state-digest/v1"
    );
}

#[test]
fn a_snapshot_restores_to_the_same_bytes_and_the_same_digest() {
    let mut w = scene();
    w.step_n(37, dt());
    let blob = w.snapshot_world();
    let back = PhysicsWorld::from_world_snapshot(&blob).expect("restore");
    assert_eq!(back.snapshot_world(), blob);
    assert_eq!(back.state_digest(), w.state_digest());
}

#[test]
fn the_same_inputs_give_the_same_digest_and_one_impulse_changes_it() {
    let (mut a, mut b) = (scene(), scene());
    a.step_n(60, dt());
    b.step_n(60, dt());
    assert_eq!(a.state_digest(), b.state_digest());
    b.bodies[0].apply_impulse(v(0, 1, 0));
    assert_ne!(a.state_digest(), b.state_digest());
    a.step_n(1, dt());
    b.step_n(1, dt());
    assert_ne!(a.state_digest(), b.state_digest());
}

/// The order bodies are added in is part of the state (their indices are the
/// ids the caller holds), so the same bodies in another order differ.
#[test]
fn insertion_order_is_part_of_the_state() {
    let mut x = PhysicsWorld::new(PhysicsConfig::default());
    x.add_body(RigidBody::new_dynamic(v(0, 5, 0), Fix128::ONE));
    x.add_body(RigidBody::new_dynamic(v(3, 5, 0), Fix128::ONE));
    let mut y = PhysicsWorld::new(PhysicsConfig::default());
    y.add_body(RigidBody::new_dynamic(v(3, 5, 0), Fix128::ONE));
    y.add_body(RigidBody::new_dynamic(v(0, 5, 0), Fix128::ONE));
    let mut z = PhysicsWorld::new(PhysicsConfig::default());
    z.add_body(RigidBody::new_dynamic(v(0, 5, 0), Fix128::ONE));
    z.add_body(RigidBody::new_dynamic(v(3, 5, 0), Fix128::ONE));
    assert_ne!(x.state_digest(), y.state_digest());
    assert_eq!(x.state_digest(), z.state_digest());
}

/// A change made to a world before its digest is compared.
type Change = Box<dyn Fn(&mut PhysicsWorld)>;

/// One change per class of what a replay depends on: each must move the digest.
#[test]
fn every_parameter_a_replay_depends_on_changes_the_digest() {
    let base = scene().state_digest();
    let changes: Vec<(&str, Change)> = vec![
        (
            "body position",
            Box::new(|w| w.bodies[0].position = v(0, 6, 0)),
        ),
        (
            "body velocity",
            Box::new(|w| w.bodies[1].velocity = v(0, 0, 1)),
        ),
        (
            "body mass",
            Box::new(|w| w.bodies[1].inv_mass = Fix128::from_ratio(1, 3)),
        ),
        (
            "body friction",
            Box::new(|w| w.bodies[0].friction = Fix128::from_ratio(1, 7)),
        ),
        (
            "body restitution",
            Box::new(|w| w.bodies[0].restitution = Fix128::from_ratio(1, 7)),
        ),
        (
            "joint added",
            Box::new(|w| {
                w.add_joint(Joint::Ball(BallJoint::new(0, 1, v(0, 1, 0), v(0, 1, 0))));
            }),
        ),
        (
            "static collider added",
            Box::new(|w| {
                w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
                    v(1, 0, 0),
                    Fix128::ONE,
                )));
            }),
        ),
        ("gravity", Box::new(|w| w.config.gravity = v(0, -5, 0))),
        ("substeps", Box::new(|w| w.config.substeps += 1)),
    ];
    let mut compared = 0;
    for (what, change) in &changes {
        let mut w = scene();
        change(&mut w);
        assert_ne!(w.state_digest(), base, "{what} did not change the digest");
        compared += 1;
    }
    assert_eq!(compared, changes.len());
    assert!(compared >= 9);
}

#[test]
fn the_law_id_binding_value_is_the_world_law_under_this_build() {
    let w = scene();
    let mut other = scene();
    other.config.gravity = v(0, -1, 0);
    assert_ne!(
        w.law_id(&PHYSICS_SEMANTICS_ID),
        other.law_id(&PHYSICS_SEMANTICS_ID)
    );
    // the snapshot header carries it (bytes 48..80 of a version 4 blob)
    assert_eq!(
        &w.snapshot_world()[48..80],
        &w.law_id(&PHYSICS_SEMANTICS_ID)
    );
    assert_eq!(&w.snapshot_world()[16..48], &PHYSICS_SEMANTICS_ID);
}

#[test]
fn a_refused_restore_names_its_kind_and_leaves_the_world_unchanged() {
    let mut w = scene();
    let before = w.state_digest();
    let mut blob = scene().snapshot_world();
    let n = blob.len();
    blob[n - 20] ^= 1; // payload byte: the checksum no longer matches
    let err = w.restore_world(&blob).unwrap_err();
    assert_eq!(err.kind(), "ChecksumMismatch");
    assert_eq!(err.code(), 7);
    assert_eq!(w.state_digest(), before);

    let mut other = scene().snapshot_world();
    other[16] ^= 1; // another semantics identifier, checksum recomputed
    let m = other.len() - 8;
    // the snapshot's checksum multiplies by 0x1_0000_01b3 (src/solver.rs `fnv1a_fold`), which
    // is not the FNV-1a 64 prime 0x100_0000_01b3 its format table names; the blob here
    // must carry the checksum the reader computes, so it uses the reader's constant
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in &other[..m] {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0001_0000_01b3);
    }
    other[m..].copy_from_slice(&hash.to_le_bytes());
    let err = w.restore_world(&other).unwrap_err();
    assert_eq!((err.kind(), err.code()), ("SemanticsMismatch", 15));
    assert_eq!(w.state_digest(), before);
}

/// The scene `jointedScene()` of tests/js/bindings.test.mjs builds through the WebAssembly binding,
/// value for value: `withConfig(0, -10, 0, 8, 4)` (its damping 0.99), a static plane y = 0, two
/// dynamic bodies, a ball joint, every number through `Fix128::from_f64` as the binding converts it,
/// and `step(1 / 60)` as `Fix128::from_f64(1.0 / 60.0)`.
fn binding_scene() -> PhysicsWorld {
    let f = Fix128::from_f64;
    let v = |x: f64, y: f64, z: f64| Vec3Fix::new(f(x), f(y), f(z));
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps: 8,
        iterations: 4,
        gravity: v(0.0, -10.0, 0.0),
        damping: Fix128::from_ratio(99, 100),
        ..PhysicsConfig::default()
    });
    let n = v(0.0, 1.0, 0.0).try_normalize().expect("unit normal");
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(n, f(0.0))));
    let a = w.add_body(RigidBody::new_dynamic(v(0.0, 5.0, 0.0), f(1.0)));
    let b = w.add_body(RigidBody::new_dynamic(v(1.0, 5.0, 0.0), f(2.0)));
    w.add_joint(Joint::Ball(BallJoint::new(
        a,
        b,
        v(0.0, 0.0, 0.0),
        v(-1.0, 0.0, 0.0),
    )));
    for _ in 0..120 {
        w.step(f(1.0 / 60.0));
    }
    w
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// golden (regenerated on purpose only): the digest and the law identifier of `binding_scene` after
/// 120 steps. Every platform must compute these exact bytes — the 5 native OS of the CI matrix here,
/// and WebAssembly in tests/js/bindings.test.mjs, which asserts the same two constants for the same
/// scene built through the binding. They move when PHYSICS_SEMANTICS_ID moves (the step or the
/// arithmetic changes) or when the snapshot format changes; that is a deliberate golden change,
/// named in the CHANGELOG with both values.
const GOLDEN_STATE_DIGEST: &str =
    "03f37fd6f8944cc59b22bfc34a0b9cceaf0118eb2569e7769348ef06c01ce399";
const GOLDEN_LAW_ID: &str = "fa916a29691fb926a39073ef014f04bd42e0a6a001efe80c1d54efde621059fb";

#[test]
fn the_digest_of_the_binding_scene_is_the_golden_on_every_platform() {
    let w = binding_scene();
    assert_eq!(hex(&w.state_digest()), GOLDEN_STATE_DIGEST);
    assert_eq!(hex(&w.law_id(&PHYSICS_SEMANTICS_ID)), GOLDEN_LAW_ID);
}
