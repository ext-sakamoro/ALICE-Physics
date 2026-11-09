//! Change detector for the overflow flag on pairs whose radius sum leaves the
//! `Fix128` range: the flag and the whole state are pinned bit for bit, per
//! broad-phase kind and sleep skip, on the scenes where the flag is known to
//! depend on the path today (AUD-A-S6W1-011 and the static pair audit).
//!
//! # Why
//!
//! The rule that raises the flag is about to become one predicate shared by
//! every path (narrow phase, park insertion, restore). That is a change of
//! meaning of a persisted byte (`serialize_state`), so the state before the
//! change is recorded here first, on every CI platform (the wasm lane runs
//! this file through wasmtime as it runs the determinism goldens), and the
//! change commit re-records it with the two digests side by side.
//!
//! # Expected values
//!
//! Not derived from physics: recorded by running this file. Which flag is right
//! is the business of `tests/analytic_broadphase_overflow_kinds.rs`; this file
//! only guards against an unintended change of bits.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::filter::CollisionFilter;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::sleeping::SleepState;
use alice_physics_core::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};
use sha2::{Digest as _, Sha256};

const KINDS: [Broadphase; 3] = [Broadphase::Bvh, Broadphase::DynamicTree, Broadphase::Hybrid];

fn x_at(x: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO)
}

fn world(kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
    w.set_sleep_skip(skip);
    w
}

fn sleep(w: &mut PhysicsWorld, i: usize) {
    w.islands.sleep_data[i].state = SleepState::Sleeping;
    w.islands.sleep_data[i].idle_time = Fix128::from_int(100);
}

fn step(w: &mut PhysicsWorld, n: usize) {
    for _ in 0..n {
        w.step(Fix128::from_ratio(1, 60));
    }
}

/// FNV-1a and SHA-256 over the scene tag, the flag and the state.
struct Digest(u64, Sha256);

impl Digest {
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.1.update(b);
    }

    fn world(&mut self, tag: &str, w: &PhysicsWorld) {
        self.bytes(tag.as_bytes());
        self.bytes(&[u8::from(w.overflow_detected())]);
        self.bytes(&w.serialize_state());
    }
}

/// The radii of the pair: sums past the range, boxes inside it or not.
fn radii() -> [Fix128; 3] {
    [
        Fix128::from_int(1i64 << 62),
        Fix128::from_int((1i64 << 62) + (1i64 << 61)),
        Fix128::from_raw(i64::MAX, u64::MAX),
    ]
}

/// Every scene, stepped under every kind with the skip on and off; returns
/// `(scenes, flags raised, digest, sha256 hex)`.
fn run() -> (usize, usize, u64, String) {
    let mut dg = Digest(0xcbf2_9ce4_8422_2325, Sha256::new());
    let (mut scenes, mut raised) = (0, 0);
    let mut record = |tag: String, w: &PhysicsWorld| {
        dg.world(&tag, w);
        scenes += 1;
        raised += usize::from(w.overflow_detected());
    };
    for &kind in &KINDS {
        for skip in [false, true] {
            let k = format!("{kind:?} skip={skip}");
            for xb in [Fix128::from_int(1 << 40), Fix128::from_int(-(1 << 62))] {
                for r in radii() {
                    // two sleeping dynamic bodies (the parked pair of AUD-A-S6W1-011)
                    let mut w = world(kind, skip);
                    w.add_body_with_radius(
                        RigidBody::new_dynamic(x_at(Fix128::ZERO), Fix128::ONE),
                        r,
                    );
                    w.add_body_with_radius(RigidBody::new_dynamic(x_at(xb), Fix128::ONE), r);
                    sleep(&mut w, 0);
                    sleep(&mut w, 1);
                    step(&mut w, 3);
                    record(format!("{k} parked xb={xb:?} r={r:?}"), &w);
                    // the same pair awake
                    let mut w = world(kind, skip);
                    w.add_body_with_radius(
                        RigidBody::new_dynamic(x_at(Fix128::ZERO), Fix128::ONE),
                        r,
                    );
                    w.add_body_with_radius(RigidBody::new_dynamic(x_at(xb), Fix128::ONE), r);
                    step(&mut w, 3);
                    record(format!("{k} awake xb={xb:?} r={r:?}"), &w);
                    // two static bodies (the static pair audit)
                    let mut w = world(kind, skip);
                    w.add_body_with_radius(RigidBody::new_static(x_at(Fix128::ZERO)), r);
                    w.add_body_with_radius(RigidBody::new_static(x_at(xb)), r);
                    step(&mut w, 1);
                    record(format!("{k} static xb={xb:?} r={r:?}"), &w);
                    // an awake pair whose layers do not collide
                    let mut w = world(kind, skip);
                    let a = w.add_body_with_radius(
                        RigidBody::new_dynamic(x_at(Fix128::ZERO), Fix128::ONE),
                        r,
                    );
                    let b =
                        w.add_body_with_radius(RigidBody::new_dynamic(x_at(xb), Fix128::ONE), r);
                    w.set_body_filter(a, CollisionFilter::new(1, 1));
                    w.set_body_filter(b, CollisionFilter::new(2, 2));
                    step(&mut w, 1);
                    record(format!("{k} filtered xb={xb:?} r={r:?}"), &w);
                    // one static, one sleeping dynamic
                    let mut w = world(kind, skip);
                    w.add_body_with_radius(RigidBody::new_static(x_at(Fix128::ZERO)), r);
                    w.add_body_with_radius(RigidBody::new_dynamic(x_at(xb), Fix128::ONE), r);
                    sleep(&mut w, 1);
                    step(&mut w, 3);
                    record(format!("{k} static+parked xb={xb:?} r={r:?}"), &w);
                }
            }
            // two sleeping unit ellipsoids with a collision radius of 2^62, centres
            // 2 + gap apart: the collider box is not the sphere's cube
            for gap_q in [-1i64, 0, 1, 3, 5, 12] {
                let gap = Fix128::from_ratio(gap_q, 4);
                let r = Fix128::from_int(1i64 << 62);
                let mut w = world(kind, skip);
                let a = w.add_body_with_radius(
                    RigidBody::new_dynamic(x_at(Fix128::ZERO), Fix128::ONE),
                    r,
                );
                let b = w.add_body_with_radius(
                    RigidBody::new_dynamic(x_at(Fix128::from_int(2) + gap), Fix128::ONE),
                    r,
                );
                for i in [a, b] {
                    assert!(w.set_body_shape(
                        i,
                        &Shape::Ellipsoid {
                            radii: Vec3Fix::from_int(1, 1, 1)
                        }
                    ));
                    w.set_body_collision_radius(i, r);
                    sleep(&mut w, i);
                }
                step(&mut w, 3);
                record(format!("{k} ellipsoids gap={gap_q}/4"), &w);
            }
        }
    }
    let sha = dg.1.finalize();
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    (scenes, raised, dg.0, hex)
}

/// Recorded with the one-predicate rule (`pair_radius_sum_overflows`), and recorded again
/// when the state blob gained the sleep time, the island records and the cloth section (the
/// bytes each scene hashes are its `serialize_state`; the flags raised did not change). Before it,
/// on main `06b1e84b` (2026-11-07), the same scenes gave 189 flags raised and
/// the digest `0xb923ddd589b2f53d`: 33 scenes changed their flag (static pairs
/// and pairs whose layers do not collide no longer raise it under Bvh /
/// DynamicTree, parked pairs raise it with the skip on as without).
const EXPECTED_SCENES: usize = 216;
const EXPECTED_RAISED: usize = 156;
const EXPECTED_DIGEST: u64 = 0x6e5daff63bcadd9d;
/// SHA-256 over the same bytes, the entry `"overflow flag"` of
/// `PHYSICS_SEMANTICS_PINS` folds (see `tests/physics_semantics_id.rs`).
const EXPECTED_SHA256: &str = "a673ff33022aeb79a6302a34009893acb120c01019d18cdcc74dc27b25b1b802";

#[test]
fn the_overflow_flag_keeps_its_previous_bits_on_the_path_dependent_scenes() {
    let (scenes, raised, digest, sha) = run();
    assert!(
        scenes > 0 && raised > 0,
        "nothing compared or no flag raised"
    );
    assert_eq!(
        (scenes, raised, digest),
        (EXPECTED_SCENES, EXPECTED_RAISED, EXPECTED_DIGEST),
        "(scenes, raised, digest) moved; recorded ({scenes}, {raised}, 0x{digest:016x}, {sha})"
    );
}

/// The SHA-256 of the same bytes, for the semantics identifier's entry.
#[test]
fn the_overflow_flag_scenes_have_their_pinned_sha256() {
    let (_, _, digest, sha) = run();
    assert_eq!(digest, EXPECTED_DIGEST);
    assert_eq!(sha, EXPECTED_SHA256, "recorded sha256 {sha}");
}
