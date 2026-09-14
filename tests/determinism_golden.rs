//! Cross-platform determinism golden tests (v1.0 Item E, Phase 1).
//!
//! Each test runs a fixed physics scenario for a fixed number of steps,
//! serialises the final body state to bytes, computes a SHA-256 hash,
//! and compares it against a hard-coded golden hash. The golden hash is
//! recorded on Mac aarch64 and expected to match bit-for-bit on every
//! supported platform (Item E goal: macOS ARM + macOS x86 + Linux ARM +
//! Linux x86 + Windows + WASM).
//!
//! # How to regenerate a golden hash
//!
//! When a test fails because the simulation intentionally changed:
//!
//! 1. Run the failing test with `--nocapture` to see the actual hash.
//! 2. Copy the "actual" hex string into the corresponding `GOLDEN_*`
//!    constant.
//! 3. Commit the fixture update in the same PR as the simulation change
//!    so reviewers can attribute the drift.
//!
//! Intentional drift (algorithm improvement, bugfix) is a semver-minor
//! event at pre-1.0; post-1.0 it requires a v2.0 major bump unless the
//! drift is a documented bugfix.
//!
//! # Cross-platform failure diagnosis
//!
//! If the fixture matches on Mac aarch64 but fails on Linux x86 /
//! Windows / WASM, the divergence is a **platform drift** — a real
//! determinism bug that must be fixed at the source (typically an
//! errant `f32` / `f64` operation slipping into a nominally Fix128
//! code path, or a SIMD gate that produces different bit patterns).

#![cfg(feature = "std")]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use sha2::{Digest, Sha256};

/// Serialise a single body's kinematic state to a stable byte layout.
///
/// The layout is `(position.hi, position.lo, position.hi, position.lo,
/// position.hi, position.lo, velocity.hi, ...)` for a total of
/// 6 × 16 bytes = 96 bytes per body: 3 position components × 2 limbs
/// each, then 3 velocity components × 2 limbs each. Fix128 is stored
/// as `(i64 hi, u64 lo)`, so we write each limb as little-endian bytes.
///
/// The rotation quaternion + angular velocity are omitted for Phase 1
/// scenarios that don't rely on them; more elaborate scenarios can
/// extend this helper with a second variant.
fn serialise_body(out: &mut Vec<u8>, body: &RigidBody) {
    let write_fix = |out: &mut Vec<u8>, f: Fix128| {
        out.extend_from_slice(&f.hi.to_le_bytes());
        out.extend_from_slice(&f.lo.to_le_bytes());
    };
    write_fix(out, body.position.x);
    write_fix(out, body.position.y);
    write_fix(out, body.position.z);
    write_fix(out, body.velocity.x);
    write_fix(out, body.velocity.y);
    write_fix(out, body.velocity.z);
}

/// Hash a physics world's body state at the current step.
fn hash_world(world: &PhysicsWorld) -> [u8; 32] {
    let mut bytes: Vec<u8> = Vec::with_capacity(world.body_count() * 96);
    for body in &world.bodies {
        serialise_body(&mut bytes, body);
    }
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    hasher.finalize().into()
}

fn hash_hex(hash: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in hash {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn assert_golden(scenario: &str, actual: [u8; 32], expected_hex: &str) {
    let actual_hex = hash_hex(&actual);
    assert_eq!(
        actual_hex,
        expected_hex,
        "\n\
         determinism golden test failed for scenario '{scenario}'\n\
           expected: {expected_hex}\n\
           actual:   {actual_hex}\n\
         \n\
         If this is a deliberate simulation change:\n\
         - Update the GOLDEN_{scenario_upper} constant in \
           tests/determinism_golden.rs to '{actual_hex}'\n\
         - Commit the fixture update in the same PR as the simulation change\n\
         \n\
         If this is a platform drift (fixture matches on Mac aarch64 but fails\n\
         on another platform), it's a determinism bug — investigate the source\n\
         of platform-dependent bit patterns.",
        scenario = scenario,
        scenario_upper = scenario.to_uppercase(),
        expected_hex = expected_hex,
        actual_hex = actual_hex,
    );
}

// ============================================================================
// Golden fixtures
// ============================================================================

/// **Scenario 1**: five dynamic spheres falling onto a static floor for
/// 200 steps at dt = 1/60 s under gravity of (0, -10, 0).
///
/// Exercises: gravity integration, damping, contact resolution against a
/// static body. No SIMD-only paths involved (SIMD is opt-in via `simd`
/// feature; this test runs with default features).
const GOLDEN_CASCADE: &str = "0a9fb401dcc5fbf12ecd8ef36bd03caed7fb271b68d4035b764dbe24f6a47854";

#[test]
fn determinism_cascade() {
    let config = PhysicsConfig {
        substeps: 4,
        iterations: 8,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        damping: Fix128::from_ratio(99, 100),
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Static floor at y = 0
    let floor = RigidBody::new_static(Vec3Fix::new(
        Fix128::ZERO,
        Fix128::from_int(-1),
        Fix128::ZERO,
    ));
    world.add_body(floor);

    // 5 dynamic spheres at increasing heights and x-offsets
    for i in 0..5 {
        let ball = RigidBody::new_dynamic(
            Vec3Fix::new(
                Fix128::from_int(i as i64),
                Fix128::from_int(5 + i as i64 * 2),
                Fix128::ZERO,
            ),
            Fix128::ONE,
        )
        .with_restitution(Fix128::from_ratio(6, 10))
        .with_linear_damping(Fix128::from_ratio(98, 100));
        world.add_body(ball);
    }

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..200 {
        world.step(dt);
    }

    let hash = hash_world(&world);
    assert_golden("cascade", hash, GOLDEN_CASCADE);
}

/// **Scenario 2**: two dynamic spheres with only gravity, no interaction,
/// for 60 steps. Simplest possible fixture — divergence here indicates
/// a fundamental Fix128 / gravity integration bug.
const GOLDEN_FREEFALL: &str = "49598cff32da429d198e5b281f9d46d89cb5dc6a430d368942b0e5249c779905";

#[test]
fn determinism_freefall() {
    let config = PhysicsConfig {
        substeps: 1,
        iterations: 1,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        damping: Fix128::ONE, // no damping
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Two balls at (0, 100, 0) and (5, 100, 0), 1 kg each, no restitution.
    // No floor — they just fall forever.
    for i in 0..2 {
        let ball = RigidBody::new_dynamic(
            Vec3Fix::new(
                Fix128::from_int(i as i64 * 5),
                Fix128::from_int(100),
                Fix128::ZERO,
            ),
            Fix128::ONE,
        );
        world.add_body(ball);
    }

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    let hash = hash_world(&world);
    assert_golden("freefall", hash, GOLDEN_FREEFALL);
}

/// **Scenario 3**: single dynamic body with an initial velocity vector,
/// no gravity, 120 steps. Exercises pure kinematic integration without
/// any constraint / contact / gravity coupling.
const GOLDEN_KINEMATIC_DRIFT: &str =
    "c79a3ee897fe95bde1bb5660ceb49552aacec0741ec4b8a2f5d46fd6c62ae099";

#[test]
fn determinism_kinematic_drift() {
    let config = PhysicsConfig {
        substeps: 1,
        iterations: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    let ball = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_velocity(Vec3Fix::new(
        Fix128::from_int(1),
        Fix128::from_int(2),
        Fix128::from_int(3),
    ));
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..120 {
        world.step(dt);
    }

    let hash = hash_world(&world);
    assert_golden("kinematic_drift", hash, GOLDEN_KINEMATIC_DRIFT);
}

// ============================================================================
// Meta test: verify the hashing function itself is deterministic on this
// platform (guards against accidental non-determinism in the test harness).
// ============================================================================

#[test]
fn hashing_helper_is_deterministic() {
    let config = PhysicsConfig::default();
    let mut world_a = PhysicsWorld::new(config);
    let mut world_b = PhysicsWorld::new(config);

    for i in 0..3 {
        let body = RigidBody::new_dynamic(
            Vec3Fix::new(
                Fix128::from_int(i as i64),
                Fix128::from_int(10),
                Fix128::ZERO,
            ),
            Fix128::ONE,
        );
        world_a.add_body(body);
        world_b.add_body(body);
    }

    let hash_a = hash_world(&world_a);
    let hash_b = hash_world(&world_b);
    assert_eq!(
        hash_hex(&hash_a),
        hash_hex(&hash_b),
        "hash_world must return identical bytes for identical inputs on the same platform"
    );
}
