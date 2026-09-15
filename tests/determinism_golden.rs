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

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::cloth::Cloth;
use alice_physics::eulerian_grid::MacGrid;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::solver::{DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::trimesh::TriMesh;
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
const GOLDEN_CASCADE: &str = "f442b544c0eab74a004567061ab7b573d6892158b2c31480b6d133e2e5970be2";

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
// Extended fixtures (Phase 2)
// ============================================================================

fn write_fix128(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn write_vec3(out: &mut Vec<u8>, v: Vec3Fix) {
    write_fix128(out, v.x);
    write_fix128(out, v.y);
    write_fix128(out, v.z);
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// **Scenario 4**: distance-constraint pendulum. Two dynamic bodies
/// linked by a fixed-length `DistanceConstraint`; one heavy pivot,
/// one lighter swinging mass. Exercises constraint iteration under
/// gravity for 240 steps.
const GOLDEN_JOINT_PENDULUM: &str =
    "96bf5b9e1c565b5331c94eeb1e73dc209e7c7ccec1582b617779541eeb4e3c1e";

#[test]
fn determinism_joint_pendulum() {
    let config = PhysicsConfig {
        substeps: 2,
        iterations: 8,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        damping: Fix128::from_ratio(999, 1000),
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Pivot: heavy body near origin, high linear damping (near-fixed).
    let pivot = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(10), Fix128::ZERO),
        Fix128::from_int(1_000),
    )
    .with_linear_damping(Fix128::from_ratio(1, 100));
    let pivot_idx = world.add_body(pivot);

    // Bob: lighter body offset in x, will swing under gravity.
    let bob = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(3), Fix128::from_int(10), Fix128::ZERO),
        Fix128::from_int(1),
    );
    let bob_idx = world.add_body(bob);

    world.add_distance_constraint(DistanceConstraint::new(
        pivot_idx,
        bob_idx,
        Vec3Fix::ZERO, // anchor on pivot body-local
        Vec3Fix::ZERO, // anchor on bob body-local
        Fix128::from_int(3),
    ));

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..240 {
        world.step(dt);
    }

    let hash = hash_world(&world);
    assert_golden("joint_pendulum", hash, GOLDEN_JOINT_PENDULUM);
}

/// Hash all cloth particles (position + velocity).
fn hash_cloth(cloth: &Cloth) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(cloth.particle_count() * 48);
    for i in 0..cloth.particle_count() {
        write_vec3(&mut bytes, cloth.positions[i]);
        write_vec3(&mut bytes, cloth.velocities[i]);
    }
    sha256_bytes(&bytes)
}

/// **Scenario 5**: 5x5 cloth grid drape under gravity for 120 steps,
/// top row pinned. Exercises XPBD constraint iteration on a soft body.
///
/// Intentional drift 1.1.1: `Fix128::atan` / `atan2` (CORDIC shift carry bug)
/// and the cloth bending constraint (standard PBD dihedral instead of the
/// sign-blind `cos - cos` push) were fixed together. The previous value
/// `0df965ee…` pinned a state where the bottom row had been pumped *above*
/// the pinned top row (y = 6.92 with the top at 5); the cloth now hangs at
/// y ≈ 3.02.
const GOLDEN_CLOTH_DRAPE: &str = "ba4fd85c6e390518ef3326802fb3a29498e1994245901e8636cdad9c0ae1d556";

#[test]
fn determinism_cloth_drape() {
    let mut cloth = Cloth::new_grid(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(5), Fix128::ZERO),
        Fix128::from_int(2), // width
        Fix128::from_int(2), // height
        5,                   // res_x
        5,                   // res_y
        Fix128::ONE,         // mass per particle
    );
    cloth.pin_top_row(5);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..120 {
        cloth.step(dt);
    }

    let hash = hash_cloth(&cloth);
    assert_golden("cloth_drape", hash, GOLDEN_CLOTH_DRAPE);
}

/// Hash a `MacGrid`'s velocity + pressure fields.
fn hash_mac_grid(grid: &MacGrid) -> [u8; 32] {
    let capacity = 16 * (grid.u.len() + grid.v.len() + grid.w.len() + grid.pressure.len());
    let mut bytes = Vec::with_capacity(capacity);
    for &f in &grid.u {
        write_fix128(&mut bytes, f);
    }
    for &f in &grid.v {
        write_fix128(&mut bytes, f);
    }
    for &f in &grid.w {
        write_fix128(&mut bytes, f);
    }
    for &f in &grid.pressure {
        write_fix128(&mut bytes, f);
    }
    sha256_bytes(&bytes)
}

/// **Scenario 6**: small (6x6x6) CFD grid stepped for 30 frames after
/// a small initial u-velocity injection. Exercises the full CFD step
/// pipeline (advection + diffusion + pressure projection).
const GOLDEN_FLUID_STEP: &str = "20f4ba26edef79d64321fdd19c306e80d9e464707e86ba3a036769ff7b1cd3b7";

#[test]
fn determinism_fluid_step() {
    let mut solver = CfdSolver::new(6, 6, 6, Fix128::from_ratio(1, 10));

    // Inject a small u-velocity at (2,2,2) to have something to advect.
    let idx = solver.grid.u.len() / 2;
    solver.grid.u[idx] = Fix128::from_ratio(1, 10);

    let dt = Fix128::from_ratio(1, 100);
    for _ in 0..30 {
        solver.step(dt);
    }

    let hash = hash_mac_grid(&solver.grid);
    assert_golden("fluid_step", hash, GOLDEN_FLUID_STEP);
}

/// **Scenario 7**: dynamic body glancing past a static SDF sphere with
/// speculative CCD enabled, 90 steps. Exercises the SDF collision path
/// with continuous collision detection.
const GOLDEN_SDF_CCD_GLANCE: &str =
    "822813e3a278e960c30acb54ece430b0c51ddda325aa4ebb72009b26a6f62a4e";

/// Unit sphere SDF centred at origin (f32-native per `SdfField` trait).
///
/// NOTE: The SDF path in alice-physics is f32-based, unlike the rest of
/// the physics kernel which uses Fix128. IEEE 754 f32 basic ops (+, -, *,
/// /, sqrt) are bit-exact across platforms per the Rust spec, so this
/// fixture should remain deterministic — but any platform drift found
/// here would indicate an f32 non-determinism issue worth investigating.
fn unit_sphere_sdf() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt();
            if len > 0.0 {
                (x / len, y / len, z / len)
            } else {
                (1.0, 0.0, 0.0)
            }
        },
    )
}

#[test]
fn determinism_sdf_ccd_glance() {
    let config = PhysicsConfig {
        substeps: 4,
        iterations: 8,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Register static SDF sphere at (5, 0, 0).
    let sdf_collider = SdfCollider::new_static(
        Box::new(unit_sphere_sdf()),
        Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO),
        QuatFix::IDENTITY,
    );
    world.add_sdf_collider(sdf_collider);

    // Moving body approaches with slight y-offset (glancing pass).
    let moving = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(3, 10), Fix128::ZERO),
        Fix128::ONE,
    )
    .with_velocity(Vec3Fix::new(
        Fix128::from_int(2),
        Fix128::ZERO,
        Fix128::ZERO,
    ))
    .with_restitution(Fix128::from_ratio(5, 10));
    world.add_body(moving);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..90 {
        world.step(dt);
    }

    let hash = hash_world(&world);
    assert_golden("sdf_ccd_glance", hash, GOLDEN_SDF_CCD_GLANCE);
}

/// Hash a series of `TriMesh::collide_sphere` results at a lattice of
/// sample positions. Deterministic collision detection sanity check.
const GOLDEN_TRIMESH_PROBE: &str =
    "85fbff506a4cd8492163697b0d9cdb2252786d5171b0ff0c125c24a84dec485a";

#[test]
fn determinism_trimesh_probe() {
    // Simple hollow-tetra-like mesh: 4 triangles forming a pyramid.
    let vertices = vec![
        Vec3Fix::ZERO,
        Vec3Fix::new(Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(1), Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(1)),
    ];
    let indices = vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2];
    let mesh = TriMesh::from_indexed(&vertices, &indices);

    let mut bytes: Vec<u8> = Vec::with_capacity(1024);
    let radius = Fix128::from_ratio(3, 10);

    // Probe a 5x5x5 lattice around the mesh centroid.
    for ix in 0..5 {
        for iy in 0..5 {
            for iz in 0..5 {
                let p = Vec3Fix::new(
                    Fix128::from_ratio(ix as i64 * 3, 10) - Fix128::ONE,
                    Fix128::from_ratio(iy as i64 * 3, 10) - Fix128::ONE,
                    Fix128::from_ratio(iz as i64 * 3, 10) - Fix128::ONE,
                );
                if let Some(contact) = mesh.collide_sphere(p, radius) {
                    bytes.push(1); // "hit" marker
                    write_vec3(&mut bytes, contact.normal);
                    write_fix128(&mut bytes, contact.depth);
                } else {
                    bytes.push(0); // "miss" marker
                }
            }
        }
    }

    let hash = sha256_bytes(&bytes);
    assert_golden("trimesh_probe", hash, GOLDEN_TRIMESH_PROBE);
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
