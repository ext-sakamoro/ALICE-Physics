//! Cross-platform determinism goldens for scenes with contacts.
//!
//! The scenes of `tests/determinism_golden.rs` produce no body-body contact
//! (their bodies are added without a collision radius), so restitution,
//! friction, static friction and contact filters were covered by no golden.
//! Each scene here runs through `PhysicsWorld::step`, checks a closed-form
//! physical invariant first (so a changed hash comes with a statement of what
//! the physics did), and then pins the SHA-256 of the final body state
//! (position, velocity, rotation, angular velocity of every body). With the
//! `parallel` feature the same scene is also run through
//! `PhysicsWorld::step_parallel` and must give the same hash.
//!
//! Every value that enters the simulation is an exact `Fix128` ratio (no
//! `f64` conversion on the input side); the invariants read the state back
//! with `to_f64`.
//!
//! # Common setup
//!
//! `PhysicsConfig::default()` (8 substeps, `h = dt / 8`, `dt = 1/60 s`,
//! one iteration) with the frame damping set to 1, so the closed forms are
//! plain kinematics, and sleeping disabled, so every body stays in the
//! contact solve for the whole run. The ground is a static sphere of radius
//! `1e6` whose top is the plane `y = 0` (curvature `1e-6 /m`: a ball
//! `x` metres from the top sees the normal tilted by `x · 1e-6 rad`). Balls
//! have radius `1/2` and mass 1. Ground and ball share one material, so the
//! combined friction `μ` and restitution `e` are the material's values.
//!
//! # Rotation
//!
//! The contact solve is translational (contact points at the body centres)
//! and no scene applies a torque, so every body keeps the identity rotation
//! and a zero angular velocity. The hashes therefore do not depend on how
//! the angular velocity is derived from the rotation change.
//!
//! # How to regenerate a golden hash
//!
//! As in `tests/determinism_golden.rs`: only for a deliberate simulation
//! change, after the invariant of the scene still holds, copy the printed
//! "actual" hash into the `GOLDEN_*` constant in the same commit.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::{PhysicsMaterial, SleepConfig};
use sha2::{Digest, Sha256};

/// Ground sphere radius (m).
const GROUND_R: i64 = 1_000_000;

fn fr(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

fn dt() -> Fix128 {
    fr(1, 60)
}

/// Substep length `h` of the default configuration, as `f64`.
fn h() -> f64 {
    1.0 / 60.0 / PhysicsConfig::default().substeps as f64
}

/// Which entry point advances the world.
#[derive(Clone, Copy, Debug)]
enum Path {
    Serial,
    #[cfg(feature = "parallel")]
    Batched,
}

fn paths() -> Vec<Path> {
    #[cfg(feature = "parallel")]
    {
        vec![Path::Serial, Path::Batched]
    }
    #[cfg(not(feature = "parallel"))]
    {
        vec![Path::Serial]
    }
}

fn step(w: &mut PhysicsWorld, path: Path) {
    match path {
        Path::Serial => w.step(dt()),
        #[cfg(feature = "parallel")]
        Path::Batched => w.step_parallel(dt()),
    }
}

/// An empty world with the common setup and gravity `g`, plus the ground
/// (body 0) carrying a material of friction `mu` and restitution `e`.
/// Returns the world and the material id for the balls.
fn world_with_ground(
    g: Vec3Fix,
    mu: Fix128,
    e: Fix128,
) -> (PhysicsWorld, alice_physics::MaterialId) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: g,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    let r = Fix128::from_int(GROUND_R);
    let ground =
        w.add_body_with_radius(RigidBody::new_static(v3(Fix128::ZERO, -r, Fix128::ZERO)), r);
    let id = w.material_table.register(PhysicsMaterial::new(0, mu, e));
    w.set_body_material(ground, id);
    (w, id)
}

/// Adds a ball of radius 1/2 and mass 1 at `pos` with velocity `vel`.
fn add_ball(
    w: &mut PhysicsWorld,
    id: alice_physics::MaterialId,
    pos: Vec3Fix,
    vel: Vec3Fix,
) -> usize {
    let i = w.add_body_with_radius(
        RigidBody::new_dynamic(pos, Fix128::ONE).with_velocity(vel),
        fr(1, 2),
    );
    w.set_body_material(i, id);
    i
}

fn write_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

/// SHA-256 of position, velocity, rotation and angular velocity of every
/// body, as hex.
fn hash_world(w: &PhysicsWorld) -> String {
    let mut bytes = Vec::with_capacity(w.bodies.len() * 13 * 16);
    for b in &w.bodies {
        for f in [
            b.position.x,
            b.position.y,
            b.position.z,
            b.velocity.x,
            b.velocity.y,
            b.velocity.z,
            b.rotation.x,
            b.rotation.y,
            b.rotation.z,
            b.rotation.w,
            b.angular_velocity.x,
            b.angular_velocity.y,
            b.angular_velocity.z,
        ] {
            write_fix(&mut bytes, f);
        }
    }
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn assert_golden(scenario: &str, path: Path, actual: &str, expected: &str) {
    assert_eq!(
        actual,
        expected,
        "\ncontact golden '{scenario}' ({path:?})\n  expected: {expected}\n  actual:   {actual}\n\
         Update GOLDEN_{} only for a deliberate simulation change whose invariant \
         still holds; a mismatch on one platform only is a determinism bug.",
        scenario.to_uppercase()
    );
}

/// Every body keeps the identity rotation and zero angular velocity (see
/// the module docs, section Rotation).
fn assert_no_rotation(scenario: &str, w: &PhysicsWorld) {
    for (i, b) in w.bodies.iter().enumerate() {
        assert!(
            b.rotation.x.is_zero()
                && b.rotation.y.is_zero()
                && b.rotation.z.is_zero()
                && b.rotation.w == Fix128::ONE
                && b.angular_velocity.x.is_zero()
                && b.angular_velocity.y.is_zero()
                && b.angular_velocity.z.is_zero(),
            "{scenario}: body {i} rotated"
        );
    }
}

// ============================================================================
// (a) Bounce: restitution on the pre-solve normal velocity
// ============================================================================

/// **Bounce**: a ball dropped from rest with its bottom `h0 = 2 m` above the
/// ground, `e = 1/2`, `μ = 0`, gravity `(0, −10, 0)`, 180 frames.
///
/// Invariants:
/// * first rebound apex `e² h0 = 0.5 m` (Newton restitution on the impact
///   speed `√(2 g h0)`). Budget `(1 + e + e²) v_i h + g dt² / 8`: the
///   symplectic substep integration shifts each leg's turning point by at
///   most `v h` (fall `v_i h`, rise `e v_i h`, impact substep `e² v_i h`), and
///   the apex is sampled once per frame (at most `g (dt/2)² / 2` below it);
/// * after 3 s (the bounce series ends at `t_f (1 + e) / (1 − e) ≈ 1.9 s`)
///   the ball rests on the ground: `|y − 1/2| < 1e-3 m` and `|v| < 2 g h`
///   (the restitution threshold, below which an approach is treated as
///   resting and does not bounce).
const GOLDEN_CONTACT_BOUNCE: &str =
    "98ffdc63cf108e4c24c42509bc1e5f4351d6a2e06ffccd1116feb74a540d98fa";

#[test]
fn determinism_contact_bounce() {
    let g: f64 = 10.0;
    let (h0, e): (f64, f64) = (2.0, 0.5);
    for path in paths() {
        let (mut w, id) = world_with_ground(
            v3(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            Fix128::ZERO,
            fr(1, 2),
        );
        let ball = add_ball(
            &mut w,
            id,
            v3(Fix128::ZERO, fr(5, 2), Fix128::ZERO),
            Vec3Fix::ZERO,
        );
        let mut contacts = 0usize;
        let mut ys = Vec::with_capacity(180);
        for _ in 0..180 {
            step(&mut w, path);
            contacts += w.contact_constraints.len();
            ys.push(w.bodies[ball].position.y.to_f64() - 0.5);
        }
        assert!(contacts > 0, "{path:?}: no contact was generated");
        // first impact: the first frame the ball is within 1 cm of the ground
        let hit = ys
            .iter()
            .position(|&y| y < 1e-2)
            .expect("ball never landed");
        // first apex: the highest sample before the ball is back down
        let back = hit
            + ys[hit..]
                .iter()
                .skip(5)
                .position(|&y| y < 1e-2)
                .expect("ball never landed twice")
            + 5;
        let apex = ys[hit..back].iter().copied().fold(f64::MIN, f64::max);
        let v_i = (2.0 * g * h0).sqrt();
        let want = e * e * h0;
        let tol = (1.0 + e + e * e) * v_i * h() + g / (60.0 * 60.0 * 8.0);
        println!("{path:?}: apex {apex:.6} (closed form {want}, budget {tol:.3e})");
        assert!(
            (apex - want).abs() < tol,
            "{path:?}: first apex {apex:.6}, closed form e² h0 = {want}"
        );
        let b = &w.bodies[ball];
        let (y, v) = (b.position.y.to_f64() - 0.5, b.velocity.length().to_f64());
        assert!(
            y.abs() < 1e-3 && v < 2.0 * g * h(),
            "{path:?}: not at rest after 3 s: y − r = {y:.3e}, |v| = {v:.3e}"
        );
        assert_no_rotation("contact_bounce", &w);
        assert_golden(
            "contact_bounce",
            path,
            &hash_world(&w),
            GOLDEN_CONTACT_BOUNCE,
        );
    }
}

// ============================================================================
// (b) Slide: Coulomb friction capped by the normal multiplier
// ============================================================================

/// **Slide**: a ball resting on the ground launched at `v0 = 3 m/s` along
/// `x`, `μ = 1/5`, `e = 0`, gravity `(0, −10, 0)`, 120 frames.
///
/// Invariants (kinetic friction decelerates at `μ g`):
/// * at `t = 1 s`: `v = v0 − μ g t = 1 m/s`, `x = v0 t − μ g t² / 2 = 2 m`;
/// * from `t_s = v0 / (μ g) = 1.5 s` on the ball is at rest at
///   `x_s = v0² / (2 μ g) = 2.25 m`; at `t = 2 s`, `|v| < 1e-9`.
///
/// Budget: the position of a substep advances with the velocity before the
/// velocity-level friction removes `μ g h` from it, so the velocity lags the
/// closed form by up to `μ g h` and the position by `μ g h t`; doubled:
/// `2 μ g h` for `v` and `2 μ g h t` for `x`. The ground tilt at `x` adds a
/// tangential `g x / R ≤ 2.3e-5 m/s²`, below `1e-4 m` over the run.
const GOLDEN_CONTACT_SLIDE: &str =
    "0dd44b4f158d51042a09e46c737a285159046a47795f6cb45f1a732fded01b91";

#[test]
fn determinism_contact_slide() {
    let (g, mu, v0) = (10.0, 0.2, 3.0);
    for path in paths() {
        let (mut w, id) = world_with_ground(
            v3(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            fr(1, 5),
            Fix128::ZERO,
        );
        let ball = add_ball(
            &mut w,
            id,
            v3(Fix128::ZERO, fr(1, 2), Fix128::ZERO),
            v3(Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO),
        );
        let mut contacts = 0usize;
        for frame in 1..=120 {
            step(&mut w, path);
            contacts += w.contact_constraints.len();
            if frame == 60 {
                let b = &w.bodies[ball];
                let (x, v) = (b.position.x.to_f64(), b.velocity.x.to_f64());
                let (x_want, v_want) = (v0 - mu * g / 2.0, v0 - mu * g);
                println!("{path:?}: t=1 x {x:.6} ({x_want}), v {v:.6} ({v_want})");
                assert!(
                    (v - v_want).abs() < 2.0 * mu * g * h() + 1e-4
                        && (x - x_want).abs() < 2.0 * mu * g * h() + 1e-4,
                    "{path:?}: t = 1 s: x {x:.6} / {x_want}, v {v:.6} / {v_want}"
                );
            }
        }
        assert!(contacts > 0, "{path:?}: no contact was generated");
        let b = &w.bodies[ball];
        let (x, v) = (b.position.x.to_f64(), b.velocity.length().to_f64());
        let x_want = v0 * v0 / (2.0 * mu * g);
        let t_s = v0 / (mu * g);
        println!("{path:?}: t=2 x {x:.6} ({x_want}), |v| {v:.3e}");
        assert!(
            (x - x_want).abs() < 2.0 * mu * g * h() * t_s + 1e-4 && v < 1e-9,
            "{path:?}: t = 2 s: x {x:.6} / {x_want}, |v| {v:.3e}"
        );
        assert_no_rotation("contact_slide", &w);
        assert_golden("contact_slide", path, &hash_world(&w), GOLDEN_CONTACT_SLIDE);
    }
}

// ============================================================================
// (c) Slope and stack: resting contacts
// ============================================================================

/// **Slope**: a ball at rest on the ground under a gravity tilted by `θ`
/// with `sin θ = 3/5`, `cos θ = 4/5` (`g = (6, −8, 0)`, a slope of 36.9°),
/// `μ = 1` (friction angle 45°), `e = 0`, 120 frames.
///
/// Invariants (a contact inside the friction cone does not slip):
/// * the ball keeps `x` within `1e-12 m` and `|v_x| < 1e-12 m/s`; a
///   velocity-only friction leaves a creep of `g sin θ h²` per substep,
///   `2.5e-2 m` over the run. The bound covers fixed-point rounding only:
///   the ground tilt is zero at `x = 0`;
/// * the ball keeps its height within `1e-6 m` of `1/2` (one substep of
///   normal gravity `g cos θ h² = 3.5e-5 m` is resolved by the next solve).
///
/// A stack of balls on the same slope is not pinned: with one iteration the
/// static friction of the ground contact is solved before the ball-ball
/// contact pushes the bottom ball sideways, and a two-ball stack creeps
/// `1.7e-2 m` in 2 s.
const GOLDEN_CONTACT_SLOPE: &str =
    "bfd18c81157e4af821687782b638f6bda05d8caf762eba605580da29d9424549";

#[test]
fn determinism_contact_slope() {
    for path in paths() {
        let (mut w, id) = world_with_ground(
            v3(Fix128::from_int(6), Fix128::from_int(-8), Fix128::ZERO),
            Fix128::ONE,
            Fix128::ZERO,
        );
        let ball = add_ball(
            &mut w,
            id,
            v3(Fix128::ZERO, fr(1, 2), Fix128::ZERO),
            Vec3Fix::ZERO,
        );
        let mut contacts = 0usize;
        for _ in 0..120 {
            step(&mut w, path);
            contacts += w.contact_constraints.len();
        }
        assert!(contacts > 0, "{path:?}: no contact was generated");
        let b = &w.bodies[ball];
        let (x, y, vx) = (
            b.position.x.to_f64(),
            b.position.y.to_f64(),
            b.velocity.x.to_f64(),
        );
        println!("{path:?}: x {x:.3e}, y {y:.9}, vx {vx:.3e}");
        assert!(
            x.abs() < 1e-12 && vx.abs() < 1e-12,
            "{path:?}: the ball slipped: x {x:.3e}, vx {vx:.3e}"
        );
        assert!((y - 0.5).abs() < 1e-6, "{path:?}: the ball at y {y:.9}");
        assert_no_rotation("contact_slope", &w);
        assert_golden("contact_slope", path, &hash_world(&w), GOLDEN_CONTACT_SLOPE);
    }
}

/// **Stack**: three balls (bottoms at 0, 1, 2 m) at rest on the ground,
/// `μ = 1/2`, `e = 1/2`, gravity `(0, −10, 0)`, 120 frames.
///
/// Invariants (the contacts carry the stack and a resting contact does not
/// bounce):
/// * every ball ends at rest, `|v| < 1e-9 m/s`: each contact approaches at
///   most `g h` per substep, below the restitution threshold `2 g h`, so `e`
///   never applies;
/// * ball `k` stays within `1e-3 m` of `1/2 + k` and keeps `x = 0` exactly
///   (all normals are vertical). The sag bound is `n³ g h² ≈ 1.2e-3 m` for
///   `n = 3`: one iteration leaves each contact an overlap of the order of
///   the weight it carries times `g h²`.
const GOLDEN_CONTACT_STACK: &str =
    "7ae4d4fc88416f79adb5bbcde4185a62cc2cfad22e8dfdc682b3adcfd5a48f61";

#[test]
fn determinism_contact_stack() {
    for path in paths() {
        let (mut w, id) = world_with_ground(
            v3(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            fr(1, 2),
            fr(1, 2),
        );
        let balls: Vec<usize> = (0..3)
            .map(|k| {
                add_ball(
                    &mut w,
                    id,
                    v3(Fix128::ZERO, fr(1 + 2 * k, 2), Fix128::ZERO),
                    Vec3Fix::ZERO,
                )
            })
            .collect();
        let mut contacts = 0usize;
        for _ in 0..120 {
            step(&mut w, path);
            contacts += w.contact_constraints.len();
        }
        assert!(contacts > 0, "{path:?}: no contact was generated");
        for (k, &i) in balls.iter().enumerate() {
            let b = &w.bodies[i];
            let (x, y, v) = (
                b.position.x.to_f64(),
                b.position.y.to_f64(),
                b.velocity.length().to_f64(),
            );
            println!("{path:?}: ball {k}: x {x:.3e}, y {y:.6}, |v| {v:.3e}");
            assert!(
                b.position.x.is_zero() && v < 1e-9 && (y - (0.5 + k as f64)).abs() < 1e-3,
                "{path:?}: ball {k} not resting: x {x:.3e}, y {y:.6}, |v| {v:.3e}"
            );
        }
        assert_no_rotation("contact_stack", &w);
        assert_golden("contact_stack", path, &hash_world(&w), GOLDEN_CONTACT_STACK);
    }
}

// ============================================================================
// (d) Veto: a pre-solve hook discards one ball's contacts
// ============================================================================

/// **Veto**: two balls dropped from rest with their bottoms 1 m above the
/// ground at `x = −3` and `x = 3`, `μ = 1/2`, `e = 1/2`, gravity
/// `(0, −10, 0)`, 120 frames. A pre-solve hook discards every contact of
/// the first ball.
///
/// Invariants:
/// * the vetoed ball falls freely through the ground: after `n` substeps
///   from rest the symplectic integration gives
///   `y = y0 − g h² n (n + 1) / 2` and `v = −g h n`, `x` unchanged
///   (tolerance `1e-9`, fixed-point rounding only);
/// * the other ball lands, bounces and rests on the ground:
///   `|y − 1/2| < 1e-3 m` and `|v| < 2 g h` after 2 s (bounce series ends at
///   `t_f (1 + e) / (1 − e) ≈ 1.3 s`), `x` within `1e-4` of `3` (the ground
///   tilt `3e-6` at `x = 3`).
const GOLDEN_CONTACT_VETO: &str =
    "9dad1dc782bc7b83bc77ceeba69d3612a18fdc0f33060231bbdd4b0d7915cb94";

#[test]
fn determinism_contact_veto() {
    let g = 10.0;
    for path in paths() {
        let (mut w, id) = world_with_ground(
            v3(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            fr(1, 2),
            fr(1, 2),
        );
        let vetoed = add_ball(
            &mut w,
            id,
            v3(Fix128::from_int(-3), fr(3, 2), Fix128::ZERO),
            Vec3Fix::ZERO,
        );
        let kept = add_ball(
            &mut w,
            id,
            v3(Fix128::from_int(3), fr(3, 2), Fix128::ZERO),
            Vec3Fix::ZERO,
        );
        w.add_pre_solve_hook(Box::new(move |a, b, _c: &Contact| {
            a != vetoed && b != vetoed
        }));
        let mut contacts = 0usize;
        for _ in 0..120 {
            step(&mut w, path);
            contacts += w.contact_constraints.len();
        }
        assert!(contacts > 0, "{path:?}: no contact was generated");
        let n = (120 * PhysicsConfig::default().substeps) as f64;
        let y_want = 1.5 - g * h() * h() * n * (n + 1.0) / 2.0;
        let v_want = -g * h() * n;
        let b = &w.bodies[vetoed];
        let (x, y, vy) = (
            b.position.x.to_f64(),
            b.position.y.to_f64(),
            b.velocity.y.to_f64(),
        );
        println!("{path:?}: vetoed y {y:.9} ({y_want:.9}), vy {vy:.9} ({v_want:.9})");
        assert!(
            (y - y_want).abs() < 1e-9 && (vy - v_want).abs() < 1e-9 && (x + 3.0).abs() < 1e-9,
            "{path:?}: vetoed ball ({x:.9}, {y:.9}), vy {vy:.9}: free fall gives \
             (-3, {y_want:.9}), vy {v_want:.9}"
        );
        let b = &w.bodies[kept];
        let (x, y, v) = (
            b.position.x.to_f64(),
            b.position.y.to_f64(),
            b.velocity.length().to_f64(),
        );
        println!("{path:?}: kept x {x:.6}, y {y:.6}, |v| {v:.3e}");
        assert!(
            (y - 0.5).abs() < 1e-3 && v < 2.0 * g * h() && (x - 3.0).abs() < 1e-4,
            "{path:?}: kept ball not at rest: ({x:.6}, {y:.6}), |v| {v:.3e}"
        );
        assert_no_rotation("contact_veto", &w);
        assert_golden("contact_veto", path, &hash_world(&w), GOLDEN_CONTACT_VETO);
    }
}
