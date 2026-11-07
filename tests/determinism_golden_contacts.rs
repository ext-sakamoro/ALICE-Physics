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

use alice_physics_core::collider::Contact;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics_core::{PhysicsMaterial, SleepConfig};
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
) -> (PhysicsWorld, alice_physics_core::MaterialId) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: g,
        damping_rate: Fix128::ZERO,
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
    id: alice_physics_core::MaterialId,
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
///
/// Re-recorded when XPBD began keeping the predicted velocity of bodies the
/// solve did not move (`v = v_pred + Δx_corr / h`).
const GOLDEN_CONTACT_BOUNCE: &str =
    "90afe915e76856fcc40198c731e8df34594f7b455c0a39324ef5878c294fc7e0";

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

/// **Slide**: a ball that cannot turn (`inv_inertia = 0`) resting on the
/// ground launched at `v0 = 3 m/s` along `x`, `μ = 1/5`, `e = 0`, gravity
/// `(0, −10, 0)`, 120 frames.
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
///
/// Re-recorded when XPBD began keeping the predicted velocity of bodies the
/// solve did not move (`v = v_pred + Δx_corr / h`).
const GOLDEN_CONTACT_SLIDE: &str =
    "82c881583c268069f3b7751ce6afa09353343620d4354028592e06363ce8a591";

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
        // a ball that cannot turn: the closed forms above are those of pure
        // sliding (a ball with inertia spins up and rolls, see `roll`)
        w.bodies[ball].inv_inertia = Vec3Fix::ZERO;
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
// (b') Roll: friction at the contact point spins the ball up
// ============================================================================

/// **Roll**: the slide scene with a solid ball (`I = 2/5 m r²`, `r = 1/2`):
/// `v0 = 3 m/s`, `μ = 1/5`, `e = 0`, 120 frames.
///
/// Invariants (kinetic friction at the contact point decelerates the centre
/// at `μ g` and spins the ball up at `5 μ g / (2 r)` until it rolls):
/// * the ball rolls from `t* = 2 v0 / (7 μ g) = 3/7 s` on at `v = 5 v0 / 7`
///   with `v = −ω_z r`;
/// * at `t = 1 s` and `t = 2 s`: `v = 15/7 m/s`, `|v + ω_z r| < 1e-3`,
///   `x = v0 t* − μ g t*² / 2 + v (t − t*)`;
/// * the ball has turned about `z` only.
///
/// Budget: `2 μ g h` on `v` as in `slide`, plus the slip the velocity
/// derivation leaves (`≤ 1e-3`); `2e-2` on `x`.
///
/// Re-recorded when the static friction began acting at the contact point:
/// once the ball rolls, its material point's displacement is under `μ λ`
/// and the position pass now keeps it (`x` at 2 s 4.470471 → 4.470459).
const GOLDEN_CONTACT_ROLL: &str =
    "0531ae7c5d21526d5d3909d213e82c385c0cdb2d5a529bd9db6ecba9be14660a";

#[test]
fn determinism_contact_roll() {
    let (g, mu, v0) = (10.0, 0.2, 3.0);
    let t_star = 2.0 * v0 / (7.0 * mu * g);
    let v_roll = 5.0 * v0 / 7.0;
    let x_at = |t: f64| v0 * t_star - mu * g * t_star * t_star / 2.0 + v_roll * (t - t_star);
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
        // I = 2/5 m r² with m = 1, r = 1/2: inv_inertia = 10
        w.bodies[ball].inv_inertia = v3(
            Fix128::from_int(10),
            Fix128::from_int(10),
            Fix128::from_int(10),
        );
        let mut contacts = 0usize;
        for frame in 1..=120 {
            step(&mut w, path);
            contacts += w.contact_constraints.len();
            if frame == 60 || frame == 120 {
                let t = frame as f64 / 60.0;
                let b = &w.bodies[ball];
                let (x, v, wr) = (
                    b.position.x.to_f64(),
                    b.velocity.x.to_f64(),
                    b.angular_velocity.z.to_f64() / 2.0,
                );
                println!(
                    "{path:?}: t={t} x {x:.6} ({:.6}), v {v:.6} ({v_roll:.6}), ω_z r {wr:.6}",
                    x_at(t)
                );
                assert!(
                    (v - v_roll).abs() < 2.0 * mu * g * h() + 1e-3
                        && (v + wr).abs() < 1e-3
                        && (x - x_at(t)).abs() < 2e-2,
                    "{path:?}: t = {t} s: x {x:.6} / {:.6}, v {v:.6} / {v_roll:.6}, ω_z r {wr:.6}",
                    x_at(t)
                );
            }
        }
        assert!(contacts > 0, "{path:?}: no contact was generated");
        let b = &w.bodies[ball];
        assert!(
            b.angular_velocity.x.is_zero()
                && b.angular_velocity.y.is_zero()
                && b.rotation.x.is_zero()
                && b.rotation.y.is_zero()
                && !b.angular_velocity.z.is_zero(),
            "{path:?}: the ball must turn about z only"
        );
        assert_golden("contact_roll", path, &hash_world(&w), GOLDEN_CONTACT_ROLL);
    }
}

// ============================================================================
// (c) Slope and stack: resting contacts
// ============================================================================

/// **Slope**: a ball that cannot turn (`inv_inertia = 0`) at rest on the
/// ground under a gravity tilted by `θ`
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
        // a ball that cannot turn: a free ball on a slope rolls (see
        // `slope_roll`), only one without inertia is held inside the cone
        w.bodies[ball].inv_inertia = Vec3Fix::ZERO;
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

/// **Slope, rolling**: the slope scene with a solid ball (`I = 2/5 m r²`,
/// `r = 1/2`): `g = (6, −8, 0)`, `μ = 1`, `e = 0`, released from rest, 120
/// frames.
///
/// Invariants (a free ball on a slope rolls without slipping whenever
/// `tan θ ≤ 7μ/2`; here `tan θ = 3/4`): `a = (5/7) g sin θ = 30/7 m/s²`, so
/// at `t = 2 s` `v = 60/7 m/s`, `x = 60/7 m`, and `v = −ω_z r`.
///
/// Budget: `2 %` plus one substep of phase (`a h t`) on `x` and `v`; `1e-3`
/// on the slip `v + ω_z r`.
const GOLDEN_CONTACT_SLOPE_ROLL: &str =
    "06d66f0d5952011228c8ad897d0dba431e0bb4ffc276f7f1f97f492bf93691a5";

#[test]
fn determinism_contact_slope_roll() {
    let a = 5.0 / 7.0 * 6.0;
    let t = 2.0;
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
        w.bodies[ball].inv_inertia = v3(
            Fix128::from_int(10),
            Fix128::from_int(10),
            Fix128::from_int(10),
        );
        let mut contacts = 0usize;
        for _ in 0..120 {
            step(&mut w, path);
            contacts += w.contact_constraints.len();
        }
        assert!(contacts > 0, "{path:?}: no contact was generated");
        let b = &w.bodies[ball];
        let (x, v, wr) = (
            b.position.x.to_f64(),
            b.velocity.x.to_f64(),
            b.angular_velocity.z.to_f64() / 2.0,
        );
        let (x_want, v_want) = (a * t * t / 2.0, a * t);
        println!("{path:?}: t=2 x {x:.6} ({x_want:.6}), v {v:.6} ({v_want:.6}), ω_z r {wr:.6}");
        assert!(
            (x - x_want).abs() < 0.02 * x_want + a * h() * t
                && (v - v_want).abs() < 0.02 * v_want + a * h()
                && (v + wr).abs() < 1e-3,
            "{path:?}: t = 2 s: x {x:.6} / {x_want:.6}, v {v:.6} / {v_want:.6}, v + ω_z r {:.3e}",
            v + wr
        );
        assert_golden(
            "contact_slope_roll",
            path,
            &hash_world(&w),
            GOLDEN_CONTACT_SLOPE_ROLL,
        );
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
///
/// Re-recorded when XPBD began keeping the predicted velocity of bodies the
/// solve did not move (`v = v_pred + Δx_corr / h`).
const GOLDEN_CONTACT_STACK: &str =
    "532dc2852b71eb9da1ca3a62a4f4063f2e1151561744d15852f9fd9dee6b28ad";

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
///
/// Re-recorded when XPBD began keeping the predicted velocity of bodies the
/// solve did not move (`v = v_pred + Δx_corr / h`), and again when the
/// contact friction began acting at the contact point (the kept ball's
/// bounces on the tilted ground leave it a spin of `1e-7`), and once more
/// when the static friction did too (the spin is `5e-5` and the ball sits
/// `2.6e-5` downhill, `|v|` `2.3e-5`: a free ball on a `3e-4` slope rolls).
const GOLDEN_CONTACT_VETO: &str =
    "ec659b05ed9fbd38463d8041828a6ed85bb96d1ff2ac5f4c328d00bf6d7d709a";

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
        // the vetoed ball never touched anything; the kept ball bounced on
        // the ground sphere, whose surface at `x = 3` is tilted by `3e-4`,
        // so the friction at the contact point left it a trace of spin
        let b = &w.bodies[vetoed];
        assert!(
            b.angular_velocity.length_squared().is_zero()
                && b.rotation.x.is_zero()
                && b.rotation.y.is_zero()
                && b.rotation.z.is_zero()
                && b.rotation.w == Fix128::ONE,
            "{path:?}: the vetoed ball turned"
        );
        let b = &w.bodies[kept];
        let spin = b.angular_velocity.length().to_f64();
        assert!(
            spin < 1e-4 && b.rotation.z.to_f64().abs() < 1e-4,
            "{path:?}: kept ball spin {spin:.3e}, rotation z {:.3e}",
            b.rotation.z.to_f64()
        );
        assert_golden("contact_veto", path, &hash_world(&w), GOLDEN_CONTACT_VETO);
    }
}
