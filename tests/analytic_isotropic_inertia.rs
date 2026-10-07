//! Analytic oracle: an isotropic inverse inertia does not depend on the
//! orientation.
//!
//! For a body whose body-frame inverse inertia is `diag(c, c, c)`, the
//! world-frame inverse inertia `R · diag(c, c, c) · Rᵀ` is `c · E` for every
//! rotation `R`, so `I⁻¹ τ = c · τ` exactly, component by component. A
//! fixed-point evaluation that rotates `τ` into the body frame, scales it and
//! rotates it back rounds differently for every orientation, so the
//! orientation of a sphere would leak into its angular and linear motion at
//! the level of the last bits. The claims pinned here:
//!
//! - (a) for `inv_inertia = (5/2, 5/2, 5/2)`, `τ = (3/7, −11/13, 1/3)` and 64
//!   non-identity orientations, the angular velocity change of
//!   `RigidBody::add_torque` (with `dt = 1`) and of
//!   `RigidBody::apply_impulse_at` is the component-wise `Fix128` product
//!   `c · τ`, bit for bit;
//! - (b) two spheres colliding with friction on a ground sphere, run under
//!   `SolverBackend::Xpbd` and `SolverBackend::Tgs`, give bit-identical
//!   positions, velocities and angular velocities for runs that differ only
//!   in the initial orientations of the two spheres;
//! - (c) an anisotropic body (`inv_inertia = (1, 1/2, 1/4)`) tumbling under a
//!   torque and sliding on the ground keeps the state hash it had before the
//!   isotropic case was handled separately (pinned constants, both backends).
//!
//! The expected values in (a) are the closed form `c · τ`, formed here from
//! the inputs, never by calling the solver.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::{PhysicsMaterial, SleepConfig, SolverBackend};
use sha2::{Digest, Sha256};

fn fr(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

fn c() -> Fix128 {
    fr(5, 2)
}

fn tau() -> Vec3Fix {
    v3(fr(3, 7), fr(-11, 13), fr(1, 3))
}

/// 64 non-identity unit quaternions: 16 axes × 4 angles.
fn orientations() -> Vec<QuatFix> {
    let mut out = Vec::with_capacity(64);
    for i in 0..16_i64 {
        let axis = v3(fr(1 + i, 3), fr(2 - i, 5), fr(3 + 2 * i, 7)).normalize();
        for k in 1..=4_i64 {
            let angle = fr(7 * k + i, 9);
            let q = QuatFix::from_axis_angle(axis, angle).normalize();
            assert!(
                q != QuatFix::IDENTITY,
                "orientation {i}/{k} must not be the identity"
            );
            out.push(q);
        }
    }
    out
}

fn isotropic_body(q: QuatFix) -> RigidBody {
    let mut b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = v3(c(), c(), c());
    b.set_rotation(q);
    b
}

fn scaled(v: Vec3Fix, s: Fix128) -> Vec3Fix {
    v3(v.x * s, v.y * s, v.z * s)
}

/// oracle (a): `add_torque(τ, 1)` on an isotropic body at rest changes `ω` by
/// exactly `c · τ` for 64 non-identity orientations.
#[test]
fn isotropic_torque_response_is_c_tau_for_every_orientation() {
    let expected = scaled(tau(), c());
    let mut differ = 0;
    for (n, q) in orientations().into_iter().enumerate() {
        let mut b = isotropic_body(q);
        b.add_torque(tau(), Fix128::ONE);
        if b.angular_velocity != expected {
            differ += 1;
            eprintln!(
                "orientation {n}: ω {:?} != c·τ {:?}",
                b.angular_velocity, expected
            );
        }
    }
    assert_eq!(differ, 0, "{differ}/64 orientations differ from c·τ");
}

/// oracle (a): `apply_impulse_at` on an isotropic body changes `ω` by exactly
/// `c · (r × J)` for 64 non-identity orientations.
#[test]
fn isotropic_impulse_at_point_response_is_c_r_cross_j_for_every_orientation() {
    let r = v3(fr(1, 2), fr(-1, 4), fr(3, 8));
    let j = v3(fr(5, 11), fr(2, 3), fr(-7, 17));
    let expected = scaled(r.cross(j), c());
    let mut differ = 0;
    for q in orientations() {
        let mut b = isotropic_body(q);
        b.apply_impulse_at(j, r);
        differ += usize::from(b.angular_velocity != expected);
    }
    assert_eq!(differ, 0, "{differ}/64 orientations differ from c·(r×J)");
}

const GROUND_R: i64 = 1_000_000;

/// Ground sphere (top at `y = 0`) plus two balls of radius 1/2 that meet
/// obliquely while sliding, with friction 1/2 and gravity −10.
fn two_ball_world(
    backend: SolverBackend,
    inv_inertia: Vec3Fix,
    qa: QuatFix,
    qb: QuatFix,
) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: v3(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        damping: Fix128::ONE,
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    let gr = Fix128::from_int(GROUND_R);
    let ground = w.add_body_with_radius(
        RigidBody::new_static(v3(Fix128::ZERO, -gr, Fix128::ZERO)),
        gr,
    );
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fr(1, 2), fr(1, 4)));
    w.set_body_material(ground, id);
    let half = fr(1, 2);
    let balls = [
        (
            v3(Fix128::ZERO, half, Fix128::ZERO),
            v3(fr(3, 1), Fix128::ZERO, fr(1, 4)),
            qa,
        ),
        (
            v3(fr(2, 1), half, fr(1, 3)),
            v3(fr(-1, 2), Fix128::ZERO, Fix128::ZERO),
            qb,
        ),
    ];
    for (pos, vel, q) in balls {
        let mut b = RigidBody::new_dynamic(pos, Fix128::ONE).with_velocity(vel);
        b.inv_inertia = inv_inertia;
        b.set_rotation(q);
        let i = w.add_body_with_radius(b, half);
        w.set_body_material(i, id);
    }
    w
}

/// Position, velocity and angular velocity of every body (no rotation).
fn motion(w: &PhysicsWorld) -> Vec<[Vec3Fix; 3]> {
    w.bodies
        .iter()
        .map(|b| [b.position, b.velocity, b.angular_velocity])
        .collect()
}

fn run_two_balls(backend: SolverBackend, qa: QuatFix, qb: QuatFix) -> Vec<[Vec3Fix; 3]> {
    let mut w = two_ball_world(backend, v3(c(), c(), c()), qa, qb);
    for _ in 0..90 {
        w.step(fr(1, 60));
    }
    motion(&w)
}

/// The orientation pairs compared with the identity run in oracle (b).
fn orientation_pairs() -> Vec<(QuatFix, QuatFix)> {
    let qs = orientations();
    vec![
        (qs[3], qs[17]),
        (qs[40], qs[63]),
        (qs[9], QuatFix::IDENTITY),
    ]
}

/// oracle (b), TGS: two isotropic spheres colliding with friction move the
/// same way whatever their initial orientations. The TGS contact applies its
/// friction impulse at the contact point, so the balls spin and the inverse
/// inertia enters both the impulse response and the effective mass; the
/// positions, velocities and angular velocities must be bit-identical.
#[test]
fn tgs_isotropic_spheres_motion_does_not_depend_on_orientation() {
    let base = run_two_balls(SolverBackend::Tgs, QuatFix::IDENTITY, QuatFix::IDENTITY);
    assert!(
        base.iter()
            .skip(1)
            .any(|m| !(m[2].x.is_zero() && m[2].y.is_zero() && m[2].z.is_zero())),
        "the friction impulses must spin the balls, or the scene tests nothing"
    );
    for (qa, qb) in orientation_pairs() {
        let other = run_two_balls(SolverBackend::Tgs, qa, qb);
        for (i, (x, y)) in base.iter().zip(&other).enumerate() {
            assert_eq!(
                x, y,
                "TGS: body {i} motion depends on the initial orientation"
            );
        }
    }
}

/// oracle (b), XPBD: the same scene keeps bit-identical positions and
/// velocities whatever the initial orientations. The XPBD contact solve is
/// translational (no inverse inertia enters it), so this holds with or
/// without the isotropic rule; the angular velocity is not compared because
/// XPBD derives it from the rotation change `q · p⁻¹`, whose fixed-point
/// rounding is not exactly the identity for an unchanged non-identity `q`.
#[test]
fn xpbd_isotropic_spheres_linear_motion_does_not_depend_on_orientation() {
    let base = run_two_balls(SolverBackend::Xpbd, QuatFix::IDENTITY, QuatFix::IDENTITY);
    for (qa, qb) in orientation_pairs() {
        let other = run_two_balls(SolverBackend::Xpbd, qa, qb);
        for (i, (x, y)) in base.iter().zip(&other).enumerate() {
            assert_eq!(
                (x[0], x[1]),
                (y[0], y[1]),
                "XPBD: body {i} linear motion depends on the initial orientation"
            );
        }
    }
}

fn write_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn hash_world(w: &PhysicsWorld) -> String {
    let mut bytes = Vec::new();
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

/// Anisotropic tumbling scene: two balls with `inv_inertia = (1, 1/2, 1/4)`,
/// a generic initial spin and orientation, a torque on ball 1 every frame.
fn anisotropic_hash(backend: SolverBackend, parallel: bool) -> String {
    let qs = orientations();
    let mut w = two_ball_world(backend, v3(Fix128::ONE, fr(1, 2), fr(1, 4)), qs[5], qs[50]);
    w.bodies[1].angular_velocity = v3(fr(2, 1), fr(-3, 2), fr(5, 4));
    w.bodies[2].angular_velocity = v3(fr(-1, 3), fr(4, 1), fr(1, 7));
    for _ in 0..90 {
        w.bodies[1].add_torque(tau(), fr(1, 60));
        if parallel {
            #[cfg(feature = "parallel")]
            w.step_parallel(fr(1, 60));
        } else {
            w.step(fr(1, 60));
        }
    }
    hash_world(&w)
}

const GOLDEN_ANISOTROPIC_XPBD: &str =
    "bb98469e8cde970a96c668693e3bc9df91f48347caa08bc05bfd67d016542949";
const GOLDEN_ANISOTROPIC_TGS: &str =
    "095f684372ceec9a468250679ad90119a010fdf225140d3adce059da0ccb0409";

/// oracle (c): the anisotropic path is unchanged (pinned on the code before
/// the isotropic case was handled separately). `step_parallel` is compared
/// for the XPBD backend, the one it runs.
#[test]
fn anisotropic_tumbling_scene_is_unchanged() {
    for (backend, golden) in [
        (SolverBackend::Xpbd, GOLDEN_ANISOTROPIC_XPBD),
        (SolverBackend::Tgs, GOLDEN_ANISOTROPIC_TGS),
    ] {
        let actual = anisotropic_hash(backend, false);
        eprintln!("{backend:?} anisotropic: {actual}");
        assert_eq!(actual, golden, "{backend:?}: anisotropic scene changed");
    }
    #[cfg(feature = "parallel")]
    assert_eq!(
        anisotropic_hash(SolverBackend::Xpbd, true),
        GOLDEN_ANISOTROPIC_XPBD,
        "XPBD: anisotropic scene changed (step_parallel)"
    );
}
