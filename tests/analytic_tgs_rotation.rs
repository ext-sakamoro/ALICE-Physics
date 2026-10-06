//! Analytic oracle for the orientation integration of the TGS backend.
//!
//! A torque-free body with isotropic inertia `I = s·E` has `ω × Iω = 0`, so
//! its angular velocity is constant and its orientation after time `t` is the
//! closed form `R(t) = exp(t·[ω]×) R(0)`: a turn by `|ω| t` about the fixed
//! axis `ω / |ω|`. The accumulated turn angle therefore equals `|ω| t` for any
//! sub-step count, with no `h²` lag.
//!
//! The scene: `I = 1.2 E`, `|ω| = 6 rad/s` about a generic axis, 600 frames of
//! `1/60 s` (10 s, closed-form angle 60 rad) under `SolverBackend::Tgs` with
//! 1, 2, 4 and 8 sub-steps, through the production entry `PhysicsWorld::step`.
//! The expected angle is computed here from the formula, never by calling the
//! solver; the XPBD backend is used only as a cross-check for the same scene.
//!
//! A principal-axis spin of an anisotropic body is pinned by a golden hash:
//! its free rotation is integrated by the gyroscopic splitting, which this
//! oracle does not concern, so its path must stay bit-identical.

#![cfg(feature = "std")]

use alice_physics::det_math::atan2_64;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::SolverBackend;

const FRAME_DT: f64 = 1.0 / 60.0;
const FRAMES: usize = 600;
const OMEGA_NORM: f64 = 6.0;
const INERTIA: f64 = 1.2;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn world(backend: SolverBackend, substeps: usize, body: RigidBody) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps,
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    w.add_body(body);
    w
}

/// `ω = 6 · (2, 3, 6) / 7`, a generic axis with `|ω| = 6`.
fn isotropic_body() -> RigidBody {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    let inv = 1.0 / INERTIA;
    b.inv_inertia = v3(inv, inv, inv);
    let s = OMEGA_NORM / 7.0;
    b.angular_velocity = v3(2.0 * s, 3.0 * s, 6.0 * s);
    b
}

/// Turn angle of the rotation taking `prev` to `next` (`next · prev⁻¹`).
fn turn_angle(prev: alice_physics::math::QuatFix, next: alice_physics::math::QuatFix) -> f64 {
    let d = next.mul(prev.conjugate());
    let v = [d.x.to_f64(), d.y.to_f64(), d.z.to_f64()];
    let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    2.0 * atan2_64(s, d.w.to_f64().abs())
}

struct Run {
    angle: f64,
    omega_unchanged: bool,
    omega_norm: f64,
}

fn run_isotropic(backend: SolverBackend, substeps: usize) -> Run {
    let body = isotropic_body();
    let omega0 = body.angular_velocity;
    let mut w = world(backend, substeps, body);
    let mut angle = 0.0;
    let mut omega_unchanged = true;
    for _ in 0..FRAMES {
        let prev = w.bodies[0].rotation;
        w.step(fx(FRAME_DT));
        angle += turn_angle(prev, w.bodies[0].rotation);
        omega_unchanged &= w.bodies[0].angular_velocity == omega0;
    }
    let o = w.bodies[0].angular_velocity;
    let on = [o.x.to_f64(), o.y.to_f64(), o.z.to_f64()];
    Run {
        angle,
        omega_unchanged,
        omega_norm: (on[0] * on[0] + on[1] * on[1] + on[2] * on[2]).sqrt(),
    }
}

/// oracle: isotropic `I = 1.2 E`, `|ω| = 6`, 10 s ⇒ accumulated turn
/// `|ω| t = 60 rad` exactly in the closed form, for 1, 2, 4 and 8 sub-steps;
/// `ω` is constant (bit-identical to its initial value, `|ω| = 6`), and the
/// TGS angle equals the XPBD angle of the same scene.
#[test]
fn tgs_isotropic_rotation_follows_the_closed_form_angle() {
    let expected = OMEGA_NORM * FRAME_DT * FRAMES as f64;
    for substeps in [1, 2, 4, 8] {
        let tgs = run_isotropic(SolverBackend::Tgs, substeps);
        let xpbd = run_isotropic(SolverBackend::Xpbd, substeps);
        let err = (tgs.angle - expected).abs();
        let cross = (tgs.angle - xpbd.angle).abs();
        eprintln!(
            "substeps {substeps}: TGS angle {:.15} (err {err:.3e}), XPBD {:.15}, |TGS-XPBD| {cross:.3e}, |ω| {}",
            tgs.angle, xpbd.angle, tgs.omega_norm
        );
        assert!(
            err < 1e-9,
            "substeps {substeps}: TGS angle {} differs from |ω|t = {expected} by {err:e}",
            tgs.angle
        );
        assert!(
            tgs.omega_unchanged,
            "substeps {substeps}: ω of an isotropic body must stay constant"
        );
        assert!(
            (tgs.omega_norm - OMEGA_NORM).abs() < 1e-15,
            "substeps {substeps}: |ω| = {}",
            tgs.omega_norm
        );
        assert!(
            cross < 1e-12,
            "substeps {substeps}: TGS angle {} vs XPBD {} differ by {cross:e}",
            tgs.angle,
            xpbd.angle
        );
    }
}

fn mix(h: &mut u64, f: Fix128) {
    for b in f.hi.to_le_bytes().iter().chain(f.lo.to_le_bytes().iter()) {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn hash_body(b: &RigidBody) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in [b.position, b.velocity, b.angular_velocity] {
        mix(&mut h, v.x);
        mix(&mut h, v.y);
        mix(&mut h, v.z);
    }
    for f in [b.rotation.x, b.rotation.y, b.rotation.z, b.rotation.w] {
        mix(&mut h, f);
    }
    h
}

/// Golden hash of the anisotropic principal-axis spin after 600 frames under
/// TGS with 8 sub-steps.
const GOLDEN_ANISOTROPIC_TGS: u64 = 0x574b_a698_19db_f022;

/// An anisotropic body (`I = diag(1, 2, 3)`) spinning about its major
/// principal axis: the free rotation is integrated by the gyroscopic
/// splitting, so the change to the isotropic orientation rule must leave its
/// path bit-identical.
#[test]
fn tgs_anisotropic_principal_spin_is_unchanged() {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = v3(1.0, 0.5, 1.0 / 3.0);
    b.angular_velocity = v3(0.0, 0.0, 5.0);
    let mut w = world(SolverBackend::Tgs, 8, b);
    for _ in 0..FRAMES {
        w.step(fx(FRAME_DT));
    }
    let got = hash_body(&w.bodies[0]);
    eprintln!("anisotropic TGS hash {got:#018x}");
    assert_eq!(got, GOLDEN_ANISOTROPIC_TGS, "hash {got:#018x}");
}
