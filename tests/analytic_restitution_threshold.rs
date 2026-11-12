//! The restitution threshold: a contact bounces only when it approaches faster
//! than `2 |g| h` (`h` the substep), on both backends.
//!
//! Derivation (the factor 2). A body resting on a surface approaches it in one
//! substep by at most `|g| h` from gravity, plus, on the position-based (XPBD)
//! backend, the velocity the previous substep's push-out of the gravity sag
//! (`|g| h²` over `h`) left in the derived velocity: `|g| h`. So a resting contact
//! never approaches faster than `2 |g| h`, and restitution (an answer to an impact)
//! is applied only above it; the TGS backend keeps its velocities and rests at
//! `|g| h`, under the same bound.
//!
//! The compared speed is the pre-solve approach `a = v0 + |g| h` (the incoming speed
//! plus the substep's own gravity, which both backends integrate before the contact
//! solve).
//!
//! Observability: a bounce of `e a ≈ |g| h` is undone by the remaining substeps'
//! gravity within one step, so a per-step reading never sees it. The scene therefore
//! steps one substep at a time (`substeps = 1`, `dt = h`); the ball starts touching
//! the plane, so detection at the step start already holds the contact.
//!
//! oracle: at `a = 1.25 × 2 |g| h` the ball leaves upward at `e a` (within 10 %), at
//! `a = 0.75 × 2 |g| h` it leaves at less than a tenth of that (rest).

use alice_physics_core::material::{PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverBackend, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;

const G: f64 = 10.0;
const DT: f64 = 1.0 / 60.0;
const SUBSTEPS: usize = 8;
/// The substep `h`.
const H: f64 = DT / SUBSTEPS as f64;
const E: f64 = 0.5;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// The fastest upward speed of the ball after a downward speed `v0` into the
/// plane (touching it at the start), restitution 0.5.
fn rise(backend: SolverBackend, v0: f64) -> f64 {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, fx(-G), Fix128::ZERO),
        solver_backend: backend,
        substeps: 1,
        ..SolverConfig::default()
    });
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        Fix128::ZERO,
    )));
    let m = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(0.5), fx(E)));
    w.material_table
        .set_pair_override(m, DEFAULT_MATERIAL, fx(0.5), fx(E));
    let start = 0.5 - 1e-4;
    let mut ball = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, fx(start), Fix128::ZERO),
        Fix128::ONE,
    );
    ball.velocity = Vec3Fix::new(Fix128::ZERO, fx(-v0), Fix128::ZERO);
    let b = w.add_body_with_radius(ball, fx(0.5));
    w.set_body_material(b, m);
    let mut top = f64::NEG_INFINITY;
    for _ in 0..(30 * SUBSTEPS) {
        w.step(fx(H));
        top = top.max(w.bodies[b].velocity.y.to_f64());
    }
    top
}

#[test]
fn a_contact_bounces_just_above_two_g_h_and_rests_just_below() {
    let threshold = 2.0 * G * H;
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let (fast, slow) = (1.25 * threshold, 0.75 * threshold);
        let above = rise(backend, fast - G * H);
        let below = rise(backend, slow - G * H);
        assert!(
            (above - E * fast).abs() < 0.1 * E * fast,
            "{backend:?}: approach 1.25 × 2|g|h left at {above:.4e}, want e a = {:.4e}",
            E * fast
        );
        assert!(
            below < 0.1 * E * fast,
            "{backend:?}: approach 0.75 × 2|g|h left at {below:.4e} (want rest)"
        );
    }
}
