//! A sphere dropped onto a static plane does not pass into it, on both
//! backends.
//!
//! oracle (speculative contact, Catto, "Continuous Collision", GDC 2013, and
//! the compliant contact's give): a sphere of radius `r` falling from rest at
//! height `y₀` under gravity `g` reaches the plane at `v = √(2 g (y₀ − r))`.
//! The point within the speculative margin `(|v| + |ω| R) h + |g| h²` of the
//! plane is a contact before the sphere arrives, so the step stops it at the
//! surface; the only depth left is the compliant contact's give under the
//! impact, `α F` with `F ≤ m (1 + e) v / h` (`α` the pair's compliance, `h`
//! the substep). The lowest centre over the run is at least `r − α m (1 + e)
//! v / h`.
//!
//! What it catches: a plain sphere (no shape) tested against a static collider
//! without the margin. The temporal Gauss–Seidel step detects once per tick,
//! so such a sphere was met only once it overlapped and sank 8.3 mm into the
//! plane before it was stopped.

use alice_physics_core::material::DEFAULT_CONTACT_COMPLIANCE;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend};
use alice_physics_core::static_collider::StaticCollider;

#[test]
fn a_dropped_sphere_stops_at_the_plane_within_the_compliant_give() {
    let (g, y0, r) = (10.0f64, 3.0f64, 0.5f64);
    let v = (2.0 * g * (y0 - r)).sqrt();
    let mut compared = 0;
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        for substeps in [1usize, 4] {
            for e in [0.0, 0.5] {
                let mut w = PhysicsWorld::new(PhysicsConfig {
                    gravity: Vec3Fix::from_int(0, -10, 0),
                    substeps,
                    solver_backend: backend,
                    ..PhysicsConfig::default()
                });
                w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
                    Vec3Fix::UNIT_Y,
                    Fix128::ZERO,
                )));
                let mut ball = RigidBody::new_dynamic(Vec3Fix::from_int(0, 3, 0), Fix128::ONE);
                ball.restitution = Fix128::from_f64(e);
                let b = w.add_body(ball);
                w.set_body_collision_radius(b, Fix128::from_ratio(1, 2));
                let mut lowest = f64::MAX;
                for _ in 0..120 {
                    w.step(Fix128::from_ratio(1, 60));
                    lowest = lowest.min(w.bodies[b].position.y.to_f64());
                }
                // two materials in series; e ≤ 1 for the pair whatever the combine rule
                let alpha = 2.0 * DEFAULT_CONTACT_COMPLIANCE.to_f64();
                let h = 1.0 / (60.0 * substeps as f64);
                let give = alpha * 1.0 * 2.0 * v / h;
                assert!(
                    lowest >= r - give,
                    "{backend:?} substeps {substeps} e {e}: lowest {lowest} (r − give {})",
                    r - give
                );
                assert!(
                    lowest < r + 1e-3,
                    "{backend:?}: the sphere never reached the plane"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 8);
}
