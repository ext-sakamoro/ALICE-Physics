//! A speculative contact (a point kept while it is near the floor, not touching)
//! must not stop a body that does not reach the surface: no ghost contacts.
//!
//! oracle (free motion): a box sliding on a frictionless floor alongside a wall,
//! closer to the wall than the speculative margin but moving parallel to it,
//! keeps its speed exactly (nothing acts along the motion); a ball rolling past
//! the edge of a static box at a distance keeps its speed (within the rolling
//! loss of `0` on a frictionless path, so exactly). Both at 240 and 960 Hz.
//!
//! Why it holds: a speculative point only bounds the approach to `vn ≥ −gap / h`
//! (closing the gap in one substep at most), takes no restitution and no
//! friction (its normal multiplier is zero), and is inactive in the position
//! solve; a body moving parallel to the surface has `vn = 0 ≥ −gap / h`.
//!
//! Teeth: speculative points are kept against planes and height fields only,
//! whose gap is measured exactly, so even a margin of 10 m cannot stop a body
//! that does not reach the surface (measured: this oracle, the rocking block, the
//! compliant rest and the flat landing all stay green with it). An oversized
//! margin becomes able to stop a body once speculative points reach finite
//! features (box edges, mesh triangles), whose gap is an estimate; the mutant
//! gets its teeth with that step.

use alice_physics_core::material::{PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverBackend, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn frictionless(w: &mut PhysicsWorld) -> u16 {
    let m = w
        .material_table
        .register(PhysicsMaterial::new(0, Fix128::ZERO, Fix128::ZERO));
    w.material_table
        .set_pair_override(m, DEFAULT_MATERIAL, Fix128::ZERO, Fix128::ZERO);
    m
}

/// A box sliding along `+x` at 3 m/s on a frictionless floor, its side
/// `gap` from a wall at `z = 0.3 + gap` (a plane facing `−z`), for 0.5 s: its
/// speed along `x`, its velocity across, at `hz`.
fn slide_by_wall(backend: SolverBackend, hz: f64, gap: f64) -> (f64, f64) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, fx(-10.0), Fix128::ZERO),
        solver_backend: backend,
        ..SolverConfig::default()
    });
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        Fix128::ZERO,
    )));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, -Fix128::ONE),
        fx(-(0.3 + gap)),
    )));
    let m = frictionless(&mut w);
    let b = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: Vec3Fix::new(fx(0.45), fx(0.06), fx(0.3)),
            },
            fx(500.0),
            Vec3Fix::new(Fix128::ZERO, fx(0.06), Fix128::ZERO),
        )
        .expect("box");
    w.set_body_material(b, m);
    w.get_body_mut(b).expect("box").velocity = Vec3Fix::new(fx(3.0), Fix128::ZERO, Fix128::ZERO);
    for _ in 0..(0.5 * hz) as usize {
        w.step(fx(1.0 / hz));
    }
    let v = w.get_body(b).expect("box").velocity;
    (v.x.to_f64(), v.z.to_f64())
}

/// A ball (radius 0.25) rolling along `+x` at 2 m/s on a frictionless floor past
/// a static box whose near face is 5 mm from the ball's path, for 1 s.
fn roll_past_box(backend: SolverBackend, hz: f64) -> (f64, f64) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, fx(-10.0), Fix128::ZERO),
        solver_backend: backend,
        ..SolverConfig::default()
    });
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        Fix128::ZERO,
    )));
    let m = frictionless(&mut w);
    let k = w.add_body(RigidBody::new_static(Vec3Fix::new(
        fx(1.0),
        fx(0.25),
        fx(0.255 + 0.25),
    )));
    w.set_body_shape(
        k,
        &Shape::Box {
            half_extents: Vec3Fix::new(fx(0.25), fx(0.25), fx(0.25)),
        },
    );
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::ZERO, fx(0.25), Fix128::ZERO),
            Fix128::ONE,
        ),
        fx(0.25),
    );
    w.set_body_material(b, m);
    w.get_body_mut(b).expect("ball").velocity = Vec3Fix::new(fx(2.0), Fix128::ZERO, Fix128::ZERO);
    for _ in 0..hz as usize {
        w.step(fx(1.0 / hz));
    }
    let v = w.get_body(b).expect("ball").velocity;
    (v.x.to_f64(), v.z.to_f64())
}

#[test]
fn a_body_passing_close_to_a_surface_is_not_stopped_by_it() {
    let mut wrong = Vec::new();
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        for hz in [240.0, 960.0] {
            let (vx, vz) = slide_by_wall(backend, hz, 1e-3);
            if (vx - 3.0).abs() > 1e-9 || vz.abs() > 1e-9 {
                wrong.push(format!(
                    "{backend:?} {hz} Hz box by the wall: v = ({vx}, {vz}), want (3, 0)"
                ));
            }
            let (vx, vz) = roll_past_box(backend, hz);
            if (vx - 2.0).abs() > 1e-9 || vz.abs() > 1e-9 {
                wrong.push(format!(
                    "{backend:?} {hz} Hz ball past the box: v = ({vx}, {vz}), want (2, 0)"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
