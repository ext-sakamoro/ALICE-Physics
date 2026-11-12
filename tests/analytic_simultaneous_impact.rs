//! A rod falling flat onto two supports bounces straight up: the two contacts
//! act at once.
//!
//! oracle (simultaneous contacts, Moreau's impact law as a linear
//! complementarity problem): a rod (a box `2.4 × 0.1 × 0.2` m) falls flat at
//! 3 m/s onto two static balls at `x = ±0.3` under it, restitution 1, no
//! friction.
//! By the mirror symmetry the two impulses are equal; both contacts' targets
//! `vn ≥ e v̄ₙ = 3` m/s then hold with a pure force through the centre of
//! mass, so the rod leaves at `v = +3` m/s with `ω = 0`. The velocity stage
//! reaches it by sequential impulses with accumulated impulses (Catto 2005).
//! The supports are close to the centre (`0.3² < I/m ≈ 0.48`), so an impulse at
//! one also lifts the other point (`1/m − d²/I > 0`): the first contact solved
//! alone lifts both, the second then takes less, and on the next pass the
//! first contact's accumulated impulse must drop (its point now separates
//! faster than its target). Supports beyond `√(I/m)` would push the other
//! point down and every pass would only add. The distance from
//! `(v, ω) = (3, 0)` falls with 1, 4 and 16 iterations and is under `1e-2`
//! at 16.
//!
//! What it catches: an accumulated impulse that may only grow (the first
//! contact keeps its first-pass impulse and the rod leaves spinning, at any
//! iteration count).

use alice_physics_core::material::{PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// The rod's vertical velocity and `|ω|` one substep (1/240 s) after its
/// ends meet the two balls.
fn rod_on_two_balls(iterations: usize) -> (f64, f64) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        iterations,
        ..SolverConfig::default()
    });
    let m = w
        .material_table
        .register(PhysicsMaterial::new(0, Fix128::ZERO, Fix128::ONE));
    w.material_table
        .set_pair_override(m, m, Fix128::ZERO, Fix128::ONE);
    w.material_table
        .set_pair_override(m, DEFAULT_MATERIAL, Fix128::ZERO, Fix128::ONE);
    for x in [-0.3, 0.3] {
        let k = w.add_body_with_radius(RigidBody::new_static(v3(x, 0.0, 0.0)), fx(0.25));
        w.set_body_material(k, m);
    }
    // the rod's bottom face just touching the balls' tops (y = 0.25)
    let rod = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(1.2, 0.05, 0.1),
            },
            fx(500.0),
            v3(0.0, 0.25 + 0.05, 0.0),
        )
        .expect("rod");
    w.set_body_material(rod, m);
    w.get_body_mut(rod).expect("rod").velocity = v3(0.0, -3.0, 0.0);
    w.step(Fix128::from_ratio(1, 240));
    let b = w.get_body(rod).expect("rod");
    (b.velocity.y.to_f64(), b.angular_velocity.length().to_f64())
}

#[test]
fn a_rod_falling_flat_on_two_supports_bounces_straight_up() {
    let mut last = f64::INFINITY;
    let mut wrong = Vec::new();
    for iterations in [1usize, 4, 16] {
        let (vy, spin) = rod_on_two_balls(iterations);
        let off = (vy - 3.0).abs().max(spin);
        eprintln!("{iterations} it: v {vy:.6} |ω| {spin:.3e}, off {off:.3e}");
        if off >= last {
            wrong.push(format!(
                "{iterations} it: off {off:.3e} does not fall from {last:.3e}"
            ));
        }
        last = off;
    }
    if last > 1e-2 {
        wrong.push(format!("16 it: off {last:.3e} from (3, 0)"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
