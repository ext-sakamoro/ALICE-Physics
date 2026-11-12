//! In a stack at rest each contact carries the weight of everything above it,
//! whatever the substep rate and the iteration count.
//!
//! oracle (statics): two spheres of masses 2 and 3 kg (radius 0.5) stacked on a
//! static one under `g = 9.81`: the upper contact carries `m₂ g = 29.43` N and
//! the lower `(m₁ + m₂) g = 49.05` N. The force is the position solve's
//! multiplier over the substep squared (`λ / h²`), read after 40 frames of
//! 1/60 s at 240, 480 and 960 Hz and with 1, 4 and 16 iterations; it is the
//! weight to `1e-6` relative in every case (the iteration count changes how
//! far the stack has settled, not what it carries), and the stack's sink
//! shrinks as the iterations grow.
//!
//! What it catches: (1) a velocity stage that pushes one contact's impulse
//! into a body whose other contact was already solved, in a single pass (the
//! lower body leaves the substep moving into its support and the stack falls
//! into a two-substep cycle, about 1.2–2 × the weight on alternate substeps);
//! (2) a speculative point (a gap at detection) left out of the position solve
//! after the solve has moved the bodies into each other (the same cycle).

use alice_physics_core::material::{PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

const G: f64 = 9.81;

/// The two contact forces (lower, upper) and the lower sphere's sink below
/// rest, after 40 frames.
fn stack(substeps: usize, iterations: usize) -> (f64, f64, f64) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -G, 0.0),
        substeps,
        iterations,
        ..SolverConfig::default()
    });
    w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.0, 0.0)), fx(0.5));
    let lower = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 1.0, 0.0), fx(2.0)), fx(0.5));
    w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 2.0, 0.0), fx(3.0)), fx(0.5));
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..40 {
        w.step(dt);
    }
    let mut f: Vec<(f64, f64)> = w
        .contact_forces(dt)
        .iter()
        .map(|(p, _, force)| (p.y.to_f64(), force.to_f64()))
        .collect();
    f.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(
        f.len(),
        2,
        "{substeps} substeps, {iterations} iterations: two contacts"
    );
    let sink = 1.0 - w.get_body(lower).expect("lower").position.y.to_f64();
    (f[0].1, f[1].1, sink)
}

#[test]
fn each_contact_carries_the_weight_above_at_every_rate_and_iteration_count() {
    let mut wrong = Vec::new();
    for substeps in [4usize, 8, 16] {
        let mut last_sink = f64::INFINITY;
        for iterations in [1usize, 4, 16] {
            let (lower, upper, sink) = stack(substeps, iterations);
            let hz = 60 * substeps;
            eprintln!(
                "{hz} Hz, {iterations} it: lower {lower:.6} upper {upper:.6} sink {sink:.3e}"
            );
            let want = [(lower, 5.0 * G, "lower"), (upper, 3.0 * G, "upper")];
            for (got, w, name) in want {
                if (got - w).abs() > 1e-6 * w {
                    wrong.push(format!(
                        "{hz} Hz, {iterations} it: {name} {got} N, want {w}"
                    ));
                }
            }
            if sink >= last_sink {
                wrong.push(format!(
                    "{hz} Hz: the sink {sink:.3e} at {iterations} it does not shrink from {last_sink:.3e}"
                ));
            }
            last_sink = sink;
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The velocities `(lower, upper)` one substep (1/240 s, one substep per step)
/// after a 3 kg ball moving down at 3 m/s meets the top of a 2 kg ball resting
/// on a static one, every contact with restitution 1 and no friction.
fn elastic_hit(iterations: usize) -> (f64, f64) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -G, 0.0),
        substeps: 1,
        iterations,
        ..SolverConfig::default()
    });
    w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.0, 0.0)), fx(0.5));
    let lower = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 1.0, 0.0), fx(2.0)), fx(0.5));
    let upper = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 2.0, 0.0), fx(3.0)), fx(0.5));
    let m = w
        .material_table
        .register(PhysicsMaterial::new(0, Fix128::ZERO, Fix128::ONE));
    w.material_table
        .set_pair_override(m, m, Fix128::ZERO, Fix128::ONE);
    w.material_table
        .set_pair_override(m, DEFAULT_MATERIAL, Fix128::ZERO, Fix128::ONE);
    w.set_body_material(lower, m);
    w.set_body_material(upper, m);
    w.get_body_mut(upper).expect("upper").velocity = v3(0.0, -3.0, 0.0);
    w.step(Fix128::from_ratio(1, 240));
    (
        w.get_body(lower).expect("lower").velocity.y.to_f64(),
        w.get_body(upper).expect("upper").velocity.y.to_f64(),
    )
}

/// oracle (simultaneous contacts, Moreau's impact law as a linear
/// complementarity problem): at the impact the upper contact's target is
/// `vn ≥ e v̄ₙ = 3` m/s of separation and the support's, whose pre-impact
/// approach is below the resting threshold, `vn ≥ 0`. Both active, the
/// solution is the support holding the lower ball still and the upper one
/// rebounding at 3 m/s (`(0, +3)`; the kinetic energy is kept, and no
/// velocity points into the support). The velocity stage solves it by
/// sequential impulses with accumulated impulses (Catto 2005), which converge
/// with the passes: the distance from `(0, 3)` falls with 1, 4 and 16
/// iterations and is under `1e-2` m/s at 16.
///
/// What it catches: a single pass of one-sided impulses (the lower ball keeps
/// the upper contact's impulse and leaves the substep moving into its support
/// at about 2 m/s, whatever the iteration count).
#[test]
fn an_elastic_hit_on_a_resting_stack_converges_on_the_simultaneous_impact() {
    let mut last = f64::INFINITY;
    let mut wrong = Vec::new();
    for iterations in [1usize, 4, 16] {
        let (lower, upper) = elastic_hit(iterations);
        let off = lower.abs().max((upper - 3.0).abs());
        eprintln!("{iterations} it: lower {lower:.6} upper {upper:.6}, off {off:.3e}");
        if off >= last {
            wrong.push(format!(
                "{iterations} it: off {off:.3e} does not fall from {last:.3e}"
            ));
        }
        last = off;
    }
    if last > 1e-2 {
        wrong.push(format!("16 it: off {last:.3e} m/s from (0, 3)"));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
