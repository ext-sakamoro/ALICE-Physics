//! Oracles for a critically damped tether built from existing 3D pieces: a static anchor,
//! a `DistanceConstraint` of rest length 0 and compliance `1 / (m ω²)`, and the body's
//! `linear_damping` at the discrete critical value (see `examples/critically_damped_tether.rs`).
//!
//! Discrete law (substeps 1, iterations 1, `SolverConfig::damping = 1`, a = ω·dt):
//! `x' = (x + dt·v) / (1 + a²)`, `v' = d·(x' − x) / dt` with `d = 1 + 2a² − 2a·sqrt(1 + a²)`.
//! The step matrix has trace `(1 + d) / (1 + a²)` and determinant `d / (1 + a²)`; this `d`
//! makes the discriminant zero, so both eigenvalues are `λ = 1 − a / sqrt(1 + a²)` and every
//! component follows `x_n = (A + B·n)·λⁿ` (`A = x0`, `B = x1/λ − x0`), the discrete twin of
//! `x(t) = (x0 + (v0 + ω x0)·t)·e^(−ωt)`.
//!
//! Distance to the closed form (v0 = 0): `ln λ = −a − a²/2 + O(a³)` and `B = a·x0 + O(a³)`,
//! so `x_n − x(t_n) ≈ −x0·(1 + ωt)·e^(−ωt)·(ωt)·a/2`; the maximum of `s(1 + s)e^(−s)` is
//! 0.840 at `s = (1 + √5)/2`, giving `|x_n − x(t_n)| ≤ 0.42·a·|x0| + O(a²)`. The measured
//! constant approaches 0.42 from below as `a → 0` (0.398 at a = 0.13, 0.4175 at a = 0.017),
//! so the assertion uses 0.43.

use alice_physics::det_math::{exp64, powf64};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sleeping::SleepConfig;
use alice_physics::solver::{DistanceConstraint, PhysicsWorld, RigidBody, SolverConfig};

fn critical_retention(omega: Fix128, dt: Fix128) -> Fix128 {
    let a = omega * dt;
    let a2 = a * a;
    Fix128::ONE + a2 + a2 - (a + a) * (Fix128::ONE + a2).sqrt()
}

fn tether_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(SolverConfig {
        substeps: 1,
        iterations: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..SolverConfig::default()
    });
    // the default sleep rule (|v| < 0.01 for 60 frames) freezes the tail of the decay;
    // the oracles compare the whole trajectory, so sleeping is switched off here
    world.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    world
}

fn add_tether(
    world: &mut PhysicsWorld,
    target: Vec3Fix,
    start: Vec3Fix,
    velocity: Vec3Fix,
    mass: Fix128,
    omega: Fix128,
    dt: Fix128,
) -> usize {
    let anchor = world.add_body(RigidBody::new_static(target));
    let mut body =
        RigidBody::new_dynamic(start, mass).with_linear_damping(critical_retention(omega, dt));
    body.velocity = velocity;
    let body = world.add_body(body);
    world.add_distance_constraint(
        DistanceConstraint::new(anchor, body, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ZERO)
            .with_compliance(Fix128::ONE / (mass * omega * omega)),
    );
    body
}

fn f64v(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

#[test]
fn tether_from_rest_tracks_the_closed_form_within_the_derived_bound() {
    for (omega, den) in [(4_i64, 60_i64), (8, 60), (8, 240), (20, 120), (3, 30)] {
        let dt = Fix128::from_ratio(1, den);
        let (h, w) = (1.0 / den as f64, omega as f64);
        let mut world = tether_world();
        let x0 = [1.5, -0.75, 0.25];
        let start = Vec3Fix::new(
            Fix128::from_f64(x0[0]),
            Fix128::from_f64(x0[1]),
            Fix128::from_f64(x0[2]),
        );
        let b = add_tether(
            &mut world,
            Vec3Fix::ZERO,
            start,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::from_int(omega),
            dt,
        );
        let start = f64v(world.bodies[b].position);
        let mut worst = 0.0_f64;
        for n in 1..=(6 * den / omega.min(6)) {
            world.step(dt);
            let t = n as f64 * h;
            let p = f64v(world.bodies[b].position);
            for c in 0..3 {
                let exact = (start[c] + w * start[c] * t) * exp64(-w * t);
                worst = worst.max((p[c] - exact).abs());
                // overshoot 0: no component changes sign on the way to the target
                assert!(
                    p[c] * start[c] >= 0.0,
                    "component {c} crossed the target at n = {n}"
                );
            }
        }
        let bound = 0.43 * w * h * 1.5; // |x0|_inf = 1.5
        assert!(
            worst <= bound,
            "omega {omega}, dt 1/{den}: {worst} > {bound}"
        );
        // the bound is tight enough to have teeth: the error is at least half of it
        assert!(
            worst >= 0.5 * bound,
            "omega {omega}, dt 1/{den}: {worst} suspiciously small"
        );
    }
}

#[test]
fn tether_follows_the_discrete_closed_form_including_tangential_velocity() {
    // initial velocity perpendicular to the displacement: a radial-only damper would orbit
    let dt = Fix128::from_ratio(1, 60);
    let (h, w): (f64, f64) = (1.0 / 60.0, 6.0);
    let mut world = tether_world();
    let target = Vec3Fix::from_int(2, 1, -1);
    let b = add_tether(
        &mut world,
        target,
        Vec3Fix::from_int(3, 1, -1),
        Vec3Fix::from_int(0, 4, 1),
        Fix128::from_int(3),
        Fix128::from_int(6),
        dt,
    );
    let tgt = f64v(target);
    let x0: Vec<f64> = f64v(world.bodies[b].position)
        .iter()
        .zip(tgt)
        .map(|(p, t)| p - t)
        .collect();
    let v0 = [0.0, 4.0, 1.0];
    let a = w * h;
    let lambda = 1.0 - a / (1.0 + a * a).sqrt();
    let coef: Vec<(f64, f64)> = (0..3)
        .map(|c| {
            let x1 = (x0[c] + h * v0[c]) / (1.0 + a * a);
            (x0[c], x1 / lambda - x0[c])
        })
        .collect();
    for n in 1..=240 {
        world.step(dt);
        let p = f64v(world.bodies[b].position);
        for c in 0..3 {
            let (aa, bb) = coef[c];
            let discrete = (aa + bb * n as f64) * powf64(lambda, f64::from(n));
            assert!(
                (p[c] - tgt[c] - discrete).abs() < 1e-9,
                "n {n} component {c}: {} vs {discrete}",
                p[c] - tgt[c]
            );
        }
    }
    let rest: f64 = f64v(world.bodies[b].position)
        .iter()
        .zip(tgt)
        .map(|(p, t)| (p - t).abs())
        .sum();
    assert!(rest < 1e-3, "settled at the target, residual {rest}");
}

#[test]
fn two_tethers_with_different_omega_in_one_world_follow_their_own_closed_forms() {
    let dt = Fix128::from_ratio(1, 120);
    let h = 1.0 / 120.0;
    let mut world = tether_world();
    let slow = add_tether(
        &mut world,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::from_int(3),
        dt,
    );
    let fast = add_tether(
        &mut world,
        Vec3Fix::from_int(0, 5, 0),
        Vec3Fix::from_int(0, 6, 0),
        Vec3Fix::ZERO,
        Fix128::from_int(4),
        Fix128::from_int(15),
        dt,
    );
    let mut worst = [0.0_f64; 2];
    for n in 1..=360 {
        world.step(dt);
        let t = n as f64 * h;
        let xs = world.bodies[slow].position.x.to_f64();
        let xf = world.bodies[fast].position.y.to_f64() - 5.0;
        worst[0] = worst[0].max((xs - (1.0 + 3.0 * t) * exp64(-3.0 * t)).abs());
        worst[1] = worst[1].max((xf - (1.0 + 15.0 * t) * exp64(-15.0 * t)).abs());
    }
    assert!(worst[0] <= 0.43 * 3.0 * h, "slow: {}", worst[0]);
    assert!(worst[1] <= 0.43 * 15.0 * h, "fast: {}", worst[1]);
}

#[test]
fn tether_replay_is_bit_identical() {
    let run = || {
        let dt = Fix128::from_ratio(1, 60);
        let mut world = tether_world();
        let b = add_tether(
            &mut world,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(1, 2, 3),
            Vec3Fix::from_int(-1, 0, 2),
            Fix128::ONE,
            Fix128::from_int(7),
            dt,
        );
        (0..200)
            .map(|_| {
                world.step(dt);
                (world.bodies[b].position, world.bodies[b].velocity)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}
