//! Oracles for the 2D `Joint2D::Mouse` and `Joint2D::Distance` laws inside
//! `PhysicsWorld2D::step`.
//!
//! Mouse: `stiffness` k (N/m) and `damping` c (N·s/m) define the spring-damper
//! `F = k·(target − x) − c·v`, clamped to `|F| <= max_force`. With `k = m ω²` and
//! `c = 2 m ω` (damping ratio 1) the continuous law is the critically damped
//! `x(t) = (x0 + (v0 + ω x0)·t)·e^(−ωt)` about the target.
//!
//! Discrete law: XPBD with the damping term (Macklin et al. 2016, eq. 26) and the
//! multiplier accumulated over the iterations of a substep. For this linear constraint the
//! first iteration already solves the substep exactly, and the update is backward Euler:
//! `x' = (x·(1 + 2ζa) + h·v) / (1 + 2ζa + a²)`, `a = ω h`, `h = dt / substeps`. At ζ = 1 both
//! eigenvalues are `λ = 1 / (1 + a)` (positive), so every component follows
//! `x_n = (A + B·n)·λⁿ` with `A = x0`, `B = x1/λ − x0`, and never overshoots from rest.
//! Distance to the continuous form (v0 = 0): `ln λ = −a + a²/2 + O(a³)` and
//! `B = x0·a/(1 + a) = x0·(a − a²) + O(a³)`, so with `s = ωt`
//! `x_n − x(t_n) ≈ x0·e^(−s)·a·s·(s − 1)/2`. `|s(s − 1)e^(−s)|` peaks at
//! `s = (3 ± √5)/2` with 0.309 (and 0.161), so `|x_n − x(t_n)| <= 0.155·a·|x0| + O(a²)`;
//! the assertions use 0.16 and require at least half of it (the bound has teeth).
//!
//! Distance: XPBD with compliance α (stiffness 1/α). Under gravity the steady extension is
//! `α·m·g` for every substep length and iteration count (the multiplier balances gravity:
//! `λ = m g h²`, `C = −(α/h²)·λ`); without accumulating the multiplier across iterations
//! each extra iteration removes a further fraction of `C` and the joint stiffens.

use alice_physics::det_math::{exp64, powf64};
use alice_physics::math::Fix128;
use alice_physics::physics2d::{
    Joint2D, PhysicsConfig2D, PhysicsWorld2D, RigidBody2D, Shape2D, Vec2Fix,
};

fn quiet_world(substeps: usize, iterations: usize) -> PhysicsWorld2D {
    PhysicsWorld2D::new(PhysicsConfig2D {
        gravity: Vec2Fix::ZERO,
        substeps,
        iterations,
        damping: Fix128::ONE,
    })
}

fn circle() -> Shape2D {
    Shape2D::Circle {
        radius: Fix128::from_ratio(1, 100),
    }
}

/// Mouse with `k = m ω²`, `c = 2 ζ m ω` and an effectively unbounded force.
fn add_mouse(
    world: &mut PhysicsWorld2D,
    start: Vec2Fix,
    target: Vec2Fix,
    mass: i64,
    omega: i64,
    zeta: i64,
) -> usize {
    let body = world.add_body(RigidBody2D::new_dynamic(
        start,
        Fix128::from_int(mass),
        circle(),
    ));
    world.add_joint(Joint2D::Mouse {
        body,
        target,
        max_force: Fix128::from_int(1_000_000_000),
        stiffness: Fix128::from_int(mass * omega * omega),
        damping: Fix128::from_int(2 * zeta * mass * omega),
    });
    body
}

fn xy(v: Vec2Fix) -> [f64; 2] {
    [v.x.to_f64(), v.y.to_f64()]
}

/// Mouse bodies placed far apart so their circles never touch.
fn spread(i: i64) -> Vec2Fix {
    Vec2Fix::from_int(100 * i, 0)
}

#[test]
fn mouse_from_rest_tracks_the_critically_damped_closed_form() {
    for (omega, den, substeps) in [
        (4_i64, 60_i64, 1_usize),
        (8, 60, 1),
        (8, 240, 1),
        (20, 60, 4),
        (3, 30, 2),
    ] {
        let dt = Fix128::from_ratio(1, den);
        let h = 1.0 / (den as f64 * substeps as f64);
        let w = omega as f64;
        let mut world = quiet_world(substeps, 1);
        let target = Vec2Fix::from_int(2, -1);
        let x0 = [1.5_f64, -0.75];
        let start = Vec2Fix::new(
            Fix128::from_int(2) + Fix128::from_ratio(3, 2),
            Fix128::from_int(-1) + Fix128::from_ratio(-3, 4),
        );
        let b = add_mouse(&mut world, start, target, 2, omega, 1);
        let mut worst = 0.0_f64;
        let frames = 6 * den / omega.min(6);
        for n in 1..=frames {
            world.step(dt);
            let t = n as f64 / den as f64;
            let p = xy(world.bodies[b].position);
            let rel = [p[0] - 2.0, p[1] + 1.0];
            for c in 0..2 {
                let exact = x0[c] * (1.0 + w * t) * exp64(-w * t);
                worst = worst.max((rel[c] - exact).abs());
                assert!(rel[c] * x0[c] >= 0.0, "component {c} overshot at frame {n}");
            }
        }
        let bound = 0.16 * w * h * 1.5;
        assert!(
            worst <= bound,
            "omega {omega}, dt 1/{den}, substeps {substeps}: {worst} > {bound}"
        );
        assert!(
            worst >= 0.5 * bound,
            "omega {omega}: error {worst} suspiciously small (bound {bound})"
        );
    }
}

#[test]
fn mouse_follows_the_discrete_closed_form_with_tangential_velocity() {
    let dt = Fix128::from_ratio(1, 60);
    let (h, w): (f64, f64) = (1.0 / 60.0, 6.0);
    let mut world = quiet_world(1, 1);
    let target = Vec2Fix::from_int(3, 4);
    let b = add_mouse(&mut world, Vec2Fix::from_int(4, 4), target, 3, 6, 1);
    world.bodies[b].velocity = Vec2Fix::from_int(0, 5);
    let x0 = [1.0, 0.0];
    let v0 = [0.0, 5.0];
    let a = w * h;
    let lambda = 1.0 / (1.0 + a);
    let coef: Vec<(f64, f64)> = (0..2)
        .map(|c| {
            let x1 = (x0[c] * (1.0 + 2.0 * a) + h * v0[c]) / ((1.0 + a) * (1.0 + a));
            (x0[c], x1 / lambda - x0[c])
        })
        .collect();
    for n in 1..=240 {
        world.step(dt);
        let p = xy(world.bodies[b].position);
        let rel = [p[0] - 3.0, p[1] - 4.0];
        for c in 0..2 {
            let (aa, bb) = coef[c];
            let discrete = (aa + bb * f64::from(n)) * powf64(lambda, f64::from(n));
            assert!(
                (rel[c] - discrete).abs() < 1e-9,
                "frame {n} component {c}: {} vs {discrete}",
                rel[c]
            );
        }
    }
}

#[test]
fn two_mouse_joints_with_different_omega_in_one_world_follow_their_own_closed_forms() {
    let dt = Fix128::from_ratio(1, 120);
    let h = 1.0 / 120.0;
    let mut world = quiet_world(1, 1);
    let slow = add_mouse(
        &mut world,
        spread(0) + Vec2Fix::from_int(1, 0),
        spread(0),
        1,
        3,
        1,
    );
    let fast = add_mouse(
        &mut world,
        spread(1) + Vec2Fix::from_int(0, 1),
        spread(1),
        4,
        15,
        1,
    );
    let mut worst = [0.0_f64; 2];
    for n in 1..=360 {
        world.step(dt);
        let t = f64::from(n) * h;
        let xs = world.bodies[slow].position.x.to_f64();
        let yf = world.bodies[fast].position.y.to_f64();
        worst[0] = worst[0].max((xs - (1.0 + 3.0 * t) * exp64(-3.0 * t)).abs());
        worst[1] = worst[1].max((yf - (1.0 + 15.0 * t) * exp64(-15.0 * t)).abs());
    }
    assert!(
        worst[0] <= 0.16 * 3.0 * h && worst[0] >= 0.08 * 3.0 * h,
        "slow: {}",
        worst[0]
    );
    assert!(
        worst[1] <= 0.16 * 15.0 * h && worst[1] >= 0.08 * 15.0 * h,
        "fast: {}",
        worst[1]
    );
}

#[test]
fn mouse_result_does_not_depend_on_the_iteration_count() {
    // the substep solve is exact after the first iteration; later iterations only see the
    // rounding residual of the first (Fix128, 2^-64 per operation)
    let run = |iterations: usize| {
        let mut world = quiet_world(2, iterations);
        let b = add_mouse(&mut world, Vec2Fix::from_int(1, 2), Vec2Fix::ZERO, 1, 9, 1);
        for _ in 0..120 {
            world.step(Fix128::from_ratio(1, 60));
        }
        xy(world.bodies[b].position)
    };
    let one = run(1);
    for it in [4, 16] {
        let p = run(it);
        for c in 0..2 {
            assert!(
                (p[c] - one[c]).abs() < 1e-15,
                "iterations {it}: {:?} vs {:?}",
                p,
                one
            );
        }
    }
}

#[test]
fn mouse_force_is_clamped_to_max_force() {
    // far from the target the spring force k·d = 1e4 N exceeds max_force = 2 N, so the
    // first substep moves the body with |F| = 2 N: |v1| = F·h / m
    let mut world = quiet_world(1, 1);
    let body = world.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::ZERO,
        Fix128::from_int(4),
        circle(),
    ));
    world.add_joint(Joint2D::Mouse {
        body,
        target: Vec2Fix::from_int(30, 40),
        max_force: Fix128::from_int(2),
        stiffness: Fix128::from_int(200),
        damping: Fix128::from_int(10),
    });
    world.step(Fix128::from_ratio(1, 10));
    let v = world.bodies[body].velocity;
    let speed = v.length().to_f64();
    assert!((speed - 2.0 * 0.1 / 4.0).abs() < 1e-12, "speed {speed}");
    // along the direction of the target (3, 4) / 5
    assert!((v.x.to_f64() / speed - 0.6).abs() < 1e-9);
    assert!((v.y.to_f64() / speed - 0.8).abs() < 1e-9);
}

#[test]
fn mouse_replay_is_bit_identical() {
    let run = || {
        let mut world = quiet_world(4, 3);
        let a = add_mouse(&mut world, Vec2Fix::from_int(1, 2), Vec2Fix::ZERO, 1, 7, 1);
        let b = add_mouse(
            &mut world,
            spread(1) + Vec2Fix::from_int(-1, 3),
            spread(1),
            2,
            11,
            1,
        );
        (0..200)
            .map(|_| {
                world.step(Fix128::from_ratio(1, 60));
                (world.bodies[a].position, world.bodies[b].velocity)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

/// Hang a body of mass `m` under gravity from a static anchor with a compliant distance
/// joint of rest length `rest` and return the settled extension.
fn hanging_extension(substeps: usize, iterations: usize) -> f64 {
    let mut world = PhysicsWorld2D::new(PhysicsConfig2D {
        gravity: Vec2Fix::from_int(0, -10),
        substeps,
        iterations,
        damping: Fix128::from_ratio(9, 10),
    });
    let anchor = world.add_body(RigidBody2D::new_static(Vec2Fix::ZERO, circle()));
    let body = world.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::from_int(0, -2),
        Fix128::from_int(2),
        circle(),
    ));
    world.add_joint(Joint2D::Distance {
        body_a: anchor,
        body_b: body,
        local_anchor_a: Vec2Fix::ZERO,
        local_anchor_b: Vec2Fix::ZERO,
        target_distance: Fix128::from_int(2),
        compliance: Fix128::from_ratio(1, 100), // k = 100 N/m
    });
    for _ in 0..600 {
        world.step(Fix128::from_ratio(1, 60));
    }
    -world.bodies[body].position.y.to_f64() - 2.0
}

#[test]
fn distance_joint_stiffness_does_not_depend_on_the_iteration_count() {
    // oracle: extension = α m g = 0.01 · 2 · 10 = 0.2, i.e. k_eff = m g / ext = 100 N/m.
    // After 600 frames at 0.9 velocity retention per frame the residual motion is below
    // 0.9^600 ≈ 1e-27 of the initial one; the remaining error is Fix128 rounding.
    for substeps in [1_usize, 4] {
        for iterations in [1_usize, 4, 16] {
            let ext = hanging_extension(substeps, iterations);
            let k_eff = 20.0 / ext;
            assert!(
                (ext - 0.2).abs() < 1e-12,
                "substeps {substeps}, iterations {iterations}: extension {ext} (k_eff {k_eff})"
            );
        }
    }
}
