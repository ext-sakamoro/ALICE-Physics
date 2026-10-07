//! Oracles for `AngularTether2D` and `KinematicDrive2D` inside
//! `PhysicsWorld2D::step_with_tethers` (returning a body to a rest pose).
//!
//! Angular tether: `τ = k_θ·(θ* − θ) − c_θ·ω` with `k_θ = I ω²`, `c_θ = 2 ζ I ω`. At ζ = 1 the
//! continuous law is `θ(t) − θ* = (θ0 + (ω0 + ω θ0) t)·e^(−ωt)`.
//! Discrete law: XPBD with damping on the scalar `C = θ − θ*` (gradient 1, inverse mass
//! `1/I`), multiplier accumulated over the iterations of a substep. The constraint is
//! linear, so the first iteration solves the substep exactly and the update is backward
//! Euler, `θ' = (θ·(1 + 2ζa) + h·θ̇) / (1 + 2ζa + a²)`, `a = ω h`, `h = dt / substeps`.
//! At ζ = 1 both eigenvalues are `λ = 1/(1 + a) > 0`, so from rest `θ_n = (A + B n)·λⁿ`
//! with `A = θ0`, `B = θ1/λ − θ0` and no overshoot. Leading error against the continuous
//! form (ω0 = 0): `ln λ = −a + a²/2 + O(a³)`, `B = θ0·(a − a²) + O(a³)`, so with `s = ωt`
//! `θ_n − θ(t_n) ≈ θ0·e^(−s)·a·s·(s − 1)/2`; `max |s(s − 1)e^(−s)| = 0.309` (at
//! `s = (3 + √5)/2`), hence `|θ_n − θ(t_n)| <= 0.155·a·|θ0| + O(a²)`. The assertions allow
//! `0.16·a·|θ0|·1.5` and require at least half of it (the bound has teeth).
//!
//! Kinematic drive: the error `e = x − x*` is advanced per substep by the exact solution of
//! `ë = −2ω ė − ω² e`, so the discrete trajectory equals the continuous
//! `e(t) = (e0 + (v0 + ω e0) t)·e^(−ωt)` for every substep count; the only error is
//! `Fix128` rounding (and `e^(−ωh)` to a few units of 2⁻⁶⁴), asserted below 1e-12.

use alice_physics::det_math::{exp64, powf64};
use alice_physics::math::Fix128;
use alice_physics::physics2d::{
    AngularTether2D, BodyType2D, Joint2D, KinematicDrive2D, PhysicsConfig2D, PhysicsWorld2D,
    RigidBody2D, Shape2D, Tethers2D, Vec2Fix,
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

/// Dynamic body at `100·i` on the x axis (far from every other) with `I = inertia`.
fn spinner(world: &mut PhysicsWorld2D, i: i64, theta0: Fix128, inertia: i64) -> usize {
    let mut b = RigidBody2D::new_dynamic(Vec2Fix::from_int(100 * i, 0), Fix128::ONE, circle());
    b.inv_inertia = Fix128::ONE / Fix128::from_int(inertia);
    b.angle = theta0;
    world.add_body(b)
}

fn tether(
    body: usize,
    target: Fix128,
    inertia: i64,
    omega: i64,
    zeta_num: i64,
    zeta_den: i64,
) -> AngularTether2D {
    let i = Fix128::from_int(inertia);
    let w = Fix128::from_int(omega);
    AngularTether2D::new(
        body,
        target,
        i * w * w,
        Fix128::from_int(2) * i * w * Fix128::from_ratio(zeta_num, zeta_den),
    )
}

fn critical_theta(theta0: f64, omega0: f64, w: f64, t: f64) -> f64 {
    (theta0 + (omega0 + w * theta0) * t) * exp64(-w * t)
}

#[test]
fn angular_tether_from_rest_tracks_the_critically_damped_closed_form() {
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
        let target = Fix128::from_ratio(1, 4);
        let theta0 = 1.25_f64; // relative to the target
        let b = spinner(&mut world, 0, Fix128::from_ratio(3, 2), 3);
        let mut set = Tethers2D::new();
        set.add_angular(tether(b, target, 3, omega, 1, 1));
        let mut worst = 0.0_f64;
        let frames = 6 * den / omega.min(6);
        for n in 1..=frames {
            world.step_with_tethers(dt, &set);
            let t = n as f64 / den as f64;
            let rel = world.bodies[b].angle.to_f64() - 0.25;
            worst = worst.max((rel - critical_theta(theta0, 0.0, w, t)).abs());
            assert!(rel >= 0.0, "omega {omega}: overshot at frame {n} ({rel})");
        }
        let bound = 0.16 * w * h * theta0 * 1.5;
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
fn angular_tether_follows_the_discrete_closed_form_with_initial_spin() {
    let dt = Fix128::from_ratio(1, 60);
    let (h, w): (f64, f64) = (1.0 / 60.0, 6.0);
    let mut world = quiet_world(1, 1);
    let b = spinner(&mut world, 0, Fix128::ONE, 2);
    world.bodies[b].angular_velocity = Fix128::from_int(-5);
    let mut set = Tethers2D::new();
    set.add_angular(tether(b, Fix128::ZERO, 2, 6, 1, 1));
    let (x0, v0) = (1.0_f64, -5.0_f64);
    let a = w * h;
    let lambda = 1.0 / (1.0 + a);
    let x1 = (x0 * (1.0 + 2.0 * a) + h * v0) / ((1.0 + a) * (1.0 + a));
    let (aa, bb) = (x0, x1 / lambda - x0);
    for n in 1..=240 {
        world.step_with_tethers(dt, &set);
        let discrete = (aa + bb * f64::from(n)) * powf64(lambda, f64::from(n));
        let got = world.bodies[b].angle.to_f64();
        assert!(
            (got - discrete).abs() < 1e-9,
            "frame {n}: {got} vs {discrete}"
        );
    }
}

/// ζ < 1 oscillates through the target; ζ > 1 never crosses it (the law is the
/// spring-damper, not a fixed critical return).
#[test]
fn angular_tether_damping_ratio_controls_overshoot() {
    let dt = Fix128::from_ratio(1, 240);
    let run = |zn: i64, zd: i64| {
        let mut world = quiet_world(1, 1);
        let b = spinner(&mut world, 0, Fix128::ONE, 1);
        let mut set = Tethers2D::new();
        set.add_angular(tether(b, Fix128::ZERO, 1, 10, zn, zd));
        let mut min = f64::MAX;
        for _ in 0..480 {
            world.step_with_tethers(dt, &set);
            min = min.min(world.bodies[b].angle.to_f64());
        }
        min
    };
    // continuous overshoot at ζ = 0.42 is exp(−πζ/√(1−ζ²)) = 0.234
    let under = run(42, 100);
    assert!(under < -0.18 && under > -0.26, "ζ = 0.42 overshoot {under}");
    assert!(run(2, 1) >= 0.0);
    assert!(run(1, 1) >= 0.0);
}

#[test]
fn angular_tether_is_independent_of_iterations() {
    let dt = Fix128::from_ratio(1, 60);
    let run = |iterations: usize| {
        let mut world = quiet_world(2, iterations);
        let b = spinner(&mut world, 0, Fix128::from_int(2), 2);
        world.bodies[b].angular_velocity = Fix128::from_int(3);
        let mut set = Tethers2D::new();
        set.add_angular(tether(b, Fix128::ZERO, 2, 7, 1, 1));
        (0..90)
            .map(|_| {
                world.step_with_tethers(dt, &set);
                world.bodies[b].angle.to_f64()
            })
            .collect::<Vec<_>>()
    };
    let one = run(1);
    for iterations in [2, 4, 16] {
        for (n, (a, b)) in one.iter().zip(run(iterations)).enumerate() {
            assert!(
                (a - b).abs() < 1e-15,
                "iterations {iterations}, frame {n}: {a} vs {b}"
            );
        }
    }
}

/// Every substep count tracks the continuous law within its own `0.16·a·|θ0|·1.5`, and the
/// error shrinks as substeps grow.
#[test]
fn angular_tether_converges_with_substeps() {
    let (omega, den) = (10_i64, 30_i64);
    let w = omega as f64;
    let mut prev = f64::MAX;
    for substeps in [1_usize, 2, 4, 8] {
        let h = 1.0 / (den as f64 * substeps as f64);
        let mut world = quiet_world(substeps, 1);
        let b = spinner(&mut world, 0, Fix128::ONE, 1);
        let mut set = Tethers2D::new();
        set.add_angular(tether(b, Fix128::ZERO, 1, omega, 1, 1));
        let mut worst = 0.0_f64;
        for n in 1..=60 {
            world.step_with_tethers(Fix128::from_ratio(1, den), &set);
            let t = n as f64 / den as f64;
            worst =
                worst.max((world.bodies[b].angle.to_f64() - critical_theta(1.0, 0.0, w, t)).abs());
        }
        assert!(worst <= 0.16 * w * h * 1.5, "substeps {substeps}: {worst}");
        assert!(
            worst < prev,
            "substeps {substeps}: error did not shrink ({worst} >= {prev})"
        );
        prev = worst;
    }
}

fn state_bits(world: &PhysicsWorld2D) -> Vec<(i64, u64)> {
    world
        .bodies
        .iter()
        .flat_map(|b| {
            [
                b.position.x,
                b.position.y,
                b.angle,
                b.velocity.x,
                b.velocity.y,
                b.angular_velocity,
            ]
        })
        .map(|f| (f.hi, f.lo))
        .collect()
}

/// Tethers of different ω on different bodies in one world each evolve exactly as alone,
/// and a replay is bit-identical.
#[test]
fn tethers_are_independent_and_replay_bit_identically() {
    let dt = Fix128::from_ratio(1, 60);
    let specs = [(0_i64, 4_i64, 1_i64), (1, 9, 2), (2, 15, 3)];
    let together = || {
        let mut world = quiet_world(3, 4);
        let mut set = Tethers2D::new();
        for &(i, omega, inertia) in &specs {
            let b = spinner(&mut world, i, Fix128::from_int(i + 1), inertia);
            set.add_angular(tether(b, Fix128::ZERO, inertia, omega, 1, 1));
        }
        for _ in 0..120 {
            world.step_with_tethers(dt, &set);
        }
        world
    };
    let w1 = together();
    assert_eq!(state_bits(&w1), state_bits(&together()), "replay differs");
    for (k, &(i, omega, inertia)) in specs.iter().enumerate() {
        let mut alone = quiet_world(3, 4);
        let b = spinner(&mut alone, i, Fix128::from_int(i + 1), inertia);
        let mut set = Tethers2D::new();
        set.add_angular(tether(b, Fix128::ZERO, inertia, omega, 1, 1));
        for _ in 0..120 {
            alone.step_with_tethers(dt, &set);
        }
        assert_eq!(
            (w1.bodies[k].angle.hi, w1.bodies[k].angle.lo),
            (alone.bodies[b].angle.hi, alone.bodies[b].angle.lo),
            "tether {k} is not independent"
        );
    }
}

/// A mouse joint plus an angular tether returns both position and orientation.
#[test]
fn mouse_plus_angular_tether_returns_the_full_pose() {
    let dt = Fix128::from_ratio(1, 60);
    let mut world = quiet_world(4, 4);
    let b = spinner(&mut world, 0, Fix128::from_int(3), 2);
    world.bodies[b].velocity = Vec2Fix::from_int(2, -1);
    world.add_joint(Joint2D::Mouse {
        body: b,
        target: Vec2Fix::from_int(5, 5),
        max_force: Fix128::from_int(1_000_000),
        stiffness: Fix128::from_int(36),
        damping: Fix128::from_int(12),
    });
    let mut set = Tethers2D::new();
    set.add_angular(AngularTether2D::critically_damped(
        b,
        Fix128::ZERO,
        Fix128::from_int(2),
        Fix128::from_int(6),
    ));
    for _ in 0..360 {
        world.step_with_tethers(dt, &set);
    }
    let body = &world.bodies[b];
    assert!((body.position.x.to_f64() - 5.0).abs() < 1e-6);
    assert!((body.position.y.to_f64() - 5.0).abs() < 1e-6);
    assert!(
        body.angle.to_f64().abs() < 1e-6,
        "angle {}",
        body.angle.to_f64()
    );
}

// ---- kinematic drive ----

fn kinematic(world: &mut PhysicsWorld2D, at: Vec2Fix, angle: Fix128) -> usize {
    let mut b = RigidBody2D::new_kinematic(at, circle());
    b.angle = angle;
    world.add_body(b)
}

#[test]
fn kinematic_drive_equals_the_continuous_closed_form_for_any_substeps() {
    // ω h up to 1.16 (ω = 2π/0.09 at 60 Hz)
    for (omega_num, omega_den, substeps) in [
        (8_i64, 1_i64, 1_usize),
        (8, 1, 5),
        (628_318, 9_0000, 1),
        (3, 1, 3),
    ] {
        let w = omega_num as f64 / omega_den as f64;
        let mut world = quiet_world(substeps, 2);
        let target = Vec2Fix::from_int(1, -2);
        let b = kinematic(&mut world, Vec2Fix::from_int(4, 2), Fix128::from_int(2));
        world.bodies[b].velocity = Vec2Fix::from_int(-1, 3);
        world.bodies[b].angular_velocity = Fix128::from_int(-4);
        let mut set = Tethers2D::new();
        set.add_drive(KinematicDrive2D::new(
            b,
            target,
            Fix128::from_ratio(1, 2),
            Fix128::from_ratio(omega_num, omega_den),
        ));
        let (e0, v0) = ([3.0, 4.0, 1.5], [-1.0, 3.0, -4.0]);
        for n in 1..=240 {
            world.step_with_tethers(Fix128::from_ratio(1, 60), &set);
            let t = f64::from(n) / 60.0;
            let body = &world.bodies[b];
            let got = [
                body.position.x.to_f64() - 1.0,
                body.position.y.to_f64() + 2.0,
                body.angle.to_f64() - 0.5,
            ];
            let vel = [
                body.velocity.x.to_f64(),
                body.velocity.y.to_f64(),
                body.angular_velocity.to_f64(),
            ];
            for c in 0..3 {
                let s = v0[c] + w * e0[c];
                let exact = (e0[c] + s * t) * exp64(-w * t);
                let exact_v = (v0[c] - w * s * t) * exp64(-w * t);
                assert!(
                    (got[c] - exact).abs() < 1e-12,
                    "ω {w}, substeps {substeps}, frame {n}, c {c}: {} vs {exact}",
                    got[c]
                );
                assert!(
                    (vel[c] - exact_v).abs() < 1e-12 * (1.0 + w),
                    "velocity c {c}: {} vs {exact_v}",
                    vel[c]
                );
            }
        }
    }
}

#[test]
fn kinematic_drive_from_rest_converges_without_overshoot() {
    let mut world = quiet_world(4, 4);
    let b = kinematic(&mut world, Vec2Fix::from_int(-3, 7), Fix128::from_int(-2));
    let mut set = Tethers2D::new();
    let id = set.add_drive(KinematicDrive2D::new(
        b,
        Vec2Fix::ZERO,
        Fix128::ZERO,
        Fix128::from_int(5),
    ));
    for _ in 0..240 {
        world.step_with_tethers(Fix128::from_ratio(1, 60), &set);
        let body = &world.bodies[b];
        assert!(body.position.x <= Fix128::ZERO && body.position.y >= Fix128::ZERO);
        assert!(body.angle <= Fix128::ZERO);
    }
    let body = &world.bodies[b];
    assert!(body.position.length().to_f64() < 1e-6 && body.angle.to_f64().abs() < 1e-6);
    // moving the target re-aims the drive; removing it leaves the body coasting
    set.drive_mut(id).unwrap().target_position = Vec2Fix::from_int(1, 0);
    world.step_with_tethers(Fix128::from_ratio(1, 60), &set);
    assert!(world.bodies[b].velocity.x > Fix128::ZERO);
    let v = world.bodies[b].velocity;
    assert!(set.remove_drive(id).is_some());
    let x = world.bodies[b].position;
    world.step_with_tethers(Fix128::from_ratio(1, 60), &set);
    assert_eq!(world.bodies[b].velocity, v);
    let coast = world.bodies[b].position - (x + v * Fix128::from_ratio(1, 60));
    assert!(
        coast.length().to_f64() < 1e-15,
        "coasting drifted by {}",
        coast.length().to_f64()
    );
}

/// Kinematic bodies driven through each other do not collide; a driven body still pushes a
/// dynamic one (contacts see its velocity).
#[test]
fn driven_kinematic_bodies_pass_through_each_other_and_push_dynamic_ones() {
    let mut world = quiet_world(4, 4);
    let a = kinematic(&mut world, Vec2Fix::from_int(-2, 0), Fix128::ZERO);
    let b = kinematic(&mut world, Vec2Fix::from_int(2, 0), Fix128::ZERO);
    let ball = world.add_body(RigidBody2D::new_dynamic(
        Vec2Fix::new(Fix128::ZERO, Fix128::from_int(5)),
        Fix128::ONE,
        Shape2D::Circle {
            radius: Fix128::from_ratio(1, 2),
        },
    ));
    let pusher = world.add_body(RigidBody2D::new_kinematic(
        Vec2Fix::new(Fix128::from_int(-2), Fix128::from_int(5)),
        Shape2D::Circle {
            radius: Fix128::from_ratio(1, 2),
        },
    ));
    let mut set = Tethers2D::new();
    let omega = Fix128::from_int(6);
    set.add_drive(KinematicDrive2D::new(
        a,
        Vec2Fix::from_int(2, 0),
        Fix128::ZERO,
        omega,
    ));
    set.add_drive(KinematicDrive2D::new(
        b,
        Vec2Fix::from_int(-2, 0),
        Fix128::ZERO,
        omega,
    ));
    set.add_drive(KinematicDrive2D::new(
        pusher,
        Vec2Fix::from_int(3, 5),
        Fix128::ZERO,
        omega,
    ));
    // e(t) = 4·(1 + 6t)·e^(−6t) < 1e-7 at t = 5 s
    for _ in 0..300 {
        world.step_with_tethers(Fix128::from_ratio(1, 60), &set);
    }
    assert!((world.bodies[a].position.x.to_f64() - 2.0).abs() < 1e-6);
    assert!((world.bodies[b].position.x.to_f64() + 2.0).abs() < 1e-6);
    assert!(
        world.bodies[ball].position.x.to_f64() > 3.5,
        "ball not pushed: {}",
        world.bodies[ball].position.x.to_f64()
    );
}

// ---- degenerate input ----

/// Out-of-range indices and mismatched body types are ignored (state unchanged vs no set);
/// `ω <= 0` deactivates a drive; `dt = 0` changes nothing; a huge `ω h` lands exactly on the
/// target at rest.
#[test]
fn degenerate_tethers_are_ignored_or_saturate() {
    let dt = Fix128::from_ratio(1, 60);
    let base = || {
        let mut world = quiet_world(2, 2);
        let d = spinner(&mut world, 0, Fix128::ONE, 1);
        world.bodies[d].angular_velocity = Fix128::from_int(2);
        let k = kinematic(&mut world, Vec2Fix::from_int(100, 0), Fix128::ONE);
        world.bodies[k].velocity = Vec2Fix::from_int(1, 0);
        world.bodies[k].angular_velocity = Fix128::ONE;
        // a nonzero inverse inertia on the kinematic body: only the body type keeps the
        // angular tether off it
        world.bodies[k].inv_inertia = Fix128::ONE;
        (world, d, k)
    };
    let (mut plain, _, _) = base();
    let (mut tethered, d, k) = base();
    let mut set = Tethers2D::new();
    set.add_angular(AngularTether2D::new(
        99,
        Fix128::ZERO,
        Fix128::ONE,
        Fix128::ONE,
    ));
    set.add_angular(AngularTether2D::new(
        k,
        Fix128::ZERO,
        Fix128::from_int(50),
        Fix128::ONE,
    ));
    set.add_drive(KinematicDrive2D::new(
        99,
        Vec2Fix::ZERO,
        Fix128::ZERO,
        Fix128::ONE,
    ));
    set.add_drive(KinematicDrive2D::new(
        d,
        Vec2Fix::ZERO,
        Fix128::ZERO,
        Fix128::from_int(5),
    ));
    set.add_drive(KinematicDrive2D::new(
        k,
        Vec2Fix::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
    ));
    set.add_drive(KinematicDrive2D::new(
        k,
        Vec2Fix::ZERO,
        Fix128::ZERO,
        Fix128::from_int(-3),
    ));
    for _ in 0..30 {
        plain.step(dt);
        tethered.step_with_tethers(dt, &set);
    }
    assert_eq!(state_bits(&plain), state_bits(&tethered));
    assert_eq!(tethered.bodies[k].body_type, BodyType2D::Kinematic);

    // dt = 0: nothing moves
    let (mut zero, _, k0) = base();
    let mut s0 = Tethers2D::new();
    s0.add_drive(KinematicDrive2D::new(
        k0,
        Vec2Fix::ZERO,
        Fix128::ZERO,
        Fix128::from_int(5),
    ));
    let before = state_bits(&zero);
    zero.step_with_tethers(Fix128::ZERO, &s0);
    assert_eq!(before, state_bits(&zero));

    // ω h = 1e6: exactly at the target, at rest
    let (mut stiff, _, ks) = base();
    let mut ss = Tethers2D::new();
    ss.add_drive(KinematicDrive2D::new(
        ks,
        Vec2Fix::from_int(7, 7),
        Fix128::from_int(3),
        Fix128::from_int(120_000_000),
    ));
    stiff.step_with_tethers(dt, &ss);
    let body = &stiff.bodies[ks];
    assert_eq!(body.position, Vec2Fix::from_int(7, 7));
    assert_eq!(body.angle, Fix128::from_int(3));
    assert_eq!(body.velocity, Vec2Fix::ZERO);
    assert_eq!(body.angular_velocity, Fix128::ZERO);

    // handles of removed / unknown entries
    let mut s = Tethers2D::new();
    let id = s.add_angular(AngularTether2D::new(
        0,
        Fix128::ZERO,
        Fix128::ONE,
        Fix128::ONE,
    ));
    assert!(s.angular(id).is_some());
    s.remove_angular(id);
    assert!(
        s.angular(id).is_none() && s.angular_mut(id).is_none() && s.remove_angular(id).is_none()
    );
}
