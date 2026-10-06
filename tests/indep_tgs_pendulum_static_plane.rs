//! Independent closed-form checks of the TGS backend's joints (projected
//! inside the substep loop) and static colliders.
//!
//! Expected values are derived here:
//!
//! - a body pendulum: unit-mass bob of central moment `I = 0.4` at arm `L`
//!   from a fixed pivot. Small oscillations obey
//!   `θ'' + β θ' + ω₀² θ = 0` with `ω₀² = m g L / (I + m L²)` and
//!   `β = −ln(d) / Δt` the per-frame damping `d` as a rate, so
//!   `T = 2π / √(ω₀² − β²/4)`; at `L = 1`, `g = 10`, `d = 0.99`, `Δt = 1/60`
//!   this is `2.36606` s. The finite amplitude `θ` lengthens the period by
//!   `θ²/16` to leading order; with damping the amplitude decays as
//!   `θ₀ e^{−βt/2}`, so the correction over a window is
//!   `(θ₀²/16) · mean(e^{−βt})`;
//! - without damping, the exact period of the physical pendulum at amplitude
//!   `θ₀` is `T = 4 √((I + m L²) / (m g L)) · K(sin(θ₀/2))` with `K` the
//!   complete elliptic integral of the first kind (by the arithmetic-geometric
//!   mean: `K(k) = π / (2 AGM(1, √(1 − k²)))`), and the energy is constant;
//! - a ball on a static plane tilted by `θ` stays at distance `r` from it.
//!
//! Every scene runs `PhysicsWorld::step` with `solver_backend: Tgs`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::joint::{BallJoint, HingeJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::static_collider::StaticCollider;
use alice_physics::SolverBackend;

const DT: f64 = 1.0 / 60.0;
const I_BOB: f64 = 0.4;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn tgs(substeps: usize, damping: Fix128) -> SolverConfig {
    SolverConfig {
        substeps,
        damping,
        solver_backend: SolverBackend::Tgs,
        ..SolverConfig::default()
    }
}

fn g() -> f64 {
    -SolverConfig::default().gravity.y.to_f64()
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Ball,
    Hinge,
}

fn pendulum(kind: Kind, cfg: SolverConfig, l: f64, theta0: f64) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(cfg);
    let pivot = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let mut bob = RigidBody::new_dynamic(v3(l * theta0.sin(), -l * theta0.cos(), 0.0), Fix128::ONE);
    bob.inv_inertia = v3(1.0 / I_BOB, 1.0 / I_BOB, 1.0 / I_BOB);
    let bob = w.add_body(bob);
    let anchor_b = v3(-l * theta0.sin(), l * theta0.cos(), 0.0);
    let joint = match kind {
        Kind::Ball => Joint::Ball(BallJoint::new(pivot, bob, Vec3Fix::ZERO, anchor_b)),
        Kind::Hinge => Joint::Hinge(HingeJoint::new(
            pivot,
            bob,
            Vec3Fix::ZERO,
            anchor_b,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )),
    };
    w.add_joint(joint);
    (w, bob)
}

/// Times (s) of the bob's upward crossings of `x = 0` (left to right),
/// linearly interpolated inside a frame, plus the worst relative energy
/// change `|E − E₀| / (E₀ − E_bottom)` (translational + rotational +
/// potential).
fn crossings(w: &mut PhysicsWorld, bob: usize, l: f64, seconds: f64) -> (Vec<f64>, f64) {
    let energy = |w: &PhysicsWorld| {
        let b = w.get_body(bob).expect("bob");
        let v = b.velocity.length().to_f64();
        let om = b.angular_velocity.length().to_f64();
        0.5 * v * v + 0.5 * I_BOB * om * om + g() * b.position.y.to_f64()
    };
    let e0 = energy(w);
    let swing = e0 + g() * l;
    let mut worst: f64 = 0.0;
    let steps = (seconds / DT).round() as usize;
    let mut out = Vec::new();
    let mut prev = w.get_body(bob).expect("bob").position.x.to_f64();
    for n in 1..=steps {
        w.step(fx(DT));
        let x = w.get_body(bob).expect("bob").position.x.to_f64();
        if prev < 0.0 && x >= 0.0 {
            out.push((n as f64 - 1.0 - prev / (x - prev)) * DT);
        }
        prev = x;
        worst = worst.max((energy(w) - e0).abs() / swing);
    }
    (out, worst)
}

fn mean_period(c: &[f64]) -> f64 {
    assert!(c.len() >= 3, "only {} crossings", c.len());
    (c[c.len() - 1] - c[0]) / (c.len() - 1) as f64
}

/// The claim's scene (L = 1, θ₀ = 0.1, default damping) at substeps 4 and 8.
/// Budget `1.5e-3` relative: crossing interpolation (`< dt²` per crossing,
/// `1e-4` relative over 3 periods) and the per-frame instead of continuous
/// damping (`O(β Δt)` of the `β²/4` term, `1e-4`). Dropping the bob's
/// rotation (`ω₀² = g / L`) moves the period by 18 %.
#[test]
fn the_damped_pendulum_period_is_the_closed_form_with_amplitude_correction() {
    let l = 1.0;
    let theta0: f64 = 0.1;
    let d = SolverConfig::default().damping;
    let beta = -d.to_f64().ln() / DT;
    let w0_sq = g() * l / (I_BOB + l * l);
    let t_damped = 2.0 * std::f64::consts::PI / (w0_sq - beta * beta / 4.0).sqrt();
    assert!(
        (t_damped - 2.366).abs() < 1e-3,
        "derivation: {t_damped} vs the stated 2.366"
    );
    let mut failures = Vec::new();
    for kind in [Kind::Ball, Kind::Hinge] {
        for substeps in [4, 8] {
            let (mut w, bob) = pendulum(kind, tgs(substeps, d), l, theta0);
            let (c, _) = crossings(&mut w, bob, l, 3.5 * t_damped);
            let period = mean_period(&c);
            // amplitude correction averaged over the measured window
            let (t0, t1) = (c[0], c[c.len() - 1]);
            let mean_decay = ((-beta * t0).exp() - (-beta * t1).exp()) / (beta * (t1 - t0));
            let want = t_damped * (1.0 + theta0 * theta0 / 16.0 * mean_decay);
            let rel = (period - want).abs() / want;
            eprintln!(
                "{kind:?} s={substeps}: period {period:.6}, closed form {want:.6}, rel {rel:.2e}"
            );
            if rel > 1.5e-3 {
                failures.push(format!(
                    "{kind:?} s={substeps}: period {period:.6}, closed form {want:.6} (rel {rel:.2e})"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn elliptic_k(k: f64) -> f64 {
    let (mut a, mut b) = (1.0f64, (1.0 - k * k).sqrt());
    for _ in 0..40 {
        let (an, bn) = ((a + b) / 2.0, (a * b).sqrt());
        a = an;
        b = bn;
    }
    std::f64::consts::PI / (2.0 * a)
}

/// Upward crossings of `x = 0` and, for each period between two successive
/// crossings, the largest `|θ|` the bob reached in it (`θ = atan2(x, −y)`).
fn periods_and_amplitudes(w: &mut PhysicsWorld, bob: usize, seconds: f64) -> Vec<(f64, f64)> {
    let steps = (seconds / DT).round() as usize;
    let mut out = Vec::new();
    let mut last_cross: Option<f64> = None;
    let mut amp: f64 = 0.0;
    let mut prev = w.get_body(bob).expect("bob").position.x.to_f64();
    for n in 1..=steps {
        w.step(fx(DT));
        let p = w.get_body(bob).expect("bob").position;
        let (x, y) = (p.x.to_f64(), p.y.to_f64());
        amp = amp.max(x.atan2(-y).abs());
        if prev < 0.0 && x >= 0.0 {
            let t = (n as f64 - 1.0 - prev / (x - prev)) * DT;
            if let Some(t0) = last_cross {
                out.push((t - t0, amp));
            }
            last_cross = Some(t);
            amp = 0.0;
        }
        prev = x;
    }
    out
}

fn elliptic_period(l: f64, theta: f64) -> f64 {
    4.0 * ((I_BOB + l * l) / (g() * l)).sqrt() * elliptic_k((theta / 2.0).sin())
}

/// No damping, L = 1.5, θ₀ = 0.6: each full period equals the exact
/// large-amplitude period at the amplitude the bob reached in it (2.3 % above
/// the small-angle period), within `1e-3` (crossing interpolation `< dt²`,
/// and the amplitude moving within one period by up to `1 %`, a `2e-4`
/// effect on the period). The amplitude itself is not conserved (next test).
#[test]
fn the_undamped_large_amplitude_period_is_the_elliptic_closed_form() {
    let l = 1.5;
    let theta0: f64 = 0.6;
    let t0 = elliptic_period(l, theta0);
    let small = 2.0 * std::f64::consts::PI * ((I_BOB + l * l) / (g() * l)).sqrt();
    assert!(
        t0 / small - 1.0 > 2e-2,
        "the amplitude correction must be resolvable"
    );
    let mut failures = Vec::new();
    for kind in [Kind::Ball, Kind::Hinge] {
        for substeps in [4, 8] {
            let (mut w, bob) = pendulum(kind, tgs(substeps, Fix128::ONE), l, theta0);
            let runs = periods_and_amplitudes(&mut w, bob, 5.5 * t0);
            assert!(
                runs.len() >= 3,
                "{kind:?} s={substeps}: {} periods",
                runs.len()
            );
            for (i, (period, amp)) in runs.iter().enumerate() {
                let want = elliptic_period(l, *amp);
                let rel = (period - want).abs() / want;
                if rel > 1e-3 {
                    failures.push(format!(
                        "{kind:?} s={substeps} period {i}: {period:.6} at amplitude {amp:.4}, \
                         elliptic {want:.6} (rel {rel:.2e})"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Without damping the swing energy is constant, so the amplitude stays
/// `θ₀ = 0.6`. Measured (hinge and ball, TGS, L = 1.5, 12 s): the amplitude
/// decays 0.6 → 0.517 / 0.556 / 0.578 at substeps 4 / 8 / 16, i.e. a loss
/// first order in `h`, bit-for-bit the same as the XPBD backend; the bob's
/// `ω_z` differs from the arm rate `θ'` by up to 0.18 / 0.092 / 0.046 rad/s.
/// The joint projection is numerically dissipative; the frame damping of the
/// default configuration (`β = 0.60 /s`) dwarfs it (`0.012 /s` here).
#[test]
#[ignore = "src gap: joint projection loses swing energy at first order in h (amplitude 0.6 to 0.517 in 12 s at 4 substeps), same on XPBD"]
fn the_undamped_pendulum_keeps_its_amplitude() {
    let l = 1.5;
    let theta0: f64 = 0.6;
    let t0 = elliptic_period(l, theta0);
    let mut failures = Vec::new();
    for kind in [Kind::Ball, Kind::Hinge] {
        for substeps in [4, 8] {
            let (mut w, bob) = pendulum(kind, tgs(substeps, Fix128::ONE), l, theta0);
            let runs = periods_and_amplitudes(&mut w, bob, 4.5 * t0);
            let last = runs.last().expect("a period").1;
            if (last - theta0).abs() > 1e-3 {
                failures.push(format!(
                    "{kind:?} s={substeps}: amplitude {last:.4}, want {theta0}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A ball of radius 0.5 dropped on a static plane tilted by 0.3 rad (normal
/// `(−sin θ, cos θ, 0)` through the origin) stays at distance `r` from it
/// (within `2e-2` below, `5e-3` above) and moves along it, at substeps 1, 4
/// and 8 under TGS, while it slides down the slope as on a frictionless plane
/// (the static-collider response removes the approaching normal velocity
/// only): downhill speed `g sin θ t` after 2 s within `1e-3` m/s.
#[test]
fn a_ball_on_a_tilted_static_plane_stays_at_its_radius_under_tgs() {
    let theta: f64 = 0.3;
    let r = 0.5;
    let n = [-theta.sin(), theta.cos(), 0.0];
    let mut failures = Vec::new();
    for substeps in [1, 4, 8] {
        let mut w = PhysicsWorld::new(tgs(substeps, Fix128::ONE));
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            v3(n[0], n[1], n[2]),
            Fix128::ZERO,
        )));
        let b = w.add_body_with_radius(
            RigidBody::new_dynamic(v3(n[0] * 0.8, n[1] * 0.8, 0.0), Fix128::ONE),
            fx(r),
        );
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for k in 0..120 {
            w.step(fx(DT));
            let p = w.get_body(b).expect("ball").position;
            let dist = p.x.to_f64() * n[0] + p.y.to_f64() * n[1];
            if k >= 30 {
                lo = lo.min(dist);
                hi = hi.max(dist);
            }
        }
        let body = w.get_body(b).expect("ball");
        let vn = body.velocity.x.to_f64() * n[0] + body.velocity.y.to_f64() * n[1];
        let along =
            -(body.velocity.x.to_f64() * theta.cos() + body.velocity.y.to_f64() * theta.sin());
        eprintln!(
            "s={substeps}: distance [{lo:.5}, {hi:.5}], v_n {vn:.2e}, downhill speed {along:.4} \
             (frictionless g sin θ t = {:.4})",
            g() * theta.sin() * 2.0
        );
        let free = g() * theta.sin() * 2.0;
        if lo < r - 2e-2
            || hi > r + 5e-3
            || vn.abs() > 5e-2 * g() * DT
            || (along - free).abs() > 1e-3
        {
            failures.push(format!(
                "s={substeps}: distance [{lo:.5}, {hi:.5}], normal velocity {vn:.3e}, \
                 downhill speed {along} (want {free})"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
