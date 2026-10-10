//! Analytic oracles for the 3D `Joint::Spring` (`SpringJoint`) spring-damper.
//!
//! The joint is the force `F = −k d − c d'` (rest length 0) between two
//! anchors, solved once per substep as XPBD with damping (Macklin, Müller,
//! Chentanez 2016, eq. 26), i.e. one backward Euler step of `m x'' = F` per
//! substep. For a body of mass `m` against a static anchor with `k = m ω²`,
//! `c = 2 m ω` (critical damping) and `a = ω h` the substep map is
//!
//! `x' = (x (1 + 2a) + h v) / (1 + a)²`,  `v' = (x' − x) / h`
//!
//! whose discrete solution is, with `r = 1 / (1 + a)` and `t = n h`,
//!
//! `x_n = (x0 (1 + n a / (1 + a)) + n h v0 r) r^n`.
//!
//! # Error bound against the continuous solution
//!
//! The continuous solution is `x(t) = (x0 + (v0 + ω x0) t) e^(−ωt)`. With
//! `τ = ω t` and `r^n = e^(−τ (1 − a/2 + O(a²)))`, expanding to first order in
//! `a`:
//!
//! - the `x0` part: `x0 e^(−τ) (1 + τ) + x0 a e^(−τ) τ (τ − 1) / 2`; the
//!   coefficient `|τ (τ − 1) e^(−τ) / 2|` peaks at `τ = (3 + √5) / 2` with
//!   `0.1545`
//! - the `v0` part: `(v0 / ω) τ e^(−τ) + (v0 / ω) a e^(−τ) τ (τ / 2 − 1)`;
//!   the coefficient `|τ (τ / 2 − 1) e^(−τ)|` peaks at `τ = 2 − √2` with
//!   `0.2305`
//!
//! so `|x_n − x(t_n)| <= a (0.1545 |x0| + 0.2305 |v0| / ω) + O(a²)`. The tests
//! use `1.1 ×` that (the `O(a²)` margin; `a <= 0.1` here) and check that the
//! measured error reaches at least 80 % of the first-order coefficient, so the
//! bound is not loose. Every component obeys it independently because the
//! rest-length-0 spring and its damper are isotropic.
//!
//! Besides the continuous solution, each run is compared with the same map
//! iterated in `f64` (`1e-9`): that pins the discrete law itself, which a
//! first-order-consistent but different scheme (explicit damping, a
//! non-accumulated second pass) could pass the `O(a)` bound with.

use alice_physics::det_math::{atan2_64, exp64, sqrt64};
#[allow(deprecated)] // the separation-based variant is pinned below
use alice_physics::joint::solve_joints_breakable;
use alice_physics::joint::{Joint, SpringJoint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::{SleepConfig, SolverBackend};

const DT: f64 = 1.0 / 60.0;
const C_X0: f64 = 0.1545;
const C_V0: f64 = 0.2305;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn to3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn config(substeps: usize, iterations: usize, damping: f64) -> SolverConfig {
    SolverConfig {
        substeps,
        iterations,
        gravity: Vec3Fix::ZERO,
        damping: fx(damping),
        ..SolverConfig::default()
    }
}

/// A world that never puts a body to sleep (the default freezes the tail of a
/// decay once `|v| < 0.01` for 60 frames).
fn world(cfg: SolverConfig) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(cfg);
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::ZERO,
        angular_threshold: Fix128::ZERO,
        frames_to_sleep: u32::MAX,
    });
    w
}

/// Rest-length-0 spring of angular frequency `omega` and damping ratio `zeta`
/// from a static anchor at the origin to a new body of mass `m`.
fn add_spring_body(
    w: &mut PhysicsWorld,
    m: f64,
    omega: f64,
    zeta: f64,
    x0: [f64; 3],
    v0: [f64; 3],
) -> usize {
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let mut b = RigidBody::new_dynamic(v3(x0[0], x0[1], x0[2]), fx(m));
    b.velocity = v3(v0[0], v0[1], v0[2]);
    let body = w.add_body(b);
    w.add_joint(Joint::Spring(SpringJoint::new(
        anchor,
        body,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::ZERO,
        fx(m * omega * omega),
        fx(2.0 * zeta * m * omega),
    )));
    body
}

/// Critically damped closed form per component.
fn closed(t: f64, x0: f64, v0: f64, omega: f64) -> f64 {
    (x0 + (v0 + omega * x0) * t) * exp64(-omega * t)
}

/// First-order discretisation error bound (module doc), with a 10 % margin.
fn bound(x0: f64, v0: f64, omega: f64, h: f64) -> f64 {
    1.1 * omega * h * (C_X0 * x0.abs() + C_V0 * v0.abs() / omega)
}

/// The substep map iterated in `f64`: `frames` frames of `substeps` substeps,
/// velocity scaled by `retain` after each frame (`SolverConfig::damping`).
/// Returns the position after every frame.
fn reference(
    x0: f64,
    v0: f64,
    omega: f64,
    zeta: f64,
    substeps: usize,
    frames: usize,
    retain: f64,
) -> Vec<f64> {
    let h = DT / substeps as f64;
    let a = omega * h;
    let (mut x, mut v) = (x0, v0);
    let mut out = Vec::with_capacity(frames);
    for _ in 0..frames {
        for _ in 0..substeps {
            let xn = (x * (1.0 + 2.0 * zeta * a) + h * v) / (1.0 + 2.0 * zeta * a + a * a);
            v = (xn - x) / h;
            x = xn;
        }
        v *= retain;
        out.push(x);
    }
    out
}

fn pos(w: &PhysicsWorld, i: usize) -> [f64; 3] {
    to3(w.get_body(i).expect("body").position)
}

/// Runs `frames` frames and returns every frame's position of `body`.
fn run(w: &mut PhysicsWorld, body: usize, frames: usize) -> Vec<[f64; 3]> {
    (0..frames)
        .map(|_| {
            w.step(fx(DT));
            pos(w, body)
        })
        .collect()
}

/// (1) Critically damped return from rest: the closed form within the
/// derived bound, the discrete law within `1e-9`, and no overshoot.
#[test]
fn critical_return_follows_the_closed_form_without_overshoot() {
    let (m, omega, x0) = (2.0, 8.0, 1.0);
    let substeps = 4;
    let h = DT / substeps as f64;
    let frames = 120;
    let mut w = world(config(substeps, 1, 1.0));
    let body = add_spring_body(&mut w, m, omega, 1.0, [x0, 0.0, 0.0], [0.0; 3]);
    let got = run(&mut w, body, frames);
    let refx = reference(x0, 0.0, omega, 1.0, substeps, frames, 1.0);
    let lim = bound(x0, 0.0, omega, h);
    let mut max_err: f64 = 0.0;
    for (n, p) in got.iter().enumerate() {
        let t = (n + 1) as f64 * DT;
        let err = (p[0] - closed(t, x0, 0.0, omega)).abs();
        max_err = max_err.max(err);
        assert!(err <= lim, "t {t:.4}: |x - closed| {err:.3e} > {lim:.3e}");
        assert!(
            (p[0] - refx[n]).abs() < 1e-9,
            "t {t:.4}: x {} vs discrete law {}",
            p[0],
            refx[n]
        );
        assert!(p[0] >= -1e-12, "t {t:.4}: overshoot to {}", p[0]);
        assert!(p[1].abs() < 1e-12 && p[2].abs() < 1e-12, "off-axis drift");
    }
    let coeff = max_err / (omega * h * x0);
    assert!(
        coeff > 0.8 * C_X0,
        "measured error coefficient {coeff:.4} far below the derived {C_X0}: bound is loose"
    );
}

/// (2) A tangential initial velocity does not make the body orbit: every
/// component decays with the closed form, so the polar angle stays below
/// `π/4` (`atan(ωt / (1 + ωt))`) and the angular momentum dies out.
#[test]
fn tangential_velocity_is_damped_and_the_body_does_not_orbit() {
    let (m, omega, r0) = (1.5, 8.0, 1.0);
    let v0 = omega * r0; // the speed of a circular orbit under the spring alone
    let substeps = 4;
    let h = DT / substeps as f64;
    let mut w = world(config(substeps, 1, 1.0));
    let body = add_spring_body(&mut w, m, omega, 1.0, [r0, 0.0, 0.0], [0.0, v0, 0.0]);
    let l0 = m * r0 * v0;
    let mut prev = [r0, 0.0, 0.0];
    let mut l_end = f64::NAN;
    for n in 0..120 {
        w.step(fx(DT));
        let p = pos(&w, body);
        let t = (n + 1) as f64 * DT;
        assert!(
            (p[0] - closed(t, r0, 0.0, omega)).abs() <= bound(r0, 0.0, omega, h),
            "t {t:.4}: x {}",
            p[0]
        );
        assert!(
            (p[1] - closed(t, 0.0, v0, omega)).abs() <= bound(0.0, v0, omega, h),
            "t {t:.4}: y {}",
            p[1]
        );
        let angle = atan2_64(p[1], p[0]);
        assert!(
            angle < std::f64::consts::FRAC_PI_4 + 0.01,
            "t {t:.4}: polar angle {angle:.4} rad, the body is orbiting"
        );
        let v = w.get_body(body).expect("body").velocity;
        l_end = m * (p[0] * v.y.to_f64() - p[1] * v.x.to_f64());
        prev = p;
    }
    assert!(
        l_end.abs() < 1e-4 * l0,
        "angular momentum {l_end:.3e} after 2 s (L0 {l0}), last position {prev:?}"
    );
}

/// (3) The same continuous solution for substeps 1 / 4 / 8 (each within its
/// own bound, so pairwise within the sum), and bit-identical results for
/// iterations 1 / 4 / 16: the joint is solved once per substep.
#[test]
fn result_does_not_depend_on_substeps_or_iterations() {
    let (m, omega, x0, v0) = (1.0, 6.0, 0.8, -1.5);
    let frames = 60;
    let mut runs = Vec::new();
    for substeps in [1usize, 4, 8] {
        let h = DT / substeps as f64;
        let mut w = world(config(substeps, 1, 1.0));
        let body = add_spring_body(&mut w, m, omega, 1.0, [x0, 0.0, 0.0], [v0, 0.0, 0.0]);
        let got = run(&mut w, body, frames);
        let refx = reference(x0, v0, omega, 1.0, substeps, frames, 1.0);
        for (n, p) in got.iter().enumerate() {
            let t = (n + 1) as f64 * DT;
            let err = (p[0] - closed(t, x0, v0, omega)).abs();
            let lim = bound(x0, v0, omega, h);
            assert!(
                err <= lim,
                "substeps {substeps} t {t:.4}: {err:.3e} > {lim:.3e}"
            );
            assert!(
                (p[0] - refx[n]).abs() < 1e-9,
                "substeps {substeps} t {t:.4}"
            );
        }
        runs.push((substeps, got));
    }
    for i in 0..runs.len() {
        for j in i + 1..runs.len() {
            let (si, ref a) = runs[i];
            let (sj, ref b) = runs[j];
            let lim = bound(x0, v0, omega, DT / si as f64) + bound(x0, v0, omega, DT / sj as f64);
            for n in 0..frames {
                assert!(
                    (a[n][0] - b[n][0]).abs() <= lim,
                    "substeps {si} vs {sj} frame {n}"
                );
            }
        }
    }

    let bits = |iterations: usize| {
        let mut w = world(config(4, iterations, 1.0));
        let body = add_spring_body(&mut w, m, omega, 1.0, [x0, 0.3, 0.0], [v0, 0.0, 0.7]);
        (0..frames)
            .map(|_| {
                w.step(fx(DT));
                w.get_body(body).expect("body").position
            })
            .collect::<Vec<_>>()
    };
    let one = bits(1);
    assert_eq!(one, bits(4), "iterations 4 differs from 1");
    assert_eq!(one, bits(16), "iterations 16 differs from 1");
}

/// (4) Two springs of different `ω` in one world each follow their own
/// closed form.
#[test]
fn springs_of_different_frequency_share_a_world() {
    let substeps = 4;
    let h = DT / substeps as f64;
    let frames = 90;
    let mut w = world(config(substeps, 1, 1.0));
    let slow = add_spring_body(&mut w, 3.0, 5.0, 1.0, [0.0, 1.0, 0.0], [0.0; 3]);
    let fast = add_spring_body(&mut w, 0.5, 20.0, 1.0, [0.0, 0.0, -0.6], [0.0; 3]);
    let ref_slow = reference(1.0, 0.0, 5.0, 1.0, substeps, frames, 1.0);
    let ref_fast = reference(-0.6, 0.0, 20.0, 1.0, substeps, frames, 1.0);
    for n in 0..frames {
        w.step(fx(DT));
        let t = (n + 1) as f64 * DT;
        let ps = pos(&w, slow);
        let pf = pos(&w, fast);
        assert!((ps[1] - closed(t, 1.0, 0.0, 5.0)).abs() <= bound(1.0, 0.0, 5.0, h));
        assert!((pf[2] - closed(t, -0.6, 0.0, 20.0)).abs() <= bound(0.6, 0.0, 20.0, h));
        assert!((ps[1] - ref_slow[n]).abs() < 1e-9, "slow t {t:.4}");
        assert!((pf[2] - ref_fast[n]).abs() < 1e-9, "fast t {t:.4}");
    }
}

/// (5) With the default `SolverConfig::damping = 0.99` and 4 substeps the
/// result is the same substep law with the velocity scaled by 0.99 after every
/// frame, and differs measurably from the undamped world.
#[test]
fn global_damping_multiplies_on_top_of_the_spring_law() {
    let (m, omega, x0) = (4.0, 1.0 / 0.12, 0.5);
    let substeps = 4;
    let frames = 90;
    let mut w = world(config(substeps, 1, 0.99));
    let body = add_spring_body(&mut w, m, omega, 1.0, [x0, 0.0, 0.0], [0.0; 3]);
    let got = run(&mut w, body, frames);
    let damped = reference(x0, 0.0, omega, 1.0, substeps, frames, 0.99);
    let plain = reference(x0, 0.0, omega, 1.0, substeps, frames, 1.0);
    let mut max_gap: f64 = 0.0;
    for n in 0..frames {
        assert!(
            (got[n][0] - damped[n]).abs() < 1e-9,
            "frame {n}: {} vs damped law {}",
            got[n][0],
            damped[n]
        );
        max_gap = max_gap.max((damped[n] - plain[n]).abs());
    }
    assert!(
        max_gap > 1e-4,
        "damping 0.99 left the result unchanged ({max_gap:.3e})"
    );
}

/// (6) Two runs of the same scene (offset anchors, rotating body, two springs)
/// produce the same bits.
#[test]
fn replay_is_bit_identical() {
    let scene = || {
        let mut w = world(SolverConfig::default());
        let a = w.add_body(RigidBody::new_static(v3(0.0, 2.0, 0.0)));
        let mut b = RigidBody::new_dynamic(v3(0.7, 1.0, 0.2), fx(1.3));
        b.angular_velocity = v3(0.4, -1.0, 2.0);
        let b = w.add_body(b);
        let c = w.add_body(RigidBody::new_dynamic(v3(-0.5, 0.4, 0.0), fx(0.7)));
        w.add_joint(Joint::Spring(SpringJoint::new(
            a,
            b,
            Vec3Fix::ZERO,
            v3(0.1, 0.2, -0.1),
            Fix128::ZERO,
            fx(30.0),
            fx(4.0),
        )));
        w.add_joint(Joint::Spring(SpringJoint::new(
            b,
            c,
            v3(-0.1, 0.0, 0.1),
            Vec3Fix::ZERO,
            fx(0.5),
            fx(12.0),
            fx(1.0),
        )));
        let mut trace = Vec::new();
        for _ in 0..200 {
            w.step(fx(DT));
            for i in [b, c] {
                let body = w.get_body(i).expect("body");
                trace.push((body.position, body.rotation, body.velocity));
            }
        }
        trace
    };
    assert_eq!(scene(), scene());
}

/// (7a) The TGS backend applies the same law: the closed form within the
/// bound and the discrete law within `1e-9`.
#[test]
fn tgs_backend_follows_the_same_law() {
    let (m, omega, x0, v0) = (2.0, 8.0, 1.0, 2.0);
    let substeps = 4;
    let h = DT / substeps as f64;
    let frames = 90;
    let mut w = world(SolverConfig {
        solver_backend: SolverBackend::Tgs,
        ..config(substeps, 1, 1.0)
    });
    let body = add_spring_body(&mut w, m, omega, 1.0, [x0, 0.0, 0.0], [0.0, v0, 0.0]);
    let refx = reference(x0, 0.0, omega, 1.0, substeps, frames, 1.0);
    let refy = reference(0.0, v0, omega, 1.0, substeps, frames, 1.0);
    for n in 0..frames {
        w.step(fx(DT));
        let p = pos(&w, body);
        let t = (n + 1) as f64 * DT;
        assert!((p[0] - closed(t, x0, 0.0, omega)).abs() <= bound(x0, 0.0, omega, h));
        assert!((p[1] - closed(t, 0.0, v0, omega)).abs() <= bound(0.0, v0, omega, h));
        assert!(
            (p[0] - refx[n]).abs() < 1e-9,
            "TGS x t {t:.4}: {} vs {}",
            p[0],
            refx[n]
        );
        assert!(
            (p[1] - refy[n]).abs() < 1e-9,
            "TGS y t {t:.4}: {} vs {}",
            p[1],
            refy[n]
        );
    }
}

/// (7b) `step_parallel` applies the same law and matches `step` bit for bit
/// (a single joint, so there is no batch reordering).
#[cfg(feature = "parallel")]
#[test]
fn parallel_step_follows_the_same_law() {
    let (m, omega, x0, v0) = (2.0, 8.0, 1.0, 2.0);
    let substeps = 4;
    let frames = 90;
    let make = || {
        let mut w = world(config(substeps, 1, 1.0));
        let body = add_spring_body(&mut w, m, omega, 1.0, [x0, 0.0, 0.0], [0.0, v0, 0.0]);
        (w, body)
    };
    let (mut seq, b0) = make();
    let (mut par, b1) = make();
    let refx = reference(x0, 0.0, omega, 1.0, substeps, frames, 1.0);
    let refy = reference(0.0, v0, omega, 1.0, substeps, frames, 1.0);
    for n in 0..frames {
        seq.step(fx(DT));
        par.step_parallel(fx(DT));
        let p = pos(&par, b1);
        assert_eq!(
            seq.get_body(b0).expect("body").position,
            par.get_body(b1).expect("body").position,
            "frame {n}"
        );
        assert!((p[0] - refx[n]).abs() < 1e-9, "parallel x frame {n}");
        assert!((p[1] - refy[n]).abs() < 1e-9, "parallel y frame {n}");
    }
}

/// A hanging body settles at the static extension `m g / k` exactly, for
/// rest length 0 and rest length > 0 and every backend (an explicit damping
/// term would leave it short by `c g h / k`).
#[test]
fn hanging_body_settles_at_the_static_extension() {
    let (m, k, g): (f64, f64, f64) = (2.0, 50.0, 10.0);
    let c = 2.0 * sqrt64(k * m);
    for rest in [0.0, 0.5] {
        for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
            let mut w = world(SolverConfig {
                gravity: v3(0.0, -g, 0.0),
                solver_backend: backend,
                ..config(4, 1, 1.0)
            });
            let top = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
            let body = w.add_body(RigidBody::new_dynamic(v3(0.0, -rest - 0.1, 0.0), fx(m)));
            w.add_joint(Joint::Spring(SpringJoint::new(
                top,
                body,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                fx(rest),
                fx(k),
                fx(c),
            )));
            for _ in 0..600 {
                w.step(fx(DT));
            }
            let y = pos(&w, body)[1];
            let expected = -(rest + m * g / k);
            assert!(
                (y - expected).abs() < 1e-9,
                "rest {rest} {backend:?}: y {y} vs -(L + m g / k) = {expected}"
            );
        }
    }
}

/// The damper acts on the relative velocity: a body held by a rest-length-0
/// spring behind a kinematic anchor moving at constant velocity `V` closes the
/// gap with the closed form of `e = x − x_anchor` (`e0`, `e'0 = −V`) and then
/// follows without lag (an absolute damper would lag by `c V / k`).
#[test]
fn damper_acts_on_relative_velocity_of_a_moving_anchor() {
    let (m, omega, speed) = (5.0, 1.0 / 0.12, 3.0);
    let substeps = 4;
    let h = DT / substeps as f64;
    let mut w = world(config(substeps, 1, 1.0));
    let anchor = w.add_body(RigidBody::new_kinematic(Vec3Fix::ZERO));
    let body = w.add_body(RigidBody::new_dynamic(v3(0.0, 0.4, 0.0), fx(m)));
    w.add_joint(Joint::Spring(SpringJoint::new(
        anchor,
        body,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::ZERO,
        fx(m * omega * omega),
        fx(2.0 * m * omega),
    )));
    for n in 0..120 {
        let t = (n + 1) as f64 * DT;
        w.get_body_mut(anchor)
            .expect("anchor")
            .set_kinematic_target(v3(speed * t, 0.0, 0.0), QuatFix::IDENTITY);
        w.step(fx(DT));
        let p = pos(&w, body);
        let a = pos(&w, anchor);
        let ex = p[0] - a[0];
        let ey = p[1] - a[1];
        assert!(
            (ex - closed(t, 0.0, -speed, omega)).abs() <= bound(0.0, speed, omega, h),
            "t {t:.4}: e_x {ex}"
        );
        assert!(
            (ey - closed(t, 0.4, 0.0, omega)).abs() <= bound(0.4, 0.0, omega, h),
            "t {t:.4}: e_y {ey}"
        );
    }
    let lag = pos(&w, anchor)[0] - pos(&w, body)[0];
    assert!(
        lag.abs() < 1e-3,
        "steady lag {lag} (absolute damping: c V / k = {})",
        2.0 * speed / omega
    );
}

/// Rest length > 0: the dashpot acts along the spring only, so a body swinging
/// round the anchor at the rest length keeps (almost all of) its angular
/// momentum; an isotropic damper of the same `c` would remove it as `e^(−c t / m)`.
#[test]
fn positive_rest_length_damps_along_the_axis_only() {
    let (m, omega, rest, speed) = (1.0, 8.0, 1.0, 2.0);
    let mut w = world(config(4, 1, 1.0));
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let mut b = RigidBody::new_dynamic(v3(rest, 0.0, 0.0), fx(m));
    b.velocity = v3(0.0, speed, 0.0);
    let body = w.add_body(b);
    w.add_joint(Joint::Spring(SpringJoint::new(
        anchor,
        body,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        fx(rest),
        fx(m * omega * omega),
        fx(2.0 * m * omega),
    )));
    let l0 = m * rest * speed;
    for _ in 0..60 {
        w.step(fx(DT));
    }
    let p = pos(&w, body);
    let v = to3(w.get_body(body).expect("body").velocity);
    let l = m * (p[0] * v[1] - p[1] * v[0]);
    assert!(l > 0.9 * l0, "angular momentum {l:.4} of {l0} after 1 s");
}

/// `break_force` still breaks the joint on the spring force `|k (|d| − L)|`.
#[allow(deprecated)] // pins the separation-based solve_joints_breakable
#[test]
fn break_force_is_kept() {
    let o = Vec3Fix::ZERO;
    let mut bodies = vec![
        RigidBody::new_static(o),
        RigidBody::new_dynamic(v3(2.0, 0.0, 0.0), fx(1.0)),
    ];
    let holds = Joint::Spring(
        SpringJoint::new(0, 1, o, o, Fix128::ZERO, fx(3.0), fx(1.0)).with_break_force(fx(6.5)),
    );
    let breaks = Joint::Spring(
        SpringJoint::new(0, 1, o, o, Fix128::ZERO, fx(3.0), fx(1.0)).with_break_force(fx(5.5)),
    );
    assert!(solve_joints_breakable(&[holds], &mut bodies, fx(DT)).is_empty());
    let mut bodies = vec![
        RigidBody::new_static(o),
        RigidBody::new_dynamic(v3(2.0, 0.0, 0.0), fx(1.0)),
    ];
    assert_eq!(
        solve_joints_breakable(&[breaks], &mut bodies, fx(DT)),
        vec![0]
    );
    assert_eq!(
        bodies[1].position,
        v3(2.0, 0.0, 0.0),
        "a broken joint is not solved"
    );
}

/// The relative velocity is that of the anchors, `v + ω × r`: a spinning body
/// whose offset anchor sits on a static anchor (spring force 0) is damped by
/// `c |ω × r|`. One solve with `h = 1/60`, `k = 0`, `c = 3`, `ω = (0, 0, 2)`,
/// `r_b = (0.5, 0, 0)`: `u = (0, 1, 0)`, `w = 1 + |r × n|² I⁻¹ = 1.25` and
/// `λ = h² c |u| / (1 + c h w)`; B moves `−λ` along `y` and turns `−0.5 λ` about `z`.
#[allow(deprecated)] // pins the separation-based solve_joints_breakable
#[test]
fn damper_reads_the_anchor_velocity_including_spin() {
    let h: f64 = 1.0 / 60.0;
    let c = 3.0;
    let mut b = RigidBody::new_dynamic(v3(-0.5, 0.0, 0.0), fx(1.0));
    b.inv_inertia = v3(1.0, 1.0, 1.0);
    b.angular_velocity = v3(0.0, 0.0, 2.0);
    let mut bodies = vec![RigidBody::new_static(Vec3Fix::ZERO), b];
    let j = Joint::Spring(SpringJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        v3(0.5, 0.0, 0.0),
        Fix128::ZERO,
        Fix128::ZERO,
        fx(c),
    ));
    assert!(solve_joints_breakable(&[j], &mut bodies, fx(h)).is_empty());
    let lambda = h * h * c / (1.0 + c * h * 1.25);
    let p = to3(bodies[1].position);
    assert!((p[0] + 0.5).abs() < 1e-12, "x {}", p[0]);
    assert!(
        (p[1] + lambda).abs() < 1e-12,
        "y {} vs -lambda {}",
        p[1],
        -lambda
    );
    let q = bodies[1].rotation;
    let twist = 2.0 * atan2_64(q.z.to_f64(), q.w.to_f64());
    assert!(
        (twist + 0.5 * lambda).abs() < 1e-12,
        "twist {twist} vs {}",
        -0.5 * lambda
    );
}
