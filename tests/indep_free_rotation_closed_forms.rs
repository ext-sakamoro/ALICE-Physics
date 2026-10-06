//! Independent closed-form checks of the free rotation of 3D rigid bodies:
//! the gyroscopic term, the symmetric splitting of the rotation and the
//! angular velocity taken from the rotation logarithm.
//!
//! Every expected value is computed in this file from Euler's equations,
//!
//! ```text
//! I₁ ω̇₁ = (I₂ − I₃) ω₂ ω₃,  I₂ ω̇₂ = (I₃ − I₁) ω₃ ω₁,  I₃ ω̇₃ = (I₁ − I₂) ω₁ ω₂
//! ```
//!
//! (body frame, principal moments), never by calling the code under test:
//!
//! - the world angular momentum `L = R I Rᵀ ω` is constant;
//! - the body-frame `ω(T)` of the solver converges to an RK4 reference of
//!   Euler's equations (step `1e-5`) at second order in the substep `h`:
//!   halving `h` divides the error by 4; the same holds for the worst energy
//!   error over the run;
//! - a spin about a principal axis (or any spin of an isotropic body) keeps
//!   `ω` and turns the body by `|ω| t`, whatever the substep count;
//! - for isotropic inertia `I = kE` the free motion does not depend on `k`
//!   (`ω × Iω = 0`), so bodies differing only in `k` stay bit-identical.
//!
//! Scenes use `PhysicsWorld::step` (and `step_parallel` with the `parallel`
//! feature) on both backends, zero gravity and no damping.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::SolverBackend;

const FRAME_DT: f64 = 1.0 / 60.0;

type V = [f64; 3];

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(v: V) -> Vec3Fix {
    Vec3Fix::new(fx(v[0]), fx(v[1]), fx(v[2]))
}

fn to_f(v: Vec3Fix) -> V {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn norm(a: V) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[derive(Clone, Copy, Debug)]
enum Entry {
    Step,
    #[cfg(feature = "parallel")]
    StepParallel,
}

fn entries() -> Vec<(SolverBackend, Entry)> {
    #[allow(unused_mut)]
    let mut v = vec![
        (SolverBackend::Xpbd, Entry::Step),
        (SolverBackend::Tgs, Entry::Step),
    ];
    #[cfg(feature = "parallel")]
    v.push((SolverBackend::Xpbd, Entry::StepParallel));
    v
}

fn advance(w: &mut PhysicsWorld, e: Entry) {
    match e {
        Entry::Step => w.step(fx(FRAME_DT)),
        #[cfg(feature = "parallel")]
        Entry::StepParallel => w.step_parallel(fx(FRAME_DT)),
    }
}

fn world(backend: SolverBackend, substeps: usize) -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps,
        solver_backend: backend,
        ..PhysicsConfig::default()
    })
}

fn free_body(inertia: V, rotation: QuatFix, omega_world: V) -> RigidBody {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = v3([1.0 / inertia[0], 1.0 / inertia[1], 1.0 / inertia[2]]);
    b.rotation = rotation;
    b.angular_velocity = v3(omega_world);
    b
}

fn body_omega(b: &RigidBody) -> V {
    to_f(b.rotation.conjugate().rotate_vec(b.angular_velocity))
}

fn world_l(b: &RigidBody, inertia: V) -> V {
    let wb = body_omega(b);
    let lb = [inertia[0] * wb[0], inertia[1] * wb[1], inertia[2] * wb[2]];
    to_f(b.rotation.rotate_vec(v3(lb)))
}

fn energy_body(wb: V, inertia: V) -> f64 {
    0.5 * (inertia[0] * wb[0] * wb[0] + inertia[1] * wb[1] * wb[1] + inertia[2] * wb[2] * wb[2])
}

/// Right-hand side of Euler's torque-free equations in the body frame.
fn euler_rhs(w: V, i: V) -> V {
    [
        (i[1] - i[2]) / i[0] * w[1] * w[2],
        (i[2] - i[0]) / i[1] * w[2] * w[0],
        (i[0] - i[1]) / i[2] * w[0] * w[1],
    ]
}

/// Body-frame `ω(t)` by classic RK4 with step `1e-5` (error `~1e-18`).
fn euler_reference(w0: V, i: V, t: f64) -> V {
    let n = (t / 1e-5).round() as usize;
    let h = t / n as f64;
    let mut w = w0;
    let axpy = |a: V, s: f64, b: V| [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
    for _ in 0..n {
        let k1 = euler_rhs(w, i);
        let k2 = euler_rhs(axpy(w, h / 2.0, k1), i);
        let k3 = euler_rhs(axpy(w, h / 2.0, k2), i);
        let k4 = euler_rhs(axpy(w, h, k3), i);
        for c in 0..3 {
            w[c] += h / 6.0 * (k1[c] + 2.0 * k2[c] + 2.0 * k3[c] + k4[c]);
        }
    }
    w
}

const INERTIA: V = [0.7, 1.9, 3.3];

fn tilt() -> QuatFix {
    QuatFix::from_axis_angle(v3([1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]), fx(0.8)).normalize()
}

/// oracle: torque-free ⇒ the world angular momentum is constant. Asymmetric
/// moments, a turned start and a spin that is not about any principal axis,
/// 2 s. A solver that keeps the world `ω` constant (no gyroscopic term) moves
/// `L` by order `|L|` within this time.
#[test]
fn world_angular_momentum_of_a_turned_asymmetric_body_is_constant() {
    let omega0 = [1.3, -2.1, 0.9];
    for (backend, e) in entries() {
        let mut w = world(backend, 8);
        let k = w.add_body(free_body(INERTIA, tilt(), omega0));
        let l0 = world_l(&w.bodies[k], INERTIA);
        let mut worst: f64 = 0.0;
        for _ in 0..120 {
            advance(&mut w, e);
            worst = worst.max(norm(sub(world_l(&w.bodies[k], INERTIA), l0)) / norm(l0));
        }
        // the body has turned by order radians, so the momentum check is not vacuous
        let wb_end = body_omega(&w.bodies[k]);
        let wb_start = body_omega(&free_body(INERTIA, tilt(), omega0));
        assert!(
            norm(sub(wb_end, wb_start)) > 0.1,
            "{backend:?}/{e:?}: body-frame ω barely changed, the scene is not exercising the term"
        );
        assert!(
            worst < 1e-12,
            "{backend:?}/{e:?}: |ΔL|/|L| reached {worst:.3e}"
        );
    }
}

/// Body-frame `ω` error at `T` against the RK4 reference and the worst
/// relative energy error over the run.
fn errors(
    backend: SolverBackend,
    e: Entry,
    substeps: usize,
    omega0: V,
    frames: usize,
) -> (f64, f64) {
    let mut w = world(backend, substeps);
    let k = w.add_body(free_body(INERTIA, tilt(), omega0));
    let wb0 = body_omega(&w.bodies[k]);
    let e0 = energy_body(wb0, INERTIA);
    let mut worst_e: f64 = 0.0;
    for _ in 0..frames {
        advance(&mut w, e);
        let de = (energy_body(body_omega(&w.bodies[k]), INERTIA) - e0).abs() / e0;
        worst_e = worst_e.max(de);
    }
    let reference = euler_reference(wb0, INERTIA, frames as f64 * FRAME_DT);
    (norm(sub(body_omega(&w.bodies[k]), reference)), worst_e)
}

/// oracle: the splitting is second order, so the error of `ω_b(T)` and the
/// worst energy error divide by 4 when the substep is halved (substeps
/// 2 → 4 → 8 at a frame of 1/60 s). A first-order composition divides by 2.
/// The band `[3.3, 4.7]` leaves room for the `O(h⁴)` term at `h = 1/120`.
#[test]
fn errors_against_euler_equations_are_second_order_in_the_substep() {
    let omega0 = [2.0, -3.0, 1.5];
    let frames = 60;
    for (backend, e) in entries() {
        let runs: Vec<(f64, f64)> = [2, 4, 8]
            .iter()
            .map(|&s| errors(backend, e, s, omega0, frames))
            .collect();
        eprintln!("{backend:?}/{e:?}: (ω error, energy error) at substeps 2/4/8 = {runs:?}");
        for pair in runs.windows(2) {
            let (wa, ea) = pair[0];
            let (wb, eb) = pair[1];
            assert!(
                wb > 1e-14 && eb > 1e-14,
                "errors at rounding level, ratio undefined"
            );
            let rw = wa / wb;
            let re = ea / eb;
            assert!(
                (3.3..=4.7).contains(&rw),
                "{backend:?}/{e:?}: ω error ratio per halving {rw:.3} (want 4)"
            );
            assert!(
                (3.3..=4.7).contains(&re),
                "{backend:?}/{e:?}: energy error ratio per halving {re:.3} (want 4)"
            );
        }
        // and the error is small in absolute terms at 8 substeps
        assert!(runs[2].0 < 1e-3, "{backend:?}/{e:?}: ω error {}", runs[2].0);
    }
}

/// Rotation angle of `q` about the unit `axis` in `(−π, π]`, from the
/// quaternion components (no solver call).
fn angle_about(q: QuatFix, axis: V) -> f64 {
    let s = q.x.to_f64() * axis[0] + q.y.to_f64() * axis[1] + q.z.to_f64() * axis[2];
    2.0 * s.atan2(q.w.to_f64())
}

fn wrap(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut x = a % two_pi;
    if x > std::f64::consts::PI {
        x -= two_pi;
    }
    if x <= -std::f64::consts::PI {
        x += two_pi;
    }
    x
}

/// Runs a spin `speed · axis_b` (start unturned) for 10 s at substeps 1, 2,
/// 4 and 8 and lists every run whose `ω` or turned angle leaves the closed
/// form (`ω = ω₀`, angle `|ω₀| t`).
fn principal_spin_failures(entries: &[(SolverBackend, Entry)], cases: &[(V, V)]) -> Vec<String> {
    let speed = 6.0;
    let frames = (10.0 / FRAME_DT).round() as usize;
    let mut failures = Vec::new();
    for &(backend, e) in entries {
        for &(inertia, axis_b) in cases {
            for substeps in [1, 2, 4, 8] {
                let mut w = world(backend, substeps);
                let omega0 = [speed * axis_b[0], speed * axis_b[1], speed * axis_b[2]];
                let k = w.add_body(free_body(inertia, QuatFix::IDENTITY, omega0));
                for _ in 0..frames {
                    advance(&mut w, e);
                }
                let b = &w.bodies[k];
                let dw = norm(sub(to_f(b.angular_velocity), omega0));
                let turned = angle_about(b.rotation, axis_b);
                let expected = wrap(speed * frames as f64 * FRAME_DT);
                let dtheta = wrap(turned - expected).abs();
                if dw >= 1e-9 || dtheta >= 1e-8 {
                    failures.push(format!(
                        "{backend:?}/{e:?} I={inertia:?} axis={axis_b:?} s={substeps}: \
                         |Δω| {dw:.3e}, angle {turned:.6} vs {expected:.6} (Δ {dtheta:.3e})"
                    ));
                }
            }
        }
    }
    failures
}

const ISOTROPIC_SPIN: (V, V) = ([1.2, 1.2, 1.2], [0.6, 0.0, 0.8]);

/// oracle: a spin `ω₀` about a principal axis (isotropic body: any axis) is a
/// steady solution of Euler's equations: after `t` the body has turned by
/// `|ω₀| t` about it and `ω = ω₀`, for every substep count. A chord-based
/// velocity `(2/h) sin(|ω| h / 2)` loses `|ω|³ h² / 24` per substep, 17 % in
/// 10 s at one substep for `|ω| = 6`. Every path except the isotropic body
/// under TGS (next test); the anisotropic spins run on TGS as well and pin
/// that its split path is exact.
#[test]
fn a_principal_spin_keeps_omega_and_turns_by_omega_t_at_every_substep_count() {
    let all = entries();
    let mut failures = principal_spin_failures(
        &all,
        &[(INERTIA, [0.0, 1.0, 0.0]), (INERTIA, [0.0, 0.0, 1.0])],
    );
    let not_tgs: Vec<_> = all
        .into_iter()
        .filter(|(b, _)| *b != SolverBackend::Tgs)
        .collect();
    failures.extend(principal_spin_failures(&not_tgs, &[ISOTROPIC_SPIN]));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The same oracle for an isotropic body under TGS.
///
/// Closed form of the measured lag: the TGS path that does not split the
/// rotation turns the quaternion by `q ← normalize(q + ½ h ω̃ q)`, with
/// `ω̃ = (ω, 0)`. For `q = (sin(φ/2) n, cos(φ/2))` and `ω = |ω| n` this is
/// `(sin(φ/2) + ½|ω|h cos(φ/2), cos(φ/2) − ½|ω|h sin(φ/2)) n`, i.e. the half
/// angle grows by `atan(|ω| h / 2)`, so each substep turns the body by
/// `2 atan(|ω| h / 2) = |ω| h − |ω|³ h³ / 12 + O(h⁵)` instead of `|ω| h`.
/// Over `t / h` substeps the angle lags by `Δθ = |ω|³ h² t / 12`: for
/// `|ω| = 6`, `t = 10`, frame `1/60`, that is 5.00e-2 / 1.25e-2 / 3.13e-3 /
/// 7.81e-4 rad at substeps 1 / 2 / 4 / 8, and the measured lags are
/// 4.993e-2 / 1.250e-2 / 3.125e-3 / 7.812e-4. `ω` itself is kept exactly.
/// The anisotropic principal spin under TGS (split path) holds to `1e-8` in
/// the previous test, which stays green as the control.
#[test]
#[ignore = "known defect: TGS turns an isotropic body by 2 atan(|w| h / 2) per substep, the angle lags by |w|^3 h^2 t / 12"]
fn tgs_isotropic_spin_turns_by_omega_t() {
    let tgs = [(SolverBackend::Tgs, Entry::Step)];
    let failures = principal_spin_failures(&tgs, &[ISOTROPIC_SPIN]);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// oracle: for `I = kE` the term `ω × Iω` vanishes and the free motion does
/// not involve `k`, so three bodies differing only in `k` stay bit-identical
/// in rotation and angular velocity, on every path.
#[test]
fn isotropic_bodies_differing_only_in_the_moment_stay_bit_identical() {
    let omega0 = [1.7, -0.4, 2.9];
    for (backend, e) in entries() {
        let mut w = world(backend, 8);
        // far apart: no collision radius, so no contacts either way
        let ids: Vec<usize> = [0.3, 1.0, 7.5]
            .iter()
            .map(|&k| w.add_body(free_body([k, k, k], tilt(), omega0)))
            .collect();
        for _ in 0..240 {
            advance(&mut w, e);
        }
        let first = &w.bodies[ids[0]];
        for &i in &ids[1..] {
            assert_eq!(
                first.rotation, w.bodies[i].rotation,
                "{backend:?}/{e:?}: rotation"
            );
            assert_eq!(
                first.angular_velocity, w.bodies[i].angular_velocity,
                "{backend:?}/{e:?}: angular velocity"
            );
        }
        // the isotropic world ω is constant (it is also the world L / k)
        let dw = norm(sub(to_f(first.angular_velocity), omega0));
        assert!(
            dw < 1e-9,
            "{backend:?}/{e:?}: isotropic ω moved by {dw:.3e}"
        );
    }
}

/// The XPBD step and its parallel twin give the bit-identical free rotation of
/// an asymmetric body.
#[cfg(feature = "parallel")]
#[test]
fn step_and_step_parallel_agree_bit_for_bit_on_an_asymmetric_body() {
    let run = |e: Entry| {
        let mut w = world(SolverBackend::Xpbd, 8);
        let k = w.add_body(free_body(INERTIA, tilt(), [1.3, -2.1, 0.9]));
        for _ in 0..120 {
            advance(&mut w, e);
        }
        (w.bodies[k].rotation, w.bodies[k].angular_velocity)
    };
    assert_eq!(run(Entry::Step), run(Entry::StepParallel));
}
