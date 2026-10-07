//! Momentum of rigid bodies and a [`DragMedium`], checked on scenes the
//! medium's own tests do not use: five bodies with asymmetric mass ratios,
//! a light medium whose velocity swings while the bodies pull it, frame
//! widths `1/50`, `1/48`, `1/30` with `5`, `6`, `2` substeps, and the
//! batched path.
//!
//! # The conserved quantity
//!
//! `T = P + Σ_i q(v_i, inv_mass_i)` with `q(a, b) = a / b` the [`Fix128`]
//! division (truncation toward zero, error below one raw unit `2⁻⁶⁴`), `P`
//! the medium momentum. This is the sum the module documentation of
//! `coupling_medium` keeps. It is computed here with `/` directly and, as a
//! cross-check, with [`DragMedium::total_momentum`] on a copy of the medium
//! rebuilt from the world's payload.
//!
//! # Bounds
//!
//! * **Powers of two, TGS.** With `inv_mass_i = 2^(−k_i)`, `k_i ≥ 0`, every
//!   `q` is exact and TGS keeps the integrated velocity: `T` keeps its bit
//!   pattern in every frame (`assert_eq!` on the raw `hi`, `lo`).
//! * **Any mass, TGS.** In one exchange body `i` gets `dv`, the medium gives
//!   up `q(dv)`, and `T` changes by `q(v + dv) − q(v) − q(dv)` for that
//!   body. The exact quotients add up exactly; each of the three truncations
//!   is in `(−1, 1)` raw units, and the result is an integer number of raw
//!   units, so the change is at most `2` raw units per body, component and
//!   substep: `|ΔT| ≤ 2 · N · n · 2⁻⁶⁴` after `n` substeps of `N` bodies.
//! * **XPBD and the batched path.** XPBD re-derives `v` from the position
//!   change, losing at most `2⁻⁶⁴ (1/h + |v| h + 2)` of velocity per body,
//!   component and substep (module documentation of `coupling_medium`); in
//!   `T` that is `q` of the loss, at most `m_i` times it plus one raw unit,
//!   on top of the `2` raw units of the exchange:
//!   `|ΔT| ≤ n · Σ_i (m_i (1/h + V h + 2) + 3) · 2⁻⁶⁴` with `|v| ≤ V`. The
//!   scenes stay in the no-overshoot regime (`c_i h / m_i ≤ 1`,
//!   `Σ c_i h / M ≤ 1`), so `V` is the largest initial speed component plus
//!   one.

use alice_physics::coupling_medium::{DragMedium, MEDIUM_OBS_MOMENTUM, MEDIUM_OBS_VELOCITY};
use alice_physics::sleeping::SleepConfig;
use alice_physics::world_participant::{Observed, Participant};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix};

const RAW: f64 = 1.0 / 18_446_744_073_709_551_616.0; // 2⁻⁶⁴

/// `(num, den, substeps)`: none of them is a case of the medium's own tests.
const CASES: [(i64, i64, usize); 3] = [(1, 50, 5), (1, 48, 6), (1, 30, 2)];

const FRAMES: usize = 240;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn bits(v: Vec3Fix) -> [(i64, u64); 3] {
    [(v.x.hi, v.x.lo), (v.y.hi, v.y.lo), (v.z.hi, v.z.lo)]
}

/// Free bodies (no gravity, damping, sleep or contact).
fn bare_world(backend: SolverBackend, substeps: usize) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps,
        iterations: 3,
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    w
}

/// A scene: body masses, velocities, couplings, medium mass and velocity.
struct Scene {
    masses: &'static [f64],
    vel: &'static [[f64; 3]],
    coeff: &'static [(usize, f64)],
    medium_mass: f64,
    medium_vel: [f64; 3],
}

/// Masses `4 : 32 : 1 : 2 : 64` (`1 : 8 : 0.25 : …`, powers of two `≥ 1`):
/// the conditions of exact conservation.
const DYADIC: Scene = Scene {
    masses: &[4.0, 32.0, 1.0, 2.0, 64.0],
    vel: &[
        [1.5, -0.25, 0.75],
        [-0.5, 0.125, -1.0],
        [2.75, 1.0, -2.0],
        [-3.0, -1.5, 0.5],
        [0.25, 0.5, 0.0],
    ],
    // Coupled out of order, one body (3) left uncoupled.
    coeff: &[(4, 23.0), (1, 11.0), (0, 2.5), (2, 0.9)],
    medium_mass: 1.5,
    medium_vel: [0.75, -1.25, 0.4],
};

/// Masses `1 : 7 : 0.3 : 2.5 : 7` (not powers of two).
const RATIO: Scene = Scene {
    masses: &[1.0, 7.0, 0.3, 2.5, 7.0],
    vel: &[
        [2.0, -1.0, 0.5],
        [-0.75, 0.3, -0.2],
        [3.1, 2.2, -2.9],
        [-1.7, -0.4, 1.3],
        [0.6, -0.9, 0.1],
    ],
    coeff: &[(2, 3.3), (0, 1.9), (3, 6.1), (1, 13.0), (4, 9.7)],
    medium_mass: 0.6,
    medium_vel: [-1.1, 0.8, 2.0],
};

/// Largest initial speed component of every scene, plus one.
const V_BOUND: f64 = 4.1;

fn medium(s: &Scene) -> DragMedium {
    let mut m = DragMedium::new(
        fx(s.medium_mass),
        v3(s.medium_vel[0], s.medium_vel[1], s.medium_vel[2]),
    )
    .expect("medium");
    for &(b, c) in s.coeff {
        m.couple(b, fx(c)).expect("couple");
    }
    m
}

fn build(s: &Scene, backend: SolverBackend, substeps: usize) -> PhysicsWorld {
    let mut w = bare_world(backend, substeps);
    for (k, (&m, v)) in s.masses.iter().zip(s.vel).enumerate() {
        let mut b = RigidBody::new_dynamic(
            v3(37.0 * k as f64, -150.0 * k as f64, 90.0 * (k % 2) as f64),
            fx(m),
        );
        b.velocity = v3(v[0], v[1], v[2]);
        w.add_body(b);
    }
    w.add_participant(Box::new(medium(s))).expect("register");
    w
}

fn channel(w: &PhysicsWorld, ch: u32) -> Vec3Fix {
    let Some(Observed::Exact(sink)) = w.observe_participant(0) else {
        panic!("medium observation undecided");
    };
    let get = |c: u32| {
        sink.values()
            .iter()
            .find(|(k, _)| *k == c)
            .map(|(_, v)| *v)
            .expect("channel")
    };
    Vec3Fix::new(get(ch), get(ch + 1), get(ch + 2))
}

/// `T` of the module documentation, computed here with `/`; only coupled
/// bodies take part (the uncoupled one keeps its own momentum anyway).
fn total(w: &PhysicsWorld, s: &Scene) -> Vec3Fix {
    let mut t = channel(w, MEDIUM_OBS_MOMENTUM);
    for &(b, _) in s.coeff {
        let body = &w.bodies[b];
        t = t + Vec3Fix::new(
            body.velocity.x / body.inv_mass,
            body.velocity.y / body.inv_mass,
            body.velocity.z / body.inv_mass,
        );
    }
    t
}

/// The participant's own sum, on a medium rebuilt from the world's payload.
fn total_by_medium(w: &PhysicsWorld, s: &Scene) -> Vec3Fix {
    let mut m = medium(s);
    let payload = w.participant_state(0).expect("payload");
    m.check_state(&payload).expect("own payload");
    m.read_state(&payload);
    m.total_momentum(&w.bodies).expect("in range")
}

fn max_raw(a: Vec3Fix, b: Vec3Fix) -> f64 {
    let d = a - b;
    let f = |x: Fix128| x.to_f64().abs();
    f(d.x).max(f(d.y)).max(f(d.z)) / RAW
}

fn xpbd_bound_raw(s: &Scene, h: f64, substeps_run: usize) -> f64 {
    let per: f64 = s
        .coeff
        .iter()
        .map(|&(b, _)| s.masses[b] * (1.0 / h + V_BOUND * h + 2.0) + 3.0)
        .sum();
    substeps_run as f64 * per
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Step,
    #[cfg(feature = "parallel")]
    Parallel,
}

fn advance(w: &mut PhysicsWorld, path: Path, dt: Fix128) {
    match path {
        Path::Step => w.try_step(dt).expect("step"),
        #[cfg(feature = "parallel")]
        Path::Parallel => w.try_step_parallel(dt).expect("step"),
    }
}

/// The no-overshoot conditions hold for every case (they bound `V`).
#[test]
fn every_scene_stays_in_the_no_overshoot_regime() {
    for s in [&DYADIC, &RATIO] {
        for (num, den, substeps) in CASES {
            let h = num as f64 / den as f64 / substeps as f64;
            let sum_c: f64 = s.coeff.iter().map(|&(_, c)| c).sum();
            assert!(sum_c * h / s.medium_mass <= 1.0, "{num}/{den} s{substeps}");
            for &(b, c) in s.coeff {
                assert!(
                    c * h / s.masses[b] <= 1.0,
                    "{num}/{den} s{substeps} body {b}"
                );
            }
        }
    }
}

/// Powers of two on TGS: `T` keeps its raw bits in every one of 240 frames,
/// the participant's own sum agrees with it, and the medium velocity really
/// changes over the run (it is not conservation by standing still).
#[test]
fn dyadic_masses_conserve_the_raw_bits_on_tgs() {
    for (num, den, substeps) in CASES {
        let dt = Fix128::from_ratio(num, den);
        let mut w = build(&DYADIC, SolverBackend::Tgs, substeps);
        let start = total(&w, &DYADIC);
        assert_eq!(total_by_medium(&w, &DYADIC), start);
        let u0 = channel(&w, MEDIUM_OBS_VELOCITY);
        let mut u_seen_x = (u0.x.to_f64(), u0.x.to_f64());
        for frame in 1..=FRAMES {
            advance(&mut w, Path::Step, dt);
            let now = total(&w, &DYADIC);
            assert_eq!(
                bits(now),
                bits(start),
                "TGS {num}/{den} s{substeps} frame {frame}"
            );
            assert_eq!(total_by_medium(&w, &DYADIC), now, "frame {frame}");
            let ux = channel(&w, MEDIUM_OBS_VELOCITY).x.to_f64();
            u_seen_x = (u_seen_x.0.min(ux), u_seen_x.1.max(ux));
        }
        assert!(
            u_seen_x.1 - u_seen_x.0 > 0.5,
            "{num}/{den}: medium velocity barely moved {u_seen_x:?}"
        );
        // The uncoupled body 3 kept its velocity exactly.
        assert_eq!(w.bodies[3].velocity, v3(-3.0, -1.5, 0.5));
    }
}

/// Masses `1 : 7 : 0.3 : 2.5 : 7` on TGS: within `2 · N · n` raw units.
#[test]
fn ratio_masses_stay_within_the_quotient_bound_on_tgs() {
    let n_bodies = RATIO.coeff.len() as f64;
    for (num, den, substeps) in CASES {
        let dt = Fix128::from_ratio(num, den);
        let mut w = build(&RATIO, SolverBackend::Tgs, substeps);
        let start = total(&w, &RATIO);
        let mut worst = 0.0f64;
        let mut changed = 0usize;
        for frame in 1..=FRAMES {
            advance(&mut w, Path::Step, dt);
            let now = total(&w, &RATIO);
            assert_eq!(total_by_medium(&w, &RATIO), now, "frame {frame}");
            let d = max_raw(now, start);
            if d > 0.0 {
                changed += 1;
            }
            let bound = 2.0 * n_bodies * (frame * substeps) as f64;
            worst = worst.max(d / bound);
            assert!(
                d <= bound,
                "TGS {num}/{den} s{substeps} frame {frame}: {d} raw > {bound}"
            );
        }
        println!(
            "ratio TGS {num}/{den} s{substeps}: worst |ΔT| / bound = {worst:.4}, \
             frames with ΔT ≠ 0: {changed}/{FRAMES}"
        );
    }
}

fn check_xpbd_bound(s: &Scene, backend: SolverBackend, path: Path, label: &str) {
    for (num, den, substeps) in CASES {
        let dt = Fix128::from_ratio(num, den);
        let h = num as f64 / den as f64 / substeps as f64;
        let mut w = build(s, backend, substeps);
        let start = total(&w, s);
        let mut worst = 0.0f64;
        let mut worst_raw = 0.0f64;
        for frame in 1..=FRAMES {
            advance(&mut w, path, dt);
            let now = total(&w, s);
            let d = max_raw(now, start);
            let bound = xpbd_bound_raw(s, h, frame * substeps);
            worst = worst.max(d / bound);
            worst_raw = worst_raw.max(d);
            assert!(
                d <= bound,
                "{label} {backend:?} {path:?} {num}/{den} s{substeps} frame {frame}: \
                 {d} raw > {bound}"
            );
        }
        println!(
            "{label} {backend:?} {path:?} {num}/{den} s{substeps}: max |ΔT| = {worst_raw} raw \
             ({:e}), worst ratio to bound = {worst:.4}",
            worst_raw * RAW
        );
    }
}

#[test]
fn xpbd_stays_within_the_documented_bound() {
    check_xpbd_bound(&DYADIC, SolverBackend::Xpbd, Path::Step, "dyadic");
    check_xpbd_bound(&RATIO, SolverBackend::Xpbd, Path::Step, "ratio");
}

/// The batched path runs the XPBD loop whatever the backend: the XPBD bound
/// for both configured backends.
#[cfg(feature = "parallel")]
#[test]
fn the_batched_path_stays_within_the_xpbd_bound() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        check_xpbd_bound(&DYADIC, backend, Path::Parallel, "dyadic");
        check_xpbd_bound(&RATIO, backend, Path::Parallel, "ratio");
    }
}

/// The medium really exchanges on the batched path (a path that skipped the
/// participant would keep `T` trivially): after one frame the medium momentum
/// equals the one of `try_step` on an XPBD world with the same width.
#[cfg(feature = "parallel")]
#[test]
fn the_batched_path_runs_the_medium_like_the_xpbd_step() {
    for (num, den, substeps) in CASES {
        let dt = Fix128::from_ratio(num, den);
        let mut a = build(&RATIO, SolverBackend::Xpbd, substeps);
        let mut b = build(&RATIO, SolverBackend::Xpbd, substeps);
        let p0 = channel(&a, MEDIUM_OBS_MOMENTUM);
        for frame in 0..FRAMES {
            a.try_step(dt).expect("step");
            b.try_step_parallel(dt).expect("step");
            assert_eq!(
                a.participant_state(0),
                b.participant_state(0),
                "{num}/{den} frame {frame}"
            );
            for (x, y) in a.bodies.iter().zip(&b.bodies) {
                assert_eq!(x.velocity, y.velocity, "{num}/{den} frame {frame}");
            }
        }
        assert!(max_raw(channel(&a, MEDIUM_OBS_MOMENTUM), p0) * RAW > 0.1);
    }
}
