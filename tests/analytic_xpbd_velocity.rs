//! Exact velocity of free rigid bodies under the XPBD backend.
//!
//! A body that no constraint or contact moves during a substep must leave the
//! substep with exactly the velocity it was predicted with: the post-solve
//! velocity is `v = v_pred + Δx_corr / h`, `Δx_corr` being the positional
//! correction the solve applied in that substep (zero for a free body). The
//! earlier derivation `v = (x − x_prev) / h` is the same in exact arithmetic
//! but re-derives `v` from `x_prev + v·h`, whose fixed-point product truncates,
//! so the low bits of every free body's velocity were dropped each substep.
//!
//! Closed forms used below (all in `Fix128`, no tolerance):
//!
//! * no gravity, no damping, no contacts: `v_n = v_0` bit for bit after any
//!   number of frames, for any `dt` and `substeps`;
//! * linear momentum `P = Σ m_i v_i`: the masses are `2^k` (`k ≥ 0`), so
//!   `m · v` is a left shift and exact, and `P` is a sum of exact terms; with
//!   every `v_i` unchanged, `P` is unchanged bit for bit;
//! * uniform gravity `g`, gravity scale 1, no damping: each substep adds the
//!   same increment `a = (g · 1) · h` (the expression the integrator
//!   evaluates, `h = dt / substeps`), so after `n` substeps
//!   `v_n = v_0 + a + a + … + a` (`n` additions in this order, each exact
//!   because `Fix128` addition is exact when it does not overflow);
//! * the TGS backend keeps a free body's velocity untouched by its solve, so
//!   XPBD and TGS give bit-identical velocities for free bodies.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::SolverBackend;

const FRAMES: usize = 240;
const SUBSTEPS: [usize; 4] = [1, 3, 4, 8];

fn dts() -> [Fix128; 2] {
    [Fix128::from_ratio(1, 60), Fix128::from_ratio(1, 64)]
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// Initial velocities with non-dyadic components (none of them is a multiple
/// of `2^-64 / h`, so a truncating `x += v·h` loses low bits on every one).
fn initial_velocities() -> [Vec3Fix; 4] {
    [
        Vec3Fix::new(r(4, 3), r(-16, 7), r(16, 11)),
        Vec3Fix::new(r(-13, 9), r(22, 13), r(-3, 17)),
        Vec3Fix::new(r(7, 5), r(5, 19), r(-29, 23)),
        Vec3Fix::new(r(-1, 3), r(-31, 29), r(41, 37)),
    ]
}

/// Masses `2^k`, `k = 0..3`: `m · v` is exact.
fn masses() -> [Fix128; 4] {
    [
        Fix128::from_int(1),
        Fix128::from_int(2),
        Fix128::from_int(4),
        Fix128::from_int(8),
    ]
}

fn config(backend: SolverBackend, substeps: usize, gravity: Vec3Fix) -> SolverConfig {
    SolverConfig {
        substeps,
        gravity,
        // A retention of exactly 1: `v · 1` is exact in `Fix128`, so frame
        // damping does not touch the velocity.
        damping: Fix128::ONE,
        solver_backend: backend,
        ..SolverConfig::default()
    }
}

/// Four free bodies, 1000 m apart, moving apart-or-sideways slowly enough
/// that they never come within contact range in `FRAMES` frames.
fn world(backend: SolverBackend, substeps: usize, gravity: Vec3Fix) -> (PhysicsWorld, Vec<usize>) {
    let mut w = PhysicsWorld::new(config(backend, substeps, gravity));
    let vs = initial_velocities();
    let ms = masses();
    let ids = (0..4)
        .map(|i| {
            let p = Vec3Fix::new(
                Fix128::from_int(1000 * i as i64),
                Fix128::from_int(-500 * i as i64),
                Fix128::from_int(250 * i as i64),
            );
            let mut b = RigidBody::new_dynamic(p, ms[i]);
            b.velocity = vs[i];
            w.add_body(b)
        })
        .collect();
    (w, ids)
}

fn momentum(w: &PhysicsWorld, ids: &[usize]) -> Vec3Fix {
    let ms = masses();
    let mut p = Vec3Fix::ZERO;
    for (k, &i) in ids.iter().enumerate() {
        let v = w.bodies[i].velocity;
        p = p + Vec3Fix::new(v.x * ms[k], v.y * ms[k], v.z * ms[k]);
    }
    p
}

fn initial_momentum() -> Vec3Fix {
    let ms = masses();
    let mut p = Vec3Fix::ZERO;
    for (v, m) in initial_velocities().iter().zip(ms) {
        p = p + Vec3Fix::new(v.x * m, v.y * m, v.z * m);
    }
    p
}

/// A free body under XPBD keeps its velocity bit for bit, every frame.
#[test]
fn xpbd_free_body_velocity_is_bit_identical_after_every_frame() {
    for dt in dts() {
        for substeps in SUBSTEPS {
            let (mut w, ids) = world(SolverBackend::Xpbd, substeps, Vec3Fix::ZERO);
            let v0 = initial_velocities();
            for frame in 1..=FRAMES {
                w.step(dt);
                for (k, &i) in ids.iter().enumerate() {
                    assert_eq!(
                        w.bodies[i].velocity, v0[k],
                        "dt {dt:?} substeps {substeps} frame {frame} body {k}"
                    );
                }
            }
        }
    }
}

/// `Σ m v` of free bodies under XPBD is bit-identical to its initial value.
#[test]
fn xpbd_free_body_momentum_is_bit_conserved() {
    let p0 = initial_momentum();
    for dt in dts() {
        for substeps in SUBSTEPS {
            let (mut w, ids) = world(SolverBackend::Xpbd, substeps, Vec3Fix::ZERO);
            for frame in 1..=FRAMES {
                w.step(dt);
                assert_eq!(
                    momentum(&w, &ids),
                    p0,
                    "dt {dt:?} substeps {substeps} frame {frame}"
                );
            }
        }
    }
}

/// Under uniform gravity each substep adds the same increment
/// `a = (g · 1) · h`, so the velocity after `n` substeps is the `Fix128` sum
/// `v_0 + a + … + a`.
#[test]
fn xpbd_free_body_under_gravity_accumulates_the_substep_increment_exactly() {
    let g = Vec3Fix::new(r(1, 7), Fix128::from_int(-10), r(-2, 3));
    for dt in dts() {
        for substeps in SUBSTEPS {
            let (mut w, ids) = world(SolverBackend::Xpbd, substeps, g);
            let h = dt / Fix128::from_int(substeps as i64);
            let a = g * Fix128::ONE * h;
            let mut want = initial_velocities();
            for frame in 1..=FRAMES {
                w.step(dt);
                for v in &mut want {
                    for _ in 0..substeps {
                        *v = *v + a;
                    }
                }
                for (k, &i) in ids.iter().enumerate() {
                    assert_eq!(
                        w.bodies[i].velocity, want[k],
                        "dt {dt:?} substeps {substeps} frame {frame} body {k}"
                    );
                }
            }
        }
    }
}

/// XPBD and TGS give bit-identical velocities and positions for free bodies,
/// with and without gravity.
#[test]
fn xpbd_and_tgs_agree_bit_for_bit_on_free_bodies() {
    let gs = [
        Vec3Fix::ZERO,
        Vec3Fix::new(r(1, 7), Fix128::from_int(-10), r(-2, 3)),
    ];
    for g in gs {
        for dt in dts() {
            for substeps in SUBSTEPS {
                let (mut x, ids) = world(SolverBackend::Xpbd, substeps, g);
                let (mut t, _) = world(SolverBackend::Tgs, substeps, g);
                for frame in 1..=FRAMES {
                    x.step(dt);
                    t.step(dt);
                    for (k, &i) in ids.iter().enumerate() {
                        assert_eq!(
                            x.bodies[i].velocity, t.bodies[i].velocity,
                            "velocity g {g:?} dt {dt:?} substeps {substeps} frame {frame} body {k}"
                        );
                        assert_eq!(
                            x.bodies[i].position, t.bodies[i].position,
                            "position g {g:?} dt {dt:?} substeps {substeps} frame {frame} body {k}"
                        );
                    }
                }
            }
        }
    }
}
