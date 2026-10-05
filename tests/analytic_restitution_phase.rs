//! Oracles for Newton restitution in the XPBD velocity pass: the separation
//! speed of a head-on impact is `e` times the approach speed, whatever the
//! point inside a substep at which the two bodies first touch.
//!
//! Every scene runs through `PhysicsWorld::step` and, with the `parallel`
//! feature, `PhysicsWorld::step_parallel`, with `PhysicsConfig::default()`.
//! Expected values are closed forms, never obtained by running the solver:
//!
//! * a head-on impact of two spheres (masses `m_a`, `m_b`, velocities `v_a`,
//!   `v_b` along `x`, restitution `e`) leaves them at
//!   `v_a' = v_c − e (v_a − v_c)` and `v_b' = v_c − e (v_b − v_c)`, with
//!   `v_c = (m_a v_a + m_b v_b) / (m_a + m_b)` (`v_c = 0` and `v_b' = 0` for a
//!   static `b`), so the separation speed is `e |v_a − v_b|` and the momentum
//!   is unchanged by the impact;
//! * frame damping: `step` multiplies every dynamic body's velocity by
//!   `d = config.damping` once at the end of each frame, so after the frame
//!   of the impact both velocities above carry one factor `d`.
//!
//! The scenes place the first touch at `(3 + k/16) h` (`h = dt / substeps`,
//! `k = 0..15`) into the first frame, so `k` sweeps the phase of the contact
//! inside a substep. Gravity acts along `y` on both bodies alike and does not
//! touch the `x` axis.
//!
//! Tolerance `1e-9` for two dynamic bodies: both fall alike, the contact
//! normal stays on `x`, and the only error is fixed-point rounding (`~1e-30`
//! per operation). Against a static body the falling sphere tilts the normal
//! by `θ ≤ Δy / (2R)` with `Δy ≤ g h² s(s+1)/2` after `s ≤ 4` substeps, which
//! couples the vertical velocity `|v_y| ≤ 4 g h` into the normal; the budget
//! is `2 (1 + e) (|v_y| θ + |v| θ²)` (`≈ 1.4e-4` at `e = 1`), a factor 2 over
//! that first-order estimate.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

const DT: f64 = 1.0 / 60.0;
const R: f64 = 0.5;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// The two entry points under test.
#[derive(Clone, Copy, Debug)]
enum Path {
    Serial,
    #[cfg(feature = "parallel")]
    Batched,
}

fn paths() -> Vec<Path> {
    #[cfg(feature = "parallel")]
    {
        vec![Path::Serial, Path::Batched]
    }
    #[cfg(not(feature = "parallel"))]
    {
        vec![Path::Serial]
    }
}

/// A head-on pair. `m_b = None` makes `b` static (and `v_b` must be 0).
#[derive(Clone, Copy, Debug)]
struct Pair {
    m_a: f64,
    m_b: Option<f64>,
    v_a: f64,
    v_b: f64,
    e: f64,
}

/// Velocities of `a` and `b` after the impact frame: `(v_a', v_b')`.
fn closed_form(p: Pair, d: f64) -> (f64, f64) {
    match p.m_b {
        Some(m_b) => {
            let vc = (p.m_a * p.v_a + m_b * p.v_b) / (p.m_a + m_b);
            ((vc - p.e * (p.v_a - vc)) * d, (vc - p.e * (p.v_b - vc)) * d)
        }
        None => (-p.e * p.v_a * d, 0.0),
    }
}

/// Runs one frame with the first touch at `(3 + k/16) h` and returns
/// `(v_a', v_b')` along `x`.
fn run(p: Pair, k: usize, path: Path) -> (f64, f64) {
    let config = PhysicsConfig::default();
    let h = DT / config.substeps as f64;
    let mut w = PhysicsWorld::new(config);
    let gap = (3.0 + k as f64 / 16.0) * h * (p.v_a - p.v_b);
    let a_body =
        RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), fx(p.m_a)).with_velocity(v3(p.v_a, 0.0, 0.0));
    let b_pos = v3(2.0 * R + gap, 0.0, 0.0);
    let b_body = match p.m_b {
        Some(m_b) => RigidBody::new_dynamic(b_pos, fx(m_b)).with_velocity(v3(p.v_b, 0.0, 0.0)),
        None => RigidBody::new_static(b_pos),
    };
    let a = w.add_body_with_radius(a_body, fx(R));
    let b = w.add_body_with_radius(b_body, fx(R));
    let id = w
        .material_table
        .register(alice_physics::PhysicsMaterial::new(
            0,
            Fix128::ZERO,
            fx(p.e),
        ));
    w.set_body_material(a, id);
    w.set_body_material(b, id);
    match path {
        Path::Serial => w.step(fx(DT)),
        #[cfg(feature = "parallel")]
        Path::Batched => w.step_parallel(fx(DT)),
    }
    (
        w.get_body(a).expect("a").velocity.x.to_f64(),
        w.get_body(b).expect("b").velocity.x.to_f64(),
    )
}

fn cases() -> Vec<(&'static str, Pair)> {
    let base = Pair {
        m_a: 1.0,
        m_b: Some(3.0),
        v_a: 4.0,
        v_b: -2.0,
        e: 0.5,
    };
    vec![
        ("m 1/3, e 0.5", base),
        ("m 1/3, e 0.2", Pair { e: 0.2, ..base }),
        ("m 1/3, e 0.8", Pair { e: 0.8, ..base }),
        (
            "equal m, e 1",
            Pair {
                m_a: 1.0,
                m_b: Some(1.0),
                v_a: 2.0,
                v_b: -2.0,
                e: 1.0,
            },
        ),
        (
            "static b, e 1",
            Pair {
                m_a: 1.0,
                m_b: None,
                v_a: 4.0,
                v_b: 0.0,
                e: 1.0,
            },
        ),
        (
            "static b, e 0.5",
            Pair {
                m_a: 1.0,
                m_b: None,
                v_a: 4.0,
                v_b: 0.0,
                e: 0.5,
            },
        ),
    ]
}

/// Error budget of the closed form for `p`, see the module docs.
fn tolerance(p: Pair) -> f64 {
    if p.m_b.is_some() {
        return 1e-9;
    }
    let config = PhysicsConfig::default();
    let g = -config.gravity.y.to_f64();
    let h = DT / config.substeps as f64;
    let theta = g * h * h * 10.0 / (2.0 * R);
    let v_y = 4.0 * g * h;
    2.0 * (1.0 + p.e) * (v_y * theta + p.v_a.abs() * theta * theta)
}

/// The separation speed `v_b' − v_a'` equals `e |v_a − v_b| d` for every
/// phase `k` of the first touch inside a substep, each body leaves with its
/// closed-form velocity, in both paths.
#[test]
fn the_separation_speed_is_e_times_the_approach_speed_at_every_phase() {
    let d = PhysicsConfig::default().damping.to_f64();
    let mut failures = Vec::new();
    for path in paths() {
        for (name, p) in cases() {
            let (va_want, vb_want) = closed_form(p, d);
            let sep_want = vb_want - va_want;
            let tol = tolerance(p);
            for k in 0..16 {
                let (va, vb) = run(p, k, path);
                let sep = vb - va;
                println!(
                    "{path:?} {name} k={k:2}: separation {:.6} (closed form {:.6})",
                    sep / d,
                    sep_want / d
                );
                if (va - va_want).abs() > tol || (vb - vb_want).abs() > tol {
                    failures.push(format!(
                        "{path:?} {name} k={k}: v_a' {va:.9} / {va_want:.9}, v_b' {vb:.9} / {vb_want:.9}"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The impact conserves momentum at every phase: `m_a v_a' + m_b v_b' =
/// d (m_a v_a + m_b v_b)` for two dynamic bodies.
#[test]
fn the_impact_conserves_momentum_at_every_phase() {
    let d = PhysicsConfig::default().damping.to_f64();
    for path in paths() {
        for (name, p) in cases() {
            let Some(m_b) = p.m_b else { continue };
            let want = d * (p.m_a * p.v_a + m_b * p.v_b);
            for k in 0..16 {
                let (va, vb) = run(p, k, path);
                let got = p.m_a * va + m_b * vb;
                assert!(
                    (got - want).abs() < 1e-9,
                    "{path:?} {name} k={k}: momentum {got:.12}, before {want:.12}"
                );
            }
        }
    }
}

/// Restitution threshold of Müller et al. 2020: an approach slower than
/// `2 |g| h` (the gravity of two substeps) is treated as resting and gets
/// `e = 0`, so the sphere stops against the static body (`v_a' = 0`); an
/// approach above it bounces with `e`. Static `b`, `e = 1`, approach speeds
/// `1.5 g h` and `2.5 g h`, every phase `k`.
#[test]
fn approaches_below_the_restitution_threshold_do_not_bounce() {
    let config = PhysicsConfig::default();
    let d = config.damping.to_f64();
    let g = -config.gravity.y.to_f64();
    let h = DT / config.substeps as f64;
    for path in paths() {
        for (factor, e_eff) in [(1.5, 0.0), (2.5, 1.0)] {
            let p = Pair {
                m_a: 1.0,
                m_b: None,
                v_a: factor * g * h,
                v_b: 0.0,
                e: 1.0,
            };
            let want = -e_eff * p.v_a * d;
            let tol = tolerance(p);
            for k in 0..16 {
                let (va, _) = run(p, k, path);
                assert!(
                    (va - want).abs() < tol,
                    "{path:?} approach {factor} g h, k={k}: v_a' {va:.9}, closed form {want:.9}"
                );
            }
        }
    }
}
