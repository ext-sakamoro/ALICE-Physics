//! Oracles for the joint compliance contract documented on
//! [`solve_joints`], [`solve_joints_breakable`], [`solve_extra_joints`] and
//! `PhysicsWorld::step`.
//!
//! XPBD (Macklin, Müller, Chentanez, MIG 2016, eq. 18) moves a compliant
//! constraint row by `Δλ = (−C − α̃ λ) / (w + α̃)` with `α̃ = α / h²`, `λ`
//! reset to zero at the start of each substep. Closed forms used below:
//!
//! * `step` solves the joints once per substep, outside the `iterations`
//!   loop, so `λ = 0` at every solve and the state after any number of frames
//!   does not depend on `iterations` at all (bit for bit);
//! * a compliant joint is a spring of stiffness `1/α`, so a mass under the
//!   constant load `F = m g` settles at the stretch `F · α`;
//! * a free-function call carries no `λ`, so `k` calls at the same `h` leave
//!   the gap `d · (α̃ / (w + α̃))^k`;
//! * `step_parallel` runs the same joint solve as `step`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

#[allow(deprecated)] // the separation-based variant is pinned below
use alice_physics::joint::solve_joints_breakable;
use alice_physics::joint::{solve_joints, BallJoint, FixedJoint, HingeJoint, Joint};
use alice_physics::joint_extra::{solve_extra_joints, ExtraJoint, WeldJoint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

const G: f64 = 10.0;
const ALPHA: f64 = 1e-3;
/// Frames to settle the oscillation (10 s) for the rest-state check.
const SETTLE_FRAMES: usize = 600;
/// Frames for the bit-identity checks: one period is 12 frames, so 2 s cover
/// the transient where an iteration-dependent solve would differ most.
const SHORT_FRAMES: usize = 120;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Ball,
    Hinge,
    Fixed,
}

const KINDS: [Kind; 3] = [Kind::Ball, Kind::Hinge, Kind::Fixed];

/// A unit mass hanging from a static body by a compliant joint whose anchors
/// coincide with the mass's centre (no lever arm, so the joint acts as a pure
/// linear spring along gravity).
fn hanging(kind: Kind, substeps: usize, iterations: usize) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(SolverConfig {
        substeps,
        iterations,
        gravity: v3(0.0, -G, 0.0),
        ..SolverConfig::default()
    });
    let a = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let b = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    // settle the oscillation; damping has no effect on the rest state (v = 0)
    w.bodies[b].linear_damping = fx(0.9);
    let joint = match kind {
        Kind::Ball => Joint::Ball(
            BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO).with_compliance(fx(ALPHA)),
        ),
        Kind::Hinge => Joint::Hinge(
            HingeJoint::new(
                a,
                b,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Vec3Fix::UNIT_Z,
                Vec3Fix::UNIT_Z,
            )
            .with_compliance(fx(ALPHA)),
        ),
        Kind::Fixed => {
            let mut j = FixedJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
            j.compliance = fx(ALPHA);
            Joint::Fixed(j)
        }
    };
    w.add_joint(joint);
    (w, b)
}

fn run(kind: Kind, substeps: usize, iterations: usize, frames: usize) -> Vec3Fix {
    let (mut w, b) = hanging(kind, substeps, iterations);
    for _ in 0..frames {
        w.step(fx(1.0 / 60.0));
    }
    w.bodies[b].position
}

/// The rest state is independent of `iterations`, bit for bit, for every
/// substep count, and approaches `F · α` once the frame is split.
#[test]
fn compliant_joint_stretch_is_iteration_independent_and_equals_f_alpha() {
    let f_alpha = G * ALPHA; // m = 1
    for kind in KINDS {
        for substeps in [1usize, 4, 8] {
            let base = run(kind, substeps, 1, SHORT_FRAMES);
            for iterations in [4usize, 16] {
                let p = run(kind, substeps, iterations, SHORT_FRAMES);
                assert_eq!(
                    p, base,
                    "{kind:?} substeps {substeps}: iterations {iterations} differs from 1"
                );
            }
            // Budget: after 10 s the residual of the decaying oscillation is
            // below 0.1 % of F·α at 4 and 8 substeps (measured 2.3e-4 and
            // 7.5e-4 relative). With 1 substep the default frame damping
            // removes part of each substep's gravity velocity, which shifts
            // the rest point; that case is pinned only for independence.
            if substeps > 1 {
                let stretch = -run(kind, substeps, 1, SETTLE_FRAMES).y.to_f64();
                assert!(
                    ((stretch - f_alpha) / f_alpha).abs() < 1e-3,
                    "{kind:?} substeps {substeps}: stretch {stretch:e}, F·α {f_alpha:e}"
                );
            }
        }
    }
}

/// `k` free-function calls at the same `h` leave `d · (α̃ / (w + α̃))^k`: no
/// `λ` is carried between calls. `α = 1`, `h = 1`, `w = 1` gives the ratio
/// `1/2`, exact in fixed point; `h = 1/2` gives `α̃ = 4` and the ratio `4/5`.
#[allow(deprecated)] // pins the separation-based solve_joints_breakable
#[test]
fn repeated_free_function_calls_compound_the_gap_fraction() {
    let d0 = 2.0;
    for (h, ratio) in [(1.0, 0.5), (0.5, 0.8)] {
        for k in [1usize, 2, 4, 16] {
            let expected = d0 * f64::powi(ratio, k as i32);
            let joints = [Joint::Ball(
                BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_compliance(Fix128::ONE),
            )];

            let fresh = || {
                vec![
                    RigidBody::new_static(Vec3Fix::ZERO),
                    RigidBody::new_dynamic(v3(0.0, -d0, 0.0), Fix128::ONE),
                ]
            };

            let mut bodies = fresh();
            for _ in 0..k {
                solve_joints(&joints, &mut bodies, fx(h));
            }
            let gap = -bodies[1].position.y.to_f64();

            let mut bodies_b = fresh();
            for _ in 0..k {
                let broken = solve_joints_breakable(&joints, &mut bodies_b, fx(h));
                assert!(broken.is_empty());
            }
            let gap_b = -bodies_b[1].position.y.to_f64();

            let mut weld = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
            weld.compliance = Fix128::ONE;
            let extra = [ExtraJoint::Weld(weld)];
            let mut bodies_e = fresh();
            for _ in 0..k {
                solve_extra_joints(&mut bodies_e, &extra, fx(h));
            }
            let gap_e = -bodies_e[1].position.y.to_f64();

            // Budget: fixed-point rounding of one division per call, < 1e-15
            for (name, g) in [
                ("solve_joints", gap),
                ("solve_joints_breakable", gap_b),
                ("solve_extra_joints", gap_e),
            ] {
                assert!(
                    (g - expected).abs() < 1e-12,
                    "{name} h {h} k {k}: gap {g:e}, closed form {expected:e}"
                );
            }
        }
    }
}

/// `step_parallel` gives the same state as `step`, bit for bit.
#[cfg(feature = "parallel")]
#[test]
fn step_parallel_matches_step_for_compliant_joints() {
    for kind in KINDS {
        for substeps in [1usize, 4] {
            for iterations in [1usize, 4] {
                let serial = run(kind, substeps, iterations, SHORT_FRAMES);
                let (mut w, b) = hanging(kind, substeps, iterations);
                for _ in 0..SHORT_FRAMES {
                    w.step_parallel(fx(1.0 / 60.0));
                }
                assert_eq!(
                    w.bodies[b].position, serial,
                    "{kind:?} substeps {substeps} iterations {iterations}"
                );
            }
        }
    }
}
