//! A body held at a place by a critically damped spring joint must stay there for good, with
//! the world's sleeping on as with it off. The spring is the world's own (`rest_length` 0,
//! `k = m ω²`, `c = 2 m ω`) from a static anchor lifted by the sag `g / ω²` above the place,
//! so the place is the spring's equilibrium under gravity.
//!
//! Each substep of the spring solve is the backward Euler step of the critically damped
//! oscillator, `x' = (x (1 + 2a) + h v) / (1 + a)²` with `a = ω h`, whose step matrix has the
//! double eigenvalue `λ = 1 / (1 + a)`; from rest at `x0` the distance from the place after
//! `n` substeps is at most `|x0| (1 + n a / (1 + a)) λⁿ` (the map is iterated below to confirm
//! the bound), plus the rounding of Q64.64 positions, far below `1e-12` m.
//!
//! The scene and the measured defect are the reproduction of a hold of the site's touch
//! layer: with sleeping on, the body goes to sleep while the spring still pulls it (it is
//! moving below the sleep threshold), and while asleep it still moves (0.17 mm at 2 s,
//! creeping to 1.46 mm at 9 s, snapping back, creeping again); with sleeping off it settles.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the backward Euler bound is computed in f64, never by the solver under test"
)]

use alice_physics_core::joint::{BallJoint, Joint, SpringJoint};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::solver::{BodyType, PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::SleepConfig;

const OMEGA: f64 = 6.0;
const DT: f64 = 1.0 / 60.0;
const SUBSTEPS: usize = 8;
const G: f64 = 9.81;
/// frames run (20 s) and the frame the comparison starts at (3 s)
const FRAMES: u64 = 1200;
const FROM: u64 = 180;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// the backward Euler spring's distance bound after `n` substeps from rest at `x0`
fn settled_bound(x0: f64, n: u64) -> f64 {
    let a = OMEGA * DT / SUBSTEPS as f64;
    x0.abs() * (1.0 + n as f64 * a / (1.0 + a)) * (1.0 + a).powi(-(n as i32))
}

/// (largest excess of the distance over its bound, the bound there, frames compared)
fn held_for_long(sleep: bool) -> (f64, f64, u64) {
    // the bound is the scheme's: its 2×2 map, iterated, never exceeds it
    let h = DT / SUBSTEPS as f64;
    let a = OMEGA * h;
    let (mut x, mut vel) = (2.0f64, 0.0f64);
    for n in 1..=FRAMES * SUBSTEPS as u64 {
        let x1 = (x * (1.0 + 2.0 * a) + h * vel) / ((1.0 + a) * (1.0 + a));
        vel = (x1 - x) / h;
        x = x1;
        assert!(
            x.abs() <= settled_bound(2.0, n) * (1.0 + 1e-9) + 1e-300,
            "{n}: {x}"
        );
    }
    let mut w = PhysicsWorld::new(SolverConfig {
        substeps: SUBSTEPS,
        gravity: v3(0.0, -G, 0.0),
        ..SolverConfig::default()
    });
    if !sleep {
        w.set_sleep_config(SleepConfig {
            time_to_sleep: Fix128 {
                hi: i64::MAX,
                lo: u64::MAX,
            },
            ..SleepConfig::default()
        });
    }
    let m = 1.0;
    let place = [0.0, 1.0, 5.0];
    let anchor = w.add_body(RigidBody::new_static(v3(
        place[0],
        place[1] + G / (OMEGA * OMEGA),
        place[2],
    )));
    let body = w.add_body(RigidBody::new_dynamic(v3(0.0, 3.0, 5.0), fx(m)));
    w.add_joint(Joint::Spring(SpringJoint::new(
        anchor,
        body,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::ZERO,
        fx(m * OMEGA * OMEGA),
        fx(2.0 * m * OMEGA),
    )));
    let (mut worst, mut allowed, mut compared) = (f64::NEG_INFINITY, 0.0f64, 0u64);
    for frame in 1..=FRAMES {
        w.step(fx(DT));
        if frame >= FROM {
            let p = w.bodies[body].position;
            let off = ((p.x.to_f64() - place[0]).powi(2)
                + (p.y.to_f64() - place[1]).powi(2)
                + (p.z.to_f64() - place[2]).powi(2))
            .sqrt();
            let bound = settled_bound(2.0, frame * SUBSTEPS as u64) + 1e-12;
            if off - bound > worst - allowed {
                (worst, allowed) = (off, bound);
            }
            compared += 1;
        }
    }
    (worst, allowed, compared)
}

/// With sleeping off, from 3 s to 20 s the body is never further from its place than the
/// scheme's transient allows: the bound is the right one.
#[test]
fn a_spring_held_body_stays_at_its_place_with_sleep_off() {
    let (worst, allowed, compared) = held_for_long(false);
    assert_eq!(compared, FRAMES - FROM + 1);
    assert!(worst <= allowed, "{worst:.3e} m, allowed {allowed:.3e} m");
}

/// The same with the world's sleeping on (the default): a body whose spring still pulls it
/// does not go to sleep, and a sleeping body is not moved, so it stays as close.
#[test]
fn a_spring_held_body_stays_at_its_place_with_sleep_on() {
    let (worst, allowed, compared) = held_for_long(true);
    assert_eq!(compared, FRAMES - FROM + 1);
    assert!(worst <= allowed, "{worst:.3e} m, allowed {allowed:.3e} m");
}

/// A measurement, not a check: for a softer, the same and a stiffer spring (ω 0.6, 6, 60),
/// the frame the held body first falls asleep (its island reaches the `Q64.64` fixed point
/// of the backward Euler spring and has been idle `time_to_sleep`), its offset from the
/// place then, and whether it stays asleep and in place for the rest of the run.
#[test]
#[ignore = "a measurement, not a check"]
fn measure_when_the_held_body_reaches_its_fixed_point_and_sleeps() {
    for omega in [0.6f64, 6.0, 60.0] {
        let mut w = PhysicsWorld::new(SolverConfig {
            substeps: SUBSTEPS,
            gravity: v3(0.0, -G, 0.0),
            ..SolverConfig::default()
        });
        let anchor = w.add_body(RigidBody::new_static(v3(
            0.0,
            1.0 + G / (omega * omega),
            5.0,
        )));
        let body = w.add_body(RigidBody::new_dynamic(v3(0.0, 3.0, 5.0), fx(1.0)));
        w.add_joint(Joint::Spring(SpringJoint::new(
            anchor,
            body,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::ZERO,
            fx(omega * omega),
            fx(2.0 * omega),
        )));
        let mut first = None;
        let mut woke_after = false;
        let mut settled_at = w.bodies[body].position;
        for frame in 1..=60 * 600u64 {
            w.step(fx(DT));
            let s = w.is_sleeping(body);
            if s && first.is_none() {
                first = Some(frame);
                settled_at = w.bodies[body].position;
            }
            if first.is_some() && (!s || w.bodies[body].position != settled_at) {
                woke_after = true;
            }
        }
        let offset = (settled_at.y.to_f64() - 1.0).abs();
        println!(
            "ω {omega}: first asleep at frame {:?} ({:.1} s), offset {offset:.3e} m, woke or moved afterwards: {woke_after}",
            first,
            first.map_or(f64::NAN, |f| f as f64 * DT)
        );
    }
}

/// A body jointed to a moving one does not stay asleep, and sleeping is invisible to it:
/// a dynamic body hangs from a kinematic one by a ball joint (zero gravity), both at rest
/// for 2 s, so with sleeping on the hung body falls asleep; then the kinematic body moves
/// 1 mm a step for 60 steps. The hung body must follow it exactly as it does in the same
/// world with sleeping off (positions bit for bit at every step: at rest, sleeping or not,
/// the two worlds are the same, and once the neighbour moves the island must be awake),
/// and it must not be flagged asleep in any step its neighbour moved.
#[test]
fn a_sleeper_jointed_to_a_moving_body_wakes_with_it() {
    let scene = |sleep: bool| {
        let mut w = PhysicsWorld::new(SolverConfig {
            substeps: SUBSTEPS,
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        });
        if !sleep {
            w.set_sleep_config(SleepConfig {
                time_to_sleep: Fix128 {
                    hi: i64::MAX,
                    lo: u64::MAX,
                },
                ..SleepConfig::default()
            });
        }
        let mut k = RigidBody::new_dynamic(v3(0.0, 2.0, 0.0), fx(1.0));
        k.body_type = BodyType::Kinematic;
        let kin = w.add_body(k);
        let hung = w.add_body(RigidBody::new_dynamic(v3(0.0, 1.0, 0.0), fx(1.0)));
        w.add_joint(Joint::Ball(BallJoint::new(
            kin,
            hung,
            v3(0.0, -0.5, 0.0),
            v3(0.0, 0.5, 0.0),
        )));
        for _ in 0..120 {
            w.step(fx(DT));
        }
        (w, kin, hung)
    };
    let (mut on, kin, hung) = scene(true);
    let (mut off, _, _) = scene(false);
    assert!(
        on.is_sleeping(hung),
        "fixture: the hung body must be asleep"
    );
    assert_eq!(
        on.bodies[hung].position, off.bodies[hung].position,
        "at rest the two agree"
    );
    let mut asleep_while_pulled = 0usize;
    for step in 1..=60i64 {
        let target = v3(0.001 * step as f64, 2.0, 0.0);
        on.bodies[kin].set_kinematic_target(target, QuatFix::IDENTITY);
        off.bodies[kin].set_kinematic_target(target, QuatFix::IDENTITY);
        on.step(fx(DT));
        off.step(fx(DT));
        if on.is_sleeping(hung) {
            asleep_while_pulled += 1;
        }
        assert_eq!(
            on.bodies[hung].position, off.bodies[hung].position,
            "step {step}: sleeping changed the hung body's motion"
        );
    }
    assert_eq!(
        asleep_while_pulled, 0,
        "steps the hung body stayed asleep while pulled"
    );
}

/// A measurement, not a check: with sleeping off, the held body's state after the
/// transient (a minute, then up to ten): the first frame its whole state (position and
/// velocity, every bit) recurs, the period of that recurrence, and the largest offset from
/// the place and speed over one period, for a softer, the same and a stiffer spring.
#[test]
#[ignore = "a measurement, not a check"]
fn measure_the_late_orbit_of_the_held_body() {
    for omega in [0.6f64, 6.0, 60.0] {
        let mut w = PhysicsWorld::new(SolverConfig {
            substeps: SUBSTEPS,
            gravity: v3(0.0, -G, 0.0),
            ..SolverConfig::default()
        });
        w.set_sleep_config(SleepConfig {
            time_to_sleep: Fix128 {
                hi: i64::MAX,
                lo: u64::MAX,
            },
            ..SleepConfig::default()
        });
        let anchor = w.add_body(RigidBody::new_static(v3(
            0.0,
            1.0 + G / (omega * omega),
            5.0,
        )));
        let body = w.add_body(RigidBody::new_dynamic(v3(0.0, 3.0, 5.0), fx(1.0)));
        w.add_joint(Joint::Spring(SpringJoint::new(
            anchor,
            body,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::ZERO,
            fx(omega * omega),
            fx(2.0 * omega),
        )));
        let mut seen: Vec<(Vec3Fix, Vec3Fix)> = Vec::new();
        let mut found = None;
        for frame in 1..=60 * 600u64 {
            w.step(fx(DT));
            let b = &w.bodies[body];
            let state = (b.position, b.velocity);
            if let Some(k) = seen.iter().rposition(|s| *s == state) {
                let period = seen.len() - k;
                let orbit = &seen[k..];
                let off = orbit
                    .iter()
                    .map(|(p, _)| (p.y.to_f64() - 1.0).abs())
                    .fold(0.0, f64::max);
                let speed = orbit
                    .iter()
                    .map(|(_, v)| v.length().to_f64())
                    .fold(0.0, f64::max);
                found = Some((frame, period, off, speed));
                break;
            }
            seen.push(state);
            if seen.len() > 4096 {
                seen.remove(0);
            }
        }
        println!("ω {omega}: recurrence (frame, period, max offset m, max speed m/s) = {found:?}");
    }
}
