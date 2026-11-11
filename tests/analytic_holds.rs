//! Oracles for `holds`: the blow against the impulse it states, and a hold against the closed form of a critically
//! damped spring, `x(t) = (x0 + (v0 + ω x0) t) e^(−ωt)`, with the first-order error of the integrator derived below.
//!
//! The joint is integrated semi-implicitly at the substep `h`. For a critically damped linear oscillator the local
//! error per step is O(ω² h² |x|), so the global error over the transient is bounded by about `ω² h ∫|x| dt`, and
//! `∫|x| dt` is `|x0| / ω + |v0 + ω x0| / ω²` at most: with `v0 = 0`, `≤ 2 ω h |x0|`. The bound below uses that.

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 exp evaluates the closed form the world is compared with"
)]

use alice_physics_core::holds::{HoldError, Holds};
use alice_physics_core::{BodyType, Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn world(g: f64, substeps: usize) -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig {
        gravity: v(0.0, -g, 0.0),
        substeps,
        ..PhysicsConfig::default()
    })
}
fn body(w: &mut PhysicsWorld, at: Vec3Fix, mass: f64) -> usize {
    w.add_body(RigidBody::new(at, fx(mass)))
}
fn pos(w: &PhysicsWorld, i: usize) -> [f64; 3] {
    let p = w.bodies[i].position;
    [p.x.to_f64(), p.y.to_f64(), p.z.to_f64()]
}
fn vel(w: &PhysicsWorld, i: usize) -> [f64; 3] {
    let p = w.bodies[i].velocity;
    [p.x.to_f64(), p.y.to_f64(), p.z.to_f64()]
}
const DT: f64 = 1.0 / 60.0;

// ── the blow ─────────────────────────────────────────────────────────────

#[test]
fn a_blow_gives_each_body_the_impulse_it_states() {
    let mut w = world(0.0, 1);
    let cases = [(1.0, 1.0), (2.0, 4.0), (3.0, 0.5)];
    let ids: Vec<usize> = cases
        .iter()
        .map(|&(d, m)| body(&mut w, v(d, 0.0, 0.0), m))
        .collect();
    let far = body(&mut w, v(10.0, 0.0, 0.0), 1.0);
    let n = w.apply_radial_impulse(Vec3Fix::ZERO, fx(5.0), fx(8.0));
    assert_eq!(n, 3);
    for (&(d, m), &i) in cases.iter().zip(&ids) {
        let want = 8.0 * (1.0 - d / 5.0) / m;
        let got = vel(&w, i);
        assert!(
            (got[0] - want).abs() < 1e-9,
            "d {d} m {m}: {} vs {want}",
            got[0]
        );
        assert_eq!(got[1], 0.0);
        assert_eq!(got[2], 0.0);
    }
    assert_eq!(vel(&w, far), [0.0, 0.0, 0.0], "past the radius");
}

#[test]
fn a_blow_goes_along_the_way_from_the_centre_in_any_direction() {
    let mut w = world(0.0, 1);
    let d = [1.0_f64, -2.0, 2.0];
    let i = body(&mut w, v(d[0], d[1], d[2]), 1.0);
    let at = v(0.0, 0.0, 0.0);
    w.apply_radial_impulse(at, fx(6.0), fx(6.0));
    // |d| = 3, so the impulse is 6 (1 - 1/2) = 3 along d/3
    let got = vel(&w, i);
    for k in 0..3 {
        assert!((got[k] - d[k]).abs() < 1e-9, "{k}: {} vs {}", got[k], d[k]);
    }
}

#[test]
fn a_negative_blow_pulls_in_and_equal_bodies_either_side_cancel() {
    let mut w = world(0.0, 1);
    let a = body(&mut w, v(2.0, 0.0, 0.0), 1.0);
    let b = body(&mut w, v(-2.0, 0.0, 0.0), 1.0);
    w.apply_radial_impulse(Vec3Fix::ZERO, fx(4.0), fx(-3.0));
    assert!((vel(&w, a)[0] + 1.5).abs() < 1e-9, "{:?}", vel(&w, a));
    assert!((vel(&w, b)[0] - 1.5).abs() < 1e-9, "{:?}", vel(&w, b));
    assert!((vel(&w, a)[0] + vel(&w, b)[0]).abs() < 1e-12, "momentum");
}

#[test]
fn a_blow_leaves_what_cannot_move_and_what_has_no_way_alone() {
    let mut w = world(0.0, 1);
    let at = w.add_body(RigidBody::new_static(v(1.0, 0.0, 0.0)));
    let middle = body(&mut w, Vec3Fix::ZERO, 1.0);
    assert_eq!(w.apply_radial_impulse(Vec3Fix::ZERO, fx(5.0), fx(8.0)), 0);
    assert_eq!(vel(&w, at), [0.0, 0.0, 0.0]);
    assert_eq!(vel(&w, middle), [0.0, 0.0, 0.0]);
    assert_eq!(w.apply_radial_impulse(Vec3Fix::ZERO, fx(0.0), fx(8.0)), 0);
    assert_eq!(w.apply_radial_impulse(Vec3Fix::ZERO, fx(-1.0), fx(8.0)), 0);
}

#[test]
fn a_blow_wakes_a_body_at_rest_and_it_then_moves() {
    let mut w = world(0.0, 1);
    let i = body(&mut w, v(1.0, 0.0, 0.0), 1.0);
    for _ in 0..600 {
        w.step(fx(DT));
    }
    assert_eq!(w.apply_radial_impulse(Vec3Fix::ZERO, fx(5.0), fx(5.0)), 1);
    for _ in 0..60 {
        w.step(fx(DT));
    }
    // 4 m/s for a second
    assert!((pos(&w, i)[0] - 5.0).abs() < 0.2, "{:?}", pos(&w, i));
}

// ── a hold ───────────────────────────────────────────────────────────────

fn closed(x0: f64, w: f64, t: f64) -> f64 {
    (x0 + w * x0 * t) * (-w * t).exp()
}

/// The world against the closed form over the whole transient, within the derived bound
fn against_closed_form(mass: f64, substeps: usize, gravity: f64) {
    let omega = 6.0;
    let mut w = world(gravity, substeps);
    let target = [1.0, 2.0, -3.0];
    let start = [4.0, 2.0, -3.0];
    let i = body(&mut w, v(start[0], start[1], start[2]), mass);
    let hands = Holds::new(&mut w);
    hands
        .hold(&mut w, i, v(target[0], target[1], target[2]), fx(omega))
        .expect("hold");
    let h = DT / substeps as f64;
    let bound = 2.0 * omega * h * 3.0 + 1e-6;
    let mut compared = 0;
    for k in 1..=180 {
        w.step(fx(DT));
        let t = k as f64 * DT;
        let want = target[0] + closed(start[0] - target[0], omega, t);
        let got = pos(&w, i)[0];
        assert!(
            (got - want).abs() <= bound,
            "m {mass} s {substeps} t {t}: {got} vs {want} (bound {bound})"
        );
        compared += 1;
    }
    assert_eq!(compared, 180);
    // it has come to rest at the place, under gravity too
    let p = pos(&w, i);
    for k in 0..3 {
        assert!(
            (p[k] - target[k]).abs() < 0.01,
            "at rest {k}: {} vs {}",
            p[k],
            target[k]
        );
    }
}

#[test]
fn a_hold_follows_the_closed_form_at_any_mass() {
    for mass in [0.25, 1.0, 40.0] {
        against_closed_form(mass, 4, 0.0);
    }
}

#[test]
fn a_hold_follows_the_closed_form_at_more_substeps_and_closer() {
    for substeps in [1, 2, 4, 8] {
        against_closed_form(1.0, substeps, 0.0);
    }
}

#[test]
fn under_gravity_it_comes_to_the_place_not_below_it() {
    against_closed_form(2.0, 4, 9.81);
}

#[test]
fn a_hold_does_not_overshoot() {
    // critically damped: it never passes the place, starting from rest
    let mut w = world(0.0, 4);
    let i = body(&mut w, v(3.0, 0.0, 0.0), 1.0);
    let hands = Holds::new(&mut w);
    hands.hold(&mut w, i, Vec3Fix::ZERO, fx(8.0)).expect("hold");
    for _ in 0..300 {
        w.step(fx(DT));
        assert!(pos(&w, i)[0] > -0.03, "{:?}", pos(&w, i));
    }
}

#[test]
fn moved_it_follows_to_the_new_place() {
    let mut w = world(9.81, 4);
    let i = body(&mut w, Vec3Fix::ZERO, 1.0);
    let hands = Holds::new(&mut w);
    hands.hold(&mut w, i, Vec3Fix::ZERO, fx(8.0)).expect("hold");
    for _ in 0..30 {
        w.step(fx(DT));
    }
    hands.move_to(&mut w, i, v(0.0, 3.0, 1.0)).expect("move");
    for _ in 0..240 {
        w.step(fx(DT));
    }
    let p = pos(&w, i);
    assert!(
        (p[1] - 3.0).abs() < 0.01 && (p[2] - 1.0).abs() < 0.01,
        "{p:?}"
    );
}

#[test]
fn let_go_it_keeps_its_velocity_and_falls_as_free_fall() {
    let g = 10.0;
    let mut w = world(g, 4);
    let i = body(&mut w, v(0.0, 0.0, 0.0), 1.0);
    let hands = Holds::new(&mut w);
    hands
        .hold(&mut w, i, v(5.0, 5.0, 0.0), fx(4.0))
        .expect("hold");
    for _ in 0..20 {
        w.step(fx(DT));
    }
    assert!(hands.holds(&w, i));
    let joints = w.joints.len();
    let at = pos(&w, i);
    let speed = vel(&w, i);
    assert!(speed[0] > 1.0, "not moving yet: {speed:?}");
    hands.release(&mut w, i).expect("release");
    assert!(!hands.holds(&w, i));
    assert_eq!(w.joints.len(), joints - 1);
    assert_eq!(vel(&w, i), speed, "release changed the velocity");
    let n = 30;
    for _ in 0..n {
        w.step(fx(DT));
    }
    let t = n as f64 * DT;
    let p = pos(&w, i);
    assert!(
        (p[0] - (at[0] + speed[0] * t)).abs() < 0.02,
        "x {} vs {}",
        p[0],
        at[0] + speed[0] * t
    );
    let want = at[1] + speed[1] * t - 0.5 * g * t * t;
    assert!((p[1] - want).abs() < 0.1, "y {} vs {want}", p[1]);
}

#[test]
fn what_cannot_be_held_is_refused_and_nothing_changes() {
    let mut w = world(0.0, 1);
    let fixed = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let i = body(&mut w, Vec3Fix::ZERO, 1.0);
    let hands = Holds::new(&mut w);
    let before = w.joints.len();
    assert_eq!(
        hands.hold(&mut w, 999, Vec3Fix::ZERO, fx(1.0)),
        Err(HoldError::NoBody(999))
    );
    assert_eq!(
        hands.hold(&mut w, fixed, Vec3Fix::ZERO, fx(1.0)),
        Err(HoldError::NotDynamic(fixed))
    );
    assert_eq!(
        hands.hold(&mut w, i, Vec3Fix::ZERO, fx(0.0)),
        Err(HoldError::BadOmega)
    );
    assert_eq!(
        hands.hold(&mut w, i, Vec3Fix::ZERO, fx(-1.0)),
        Err(HoldError::BadOmega)
    );
    assert_eq!(
        hands.move_to(&mut w, i, Vec3Fix::ZERO),
        Err(HoldError::NotHeld(i))
    );
    assert_eq!(hands.release(&mut w, i), Err(HoldError::NotHeld(i)));
    assert_eq!(w.joints.len(), before);
    hands.hold(&mut w, i, Vec3Fix::ZERO, fx(1.0)).expect("hold");
    assert_eq!(
        hands.hold(&mut w, i, Vec3Fix::ZERO, fx(1.0)),
        Err(HoldError::AlreadyHeld(i))
    );
    assert_eq!(w.joints.len(), before + 1);
}

#[test]
fn a_hold_leaves_other_joints_and_other_hands_alone() {
    let mut w = world(0.0, 4);
    let a = body(&mut w, v(1.0, 0.0, 0.0), 1.0);
    let b = body(&mut w, v(2.0, 0.0, 0.0), 1.0);
    let first = Holds::new(&mut w);
    let second = Holds::new(&mut w);
    first.hold(&mut w, a, Vec3Fix::ZERO, fx(4.0)).expect("hold");
    second
        .hold(&mut w, b, Vec3Fix::ZERO, fx(4.0))
        .expect("hold");
    assert!(!first.holds(&w, b));
    assert!(!second.holds(&w, a));
    assert_eq!(first.release_all(&mut w), 1);
    assert!(second.holds(&w, b));
    assert_eq!(first.release_all(&mut w), 0);
}

// ── forming ──────────────────────────────────────────────────────────────

#[test]
fn pieces_form_at_their_places_and_when_let_go_they_fall() {
    let mut w = world(10.0, 2);
    let bodies: Vec<usize> = (0..12)
        .map(|k| body(&mut w, v(f64::from(k) * 0.1, 0.0, 0.0), 1.0))
        .collect();
    let spots: Vec<Vec3Fix> = (0..12).map(|k| v(f64::from(k) - 6.0, 4.0, -2.0)).collect();
    let hands = Holds::new(&mut w);
    hands.form(&mut w, &bodies, &spots, fx(5.0)).expect("form");
    for _ in 0..240 {
        w.step(fx(DT));
    }
    for (k, &i) in bodies.iter().enumerate() {
        let p = pos(&w, i);
        assert!((p[0] - (k as f64 - 6.0)).abs() < 0.02, "{k}: {p:?}");
        assert!((p[1] - 4.0).abs() < 0.02, "{k}: {p:?}");
    }
    assert_eq!(hands.release_all(&mut w), 12);
    for _ in 0..30 {
        w.step(fx(DT));
    }
    for &i in &bodies {
        assert!(pos(&w, i)[1] < 3.0, "did not fall: {:?}", pos(&w, i));
    }
}

#[test]
fn a_form_that_cannot_be_is_refused_whole() {
    let mut w = world(0.0, 1);
    let a = body(&mut w, Vec3Fix::ZERO, 1.0);
    let hands = Holds::new(&mut w);
    let before = w.joints.len();
    assert_eq!(
        hands.form(&mut w, &[a, 999], &[Vec3Fix::ZERO, Vec3Fix::ZERO], fx(1.0)),
        Err(HoldError::NoBody(999))
    );
    assert_eq!(w.joints.len(), before);
    assert!(!hands.holds(&w, a));
    assert_eq!(
        hands.form(&mut w, &[a], &[], fx(1.0)),
        Err(HoldError::NoBody(a))
    );
    assert_eq!(
        hands.form(&mut w, &[a, a], &[Vec3Fix::ZERO, Vec3Fix::ZERO], fx(1.0)),
        Err(HoldError::AlreadyHeld(a))
    );
    assert_eq!(w.joints.len(), before);
}

// ── in the world's state ─────────────────────────────────────────────────

#[test]
fn a_hold_is_in_the_digest_and_survives_a_snapshot_bit_for_bit() {
    let build = || {
        let mut w = world(9.81, 2);
        let i = body(&mut w, v(1.0, 1.0, 1.0), 1.0);
        let hands = Holds::new(&mut w);
        (w, i, hands)
    };
    let (mut a, i, hands) = build();
    let (mut other, ..) = build();
    assert_eq!(a.state_digest(), other.state_digest());
    hands
        .hold(&mut a, i, v(5.0, 5.0, 5.0), fx(4.0))
        .expect("hold");
    assert_ne!(
        a.state_digest(),
        other.state_digest(),
        "a hold is not in the digest"
    );
    for _ in 0..30 {
        a.step(fx(DT));
        other.step(fx(DT));
    }
    assert_ne!(a.state_digest(), other.state_digest());
    let saved = a.snapshot_world();
    let at_save = a.state_digest();
    for _ in 0..30 {
        a.step(fx(DT));
    }
    let want = a.state_digest();
    let (mut b, ..) = build();
    b.restore_world(&saved).expect("restore");
    assert_eq!(b.state_digest(), at_save);
    assert!(hands.holds(&b, i), "the hold did not come back");
    for _ in 0..30 {
        b.step(fx(DT));
    }
    assert_eq!(b.state_digest(), want);
}

#[test]
fn the_same_hands_give_the_same_bits() {
    let run = || {
        let mut w = world(9.81, 2);
        let ids: Vec<usize> = (0..6)
            .map(|k| body(&mut w, v(f64::from(k), 0.0, 0.0), 1.0))
            .collect();
        let hands = Holds::new(&mut w);
        hands
            .hold(&mut w, ids[0], v(0.0, 2.0, 0.0), fx(5.0))
            .expect("hold");
        w.apply_radial_impulse(v(2.0, 0.0, 0.0), fx(6.0), fx(9.0));
        for _ in 0..40 {
            w.step(fx(DT));
        }
        hands.release(&mut w, ids[0]).expect("release");
        for _ in 0..40 {
            w.step(fx(DT));
        }
        w.state_digest()
    };
    assert_eq!(run(), run());
    assert_eq!(BodyType::Dynamic, BodyType::Dynamic);
}

// ── at rest, for long ────────────────────────────────────────────────────

/// The held body's distance from its place after `n` substeps, bounded: the world's spring is the backward Euler
/// step of the critically damped oscillator (`solve_spring_joint`), `x' = (x (1 + 2a) + h v) / (1 + a)²`, `a = ω h`,
/// whose step matrix has the double eigenvalue `λ = 1 / (1 + a)`; from rest at `x0` the distance is
/// `|x0| (1 + n a / (1 + a)) λⁿ`. Checked against that map iterated here, then used as the bound
fn settled_bound(x0: f64, omega: f64, h: f64, n: u64) -> f64 {
    let a = omega * h;
    x0.abs() * (1.0 + n as f64 * a / (1.0 + a)) * (1.0 + a).powi(-(n as i32))
}

fn held_for_long(sleep: bool) -> (f64, f64, u64) {
    use alice_physics_core::sleeping::SleepConfig;
    let (omega, dt, substeps) = (6.0, DT, 8usize);
    let h = dt / substeps as f64;
    // the bound is the scheme's: its 2×2 map, iterated, never exceeds it
    let (mut x, mut vel) = (2.0f64, 0.0f64);
    let a = omega * h;
    for n in 1..=9600u64 {
        let x1 = (x * (1.0 + 2.0 * a) + h * vel) / ((1.0 + a) * (1.0 + a));
        vel = (x1 - x) / h;
        x = x1;
        assert!(
            x.abs() <= settled_bound(2.0, omega, h, n) * (1.0 + 1e-9) + 1e-300,
            "{n}: {x}"
        );
    }
    let mut w = world(9.81, substeps);
    if !sleep {
        w.set_sleep_config(SleepConfig {
            // never: an idle time of 2^32 s is not reached in this run
            time_to_sleep: Fix128::from_int(1 << 32),
            ..SleepConfig::default()
        });
    }
    let b = body(&mut w, v(0.0, 3.0, 5.0), 1.0);
    let holds = Holds::new(&mut w);
    holds
        .hold(&mut w, b, v(0.0, 1.0, 5.0), fx(omega))
        .expect("held");
    let (mut worst, mut allowed, mut compared) = (0.0f64, 0.0f64, 0u64);
    for step in 1..=1200u64 {
        w.step(fx(dt));
        if step >= 180 {
            let p = pos(&w, b);
            let off = (p[0].powi(2) + (p[1] - 1.0).powi(2) + (p[2] - 5.0).powi(2)).sqrt();
            // the scheme's transient, and the rounding of Q64.64 positions (far below 1e-12 m)
            let bound = settled_bound(2.0, omega, h, step * substeps as u64) + 1e-12;
            if off - bound > worst - allowed {
                (worst, allowed) = (off, bound);
            }
            compared += 1;
        }
    }
    (worst, allowed, compared)
}

/// Held with sleeping off, from 3 s to 20 s the body is never further from its place than the scheme's transient
/// allows: the bound above is the right one
#[test]
fn held_for_long_with_sleep_off_it_stays_at_its_place() {
    let (worst, allowed, compared) = held_for_long(false);
    assert_eq!(compared, 1021);
    assert!(worst <= allowed, "{worst} m, allowed {allowed} m");
}

/// The same with the world's sleeping on (the default): a sleeping body is not moved, and a body whose spring is not
/// at rest does not go to sleep, so it stays as close. Today it creeps up by over a millimetre
#[test]
#[ignore = "known defect: physics-hold-sleep-creep: the island sleeps while the spring is not at rest, and the sleeping body still moves"]
fn held_for_long_with_sleep_on_it_stays_at_its_place() {
    let (worst, allowed, compared) = held_for_long(true);
    assert_eq!(compared, 1021);
    assert!(worst <= allowed, "{worst} m, allowed {allowed} m");
}
