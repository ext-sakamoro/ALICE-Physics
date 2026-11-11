//! A visitor's touch on a body (grab at a point, drag the point, let go), as inputs a log carries: what each does to
//! a cloth held to its rest shape, against the closed form, and that the state digest covers it.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::cloth::Cloth;
use alice_physics_core::det_math::exp64;
use alice_physics_core::input_log::{InputLog, LogInput};
use alice_physics_core::touch::{Hand, Touch};
use alice_physics_core::{Fix128, Vec3Fix};

/// Two particles 100 apart (the bond between them hardly moves when one is drawn aside), no gravity, no drag of
/// their own, each held to its rest place by `k = m ω²`, `c = 2 ζ m ω` (ω = 2π rad/s here: period 1 s, ζ = 0.4)
fn sheet() -> Cloth {
    sheet_of(1.0)
}

/// The same with particles of mass `m` (kg), the tether's ω and ζ kept: `k = m ω²`, `c = 2 ζ m ω`
fn sheet_of(m: f64) -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(100),
        Fix128::ONE,
        2,
        1,
        Fix128::from_f64(m),
    );
    c.config.gravity = Vec3Fix::ZERO;
    c.config.damping = Fix128::ONE;
    let w = Fix128::from_f64(std::f64::consts::TAU);
    let m = Fix128::from_f64(m);
    c.set_rest_tether(
        m * w * w,
        Fix128::from_int(2) * m * w * Fix128::from_ratio(2, 5),
    );
    c
}

const DT: f64 = 1.0 / 60.0;

/// `(sin x, cos x)` by Taylor series after reduction to `[−π, π]` (the platform `f64::sin` is not used in this crate's
/// tests; error ≲ 1e-15 here), as in `tests/analytic_cloth_rest_tether.rs`
fn sin_cos64(x: f64) -> (f64, f64) {
    let tau = 2.0 * core::f64::consts::PI;
    let mut r = x.rem_euclid(tau);
    if r > core::f64::consts::PI {
        r -= tau;
    }
    let (mut s, mut c) = (0.0_f64, 0.0_f64);
    let mut term = 1.0_f64;
    for n in 0..40_u32 {
        match n % 4 {
            0 => c += term,
            1 => s += term,
            2 => c -= term,
            _ => s -= term,
        }
        term *= r / f64::from(n + 1);
    }
    (s, c)
}

/// A change to one part of a cloth's state, by a small amount
type Change = (&'static str, fn(&mut Cloth, Fix128));

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

// oracle: the underdamped oscillator let go from rest at x0, x(t) = x0 e^(−ζωt) (cos ω_d t + ζω/ω_d sin ω_d t)
fn let_go(x0: f64, t: f64) -> f64 {
    let (w, z) = (std::f64::consts::TAU, 0.4);
    let wd = w * (1.0f64 - z * z).sqrt();
    let (sn, cs) = sin_cos64(wd * t);
    x0 * exp64(-z * w * t) * (cs + z * w / wd * sn)
}

#[test]
fn a_letter_let_go_springs_back_by_its_tether() {
    let mut bodies = vec![sheet()];
    let mut hand = Hand::new(Fix128::from_ratio(1, 2));
    assert!(hand.apply(&mut bodies, &Touch::grab(0, v(0.1, 0.0, 0.0))));
    assert_eq!(hand.held(), Some((0, 0)), "the particle nearest the touch");
    // drawn 0.2 aside and held there for a while: it stays where the hand is
    hand.apply(&mut bodies, &Touch::drag(v(0.0, 0.2, 0.0)));
    for _ in 0..30 {
        bodies[0].step(Fix128::from_f64(DT));
    }
    // where the hand is, on the grid: within half a grid step of 0.2
    let y = bodies[0].positions[0].y.to_f64();
    assert!((y - 0.2).abs() <= 0.5 / 65536.0, "held at {y}");
    hand.apply(&mut bodies, &Touch::Release);
    assert_eq!(hand.held(), None);
    let mut worst = 0.0f64;
    for n in 1..=90 {
        bodies[0].step(Fix128::from_f64(DT));
        let have = bodies[0].positions[0].y.to_f64();
        worst = worst.max((have - let_go(y, n as f64 * DT)).abs());
    }
    println!("let go from 0.2: worst {worst:.2e} over 1.5 s");
    assert!(worst < 1e-12, "{worst}");
}

#[test]
fn a_touch_out_of_reach_takes_nothing() {
    let mut bodies = vec![sheet()];
    let before = bodies[0].state_digest();
    let mut hand = Hand::new(Fix128::from_ratio(1, 10));
    assert!(!hand.apply(&mut bodies, &Touch::grab(0, v(50.0, 3.0, 0.0))));
    assert!(
        !hand.apply(&mut bodies, &Touch::grab(7, v(0.0, 0.0, 0.0))),
        "no such body"
    );
    assert!(
        !hand.apply(&mut bodies, &Touch::drag(v(1.0, 1.0, 0.0))),
        "nothing held"
    );
    assert!(!hand.apply(&mut bodies, &Touch::Release), "nothing held");
    assert_eq!(hand.held(), None);
    assert_eq!(bodies[0].state_digest(), before, "nothing changed");
    // a reach of zero or less takes nothing either
    let mut none = Hand::new(Fix128::ZERO);
    assert!(!none.apply(&mut bodies, &Touch::grab(0, v(0.0, 0.0, 0.0))));
}

/// With both particles in reach, a touch takes the nearer one, whichever index it has (30 away against 70, then
/// 70 against 30)
#[test]
fn a_touch_takes_the_nearer_of_two_particles_in_reach() {
    for (x, want) in [(70.0, 1usize), (30.0, 0usize)] {
        let mut bodies = vec![sheet()];
        let mut hand = Hand::new(Fix128::from_int(200));
        assert!(hand.apply(&mut bodies, &Touch::grab(0, v(x, 0.0, 0.0))));
        assert_eq!(hand.held(), Some((0, want)), "touch at x = {x}");
    }
}

/// A particle pinned before it is grabbed stays pinned when it is let go
#[test]
fn letting_go_of_a_pinned_point_leaves_it_pinned() {
    let mut bodies = vec![sheet()];
    bodies[0].pin(0);
    let mut hand = Hand::new(Fix128::ONE);
    hand.apply(&mut bodies, &Touch::grab(0, v(0.0, 0.0, 0.0)));
    hand.apply(&mut bodies, &Touch::Release);
    assert!(bodies[0].pinned.contains(&0));
    assert!(bodies[0].inv_masses[0].is_zero());
}

fn played(log: &InputLog<Touch>, steps: u64) -> [u8; 32] {
    let mut bodies = vec![sheet(), sheet()];
    let mut hand = Hand::new(Fix128::from_ratio(1, 2));
    let mut next = 0;
    for step in 0..steps {
        while let Some((at, t)) = log.entries().get(next) {
            if *at != step {
                break;
            }
            hand.apply(&mut bodies, t);
            next += 1;
        }
        for b in &mut bodies {
            b.step(Fix128::from_f64(DT));
        }
    }
    let mut all = Vec::new();
    for b in &bodies {
        all.extend_from_slice(&b.state_digest());
    }
    all.extend_from_slice(&hand.state_bytes());
    let mut d = [0u8; 32];
    for (i, b) in all.iter().enumerate() {
        d[i % 32] ^= b.rotate_left((i / 32) as u32);
    }
    d
}

fn log(drag_y: f64) -> InputLog<Touch> {
    let mut l = InputLog::new();
    l.record(3, Touch::grab(1, v(100.0, 0.0, 0.0))).unwrap();
    l.record(4, Touch::drag(v(100.0, drag_y, 0.0))).unwrap();
    l.record(20, Touch::Release).unwrap();
    l
}

/// The log replays to the same state, a different drag to a different one, and the code carries it whole
#[test]
fn the_digest_covers_what_the_hand_did() {
    let a = log(0.3);
    assert_eq!(played(&a, 60), played(&a, 60));
    assert_ne!(
        played(&a, 60),
        played(&log(0.31), 60),
        "a drag 1 cm further is another state"
    );
    // still held at step 10: the hand's own state counts
    assert_ne!(played(&a, 10), played(&InputLog::new(), 10));
    let code = a.to_code(&[7; 32]);
    let (_, back) = InputLog::<Touch>::from_code(&code).unwrap();
    assert_eq!(back, a);
    assert_eq!(played(&back, 60), played(&a, 60));
}

/// Positions are carried on a grid of 2⁻¹⁶ m (15 µm): what is written is read back exactly, and a touch made from
/// a float lands on the grid
#[test]
fn touches_are_written_on_their_grid_and_read_back_exactly() {
    for t in [
        Touch::grab(0, v(0.123_456, -2.5, 1e-7)),
        Touch::grab(u32::MAX, v(-1000.0, 0.0, 3.0)),
        Touch::drag(v(0.5, 0.25, -0.125)),
        Touch::Release,
    ] {
        let mut bytes = Vec::new();
        t.write(&mut bytes);
        let mut rest = bytes.as_slice();
        assert_eq!(Touch::read(&mut rest), Some(t));
        assert!(rest.is_empty());
        if let Touch::Grab { at: p, .. } | Touch::Drag { to: p } = t {
            for c in [p.x, p.y, p.z] {
                assert_eq!(c.lo & ((1u64 << 48) - 1), 0, "{c:?} is on the grid");
            }
        }
    }
    // one rounding: within half a grid step of what was asked
    let Touch::Drag { to } = Touch::drag(v(0.123_456, 0.0, 0.0)) else {
        unreachable!()
    };
    assert!((to.x.to_f64() - 0.123_456).abs() <= 0.5 / 65536.0);
    // not a touch
    for bad in [&[][..], &[9][..], &[1][..], &[2, 0x80][..]] {
        let mut rest = bad;
        assert_eq!(Touch::read(&mut rest), None, "{bad:?}");
    }
}

/// Each part of a cloth's state moves its digest on its own: a position, a previous position, a velocity, a mass, a
/// pin, the tether
#[test]
fn every_part_of_the_cloth_is_in_its_digest() {
    let base = sheet().state_digest();
    let tiny = Fix128::from_ratio(1, 1 << 20);
    let changed: [Change; 6] = [
        ("position", |c, t| c.positions[1].x = c.positions[1].x + t),
        ("previous position", |c, t| {
            c.prev_positions[1].y = c.prev_positions[1].y + t
        }),
        ("velocity", |c, t| c.velocities[0].z = c.velocities[0].z + t),
        ("mass", |c, t| c.inv_masses[0] = c.inv_masses[0] + t),
        ("pin", |c, _| c.pinned.push(1)),
        ("tether", |c, t| {
            let (k, d) = c.rest_tether().unwrap();
            c.set_rest_tether(k + t, d);
        }),
    ];
    for (what, f) in changed {
        let mut c = sheet();
        f(&mut c, tiny);
        assert_ne!(c.state_digest(), base, "{what}");
    }
}

/// A hand holding with a finite grip: a spring of stiffness `kg` from the point to the hand, which lets go when its
/// pull reaches `fmax`. The point is held to rest by the cloth's tether `kt` (sheet(): kt = m ω², m = 1, ω = 2π)
fn gripped(kg: f64, fmax: f64) -> (Vec<Cloth>, Hand) {
    let bodies = vec![sheet()];
    let hand = Hand::with_grip(
        Fix128::from_ratio(1, 2),
        Fix128::from_f64(kg),
        Fix128::from_f64(fmax),
    );
    (bodies, hand)
}

fn steps(bodies: &mut [Cloth], hand: &mut Hand, n: usize) {
    steps_at(bodies, hand, n, DT);
}

fn steps_at(bodies: &mut [Cloth], hand: &mut Hand, n: usize, dt: f64) {
    for _ in 0..n {
        hand.pull(bodies, Fix128::from_f64(dt));
        bodies[0].step(Fix128::from_f64(dt));
    }
}

// oracle: the scheme's own rest. Each step the grip gives the backward Euler impulse of its spring,
// J = kg dt (d − x − dt v) / (1 + kg w dt²), a velocity kick v ← (1 − c dt) v + c (d − x) with c = kg dt w / (1 + kg w dt²)
// (w = 1/m), then the tether runs exactly: (x, v) ← Φ (x, v), Φ the underdamped oscillator over dt,
//   x(t) = e^(−ζωt) [x0 cos ω_d t + (v0 + ζω x0) / ω_d sin ω_d t]
//   v(t) = e^(−ζωt) [v0 cos ω_d t − (ω² x0 + ζω v0) / ω_d sin ω_d t]
// Its fixed point solves two linear equations; as dt → 0 it is the continuous balance kg (d − x) = kt x
fn scheme_rest_at(kg: f64, d: f64, dt: f64, m: f64) -> f64 {
    let (w, z) = (std::f64::consts::TAU, 0.4);
    let wd = w * (1.0f64 - z * z).sqrt();
    let e = exp64(-z * w * dt);
    let (sn, cs) = sin_cos64(wd * dt);
    let (a11, a12) = (e * (cs + z * w / wd * sn), e * sn / wd);
    let (a21, a22) = (-e * w * w / wd * sn, e * (cs - z * w / wd * sn));
    let c = kg * dt / m / (1.0 + kg * dt * dt / m);
    let keep = 1.0 - c * dt;
    // x = a11 x + a12 (keep v + c (d − x)),  v = a21 x + a22 (keep v + c (d − x))
    let (p, q, r) = (1.0 - a11 + a12 * c, -a12 * keep, a12 * c * d);
    let (s, t, u) = (-a21 + a22 * c, 1.0 - a22 * keep, a22 * c * d);
    (r * t - q * u) / (p * t - q * s)
}

fn scheme_rest(kg: f64, d: f64) -> f64 {
    scheme_rest_at(kg, d, DT, 1.0)
}

// oracle: two springs in series against the hand, at rest: kg (d − x) = kt x, so x = kg d / (kg + kt) (continuous),
// and the scheme's rest above (discrete), to which the cloth comes
#[test]
fn a_finite_grip_holds_the_point_where_the_springs_balance() {
    let kt = std::f64::consts::TAU * std::f64::consts::TAU;
    for (kg, d) in [(40.0, 0.05), (100.0, 0.03), (20.0, 0.08)] {
        let (mut bodies, mut hand) = gripped(kg, 1000.0);
        assert!(hand.apply(&mut bodies, &Touch::grab(0, v(0.0, 0.0, 0.0))));
        hand.apply(&mut bodies, &Touch::drag(v(0.0, d, 0.0)));
        // it is not pinned: the cloth moves it, the grip only pulls
        assert!(!bodies[0].pinned.contains(&0));
        steps(&mut bodies, &mut hand, 600);
        let x = bodies[0].positions[0].y.to_f64();
        let d_grid = match Touch::drag(v(0.0, d, 0.0)) {
            Touch::Drag { to } => to.y.to_f64(),
            _ => unreachable!(),
        };
        let balance = kg * d_grid / (kg + kt);
        let want = scheme_rest(kg, d_grid);
        println!("kg {kg} d {d}: held at {x:.9}, the scheme's rest {want:.9}, the continuous balance {balance:.9}");
        // the cloth settles onto the scheme's rest (600 steps of a decay e^(−ζω t) at ζω ≈ 2.5/s leave e^(−25) of it)
        assert!((x - want).abs() < 1e-9, "kg {kg}: {x} against {want}");
        // and the scheme is the continuous law to first order in the step: within (kg/m) dt² of the balance per unit d
        assert!(
            (want - balance).abs() < kg * DT * DT * d_grid,
            "kg {kg}: {want} against {balance}"
        );
        assert!(hand.held().is_some(), "still held");
    }
}

// oracle: a slow drag lets go where the grip's pull at balance reaches fmax: kg kt d / (kg + kt) = fmax
#[test]
fn a_finite_grip_slips_where_its_pull_reaches_its_strength() {
    let kt = std::f64::consts::TAU * std::f64::consts::TAU;
    let (kg, fmax) = (60.0, 2.0);
    let d_slip = fmax * (kg + kt) / (kg * kt);
    let (mut bodies, mut hand) = gripped(kg, fmax);
    hand.apply(&mut bodies, &Touch::grab(0, v(0.0, 0.0, 0.0)));
    // the hand moves 0.1 mm a step: slow beside the springs' periods, so the point is near its balance throughout
    let mut slipped_at = None;
    for n in 1..=4000 {
        let d = n as f64 * 1e-4;
        hand.apply(&mut bodies, &Touch::drag(v(0.0, d, 0.0)));
        steps(&mut bodies, &mut hand, 1);
        if hand.held().is_none() {
            slipped_at = Some(d);
            break;
        }
    }
    let at = slipped_at.expect("the grip never let go");
    // the scheme's rest is linear in d, so it slips where kg (d − x(d)) = fmax: d = fmax / (kg (1 − x(1)))
    let d_scheme = fmax / (kg * (1.0 - scheme_rest(kg, 1.0)));
    println!("slipped at d = {at:.5}, the scheme's {d_scheme:.5}, the continuous {d_slip:.5}");
    // the drag moves 1e-4 a step and the point trails its rest by the drag's speed over the decay rate (1e-4 / (ζω dt)
    // ≈ 2.4e-3 of travel at most, reached only after many steps): it slips within that of the scheme's point, early
    assert!(
        at <= d_scheme + 1e-4 && d_scheme - at < 1e-4 / (0.4 * std::f64::consts::TAU * DT),
        "{at} against {d_scheme}"
    );
    // and the scheme's point is the continuous one to first order in the step
    assert!(
        (d_scheme - d_slip).abs() < kg * DT * DT * d_slip,
        "{d_scheme} against {d_slip}"
    );
    // let go, the point is the cloth's again and springs back to rest
    steps(&mut bodies, &mut hand, 600);
    assert!(bodies[0].positions[0].y.to_f64().abs() < 1e-3);
}

/// Degenerate grips take nothing: no strength, no stiffness, or not finite
#[test]
fn a_grip_of_no_strength_takes_nothing() {
    for (kg, fmax) in [(0.0, 1.0), (10.0, 0.0), (-1.0, 1.0), (10.0, -1.0)] {
        let (mut bodies, mut hand) = gripped(kg, fmax);
        assert!(
            !hand.apply(&mut bodies, &Touch::grab(0, v(0.0, 0.0, 0.0))),
            "kg {kg} fmax {fmax}"
        );
        assert_eq!(hand.held(), None);
    }
}

/// The scheme's rest comes to the continuous balance `kg d / (kg + kt)` as the step shrinks, at second order. The grip's
/// impulse and the tether's exact step are taken one after the other (a Lie splitting, O(dt) along a motion), but at a
/// fixed point the Lie splitting's state is the symmetric (Strang) splitting's carried by half a tether step: the Strang
/// fixed point is within O(dt²) of the balance, and half a step from a state at rest moves it by O(dt²) too. So halving
/// the step quarters the distance from the balance. The cloth comes to the scheme's rest at each step size
#[test]
fn the_grip_comes_to_the_continuous_balance_as_the_step_shrinks() {
    let kt = std::f64::consts::TAU * std::f64::consts::TAU;
    let (kg, d) = (40.0, 0.05);
    let balance = kg * d / (kg + kt);
    let mut errors = Vec::new();
    for halvings in 0..4 {
        let dt = DT / f64::from(1u32 << halvings);
        let want = scheme_rest_at(kg, d, dt, 1.0);
        errors.push((want - balance).abs());
        if halvings < 2 {
            let (mut bodies, mut hand) = gripped(kg, 1000.0);
            hand.apply(&mut bodies, &Touch::grab(0, v(0.0, 0.0, 0.0)));
            hand.apply(&mut bodies, &Touch::drag(v(0.0, d, 0.0)));
            steps_at(&mut bodies, &mut hand, 600 << halvings, dt);
            let x = bodies[0].positions[0].y.to_f64();
            // the drag's grid (2^-16) moves the balance by under 1e-5 of d; the cloth settles onto the scheme's rest
            let d_grid = match Touch::drag(v(0.0, d, 0.0)) {
                Touch::Drag { to } => to.y.to_f64(),
                _ => unreachable!(),
            };
            let want_grid = scheme_rest_at(kg, d_grid, dt, 1.0);
            assert!(
                (x - want_grid).abs() < 1e-9,
                "dt {dt}: {x} against {want_grid}"
            );
        }
    }
    println!("distance from the balance as the step halves: {errors:?}");
    for w in errors.windows(2) {
        let ratio = w[0] / w[1];
        assert!((3.6..4.4).contains(&ratio), "{errors:?}");
    }
}

/// A light point under the law's grip: 0.1 kg at kg = 5181 N/m and a 1/60 s step, `ω dt = √(kg / m) dt ≈ 3.8`, past
/// where an explicit spring kick is stable (2). The backward Euler grip holds it at the scheme's rest, still held
#[test]
fn a_light_point_under_a_stiff_grip_stays_where_the_springs_balance() {
    let (m, kg, d): (f64, f64, f64) = (0.1, 5181.454930295214, 0.002);
    assert!(
        (kg / m).sqrt() * DT > 2.0,
        "the case is past the explicit bound"
    );
    let bodies = vec![sheet_of(m)];
    let mut hand = Hand::with_grip(
        Fix128::from_ratio(1, 2),
        Fix128::from_f64(kg),
        Fix128::from_f64(1000.0),
    );
    let mut bodies = bodies;
    assert!(hand.apply(&mut bodies, &Touch::grab(0, v(0.0, 0.0, 0.0))));
    hand.apply(&mut bodies, &Touch::drag(v(0.0, d, 0.0)));
    let d_grid = match Touch::drag(v(0.0, d, 0.0)) {
        Touch::Drag { to } => to.y.to_f64(),
        _ => unreachable!(),
    };
    let mut largest = 0.0f64;
    for _ in 0..600 {
        steps(&mut bodies, &mut hand, 1);
        largest = largest.max(bodies[0].positions[0].y.to_f64().abs());
    }
    let x = bodies[0].positions[0].y.to_f64();
    let want = scheme_rest_at(kg, d_grid, DT, m);
    println!("light point: held at {x:.9}, the scheme's rest {want:.9}, largest {largest:.6}");
    assert!(hand.held().is_some(), "still held");
    // never further than the hand: a stable spring does not overshoot past twice the pull
    assert!(largest <= 2.0 * d_grid, "{largest}");
    assert!((x - want).abs() < 1e-9, "{x} against {want}");
}
