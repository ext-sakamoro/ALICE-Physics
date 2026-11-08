//! A visitor's touch on a body (grab at a point, drag the point, let go), as inputs a log carries: what each does to
//! a cloth held to its rest shape, against the closed form, and that the state digest covers it.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::cloth::Cloth;
use alice_physics_core::input_log::{InputLog, LogInput};
use alice_physics_core::touch::{Hand, Touch};
use alice_physics_core::{Fix128, Vec3Fix};

/// Two particles 100 apart (the bond between them hardly moves when one is drawn aside), no gravity, no drag of
/// their own, each held to its rest place by `k = m ω²`, `c = 2 ζ m ω` (ω = 2π rad/s here: period 1 s, ζ = 0.4)
fn sheet() -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(100),
        Fix128::ONE,
        2,
        1,
        Fix128::ONE,
    );
    c.config.gravity = Vec3Fix::ZERO;
    c.config.damping = Fix128::ONE;
    let w = Fix128::from_f64(std::f64::consts::TAU);
    c.set_rest_tether(w * w, Fix128::from_int(2) * w * Fix128::from_ratio(2, 5));
    c
}

const DT: f64 = 1.0 / 60.0;

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
    x0 * (-z * w * t).exp() * ((wd * t).cos() + z * w / wd * (wd * t).sin())
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
    let changed: [(&str, fn(&mut Cloth, Fix128)); 6] = [
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
