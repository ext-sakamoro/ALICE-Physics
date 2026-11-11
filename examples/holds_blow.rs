//! A blow, a hold and a log: a body is hit from a point, another is held at a place and settles there under its
//! weight, and the hands are written as inputs of a log that replays the run to the same digest.
//!
//! Checked as it runs: the blow's velocity against the impulse it states (`s (1 − d/R) / m`), the held body at its
//! place within 1 mm after 4 s (critically damped at ω = 6 rad/s, so `e^(−24)` of the start is left), and the replay's
//! digest against the run's, bit for bit. The full oracles are `tests/analytic_holds.rs` and `tests/input_log.rs`.
//!
//! ```bash
//! cargo run --example holds_blow
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics_core::holds::Holds;
use alice_physics_core::input_log::{InputLog, LogInput};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

const DT: f64 = 1.0 / 60.0;

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: v(0.0, -9.81, 0.0),
        ..PhysicsConfig::default()
    });
    w.add_body(RigidBody::new(v(2.0, 0.0, 0.0), Fix128::from_f64(2.0)));
    w.add_body(RigidBody::new(v(0.0, 3.0, 5.0), Fix128::from_f64(1.0)));
    w
}

/// The hands, as bytes a host writes: 0 a blow from the origin of radius 4 and strength 6, 1 hold body 1 at (0, 1, 5)
fn hand(w: &mut PhysicsWorld, holds: &Holds, input: &[u8]) {
    match input.first() {
        Some(0) => {
            w.apply_radial_impulse(
                v(0.0, 0.0, 0.0),
                Fix128::from_f64(4.0),
                Fix128::from_f64(6.0),
            );
        }
        Some(1) => {
            let _ = holds.hold(w, 1, v(0.0, 1.0, 5.0), Fix128::from_f64(6.0));
        }
        _ => {}
    }
}

fn main() {
    let mut w = scene();
    let holds = Holds::new(&mut w);
    let mut log: InputLog<Vec<u8>> = InputLog::new();
    let dt = Fix128::from_f64(DT);
    let steps = 240;
    for step in 0..steps {
        if step == 0 || step == 1 {
            let input = vec![step as u8];
            hand(&mut w, &holds, &input);
            log.record(step, input).expect("the steps go forward");
        }
        if step == 0 {
            // the blow, before any step: (6 (1 − 2/4)) / 2 kg = 1.5 m/s along x
            let got = w.bodies[0].velocity.x.to_f64();
            println!("blow: {got:.6} m/s (stated 1.5)");
            assert!((got - 1.5).abs() < 1e-9, "{got}");
        }
        w.step(dt);
    }
    let held = w.bodies[1].position;
    let off = (held - v(0.0, 1.0, 5.0)).length().to_f64();
    println!(
        "held: {:.6} m from the place, held {}",
        off,
        holds.holds(&w, 1)
    );
    assert!(holds.holds(&w, 1) && off < 1e-3, "{off}");

    // an input's bytes read back as the input
    let mut bytes = Vec::new();
    vec![1u8, 2, 3].write(&mut bytes);
    let mut rest = bytes.as_slice();
    assert_eq!(Vec::<u8>::read(&mut rest), Some(vec![1, 2, 3]));
    assert!(rest.is_empty());

    // the log as a code, and replayed from nothing to the same digest
    let digest = w.state_digest();
    let code = log.to_code(&digest);
    let (head, back) = InputLog::<Vec<u8>>::from_code(&code).expect("the code reads");
    assert_eq!(head, [digest[0], digest[1], digest[2], digest[3]]);
    let mut again = scene();
    let holds_again = Holds::new(&mut again);
    back.replay(&mut again, steps, dt, |world, input| {
        hand(world, &holds_again, input)
    });
    let same = again.state_digest() == digest;
    println!(
        "replayed from the code ({} characters): same digest {same}",
        code.len()
    );
    assert!(same);
}
