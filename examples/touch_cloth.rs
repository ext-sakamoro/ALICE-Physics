//! A touch on a cloth held to its rest shape: grab a point, draw it aside, let go, and watch it spring back. The touches
//! go into a log; the log's code replays to the same cloth.
//!
//!     cargo run --release --example touch_cloth
//!
//! Author: Moroya Sakamoto

use alice_physics_core::cloth::Cloth;
use alice_physics_core::input_log::InputLog;
use alice_physics_core::touch::{Grabbable, Hand, Touch};
use alice_physics_core::{Fix128, Vec3Fix};

fn sheet() -> Cloth {
    let mut c = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::ONE,
        5,
        5,
        Fix128::from_ratio(1, 10),
    );
    c.config.gravity = Vec3Fix::ZERO;
    let w = Fix128::from_int(12);
    let m = Fix128::from_ratio(1, 10);
    c.set_rest_tether(
        m * w * w,
        Fix128::from_int(2) * m * w * Fix128::from_ratio(2, 5),
    );
    c
}

/// Play `log` for `steps`, calling `seen` after each step with the step, the bodies and the hand
fn play<B: Grabbable>(
    bodies: &mut [B],
    log: &InputLog<Touch>,
    steps: u64,
    step: impl Fn(&mut B),
    mut seen: impl FnMut(u64, &[B], &Hand),
) -> Vec<u8> {
    let mut hand = Hand::new(Fix128::from_ratio(1, 5));
    let mut next = 0;
    for s in 0..steps {
        while let Some((_, t)) = log.entries().get(next).filter(|(at, _)| *at == s) {
            hand.apply(bodies, t);
            next += 1;
        }
        for b in bodies.iter_mut() {
            step(b);
        }
        seen(s, bodies, &hand);
    }
    hand.state_bytes()
}

fn main() {
    let mut log = InputLog::new();
    let p = |x: f64, y: f64| Vec3Fix::new(Fix128::from_f64(x), Fix128::from_f64(y), Fix128::ZERO);
    log.record(5, Touch::grab(0, p(0.5, 0.0)))
        .expect("in order");
    log.record(6, Touch::drag(p(0.5, 0.3))).expect("in order");
    log.record(30, Touch::Release).expect("in order");
    let dt = Fix128::from_ratio(1, 60);
    let mut bodies = vec![sheet()];
    let hand = play(
        &mut bodies,
        &log,
        90,
        |c: &mut Cloth| c.step(dt),
        |s, b, h| {
            if s == 29 {
                println!(
                    "held {:?}: the middle of the edge at y = {:.4}",
                    h.held(),
                    b[0].positions[2].y.to_f64()
                );
            }
        },
    );
    println!(
        "a second after it is let go: y = {:.6}",
        bodies[0].positions[2].y.to_f64()
    );
    let code = log.to_code(&bodies[0].state_digest());
    let (_, again) = InputLog::<Touch>::from_code(&code).expect("a code");
    let mut replay = vec![sheet()];
    let hand_again = play(
        &mut replay,
        &again,
        90,
        |c: &mut Cloth| c.step(dt),
        |_, _, _| {},
    );
    println!(
        "the code {code} replays to the same cloth and hand: {}",
        replay[0].state_digest() == bodies[0].state_digest() && hand_again == hand
    );
}
