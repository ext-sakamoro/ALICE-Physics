//! Rigid bodies exchanging momentum with a medium through linear drag.
//!
//! Three free bodies (masses 1, 2, 4) are coupled to a medium of mass 3 that
//! starts moving the other way. The example prints the relative velocity of
//! the first body against the closed form of the explicit exchange and the
//! total momentum `Σ m v + P` on XPBD and TGS: TGS keeps it bit for bit,
//! XPBD to within the rounding of its velocity re-derivation.
//!
//! Run: `cargo run --example coupling_medium`

use alice_physics::coupling_medium::{DragMedium, MEDIUM_OBS_MOMENTUM, MEDIUM_OBS_VELOCITY};
use alice_physics::sleeping::SleepConfig;
use alice_physics::world_participant::Observed;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix};

const MASSES: [i64; 3] = [1, 2, 4];

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

fn total(w: &PhysicsWorld) -> Vec3Fix {
    let mut p = channel(w, MEDIUM_OBS_MOMENTUM);
    for (b, &m) in w.bodies.iter().zip(&MASSES) {
        p = p + b.velocity * Fix128::from_int(m);
    }
    p
}

fn run(backend: SolverBackend) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps: 4,
        solver_backend: backend,
        ..PhysicsConfig::default()
    });
    w.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    let mut medium =
        DragMedium::new(Fix128::from_int(3), Vec3Fix::from_int(-2, 0, 0)).expect("positive mass");
    for (k, &m) in MASSES.iter().enumerate() {
        let mut b =
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 100 * k as i64, 0), Fix128::from_int(m));
        b.velocity = Vec3Fix::from_int(1 + k as i64, 0, 0);
        w.add_body(b);
        medium
            .couple(k, Fix128::from_ratio(1 + k as i64, 2))
            .expect("non-negative coefficient");
    }
    let initial = medium
        .total_momentum(&w.bodies)
        .expect("coupled bodies exist");
    println!(
        "{backend:?}: M = {}, P = {:.3}, {} couplings, initial Σ m v + P = {:.3}",
        medium.mass().to_f64(),
        medium.momentum().x.to_f64(),
        medium.couplings().len(),
        initial.x.to_f64()
    );
    w.add_participant(Box::new(medium)).expect("register");

    let start = total(&w);
    assert_eq!(start, initial, "the medium's sum and the one here agree");
    println!("{backend:?}: total momentum x = {:.15}", start.x.to_f64());
    println!("  frame   v0 - u        u (medium)    |Σ m v + P - start|");
    let dt = Fix128::from_ratio(1, 64);
    for frame in 1..=240 {
        w.try_step(dt).expect("step");
        if frame % 40 == 0 {
            let u = channel(&w, MEDIUM_OBS_VELOCITY);
            let d = total(&w) - start;
            let dev =
                d.x.to_f64()
                    .abs()
                    .max(d.y.to_f64().abs())
                    .max(d.z.to_f64().abs());
            println!(
                "  {frame:5}   {:+.9}  {:+.9}  {dev:.3e}",
                (w.bodies[0].velocity.x - u.x).to_f64(),
                u.x.to_f64()
            );
        }
    }
    if backend == SolverBackend::Tgs {
        assert_eq!(total(&w), start, "TGS keeps Σ m v + P bit for bit");
        println!("  Σ m v + P unchanged bit for bit");
    }
}

fn main() {
    run(SolverBackend::Xpbd);
    run(SolverBackend::Tgs);
}
