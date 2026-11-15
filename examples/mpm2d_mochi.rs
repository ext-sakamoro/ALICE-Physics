//! A soft 2D blob (mochi) with the material point method (feature `mpm2d`).
//!
//! A disk seeded from its signed distance falls onto the ground, a hand presses
//! the top of it through the particles within its radius, and the pressed shape, rebuilt
//! at rest and held by a damped elastic foundation, is then stepped with the implicit solver at a frame's
//! step. Mass is the same at every stage, the blob never goes through the ground, and a
//! snapshot taken at the end restores to the same digest.
//!
//! ```bash
//! cargo run --release --example mpm2d_mochi --features mpm2d
//! ```

use alice_physics_core::math::Fix128;
use alice_physics_core::mpm2d::{ImplicitSolve, Mpm2d, Mpm2dConfig, Mpm2dMaterial, Mpm2dModel};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

const CELL: f64 = 0.01;
const GROUND: f64 = 0.05;
const RADIUS: f64 = 0.04;

fn config(dt: f64) -> Mpm2dConfig {
    Mpm2dConfig {
        origin: [Fix128::ZERO, Fix128::ZERO],
        cell: fx(CELL),
        nodes: [24, 40],
        gravity: [Fix128::ZERO, fx(-9.81)],
        dt: fx(dt),
        ground: Some(fx(GROUND)),
    }
}

fn material() -> Mpm2dMaterial {
    Mpm2dMaterial {
        youngs: fx(2.0e4),
        poisson: fx(0.3),
        density: fx(1000.0),
        model: Mpm2dModel::NeoHookean,
        maxwell: None,
    }
}

fn lowest(w: &Mpm2d) -> f64 {
    w.particles()
        .iter()
        .map(|p| p.x[1].to_f64())
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    // the explicit step: a third of the time the bar wave takes to cross a cell
    let wave = (2.0e4 / (0.91 * 1000.0f64)).sqrt();
    let mut w = Mpm2d::new(config(CELL / wave / 3.0), material()).expect("a valid world");
    let centre = [0.12, GROUND + RADIUS + 0.02];
    let seeded = w.seed(
        |x| {
            let (dx, dy) = (x[0].to_f64() - centre[0], x[1].to_f64() - centre[1]);
            fx((dx * dx + dy * dy).sqrt() - RADIUS)
        },
        2,
    );
    // one extra crumb on top, added by hand
    w.add_particle(
        [fx(centre[0]), fx(centre[1] + RADIUS - CELL / 4.0)],
        [Fix128::ZERO, Fix128::ZERO],
        fx(CELL * CELL / 4.0),
    );
    let mass = w.mass();
    println!(
        "seeded {seeded} particles (+1), mass {:.6} kg/m",
        mass.to_f64()
    );

    // the fall
    for _ in 0..1500 {
        w.step().expect("an explicit step");
    }
    assert_eq!(w.mass(), mass);
    assert!(
        lowest(&w) >= GROUND - 1e-9,
        "the blob stays above the ground"
    );
    println!(
        "after the fall: lowest {:.4} m, kinetic {:.3e} J/m, elastic {:.3e} J/m, angular momentum {:.3e}",
        lowest(&w),
        w.kinetic_energy().to_f64(),
        w.elastic_energy().to_f64(),
        w.angular_momentum().to_f64()
    );

    // a hand presses the top with 5 N/m, shared by rest area
    let top = w
        .particles()
        .iter()
        .map(|p| p.x[1].to_f64())
        .fold(f64::NEG_INFINITY, f64::max);
    let hand = w.particles_within([fx(centre[0]), fx(top)], fx(0.015));
    assert!(!hand.is_empty(), "the hand reaches the top");
    let before = top;
    for _ in 0..600 {
        let loads = w
            .shared_loads(&hand, [Fix128::ZERO, fx(-5.0)])
            .expect("hand");
        w.step_with_loads(&loads).expect("a pressed step");
    }
    let pressed = hand
        .iter()
        .map(|&i| w.particles()[i].x[1].to_f64())
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(pressed < before, "the hand pushes the top down");
    assert_eq!(w.mass(), mass);
    println!("pressed: the top went from {before:.4} m to {pressed:.4} m");

    // the pressed shape, rebuilt at rest in a world stepped at a frame (1/60 s) and held there
    // by a damped foundation, stepped implicitly
    let anchors: Vec<_> = w.particles().iter().map(|p| p.x).collect();
    let mut frame = Mpm2d::new(config(1.0 / 60.0), material()).expect("a valid world");
    for p in w.particles() {
        frame.add_particle(p.x, [Fix128::ZERO, Fix128::ZERO], p.area0);
    }
    assert_eq!(frame.mass(), mass);
    let solve = ImplicitSolve::default();
    for _ in 0..10 {
        let mut loads = frame
            .foundation_loads(&anchors, fx(2.0e5), fx(20.0))
            .expect("foundation");
        loads.extend(
            frame
                .shared_loads(&hand, [Fix128::ZERO, fx(-5.0)])
                .expect("hand"),
        );
        let report = frame
            .step_implicit_with_loads(&solve, &loads)
            .expect("an implicit step");
        assert!(report.newton <= solve.max_newton);
    }
    let _ = frame
        .step_implicit(&solve)
        .expect("an implicit step without loads");
    assert_eq!(frame.mass(), mass);
    assert!(lowest(&frame) >= GROUND - 1e-9);

    let digest = frame.state_digest();
    let again = Mpm2d::restore(&frame.snapshot()).expect("restores");
    assert_eq!(again.state_digest(), digest);
    println!("implicit frames done, digest {:02x?}", &digest[..4]);
}
