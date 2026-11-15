//! Mass of the 2D material point method ([`alice_physics_core::mpm2d`]), stated on its own.
//!
//! Each particle carries the mass `ρ A⁰` it was given when it was added ([`Mpm2d::add_particle`]);
//! the grid's node masses are rebuilt from the particles every step and never written back. So
//! every path that advances or rebuilds a world has to leave each particle's mass, the particle
//! count and the total exactly (bit for bit) as they were: the explicit step, the step with
//! external loads (a shared hand force and an elastic, damped foundation), the implicit step
//! with loads, a viscoelastic material, a snapshot restored mid-run and a refused step. The
//! total is compared with `ρ Σ A⁰` summed in particle order, which is how [`Mpm2d::mass`] adds.

use alice_physics_core::math::Fix128;
use alice_physics_core::mpm2d::{
    ImplicitSolve, Maxwell, Mpm2d, Mpm2dConfig, Mpm2dMaterial, Mpm2dModel,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

const CELL: f64 = 0.01;
const GROUND: f64 = 0.05;
const MODELS: [Mpm2dModel; 2] = [Mpm2dModel::FixedCorotated, Mpm2dModel::NeoHookean];

fn world(model: Mpm2dModel, maxwell: bool) -> Mpm2d {
    let density = fx(1000.0);
    let mut w = Mpm2d::new(
        Mpm2dConfig {
            origin: [Fix128::ZERO, Fix128::ZERO],
            cell: fx(CELL),
            nodes: [24, 40],
            gravity: [Fix128::ZERO, fx(-9.81)],
            dt: fx(CELL / (1.0e5f64 / (0.91 * 1000.0)).sqrt() / 3.0),
            ground: Some(fx(GROUND)),
        },
        Mpm2dMaterial {
            youngs: fx(1.0e5),
            poisson: fx(0.3),
            density,
            model,
            maxwell: maxwell.then(|| Maxwell {
                fraction: fx(0.5),
                relaxation_time: fx(0.05),
            }),
        },
    )
    .expect("a valid world");
    // a 4 × 10 cell column, 2 × 2 particles a cell, two particle sizes so the masses differ
    for i in 0..8 {
        for j in 0..20 {
            let x = 0.1 + (f64::from(i) + 0.5) * CELL / 2.0;
            let y = GROUND + (f64::from(j) + 0.5) * CELL / 2.0;
            let area = if (i + j) % 3 == 0 { 2.0e-5 } else { 2.5e-5 };
            w.add_particle([fx(x), fx(y)], [fx(0.0), fx(0.1)], fx(area));
        }
    }
    w
}

/// each particle's mass, and the total as `ρ Σ A⁰` in particle order
fn ledger(w: &Mpm2d, density: Fix128) -> (Vec<Fix128>, Fix128) {
    let masses = w.particles().iter().map(|p| p.mass).collect();
    let total = w
        .particles()
        .iter()
        .fold(Fix128::ZERO, |a, p| a + density * p.area0);
    (masses, total)
}

fn check(w: &Mpm2d, start: &(Vec<Fix128>, Fix128), what: &str) {
    let now: Vec<Fix128> = w.particles().iter().map(|p| p.mass).collect();
    assert_eq!(
        now.len(),
        start.0.len(),
        "{what}: the particle count changed"
    );
    assert_eq!(now, start.0, "{what}: a particle's mass changed");
    assert_eq!(
        w.mass(),
        start.1,
        "{what}: the total is not rho times the summed rest areas"
    );
}

#[test]
fn every_step_keeps_each_particle_mass_and_the_total_exactly() {
    let density = fx(1000.0);
    let solve = ImplicitSolve::default();
    let mut compared = 0;
    for model in MODELS {
        for maxwell in [false, true] {
            let mut w = world(model, maxwell);
            let start = ledger(&w, density);
            assert_eq!(start.0.len(), 160);
            check(&w, &start, "as added");
            let anchors: Vec<_> = w.particles().iter().map(|p| p.x).collect();
            let hand = w.particles_within([fx(0.12), fx(GROUND + 0.1)], fx(0.012));
            assert!(!hand.is_empty(), "the hand touches the top of the column");
            let tag = format!("{model:?} maxwell {maxwell}");

            for n in 0..40 {
                w.step().expect("an explicit step");
                check(&w, &start, &format!("{tag}: explicit step {n}"));
            }
            for n in 0..40 {
                let mut loads = w
                    .foundation_loads(&anchors, fx(2.0e6), fx(50.0))
                    .expect("foundation loads");
                loads.extend(w.shared_loads(&hand, [fx(0.0), fx(-2.0)]).expect("hand"));
                w.step_with_loads(&loads).expect("a loaded step");
                check(&w, &start, &format!("{tag}: loaded step {n}"));
            }

            let restored = Mpm2d::restore(&w.snapshot()).expect("a snapshot restores");
            check(&restored, &start, &format!("{tag}: restored"));
            w = restored;

            let refused = w.step_with_loads(&[(start.0.len(), [fx(1.0), fx(0.0)])]);
            assert!(refused.is_err(), "{tag}: a load on no particle is refused");
            check(&w, &start, &format!("{tag}: refused step"));

            for n in 0..5 {
                let loads = w.shared_loads(&hand, [fx(0.0), fx(-2.0)]).expect("hand");
                w.step_implicit_with_loads(&solve, &loads)
                    .expect("an implicit step");
                check(&w, &start, &format!("{tag}: implicit step {n}"));
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 4);
}
