//! Closed-form oracles of the thick chain's neo-Hookean stretch law
//! ([`alice_physics_core::thick_chain::ThickChainStretch::NeoHookean`]): the incompressible
//! neo-Hookean solid in plane strain, whose energy per rest area at the stretch `λ` is
//! `W = (μ / 2)(λ² + λ⁻² − 2)` (the width follows as `1/λ`), so a link of rest width `w0`
//! carries the tension `T(λ) = dW/dλ · w0 = μ w0 (λ − λ⁻³)` (Ogden, "Non-linear Elastic
//! Deformations", 1984, the plane strain case of the neo-Hookean strain energy).
//!
//! 1. One hanging link: it settles where `T(λ) = m g`. The shear modulus is chosen so that
//!    the closed form gives a stated `λ` (1.05, 1.3, 1.7), and the chain must come to it.
//! 2. Small stretch: `T → 4 μ w0 (λ − 1)` (the plane strain modulus `4 μ`); the next term of
//!    the series is `−6 μ w0 (λ − 1)²`, so the extension is above the linear one by between
//!    `1.5 ε² − 2.5 ε³` and `1.5 ε²`.
//! 3. One link standing on a held particle: the weight shortens it to where
//!    `T(λ) = −m g`, `λ = 0.8`.
//! 4. A chain sliding along a channel that holds it thin stops after `v0² / (2 a)`,
//!    `a = μ_f T(λc) l / (c m)`: the walls press with `T l / c` whatever the law of `T`.
//! 5. Entering a narrower channel: `tests/analytic_thick_chain_entry.rs`.
//! 6. The law is part of the state: the snapshot tells the two laws apart, and a run repeats
//!    bit for bit.
//! 7. Held by a channel with nothing out of balance, a chain comes to rest and stays, and
//!    the walls carry `T(λc) l / c` for each link less the substep's share (a ratchet at
//!    the measured share).
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: closed forms are evaluated in f64, never by the code under test"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_collider::DistanceSdfQuery;
use alice_physics_core::thick_chain::{
    ThickChain, ThickChainConfig, ThickChainError, ThickChainFit, ThickChainStretch, P2,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn f(v: Fix128) -> f64 {
    v.to_f64()
}

const ZERO2: P2 = [Fix128::ZERO, Fix128::ZERO];

fn config(shear: f64) -> ThickChainConfig {
    ThickChainConfig {
        substeps: 32,
        iterations: 4,
        gravity: ZERO2,
        stretch: ThickChainStretch::NeoHookean { shear: fx(shear) },
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(0.002),
        splits: 0,
        relaxation: Fix128::ZERO,
        fit: ThickChainFit::Disc,
    }
}

/// `T / (μ w0)` at the stretch `λ`
fn pull(lambda: f64) -> f64 {
    lambda - lambda.powi(-3)
}

/// One vertical link, one end held, under gravity for 3 s; returns its stretch and the
/// tolerance on it. `up`: the free particle is the upper one (the weight shortens the link).
///
/// The tolerance: a substep at rest moves the free particle by `δ = h² g` and the constraint
/// takes it back; with the constraint's slope `(1 + λ⁻²) / l0` constant that is the weight
/// exactly, and over `δ` the slope changes by the fraction `2 λ⁻³ / (1 + λ⁻²) · δ / l0` of
/// itself, so the tension is the weight to twice that fraction (the second order with it).
/// The swing left under the drag `k` after the time `t` is below the first extension times
/// `e^(−k t / 2)`.
fn hang(lambda: f64, up: bool) -> (f64, f64) {
    let (l0, r0, rho, g, drag) = (0.02, 0.01, 1000.0, 9.81, 30.0);
    // the free particle carries half the link: rho * (2 r0 l0) / 2
    let m = rho * r0 * l0;
    let shear = m * g / (2.0 * r0 * pull(lambda).abs());
    let mut cfg = config(shear);
    cfg.gravity = [Fix128::ZERO, fx(-g)];
    cfg.drag = fx(drag);
    let points = if up {
        [ZERO2, [Fix128::ZERO, fx(l0)]]
    } else {
        [[Fix128::ZERO, fx(l0)], ZERO2]
    };
    let mut chain = ThickChain::new(cfg, &points, fx(r0), fx(rho)).expect("a valid chain");
    chain.pin(0);
    let seconds = 3.0;
    for _ in 0..(seconds * 60.0) as usize {
        chain.step(Fix128::from_ratio(1, 60)).expect("a step");
    }
    let h = 1.0 / 60.0 / f64::from(cfg.substeps);
    let turn = 2.0 * lambda.powi(-3) / (1.0 + lambda.powi(-2)) * h * h * g / l0;
    let stiff = 1.0 + 3.0 * lambda.powi(-4);
    let tol = 2.0 * turn * pull(lambda).abs() / stiff
        + (lambda - 1.0).abs() * (-drag * seconds / 2.0).exp()
        + 1.0e-12;
    (f(chain.length(0)) / l0, tol)
}

#[test]
fn a_hanging_link_settles_where_the_neo_hookean_tension_is_its_weight() {
    for lambda in [1.05, 1.3, 1.7] {
        let (got, tol) = hang(lambda, false);
        assert!(
            (got - lambda).abs() <= tol,
            "stretch {got:e}, expected {lambda:e} within {tol:e}"
        );
    }
}

#[test]
fn a_small_stretch_meets_the_plane_strain_modulus() {
    // the load that the linear law 4 μ w0 ε would answer with ε = 0.01
    let linear = 0.01;
    // the stretch whose pull is 4 ε: found on the closed form by bisection
    let (mut lo, mut hi) = (1.0f64, 1.1f64);
    for _ in 0..80 {
        let mid = (lo + hi) / 2.0;
        if pull(mid) < 4.0 * linear {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (got, tol) = hang(lo, false);
    let over = got - 1.0 - linear;
    let (least, most) = (
        1.5 * linear * linear - 2.5 * linear.powi(3) - tol,
        // ε itself is above the linear one, by under 2 %: 1.5 ε² is within 1.6 ε_linear²
        1.6 * linear * linear + tol,
    );
    assert!(
        over >= least && over <= most,
        "extension over the linear one {over:e}, expected {least:e}..{most:e}"
    );
}

#[test]
fn a_standing_link_is_shortened_to_where_the_tension_is_minus_its_weight() {
    let lambda = 0.8;
    let (got, tol) = hang(lambda, true);
    assert!(
        (got - lambda).abs() <= tol,
        "stretch {got:e}, expected {lambda:e} within {tol:e}"
    );
}

/// the free space between two walls at `y = ±c`
fn channel(c: f32) -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(move |_x: f32, y: f32, _z: f32| c - y.abs())
}

fn row(start: f64, l: f64, links: usize) -> Vec<P2> {
    (0..=links)
        .map(|i| [fx(start + l * i as f64), Fix128::ZERO])
        .collect()
}

#[test]
fn a_neo_hookean_chain_held_thin_stops_after_the_squeeze_friction_distance() {
    let (l0, r0, rho, shear, c, mu, v0) = (0.01, 0.05, 1000.0, 500.0, 0.027f32, 0.05, 0.3);
    let mut cfg = config(shear);
    cfg.substeps = 8;
    cfg.min_radius = fx(0.005);
    cfg.friction = fx(mu);
    let area = 2.0 * r0 * l0;
    let held = area / (2.0 * f64::from(c));
    let mut chain = ThickChain::new(cfg, &row(0.0, l0, 6), fx(r0), fx(rho)).expect("a valid chain");
    for i in 0..chain.particle_count() {
        chain.set_position(i, [fx(held * i as f64), Fix128::ZERO]);
        chain.set_velocity(i, [fx(v0), Fix128::ZERO]);
    }
    let walls = channel(c);
    let start: f64 = chain.positions().iter().map(|p| f(p[0])).sum();
    let dt = Fix128::from_ratio(1, 60);
    let mut frames = 0;
    while chain.velocities().iter().any(|v| f(v[0]).abs() > 1.0e-9) {
        chain.step_in(dt, &walls, &[]).expect("a step");
        frames += 1;
        assert!(frames < 600, "still moving after 10 s");
    }
    let n = chain.particle_count() as f64;
    let got = (chain.positions().iter().map(|p| f(p[0])).sum::<f64>() - start) / n;
    // both walls together press on a link with T l / c; the chain's mass is the links'
    let lambda = held / l0;
    let tension = shear * 2.0 * r0 * pull(lambda);
    let a = mu * tension * held / f64::from(c) / (rho * area);
    let expect = v0 * v0 / (2.0 * a);
    // the staircase of the Coulomb stop (v0 h a frame's substep), and the tension known to
    // (ω h)² of it: ω² = (w + w) dT/dl, dT/dl = μ w0 (1 + 3 λ⁻⁴) / l0
    let h = f(dt) / f64::from(cfg.substeps);
    let w = 1.0 / (rho * r0 * l0);
    let wh2 = (w + w) * shear * 2.0 * r0 * (1.0 + 3.0 * lambda.powi(-4)) / l0 * h * h;
    let tol = 2.0 * v0 * h + 2.0 * wh2 * expect;
    assert!(
        (got - expect).abs() <= tol,
        "travel {got:e}, expected {expect:e} within {tol:e}"
    );
}

#[test]
fn held_by_a_channel_a_chain_comes_to_rest_and_the_walls_carry_most_of_its_tension() {
    // links laid as the channel holds them, no friction, no drag, no speed. The solver's own
    // rest state is a little off the laid one (the stretch and the walls are solved in turn),
    // so it is given 10 frames; then nothing moves: no particle has a speed and no link
    // changes its length for 20 frames. The walls carry T(λc) l / c for each link (a strip
    // squeezed to λc with free ends presses each side with μ (λc² − λc⁻²) l), less what the
    // turn-taking leaves: a share that goes with (ω h)² and that more passes do not remove
    // (one link: 0.888 with 4 passes, 0.889 with 8 and with 16). This run is at
    // (ω h)² = 0.13 with 4 passes, where the walls' force was measured at 0.888 of the closed
    // form for one link and 0.937 to 0.967 for six: the floor below is a ratchet at that
    // measure (it may only be raised), the closed form is the ceiling.
    let (l0, r0, shear, c) = (0.01, 0.05, 500.0, 0.027f32);
    let held = r0 * l0 / f64::from(c);
    let at = held / l0;
    let expect = shear * 2.0 * r0 * pull(at) * held / f64::from(c);
    let walls = channel(c);
    for (links, floor) in [(1usize, 0.88), (6, 0.93)] {
        let mut cfg = config(shear);
        cfg.substeps = 8;
        cfg.min_radius = fx(0.005);
        let mut chain =
            ThickChain::new(cfg, &row(0.0, l0, links), fx(r0), fx(DENSITY)).expect("a chain");
        for i in 0..chain.particle_count() {
            chain.set_position(i, [fx(held * i as f64), Fix128::ZERO]);
        }
        for _ in 0..10 {
            chain
                .step_in(Fix128::from_ratio(1, 60), &walls, &[])
                .expect("a step");
        }
        let lengths: Vec<Fix128> = (0..links).map(|k| chain.length(k)).collect();
        for n in 0..20 {
            chain
                .step_in(Fix128::from_ratio(1, 60), &walls, &[])
                .expect("a step");
            for (i, v) in chain.velocities().iter().enumerate() {
                assert!(
                    f(v[0]).abs() < 1.0e-9 && f(v[1]).abs() < 1.0e-9,
                    "{links} links: frame {n}: particle {i} moves at {:e}, {:e}",
                    f(v[0]),
                    f(v[1])
                );
            }
            for (k, before) in lengths.iter().enumerate() {
                // at rest to 1e-9 m/s over 20 frames: under 1e-9 m
                assert!(
                    (f(chain.length(k)) - f(*before)).abs() < 1.0e-9,
                    "{links} links: frame {n}: link {k} changed its length"
                );
            }
        }
        for k in 0..links {
            let share = f(chain.wall_force(k)) / expect;
            assert!(
                share >= floor && share <= 1.0 + 1.0e-6,
                "{links} links: the walls carry {share} of T l / c for link {k} (floor {floor})"
            );
            // and the link is no longer than the walls make it, and not 1 % shorter
            let l = f(chain.length(k)) / held;
            assert!(
                l <= 1.0 + 2.0f64.powi(-22) && l >= 0.99,
                "{links} links: link {k} is {l} of the walls' length"
            );
        }
    }
    // without walls there is no press
    let mut free =
        ThickChain::new(config(shear), &row(0.0, l0, 2), fx(r0), fx(DENSITY)).expect("a chain");
    free.step(Fix128::from_ratio(1, 60)).expect("a step");
    assert_eq!(free.wall_force(0), Fix128::ZERO);
}

const LINK: f64 = 0.01;
const RADIUS: f64 = 0.02;
const SHEAR: f64 = 2000.0;
const DENSITY: f64 = 1000.0;

#[test]
fn the_stretch_law_is_part_of_the_state() {
    let points = row(0.0, LINK, 3);
    let make = |stretch| {
        let mut cfg = config(SHEAR);
        cfg.stretch = stretch;
        ThickChain::new(cfg, &points, fx(RADIUS), fx(DENSITY)).expect("a valid chain")
    };
    let value = fx(SHEAR);
    let neo = make(ThickChainStretch::NeoHookean { shear: value });
    let linear = make(ThickChainStretch::Linear { compliance: value });
    assert_ne!(neo.snapshot(), linear.snapshot());
    assert_ne!(neo.state_digest(), linear.state_digest());
    // the same run twice: the same bits
    let run = || {
        let mut chain = make(ThickChainStretch::NeoHookean { shear: value });
        chain.set_position(3, [fx(LINK * 3.1), Fix128::ZERO]);
        for _ in 0..30 {
            chain.step(Fix128::from_ratio(1, 60)).expect("a step");
        }
        chain.state_digest()
    };
    assert_eq!(run(), run());
    // a shear modulus of 0 is refused by name
    let mut cfg = config(SHEAR);
    cfg.stretch = ThickChainStretch::NeoHookean {
        shear: Fix128::ZERO,
    };
    assert_eq!(
        ThickChain::new(cfg, &points, fx(RADIUS), fx(DENSITY)).err(),
        Some(ThickChainError::Invalid("shear"))
    );
}
