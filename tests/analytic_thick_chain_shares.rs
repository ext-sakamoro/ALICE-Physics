//! Closed-form oracles of the thick chain's conforming rule
//! ([`alice_physics_core::thick_chain::ThickChainFit::Conforming`]): a link takes the shape
//! of the space it has. Its samples' shares `ω_j` (trapezoid weights, adding up to 1) are
//! each as long as the wall there makes them, `μ_j = r_rest / d_j` (a planar share keeps
//! its area: width times length), or as the link's tension does, `λ*`, whichever is more:
//! `l / l0 = Σ ω_j max(λ*, μ_j)`. The link's energy is `A Σ ω_j W(max(λ*, μ_j))`,
//! `W = (μ / 2)(λ − 1/λ)²`, its tension `T(λ*)`; no `λ*` exists when `l < l0 Σ ω_j μ_j`,
//! and that is the hard wall.
//!
//! 1. A straight channel: every share is held alike, so a link settles at `A / (2 c)` on
//!    the middle line, as under the disc rule.
//! 2. Sliding along that channel with friction it stops after `v0² / (2 a)`,
//!    `a = μ_f T(λc) l / (c m)`: the held shares press on the walls with the gel's own
//!    stress, `T l / c` for both walls together.
//! 3. In a straight frictionless channel with no drag the chain's momentum along it does
//!    not change, whatever its links do among themselves: the walls only push across.
//! 4. `λ*` as the chain reports it is the solution of `l / l0 = Σ ω_j max(λ*, μ_j)` found
//!    independently; the least `μ_j` when the link is no longer than the walls ask; and as a
//!    link is moved across a
//!    mouth it changes by no more than `(1 − Ω) / Ω · r / c²` times the way (`Ω` the free
//!    shares' weight): no jump when a share passes from held to free.
//! 5. Let go with its head at the open end of a channel narrower than itself, the chain
//!    leaves by itself (the shares at the mouth are pushed out by their own stress and pull
//!    the rest), and its kinetic energy is never above the energy the squeeze held,
//!    `A W(r0 / c)`.
//! 6. The rule is part of the state, and it needs neo-Hookean links.
//! 7. Held by a channel with nothing out of balance, a chain comes to rest and stays (no
//!    particle has a speed, no link changes its length), and the walls carry `T(λc) l / c`
//!    for each link less the substep's share.
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
const DENSITY: f64 = 1000.0;
const LEAST: f64 = 0.005;
/// the longest a link is sampled for, over its rest length
const REACH: f64 = 8.0;
const SUBSTEPS: u32 = 16;

fn config(shear: f64, fit: ThickChainFit) -> ThickChainConfig {
    ThickChainConfig {
        substeps: SUBSTEPS,
        iterations: 4,
        gravity: ZERO2,
        stretch: ThickChainStretch::NeoHookean { shear: fx(shear) },
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(LEAST),
        splits: 8,
        relaxation: Fix128::ZERO,
        fit,
        reach: fx(REACH),
    }
}

fn frame() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn row(start: f64, l: f64, links: usize) -> Vec<P2> {
    (0..=links)
        .map(|i| [fx(start + l * i as f64), Fix128::ZERO])
        .collect()
}

/// `T / (μ w0)` at the stretch `λ`
fn pull(lambda: f64) -> f64 {
    lambda - lambda.powi(-3)
}

/// the free space between two walls at `y = ±c`
fn channel(c: f32) -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(move |_x: f32, y: f32, _z: f32| c - y.abs())
}

#[test]
fn a_chain_in_a_channel_is_as_long_under_either_rule() {
    let (l0, r0, shear, c) = (0.01, 0.05, 500.0, 0.027f32);
    let area = 2.0 * r0 * l0;
    let held = area / (2.0 * f64::from(c));
    let walls = channel(c);
    for fit in [ThickChainFit::Disc, ThickChainFit::Conforming] {
        let mut cfg = config(shear, fit);
        cfg.drag = fx(30.0);
        let mut chain =
            ThickChain::new(cfg, &row(0.0, l0, 6), fx(r0), fx(DENSITY)).expect("a chain");
        // let go a little longer than it settles, 2 mm off the middle
        for i in 0..chain.particle_count() {
            chain.set_position(i, [fx(1.05 * held * i as f64), fx(0.002)]);
        }
        for _ in 0..180 {
            chain.step_in(frame(), &walls, &[]).expect("a step");
        }
        // a substep at rest: the stretch shortens a link by (ω h)² of its extension (to first
        // order), the walls give it back; the walls are f32
        let h = 1.0 / 60.0 / f64::from(SUBSTEPS);
        let at = held / l0;
        let w = 1.0 / (DENSITY * r0 * l0);
        let wh2 = (w + w) * shear * 2.0 * r0 * (1.0 + 3.0 * at.powi(-4)) / l0 * h * h;
        let tol = wh2 * (held - l0) + held * 2.0f64.powi(-23);
        for k in 0..chain.link_count() {
            let l = f(chain.length(k));
            assert!(
                (l - held).abs() <= tol,
                "{fit:?}: link {k}: {l:e}, expected {held:e} within {tol:e}"
            );
        }
        // on the middle line: the wall's slope is taken over ±e (an eighth of the least
        // radius) and is zero only there
        let band = LEAST / 32.0;
        for (i, p) in chain.positions().iter().enumerate() {
            assert!(
                f(p[1]).abs() <= band,
                "{fit:?}: particle {i}: y {:e}",
                f(p[1])
            );
        }
    }
}

#[test]
fn a_conforming_chain_held_thin_stops_after_the_squeeze_friction_distance() {
    let (l0, r0, shear, c, mu, v0) = (0.01, 0.05, 500.0, 0.027f32, 0.05, 0.3);
    let mut cfg = config(shear, ThickChainFit::Conforming);
    cfg.substeps = 8;
    cfg.friction = fx(mu);
    let area = 2.0 * r0 * l0;
    let held = area / (2.0 * f64::from(c));
    let mut chain = ThickChain::new(cfg, &row(0.0, l0, 6), fx(r0), fx(DENSITY)).expect("a chain");
    for i in 0..chain.particle_count() {
        chain.set_position(i, [fx(held * i as f64), Fix128::ZERO]);
        chain.set_velocity(i, [fx(v0), Fix128::ZERO]);
    }
    let walls = channel(c);
    let start: f64 = chain.positions().iter().map(|p| f(p[0])).sum();
    let mut frames = 0;
    while chain.velocities().iter().any(|v| f(v[0]).abs() > 1.0e-9) {
        chain.step_in(frame(), &walls, &[]).expect("a step");
        frames += 1;
        assert!(frames < 600, "still moving after 10 s");
    }
    let n = chain.particle_count() as f64;
    let got = (chain.positions().iter().map(|p| f(p[0])).sum::<f64>() - start) / n;
    // both walls together press on a link with T l / c; the chain's mass is the links'
    let at = held / l0;
    let tension = shear * 2.0 * r0 * pull(at);
    let a = mu * tension * held / f64::from(c) / (DENSITY * area);
    let expect = v0 * v0 / (2.0 * a);
    // the staircase of the Coulomb stop (v0 h a frame's substep), and the walls' force
    // known to (ω h)² of it, as under the disc rule (the stretch and the walls are solved
    // in turn): ω² = (w + w) dT/dl, dT/dl = μ w0 (1 + 3 λ⁻⁴) / l0
    let h = 1.0 / 60.0 / f64::from(cfg.substeps);
    let w = 1.0 / (DENSITY * r0 * l0);
    let wh2 = (w + w) * shear * 2.0 * r0 * (1.0 + 3.0 * at.powi(-4)) / l0 * h * h;
    let tol = 2.0 * v0 * h + 2.0 * wh2 * expect;
    assert!(
        (got - expect).abs() <= tol,
        "travel {got:e}, expected {expect:e} within {tol:e}"
    );
}

#[test]
fn the_walls_of_a_straight_channel_do_not_change_the_momentum_along_it() {
    let (l0, r0, shear, c) = (0.01, 0.02, 500.0, 0.012f32);
    let links = 8;
    let held = r0 * l0 / f64::from(c);
    let cfg = config(shear, ThickChainFit::Conforming);
    let mut chain =
        ThickChain::new(cfg, &row(0.0, l0, links), fx(r0), fx(DENSITY)).expect("a chain");
    // held by the channel, the particles sent along it at speeds that set the links against
    // each other
    for i in 0..=links {
        chain.set_position(i, [fx(1.1 * held * i as f64), Fix128::ZERO]);
        chain.set_velocity(i, [fx(0.05 * (i % 3) as f64), Fix128::ZERO]);
    }
    // an end particle carries half a link, the others two halves
    let mass = |i: usize| DENSITY * 2.0 * r0 * l0 * if i == 0 || i == links { 0.5 } else { 1.0 };
    let momentum = |chain: &ThickChain| -> f64 {
        chain
            .velocities()
            .iter()
            .enumerate()
            .map(|(i, v)| mass(i) * f(v[0]))
            .sum()
    };
    let first = momentum(&chain);
    let walls = channel(c);
    let mut moved = false;
    for n in 0..60 {
        let before = chain.length(0);
        chain.step_in(frame(), &walls, &[]).expect("a step");
        moved |= chain.length(0) != before;
        let now = momentum(&chain);
        // sums of products in Q64.64 over 9 particles and up to 16 × 2^8 parts of a frame
        assert!(
            (now - first).abs() <= 1.0e-12,
            "frame {n}: the momentum went from {first:e} to {now:e}"
        );
        for (i, p) in chain.positions().iter().enumerate() {
            assert_eq!(p[1], Fix128::ZERO, "particle {i} left the middle line");
        }
    }
    assert!(moved, "the links did not move among themselves");
}

#[test]
fn held_by_a_channel_a_conforming_chain_comes_to_rest_and_the_walls_carry_its_tension() {
    // links laid as the channel holds them, no friction, no drag, no speed. After 10 frames
    // for the solver's own rest state, nothing moves for 20: no particle has a speed and no
    // link changes its length. The walls carry T(λc) l / c for each link, less what solving
    // the stretch and the walls in turn leaves (a share that goes with (ω h)², here 0.13
    // with 4 passes: the disc rule's measured floor at these settings is 0.88 for one link
    // and 0.93 for six, and in a straight channel the conforming rule does what it does).
    let (l0, r0, shear, c) = (0.01, 0.05, 500.0, 0.027f32);
    let held = r0 * l0 / f64::from(c);
    let at = held / l0;
    let expect = shear * 2.0 * r0 * pull(at) * held / f64::from(c);
    let walls = channel(c);
    for (links, floor) in [(1usize, 0.88), (6, 0.93)] {
        let mut cfg = config(shear, ThickChainFit::Conforming);
        cfg.substeps = 8;
        let mut chain =
            ThickChain::new(cfg, &row(0.0, l0, links), fx(r0), fx(DENSITY)).expect("a chain");
        for i in 0..chain.particle_count() {
            chain.set_position(i, [fx(held * i as f64), Fix128::ZERO]);
        }
        for _ in 0..10 {
            chain.step_in(frame(), &walls, &[]).expect("a step");
        }
        let lengths: Vec<Fix128> = (0..links).map(|k| chain.length(k)).collect();
        for n in 0..20 {
            chain.step_in(frame(), &walls, &[]).expect("a step");
            for (i, v) in chain.velocities().iter().enumerate() {
                assert!(
                    f(v[0]).abs() < 1.0e-9 && f(v[1]).abs() < 1.0e-9,
                    "{links} links: frame {n}: particle {i} moves at {:e}, {:e}",
                    f(v[0]),
                    f(v[1])
                );
            }
            for (k, before) in lengths.iter().enumerate() {
                assert!(
                    (f(chain.length(k)) - f(*before)).abs() < 1.0e-9,
                    "{links} links: frame {n}: link {k} changed its length"
                );
            }
        }
        assert_eq!(
            chain.walls_unsettled(),
            0,
            "a pass left its walls unsettled"
        );
        for k in 0..links {
            let share = f(chain.wall_force(k)) / expect;
            println!(
                "conforming, {links} links: the walls carry {share:.4} of T l / c for link {k}"
            );
            assert!(
                share >= floor && share <= 1.0 + 1.0e-6,
                "{links} links: the walls carry {share} of T l / c for link {k} (floor {floor})"
            );
            let l = f(chain.length(k)) / held;
            assert!(
                l <= 1.0 + 2.0f64.powi(-22) && l >= 0.99,
                "{links} links: link {k} is {l} of the walls' length"
            );
        }
    }
}

/// the free space of a mouth: a channel of half-width `c` for `x < 0` that opens at `x = 0`
/// into a channel of half-width `wide` (exact: the nearest of the narrow walls, the faces
/// at `x = 0` and the wide walls)
fn step(c: f32, wide: f32) -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(move |x: f32, y: f32, _z: f32| step_distance(c, wide, x, y))
}

fn step_distance(c: f32, wide: f32, x: f32, y: f32) -> f32 {
    let y = y.abs();
    if x < 0.0 {
        if y < c {
            // in the narrow channel: its wall, or the corner seen from inside is the wall's end
            c - y
        } else {
            // in the wall: the narrow wall or the face
            -(y - c).min(-x)
        }
    } else if y < wide {
        // in the wide channel: its wall, or the face from the corner (0, c) outwards
        let face = if y >= c {
            x
        } else {
            (x * x + (c - y) * (c - y)).sqrt()
        };
        (wide - y).min(face)
    } else {
        -(y - wide)
    }
}

/// the conforming rule's shares of a straight link from `x0` to `x0 + l` on the middle line
/// of the mouth: the weights and the stretches the walls ask for, as the MODEL section
/// states them (the samples at most half the least radius apart, one more interval than the
/// length asks for)
fn shares(x0: f64, l: f64, n: usize, rest_radius: f64, c: f32, wide: f32) -> (Vec<f64>, Vec<f64>) {
    let mut weights = Vec::new();
    let mut asked = Vec::new();
    for j in 0..=n {
        let x = x0 + l * j as f64 / n as f64;
        let d = f64::from(step_distance(c, wide, x as f32, 0.0)).max(LEAST);
        weights.push(if j == 0 || j == n { 0.5 } else { 1.0 } / n as f64);
        asked.push(rest_radius / d);
    }
    (weights, asked)
}

/// `λ*` and the free shares' weight for `ratio = l / l0`: `ratio = Σ ω max(λ*, μ)`, by
/// bisection on `λ*` (the sum grows with it); none when the walls ask for more than `ratio`
fn free_stretch(ratio: f64, weights: &[f64], asked: &[f64]) -> Option<(f64, f64)> {
    let need: f64 = weights.iter().zip(asked).map(|(w, m)| w * m).sum();
    if ratio <= need {
        return None;
    }
    let (mut lo, mut hi) = (0.0f64, ratio * 4.0);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let sum: f64 = weights.iter().zip(asked).map(|(w, m)| w * m.max(mid)).sum();
        if sum < ratio {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let free = weights
        .iter()
        .zip(asked)
        .filter(|(_, m)| **m <= lo)
        .map(|(w, _)| w)
        .sum();
    Some((lo, free))
}

#[test]
fn the_free_stretch_is_the_shares_equation_and_has_no_jump_across_a_mouth() {
    let (l0, r0, c, wide) = (0.02, 0.02, 0.012f32, 0.2f32);
    // a link 1.3 times its rest length, moved along the middle line across the mouth
    let l = 1.3 * l0;
    let cfg = config(500.0, ThickChainFit::Conforming);
    let mut chain = ThickChain::new(cfg, &row(0.0, l0, 1), fx(r0), fx(DENSITY)).expect("a chain");
    let walls = step(c, wide);
    let way = 0.0002;
    let (mut seen_none, mut seen_some, mut last) = (false, 0, None::<(f64, f64)>);
    for n in 0..300 {
        let x0 = -0.045 + way * f64::from(n);
        chain.set_position(0, [fx(x0), Fix128::ZERO]);
        chain.set_position(1, [fx(x0 + l), Fix128::ZERO]);
        let got = f(chain.free_stretch(0, &walls).expect("a link"));
        // the chain's own count (a ceiling taken in Q64.64; pinned at a whole number below)
        let spans = usize::try_from(chain.spans(0).expect("a count")).expect("a count");
        let (weights, asked) = shares(x0, l, spans, r0, c, wide);
        match free_stretch(l / l0, &weights, &asked) {
            None => {
                // no longer than the walls ask: the least stretch they ask of a share
                let least = asked.iter().copied().fold(f64::INFINITY, f64::min);
                assert!(
                    (got - least).abs() <= 1.0e-5 * least,
                    "at {x0}: {got:e}, the least asked is {least:e}"
                );
                seen_none = true;
                last = None;
            }
            Some((want, free)) => {
                // the walls are f32 (2^-23 of a distance) and the weights add up to 1 to
                // Q64.64 in the chain, to f64 here; where the link is just as long as the
                // walls ask the two may be on different sides, and the least asked is within
                // that of λ* there
                let need: f64 = weights.iter().zip(&asked).map(|(w, m)| w * m).sum();
                let edge = (l / l0 - need).abs() <= 1.0e-5 * need;
                let least = asked.iter().copied().fold(f64::INFINITY, f64::min);
                assert!(
                    (got - want).abs() <= 1.0e-5 * want
                        || (edge && (got - least).abs() <= 1.0e-5 * least),
                    "at {x0}: {got:e}, expected {want:e}"
                );
                if let Some((before, free_before)) = last {
                    // dλ*/dx = −Σ_held ω dμ/dx / Ω and |dμ/dx| ≤ r / c² (the distance
                    // changes no faster than the way): no more than (1 − Ω) / Ω · r / c² a
                    // metre, with the smaller Ω of the two places, and the f32 walls twice
                    let least_free = free.min(free_before);
                    let most = (1.0 - least_free) / least_free * r0 / f64::from(c * c) * way
                        + 2.0e-5 * want;
                    assert!(
                        (got - before).abs() <= most,
                        "at {x0}: λ* went from {before:e} to {got:e}, at most {most:e}"
                    );
                }
                seen_some += 1;
                last = Some((got, free));
            }
        }
    }
    // deep in the channel the walls ask for 5/3 of the rest length and the link is 1.3 of
    // it: every share is held; out of it the link is free at 1.3
    assert!(seen_none, "the link was never as short as the walls ask");
    // the link is 26 mm long and moved 60 mm in 300 steps: it is partly out of the channel
    // for at least a quarter of them
    assert!(seen_some >= 75, "λ* was read at {seen_some} places only");
    let (got, _) = last.expect("free at the end");
    assert!((got - 1.3).abs() <= 1.0e-3, "free of the mouth λ* is {got}");
}

#[test]
fn let_go_at_the_open_end_a_squeezed_chain_leaves_with_no_more_than_the_squeezes_energy() {
    let (l0, r0, shear, c, wide) = (0.01, 0.02, 500.0, 0.012f32, 0.2f32);
    let links = 8;
    let at = r0 / f64::from(c);
    let held = at * l0;
    let cfg = config(shear, ThickChainFit::Conforming);
    let mut chain =
        ThickChain::new(cfg, &row(0.0, l0, links), fx(r0), fx(DENSITY)).expect("a chain");
    // held by the channel, its head at the mouth
    for i in 0..=links {
        chain.set_position(i, [fx(-held * (links - i) as f64), Fix128::ZERO]);
    }
    let walls = step(c, wide);
    // the squeeze holds A W(λc) in every link
    let stored = links as f64 * 2.0 * r0 * l0 * shear / 2.0 * (at - 1.0 / at).powi(2);
    let mass = |i: usize| DENSITY * 2.0 * r0 * l0 * if i == 0 || i == links { 0.5 } else { 1.0 };
    let mut most = 0.0f64;
    let mut out = None;
    for n in 0..600 {
        chain.step_in(frame(), &walls, &[]).expect("a step");
        let kinetic: f64 = chain
            .velocities()
            .iter()
            .enumerate()
            .map(|(i, v)| 0.5 * mass(i) * (f(v[0]).powi(2) + f(v[1]).powi(2)))
            .sum();
        most = most.max(kinetic);
        // the walls are f32 and the sums f64
        assert!(
            kinetic <= stored * (1.0 + 1.0e-6),
            "frame {n}: the kinetic energy {kinetic:e} is above the squeeze's {stored:e}"
        );
        if out.is_none() && f(chain.positions()[0][0]) > 0.0 {
            out = Some(n);
        }
    }
    assert!(out.is_some(), "the tail is still in the channel after 10 s");
    println!(
        "left at frame {:?}; the most kinetic energy was {:.3} of the squeeze's",
        out,
        most / stored
    );
}

#[test]
fn the_fit_rule_is_part_of_the_state_and_needs_neo_hookean_links() {
    let points = row(0.0, 0.01, 2);
    let made = |fit| {
        ThickChain::new(config(500.0, fit), &points, fx(0.02), fx(DENSITY))
            .expect("a chain")
            .snapshot()
    };
    assert_ne!(made(ThickChainFit::Disc), made(ThickChainFit::Conforming));
    let mut cfg = config(500.0, ThickChainFit::Conforming);
    cfg.stretch = ThickChainStretch::Linear {
        compliance: fx(1.0e-4),
    };
    assert_eq!(
        ThickChain::new(cfg, &points, fx(0.02), fx(DENSITY)).err(),
        Some(ThickChainError::Invalid("fit"))
    );
}

#[test]
fn a_links_samples_are_laid_out_once_for_its_reach_and_counted_as_stated() {
    // lengths that are exact in binary: l0 = 1/64, r_min = 1/128, reach 8, so
    // 2 l0 reach / r_min = 32 exactly and the spans are ceil(32) + 1 = 33; a reach one part in
    // 2^20 longer asks for one more
    let count = |reach: f64, stretch: f64| {
        let mut cfg = config(500.0, ThickChainFit::Conforming);
        cfg.min_radius = fx(1.0 / 128.0);
        cfg.reach = fx(reach);
        let l0 = 1.0 / 64.0;
        let mut chain =
            ThickChain::new(cfg, &row(0.0, l0, 1), fx(0.02), fx(DENSITY)).expect("a chain");
        chain.set_position(1, [fx(l0 * stretch), Fix128::ZERO]);
        chain.spans(0)
    };
    assert_eq!(count(8.0, 1.0), Ok(33));
    assert_eq!(count(8.0 + 1.0 / 1_048_576.0, 1.0), Ok(34));
    assert_eq!(count(4.0, 1.0), Ok(17));
    // the count does not follow the link's length
    assert_eq!(count(8.0, 0.5), Ok(33));
    assert_eq!(count(8.0, 8.0), Ok(33));
    // beyond its reach the link is refused
    assert_eq!(
        count(8.0, 8.0 + 1.0 / 1024.0),
        Err(ThickChainError::Stretched { link: 0 })
    );
}

/// a channel of half-width `c` with a pocket of half-width `wide` over `from..to`, its ends
/// sloped over 4 mm (a field with the walls as its zero set; not a distance in the slopes)
fn pocket(
    c: f32,
    wide: f32,
    from: f32,
    to: f32,
) -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(move |x: f32, y: f32, _z: f32| {
        let ramp = 0.004;
        let open = ((x - from) / ramp)
            .clamp(0.0, 1.0)
            .min(((to - x) / ramp).clamp(0.0, 1.0));
        c + (wide - c) * open - y.abs()
    })
}

#[test]
fn between_links_held_by_a_channel_a_link_in_a_pocket_is_off_its_walls_and_takes_the_push() {
    // nine links in a channel that holds them at λc = r0 / c, the head fixed, the tail pushed
    // with F; no friction. Where the channel has a pocket wide enough to leave a link free,
    // that link is not on its walls: it is as long as its own stretch makes it under the
    // push, T(λ) = −F (μ w0 (λ − λ⁻³) = −F), longer than the pocket's walls would make it,
    // while the links in the channel on both sides stay at the walls' length (they cannot be
    // shorter) and pass the push on. The walls of the two held runs are solved with the
    // free links between them off the condition.
    let (l0, r0, shear, c, wide) = (0.01, 0.02, 500.0, 0.012f32, 0.03f32);
    let (links, push) = (9usize, 9.0);
    let held = r0 * l0 / f64::from(c);
    let walls = pocket(c, wide, 0.05, 0.09);
    let mut cfg = config(shear, ThickChainFit::Conforming);
    cfg.drag = fx(30.0);
    let mut chain =
        ThickChain::new(cfg, &row(0.0, l0, links), fx(r0), fx(DENSITY)).expect("a chain");
    for i in 0..=links {
        // laid from the fixed head back, as long as the channel holds them
        chain.set_position(i, [fx(0.15 - held * (links - i) as f64), Fix128::ZERO]);
    }
    chain.pin(links);
    let mut accel = vec![ZERO2; links + 1];
    accel[0] = [fx(push / (DENSITY * r0 * l0)), Fix128::ZERO];
    let mut rested = false;
    for _ in 0..1800 {
        chain.step_in(frame(), &walls, &accel).expect("a step");
        if chain.velocities().iter().all(|v| f(v[0]).abs() < 1.0e-6) {
            rested = true;
            break;
        }
    }
    assert!(rested, "the chain does not come to rest");
    assert_eq!(
        chain.walls_unsettled(),
        0,
        "a pass left its walls unsettled"
    );
    // λ with pull(λ) = −F / (μ w0), by bisection (pull grows with λ)
    let (mut lo, mut hi) = (0.5f64, 1.0f64);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if shear * 2.0 * r0 * pull(mid) < -push {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let free = (lo + hi) / 2.0;
    let (mut in_pocket, mut before, mut after) = (0, 0, 0);
    for k in 0..links {
        let (a, b) = (f(chain.positions()[k][0]), f(chain.positions()[k + 1][0]));
        let ratio = f(chain.length(k)) / l0;
        if a >= 0.054 && b <= 0.086 {
            // wholly in the pocket's flat part: the drag at rest is below
            // 10 particles × 0.4 kg × 30 /s × 1e-6 m/s = 1.2e-4 N of the 9 N
            assert!(
                (ratio - free).abs() <= 1.0e-4 * free,
                "link {k} in the pocket is {ratio} of its rest length, its stretch under the push is {free}"
            );
            assert!(
                ratio > r0 / f64::from(wide) * 1.2,
                "link {k} is on the pocket's walls"
            );
            in_pocket += 1;
        } else if b <= 0.05 || a >= 0.09 {
            // wholly in the channel: at the walls' length (f32 walls, Q64.64 against f64)
            assert!(
                (ratio - held / l0).abs() <= 1.0e-6 * held / l0,
                "link {k} in the channel is {ratio} of its rest length, the walls make it {}",
                held / l0
            );
            if b <= 0.05 {
                before += 1;
            } else {
                after += 1;
            }
        }
    }
    assert!(
        in_pocket >= 1 && before >= 2 && after >= 2,
        "{in_pocket} links in the pocket, {before} and {after} in the channel on its sides"
    );
}
