//! Oracles of the thick chain's conforming rule
//! ([`alice_physics_core::thick_chain::ThickChainFit::Conforming`]) that settle a chain
//! over many frames; the closed forms and the rest of the rule's oracles are in
//! `tests/analytic_thick_chain_shares.rs`. This file is an integration test only (the
//! shares file also runs as a lib test, without optimisation, where these two take four
//! minutes).
//!
//! 1. Let go with its head at the open end of a channel narrower than itself, the chain
//!    leaves by itself (the shares at the mouth are pushed out by their own stress and pull
//!    the rest), and its kinetic energy is never above the energy the squeeze held,
//!    `A W(r0 / c)`.
//! 2. Between links held by a channel, the links in a pocket wider than their rest width
//!    are off their walls: at rest every link is either on its hard wall with a push that
//!    does not pull, or longer than its walls ask with no push.
//! 3. The walls of this file are fields the chain can be given.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: closed forms are evaluated in f64, never by the code under test"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_collider::DistanceSdfQuery;
use alice_physics_core::thick_chain::{
    ThickChain, ThickChainConfig, ThickChainFit, ThickChainStretch, P2,
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

/// the distance to the walls of a channel of half-width `c` with a pocket of half-width
/// `wide` over `from..to`, its two ends sloped over `ramp` (exact: the nearest point of the
/// upper wall's five runs, the lower wall by symmetry), positive between the walls
fn pocket_distance(c: f32, wide: f32, from: f32, to: f32, ramp: f32, x: f32, y: f32) -> f32 {
    let y = y.abs();
    let runs = [
        // the far ends are half a metre out, not further: a run's nearest point is found
        // along its length, and f32 keeps 3e-8 m at that size
        ([-0.5f32, c], [from, c]),
        ([from, c], [from + ramp, wide]),
        ([from + ramp, wide], [to - ramp, wide]),
        ([to - ramp, wide], [to, c]),
        ([to, c], [0.5, c]),
    ];
    let mut near = f32::INFINITY;
    for (p, q) in runs {
        let d = [q[0] - p[0], q[1] - p[1]];
        let t =
            (((x - p[0]) * d[0] + (y - p[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1])).clamp(0.0, 1.0);
        let (cx, cy) = (p[0] + t * d[0], p[1] + t * d[1]);
        near = near.min(((cx - x) * (cx - x) + (cy - y) * (cy - y)).sqrt());
    }
    let open = ((x - from) / ramp)
        .clamp(0.0, 1.0)
        .min(((to - x) / ramp).clamp(0.0, 1.0));
    if y < c + (wide - c) * open {
        near
    } else {
        -near
    }
}

fn pocket(
    c: f32,
    wide: f32,
    from: f32,
    to: f32,
    ramp: f32,
) -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(move |x: f32, y: f32, _z: f32| {
        pocket_distance(c, wide, from, to, ramp, x, y)
    })
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
fn between_links_held_by_a_channel_the_links_in_a_pocket_are_off_their_walls() {
    // sixteen links in a channel that holds them at λc = r0 / c, the head fixed, the tail pushed
    // with 9 N; no friction. The channel has a pocket wide enough to leave a link free. At
    // rest the links wholly in the channel are at the walls' length (they cannot be shorter,
    // and the push keeps them from being longer), and the links in the pocket whose
    // every sample asks less than their stretch are free of their walls: no press on them. The two runs of held links are solved with the free links between
    // them off the condition, and no pass leaves its walls unsettled. (How short the pocket's
    // links are is not asserted: they carry the push and what the two mouths add, a squeezed
    // link at a mouth being pushed towards the wide side.)
    let (l0, r0, shear, c, wide) = (0.01, 0.02, 500.0, 0.012f32, 0.03f32);
    let (links, push) = (16usize, 9.0);
    let (from, to, ramp) = (0.04f32, 0.12f32, 0.02f32);
    let held = r0 * l0 / f64::from(c);
    let walls = pocket(c, wide, from, to, ramp);
    let mut cfg = config(shear, ThickChainFit::Conforming);
    cfg.drag = fx(300.0);
    let mut chain =
        ThickChain::new(cfg, &row(0.0, l0, links), fx(r0), fx(DENSITY)).expect("a chain");
    for i in 0..=links {
        // laid from the fixed head back, as long as the channel holds them
        chain.set_position(i, [fx(0.17 - held * (links - i) as f64), Fix128::ZERO]);
    }
    chain.pin(links);
    let mut accel = vec![ZERO2; links + 1];
    accel[0] = [fx(push / (DENSITY * r0 * l0)), Fix128::ZERO];
    let mut rested = false;
    for _ in 0..3600 {
        chain.step_in(frame(), &walls, &accel).expect("a step");
        if chain.velocities().iter().all(|v| f(v[0]).abs() < 1.0e-6) {
            rested = true;
            break;
        }
    }
    let state: Vec<String> = (0..links)
        .map(|k| {
            format!(
                "{:.1}..{:.1} mm {:.4} ({:.2} N)",
                f(chain.positions()[k][0]) * 1000.0,
                f(chain.positions()[k + 1][0]) * 1000.0,
                f(chain.length(k)) / l0,
                f(chain.wall_force(k))
            )
        })
        .collect();
    println!("pocket: {}", state.join("; "));
    assert!(rested, "the chain does not come to rest");
    assert_eq!(
        chain.walls_unsettled(),
        0,
        "a pass left its walls unsettled"
    );
    // a link is free of its walls when every one of its samples asks less than its stretch
    // (the nearest wall of a sample near a mouth is the mouth's slope, not the pocket's flat
    // wall, so this is read from the field, not from the pocket's drawing)
    let (mut in_pocket, mut before, mut after) = (0, 0, 0);
    for k in 0..links {
        let (a, b) = (f(chain.positions()[k][0]), f(chain.positions()[k + 1][0]));
        let ratio = f(chain.length(k)) / l0;
        let spans = usize::try_from(chain.spans(k).expect("a count")).expect("a count");
        let most_asked = (0..=spans)
            .map(|j| {
                let x = a + (b - a) * j as f64 / spans as f64;
                r0 / f64::from(pocket_distance(c, wide, from, to, ramp, x as f32, 0.0)).max(LEAST)
            })
            .fold(0.0f64, f64::max);
        if most_asked < ratio * (1.0 - 1.0e-3) {
            assert!(
                chain.wall_force(k).is_zero(),
                "link {k} is free of its walls and presses on them"
            );
            in_pocket += 1;
        } else if b <= f64::from(from) || a >= f64::from(to) {
            // at the walls' length: f32 walls, Q64.64 against f64
            assert!(
                (ratio - held / l0).abs() <= 1.0e-6 * held / l0,
                "link {k} in the channel is {ratio} of its rest length, the walls make it {}",
                held / l0
            );
            assert!(
                chain.wall_force(k) > Fix128::ZERO,
                "link {k} in the channel does not press on its walls"
            );
            if b <= f64::from(from) {
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

/// a field the chain can be given: it never changes faster than the way between two points
/// (so it does not overstate the distance to a wall, the range the module states). Checked
/// on a lattice of `step` over the box, each point against its neighbours to the right,
/// above and on both diagonals; f32 values of points up to half a metre out (3e-8 m each), so
/// 2.5e-7 m is allowed.
fn assert_no_faster_than_the_way(
    what: &str,
    field: &dyn Fn(f32, f32) -> f32,
    (x0, x1): (f32, f32),
    (y0, y1): (f32, f32),
    step: f32,
) {
    let (nx, ny) = (((x1 - x0) / step) as i32, ((y1 - y0) / step) as i32);
    let mut pairs = 0u32;
    for i in 0..nx {
        for j in 0..ny {
            let a = [x0 + step * i as f32, y0 + step * j as f32];
            for (dx, dy) in [(step, 0.0), (0.0, step), (step, step), (step, -step)] {
                let b = [a[0] + dx, a[1] + dy];
                let way = (dx * dx + dy * dy).sqrt();
                let change = (field(a[0], a[1]) - field(b[0], b[1])).abs();
                assert!(
                    change <= way + 2.5e-7,
                    "{what}: from {a:?} to {b:?} the field changes by {change} over {way}"
                );
                pairs += 1;
            }
        }
    }
    assert!(pairs >= 1000, "{what}: {pairs} pairs compared");
}

#[test]
fn the_walls_of_the_pocket_tests_are_fields_the_chain_can_be_given() {
    // at 0.5 mm, finer than any ramp or step here
    assert_no_faster_than_the_way(
        "the mouth with a step",
        &|x, y| step_distance(0.012, 0.2, x, y),
        (-0.06, 0.06),
        (-0.03, 0.03),
        0.0005,
    );
    assert_no_faster_than_the_way(
        "the pocket",
        &|x, y| pocket_distance(0.012, 0.03, 0.04, 0.12, 0.02, x, y),
        (-0.12, 0.20),
        (-0.04, 0.04),
        0.0005,
    );
}
