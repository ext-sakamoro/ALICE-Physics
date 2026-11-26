//! Closed-form oracles of the thick chain's bending law
//! ([`alice_physics_core::thick_chain::ThickChainBend::Section`]): what a bent chain weighs,
//! carries and does, against the energy of the law written here in `f64`.
//!
//! The law: a link of area `A = 2 r0 l0`, stretched to `λ` at the middle of its width and
//! bent to the curvature `κ`, is Rivlin's flexure of an incompressible neo-Hookean block in
//! plane strain (1949): `E = (μ A / 2) [λ² (1 + a²) + λ⁻² artanh(a) / a − 2]`, `a = κ r`,
//! `r = r0 / λ`. At `a = 0` it is the link's stretch energy `(μ A / 2)(λ − 1/λ)²`; the rest,
//! `(μ A / 2) κ² r0² [1 + λ⁻⁴ g(a)]` with `g(a) = (artanh(a) / a − 1) / a²` (`1/3` at 0), is
//! the bending. A joint between two links has the curvature `κ = 2 tan(θ / 2) / l̄` (`θ` the
//! angle the links turn by, `l̄` the mean of their lengths), and each of the two links gives
//! the joint half of its bending energy at that curvature. So a chain's energy is a function
//! of its particles' places alone ([`energy`] below), and everything here follows from it:
//!
//! 1. Pure bending. A couple on the last link of a clamped chain bends the links between
//!    into one arc: every inner joint turns by the `θ` and every inner link is as long as the
//!    `λ` at which that energy less the couple's work is least; and every joint, those
//!    by the clamp and by the loaded link too, carries the couple, and no link a tension
//!    (the slopes of the energy in each turn and each length) — at three couples, up to
//!    most of the way to the radius of the half-width (bent, the links lengthen and thin of
//!    themselves). A pull on the head does not leave an arc (its line misses the joints of a
//!    bent chain), so the law under a pull is tested by 4. The arc is the same at 16, 32
//!    and 64 substeps to a frame and at 4, 8 and 16 passes to a substep.
//! 2. A clamped chain under its weight hangs where the discrete statics of small angles
//!    put it (the moment at a joint over `B / l`, `B = (8/3) μ r0³`), and that is above the
//!    beam's `q L⁴ / (8 B)` by a share that halves when the links are half as long.
//! 3. Pushed along itself at the head and a little across, a clamped chain leans by the
//!    beam-column's amplified deflection below the buckling load (the discrete statics
//!    solved here), and leaves its line above it.
//! 4. At rest under a dead load that bends it and stretches it unevenly, to 1.2 of its
//!    length and more, the load on every free particle is the slope of the energy there
//!    (central differences of [`energy`]): the law's force has every term of the energy's
//!    slope, the links' lengths among them.
//! 5. Let go bent and stretched, with no drag, the energy never grows; let go a hundredth of
//!    its length off its line, the head swings with the period of the first mode of the
//!    discrete chain; with a retardation time that swing, and the first swing along the
//!    chain, die as the substep's recurrence for a Kelvin-Voigt solid's mode does (the rate
//!    `τ ω² / 2`), at 16, 32 and 64 substeps to a frame and at 1, 8, 16 and 64 passes.
//! 6. Let down a centimetre onto a floor from a fixed tail, no link turns back; without the
//!    law one does.
//! 7. Pushed at the head with twenty times the buckling load, no joint is bent to the radius
//!    of its half-width; how many links far apart along the chain overlap is printed.
//! 8. A joint whose thicker link alone is past the radius of its half-width is refused.
//!
//! Integration test only: the chains settle over many frames.
#![allow(
    clippy::disallowed_methods,
    clippy::needless_range_loop,
    reason = "analytic reference: closed forms are evaluated in f64, never by the code under test; its solves index rows and columns as the algebra is written"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_collider::DistanceSdfQuery;
use alice_physics_core::thick_chain::{
    ThickChain, ThickChainBend, ThickChainConfig, ThickChainError, ThickChainFit,
    ThickChainStretch, P2,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn f(v: Fix128) -> f64 {
    v.to_f64()
}

const ZERO2: P2 = [Fix128::ZERO, Fix128::ZERO];
const LINK: f64 = 0.01;
const RADIUS: f64 = 0.02;
const DENSITY: f64 = 1000.0;
const SHEAR: f64 = 1000.0;
/// the area of a link, m²
const AREA: f64 = 2.0 * RADIUS * LINK;
/// the mass of a particle between two links, kg per metre of depth
const MASS: f64 = DENSITY * AREA;
/// the bending stiffness of the strip at rest, `(8/3) μ r0³`, N m per metre of depth
const BEND: f64 = 8.0 / 3.0 * SHEAR * RADIUS * RADIUS * RADIUS;

fn frame() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn config(drag: f64) -> ThickChainConfig {
    ThickChainConfig {
        substeps: 32,
        iterations: 8,
        gravity: ZERO2,
        stretch: ThickChainStretch::NeoHookean { shear: fx(SHEAR) },
        drag: fx(drag),
        friction: Fix128::ZERO,
        min_radius: fx(0.01),
        splits: 8,
        relaxation: Fix128::ZERO,
        fit: ThickChainFit::Disc,
        reach: Fix128::ONE,
    }
}

fn row(links: usize) -> Vec<P2> {
    (0..=links)
        .map(|i| [Fix128::from_ratio(i as i64, 100), Fix128::ZERO])
        .collect()
}

/// a straight chain of `links` links with the law, its first two particles fixed
fn clamped(cfg: ThickChainConfig, links: usize) -> ThickChain {
    let mut chain = ThickChain::new(cfg, &row(links), fx(RADIUS), fx(DENSITY))
        .expect("a chain")
        .bending(ThickChainBend::Section)
        .expect("the law");
    chain.pin(0);
    chain.pin(1);
    chain
}

/// no wall anywhere: the free space of the runs that carry a load on their particles
fn open() -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(|_x: f32, _y: f32, _z: f32| 1.0e3)
}

fn places(chain: &ThickChain) -> Vec<[f64; 2]> {
    chain
        .positions()
        .iter()
        .map(|p| [f(p[0]), f(p[1])])
        .collect()
}

// ---- the law in f64 ----

/// `g(a) = (artanh(a) / a − 1) / a²`
fn g(a: f64) -> f64 {
    if a.abs() < 1.0e-3 {
        1.0 / 3.0 + a * a / 5.0 + a.powi(4) / 7.0
    } else {
        (a.atanh() / a - 1.0) / (a * a)
    }
}

/// the bending energy a link of length `l` gives one of its joints at the curvature `kappa`
fn half(kappa: f64, l: f64) -> f64 {
    let lambda = l / LINK;
    let a = kappa * RADIUS / lambda;
    assert!(a.abs() < 1.0, "past the radius of the half-width: a = {a}");
    SHEAR * AREA / 4.0 * kappa * kappa * RADIUS * RADIUS * (1.0 + lambda.powi(-4) * g(a))
}

fn stretch(l: f64) -> f64 {
    let lambda = l / LINK;
    SHEAR * AREA / 2.0 * (lambda - 1.0 / lambda).powi(2)
}

/// the angle the chain turns by at particle `i`
fn turn(x: &[[f64; 2]], i: usize) -> f64 {
    let (a, b) = (
        [x[i][0] - x[i - 1][0], x[i][1] - x[i - 1][1]],
        [x[i + 1][0] - x[i][0], x[i + 1][1] - x[i][1]],
    );
    (a[0] * b[1] - a[1] * b[0]).atan2(a[0] * b[0] + a[1] * b[1])
}

fn length(x: &[[f64; 2]], k: usize) -> f64 {
    (x[k + 1][0] - x[k][0]).hypot(x[k + 1][1] - x[k][1])
}

fn curvature(x: &[[f64; 2]], i: usize) -> f64 {
    2.0 * (turn(x, i) / 2.0).tan() / ((length(x, i - 1) + length(x, i)) / 2.0)
}

/// the elastic energy of a chain through `x`, J per metre of depth
fn energy(x: &[[f64; 2]]) -> f64 {
    let links = x.len() - 1;
    let mut total: f64 = (0..links).map(|k| stretch(length(x, k))).sum();
    for i in 1..links {
        let kappa = curvature(x, i);
        total += half(kappa, length(x, i - 1)) + half(kappa, length(x, i));
    }
    total
}

/// the largest `a = κ r` of a joint's two links, over the chain
fn most_bent(x: &[[f64; 2]]) -> f64 {
    let links = x.len() - 1;
    (1..links)
        .map(|i| {
            let kappa = curvature(x, i).abs();
            let thin = |k: usize| kappa * RADIUS * LINK / length(x, k);
            thin(i - 1).max(thin(i))
        })
        .fold(0.0, f64::max)
}

// ---- 1. pure bending ----

/// the energy of one inner link and one inner joint of a uniform arc: links `lambda` long,
/// joints turning by `theta`
fn cell(lambda: f64, theta: f64) -> f64 {
    let l = lambda * LINK;
    stretch(l) + 2.0 * half(2.0 * (theta / 2.0).tan() / l, l)
}

/// the root of a function that grows through 0 between `lo` and `hi`
fn root(mut lo: f64, mut hi: f64, grows: impl Fn(f64) -> f64) -> f64 {
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if grows(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// the largest turn of a joint between links `lambda` long, short of the radius of the
/// half-width by a part in a thousand
fn most(lambda: f64) -> f64 {
    2.0 * (0.999 * lambda * lambda * LINK / (2.0 * RADIUS)).atan()
}

/// the stretch at which a link in an arc of turn `theta` carries the pull `pull`: where
/// `cell − pull · l` is least in `lambda`
fn stretch_at(theta: f64, pull: f64) -> f64 {
    // the least stretch at which this turn is inside the radius of the half-width
    let least = (2.0 * RADIUS * (theta / 2.0).tan() / (0.999 * LINK)).sqrt();
    let d = 1.0e-7;
    root(least.max(0.3) + 2.0 * d, 6.0, |lambda| {
        (cell(lambda + d, theta) - cell(lambda - d, theta)) / (2.0 * d) - pull * LINK
    })
}

/// the turn and the stretch of the arc under the couple `couple` and the pull `pull`
fn arc(couple: f64, pull: f64) -> (f64, f64) {
    let d = 1.0e-7;
    let work = |theta: f64| {
        let lambda = stretch_at(theta, pull);
        cell(lambda, theta) - pull * lambda * LINK
    };
    let theta = root(1.0e-4, 3.0, |theta| {
        (work(theta + d) - work(theta - d)) / (2.0 * d) - couple
    });
    (theta, stretch_at(theta, pull))
}

/// A clamped chain of ten links under a couple and a pull on its last link, both turning
/// with the link, until it rests, at `substeps` substeps to a frame and `passes` passes a
/// substep; its places.
fn bent_at(couple: f64, pull: f64, substeps: u32, passes: u32) -> Vec<[f64; 2]> {
    bent_for(couple, pull, substeps, passes, 480)
}

/// [`bent_at`] over `frames` frames. The load is set anew every substep (each substep is
/// its own step): a couple that turns with its link does work by the link's angle alone,
/// so it has an energy, and the chain comes to rest; set once a frame it is a pair of
/// forces of fixed directions for 32 substeps, which near the fold keeps the chain moving
/// (measured: 3.9e-3 J of motion left after 32 s, the joints 2 % from the arc and
/// different at every substep count). The chain is at rest at the end, or this panics.
fn bent_for(couple: f64, pull: f64, substeps: u32, passes: u32, frames: usize) -> Vec<[f64; 2]> {
    const LINKS: usize = 10;
    let mut cfg = config(20.0);
    cfg.substeps = 1;
    cfg.iterations = passes;
    let mut chain = clamped(cfg, LINKS);
    let walls = open();
    let substep = Fix128::from_ratio(1, 60 * i64::from(substeps));
    let ramp = 120.0 * f64::from(substeps);
    for step_no in 0..frames * substeps as usize {
        let x = places(&chain);
        let l = length(&x, LINKS - 1);
        let u = [
            (x[LINKS][0] - x[LINKS - 1][0]) / l,
            (x[LINKS][1] - x[LINKS - 1][1]) / l,
        ];
        let n = [-u[1], u[0]];
        // the couple as a pair of forces across the link, the pull along it on the head;
        // both come on over the first two seconds, so the chain is not thrown at the fold
        let on = (step_no as f64 / ramp).min(1.0);
        let (pair, pull) = (on * couple / l, on * pull);
        let mut accel = vec![ZERO2; LINKS + 1];
        accel[LINKS - 1] = [fx(-pair * n[0] / MASS), fx(-pair * n[1] / MASS)];
        accel[LINKS] = [
            fx((pair * n[0] + pull * u[0]) / (MASS / 2.0)),
            fx((pair * n[1] + pull * u[1]) / (MASS / 2.0)),
        ];
        chain.step_in(substep, &walls, &accel).expect("a step");
    }
    let (moving, unsettled) = (kinetic(&chain), f(chain.last_pass_move()));
    println!(
        "[thick chain bend] couple {couple:.4}, {substeps} substeps, {passes} passes, {frames} frames: kinetic energy {moving:.3e} J, the last pass moved {unsettled:.3e} m, steps halved {}",
        chain.bend_halved()
    );
    assert!(
        moving < 1.0e-12 * energy(&places(&chain)),
        "not at rest: {moving} J of motion"
    );
    places(&chain)
}

fn bent_by(couple: f64, pull: f64) -> Vec<[f64; 2]> {
    bent_at(couple, pull, 32, 8)
}

/// the places of a chain from its joints' turns (`turns[i]` at particle `i`) and its links'
/// lengths, the first link along `x`
fn shape(turns: &[f64], lengths: &[f64]) -> Vec<[f64; 2]> {
    let mut x = vec![[0.0, 0.0]];
    let mut way = 0.0;
    for (k, l) in lengths.iter().enumerate() {
        way += turns[k];
        let last = x[k];
        x.push([last[0] + l * way.cos(), last[1] + l * way.sin()]);
    }
    x
}

/// The moment each joint carries and the tension each link carries in the chain through
/// `x`: the slopes of [`energy`] in the joint's turn and in the link's length (central
/// differences, the other turns and lengths held).
fn carried(x: &[[f64; 2]]) -> (Vec<f64>, Vec<f64>) {
    let links = x.len() - 1;
    let mut turns = vec![0.0; links];
    for (i, turned) in turns.iter_mut().enumerate().skip(1) {
        *turned = turn(x, i);
    }
    let lengths: Vec<f64> = (0..links).map(|k| length(x, k)).collect();
    let d = 1.0e-7;
    let slope =
        |turns: &mut Vec<f64>, lengths: &mut Vec<f64>, turned: bool, at: usize, step: f64| {
            let value = if turned {
                &mut turns[at]
            } else {
                &mut lengths[at]
            };
            let kept = *value;
            *value = kept + step;
            let up = energy(&shape(turns, lengths));
            let value = if turned {
                &mut turns[at]
            } else {
                &mut lengths[at]
            };
            *value = kept - step;
            let down = energy(&shape(turns, lengths));
            let value = if turned {
                &mut turns[at]
            } else {
                &mut lengths[at]
            };
            *value = kept;
            (up - down) / (2.0 * step)
        };
    let (mut t, mut l) = (turns.clone(), lengths.clone());
    let moments = (1..links)
        .map(|i| slope(&mut t, &mut l, true, i, d))
        .collect();
    let tensions = (1..links)
        .map(|k| slope(&mut t, &mut l, false, k, d * LINK))
        .collect();
    (moments, tensions)
}

fn pure_bending(couple: f64, pull: f64) {
    let (theta, lambda) = arc(couple, pull);
    let x = bent_by(couple, pull);
    println!(
        "[thick chain bend] couple {couple:.4}: the arc turns {theta:.6} with links {lambda:.6}; joints 1..=9 turn {:?}; links 0..=9 are {:?} long",
        (1..=9).map(|i| (turn(&x, i).abs() * 1e6).round() / 1e6).collect::<Vec<_>>(),
        (0..=9).map(|k| (length(&x, k) / LINK * 1e6).round() / 1e6).collect::<Vec<_>>()
    );
    // at rest under a couple alone every joint carries the couple and no link a tension,
    // the joints and links by the clamp and by the loaded link among them (there the chain
    // is not the arc: the fixed link is not stretched, the last has a joint at one end only)
    let (moments, tensions) = carried(&x);
    for (i, moment) in moments.iter().enumerate() {
        assert!(
            (moment.abs() / couple - 1.0).abs() < 1.0e-3,
            "couple {couple}: joint {} carries {moment}",
            i + 1
        );
    }
    // (a tension against the pull that alone would hold a link at 1.01 of its length)
    let small = 1.0e-3 * SHEAR * 2.0 * RADIUS * (1.01 - 1.01f64.powi(-3));
    for (k, tension) in tensions.iter().enumerate() {
        assert!(
            (tension - pull).abs() < small,
            "couple {couple} pull {pull}: link {} carries {tension}",
            k + 1
        );
    }
    // away from the ends the chain is the arc of least energy
    for i in 4..=7 {
        let got = turn(&x, i).abs();
        assert!(
            (got / theta - 1.0).abs() < 2.0e-3,
            "couple {couple} pull {pull}: joint {i} turns {got}, the arc {theta}"
        );
    }
    for k in 4..=6 {
        let got = length(&x, k) / LINK;
        assert!(
            (got / lambda - 1.0).abs() < 2.0e-4,
            "couple {couple} pull {pull}: link {k} is {got} long, the arc {lambda}"
        );
    }
}

/// the couple that bends links of rest length to `a = κ r` (without their lengthening)
fn couple_for(a: f64) -> f64 {
    let theta = 2.0 * (a * LINK / (2.0 * RADIUS)).atan();
    let d = 1.0e-7;
    (cell(1.0, theta + d) - cell(1.0, theta - d)) / (2.0 * d)
}

#[test]
fn a_couple_bends_the_inner_links_to_the_arc_of_least_energy_gently() {
    pure_bending(couple_for(0.1), 0.0);
}

#[test]
fn a_couple_bends_the_inner_links_to_the_arc_of_least_energy_halfway_to_the_fold() {
    pure_bending(couple_for(0.5), 0.0);
}

#[test]
fn a_couple_bends_the_inner_links_to_the_arc_of_least_energy_near_the_fold() {
    let couple = couple_for(0.8);
    // the arc is where the law says, and that is not where a law without the wall would
    // put it: the links lengthen and thin as they bend
    let (theta, lambda) = arc(couple, 0.0);
    assert!(lambda > 1.02, "the arc's links: {lambda}");
    assert!(theta < most(lambda));
    pure_bending(couple, 0.0);
}

#[test]
fn the_arc_under_a_couple_is_the_same_whatever_the_substep_and_the_passes() {
    // a rest state is a property of the energy, not of the step that reaches it: halfway to
    // the fold, with 16, 32 and 64 substeps to a frame and with 4 and 16 passes to a substep
    let couple = couple_for(0.5);
    let (theta, lambda) = arc(couple, 0.0);
    let mut seen = Vec::new();
    for (substeps, passes) in [(16, 8), (32, 8), (64, 8), (32, 4), (32, 16)] {
        let x = bent_at(couple, 0.0, substeps, passes);
        let (turned, long) = (turn(&x, 5).abs(), length(&x, 5) / LINK);
        assert!(
            (turned / theta - 1.0).abs() < 2.0e-3 && (long / lambda - 1.0).abs() < 2.0e-4,
            "{substeps} substeps, {passes} passes: joint 5 turns {turned} (the arc {theta}), link 5 is {long} long (the arc {lambda})"
        );
        seen.push(turned);
    }
    let (least, most) = seen.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &t| {
        (lo.min(t), hi.max(t))
    });
    assert!(most / least - 1.0 < 5.0e-4, "{seen:?}");
}

// ---- 2 and 3. small angles: the discrete statics ----

/// The deflections of a clamped chain of `links` free links of length `l` under the forces
/// `load` across it on its particles (particle 0 is the clamp's joint) and the push `push`
/// along it on its head, by small angles: the moment at joint `j` is
/// `Σ load_i (x_i − x_j) + push (y_head − y_j)` and turns the joint by `moment · l / bend`.
/// Solved by elimination.
fn deflections(links: usize, l: f64, bend: f64, load: &[f64], push: f64) -> Vec<f64> {
    // unknowns: the turns of joints 0..links; y_i = l Σ_{j<i} (i − j) turn_j
    let n = links;
    let lever = |i: usize, j: usize| if i > j { (i - j) as f64 * l } else { 0.0 };
    let mut m = vec![vec![0.0f64; n + 1]; n];
    for j in 0..n {
        m[j][j] += bend / l;
        // push (y_head − y_j) in the turns
        for (t, entry) in m[j].iter_mut().enumerate().take(n) {
            *entry -= push * (lever(n, t) - lever(j, t));
        }
        m[j][n] = (j + 1..=n).map(|i| load[i] * lever(i, j)).sum();
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))
            .expect("a row");
        m.swap(col, pivot);
        for r in 0..n {
            if r != col {
                let by = m[r][col] / m[col][col];
                for c in col..=n {
                    m[r][c] -= by * m[col][c];
                }
            }
        }
    }
    let turns: Vec<f64> = (0..n).map(|j| m[j][n] / m[j][j]).collect();
    (0..=n)
        .map(|i| (0..i).map(|j| lever(i, j) * turns[j]).sum())
        .collect()
}

/// the head's sag of a clamped chain of `links` free links under the gravity `g`, at rest
fn sag(links: usize, g: f64, drag: f64, frames: usize) -> f64 {
    let mut cfg = config(drag);
    cfg.gravity = [Fix128::ZERO, fx(-g)];
    let mut chain = clamped(cfg, links + 1);
    for _ in 0..frames {
        chain.step(frame()).expect("a step");
    }
    -places(&chain)[links + 1][1]
}

#[test]
fn a_clamped_chain_sags_by_the_discrete_statics_and_nears_the_beam_as_its_links_shorten() {
    let mut over = Vec::new();
    for (links, drag, frames) in [(10usize, 16.0, 150usize), (20, 4.0, 480)] {
        let reach = links as f64 * LINK;
        // the gravity under which the beam's head sags a hundredth of its length
        let beam = 0.01 * reach;
        let g = 8.0 * BEND * beam / (reach.powi(4) * DENSITY * 2.0 * RADIUS);
        let mut load = vec![-MASS * g; links + 1];
        load[links] /= 2.0;
        let statics = -deflections(links, LINK, BEND, &load, 0.0)[links];
        let got = sag(links, g, drag, frames);
        assert!(
            (got / statics - 1.0).abs() < 5.0e-3,
            "{links} links: sags {got}, the statics {statics}"
        );
        over.push((statics / beam - 1.0) * links as f64);
    }
    // the joints carry the bending in lumps: above the beam by about 2.1 / N
    for share in &over {
        assert!((share - 2.1).abs() < 0.1, "{over:?}");
    }
}

/// A clamped chain of twenty free links pushed along itself by `push` on its head and
/// across by `across`: the head's deflection after `frames`, or the frame it was a tenth of
/// the chain's length off its line.
fn leaning(push: f64, across: f64, drag: f64, frames: usize) -> Result<f64, usize> {
    const LINKS: usize = 20;
    let mut chain = clamped(config(drag), LINKS + 1);
    let walls = open();
    let mut accel = vec![ZERO2; LINKS + 2];
    accel[LINKS + 1] = [fx(-push / (MASS / 2.0)), fx(across / (MASS / 2.0))];
    for frame_no in 0..frames {
        chain.step_in(frame(), &walls, &accel).expect("a step");
        let y = f(chain.positions()[LINKS + 1][1]);
        if y.abs() > 0.1 * LINKS as f64 * LINK {
            return Err(frame_no);
        }
    }
    Ok(f(chain.positions()[LINKS + 1][1]))
}

/// the stretch of a link pushed along by `push` (`μ w0 (λ − λ⁻³) = −push`), and the
/// bending stiffness of the strip at that stretch, `2 μ r0³ (1 + λ⁻⁴ / 3) / λ`
fn pushed(push: f64) -> (f64, f64) {
    let lambda = root(0.3, 1.0, |lambda: f64| {
        SHEAR * 2.0 * RADIUS * (lambda - lambda.powi(-3)) + push
    });
    (
        lambda,
        2.0 * SHEAR * RADIUS.powi(3) * (1.0 + lambda.powi(-4) / 3.0) / lambda,
    )
}

#[test]
fn pushed_along_below_the_buckling_load_a_clamped_chain_leans_by_the_beam_columns_share() {
    const LINKS: usize = 20;
    let reach = LINKS as f64 * LINK;
    let euler = std::f64::consts::PI.powi(2) * BEND / (4.0 * reach * reach);
    // the push across that alone moves the beam's head 2e-4 of a metre
    let across = 3.0 * BEND * 2.0e-4 / reach.powi(3);
    let mut seen = Vec::new();
    for share in [0.0, 0.5, 0.8] {
        let push = share * euler;
        let (lambda, bend) = pushed(push);
        let mut load = vec![0.0; LINKS + 1];
        load[LINKS] = across;
        let statics = deflections(LINKS, lambda * LINK, bend, &load, push)[LINKS];
        // the drag at which the softened first mode is critically damped
        let beam = 3.516 * (BEND / (DENSITY * 2.0 * RADIUS * reach.powi(4))).sqrt();
        let drag = 2.0 * beam * (1.0 - share).sqrt();
        let got = leaning(push, across, drag, 900).expect("on its line");
        assert!(
            (got / statics - 1.0).abs() < 1.0e-2,
            "push {share} of the buckling load: leans {got}, the statics {statics}"
        );
        seen.push(got);
    }
    // the push along is what amplifies: about 2 and 5 times at a half and at four fifths
    assert!(
        seen[1] / seen[0] > 1.7 && seen[1] / seen[0] < 2.3,
        "{seen:?}"
    );
    assert!(
        seen[2] / seen[0] > 4.0 && seen[2] / seen[0] < 6.0,
        "{seen:?}"
    );
}

#[test]
fn pushed_along_above_the_buckling_load_a_clamped_chain_leaves_its_line() {
    const LINKS: usize = 20;
    let reach = LINKS as f64 * LINK;
    let euler = std::f64::consts::PI.powi(2) * BEND / (4.0 * reach * reach);
    let across = 3.0 * BEND * 2.0e-4 / reach.powi(3);
    assert!(leaning(1.3 * euler, across, 0.5, 1500).is_err());
    // and with nothing across it, a straight chain stays on its line whatever the push
    assert_eq!(leaning(1.3 * euler, 0.0, 0.5, 120), Ok(0.0));
}

// ---- 4. the force is the slope of the energy ----

#[test]
fn at_rest_under_a_dead_load_the_load_on_every_particle_is_the_slope_of_the_energy() {
    const LINKS: usize = 8;
    let mut chain = clamped(config(20.0), LINKS);
    let walls = open();
    // across and along at the head, and against it in the middle: the links are bent and
    // stretched unevenly
    let mut force = [[0.0f64; 2]; LINKS + 1];
    force[LINKS] = [30.0, 12.0];
    force[5] = [-8.0, -22.0];
    let accel: Vec<P2> = force
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mass = if i == LINKS { MASS / 2.0 } else { MASS };
            [fx(p[0] / mass), fx(p[1] / mass)]
        })
        .collect();
    for _ in 0..480 {
        chain.step_in(frame(), &walls, &accel).expect("a step");
    }
    let x = places(&chain);
    // the scene bends and stretches: neither alone would test the other's terms
    assert!(most_bent(&x) > 0.2, "bent to a = {}", most_bent(&x));
    let longest = (0..LINKS).map(|k| length(&x, k)).fold(0.0, f64::max);
    let shortest = (0..LINKS).map(|k| length(&x, k)).fold(1.0, f64::min);
    assert!(longest / shortest > 1.03, "{shortest} to {longest}");
    assert!(longest / LINK > 1.2, "the longest link {longest}");
    let d = 1.0e-8;
    for i in 2..=LINKS {
        for axis in 0..2 {
            let (mut up, mut down) = (x.clone(), x.clone());
            up[i][axis] += d;
            down[i][axis] -= d;
            let slope = (energy(&up) - energy(&down)) / (2.0 * d);
            assert!(
                (slope - force[i][axis]).abs() < 1.0e-3 * 30.0,
                "particle {i} axis {axis}: the energy's slope {slope}, the load {}",
                force[i][axis]
            );
        }
    }
}

// ---- 5. let go ----

fn kinetic(chain: &ThickChain) -> f64 {
    let last = chain.particle_count() - 1;
    chain
        .velocities()
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mass = if i == 0 || i == last {
                MASS / 2.0
            } else {
                MASS
            };
            0.5 * mass * (f(v[0]).powi(2) + f(v[1]).powi(2))
        })
        .sum()
}

#[test]
fn let_go_bent_and_stretched_the_energy_never_grows() {
    const LINKS: usize = 10;
    let mut cfg = config(0.0);
    cfg.iterations = 32;
    let mut chain = clamped(cfg, LINKS);
    // an arc of links 1.15 long, each joint turning 0.12 rad
    let (mut at, mut way) = ([LINK, 0.0f64], 0.0f64);
    for i in 2..=LINKS {
        way += 0.12;
        at = [
            at[0] + 1.15 * LINK * way.cos(),
            at[1] + 1.15 * LINK * way.sin(),
        ];
        chain.set_position(i, [fx(at[0]), fx(at[1])]);
    }
    let mut before = energy(&places(&chain));
    let first = before;
    let mut least = before;
    let mut grew = 0;
    for step in 0..180 {
        chain.step(frame()).expect("a step");
        let now = energy(&places(&chain)) + kinetic(&chain);
        if now > before + 1.0e-7 * first {
            println!(
                "[probe] frame {step}: +{:.3e} of {:.3e}",
                now - before,
                before
            );
            grew += 1;
        }
        before = now;
        least = least.min(now);
    }
    assert_eq!(grew, 0, "frames in which the energy grew");
    // it moved: the implicit step took a share of the energy
    assert!(least < 0.999 * first, "{least} of {first}");
}

#[test]
fn let_go_a_little_off_its_line_the_head_swings_with_the_first_modes_period() {
    const LINKS: usize = 20;
    let reach = LINKS as f64 * LINK;
    // the first mode of the discrete chain: the largest eigenvalue of (deflections under a
    // unit load) × (masses) is 1 / ω², by repeated application
    let mass: Vec<f64> = (0..=LINKS)
        .map(|i| if i == LINKS { MASS / 2.0 } else { MASS })
        .collect();
    let mut shape = vec![1.0f64; LINKS + 1];
    let mut omega2 = 0.0;
    for _ in 0..60 {
        let load: Vec<f64> = shape.iter().zip(&mass).map(|(y, m)| y * m).collect();
        let next = deflections(LINKS, LINK, BEND, &load, 0.0);
        omega2 = shape[LINKS] / next[LINKS];
        let top = next[LINKS];
        shape = next.iter().map(|y| y / top).collect();
    }
    let period = 2.0 * std::f64::consts::PI / omega2.sqrt();
    // (the beam's first mode, 3.516 √(B / (m L⁴)), is within a tenth of it)
    let beam = 3.516 * (BEND / (DENSITY * 2.0 * RADIUS * reach.powi(4))).sqrt();
    assert!((omega2.sqrt() / beam - 1.0).abs() < 0.1);

    let swing = |side: f64| {
        let mut chain = clamped(config(0.0), LINKS + 1);
        // the first mode's shape, a hundredth of the chain's length at the head
        for i in 1..=LINKS {
            chain.set_position(
                i + 1,
                [
                    fx(LINK * (i + 1) as f64),
                    fx(side * 0.01 * reach * shape[i]),
                ],
            );
        }
        let mut crossings = Vec::new();
        let mut last = f(chain.positions()[LINKS + 1][1]);
        let mut track = Vec::new();
        for step in 0..(2.6 * period * 60.0) as usize {
            chain.step(frame()).expect("a step");
            let y = f(chain.positions()[LINKS + 1][1]);
            if last * side > 0.0 && y * side <= 0.0 {
                // the frame's fraction at which the head crossed its line
                crossings.push((step as f64 + last / (last - y)) / 60.0);
            }
            last = y;
            track.push(chain.positions()[LINKS + 1][1]);
        }
        (crossings, track)
    };
    let (crossings, up) = swing(1.0);
    assert!(crossings.len() >= 2, "{crossings:?}");
    let got = crossings[1] - crossings[0];
    assert!(
        (got / period - 1.0).abs() < 1.5e-2,
        "the head's period {got}, the first mode's {period}"
    );
    // let go on the other side it swings mirrored, bit for bit
    let (_, down) = swing(-1.0);
    assert!(up.iter().zip(&down).all(|(a, b)| *a == -*b));
}

/// The first mode's shape (the head at 1) and its `ω²`, of a clamped chain of `links` free
/// links: the largest eigenvalue of (deflections under a unit load) × (masses) is `1 / ω²`.
fn first_mode(links: usize) -> (Vec<f64>, f64) {
    let mass: Vec<f64> = (0..=links)
        .map(|i| if i == links { MASS / 2.0 } else { MASS })
        .collect();
    let mut shape = vec![1.0f64; links + 1];
    let mut omega2 = 0.0;
    for _ in 0..60 {
        let load: Vec<f64> = shape.iter().zip(&mass).map(|(y, m)| y * m).collect();
        let next = deflections(links, LINK, BEND, &load, 0.0);
        omega2 = shape[links] / next[links];
        let top = next[links];
        shape = next.iter().map(|y| y / top).collect();
    }
    (shape, omega2)
}

/// The steps of 1/240 s in one and a half periods of `omega2`, and the step from which the
/// swing is read (half a period in).
fn window(omega2: f64) -> (usize, usize) {
    let period = 2.0 * std::f64::consts::PI / omega2.sqrt();
    (
        (1.5 * period * 240.0) as usize,
        (0.5 * period * 240.0) as usize,
    )
}

/// oracle: the substep's recurrence for a mode of frequency `ω` with the retardation time
/// `τ` (the MODEL section of the chain), `e₊ (1 + ω² h² + τ ω² h) = e (1 + τ ω² h) +
/// (e − e₋)`, from rest at 1, stepped `per_step` substeps to each step of 1/240 s: the
/// most it is in the window, the swing left a period on (`exp(−τ ω² T / 2)` nearly, as a
/// Kelvin-Voigt solid's mode dies).
fn recurrence_left(omega2: f64, tau: f64, per_step: u32) -> f64 {
    let h = 1.0 / 240.0 / f64::from(per_step);
    let (stiff, slow) = (omega2 * h * h, tau * omega2 * h);
    let (steps, from) = window(omega2);
    let (mut e, mut before) = (1.0f64, 1.0f64);
    let mut top = f64::MIN;
    for step in 0..steps {
        for _ in 0..per_step {
            let next = (e * (1.0 + slow) + (e - before)) / (1.0 + stiff + slow);
            before = e;
            e = next;
        }
        if step > from {
            top = top.max(e);
        }
    }
    top
}

const SWING_LINKS: usize = 12;

/// The head's height a period after a clamped chain of [`SWING_LINKS`] free links is let
/// go in its first bending mode (a hundredth of its length at the head), over its height
/// when let go; and the most a particle moved in the last pass of the last substep.
fn bend_swing_left(tau: f64, per_step: u32, passes: u32) -> (f64, f64) {
    let (shape, omega2) = first_mode(SWING_LINKS);
    let mut cfg = config(0.0);
    cfg.relaxation = fx(tau);
    cfg.substeps = per_step;
    cfg.iterations = passes;
    let mut chain = clamped(cfg, SWING_LINKS + 1);
    let first = 0.01 * SWING_LINKS as f64 * LINK;
    for i in 1..=SWING_LINKS {
        chain.set_position(i + 1, [fx(LINK * (i + 1) as f64), fx(first * shape[i])]);
    }
    let (steps, from) = window(omega2);
    let mut top = f64::MIN;
    for step in 0..steps {
        chain.step(Fix128::from_ratio(1, 240)).expect("a step");
        if step > from {
            top = top.max(f(chain.positions()[SWING_LINKS + 1][1]));
        }
    }
    (top / first, f(chain.last_pass_move()))
}

/// The same of a chain held at its tail and let go stretched in its first mode along
/// itself (a five-hundredth of its length at the head): masses `m` (the head `m / 2`) on
/// springs `k = 4 μ w0 / l0`, `u_i = sin(i θ)`, `θ = π / (2 N)`, `ω = 2 √(k / m) sin(θ / 2)`.
fn stretch_swing_left(tau: f64, per_step: u32, passes: u32) -> (f64, f64) {
    let omega2 = stretch_mode();
    let theta = std::f64::consts::PI / (2.0 * SWING_LINKS as f64);
    let mut cfg = config(0.0);
    cfg.relaxation = fx(tau);
    cfg.substeps = per_step;
    cfg.iterations = passes;
    let mut chain = ThickChain::new(cfg, &row(SWING_LINKS), fx(RADIUS), fx(DENSITY))
        .expect("a chain")
        .bending(ThickChainBend::Section)
        .expect("the law");
    chain.pin(0);
    let first = 0.002 * SWING_LINKS as f64 * LINK;
    for i in 1..=SWING_LINKS {
        let x = LINK * i as f64 + first * (i as f64 * theta).sin();
        chain.set_position(i, [fx(x), Fix128::ZERO]);
    }
    let (steps, from) = window(omega2);
    let mut top = f64::MIN;
    for step in 0..steps {
        chain.step(Fix128::from_ratio(1, 240)).expect("a step");
        if step > from {
            top = top.max(f(chain.positions()[SWING_LINKS][0]) - LINK * SWING_LINKS as f64);
        }
    }
    (top / first, f(chain.last_pass_move()))
}

fn stretch_mode() -> f64 {
    let k = 4.0 * SHEAR * 2.0 * RADIUS / LINK;
    let theta = std::f64::consts::PI / (2.0 * SWING_LINKS as f64);
    let omega = 2.0 * (k / MASS).sqrt() * (theta / 2.0).sin();
    omega * omega
}

/// What is left of a swing may differ from the recurrence's by this share of itself,
/// whatever the substep and the passes: both are read at the same instants, 1/240 s apart,
/// so what differs is the mode itself, which is the small-angle one only to the square of
/// the swing over the chain's length and which the recurrence takes as one frequency. (On
/// the solve of joint after joint the shares were 5.4 % at 32 substeps to a frame and 8
/// passes, 16 % at 16 substeps, and 0.003 % at 64 passes.) Where there is more than one
/// pass the last one moved nothing that shows: under a millionth of the swing.
const SWING_TOLERANCE: f64 = 2e-3;

/// substeps to a step of 1/240 s (4, 8, 16: 16, 32, 64 to 1/60 s) and passes
const SWING_GRID: [(u32, u32); 7] = [(8, 1), (8, 8), (8, 16), (8, 64), (4, 8), (16, 8), (16, 1)];

fn dies_as_the_recurrence(
    name: &str,
    omega2: f64,
    first: f64,
    taus: [f64; 2],
    swing: impl Fn(f64, u32, u32) -> (f64, f64),
) {
    let mut off = Vec::new();
    for (per_step, passes) in SWING_GRID {
        for tau in taus {
            let want = recurrence_left(omega2, tau, per_step);
            let (got, moved) = swing(tau, per_step, passes);
            let ratio = got / want;
            println!(
                "{name} substeps {per_step} passes {passes} tau {tau}: left {got:.6}, the recurrence's {want:.6}, of it {ratio:.5}, last pass moved {moved:e}"
            );
            // (a single pass is all of the substep's step: what it moved says nothing)
            if (ratio - 1.0).abs() > SWING_TOLERANCE || (passes > 1 && moved > 1e-6 * first) {
                off.push((per_step, passes, tau, ratio, moved));
            }
        }
    }
    assert!(
        off.is_empty(),
        "{name}: (substeps, passes, τ, left of the recurrence's, moved) {off:?}"
    );
}

#[test]
fn with_a_retardation_time_the_bending_swing_dies_at_the_kelvin_voigt_rate_whatever_the_passes() {
    let (_, omega2) = first_mode(SWING_LINKS);
    // the case tells: with a twentieth of a second more than a quarter goes in a period,
    // without it under a hundredth
    assert!(recurrence_left(omega2, 0.0, 8) > 0.99);
    assert!(recurrence_left(omega2, 0.05, 8) < 0.75);
    dies_as_the_recurrence(
        "bending",
        omega2,
        0.01 * SWING_LINKS as f64 * LINK,
        [0.0, 0.05],
        bend_swing_left,
    );
}

#[test]
fn with_a_retardation_time_the_stretch_swing_dies_at_the_kelvin_voigt_rate_whatever_the_passes() {
    let omega2 = stretch_mode();
    assert!(recurrence_left(omega2, 0.004, 8) < 0.75);
    dies_as_the_recurrence(
        "stretch",
        omega2,
        0.002 * SWING_LINKS as f64 * LINK,
        [0.0, 0.004],
        stretch_swing_left,
    );
}

// ---- 6 and 7. what the law is for ----

/// the free space above a floor at `y = 0`
fn floor() -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(|_x: f32, y: f32, _z: f32| y)
}

/// A chain of twenty links 30 mm in half-width let down a centimetre onto a floor from its
/// fixed tail, for three seconds: its places, or the step's refusal.
fn let_down(law: ThickChainBend) -> Result<Vec<[f64; 2]>, ThickChainError> {
    let mut cfg = config(1.0);
    cfg.gravity = [Fix128::ZERO, fx(-9.81)];
    cfg.iterations = 4;
    let points: Vec<P2> = (0..=20)
        .map(|i| [Fix128::from_ratio(i as i64, 100), fx(0.04)])
        .collect();
    let mut chain = ThickChain::new(cfg, &points, fx(0.03), fx(DENSITY))
        .expect("a chain")
        .bending(law)?;
    chain.pin(0);
    let walls = floor();
    for _ in 0..180 {
        chain.step_in(frame(), &walls, &[])?;
    }
    Ok(places(&chain))
}

fn turned_back(x: &[[f64; 2]]) -> usize {
    (0..x.len() - 1).filter(|&k| x[k + 1][0] < x[k][0]).count()
}

#[test]
fn let_down_onto_a_floor_from_a_fixed_tail_no_link_turns_back() {
    let with = let_down(ThickChainBend::Section).expect("the run");
    assert_eq!(turned_back(&with), 0);
    // the scene is one that folds a chain without the law
    let without = let_down(ThickChainBend::None).expect("the run");
    assert!(turned_back(&without) > 0, "{without:?}");
}

#[test]
fn pushed_with_twenty_times_the_buckling_load_no_joint_reaches_the_radius_of_its_half_width() {
    const LINKS: usize = 20;
    let reach = LINKS as f64 * LINK;
    let euler = std::f64::consts::PI.powi(2) * BEND / (4.0 * reach * reach);
    let mut chain = clamped(config(2.0), LINKS + 1);
    let walls = open();
    let mut accel = vec![ZERO2; LINKS + 2];
    accel[LINKS + 1] = [fx(-20.0 * euler / (MASS / 2.0)), fx(1.0e-3)];
    let (mut most, mut refused) = (0.0f64, 0usize);
    for _ in 0..180 {
        match chain.step_in(frame(), &walls, &accel) {
            Ok(()) => {}
            // a step the halvings cannot take is refused and the chain stays where it was
            Err(ThickChainError::Folded { .. } | ThickChainError::TooStiff { .. }) => refused += 1,
            Err(other) => panic!("{other:?}"),
        }
        most = most.max(most_bent(&places(&chain)));
    }
    assert!(most < 1.0, "bent to a = {most}");
    assert!(most > 0.3, "the push bent it: a = {most}");
    // measured, not required to be none: links far apart along the chain that overlap
    let x = places(&chain);
    let apart = (2.0 * RADIUS / LINK).ceil() as usize + 2;
    let mut overlaps = 0;
    for i in 0..x.len() {
        for j in i + apart..x.len() {
            if (x[i][0] - x[j][0]).hypot(x[i][1] - x[j][1]) < 2.0 * RADIUS {
                overlaps += 1;
            }
        }
    }
    println!(
        "[thick chain bend] 20 x the buckling load: most a {most:.3}, steps refused {refused} of 180, particle pairs more than {apart} links apart closer than the width {overlaps}"
    );
}

// ---- 8. the wall is each link's own ----

#[test]
fn a_joint_whose_thicker_link_alone_is_past_the_radius_of_its_half_width_is_refused() {
    // link 0 at rest (20 mm in half-width), link 1 twice its rest length (10 mm): at the
    // joint's curvature `κ = 2 tan(θ/2) / l̄`, `l̄` 15 mm, link 0 is at `a = 1` when
    // `tan(θ/2) = l̄ / (2 r0)` and link 1 only at half of that
    let limit = 2.0 * (0.015f64 / (2.0 * RADIUS)).atan();
    let chain_at = |turn: f64| {
        let rest = vec![
            [Fix128::ZERO, Fix128::ZERO],
            [fx(LINK), Fix128::ZERO],
            [fx(2.0 * LINK), Fix128::ZERO],
        ];
        let mut chain =
            ThickChain::new(config(0.0), &rest, fx(RADIUS), fx(DENSITY)).expect("a chain");
        chain.set_position(
            2,
            [
                fx(LINK + 2.0 * LINK * turn.cos()),
                fx(2.0 * LINK * turn.sin()),
            ],
        );
        chain.bending(ThickChainBend::Section)
    };
    assert_eq!(
        chain_at(1.02 * limit).err(),
        Some(ThickChainError::Folded { joint: 1 })
    );
    assert!(chain_at(0.97 * limit).is_ok());
    // the thin link alone would be inside at either turn
    let x = [
        [0.0, 0.0],
        [LINK, 0.0],
        [
            LINK + 2.0 * LINK * (1.02 * limit).cos(),
            2.0 * LINK * (1.02 * limit).sin(),
        ],
    ];
    assert!(curvature(&x, 1) * RADIUS / 2.0 < 0.6);
}
