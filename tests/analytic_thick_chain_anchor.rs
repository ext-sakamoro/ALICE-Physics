//! Closed-form oracles of the thick chain held by one end
//! ([`alice_physics_core::thick_chain::ThickChain`]): split substeps, the material's inner
//! friction, a pull on the head against a fixed tail, and the way back after the pull ends.
//!
//! 1. Split substeps. A free link at 20 m/s travels 20 t exactly (no force acts): refused
//!    when a substep may not be halved, done when it may; a run that needs no halving is the
//!    same bits either way; a link thrown at a floor stops on it; a link too stiff for its
//!    substep (`(ω h)² > 1/2`) is given the halvings that resolve it, the same numbers as
//!    those substeps taken outright.
//! 2. Inner friction (the Kelvin-Voigt solid, retardation time `τ`). One free link let go
//!    stretched under the drag `k`: its extension goes by
//!    `e₊ (1 + ω² h² + τ ω² h) = e (1 + τ ω² h) + (1 − k h)(e − e₋)` a substep,
//!    `ω² = (w0 + w1) / α` (the implicit step of `m ë + (k m + τ K) ė + K e = 0`), below and
//!    above the critical damping.
//! 3. The fixed tail does not move: the pinned particles keep their bits through a pull
//!    between walls.
//! 4. Pulled by `F` on the head with the tail fixed and no walls, every free link carries
//!    `F`: its stretch is where `μ w0 (λ − λ⁻³) = F` (neo-Hookean, plane strain).
//! 5. In a straight frictionless channel of half-width `c < r0` the walls already hold
//!    `λc = r0 / c`: the head is `max(λc, λ(F))` times the free rest length from the fixed
//!    part, and above `T(λc)` the chain is clear of the walls.
//! 6. Let go, the free part swings back. In its first mode (fixed at one end, the head's
//!    particle of half a link's mass: `u_i ∝ sin(i π / (2 N))`,
//!    `ω² = 4 (K / m) sin²(π / (4 N))`, `K = 4 μ w0 / l0`) it follows the recurrence of 2,
//!    and it is back within a hundredth of its first amplitude when the recurrence is.
//! 7. After the pull ends the energy (the particles' kinetic energy and the links'
//!    `(μ / 2)(λ − 1/λ)² A`) never grows.
//!
//! The solves of 4 to 7 run 64 passes with `α / h²` 7 times `(w_i + w_j) |∇C|²` or more at
//! rest, so a pass leaves a small part of the error before it (Gauss-Seidel on a diagonally
//! dominant chain): what is left after 64 is far below the tolerances stated.
//!
//! Integration test only (not brought into the lib tests with `#[path]`): its chains settle
//! over many frames, about 70 s at opt-level 2 and six minutes without optimisation, which
//! the lib tests of the minimum-toolchain job run twice.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: closed forms are evaluated in f64, never by the code under test"
)]

use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_collider::DistanceSdfQuery;
use alice_physics_core::thick_chain::{
    ThickChain, ThickChainConfig, ThickChainError, ThickChainStretch, P2,
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
const SUBSTEPS: u32 = 16;
/// the fixed links and the free ones
const FIXED: usize = 3;
const FREE: usize = 7;

fn frame() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn step_time() -> f64 {
    1.0 / 60.0 / f64::from(SUBSTEPS)
}

fn config(stretch: ThickChainStretch) -> ThickChainConfig {
    ThickChainConfig {
        substeps: SUBSTEPS,
        iterations: 64,
        gravity: ZERO2,
        stretch,
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(0.002),
        splits: 0,
        relaxation: Fix128::ZERO,
    }
}

fn neo() -> ThickChainStretch {
    ThickChainStretch::NeoHookean { shear: fx(SHEAR) }
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

/// the stretch at which `T / (μ w0)` is `load` (the pull grows with the stretch)
fn stretch_under(load: f64) -> f64 {
    let (mut lo, mut hi) = (0.05f64, 100.0f64);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if pull(mid) < load {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

#[test]
fn a_fast_link_is_refused_without_split_substeps_and_flies_straight_with_them() {
    let v = 20.0;
    let fly = |splits: u32| {
        let mut cfg = config(neo());
        cfg.min_radius = fx(0.005);
        cfg.splits = splits;
        let mut chain =
            ThickChain::new(cfg, &row(0.0, LINK, 1), fx(RADIUS), fx(DENSITY)).expect("a chain");
        for i in 0..2 {
            chain.set_velocity(i, [fx(v), Fix128::ZERO]);
        }
        let before = chain.snapshot();
        let result = chain.step(frame());
        (result, chain, before)
    };
    // a substep is 1/960 s: 20.8 mm of travel against the 2.5 mm allowed; a sixteenth of it
    // is 1.3 mm
    let (result, chain, before) = fly(0);
    assert_eq!(result, Err(ThickChainError::TooFast { particle: 0 }));
    assert_eq!(chain.snapshot(), before);
    // three halvings leave 2.6 mm: still too far, and nothing changes
    let (result, chain, before) = fly(3);
    assert_eq!(result, Err(ThickChainError::TooFast { particle: 0 }));
    assert_eq!(chain.snapshot(), before);
    let (result, chain, _) = fly(8);
    assert_eq!(result, Ok(()));
    for (i, p) in chain.positions().iter().enumerate() {
        let expect = LINK * i as f64 + v / 60.0;
        // sums of 512 products in Q64.64
        assert!(
            (f(p[0]) - expect).abs() <= 1.0e-12,
            "particle {i}: {:e}, expected {expect:e}",
            f(p[0])
        );
    }
    let mut cfg = config(neo());
    cfg.splits = 17;
    assert_eq!(
        ThickChain::new(cfg, &row(0.0, LINK, 1), fx(RADIUS), fx(DENSITY)).err(),
        Some(ThickChainError::Invalid("splits"))
    );
}

#[test]
fn a_run_that_needs_no_split_is_the_same_bits_with_and_without() {
    let run = |splits: u32| {
        let mut cfg = config(neo());
        cfg.gravity = [Fix128::ZERO, fx(-9.81)];
        cfg.drag = fx(5.0);
        cfg.splits = splits;
        let points = [[Fix128::ZERO, fx(LINK)], ZERO2];
        let mut chain = ThickChain::new(cfg, &points, fx(RADIUS), fx(DENSITY)).expect("a chain");
        chain.pin(0);
        for _ in 0..30 {
            chain.step(frame()).expect("a step");
        }
        (chain.positions().to_vec(), chain.velocities().to_vec())
    };
    assert_eq!(run(0), run(8));
    // the two settings are part of the state all the same: a snapshot tells them apart
    let made = |splits: u32, relaxation: f64| {
        let mut cfg = config(neo());
        cfg.splits = splits;
        cfg.relaxation = fx(relaxation);
        ThickChain::new(cfg, &row(0.0, LINK, 2), fx(RADIUS), fx(DENSITY))
            .expect("a chain")
            .snapshot()
    };
    assert_ne!(made(0, 0.0), made(8, 0.0));
    assert_ne!(made(0, 0.0), made(0, 0.004));
    // and a retardation time below 0 is refused by name
    let mut cfg = config(neo());
    cfg.relaxation = fx(-0.001);
    assert_eq!(
        ThickChain::new(cfg, &row(0.0, LINK, 2), fx(RADIUS), fx(DENSITY)).err(),
        Some(ThickChainError::Invalid("relaxation"))
    );
}

#[test]
fn a_link_thrown_at_a_floor_stops_on_it() {
    let mut cfg = config(neo());
    cfg.min_radius = fx(0.005);
    cfg.splits = 8;
    let points = [[Fix128::ZERO, fx(0.05)], [fx(LINK), fx(0.05)]];
    let mut chain = ThickChain::new(cfg, &points, fx(RADIUS), fx(DENSITY)).expect("a chain");
    for i in 0..2 {
        chain.set_velocity(i, [Fix128::ZERO, fx(-20.0)]);
    }
    let floor = DistanceSdfQuery::new(|_x: f32, y: f32, _z: f32| y);
    // a frame at 20 m/s is 0.33 m: six times the way to the floor
    for _ in 0..3 {
        chain.step_in(frame(), &floor, &[]).expect("a step");
        // the link may be flattened by the blow (thinner and longer); its outline is never
        // further into the floor than a part's travel, half the least radius
        let radius = f(chain.radius(0));
        for (i, p) in chain.positions().iter().enumerate() {
            assert!(
                f(p[1]) - radius >= -0.0025,
                "particle {i} is at {:e} with the radius {radius:e}",
                f(p[1])
            );
        }
    }
    // and it is above the floor, not through it (flattened by the blow, it springs back up)
    for p in chain.positions() {
        assert!(f(p[1]) > 0.0, "at {:e}", f(p[1]));
    }
}

#[test]
fn a_link_too_stiff_for_its_substep_is_given_the_halvings_that_resolve_it() {
    // 4 links of shear modulus 2 kPa at 4 substeps to a frame of 1/64 s (h = 1/256 s):
    // (ω h)² = (w_i + w_j) |∇C|² h² μ A is 7.5 · 200² · 0.8 / 256² = 3.7 at the end links and
    // 2.4 between, so two halvings bring every link under 1/2 and one does not. The frame is
    // a power of two, so the parts are the same numbers as 16 substeps taken outright.
    let run = |substeps: u32, splits: u32| {
        let mut cfg = config(ThickChainStretch::NeoHookean { shear: fx(2000.0) });
        cfg.iterations = 4;
        cfg.substeps = substeps;
        cfg.splits = splits;
        let mut chain =
            ThickChain::new(cfg, &row(0.0, LINK, 4), fx(RADIUS), fx(DENSITY)).expect("a chain");
        // let go with one particle 0.2 mm out
        chain.set_position(2, [fx(2.0 * LINK + 0.0002), Fix128::ZERO]);
        let before = chain.snapshot();
        let mut result = Ok(());
        for _ in 0..20 {
            result = chain.step(Fix128::from_ratio(1, 64));
            if result.is_err() {
                break;
            }
        }
        (result, chain, before)
    };
    let (result, split, _) = run(4, 8);
    assert_eq!(result, Ok(()));
    let (result, outright, _) = run(16, 0);
    assert_eq!(result, Ok(()));
    assert_eq!(split.positions(), outright.positions());
    assert_eq!(split.velocities(), outright.velocities());
    // and not the same as 4 substeps left unresolved
    let (result, coarse, _) = run(4, 0);
    assert_eq!(result, Ok(()));
    assert_ne!(coarse.positions(), outright.positions());
    // with one halving allowed the step is refused, by the link, and nothing changes
    let (result, refused, before) = run(4, 1);
    assert_eq!(result, Err(ThickChainError::TooStiff { link: 0 }));
    assert_eq!(refused.snapshot(), before);
}

/// `count` terms of the recurrence of a mode of frequency `omega` let go at `first` under the
/// drag `drag` and the retardation time `tau`, a substep `h` apart
fn recurrence(first: f64, omega: f64, drag: f64, tau: f64, h: f64, count: usize) -> Vec<f64> {
    let (stiff, slow) = (omega * omega * h * h, tau * omega * omega * h);
    let mut out = vec![first];
    let mut before = first;
    for n in 0..count {
        let now = out[n];
        out.push((now * (1.0 + slow) + (1.0 - drag * h) * (now - before)) / (1.0 + stiff + slow));
        before = now;
    }
    out
}

#[test]
fn a_free_link_with_inner_friction_swings_on_the_recurrence_of_its_implicit_step() {
    let (alpha, first) = (1.0e-4, 0.002);
    let h = step_time();
    // each particle carries half the link: rho * (2 r0 l0) / 2
    let w = 1.0 / (DENSITY * RADIUS * LINK);
    let omega = ((w + w) / alpha).sqrt();
    // the critical retardation time is 2 / ω: a tenth of it swings, four times it creeps
    for (tau, drag) in [(0.2 / omega, 0.0), (0.2 / omega, 3.0), (8.0 / omega, 0.0)] {
        let mut cfg = config(ThickChainStretch::Linear {
            compliance: fx(alpha),
        });
        cfg.iterations = 4;
        cfg.drag = fx(drag);
        cfg.relaxation = fx(tau);
        let mut chain =
            ThickChain::new(cfg, &row(0.0, LINK, 1), fx(RADIUS), fx(DENSITY)).expect("a chain");
        chain.set_position(1, [fx(LINK + first), Fix128::ZERO]);
        let frames = 12;
        let expect = recurrence(first, omega, drag, tau, h, frames * SUBSTEPS as usize);
        let mut crossed = false;
        for n in 1..=frames {
            chain.step(frame()).expect("a step");
            let got = f(chain.length(0)) - LINK;
            let want = expect[n * SUBSTEPS as usize];
            crossed |= want < 0.0;
            // the settings are f64 made Q64.64 (2^-64 each) and a term of the recurrence is
            // well conditioned: a millionth of the first extension covers 192 substeps
            assert!(
                (got - want).abs() <= first * 1.0e-6,
                "τ {tau:e} k {drag}: frame {n}: {got:e}, expected {want:e}"
            );
        }
        // the references are what they are called: the first two swing through, the third
        // never does
        assert_eq!(crossed, tau * omega < 2.0, "τ ω = {}", tau * omega);
    }
}

/// a chain of `FIXED + FREE` links from `x = 0`, its first `FIXED` links fixed
fn anchored(cfg: ThickChainConfig) -> ThickChain {
    let mut chain = ThickChain::new(cfg, &row(0.0, LINK, FIXED + FREE), fx(RADIUS), fx(DENSITY))
        .expect("a chain");
    for i in 0..=FIXED {
        chain.pin(i);
    }
    chain
}

/// the acceleration that is the force `force` (N per metre of depth) on the head's particle
fn on_the_head(force: f64) -> Vec<P2> {
    let mut accel = vec![ZERO2; FIXED + FREE + 1];
    // the head's particle carries half a link
    accel[FIXED + FREE] = [fx(force / (DENSITY * RADIUS * LINK)), Fix128::ZERO];
    accel
}

/// the free space between two walls at `y = ±c`
fn channel(c: f32) -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(move |_x: f32, y: f32, _z: f32| c - y.abs())
}

#[test]
fn the_fixed_tail_keeps_its_bits_through_a_pull_between_walls() {
    let mut cfg = config(neo());
    cfg.drag = fx(10.0);
    cfg.friction = fx(0.3);
    cfg.gravity = [Fix128::ZERO, fx(-9.81)];
    cfg.splits = 8;
    let mut chain = anchored(cfg);
    let before = chain.positions().to_vec();
    let walls = channel(0.03);
    let accel = on_the_head(2.0 * SHEAR * 2.0 * RADIUS);
    for _ in 0..60 {
        chain.step_in(frame(), &walls, &accel).expect("a step");
    }
    assert_eq!(chain.positions()[..=FIXED], before[..=FIXED]);
    for v in &chain.velocities()[..=FIXED] {
        assert_eq!(*v, ZERO2);
    }
    // and the pull did pull: the head went on by more than the free part's rest length
    let head = f(chain.positions()[FIXED + FREE][0]) - f(before[FIXED + FREE][0]);
    assert!(head > LINK * FREE as f64, "the head went {head}");
    // without the fixed tail the same pull takes the whole chain along
    let mut loose = ThickChain::new(cfg, &before, fx(RADIUS), fx(DENSITY)).expect("a chain");
    for _ in 0..60 {
        loose.step_in(frame(), &walls, &accel).expect("a step");
    }
    assert!(f(loose.positions()[0][0]) > LINK, "the loose tail stayed");
}

/// the tolerance on a link's stretch at rest under the pull `force` on the head, after
/// `seconds` under the drag `drag`: a substep at rest moves the head by `δ = h² F / m` and
/// the solve takes it back; over `δ` the constraint's slope changes by the fraction
/// `2 λ⁻³ / (1 + λ⁻²) · δ / l0`, so the tension is the pull to twice that fraction, and the
/// swing left is below the first extension times `e^(−k t / 2)`.
fn rest_tolerance(lambda: f64, force: f64, drag: f64, seconds: f64) -> f64 {
    let h = step_time();
    let moved = h * h * force / (DENSITY * RADIUS * LINK);
    let turn = 2.0 * lambda.powi(-3) / (1.0 + lambda.powi(-2)) * moved / LINK;
    2.0 * turn * pull(lambda).abs() / (1.0 + 3.0 * lambda.powi(-4))
        + (lambda - 1.0).abs() * (-drag * seconds / 2.0).exp()
        + 1.0e-9
}

#[test]
fn every_free_link_of_a_pulled_chain_is_where_its_tension_is_the_pull() {
    let (drag, seconds) = (30.0, 3.0);
    // walls far away: the drive goes in with them
    let open = channel(10.0);
    for lambda in [1.2, 2.0, 3.5] {
        let force = SHEAR * 2.0 * RADIUS * pull(lambda);
        let mut cfg = config(neo());
        cfg.drag = fx(drag);
        cfg.splits = 8;
        let accel = on_the_head(force);
        let mut pulled = anchored(cfg);
        for _ in 0..(seconds * 60.0) as usize {
            pulled.step_in(frame(), &open, &accel).expect("a step");
        }
        let tol = rest_tolerance(lambda, force, drag, seconds);
        for k in 0..FIXED {
            assert_eq!(pulled.length(k), pulled.rest_length(k), "fixed link {k}");
        }
        for k in FIXED..FIXED + FREE {
            let got = f(pulled.length(k)) / LINK;
            assert!(
                (got - lambda).abs() <= tol,
                "λ {lambda}: link {k}: {got:e} within {tol:e}"
            );
        }
        // pulled on the tail instead, nothing moves: the tail is fixed
        let mut wrong = vec![ZERO2; FIXED + FREE + 1];
        wrong[0] = accel[FIXED + FREE];
        let mut held = anchored(cfg);
        let before = held.snapshot();
        for _ in 0..30 {
            held.step_in(frame(), &open, &wrong).expect("a step");
        }
        assert_eq!(held.snapshot(), before);
    }
}

#[test]
fn the_reach_in_a_channel_is_the_longer_of_the_walls_hold_and_the_pulls_stretch() {
    let c = 0.012f32;
    let at = RADIUS / f64::from(c);
    let held = at * LINK;
    let (drag, seconds) = (30.0, 3.0);
    let limit = SHEAR * 2.0 * RADIUS * pull(at);
    let walls = channel(c);
    for share in [0.5, 2.0] {
        let force = share * limit;
        let mut cfg = config(neo());
        cfg.drag = fx(drag);
        cfg.splits = 8;
        // laid out as the channel holds it (every link `held` long), the first links fixed so
        let mut chain =
            ThickChain::new(cfg, &row(0.0, LINK, FIXED + FREE), fx(RADIUS), fx(DENSITY))
                .expect("a chain");
        for i in 0..=FIXED + FREE {
            chain.set_position(i, [fx(held * i as f64), Fix128::ZERO]);
        }
        for i in 0..=FIXED {
            chain.pin(i);
        }
        let accel = on_the_head(force);
        for _ in 0..(seconds * 60.0) as usize {
            chain.step_in(frame(), &walls, &accel).expect("a step");
        }
        let lambda = if share < 1.0 {
            at
        } else {
            stretch_under(share * pull(at))
        };
        let reach = f(chain.positions()[FIXED + FREE][0]) - f(chain.positions()[FIXED][0]);
        let expect = lambda * LINK * FREE as f64;
        // held by the walls: a link at rest is within (ω h)² of its extension of the walls'
        // length, ω² = (w + w) dT/dl (the stretch solve shortens it by that, the walls give
        // it back), and the walls are f32; pulled clear: the tolerance of a pulled link
        let h = step_time();
        let w = 1.0 / (DENSITY * RADIUS * LINK);
        let wh2 = (w + w) * SHEAR * 2.0 * RADIUS * (1.0 + 3.0 * at.powi(-4)) / LINK * h * h;
        let tol = if share < 1.0 {
            FREE as f64 * (wh2 * (held - LINK) + held * 2.0f64.powi(-23))
        } else {
            FREE as f64 * LINK * rest_tolerance(lambda, force, drag, seconds)
        };
        assert!(
            (reach - expect).abs() <= tol,
            "pull {share} of the limit: reach {reach:e}, expected {expect:e} within {tol:e}"
        );
        if share > 1.0 {
            for k in FIXED..FIXED + FREE {
                let clear = f64::from(c) - f(chain.radius(k));
                assert!(clear > 0.0, "link {k} is {clear:e} from clearing the walls");
            }
        }
    }
}

/// the free part's first mode: the frequency, and the shape at free particle `i` of `FREE`
fn first_mode() -> (f64, impl Fn(usize) -> f64) {
    let stiffness = 4.0 * SHEAR * 2.0 * RADIUS / LINK;
    let mass = DENSITY * 2.0 * RADIUS * LINK;
    let angle = core::f64::consts::PI / (2.0 * FREE as f64);
    (
        2.0 * (stiffness / mass).sqrt() * (angle / 2.0).sin(),
        move |i: usize| (angle * i as f64).sin(),
    )
}

#[test]
fn let_go_the_free_part_swings_back_on_its_first_modes_recurrence() {
    let (omega, shape) = first_mode();
    // a head 0.14 mm out: the largest strain is 0.3 %
    let first = 0.002 * LINK * FREE as f64;
    let strain = first * core::f64::consts::PI / (2.0 * FREE as f64) / LINK;
    let h = step_time();
    for (tau, drag) in [(0.0, 1.0), (0.2 / omega, 1.0)] {
        let mut cfg = config(neo());
        cfg.drag = fx(drag);
        cfg.relaxation = fx(tau);
        let mut chain = anchored(cfg);
        for i in 1..=FREE {
            let x = LINK * (FIXED + i) as f64 + first * shape(i);
            chain.set_position(FIXED + i, [fx(x), Fix128::ZERO]);
        }
        let frames = 18;
        let expect = recurrence(first, omega, drag, tau, h, 240 * SUBSTEPS as usize);
        for n in 1..=frames {
            chain.step(frame()).expect("a step");
            let want = expect[n * SUBSTEPS as usize];
            // the law is stiffer or softer than its tangent by 1.5 times the strain, so the
            // frequency by 0.75 of it: the phase is off by that times ω t, of the amplitude
            let time = n as f64 / 60.0;
            let tol = first * (0.75 * strain * omega * time + 1.0e-6);
            for i in 1..=FREE {
                let got = f(chain.positions()[FIXED + i][0]) - LINK * (FIXED + i) as f64;
                assert!(
                    (got - want * shape(i)).abs() <= tol,
                    "τ {tau:e}: frame {n}, particle {i}: {got:e}, expected {:e} within {tol:e}",
                    want * shape(i)
                );
            }
        }
        // back: from the substep after which the recurrence stays within a hundredth of its
        // first amplitude, so does every particle (with the phase's share at that time)
        let back = (0..expect.len())
            .rev()
            .find(|&n| expect[n].abs() > 0.01 * first)
            .map_or(0, |n| n + 1);
        let frame_back = back.div_ceil(SUBSTEPS as usize);
        assert!(frame_back < 240, "the recurrence is not back in 4 s");
        for _ in frames..frame_back {
            chain.step(frame()).expect("a step");
        }
        let time = frame_back as f64 / 60.0;
        let tol = first * (0.01 + 0.01 * 0.75 * strain * omega * time + 1.0e-6);
        for i in 1..=FREE {
            let got = f(chain.positions()[FIXED + i][0]) - LINK * (FIXED + i) as f64;
            assert!(
                got.abs() <= tol,
                "τ {tau:e}: back at frame {frame_back}: particle {i} is {got:e} out"
            );
        }
    }
}

fn energy(chain: &ThickChain) -> f64 {
    let mut total = 0.0;
    for (i, v) in chain.velocities().iter().enumerate() {
        // an end particle carries half a link, the others two halves
        let mass = DENSITY
            * 2.0
            * RADIUS
            * LINK
            * if i == 0 || i == FIXED + FREE {
                0.5
            } else {
                1.0
            };
        total += 0.5 * mass * (f(v[0]).powi(2) + f(v[1]).powi(2));
    }
    for k in 0..chain.link_count() {
        let lambda = f(chain.length(k)) / f(chain.rest_length(k));
        total += SHEAR / 2.0 * (lambda - 1.0 / lambda).powi(2) * f(chain.area(k));
    }
    total
}

#[test]
fn after_the_pull_ends_the_energy_never_grows() {
    for tau in [0.0, 0.004] {
        let mut cfg = config(neo());
        cfg.drag = fx(1.0);
        cfg.relaxation = fx(tau);
        cfg.splits = 8;
        let mut chain = anchored(cfg);
        // let go from three times the free part's rest length
        for i in 1..=FREE {
            let x = LINK * FIXED as f64 + 3.0 * LINK * i as f64;
            chain.set_position(FIXED + i, [fx(x), Fix128::ZERO]);
        }
        let first = energy(&chain);
        let mut last = first;
        for n in 0..180 {
            chain.step(frame()).expect("a step");
            let now = energy(&chain);
            // f64 sums of 21 terms
            assert!(
                now <= last + first * 1.0e-12,
                "τ {tau}: frame {n}: the energy went from {last:e} to {now:e}"
            );
            last = now;
        }
        // and it went: under a hundredth of the first after 3 s
        assert!(
            last < 0.01 * first,
            "τ {tau}: {last:e} of {first:e} is left"
        );
    }
}
