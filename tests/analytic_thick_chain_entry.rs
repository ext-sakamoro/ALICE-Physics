//! Closed-form oracles of the thick chain entering a narrower channel
//! ([`alice_physics_core::thick_chain::ThickChain`], neo-Hookean links, the disc rule).
//! Integration tests only: they run for tens of seconds.
//!
//! The closed form. Pushed on its tail by a force `F`, a chain is fed into a frictionless
//! channel of half-width `c < r0` through a taper. In steady feeding the part outside is
//! shortened to `λ_F` by the push (`T(λ_F) = −F`, `T(λ) = μ w0 (λ − λ⁻³)`), a unit of rest
//! length fed moves the tail by `λ_F` and takes the energy `(W(λc) − W(λ_F)) w0`,
//! `W = (μ / 2)(λ − 1/λ)²`, `λc = r0 / c`. So over the feed the links take `F* λ_F` per unit
//! of rest length with `F* λ_F = (W(λc) − W(λ_F)) w0`: with `q = λ_F²` and
//! `S = λc² + λc⁻²`, `q² + S q − 3 = 0` and `F* = μ w0 (λ_F⁻³ − λ_F)`.
//!
//! What the chain adds. A link has one radius, that of its narrowest sample, so it pays the
//! squeeze in lumps, a link at a time; when a link clears its lump the chain jumps on and the
//! drag takes the speed. So:
//!
//! 1. Pushed slowly (a drag that leaves it no speed to carry it over a lump), no chain is fed
//!    below `F*`; the least push of a ladder that feeds it does not grow when its links are
//!    halved. A chain too stiff for its substep is given the substeps (the split rule), so
//!    it is not fed below `F*` either.
//! 2. Moved slowly by its tail, the push's work is never less than the energy the links
//!    take (smooth walls at rest do no work; a jump loses energy to the drag), the loss is
//!    smaller with shorter links, and the energy taken over the way is `F*` times the way to
//!    within two links' share (a lump may be pending at either end of the way).
//! 3. Fed, every link is as long as the channel makes it: `A / (2 c)`.
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

/// the mouth: a channel of half-width `WIDE` for `x < 0` that narrows evenly over `TAPER` to
/// the half-width `NARROW`
const WIDE: f32 = 0.03;
const NARROW: f32 = 0.012;
const TAPER: f32 = 0.1;
/// the chain: 0.2 m of rest radius 0.02 m, its head 5 mm before the mouth
const BODY: f64 = 0.2;
const RADIUS: f64 = 0.02;
const DENSITY: f64 = 1000.0;
/// the soft chain of 1 and 2, and the drag that leaves it slow
const SHEAR: f64 = 200.0;
const DRAG: f64 = 2.0;
/// the frames a push is given: 15 s
const FRAMES: usize = 900;

/// the distance to the mouth's walls (exact: the nearest point of the upper wall's three
/// runs, the lower wall by symmetry), positive between them
fn mouth(x: f32, y: f32) -> f32 {
    let y = y.abs();
    let runs = [
        ([-10.0f32, WIDE], [0.0f32, WIDE]),
        ([0.0, WIDE], [TAPER, NARROW]),
        ([TAPER, NARROW], [10.0, NARROW]),
    ];
    let mut near = f32::INFINITY;
    for (p, q) in runs {
        let d = [q[0] - p[0], q[1] - p[1]];
        let t =
            (((x - p[0]) * d[0] + (y - p[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1])).clamp(0.0, 1.0);
        let (cx, cy) = (p[0] + t * d[0], p[1] + t * d[1]);
        near = near.min(((cx - x) * (cx - x) + (cy - y) * (cy - y)).sqrt());
    }
    let half = if x <= 0.0 {
        WIDE
    } else if x >= TAPER {
        NARROW
    } else {
        WIDE + (NARROW - WIDE) * x / TAPER
    };
    if y < half {
        near
    } else {
        -near
    }
}

/// `T / (μ w0)` at the stretch `λ`
fn pull(lambda: f64) -> f64 {
    lambda - lambda.powi(-3)
}

/// `F* / (μ w0)` and the stretch of the part outside under it
fn threshold() -> (f64, f64) {
    let at = RADIUS / f64::from(NARROW);
    let s = at * at + 1.0 / (at * at);
    let q = (-s + (s * s + 12.0).sqrt()) / 2.0;
    let outside = q.sqrt();
    (-pull(outside), outside)
}

fn chain(shear: f64, drag: f64, links: usize) -> ThickChain {
    chain_of(shear, drag, links, ThickChainFit::Disc)
}

fn chain_of(shear: f64, drag: f64, links: usize, fit: ThickChainFit) -> ThickChain {
    let cfg = ThickChainConfig {
        substeps: 16,
        iterations: 4,
        gravity: ZERO2,
        stretch: ThickChainStretch::NeoHookean { shear: fx(shear) },
        drag: fx(drag),
        friction: Fix128::ZERO,
        min_radius: fx(0.005),
        splits: 8,
        relaxation: Fix128::ZERO,
        fit,
        reach: fx(RADIUS / 0.005),
    };
    let link = BODY / links as f64;
    let points: Vec<P2> = (0..=links)
        .map(|i| [fx(-0.005 - BODY + link * i as f64), Fix128::ZERO])
        .collect();
    ThickChain::new(cfg, &points, fx(RADIUS), fx(DENSITY)).expect("a valid chain")
}

fn walls() -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(|x: f32, y: f32, _z: f32| mouth(x, y))
}

/// pushes the chain's tail with `push` (N per metre of depth) for 15 s, or until the tail is
/// `until` along (m)
fn pushed(mut chain: ThickChain, push: f64, until: f64) -> ThickChain {
    let links = chain.link_count();
    // the tail particle carries half a link
    let mut accel = vec![ZERO2; links + 1];
    accel[0] = [
        fx(push / (DENSITY * RADIUS * BODY / links as f64)),
        Fix128::ZERO,
    ];
    let walls = walls();
    for _ in 0..FRAMES {
        chain
            .step_in(Fix128::from_ratio(1, 60), &walls, &accel)
            .expect("a step");
        if f(chain.positions()[0][0]) > until {
            break;
        }
    }
    chain
}

/// whether the push brings the chain's tail into the mouth
fn fed(shear: f64, drag: f64, links: usize, share: f64) -> bool {
    let push = share * threshold().0 * shear * 2.0 * RADIUS;
    f(pushed(chain(shear, drag, links), push, 0.0).positions()[0][0]) > 0.0
}

#[test]
fn a_chain_of_short_links_is_not_fed_below_the_push_of_its_squeeze_energy_and_is_fed_above_it() {
    // the closed form's push at one resolution (40 links of 5 mm): not fed at 0.95 of it, fed
    // by 1.6 of it at the most (the links' lumps: see the module's Approximations). The other
    // resolutions and the order between them are in the test below.
    assert!(
        !fed(SHEAR, DRAG, 40, 0.95),
        "40 links are fed below the limit"
    );
    assert!(
        [1.1, 1.3, 1.6]
            .iter()
            .any(|&share| fed(SHEAR, DRAG, 40, share)),
        "no push of the ladder feeds 40 links"
    );
}

#[test]
#[ignore = "runtime: about 45 s at opt-level 2, measured 2026-11-19 (nine pushes of 15 s each, up to 80 links); run by the weekly job"]
fn pushed_slowly_no_chain_is_fed_below_the_push_of_its_squeeze_energy() {
    let (limit, outside) = threshold();
    // the closed form's own numbers: λc = 5/3, the part outside 12 % shorter, 0.609 μ w0
    assert!((outside - 0.876).abs() < 1.0e-3, "outside {outside}");
    assert!((limit - 0.609).abs() < 1.0e-3, "limit {limit}");
    let ladder = [1.1, 1.3, 1.6];
    let mut last = f64::INFINITY;
    for links in [20, 40] {
        assert!(
            !fed(SHEAR, DRAG, links, 0.95),
            "{links} links are fed below the limit"
        );
        let least = ladder
            .iter()
            .copied()
            .find(|&share| fed(SHEAR, DRAG, links, share))
            .unwrap_or(f64::INFINITY);
        assert!(
            least <= last,
            "{links} links need {least} of the limit, links twice as long {last}"
        );
        last = least;
    }
    assert!(last.is_finite(), "no push of the ladder feeds 40 links");
    // ten times as stiff with a quarter of the links' length: (ω h)² is 0.96 at 16 substeps,
    // and the chain is given the halving that resolves it
    assert!(
        !fed(10.0 * SHEAR, 10.0 * DRAG, 80, 0.95),
        "80 stiff links are fed below the limit"
    );
}

/// the links' elastic energy, J per metre of depth
fn stored(chain: &ThickChain) -> f64 {
    (0..chain.link_count())
        .map(|k| {
            let lambda = f(chain.length(k)) / f(chain.rest_length(k));
            SHEAR / 2.0 * (lambda - 1.0 / lambda).powi(2) * f(chain.area(k))
        })
        .sum()
}

/// moves a stalled chain of `links` links on by its tail, slowly; returns the push's work
/// over the energy the links took, and that energy over `F*` times the way
fn slow_feed(links: usize) -> (f64, f64) {
    let link = BODY / links as f64;
    let limit = threshold().0 * SHEAR * 2.0 * RADIUS;
    let stalled = pushed(chain(SHEAR, DRAG, links), 1.05 * limit, 0.0);
    assert!(f(stalled.positions()[0][0]) < 0.0, "not stalled");
    assert!(
        f(stalled.positions()[links][0]) > 0.0,
        "the head is not in the mouth"
    );
    // the same chain where it stalled, with a drag near the critical one of its slowest
    // mode (about 12 rad/s), held by its tail
    let mut chain = chain(SHEAR, 24.0, links);
    for (i, p) in stalled.positions().iter().enumerate() {
        chain.set_position(i, *p);
    }
    chain.pin(0);
    let walls = walls();
    let rest = |chain: &mut ThickChain| {
        for _ in 0..900 {
            chain
                .step_in(Fix128::from_ratio(1, 60), &walls, &[])
                .expect("a step");
            let still = chain
                .velocities()
                .iter()
                .all(|v| f(v[0]).abs() < 1.0e-5 && f(v[1]).abs() < 1.0e-5);
            if still {
                return;
            }
        }
        panic!("the chain does not come to rest");
    };
    // the push is read from the first link: outside the mouth nothing but its neighbours
    // holds it, F = −T(λ)
    let push = |chain: &ThickChain| -SHEAR * 2.0 * RADIUS * pull(f(chain.length(0)) / link);
    rest(&mut chain);
    let (first, mut before) = (stored(&chain), push(&chain));
    // the tail's way: 0.25 mm at a time over 25 mm
    let (step, count) = (2.5e-4, 100);
    let (mut work, mut variation) = (0.0, 0.0);
    let at = f(chain.positions()[0][0]);
    for n in 1..=count {
        chain.set_position(0, [fx(at + step * f64::from(n)), Fix128::ZERO]);
        rest(&mut chain);
        let now = push(&chain);
        work += (before + now) / 2.0 * step;
        variation += (now - before).abs();
        before = now;
    }
    let gain = stored(&chain) - first;
    let way = step * f64::from(count);
    // the trapezoid sum of a push of bounded variation is within half a step of its
    // variation; at rest to 1e-5 m/s the drag's force is below 1e-3 N
    let tol = step * variation / 2.0 + 1.0e-3 * way;
    assert!(
        work >= gain - tol,
        "{links} links: the push worked {work:e}, the links took {gain:e} (within {tol:e})"
    );
    // a link's share of the energy: its feed (λ_F of its rest length) times F*
    let share = limit * threshold().1 * link;
    assert!(
        (gain - limit * way).abs() <= 2.0 * share,
        "{links} links took {gain:e} over {way} m, F* times the way is {:e} (a link's share {share:e})",
        limit * way
    );
    (work / gain, gain / (limit * way))
}

#[test]
fn moved_slowly_the_push_works_no_less_than_the_links_take() {
    // one resolution (20 mm links); the other and the order between them are in the test below
    let (ratio, _) = slow_feed(10);
    assert!(ratio >= 1.0, "the work over the energy is {ratio}");
}

#[test]
#[ignore = "runtime: about 43 s at opt-level 2, measured 2026-11-19 (two chains moved through 100 rests each); run by the weekly job"]
fn moved_slowly_the_push_works_no_less_than_the_links_take_and_the_loss_goes_with_the_links_length()
{
    let (long, _) = slow_feed(10);
    let (short, taken) = slow_feed(20);
    assert!(
        short < long,
        "the work over the energy is {long} with 20 mm links and {short} with 10 mm"
    );
    println!("work over energy: {long:.3} (20 mm links), {short:.3} (10 mm); energy over F* way {taken:.4}");
}

#[test]
fn fed_every_link_is_as_long_as_the_channel_makes_it() {
    let limit = threshold().0 * SHEAR * 2.0 * RADIUS;
    let links = 20;
    let chain = pushed(
        chain(SHEAR, DRAG, links),
        2.0 * limit,
        f64::from(TAPER) + 0.02,
    );
    assert!(f(chain.positions()[0][0]) > f64::from(TAPER), "not fed");
    let held = 2.0 * RADIUS * (BODY / links as f64) / (2.0 * f64::from(NARROW));
    for k in 0..chain.link_count() {
        let l = f(chain.length(k));
        // the walls are f32 and the chain is moving: the drag's share of the tension is
        // below a thousandth of the length
        assert!(
            (l - held).abs() <= held * 1.0e-3,
            "link {k}: {l:e}, expected {held:e}"
        );
    }
}

/// the conforming rule's energy of the chain, J per metre of depth, from its definition:
/// per link the samples' shares (laid out once for the longest the link may be, at most half
/// the least radius apart there; trapezoid weights), each as stretched as the wall asks,
/// `μ = r_rest / d`, or as the free stretch `λ*`, whichever is more, with
/// `l / l0 = Σ ω max(λ*, μ)`
fn stored_in_shares(chain: &ThickChain) -> f64 {
    let least = 0.005;
    let mut total = 0.0;
    for k in 0..chain.link_count() {
        let (a, b) = (chain.positions()[k], chain.positions()[k + 1]);
        let (l, rest) = (f(chain.length(k)), f(chain.rest_length(k)));
        // the chain's own layout: laid out once for the link's reach, whatever its length
        // (the count itself is pinned in analytic_thick_chain_shares.rs)
        let n = usize::try_from(chain.spans(k).expect("a count")).expect("a count");
        let mut weights = Vec::new();
        let mut asked = Vec::new();
        for j in 0..=n {
            let t = j as f64 / n as f64;
            let x = f(a[0]) * (1.0 - t) + f(b[0]) * t;
            let y = f(a[1]) * (1.0 - t) + f(b[1]) * t;
            let d = f64::from(mouth(x as f32, y as f32)).max(least);
            weights.push(if j == 0 || j == n { 0.5 } else { 1.0 } / n as f64);
            asked.push(RADIUS / d);
        }
        let ratio = l / rest;
        let need: f64 = weights.iter().zip(&asked).map(|(w, m)| w * m).sum();
        // the free stretch by bisection (the sum grows with it); 0 when every share is held
        let (mut lo, mut hi) = (0.0f64, ratio * 4.0);
        if ratio > need {
            for _ in 0..200 {
                let mid = (lo + hi) / 2.0;
                let sum: f64 = weights
                    .iter()
                    .zip(&asked)
                    .map(|(w, m)| w * m.max(mid))
                    .sum();
                if sum < ratio {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
        let energy = |stretch: f64| SHEAR / 2.0 * (stretch - 1.0 / stretch).powi(2);
        total += f(chain.area(k))
            * weights
                .iter()
                .zip(&asked)
                .map(|(w, m)| w * energy(m.max(lo)))
                .sum::<f64>();
    }
    total
}

#[test]
fn a_conforming_chain_pays_the_squeeze_as_it_goes_not_a_link_at_a_time() {
    let limit = threshold().0 * SHEAR * 2.0 * RADIUS;
    let links = 20;
    let fed = |share: f64| {
        let chain = chain_of(SHEAR, DRAG, links, ThickChainFit::Conforming);
        f(pushed(chain, share * limit, 0.0).positions()[0][0]) > 0.0
    };
    // not below the closed form; and at 1.1 of it, where the disc rule's 10 mm links stall
    // on their lumps (the first test of this file)
    assert!(!fed(0.95), "fed below the limit");
    assert!(fed(1.1), "not fed at 1.1 of the limit");
    // moved slowly by its tail from where 0.9 of the limit left it: the push's work is the
    // energy the shares took, both ways
    let stalled = pushed(
        chain_of(SHEAR, DRAG, links, ThickChainFit::Conforming),
        0.9 * limit,
        0.0,
    );
    assert!(f(stalled.positions()[0][0]) < 0.0, "not stalled");
    let link = BODY / links as f64;
    let mut chain = chain_of(SHEAR, 24.0, links, ThickChainFit::Conforming);
    for (i, p) in stalled.positions().iter().enumerate() {
        chain.set_position(i, *p);
    }
    chain.pin(0);
    let walls = walls();
    let rest = |chain: &mut ThickChain| {
        for _ in 0..900 {
            chain
                .step_in(Fix128::from_ratio(1, 60), &walls, &[])
                .expect("a step");
            let still = chain
                .velocities()
                .iter()
                .all(|v| f(v[0]).abs() < 1.0e-5 && f(v[1]).abs() < 1.0e-5);
            if still {
                return;
            }
        }
        panic!("the chain does not come to rest");
    };
    let push = |chain: &ThickChain| -SHEAR * 2.0 * RADIUS * pull(f(chain.length(0)) / link);
    rest(&mut chain);
    let (first, mut before) = (stored_in_shares(&chain), push(&chain));
    let (step, count) = (2.5e-4, 100);
    let (mut work, mut variation) = (0.0, 0.0);
    let at = f(chain.positions()[0][0]);
    for n in 1..=count {
        chain.set_position(0, [fx(at + step * f64::from(n)), Fix128::ZERO]);
        rest(&mut chain);
        let now = push(&chain);
        work += (before + now) / 2.0 * step;
        variation += (now - before).abs();
        before = now;
    }
    let gain = stored_in_shares(&chain) - first;
    let way = step * f64::from(count);
    // the trapezoid sum of a push of bounded variation is within half a step of its
    // variation; at rest to 1e-5 m/s the drag's force is below 1e-3 N
    let tol = step * variation / 2.0 + 1.0e-3 * way;
    println!(
        "conforming slow feed: work {work:.5} gain {gain:.5} (within {tol:.5}); over F* way {:.4}",
        gain / (limit * way)
    );
    assert_eq!(
        chain.walls_unsettled(),
        0,
        "a pass left its walls unsettled"
    );
    assert!(
        (work - gain).abs() <= tol,
        "the push worked {work:e}, the shares took {gain:e}, within {tol:e}"
    );
}

/// what the walls ask of link `k`'s length over its rest length, `Σ ω_j μ_j` over the chain's
/// own layout: the shares as stretched as the walls make them, `μ = r_rest / d`
fn walls_ask(chain: &ThickChain, k: usize) -> f64 {
    let (a, b) = (chain.positions()[k], chain.positions()[k + 1]);
    let n = usize::try_from(chain.spans(k).expect("a count")).expect("a count");
    (0..=n)
        .map(|j| {
            let t = j as f64 / n as f64;
            let x = f(a[0]) * (1.0 - t) + f(b[0]) * t;
            let y = f(a[1]) * (1.0 - t) + f(b[1]) * t;
            let d = f64::from(mouth(x as f32, y as f32)).max(0.005);
            let weight = if j == 0 || j == n { 0.5 } else { 1.0 } / n as f64;
            weight * RADIUS / d
        })
        .sum()
}

#[test]
fn at_rest_no_link_is_shorter_than_its_walls_make_it() {
    // a body that keeps its area between walls is as long as they make it, and no shorter:
    // `l >= l0 Σ ω_j μ_j` for every link, the hard condition of the conforming rule. A chain
    // pushed into the taper by 0.9 of the feeding push and left to rest has a run of links
    // side by side on that condition; each is on it, not below it. The walls are f32 (2^-23
    // of a distance), the sum here f64 against the chain's Q64.64, and the rest is to
    // 1e-5 m/s: 1e-6 of the length the walls ask covers them.
    let limit = threshold().0 * SHEAR * 2.0 * RADIUS;
    let links = 20;
    let stalled = pushed(
        chain_of(SHEAR, DRAG, links, ThickChainFit::Conforming),
        0.9 * limit,
        0.0,
    );
    let mut chain = chain_of(SHEAR, 24.0, links, ThickChainFit::Conforming);
    for (i, p) in stalled.positions().iter().enumerate() {
        chain.set_position(i, *p);
    }
    chain.pin(0);
    let walls = walls();
    let mut rested = false;
    for _ in 0..900 {
        chain
            .step_in(Fix128::from_ratio(1, 60), &walls, &[])
            .expect("a step");
        if chain
            .velocities()
            .iter()
            .all(|v| f(v[0]).abs() < 1.0e-5 && f(v[1]).abs() < 1.0e-5)
        {
            rested = true;
            break;
        }
    }
    assert!(rested, "the chain does not come to rest");
    let link = BODY / links as f64;
    let mut on_the_walls = 0;
    for k in 0..links {
        let (ratio, ask) = (f(chain.length(k)) / link, walls_ask(&chain, k));
        assert!(
            ratio >= ask * (1.0 - 1.0e-6),
            "link {k} is {ratio} of its rest length, the walls make it {ask}: short by {:e}",
            ask - ratio
        );
        if ratio <= ask * (1.0 + 1.0e-6) {
            on_the_walls += 1;
        }
    }
    // the scene has what it is about: several links side by side on the condition
    assert!(on_the_walls >= 3, "{on_the_walls} links on their walls");
    assert_eq!(
        chain.walls_unsettled(),
        0,
        "a pass left its walls unsettled"
    );
    // and the rest state is the least of the stated energy among the states the walls allow:
    // one particle moved a micrometre either way along the pipe, the others where they are.
    // A move that makes a link shorter than its walls allow is not one of those states; any
    // other must not lower the energy. The walls are f32: of the 0.1 J/m here that is
    // 1e-8; the force left at rest (below 1e-3 N) over a micrometre is 1e-9.
    let energy = stored_in_shares(&chain);
    let (way, mut tried) = (1.0e-6, 0);
    for i in 1..=links {
        for side in [-1.0, 1.0] {
            let mut moved = chain_of(SHEAR, 24.0, links, ThickChainFit::Conforming);
            for (j, p) in chain.positions().iter().enumerate() {
                moved.set_position(j, *p);
            }
            let at = chain.positions()[i];
            moved.set_position(i, [fx(f(at[0]) + side * way), at[1]]);
            let allowed = (0..links)
                .all(|k| f(moved.length(k)) / link >= walls_ask(&moved, k) * (1.0 - 1.0e-6));
            if !allowed {
                continue;
            }
            tried += 1;
            let after = stored_in_shares(&moved);
            assert!(
                after >= energy - 2.0e-8,
                "particle {i} moved by {:e}: the energy goes from {energy:e} to {after:e}",
                side * way
            );
        }
    }
    // the free links' particles can go both ways, the held ones one way or neither
    assert!(tried >= 20, "{tried} moves tried");
}
