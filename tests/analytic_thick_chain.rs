//! Closed-form oracles of the thick chain ([`alice_physics_core::thick_chain::ThickChain`]): a
//! planar chain of links, each of a fixed area, so a stretched link is thinner and a link
//! held between walls is longer.
//!
//! 1. Radius law: a link of rest length `l0` and rest radius `r0` has area `2 r0 l0`; at
//!    length `2 l0` its radius is `r0 / 2`, at `l0 / 2` it is `2 r0`, and it never goes below
//!    the least radius. The values are dyadic, so the only error is the square root's.
//! 2. One hanging link: the upper particle held, the lower of mass `m` under gravity `g`
//!    settles at the extension `α m g` (`α` the link's compliance). A substep at rest is a
//!    fixed point exactly there, whatever the step; the swing left after a time `T` under the
//!    drag `k` is below the first extension times `e^(−k T / 2)`.
//! 3. One free link let go stretched: the constraint is solved implicitly (backward Euler),
//!    so the extension after `n` substeps is `e ρⁿ cos(n θ)` with `ρ = 1 / √(1 + (ω h)²)`,
//!    `θ = atan(ω h)`, `ω² = (w0 + w1) / α`: the period is `2 π h / θ` (`2 π / ω` as `h → 0`)
//!    and it does not depend on the number of iterations.
//! 4. A straight channel narrower than the chain: every link settles at the length
//!    `A / (2 c)` (`c` the channel's half-width) on the channel's middle line.
//! 5. A chain sliding on a floor under gravity stops after `v0² / (2 μ g)` (Coulomb).
//! 6. A chain sliding along a channel that holds it thin stops after `v0² / (2 a)`,
//!    `a = μ T l / (c m)`: the two walls together press on a link with `T l / c`, `T = (l −
//!    l0) / α` the link's tension, `m` its mass.
//! 7. A bent pipe, driven: no sample of the chain is closer to a wall than its link's radius,
//!    no point of the outline is in a wall by more than the bound of the sample spacing, the
//!    middle line never touches a wall, and past the mouth the links are back at their rest
//!    radius. For a pipe wider than the chain and for one narrower.
//! 8. A run repeats bit for bit; a refused step changes nothing.
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

fn linear(compliance: f64) -> ThickChainStretch {
    ThickChainStretch::Linear {
        compliance: fx(compliance),
    }
}

fn config() -> ThickChainConfig {
    ThickChainConfig {
        substeps: 8,
        iterations: 4,
        gravity: ZERO2,
        stretch: linear(1.0e-4),
        drag: Fix128::ZERO,
        friction: Fix128::ZERO,
        min_radius: fx(0.005),
        splits: 0,
        relaxation: Fix128::ZERO,
    }
}

/// a straight chain of `links` links of length `l` from `start` along x
fn row(start: [f64; 2], l: f64, links: usize) -> Vec<P2> {
    (0..=links)
        .map(|i| [fx(start[0] + l * i as f64), fx(start[1])])
        .collect()
}

#[test]
fn a_link_of_fixed_area_is_thinner_when_longer() {
    let (l0, r0) = (Fix128::from_ratio(1, 64), Fix128::from_ratio(1, 32));
    let mut cfg = config();
    cfg.min_radius = Fix128::from_ratio(1, 1024);
    let points = [ZERO2, [l0, Fix128::ZERO]];
    let mut chain = ThickChain::new(cfg, &points, r0, fx(1000.0)).expect("a valid chain");
    // the square root of a dyadic square: within 2 units of the last place, and the division
    // by twice the length doubles that at most (lengths below 1)
    let ulp = |n: u64| Fix128::from_raw(0, n);
    let near = |a: Fix128, b: Fix128| (a - b).abs() <= ulp(4);
    assert_eq!(chain.area(0), r0 * l0 + r0 * l0);
    assert!(near(chain.radius(0), r0));
    chain.set_position(1, [l0 + l0, Fix128::ZERO]);
    assert!(near(chain.length(0), l0 + l0));
    assert!(near(chain.radius(0), r0.half()));
    chain.set_position(1, [l0.half(), Fix128::ZERO]);
    assert!(near(chain.radius(0), r0 + r0));
    // 64 times the rest length would be r0 / 64 = 1/2048: the least radius holds
    chain.set_position(1, [Fix128::ONE, Fix128::ZERO]);
    assert_eq!(chain.radius(0), cfg.min_radius);
}

#[test]
fn a_hanging_link_stretches_by_its_compliance_times_the_weight() {
    let (l0, r0, rho, g, alpha, drag) = (0.02, 0.01, 1000.0, 9.81, 1.0e-4, 30.0);
    let mut cfg = config();
    cfg.gravity = [Fix128::ZERO, fx(-g)];
    cfg.stretch = linear(alpha);
    cfg.drag = fx(drag);
    let points = [[Fix128::ZERO, fx(l0)], ZERO2];
    let mut chain = ThickChain::new(cfg, &points, fx(r0), fx(rho)).expect("a valid chain");
    chain.pin(0);
    // the lower particle carries half the link: rho * (2 r0 l0) / 2
    let m = rho * r0 * l0;
    let expect = alpha * m * g;
    let seconds = 2.0;
    for _ in 0..(seconds * 60.0) as usize {
        chain.step(Fix128::from_ratio(1, 60)).expect("a step");
    }
    let got = f(chain.length(0)) - l0;
    // let go at rest length, the swing starts at the extension itself and the drag takes it
    // down by e^(-k t / 2)
    let bound = expect * (-drag * seconds / 2.0).exp();
    assert!(
        (got - expect).abs() <= bound,
        "extension {got:e}, expected {expect:e} within {bound:e}"
    );
}

/// the extensions of a free link let go stretched by `e`, one per substep
fn swing(iterations: u32, steps: usize) -> (Vec<f64>, f64, f64) {
    let (l0, r0, rho, alpha, e) = (0.02, 0.01, 1000.0, 1.0e-4, 0.002);
    let mut cfg = config();
    cfg.substeps = 1;
    cfg.iterations = iterations;
    cfg.stretch = linear(alpha);
    cfg.min_radius = fx(0.001);
    let points = [ZERO2, [fx(l0), Fix128::ZERO]];
    let mut chain = ThickChain::new(cfg, &points, fx(r0), fx(rho)).expect("a valid chain");
    chain.set_position(1, [fx(l0 + e), Fix128::ZERO]);
    let e = f(chain.length(0)) - l0;
    let w = 1.0 / (rho * r0 * l0);
    let omega = ((w + w) / alpha).sqrt();
    let dt = Fix128::from_ratio(1, 2000);
    let mut out = vec![e];
    for _ in 0..steps {
        chain.step(dt).expect("a step");
        out.push(f(chain.length(0)) - l0);
    }
    (out, omega * f(dt), e)
}

#[test]
fn a_free_link_swings_on_the_backward_euler_solution_whatever_the_iterations() {
    let steps = 400;
    for iterations in [1, 8] {
        let (got, wh, e) = swing(iterations, steps);
        let (rho, theta) = (1.0 / (1.0 + wh * wh).sqrt(), wh.atan());
        // at least two periods in the run
        assert!(steps as f64 * theta > 4.0 * core::f64::consts::PI);
        let mut worst = 0.0f64;
        for (n, x) in got.iter().enumerate() {
            let expect = e * rho.powi(n as i32) * (n as f64 * theta).cos();
            worst = worst.max((x - expect).abs());
        }
        // the oracle is evaluated in f64 (relative 2^-52 per operation, a few per sample) and
        // the chain rounds at 2^-64 after dividing by h^2 (4e6): both far below 1e-9 of the
        // first extension, which a wrong period (one part in 1e3) exceeds by six orders
        assert!(
            worst <= 1.0e-9 * e,
            "iterations {iterations}: worst {worst:e} of {e:e}"
        );
    }
}

/// the free space between two walls at `y = ±c`
fn channel(c: f32) -> DistanceSdfQuery<impl Fn(f32, f32, f32) -> f32> {
    DistanceSdfQuery::new(move |_x: f32, y: f32, _z: f32| c - y.abs())
}

#[test]
fn a_chain_in_a_narrow_channel_is_longer_by_the_ratio_of_the_widths() {
    let (l0, r0, rho, alpha, c) = (0.01, 0.05, 1000.0, 1.0e-4, 0.027f32);
    let mut cfg = config();
    cfg.stretch = linear(alpha);
    cfg.drag = fx(30.0);
    let area = 2.0 * r0 * l0;
    let held = area / (2.0 * f64::from(c));
    // let go a little longer than it settles, 2 mm off the middle
    let mut chain =
        ThickChain::new(cfg, &row([0.0, 0.0], l0, 6), fx(r0), fx(rho)).expect("a valid chain");
    for i in 0..chain.particle_count() {
        chain.set_position(i, [fx(1.05 * held * i as f64), fx(0.002)]);
    }
    let walls = channel(c);
    let seconds = 3.0;
    for _ in 0..(seconds * 60.0) as usize {
        chain
            .step_in(Fix128::from_ratio(1, 60), &walls, &[])
            .expect("a step");
    }
    // a substep at rest: the stretch constraint shortens a link by (ω h)² of its extension
    // (to first order), the walls give it back; so a link at rest is within that of `held`.
    // The walls are f32: the half-width is known to 2^-24 of it.
    let w = 1.0 / (rho * r0 * l0);
    let h = 1.0 / 60.0 / f64::from(cfg.substeps);
    let wh2 = (w + w) / alpha * h * h;
    let tol = wh2 * (held - l0) + held * 2.0f64.powi(-23);
    for k in 0..chain.link_count() {
        let l = f(chain.length(k));
        assert!(
            (l - held).abs() <= tol,
            "link {k}: {l:e}, expected {held:e} within {tol:e}"
        );
    }
    // on the middle line: the wall's slope is a central difference over ±e (an eighth of the
    // least radius), which pulls a link towards the middle from either side and is zero only
    // on it; a slope taken to one side would settle e / 2 off. Within e / 4.
    let band = f(cfg.min_radius) / 32.0;
    for (i, p) in chain.positions().iter().enumerate() {
        assert!(f(p[1]).abs() <= band, "particle {i}: y {:e}", f(p[1]));
    }
}

#[test]
fn a_chain_sliding_on_a_floor_stops_after_the_coulomb_distance() {
    let (l0, r0, rho, g, mu, v0) = (0.01, 0.01, 1000.0, 9.81, 0.3, 0.5);
    let mut cfg = config();
    cfg.gravity = [Fix128::ZERO, fx(-g)];
    cfg.friction = fx(mu);
    // stiff links: the weight flattens them by less than a part in 1e4
    cfg.stretch = linear(1.0e-7);
    let mut chain =
        ThickChain::new(cfg, &row([0.0, r0], l0, 4), fx(r0), fx(rho)).expect("a valid chain");
    for i in 0..chain.particle_count() {
        chain.set_velocity(i, [fx(v0), Fix128::ZERO]);
    }
    let floor = DistanceSdfQuery::new(|_x: f32, y: f32, _z: f32| y);
    let start = f(chain.positions()[0][0]);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        chain.step_in(dt, &floor, &[]).expect("a step");
    }
    let got = f(chain.positions()[0][0]) - start;
    let expect = v0 * v0 / (2.0 * mu * g);
    // the speed drops by μ g h at the end of each substep, after the substep's travel: the
    // travel is the sum of a staircase above the parabola, v0 h / 2 more than it
    let h = f(dt) / f64::from(cfg.substeps);
    assert!(
        got >= expect && got <= expect + v0 * h,
        "travel {got:e}, expected {expect:e} to {:e} more",
        v0 * h
    );
    // at rest: what is left is the links' swing, which the friction takes down by μ g h a
    // substep at the most, so no particle moves faster than that
    for v in chain.velocities() {
        assert!(f(v[0]).abs() <= mu * g * h, "still moving: {:e}", f(v[0]));
    }
}

#[test]
fn a_chain_held_thin_by_a_channel_stops_after_the_squeeze_friction_distance() {
    let (l0, r0, rho, alpha, c, mu, v0) = (0.01, 0.05, 1000.0, 1.0e-4, 0.027f32, 0.05, 0.3);
    let mut cfg = config();
    cfg.stretch = linear(alpha);
    cfg.friction = fx(mu);
    let area = 2.0 * r0 * l0;
    let held = area / (2.0 * f64::from(c));
    let links = 6;
    let mut chain =
        ThickChain::new(cfg, &row([0.0, 0.0], l0, links), fx(r0), fx(rho)).expect("a valid chain");
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
    let tension = (held - l0) / alpha;
    let a = mu * tension * held / f64::from(c) / (rho * area);
    let expect = v0 * v0 / (2.0 * a);
    // the staircase of 5, and the tension known to (ω h)² of it (oracle 4)
    let h = f(dt) / f64::from(cfg.substeps);
    let w = 1.0 / (rho * r0 * l0);
    let wh2 = (w + w) / alpha * h * h;
    let tol = 2.0 * v0 * h + 2.0 * wh2 * expect;
    assert!(
        (got - expect).abs() <= tol,
        "travel {got:e}, expected {expect:e} within {tol:e}"
    );
}

/// the bent pipe: a level run from `A` to `B`, then up to the mouth, free space above it
const A: [f32; 2] = [0.10, 0.20];
const B: [f32; 2] = [0.60, 0.20];
const MOUTH: f32 = 0.60;

fn nearest(x: [f32; 2], p: [f32; 2], q: [f32; 2]) -> [f32; 2] {
    let d = [q[0] - p[0], q[1] - p[1]];
    let t = (((x[0] - p[0]) * d[0] + (x[1] - p[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]))
        .clamp(0.0, 1.0);
    [p[0] + t * d[0], p[1] + t * d[1]]
}

/// the distance to the pipe's wall (positive in the free space), a lower bound of it where the
/// two runs meet
fn pipe(half: f32, x: f32, y: f32) -> f32 {
    let top = [B[0], MOUTH + 10.0];
    let mut best = y - MOUTH;
    for (p, q) in [(A, B), (B, top)] {
        let c = nearest([x, y], p, q);
        let len = ((c[0] - x) * (c[0] - x) + (c[1] - y) * (c[1] - y)).sqrt();
        best = best.max(half - len);
    }
    best
}

/// how far along the pipe's line a point is, and the line's direction there
fn along(x: [f32; 2]) -> (f32, [f64; 2]) {
    let level = nearest(x, A, B);
    let rise = [B[0], x[1].max(B[1])];
    let d = |c: [f32; 2]| (c[0] - x[0]) * (c[0] - x[0]) + (c[1] - x[1]) * (c[1] - x[1]);
    if d(level) <= d(rise) {
        (level[0] - A[0], [1.0, 0.0])
    } else {
        (B[0] - A[0] + rise[1] - B[1], [0.0, 1.0])
    }
}

/// how far past the mouth the tail is driven, m: the rest radius 0.05 and the 0.02 the drag
/// lets it coast, with room
const CLEAR: f32 = 0.10;
/// the frames the chain is left to settle after the drive
const SETTLE: usize = 120;

/// drives a chain of rest radius 0.05 through the pipe of half-width `half`; returns the
/// digest at the end
fn through_the_pipe(half: f32) -> [u8; 32] {
    let (l0, r0, rho) = (0.01, 0.05, 1000.0);
    let mut cfg = config();
    // the squeeze lets go at ω times the extension, about 1.2 m/s: 16 substeps keep a
    // substep's travel under half the least radius
    cfg.substeps = 16;
    cfg.drag = fx(40.0);
    cfg.friction = fx(0.05);
    let r_min = f(cfg.min_radius);
    let links = 20;
    // in the level run, already as thin as the pipe holds it
    let fit = f64::from(half).min(r0);
    let l = 2.0 * r0 * l0 / (2.0 * fit) * 1.02;
    let start = [f64::from(A[0]) + 0.02, f64::from(A[1])];
    let mut chain =
        ThickChain::new(cfg, &row(start, l0, links), fx(r0), fx(rho)).expect("a valid chain");
    for i in 0..chain.particle_count() {
        chain.set_position(i, [fx(start[0] + l * i as f64), fx(start[1])]);
    }
    let walls = DistanceSdfQuery::new(move |x: f32, y: f32, _z: f32| pipe(half, x, y));
    let dist = |p: [f64; 2]| f64::from(pipe(half, p[0] as f32, p[1] as f32));
    let out = B[0] - A[0] + MOUTH - B[1];
    let dt = Fix128::from_ratio(1, 60);
    let (mut least_sample, mut least_outline, mut least_middle) =
        (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut frames_out = 0;
    for frame in 0..1200 {
        let pos: Vec<[f64; 2]> = chain
            .positions()
            .iter()
            .map(|p| [f(p[0]), f(p[1])])
            .collect();
        let tail = pos
            .iter()
            .map(|p| along([p[0] as f32, p[1] as f32]).0)
            .fold(f32::INFINITY, f32::min);
        // drive until the tail is clear of the mouth's rim by more than the rest radius (the
        // wall around the mouth is a floor to the body above it), then two seconds to settle
        let drive: Vec<P2> = pos
            .iter()
            .map(|p| {
                let way = along([p[0] as f32, p[1] as f32]).1;
                let a = if tail < out + CLEAR { 30.0 } else { 0.0 };
                [fx(a * way[0]), fx(a * way[1])]
            })
            .collect();
        chain
            .step_in(dt, &walls, &drive)
            .unwrap_or_else(|e| panic!("frame {frame}: {e:?}"));
        // 16 points to a link: the samples' clearance against the link's radius, the outline
        // and the middle line
        for k in 0..chain.link_count() {
            let (p, q) = (chain.positions()[k], chain.positions()[k + 1]);
            let r = f(chain.radius(k));
            for j in 0..=16 {
                let t = f64::from(j) / 16.0;
                let x = [
                    f(p[0]) * (1.0 - t) + f(q[0]) * t,
                    f(p[1]) * (1.0 - t) + f(q[1]) * t,
                ];
                let d = dist(x);
                least_middle = least_middle.min(d);
                least_outline = least_outline.min(d - r);
                if j == 0 || j == 16 {
                    least_sample = least_sample.min(d - r);
                }
            }
        }
        if tail >= out + CLEAR {
            frames_out += 1;
            if frames_out == SETTLE {
                break;
            }
        }
    }
    assert_eq!(
        frames_out, SETTLE,
        "half {half}: the tail did not leave the pipe"
    );
    println!(
        "half {half}: least sample {least_sample:+.3e}, outline {least_outline:+.3e}, middle {least_middle:+.3e}"
    );
    // a wall point at least r from two samples s <= r_min / 2 apart is at least
    // sqrt(r² − s² / 4) from the line between them: with r >= r_min the outline is in a wall by
    // at most r_min (1 − sqrt(15) / 4); a sample itself is short of its radius by what the
    // last pass of the iteration left, bounded by a substep's travel at the speed limit of
    // the range (r_min / 2), measured far below it
    let between = r_min * (1.0 - 15.0f64.sqrt() / 4.0);
    assert!(
        least_middle > 0.0,
        "half {half}: the middle line touched a wall"
    );
    assert!(
        least_sample >= -r_min / 2.0,
        "half {half}: a sample {least_sample:e} short of its radius"
    );
    assert!(
        least_outline >= -(r_min / 2.0 + between),
        "half {half}: the outline {least_outline:e} in a wall"
    );
    // out of the pipe and settled: back at the rest radius. Under the drag k a stretch mode
    // of frequency ω decays like e^(−k t / 2) when it swings (ω > k / 2) and like
    // e^(−(k / 2 − √(k² / 4 − ω²)) t) when it does not; the slowest is the longest wave of the
    // free chain, ω1 >= 2 ω0 sin(π / (2 (N + 1))), ω0² = 1 / (m α), m a link's mass
    let k = f(cfg.drag);
    let w0 = (1.0 / (rho * 2.0 * r0 * l0 * 1.0e-4)).sqrt();
    let w1 = 2.0 * w0 * (core::f64::consts::PI / (2.0 * (links + 1) as f64)).sin();
    let rate = if w1 > k / 2.0 {
        k / 2.0
    } else {
        k / 2.0 - (k * k / 4.0 - w1 * w1).sqrt()
    };
    let seconds = SETTLE as f64 / 60.0;
    let left = (r0 - fit).max(r0 * 0.02) * (-rate * seconds).exp();
    for k in 0..chain.link_count() {
        let r = f(chain.radius(k));
        assert!(
            (r - r0).abs() <= left + r0 * 1.0e-6,
            "half {half}: link {k} radius {r:e}, rest {r0:e} within {left:e}"
        );
    }
    chain.state_digest()
}

#[test]
fn a_chain_driven_through_a_wider_pipe_stays_out_of_the_walls() {
    let _ = through_the_pipe(0.065);
}

#[test]
fn a_chain_driven_through_a_narrower_pipe_is_squeezed_and_comes_back() {
    let _ = through_the_pipe(0.027);
}

#[test]
fn a_run_repeats_bit_for_bit() {
    assert_eq!(through_the_pipe(0.027), through_the_pipe(0.027));
    assert_ne!(through_the_pipe(0.027), through_the_pipe(0.065));
}

#[test]
fn a_refused_chain_or_step_changes_nothing() {
    let points = row([0.0, 0.0], 0.01, 3);
    let bad = |edit: &dyn Fn(&mut ThickChainConfig), field: &str| {
        let mut cfg = config();
        edit(&mut cfg);
        assert_eq!(
            ThickChain::new(cfg, &points, fx(0.02), fx(1000.0)).err(),
            Some(ThickChainError::Invalid(match field {
                "substeps" => "substeps",
                "iterations" => "iterations",
                "compliance" => "compliance",
                "shear" => "shear",
                "drag" => "drag",
                "friction" => "friction",
                _ => "min_radius",
            })),
            "{field}"
        );
    };
    bad(&|c| c.substeps = 0, "substeps");
    bad(&|c| c.iterations = 0, "iterations");
    bad(&|c| c.stretch = linear(0.0), "compliance");
    bad(
        &|c| {
            c.stretch = ThickChainStretch::NeoHookean {
                shear: Fix128::ZERO,
            }
        },
        "shear",
    );
    bad(&|c| c.drag = fx(-1.0), "drag");
    bad(&|c| c.friction = fx(-0.1), "friction");
    bad(&|c| c.min_radius = Fix128::ZERO, "min_radius");
    let make = |pts: &[P2], r: f64, rho: f64| ThickChain::new(config(), pts, fx(r), fx(rho));
    assert_eq!(
        make(&points[..1], 0.02, 1000.0).err(),
        Some(ThickChainError::Invalid("points"))
    );
    assert_eq!(
        make(&points, 0.001, 1000.0).err(),
        Some(ThickChainError::Invalid("radius"))
    );
    assert_eq!(
        make(&points, 0.02, 0.0).err(),
        Some(ThickChainError::Invalid("density"))
    );
    assert_eq!(
        make(&[points[0], points[0]], 0.02, 1000.0).err(),
        Some(ThickChainError::Degenerate { link: 0 })
    );

    let mut chain = make(&points, 0.02, 1000.0).expect("a valid chain");
    let dt = Fix128::from_ratio(1, 60);
    chain.step(dt).expect("a step");
    let before = chain.snapshot();
    // a step that is not a time
    assert_eq!(
        chain.step(Fix128::ZERO),
        Err(ThickChainError::Invalid("dt"))
    );
    // a drive for another chain
    let walls = channel(1.0);
    assert_eq!(
        chain.step_in(dt, &walls, &[ZERO2]),
        Err(ThickChainError::Invalid("accel"))
    );
    // a field that is not a number
    let broken = DistanceSdfQuery::new(|_x: f32, _y: f32, _z: f32| f32::NAN);
    assert_eq!(
        chain.step_in(dt, &broken, &[]),
        Err(ThickChainError::Field { link: 0 })
    );
    // faster than half the least radius a substep, in the first substep
    chain.set_velocity(2, [fx(10.0), Fix128::ZERO]);
    let fast = chain.snapshot();
    assert_ne!(fast, before);
    assert_eq!(
        chain.step(dt),
        Err(ThickChainError::TooFast { particle: 2 })
    );
    assert_eq!(chain.snapshot(), fast);
    // and in the second: a drive of 400 m/s² moves a particle 1.7 mm in the first substep
    // (h = 1/480 s) and 3.5 mm in the second, past the 2.5 mm; the first substep, which
    // moved every particle, is not kept
    chain.set_velocity(2, ZERO2);
    let rest = chain.snapshot();
    let drive = vec![[fx(400.0), Fix128::ZERO]; chain.particle_count()];
    assert_eq!(
        chain.step_in(dt, &walls, &drive),
        Err(ThickChainError::TooFast { particle: 0 })
    );
    assert_eq!(chain.snapshot(), rest);
    let fast = rest;
    assert_eq!(chain.state_digest(), {
        use sha2::{Digest, Sha256};
        let d: [u8; 32] = Sha256::digest(&fast).into();
        d
    });
}
