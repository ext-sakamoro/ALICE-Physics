//! A thick chain driven through a bent pipe narrower than itself: links of fixed area, so
//! the body is thinner and longer in the pipe and back at its rest width past the mouth.
//!
//! The pipe is the set of points within `HALF` of an L-shaped line (a level run, a corner, a
//! rise) that opens into free space at the top. Printed per sampled frame: where the chain's
//! ends are, its thinnest and widest link, its length against its rest length and how near
//! its outline came to a wall; at the end the digest (the same on every machine and run).
//!
//! ```bash
//! cargo run --release --example thick_chain_pipe
//! ```

use alice_physics_core::math::Fix128;
use alice_physics_core::sdf_collider::DistanceSdfQuery;
use alice_physics_core::thick_chain::{
    ThickChain, ThickChainConfig, ThickChainFit, ThickChainStretch, P2,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// the pipe's half-width, m: the chain's rest radius is 0.05
const HALF: f32 = 0.027;
const A: [f32; 2] = [0.10, 0.20];
const B: [f32; 2] = [0.60, 0.20];
const MOUTH: f32 = 0.60;
const DRIVE: f64 = 30.0;

fn nearest(x: [f32; 2], p: [f32; 2], q: [f32; 2]) -> [f32; 2] {
    let d = [q[0] - p[0], q[1] - p[1]];
    let t = (((x[0] - p[0]) * d[0] + (x[1] - p[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]))
        .clamp(0.0, 1.0);
    [p[0] + t * d[0], p[1] + t * d[1]]
}

/// the distance to the pipe's wall, positive in the free space
fn pipe(x: f32, y: f32) -> f32 {
    let top = [B[0], MOUTH + 10.0];
    let mut best = y - MOUTH;
    for (p, q) in [(A, B), (B, top)] {
        let c = nearest([x, y], p, q);
        let len = ((c[0] - x) * (c[0] - x) + (c[1] - y) * (c[1] - y)).sqrt();
        best = best.max(HALF - len);
    }
    best
}

/// the pipe's direction at a point: along the level run, then up
fn way(x: [f32; 2]) -> [f64; 2] {
    let level = nearest(x, A, B);
    let rise = [B[0], x[1].max(B[1])];
    let d = |c: [f32; 2]| (c[0] - x[0]) * (c[0] - x[0]) + (c[1] - x[1]) * (c[1] - x[1]);
    if d(level) <= d(rise) {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    }
}

/// The conforming rule in the level run: every share of a link takes the width the walls
/// leave it, so a link at rest is as long as its walls make it and no longer.
fn conforming_at_rest() {
    let (l0, r0, links) = (0.01, 0.05, 8);
    let config = ThickChainConfig {
        substeps: 16,
        iterations: 4,
        gravity: [Fix128::ZERO; 2],
        stretch: ThickChainStretch::NeoHookean { shear: fx(3000.0) },
        drag: fx(40.0),
        friction: Fix128::ZERO,
        min_radius: fx(0.005),
        splits: 8,
        relaxation: Fix128::ZERO,
        fit: ThickChainFit::Conforming,
        // the level run asks a link to be r0 / HALF = 1.85 times its rest length
        reach: fx(2.0),
    };
    let start = [f64::from(A[0]) + 0.05, f64::from(A[1])];
    let held = l0 * r0 / f64::from(HALF);
    let points: Vec<P2> = (0..=links)
        .map(|i| [fx(start[0] + l0 * i as f64), fx(start[1])])
        .collect();
    let mut chain = ThickChain::new(config, &points, fx(r0), fx(1000.0)).expect("a valid chain");
    // set in the level run as long as its walls make it
    for i in 0..chain.particle_count() {
        chain.set_position(i, [fx(start[0] + held * i as f64), fx(start[1])]);
    }
    let walls = DistanceSdfQuery::new(|x: f32, y: f32, _z: f32| pipe(x, y));
    for _ in 0..120 {
        chain
            .step_in(Fix128::from_ratio(1, 60), &walls, &[])
            .expect("a step in the level run");
    }
    let middle = links / 2;
    let asked = chain
        .free_stretch(middle, &walls)
        .expect("a link inside the walls")
        .to_f64();
    println!(
        "conforming, {links} links at rest in the level run: link {middle} is {:.4} of its rest length, the walls ask {asked:.4}, in {} shares, its walls press {:.1} N/m; passes that left walls unsettled: {}",
        chain.length(middle).to_f64() / chain.rest_length(middle).to_f64(),
        chain.spans(middle).expect("a link of the chain"),
        chain.wall_force(middle).to_f64(),
        chain.walls_unsettled()
    );
}

fn main() {
    let (l0, r0) = (0.01, 0.05);
    let config = ThickChainConfig {
        substeps: 16,
        iterations: 4,
        gravity: [Fix128::ZERO; 2],
        stretch: ThickChainStretch::Linear {
            compliance: fx(1.0e-4),
        },
        drag: fx(40.0),
        friction: fx(0.05),
        min_radius: fx(0.005),
        splits: 0,
        relaxation: Fix128::ZERO,
        fit: ThickChainFit::Disc,
        reach: Fix128::ONE,
    };
    let links = 20;
    let start = [f64::from(A[0]) + 0.02, f64::from(A[1])];
    let points: Vec<P2> = (0..=links)
        .map(|i| [fx(start[0] + l0 * i as f64), fx(start[1])])
        .collect();
    let mut chain = ThickChain::new(config, &points, fx(r0), fx(1000.0)).expect("a valid chain");
    // set in the level run as long as the pipe holds it
    let held = chain.area(0).to_f64() / (2.0 * f64::from(HALF));
    for i in 0..chain.particle_count() {
        chain.set_position(i, [fx(start[0] + 1.02 * held * i as f64), fx(start[1])]);
    }
    let rest: f64 = (0..chain.link_count())
        .map(|k| chain.rest_length(k).to_f64())
        .sum();
    println!(
        "{} links, rest length {rest:.3} m, rest radius {r0} m, pipe half-width {HALF} m, {} substeps a frame",
        chain.link_count(),
        chain.config().substeps
    );
    let walls = DistanceSdfQuery::new(|x: f32, y: f32, _z: f32| pipe(x, y));
    let dt = Fix128::from_ratio(1, 60);
    for frame in 0..150 {
        let tail = chain.positions()[0];
        // the drive ends once the tail is well past the mouth
        let on = tail[1].to_f32() < MOUTH + 0.10;
        let drive: Vec<P2> = chain
            .positions()
            .iter()
            .map(|p| {
                let w = way([p[0].to_f32(), p[1].to_f32()]);
                let a = if on { DRIVE } else { 0.0 };
                [fx(a * w[0]), fx(a * w[1])]
            })
            .collect();
        if let Err(error) = chain.step_in(dt, &walls, &drive) {
            println!("frame {frame}: the step was refused: {error:?}");
            return;
        }
        if frame % 10 == 9 {
            let (mut thin, mut wide, mut length, mut near) =
                (f64::INFINITY, 0.0f64, 0.0, f64::INFINITY);
            // the walls' press on the whole body, N per metre of depth
            let mut press = 0.0;
            for k in 0..chain.link_count() {
                let r = chain.radius(k).to_f64();
                thin = thin.min(r);
                wide = wide.max(r);
                length += chain.length(k).to_f64();
                press += chain.wall_force(k).to_f64();
                for p in [chain.positions()[k], chain.positions()[k + 1]] {
                    near = near.min(f64::from(pipe(p[0].to_f32(), p[1].to_f32())) - r);
                }
            }
            let head = chain.positions()[links];
            let speed = chain.velocities()[links];
            println!(
                "t {:.2} s{}: tail ({:.3}, {:.3}) head ({:.3}, {:.3}) at {:.2} m/s, radius {:.4}..{:.4} m, length {:.2} of rest, outline to wall {:+.4} m, the walls press {:.1} N/m",
                f64::from(frame + 1) / 60.0,
                if on { " drive" } else { "" },
                tail[0].to_f64(),
                tail[1].to_f64(),
                head[0].to_f64(),
                head[1].to_f64(),
                (speed[0] * speed[0] + speed[1] * speed[1]).sqrt().to_f64(),
                thin,
                wide,
                length / rest,
                near,
                press
            );
        }
    }
    conforming_at_rest();
    // held by its tail and let go, the chain keeps its place there
    chain.pin(0);
    let at = chain.positions()[0];
    chain.set_velocity(links, [fx(0.1), Fix128::ZERO]);
    chain.step(dt).expect("a step");
    assert_eq!(chain.positions()[0], at);
    let digest = chain.state_digest();
    println!(
        "snapshot {} bytes, digest {}",
        chain.snapshot().len(),
        digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
}
