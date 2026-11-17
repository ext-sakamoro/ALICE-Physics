//! A soft body pushed through a bent pipe (feature `mpm2d`): walls from a signed distance
//! field, Coulomb friction, and a push on the body's tail.
//!
//! The pipe is the set of points within `RADIUS` of an L-shaped line (a level run, a corner, a
//! rise) that opens into free space at the top. The body starts in the level run; a force
//! shared over the particles of its tail pushes it round the corner for `PUSH_TIME`, then
//! lets go. Printed per sampled frame: the body's extent, the wall time of the frame, and at
//! the end the digest (the same on every machine and every run).
//!
//! ```bash
//! cargo run --release --example mpm2d_pipe --features mpm2d
//! ```

use alice_physics_core::math::Fix128;
use alice_physics_core::mpm2d::{Mpm2d, Mpm2dConfig, Mpm2dMaterial, Mpm2dModel, V2};
use alice_physics_core::sdf_collider::ClosureSdfQuery;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

const CELL: f64 = 0.005;
const RADIUS: f32 = 0.04;
/// the pipe's line: from `A` level to `B`, then up to the mouth at `MOUTH`
const A: [f32; 2] = [0.10, 0.15];
const B: [f32; 2] = [0.45, 0.15];
const MOUTH: f32 = 0.50;
const FRICTION: f64 = 0.1;
/// the push on the tail, N per metre of depth
const PUSH: f64 = 400.0;
const PUSH_TIME: f64 = 1.0;
const RUN_TIME: f64 = 1.5;
const FRAME: f64 = 1.0 / 60.0;

/// the nearest point of the segment `p q` to `x`
fn nearest(x: [f32; 2], p: [f32; 2], q: [f32; 2]) -> [f32; 2] {
    let d = [q[0] - p[0], q[1] - p[1]];
    let t = (((x[0] - p[0]) * d[0] + (x[1] - p[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]))
        .clamp(0.0, 1.0);
    [p[0] + t * d[0], p[1] + t * d[1]]
}

/// the free space as a distance (positive in it) and the direction further into it: the pipe
/// below the mouth, everything above it
fn free_space(x: f32, y: f32) -> (f32, [f32; 2]) {
    let top = [B[0], MOUTH + 1.0];
    let mut best = (f32::NEG_INFINITY, [0.0, 0.0]);
    for (p, q) in [(A, B), (B, top)] {
        let c = nearest([x, y], p, q);
        let (dx, dy) = (c[0] - x, c[1] - y);
        let len = (dx * dx + dy * dy).sqrt();
        let d = RADIUS - len;
        if d > best.0 {
            // on the line itself any direction serves: such a point is not in a wall
            best = (d, if len > 0.0 { [dx / len, dy / len] } else { [0.0, 1.0] });
        }
    }
    let open = y - MOUTH;
    if open > best.0 {
        best = (open, [0.0, 1.0]);
    }
    best
}

fn main() {
    // gel: 20 kPa, a step a third of the cell crossing time of its wave
    let (e, nu, rho) = (2.0e4f64, 0.4f64, 1000.0f64);
    let wave = (e / ((1.0 - nu * nu) * rho)).sqrt();
    let per_frame = (FRAME / (CELL / wave / 3.0)).ceil() as usize;
    let dt = FRAME / per_frame as f64;
    let mut world = Mpm2d::new(
        Mpm2dConfig {
            origin: [Fix128::ZERO, Fix128::ZERO],
            cell: fx(CELL),
            nodes: [128, 180],
            gravity: [Fix128::ZERO, fx(-9.81)],
            dt: fx(dt),
            ground: None,
        },
        Mpm2dMaterial {
            youngs: fx(e),
            poisson: fx(nu),
            density: fx(rho),
            model: Mpm2dModel::NeoHookean,
            maxwell: None,
        },
    )
    .expect("a valid world");
    let pipe = ClosureSdfQuery::new(
        |x: f32, y: f32, _z: f32| free_space(x, y).0,
        |x: f32, y: f32, _z: f32| {
            let n = free_space(x, y).1;
            (n[0], n[1], 0.0)
        },
    );
    world
        .set_walls_from_sdf(&pipe, fx(FRICTION))
        .expect("valid walls");
    // the body: a slab in the level run, 0.2 m long, 1.5 cells clear of the walls (where a
    // body rests against them)
    let half = f64::from(RADIUS) - 1.5 * CELL;
    let count = world.seed(
        |x: V2| {
            let (dx, dy) = ((x[0].to_f64() - 0.22).abs() - 0.1, (x[1].to_f64() - 0.15).abs() - half);
            fx(dx.max(dy))
        },
        2,
    );
    println!(
        "{count} particles, {per_frame} steps a frame (dt {dt:.2e} s), wall friction {:?}",
        world.wall_friction().map(Fix128::to_f64)
    );
    let frames = (RUN_TIME / FRAME).round() as usize;
    let mut worst = 0.0f64;
    for frame in 0..frames {
        let pushing = (frame as f64) * FRAME < PUSH_TIME;
        #[allow(
            clippy::disallowed_methods,
            reason = "example: the wall time of a frame, printed, never fed back"
        )]
        let clock = std::time::Instant::now();
        for _ in 0..per_frame {
            let result = if pushing {
                // the tail: the particles within a pipe radius of the body's last point along
                // the level run
                let tail = world
                    .particles()
                    .iter()
                    .min_by_key(|p| p.x[0])
                    .map(|p| p.x)
                    .expect("particles");
                let patch = world.particles_within(tail, fx(f64::from(RADIUS)));
                let loads = world
                    .shared_loads(&patch, [fx(PUSH), Fix128::ZERO])
                    .expect("a patch with area");
                world.step_with_loads(&loads)
            } else {
                world.step()
            };
            if let Err(error) = result {
                println!("frame {frame}: the step was refused: {error:?}");
                return;
            }
        }
        let secs = clock.elapsed().as_secs_f64();
        worst = worst.max(secs);
        if frame % 6 == 5 || frame + 1 == frames {
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            let mut deepest = f64::INFINITY;
            for p in world.particles() {
                let x = [p.x[0].to_f64(), p.x[1].to_f64()];
                for k in 0..2 {
                    lo[k] = lo[k].min(x[k]);
                    hi[k] = hi[k].max(x[k]);
                }
                deepest = deepest.min(f64::from(free_space(x[0] as f32, x[1] as f32).0));
            }
            println!(
                "t {:.2} s{}: x {:.3}..{:.3}, y {:.3}..{:.3}, nearest to a wall {:+.4} m, {:.1} ms",
                (frame + 1) as f64 * FRAME,
                if pushing { " push" } else { "" },
                lo[0],
                hi[0],
                lo[1],
                hi[1],
                deepest,
                secs * 1e3
            );
        }
    }
    let digest = world.state_digest();
    println!(
        "worst frame {:.1} ms; digest {}",
        worst * 1e3,
        digest.iter().map(|b| format!("{b:02x}")).collect::<String>()
    );
    // without the walls the same world is the plain one
    world.clear_walls();
    println!("walls cleared: friction {:?}", world.wall_friction());
}
