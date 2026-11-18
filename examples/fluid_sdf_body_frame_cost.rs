//! Cost of one PBF frame with SDF density-map bodies, split into the base PBF and the map
//! terms: the same fluid (a 1.6 m tank filled to 0.8 m around a 0.4 m cube lying on the
//! floor, s = 0.1 m, h = 2 s) stepped with the tank and the cube as density maps, and stepped
//! with no solids at all; then the map lookups alone (one per particle, solid and constraint
//! iteration, plus the colour pass), and the SDF normals the attribution asks for.
//!
//! Prints milliseconds per frame (mean over the timed frames after one warm-up frame).
//!
//! Author: Moroya Sakamoto

use alice_physics_core::fluid::{Fluid, FluidConfig};
use alice_physics_core::fluid_boundary_map::BoundaryMap;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};
use std::time::Instant;

const S: f64 = 0.1;
const W: f64 = 1.6;
const CUBE: f64 = 0.4;
const TOP: f64 = 0.8;
const RHO0: f64 = 1000.0;
const TIMED: usize = 5;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn box_sdf(p: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let q: Vec<f32> = (0..3).map(|i| (lo[i] - p[i]).max(p[i] - hi[i])).collect();
    let outside = q
        .iter()
        .map(|x| {
            let m = x.max(0.0);
            m * m
        })
        .sum::<f32>()
        .sqrt();
    let inside = q[0].max(q[1]).max(q[2]).min(0.0);
    outside + inside
}

fn collider(f: impl Fn([f32; 3]) -> f32 + Send + Sync + Clone + 'static) -> SdfCollider {
    let g = f.clone();
    let field = ClosureSdf::new(
        move |x, y, z| f([x, y, z]),
        move |x, y, z| {
            let e = 1e-4_f32;
            let p = [x, y, z];
            let mut n = [0.0_f32; 3];
            for a in 0..3 {
                let (mut p1, mut p0) = (p, p);
                p1[a] += e;
                p0[a] -= e;
                n[a] = (g(p1) - g(p0)) / (2.0 * e);
            }
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
            (n[0] / l, n[1] / l, n[2] / l)
        },
    );
    SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn tank() -> SdfCollider {
    let (w, l, t) = (W as f32, (2.0 * S) as f32, (2.0 * TOP) as f32);
    collider(move |p| {
        let outer = box_sdf(p, [-l, -l, -l], [w + l, t, w + l]);
        let hole = box_sdf(p, [0.0, 0.0, 0.0], [w, t + 1.0, w]);
        outer.max(-hole)
    })
}

fn cube() -> SdfCollider {
    let lo = ((W - CUBE) / 2.0) as f32;
    let c = CUBE as f32;
    collider(move |p| box_sdf(p, [lo, 0.0, lo], [lo + c, c, lo + c]))
}

fn config() -> FluidConfig {
    let s = fx(S);
    let h = s * Fix128::from_int(2);
    FluidConfig {
        rest_spacing: s,
        kernel_radius: h,
        rest_density: fx(RHO0),
        particle_mass: FluidConfig::lattice_mass(fx(RHO0), h, s),
        ..FluidConfig::default()
    }
}

fn lattice() -> Vec<Vec3Fix> {
    let nx = (W / S).round() as i64;
    let ny = (TOP / S).round() as i64;
    let lo = ((W - CUBE) / 2.0 / S).round() as i64;
    let size = (CUBE / S).round() as i64;
    let mut pts = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nx {
                let inside = (lo..lo + size).contains(&i)
                    && (0..size).contains(&j)
                    && (lo..lo + size).contains(&k);
                if !inside {
                    pts.push(v(
                        (i as f64 + 0.5) * S,
                        (j as f64 + 0.5) * S,
                        (k as f64 + 0.5) * S,
                    ));
                }
            }
        }
    }
    pts
}

fn ms_per_frame(fluid: &mut Fluid) -> f64 {
    let dt = Fix128::from_ratio(1, 60);
    fluid.step(dt);
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: wall time printed, never asserted"
    )]
    let t0 = Instant::now();
    for _ in 0..TIMED {
        fluid.step(dt);
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: wall time printed, never asserted"
    )]
    let total_s = t0.elapsed().as_secs_f64();
    total_s * 1e3 / TIMED as f64
}

fn main() {
    let cfg = config();
    let pts = lattice();
    let n = pts.len();
    let r = 2.0 * S + 0.3;

    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: wall time printed, never asserted"
    )]
    let t0 = Instant::now();
    let mut with_bodies = Fluid::new(pts.clone(), cfg)
        .with_sdf_body(tank(), v(-r, -r, -r), v(W + r, TOP + 2.0 * S, W + r))
        .with_sdf_body(
            cube(),
            v((W - CUBE) / 2.0 - 0.3, -0.3, (W - CUBE) / 2.0 - 0.3),
            v((W + CUBE) / 2.0 + 0.3, CUBE + 0.3, (W + CUBE) / 2.0 + 0.3),
        );
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: wall time printed, never asserted"
    )]
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut no_bodies = Fluid::new(pts.clone(), cfg);

    let with_ms = ms_per_frame(&mut with_bodies);
    let base_ms = ms_per_frame(&mut no_bodies);

    // the lookups a frame makes: per substep, (iterations + 1) samples per particle and solid
    let s = fx(S);
    let h = s * Fix128::from_int(2);
    let step = s / Fix128::from_int(4);
    let maps = [
        BoundaryMap::from_sdf(
            &tank(),
            v(-r, -r, -r),
            v(W + r, TOP + 2.0 * S, W + r),
            step,
            fx(RHO0),
            h,
        ),
        BoundaryMap::from_sdf(
            &cube(),
            v((W - CUBE) / 2.0 - 0.3, -0.3, (W - CUBE) / 2.0 - 0.3),
            v((W + CUBE) / 2.0 + 0.3, CUBE + 0.3, (W + CUBE) / 2.0 + 0.3),
            step,
            fx(RHO0),
            h,
        ),
    ];
    let lookups_per_particle = cfg.substeps * (cfg.iterations + 1);
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: wall time printed, never asserted"
    )]
    let t1 = Instant::now();
    let mut acc = Fix128::ZERO;
    for _ in 0..lookups_per_particle {
        for p in &with_bodies.positions {
            for m in &maps {
                acc = acc + m.sample(*p).0;
            }
        }
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: wall time printed, never asserted"
    )]
    let sample_ms = t1.elapsed().as_secs_f64() * 1e3;
    // SDF normals: two per particle and constraint iteration where both solids are near
    // (upper bound: every particle)
    let sdfs = [tank(), cube()];
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: wall time printed, never asserted"
    )]
    let t2 = Instant::now();
    let mut nsum = 0.0_f32;
    for _ in 0..(cfg.substeps * cfg.iterations) {
        for p in &with_bodies.positions {
            for sdf in &sdfs {
                let (x, y, z) = p.to_f32();
                nsum += sdf.field.normal(x, y, z).1;
            }
        }
    }
    #[allow(
        clippy::disallowed_methods,
        reason = "timing harness: wall time printed, never asserted"
    )]
    let normal_ms = t2.elapsed().as_secs_f64() * 1e3;

    println!(
        "particles {n}, substeps {}, iterations {}",
        cfg.substeps, cfg.iterations
    );
    println!(
        "map build (both solids): {build_ms:.0} ms, grid nodes {} + {}",
        maps[0].node_count(),
        maps[1].node_count()
    );
    println!("frame with the two solids: {with_ms:.1} ms");
    println!("frame with no solids (base PBF): {base_ms:.1} ms");
    println!("map lookups of one frame: {sample_ms:.1} ms");
    println!("SDF normals of one frame (every particle, upper bound): {normal_ms:.1} ms");
    println!("(checksums {} {nsum})", acc.to_f64());
    // the forces the fluid put on the solids in the last timed frame (N): Newton 3 says they
    // sum to minus the solids' push on the fluid
    let forces = with_bodies.body_forces();
    let sum = forces
        .iter()
        .fold(with_bodies.boundary_force(), |a, f| a + *f);
    println!(
        "force on the tank {:.1} N, on the cube {:.1} N (y), sum with the fluid side {:.3e} N",
        forces[0].y.to_f64(),
        forces[1].y.to_f64(),
        sum.length().to_f64()
    );
    // the same force on the cube (it rests on the floor), particle by particle, split above
    // / below / beside it
    let per = with_bodies.body_particle_forces();
    let nb = with_bodies.body_count();
    let (mut above, mut below, mut beside) = (0.0, 0.0, 0.0);
    for (i, p) in with_bodies.positions.iter().enumerate() {
        let fy = per[i * nb + 1].y.to_f64();
        let y = p.y.to_f64();
        if y > CUBE {
            above += fy;
        } else if y < 0.0 {
            below += fy;
        } else {
            beside += fy;
        }
    }
    println!("cube by face (y): above {above:.1} N, below {below:.1} N, beside {beside:.1} N");
}
