//! The cube scene of `tests/analytic_fluid_sdf_body.rs` (a 1.6 m tank, s = 0.1 m, h = 2 s, a
//! 0.4 m cube held 2 h above the floor, fluid to 1.2 m) and the stored state of its run at frame
//! 124, shared by `tests/analytic_fluid_newton_fixture.rs` and the example
//! `fluid_newton_fixture`, which writes `tests/fixtures/fluid_newton_cube_f124.bin`.
//!
//! The file is a short text header (lines starting with `#`, then the line `data`) followed by
//! the particle count as a little-endian `u32` and, per particle, the position then the velocity,
//! each component a `Fix128` as `hi` (`i64`) then `lo` (`u64`), little-endian. The header holds
//! no paths beyond the repository's own.

#![allow(
    dead_code,
    reason = "each user takes a different part of the shared scene"
)]

use alice_physics_core::fluid::{Fluid, FluidConfig};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};

const S: f64 = 0.1;
const W: f64 = 1.6;
const CUBE: f64 = 0.4;
const RHO0: f64 = 1000.0;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// Exact SDF of an axis-aligned box `[lo, hi]` (negative inside)
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

/// Central-difference normal of an f32 SDF closure
fn fd_normal(f: &dyn Fn([f32; 3]) -> f32, p: [f32; 3]) -> (f32, f32, f32) {
    let e = 1e-4_f32;
    let mut g = [0.0_f32; 3];
    for a in 0..3 {
        let mut p1 = p;
        let mut p0 = p;
        p1[a] += e;
        p0[a] -= e;
        g[a] = (f(p1) - f(p0)) / (2.0 * e);
    }
    let n = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-12);
    (g[0] / n, g[1] / n, g[2] / n)
}

fn collider(f: impl Fn([f32; 3]) -> f32 + Send + Sync + Clone + 'static) -> SdfCollider {
    let g = f.clone();
    let field = ClosureSdf::new(
        move |x, y, z| f([x, y, z]),
        move |x, y, z| fd_normal(&g, [x, y, z]),
    );
    SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

/// Open-topped tank: floor at y = 0 and four walls of thickness 2 s around `[0, W]^2`
fn tank(top: f64) -> SdfCollider {
    let (w, l, t) = (W as f32, (2.0 * S) as f32, top as f32);
    collider(move |p| {
        let outer = box_sdf(p, [-l, -l, -l], [w + l, t, w + l]);
        let hole = box_sdf(p, [0.0, 0.0, 0.0], [w, t + 1.0, w]);
        outer.max(-hole)
    })
}

fn cube(bottom: f64) -> SdfCollider {
    let lo = ((W - CUBE) / 2.0) as f32;
    let (b, c) = (bottom as f32, CUBE as f32);
    collider(move |p| box_sdf(p, [lo, b, lo], [lo + c, b + c, lo + c]))
}

pub struct Scene {
    pub fluid: Fluid,
    pub n: usize,
}

/// Fluid on the lattice `(i + 1/2) s` filling `[0, W]^2 x [0, top]` outside the cube
pub fn scene(bottom: f64, top: f64) -> Scene {
    let s = fx(S);
    let h = s * Fix128::from_int(2);
    let config = FluidConfig {
        rest_spacing: s,
        kernel_radius: h,
        rest_density: fx(RHO0),
        particle_mass: FluidConfig::lattice_mass(fx(RHO0), h, s),
        ..FluidConfig::default()
    };
    let nx = (W / S).round() as i64;
    let ny = (top / S).round() as i64;
    let lo = ((W - CUBE) / 2.0 / S).round() as i64;
    let size = (CUBE / S).round() as i64;
    let b0 = (bottom / S).round() as i64;
    let mut pts = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nx {
                let inside = (lo..lo + size).contains(&i)
                    && (b0..b0 + size).contains(&j)
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
    let n = pts.len();
    let r = 2.0 * S + 0.3;
    let fluid = Fluid::new(pts, config)
        .with_sdf_body(
            tank(2.0 * top),
            v(-r, -r, -r),
            v(W + r, top + 2.0 * S, W + r),
        )
        .with_sdf_body(
            cube(bottom),
            v((W - CUBE) / 2.0 - 0.3, bottom - 0.3, (W - CUBE) / 2.0 - 0.3),
            v(
                (W + CUBE) / 2.0 + 0.3,
                bottom + CUBE + 0.3,
                (W + CUBE) / 2.0 + 0.3,
            ),
        );
    Scene { fluid, n }
}

/// The rounding bound on Newton 3 between the fluid and the solids, per component, in N
/// (the fluid MODEL "Newton 3"): each particle correction is rounded twice (by the particle
/// volume and by the relaxation, round to nearest), at most one ulp `2^-64` m in all, for
/// every particle on every constraint iteration of the last substep; the fluid-side total and
/// each solid's share are then multiplied by `m / dt^2` once each (half an ulp of force
/// apiece). The pair terms cancel exactly, so nothing else separates the two sides.
pub fn newton_bound(f: &Fluid, frame_dt: f64) -> f64 {
    let ulp = 1.0 / 18_446_744_073_709_551_616.0_f64; // 2^-64, exact
    let dt = frame_dt / f.config.substeps as f64;
    let force_per_correction = f.config.particle_mass.to_f64() / (dt * dt);
    let count = (f.particle_count() * f.config.iterations) as f64;
    (count * force_per_correction + f.body_count() as f64 + 1.0) * ulp
}

/// The largest component of `v`, in N
pub fn largest(v: Vec3Fix) -> f64 {
    v.x.to_f64()
        .abs()
        .max(v.y.to_f64().abs())
        .max(v.z.to_f64().abs())
}

/// The cube's bottom and the fluid's top of the stored scene, in m
pub const BOTTOM: f64 = 0.4;
pub const TOP: f64 = BOTTOM + 0.8;
/// Frames of 1/60 s stepped from the lattice before the state is stored
pub const FRAME: usize = 124;

const HEADER: &str = "# PBF fluid state: the cube scene of tests/support/fluid_cube_fixture.rs\n\
# (cube 0.4 m above the floor, fluid to 1.2 m) stepped 124 frames of 1/60 s from the lattice.\n\
# Read by tests/analytic_fluid_newton_fixture.rs; written by\n\
# cargo run --example fluid_newton_fixture -- tests/fixtures/fluid_newton_cube_f124.bin\n\
# After the line `data`: u32 particle count, then per particle position xyz and velocity xyz,\n\
# each Fix128 as hi i64 then lo u64, all little-endian.\n\
data\n";

/// The scene stepped [`FRAME`] frames from the lattice
pub fn run_to_fixture_frame() -> Scene {
    let mut sc = scene(BOTTOM, TOP);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..FRAME {
        sc.fluid.step(dt);
    }
    sc
}

/// The stored form of the fluid's positions and velocities
pub fn encode(f: &Fluid) -> Vec<u8> {
    let mut out = HEADER.as_bytes().to_vec();
    let n = f.positions.len();
    out.extend_from_slice(&(n as u32).to_le_bytes());
    for i in 0..n {
        for w in [f.positions[i], f.velocities[i]] {
            for c in [w.x, w.y, w.z] {
                out.extend_from_slice(&c.hi.to_le_bytes());
                out.extend_from_slice(&c.lo.to_le_bytes());
            }
        }
    }
    out
}

/// Positions and velocities read back from [`encode`]'s form
pub fn decode(bytes: &[u8]) -> (Vec<Vec3Fix>, Vec<Vec3Fix>) {
    assert!(bytes.starts_with(HEADER.as_bytes()), "fixture header");
    let body = &bytes[HEADER.len()..];
    let n = u32::from_le_bytes(body[..4].try_into().unwrap()) as usize;
    assert_eq!(body.len(), 4 + n * 6 * 16, "fixture length");
    let fix = |k: usize| {
        let at = 4 + 16 * k;
        let hi = i64::from_le_bytes(body[at..at + 8].try_into().unwrap());
        let lo = u64::from_le_bytes(body[at + 8..at + 16].try_into().unwrap());
        Fix128::from_raw(hi, lo)
    };
    let vec = |k: usize| Vec3Fix::new(fix(k), fix(k + 1), fix(k + 2));
    let pos = (0..n).map(|i| vec(6 * i)).collect();
    let vel = (0..n).map(|i| vec(6 * i + 3)).collect();
    (pos, vel)
}
