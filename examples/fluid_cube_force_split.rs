//! Measurement only: the cube two kernel radii above the floor (tests/analytic_fluid_sdf_body.rs),
//! its vertical force split by term (density map vs contact) and by face, over the last 60 frames.
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

struct Scene {
    fluid: Fluid,
    n: usize,
}

/// Fluid on the lattice `(i + 1/2) s` filling `[0, W]^2 x [0, top]` outside the cube
fn scene(bottom: f64, top: f64) -> Scene {
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


fn main() {
    let (bottom, top) = (0.4, 1.2);
    let mut sc = scene(bottom, top);
    let g = FluidConfig::default().gravity.y.to_f64().abs();
    let lo = (W - CUBE) / 2.0;
    let (mut f_face, mut f_bins) = ([[0.0_f64; 2]; 3], [[0.0_f64; 2]; 4]);
    let mut per_frame: Vec<[[f64; 2]; 3]> = Vec::new();
    let frames = 900;
    for f in 0..frames {
        sc.fluid.step(Fix128::from_ratio(1, 60));
        if f + 60 < frames {
            continue;
        }
        let mut this = [[0.0_f64; 2]; 3];
        for &(k, p, force, kind) in &sc.fluid.diag {
            if k != 1 {
                continue;
            }
            let (x, y) = (p.x.to_f64(), p.y.to_f64());
            let face = if y > bottom + CUBE { 0 } else if y < bottom { 1 } else { 2 };
            f_face[face][kind as usize] += force.y.to_f64() / 60.0;
            this[face][kind as usize] += force.y.to_f64();
            if face == 1 {
                let bin = (((x - lo) / S).floor().max(0.0) as usize).min(3);
                f_bins[bin][kind as usize] += force.y.to_f64() / 60.0;
            }
        }
        per_frame.push(this);
    }
    let sd = |face: usize, kind: usize| {
        let m = per_frame.iter().map(|r| r[face][kind]).sum::<f64>() / per_frame.len() as f64;
        (per_frame.iter().map(|r| (r[face][kind] - m).powi(2)).sum::<f64>() / per_frame.len() as f64).sqrt()
    };
    let h = (sc.n as f64 * S * S * S + CUBE * CUBE * CUBE) / (W * W);
    let closed = RHO0 * g * CUBE * CUBE * CUBE;
    let p_bot = RHO0 * g * (h - bottom) * CUBE * CUBE;
    let p_top = RHO0 * g * (h - bottom - CUBE) * CUBE * CUBE;
    println!("contact factor {:?}; H {h:.4} m; closed Archimedes {closed:.2} N; bottom face closed {p_bot:.2} N up, top face closed {p_top:.2} N down", std::env::var("DIAG_CONTACT_FACTOR").ok());
    for (name, i) in [("top", 0), ("bottom", 1), ("side", 2)] {
        println!("{name}: map F_y {:.2} N (sd {:.2}), contact F_y {:.2} N (sd {:.2}), sum {:.2} N", f_face[i][0], sd(i, 0), f_face[i][1], sd(i, 1), f_face[i][0] + f_face[i][1]);
    }
    let tot: f64 = f_face.iter().map(|r| r[0] + r[1]).sum();
    let bf = sc.fluid.body_forces()[1].y.to_f64();
    println!("total diag F_y {tot:.2} N = {:.4} of closed (body_forces last frame {bf:.2})", tot / closed);
    for (b, r) in f_bins.iter().enumerate() {
        let closed_bin = RHO0 * g * (h - bottom) * S * CUBE;
        println!("bottom bin x {:.1}-{:.1}: map {:.2} contact {:.2} N, / closed {:.3}", lo + b as f64 * S, lo + (b as f64 + 1.0) * S, r[0], r[1], (r[0] + r[1]) / closed_bin);
    }
}
