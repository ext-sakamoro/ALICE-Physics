//! Convergence of the PBF density constraint next to an SDF wall: fluid on the rest lattice
//! (`(i + 1/2) s`) filling a 0.8 m tank 0.6 m deep, the tank given only by its SDF, stepped
//! once (one substep, no settling) with 1 to 10 constraint iterations from the same state.
//! Prints the mean `|rho / rho_0 - 1|` of the particles within one kernel radius of the floor
//! or walls, as measured by the last iteration (after `k - 1` corrections), and of the
//! interior particles.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::fluid::{Fluid, FluidConfig};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};

const S: f64 = 0.1;
const W: f64 = 0.8;
const DEPTH: f64 = 0.6;
const RHO0: f64 = 1000.0;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn box_sdf(p: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let q: Vec<f32> = (0..3).map(|i| (lo[i] - p[i]).max(p[i] - hi[i])).collect();
    let outside = q.iter().map(|x| x.max(0.0).powi(2)).sum::<f32>().sqrt();
    let inside = q[0].max(q[1]).max(q[2]).min(0.0);
    outside + inside
}

fn tank() -> SdfCollider {
    let (w, l, t) = (W as f32, (2.0 * S) as f32, (2.0 * DEPTH) as f32);
    let f = move |p: [f32; 3]| {
        let outer = box_sdf(p, [-l, -l, -l], [w + l, t, w + l]);
        let hole = box_sdf(p, [0.0, 0.0, 0.0], [w, t + 1.0, w]);
        outer.max(-hole)
    };
    let g = f;
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

fn main() {
    let s = fx(S);
    let h = s * Fix128::from_int(2);
    let (nx, ny) = ((W / S).round() as i64, (DEPTH / S).round() as i64);
    let mut pts = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nx {
                pts.push(v(
                    (i as f64 + 0.5) * S,
                    (j as f64 + 0.5) * S,
                    (k as f64 + 0.5) * S,
                ));
            }
        }
    }
    let r = 2.0 * S + 0.3;
    let hf = 2.0 * S;
    let near_wall = |p: &Vec3Fix| {
        let (x, y, z) = (p.x.to_f64(), p.y.to_f64(), p.z.to_f64());
        y < hf || x < hf || z < hf || x > W - hf || z > W - hf
    };
    println!("iterations | wall density error | interior density error");
    for k in 1..=10usize {
        let config = FluidConfig {
            rest_spacing: s,
            kernel_radius: h,
            rest_density: fx(RHO0),
            particle_mass: FluidConfig::lattice_mass(fx(RHO0), h, s),
            substeps: 1,
            iterations: k,
            ..FluidConfig::default()
        };
        let mut fluid = Fluid::new(pts.clone(), config).with_sdf_body(
            tank(),
            v(-r, -r, -r),
            v(W + r, DEPTH + 2.0 * S, W + r),
        );
        fluid.step(Fix128::from_ratio(1, 120));
        let (mut wall, mut nw, mut inner, mut ni) = (0.0, 0usize, 0.0, 0usize);
        for (p, d) in fluid.positions.iter().zip(&fluid.densities) {
            let e = (d.to_f64() / RHO0 - 1.0).abs();
            if near_wall(p) {
                wall += e;
                nw += 1;
            } else {
                inner += e;
                ni += 1;
            }
        }
        println!(
            "{k:2} | {:.6e} ({nw}) | {:.6e} ({ni})",
            wall / nw.max(1) as f64,
            inner / ni.max(1) as f64
        );
    }
}
