//! The resting 4 x 8 x 4 column of tests/analytic_fluid_pbf_density.rs after 900 frames:
//! the x+ wall force per height bin against the hydrostatic closed form
//! `integral rho_0 g (H - y) D dy` over the bin, the density of the fluid layer next to the
//! wall per bin against the interior column at the same height, and how far the wall layer
//! sits from the wall.
use alice_physics_core::fluid::{Fluid, FluidConfig};
use alice_physics_core::math::{Fix128, Vec3Fix};

const NX: i64 = 4;
const NY: i64 = 8;
const NZ: i64 = 4;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let it: usize = a.get(1).map_or(10, |x| x.parse().unwrap());
    let sub: usize = a.get(2).map_or(2, |x| x.parse().unwrap());
    let config = FluidConfig {
        iterations: it,
        substeps: sub,
        ..FluidConfig::default()
    };
    println!("iterations {it} substeps {sub}");
    let s = config.rest_spacing;
    let half = s / Fix128::from_int(2);
    let fluid = Fluid::new_block(
        Vec3Fix::new(half, half, half),
        Vec3Fix::new(
            Fix128::from_int(NX) * s - half,
            Fix128::from_int(NY) * s - half,
            Fix128::from_int(NZ) * s - half,
        ),
        s,
        config,
    );
    let boundary = Fluid::box_boundary(
        Vec3Fix::ZERO,
        Vec3Fix::new(
            Fix128::from_int(NX) * s,
            Fix128::from_int(2 * NY) * s,
            Fix128::from_int(NZ) * s,
        ),
        s,
        2,
    );
    let mut fluid = fluid.with_boundary(boundary);
    for _ in 0..900 {
        fluid.step(Fix128::from_ratio(1, 60));
    }
    let sf = s.to_f64();
    let rho0 = config.rest_density.to_f64();
    let g = config.gravity.y.to_f64().abs();
    let (w, d) = (NX as f64 * sf, NZ as f64 * sf);
    let mass = fluid.particle_count() as f64 * config.particle_mass.to_f64();
    let hgt = mass / (rho0 * w * d);
    println!(
        "relaxation {} | H {hgt:.4} m, closed-form side force {:.2} N",
        config.relaxation.to_f64(),
        0.5 * rho0 * g * hgt * hgt * d
    );
    let forces = fluid.boundary_forces();
    let nb = (2 * NY) as usize;
    let mut wall_f = vec![0.0_f64; nb];
    for (p, f) in fluid.boundary.iter().zip(forces.iter()) {
        let (x, y) = (p.x.to_f64(), p.y.to_f64());
        if x > w + 1e-9 && y > 0.0 {
            let b = ((y / sf).floor() as usize).min(nb - 1);
            wall_f[b] += -f.x.to_f64();
        }
    }
    let (mut wl_rho, mut wl_n, mut wl_x, mut in_rho, mut in_n) = (
        vec![0.0; nb],
        vec![0usize; nb],
        vec![0.0; nb],
        vec![0.0; nb],
        vec![0usize; nb],
    );
    for (p, r) in fluid.positions.iter().zip(&fluid.densities) {
        let (x, y) = (p.x.to_f64(), p.y.to_f64());
        let b = ((y / sf).floor().max(0.0) as usize).min(nb - 1);
        let rr = r.to_f64() / rho0;
        if x > w - sf {
            wl_rho[b] += rr;
            wl_n[b] += 1;
            wl_x[b] += w - x;
        } else if x > sf && x < w - sf {
            in_rho[b] += rr;
            in_n[b] += 1;
        }
    }
    println!("bin y0-y1 (m) | wall Fx N | closed N | ratio | wall-layer rho/rho0 (n) | gap to wall / s | interior rho/rho0 (n)");
    let (mut tw, mut tc) = (0.0, 0.0);
    for b in 0..nb {
        let (y0, y1) = (b as f64 * sf, (b as f64 + 1.0) * sf);
        let (a, c) = (y0.min(hgt), y1.min(hgt));
        let closed = rho0 * g * d * (hgt * (c - a) - 0.5 * (c * c - a * a));
        tw += wall_f[b];
        tc += closed;
        if wall_f[b].abs() < 1e-9 && closed == 0.0 && wl_n[b] == 0 {
            continue;
        }
        let f = |s: f64, n: usize| if n == 0 { f64::NAN } else { s / n as f64 };
        println!(
            "{y0:.1}-{y1:.1} | {:9.3} | {closed:8.3} | {:6.3} | {:.4} ({}) | {:.3} | {:.4} ({})",
            wall_f[b],
            if closed > 0.0 {
                wall_f[b] / closed
            } else {
                f64::NAN
            },
            f(wl_rho[b], wl_n[b]),
            wl_n[b],
            f(wl_x[b], wl_n[b]) / sf,
            f(in_rho[b], in_n[b]),
            in_n[b]
        );
    }
    println!(
        "total wall {tw:.3} N, closed {tc:.3} N, ratio {:.4}",
        tw / tc
    );
    // (h2) x+ wall particles below the floor level (corner block, excluded by the oracle)
    let mut corner = 0.0;
    for (p, f) in fluid.boundary.iter().zip(forces.iter()) {
        if p.x.to_f64() > w + 1e-9 && p.y.to_f64() < 0.0 {
            corner += -f.x.to_f64();
        }
    }
    println!(
        "h2: x+ corner block below y = 0 carries {corner:.3} N ({:.4} of closed)",
        corner / tc
    );
    // (h3) the hydrostatic pressure p = rho_0 g (H - y) at the settled positions, carried to
    // each wall particle by the SPH pressure force vol^2 p |grad W| (shape only: scaled so
    // the 0.2-0.5 m bins match the closed form)
    let h = config.kernel_radius.to_f64();
    let vol = config.particle_mass.to_f64() / rho0;
    let cg = 45.0 / (std::f64::consts::PI * (h * h * h) * (h * h * h));
    let mut sm = vec![0.0_f64; nb + 1];
    for b in fluid.boundary.iter() {
        let (bx, by, bz) = (b.x.to_f64(), b.y.to_f64(), b.z.to_f64());
        if bx <= w + 1e-9 {
            continue;
        }
        let mut fxs = 0.0;
        for q in &fluid.positions {
            let (dx, dy, dz) = (q.x.to_f64() - bx, q.y.to_f64() - by, q.z.to_f64() - bz);
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r > 0.0 && r < h {
                let pr = rho0 * g * (hgt - q.y.to_f64()).max(0.0);
                fxs += vol * vol * pr * cg * (h - r) * (h - r) * (-dx / r);
            }
        }
        let k = if by < 0.0 {
            nb
        } else {
            ((by / sf).floor() as usize).min(nb - 1)
        };
        sm[k] += fxs.abs();
    }
    let mut cl = vec![0.0_f64; nb];
    for (b, c) in cl.iter_mut().enumerate() {
        let (a, cc) = ((b as f64 * sf).min(hgt), ((b as f64 + 1.0) * sf).min(hgt));
        *c = rho0 * g * d * (hgt * (cc - a) - 0.5 * (cc * cc - a * a));
    }
    let scale = (cl[2] + cl[3] + cl[4]) / (sm[2] + sm[3] + sm[4]);
    println!("h3: smeared hydrostatic / closed per bin (scaled on 0.2-0.5 m); corner block below y = 0 {:.4} of closed", sm[nb] * scale / tc);
    for b in 0..nb {
        if cl[b] > 0.0 {
            println!(
                "h3 {:.1}-{:.1} | {:.3} | measured {:.3} | interior C {:.5}",
                b as f64 * sf,
                (b as f64 + 1.0) * sf,
                sm[b] * scale / cl[b],
                wall_f[b] / cl[b],
                if in_n[b] > 0 {
                    in_rho[b] / in_n[b] as f64 - 1.0
                } else {
                    f64::NAN
                }
            );
        }
    }
}
