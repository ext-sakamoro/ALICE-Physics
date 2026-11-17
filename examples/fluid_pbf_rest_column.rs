//! A PBF water column resting in an open box of boundary particles.
//!
//! The box is lined with two layers of fixed boundary particles (`Fluid::box_boundary`),
//! the fluid is a 4 x 8 x 4 lattice at the rest spacing `h / 2` with the
//! lattice-calibrated mass (`FluidConfig::lattice_mass`). Two runs:
//!
//! - no gravity: the block must stay where it is (largest speed over the last 60 frames)
//! - gravity: the column settles and the reaction of the floor (`Fluid::boundary_force`) must carry the weight `M g`, the
//!   integral of `rho g h` over the floor
//!
//! ```bash
//! cargo run --release --example fluid_pbf_rest_column --features std
//! ```

use alice_physics_core::fluid::{Fluid, FluidConfig};
use alice_physics_core::math::{Fix128, Vec3Fix};

fn column(gravity: bool) -> Fluid {
    let mut config = FluidConfig::default();
    if !gravity {
        config.gravity = Vec3Fix::ZERO;
    }
    let s = config.rest_spacing;
    let half = s / Fix128::from_int(2);
    let (nx, ny, nz) = (4i64, 8i64, 4i64);
    let min = Vec3Fix::ZERO;
    let max = Vec3Fix::new(
        Fix128::from_int(nx) * s,
        Fix128::from_int(2 * ny) * s,
        Fix128::from_int(nz) * s,
    );
    let fluid = Fluid::new_block(
        Vec3Fix::new(half, half, half),
        Vec3Fix::new(
            Fix128::from_int(nx) * s - half,
            Fix128::from_int(ny) * s - half,
            Fix128::from_int(nz) * s - half,
        ),
        s,
        config,
    );
    fluid.with_boundary(Fluid::box_boundary(min, max, s, 2))
}

fn run(gravity: bool, frames: usize) -> (f64, f64, f64, f64) {
    let mut fluid = column(gravity);
    let dt = Fix128::from_ratio(1, 60);
    let rest_top = 8.0 * fluid.config.rest_spacing.to_f64();
    let mut tail = 0.0f64;
    for frame in 0..frames {
        fluid.step(dt);
        if frame + 60 >= frames {
            let v = fluid
                .velocities
                .iter()
                .map(|v| v.length().to_f64())
                .fold(0.0, f64::max);
            tail = tail.max(v);
        }
    }
    let top = fluid
        .positions
        .iter()
        .map(|p| p.y.to_f64())
        .fold(f64::MIN, f64::max);
    let rho0 = fluid.config.rest_density.to_f64();
    let mut d: Vec<f64> = fluid.densities.iter().map(|d| d.to_f64()).collect();
    d.sort_by(f64::total_cmp);
    let mid = &d[d.len() / 4..3 * d.len() / 4];
    let err = mid.iter().map(|x| (x / rho0 - 1.0).abs()).sum::<f64>() / mid.len() as f64;
    let weight = fluid.particle_count() as f64 * fluid.config.particle_mass.to_f64() * 10.0;
    let floor = fluid.boundary_force().y.to_f64();
    if gravity {
        // side wall x+ against 1/2 rho g H^2 D, H = M / (rho_0 W D)
        let s = fluid.config.rest_spacing.to_f64();
        let (w, d) = (4.0 * s, 4.0 * s);
        let mass = fluid.particle_count() as f64 * fluid.config.particle_mass.to_f64();
        let height = mass / (rho0 * w * d);
        let expected = 0.5 * rho0 * 10.0 * height * height * d;
        let wall = Fix128::from_int(4) * fluid.config.rest_spacing;
        let side: f64 = fluid
            .boundary
            .iter()
            .zip(fluid.boundary_forces())
            .filter(|(p, _)| p.x > wall && p.y > Fix128::ZERO)
            .map(|(_, f)| -f.x.to_f64())
            .sum();
        println!("side wall force / (1/2 rho g H^2 D) {:.4}", side / expected);
    }
    (
        top / rest_top,
        err,
        tail,
        if gravity { floor / weight } else { floor },
    )
}

fn main() {
    let (top, err, tail, floor) = run(false, 300);
    println!("no gravity: top/rest {top:.3} (rest 0.938), density error {err:.4}, max speed over last 60 frames {tail:.4} m/s, wall force {floor:.4} N");
    let (top, err, tail, floor) = run(true, 900);
    println!("gravity:    top/rest {top:.3}, density error {err:.4}, max speed over last 60 frames {tail:.4} m/s, floor reaction / weight {floor:.5}");
}
