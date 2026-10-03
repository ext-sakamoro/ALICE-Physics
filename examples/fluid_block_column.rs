//! A column of PBF water built with `Fluid::new_block`.
//!
//! The lattice is checked against its closed form (`prod floor(extent/s) + 1`
//! particles, x-major order, exact dyadic coordinates), then the block is stepped
//! for a quarter of a second under gravity: the centre of mass must fall, and
//! because the density constraint starts satisfied or over-compressed it must
//! fall slower than a free particle (`g t^2 / 2`).
//!
//! ```bash
//! cargo run --release --example fluid_block_column --features std
//! ```

use alice_physics::fluid::{Fluid, FluidConfig};
use alice_physics::math::{Fix128, Vec3Fix};

fn main() {
    let spacing = Fix128::from_ratio(1, 10);
    let min = Vec3Fix::ZERO;
    let max = Vec3Fix::new(
        Fix128::from_ratio(3, 10),
        Fix128::from_ratio(5, 10),
        Fix128::from_ratio(3, 10),
    );
    let mut fluid = Fluid::new_block(min, max, spacing, FluidConfig::default());

    // 4 x 6 x 4 lattice: floor(0.3/0.1)+1, floor(0.5/0.1)+1, floor(0.3/0.1)+1
    let (nx, ny, nz) = (4usize, 6usize, 4usize);
    println!(
        "particles: {} (closed form {})",
        fluid.particle_count(),
        nx * ny * nz
    );
    assert_eq!(fluid.particle_count(), nx * ny * nz);
    // x-major: index = (ix * ny + iy) * nz + iz
    let probe = (2 * ny + 3) * nz + 1;
    let p = fluid.positions[probe];
    println!(
        "particle {probe} sits at ({:.3}, {:.3}, {:.3})",
        p.x.to_f64(),
        p.y.to_f64(),
        p.z.to_f64()
    );
    assert!(
        (p.x.to_f64() - 0.2).abs() < 1e-12
            && (p.y.to_f64() - 0.3).abs() < 1e-12
            && (p.z.to_f64() - 0.1).abs() < 1e-12
    );

    let com = |f: &Fluid| {
        f.positions.iter().map(|q| q.y.to_f64()).sum::<f64>() / f.particle_count() as f64
    };
    let y0 = com(&fluid);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..15 {
        fluid.step(dt);
    }
    let fall = y0 - com(&fluid);
    let free_fall = 0.5 * 10.0 * 0.25 * 0.25;
    println!("centre of mass fell {fall:.4} m in 0.25 s (free fall would be {free_fall:.4} m)");
    assert!(fall > 0.0 && fall < free_fall);
}
